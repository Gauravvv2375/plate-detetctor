"""Shared training helpers; not a user-facing command."""

from __future__ import annotations

import shutil
import os
import re
import signal
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from src.utils import PROJECT_ROOT


@dataclass
class YoloCheckpoint:
    path: Path
    completed_epochs: int
    target_epochs: int
    train_args: dict


@contextmanager
def training_lock(name: str):
    """Prevent two local processes from training the same model concurrently."""
    import fcntl

    lock_dir = PROJECT_ROOT / "runs/.locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{name}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "unknown process"
        handle.close()
        raise RuntimeError(f"Another {name} training job is already running ({owner}).") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started={datetime.now().isoformat(timespec='seconds')}\n")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def graceful_training_interrupts():
    """Turn SIGTERM into the same safe last-completed-epoch path as Ctrl+C."""
    previous = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(signum, frame):
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def backup_yolo_previous(trainer) -> None:
    """Before an epoch save, preserve the prior valid last.pt as previous.pt."""
    last = Path(trainer.last)
    if not last.is_file():
        return
    try:
        import torch

        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("optimizer") is None:
            return
    except Exception:
        return
    _atomic_copy(last, last.with_name("previous.pt"))


def _run_directory_matches(path: Path, run_name: str) -> bool:
    directory = path.parents[1].name
    return bool(re.fullmatch(rf"{re.escape(run_name)}\d*", directory))


def _checkpoint_completed_epochs(checkpoint: dict) -> int:
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch + 1
    results = checkpoint.get("train_results") or {}
    recorded = results.get("epoch") or []
    return int(max(recorded)) if recorded else 0


def _architecture_signature(model) -> tuple:
    """Describe architecture by model class and parameter/buffer shapes, not checkpoint filename."""
    return (
        type(model).__module__,
        type(model).__qualname__,
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items()),
    )


@lru_cache(maxsize=16)
def _requested_architecture(model_name: str) -> tuple[str, tuple | str]:
    """Resolve a local .pt initializer to its real architecture when possible."""
    import torch

    requested = Path(model_name).expanduser()
    if not requested.is_absolute():
        project_relative = PROJECT_ROOT / requested
        if project_relative.is_file():
            requested = project_relative
    if requested.is_file():
        initializer = torch.load(requested, map_location="cpu", weights_only=False)
        if not isinstance(initializer, dict):
            raise ValueError(f"initializer {requested} is not an Ultralytics checkpoint")
        model = initializer.get("ema") or initializer.get("model")
        if model is None:
            raise ValueError(f"initializer {requested} has no model/EMA state")
        return "state_shapes", _architecture_signature(model)
    return "yaml_stem", Path(model_name).stem


def inspect_yolo_checkpoint(
    path: Path,
    *,
    task: str,
    model_name: str,
    yaml_path: Path,
    imgsz: int,
    epochs: int,
    expected_names: dict[int, str],
) -> YoloCheckpoint:
    """Load and verify a resumable Ultralytics checkpoint without altering it."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "train_args" not in checkpoint:
        raise ValueError("not an Ultralytics training checkpoint")
    train_args = checkpoint["train_args"]
    mismatches = []
    if train_args.get("task") != task:
        mismatches.append(f"task={train_args.get('task')!r}, expected {task!r}")
    if int(train_args.get("imgsz", -1)) != int(imgsz):
        mismatches.append(f"imgsz={train_args.get('imgsz')!r}, requested {imgsz}")
    target_epochs = int(train_args.get("epochs", 0))
    if target_epochs != int(epochs):
        mismatches.append(f"target epochs={target_epochs}, requested {epochs}")
    stored_data = Path(str(train_args.get("data", ""))).expanduser()
    if stored_data.resolve() != yaml_path.resolve():
        mismatches.append(f"data={stored_data}, expected {yaml_path.resolve()}")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        mismatches.append("model/EMA state is missing")
    else:
        names = {int(key): value for key, value in dict(getattr(model, "names", {})).items()}
        if names != expected_names:
            mismatches.append(f"class names={names!r}, expected {expected_names!r}")
        architecture_kind, expected_architecture = _requested_architecture(str(model_name))
        if architecture_kind == "state_shapes":
            if _architecture_signature(model) != expected_architecture:
                mismatches.append("model architecture/state shapes differ from the requested initializer")
        else:
            yaml_file = Path(str(getattr(model, "yaml", {}).get("yaml_file", ""))).stem
            if yaml_file and yaml_file != expected_architecture:
                mismatches.append(f"model architecture={yaml_file!r}, requested {expected_architecture!r}")
    completed_epochs = _checkpoint_completed_epochs(checkpoint)
    if 0 < completed_epochs < target_epochs and checkpoint.get("optimizer") is None:
        mismatches.append("optimizer state is missing from incomplete checkpoint")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    return YoloCheckpoint(path.resolve(), completed_epochs, target_epochs, train_args)


def find_yolo_checkpoint(
    *,
    run_name: str,
    task: str,
    model_name: str,
    yaml_path: Path,
    imgsz: int,
    epochs: int,
    expected_names: dict[int, str],
    require: bool,
) -> YoloCheckpoint | None:
    candidates = [
        path
        for path in PROJECT_ROOT.glob(f"runs/{run_name}*/weights/*.pt")
        if path.name in {"last.pt", "previous.pt"} and _run_directory_matches(path, run_name)
    ]
    candidates.sort(key=lambda path: (path.stat().st_mtime, path.name == "last.pt"), reverse=True)
    errors = []
    compatible = []
    for path in candidates:
        try:
            checkpoint = inspect_yolo_checkpoint(
                path,
                task=task,
                model_name=model_name,
                yaml_path=yaml_path,
                imgsz=imgsz,
                epochs=epochs,
                expected_names=expected_names,
            )
            if checkpoint.completed_epochs:
                compatible.append(checkpoint)
            else:
                errors.append(f"{path}: no completed epoch recorded")
        except Exception as error:
            errors.append(f"{path}: {error}")
    if compatible:
        selected = compatible[0]
        if selected.path.name == "previous.pt":
            print(f"Newest compatible last.pt was unusable; falling back to valid checkpoint: {selected.path}")
        return selected
    if candidates or require:
        detail = "\n".join(errors) if errors else "no checkpoint files found"
        raise RuntimeError(f"No compatible resumable checkpoint for {run_name}:\n{detail}")
    return None


def print_training_banner(title: str, mode: str, target: int, device: str, checkpoint: YoloCheckpoint | None = None) -> None:
    print("=" * 40)
    print(title)
    print(f"Mode: {mode}")
    if checkpoint:
        print(f"Checkpoint: {checkpoint.path}")
        print(f"Completed epochs: {checkpoint.completed_epochs}")
        print(f"Resuming from: epoch {checkpoint.completed_epochs + 1}")
    print(f"Target epochs: {target}")
    print(f"Device: {device}")
    print("=" * 40)


def ensure_ultralytics_mps_safe(device: str) -> None:
    """Reject releases affected by upstream MPS non-blocking transfer corruption."""
    if str(device).lower() != "mps":
        return
    from packaging.version import Version

    installed = Version(version("ultralytics"))
    minimum = Version("8.3.204")
    if installed < minimum:
        raise RuntimeError(
            f"Ultralytics {installed} is unsafe for MPS validation. Versions before {minimum} can corrupt "
            "class tensors through non-blocking MPS transfers. Run: pip install -r requirements.txt"
        )


def runtime_yolo_yaml(dataset: Path) -> Path:
    """Create a local absolute YAML at runtime while keeping committed YAML portable."""
    dataset = dataset.resolve()
    original = dataset / "data.yaml"
    lines = original.read_text(encoding="utf-8").splitlines()
    lines = [f"path: {dataset}" if line.strip().startswith("path:") else line for line in lines]
    target = PROJECT_ROOT / "reports" / f"{dataset.name}_runtime.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def filtered_detection_test_yaml(dataset: Path) -> tuple[Path, int, list[str]]:
    """Build an evaluation YAML that excludes unlabelled or visibly annotated test images."""
    dataset = dataset.resolve()
    accepted, excluded = [], []
    for image in sorted((dataset / "images/test").glob("*")):
        if not image.is_file():
            continue
        label = dataset / "labels/test" / f"{image.stem}.txt"
        suspicious_name = any(word in image.stem.lower() for word in ("annotated", "drawn", "visualized"))
        if not label.is_file() or not label.read_text(encoding="utf-8", errors="replace").strip() or suspicious_name:
            excluded.append(str(image))
        else:
            accepted.append(str(image))
    image_list = PROJECT_ROOT / "reports/det_test_valid.txt"
    image_list.parent.mkdir(parents=True, exist_ok=True)
    image_list.write_text("\n".join(accepted) + "\n", encoding="utf-8")
    lines = (dataset / "data.yaml").read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        if line.strip().startswith("path:"):
            rewritten.append(f"path: {dataset}")
        elif line.strip().startswith("test:"):
            rewritten.append(f"test: {image_list}")
        else:
            rewritten.append(line)
    yaml = PROJECT_ROOT / "reports/det_test_filtered_runtime.yaml"
    yaml.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return yaml, len(accepted), excluded


def save_yolo_best(model, destination: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics trainer state is unavailable after training")
    source = Path(getattr(trainer, "best", Path(trainer.save_dir) / "weights" / "best.pt"))
    if not source.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
