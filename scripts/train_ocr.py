"""Train PARSeq OCR. Input: stitched JSONL crops/text. Processing: realistic augmentation and docTR PARSeq optimization. Output: best validation checkpoint."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm.auto import tqdm

from src.utils import PLATE_CHARSET, PROJECT_ROOT, choose_device, clean_plate_text, read_jsonl, resolve_manifest_path
from scripts._training import graceful_training_interrupts, training_lock


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def augment(image: Image.Image) -> Image.Image:
    """Apply mild geometry before the final fixed-size resize."""
    if random.random() < 0.45:
        image = image.rotate(random.uniform(-4, 4), resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if random.random() < 0.35:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.3))
    if random.random() < 0.2:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.1)))
    return image


class OCRDataset:
    def __init__(self, manifest: Path, training: bool, limit: int = 0):
        self.records = []
        for record in read_jsonl(manifest):
            image_path = resolve_manifest_path(str(record.get("path", "")), manifest)
            label = clean_plate_text(str(record.get("text", "")))
            if image_path.is_file() and label and set(label) <= set(PLATE_CHARSET):
                self.records.append((image_path, label))
                if limit and len(self.records) >= limit:
                    break
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        path, label = self.records[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.training:
            image = augment(image)
        image = image.resize((128, 32), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array), label


def collate(batch):
    import torch

    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


def prediction_pairs(output, labels):
    predictions = output.get("preds", [])
    texts = [clean_plate_text(str(item[0] if isinstance(item, (tuple, list)) else item)) for item in predictions]
    return zip(texts, labels)

 
def run_epoch(model, loader, device, optimizer=None, epoch=None, total_epochs=None):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = exact = edits = characters = samples = seen = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        batches = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="batch",
            dynamic_ncols=True,
        ) if training and epoch is not None else loader
        for images, labels in batches:
            images = images.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # docTR flattens training logits for the permutation loss, so its
            # postprocessor is intentionally requested only in evaluation mode.
            output = model(images, target=labels, return_preds=not training)
            loss = output["loss"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            loss_value = float(loss.detach())
            total_loss += loss_value * len(labels)
            seen += len(labels)
            if training and epoch is not None:
                batches.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6g}")
            for prediction, truth in prediction_pairs(output, labels):
                exact += prediction == truth
                edits += edit_distance(prediction, truth)
                characters += len(truth)
                samples += 1
    return {"loss": total_loss / max(1, seen), "exact": (exact / max(1, samples)) if not training else None, "cer": (edits / max(1, characters)) if not training else None}


def ocr_training_config(args: argparse.Namespace) -> dict:
    return {
        "task": "parseq_ocr",
        "architecture": "doctr_parseq",
        "data_dir": str(args.data_dir.resolve()),
        "target_epochs": int(args.epochs),
        "batch": int(args.batch),
        "learning_rate": float(args.lr),
        "vocab": PLATE_CHARSET,
        "image_size": [32, 128],
        "max_records": int(args.max_records),
    }


def validate_ocr_checkpoint(checkpoint: dict, expected_config: dict) -> tuple[int, float]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not a dictionary")
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_exact_accuracy",
        "config",
        "vocab",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"missing keys: {missing}")
    mismatches = []
    stored = checkpoint["config"]
    for key, expected in expected_config.items():
        if stored.get(key) != expected:
            mismatches.append(f"{key}={stored.get(key)!r}, requested {expected!r}")
    if checkpoint["vocab"] != PLATE_CHARSET:
        mismatches.append("character vocabulary differs")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    epoch = int(checkpoint["epoch"])
    if epoch < 1:
        raise ValueError(f"invalid completed epoch {epoch}")
    return epoch, float(checkpoint["best_exact_accuracy"])


def load_ocr_recovery(directory: Path, config: dict, require: bool):
    import torch

    errors = []
    found = False
    for name in ("last.pt", "previous.pt"):
        path = directory / name
        if not path.is_file():
            continue
        found = True
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            completed, best = validate_ocr_checkpoint(checkpoint, config)
            if name == "previous.pt":
                print(f"Newest OCR checkpoint was unusable; falling back to: {path}")
            return path, checkpoint, completed, best
        except Exception as error:
            errors.append(f"{path}: {error}")
    if found or require:
        detail = "\n".join(errors) if errors else "no last.pt or previous.pt found"
        raise RuntimeError(f"No compatible PARSeq recovery checkpoint:\n{detail}")
    return None


def _atomic_checkpoint_write(checkpoint: dict, destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # A successful reload is required before replacing a known-good file.
        candidate = torch.load(temporary, map_location="cpu", weights_only=False)
        if not isinstance(candidate, dict) or "model_state_dict" not in candidate:
            raise ValueError("temporary checkpoint validation failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_ocr_recovery(checkpoint: dict, directory: Path, is_best: bool) -> None:
    last, previous = directory / "last.pt", directory / "previous.pt"
    if last.is_file():
        try:
            import torch

            existing = torch.load(last, map_location="cpu", weights_only=False)
            if isinstance(existing, dict) and "model_state_dict" in existing:
                descriptor, temporary_name = tempfile.mkstemp(prefix=".previous.pt.", suffix=".tmp", dir=directory)
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    shutil.copy2(last, temporary)
                    temporary.replace(previous)
                finally:
                    temporary.unlink(missing_ok=True)
        except Exception:
            pass
    _atomic_checkpoint_write(checkpoint, last)
    if is_best:
        _atomic_checkpoint_write(checkpoint, directory / "best.pt")


def archive_ocr_checkpoints(directory: Path) -> Path | None:
    existing = [directory / name for name in ("last.pt", "previous.pt", "best.pt") if (directory / name).is_file()]
    if not existing:
        return None
    archive = directory / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        path.replace(archive / path.name)
    return archive


def move_optimizer_state(optimizer, device: str) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/ocr_stitched")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Require a valid incomplete recovery checkpoint")
    mode.add_argument("--fresh", action="store_true", help="Archive current OCR checkpoints and start again")
    parser.add_argument("--max-records", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models/ocr", help=argparse.SUPPRESS)
    parser.add_argument("--run-name", default="parseq_ocr", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import torch
    from doctr.models import parseq
    from torch.utils.data import DataLoader

    device = choose_device(args.device)
    destination_dir = args.checkpoint_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    config = ocr_training_config(args)
    if args.fresh:
        with training_lock(args.run_name):
            archive = archive_ocr_checkpoints(destination_dir)
        if archive:
            print(f"Archived previous OCR checkpoints: {archive}")
        recovery = None
    else:
        recovery = load_ocr_recovery(destination_dir, config, args.resume)
    if recovery and recovery[2] >= args.epochs:
        print(f"Training already completed: {recovery[2]}/{args.epochs} epochs.")
        print("Use --fresh to intentionally train again.")
        return 0
    print("=" * 40)
    print("PARSEQ OCR")
    print(f"Mode: {'AUTO RESUME' if recovery else 'NEW TRAINING'}")
    if recovery:
        print(f"Checkpoint: {recovery[0]}")
        print(f"Completed epochs: {recovery[2]}")
        print(f"Resuming from: epoch {recovery[2] + 1}")
    print(f"Target epochs: {args.epochs}")
    print(f"Device: {device}")
    print("=" * 40)
    train_data = OCRDataset(args.data_dir / "train.jsonl", True, args.max_records)
    val_data = OCRDataset(args.data_dir / "val.jsonl", False, args.max_records)
    if not train_data.records or not val_data.records:
        raise RuntimeError("No valid OCR records. Run scripts/prepare_dataset.py and inspect the report.")
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)
    # A resumed run loads its exact model state and does not redownload weights.
    model = parseq(pretrained=not bool(recovery), vocab=PLATE_CHARSET).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_exact = 1, -1.0
    if recovery:
        checkpoint = recovery[1]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best_exact = recovery[2] + 1, recovery[3]
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    try:
        with training_lock(args.run_name), graceful_training_interrupts():
            for epoch in range(start_epoch, args.epochs + 1):
                print(f"\nEpoch {epoch}/{args.epochs}")
                train_metrics = run_epoch(model, train_loader, device, optimizer, epoch, args.epochs)
                val_metrics = run_epoch(model, val_loader, device)
                scheduler.step()
                print(f"Epoch {epoch}/{args.epochs} completed")
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Accuracy: {val_metrics['exact']:.2%}")
                print(f"Character Error Rate: {val_metrics['cer']:.2%}")
                improved = val_metrics["exact"] > best_exact
                if improved:
                    best_exact = val_metrics["exact"]
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_exact_accuracy": best_exact,
                    "val_metrics": val_metrics,
                    "config": config,
                    "vocab": PLATE_CHARSET,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                }
                save_ocr_recovery(checkpoint, destination_dir, improved)
                if args.stop_after_epoch and epoch >= args.stop_after_epoch:
                    raise KeyboardInterrupt("controlled checkpoint smoke-test interruption")
    except KeyboardInterrupt:
        print(f"Training interrupted. Resume by running the same command. Last completed checkpoint: {destination_dir / 'last.pt'}")
        return 130
    print(destination_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
