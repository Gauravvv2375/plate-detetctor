"""Train PARSeq OCR from split JSONL or UTF-8 labels.csv data."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm.auto import tqdm

from src.utils import PLATE_CHARSET, PROJECT_ROOT, choose_device, clean_plate_text, read_jsonl, resolve_manifest_path
from scripts._training import graceful_training_interrupts, training_lock

DOCTR_MEAN = np.asarray((0.694, 0.695, 0.693), dtype=np.float32).reshape(3, 1, 1)
DOCTR_STD = np.asarray((0.299, 0.296, 0.301), dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class OCRRecord:
    path: Path
    label: str
    layout: str = "one_line"
    split: str = ""
    group_id: str = ""


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def aligned_character_matches(prediction: str, truth: str) -> int:
    """Count exact characters on a minimum-edit alignment."""
    costs = [[0] * (len(truth) + 1) for _ in range(len(prediction) + 1)]
    for i in range(len(prediction) + 1):
        costs[i][0] = i
    for j in range(len(truth) + 1):
        costs[0][j] = j
    for i, predicted in enumerate(prediction, 1):
        for j, expected in enumerate(truth, 1):
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1] + (predicted != expected),
            )
    matches, i, j = 0, len(prediction), len(truth)
    while i or j:
        if i and j and prediction[i - 1] == truth[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            matches += 1
            i, j = i - 1, j - 1
        elif i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return matches


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


def normalize_ocr_text(text: str, vocab: str) -> str:
    """Normalize Unicode without changing Devanagari combining-character order."""
    text = unicodedata.normalize("NFC", text)
    if vocab == PLATE_CHARSET:
        text = text.upper()
    allowed = set(vocab)
    return "".join(character for character in text if character in allowed)


def _manifest_records(manifest: Path) -> list[OCRRecord]:
    records = []
    for record in read_jsonl(manifest):
        image_path = resolve_manifest_path(str(record.get("path", "")), manifest)
        label = clean_plate_text(str(record.get("text", "")))
        if image_path.is_file() and label and set(label) <= set(PLATE_CHARSET):
            records.append(OCRRecord(image_path, label))
    return records


def _csv_records(data_dir: Path) -> list[OCRRecord]:
    labels_path = data_dir / "labels.csv"
    records = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "text"} <= set(reader.fieldnames):
            raise ValueError(f"{labels_path} must contain 'image' and 'text' columns")
        for line_number, row in enumerate(reader, 2):
            label = unicodedata.normalize("NFC", row["text"]).strip()
            image_path = (data_dir / row["image"]).resolve()
            if not image_path.is_relative_to(data_dir.resolve()):
                raise ValueError(f"Image path escapes dataset directory at {labels_path}:{line_number}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing OCR image at {labels_path}:{line_number}: {image_path}")
            if not label:
                raise ValueError(f"Empty OCR label at {labels_path}:{line_number}")
            layout = row.get("layout", "one_line")
            if layout not in {"one_line", "two_line", "header_registration"}:
                raise ValueError(f"Unsupported layout at {labels_path}:{line_number}: {layout!r}")
            if any(unicodedata.category(character).startswith("C") for character in label):
                raise ValueError(f"Control/format character in OCR label at {labels_path}:{line_number}")
            split = row.get("split", "").strip()
            if split and split not in {"train", "val"}:
                raise ValueError(f"Unsupported split at {labels_path}:{line_number}: {split!r}")
            records.append(OCRRecord(image_path, label, layout, split, row.get("group_id", "").strip()))
    if not records:
        raise ValueError(f"No OCR records found in {labels_path}")
    return records


def _dataset_vocabulary(data_dir: Path, records: list[OCRRecord]) -> str:
    derived = "".join(sorted({character for record in records for character in record.label}, key=ord))
    charset_path = data_dir / "charset.txt"
    if not charset_path.is_file():
        return derived
    lines = charset_path.read_text(encoding="utf-8-sig").splitlines()
    if any(len(line) != 1 for line in lines) or len(lines) != len(set(lines)):
        raise ValueError(f"{charset_path} must contain one unique character per line")
    declared = "".join(lines)
    if set(declared) != set(derived):
        missing = "".join(sorted(set(derived) - set(declared), key=ord))
        unused = "".join(sorted(set(declared) - set(derived), key=ord))
        raise ValueError(f"{charset_path} disagrees with labels.csv: missing={missing!r}, unused={unused!r}")
    return declared


def load_datasets(data_dir: Path, val_fraction: float, split_seed: int):
    """Load legacy JSONL splits or deterministically split a UTF-8 labels.csv dataset."""
    train_manifest, val_manifest = data_dir / "train.jsonl", data_dir / "val.jsonl"
    if train_manifest.is_file() and val_manifest.is_file():
        train_records = _manifest_records(train_manifest)
        val_records = _manifest_records(val_manifest)
        return train_records, val_records, PLATE_CHARSET, 32, "jsonl"
    if not (data_dir / "labels.csv").is_file():
        raise FileNotFoundError(f"Expected train.jsonl + val.jsonl or labels.csv in {data_dir}")
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    records = _csv_records(data_dir)
    has_explicit_split = [bool(record.split) for record in records]
    if any(has_explicit_split) and not all(has_explicit_split):
        raise ValueError("labels.csv must specify split for every row or for no rows")
    if all(has_explicit_split):
        train_records = [record for record in records if record.split == "train"]
        val_records = [record for record in records if record.split == "val"]
        train_groups = {record.group_id for record in train_records if record.group_id}
        val_groups = {record.group_id for record in val_records if record.group_id}
        overlap = train_groups & val_groups
        if overlap:
            raise ValueError(f"Train/validation group leakage detected: {sorted(overlap)[:5]}")
    else:
        random.Random(split_seed).shuffle(records)
        val_count = max(1, round(len(records) * val_fraction))
        val_records, train_records = records[:val_count], records[val_count:]
    if not train_records or not val_records:
        raise ValueError("Both training and validation splits must contain at least one record")
    vocab = _dataset_vocabulary(data_dir, records)
    # docTR 0.11's target_size includes the SOS and EOS positions. Reserve both
    # so the longest label is encoded in full instead of being silently cut.
    max_length = max(len(record.label) for record in records) + 2
    return train_records, val_records, vocab, max_length, "csv"


def _stitch_two_line_image(image: Image.Image) -> Image.Image:
    """Convert a synthetic stacked plate into the horizontal row order used at inference."""
    width, height = image.size
    gray = np.asarray(image.convert("L"))
    low, high = max(1, int(height * 0.35)), min(height - 1, int(height * 0.65))
    darkness = (255 - gray).mean(axis=1)
    split = low + int(np.argmin(darkness[low:high]))
    top = image.crop((0, 0, width, split))
    bottom = image.crop((0, split, width, height))
    target_height = max(top.height, bottom.height)

    def resize_row(row: Image.Image) -> Image.Image:
        row_width = max(1, round(row.width * target_height / row.height))
        return row.resize((row_width, target_height), Image.Resampling.BILINEAR)

    top, bottom = resize_row(top), resize_row(bottom)
    stitched = Image.new("RGB", (top.width + bottom.width, target_height), "white")
    stitched.paste(top, (0, 0))
    stitched.paste(bottom, (top.width, 0))
    return stitched


def _crop_registration_below_header(image: Image.Image) -> Image.Image:
    """Mirror row filtering for synthetic plates that include a small decorative header."""
    top = max(0, round(image.height * 0.29))
    return image.crop((0, top, image.width, image.height))


class OCRDataset:
    def __init__(self, records: list[OCRRecord], training: bool, normalize: bool = False, limit: int = 0):
        self.records = records[:limit] if limit else records
        self.training = training
        self.normalize = normalize

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        with Image.open(record.path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if record.layout == "two_line":
            image = _stitch_two_line_image(image)
        elif record.layout == "header_registration":
            image = _crop_registration_below_header(image)
        if self.training:
            image = augment(image)
        image = image.resize((128, 32), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.normalize:
            array = (array - DOCTR_MEAN) / DOCTR_STD
        return torch.from_numpy(array), record.label, str(record.path)


def collate(batch):
    import torch

    images, labels, paths = zip(*batch)
    return torch.stack(images), list(labels), list(paths)


def codepoints(text: str) -> str:
    return " ".join(f"U+{ord(character):04X}" for character in text)

 
def run_epoch(model, loader, device, vocab, optimizer=None, epoch=None, total_epochs=None, diagnostic_samples=0):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = exact = edits = character_matches = characters = samples = seen = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        batches = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="batch",
            dynamic_ncols=True,
        ) if training and epoch is not None else loader
        diagnostics_printed = 0
        for images, labels, paths in batches:
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
            for item, truth, path in zip(output.get("preds", []), labels, paths):
                raw = str(item[0] if isinstance(item, (tuple, list)) else item)
                prediction = normalize_ocr_text(raw, vocab)
                exact += prediction == truth
                edits += edit_distance(prediction, truth)
                character_matches += aligned_character_matches(prediction, truth)
                characters += len(truth)
                samples += 1
                if not training and diagnostics_printed < diagnostic_samples:
                    print(f"\nIMAGE: {path}")
                    print(f"GROUND TRUTH: {truth}")
                    print(f"RAW PREDICTION: {raw}")
                    print(f"NORMALIZED PREDICTION: {prediction}")
                    print(f"GROUND TRUTH CODEPOINTS: {codepoints(truth)}")
                    print(f"PREDICTION CODEPOINTS: {codepoints(prediction)}")
                    diagnostics_printed += 1
    cer = (edits / max(1, characters)) if not training else None
    return {
        "loss": total_loss / max(1, seen),
        "exact": (exact / max(1, samples)) if not training else None,
        "cer": cer,
        "character_accuracy": (character_matches / max(1, characters)) if not training else None,
    }


def ocr_training_config(args: argparse.Namespace, vocab: str, max_length: int, dataset_format: str) -> dict:
    config = {
        "task": "parseq_ocr",
        "architecture": "doctr_parseq",
        "data_dir": str(args.data_dir.resolve()),
        "target_epochs": int(args.epochs),
        "batch": int(args.batch),
        "learning_rate": float(args.lr),
        "vocab": vocab,
        "image_size": [32, 128],
        "max_records": int(args.max_records),
    }
    if dataset_format == "csv":
        config.update({
            "dataset_format": "csv",
            "validation_fraction": float(args.val_fraction),
            "split_seed": int(args.split_seed),
            "max_length": int(max_length),
            "input_normalization": "doctr_parseq",
            "two_line_preprocess": "horizontal_stitch_v1",
            "initialization": "doctr_pretrained_except_token_layers",
        })
        dataset_info_path = args.data_dir / "dataset_info.json"
        if dataset_info_path.is_file():
            import json

            dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
            if dataset_info.get("generator") == "scripts/generate_mixed_ocr_dataset.py":
                config.update({
                    "preprocessing": {
                        "input_normalization": "doctr_parseq",
                        "two_line": "horizontal_stitch_v1",
                        "decorative_header": "registration_row_crop_v1",
                    },
                    "initialization_version": "doctr_compatible_transfer_v1",
                    "split_strategy": dataset_info.get("split_strategy"),
                })
    return config


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
        if key == "target_epochs":
            continue
        if stored.get(key) != expected:
            mismatches.append(f"{key}={stored.get(key)!r}, requested {expected!r}")
    stored_target = int(stored.get("target_epochs", 0))
    requested_target = int(expected_config["target_epochs"])
    if requested_target < stored_target:
        mismatches.append(f"target_epochs cannot decrease from {stored_target} to {requested_target}")
    if checkpoint["vocab"] != expected_config["vocab"]:
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
            best_cer = checkpoint.get("best_cer")
            if best_cer is None and (directory / "best.pt").is_file():
                best_checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
                validate_ocr_checkpoint(best_checkpoint, config)
                best_cer = best_checkpoint.get("val_metrics", {}).get("cer")
            best_cer = float(best_cer if best_cer is not None else float("inf"))
            if name == "previous.pt":
                print(f"Newest OCR checkpoint was unusable; falling back to: {path}")
            return path, checkpoint, completed, best, best_cer
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


def build_parseq_with_compatible_transfer(parseq, vocabulary: str, max_length: int):
    """Transfer compatible docTR PARSeq features while replacing vocabulary-specific layers."""
    import torch

    model = parseq(pretrained=False, pretrained_backbone=False, vocab=vocabulary, max_length=max_length)
    pretrained = parseq(pretrained=True)
    source, target = pretrained.state_dict(), model.state_dict()
    excluded = ("head.", "embed.", "pos_queries")
    compatible = {
        key: value for key, value in source.items()
        if not key.startswith(excluded) and key in target and value.shape == target[key].shape
    }
    model.load_state_dict(compatible, strict=False)
    with torch.no_grad():
        positions = min(pretrained.pos_queries.shape[1], model.pos_queries.shape[1])
        model.pos_queries[:, :positions].copy_(pretrained.pos_queries[:, :positions])
    return model, len(compatible), sum(value.numel() for value in compatible.values())


def restore_extended_scheduler(optimizer, saved_state: dict, completed_epochs: int, target_epochs: int):
    """Restore cosine scheduling, recomputing its position when the total target grows."""
    import torch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_epochs)
    saved_target = int(saved_state["T_max"])
    if saved_target == target_epochs:
        scheduler.load_state_dict(saved_state)
        return scheduler, False
    base_lrs = list(saved_state["base_lrs"])
    eta_min = float(saved_state.get("eta_min", 0.0))
    lrs = [
        eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * completed_epochs / target_epochs)) / 2
        for base_lr in base_lrs
    ]
    for group, lr, base_lr in zip(optimizer.param_groups, lrs, base_lrs):
        group["lr"] = lr
        group["initial_lr"] = base_lr
    scheduler.base_lrs = base_lrs
    scheduler.last_epoch = completed_epochs
    scheduler._step_count = completed_epochs + 1
    scheduler._last_lr = lrs
    return scheduler, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/ocr_stitched")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction for labels.csv datasets")
    parser.add_argument("--split-seed", type=int, default=42, help="Deterministic labels.csv split seed")
    parser.add_argument("--diagnostic-samples", type=int, default=3, help="Validation predictions to print per epoch")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Require a valid incomplete recovery checkpoint")
    mode.add_argument("--fresh", action="store_true", help="Archive current OCR checkpoints and start again")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate best.pt without training")
    parser.add_argument("--max-records", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models/ocr")
    parser.add_argument("--run-name", default="parseq_ocr", help=argparse.SUPPRESS)
    args = parser.parse_args()
    import torch
    from doctr.models import parseq
    from torch.utils.data import DataLoader

    device = choose_device(args.device)
    train_records, val_records, vocabulary, max_length, dataset_format = load_datasets(
        args.data_dir.resolve(), args.val_fraction, args.split_seed
    )
    destination_dir = args.checkpoint_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    config = ocr_training_config(args, vocabulary, max_length, dataset_format)
    if args.evaluate_only:
        checkpoint_path = destination_dir / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PARSeq checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stored_config = checkpoint.get("config", {})
        for training_only_key in ("target_epochs", "batch", "learning_rate"):
            if training_only_key in stored_config:
                config[training_only_key] = stored_config[training_only_key]
        completed, best = validate_ocr_checkpoint(checkpoint, config)
        best_cer = float(checkpoint.get("best_cer", checkpoint.get("val_metrics", {}).get("cer", float("inf"))))
        recovery = checkpoint_path, checkpoint, completed, best, best_cer
    elif args.fresh:
        with training_lock(args.run_name):
            archive = archive_ocr_checkpoints(destination_dir)
        if archive:
            print(f"Archived previous OCR checkpoints: {archive}")
        recovery = None
    else:
        recovery = load_ocr_recovery(destination_dir, config, args.resume)
    if recovery and recovery[2] >= args.epochs and not args.evaluate_only:
        print(f"Training already completed: {recovery[2]}/{args.epochs} epochs.")
        print("Use --fresh to intentionally train again.")
        return 0
    normalize_inputs = dataset_format == "csv"
    print("=" * 40)
    print("PARSEQ OCR")
    print(f"Mode: {'EVALUATION' if args.evaluate_only else ('AUTO RESUME' if recovery else 'NEW TRAINING')}")
    if recovery:
        print(f"Checkpoint: {recovery[0]}")
        print(f"Completed epochs: {recovery[2]}")
        if not args.evaluate_only:
            print(f"Resuming from: epoch {recovery[2] + 1}")
    print(f"Target epochs: {config['target_epochs']}")
    print(f"Device: {device}")
    print(f"Dataset format: {dataset_format}")
    print(f"Training samples: {min(len(train_records), args.max_records) if args.max_records else len(train_records)}")
    print(f"Validation samples: {min(len(val_records), args.max_records) if args.max_records else len(val_records)}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Maximum label length: {max(len(record.label) for record in train_records + val_records)}")
    print(f"Model max_length: {max_length}")
    print(f"Checkpoint directory: {destination_dir}")
    print("=" * 40)
    # CSV images already include clean-to-difficult synthetic variants. Avoid
    # adding a second random augmentation layer that validation never sees.
    train_data = OCRDataset(train_records, dataset_format == "jsonl", normalize_inputs, args.max_records)
    val_data = OCRDataset(val_records, False, normalize_inputs, args.max_records)
    if not train_data.records or not val_data.records:
        raise RuntimeError("No valid OCR records. Run scripts/prepare_dataset.py and inspect the report.")
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)
    if recovery:
        model = parseq(pretrained=False, pretrained_backbone=False, vocab=vocabulary, max_length=max_length)
        pretrained_status = "checkpoint resume"
    elif dataset_format == "csv":
        model, transferred_tensors, transferred_parameters = build_parseq_with_compatible_transfer(parseq, vocabulary, max_length)
        pretrained_status = f"docTR visual/decoder transfer ({transferred_tensors} tensors, {transferred_parameters:,} parameters)"
    else:
        model = parseq(pretrained=True, vocab=vocabulary, max_length=max_length)
        pretrained_status = "docTR pretrained PARSeq"
    model = model.to(device)
    print(f"Pretrained model/backbone status: {pretrained_status}")
    if args.evaluate_only:
        model.load_state_dict(recovery[1]["model_state_dict"])
        metrics = run_epoch(
            model, val_loader, device, vocabulary, diagnostic_samples=args.diagnostic_samples
        )
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"Exact Sequence Accuracy: {metrics['exact']:.2%}")
        print(f"Character Accuracy: {metrics['character_accuracy']:.2%}")
        print(f"Character Error Rate: {metrics['cer']:.2%}")
        return 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_exact, best_cer = 1, -1.0, float("inf")
    if recovery:
        checkpoint = recovery[1]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state(optimizer, device)
        scheduler, extended = restore_extended_scheduler(
            optimizer, checkpoint["scheduler_state_dict"], recovery[2], args.epochs
        )
        if extended:
            print(f"Extended cosine schedule to {args.epochs} total epochs; current LR: {scheduler.get_last_lr()[0]:.6g}")
        start_epoch, best_exact, best_cer = recovery[2] + 1, recovery[3], recovery[4]
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    try:
        with training_lock(args.run_name), graceful_training_interrupts():
            for epoch in range(start_epoch, args.epochs + 1):
                print(f"\nEpoch {epoch}/{args.epochs}")
                train_metrics = run_epoch(model, train_loader, device, vocabulary, optimizer, epoch, args.epochs)
                val_metrics = run_epoch(
                    model, val_loader, device, vocabulary, diagnostic_samples=args.diagnostic_samples
                )
                scheduler.step()
                print(f"Epoch {epoch}/{args.epochs} completed")
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Accuracy: {val_metrics['exact']:.2%}")
                print(f"Character Accuracy: {val_metrics['character_accuracy']:.2%}")
                print(f"Character Error Rate: {val_metrics['cer']:.2%}")
                improved = (
                    val_metrics["exact"] > best_exact
                    or (val_metrics["exact"] == best_exact and val_metrics["cer"] < best_cer)
                )
                if improved:
                    best_exact = val_metrics["exact"]
                    best_cer = val_metrics["cer"]
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_exact_accuracy": best_exact,
                    "best_cer": best_cer,
                    "val_metrics": val_metrics,
                    "config": config,
                    "vocab": vocabulary,
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
