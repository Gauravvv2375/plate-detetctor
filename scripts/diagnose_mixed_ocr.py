"""Read-only, stratified diagnostics for a mixed-script PARSeq checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from doctr.models import parseq
from fontTools.ttLib import TTCollection, TTFont
from torch.utils.data import DataLoader

from scripts.generate_mixed_ocr_dataset import FONT_AUDIT_SPECS
from scripts.train_ocr import OCRDataset, collate, load_datasets, normalize_ocr_text
from src.utils import PROJECT_ROOT, choose_device


CATEGORIES = (
    "latin_only",
    "devanagari_only",
    "latin_letters_devanagari_digits",
    "devanagari_letters_ascii_digits",
    "mixed_script",
)
SAFE_MIXED_FONTS = {"Arial Unicode", "Devanagari Sangam MN 0", "Devanagari Sangam MN 1"}


def alignment(truth: str, prediction: str) -> list[tuple[str, str]]:
    """Return a deterministic minimum-edit truth/prediction alignment."""
    rows, columns = len(truth) + 1, len(prediction) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        costs[row][0] = row
    for column in range(columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row][column] = min(
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
                costs[row - 1][column - 1] + (truth[row - 1] != prediction[column - 1]),
            )
    pairs: list[tuple[str, str]] = []
    row, column = len(truth), len(prediction)
    while row or column:
        if row and column and costs[row][column] == costs[row - 1][column - 1] + (
            truth[row - 1] != prediction[column - 1]
        ):
            pairs.append((truth[row - 1], prediction[column - 1]))
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            pairs.append((truth[row - 1], "<DEL>"))
            row -= 1
        else:
            pairs.append(("<INS>", prediction[column - 1]))
            column -= 1
    return list(reversed(pairs))


def script_group(character: str) -> str:
    if "A" <= character <= "Z":
        return "latin_letters"
    if "0" <= character <= "9":
        return "ascii_digits"
    if "\u0966" <= character <= "\u096f":
        return "devanagari_digits"
    if "\u0900" <= character <= "\u097f":
        return "devanagari_letters_signs"
    return "separators"


def empty_metrics() -> dict:
    return {"samples": 0, "exact": 0, "truth_characters": 0, "matches": 0, "edits": 0}


def add_result(metrics: dict, truth: str, prediction: str, pairs: list[tuple[str, str]]) -> None:
    metrics["samples"] += 1
    metrics["exact"] += truth == prediction
    metrics["truth_characters"] += len(truth)
    metrics["matches"] += sum(expected == predicted for expected, predicted in pairs if expected != "<INS>")
    metrics["edits"] += sum(expected != predicted for expected, predicted in pairs)


def finish_metrics(metrics: dict) -> dict:
    samples = max(1, metrics["samples"])
    characters = max(1, metrics["truth_characters"])
    return {
        "sample_count": metrics["samples"],
        "exact_accuracy": metrics["exact"] / samples,
        "character_accuracy": metrics["matches"] / characters,
        "cer": metrics["edits"] / characters,
    }


def character_frequencies(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(character for row in rows for character in row["text"])
    return {character: counts[character] for character in sorted(counts, key=ord)}


def distribution(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def font_capability_report(dataset_characters: set[str]) -> dict[str, dict]:
    latin = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
    devanagari = {character for character in dataset_characters if "\u0900" <= character <= "\u097f"}
    report = {}
    for path, index, name in FONT_AUDIT_SPECS:
        if not path.is_file():
            report[name] = {"available": False, "safe_for_mixed_script": False}
            continue
        font = TTCollection(str(path)).fonts[index] if path.suffix.lower() == ".ttc" else TTFont(str(path), fontNumber=index)
        codepoints = {codepoint for table in font["cmap"].tables for codepoint in table.cmap}
        missing_latin = sorted(character for character in latin if ord(character) not in codepoints)
        missing_devanagari = sorted(character for character in devanagari if ord(character) not in codepoints)
        full_latin, full_devanagari = not missing_latin, not missing_devanagari
        report[name] = {
            "available": True, "path": str(path), "face_index": index,
            "full_latin_support": full_latin, "full_devanagari_support": full_devanagari,
            "safe_for_latin": full_latin, "safe_for_devanagari": full_devanagari,
            "safe_for_mixed_script": full_latin and full_devanagari,
            "missing_latin": missing_latin, "missing_devanagari": missing_devanagari,
            "has_gsub": "GSUB" in font, "has_gpos": "GPOS" in font,
            "pillow_fallback": False,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/mixed_ocr_dataset")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/ocr_mixed/best.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/mixed_ocr_diagnosis.json")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir, checkpoint_path = args.data_dir.resolve(), args.checkpoint.resolve()
    with (data_dir / "labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row_by_path = {str((data_dir / row["image"]).resolve()): row for row in rows}
    train_records, val_records, vocab, max_length, _ = load_datasets(data_dir, 0.1, 42)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = choose_device(args.device)
    model = parseq(pretrained=False, pretrained_backbone=False, vocab=vocab, max_length=max_length)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    category_metrics = {category: empty_metrics() for category in CATEGORIES}
    difficulty_metrics = defaultdict(empty_metrics)
    font_metrics = defaultdict(empty_metrics)
    layout_metrics = defaultdict(empty_metrics)
    safe_font_metrics = {
        "overall": empty_metrics(),
        "category": defaultdict(empty_metrics),
        "difficulty": defaultdict(empty_metrics),
        "layout": defaultdict(empty_metrics),
    }
    script_totals = defaultdict(lambda: {"truth_characters": 0, "matches": 0})
    confusions: Counter[tuple[str, str]] = Counter()
    failures: list[dict] = []
    category_losses: dict[str, float] = {}

    for category in CATEGORIES:
        selected = [record for record in val_records if row_by_path[str(record.path)]["category"] == category]
        loader = DataLoader(
            OCRDataset(selected, False, True), batch_size=args.batch, shuffle=False,
            num_workers=args.workers, collate_fn=collate,
        )
        loss_sum = seen = 0
        with torch.inference_mode():
            for images, labels, paths in loader:
                output = model(images.to(device), target=labels, return_preds=True)
                loss_sum += float(output["loss"].detach()) * len(labels)
                seen += len(labels)
                for item, truth, path in zip(output["preds"], labels, paths):
                    raw = str(item[0] if isinstance(item, (tuple, list)) else item)
                    confidence = float(item[1]) if isinstance(item, (tuple, list)) and len(item) > 1 else 0.0
                    prediction = normalize_ocr_text(raw, vocab)
                    pairs = alignment(truth, prediction)
                    metadata = row_by_path[path]
                    add_result(category_metrics[category], truth, prediction, pairs)
                    add_result(difficulty_metrics[metadata["difficulty"]], truth, prediction, pairs)
                    add_result(font_metrics[metadata["font"]], truth, prediction, pairs)
                    add_result(layout_metrics[metadata["layout"]], truth, prediction, pairs)
                    if metadata["font"] in SAFE_MIXED_FONTS:
                        add_result(safe_font_metrics["overall"], truth, prediction, pairs)
                        add_result(safe_font_metrics["category"][category], truth, prediction, pairs)
                        add_result(safe_font_metrics["difficulty"][metadata["difficulty"]], truth, prediction, pairs)
                        add_result(safe_font_metrics["layout"][metadata["layout"]], truth, prediction, pairs)
                    for expected, predicted in pairs:
                        if expected != "<INS>":
                            group = script_group(expected)
                            script_totals[group]["truth_characters"] += 1
                            script_totals[group]["matches"] += expected == predicted
                        if expected != predicted:
                            confusions[(expected, predicted)] += 1
                    if prediction != truth:
                        failures.append({
                            "image": metadata["image"], "category": category,
                            "difficulty": metadata["difficulty"], "font": metadata["font"],
                            "layout": metadata["layout"], "ground_truth": truth,
                            "prediction": prediction, "confidence": confidence,
                            "character_errors": [
                                {"expected": expected, "predicted": predicted}
                                for expected, predicted in pairs if expected != predicted
                            ],
                        })
        category_losses[category] = loss_sum / max(1, seen)

    val_rows = [row_by_path[str(record.path)] for record in val_records]
    train_rows = [row_by_path[str(record.path)] for record in train_records]
    for category in CATEGORIES:
        category_metrics[category] = finish_metrics(category_metrics[category])
        category_metrics[category]["average_loss"] = category_losses[category]
    named = {"images/plate_000045.jpg", "images/plate_000078.jpg"}
    named_failures = [failure for failure in failures if failure["image"] in named]
    other_failures = [failure for failure in failures if failure["image"] not in named]
    selected_failures = named_failures + sorted(other_failures, key=lambda item: item["confidence"], reverse=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "stored_validation_metrics": checkpoint.get("val_metrics"),
        "vocabulary_size": len(vocab),
        "max_length": max_length,
        "input_normalization": checkpoint.get("config", {}).get("input_normalization"),
        "category_metrics": category_metrics,
        "difficulty_metrics": {key: finish_metrics(value) for key, value in difficulty_metrics.items()},
        "font_metrics": {key: finish_metrics(value) for key, value in font_metrics.items()},
        "layout_metrics": {key: finish_metrics(value) for key, value in layout_metrics.items()},
        "safe_mixed_font_metrics": {
            "overall": finish_metrics(safe_font_metrics["overall"]),
            **{
                group: {key: finish_metrics(value) for key, value in grouped.items()}
                for group, grouped in safe_font_metrics.items() if group != "overall"
            },
        },
        "script_character_accuracy": {
            key: {
                **value,
                "accuracy": value["matches"] / max(1, value["truth_characters"]),
            } for key, value in script_totals.items()
        },
        "top_30_confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in confusions.most_common(30)
        ],
        "failed_sample_count": len(failures),
        "failed_examples": selected_failures[:20],
        "character_frequency": {"train": character_frequencies(train_rows), "val": character_frequencies(val_rows)},
        "distribution": {
            split: {
                field: distribution(split_rows, field)
                for field in ("category", "difficulty", "font", "background", "layout")
            } | {"label_length": distribution([row | {"label_length": str(len(row["text"]))} for row in split_rows], "label_length")}
            for split, split_rows in (("train", train_rows), ("val", val_rows))
        },
        "state_frequency": {"train": distribution(train_rows, "state"), "val": distribution(val_rows, "state")},
        "series_frequency": {"train": distribution(train_rows, "series"), "val": distribution(val_rows, "series")},
        "font_capability": font_capability_report(set(vocab)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
