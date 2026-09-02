"""Strict YOLO detection-label preflight.

Input: data/det and its runtime YAML.
Processing: byte-level text checks, numeric/range validation, split pairing,
Ultralytics parser inspection, and optional cache invalidation.
Output: a console/JSON report; source annotations are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._training import runtime_yolo_yaml
from src.utils import IMAGE_SUFFIXES, PROJECT_ROOT


@dataclass
class LabelProblem:
    split: str
    filename: str
    line_number: int | None
    raw_line: str
    problem_type: str


def _problem(split: str, path: Path, line_number: int | None, raw: str, kind: str) -> LabelProblem:
    return LabelProblem(split, path.name, line_number, raw, kind)


def validate_label_file(path: Path, split: str) -> list[LabelProblem]:
    problems: list[LabelProblem] = []
    raw_bytes = path.read_bytes()
    if not raw_bytes.strip():
        return problems  # Intentional background sample.
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        problems.append(_problem(split, path, 1, "<UTF-8 BOM>", "UTF-8 BOM"))
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        return [_problem(split, path, None, repr(raw_bytes[:100]), f"invalid UTF-8: {error}")]
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            if raw_line and any(not char.isspace() for char in raw_line):
                problems.append(_problem(split, path, line_number, repr(raw_line), "hidden characters in blank line"))
            continue
        hidden = [char for char in raw_line if unicodedata.category(char).startswith("C") and char not in "\t"]
        if hidden:
            problems.append(_problem(split, path, line_number, raw_line, f"hidden control character U+{ord(hidden[0]):04X}"))
            continue
        fields = stripped.split()
        if len(fields) != 5:
            problems.append(_problem(split, path, line_number, raw_line, f"expected 5 values, found {len(fields)}"))
            continue
        try:
            values = [float(value) for value in fields]
        except ValueError:
            problems.append(_problem(split, path, line_number, raw_line, "non-numeric value"))
            continue
        if not all(math.isfinite(value) for value in values):
            problems.append(_problem(split, path, line_number, raw_line, "NaN or Inf value"))
            continue
        class_id, cx, cy, width, height = values
        if class_id != 0.0 or not class_id.is_integer():
            kind = "negative class ID" if class_id < 0 else f"class ID must be exactly 0, found {fields[0]}"
            problems.append(_problem(split, path, line_number, raw_line, kind))
        for name, value in (("cx", cx), ("cy", cy)):
            if not 0.0 <= value <= 1.0:
                problems.append(_problem(split, path, line_number, raw_line, f"{name} outside [0,1]"))
        for name, value in (("width", width), ("height", height)):
            if not 0.0 < value <= 1.0:
                problems.append(_problem(split, path, line_number, raw_line, f"{name} outside (0,1]"))
    return problems


def validate_dataset(dataset: Path) -> dict:
    dataset = dataset.resolve()
    report: dict = {"dataset": str(dataset), "splits": {}, "problems": []}
    all_problems: list[LabelProblem] = []
    for split in ("train", "val", "test"):
        image_dir, label_dir = dataset / "images" / split, dataset / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems, label_stems = {path.stem for path in images}, {path.stem for path in labels}
        empty = 0
        for label in labels:
            if not label.read_bytes().strip():
                empty += 1
            all_problems.extend(validate_label_file(label, split))
        # A missing test annotation is reported but does not invalidate train/val.
        missing = sorted(image_stems - label_stems)
        orphan = sorted(label_stems - image_stems)
        report["splits"][split] = {
            "images": len(images), "labels": len(labels), "empty_labels": empty,
            "missing_labels": missing, "orphan_labels": orphan,
        }
        if split in {"train", "val"}:
            for stem in missing:
                all_problems.append(LabelProblem(split, stem, None, "", "missing label file"))
            for stem in orphan:
                all_problems.append(LabelProblem(split, stem, None, "", "orphan label file"))
    report["problems"] = [asdict(problem) for problem in all_problems]
    return report


def validate_runtime_yaml(path: Path, dataset: Path) -> dict:
    import yaml

    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = content.get("names")
    normalized_names = dict(enumerate(names)) if isinstance(names, list) else names
    expected_root = dataset.resolve()
    root = Path(str(content.get("path", ""))).expanduser().resolve()
    issues = []
    if root != expected_root:
        issues.append(f"path resolves to {root}, expected {expected_root}")
    if content.get("train") != "images/train":
        issues.append(f"train must be images/train, found {content.get('train')!r}")
    if content.get("val") != "images/val":
        issues.append(f"val must be images/val, found {content.get('val')!r}")
    if content.get("nc") != 1:
        issues.append(f"nc must be 1, found {content.get('nc')!r}")
    if normalized_names != {0: "license_plate"}:
        issues.append(f"names must be {{0: 'license_plate'}}, found {normalized_names!r}")
    return {"path": str(path), "resolved_root": str(root), "names": normalized_names, "nc": content.get("nc"), "issues": issues}


def clear_ultralytics_caches(dataset: Path) -> list[str]:
    removed = []
    for cache in sorted((dataset / "labels").glob("*.cache")):
        cache.unlink()
        removed.append(str(cache))
    return removed


def inspect_ultralytics_targets(yaml_path: Path, split: str, batch: int, imgsz: int) -> dict:
    """Inspect the exact class arrays produced by Ultralytics' YOLODataset parser."""
    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    data = check_det_dataset(str(yaml_path))
    cfg = get_cfg(DEFAULT_CFG, overrides={"data": str(yaml_path), "imgsz": imgsz, "batch": batch, "task": "detect", "mode": "train"})
    parsed = build_yolo_dataset(cfg, data[split], batch, data, mode="train" if split == "train" else "val", rect=split != "train", stride=32)
    arrays = [label["cls"].reshape(-1) for label in parsed.labels if label["cls"].size]
    classes = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
    return {
        "split": split, "images": len(parsed), "targets": int(classes.size),
        "unique_class_ids": sorted(float(value) for value in np.unique(classes)),
        "finite": bool(np.isfinite(classes).all()), "minimum": float(classes.min()) if classes.size else None,
        "maximum": float(classes.max()) if classes.size else None,
    }


def print_report(report: dict) -> None:
    for split, details in report["splits"].items():
        print(f"{split}: images={details['images']} labels={details['labels']} empty={details['empty_labels']} missing={len(details['missing_labels'])} orphan={len(details['orphan_labels'])}")
    if report["problems"]:
        print("Invalid annotations:")
        for item in report["problems"]:
            print(f"{item['split']}/{item['filename']} | line {item['line_number']} | {item['problem_type']} | {item['raw_line']}")
    else:
        print("Invalid train/val annotations: 0")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly validate YOLO plate labels before training.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/det")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--loader-check", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/yolo_label_validation.json")
    args = parser.parse_args()
    report = validate_dataset(args.data)
    yaml_path = runtime_yolo_yaml(args.data)
    report["runtime_yaml"] = validate_runtime_yaml(yaml_path, args.data)
    if args.clear_cache:
        report["removed_caches"] = clear_ultralytics_caches(args.data)
    if args.loader_check and not report["problems"] and not report["runtime_yaml"]["issues"]:
        report["ultralytics_targets"] = [inspect_ultralytics_targets(yaml_path, split, args.batch, args.imgsz) for split in ("train", "val")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    for parsed in report.get("ultralytics_targets", []):
        print(f"Ultralytics {parsed['split']} targets: unique={set(parsed['unique_class_ids'])} finite={parsed['finite']} count={parsed['targets']}")
    print(f"Report: {args.output}")
    return 1 if report["problems"] or report["runtime_yaml"]["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
