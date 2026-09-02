"""Dataset audit. Input: data directory. Processing: counts, annotation/manifests, image verification and hashes. Output: readable and JSON reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps

from src.utils import IMAGE_SUFFIXES, PROJECT_ROOT, clean_plate_text, image_files, read_jsonl, resolve_manifest_path


def hash_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_labels(dataset: Path, split: str, columns: int) -> dict:
    images = image_files(dataset / "images" / split) if (dataset / "images" / split).exists() else []
    labels = sorted((dataset / "labels" / split).glob("*.txt")) if (dataset / "labels" / split).exists() else []
    image_stems, label_stems = {x.stem for x in images}, {x.stem for x in labels}
    empty, corrupt = [], []
    for label in labels:
        content = label.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            empty.append(str(label.relative_to(PROJECT_ROOT)))
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            fields = line.split()
            try:
                values = [float(x) for x in fields]
            except ValueError:
                values = []
            if len(values) != columns or not values or int(values[0]) != 0 or any(x < 0 or x > 1 for x in values[1:]):
                corrupt.append(f"{label.relative_to(PROJECT_ROOT)}:{line_number}")
    return {
        "images": len(images), "labels": len(labels), "missing_annotations": sorted(image_stems - label_stems),
        "orphan_labels": sorted(label_stems - image_stems), "empty_annotations": empty, "corrupt_annotations": corrupt,
    }


def check_manifest(path: Path) -> dict:
    records = invalid_paths = invalid_text = invalid_json = 0
    sources, rows = Counter(), Counter()
    samples = []
    try:
        iterator = read_jsonl(path)
        for record in iterator:
            records += 1
            stored = str(record.get("path", ""))
            resolved = resolve_manifest_path(stored, path)
            if not resolved.is_file():
                invalid_paths += 1
                if len(samples) < 10:
                    samples.append(stored)
            text = clean_plate_text(str(record.get("text", "")))
            if not text or text != str(record.get("text", "")).upper():
                invalid_text += 1
            sources[str(record.get("src", "unknown"))] += 1
            if "n_rows" in record:
                rows[str(record["n_rows"])] += 1
    except ValueError:
        invalid_json += 1
    return {"records": records, "invalid_paths": invalid_paths, "invalid_text": invalid_text, "invalid_json": invalid_json, "sources": dict(sources), "rows": dict(rows), "missing_samples": samples}


def inspect_images(paths: list[Path]) -> tuple[list[str], dict[str, list[str]]]:
    corrupt, hashes = [], defaultdict(list)
    for index, path in enumerate(paths, 1):
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).verify()
        except Exception as error:
            corrupt.append(f"{path.relative_to(PROJECT_ROOT)}: {type(error).__name__}: {error}")
            continue
        hashes[hash_file(path)].append(str(path.relative_to(PROJECT_ROOT)))
    return corrupt, {key: values for key, values in hashes.items() if len(values) > 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--quick", action="store_true", help="Skip opening and hashing every core image")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/dataset_report.json")
    args = parser.parse_args()
    report = {"detection": {}, "row_detection": {}, "ocr": {}, "issues": {}}
    report["total_image_files_under_data"] = len(image_files(args.data))
    report["total_yolo_label_files"] = sum(1 for path in args.data.rglob("*.txt") if "/labels/" in path.as_posix())
    datasets = [("detection", args.data / "det", 5), ("row_axis", args.data / "rowdet", 5), ("row_obb", args.data / "rowdet_obb", 9), ("row_realtrain", args.data / "rowdet_realtrain", 5), ("row_realval", args.data / "rowdet_realval", 5)]
    core_images: list[Path] = []
    for name, dataset, columns in datasets:
        destination = report["detection"] if name == "detection" else report["row_detection"]
        destination[name] = {}
        for split in ("train", "val", "test"):
            result = check_labels(dataset, split, columns)
            destination[name][split] = result
            core_images.extend(image_files(dataset / "images" / split) if (dataset / "images" / split).exists() else [])
    for dataset_name in ("ocr", "ocr_stitched"):
        report["ocr"][dataset_name] = {}
        for split in ("train", "val", "test"):
            manifest = args.data / dataset_name / f"{split}.jsonl"
            if manifest.is_file():
                result = check_manifest(manifest)
                report["ocr"][dataset_name][split] = result
                for record in read_jsonl(manifest):
                    path = resolve_manifest_path(str(record.get("path", "")), manifest)
                    if path.is_file():
                        core_images.append(path)
    report["available_weights"] = [str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}]
    report["issues"]["foreign_absolute_yaml_paths"] = []
    for yaml in args.data.rglob("*.yaml"):
        for line in yaml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("path:") and line.split(":", 1)[1].strip().startswith("/"):
                report["issues"]["foreign_absolute_yaml_paths"].append(f"{yaml.relative_to(PROJECT_ROOT)}: {line.strip()}")
    report["issues"]["possible_burned_annotation_images"] = [str(path.relative_to(PROJECT_ROOT)) for path in args.data.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and any(word in path.name.lower() for word in ("annotated", "drawn", "visualized"))]
    unique_core = sorted(set(core_images))
    report["core_image_references"] = len(core_images)
    report["unique_core_images"] = len(unique_core)
    if not args.quick:
        corrupt, duplicates = inspect_images(unique_core)
        report["issues"]["corrupt_images"] = corrupt
        report["issues"]["duplicate_content_groups"] = list(duplicates.values())
    else:
        report["issues"]["corrupt_images"] = "not checked (--quick)"
        report["issues"]["duplicate_content_groups"] = "not checked (--quick)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    det = report["detection"]["detection"]
    print("Detection dataset:")
    print(f"  all image files under data/: {report['total_image_files_under_data']}")
    print(f"  all YOLO label files: {report['total_yolo_label_files']}")
    for split in ("train", "val", "test"):
        item = det[split]
        print(f"  {split}: images={item['images']} labels={item['labels']} missing={len(item['missing_annotations'])} empty={len(item['empty_annotations'])} corrupt={len(item['corrupt_annotations'])}")
    print("Row detection:")
    for name, splits in report["row_detection"].items():
        print(f"  {name}: " + ", ".join(f"{split}={values['images']}" for split, values in splits.items() if values["images"]))
    print("OCR dataset:")
    for name, splits in report["ocr"].items():
        print(f"  {name}: " + ", ".join(f"{split}={values['records']} (missing={values['invalid_paths']})" for split, values in splits.items()))
    print(f"Available pretrained weights: {len(report['available_weights'])}")
    print(f"Corrupt images: {len(report['issues']['corrupt_images']) if isinstance(report['issues']['corrupt_images'], list) else report['issues']['corrupt_images']}")
    groups = report["issues"]["duplicate_content_groups"]
    print(f"Duplicate content groups: {len(groups) if isinstance(groups, list) else groups}")
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
