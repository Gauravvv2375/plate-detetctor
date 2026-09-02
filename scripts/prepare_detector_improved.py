"""Build a deduplicated, one-class YOLO plate dataset from read-only sources.

This utility performs dataset conversion and validation only. It never loads a
model or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageOps

from scripts._training import runtime_yolo_yaml
from scripts.validate_yolo_labels import inspect_ultralytics_targets, validate_dataset, validate_runtime_yaml
from src.utils import IMAGE_SUFFIXES, PROJECT_ROOT


SPLITS = ("train", "val", "test")
SOURCE_ORDER = {"original": 0, "kaggle1": 1, "kaggle2": 2, "kaggle3": 3}
HOLDOUT_ORDER = {"test": 0, "val": 1, "train": 2}


@dataclass
class Sample:
    source: str
    split: str
    image: Path
    boxes: list[tuple[float, float, float, float]]
    content_hash: str
    original_annotation: str


@dataclass
class SourceAudit:
    path: str
    annotation_format: str
    data_yaml: str | None = None
    declared_class_names: list[str] = field(default_factory=list)
    original_class_ids: set[int] = field(default_factory=set)
    object_names: Counter = field(default_factory=Counter)
    image_formats: Counter = field(default_factory=Counter)
    split_images: Counter = field(default_factory=Counter)
    total_images: int = 0
    total_label_files: int = 0
    plate_annotations: int = 0
    empty_labels: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    jpeg_eoi_warnings: list[str] = field(default_factory=list)
    malformed_annotations: list[str] = field(default_factory=list)
    duplicate_groups: int = 0
    duplicate_images: int = 0
    accepted_before_dedup: int = 0
    contributed: Counter = field(default_factory=Counter)
    duplicates_skipped: int = 0
    conflicts_rejected: int = 0

    def as_dict(self) -> dict:
        result = dict(self.__dict__)
        result["original_class_ids"] = sorted(self.original_class_ids)
        result["object_names"] = dict(self.object_names)
        result["image_formats"] = dict(self.image_formats)
        result["split_images"] = {split: self.split_images[split] for split in SPLITS}
        result["contributed"] = {split: self.contributed[split] for split in SPLITS}
        return result


def decoded_hash(path: Path) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}:RGB:".encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest(), rgb.size


def inspect_image(path: Path, audit: SourceAudit) -> tuple[str, tuple[int, int]] | None:
    if path.suffix.lower() in {".jpg", ".jpeg"} and path.read_bytes()[-2:] != b"\xff\xd9":
        audit.jpeg_eoi_warnings.append(str(path))
    try:
        return decoded_hash(path)
    except Exception as error:
        audit.corrupt_images.append(f"{path}: {type(error).__name__}: {error}")
        return None


def valid_box(values: list[float]) -> str | None:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return "box must contain four finite coordinates"
    cx, cy, width, height = values
    if not 0.0 <= cx <= 1.0 or not 0.0 <= cy <= 1.0:
        return "box center is outside [0,1]"
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        return "box size is outside (0,1]"
    return None


def deterministic_split(content_hash: str) -> str:
    bucket = int(content_hash[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def audit_yolo(source: str, root: Path, split_dirs: dict[str, str], audit: SourceAudit) -> list[Sample]:
    samples: list[Sample] = []
    yaml = root / "data.yaml"
    if yaml.is_file():
        audit.data_yaml = str(yaml)
        import yaml as yaml_module

        document = yaml_module.safe_load(yaml.read_text(encoding="utf-8"))
        names = document.get("names", [])
        audit.declared_class_names = list(names.values()) if isinstance(names, dict) else list(names)
    for split, source_split in split_dirs.items():
        image_dir = root / source_split / "images" if source_split else root / "images" / split
        label_dir = root / source_split / "labels" if source_split else root / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        audit.total_images += len(images)
        audit.total_label_files += len(labels)
        audit.image_formats.update(path.suffix.lower() for path in images)
        audit.split_images[split] += len(images)
        by_stem = {path.stem: path for path in images}
        label_stems = {path.stem for path in labels}
        audit.missing_labels.extend(str(path) for stem, path in by_stem.items() if stem not in label_stems)
        audit.orphan_labels.extend(str(path) for path in labels if path.stem not in by_stem)
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                continue
            inspected = inspect_image(image, audit)
            if inspected is None:
                continue
            content_hash, _ = inspected
            raw = label.read_text(encoding="utf-8", errors="replace")
            if not raw.strip():
                audit.empty_labels += 1
                samples.append(Sample(source, split, image, [], content_hash, str(label)))
                continue
            boxes: list[tuple[float, float, float, float]] = []
            bad = []
            for line_number, line in enumerate(raw.splitlines(), 1):
                fields = line.split()
                try:
                    values = [float(field) for field in fields]
                except ValueError:
                    bad.append(f"{label}:{line_number}: non-numeric label")
                    continue
                if len(values) != 5:
                    bad.append(f"{label}:{line_number}: expected 5 values, found {len(values)}")
                    continue
                class_id = values[0]
                if not math.isfinite(class_id) or not class_id.is_integer():
                    bad.append(f"{label}:{line_number}: invalid class ID {fields[0]}")
                    continue
                audit.original_class_ids.add(int(class_id))
                if int(class_id) != 0:
                    bad.append(f"{label}:{line_number}: unrelated/unknown class ID {int(class_id)}")
                    continue
                problem = valid_box(values[1:])
                if problem:
                    bad.append(f"{label}:{line_number}: {problem}")
                else:
                    boxes.append(tuple(values[1:]))
            if bad:
                audit.malformed_annotations.extend(bad)
                continue
            audit.plate_annotations += len(boxes)
            samples.append(Sample(source, split, image, boxes, content_hash, str(label)))
    audit.accepted_before_dedup = len(samples)
    return samples


def voc_image_for(xml: Path, filename: str, source: str) -> Path:
    if source == "kaggle1":
        root = PROJECT_ROOT / "data/kaggle 1"
        if "number_plate_annos_ocr" in xml.parts:
            return root / "number_plate_images_ocr/number_plate_images_ocr" / filename
        return root / "Indian_Number_Plates/Sample_Images" / filename
    return xml.parent / filename


def audit_voc(source: str, root: Path, audit: SourceAudit) -> list[Sample]:
    samples: list[Sample] = []
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    xmls = sorted(root.rglob("*.xml"))
    audit.total_images = len(images)
    audit.total_label_files = len(xmls)
    referenced_images: set[Path] = set()
    for xml in xmls:
        try:
            tree = ET.parse(xml).getroot()
        except Exception as error:
            audit.malformed_annotations.append(f"{xml}: invalid XML: {error}")
            continue
        filename = (tree.findtext("filename") or "").strip()
        image = voc_image_for(xml, filename, source)
        referenced_images.add(image.resolve())
        if not filename or not image.is_file():
            audit.orphan_labels.append(f"{xml}: referenced image not found: {image}")
            continue
        inspected = inspect_image(image, audit)
        if inspected is None:
            continue
        content_hash, (width, height) = inspected
        boxes: list[tuple[float, float, float, float]] = []
        bad = []
        objects = tree.findall("object")
        if not objects:
            audit.empty_labels += 1
        for index, obj in enumerate(objects, 1):
            object_name = (obj.findtext("name") or "").strip()
            audit.object_names[object_name or "<missing>"] += 1
            if source == "kaggle1" and object_name.lower() not in {"number_plate", "license_plate", "licence_plate", "plate", "vehicle_plate"}:
                bad.append(f"{xml}:object {index}: unrelated class {object_name!r}")
                continue
            box = obj.find("bndbox")
            try:
                xmin = float(box.findtext("xmin"))
                ymin = float(box.findtext("ymin"))
                xmax = float(box.findtext("xmax"))
                ymax = float(box.findtext("ymax"))
            except (AttributeError, TypeError, ValueError) as error:
                bad.append(f"{xml}:object {index}: malformed Pascal VOC box: {error}")
                continue
            raw_values = [xmin, ymin, xmax, ymax]
            if not all(math.isfinite(value) for value in raw_values) or xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
                bad.append(f"{xml}:object {index}: invalid pixel box {raw_values} for {width}x{height}")
                continue
            normalized = ((xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height), (xmax - xmin) / width, (ymax - ymin) / height)
            problem = valid_box(list(normalized))
            if problem:
                bad.append(f"{xml}:object {index}: {problem}")
            else:
                boxes.append(normalized)
        if bad:
            audit.malformed_annotations.extend(bad)
            continue
        audit.plate_annotations += len(boxes)
        split = deterministic_split(content_hash)
        audit.split_images[split] += 1
        samples.append(Sample(source, split, image, boxes, content_hash, str(xml)))
    audit.missing_labels = [str(path) for path in images if path.resolve() not in referenced_images]
    for image in images:
        audit.image_formats[image.suffix.lower()] += 0  # Ensure formats from missing-label images are represented below.
    missing_format_counts = Counter(path.suffix.lower() for path in images)
    audit.image_formats = missing_format_counts
    audit.accepted_before_dedup = len(samples)
    return samples


def record_source_duplicates(samples: list[Sample], audit: SourceAudit) -> None:
    groups = Counter(sample.content_hash for sample in samples)
    audit.duplicate_groups = sum(count > 1 for count in groups.values())
    audit.duplicate_images = sum(count - 1 for count in groups.values() if count > 1)


def canonical_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple:
    return tuple(sorted(tuple(round(value, 7) for value in box) for box in boxes))


def deduplicate(samples: list[Sample], audits: dict[str, SourceAudit]) -> tuple[list[Sample], list[dict], list[dict]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.content_hash].append(sample)
    retained, duplicate_details, conflicts = [], [], []
    for content_hash, group in sorted(groups.items()):
        annotations = {canonical_boxes(sample.boxes) for sample in group}
        if len(annotations) > 1:
            detail = {"hash": content_hash, "samples": [{"source": item.source, "split": item.split, "image": str(item.image), "boxes": item.boxes} for item in group]}
            conflicts.append(detail)
            for item in group:
                audits[item.source].conflicts_rejected += 1
            continue
        chosen = min(group, key=lambda item: (HOLDOUT_ORDER[item.split], SOURCE_ORDER[item.source], str(item.image)))
        retained.append(chosen)
        if len(group) > 1:
            skipped = [item for item in group if item is not chosen]
            duplicate_details.append({"hash": content_hash, "kept": str(chosen.image), "kept_split": chosen.split, "skipped": [str(item.image) for item in skipped]})
            for item in skipped:
                audits[item.source].duplicates_skipped += 1
    return retained, duplicate_details, conflicts


def safe_stem(sample: Sample) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", sample.image.stem).strip("._") or "image"
    return f"{sample.source}_{sample.content_hash[:12]}_{stem}"


def write_dataset(samples: list[Sample], destination: Path, audits: dict[str, SourceAudit]) -> None:
    stage = destination.parent / f".{destination.name}.building-{os.getpid()}"
    if destination.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {destination if destination.exists() else stage}")
    try:
        for split in SPLITS:
            (stage / "images" / split).mkdir(parents=True, exist_ok=True)
            (stage / "labels" / split).mkdir(parents=True, exist_ok=True)
        for sample in sorted(samples, key=lambda item: (item.split, item.source, str(item.image))):
            stem = safe_stem(sample)
            image_destination = stage / "images" / sample.split / f"{stem}{sample.image.suffix.lower()}"
            label_destination = stage / "labels" / sample.split / f"{stem}.txt"
            shutil.copy2(sample.image, image_destination)
            if image_destination.suffix.lower() in {".jpg", ".jpeg"} and image_destination.read_bytes()[-2:] != b"\xff\xd9":
                with image_destination.open("ab") as handle:
                    handle.write(b"\xff\xd9")
            lines = ["0 " + " ".join(f"{value:.10f}" for value in box) for box in sample.boxes]
            label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            audits[sample.source].contributed[sample.split] += 1
        (stage / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: license_plate\n",
            encoding="utf-8",
        )
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def final_image_and_duplicate_check(destination: Path) -> dict:
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    corrupt = []
    annotations = backgrounds = 0
    split_counts = {}
    for split in SPLITS:
        images = sorted(path for path in (destination / "images" / split).glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        split_counts[split] = len(images)
        for image in images:
            try:
                content_hash, _ = decoded_hash(image)
            except Exception as error:
                corrupt.append(f"{image}: {error}")
                continue
            hashes[content_hash].append((split, str(image)))
            label = destination / "labels" / split / f"{image.stem}.txt"
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            annotations += len(lines)
            backgrounds += not lines
    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    leakage = [items for items in duplicate_groups if len({split for split, _ in items}) > 1]
    return {
        "split_images": split_counts,
        "total_images": sum(split_counts.values()),
        "total_plate_annotations": annotations,
        "background_images": backgrounds,
        "corrupt_images": corrupt,
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_duplicate_leakage": leakage,
    }


def write_visual_audit(destination: Path, output: Path) -> None:
    """Render deterministic box overlays for four samples from each new source."""
    cells = []
    for source in ("kaggle1", "kaggle2", "kaggle3"):
        candidates = []
        for split in SPLITS:
            candidates.extend(sorted((destination / "images" / split).glob(f"{source}_*")))
        indices = sorted({0, len(candidates) // 3, (2 * len(candidates)) // 3, len(candidates) - 1}) if candidates else []
        for index in indices:
            image_path = candidates[index]
            split = image_path.parent.name
            label_path = destination / "labels" / split / f"{image_path.stem}.txt"
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((360, 250), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (380, 290), "white")
            x_offset, y_offset = (380 - image.width) // 2, 30 + (250 - image.height) // 2
            canvas.paste(image, (x_offset, y_offset))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{source} / {split}", fill="black")
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                _, cx, cy, width, height = map(float, line.split())
                left = x_offset + (cx - width / 2) * image.width
                top = y_offset + (cy - height / 2) * image.height
                right = x_offset + (cx + width / 2) * image.width
                bottom = y_offset + (cy + height / 2) * image.height
                draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=3)
            cells.append(canvas)
    sheet = Image.new("RGB", (380 * 4, 290 * 3), (225, 225, 225))
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % 4) * 380, (index // 4) * 290))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def markdown_report(report: dict) -> str:
    lines = ["# Improved detector dataset report", "", "No model training was run.", ""]
    for name, audit in report["sources"].items():
        lines.extend([
            f"## {name}", "",
            f"- Path: `{audit['path']}`",
            f"- Format: {audit['annotation_format']}",
            f"- Images / label files: {audit['total_images']} / {audit['total_label_files']}",
            f"- Input/assigned splits (train/val/test): {audit['split_images']['train']} / {audit['split_images']['val']} / {audit['split_images']['test']}",
            f"- Original class IDs: {audit['original_class_ids'] or 'not applicable (Pascal VOC names)' }",
            f"- Declared class names: {audit['declared_class_names'] or 'none'}",
            f"- Empty/background labels: {audit['empty_labels']}",
            f"- Missing / orphan labels: {len(audit['missing_labels'])} / {len(audit['orphan_labels'])}",
            f"- Corrupt images / malformed annotations: {len(audit['corrupt_images'])} / {len(audit['malformed_annotations'])}",
            f"- Readable JPEGs with incomplete EOI markers: {len(audit['jpeg_eoi_warnings'])}",
            f"- Exact duplicate images within source: {audit['duplicate_images']} in {audit['duplicate_groups']} groups",
            f"- Final contribution (train/val/test): {audit['contributed']['train']} / {audit['contributed']['val']} / {audit['contributed']['test']}", "",
        ])
    final = report["final"]
    lines.extend([
        "## Final data/det_improved", "",
        f"- Train / val / test: {final['split_images']['train']} / {final['split_images']['val']} / {final['split_images']['test']}",
        f"- Total images: {final['total_images']}",
        f"- Plate annotations: {final['total_plate_annotations']}",
        f"- Background images: {final['background_images']}",
        f"- Exact duplicate groups found: {report['deduplication']['exact_duplicate_groups_found']}",
        f"- Non-conflicting duplicate copies skipped: {report['deduplication']['duplicates_skipped']}",
        f"- Conflicting duplicate images rejected: {report['deduplication']['conflicting_images_rejected']} in {len(report['deduplication']['conflicting_groups'])} groups",
        f"- Final class: `0: license_plate`", "",
        "## Validation", "",
        f"- Passed: **{report['validation']['passed']}**",
        f"- Label problems: {len(report['validation']['label_report']['problems'])}",
        f"- YAML issues: {len(report['validation']['yaml_report']['issues'])}",
        f"- Corrupt final images: {len(final['corrupt_images'])}",
        f"- Cross-split exact duplicate leakage: {len(final['cross_split_duplicate_leakage'])}",
        f"- Ultralytics parsed targets: {report['validation']['ultralytics_targets']}", "",
        "See the JSON report for exact filenames, rejected annotations, duplicate groups, and original Pascal VOC object names.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a deduplicated one-class plate detector dataset without training.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/det_improved")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/det_improved_dataset_report.json")
    args = parser.parse_args()
    roots = {
        "original": PROJECT_ROOT / "data/det",
        "kaggle1": PROJECT_ROOT / "data/kaggle 1",
        "kaggle2": PROJECT_ROOT / "data/kaggle 2",
        "kaggle3": PROJECT_ROOT / "data/kaggle 3",
    }
    audits = {
        "original": SourceAudit(str(roots["original"]), "YOLO detection", str(roots["original"] / "data.yaml"), ["license_plate"]),
        "kaggle1": SourceAudit(str(roots["kaggle1"]), "Pascal VOC XML (pixel xyxy)"),
        "kaggle2": SourceAudit(str(roots["kaggle2"]), "Pascal VOC XML (pixel xyxy; object names are plate transcriptions)"),
        "kaggle3": SourceAudit(str(roots["kaggle3"]), "YOLO detection", str(roots["kaggle3"] / "data.yaml"), ["number_plate"]),
    }
    samples = []
    samples += audit_yolo("original", roots["original"], {split: "" for split in SPLITS}, audits["original"])
    samples += audit_voc("kaggle1", roots["kaggle1"], audits["kaggle1"])
    samples += audit_voc("kaggle2", roots["kaggle2"], audits["kaggle2"])
    samples += audit_yolo("kaggle3", roots["kaggle3"], {"train": "train", "val": "valid", "test": "test"}, audits["kaggle3"])
    by_source = defaultdict(list)
    for sample in samples:
        by_source[sample.source].append(sample)
    for name, audit in audits.items():
        record_source_duplicates(by_source[name], audit)
    retained, duplicate_details, conflicts = deduplicate(samples, audits)
    write_dataset(retained, args.output.resolve(), audits)
    final = final_image_and_duplicate_check(args.output.resolve())
    runtime_yaml = runtime_yolo_yaml(args.output.resolve())
    label_report = validate_dataset(args.output.resolve())
    yaml_report = validate_runtime_yaml(runtime_yaml, args.output.resolve())
    ultralytics_targets = []
    parser_error = None
    try:
        ultralytics_targets = [inspect_ultralytics_targets(runtime_yaml, split, 16, 640) for split in SPLITS]
    except Exception as error:
        parser_error = f"{type(error).__name__}: {error}"
    # Ultralytics may losslessly repair truncated JPEG endings in the copied
    # dataset while parsing. Re-open and re-hash the final on-disk files after
    # that operation so the final integrity/leakage result reflects reality.
    final = final_image_and_duplicate_check(args.output.resolve())
    passed = not (
        label_report["problems"] or yaml_report["issues"] or final["corrupt_images"]
        or final["exact_duplicate_groups"] or final["cross_split_duplicate_leakage"] or parser_error
    )
    report = {
        "sources": {name: audit.as_dict() for name, audit in audits.items()},
        "normalization": {
            "final_class_id": 0,
            "final_class_name": "license_plate",
            "axis_aligned_yolo": True,
            "readable_source_jpegs_with_eoi_repaired_in_copy": sum(len(audit.jpeg_eoi_warnings) for audit in audits.values()),
        },
        "deduplication": {
            "method": "SHA-256 over decoded RGB dimensions and pixels",
            "exact_duplicate_groups_found": len(duplicate_details) + len(conflicts),
            "exact_duplicate_images_beyond_unique": sum(len(detail["skipped"]) for detail in duplicate_details) + sum(len(group["samples"]) - 1 for group in conflicts),
            "duplicates_skipped": sum(audit.duplicates_skipped for audit in audits.values()),
            "conflicting_images_rejected": sum(len(group["samples"]) for group in conflicts),
            "duplicate_groups": duplicate_details,
            "conflicting_groups": conflicts,
        },
        "final": final,
        "validation": {
            "passed": passed,
            "label_report": label_report,
            "yaml_report": yaml_report,
            "ultralytics_targets": ultralytics_targets,
            "ultralytics_parser_error": parser_error,
        },
        "paths": {
            "dataset": str(args.output.resolve()),
            "data_yaml": str((args.output / "data.yaml").resolve()),
            "visual_audit": str((args.report.parent / "det_improved_visual_audit.jpg").resolve()),
            "validation_report": str((args.report.parent / "det_improved_validation.json").resolve()),
        },
    }
    write_visual_audit(args.output.resolve(), Path(report["paths"]["visual_audit"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.report.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Dataset: {args.output.resolve()}")
    print(f"Report: {args.report.resolve()}")
    print(f"Images: train={final['split_images']['train']} val={final['split_images']['val']} test={final['split_images']['test']} total={final['total_images']}")
    print(f"Plate annotations: {final['total_plate_annotations']}; backgrounds: {final['background_images']}")
    print(f"Duplicates skipped: {report['deduplication']['duplicates_skipped']}; conflicting groups rejected: {len(conflicts)}")
    print(f"Validation passed: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
