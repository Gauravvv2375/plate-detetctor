"""ANPR command line. Input: one image or folder. Processing: detection, row rectification, PARSeq OCR. Output: registration text."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from src.detector import PlateDetection, PlateDetector
from src.preprocess import load_image, mild_ocr_preprocess
from src.recognizer import PARSeqRecognizer
from src.row_detector import RowDetector
from src.utils import IMAGE_SUFFIXES, PROJECT_ROOT, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a vehicle registration number from an image.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--folder", type=Path)
    parser.add_argument("--det-weights", type=Path, default=PROJECT_ROOT / "models/plate_detector/best.pt")
    parser.add_argument("--row-weights", type=Path, default=PROJECT_ROOT / "models/row_detector/best.pt")
    parser.add_argument("--ocr-weights", type=Path, default=PROJECT_ROOT / "models/ocr/best.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or a CUDA index")
    parser.add_argument("--det-imgsz", type=int, default=640, help="Use 1920 for small plates in 1080p dashcam frames")
    parser.add_argument("--det-conf", type=float, default=0.30)
    parser.add_argument("--det-nms-iou", type=float, default=0.6, help="Post-detection duplicate IoU threshold")
    parser.add_argument("--row-conf", type=float, default=0.25, help="Retained for compatibility; row confidence is diagnostic only")
    parser.add_argument("--ocr-conf", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--save-results",
        "--save-debug",
        dest="save_results",
        action="store_true",
        help="Save artifacts for successful plate results only",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


@dataclass
class PlateInference:
    index: int
    text: str
    box: tuple[float, float, float, float]
    detection_confidence: float
    ocr_confidence: float | None = None
    raw_ocr: str = ""
    rows: int = 0
    row_angles: list[float] = field(default_factory=list)
    row_confidence: float = 0.0
    row_detector_succeeded: bool = False
    row_detector_message: str = "NOT RUN"
    fallback_ocr_used: bool = False
    ocr_ran: bool = False
    validation_passed: bool = False
    error: str | None = None
    artifact_warnings: list[str] = field(default_factory=list)


def _box_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _box_intersection(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _box_size(box: tuple[float, float, float, float]) -> tuple[float, float, float]:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width, height, width * height


def _deduplicate_plates(plates: list[PlateDetection], iou_threshold: float = 0.6) -> list[PlateDetection]:
    """Greedy NMS: keep confidence leaders while retaining distinct nearby plates."""
    kept: list[PlateDetection] = []
    for candidate in sorted(plates, key=lambda item: item.confidence, reverse=True):
        if all(_box_iou(candidate.box, accepted.box) < iou_threshold for accepted in kept):
            kept.append(candidate)
    return kept


def _filter_nested_duplicates(plates: list[PlateDetection]) -> list[PlateDetection]:
    """Remove same-plate boxes missed by IoU NMS because one box sits inside another."""
    kept: list[PlateDetection] = []
    for candidate in sorted(plates, key=lambda item: item.confidence, reverse=True):
        candidate_width, candidate_height, candidate_area = _box_size(candidate.box)
        duplicate = False
        for accepted in kept:
            accepted_width, accepted_height, accepted_area = _box_size(accepted.box)
            smaller_area = min(candidate_area, accepted_area)
            if smaller_area <= 0:
                continue
            containment = _box_intersection(candidate.box, accepted.box) / smaller_area
            width_similarity = min(candidate_width, accepted_width) / max(candidate_width, accepted_width)
            height_similarity = min(candidate_height, accepted_height) / max(candidate_height, accepted_height)
            candidate_center = ((candidate.box[0] + candidate.box[2]) / 2, (candidate.box[1] + candidate.box[3]) / 2)
            accepted_center = ((accepted.box[0] + accepted.box[2]) / 2, (accepted.box[1] + accepted.box[3]) / 2)
            center_x_close = abs(candidate_center[0] - accepted_center[0]) <= 0.25 * max(candidate_width, accepted_width)
            center_y_close = abs(candidate_center[1] - accepted_center[1]) <= 0.25 * max(candidate_height, accepted_height)
            if containment >= 0.80 and width_similarity >= 0.45 and height_similarity >= 0.45 and center_x_close and center_y_close:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _reading_order(plates: list[PlateDetection]) -> list[PlateDetection]:
    """Group detections into visual rows, then order each row left-to-right."""
    rows: list[list[PlateDetection]] = []
    by_height = sorted(
        plates,
        key=lambda item: ((item.box[1] + item.box[3]) / 2, (item.box[0] + item.box[2]) / 2),
    )
    for plate in by_height:
        center_y = (plate.box[1] + plate.box[3]) / 2
        height = max(1.0, plate.box[3] - plate.box[1])
        for row in rows:
            row_center = sum((item.box[1] + item.box[3]) / 2 for item in row) / len(row)
            row_height = sum(max(1.0, item.box[3] - item.box[1]) for item in row) / len(row)
            if abs(center_y - row_center) <= 0.6 * max(height, row_height):
                row.append(plate)
                break
        else:
            rows.append([plate])
    rows.sort(key=lambda row: sum((item.box[1] + item.box[3]) / 2 for item in row) / len(row))
    return [
        plate
        for row in rows
        for plate in sorted(row, key=lambda item: ((item.box[0] + item.box[2]) / 2, item.box))
    ]


def _artifact_name(path: Path, index: int) -> str:
    identity = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}_{identity}_plate_{index:03d}.png"


def _normalized_rotation(angle: float) -> float:
    # OBB point order may report either the long edge or its perpendicular short
    # edge, so 0 and +/-90 degrees describe the same plate-row orientation.
    return (angle + 45.0) % 90.0 - 45.0


def _row_rejection(row_result, _confidence_threshold: float) -> str | None:
    if row_result.rows not in (1, 2):
        return "row count is outside the expected range"
    if len(row_result.angles) != row_result.rows:
        return "row geometry is incomplete"
    if any(abs(_normalized_rotation(angle)) > 35.0 for angle in row_result.angles):
        return "row rotation is implausible"
    width, height = row_result.image.size
    if width < 24 or height < 8 or width * height < 300:
        return "rectified row image is too small"
    return None


def _is_valid_indian_registration(text: str) -> bool:
    """Accept common Indian plate variants while rejecting words and short fragments."""
    standard = r"[A-Z]{2}\d{1,2}[A-Z]{1,4}\d{2,4}"
    state_number = r"[A-Z]{2}\d{4,6}"
    temporary = r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{2,4}[A-Z]{1,2}"
    bharat_series = r"\d{2}BH\d{4}[A-Z]{1,2}"
    return re.fullmatch(rf"(?:{standard}|{state_number}|{temporary}|{bharat_series})", text) is not None


def _postprocess_ocr_text(text: str) -> str:
    """Remove an invented state prefix only when followed by a complete BH plate."""
    prefixed_bharat_series = re.fullmatch(r"[A-Z]{2}(\d{2}BH\d{4}[A-Z]{1,2})", text)
    return prefixed_bharat_series.group(1) if prefixed_bharat_series else text


def _ocr_rejection(ocr_result, confidence_threshold: float, text: str | None = None) -> str | None:
    accepted_text = ocr_result.text if text is None else text
    if ocr_result.confidence < confidence_threshold:
        return "OCR confidence is below threshold"
    if not _is_valid_indian_registration(accepted_text):
        return "OCR text does not match an Indian registration format"
    return None


def _save_image(image: Image.Image, path: Path, warnings: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    except Exception as error:
        warnings.append(f"Could not save {path}: {type(error).__name__}: {error}")


def _save_detection(image: Image.Image, plate: PlateDetection, path: Path, warnings: list[str]) -> None:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(plate.box, outline=(255, 0, 0), width=max(2, round(min(image.size) / 300)))
    draw.text((plate.box[0], max(0, plate.box[1] - 14)), f"plate {plate.confidence:.3f}", fill=(255, 0, 0))
    _save_image(annotated, path, warnings)


def infer_all(
    path: Path,
    detector: PlateDetector,
    row_detector: RowDetector,
    recognizer: PARSeqRecognizer,
    ocr_threshold: float,
    output_dir: Path | None = None,
    nms_iou_threshold: float = 0.6,
    row_confidence_threshold: float = 0.25,
) -> list[PlateInference]:
    """Process every detected plate; failure of one plate never aborts the rest."""
    image = load_image(path)
    plates = _reading_order(_filter_nested_duplicates(_deduplicate_plates(detector.detect(image), nms_iou_threshold)))
    results: list[PlateInference] = []
    for index, plate in enumerate(plates, 1):
        result = PlateInference(index, "PLATE_UNREADABLE", plate.box, plate.confidence)
        name = _artifact_name(path, index)

        row_failure = "row path did not complete"
        used_no_obb_crop = False
        try:
            row_result = row_detector.process(plate.crop)
            result.rows = row_result.rows
            result.row_angles = row_result.angles
            result.row_confidence = row_result.confidence
            row_rejection = _row_rejection(row_result, row_confidence_threshold)
            if row_rejection:
                row_failure = row_rejection
                result.row_detector_message = f"FAILED: {row_rejection}\nUsing original crop."
            else:
                no_obb_fallback = row_result.rows == 1 and row_result.angles == [0.0] and row_result.confidence == 0.0
                used_no_obb_crop = no_obb_fallback
                result.row_detector_succeeded = not no_obb_fallback
                result.fallback_ocr_used = no_obb_fallback
                result.row_detector_message = "No OBB found.\nUsing original crop." if no_obb_fallback else "SUCCESS"
                row_ocr_input = mild_ocr_preprocess(row_result.image)
                result.ocr_ran = True
                row_ocr_result = recognizer.recognize(row_ocr_input)
                result.raw_ocr = row_ocr_result.raw_text
                result.ocr_confidence = row_ocr_result.confidence
                row_ocr_text = _postprocess_ocr_text(row_ocr_result.text)
                row_failure = _ocr_rejection(row_ocr_result, ocr_threshold, row_ocr_text) or ""
            if not row_failure:
                result.text = row_ocr_text
                result.validation_passed = True
                if output_dir is not None:
                    _save_detection(image, plate, output_dir / "detections" / name, result.artifact_warnings)
                    _save_image(plate.crop, output_dir / "crops" / name, result.artifact_warnings)
                    _save_image(row_result.image, output_dir / "rectified" / name, result.artifact_warnings)
                    _save_image(row_ocr_input, output_dir / "stitched" / name, result.artifact_warnings)
                results.append(result)
                continue
        except Exception as error:
            row_failure = f"{type(error).__name__}: {error}"
            if result.row_detector_message == "NOT RUN":
                result.row_detector_message = f"FAILED: {row_failure}\nUsing original crop."

        if used_no_obb_crop:
            result.error = f"Row path: no OBB found; fallback path: {row_failure}"
            results.append(result)
            continue

        result.fallback_ocr_used = True
        try:
            fallback_ocr_input = mild_ocr_preprocess(plate.crop)
            result.ocr_ran = True
            fallback_ocr_result = recognizer.recognize(fallback_ocr_input)
            result.raw_ocr = fallback_ocr_result.raw_text
            result.ocr_confidence = fallback_ocr_result.confidence
            fallback_ocr_text = _postprocess_ocr_text(fallback_ocr_result.text)
            fallback_failure = _ocr_rejection(fallback_ocr_result, ocr_threshold, fallback_ocr_text)
            if fallback_failure is None:
                result.text = fallback_ocr_text
                result.error = None
                result.validation_passed = True
                if output_dir is not None:
                    _save_detection(image, plate, output_dir / "detections" / name, result.artifact_warnings)
                    _save_image(plate.crop, output_dir / "crops" / name, result.artifact_warnings)
                    _save_image(fallback_ocr_input, output_dir / "stitched" / name, result.artifact_warnings)
            else:
                result.error = f"Row path: {row_failure}; fallback path: {fallback_failure}"
        except Exception as error:
            fallback_failure = f"{type(error).__name__}: {error}"
            result.error = f"Row path: {row_failure}; fallback path: {fallback_failure}"
        results.append(result)
    return results


def _print_results(path: Path, results: list[PlateInference], verbose: bool, folder: bool = False) -> None:
    if not results:
        if verbose or folder:
            print(f"Image: {path}")
        print("Detected Plates: 0")
        return
    if verbose or folder:
        print(f"Image: {path}")
    print(f"Detected Plates: {len(results)}")
    for result in results:
        print(f"\nPlate {result.index}")
        print(f"Text: {result.text}")
        print(f"Detection Confidence: {result.detection_confidence:.4f}")
        print(f"OCR Confidence: {result.ocr_confidence:.4f}" if result.ocr_confidence is not None else "OCR Confidence: N/A")
        if verbose:
            print(f"Plate bbox: {[round(value, 1) for value in result.box]}")
            print(f"Row angles: {[round(value, 2) for value in result.row_angles]}")
            print(f"Row confidence: {result.row_confidence:.4f}")
            print("Detector: SUCCESS")
            print("Row Detector:")
            print(result.row_detector_message)
            print("OCR:")
            print("RUNNING" if result.ocr_ran else "NOT RUN")
            print("OCR Result:")
            print(result.text if result.validation_passed else result.raw_ocr)
            print("Validation:")
            print("PASSED" if result.validation_passed else "FAILED")
            print(f"Fallback OCR: {'USED' if result.fallback_ocr_used else 'NOT USED'}")


def infer(path: Path, detector: PlateDetector, row_detector: RowDetector, recognizer: PARSeqRecognizer, ocr_threshold: float, verbose: bool) -> str:
    """Backward-compatible single-string API; all plates are still processed."""
    results = infer_all(path, detector, row_detector, recognizer, ocr_threshold)
    if verbose:
        _print_results(path, results, True)
    return results[0].text if results else "NO_PLATE_DETECTED"


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    try:
        detector = PlateDetector(args.det_weights, device, args.det_conf, args.det_imgsz)
        row_detector = RowDetector(args.row_weights, device)
        recognizer = PARSeqRecognizer(args.ocr_weights, device)
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.image:
        if not args.image.is_file():
            print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
            return 2
        output_dir = args.output_dir if args.save_results else None
        results = infer_all(
            args.image,
            detector,
            row_detector,
            recognizer,
            args.ocr_conf,
            output_dir,
            args.det_nms_iou,
            args.row_conf,
        )
        _print_results(args.image, results, args.verbose)
        return 0
    if not args.folder.is_dir():
        print(f"ERROR: Folder not found: {args.folder}", file=sys.stderr)
        return 2
    paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    output_dir = args.output_dir if args.save_results else None
    for path in paths:
        results = infer_all(
            path,
            detector,
            row_detector,
            recognizer,
            args.ocr_conf,
            output_dir,
            args.det_nms_iou,
            args.row_conf,
        )
        _print_results(path, results, args.verbose, folder=True)
        if args.verbose:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
