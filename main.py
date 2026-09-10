"""ANPR command line. Input: one image or folder. Processing: detection, row rectification, PARSeq OCR. Output: registration text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.detector import PlateDetection, PlateDetector
from src.preprocess import expanded_crop, load_image, mild_ocr_preprocess
from src.recognizer import OCRResult, PARSeqRecognizer
from src.row_detector import RowDetector
from src.header_ocr import tighten_header, load_header_recognizer
from src.utils import IMAGE_SUFFIXES, PROJECT_ROOT, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a vehicle registration number from an image.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--folder", type=Path)
    parser.add_argument("--det-weights", type=Path, default=PROJECT_ROOT / "models/plate_detector/best.pt")
    parser.add_argument("--row-weights", type=Path, default=PROJECT_ROOT / "models/row_detector/best.pt")
    parser.add_argument("--ocr-weights", type=Path, default=PROJECT_ROOT / "models/ocr/best.pt")
    parser.add_argument(
        "--devanagari-ocr-weights",
        type=Path,
        default=PROJECT_ROOT / "models/ocr_devanagari/best.pt",
        help="Optional second OCR checkpoint for Devanagari routing",
    )
    parser.add_argument(
        "--mixed-ocr-weights",
        type=Path,
        default=PROJECT_ROOT / "models/ocr_mixed_v2/best.pt",
        help="Optional combined Latin + Devanagari PARSeq checkpoint",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or a CUDA index")
    parser.add_argument('--header-ocr-weights', type=Path, default=PROJECT_ROOT / 'models/ocr_header_real_v1/best.pt')
    parser.add_argument("--det-imgsz", type=int, default=640, help="Use 1920 for small plates in 1080p dashcam frames")
    parser.add_argument("--det-conf", type=float, default=0.30)
    parser.add_argument("--det-nms-iou", type=float, default=0.6, help="Post-detection duplicate IoU threshold")
    parser.add_argument("--det-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--det-fallback-imgsz", type=int, default=2560)
    parser.add_argument("--det-fallback-conf", type=float, default=0.05)
    parser.add_argument("--det-tile-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--det-tile-size", type=int, default=1280)
    parser.add_argument("--det-tile-overlap", type=float, default=0.25)
    parser.add_argument("--row-conf", type=float, default=0.25, help="Retained for compatibility; row confidence is diagnostic only")
    parser.add_argument("--ocr-conf", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--save-results", action="store_true", help="Save artifacts for successful plate results only")
    parser.add_argument("--save-debug", action="store_true", help="Save structured diagnostics for every final detection")
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
    detected_rows: int = 0
    selected_rows: list[dict] = field(default_factory=list)
    ignored_rows: list[dict] = field(default_factory=list)
    row_angles: list[float] = field(default_factory=list)
    row_confidence: float = 0.0
    row_detector_succeeded: bool = False
    row_detector_message: str = "NOT RUN"
    fallback_ocr_used: bool = False
    ocr_ran: bool = False
    validation_passed: bool = False
    normalized_text: str = ""
    validation_status: str = "NOT_RUN"
    final_status: str = "UNREADABLE"
    status_reason: str = "OCR did not produce a usable result"
    postprocessing_applied: bool = False
    detection_source: str = "primary"
    accepted: bool = False
    row_direct_disagreement: bool = False
    ocr_strategy: str = "NOT RUN"
    latin_ocr_prediction: str = ""
    devanagari_ocr_prediction: str = ""
    mixed_ocr_prediction: str = ""
    risk_flags: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    artifact_warnings: list[str] = field(default_factory=list)
    debug_crop: Image.Image | None = field(default=None, repr=False)
    debug_row_image: Image.Image | None = field(default=None, repr=False)
    debug_stitched_image: Image.Image | None = field(default=None, repr=False)
    debug_ocr_input: Image.Image | None = field(default=None, repr=False)
    debug_images: dict = field(default_factory=dict, repr=False)
    ocr_calls: list[dict] = field(default_factory=list)
    debug_stage: str = ''
    header_text: str = ''
    registration_text: str = ''
    full_text: str = ''
    text_rows: list[dict] = field(default_factory=list)
    row_result: object = field(default=None, repr=False)

    def as_dict(self):
        return {'plate_number': self.registration_text, 'registration_text': self.registration_text,
                'header_text': self.header_text, 'full_text': self.full_text,
                'header_status': 'REVIEW_REQUIRED' if self.header_text else 'NONE_OR_UNREADABLE',
                'confidence': self.ocr_confidence, 'status': self.final_status,
                'rows': self.text_rows}


@dataclass
class DetectionDiagnostics:
    primary_count: int = 0
    fallback_count: int = 0
    tile_count: int = 0
    final_count: int = 0
    fallback_attempted: bool = False
    tile_attempted: bool = False


@dataclass
class SourcedDetection:
    box: tuple[float, float, float, float]
    confidence: float
    crop: Image.Image
    source: str


@dataclass
class OCREvaluation:
    raw_text: str
    normalized_text: str
    confidence: float
    validation_status: str
    final_status: str
    postprocessing_applied: bool
    reason: str

    @property
    def usable(self) -> bool:
        return bool(self.normalized_text) and self.final_status != "INVALID_FORMAT"

    @property
    def needs_fallback(self) -> bool:
        return self.final_status in {"LOW_CONFIDENCE", "UNCERTAIN", "INVALID_FORMAT"}

    @property
    def saveable(self) -> bool:
        return self.final_status in {"FORMAT_CONFIDENT", "PARTIAL_VISIBLE"}


@dataclass
class OCRRouting:
    evaluation: OCREvaluation
    strategy: str
    latin_prediction: str = ""
    devanagari_prediction: str = ""
    mixed_prediction: str = ""
    visual_groups: int = 0
    row_predictions: list[dict] = field(default_factory=list)


class _DebugRecognizer:
    """Observe actual inference calls only when debug was explicitly requested."""
    def __init__(self, recognizer, name, result, threshold):
        self.recognizer, self.name = recognizer, name
        self.result, self.threshold = result, threshold
        self.input_normalization = getattr(recognizer, 'input_normalization', None)

    def recognize(self, image):
        prediction = self.recognizer.recognize(image)
        is_header = self.result.debug_stage.startswith('header_row_')
        evaluation = _evaluate_header(prediction, self.threshold) if is_header else _evaluate_ocr(prediction, self.threshold)
        groups = 0 if is_header else _estimate_visual_groups(image)
        sequence = len(self.result.ocr_calls) + 1
        prefix = f'ocr_call_{sequence:02d}_{self.name}'
        self.result.debug_images[prefix + '_before_resize.png'] = image.copy()
        self.result.debug_images[prefix + '_final.png'] = PARSeqRecognizer.input_image(image)
        text = prediction.text
        latin = any('A' <= c <= 'Z' or '0' <= c <= '9' for c in text)
        dev = _contains_devanagari(text)
        score = _routing_score(evaluation, groups)
        if self.name == 'mixed' and latin and dev:
            score += .08
        self.result.ocr_calls.append({
            'model': self.name, 'input': prefix, 'size_before_resize': list(image.size),
            'stage': self.result.debug_stage,
            'checkpoint': getattr(self.recognizer, 'checkpoint_path', None),
            'epoch': getattr(self.recognizer, 'checkpoint_epoch', None),
            'vocabulary': getattr(self.recognizer, 'vocabulary', None),
            'max_length': getattr(self.recognizer, 'max_length', None),
            'normalization': getattr(self.recognizer, 'input_normalization', None),
            'mean': [0.694, 0.695, 0.693] if self.input_normalization == 'doctr_parseq' else None,
            'std': [0.299, 0.296, 0.301] if self.input_normalization == 'doctr_parseq' else None,
            'resize': [128, 32], 'interpolation': 'bilinear', 'mode': 'RGB',
            'raw_prediction': prediction.raw_text, 'prediction': text,
            'confidence': prediction.confidence, 'routing_score': score,
            'visual_groups': groups, 'validation': evaluation.validation_status,
            'script': 'mixed' if latin and dev else 'Devanagari' if dev else 'Latin',
        })
        return prediction


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


def _sourced(detections: list[PlateDetection], source: str) -> list[SourcedDetection]:
    return [SourcedDetection(item.box, item.confidence, item.crop, source) for item in detections]


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size <= 0:
        raise ValueError("detector tile size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("detector tile overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _detect_tiles(
    detector: PlateDetector,
    image: Image.Image,
    tile_size: int,
    overlap: float,
    confidence: float,
    image_size: int,
) -> list[SourcedDetection]:
    detections: list[SourcedDetection] = []
    margin = float(getattr(detector, "crop_margin", 0.08))
    for top in _tile_starts(image.height, tile_size, overlap):
        for left in _tile_starts(image.width, tile_size, overlap):
            right, bottom = min(image.width, left + tile_size), min(image.height, top + tile_size)
            tile = image.crop((left, top, right, bottom))
            for plate in detector.detect(tile, confidence=confidence, image_size=image_size):
                box = (
                    max(0.0, plate.box[0] + left),
                    max(0.0, plate.box[1] + top),
                    min(float(image.width), plate.box[2] + left),
                    min(float(image.height), plate.box[3] + top),
                )
                width, height, _ = _box_size(box)
                if width < 2 or height < 2:
                    continue
                detections.append(SourcedDetection(box, plate.confidence, expanded_crop(image, box, margin), "tile"))
    return detections


def _needs_dense_tile_recovery(
    detections: list[SourcedDetection], image: Image.Image
) -> bool:
    """Use enhanced detection only for scenes containing many small plates."""
    if len(detections) < 4 or image.width <= 0 or image.height <= 0:
        return False
    image_area = image.width * image.height
    relative_areas = [_box_size(item.box)[2] / image_area for item in detections]
    return float(np.median(relative_areas)) <= 0.03


def _detect_for_inference(
    detector: PlateDetector,
    image: Image.Image,
    nms_iou_threshold: float,
    diagnostics: DetectionDiagnostics,
    fallback_enabled: bool,
    fallback_image_size: int,
    fallback_confidence: float,
    tile_enabled: bool,
    tile_size: int,
    tile_overlap: float,
) -> list[SourcedDetection]:
    primary = _sourced(detector.detect(image), "primary")
    diagnostics.primary_count = len(primary)
    candidates = primary
    dense_tile_recovery = tile_enabled and _needs_dense_tile_recovery(primary, image)
    if not primary and fallback_enabled:
        diagnostics.fallback_attempted = True
        fallback = _sourced(
            detector.detect(image, confidence=fallback_confidence, image_size=fallback_image_size),
            "fallback",
        )
        diagnostics.fallback_count = len(fallback)
        candidates = fallback
    if tile_enabled and (not primary or dense_tile_recovery):
        diagnostics.tile_attempted = True
        tiles = _detect_tiles(
            detector,
            image,
            tile_size,
            tile_overlap,
            fallback_confidence,
            tile_size,
        )
        diagnostics.tile_count = len(tiles)
        candidates = [*candidates, *tiles]
    final = _reading_order(_filter_nested_duplicates(_deduplicate_plates(candidates, nms_iou_threshold)))
    diagnostics.final_count = len(final)
    return final


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
    """Accept common Latin formats and plausible mixed-script registrations."""
    compact = re.sub(r"[ -]+", "", unicodedata.normalize("NFC", text).strip())
    standard = r"[A-Z]{2}\d{1,2}[A-Z]{1,4}\d{2,4}"
    state_number = r"[A-Z]{2}\d{4,6}"
    temporary = r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{2,4}[A-Z]{1,2}"
    bharat_series = r"\d{2}BH\d{4}[A-Z]{1,2}"
    if re.fullmatch(rf"(?:{standard}|{state_number}|{temporary}|{bharat_series})", compact):
        return True
    if not _has_only_supported_registration_characters(text):
        return False
    units = _registration_units(compact)
    return 4 <= len(units) <= 24 and any(item.isdigit() for item in units) and any(item.isalpha() for item in units)


def _registration_units(text: str) -> list[str]:
    """Count visible base letters/digits without counting Devanagari combining marks."""
    return [
        character
        for character in unicodedata.normalize("NFC", text)
        if (character.isalpha() or character.isdigit()) and not unicodedata.category(character).startswith("M")
    ]


def _has_only_supported_registration_characters(text: str) -> bool:
    for character in unicodedata.normalize("NFC", text):
        if character in " -":
            continue
        if "A" <= character <= "Z" or "0" <= character <= "9" or "\u0900" <= character <= "\u097f":
            continue
        return False
    return True


def _contains_devanagari(text: str) -> bool:
    return any("\u0900" <= character <= "\u097f" for character in text)


def _postprocess_ocr_text(text: str) -> str:
    """Remove an invented state prefix only when followed by a complete BH plate."""
    prefixed_bharat_series = re.fullmatch(r"[A-Z]{2}(\d{2}BH\d{4}[A-Z]{1,2})", text)
    return prefixed_bharat_series.group(1) if prefixed_bharat_series else text


def _evaluate_ocr(ocr_result, confidence_threshold: float) -> OCREvaluation:
    visible = unicodedata.normalize("NFC", ocr_result.text).strip()
    normalized = _postprocess_ocr_text(visible)
    compact = re.sub(r"[ -]+", "", normalized)
    exact_standard = re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}", compact) is not None
    exact_bharat = re.fullmatch(r"\d{2}BH\d{4}[A-Z]{1,2}", compact) is not None
    units = _registration_units(compact)
    supported = _has_only_supported_registration_characters(normalized)
    has_letter = any(character.isalpha() for character in units)
    has_digit = any(character.isdigit() for character in units)
    plausible_visible = supported and 4 <= len(units) <= 24 and has_digit and (has_letter or len(units) <= 6)
    mixed_script = _contains_devanagari(normalized)
    if not normalized:
        validation_status, final_status = "INVALID_FORMAT", "INVALID_FORMAT"
        reason = "OCR returned empty text"
    elif ocr_result.confidence < confidence_threshold:
        validation_status = "VALID_FORMAT" if exact_standard or exact_bharat else "POTENTIAL_FORMAT" if plausible_visible else "INVALID_FORMAT"
        final_status = "LOW_CONFIDENCE" if plausible_visible or exact_standard or exact_bharat else "INVALID_FORMAT"
        reason = f"OCR confidence {ocr_result.confidence:.4f} is below {confidence_threshold:.4f}"
    elif exact_standard or exact_bharat:
        validation_status, final_status = "VALID_FORMAT", "FORMAT_CONFIDENT"
        reason = "format and confidence checks passed; character correctness is not guaranteed"
    elif plausible_visible:
        validation_status = "POTENTIAL_FORMAT" if mixed_script else "INVALID_FORMAT"
        final_status = "UNCERTAIN" if mixed_script else "PARTIAL_VISIBLE" if _is_valid_indian_registration(normalized) else "UNCERTAIN"
        reason = (
            "mixed-script text was preserved exactly and requires review"
            if mixed_script
            else "visible alphanumeric text was preserved without completing or guessing characters"
        )
    else:
        validation_status, final_status = "INVALID_FORMAT", "INVALID_FORMAT"
        reason = "OCR text is not a plausible registration string"
    return OCREvaluation(
        raw_text=ocr_result.raw_text,
        normalized_text=normalized,
        confidence=ocr_result.confidence,
        validation_status=validation_status,
        final_status=final_status,
        postprocessing_applied=normalized != ocr_result.raw_text,
        reason=reason,
    )


def _estimate_visual_groups(image: Image.Image) -> int:
    """Estimate character groups for routing only; never use it to invent text."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.size == 0 or gray.shape[1] < 4:
        return 0
    threshold = float(np.percentile(gray, 35))
    dark = gray <= min(210.0, threshold + 12.0)
    active = dark.mean(axis=0) >= 0.08
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= 2:
                runs.append((start, index))
            start = None
    if start is not None and len(active) - start >= 2:
        runs.append((start, len(active)))
    width = gray.shape[1]
    runs = [
        run for run in runs
        if not ((run[0] <= 1 or run[1] >= width - 1) and run[1] - run[0] < max(3, round(width * 0.04)))
    ]
    return len(runs)


def _routing_score(evaluation: OCREvaluation, visual_groups: int) -> float:
    if not evaluation.usable:
        return -10.0
    score = evaluation.confidence
    if visual_groups:
        score -= 0.07 * abs(len(_registration_units(evaluation.normalized_text)) - visual_groups)
    return score


def _evaluate_header(prediction, threshold):
    text = unicodedata.normalize('NFC', prediction.text).strip()
    meaningful = (_has_only_supported_registration_characters(text)
                  and 2 <= len(_registration_units(text)) <= 64
                  and prediction.confidence >= threshold)
    return OCREvaluation(prediction.raw_text, text, prediction.confidence,
        'NOT_APPLICABLE', 'UNCERTAIN' if meaningful else 'INVALID_FORMAT', False,
        'header candidate requires review' if meaningful else 'header below confidence/text threshold')


def _route_ocr(
    image: Image.Image,
    latin_recognizer: PARSeqRecognizer,
    devanagari_recognizer: PARSeqRecognizer | None,
    confidence_threshold: float,
    mixed_recognizer: PARSeqRecognizer | None = None,
    raw_image: Image.Image | None = None,
    header: bool = False,
) -> OCRRouting:
    def recognize(model):
        # CSV-trained checkpoints saw RGB pixels without the legacy contrast /
        # unsharp pass. Keep that older pass only for legacy OCR checkpoints.
        source = raw_image if raw_image is not None and getattr(model, 'input_normalization', None) == 'doctr_parseq' else image
        prediction = model.recognize(source)
        if not header:
            return _evaluate_ocr(prediction, confidence_threshold)
        return _evaluate_header(prediction, confidence_threshold)

    latin = recognize(latin_recognizer)
    visual_groups = 0 if header else _estimate_visual_groups(image)
    if devanagari_recognizer is None or devanagari_recognizer is latin_recognizer:
        if mixed_recognizer is None:
            return OCRRouting(latin, "single OCR model", latin.normalized_text, "", "", visual_groups)
        mixed = recognize(mixed_recognizer)
        mixed_score = _routing_score(mixed, visual_groups)
        latin_score = _routing_score(latin, visual_groups)
        chosen = mixed if mixed.usable and mixed_score >= latin_score - 0.03 else latin
        selected = "mixed" if chosen is mixed else "Latin"
        return OCRRouting(
            chosen,
            f"mixed + Latin OCR; selected {selected} (Latin score={latin_score:.3f}, mixed score={mixed_score:.3f})",
            latin.normalized_text,
            "",
            mixed.normalized_text,
            visual_groups,
        )

    devanagari = recognize(devanagari_recognizer)
    latin_score = _routing_score(latin, visual_groups)
    devanagari_score = _routing_score(devanagari, visual_groups)
    latin_length_gap = abs(len(_registration_units(latin.normalized_text)) - visual_groups) if visual_groups else 0
    devanagari_margin = 0.10 if latin.validation_status == "VALID_FORMAT" and latin_length_gap <= 2 else 0.03
    if not latin.usable and devanagari.usable:
        chosen, selected = devanagari, "Devanagari"
    elif devanagari.usable and devanagari_score > latin_score + devanagari_margin:
        chosen, selected = devanagari, "Devanagari"
    else:
        chosen, selected = latin, "Latin"

    if mixed_recognizer is not None:
        mixed = recognize(mixed_recognizer)
        mixed_score = _routing_score(mixed, visual_groups)
        visible_mixed = _contains_devanagari(mixed.normalized_text) and any("A" <= item <= "Z" or "0" <= item <= "9" for item in mixed.normalized_text)
        if visible_mixed:
            mixed_score += 0.08
        selected_score = devanagari_score if selected == "Devanagari" else latin_score
        if mixed.usable and mixed_score >= selected_score - 0.03:
            chosen, selected, selected_score = mixed, "mixed", mixed_score
        strategy = (
            f"three-model OCR; selected {selected} by confidence, script compatibility, and visual-length agreement "
            f"(visual groups={visual_groups}, Latin score={latin_score:.3f}, "
            f"Devanagari score={devanagari_score:.3f}, mixed score={mixed_score:.3f})"
        )
        alternatives = [
            score for evaluation, score in ((latin, latin_score), (devanagari, devanagari_score), (mixed, mixed_score))
            if evaluation.usable and evaluation.normalized_text != chosen.normalized_text
        ]
        if alternatives and selected_score - max(alternatives) <= 0.03:
            chosen = replace(
                chosen,
                final_status="UNCERTAIN",
                reason="OCR models disagree with similar routing scores; manual review is required",
            )
        return OCRRouting(
            chosen,
            strategy,
            latin.normalized_text,
            devanagari.normalized_text,
            mixed.normalized_text,
            visual_groups,
        )

    disagreement = latin.usable and devanagari.usable and latin.normalized_text != devanagari.normalized_text
    if disagreement and abs(latin_score - devanagari_score) <= 0.03:
        chosen = replace(
            chosen,
            validation_status="POTENTIAL_FORMAT" if _contains_devanagari(chosen.normalized_text) else chosen.validation_status,
            final_status="UNCERTAIN",
            reason="Latin and Devanagari OCR disagree with similar routing scores; manual review is required",
        )
    strategy = (
        f"dual OCR; selected {selected} by confidence and visual-length agreement "
        f"(visual groups={visual_groups}, Latin score={latin_score:.3f}, Devanagari score={devanagari_score:.3f})"
    )
    return OCRRouting(chosen, strategy, latin.normalized_text, devanagari.normalized_text, "", visual_groups)


def _apply_routing_diagnostics(result: PlateInference, routing: OCRRouting) -> None:
    result.ocr_strategy = routing.strategy
    result.latin_ocr_prediction = routing.latin_prediction
    result.devanagari_ocr_prediction = routing.devanagari_prediction
    result.mixed_ocr_prediction = routing.mixed_prediction


def _route_row_result(
    row_result,
    stitched_input: Image.Image,
    latin_recognizer: PARSeqRecognizer,
    devanagari_recognizer: PARSeqRecognizer | None,
    confidence_threshold: float,
    mixed_recognizer: PARSeqRecognizer | None = None,
) -> OCRRouting:
    """Route separate rows independently when geometry supplies two real rows."""
    if len(row_result.row_images) != 2:
        return _route_ocr(stitched_input, latin_recognizer, devanagari_recognizer, confidence_threshold, mixed_recognizer, row_result.image)
    routes = [
        _route_ocr(mild_ocr_preprocess(row), latin_recognizer, devanagari_recognizer, confidence_threshold, mixed_recognizer, row)
        for row in row_result.row_images
    ]
    if not all(route.evaluation.usable for route in routes):
        return _route_ocr(stitched_input, latin_recognizer, devanagari_recognizer, confidence_threshold, mixed_recognizer, row_result.image)
    text = " ".join(route.evaluation.normalized_text for route in routes)
    confidence = min(route.evaluation.confidence for route in routes)
    evaluation = _evaluate_ocr(OCRResult(text, text, confidence), confidence_threshold)
    return OCRRouting(
        evaluation=evaluation,
        strategy="per-row OCR routing; " + "; ".join(f"row {index}: {route.strategy}" for index, route in enumerate(routes, 1)),
        latin_prediction=" | ".join(route.latin_prediction for route in routes),
        devanagari_prediction=" | ".join(route.devanagari_prediction for route in routes),
        mixed_prediction=" | ".join(route.mixed_prediction for route in routes),
        visual_groups=sum(route.visual_groups for route in routes),
        row_predictions=[{'text': route.evaluation.normalized_text, 'confidence': route.evaluation.confidence,
                          'strategy': route.strategy} for route in routes],
    )


def _reading_order_text(rows):
    angles = [((row.get('angle', 0) + 45) % 90 - 45) for row in rows]
    theta = np.radians(float(np.median(angles))) if angles else 0
    def position(row):
        x, y = row['center']
        return x * np.cos(theta) + y * np.sin(theta), -x * np.sin(theta) + y * np.cos(theta)
    lines = []
    for row in sorted(rows, key=lambda row: position(row)[1]):
        for line in lines:
            if abs(position(row)[1] - position(line[0])[1]) <= .4 * min(row['size'][1], line[0]['size'][1]):
                line.append(row)
                break
        else:
            lines.append([row])
    return [row for line in lines for row in sorted(line, key=lambda row: position(row)[0])]


def _complete_plate_text(result, recognizers, threshold, debug, header_recognizer=None):
    """Add independent auxiliary text after registration acceptance is finalized."""
    result.registration_text = result.text if result.text != 'PLATE_UNREADABLE' else ''
    row_result = result.row_result
    classified = getattr(row_result, 'classified_rows', [])
    headers = []
    models = tuple(_DebugRecognizer(model, name, result, threshold) if debug and model else model
                   for model, name in zip(recognizers, ('latin', 'devanagari', 'mixed')))
    for metadata in classified:
        entry = dict(metadata)
        if metadata['role'] == 'HEADER':
            try:
                raw = row_result.region_images[metadata['index']]
                original = raw
                raw, crop_info = tighten_header(raw)
                entry['crop_refinement'] = crop_info
                result.debug_stage = f"header_row_{metadata['index']}"
                if debug:
                    result.debug_images[result.debug_stage + '_original.png'] = original
                    result.debug_images[result.debug_stage + '_tight.png'] = raw
                if header_recognizer is not None:
                    model = _DebugRecognizer(header_recognizer, 'header', result, threshold) if debug else header_recognizer
                    prediction = model.recognize(raw)
                    route = OCRRouting(_evaluate_header(prediction, threshold), 'dedicated real-header OCR')
                else:
                    route = _route_ocr(mild_ocr_preprocess(raw), models[0], models[1], threshold, models[2], raw, header=True)
                entry.update(text=route.evaluation.normalized_text if route.evaluation.usable else '',
                    confidence_ocr=route.evaluation.confidence, status='REVIEW_REQUIRED' if route.evaluation.usable else 'UNREADABLE',
                    latin_prediction=route.latin_prediction, devanagari_prediction=route.devanagari_prediction,
                    mixed_prediction=route.mixed_prediction, strategy=route.strategy)
                if entry['text']:
                    headers.append(entry)
            except Exception as error:
                entry.update(text='', status='UNREADABLE', error=str(error))
        result.text_rows.append(entry)
    headers = _reading_order_text(headers)
    result.header_text = ' '.join(row['text'] for row in headers)
    # Registration remains one independently reconciled value, even when it
    # spans two rows. Auxiliary text can never alter that value or its status.
    selected = getattr(row_result, 'selected_rows', [])
    registration = {'text': result.registration_text,
                    'angle': selected[0].get('angle', 0) if selected else 0,
                    'center': selected[0].get('center', [0, 0]) if selected else [0, 0],
                    'size': selected[0].get('size', [1, 1]) if selected else [1, 1]}
    result.full_text = ' '.join(row['text'] for row in _reading_order_text(headers + [registration]) if row['text'])


def _apply_ocr_evaluation(result: PlateInference, evaluation: OCREvaluation) -> None:
    result.raw_ocr = evaluation.raw_text
    result.normalized_text = evaluation.normalized_text
    result.ocr_confidence = evaluation.confidence
    result.validation_status = evaluation.validation_status
    result.final_status = evaluation.final_status
    result.status_reason = evaluation.reason
    result.validation_passed = evaluation.final_status == "FORMAT_CONFIDENT"
    result.postprocessing_applied = evaluation.postprocessing_applied
    if evaluation.usable:
        result.text = evaluation.normalized_text


def _reconcile_ocr(primary: OCREvaluation | None, direct: OCREvaluation, header_filtered: bool = False) -> OCREvaluation:
    """Choose between two OCR views without ever adding unseen characters."""
    if primary is None or not primary.usable:
        return direct
    if not direct.usable:
        return primary
    if primary.normalized_text == direct.normalized_text:
        return primary if primary.confidence >= direct.confidence else direct
    if header_filtered:
        return replace(primary, final_status='UNCERTAIN',
            reason='registration-row and full-crop OCR disagree; retained the row because the full crop includes an ignored header')

    first, second = primary.normalized_text, direct.normalized_text
    if first.startswith(second) or second.startswith(first):
        shorter = primary if len(first) < len(second) else direct
        longer = direct if shorter is primary else primary
        if shorter.confidence >= longer.confidence:
            return replace(
                shorter,
                validation_status="INVALID_FORMAT",
                final_status="PARTIAL_VISIBLE",
                reason="the higher-confidence OCR view is a strict prefix; no trailing character was invented",
            )
        return longer

    common_length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        common_length += 1
    if common_length >= 7 and len(first) - common_length <= 1 and len(second) - common_length <= 1:
        source = primary if primary.confidence >= direct.confidence else direct
        return replace(
            source,
            normalized_text=first[:common_length],
            validation_status="INVALID_FORMAT",
            final_status="PARTIAL_VISIBLE",
            postprocessing_applied=True,
            reason="OCR views disagree only at the trailing character; only their visible consensus was retained",
        )

    if primary.final_status == "FORMAT_CONFIDENT" and direct.final_status != "FORMAT_CONFIDENT":
        chosen = primary
    elif direct.final_status == "FORMAT_CONFIDENT" and primary.final_status != "FORMAT_CONFIDENT":
        chosen = direct
    else:
        chosen = primary if primary.confidence >= direct.confidence else direct
    return replace(
        chosen,
        validation_status="VALID_FORMAT" if chosen.validation_status == "VALID_FORMAT" else chosen.validation_status,
        final_status="UNCERTAIN",
        reason="row and direct-crop OCR disagree; format validation cannot establish character correctness",
    )


def _night_or_glare_candidate(crop: Image.Image) -> bool:
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    if rgb.size == 0:
        return False
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    yellow = (red >= 110) & (green >= 80) & (blue <= 0.72 * np.minimum(red, green))
    shadow = np.maximum.reduce((red, green, blue)) <= 120
    red_cast = (red >= 1.20 * green) & (red >= 1.50 * blue)
    return bool(
        float(yellow.mean()) >= 0.15
        and (float(shadow.mean()) >= 0.10 or float(red_cast.mean()) >= 0.20)
    )


def _finalize_acceptance(result: PlateInference) -> None:
    confidence = result.ocr_confidence or 0.0
    two_line = result.rows >= 2
    night_or_glare = _night_or_glare_candidate(result.debug_crop) if result.debug_crop is not None else False
    ambiguous_characters = set("IOBRNM108")
    possible_confusion = bool(set(result.normalized_text) & ambiguous_characters) and (
        confidence < 0.98 or result.row_direct_disagreement or night_or_glare
    )
    result.risk_flags = {
        "low_ocr_confidence": confidence < 0.98,
        "row_direct_disagreement": result.row_direct_disagreement,
        "two_line_plate": two_line,
        "possible_character_confusion": possible_confusion,
        "night_or_glare_candidate": night_or_glare,
    }
    format_valid = result.validation_status == "VALID_FORMAT"
    result.validation_passed = format_valid
    if result.text == "PLATE_UNREADABLE":
        result.final_status = "REJECTED"
        result.status_reason = result.status_reason or "OCR did not produce usable registration text"
    elif result.final_status == 'UNCERTAIN' and result.row_direct_disagreement:
        result.final_status = 'REVIEW_REQUIRED'
    elif confidence < 0.85:
        result.final_status = "LOW_CONFIDENCE"
        result.status_reason = f"OCR confidence {confidence:.4f} is below the review threshold 0.8500"
    elif result.final_status == "PARTIAL_VISIBLE":
        result.final_status = "PARTIAL_VISIBLE"
        result.status_reason = "visible partial text preserved without completing or guessing characters"
    elif format_valid:
        blocking_risk = result.final_status == 'UNCERTAIN' or confidence < 0.98 or result.row_direct_disagreement or night_or_glare or possible_confusion
        if two_line and confidence < 0.995:
            blocking_risk = True
        if blocking_risk:
            result.final_status = "REVIEW_REQUIRED"
            if two_line and night_or_glare:
                result.status_reason = "format valid but OCR uncertain / possible character confusion / two-line night truck plate"
            else:
                reasons = []
                if confidence < 0.98:
                    reasons.append("OCR confidence is in the review range")
                if result.row_direct_disagreement:
                    reasons.append("row and direct-crop OCR disagree")
                if two_line:
                    reasons.append("multi-row stitching was used")
                if possible_confusion:
                    reasons.append("possible visually confused characters")
                if night_or_glare:
                    reasons.append("yellow plate with night/glare characteristics")
                result.status_reason = "; ".join(reasons)
        else:
            result.final_status = "ACCEPTED"
            result.accepted = True
            result.status_reason = "very high OCR confidence, valid format, and no conflicting OCR evidence"
    elif result.validation_status == "POTENTIAL_FORMAT":
        result.final_status = "REVIEW_REQUIRED"
        result.status_reason = "mixed-script or Devanagari registration text is plausible but requires manual review"
    elif result.validation_status == "INVALID_FORMAT":
        result.final_status = "REVIEW_REQUIRED"
        result.status_reason = "visible OCR text has an invalid format and requires manual review"


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


def _save_debug_bundle(
    path: Path,
    image: Image.Image,
    results: list[PlateInference],
    diagnostics: DetectionDiagnostics,
    output_dir: Path,
) -> None:
    debug_dir = output_dir / "debug" / path.stem
    warnings: list[str] = []
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for result in results:
        draw.rectangle(result.box, outline=(255, 0, 0), width=max(2, round(min(image.size) / 300)))
        draw.text((result.box[0], max(0, result.box[1] - 14)), f"{result.index}: {result.detection_confidence:.3f}", fill=(255, 0, 0))
    _save_image(annotated, debug_dir / "final_detections.png", warnings)
    _save_image(image, debug_dir / "01_original.png", warnings)
    records = []
    for result in results:
        prefix = f"plate_{result.index:03d}"
        matching_calls = [call for call in result.ocr_calls if not call.get('stage', '').startswith('header_row_') and call['prediction'].strip() == result.normalized_text]
        if matching_calls:
            selected_input = matching_calls[0]['input']
            result.debug_ocr_input = result.debug_images[selected_input + '_before_resize.png']
            result.debug_images['10_ocr_input_before_resize.png'] = result.debug_ocr_input
            result.debug_images['11_ocr_input_final.png'] = result.debug_images[selected_input + '_final.png']
        for filename, debug_image in result.debug_images.items():
            _save_image(debug_image, debug_dir / prefix / filename, warnings)
        if result.debug_crop is not None:
            _save_image(result.debug_crop, debug_dir / f"{prefix}_detector_crop.png", warnings)
        if result.debug_row_image is not None:
            _save_image(result.debug_row_image, debug_dir / f"{prefix}_row_output.png", warnings)
        if result.debug_stitched_image is not None:
            _save_image(result.debug_stitched_image, debug_dir / f"{prefix}_stitched.png", warnings)
        if result.debug_ocr_input is not None:
            _save_image(result.debug_ocr_input, debug_dir / f"{prefix}_final_ocr_input.png", warnings)
        records.append(
            {
                "index": result.index,
                "bbox": list(result.box),
                "detection_confidence": result.detection_confidence,
                "detection_source": result.detection_source,
                "row_confidence": result.row_confidence,
                "detected_rows": result.detected_rows,
                "selected_rows": result.selected_rows,
                "ignored_rows": result.ignored_rows,
                "row_angles": result.row_angles,
                "ocr_strategy": result.ocr_strategy,
                "text_result": result.as_dict(),
                "ocr_calls": result.ocr_calls,
                "selected_ocr_inputs": [call['input'] for call in matching_calls],
                "row_detector_message": result.row_detector_message,
                "row_direct_disagreement": result.row_direct_disagreement,
                "latin_ocr_prediction": result.latin_ocr_prediction,
                "devanagari_ocr_prediction": result.devanagari_ocr_prediction,
                "mixed_ocr_prediction": result.mixed_ocr_prediction,
                "raw_ocr_text": result.raw_ocr,
                "normalized_text": result.normalized_text,
                "ocr_confidence": result.ocr_confidence,
                "validation_status": result.validation_status,
                "format_validation": result.validation_status,
                "final_status": result.final_status,
                "status_reason": result.status_reason,
                "accepted": result.accepted,
                "risk_flags": result.risk_flags,
                "fallback_used": result.fallback_ocr_used,
                "postprocessing_applied": result.postprocessing_applied,
            }
        )
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "results.json").write_text(
            json.dumps(
                {
                    "image": str(path),
                    "detection_counts": {
                        "primary": diagnostics.primary_count,
                        "fallback": diagnostics.fallback_count,
                        "tile": diagnostics.tile_count,
                        "final_nms": diagnostics.final_count,
                    },
                    "plates": records,
                    "warnings": warnings,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def infer_all(
    path: Path,
    detector: PlateDetector,
    row_detector: RowDetector,
    recognizer: PARSeqRecognizer,
    ocr_threshold: float,
    output_dir: Path | None = None,
    nms_iou_threshold: float = 0.6,
    row_confidence_threshold: float = 0.25,
    detection_diagnostics: DetectionDiagnostics | None = None,
    detector_fallback: bool = False,
    detector_fallback_image_size: int = 2560,
    detector_fallback_confidence: float = 0.05,
    detector_tile_fallback: bool = False,
    detector_tile_size: int = 1280,
    detector_tile_overlap: float = 0.25,
    debug_output_dir: Path | None = None,
    devanagari_recognizer: PARSeqRecognizer | None = None,
    mixed_recognizer: PARSeqRecognizer | None = None,
    header_recognizer: PARSeqRecognizer | None = None,
) -> list[PlateInference]:
    """Process every detected plate; failure of one plate never aborts the rest."""
    image = load_image(path)
    diagnostics = detection_diagnostics if detection_diagnostics is not None else DetectionDiagnostics()
    plates = _detect_for_inference(
        detector,
        image,
        nms_iou_threshold,
        diagnostics,
        detector_fallback,
        detector_fallback_image_size,
        detector_fallback_confidence,
        detector_tile_fallback,
        detector_tile_size,
        detector_tile_overlap,
    )
    results: list[PlateInference] = []
    base_recognizers = recognizer, devanagari_recognizer, mixed_recognizer
    for index, plate in enumerate(plates, 1):
        result = PlateInference(
            index,
            "PLATE_UNREADABLE",
            plate.box,
            plate.confidence,
            detection_source=plate.source,
            debug_crop=plate.crop,
        )
        name = _artifact_name(path, index)
        recognizer, devanagari_recognizer, mixed_recognizer = base_recognizers
        if debug_output_dir is not None:
            recognizer, devanagari_recognizer, mixed_recognizer = (
                _DebugRecognizer(model, label, result, ocr_threshold) if model is not None else None
                for model, label in zip(base_recognizers, ('latin', 'devanagari', 'mixed'))
            )
            result.debug_images['02_plate_detection_crop.png'] = image.crop(tuple(map(round, plate.box)))
            result.debug_images['03_plate_crop_expanded.png'] = plate.crop
            result.debug_images['04_row_detector_input.png'] = plate.crop

        row_failure = "row path did not complete"
        used_no_obb_crop = False
        primary_evaluation: OCREvaluation | None = None
        primary_routing: OCRRouting | None = None
        primary_ocr_input: Image.Image | None = None
        primary_row_image: Image.Image | None = None
        try:
            row_result = row_detector.process(plate.crop)
            result.row_result = row_result
            result.debug_row_image = row_result.image
            result.debug_stitched_image = row_result.image
            result.rows = row_result.rows
            result.detected_rows = row_result.detected_rows if row_result.detected_rows is not None else row_result.rows
            result.selected_rows = row_result.selected_rows
            result.ignored_rows = row_result.ignored_rows
            result.row_angles = row_result.angles
            result.row_confidence = row_result.confidence
            if debug_output_dir is not None:
                annotated_rows = plate.crop.copy()
                row_draw = ImageDraw.Draw(annotated_rows)
                for region in row_result.selected_rows + row_result.ignored_rows:
                    polygon = region.get('polygon', [])
                    if polygon:
                        row_draw.polygon([tuple(point) for point in polygon], outline='red')
                        row_draw.text(tuple(polygon[0]), str(region['index']), fill='red')
                    bounds = region.get('bounds')
                    if bounds:
                        region_prefix = f"row_{region['index']:02d}"
                        result.debug_images[region_prefix + '_raw.png'] = plate.crop.crop(tuple(map(round, bounds)))
                        regions = getattr(row_result, 'region_images', [])
                        if region['index'] < len(regions):
                            result.debug_images[region_prefix + '_rectified.png'] = regions[region['index']]
                result.debug_images['05_row_detector_all_regions.png'] = annotated_rows
            row_rejection = _row_rejection(row_result, row_confidence_threshold)
            if row_rejection:
                row_failure = row_rejection
                result.row_detector_message = f"FAILED: {row_rejection}\nUsing original crop."
            else:
                no_obb_fallback = row_result.rows == 1 and row_result.angles == [0.0] and row_result.confidence == 0.0
                used_no_obb_crop = no_obb_fallback
                result.row_detector_succeeded = not no_obb_fallback
                result.fallback_ocr_used = no_obb_fallback
                result.row_detector_message = (
                    "No OBB found.\nUsing original crop."
                    if no_obb_fallback
                    else f"SUCCESS: selected {row_result.rows} of {result.detected_rows} detected row regions"
                )
                primary_row_image = row_result.image
                primary_ocr_input = mild_ocr_preprocess(row_result.image)
                result.debug_stage = 'registration_rows'
                result.debug_ocr_input = primary_ocr_input
                result.ocr_ran = True
                primary_routing = _route_row_result(
                    row_result,
                    primary_ocr_input,
                    recognizer,
                    devanagari_recognizer,
                    ocr_threshold,
                    mixed_recognizer,
                )
                primary_evaluation = primary_routing.evaluation
                for metadata, prediction in zip(getattr(row_result, 'selected_rows', []), primary_routing.row_predictions or [
                    {'text': primary_evaluation.normalized_text, 'confidence': primary_evaluation.confidence,
                     'strategy': primary_routing.strategy}]):
                    for classified in getattr(row_result, 'classified_rows', []):
                        if classified['index'] == metadata['index']:
                            classified.update({key: value for key, value in prediction.items() if key != 'confidence'})
                            classified['confidence_ocr'] = prediction['confidence']
                _apply_ocr_evaluation(result, primary_evaluation)
                _apply_routing_diagnostics(result, primary_routing)
                row_failure = primary_evaluation.final_status
            if (
                not used_no_obb_crop
                and primary_evaluation is not None
                and primary_evaluation.final_status == "FORMAT_CONFIDENT"
                and primary_evaluation.confidence >= 0.99
            ):
                if output_dir is not None:
                    _save_detection(image, plate, output_dir / "detections" / name, result.artifact_warnings)
                    _save_image(plate.crop, output_dir / "crops" / name, result.artifact_warnings)
                    _save_image(primary_row_image, output_dir / "rectified" / name, result.artifact_warnings)
                    _save_image(primary_ocr_input, output_dir / "stitched" / name, result.artifact_warnings)
                results.append(result)
                continue
        except Exception as error:
            row_failure = f"{type(error).__name__}: {error}"
            if result.row_detector_message == "NOT RUN":
                result.row_detector_message = f"FAILED: {row_failure}\nUsing original crop."

        if used_no_obb_crop:
            if primary_evaluation is None or not primary_evaluation.usable:
                result.error = f"Row path: no OBB found; direct-crop OCR: {row_failure}"
            elif output_dir is not None and primary_evaluation.saveable:
                _save_detection(image, plate, output_dir / "detections" / name, result.artifact_warnings)
                _save_image(plate.crop, output_dir / "crops" / name, result.artifact_warnings)
                _save_image(primary_ocr_input, output_dir / "stitched" / name, result.artifact_warnings)
            results.append(result)
            continue

        result.fallback_ocr_used = True
        fallback_evaluation: OCREvaluation | None = None
        fallback_routing: OCRRouting | None = None
        fallback_ocr_input: Image.Image | None = None
        try:
            fallback_ocr_input = mild_ocr_preprocess(plate.crop)
            result.debug_stage = 'direct_plate_fallback'
            result.debug_ocr_input = fallback_ocr_input
            result.ocr_ran = True
            fallback_routing = _route_ocr(
                fallback_ocr_input,
                recognizer,
                devanagari_recognizer,
                ocr_threshold,
                mixed_recognizer,
                plate.crop,
            )
            fallback_evaluation = fallback_routing.evaluation
            result.row_direct_disagreement = bool(
                primary_evaluation is not None
                and primary_evaluation.usable
                and fallback_evaluation.usable
                and primary_evaluation.normalized_text != fallback_evaluation.normalized_text
            )
            selected_evaluation = _reconcile_ocr(primary_evaluation, fallback_evaluation,
                header_filtered=any(row.get('reason') == 'small header/decorative text' for row in result.ignored_rows))
            if selected_evaluation.usable:
                _apply_ocr_evaluation(result, selected_evaluation)
                if primary_evaluation is not None and selected_evaluation.normalized_text == primary_evaluation.normalized_text:
                    selected_routing = primary_routing
                elif fallback_evaluation is not None and selected_evaluation.normalized_text == fallback_evaluation.normalized_text:
                    selected_routing = fallback_routing
                else:
                    selected_routing = None
                if selected_routing is not None:
                    _apply_routing_diagnostics(result, selected_routing)
                    result.debug_ocr_input = primary_ocr_input if selected_routing is primary_routing else fallback_ocr_input
                else:
                    result.ocr_strategy = "row/direct fallback reconciliation retained only their visible consensus"
                if primary_evaluation is not None and fallback_evaluation is not None:
                    result.ocr_strategy = f"row/direct fallback reconciliation; {result.ocr_strategy}"
                result.error = None
                if output_dir is not None and selected_evaluation.saveable:
                    _save_detection(image, plate, output_dir / "detections" / name, result.artifact_warnings)
                    _save_image(plate.crop, output_dir / "crops" / name, result.artifact_warnings)
                    if primary_row_image is not None:
                        _save_image(primary_row_image, output_dir / "rectified" / name, result.artifact_warnings)
                    _save_image(fallback_ocr_input, output_dir / "stitched" / name, result.artifact_warnings)
            else:
                result.error = f"Row path: {row_failure}; fallback path: {fallback_evaluation.final_status}"
        except Exception as error:
            fallback_failure = f"{type(error).__name__}: {error}"
            if primary_evaluation is not None and primary_evaluation.usable:
                _apply_ocr_evaluation(result, primary_evaluation)
                result.error = None
            else:
                result.error = f"Row path: {row_failure}; fallback path: {fallback_failure}"
        results.append(result)
    for result in results:
        _finalize_acceptance(result)
        _complete_plate_text(result, base_recognizers, ocr_threshold, debug_output_dir is not None, header_recognizer)
    if debug_output_dir is not None:
        _save_debug_bundle(path, image, results, diagnostics, debug_output_dir)
    return results


def _print_detection_diagnostics(diagnostics: DetectionDiagnostics) -> None:
    print(f"Primary detections: {diagnostics.primary_count}")
    print(f"Fallback detections: {diagnostics.fallback_count}")
    print(f"Tile detections: {diagnostics.tile_count}")
    print(f"Final NMS detections: {diagnostics.final_count}")


def _print_results(path: Path, results: list[PlateInference], verbose: bool, folder: bool = False) -> None:
    accepted_count = sum(result.final_status == "ACCEPTED" for result in results)
    rejected_count = sum(result.final_status == "REJECTED" for result in results)
    review_count = len(results) - accepted_count - rejected_count
    if not results:
        if verbose or folder:
            print(f"Image: {path}")
        print("Accepted Plates: 0")
        print("Review Candidates: 0")
        print("Rejected Candidates: 0")
        return
    if verbose or folder:
        print(f"Image: {path}")
    print(f"Accepted Plates: {accepted_count}")
    print(f"Review Candidates: {review_count}")
    print(f"Rejected Candidates: {rejected_count}")
    for result in results:
        print(f"\nPlate {result.index}")
        print(f"Text: {result.text}")
        print(f"Header Text: {result.header_text or 'NONE'}")
        if result.header_text:
            print('Header Status: REVIEW_REQUIRED (independent OCR prediction)')
        print(f"Registration Text: {result.registration_text or result.text}")
        print(f"Full Plate Text: {result.full_text or result.text}")
        print(f"Raw OCR Text: {result.raw_ocr}")
        print(f"Normalized Text: {result.normalized_text}")
        print(f"Detection Confidence: {result.detection_confidence:.4f}")
        print(f"OCR Confidence: {result.ocr_confidence:.4f}" if result.ocr_confidence is not None else "OCR Confidence: N/A")
        print(f"Format Validation: {result.validation_status}")
        print(f"Final Status: {result.final_status}")
        print(f"Reason: {result.status_reason}")
        if verbose:
            print(f"Plate bbox: {[round(value, 1) for value in result.box]}")
            print(f"Detection source: {result.detection_source}")
            print(f"Row angles: {[round(value, 2) for value in result.row_angles]}")
            print(f"Row confidence: {result.row_confidence:.4f}")
            print("Plate detector: SUCCESS")
            print("Detector: SUCCESS")
            print("Row Detector:")
            print(result.row_detector_message)
            print(f"Detected row count: {result.detected_rows}")
            print(f"Selected rows: {result.selected_rows}")
            print(f"Ignored rows: {result.ignored_rows}")
            print(f"Text rows: {result.text_rows}")
            print("OCR:")
            print("RUNNING" if result.ocr_ran else "NOT RUN")
            print(f"OCR strategy: {result.ocr_strategy}")
            print(f"Latin OCR prediction: {result.latin_ocr_prediction or '<empty>'}")
            print(f"Devanagari OCR prediction: {result.devanagari_ocr_prediction or '<empty>'}")
            print(f"Mixed OCR prediction: {result.mixed_ocr_prediction or '<unavailable>'}")
            print(f"Combined/final prediction: {result.text}")
            print("OCR Result:")
            print(result.text if result.validation_passed else result.raw_ocr)
            print(f"Raw OCR: {result.raw_ocr}")
            print(f"Normalized text: {result.normalized_text}")
            print(f"Post-processing applied: {'YES' if result.postprocessing_applied else 'NO'}")
            print("Validation:")
            print(result.validation_status)
            print(f"Final status: {result.final_status}")
            print(f"Status reason: {result.status_reason}")
            print(f"Fallback OCR: {'USED' if result.fallback_ocr_used else 'NOT USED'}")
            print(f"Final result: {result.text}")


def infer(path: Path, detector: PlateDetector, row_detector: RowDetector, recognizer: PARSeqRecognizer, ocr_threshold: float, verbose: bool) -> str:
    """Backward-compatible single-string API; all plates are still processed."""
    results = infer_all(path, detector, row_detector, recognizer, ocr_threshold)
    if verbose:
        _print_results(path, results, True)
    return results[0].text if results else "NO_PLATE_DETECTED"


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    header_recognizer = None
    try:
        header_recognizer = load_header_recognizer(args.header_ocr_weights, device)
    except Exception as error:
        if args.verbose:
            print(f'Header checkpoint unavailable: {error}; using existing header candidates')
    try:
        detector = PlateDetector(args.det_weights, device, args.det_conf, args.det_imgsz)
        row_detector = RowDetector(args.row_weights, device)
        recognizer = PARSeqRecognizer(args.ocr_weights, device)
        devanagari_recognizer = None
        if args.devanagari_ocr_weights.is_file() and args.devanagari_ocr_weights.resolve() != args.ocr_weights.resolve():
            devanagari_recognizer = PARSeqRecognizer(args.devanagari_ocr_weights, device)
        mixed_recognizer = PARSeqRecognizer(args.mixed_ocr_weights, device) if args.mixed_ocr_weights.is_file() else None
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if args.image:
        if not args.image.is_file():
            print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
            return 2
        output_dir = args.output_dir if args.save_results else None
        diagnostics = DetectionDiagnostics()
        results = infer_all(
            args.image,
            detector,
            row_detector,
            recognizer,
            args.ocr_conf,
            output_dir,
            args.det_nms_iou,
            args.row_conf,
            diagnostics,
            args.det_fallback,
            args.det_fallback_imgsz,
            args.det_fallback_conf,
            args.det_tile_fallback,
            args.det_tile_size,
            args.det_tile_overlap,
            args.output_dir if args.save_debug else None,
            devanagari_recognizer,
            mixed_recognizer,
            header_recognizer,
        )
        if args.verbose:
            if mixed_recognizer is None:
                print(f"Mixed OCR checkpoint: NOT FOUND ({args.mixed_ocr_weights}); using Latin + Devanagari fallback")
            else:
                print(f"Mixed OCR checkpoint: LOADED ({args.mixed_ocr_weights})")
            _print_detection_diagnostics(diagnostics)
        _print_results(args.image, results, args.verbose)
        if not results:
            attempted = "primary, fallback, and tile detection" if diagnostics.tile_attempted else "primary and fallback detection"
            print(f"No plate detected after {attempted}. This is likely a detector training-data failure case.")
        return 0
    if not args.folder.is_dir():
        print(f"ERROR: Folder not found: {args.folder}", file=sys.stderr)
        return 2
    paths = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    output_dir = args.output_dir if args.save_results else None
    for path in paths:
        diagnostics = DetectionDiagnostics()
        results = infer_all(
            path,
            detector,
            row_detector,
            recognizer,
            args.ocr_conf,
            output_dir,
            args.det_nms_iou,
            args.row_conf,
            diagnostics,
            args.det_fallback,
            args.det_fallback_imgsz,
            args.det_fallback_conf,
            args.det_tile_fallback,
            args.det_tile_size,
            args.det_tile_overlap,
            args.output_dir if args.save_debug else None,
            devanagari_recognizer,
            mixed_recognizer,
            header_recognizer,
        )
        if args.verbose:
            if mixed_recognizer is None:
                print(f"Mixed OCR checkpoint: NOT FOUND ({args.mixed_ocr_weights}); using Latin + Devanagari fallback")
            else:
                print(f"Mixed OCR checkpoint: LOADED ({args.mixed_ocr_weights})")
            _print_detection_diagnostics(diagnostics)
        _print_results(path, results, args.verbose, folder=True)
        if not results:
            attempted = "primary, fallback, and tile detection" if diagnostics.tile_attempted else "primary and fallback detection"
            print(f"No plate detected after {attempted}. This is likely a detector training-data failure case.")
        if args.verbose:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
