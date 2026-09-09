"""Text-row detector. Input: plate crop. Processing: YOLO11-OBB plus perspective warp. Output: ordered, stitched OCR image."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .preprocess import rectify_quad, stitch_rows


@dataclass
class RowResult:
    image: Image.Image
    rows: int
    angles: list[float]
    confidence: float
    detected_rows: int | None = None
    selected_rows: list[dict] = field(default_factory=list)
    ignored_rows: list[dict] = field(default_factory=list)
    row_images: list[Image.Image] = field(default_factory=list, repr=False)
    region_images: list[Image.Image] = field(default_factory=list, repr=False)
    classified_rows: list[dict] = field(default_factory=list)


@dataclass
class _RowCandidate:
    index: int
    image: Image.Image
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    confidence: float
    bounds: tuple[float, float, float, float]
    polygon: list = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.width * self.height

    def diagnostic(self, reason: str) -> dict:
        return {
            "index": self.index,
            "confidence": round(self.confidence, 4),
            "center": [round(self.center_x, 1), round(self.center_y, 1)],
            "size": [round(self.width, 1), round(self.height, 1)],
            "reason": reason,
            "angle": self.angle,
            "bounds": list(self.bounds),
            "polygon": self.polygon,
        }


def _intersection_over_smaller(left: _RowCandidate, right: _RowCandidate) -> float:
    x1, y1 = max(left.bounds[0], right.bounds[0]), max(left.bounds[1], right.bounds[1])
    x2, y2 = min(left.bounds[2], right.bounds[2]), min(left.bounds[3], right.bounds[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.bounds[2] - left.bounds[0]) * max(0.0, left.bounds[3] - left.bounds[1])
    right_area = max(0.0, right.bounds[2] - right.bounds[0]) * max(0.0, right.bounds[3] - right.bounds[1])
    smaller = min(left_area, right_area)
    return intersection / smaller if smaller > 0 else 0.0


def _select_registration_rows(candidates: list[_RowCandidate], plate_size: tuple[int, int]):
    """Remove duplicate/tiny header detections and retain at most two registration rows."""
    ignored: list[dict] = []
    deduplicated: list[_RowCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = next(
            (
                accepted for accepted in deduplicated
                if _intersection_over_smaller(candidate, accepted) >= 0.55
                and abs(candidate.center_y - accepted.center_y) <= 0.65 * max(candidate.height, accepted.height)
            ),
            None,
        )
        if duplicate is not None:
            ignored.append(candidate.diagnostic(f"duplicate of row {duplicate.index}"))
        else:
            deduplicated.append(candidate)
    if not deduplicated:
        return [], ignored

    largest = max(deduplicated, key=lambda item: item.area)
    substantial: list[_RowCandidate] = []
    for candidate in deduplicated:
        if candidate is largest:
            substantial.append(candidate)
            continue
        area_ratio = candidate.area / max(1.0, largest.area)
        height_ratio = candidate.height / max(1.0, largest.height)
        if area_ratio < 0.35 and height_ratio < 0.65:
            ignored.append(candidate.diagnostic("small header/decorative text"))
        else:
            substantial.append(candidate)

    _, plate_height = plate_size
    if len(substantial) > 2:
        max_area = max(item.area for item in substantial)
        max_height = max(item.height for item in substantial)

        def score(item: _RowCandidate) -> float:
            center_distance = abs(item.center_y - plate_height / 2) / max(1.0, plate_height / 2)
            centrality = max(0.0, 1.0 - center_distance)
            return (
                item.area / max_area
                + 0.75 * item.height / max_height
                + 0.50 * item.confidence
                + 0.20 * centrality
            )

        ranked = sorted(substantial, key=score, reverse=True)
        substantial = ranked[:2]
        ignored.extend(item.diagnostic("lower registration-row rank") for item in ranked[2:])
    selected = sorted(substantial, key=lambda item: item.center_y)
    return selected, ignored


class RowDetector:
    def __init__(self, weights: Path, device: str, confidence: float = 0.25, image_size: int = 320):
        if not weights.is_file():
            raise FileNotFoundError(f"Row detector weights not found: {weights}. Run scripts/train_row_detector.py first.")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device, self.confidence, self.image_size = device, confidence, image_size

    def process(self, plate: Image.Image) -> RowResult:
        result = self.model.predict(source=plate, imgsz=self.image_size, conf=self.confidence, device=self.device, verbose=False)[0]
        obb = result.obb
        if obb is None or len(obb) == 0:
            # A single already-cropped row is safer than fabricating geometry.
            return RowResult(plate, 1, [0.0], 0.0, 0, [], [], [plate])
        polygons = obb.xyxyxyxy.cpu().tolist()
        confidences = obb.conf.cpu().tolist()
        candidates = []
        for index, (polygon, confidence) in enumerate(zip(polygons, confidences)):
            center_y = sum(point[1] for point in polygon) / 4.0
            center_x = sum(point[0] for point in polygon) / 4.0
            dx, dy = polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1]
            edges = [
                math.hypot(
                    polygon[(edge + 1) % 4][0] - polygon[edge][0],
                    polygon[(edge + 1) % 4][1] - polygon[edge][1],
                )
                for edge in range(4)
            ]
            xs, ys = [point[0] for point in polygon], [point[1] for point in polygon]
            candidates.append(
                _RowCandidate(
                    index=index,
                    image=rectify_quad(plate, polygon),
                    center_x=center_x,
                    center_y=center_y,
                    width=max(edges),
                    height=min(edges),
                    angle=math.degrees(math.atan2(dy, dx)),
                    confidence=float(confidence),
                    bounds=(min(xs), min(ys), max(xs), max(ys)),
                    polygon=polygon,
                )
            )
        selected, ignored = _select_registration_rows(candidates, plate.size)
        if not selected:
            return RowResult(plate, 1, [0.0], 0.0, len(candidates), [], ignored, [plate])
        rows = [(candidate.image, candidate.center_y) for candidate in selected]
        output = RowResult(
            image=stitch_rows(rows),
            rows=len(selected),
            angles=[candidate.angle for candidate in selected],
            confidence=float(sum(candidate.confidence for candidate in selected) / len(selected)),
            detected_rows=len(candidates),
            selected_rows=[candidate.diagnostic("selected registration row") for candidate in selected],
            ignored_rows=ignored,
            row_images=[candidate.image for candidate in selected],
            region_images=[candidate.image for candidate in candidates],
            classified_rows=_classify_rows(selected, ignored, candidates),
        )
        if not any(row['role'] == 'HEADER' for row in output.classified_rows):
            self._recover_small_text(plate, output, selected)
        return output

    def _recover_small_text(self, plate, output, selected):
        """Higher-resolution auxiliary detection cannot alter registration geometry."""
        import cv2
        import numpy as np
        try:
            prediction = self.model.predict(source=plate, imgsz=max(960, self.image_size),
                conf=max(.4, self.confidence), device=self.device, verbose=False)[0]
            if prediction.obb is None:
                return
            primary = max(selected, key=lambda row: row.area)
            theta = math.radians((primary.angle + 45) % 90 - 45)
            normal = np.array([-math.sin(theta), math.cos(theta)])
            accepted = [np.asarray(row.polygon, dtype=np.float32) for row in selected]
            for polygon, confidence in sorted(zip(prediction.obb.xyxyxyxy.cpu().tolist(),
                                                  prediction.obb.conf.cpu().tolist()), key=lambda item: -item[1]):
                quad = np.asarray(polygon, dtype=np.float32)
                center = quad.mean(axis=0)
                edges = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
                width, height = float(max(edges)), float(min(edges))
                offset = float((center - [primary.center_x, primary.center_y]) @ normal)
                if not (8 <= height < .65 * primary.height and width >= 24 and width / height >= 2
                        and offset < -.5 * primary.height):
                    continue
                area = abs(cv2.contourArea(quad))
                if any(cv2.intersectConvexConvex(quad, other)[0] / max(1, min(area, abs(cv2.contourArea(other)))) >= .35
                       for other in accepted):
                    continue
                edge = quad[(int(np.argmax(edges)) - 1) % 4] - quad[int(np.argmax(edges))]
                angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
                if abs(((angle - primary.angle) + 45) % 90 - 45) > 20:
                    continue
                accepted.append(quad)
                index = len(output.region_images)
                crop = rectify_quad(plate, polygon)
                row = _RowCandidate(index, crop, float(center[0]), float(center[1]), width, height,
                    angle, float(confidence), (float(quad[:,0].min()), float(quad[:,1].min()),
                    float(quad[:,0].max()), float(quad[:,1].max())), polygon)
                output.region_images.append(crop)
                metadata = row.diagnostic('auxiliary small-text detection; registration selection unchanged')
                output.ignored_rows.append(metadata)
                output.classified_rows.append(metadata | {'role': 'HEADER', 'source': 'auxiliary_960'})
            # The fine-scale detector can split one physical header into words.
            # Rectify their common line from the original pixels, not a stitch
            # of separately resized words, so intervening text stays visible.
            headers = [row for row in output.classified_rows if row.get('source') == 'auxiliary_960']
            groups = []
            for row in headers:
                for group in groups:
                    delta = np.asarray(row['center']) - group[0]['center']
                    if abs(float(delta @ normal)) < .6 * min(row['size'][1], group[0]['size'][1]):
                        group.append(row)
                        break
                else:
                    groups.append([row])
            for group in groups:
                if len(group) < 2:
                    continue
                points = np.concatenate([np.asarray(row['polygon'], dtype=np.float32) for row in group])
                polygon = cv2.boxPoints(cv2.minAreaRect(points)).tolist()
                crop = rectify_quad(plate, polygon)
                center = np.mean(polygon, axis=0)
                index = len(output.region_images)
                output.region_images.append(crop)
                for row in group:
                    row['role'] = 'DECORATIVE_OR_NOISE'
                    row['reason'] = f'merged into header row {index}'
                merged = _RowCandidate(index,crop,float(center[0]),float(center[1]),crop.width,crop.height,
                    primary.angle,min(row['confidence'] for row in group),
                    (float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())),polygon)
                metadata = merged.diagnostic('same-line auxiliary regions merged geometrically')
                output.ignored_rows.append(metadata)
                output.classified_rows.append(metadata | {'role':'HEADER','source':'auxiliary_960',
                    'merged_from':[row['index'] for row in group]})
        except Exception:
            # Auxiliary detection is optional; never invalidate registration.
            return


def _classify_rows(selected, ignored, candidates):
    """Retain distinct, horizontal small text separately; never reassign registration rows."""
    primary = max(selected, key=lambda row: row.area) if selected else None
    selected_ids = {row.index for row in selected}
    reasons = {row['index']: row['reason'] for row in ignored}
    result = []
    for row in candidates:
        reason = reasons.get(row.index, 'selected registration row')
        role = 'DECORATIVE_OR_NOISE'
        if row.index in selected_ids:
            role = 'PRIMARY_REGISTRATION' if row is primary else 'SECONDARY_REGISTRATION'
        elif (primary and reason == 'small header/decorative text'
              and row.confidence >= .40 and row.height >= 8 and row.width >= 24
              and row.width / max(1, row.height) >= 2
              and abs((row.angle + 45) % 90 - 45) <= 25):
            role = 'HEADER'
        result.append(row.diagnostic(reason) | {'role': role})
    return result
