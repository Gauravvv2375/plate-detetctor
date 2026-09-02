"""Text-row detector. Input: plate crop. Processing: YOLO11-OBB plus perspective warp. Output: ordered, stitched OCR image."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .preprocess import rectify_quad, stitch_rows


@dataclass
class RowResult:
    image: Image.Image
    rows: int
    angles: list[float]
    confidence: float


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
            return RowResult(plate, 1, [0.0], 0.0)
        polygons = obb.xyxyxyxy.cpu().tolist()
        confidences = obb.conf.cpu().tolist()
        rows, angles = [], []
        for polygon in polygons:
            center_y = sum(point[1] for point in polygon) / 4.0
            rows.append((rectify_quad(plate, polygon), center_y))
            dx, dy = polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1]
            angles.append(math.degrees(math.atan2(dy, dx)))
        return RowResult(stitch_rows(rows), len(rows), angles, float(sum(confidences) / len(confidences)))
