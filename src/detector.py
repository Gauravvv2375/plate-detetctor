"""Whole-plate detector. Input: vehicle image. Processing: YOLO11 prediction. Output: ranked  plate boxes and crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .preprocess import expanded_crop


@dataclass
class PlateDetection:
    box: tuple[float, float, float, float]
    confidence: float
    crop: Image.Image


class PlateDetector:
    def __init__(self, weights: Path, device: str, confidence: float = 0.25, image_size: int = 640, crop_margin: float = 0.08):
        if not weights.is_file():
            raise FileNotFoundError(f"Plate detector weights not found: {weights}. Run scripts/train_detector.py first.")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device, self.confidence, self.image_size, self.crop_margin = device, confidence, image_size, crop_margin

    def detect(self, image: Image.Image) -> list[PlateDetection]:
        result = self.model.predict(source=image, imgsz=self.image_size, conf=self.confidence, device=self.device, verbose=False)[0]
        detections: list[PlateDetection] = []
        if result.boxes is None:
            return detections
        for coordinates, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            box = tuple(float(x) for x in coordinates)
            detections.append(PlateDetection(box, float(confidence), expanded_crop(image, box, self.crop_margin)))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)
