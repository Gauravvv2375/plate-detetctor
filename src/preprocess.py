"""Image geometry helpers. Input: PIL images/boxes. Processing: EXIF correction, crop expansion and rectification. Output: OCR-ready images."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def expanded_crop(image: Image.Image, box: Sequence[float], margin: float = 0.08) -> Image.Image:
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    bounds = (max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(image.width, int(x2 + dx)), min(image.height, int(y2 + dy)))
    return image.crop(bounds)


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(differences)], points[np.argmax(sums)], points[np.argmax(differences)]], dtype=np.float32)


def rectify_quad(image: Image.Image, points: Sequence[Sequence[float]]) -> Image.Image:
    """Perspective-flatten an oriented text-row quadrilateral."""
    import cv2

    quad = _order_quad(np.asarray(points))
    tl, tr, br, bl = quad
    width = max(2, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = max(2, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    warped = cv2.warpPerspective(np.asarray(image), matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def stitch_rows(rows: list[tuple[Image.Image, float]]) -> Image.Image:
    """Sort rectified rows top-to-bottom, resize to one height, then concatenate."""
    if not rows:
        raise ValueError("Cannot stitch an empty row list")
    ordered = [image for image, _ in sorted(rows, key=lambda item: item[1])]
    target_height = max(image.height for image in ordered)
    resized = [image.resize((max(1, round(image.width * target_height / image.height)), target_height), Image.Resampling.LANCZOS) for image in ordered]
    gap = max(2, target_height // 12) if len(resized) > 1 else 0
    canvas = Image.new("RGB", (sum(x.width for x in resized) + gap * (len(resized) - 1), target_height), "white")
    left = 0
    for row in resized:
        canvas.paste(row, (left, 0))
        left += row.width + gap
    return canvas


def mild_ocr_preprocess(image: Image.Image, min_height: int = 48) -> Image.Image:
    """Conservative contrast/sharpening; avoids thresholding away faint strokes."""
    if image.height < min_height:
        scale = min_height / max(1, image.height)
        image = image.resize((max(1, round(image.width * scale)), min_height), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.12)
    return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
