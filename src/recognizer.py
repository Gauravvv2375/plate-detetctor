"""PARSeq OCR adapter. Input: stitched text image. Processing: docTR PARSeq decoding. Output: raw text and confidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import PLATE_CHARSET, clean_plate_text


@dataclass
class OCRResult:
    raw_text: str
    text: str
    confidence: float


class PARSeqRecognizer:
    """Thin wrapper around docTR's implementation of the PARSeq architecture."""

    def __init__(self, weights: Path, device: str):
        if not weights.is_file():
            raise FileNotFoundError(f"PARSeq weights not found: {weights}. Run scripts/train_ocr.py first.")
        import torch
        from doctr.models import parseq

        self.torch, self.device = torch, device
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        vocabulary = checkpoint.get("vocab", PLATE_CHARSET)
        self.model = parseq(pretrained=False, pretrained_backbone=False, vocab=vocabulary)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def _tensor(self, image: Image.Image):
        image = image.convert("RGB").resize((128, 32), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return self.torch.from_numpy(array).unsqueeze(0).to(self.device)

    def recognize(self, image: Image.Image) -> OCRResult:
        with self.torch.inference_mode():
            output = self.model(self._tensor(image), return_preds=True)
        prediction = output["preds"][0]
        if isinstance(prediction, (tuple, list)):
            raw_text, confidence = prediction[0], float(prediction[1])
        else:
            raw_text, confidence = str(prediction), 0.0
        return OCRResult(str(raw_text), clean_plate_text(str(raw_text)), confidence)
