"""PARSeq OCR adapter. Input: stitched text image. Processing: docTR PARSeq decoding. Output: raw text and confidence."""

from __future__ import annotations

import gc
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
        # Production checkpoints also contain optimizer/scheduler state. Memory
        # mapping prevents those unused training tensors from being copied into
        # RSS while the inference model is initialized.
        checkpoint = torch.load(
            weights, map_location="cpu", weights_only=False, mmap=True
        )
        self.checkpoint_path = str(weights.resolve())
        self.checkpoint_epoch = checkpoint.get("epoch")
        vocabulary = checkpoint.get("vocab", PLATE_CHARSET)
        max_length = int(checkpoint.get("config", {}).get("max_length", 32))
        self.max_length = max_length
        self.input_normalization = checkpoint.get("config", {}).get("input_normalization")
        self.vocabulary = vocabulary
        self.model = parseq(
            pretrained=False, pretrained_backbone=False, vocab=vocabulary, max_length=max_length
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        self.model.load_state_dict(state_dict, assign=True)
        del state_dict, checkpoint
        gc.collect()
        self.model.to(device).eval()

    @staticmethod
    def input_image(image: Image.Image) -> Image.Image:
        """Exact RGB pixels before tensor normalization; also used for debug exports."""
        return image.convert("RGB").resize((128, 32), Image.Resampling.BILINEAR)

    def _tensor(self, image: Image.Image):
        image = self.input_image(image)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.input_normalization == "doctr_parseq":
            mean = np.asarray((0.694, 0.695, 0.693), dtype=np.float32).reshape(3, 1, 1)
            std = np.asarray((0.299, 0.296, 0.301), dtype=np.float32).reshape(3, 1, 1)
            array = (array - mean) / std
        return self.torch.from_numpy(array).unsqueeze(0).to(self.device)

    def recognize(self, image: Image.Image) -> OCRResult:
        with self.torch.inference_mode():
            output = self.model(self._tensor(image), return_preds=True)
        prediction = output["preds"][0]
        if isinstance(prediction, (tuple, list)):
            raw_text, confidence = prediction[0], float(prediction[1])
        else:
            raw_text, confidence = str(prediction), 0.0
        text = str(raw_text)
        if self.vocabulary == PLATE_CHARSET:
            cleaned = clean_plate_text(text)
        else:
            import unicodedata

            allowed = set(self.vocabulary)
            cleaned = "".join(character for character in unicodedata.normalize("NFC", text) if character in allowed)
        return OCRResult(text, cleaned, confidence)
