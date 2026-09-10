from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.recognizer import PARSeqRecognizer


class RecognizerLoadingTests(unittest.TestCase):
    def test_inference_checkpoint_is_memory_mapped_without_state_copy(self):
        checkpoint = {
            "epoch": 7,
            "model_state_dict": {"weight": object()},
            "config": {"max_length": 16, "input_normalization": "doctr_parseq"},
            "vocab": "ABC123",
        }
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        fake_torch = types.ModuleType("torch")
        fake_torch.load = Mock(return_value=checkpoint)
        fake_doctr = types.ModuleType("doctr")
        fake_models = types.ModuleType("doctr.models")
        fake_models.parseq = Mock(return_value=model)

        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "best.pt"
            weights.touch()
            with patch.dict(
                sys.modules,
                {"torch": fake_torch, "doctr": fake_doctr, "doctr.models": fake_models},
            ):
                recognizer = PARSeqRecognizer(weights, "cpu")

        fake_torch.load.assert_called_once_with(
            weights, map_location="cpu", weights_only=False, mmap=True
        )
        model.load_state_dict.assert_called_once_with(
            checkpoint["model_state_dict"], assign=True
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        self.assertEqual(recognizer.vocabulary, "ABC123")


if __name__ == "__main__":
    unittest.main()
