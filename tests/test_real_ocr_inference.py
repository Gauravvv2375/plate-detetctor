import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from main import (PlateInference, _DebugRecognizer, _finalize_acceptance,
                  _route_ocr, _reconcile_ocr, _evaluate_ocr, parse_args)
from src.recognizer import OCRResult, PARSeqRecognizer
from src.row_detector import _RowCandidate, _select_registration_rows


class CaptureRecognizer:
    def __init__(self, normalization=None):
        self.input_normalization = normalization
        self.images = []

    def recognize(self, image):
        self.images.append(image)
        return OCRResult('MH१२AB१२३४', 'MH१२AB१२३४', .99)


class RealInferenceTests(unittest.TestCase):
    def test_header_fallback_does_not_override_usable_registration_row(self):
        row = _evaluate_ocr(OCRResult('२३५७', '२३५७', .82), .8)
        direct = _evaluate_ocr(OCRResult('MH12AB1234', 'MH12AB1234', .99), .8)
        result = _reconcile_ocr(row, direct, header_filtered=True)
        self.assertEqual(result.normalized_text, '२३५७')
        self.assertEqual(result.final_status, 'UNCERTAIN')

    def test_v2_default_and_override(self):
        with patch.object(sys, 'argv', ['main.py', '--image', 'x.png']):
            self.assertEqual(parse_args().mixed_ocr_weights.parts[-3:], ('models', 'ocr_mixed_v2', 'best.pt'))
        with patch.object(sys, 'argv', ['main.py', '--image', 'x.png', '--mixed-ocr-weights', 'custom.pt']):
            self.assertEqual(parse_args().mixed_ocr_weights, Path('custom.pt'))

    def test_normalized_models_receive_raw_rgb_legacy_keeps_preprocess(self):
        raw, enhanced = Image.new('RGB', (90, 30), 'gray'), Image.new('RGB', (144, 48), 'white')
        latin, dev, mixed = CaptureRecognizer(), CaptureRecognizer('doctr_parseq'), CaptureRecognizer('doctr_parseq')
        route = _route_ocr(enhanced, latin, dev, .8, mixed, raw)
        self.assertIs(latin.images[0], enhanced)
        self.assertIs(dev.images[0], raw)
        self.assertIs(mixed.images[0], raw)
        self.assertEqual(route.mixed_prediction, 'MH१२AB१२३४')

    def test_tensor_matches_training(self):
        import torch
        from scripts.train_ocr import OCRDataset, OCRRecord
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.png'
            Image.fromarray(np.random.default_rng(8).integers(0, 256, (47, 231, 3), dtype=np.uint8)).save(path)
            expected = OCRDataset([OCRRecord(path, 'MH१२AB१२३४')], False, True)[0][0]
            adapter = PARSeqRecognizer.__new__(PARSeqRecognizer)
            adapter.torch, adapter.device, adapter.input_normalization = torch, 'cpu', 'doctr_parseq'
            with Image.open(path) as image:
                actual = adapter._tensor(image)[0]
            self.assertTrue(torch.equal(expected, actual))

    def test_header_and_duplicate_do_not_replace_large_row(self):
        image = Image.new('RGB', (200, 100))
        main = _RowCandidate(0, image, 100, 60, 180, 40, 0, .90, (10,40,190,80))
        duplicate = _RowCandidate(1, image, 100, 60, 160, 38, 0, .80, (20,41,180,79))
        header = _RowCandidate(2, image, 100, 20, 80, 12, 0, .95, (60,14,140,26))
        selected, ignored = _select_registration_rows([main, duplicate, header], image.size)
        self.assertEqual([row.index for row in selected], [0])
        self.assertEqual(len(ignored), 2)

    def test_model_disagreement_cannot_be_promoted_to_accepted(self):
        result = PlateInference(1, 'MH12AB1234', (0,0,200,50), .95,
            ocr_confidence=.999, normalized_text='MH12AB1234',
            validation_status='VALID_FORMAT', final_status='UNCERTAIN')
        _finalize_acceptance(result)
        self.assertEqual(result.final_status, 'REVIEW_REQUIRED')

    def test_debug_observes_actual_calls_without_changing_prediction(self):
        result = PlateInference(1, '', (0,0,200,50), .9)
        model = CaptureRecognizer('doctr_parseq')
        wrapped = _DebugRecognizer(model, 'mixed', result, .8)
        prediction = wrapped.recognize(Image.new('RGB', (200,50), 'white'))
        self.assertEqual(prediction.text, 'MH१२AB१२३४')
        self.assertEqual(result.ocr_calls[0]['script'], 'mixed')
        self.assertEqual(result.debug_images['ocr_call_01_mixed_final.png'].size, (128,32))
        self.assertEqual(len(model.images), 1)


if __name__ == '__main__':
    unittest.main()
