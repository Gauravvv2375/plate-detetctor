from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from main import (
    PlateInference,
    _evaluate_ocr,
    _finalize_acceptance,
    _indian_registration_format,
    _reconcile_ocr,
    _route_ocr,
    _route_row_result,
)
from src.recognizer import OCRResult
from src.row_detector import RowResult


class FixedRecognizer:
    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def recognize(self, _image):
        text, confidence = next(self.predictions)
        return OCRResult(text, text, confidence)


class OCRDecisionTests(unittest.TestCase):
    def setUp(self):
        self.row = Image.new("RGB", (100, 24), "white")

    def evaluate(self, text: str, confidence: float = 0.99):
        return _evaluate_ocr(OCRResult(text, text, confidence), 0.80)

    def five_character_row(self):
        image = Image.new("RGB", (120, 30), "white")
        draw = ImageDraw.Draw(image)
        for left in (8, 30, 52, 74, 96):
            draw.rectangle((left, 3, left + 8, 26), fill="black")
        return image

    def test_known_indian_registration_structures(self):
        examples = {
            "KA01AB1234": "STANDARD",
            "MH14EV3456": "STANDARD",
            "T0820AB1234CD": "TEMPORARY",
            "DLR012345": "DEALER",
            "KA01A1234TC0001": "TRADE_CERTIFICATE",
            "77CD1234": "DIPLOMATIC",
        }
        for text, expected_format in examples.items():
            with self.subTest(text=text):
                evaluation = self.evaluate(text)
                self.assertEqual(_indian_registration_format(text), expected_format)
                self.assertEqual(evaluation.validation_status, "VALID_FORMAT")
                self.assertEqual(evaluation.final_status, "FORMAT_CONFIDENT")

    def test_two_line_routing_joins_each_row_once(self):
        row_result = RowResult(
            self.row,
            2,
            [0.0, 0.0],
            0.95,
            row_images=[self.row, self.row],
        )
        routing = _route_row_result(
            row_result,
            self.row,
            FixedRecognizer([("RJ20P", 0.99), ("A1908", 0.99)]),
            None,
            0.80,
        )

        self.assertEqual(routing.evaluation.normalized_text, "RJ20P A1908")
        self.assertEqual(len(routing.row_predictions), 2)

    def test_row_fragments_do_not_reward_hallucinated_full_plate_shapes(self):
        row = self.five_character_row()
        row_result = RowResult(row, 2, [0.0, 0.0], 0.95, row_images=[row, row])
        routing = _route_row_result(
            row_result,
            row,
            FixedRecognizer([("RJ20PPJ2020", 0.90), ("AA19A1908", 0.92)]),
            FixedRecognizer([("", 0.99), ("", 0.99)]),
            0.80,
            FixedRecognizer([("RJ20P", 0.80), ("A1908", 0.80)]),
        )

        self.assertEqual(routing.evaluation.normalized_text, "RJ20P A1908")
        self.assertEqual([item["text"] for item in routing.row_predictions], ["RJ20P", "A1908"])
        self.assertNotIn("RJ20PPJ2020", routing.evaluation.normalized_text)

    def test_models_for_one_row_remain_alternative_candidates(self):
        routing = _route_ocr(
            self.five_character_row(),
            FixedRecognizer([("KA01AB1234", 0.95)]),
            FixedRecognizer([("कए०१अब१२३४", 0.96)]),
            0.80,
            FixedRecognizer([("KA01AC1234", 0.94)]),
        )

        alternatives = {"KA01AB1234", "कए०१अब१२३४", "KA01AC1234"}
        self.assertIn(routing.evaluation.normalized_text, alternatives)
        self.assertFalse(any(
            left + right == routing.evaluation.normalized_text
            for left in alternatives for right in alternatives if left != right
        ))

    def test_direct_crop_is_compared_not_appended(self):
        row = self.evaluate("KA01AB1234", 0.99)
        direct = self.evaluate("KA01AC1234", 0.98)

        selected = _reconcile_ocr(row, direct)

        self.assertIn(selected.normalized_text, {row.normalized_text, direct.normalized_text})
        self.assertEqual(selected.final_status, "UNCERTAIN")

    def test_formatting_normalization_cannot_duplicate_characters(self):
        evaluation = self.evaluate("  KA 01-AB 1234  ")
        self.assertEqual(evaluation.normalized_text, "KA 01-AB 1234")

    def test_single_row_routing_is_unchanged(self):
        row_result = RowResult(self.row, 1, [0.0], 0.95, row_images=[self.row])
        routing = _route_row_result(
            row_result,
            self.row,
            FixedRecognizer([("KA01AB1234", 0.99)]),
            None,
            0.80,
        )

        self.assertEqual(routing.evaluation.normalized_text, "KA01AB1234")
        self.assertEqual(routing.evaluation.final_status, "FORMAT_CONFIDENT")

    def test_structurally_valid_view_beats_extra_character_view(self):
        row = self.evaluate("RJ220P A1908", 0.999)
        direct = self.evaluate("RJ20P A1908", 0.94)

        result = _reconcile_ocr(row, direct)

        self.assertEqual(result.normalized_text, "RJ20P A1908")
        self.assertEqual(result.final_status, "UNCERTAIN")

    def test_high_confidence_invalid_candidate_does_not_beat_valid_latin(self):
        routing = _route_ocr(
            self.row,
            FixedRecognizer([("KA01AB1234", 0.91)]),
            FixedRecognizer([("क१२३४५", 0.999)]),
            0.80,
            FixedRecognizer([("KAA1AB1234", 0.995)]),
        )

        self.assertEqual(routing.evaluation.normalized_text, "KA01AB1234")
        self.assertEqual(routing.evaluation.validation_status, "VALID_FORMAT")

    def test_genuine_devanagari_agreement_remains_supported(self):
        visible = "एमएच१२एबी१२३४"
        routing = _route_ocr(
            self.row,
            FixedRecognizer([("MH12A81234", 0.82)]),
            FixedRecognizer([(visible, 0.96)]),
            0.80,
            FixedRecognizer([(visible, 0.95)]),
        )

        self.assertEqual(routing.evaluation.normalized_text, visible)
        self.assertNotEqual(routing.evaluation.final_status, "INVALID_FORMAT")

    def test_meaningful_row_direct_disagreement_requires_review(self):
        selected = _reconcile_ocr(
            self.evaluate("KA01AB1234", 0.99),
            self.evaluate("KA01AC1234", 0.99),
        )
        result = PlateInference(
            1,
            selected.normalized_text,
            (0, 0, 100, 30),
            0.95,
            ocr_confidence=selected.confidence,
            normalized_text=selected.normalized_text,
            validation_status=selected.validation_status,
            final_status=selected.final_status,
            row_direct_disagreement=True,
            debug_crop=Image.new("RGB", (100, 30), "white"),
        )

        _finalize_acceptance(result)

        self.assertEqual(result.final_status, "REVIEW_REQUIRED")

    def test_spacing_only_agreement_can_be_accepted(self):
        row = self.evaluate("KA 01 AB 1234", 0.999)
        direct = self.evaluate("KA01AB1234", 0.998)
        selected = _reconcile_ocr(row, direct)
        result = PlateInference(
            1,
            selected.normalized_text,
            (0, 0, 100, 30),
            0.95,
            ocr_confidence=selected.confidence,
            normalized_text=selected.normalized_text,
            validation_status=selected.validation_status,
            final_status=selected.final_status,
            debug_crop=Image.new("RGB", (100, 30), "white"),
        )

        _finalize_acceptance(result)

        self.assertEqual(result.final_status, "ACCEPTED")
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
