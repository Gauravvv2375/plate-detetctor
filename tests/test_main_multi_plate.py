from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import DetectionDiagnostics, _evaluate_ocr, _print_results, _route_ocr, _route_row_result, infer, infer_all, parse_args
from src.detector import PlateDetection
from src.recognizer import OCRResult
from src.row_detector import RowResult


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, **kwargs):
        return self.detections


class FakeRowDetector:
    def __init__(self, fail_color=None):
        self.colors = []
        self.fail_color = fail_color

    def process(self, crop):
        color = crop.getpixel((0, 0))
        self.colors.append(color)
        if color == self.fail_color:
            raise RuntimeError("controlled row failure")
        return RowResult(crop, 1, [0.0], 0.9)


class FakeRecognizer:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        text = next(self.texts)
        return OCRResult(text, text, 1.0)


def detection(box, color, confidence=0.9):
    crop = Image.new("RGB", (40, 20), color)
    draw = ImageDraw.Draw(crop)
    for index, left in enumerate(range(8, 31, 5)):
        draw.rectangle((left, 5, left + 2, 14), fill="white" if index % 2 == 0 else "black")
    return PlateDetection(box, confidence, crop)


class MultiPlateInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "vehicles.jpg"
        Image.new("RGB", (240, 180), "white").save(self.image)

    def tearDown(self):
        self.temporary.cleanup()

    def test_processes_all_plates_in_reading_order_and_saves_each_stage(self):
        blue, green, red, yellow = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)
        detector = FakeDetector([
            detection((15, 110, 55, 130), red),
            detection((110, 12, 150, 32), green),
            detection((120, 115, 160, 135), yellow),
            detection((10, 10, 50, 30), blue),
        ])
        row_detector = FakeRowDetector()
        expected_texts = ["MH46BK1902", "KA03MR4588", "DL01AB1234", "TN42R2697"]
        recognizer = FakeRecognizer(expected_texts)
        output = self.root / "outputs"

        results = infer_all(self.image, detector, row_detector, recognizer, 0.35, output)

        self.assertEqual([result.text for result in results], expected_texts)
        self.assertEqual(row_detector.colors, [blue, green, red, yellow])
        for directory in ("detections", "crops", "rectified", "stitched"):
            files = list((output / directory).glob("*.png"))
            self.assertEqual(len(files), 4)
            self.assertEqual(
                {f"plate_{index:03d}" for index in range(1, 5)},
                {path.stem.rsplit("_", 2)[-2] + "_" + path.stem.rsplit("_", 1)[-1] for path in files},
            )

    def test_one_plate_failure_does_not_stop_remaining_plates(self):
        failed, good = (255, 0, 0), (0, 0, 255)
        detector = FakeDetector([
            detection((10, 10, 50, 30), failed),
            detection((100, 10, 140, 30), good),
        ])
        row_detector = FakeRowDetector(fail_color=failed)
        recognizer = FakeRecognizer(["MH46BK1902", "KA03MR4588"])
        output = self.root / "outputs"

        results = infer_all(self.image, detector, row_detector, recognizer, 0.35, output)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "MH46BK1902")
        self.assertIsNone(results[0].error)
        self.assertEqual(results[1].text, "KA03MR4588")
        self.assertEqual(recognizer.calls, 2)
        self.assertEqual(len(list((output / "detections").glob("*.png"))), 2)
        self.assertEqual(len(list((output / "crops").glob("*.png"))), 2)
        self.assertEqual(len(list((output / "rectified").glob("*.png"))), 1)
        self.assertEqual(len(list((output / "stitched").glob("*.png"))), 2)

    def test_unreadable_ocr_saves_no_debug_artifacts(self):
        output = self.root / "outputs"
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 0, 0))]),
            FakeRowDetector(),
            FakeRecognizer(["", ""]),
            0.35,
            output,
        )

        self.assertEqual(results[0].text, "PLATE_UNREADABLE")
        self.assertEqual(results[0].final_status, "REJECTED")
        self.assertFalse(output.exists())

    def test_invalid_ocr_text_is_unreadable_and_saves_no_artifacts(self):
        output = self.root / "outputs"
        recognizer = FakeRecognizer(["HONDA", "HONDA"])
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 0, 0))]),
            FakeRowDetector(),
            recognizer,
            0.35,
            output,
        )

        self.assertEqual(results[0].text, "PLATE_UNREADABLE")
        self.assertEqual(results[0].final_status, "REJECTED")
        self.assertEqual(recognizer.calls, 2)
        self.assertFalse(output.exists())

    def test_detector_crop_always_reaches_ocr_even_when_visually_blank(self):
        logo = PlateDetection((10, 10, 50, 30), 0.95, Image.new("RGB", (40, 20), "gray"))
        row_detector = FakeRowDetector()
        recognizer = FakeRecognizer(["HONDA", "HONDA"])

        results = infer_all(self.image, FakeDetector([logo]), row_detector, recognizer, 0.35)

        self.assertEqual(results[0].text, "PLATE_UNREADABLE")
        self.assertEqual(len(row_detector.colors), 1)
        self.assertEqual(recognizer.calls, 2)

    def test_low_confidence_plate_text_saves_no_debug_artifacts(self):
        class LowConfidenceRecognizer:
            def recognize(self, image):
                return OCRResult("MH46BK1902", "MH46BK1902", 0.2)

        output = self.root / "outputs"
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 0, 0))]),
            FakeRowDetector(),
            LowConfidenceRecognizer(),
            0.35,
            output,
        )

        self.assertEqual(results[0].text, "MH46BK1902")
        self.assertEqual(results[0].final_status, "LOW_CONFIDENCE")
        self.assertFalse(output.exists())

    def test_overlapping_duplicates_keep_highest_confidence_without_suppressing_nearby_plate(self):
        duplicate_low = detection((12, 11, 52, 31), (255, 0, 0), 0.70)
        duplicate_high = detection((10, 10, 50, 30), (0, 255, 0), 0.98)
        nearby_plate = detection((48, 10, 88, 30), (0, 0, 255), 0.90)
        detector = FakeDetector([duplicate_low, nearby_plate, duplicate_high])
        row_detector = FakeRowDetector()

        results = infer_all(
            self.image,
            detector,
            row_detector,
            FakeRecognizer(["MH46BK1902", "KA03MR4588"]),
            0.35,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([result.text for result in results], ["MH46BK1902", "KA03MR4588"])
        self.assertEqual(results[0].detection_confidence, 0.98)
        self.assertEqual(row_detector.colors, [(0, 255, 0), (0, 0, 255)])

    def test_nested_duplicate_missed_by_iou_nms_is_removed(self):
        outer = detection((10, 10, 110, 50), (0, 255, 0), 0.98)
        inner = detection((25, 15, 95, 45), (255, 0, 0), 0.80)
        row_detector = FakeRowDetector()
        recognizer = FakeRecognizer(["MH46BK1902"])

        results = infer_all(self.image, FakeDetector([inner, outer]), row_detector, recognizer, 0.35)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].detection_confidence, 0.98)
        self.assertEqual(recognizer.calls, 1)

    def test_partial_crop_still_reaches_ocr(self):
        partial = detection((0, 20, 40, 40), (255, 255, 0), 0.95)
        row_detector = FakeRowDetector()
        recognizer = FakeRecognizer(["MH46BK1902"])

        results = infer_all(self.image, FakeDetector([partial]), row_detector, recognizer, 0.35)

        self.assertEqual(results[0].text, "MH46BK1902")
        self.assertEqual(len(row_detector.colors), 1)
        self.assertEqual(recognizer.calls, 1)

    def test_abnormal_row_result_uses_fallback_ocr(self):
        class AbnormalRowDetector:
            def process(self, crop):
                return RowResult(crop, 3, [0.0, 0.0, 0.0], 0.95)

        recognizer = FakeRecognizer(["MH46BK1902"])
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (240, 240, 240))]),
            AbnormalRowDetector(),
            recognizer,
            0.35,
        )

        self.assertEqual(results[0].text, "MH46BK1902")
        self.assertEqual(recognizer.calls, 1)

    def test_ranked_row_result_does_not_reject_original_three_detections(self):
        class RankedRowDetector:
            def process(self, crop):
                return RowResult(
                    crop,
                    1,
                    [0.0],
                    0.90,
                    detected_rows=3,
                    selected_rows=[{"index": 0, "reason": "selected registration row"}],
                    ignored_rows=[
                        {"index": 1, "reason": "small header/decorative text"},
                        {"index": 2, "reason": "duplicate of row 1"},
                    ],
                    row_images=[crop],
                )

        recognizer = FakeRecognizer(["MH12AB1234"])
        result = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (240, 240, 240))]),
            RankedRowDetector(),
            recognizer,
            0.35,
        )[0]

        self.assertEqual(result.text, "MH12AB1234")
        self.assertEqual(result.detected_rows, 3)
        self.assertEqual(result.rows, 1)
        self.assertFalse(result.fallback_ocr_used)
        self.assertEqual(recognizer.calls, 1)

    def test_mixed_script_validation_preserves_visible_unicode(self):
        examples = [
            "MH12AB1234",
            "एमएच१२एबी१२३४",
            "MH१२AB१२३४",
            "एमएच12AB१२३४",
            "MH १२ एबी 1234",
            "MH १२ AB १२३४",
            "एमएच 12 एबी 1234",
            "MH १२ एबी 1234",
            "एमएच 12 AB १२३४",
        ]
        for text in examples:
            with self.subTest(text=text):
                evaluation = _evaluate_ocr(OCRResult(text, text, 0.99), 0.80)
                self.assertEqual(evaluation.normalized_text, text)
                self.assertIn(evaluation.validation_status, {"VALID_FORMAT", "POTENTIAL_FORMAT"})
                self.assertNotEqual(evaluation.final_status, "INVALID_FORMAT")

    def test_mixed_checkpoint_candidate_preserves_true_mixed_prediction(self):
        class ConfidenceRecognizer:
            def __init__(self, text, confidence):
                self.result = OCRResult(text, text, confidence)

            def recognize(self, image):
                return self.result

        routing = _route_ocr(
            Image.new("RGB", (120, 32), "white"),
            ConfidenceRecognizer("MH12AB1234", 0.90),
            ConfidenceRecognizer("एमएच१२एबी१२३४", 0.91),
            0.80,
            ConfidenceRecognizer("MH१२AB१२३४", 0.92),
        )

        self.assertEqual(routing.evaluation.normalized_text, "MH१२AB१२३४")
        self.assertEqual(routing.mixed_prediction, "MH१२AB१२३४")
        self.assertIn("three-model OCR", routing.strategy)

    def test_close_three_model_disagreement_requires_review(self):
        class ConfidenceRecognizer:
            def __init__(self, text, confidence):
                self.result = OCRResult(text, text, confidence)

            def recognize(self, image):
                return self.result

        routing = _route_ocr(
            Image.new("RGB", (120, 32), "white"),
            ConfidenceRecognizer("MH12AB1234", 0.90),
            ConfidenceRecognizer("", 0.90),
            0.80,
            ConfidenceRecognizer("MH12AB1235", 0.91),
        )

        self.assertEqual(routing.evaluation.final_status, "UNCERTAIN")
        self.assertIn("manual review", routing.evaluation.reason)

    def test_two_rows_can_route_to_different_script_models(self):
        row = Image.new("RGB", (80, 24), "white")
        row_result = RowResult(row, 2, [0.0, 0.0], 0.9, row_images=[row, row])
        latin = FakeRecognizer(["MH12", "HONDA"])
        devanagari = FakeRecognizer(["शब्द", "१२३४"])

        routing = _route_row_result(row_result, row, latin, devanagari, 0.80)

        self.assertEqual(routing.evaluation.normalized_text, "MH12 १२३४")
        self.assertEqual(routing.evaluation.validation_status, "POTENTIAL_FORMAT")
        self.assertIn("per-row OCR routing", routing.strategy)

    def test_zero_rows_use_fallback_but_low_row_confidence_still_uses_returned_image(self):
        class InvalidRowDetector:
            def __init__(self, result):
                self.result = result

            def process(self, crop):
                return self.result

        crop = detection((10, 10, 50, 30), (255, 255, 0)).crop
        row_cases = [
            (RowResult(crop, 0, [], 0.0), False, True),
            (RowResult(crop, 1, [0.0], 0.10), True, False),
        ]
        for row_result, row_succeeded, fallback_used in row_cases:
            with self.subTest(rows=row_result.rows, confidence=row_result.confidence):
                recognizer = FakeRecognizer(["MH46BK1902"])
                results = infer_all(
                    self.image,
                    FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
                    InvalidRowDetector(row_result),
                    recognizer,
                    0.35,
                )

                self.assertEqual(results[0].text, "MH46BK1902")
                self.assertEqual(recognizer.calls, 1)
                self.assertEqual(results[0].row_detector_succeeded, row_succeeded)
                self.assertEqual(results[0].fallback_ocr_used, fallback_used)

    def test_no_obb_sentinel_runs_ocr_on_returned_original_crop(self):
        crop = detection((10, 10, 50, 30), (255, 255, 0)).crop

        class NoObbRowDetector:
            def process(self, plate):
                return RowResult(plate, 1, [0.0], 0.0)

        recognizer = FakeRecognizer(["MH46BK1902"])
        results = infer_all(
            self.image,
            FakeDetector([PlateDetection((10, 10, 50, 30), 0.95, crop)]),
            NoObbRowDetector(),
            recognizer,
            0.35,
        )

        self.assertEqual(results[0].text, "MH46BK1902")
        self.assertEqual(results[0].ocr_confidence, 1.0)
        self.assertEqual(recognizer.calls, 1)
        self.assertFalse(results[0].row_detector_succeeded)
        self.assertTrue(results[0].fallback_ocr_used)

    def test_bharat_series_removes_artificial_state_prefix(self):
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
            FakeRowDetector(),
            FakeRecognizer(["OD22BH6517A"]),
            0.35,
        )

        self.assertEqual(results[0].text, "22BH6517A")
        self.assertEqual(results[0].raw_ocr, "OD22BH6517A")
        self.assertEqual(results[0].normalized_text, "22BH6517A")
        self.assertTrue(results[0].postprocessing_applied)

    def test_short_visible_ocr_is_never_autocompleted(self):
        detector = FakeDetector([
            detection((10, 10, 50, 30), (255, 255, 0)),
            detection((100, 10, 140, 30), (0, 0, 255)),
        ])
        raw_texts = ["MH04BQ336", "MH11AW23"]

        results = infer_all(
            self.image,
            detector,
            FakeRowDetector(),
            FakeRecognizer([raw_texts[0], raw_texts[0], raw_texts[1], raw_texts[1]]),
            0.35,
        )

        self.assertEqual([result.raw_ocr for result in results], raw_texts)
        self.assertEqual([result.normalized_text for result in results], raw_texts)
        self.assertEqual([result.text for result in results], raw_texts)
        self.assertEqual([result.final_status for result in results], ["PARTIAL_VISIBLE", "PARTIAL_VISIBLE"])
        self.assertTrue(all(not result.postprocessing_applied for result in results))

    def test_two_view_ocr_removes_only_disputed_trailing_characters(self):
        class SequenceRecognizer:
            def __init__(self, predictions):
                self.predictions = iter(predictions)

            def recognize(self, image):
                return next(self.predictions)

        prefix_results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
            FakeRowDetector(),
            SequenceRecognizer([
                OCRResult("MH04BQ3361", "MH04BQ3361", 0.976),
                OCRResult("MH04BQ336", "MH04BQ336", 0.999),
            ]),
            0.35,
        )
        disputed_results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
            FakeRowDetector(),
            SequenceRecognizer([
                OCRResult("MH11AW231", "MH11AW231", 0.997),
                OCRResult("MH11AW237", "MH11AW237", 0.958),
            ]),
            0.35,
        )

        self.assertEqual(prefix_results[0].text, "MH04BQ336")
        self.assertEqual(disputed_results[0].text, "MH11AW23")
        self.assertEqual(prefix_results[0].final_status, "PARTIAL_VISIBLE")
        self.assertEqual(disputed_results[0].final_status, "PARTIAL_VISIBLE")

    def test_standard_state_prefixed_registrations_remain_unchanged(self):
        detector = FakeDetector([
            detection((10, 10, 50, 30), (255, 255, 0)),
            detection((100, 10, 140, 30), (0, 0, 255)),
        ])

        results = infer_all(
            self.image,
            detector,
            FakeRowDetector(),
            FakeRecognizer(["MH12AB1234", "DL1CAB1234"]),
            0.35,
        )

        self.assertEqual([result.text for result in results], ["MH12AB1234", "DL1CAB1234"])

    def test_format_valid_medium_confidence_two_line_plate_requires_review(self):
        class TwoRowDetector:
            def process(self, crop):
                return RowResult(crop, 2, [0.0, 0.0], 0.9)

        class MediumConfidenceRecognizer:
            def recognize(self, image):
                return OCRResult("MH12IN1428", "MH12IN1428", 0.92)

        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (240, 170, 20))]),
            TwoRowDetector(),
            MediumConfidenceRecognizer(),
            0.80,
        )
        result = results[0]

        self.assertEqual(result.validation_status, "VALID_FORMAT")
        self.assertEqual(result.final_status, "REVIEW_REQUIRED")
        self.assertFalse(result.accepted)
        self.assertTrue(result.risk_flags["low_ocr_confidence"])
        self.assertTrue(result.risk_flags["two_line_plate"])
        self.assertTrue(result.risk_flags["possible_character_confusion"])
        self.assertTrue(result.risk_flags["night_or_glare_candidate"])
        self.assertIn("format valid but OCR uncertain", result.status_reason)

    def test_high_confidence_clean_bike_plate_is_accepted(self):
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (240, 240, 240))]),
            FakeRowDetector(),
            FakeRecognizer(["MH12NN0456"]),
            0.80,
        )
        result = results[0]

        self.assertEqual(result.text, "MH12NN0456")
        self.assertEqual(result.validation_status, "VALID_FORMAT")
        self.assertEqual(result.final_status, "ACCEPTED")
        self.assertTrue(result.accepted)

    def test_plate_is_unreadable_only_after_row_and_fallback_ocr_fail(self):
        recognizer = FakeRecognizer(["HONDA", "SUZUKI"])

        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
            FakeRowDetector(),
            recognizer,
            0.35,
        )

        self.assertEqual(results[0].text, "PLATE_UNREADABLE")
        self.assertEqual(recognizer.calls, 2)
        self.assertIn("Row path:", results[0].error)
        self.assertIn("fallback path:", results[0].error)

    def test_verbose_output_reports_row_and_fallback_status(self):
        zero_rows = RowResult(detection((10, 10, 50, 30), (255, 255, 0)).crop, 1, [0.0], 0.0)

        class ZeroRowDetector:
            def process(self, crop):
                return zero_rows

        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
            ZeroRowDetector(),
            FakeRecognizer(["MH46BK1902"]),
            0.35,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            _print_results(self.image, results, True)

        text = output.getvalue()
        self.assertIn("Text: MH46BK1902", text)
        self.assertIn("Detector: SUCCESS", text)
        self.assertIn("Row Detector:\nNo OBB found.\nUsing original crop.", text)
        self.assertIn("OCR:\nRUNNING", text)
        self.assertIn("OCR Result:\nMH46BK1902", text)
        self.assertIn("Validation:\nVALID_FORMAT", text)
        self.assertIn("Fallback OCR: USED", text)

    def test_legacy_infer_returns_first_string_but_processes_every_plate(self):
        detector = FakeDetector([
            detection((10, 10, 50, 30), (255, 0, 0)),
            detection((100, 10, 140, 30), (0, 0, 255)),
        ])
        recognizer = FakeRecognizer(["MH46BK1902", "KA03MR4588"])

        result = infer(self.image, detector, FakeRowDetector(), recognizer, 0.35, False)

        self.assertEqual(result, "MH46BK1902")
        self.assertEqual(recognizer.calls, 2)

    def test_multi_plate_terminal_output_includes_every_result(self):
        detector = FakeDetector([
            detection((10, 10, 50, 30), (255, 0, 0), 0.98),
            detection((100, 10, 140, 30), (0, 0, 255), 0.96),
        ])
        results = infer_all(
            self.image,
            detector,
            FakeRowDetector(),
            FakeRecognizer(["MH46BK1902", "KA03MR4588"]),
            0.35,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            _print_results(self.image, results, False)
        text = output.getvalue()

        self.assertIn("Accepted Plates: 2", text)
        self.assertIn("Review Candidates: 0", text)
        self.assertIn("Rejected Candidates: 0", text)
        self.assertIn("Plate 1", text)
        self.assertIn("MH46BK1902", text)
        self.assertIn("Plate 2", text)
        self.assertIn("KA03MR4588", text)

    def test_detector_fallback_runs_only_when_primary_is_empty(self):
        class RecordingDetector:
            def __init__(self, primary, fallback):
                self.primary = primary
                self.fallback = fallback
                self.calls = []

            def detect(self, image, **kwargs):
                self.calls.append(kwargs)
                return self.fallback if kwargs else self.primary

        primary_detector = RecordingDetector(
            [detection((10, 10, 50, 30), (255, 255, 0))],
            [detection((100, 10, 140, 30), (0, 0, 255))],
        )
        primary_results = infer_all(
            self.image,
            primary_detector,
            FakeRowDetector(),
            FakeRecognizer(["MH12AB1234"]),
            0.35,
            detector_fallback=True,
        )
        self.assertEqual(len(primary_detector.calls), 1)
        self.assertEqual(primary_results[0].detection_source, "primary")

        fallback_detector = RecordingDetector([], [detection((100, 10, 140, 30), (0, 0, 255))])
        diagnostics = DetectionDiagnostics()
        fallback_results = infer_all(
            self.image,
            fallback_detector,
            FakeRowDetector(),
            FakeRecognizer(["DL1CAB1234"]),
            0.35,
            detection_diagnostics=diagnostics,
            detector_fallback=True,
        )
        self.assertEqual(len(fallback_detector.calls), 2)
        self.assertEqual(fallback_results[0].detection_source, "fallback")
        self.assertEqual(diagnostics.primary_count, 0)
        self.assertEqual(diagnostics.fallback_count, 1)

    def test_tiled_detection_converts_coordinates_to_full_image(self):
        Image.new("RGB", (300, 200), "white").save(self.image)

        class TileDetector:
            def __init__(self):
                self.calls = 0

            def detect(self, image, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    return [detection((10, 20, 50, 40), (255, 255, 0), 0.9)]
                return []

        diagnostics = DetectionDiagnostics()
        results = infer_all(
            self.image,
            TileDetector(),
            FakeRowDetector(),
            FakeRecognizer(["MH12AB1234"]),
            0.35,
            detection_diagnostics=diagnostics,
            detector_tile_fallback=True,
            detector_tile_size=128,
            detector_tile_overlap=0.0,
        )

        self.assertEqual(results[0].box, (138, 20, 178, 40))
        self.assertEqual(results[0].detection_source, "tile")
        self.assertEqual(diagnostics.tile_count, 1)

    def test_save_debug_writes_structured_per_image_bundle(self):
        diagnostics = DetectionDiagnostics()
        debug_root = self.root / "outputs"
        results = infer_all(
            self.image,
            FakeDetector([detection((10, 10, 50, 30), (240, 240, 240))]),
            FakeRowDetector(),
            FakeRecognizer(["MH12AB1234"]),
            0.35,
            detection_diagnostics=diagnostics,
            debug_output_dir=debug_root,
        )

        bundle = debug_root / "debug" / self.image.stem
        self.assertEqual(results[0].text, "MH12AB1234")
        self.assertTrue((bundle / "final_detections.png").is_file())
        self.assertTrue((bundle / "plate_001_detector_crop.png").is_file())
        self.assertTrue((bundle / "plate_001_row_output.png").is_file())
        self.assertTrue((bundle / "plate_001_stitched.png").is_file())
        self.assertTrue((bundle / "plate_001_final_ocr_input.png").is_file())
        report = json.loads((bundle / "results.json").read_text())
        self.assertEqual(report["plates"][0]["raw_ocr_text"], "MH12AB1234")
        self.assertEqual(report["plates"][0]["detection_source"], "primary")
        self.assertTrue(report["plates"][0]["accepted"])
        self.assertEqual(report["plates"][0]["format_validation"], "VALID_FORMAT")
        self.assertEqual(report["plates"][0]["final_status"], "ACCEPTED")
        self.assertIn("risk_flags", report["plates"][0])

    def test_no_detection_verbose_output_saves_nothing(self):
        output_dir = self.root / "outputs"
        results = infer_all(self.image, FakeDetector([]), FakeRowDetector(), FakeRecognizer([]), 0.35, output_dir)
        output = io.StringIO()
        with redirect_stdout(output):
            _print_results(self.image, results, True)

        self.assertEqual(results, [])
        self.assertIn("Accepted Plates: 0", output.getvalue())
        self.assertIn("Review Candidates: 0", output.getvalue())
        self.assertIn("Rejected Candidates: 0", output.getvalue())
        self.assertFalse(output_dir.exists())

    def test_save_results_is_opt_in_and_save_debug_remains_compatible(self):
        with patch.object(sys, "argv", ["main.py", "--image", str(self.image)]):
            args = parse_args()
            self.assertFalse(args.save_results)
            self.assertEqual(args.ocr_conf, 0.80)
            self.assertEqual(args.det_conf, 0.30)
            self.assertTrue(args.det_fallback)
            self.assertTrue(args.det_tile_fallback)
        with patch.object(sys, "argv", ["main.py", "--image", str(self.image), "--save-results"]):
            self.assertTrue(parse_args().save_results)
        with patch.object(sys, "argv", ["main.py", "--image", str(self.image), "--save-debug"]):
            args = parse_args()
            self.assertFalse(args.save_results)
            self.assertTrue(args.save_debug)


if __name__ == "__main__":
    unittest.main()
