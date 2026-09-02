from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _print_results, infer, infer_all, parse_args
from src.detector import PlateDetection
from src.recognizer import OCRResult
from src.row_detector import RowResult


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
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
        return OCRResult(text, text, 0.95)


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

        self.assertEqual(results[0].text, "PLATE_UNREADABLE")
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
            FakeDetector([detection((10, 10, 50, 30), (255, 255, 0))]),
            AbnormalRowDetector(),
            recognizer,
            0.35,
        )

        self.assertEqual(results[0].text, "MH46BK1902")
        self.assertEqual(recognizer.calls, 1)

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
        self.assertEqual(results[0].ocr_confidence, 0.95)
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
        self.assertIn("Validation:\nPASSED", text)
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

        self.assertIn("Detected Plates: 2", text)
        self.assertIn("Plate 1", text)
        self.assertIn("MH46BK1902", text)
        self.assertIn("Plate 2", text)
        self.assertIn("KA03MR4588", text)

    def test_no_detection_verbose_output_saves_nothing(self):
        output_dir = self.root / "outputs"
        results = infer_all(self.image, FakeDetector([]), FakeRowDetector(), FakeRecognizer([]), 0.35, output_dir)
        output = io.StringIO()
        with redirect_stdout(output):
            _print_results(self.image, results, True)

        self.assertEqual(results, [])
        self.assertIn("Detected Plates: 0", output.getvalue())
        self.assertFalse(output_dir.exists())

    def test_save_results_is_opt_in_and_save_debug_remains_compatible(self):
        with patch.object(sys, "argv", ["main.py", "--image", str(self.image)]):
            args = parse_args()
            self.assertFalse(args.save_results)
            self.assertEqual(args.ocr_conf, 0.80)
            self.assertEqual(args.det_conf, 0.30)
        with patch.object(sys, "argv", ["main.py", "--image", str(self.image), "--save-results"]):
            self.assertTrue(parse_args().save_results)
        with patch.object(sys, "argv", ["main.py", "--image", str(self.image), "--save-debug"]):
            self.assertTrue(parse_args().save_results)


if __name__ == "__main__":
    unittest.main()
