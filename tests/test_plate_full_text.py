import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image
from main import PlateInference, _complete_plate_text, _reading_order_text
from src.recognizer import OCRResult
from src.row_detector import RowDetector, RowResult, _RowCandidate, _select_registration_rows, _classify_rows


class HeaderRecognizer:
    def __init__(self, texts, confidence=.99):
        self.texts = iter(texts)
        self.confidence = confidence
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        text = next(self.texts)
        return OCRResult(text, text, self.confidence)


def fixture(header=True):
    image = Image.new('RGB', (180, 40), 'white')
    primary = _RowCandidate(0, image, 100, 65, 180, 40, 0, .9, (10,45,190,85))
    small = _RowCandidate(1, image, 100, 20, 85, 15, 0, .9, (57,12,142,27))
    candidates = [primary, small] if header else [primary]
    selected, ignored = _select_registration_rows(candidates, (200,100))
    row_result = RowResult(image, len(selected), [0]*len(selected), .9,
        selected_rows=[row.diagnostic('selected') for row in selected],
        ignored_rows=ignored, region_images=[image]*len(candidates),
        classified_rows=_classify_rows(selected, ignored, candidates))
    result = PlateInference(1, 'KA05CD6789', (0,0,200,100), .9,
        ocr_confidence=.91, final_status='REVIEW_REQUIRED', validation_status='VALID_FORMAT',
        row_result=row_result)
    return result


class FullTextTests(unittest.TestCase):
    def test_auxiliary_detection_merges_header_without_changing_registration(self):
        class Array:
            def __init__(self, value): self.value = value
            def cpu(self): return self
            def tolist(self): return self.value
        model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(obb=SimpleNamespace(
            xyxyxyxy=Array([[[20,10],[80,10],[80,25],[20,25]],
                           [[90,10],[150,10],[150,25],[90,25]],
                           [[10,45],[190,45],[190,85],[10,85]]]), conf=Array([.9,.8,.9])))])
        detector = RowDetector.__new__(RowDetector)
        detector.model, detector.device, detector.image_size, detector.confidence = model, 'cpu', 320, .25
        result = fixture(False).row_result
        image = Image.new('RGB', (200,100), 'white')
        primary = _RowCandidate(0,image,100,65,180,40,0,.9,(10,45,190,85),[[10,45],[190,45],[190,85],[10,85]])
        prior = result.image, result.rows, list(result.selected_rows)
        detector._recover_small_text(image, result, [primary])
        self.assertEqual((result.image,result.rows,result.selected_rows), prior)
        headers = [row for row in result.classified_rows if row['role']=='HEADER']
        self.assertEqual(len(headers),1)
        self.assertEqual(len(headers[0]['merged_from']),2)

    def test_headers_preserve_each_script_and_registration_state(self):
        for text in ['TRANSPORT', 'परिवहन', 'परिवहन Z9']:
            with self.subTest(text=text):
                result = fixture()
                before = result.text, result.ocr_confidence, result.final_status, result.validation_status
                _complete_plate_text(result, (HeaderRecognizer([text]), None, None), .8, False)
                self.assertEqual(result.header_text, text)
                self.assertEqual(result.registration_text, before[0])
                self.assertEqual(result.full_text, text + ' ' + before[0])
                self.assertEqual((result.text,result.ocr_confidence,result.final_status,result.validation_status), before)
                self.assertEqual(result.as_dict()['plate_number'], before[0])

    def test_no_header_does_not_call_ocr(self):
        result = fixture(False)
        model = HeaderRecognizer([])
        _complete_plate_text(result, (model,None,None), .8, False)
        self.assertEqual(result.header_text, '')
        self.assertEqual(result.full_text, result.registration_text)
        self.assertEqual(model.calls, 0)

    def test_low_confidence_header_omitted(self):
        result = fixture()
        _complete_plate_text(result, (HeaderRecognizer(['TEXT'], .3),None,None), .8, False)
        self.assertEqual(result.header_text, '')
        self.assertEqual(result.text_rows[1]['status'], 'UNREADABLE')

    def test_noise_and_duplicates_are_not_headers(self):
        result = fixture()
        result.row_result.classified_rows[1]['role'] = 'DECORATIVE_OR_NOISE'
        model = HeaderRecognizer([])
        _complete_plate_text(result, (model,None,None), .8, False)
        self.assertEqual(model.calls, 0)
        self.assertEqual(result.full_text, result.text)

    def test_two_registration_rows_remain_registration(self):
        image = Image.new('RGB', (200,100))
        candidates = [_RowCandidate(i,image,100,25+i*45,150,30,0,.9,(25,10+i*45,175,40+i*45)) for i in range(2)]
        selected, ignored = _select_registration_rows(candidates, image.size)
        classified = _classify_rows(selected,ignored,candidates)
        self.assertEqual({row['role'] for row in classified}, {'PRIMARY_REGISTRATION','SECONDARY_REGISTRATION'})

    def test_partially_overlapping_duplicate_row_contributes_only_once(self):
        image = Image.new('RGB', (200,100))
        stronger = _RowCandidate(0,image,60,55,100,30,1,.95,(10,40,110,70))
        duplicate = _RowCandidate(1,image,110,55,100,28,2,.85,(60,41,160,69))

        selected, ignored = _select_registration_rows([duplicate, stronger], image.size)

        self.assertEqual([row.index for row in selected], [0])
        self.assertEqual(len(ignored), 1)
        self.assertIn('duplicate of row 0', ignored[0]['reason'])

    def test_reading_order_groups_same_line_left_to_right(self):
        rows = [{'center':c,'size':[30,20],'text':t} for c,t in [([80,10],'B'),([20,12],'A'),([10,60],'C')]]
        self.assertEqual([row['text'] for row in _reading_order_text(rows)], ['A','B','C'])

    def test_header_failure_does_not_change_registration(self):
        result = fixture()
        _complete_plate_text(result, (HeaderRecognizer([]),None,None), .8, False)
        self.assertEqual(result.full_text, result.text)
        self.assertEqual(result.final_status, 'REVIEW_REQUIRED')

    def test_all_three_models_are_used_for_header(self):
        result = fixture()
        models = tuple(HeaderRecognizer(['परिवहन']) for _ in range(3))
        _complete_plate_text(result, models, .8, False)
        self.assertEqual([model.calls for model in models], [1,1,1])
        self.assertEqual(result.header_text, 'परिवहन')


if __name__ == '__main__':
    unittest.main()
