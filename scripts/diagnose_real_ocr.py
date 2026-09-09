"""Compare frozen OCR checkpoints on explicit views of one real image; never train."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps
from src.detector import PlateDetector
from src.row_detector import RowDetector
from src.recognizer import PARSeqRecognizer
from src.preprocess import load_image, mild_ocr_preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--tight-box', type=int, nargs=4, required=True,
                        help='Human diagnostic crop in original-image coordinates; never used by inference')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = load_image(args.image)
    plate = PlateDetector(Path('models/plate_detector/best.pt'), args.device, .30).detect(original)[0]
    rows = RowDetector(Path('models/row_detector/best.pt'), args.device).process(plate.crop)
    tight = original.crop(tuple(args.tight_box))
    views = {'original': original, 'plate': plate.crop, 'rectified_row': rows.image,
             'current_row_preprocessed': mild_ocr_preprocess(rows.image),
             'current_direct_preprocessed': mild_ocr_preprocess(plate.crop),
             'tight_digits': tight,
             'tight_padded': ImageOps.pad(tight, (128, 32), method=Image.Resampling.BILINEAR, color='white'),
             'row_padded': ImageOps.pad(rows.image, (128, 32), method=Image.Resampling.BILINEAR, color='white')}
    if rows.selected_rows and 'bounds' in rows.selected_rows[0]:
        views['selected_row_raw'] = plate.crop.crop(tuple(map(round, rows.selected_rows[0]['bounds'])))
    report = {'image': str(args.image.resolve()), 'tight_box': args.tight_box,
              'checkpoints': {}, 'views': {}}
    for name, view in views.items():
        view.save(args.output_dir / f'{name}.png')
        view.convert('RGB').resize((128,32), Image.Resampling.BILINEAR).save(args.output_dir / f'{name}_final.png')
        report['views'][name] = {'size': view.size, 'predictions': {}}
    for name, directory in [('latin','ocr'), ('devanagari','ocr_devanagari'), ('mixed','ocr_mixed_v2')]:
        path = Path('models') / directory / 'best.pt'
        recognizer = PARSeqRecognizer(path, args.device)
        report['checkpoints'][name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'vocabulary': recognizer.vocabulary, 'normalization': recognizer.input_normalization}
        for view_name, view in views.items():
            prediction = recognizer.recognize(view)
            report['views'][view_name]['predictions'][name] = vars(prediction)
        del recognizer
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
