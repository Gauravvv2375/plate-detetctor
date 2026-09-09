"""Extract header pixels and record only explicitly supplied, human-verified labels."""
import argparse
import csv
import hashlib
import json
import sys
import unicodedata
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.header_ocr import tighten_header
from src.preprocess import load_image
from src.utils import PROJECT_ROOT


def store_sample(root, crop, source, text=None):
    root = Path(root)
    if text is not None:
        text = unicodedata.normalize('NFC', text).strip()
        if not text or any(unicodedata.category(c).startswith('C') for c in text):
            raise ValueError('Supply a nonempty manually verified label without control characters')
    for directory in ['images', 'unlabeled']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    # One lock covers manifest, image and provenance changes by this utility.
    import fcntl
    with (root / '.labeling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = root / 'labels.csv'
        if not manifest.exists():
            manifest.write_text('image,text\n', encoding='utf-8')
        name = f'header_{uuid.uuid4().hex}.png'
        relative = Path('images' if text is not None else 'unlabeled') / name
        crop.convert('RGB').save(root / relative)
        info_path = root / 'dataset_info.json'
        info = json.loads(info_path.read_text()) if info_path.exists() else {'version':1,'samples':{}}
        info['samples'][relative.as_posix()] = {'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'source': str(Path(source).resolve()), 'verified': text is not None,
            'crop_sha256': hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            'label_source': 'manual_cli' if text is not None else 'unlabeled', 'text':text}
        if text is not None:
            with manifest.open('a',encoding='utf-8',newline='') as stream:
                csv.writer(stream).writerow([relative.as_posix(),text])
        with manifest.open(encoding='utf-8',newline='') as stream:
            rows = list(csv.DictReader(stream))
        info['verified_samples'] = len(rows)
        info['unlabeled_samples'] = sum(not x['verified'] for x in info['samples'].values())
        (root/'charset.txt').write_text(''.join(c+'\n' for c in sorted(set(''.join(row['text'] for row in rows)))), encoding='utf-8')
        info_path.write_text(json.dumps(info,ensure_ascii=False,indent=2), encoding='utf-8')
    return root / relative


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text',help='Exact manually verified transcription; never an OCR guess')
    group.add_argument('--extract-only',action='store_true')
    parser.add_argument('--header-index',type=int,help='Zero-based extracted header index when several are found')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'data/real_header_ocr_dataset')
    args=parser.parse_args()
    from src.detector import PlateDetector
    from src.row_detector import RowDetector
    image=load_image(args.image)
    detector=PlateDetector(PROJECT_ROOT/'models/plate_detector/best.pt',args.device)
    rows=RowDetector(PROJECT_ROOT/'models/row_detector/best.pt',args.device)
    crops=[]
    for plate in detector.detect(image):
        result=rows.process(plate.crop)
        for row in result.classified_rows:
            if row['role']=='HEADER':
                crops.append(tighten_header(result.region_images[row['index']])[0])
    if args.header_index is not None:
        if not 0 <= args.header_index < len(crops):
            parser.error('header-index is outside detected headers')
        crops=[crops[args.header_index]]
    if not crops:
        parser.error('No header detected; no label or crop was fabricated')
    if args.text is not None and len(crops)!=1:
        parser.error('Several headers detected; choose --header-index before supplying a verified label')
    for crop in crops:
        print(store_sample(args.output_dir,crop,args.image,args.text))
    print('Dataset preparation complete. More verified real samples are recommended before production fine-tuning.')


if __name__=='__main__':
    main()
