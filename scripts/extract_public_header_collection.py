"""Use existing detectors on visually selected public sources; no OCR labels."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.detector import PlateDetector
from src.row_detector import RowDetector
from src.header_ocr import tighten_header
from src.preprocess import load_image

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'data/real_header_ocr_dataset'

def main():
    decisions=json.loads((ROOT/'collection_visual_decisions.json').read_text())
    audit={str(x['id']):x for x in json.loads((ROOT/'collection_download_audit.json').read_text())}
    (ROOT/'source_images').mkdir(exist_ok=True)
    (ROOT/'extraction_staging').mkdir(exist_ok=True)
    out=ROOT/'collection_extraction_audit.json'
    results=json.loads(out.read_text()) if out.exists() else {}
    detector=PlateDetector(PROJECT/'models/plate_detector/best.pt','cpu')
    rows=RowDetector(PROJECT/'models/row_detector/best.pt','cpu')
    for key in decisions['accepted']:
        if key in results:continue
        item=audit[key]
        src=ROOT/item['download']
        dest=ROOT/'source_images'/f'source_{int(key):06d}{src.suffix}'
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError(f'Refusing overwrite: {dest}')
        else:
            shutil.copyfile(src,dest)
        entry={'source_image':dest.relative_to(ROOT).as_posix(),'headers':[],'plates':[]}
        try:
            image=load_image(dest)
            plates=detector.detect(image)
            for p,plate in enumerate(plates):
                result=rows.process(plate.crop)
                entry['plates'].append({'box':plate.box,'confidence':plate.confidence,'rows':result.classified_rows})
                for row in result.classified_rows:
                    if row['role']!='HEADER':continue
                    crop,info=tighten_header(result.region_images[row['index']])
                    name=f'header_{int(key):06d}_p{p+1}_r{row["index"]}.png'
                    path=ROOT/'extraction_staging'/name
                    if path.exists():raise ValueError(f'Refusing overwrite: {path}')
                    crop.save(path)
                    entry['headers'].append({'crop':path.relative_to(ROOT).as_posix(),'size':list(crop.size),'refinement':info})
        except Exception as exc:
            entry['error']=str(exc)
        results[key]=entry
        out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(key,'plates',len(entry['plates']),'header candidates',len(entry['headers']),entry.get('error',''),flush=True)

if __name__=='__main__':main()
