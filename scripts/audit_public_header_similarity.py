"""Flag whole-image perceptual duplicates for visual confirmation."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]/'data/real_header_ocr_dataset'

def phash(path):
    im=ImageOps.exif_transpose(Image.open(path)).convert('L').resize((32,32))
    values=cv2.dct(np.asarray(im,dtype=np.float32))[:8,:8].flatten()[1:]
    return values>np.median(values)

def main():
    records=[x for x in json.loads((ROOT/'collection_download_audit.json').read_text()) if x.get('download') and (ROOT/x['download']).exists()]
    hashes=[phash(ROOT/x['download']) for x in records]
    pairs=[]
    for i in range(len(records)):
        for j in range(i):
            distance=int(np.count_nonzero(hashes[i]!=hashes[j]))
            if distance<=10:
                pairs.append({'ids':[records[j]['id'],records[i]['id']],'phash_hamming':distance,'needs_visual_confirmation':True})
    report={'algorithm':'63-bit DCT perceptual hash; distance <= 10 proposes review, never automatic acceptance',
            'hashes':{str(x['id']):''.join(map(str,h.astype(int))) for x,h in zip(records,hashes)},'pairs':pairs}
    (ROOT/'collection_similarity_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(pairs))

if __name__=='__main__':main()
