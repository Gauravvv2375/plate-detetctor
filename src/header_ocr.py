"""Header-only crop refinement and isolated checkpoint loading."""
from pathlib import Path
import numpy as np
from PIL import Image


def tighten_header(image):
    """Find a coherent text band; leave ambiguous crops unchanged.

    Thresholding is for geometry only. Returned pixels retain their original RGB.
    Isolated edge components (including separated bolts) are not joined to text.
    """
    import cv2
    image = image.convert('RGB')
    gray = np.asarray(image.convert('L'))
    h, w = gray.shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [tuple(map(int, stats[i])) for i in range(1, count)
             if stats[i,4] >= max(3, h*h*.003) and stats[i,3] >= h*.12
             and not (stats[i,2] > w*.9 and stats[i,3] < h*.2)]
    if not boxes:
        return image, {'bounds': [0,0,w,h], 'reason': 'no coherent foreground'}
    central = [b for b in boxes if .2*w <= b[0]+b[2]/2 <= .8*w]
    seed = max(central or boxes, key=lambda b: b[2] * b[3])
    chosen = [seed]
    remaining = [b for b in boxes if b != seed]
    changed = True
    while changed:
        changed = False
        left = min(b[0] for b in chosen); right = max(b[0]+b[2] for b in chosen)
        for b in remaining[:]:
            gap = max(left-(b[0]+b[2]), b[0]-right, 0)
            if gap <= .45*h:
                chosen.append(b); remaining.remove(b); changed = True
    # Preserve separated words/marks unless the excluded component is a
    # compact, isolated object at an outer edge. Never trim arbitrary words.
    for b in remaining:
        edge = b[0]+b[2]/2 < .2*w or b[0]+b[2]/2 > .8*w
        compact = .6 <= b[2]/max(1,b[3]) <= 1.6 and b[2] <= h
        if not (edge and compact):
            chosen.append(b)
    left = min(b[0] for b in chosen); right = max(b[0]+b[2] for b in chosen)
    if right-left < w*.45:
        return image, {'bounds':[0,0,w,h], 'reason':'foreground too ambiguous to trim'}
    # Include small detached vowel marks inside the retained horizontal span.
    ys, xs = np.nonzero(mask[:,left:right])
    pad = max(2, round(h*.08))
    bounds = [max(0,left-pad), max(0,int(ys.min())-pad), min(w,right+pad), min(h,int(ys.max())+1+pad)]
    return image.crop(tuple(bounds)), {'bounds':bounds, 'reason':'coherent text band with character margin',
                                       'original_size':[w,h]}


def load_header_recognizer(path, device):
    """Only validated, non-smoke header fine-tunes may enter header inference."""
    import torch
    from .recognizer import PARSeqRecognizer
    path = Path(path)
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    if config.get('task') != 'header_finetune' or config.get('smoke_test') or not config.get('production_ready') or config.get('input_normalization') != 'doctr_parseq':
        raise ValueError('Not a production header fine-tune checkpoint')
    return PARSeqRecognizer(path, device)
