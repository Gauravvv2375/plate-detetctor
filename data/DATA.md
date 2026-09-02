# Datasets — `/home/calua/number_plate/data`

Everything below is on this machine (`spark-3859`). Use the `numplate` conda env:

    conda activate numplate

There are three separate tasks. Pick the one you're training.

---

## 1. Plate detection — `data/det/`

Standard YOLO layout. Detects the whole number plate in a full scene.

    data/det/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt     # "0 cx cy w h", normalised
      data.yaml                          # single class: license_plate

| split | images |
|---|---|
| train | 6,308 |
| val | 1,784 |
| test | 883 |

Train:  `yolo detect train data=data/det/data.yaml model=yolo11s.pt imgsz=640 epochs=60`

Current best: **test mAP50 0.989, mAP50-95 0.715**.

**Two things to know.** 132 train / 19 val images are deliberately EMPTY-label
background patches (dashcam timestamp overlays that the detector used to read as
plates). Do not delete them — they cut false positives from 64% to 4% on those
locations. Second: on dashcam footage you must run inference at `imgsz=1920`.
At the 640 default the detector finds **zero** plates in 1080p dashcam frames,
because plates there are only ~16–90 px wide.

---

## 2. Text-row detection — `data/rowdet/` and `data/rowdet_obb/`

Runs INSIDE a plate crop and finds each row of registration text. This is what
makes two-row plates readable: a CTC recogniser emits one symbol per image
column, so on a stacked plate it cannot produce the right string. Detect the
rows, lay them side by side, and the limitation disappears.

    data/rowdet/       axis-aligned boxes   "0 cx cy w h"
    data/rowdet_obb/   ORIENTED boxes       "0 x1 y1 x2 y2 x3 y3 x4 y4"

Both: 5,382 train / 747 val images, ~10,100 boxes.

Train:  `yolo detect train data=data/rowdet/data.yaml model=yolo11s.pt imgsz=320 epochs=60`
OBB:    `yolo obb    train data=data/rowdet_obb/data.yaml model=yolo11s-obb.pt imgsz=320 epochs=60`

Axis-aligned version scores mAP50 0.995 (0.995 on real held-out crops, 84%
recall on known two-row plates — that recall is the current bottleneck).

The OBB version exists because every recogniser collapses to ~14% accuracy past
15° of plate tilt. An oriented box carries the angle, so the row can be
perspective-warped FLAT before recognition instead of relying on a post-hoc
deskew that is unreliable past 25° and sign-ambiguous near 45°.

**Caveat: only 116 of the 5,382 train images are real.** The rest are synthetic.
Real-domain recall is the weak link — more hand-drawn row boxes would help here
more than any model change.

---

## 3. Text recognition — `data/ocr/` and `data/ocr_stitched/`

JSONL manifests, one record per crop:

    {"path": "/abs/path/crop.jpg", "text": "MH12HN4507", "src": "real"}

`data/ocr_stitched/` records also carry `n_rows` and `orig` (the pre-stitch crop).

| dir | train | val | test | what the image is |
|---|---|---|---|---|
| `data/ocr/` | 19,230 (1,230 real) | 203 | 240 | the plate crop as-is |
| `data/ocr_stitched/` | 15,074 (1,137 real) | 203 | 238 | rows detected, deskewed, laid side by side |

**Train on `data/ocr_stitched/`.** It is worth +12 points to a CRNN and lifts
multi-row accuracy from ~5% to ~78%. `src` is `real`, `syn` (old single-row
synthetic) or `syn2` (new two-row synthetic).

Current results (test, real crops only):

| model | exact | CER | multi-row |
|---|---|---|---|
| PARSeq (docTR, 23.8M) | **93.70%** | **2.12%** | 78.3% |
| TrOCR-small (61.6M) | 88.66% | 3.12% | 65.2% |
| PP-OCRv6-small (2.8M) | 86.55% | 2.17% | 73.9% |
| CRNN+CTC (4.1M) | 73.11% | 5.64% | 47.8% |

Reference trainers: `scripts/train_parseq.py`, `train_trocr.py`,
`train_ppocr.py`, `train_ocr.py`. All take `--data-dir`.

---

## 4. Supporting data

| path | contents |
|---|---|
| `data/ocr/synth2/` | 6,000 synthetic plates + `index.jsonl` with per-row boxes AND rotated quads. Regenerate: `python3 scripts/gen_synth_plates.py -n 6000` |
| `data/mined/` | 392 plate crops mined from dashcam video, with source video + frame + box. 91 are human-verified. |
| `data/det_hardneg/` | 151 confirmed non-plate patches (timestamp overlays) |
| `data/additional/dashcam/Dashcam_vdos/` | 35 dashcam videos, 3.9 GB, 1080p30 |
| `raw/` | original HF downloads (keremberke plates, Indian plate crops, synthetic corpus) |
| `data/ocr/illegible_index.json` | 101 crops a human judged unreadable — negatives for the legibility classifier |

---

## Pitfalls that have already cost us time

1. **`transformers` must be 5.11.0.** 4.57.x silently mis-shifts labels inside
   `VisionEncoderDecoderModel`; TrOCR trains to 0% exact-match while inference
   still looks fine. `train_trocr.py` asserts the version and aborts otherwise.
2. **Apply `ImageOps.exif_transpose()`** when loading frames with PIL before
   detection. ultralytics does it when reading a path, PIL does not — skipping
   it lost 4 of 10 detections on sideways-stored phone photos.
3. **Augment before resizing, not after.** `scripts/augment.py` changes image
   dimensions (rotate with expand, zoom-out padding), so resizing first makes
   the batch unstackable.
4. **Evaluate on test only, and report the train/test gap.** The CRNN scores
   95.7% on train two-row crops and 58.3% on test — a 37-point memorisation gap
   that made it look like the best two-row model when it was not.
5. **`labels/*.txt` have no trailing newline.** `cat *.txt | wc -l` undercounts
   badly; parse per-file.
