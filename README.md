# Automatic Number Plate Recognition (ANPR)

This command-line project detects a vehicle plate, separates and straightens one or two text rows, recognizes the stitched text with PARSeq, and prints the cleaned registration number. It deliberately has no web server, API, GUI, or frontend.

## Architecture

```text
vehicle image + EXIF correction
  -> YOLO11s plate detector
  -> expanded plate crop
  -> YOLO11s-OBB text-row detector
  -> perspective correction + top-to-bottom row stitching
  -> docTR PARSeq OCR
  -> uppercase A-Z/0-9 cleanup
```

The row stage matters for stacked Indian plates: `MH12` above `AB1234` becomes one horizontal OCR image. The OBB quadrilateral also supplies the geometry needed to flatten tilted text.

## Dataset inspection findings

The uploaded `data/` tree contains 63,244 image files and 21,512 YOLO label files across core, supporting, and duplicated task representations. It includes the documented detection splits (6,308 train, 1,784 validation, 883 test), both row datasets (5,382 train and 747 validation each), and the stitched OCR manifests (15,074 train, 203 validation, 238 test). The 132 empty detection training labels and 19 empty validation labels are retained as intentional hard negatives. The test detection split has 883 images and 882 label files, so one image is explicitly reported as missing an annotation and must not silently enter metric calculations.

All supplied YOLO YAMLs and original OCR JSONL manifests referred to `/home/calua/number_plate/...`. The YAMLs in this project now use portable relative roots. `scripts/prepare_dataset.py` creates non-destructive `*.portable.jsonl` copies; runtime code can also repair the old paths in memory. No `.pt`, `.pth`, `.ckpt`, or `.onnx` trained weights were supplied, so historical figures in `data/DATA.md` are references only, not results from this implementation.

Run the complete audit for corrupt images, invalid labels, missing/orphan annotations, manifests, duplicate content, suspicious pre-annotated image filenames, and weights:

```bash
python scripts/inspect_dataset.py
```

The detailed machine-readable result is written to `reports/dataset_report.json`. Empty annotations are reported separately from missing or corrupt annotations.

## Installation

From a fresh terminal:

```bash
cd /Users/gauravghanekar/Downloads/number_data
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch automatically uses the device selected by the scripts: CUDA first, then Apple MPS, then CPU. Pass `--device cpu`, `--device mps`, or `--device cuda` to override it.

## Dataset preparation

Expected layouts are already present:

```text
data/det/images/{train,val,test}       data/det/labels/{train,val,test}
data/rowdet_obb/images/{train,val}     data/rowdet_obb/labels/{train,val}
data/ocr_stitched/{train,val,test}     data/ocr_stitched/{train,val,test}.jsonl
```

Audit and produce portable manifest copies:

```bash
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
```

Strict detector-label and Ultralytics-parser preflight:

```bash
python scripts/validate_yolo_labels.py --loader-check --clear-cache
python scripts/train_detector.py --preflight-only --device mps
```

Empty label files remain valid background samples. The validator reports malformed values, NaN/Inf, hidden characters, missing pairs, YAML errors, and the exact class IDs produced by Ultralytics.

The original JSONLs are intentionally preserved. Training/inference resolves either original or portable paths relative to this project.

## Training

Train each stage in order:

```bash
python scripts/train_detector.py --epochs 60 --imgsz 640 --batch 16
python scripts/train_row_detector.py --epochs 60 --imgsz 320 --batch 32
python scripts/train_ocr.py --epochs 30 --batch 64
```

These commands are restart-safe. Run the same command again after Ctrl+C, terminal closure, or a machine restart and it automatically continues from the latest completed epoch. The requested `--epochs` value is the total target, not a number of extra epochs. For example, interrupting a 60-epoch detector run after epoch 17 causes the same command to continue at epoch 18 and stop at epoch 60.

At startup each trainer clearly reports `NEW TRAINING` or `AUTO RESUME`, the selected checkpoint, completed epoch, next epoch, total target, and device. An already completed run exits without training again.

Resume controls:

```bash
# Require a compatible checkpoint; fail clearly if none exists.
python scripts/train_detector.py --resume --epochs 60 --imgsz 640 --batch 16

# Intentionally start a new run while preserving previous runs/checkpoints.
python scripts/train_detector.py --fresh --epochs 60 --imgsz 640 --batch 16
python scripts/train_row_detector.py --fresh --epochs 60 --imgsz 320 --batch 32
python scripts/train_ocr.py --fresh --epochs 30 --batch 64
```

Detector and row-detector recovery checkpoints live in their Ultralytics run directories as `weights/last.pt` and `weights/previous.pt`. OCR recovery files are `models/ocr/last.pt` and `models/ocr/previous.pt`; `models/ocr/best.pt` remains the best validation model. OCR checkpoint updates are atomic and include the model, optimizer, scheduler, last completed epoch, best metric, configuration, vocabulary, and random-number-generator state. If `last.pt` is corrupt, recovery falls back to `previous.pt`. Checkpoint compatibility is validated before training, and a per-trainer process lock prevents two local jobs from writing the same model concurrently.

On macOS, optionally keep the machine awake only while a training command is running:

```bash
caffeinate -i python scripts/train_detector.py --epochs 60 --imgsz 640 --batch 16
```

`caffeinate` is optional and does not change checkpoint or resume behavior.

Outputs are copied to predictable locations:

```text
models/plate_detector/best.pt
models/row_detector/best.pt
models/ocr/best.pt
```

YOLO training starts from `yolo11s.pt` and `yolo11s-obb.pt`. OCR uses docTR's PARSeq implementation, the dataset's `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` vocabulary, mild pre-resize augmentation, and selects the checkpoint with the highest validation exact-match accuracy. Each OCR epoch reports training loss plus validation loss, exact match, and character error rate.

Ultralytics is pinned above 8.3.203 because that release used non-blocking tensor transfers on Apple MPS and could corrupt validation class IDs. Detector training now validates raw annotations and parsed targets, clears stale YOLO caches, and refuses affected MPS releases before training starts.

Training all three neural networks is substantial work; GPU or Apple Silicon is strongly recommended. The repository does not manufacture placeholder weights or accuracy numbers.

### Optional mixed-script PARSeq model

The combined Latin + Devanagari OCR model is independent of the existing Latin
and Devanagari checkpoints. Generate its grouped, deterministic synthetic data:

```bash
.venv/bin/python scripts/generate_mixed_ocr_dataset.py \
  --output data/mixed_ocr_dataset \
  --count 100000 \
  --seed 20260904 \
  --val-fraction 0.10
```

Train, resume, extend the total target, or evaluate only this third model:

```bash
.venv/bin/python scripts/train_ocr.py --data-dir data/mixed_ocr_dataset --epochs 30 --batch 64 --device mps --checkpoint-dir models/ocr_mixed --run-name parseq_ocr_mixed --diagnostic-samples 5 --fresh
.venv/bin/python scripts/train_ocr.py --data-dir data/mixed_ocr_dataset --epochs 30 --batch 64 --device mps --checkpoint-dir models/ocr_mixed --run-name parseq_ocr_mixed --diagnostic-samples 5 --resume
.venv/bin/python scripts/train_ocr.py --data-dir data/mixed_ocr_dataset --epochs 60 --batch 64 --device mps --checkpoint-dir models/ocr_mixed --run-name parseq_ocr_mixed --diagnostic-samples 5 --resume
.venv/bin/python scripts/train_ocr.py --data-dir data/mixed_ocr_dataset --epochs 60 --batch 64 --device mps --checkpoint-dir models/ocr_mixed --run-name parseq_ocr_mixed --diagnostic-samples 20 --evaluate-only
```

When `models/ocr_mixed/best.pt` exists, inference evaluates it as a third,
whole-row candidate. Predictions from separate models are never spliced. When
the checkpoint is absent, verbose inference reports that fact and continues
with the existing Latin and Devanagari models.

## Inference

One image, with clean stdout:

```bash
python main.py --image test.jpg
```

Normal production:

```bash
python main.py \
  --image "IMAGE_PATH" \
  --device mps \
  --det-imgsz 960 \
  --det-conf 0.25
```

Folder mode:

```bash
python main.py --folder test_images/
```

Diagnostics:

```bash
python main.py --image test.jpg --verbose
```

Inference runs fully in memory by default. To save detection, crop, rectified,
and stitched images for accepted plate results only:

```bash
python main.py --image test.jpg --save-results
```

For a difficult small or distant plate, full-image detection automatically
retries at the configured fallback resolution when the primary pass returns
zero detections:

```bash
python main.py \
  --image "IMAGE_PATH" \
  --device mps \
  --det-imgsz 1920 \
  --det-conf 0.10 \
  --det-fallback \
  --verbose
```

Overlapping tile fallback is enabled by default when the primary pass is empty;
use `--no-det-tile-fallback` to disable it. Tile and fallback detections are
merged through the same NMS. Structured debugging writes to
`outputs/debug/<image_stem>/`:

```bash
python main.py \
  --image "IMAGE_PATH" \
  --device mps \
  --det-imgsz 1920 \
  --det-conf 0.10 \
  --det-fallback \
  --save-debug \
  --verbose
```

For a 1080p dashcam frame whose plate is only a few dozen pixels wide:

```bash
python main.py --image dashcam.jpg --det-imgsz 1920
```

The summary separates accepted, review, and rejected candidates. `VALID_FORMAT`
only means the text shape resembles a registration; it does not establish that
each OCR character is correct. `ACCEPTED` additionally requires very high OCR
confidence and no blocking reconciliation or image-risk signal. Medium-confidence,
conflicting, multi-row, or yellow night/glare candidates become
`REVIEW_REQUIRED`. Raw OCR is never completed with invented characters.
Normalized text is stored separately; short visible text is `PARTIAL_VISIBLE`
and low-confidence visible text is `LOW_CONFIDENCE`.

## Evaluation

After training, evaluate held-out test data only:

```bash
python scripts/evaluate.py
```

This writes `reports/evaluation.json` with detector precision, recall, mAP50, mAP50-95, and OCR exact-match accuracy/CER. To diagnose memorization without confusing train results with final results:

```bash
python scripts/evaluate.py --train-sample 500
```

The evaluator creates a test image list containing only paired, non-empty annotations and excludes suspicious filenames such as the uploaded `_annotated.jpg` test image. Every exclusion is recorded in the JSON output instead of being silently counted as a background sample.

Do not quote the historical scores in `data/DATA.md` as this project's results. Only metrics produced by the evaluation command for the newly trained checkpoints are valid.

## File guide

- `src/detector.py`: full-image YOLO detection and plate cropping.
- `src/row_detector.py`: OBB row inference, row ordering, and stitching.
- `src/preprocess.py`: EXIF-safe loading, conservative enhancement, and perspective geometry.
- `src/recognizer.py`: PARSeq checkpoint loading and decoding.
- `scripts/inspect_dataset.py`: comprehensive dataset audit.
- `scripts/prepare_dataset.py`: portable manifest generation.
- `scripts/train_*.py`: separate, explainable training entry points.
- `scripts/evaluate.py`: final test-only metrics.
- `main.py`: small CLI orchestration layer.
