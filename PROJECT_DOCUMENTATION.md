# ANPR Project Documentation

**Repository:** `/Users/gauravghanekar/Downloads/number_data`  
**Inspection date:** 2026-09-09  
**Scope:** current local implementation, not a proposed redesign.

This document is a developer handoff for inference, data preparation, evaluation,
debugging and future training. Paths in commands are relative to the repository
root unless an absolute path is shown. Run commands from that root.

The source files, live CLI help, checkpoint metadata, dataset manifests and local
reports were inspected to prepare this document. Dataset counts below are current
filesystem/manifest counts. Accuracy numbers are identified as stored checkpoint
metrics or previously recorded experiments; no new full validation benchmark was
run for this documentation task.

The older `README.md` and `data/DATA.md` contain historical information. In
particular, the README's initial Latin-only flow and mixed-v1 commands do not
describe the current default pipeline. `data/DATA.md` references another machine,
other trainers and historical metrics; those are not instructions for this checkout.

## 1. Project overview

The project detects Indian vehicle number plates in an input photograph, identifies
registration rows inside each plate, recognizes their text, and returns conservative
registration decisions. It also recognizes meaningful auxiliary/header rows
independently and constructs a full-plate text representation.

The executable interface is a Python CLI. There is no implemented HTTP server,
frontend, GUI, live-video service or database-backed ANPR application.

Current capabilities:

- Multiple plate detections per image, ordered for reporting.
- Single-line and two-row registrations, including tilted rows.
- Latin OCR using a 36-character A–Z/0–9 model.
- Devanagari OCR using a separate 68-character model.
- True within-row Latin/Devanagari candidates using the 73-character mixed-v2 model.
- Script-preserving output from Unicode-capable checkpoints, including supported
  Devanagari digits, combining marks, spaces and hyphens.
- Geometry-based separation of small header text from primary registration text.
- Independent `header_text`, `registration_text` and `full_text` results.
- Conservative handling of disagreement, short visible strings and low confidence.

The system supports these input categories, but does not guarantee accurate
recognition for every Indian state format, font, language, vehicle or plate design.
Vocabulary coverage is finite. Arbitrary Marathi words can contain characters
absent from the current mixed model. Small or stylized real-world Devanagari text
remains a verified weakness despite excellent synthetic validation scores.

`REVIEW_REQUIRED` is intentional: a plausible-looking registration is not proof
that all characters are correct. A header must never make an uncertain registration
appear accepted. The implemented successful status is `ACCEPTED`, not `OK`.

## 2. Complete project architecture

```text
Input image / each direct child image in a folder
    |
    v
EXIF correction + RGB loading                    src/preprocess.py
    |
    v
YOLO11s plate detection                          src/detector.py
    |  primary empty?
    +----> optional higher-resolution full-image pass
    +----> optional overlapping tiles; boxes mapped to full image
    |
    v
IoU NMS + nested-duplicate filtering + reading order
    |
    v
For each plate: 8%-expanded crop
    |
    v
YOLO11s-OBB text-row detection                    src/row_detector.py
    |
    +--> overlap deduplication + registration-row ranking
    |       |
    |       +--> PRIMARY_REGISTRATION / SECONDARY_REGISTRATION
    |       |       |
    |       |       v
    |       |   quadrilateral rectification; one/two-row views
    |       |       |
    |       |       +--> legacy enhancement --> Latin PARSeq
    |       |       +--> raw RGB -----------> Devanagari PARSeq
    |       |       +--> raw RGB -----------> Mixed-v2 PARSeq
    |       |       |
    |       |       v
    |       |   whole-candidate selection / per-row routing
    |       |       |
    |       |       +<-- original-plate OCR fallback when applicable
    |       |       |
    |       |       v
    |       |   conservative row/direct reconciliation
    |       |       |
    |       |       v
    |       |   format + confidence + risk checks
    |       |       |
    |       |       v
    |       |   FINAL REGISTRATION TEXT / CONFIDENCE / STATUS
    |       |
    |       +--> HEADER (including optional auxiliary 960px detection)
    |       |       |
    |       |       v
    |       |   rectify + geometry-based tightening
    |       |       |
    |       |       +--> eligible dedicated header checkpoint, if present
    |       |       `--> otherwise existing Latin/Devanagari/mixed candidates
    |       |       |
    |       |       v
    |       |   independent header text, always reviewable
    |       |
    |       `--> DECORATIVE_OR_NOISE: no header OCR
    |
    v
Reading-order assembly: header_text + independent registration_text
    |
    v
PlateInference / as_dict() / terminal output
    `--> debug images + results.json ONLY when explicitly requested
```

The diagram shows logical branches; `_complete_plate_text()` runs after
`_finalize_acceptance()`. Registration finalization therefore precedes header OCR.
Each plate is processed independently; one row/OCR failure need not abort other
plates. `infer_all()` is the reusable entry point, while `infer()` is a legacy
single-string wrapper that returns the first plate's text.

## 3. Models and checkpoint inventory

### 3.1 Current inference models

| Component | Exact checkpoint | Architecture / input → output | Current use and readiness |
|---|---|---|---|
| Plate detector | `models/plate_detector/best.pt` | Ultralytics YOLO11s DetectionModel; vehicle RGB image → class-0 boxes and confidence | Trained; mandatory default detector; small/distant plates remain difficult |
| Row detector | `models/row_detector/best.pt` | Ultralytics YOLO11s-OBB OBBModel; plate crop → class-0 quadrilaterals and confidence | Trained; mandatory default row detector; geometry heuristics select registration/header roles |
| Latin OCR | `models/ocr/best.pt` | docTR PARSeq; RGB text image → decoded text/confidence | Trained; mandatory default OCR candidate; best epoch 24, recovery completed epoch 30 |
| Devanagari OCR | `models/ocr_devanagari/best.pt` | docTR PARSeq, 68-character vocabulary | Trained; optional second candidate loaded when present; best epoch 25, recovery epoch 30 |
| Mixed OCR v2 | `models/ocr_mixed_v2/best.pt` | docTR PARSeq, 73-character combined vocabulary | Trained; current optional mixed candidate and protected registration baseline; best epoch 27, recovery epoch 30 |
| Dedicated real-header OCR | `models/ocr_header_real_v1/best.pt` | Planned fine-tuned copy of mixed-v2 PARSeq | **Absent. The directory `models/ocr_header_real_v1/` does not currently exist. No production header model is trained.** |

The plate checkpoint records an improved-detector run at image size 960, batch 8,
target 20 epochs, using `reports/det_improved_runtime.yaml` and
`runs/plate_detector4/`. The main CLI nevertheless defaults to inference size 640.
The row checkpoint records target 60 epochs, image size 320 and batch 32.
Their exported checkpoints have `epoch=-1`, an Ultralytics stripped/export state;
do not interpret that as evidence of an untrained detector.

Stored detector validation metrics, not newly measured end-to-end accuracy:

| Model | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Current plate checkpoint | 0.98132 | 0.95348 | 0.98018 | 0.72989 |
| Current row checkpoint | 0.99996 | 1.00000 | 0.99500 | 0.99440 |

These values characterize their recorded validation data. They do not establish
real-header recall or correct OCR of every detected plate.

### 3.2 Legacy, recovery and archived checkpoints

The following are present and must not be deleted casually:

| Location | Files present / purpose |
|---|---|
| `models/ocr/` | `best.pt`, `last.pt`, `previous.pt`: Latin inference and recovery |
| `models/ocr_devanagari/` | `best.pt`, `last.pt`, `previous.pt`: active Devanagari run |
| `models/ocr_mixed_v2/` | `best.pt`, `last.pt`, `previous.pt`: active mixed-v2 run |
| `models/ocr_mixed/` | `best.pt`, `last.pt`, `previous.pt`: historical mixed-v1 run; best/recovery epoch 4; **not the production baseline** |
| `models/ocr_devanagari/archive/20260903_diagnosed_epoch5_scratch/` | `best.pt`, `last.pt`, `previous.pt`: preserved diagnosed historical run |
| `models/ocr_mixed_v2/archive/20260907_110737/` | `best.pt`, `last.pt`, `previous.pt`: preserved older mixed-v2 run state |
| `runs/plate_detector3/weights/` | `best.pt`, `last.pt`, `previous.pt`: earlier detector run |
| `runs/plate_detector4/weights/` | `best.pt`, `last.pt`, `previous.pt`: improved detector run |
| `runs/row_detector_obb/weights/` | `best.pt`, `last.pt`, `previous.pt`: row run |
| Repository root | `yolo11s.pt`, `yolo11s-obb.pt`: YOLO training initializers, not the active custom inference checkpoints |

There are also run directories containing configuration without a weights directory;
the existence of a run folder alone does not prove a successful trained model.
None of the archives or mixed-v1 checkpoints is selected by the default inference CLI.

## 4. OCR model details

### 4.1 PARSeq architecture and vocabulary

The installed docTR 0.11.0 `parseq()` uses a small vision transformer (`vit_s`),
patch size `(4, 8)`, 384-dimensional embeddings, and a permutation-trained
autoregressive decoder. Recognition instantiates the architecture without
downloading pretrained weights, then strictly loads the selected local state dict.
The adapter calls the model in `eval()` and `torch.inference_mode()`.

| OCR model | Vocabulary size, excluding special tokens | Stored/configured max_length | Normalization |
|---|---:|---:|---|
| Latin | 36 | 32, adapter/loader default for legacy config | Float RGB / 255 only |
| Devanagari | 68 | 62 | docTR mean/std |
| Mixed v1, historical | 73 | 34 | docTR mean/std |
| Mixed v2 | 73 | 34 | docTR mean/std |

Latin vocabulary is `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. It cannot emit
Devanagari. The Devanagari checkpoint includes Devanagari letters/digits, space
and hyphen, but **does not include Latin A–Z or ASCII digits 0–9**. Running these
two restricted models together does not itself provide reliable within-row mixed
recognition. Mixed-v2 provides that combined vocabulary.

Mixed-v2's ordered vocabulary is shown below; the initial character is a space:

```text
 -0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZआईएओकचजटडनपफबमयरलवस़ाीूे्०१२३४५६७८९
```

This is not the complete Devanagari block or a full Marathi alphabet. For example,
`ह` is absent from mixed-v2. High registration validation accuracy therefore does
not imply coverage of arbitrary Marathi header words.

CSV datasets compute `max_length = longest NFC label length + 2`, reserving SOS/EOS
capacity for docTR target encoding. Length here means Unicode code points, not
visual grapheme clusters. Mixed-v2's longest label is 32 code points; Devanagari's
is 60. Header fine-tuning preserves the source model's capacity and rejects labels
longer than `max_length - 2` instead of silently truncating them.

### 4.2 Image and tensor contract

`PARSeqRecognizer.input_image()` is the canonical inference resize:

1. Convert to RGB.
2. Bilinear stretch to **128 pixels wide × 32 pixels high**.
3. Convert to float32 CHW, divide by 255, add batch dimension.
4. For checkpoints marked `input_normalization == 'doctr_parseq'`, normalize with
   mean `(0.694, 0.695, 0.693)` and std `(0.299, 0.296, 0.301)`.

Tensor input is `[N, 3, 32, 128]`; checkpoint `image_size` is stored as `[32, 128]`.
Do not confuse width/height order with tensor height/width order.

CSV-trained Devanagari and mixed models receive the **raw rectified/cropped RGB**
view before this resize, not the legacy contrast/sharpening pass. Legacy Latin OCR
keeps `mild_ocr_preprocess()`: upscaling short images to at least height 48,
contrast factor 1.12, then mild unsharp masking. This distinction is intentional
and protected by tensor-equality/routing tests.

No production aspect-ratio padding or higher-resolution PARSeq input is currently
enabled. A stored frozen-model 256×32 experiment failed positional-embedding
compatibility (257 versus 129 tokens). Changing input dimensions is not a safe
drop-in checkpoint configuration change.

### 4.3 Unicode and output preservation

- CSV loading and Unicode OCR output use NFC normalization.
- CSV rejects empty labels, path traversal and control/format characters.
- `charset.txt` is one unique character per line; a space occupies its own line.
- Charset membership must equal characters derived from labels, not merely contain
  them. Ordering is checkpoint-significant.
- The Unicode recognizer retains only characters in its checkpoint vocabulary.
- Legacy Latin output is uppercased and stripped to A–Z/0–9.
- Main validation accepts uppercase Latin, ASCII digits, U+0900–U+097F, spaces
  and hyphens. Combining marks do not count as independent base registration units.

There is **no inference transliteration** from Latin to Devanagari or from
Devanagari digits to ASCII. `MH १२ AB १२३४` must not be converted merely to make
validation easier. This preserves the scripts predicted by the model; it is not
a promise that an incorrect model prediction matches the image.

`raw_ocr` remains available alongside normalized output. Normalization is not
entirely a no-op: NFC, filtering, trimming and legacy Latin cleanup exist, and a
narrow BH-series rule removes an extra two-letter prefix before an otherwise
complete BH registration. There is no CLI switch for generalized transliteration.

### 4.4 Verified mixed-v2 synthetic validation result

Read directly from `models/ocr_mixed_v2/best.pt`:

| Metric | Value |
|---|---:|
| Best epoch | 27 |
| Validation loss | **0.0105** (stored 0.010543792837658234) |
| Exact sequence accuracy | **98.38%** (stored 0.9837864804190571) |
| Character accuracy | **99.85%** (stored 0.9985292684439483) |
| Character error rate | **0.15%** (stored 0.0014973110420044475) |

These are **synthetic validation** results on the corrected mixed-v2 split, not
real-world header or end-to-end ANPR accuracy. `last.pt` is epoch 30, not the best
epoch. Continue using `best.pt` for inference.

Other stored OCR validation results: Latin best epoch 24 has exact 88.67% and CER
2.59%; Devanagari best epoch 25 has exact 95.42% and CER 0.60%. These use different
datasets and should not be compared as if they were one shared benchmark.

Exact accuracy compares entire normalized strings. CER is summed Levenshtein
edits divided by ground-truth code points. Character accuracy counts matches on a
minimum-edit alignment divided by ground-truth code points; it is not necessarily
exactly `1 - CER`, especially with insertions.

## 5. Datasets

### 5.1 Core data inventory

| Dataset | Current counts | Format and role |
|---|---|---|
| `data/det/` | 6,308 train / 1,784 val / 883 test images | Original whole-plate YOLO boxes |
| `data/det_improved/` | 7,640 train / 1,926 val / 1,097 test images | Improved merged/deduplicated whole-plate detector data; current plate model records this dataset |
| `data/rowdet/` | 5,382 train / 747 val images | Axis-aligned text-row labels; supporting representation, not current OBB training default |
| `data/rowdet_obb/` | 5,382 train / 747 val images | Oriented row labels used by current row trainer |
| `data/ocr/` | Original JSONL manifests: 19,230 train / 203 val / 240 test | Legacy unstitched OCR crops; additional clean/audit manifests also exist |
| `data/ocr_stitched/` | 15,074 train / 203 val / 238 test images/records | Current Latin OCR data; pre-rectified/stitched rows |
| `data/devenagari_ocr_dataset/` | 50,000 images and CSV records | Synthetic Devanagari OCR data; spelling of `devenagari` in the path is intentional |
| `data/mixed_ocr_dataset/` | 80,000 images; 71,982 train / 8,018 val | Historical synthetic mixed v1; **not production training data** |
| `data/mixed_ocr_dataset_v2/` | 80,000 images; 71,982 train / 8,018 val | Corrected synthetic mixed v2; current mixed registration baseline |
| `data/real_header_ocr_dataset/` | 9 retained sources, 3 unlabeled crops, 0 verified labels | Real header collection/preparation in progress |

Original detector labels contain 132 intentionally empty training annotations
and 19 empty validation annotations. These are hard negatives, not corrupt labels.
The original test split has 882 `.txt` annotations for 883 images: one missing
annotation. Image/label counts above exclude unrelated non-`.txt` files. Improved
detector data has matching image/label pairs in all splits and retains the 151
background samples. Its stored preparation report records no cross-split decoded
image duplicates; that report is evidence from preparation, not a new full pixel audit.

YOLO labels are normalized coordinates:

```text
# whole plate / axis-aligned row
0 cx cy width height

# OBB row
0 x1 y1 x2 y2 x3 y3 x4 y4
```

Dataset YAMLs use `path: .`, relative split directories and one class. Trainers
generate absolute runtime YAMLs under `reports/`; do not assume those generated
files are portable when moving the project.

Latin JSONL uses `path` and `text`, with optional source/row metadata. Some original
paths reference `/home/calua/number_plate/`; `resolve_manifest_path()` repairs them
in memory, and `.portable.jsonl` copies exist. The trainer still opens
`train.jsonl` and `val.jsonl`, not automatically the portable filenames.
The stitched training manifest contains 13,937 synthetic and 1,137 real records;
validation and test records are marked real.

### 5.2 Synthetic Devanagari dataset

The folder contains `images/`, `labels.csv`, `labels.txt`, `charset.txt`,
`dataset_info.json`, `integrity_report.json` and a preview image.
CSV begins with `image,text` and includes pattern, state, city, series, layout,
font, augmentation and render-hash metadata. Paths are relative to the dataset.
`labels.txt` is an auxiliary UTF-8 tab-separated path/transcription representation;
the current training loader uses **labels.csv**, not labels.txt.

There are 37,368 one-line and 12,632 two-line images. With the current defaults,
the loader shuffles deterministically using seed 42 and divides 45,000 train /
5,000 validation records. There is no explicit split column. This is a record
split, not the source-group split used for verified real-header training.

The stored integrity report has no listed errors. The documentation inspection
also ran the loader successfully; that checks paths/labels/charset/splits but does
not replace visually checking all 50,000 rendered samples.

### 5.3 Why mixed v1 is obsolete

`reports/mixed_ocr_font_capability.json` identifies Devanagari MT and ITF
Devanagari faces that lack uppercase Latin A–Z glyphs. Pillow does not supply
automatic font fallback for those missing glyphs. Some v1 images therefore
rendered identical missing-glyph boxes where labels claimed distinct Latin letters.
Training on those image/label contradictions cannot teach reliable Latin recognition.

`reports/mixed_ocr_diagnosis.json` records the old epoch-4 model's exact accuracy
around 59.20%, with much worse Latin/mixed categories than Devanagari-only examples.
Neither `data/mixed_ocr_dataset/` nor `models/ocr_mixed/best.pt` should be used as
the production baseline. Keep them for provenance and diagnosis, not default use.

### 5.4 Corrected mixed v2

The current generator uses only the audited mixed-safe faces:

- Arial Unicode.
- Devanagari Sangam MN face 0.
- Devanagari Sangam MN face 1.

The code has a fixed audited allowlist; `available_fonts()` checks that those files
load. It does not perform a fresh complete cmap audit on every run, so a new font
requires deliberate validation, not just addition to the list.

Both mixed manifests include `image,text,category,layout,difficulty,split,group_id`
plus plate-part, separator, font, background, header and hash metadata. The v2
generator version is 2. Five categories cover Latin-only, Devanagari-only, each
cross-script digit combination, and other mixed-script combinations.

V2 layouts: 53,184 one-line, 19,004 two-line and 7,812 header-plus-registration.
Splits are assigned by SHA-256 of `group_id`; related variants stay in one split.
The loader rejects group overlap and mixed specified/unspecified split rows.
Explicit split assignments take precedence over randomized splitting.

For synthetic two-line examples, the loader finds a dark-pixel valley near the
vertical midpoint and lays the two halves horizontally. For synthetic
`header_registration` examples it crops below 29% of image height before OCR.
That fixed synthetic-layout preprocessing is **not** the geometry used to crop
real inference headers. The target `text` is registration-only; `header_text`
metadata does not make mixed-v2 a trained arbitrary-header recognizer.

The generator refuses a nonempty destination. Its CLI default still points to
`data/mixed_ocr_dataset`; do not rerun it casually or regenerate v2 in place.

### 5.5 Supporting data

`data/kaggle 1/`, `data/kaggle 2/` and `data/kaggle 3/` are source datasets consumed
by improved detector preparation. The first two are parsed as Pascal VOC XML;
the third as YOLO. `data/mined/index.jsonl` has 392 mined records.
`data/det_hardneg/`, `data/rowdet_realtrain/`, `data/rowdet_realval/`,
`data/additional/` and legacy `data/ocr/` variants are supporting assets, not the
default mixed-v2 training input. Do not combine their counts as independent,
deduplicated training examples without an audit.

## 6. Real header dataset: current state and label safety

```text
data/real_header_ocr_dataset/
  source_images/          # 9 retained original public image downloads
  images/                 # 0 verified crop images
  unlabeled/              # 3 crop images: original 1 + collected 2
  source_manifest.csv     # 9 source/provenance records
  review_queue.csv        # 9 records: 2 crop paths, 7 extraction failures
  labels.csv              # image,text header only; 0 verified training records
  charset.txt             # currently empty: no verified label vocabulary
  dataset_info.json       # preparation-tool provenance; records original crop only
  collection_staging/     # rejected/unselected collection artifacts; NOT training data
  extraction_staging/     # 3 proposals, including an incorrectly detected Latin prefix
  collection_*.json       # search, download, deduplication and extraction audits
  representative_samples.jpg
```

**OCR/AI guesses must NOT automatically become training ground truth.**

`source_images/` contains full source images. `unlabeled/` contains crops, not
verified labels. `review_queue.csv` records review work, not approval. All nine
source/review records are `NEEDS_REVIEW`; candidate transcriptions are blank.
The review queue's two nonempty crop paths are:

- `unlabeled/header_000016_p1_r1.png`, 124×42.
- `unlabeled/header_000036_p1_r1.png`, 131×15.

The earlier standalone crop is also in `unlabeled/`, but is not a row in the web
collection review queue. `dataset_info.json` still reports one unlabeled sample
because the collection utilities kept their own provenance manifests. Thus its
count of 1 is not a filesystem total; **three unlabeled crops currently exist**.
The collection tools did not create verified provenance for the other two crops.

The nine sources include an RTO-prefix/two-row example and an adjacent official
placard case, explicitly described in source notes. They are candidates for human
suitability review, not nine guaranteed clean decorative-header training samples.
The collection reached nine sources, not its target of 100. Seven sources have
`HEADER_EXTRACTION_FAILED`; the original images remain available.

Public source URLs and page URLs are retained. Public access is not a verified
reuse license; permissions must be reviewed before training/distribution.
Do not train from staging folders or from the ten-panel preview: the preview
contains nine unique sources and one additional crop view, not ten unique photos.

## 7. Main source files and responsibilities

| File | Important entry points | Input → output / connection |
|---|---|---|
| `main.py` | `parse_args`, `infer_all`, `infer`, `PlateInference`, `OCRRouting`, `_route_ocr`, `_reconcile_ocr`, `_finalize_acceptance`, `_complete_plate_text` | Orchestrates models, fallbacks, independent registration/header results, CLI output and optional diagnostics |
| `src/detector.py` | `PlateDetector`, `PlateDetection` | PIL image → confidence-sorted plate boxes and expanded crops; wraps YOLO |
| `src/row_detector.py` | `RowDetector.process`, `_select_registration_rows`, `_classify_rows`, `_recover_small_text`, `RowResult` | Plate crop → rectified row images, selected/ignored/classified metadata and stitched registration image |
| `src/preprocess.py` | `load_image`, `expanded_crop`, `rectify_quad`, `stitch_rows`, `mild_ocr_preprocess` | EXIF/RGB, crop expansion, OpenCV warp, row geometry and legacy enhancement |
| `src/recognizer.py` | `PARSeqRecognizer`, `OCRResult`, `input_image`, `_tensor` | Local checkpoint + row image → raw/filtered prediction and confidence |
| `src/header_ocr.py` | `tighten_header`, `load_header_recognizer` | Geometry-only header tightening and gated loading of a dedicated header model |
| `src/utils.py` | `choose_device`, `clean_plate_text`, `resolve_manifest_path`, `read_jsonl` | Device selection, legacy Latin cleanup and portable dataset path handling |
| `scripts/train_ocr.py` | `OCRRecord`, `OCRDataset`, `load_datasets`, `run_epoch`, recovery/transfer functions | JSONL or CSV OCR data → PARSeq training/evaluation, metrics and checkpoints |
| `scripts/train_detector.py` | `main` | Whole-plate YOLO data → validation/preflight, resumable training and copied best checkpoint |
| `scripts/train_row_detector.py` | `main` | OBB row data → resumable YOLO-OBB training and copied best checkpoint |
| `scripts/_training.py` | `training_lock`, `find_yolo_checkpoint`, `inspect_yolo_checkpoint`, `runtime_yolo_yaml`, `save_yolo_best` | Shared checkpoint validation, locks, runtime YAMLs, recovery and interrupt handling |
| `scripts/train_header_ocr.py` | `verified_records`, `split_real`, `initialize_model`, `HeaderData`, `main` | Verified real crops + synthetic replay + mixed-v2 base → isolated header fine-tune/evaluation |
| `scripts/prepare_real_header_dataset.py` | `store_sample`, `main` | Full plate image + optional manually verified text → uniquely named crop and provenance |
| `scripts/generate_mixed_ocr_dataset.py` | `generate`, `render_plate`, `sample_definition`, `deterministic_split` | Seed/count/safe fonts → synthetic registration images, labels, metadata and integrity report |
| `scripts/diagnose_mixed_ocr.py` | `font_capability_report`, alignment/metrics functions, `main` | Dataset + checkpoint → font/script/category error report; defaults still refer to mixed v1 |
| `scripts/diagnose_real_ocr.py` | `main` | One image + manually supplied diagnostic box → frozen-model crop/preprocessing comparisons |
| `scripts/evaluate.py` | `evaluate_ocr`, `main` | Filtered original detector test split + legacy Latin OCR test → evaluation JSON; not mixed/header evaluation |
| `scripts/inspect_dataset.py` | Dataset audit entry point | Existing data → corruption/annotation/manifest/duplicate report; writes a report |
| `scripts/prepare_dataset.py` | `make_manifest_portable`, `main` | Original JSONL → `.portable.jsonl` copies; does not replace originals |
| `scripts/validate_yolo_labels.py` | Validation/cache/parser helpers | Labels/YAML → validation report; selected modes clear/rebuild caches |
| `scripts/prepare_detector_improved.py` | `audit_yolo`, `audit_voc`, `deduplicate`, `write_dataset`, `main` | Original/Kaggle data → improved detector copy and provenance/validation reports |

Collection-only utilities also exist:

| Script | Role and limitation |
|---|---|
| `collect_public_header_images.py` | Downloads a previously archived URL inventory, checks source-page availability, records hashes/status; not an autonomous search engine |
| `review_public_header_collection.py` | Produces contact sheets from downloaded candidates |
| `audit_public_header_similarity.py` | Computes 63-bit DCT perceptual hashes; distance ≤10 proposes visual comparison, not automatic truth |
| `extract_public_header_collection.py` | Uses existing detector/row/tightening code on explicitly accepted source IDs; saves proposals |
| `finalize_public_header_collection.py` | Publishes reviewed source/crop manifests and counts using exclusive manifest creation |

These utilities implement the existing collection batch and decisions, not a
general review application. The finalizer refuses to overwrite existing manifests;
do not blindly rerun it as an append/merge operation. A later batch needs an
intentional merge that preserves provenance and existing manually verified labels.

## 8. main.py CLI

Live help was inspected with `.venv/bin/python main.py --help`.

```bash
cd /Users/gauravghanekar/Downloads/number_data
.venv/bin/python main.py --image "/path/to/image.jpg" --device mps --verbose
```

| Option | Actual default / behavior |
|---|---|
| `--image PATH` | One image; mutually exclusive with `--folder` |
| `--folder PATH` | Processes supported files directly inside folder, sorted; **not recursive** |
| `--device` | `auto`; CUDA → MPS → CPU; explicit value passed through |
| `--det-weights` | `models/plate_detector/best.pt` |
| `--row-weights` | `models/row_detector/best.pt` |
| `--ocr-weights` | `models/ocr/best.pt` |
| `--devanagari-ocr-weights` | `models/ocr_devanagari/best.pt`; optional when missing |
| `--mixed-ocr-weights` | **`models/ocr_mixed_v2/best.pt`**; optional when missing |
| `--header-ocr-weights` | `models/ocr_header_real_v1/best.pt`; load only if present and eligible |
| `--det-imgsz` | 640; consider larger sizes for genuinely small/distant plate pixels |
| `--det-conf` | 0.30 for primary detection |
| `--det-nms-iou` | 0.60 post-detection duplicate IoU threshold |
| `--det-fallback` / `--no-det-fallback` | Higher-resolution fallback enabled by default |
| `--det-fallback-imgsz` | 2560 |
| `--det-fallback-conf` | 0.05; also used by tile fallback |
| `--det-tile-fallback` / `--no-det-tile-fallback` | Tile fallback enabled by default |
| `--det-tile-size` | 1280 |
| `--det-tile-overlap` | 0.25 |
| `--row-conf` | 0.25 compatibility argument; does not gate row acceptance or configure the RowDetector created by main |
| `--ocr-conf` | 0.80 candidate threshold; not the final auto-accept threshold |
| `--output-dir` | `outputs`; selects artifact destination, does not itself enable writes |
| `--save-results` | Opt-in intermediate artifacts for successful/saveable OCR paths |
| `--save-debug` | Opt-in structured diagnostics, including review/rejected candidates |
| `--verbose` | Prints model routing, row details and fallback diagnostics |

Fallbacks run only when **primary detection returns zero boxes**. If primary is
empty, tiles may still run after the full-image fallback finds boxes, because
their condition also checks the original primary count. Tile boxes are translated
to full-image coordinates, then merged with fallback boxes through NMS and nested
duplicate suppression.

Missing plate/row/primary Latin checkpoints cause startup errors. Missing optional
Devanagari/mixed checkpoints are omitted. A present but incompatible optional OCR
checkpoint can still cause initialization failure; do not assume every model-load
error has automatic recovery. Header checkpoint failures are caught separately
and fall back to the current candidate system.

Supported folder suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.
Normal completion can return exit code 0 even with no accepted plate; consumers
must inspect result statuses, not just the process exit code. Startup/path failures
use exit code 2 in the handled cases.

## 9. Output format and reusable results

Terminal summaries contain `Accepted Plates`, `Review Candidates`, and
`Rejected Candidates`. Review count includes every status other than `ACCEPTED`
and `REJECTED`, including `LOW_CONFIDENCE` and `PARTIAL_VISIBLE`.

| Field | Meaning |
|---|---|
| `Text:` | Current CLI label for primary plate text; the literal label is not `Plate Text:` |
| `Header Text` | Independently recognized auxiliary text, or `NONE` |
| `Header Status` | Printed as `REVIEW_REQUIRED (independent OCR prediction)` when header text exists |
| `Registration Text` | Independently reconciled registration only |
| `Full Plate Text` | Reading-order composition of registration and useful recognized headers |
| `Raw OCR Text` | Uncleaned prediction retained by selected evaluation/reconciliation |
| `Normalized Text` | Registration text after current NFC/filtering/narrow postprocessing rules |
| `Detection Confidence` | Plate-detector box confidence; not OCR confidence |
| `OCR Confidence` | Selected registration OCR confidence, not boosted by headers |
| `Format Validation` | `VALID_FORMAT`, `POTENTIAL_FORMAT`, `INVALID_FORMAT`, or unrun state |
| `Final Status` | `ACCEPTED`, `REVIEW_REQUIRED`, `LOW_CONFIDENCE`, `PARTIAL_VISIBLE`, `REJECTED` |
| `Reason` | Explanation of the final decision |

Unusable text is internally represented by `PLATE_UNREADABLE`. When no header
exists, the CLI does not print a standalone `Header Status: NONE` line; the reusable
dictionary reports `NONE_OR_UNREADABLE`. No detections produces zero summary counts.

`PlateInference.as_dict()` currently returns:

```python
{
    "plate_number": registration_text,
    "registration_text": registration_text,
    "header_text": header_text,
    "full_text": full_text,
    "header_status": "REVIEW_REQUIRED" if header_text else "NONE_OR_UNREADABLE",
    "confidence": ocr_confidence,
    "status": final_status,
    "rows": text_rows,
}
```

`confidence` can be `None` if OCR never supplied it. The full dataclass additionally
contains bounding boxes, detection confidence, risk flags, raw text, errors and
debug image objects; these are not all exposed by `as_dict()`.
This method is an in-process integration aid, not an implemented web endpoint or
CLI `--json` mode. Header text cannot modify registration confidence, validation,
acceptance or the `plate_number` value.

## 10. Row detection and header handling

### 10.1 Registration selection

RowDetector's primary pass uses confidence 0.25 and image size 320. Each OBB
quadrilateral is rectified with OpenCV perspective warping and replicated borders.
Candidates retain their original polygon, bounds, center, size, angle and confidence.

Overlapping row candidates are deduplicated in confidence order using intersection
over the smaller axis-aligned bounds ≥0.55 and compatible vertical center distance
(≤0.65 times the larger height). This is distinct from whole-plate IoU NMS.

The largest-area row seeds the size comparison. A different row is filtered from
registration if both area ratio <0.35 and height ratio <0.65. If more than two
substantial rows remain, ranking uses normalized area, height, detector confidence
and vertical centrality, with weights 1, 0.75, 0.50 and 0.20 respectively.
At most two are retained and sorted top-to-bottom.

The largest **selected** row is `PRIMARY_REGISTRATION`; another selected row is
`SECONDARY_REGISTRATION`. Geometry is a heuristic, not an OCR-based proof of which
line contains a registration. The raw detected count can exceed two without an
automatic plate rejection: the acceptance check sees selected rows.

Main checks selected count 1–2, complete angles, normalized rotation ≤35 degrees,
and rectified size at least 24×8 with area ≥300. Low aggregate row confidence alone
does not reject a plate. No OBB returns the original crop as a one-row sentinel
with angle 0 and confidence 0; main recognizes that fallback explicitly.

### 10.2 Header classification and recovery

An ignored `small header/decorative text` row becomes `HEADER` only if it passes
confidence ≥0.40, height ≥8, width ≥24, aspect ratio ≥2 and normalized angle ≤25°.
Duplicates, lower-ranked substantial rows and other unsuitable regions remain
`DECORATIVE_OR_NOISE`. The primary-pass header classification does not impose a
universal fixed top-of-image coordinate.

If no header is classified, an auxiliary pass runs at `max(960, image_size)` and
confidence at least 0.40. It looks for smaller elongated text above the primary
row in the row's coordinate frame, checks angle compatibility and rejects overlap.
Same-line fragments can be merged geometrically by re-warping original pixels.
The old fragments become noise entries pointing to the merged header.
This recovery never replaces the selected registration rows, and optional recovery
errors are swallowed rather than invalidating registration.

`detected_rows` records the primary detector's count; auxiliary header entries can
increase classified/region metadata without updating that primary count. Use
`classified_rows`, including `source='auxiliary_960'`, to understand all candidates.

### 10.3 Two-row registration is not header plus registration

```text
MH 12
AB 1234
```

These may be two substantial registration rows. They must remain selected
registration rows, not be split merely because one is above the other. When two
separate row images are available, main tries OCR on each independently. If either
candidate is unusable, it falls back to the stitched registration view. Geometry
and plausibility are imperfect; short top rows can require manual review.

### 10.4 Header crop refinement

`tighten_header()` identifies a coherent dark connected-component band, retains
character margins and detached marks in that band, and can exclude isolated
compact outer objects such as bolts. Thresholding is used only to locate geometry;
the returned image retains original RGB pixels. Blank or ambiguous narrow foreground
returns the original crop rather than aggressively cutting it.

This is not a universal bolt/logo removal model. It can retain unwanted objects or
misjudge unusual detached glyphs. Inspect actual crops before labeling. Headers
are OCRed individually; they are not stitched into the registration OCR image.

## 11. OCR reconciliation and validation

### 11.1 Model candidate selection

There is no learned script detector and no arbitrary character-by-character
splicing of different whole-row model predictions. Available models produce full
candidates for a row/view; one is selected. Two actual registration rows may select
different models and then be joined in reading order.

The code's candidate score is:

```text
unusable candidate: -10
otherwise: OCR confidence - 0.07 * abs(base-character count - visual groups)
```

The length penalty is omitted when no visual groups were estimated. Groups come
from dark column runs, not a reliable character count; this only informs routing.
Header OCR sets group count to zero and uses a header-specific text test instead
of registration-format validation.

In the three-model route:

- Devanagari generally must exceed Latin by 0.03; the margin is 0.10 when Latin
  has a valid format and its visual-length gap is at most two.
- Mixed output containing both a Devanagari character and a Latin letter/ASCII
  digit receives a 0.08 routing bonus.
- A usable mixed candidate can win within 0.03 of the previously selected score.
- Different usable predictions within a 0.03 score margin mark the result
  `UNCERTAIN`. Identical predictions are not counted as conflicting alternatives.

The Latin-plus-mixed-only branch has a simpler selection rule; do not assume every
optional-model configuration applies the complete three-model disagreement policy.
Scores are heuristics, not calibrated probabilities.

docTR confidence is the mean maximum token probability over the decoded word before
EOS. A high token average is not the probability that the complete registration is
correct and is not the same quantity as a routing score.

### 11.2 Row and original-crop reconciliation

Reliable one/two-row OCR is attempted first. A `FORMAT_CONFIDENT` row prediction at
confidence ≥0.99 can avoid original-crop OCR; final acceptance and header completion
still run later. Other eligible paths try original-plate OCR. A no-OBB sentinel
already uses that crop and does not need an identical second pass.

Reconciliation behavior includes:

- Prefer a usable result when the other view is unusable.
- For identical text, retain the higher-confidence evaluation.
- If a header was filtered and usable row/full-crop predictions disagree, retain
  the row and require review rather than let header pixels override registration.
- Preserve a higher-confidence strict prefix without inventing a suffix.
- When long predictions differ only at their final character, keep only a common
  prefix of at least seven characters and mark it partial.
- Other conflicting usable results remain uncertain even if one fits a format.

Thus “no splicing” does not mean no postprocessing: the explicitly implemented
visible-prefix consensus rule can shorten a result. No missing character is filled
from a state template or guessed solely to satisfy a registration regex.

### 11.3 Format and final acceptance

The strict confident checks are standard Latin-like registrations and BH-series
patterns. Python `\d` also matches Unicode digits. Broader plausibility permits
4–24 base letter/digit units, requires digits, and accepts letters or short digit-only
strings up to six units. Supported Devanagari is not rejected simply for being mixed.
The broad validator is a plausibility filter, not an exhaustive national format oracle.

After routing/reconciliation, finalization applies these important rules:

- Unreadable sentinel → `REJECTED`.
- Row/direct uncertain disagreement → `REVIEW_REQUIRED`.
- Confidence below 0.85 → `LOW_CONFIDENCE` in the applicable branch.
- A visible partial result remains `PARTIAL_VISIBLE`.
- Valid format normally needs confidence ≥0.98 and no blocking risk for `ACCEPTED`.
- Two-row results require confidence ≥0.995 to avoid that extra review risk.
- Disagreement, uncertain candidate state, yellow/night/glare heuristics and
  possible confusion among `IOBRNM108` can block automatic acceptance.
- Plausible Devanagari/mixed output marked `POTENTIAL_FORMAT` → `REVIEW_REQUIRED`.

Changing `--ocr-conf` does not remove the later 0.85/0.98/0.995 policy thresholds.
Format validation must never be presented as visual ground-truth verification.

## 12. Synthetic-to-real gap

The current mixed-v2 model's synthetic success does not eliminate real stylized
Devanagari failures. `reports/real_ocr_inference.md` records frozen-model tests on
`WhatsApp Image 2026-09-03 at 17.56.53.jpeg`.

That investigation checked:

- Correct mixed-v2 checkpoint, epoch, vocabulary and normalization metadata.
- Correct large registration-row selection and exclusion of header/duplicate regions.
- Raw bounding crop, rectified row and a manually inspected wider tight crop.
- Training/inference tensor compatibility at 128×32.
- Unenhanced versus contrast/sharpened input.
- Aspect-preserving padding versus the standard stretch.

Neither the manually inspected tight crop nor padding recovered the correct number
with the frozen mixed model. The selected rectified row was 275×45 inside a 309×94
expanded plate crop. The report records mixed prediction `२ू३८४६` at confidence
about 0.7965 after input correction; the final result remained `REVIEW_REQUIRED`.
These are recorded predictions, not labels or a new run on today's checkout.

Evidence points to a synthetic-to-real generalization gap involving stylized glyphs,
short-number distributions and real capture conditions. It does not prove that
every remaining failure is OCR-only: new images can still fail detection or row
selection. More epochs on unchanged synthetic data are not automatically the
solution. Obtain verified real examples and a held-out evaluation set first.

## 13. Header OCR status

Header recognition is implemented and isolated, but remains limited in real-world
accuracy and extraction recall. The dedicated fine-tuning code exists; a production
header model does not.

`reports/header_ocr_adaptation.md` records this crop study on
`images_1581599221866_marathi_number_plate.webp`:

| Stage | Recorded dimensions |
|---|---|
| Raw axis-aligned header bounds | 380×148 |
| Rectified header including bolts | 383×49 |
| Geometry-tightened header | 257×49 |
| Final standard OCR input | 128×32 |

The recorded Devanagari prediction changed from `भ दराष्ट्र१९जीसी` to
`महारुष्ट्र१२जीसीसी`; the selected header confidence increased but the text remained
incorrect and reviewable. Registration stayed `२८६८`, confidence 0.987984,
`REVIEW_REQUIRED`. Padding also left errors. Do not adopt these strings as truth.

`load_header_recognizer()` returns `None` for a missing checkpoint and rejects one
unless its config has `task='header_finetune'`, `smoke_test=False`,
`production_ready=True` and `input_normalization='doctr_parseq'`. Main catches
incompatible header loading and uses the existing candidate system.

When an eligible header model loads, it is called only for `HEADER` rows. If its
prediction is low-confidence/unusable, the current code omits that header; it does
not then rerun all general OCR models for that same header. The missing/invalid
checkpoint fallback should not be confused with a per-prediction fallback.

Header evaluation requires supported characters, 2–64 base units, and the configured
OCR confidence threshold. It does not require registration digits or a state pattern.
Any retained header still reports `REVIEW_REQUIRED`, including output from a future
eligible dedicated model. An omitted header may mean absent, missed or unreadable.

## 14. Real header dataset preparation

The following syntax is supported by the inspected `--help` output.
These are **future data-writing commands**, not steps executed for this document.

Extract only, without assigning a label:

```bash
.venv/bin/python scripts/prepare_real_header_dataset.py \
  --image "/path/to/real_plate.jpg" \
  --extract-only --device cpu
```

After a human verifies the exact text in the selected crop:

```bash
.venv/bin/python scripts/prepare_real_header_dataset.py \
  --image "/path/to/real_plate.jpg" \
  --text "YOUR_MANUALLY_VERIFIED_TRANSCRIPTION" \
  --device cpu
```

Replace the placeholder with verified text; never use it literally or substitute
an OCR guess. If several headers are detected, choose `--header-index N`, a
zero-based index in the extracted-header list. This is not necessarily the detector's
raw row index. `--extract-only` can save all detected headers; labeling requires a
single selected header. No detections produces an error, not a fabricated crop.

The tool reruns full-image plate/row detection, tightens each header and saves a
unique UUID crop. `--output-dir` defaults to `data/real_header_ocr_dataset`.
It does not offer a CLI to promote an existing crop or edit a review-queue row
in place. Pointing it at an already cropped header is not equivalent to its
documented full-plate workflow and may fail plate detection.

`store_sample()` records source SHA-256, crop SHA-256, exact NFC text and label
provenance in `dataset_info.json`, under a file lock. With `--text`, it stores the
crop under `images/`, appends `labels.csv` and rebuilds the verified charset. Without
text, the crop goes to `unlabeled/`. Repeated extraction creates another unique file;
it is not source-content deduplication.

Appending a row to `labels.csv` by hand is insufficient for the current trainer:
matching verified provenance and crop hashes are required. A future review/import
workflow must preserve those checks rather than bypass them. Current web-collected
unlabeled crops are not automatically promoted or linked into verified provenance.

## 15. Header fine-tuning

### 15.1 Safeguards and replay

`scripts/train_header_ocr.py` permits only the resolved source path
`models/ocr_mixed_v2/best.pt`. Its default output is the separate
`models/ocr_header_real_v1/`. Output must resolve to a direct `models/ocr_header_*`
directory; the source baseline is not an output target.

The trainer checks that each label:

- Points inside the dataset's `images/` directory.
- Has `verified=True` and `label_source='manual_cli'` provenance.
- Matches the recorded NFC text and crop hash.
- Fits source sequence capacity and vocabulary, unless explicit extension is enabled.

Production requires **100 distinct verified source-image hashes**, not 100 copies
or multiple crops from one image. Current count is zero, so production training is
blocked. The project adaptation report recommends aiming for 500–1,000 diverse
verified crops as an engineering starting point, not an accuracy guarantee.
Split by physical vehicle/plate as well where possible: a hash only groups the
same source file, not different photographs of the same plate.

Real data uses a deterministic 80/20 source-group split. Synthetic replay samples
up to 1,000 existing mixed-v2 training records by default without copying datasets.
A weighted sampler draws approximately 75% real / 25% synthetic examples with
replacement. Production epoch length is `max(batch, 4 * real_training_count)`.
The existing synthetic validation split remains independent.

Defaults: AdamW, LR **2e-5**, weight decay 0.01, cosine schedule, 5 epochs, batch 16,
CPU device. LR must be positive and no greater than 5e-5. Optional
`--augment-headers` adds only very mild brightness, Gaussian noise and a weak blur
blend to real training tensors. It does not implement the entire suggested list of
JPEG/perspective/bolt augmentations. Default header augmentation is off.

### 15.2 Vocabulary and production gate

Without `--extend-vocab`, missing verified characters are an error. With it,
missing characters are appended; existing visual/decoder weights, character token
weights and EOS/SOS/PAD mappings are preserved. Only genuinely new token rows start
randomly. This is fine-tuning, not training from scratch or modifying mixed-v2.

Per-epoch reports contain real-header validation, full synthetic mixed validation,
Latin-only, Devanagari-only, mixed-script, Latin-letter/Devanagari-digit and
Devanagari-letter/ASCII-digit subsets. Metrics include exact, character accuracy,
CER and sample count. Empty subsets are marked unavailable.

The current production eligibility flag requires:

- Not a smoke run.
- Real-header validation exact accuracy ≥0.80.
- Each available synthetic subset loses no more than 0.01 absolute exact accuracy
  and gains no more than 0.005 absolute CER relative to the baseline.

The original mixed-v2 model supplies the synthetic baseline, including when header
vocabulary expands. These gates are not a replacement for a real held-out test set.
Header `best.pt` is chosen by real-header exact improvement, not by eligibility
alone; check that the selected best actually carries `production_ready=True`.

### 15.3 Future commands

Only after sufficient verified data exists:

```bash
.venv/bin/python scripts/train_header_ocr.py \
  --data-dir data/real_header_ocr_dataset \
  --synthetic-dir data/mixed_ocr_dataset_v2 \
  --source models/ocr_mixed_v2/best.pt \
  --checkpoint-dir models/ocr_header_real_v1 \
  --epochs 5 --batch 16 --lr 0.00002 --synthetic-samples 1000 \
  --device mps --extend-vocab
```

Use CPU if MPS is unavailable. Add `--augment-headers` only as a deliberate,
evaluated experiment. Resume with the same command plus `--resume`, retaining
the original epoch target, dataset fingerprints, replay configuration and options.
Changing verified data should be a new intentional run, not a forced incompatible resume.

Evaluate a completed header run using the same configuration plus `--evaluate-only`
instead of `--resume`. This prints the subset report. Training creates
`baseline_evaluation.json` and `evaluation_epoch_NNN.json` inside the separate
header output directory.

A plumbing-only smoke test, also requiring at least one verified sample:

```bash
.venv/bin/python scripts/train_header_ocr.py \
  --smoke-test --extend-vocab \
  --checkpoint-dir models/ocr_header_smoke --device cpu
```

This forces one tiny four-sample epoch, may overlap tiny real train/validation
sets, and cannot qualify for production inference. It is **not** a production model.
Do not run it against `models/ocr_header_real_v1/`. There is no header `--fresh` flag.

## 16. Training and evaluation command reference

**Do not run the training commands merely to set up inference. Existing models are
already trained. These commands document future/manual operations, not actions
performed while writing this file.**

### 16.1 Plate detector — historical/recovery reference only

The current improved run's settings, rather than the older README defaults:

```bash
.venv/bin/python scripts/train_detector.py \
  --data data/det_improved --model yolo11s.pt \
  --epochs 20 --imgsz 960 --batch 8 --workers 0 --device mps --resume
```

This requires compatible recovery in `runs/plate_detector*/weights/`; it may report
that training is already completed. Inspect the selected run before authorizing
any continuation. The original trainer defaults remain `data/det`, 60 epochs,
640 pixels and batch 16, which are not the current improved run configuration.

Even `--preflight-only` writes runtime YAMLs and clears/rebuilds label caches.
It is not a strictly read-only documentation/inspection command. Full training
eventually copies its best checkpoint to the default production detector path.

### 16.2 Row detector — historical/recovery reference only

```bash
.venv/bin/python scripts/train_row_detector.py \
  --data data/rowdet_obb --model yolo11s-obb.pt \
  --epochs 60 --imgsz 320 --batch 32 --workers 0 --device mps --resume
```

Do not retrain the row model as part of a header-OCR adaptation experiment.
Recovery requires the original task, data/runtime YAML, architecture, image size
and total epoch target. Unlike generic OCR, these wrappers do not permit arbitrary
epoch-target extension of an existing YOLO run.

### 16.3 Devanagari OCR — completed-run reference

```bash
.venv/bin/python scripts/train_ocr.py \
  --data-dir data/devenagari_ocr_dataset \
  --checkpoint-dir models/ocr_devanagari --run-name parseq_ocr_devanagari \
  --epochs 30 --batch 64 --lr 0.0003 --workers 0 \
  --val-fraction 0.1 --split-seed 42 --device mps --resume
```

The current `last.pt` is epoch 30. This command should not perform new epochs on a
completed compatible run. Do not respond to “completed” by adding `--fresh`.
`--run-name` is implemented but hidden from argparse help; it distinguishes local
training locks and is not an OCR checkpoint-directory selector.

### 16.4 Mixed-v2 OCR — completed-run reference

```bash
.venv/bin/python scripts/train_ocr.py \
  --data-dir data/mixed_ocr_dataset_v2 \
  --checkpoint-dir models/ocr_mixed_v2 --run-name parseq_ocr_mixed_v2 \
  --epochs 30 --batch 64 --lr 0.0003 --workers 0 \
  --val-fraction 0.1 --split-seed 42 --diagnostic-samples 5 \
  --device mps --resume
```

This is the matching reference configuration, not a recommendation to continue
synthetic training. Generic OCR permits a deliberately increased total epoch
target with scheduler restoration; never do that merely to address one real-image
failure. Keep the known-good baseline protected and plan experiments separately.

### 16.5 Mixed-v2 evaluation — appropriate current diagnostic

```bash
.venv/bin/python scripts/train_ocr.py \
  --data-dir data/mixed_ocr_dataset_v2 \
  --checkpoint-dir models/ocr_mixed_v2 \
  --evaluate-only --batch 64 --workers 0 --device cpu \
  --diagnostic-samples 10
```

This loads `best.pt` and evaluates the existing validation split without optimizer
updates. It can be expensive on CPU. The code prints metrics and sample code points;
it does not save a new OCR evaluation report by default. Data path, vocabulary,
preprocessing and split configuration must remain compatible with the checkpoint.
Evaluation takes training-only batch/LR/target compatibility values from the stored
config. Use `mps` only where the backend actually initializes.

For legacy detector-plus-Latin test evaluation, the supported command is:

```bash
.venv/bin/python scripts/evaluate.py --device cpu
```

It writes `reports/evaluation.json`, generated filtered test inputs and Ultralytics
validation artifacts. It uses `data/det` and `data/ocr_stitched/test.jsonl` internally;
it is not a full end-to-end or mixed-v2/header evaluator.

Header fine-tuning/evaluation commands are in Section 15. New production header
training is currently blocked by insufficient verified real data.

## 17. Checkpoint safety and resume semantics

- `best.pt`: validation-selected inference candidate, not necessarily latest epoch.
- `last.pt`: latest completed epoch, with optimizer/scheduler/config/random state
  in OCR recovery checkpoints.
- `previous.pt`: backup of a prior usable recovery checkpoint, used if newest
  recovery cannot be loaded compatibly.
- `archive/`: intentionally preserved older runs; not automatic inference fallbacks.

OCR writes use temporary files and atomic replacement. Existing valid `last.pt`
is copied into `previous.pt`; a new best replaces `best.pt` only when appropriate.
Checkpoint compatibility compares vocabulary, architecture/data/preprocessing
configuration and other expected fields. The generic OCR trainer disallows a
decreased total epoch target but can extend it with restored scheduling.

Default generic OCR behavior is auto-resume when compatible recovery exists.
`--resume` makes a recovery checkpoint mandatory; `--fresh` archives current
best/last/previous and starts again. **Do not use `--fresh` accidentally on a trained
model.** A 30-epoch target means 30 total epochs, not 30 more epochs.

YOLO recovery searches matching numbered run directories, checks task/class names,
architecture, runtime dataset path, image size and exact epoch target, and requires
optimizer state for incomplete checkpoints. It falls back to compatible previous
weights and preserves numbered runs. Exported `models/.../best.pt` is not the
intended resumable trainer state.

Header fine-tuning additionally fingerprints source checkpoint bytes, verified
real crop/label/source provenance and the synthetic CSV. Resume requires the same
original epoch target. It has a separate output lock and no `--fresh` operation.
Fingerprints do not replace all data curation: the synthetic fingerprint is of
its manifest, not every synthetic image's bytes.

Keep **`models/ocr_mixed_v2/best.pt` as the registration baseline**. A successful
header model belongs in `models/ocr_header_real_v1/` and is used only for headers.
Do not overwrite detector, row or Latin/Devanagari checkpoints to experiment with
header accuracy. Checkpoints are loaded with `weights_only=False`; load trusted
project files only, not untrusted downloaded pickle checkpoints.

## 18. Real image testing

No-debug inference on an existing real image:

```bash
.venv/bin/python main.py \
  --image "/Users/gauravghanekar/Downloads/number_data/images_1581599221866_marathi_number_plate.webp" \
  --device mps --verbose
```

Repeat the stylized registration failure case:

```bash
.venv/bin/python main.py \
  --image "/Users/gauravghanekar/Downloads/number_data/WhatsApp Image 2026-09-03 at 17.56.53.jpeg" \
  --device cpu --mixed-ocr-weights models/ocr_mixed_v2/best.pt --verbose
```

Folder mode, with no artifacts saved:

```bash
.venv/bin/python main.py --folder "/path/to/test_images" --device cpu --verbose
```

Opt-in debug for a selected difficult image:

```bash
.venv/bin/python main.py \
  --image "/path/to/difficult_plate.jpg" --device cpu \
  --verbose --save-debug --output-dir outputs/manual_header_debug
```

Opt-in successful OCR-path artifacts:

```bash
.venv/bin/python main.py --image "/path/to/image.jpg" \
  --device cpu --save-results --output-dir outputs/manual_results
```

Without `--save-debug` and `--save-results`, normal inference does not save these
artifacts. `--verbose` alone prints diagnostics. Use debug only when needed: it
saves many per-model images and can consume substantial storage. Reusing an output
directory for the same image stem may overwrite prior diagnostics; use a deliberate
new output directory when preserving comparison evidence.

`--save-results` runs on intermediate saveable OCR paths before final risk status
is settled. An artifact's existence does not establish final `ACCEPTED` status.

## 19. Debugging guide

### 19.1 Troubleshooting decision flow

```text
No/wrong plate box
  -> inspect original image, EXIF orientation, detector weights and detection counts
  -> inspect small-plate resolution, full-image fallback and tile coordinates

Correct plate but wrong/missing registration row
  -> inspect all OBB regions, selected/ignored rows, overlap and relative sizes
  -> inspect original crop versus rectified row and row/direct fallback path

Correct registration row but wrong OCR
  -> inspect exact checkpoint/vocabulary, raw RGB and 128x32 tensor contract
  -> compare per-model predictions and routing/reconciliation
  -> consider a verified synthetic-to-real gap, not just confidence or format

Correct registration but missing/wrong header
  -> inspect HEADER classification and auxiliary recovery
  -> inspect raw/rectified/tight crop for bolts, clipped marks and registration pixels
  -> inspect vocabulary and independent header predictions
  -> collect verified real data; never use the wrong prediction as a label
```

### 19.2 Saved debug bundle

With `--save-debug`, inspect `OUTPUT_DIR/debug/IMAGE_STEM/`:

- `01_original.png` and `final_detections.png`.
- Per-plate detector crop, row output, stitched view and final OCR-input exports.
- `plate_001/02_plate_detection_crop.png`, `03_plate_crop_expanded.png`,
  `04_row_detector_input.png`, `05_row_detector_all_regions.png`.
- `row_NN_raw.png` and `row_NN_rectified.png` inside the per-plate directory.
- `ocr_call_NN_MODEL_before_resize.png` and `ocr_call_NN_MODEL_final.png` for
  actual Latin, Devanagari, mixed or dedicated-header calls.
- `header_row_N_original.png` and `header_row_N_tight.png` when a header is processed.
- `10_ocr_input_before_resize.png` and `11_ocr_input_final.png` when matched to a
  selected registration prediction.
- `results.json`: boxes, detection sources/counts, row polygons/reasons, per-model
  calls, checkpoint/vocabulary metadata, raw/normalized text, scores, confidence,
  risks, fallback flags, structured full-text result and final status.

Matching debug calls can include multiple calls with the same selected string;
inspect stage/model metadata rather than assuming one filename alone explains the
entire reconciliation. The final PNG is RGB before mean/std normalization, not a
visualization of the normalized floating-point tensor.

Existing diagnostic evidence is in `reports/real_ocr_inference.md`,
`reports/header_ocr_adaptation.md`, `reports/mixed_ocr_diagnosis.json`,
`reports/mixed_ocr_font_capability.json` and corresponding `outputs/` subdirectories.
Older reports describe the state at their own dates; current counts are in this
document and live manifests.

`scripts/diagnose_real_ocr.py` requires `--image`, `--tight-box x1 y1 x2 y2`,
`--output-dir` and optional `--device`. The manually chosen box is an experimental
control only, not a production crop. `scripts/diagnose_mixed_ocr.py` defaults to
mixed v1; explicitly supply v2 data/checkpoint/output paths for a new v2 study.
Both diagnostic scripts write artifacts; neither was run to regenerate evidence
for this documentation task.

## 20. Test suite

| Test file | Tests currently discovered | Protection |
|---|---:|---|
| `tests/test_main_multi_plate.py` | 32 | Multi-plate ordering, failures, NMS/nested boxes, detection fallbacks, row/crop reconciliation, Unicode, acceptance policy and artifact flags |
| `tests/test_real_ocr_inference.py` | 7 | Mixed-v2 default, raw-versus-legacy preprocessing, exact tensor compatibility, header filtering and disagreement/debug behavior |
| `tests/test_plate_full_text.py` | 9 | Independent headers, three-model routing, noise suppression, two-row registration and reading order |
| `tests/test_header_adaptation.py` | 9 | Crop tightening, verified-label/provenance gates, checkpoint transfer/recovery, vocabulary expansion and header isolation |
| `tests/test_mixed_ocr_dataset.py` | 1 | Temporary synthetic generation, Unicode, charset integrity and split-group isolation |

Standard command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

**Actual result on 2026-09-09: 58 tests ran, all passed, in 11.632 seconds.**
Tests do not establish broad real-world OCR accuracy: many inference tests use
fake detectors/recognizers to isolate logic. There are also tests that read the
real mixed-v2 checkpoint and test its tensor/transfer contract.

Test-side effects must be understood before running the full suite:

- The header transfer test makes one optimizer step on a disposable in-memory
  model and synthetic label, without saving it to a production checkpoint.
- The generator test creates 20 synthetic fixtures in a temporary directory.
- Other tests use temporary images, labels and checkpoint files for safety checks.

During this documentation inspection, an attempted in-memory skip did not take
effect, so those fixture tests ran too. No production training entry point was
launched and no project model/dataset files were changed. The complete result is
reported rather than claiming those tests were skipped. A metadata fingerprint of
320,949 project files, excluding environment/git/cache directories, was unchanged
across the test run. Tests were not modified.

If a future task prohibits even temporary fixture generation or in-memory optimizer
steps, explicitly exclude those two test IDs before executing the suite; do not
describe such a subset as a full 58-test run.

## 21. Environment

The inspected host is Apple Silicon (`arm64`) macOS with Python **3.12.14** inside
`.venv`. The application is Python, not the old conda environment mentioned in
`data/DATA.md`.

Pinned requirements and observed installed versions agree:

| Package | Version | Purpose |
|---|---|---|
| torch | 2.7.1 | Model tensors/training/inference, MPS/CPU/CUDA |
| torchvision | 0.22.1 | PyTorch vision dependencies |
| ultralytics | 8.3.253 | YOLO11 detection/OBB/training |
| python-doctr | 0.11.0 | PARSeq implementation |
| opencv-python | 4.11.0.86 | Perspective geometry, components and diagnostic image processing |
| Pillow | 11.3.0 | Image loading/EXIF/resize/rendering |
| numpy | 2.1.3 | Array transforms and geometry |

Also observed installed: `requests` 2.34.2, `fonttools` 4.63.0 and `tqdm` 4.70.0.
Collection/font diagnosis relies on these although they are not explicitly pinned
as direct entries in `requirements.txt`. Check them when recreating the environment.
FastAPI and Uvicorn are **not installed**, and no working API implementation was found.

Setup for a genuinely new checkout/environment, not a command to replace this one:

```bash
cd /Users/gauravghanekar/Downloads/number_data
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use the existing `.venv` here. macOS-specific font paths in the synthetic generator
need review on another OS. Header/other trainer locks use `fcntl`, so Windows is
not a tested drop-in platform.

`choose_device('auto')` selects CUDA, then MPS, then CPU. Explicit `--device mps`
is not an automatic CPU fallback after a backend failure. This inspection process
reported both MPS and CUDA unavailable; CPU is the verified fallback in this tool
environment even though historical training used MPS. Header preparation/training
defaults to the literal `cpu` device rather than calling the shared auto selector.

Ultralytics 8.3.203 has a documented-in-project MPS validation tensor issue; the
shared training code checks affected versions. Do not casually downgrade from the
pinned 8.3.253. Historical metrics and `/home/calua` package advice do not override
the inspected local dependencies.

## 22. NEXT PLANNED BACKEND/API STEP

**PLANNED / NOT YET IMPLEMENTED**

The next backend integration should expose one endpoint, for example:

```text
POST /anpr
  image upload
    -> existing ANPR pipeline/service
    -> structured response with one result per detected plate
```

Reuse `infer_all()` and the `PlateInference.as_dict()` boundary rather than
reimplementing detector/OCR reconciliation inside route code. The frontend should
not coordinate three OCR models or interpret header OCR as registration approval.

Intended per-plate response concept:

```json
{
  "plate_number": "<registration prediction>",
  "header_text": "<independent header prediction or empty>",
  "full_text": "<reading-order combined text>",
  "confidence": 0.0,
  "status": "REVIEW_REQUIRED"
}
```

This is a schema illustration, not an actual recognition result. The eventual
schema should retain current status vocabulary and account for multiple plates,
missing confidence, no detections, failures and independent header status.
`ACCEPTED` must not silently become “ground-truth verified”.

Future implementation considerations include loading models once, bounding upload
sizes, temporary-file cleanup, concurrency/device memory, exception handling and
clear API errors. None is implemented by this documentation. Do not tell a frontend
developer that `/anpr` already exists or that running main.py starts a server.

## 23. Current project status

| Component | Status | Checkpoint / location | Notes |
|---|---|---|---|
| Plate Detector | READY | `models/plate_detector/best.pt` | Trained YOLO11s; difficult small plates may still fail |
| Row Detector | WORKING WITH LIMITATIONS | `models/row_detector/best.pt` | Trained OBB model; imperfect real small-text recall/classification |
| Latin OCR | READY | `models/ocr/best.pt` | Preserved legacy candidate; no Devanagari vocabulary |
| Devanagari OCR | WORKING WITH LIMITATIONS | `models/ocr_devanagari/best.pt` | Strong synthetic performance; no ASCII/Latin tokens |
| Mixed v2 OCR | WORKING WITH LIMITATIONS | `models/ocr_mixed_v2/best.pt` | Current baseline; 98.38% synthetic exact, real stylized failures remain |
| Header Detection | WORKING WITH LIMITATIONS | `src/row_detector.py`, `src/header_ocr.py` | Independent extraction/tightening; only 2 usable crops from 9 collected sources |
| Header OCR | IN PROGRESS | Current candidate fallback; `models/ocr_header_real_v1/` absent | Dedicated trainer exists; no production header checkpoint |
| Real Header Dataset | IN PROGRESS | `data/real_header_ocr_dataset/` | 3 unlabeled crops, 0 verified records; not production-ready |
| Frontend API | NOT STARTED | Planned `POST /anpr` | No backend endpoint or frontend integration implemented |

“READY” here means the trained component is available to run, not an assertion of
perfect accuracy or a deployed/validated production service.

## 24. Next steps, in priority order

1. Continue collecting diverse real Marathi/Devanagari plate/header photographs,
   preserving source URLs and checking reuse permissions.
2. Obtain more successfully extracted, complete, usable header crops; inspect
   missed/incorrect geometry before treating every failure as an OCR problem.
3. Manually verify labels and record matching crop/source provenance. Build a
   held-out real set and prevent source/physical-plate leakage.
4. After the minimum verified-source gate is met, fine-tune a **separate** header
   model with mixed-v2 initialization and synthetic replay.
5. Evaluate real-header accuracy and all synthetic/Latin/Devanagari/mixed subsets;
   inspect reports and real examples beyond the automatic eligibility flag.
6. Integrate an eligible improving checkpoint only through header-specific routing,
   with the mixed-v2 registration baseline unchanged.
7. Create the single backend endpoint using the reusable result boundary.
8. Connect a frontend that exposes review status and independent header uncertainty.

In parallel with data curation, preserve a separate set of verified real
registration-row failures for future domain-gap work. Header-only fine-tuning must
not be presented as fixing stylized registration recognition when the new model is
never used on registration rows.

## 25. Simplified project directory tree

```text
number_data/
  PROJECT_DOCUMENTATION.md
  README.md
  requirements.txt
  main.py
  .venv/
  yolo11s.pt
  yolo11s-obb.pt
  models/
    plate_detector/best.pt
    row_detector/best.pt
    ocr/{best,last,previous}.pt
    ocr_devanagari/{best,last,previous}.pt
      archive/20260903_diagnosed_epoch5_scratch/
    ocr_mixed/{best,last,previous}.pt             # legacy v1
    ocr_mixed_v2/{best,last,previous}.pt
      archive/20260907_110737/
    [ocr_header_real_v1/ is planned; absent]
  data/
    det/
    det_improved/
    rowdet/
    rowdet_obb/
    rowdet_realtrain/
    rowdet_realval/
    ocr/
    ocr_stitched/
    devenagari_ocr_dataset/
    mixed_ocr_dataset/                           # obsolete training baseline
    mixed_ocr_dataset_v2/
    real_header_ocr_dataset/
    det_hardneg/
    mined/
    additional/
    kaggle 1/, kaggle 2/, kaggle 3/
    DATA.md                                     # historical provenance
  src/
    detector.py
    row_detector.py
    preprocess.py
    recognizer.py
    header_ocr.py
    utils.py
  scripts/
    _training.py
    train_detector.py
    train_row_detector.py
    train_ocr.py
    train_header_ocr.py
    prepare_real_header_dataset.py
    generate_mixed_ocr_dataset.py
    diagnose_mixed_ocr.py
    diagnose_real_ocr.py
    evaluate.py
    inspect_dataset.py
    prepare_dataset.py
    prepare_detector_improved.py
    validate_yolo_labels.py
    [public-header collection/review utilities]
  tests/
    test_main_multi_plate.py
    test_real_ocr_inference.py
    test_plate_full_text.py
    test_header_adaptation.py
    test_mixed_ocr_dataset.py
    resume_smoke/
  runs/
    plate_detector3/
    plate_detector4/
    row_detector_obb/
    [other historical run configurations]
  reports/
    dataset_report.json
    det_improved_dataset_report.json
    det_improved_dataset_report.md
    mixed_ocr_diagnosis.json
    mixed_ocr_font_capability.json
    real_ocr_inference.md
    header_ocr_adaptation.md
    real_header_web_collection.md
    [generated runtime YAMLs and validation reports]
  outputs/
    debug/
    real_debug/
    full_text_debug/
    header_crop_study/
    header_refinement_final/
    [other inference artifacts]
  images_1581599221866_marathi_number_plate.webp
  WhatsApp Image 2026-09-02 at 13.38.37.jpeg
  WhatsApp Image 2026-09-03 at 17.56.53.jpeg
```

Brace notation lists the existing named checkpoint files compactly. Bracketed
entries describe categories or explicitly absent/planned directories, not literal
filenames.

## 26. Known do-not-do items

- Do not overwrite `models/ocr_mixed_v2/best.pt` for a header experiment.
- Do not retrain plate or row detectors merely because a correctly cropped header
  is misrecognized.
- Do not turn OCR/AI guesses into verified training ground truth.
- Do not copy review-queue rows into labels without human verification and the
  required source/crop provenance.
- Do not train on `unlabeled/`, staging folders, contact sheets or duplicate images
  as if they were independent approved samples.
- Do not blindly concatenate conflicting OCR predictions or invent missing symbols.
- Do not treat every top/small row as a header or allow header OCR to replace a
  selected registration row.
- Do not use mixed v1's data/model as the production baseline.
- Do not assume the mixed vocabulary contains every Marathi character.
- Do not translate visible scripts/digits to satisfy a preferred format.
- Do not change 128×32 input shape or normalization without checkpoint-compatible
  architecture work and evaluation.
- Do not treat a high token confidence or regex match as proof of character correctness.
- Do not save debug artifacts on every inference run without a storage/debugging need.
- Do not accidentally restart completed training with `--fresh`.
- Do not change dataset/split/vocabulary settings and force an incompatible resume.
- Do not quote synthetic validation numbers as real-world header accuracy.
- Do not use stale README/DATA.md commands without checking the current CLI/code.
- Do not claim an API, trained header model or production-ready real dataset exists.

## 27. Where to continue from here

The immediate task is **verified real-header data acquisition and preparation**, not
more synthetic epochs or an inference rewrite. Start with
`data/real_header_ocr_dataset/source_manifest.csv`, `review_queue.csv`, the three
unlabeled crops and seven documented extraction failures. Review image suitability,
complete crops and source permissions; then add human-verified samples through
`scripts/prepare_real_header_dataset.py` with matching provenance.

Use `reports/header_ocr_adaptation.md` and `reports/real_header_web_collection.md`
for existing evidence, and this document for current counts/commands. The next
training file is `scripts/train_header_ocr.py`, but its production gate currently
has **zero of the required 100 verified independent sources**. Preserve
`models/ocr_mixed_v2/best.pt` and registration behavior throughout.

After a separate header model demonstrably improves real headers without synthetic
regression, load it only via `--header-ocr-weights`. Then implement the planned
single endpoint around `infer_all()`/`PlateInference.as_dict()` and connect a
frontend that keeps registration and header uncertainty separate.
