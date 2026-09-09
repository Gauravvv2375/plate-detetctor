# Header OCR adaptation

## Verified diagnosis

The detected header's raw axis-aligned bounds are 380×148. Rectification produces
383×49 pixels including mounting bolts. Header-only connected-component geometry
now retains the text band with a margin and excludes isolated compact edge objects:
257×49 pixels on the supplied real image. Thresholding locates bounds only; returned
RGB pixels are not binarized. Ambiguous/narrow foreground leaves the crop unchanged.
This heuristic cannot guarantee exclusion of every bolt or preservation of every
unusual detached glyph; inspect new extracted samples before labeling.

Actual outputs (not ground truth):

| Model | Rectified original | Tightened |
|---|---|---|
| Devanagari | भ दराष्ट्र१९जीसी | महारुष्ट्र१२जीसीसी |
| Mixed v2 | एमड११यू जी सी ८ | मZा१यू 9२जी ११ीी |

The selected header changed from the mixed prediction (.8960 confidence) to the
Devanagari prediction (.9793). It remains incorrect and REVIEW_REQUIRED. Registration
remains २८६८, confidence .987984, REVIEW_REQUIRED, unchanged from before this task.

Padding the tight header to preserve aspect ratio yielded Devanagari
`महाराष्ट्र१२जज सी` and mixed `मर१११य 9२जी ११ीी`; both still contain errors.
128×32 bilinear resize and checkpoint normalization remain in use. A frozen-model
256×32 trial failed with positional-embedding token counts 257 versus 129.
Changing input shape would require an explicit architecture/positional-embedding
adaptation and subsequent training/evaluation. It was not deployed.

Debug crops and full per-call predictions:
`outputs/header_refinement_final/debug/images_1581599221866_marathi_number_plate/`.
Inference emits diagnostic files only with `--save-debug`.

## Data collection

Dataset: `data/real_header_ocr_dataset/` with images/, unlabeled/, labels.csv
(image,text), charset.txt, dataset_info.json. Present state: zero verified labels,
one unlabeled extracted crop. OCR predictions are never passed to the dataset tool.

Extract without assigning a label:

```sh
.venv/bin/python scripts/prepare_real_header_dataset.py --image "some_plate.jpg" --extract-only --device cpu
```

After visually verifying the complete header, provide its exact transcription:

```sh
.venv/bin/python scripts/prepare_real_header_dataset.py --image "some_plate.jpg" --text "YOUR_VERIFIED_TRANSCRIPTION" --device cpu
```

Replace the placeholder; it is not training text. If several headers are detected,
use `--header-index N` (zero-based). Filenames are unique. The tool records source
and crop hashes and explicit manual-label provenance. Repeated extraction does not
overwrite earlier images. Unicode is NFC. Unlabeled entries cannot enter training.

## Fine-tuning, later

Production safeguard: at least 100 distinct verified source-image hashes. Aim for
500–1,000 diverse verified crops before judging real-world accuracy. This is an
engineering starting point, not an accuracy guarantee. Train/validation are split
80/20 by source image, so crops of the same source cannot cross the split. Distinct
photos of the same physical plate should be curated to avoid train/test leakage.

Default replay: sample 1,000 existing synthetic training records in memory; draw
75% real / 25% synthetic examples with replacement. Existing synthetic validation
splits stay untouched. Each epoch samples four times the real training-row count.
Default augmentation is off. Optional `--augment-headers` uses very mild brightness,
noise and blur only on real training rows; no aggressive geometry or fake bolts.

```sh
.venv/bin/python scripts/train_header_ocr.py \
  --data-dir data/real_header_ocr_dataset \
  --checkpoint-dir models/ocr_header_real_v1 \
  --epochs 5 --batch 16 --lr 0.00002 --device mps --extend-vocab
```

The source is exclusively `models/ocr_mixed_v2/best.pt`. Existing vocabulary and all
weights load exactly when labels fit. Marathi headers can contain letters absent
from that limited vocabulary: without `--extend-vocab`, training rejects them.
With this explicit option, only verified missing characters are appended; shared
character weights and EOS/SOS/PAD weights are copied to their correct indices.
Only new token rows are randomly initialized. The original model is never altered.
Labels beyond the checkpoint's sequence capacity are rejected, not truncated.

Resume the same command with `--resume`; retain the original epoch target and data.
Config, source hash, real image/label fingerprints and synthetic manifest fingerprints
must match. Recovery retains optimizer, scheduler and random states. Writes use the
existing atomic best.pt / last.pt / previous.pt mechanism inside the new directory.
No `--fresh`, archive, or checkpoint-deletion operation is provided.

Evaluate with the same configuration plus `--evaluate-only`. Training saves
`baseline_evaluation.json` and `evaluation_epoch_NNN.json`. Evaluation-only prints
the same report. Separate metrics (exact, character_accuracy, CER, sample count)
cover real headers, all synthetic mixed validation, Latin-only, Devanagari-only,
mixed-script and both cross-script digit categories. Registration retention is
measured against the original mixed-v2 model, including when vocabulary expands.

Header inference only loads a non-smoke checkpoint marked eligible by the training
gate: real validation exact ≥80%, each synthetic subset loses ≤1 percentage point
exact and gains ≤0.5 percentage points CER against baseline. These initial gates
do not replace reviewing the reports and a held-out real test set.

For a tiny verified dataset, explicit plumbing test only:

```sh
.venv/bin/python scripts/train_header_ocr.py --smoke-test --extend-vocab \
  --checkpoint-dir models/ocr_header_smoke --device cpu
```

This performs one four-sample epoch, may overlap tiny real train/validation splits,
and is never eligible for production header inference. It still requires at least
one verified sample. No such dataset run was launched in this task.

## Inference isolation and testing

`--header-ocr-weights` defaults to `models/ocr_header_real_v1/best.pt`. Missing,
incompatible, smoke or unqualified files fall back to the existing header candidates.
The dedicated model is called only for HEADER rows, after registration finalization.
Registration model selection, confidence, validation, text and status are unchanged.

Tests include an in-memory optimizer step using an explicitly rendered synthetic
test label, exact source-weight transfer, vocabulary special-token remapping,
recovery rotation, changed-dataset rejection, crop safeguards, manual/unlabeled
handling, header-only routing, and all existing ANPR regressions. Temporary test
checkpoints are unrelated to production models. No real label was fabricated.

Files for this task: main.py (header integration only), src/header_ocr.py,
scripts/prepare_real_header_dataset.py, scripts/train_header_ocr.py,
tests/test_header_adaptation.py, this report, the new unlabeled dataset and debug
artifacts. Detector/training code and existing checkpoints/datasets were not edited.
