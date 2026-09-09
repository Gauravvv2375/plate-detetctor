# Real-image OCR diagnosis — 2026-09-08

Image: `WhatsApp Image 2026-09-03 at 17.56.53.jpeg` (356 × 105 RGB).
No training was run. Existing checkpoints and datasets were preserved.

## Finding

The remaining primary issue is synthetic-to-real generalization: the frozen mixed
model fails on a visually verified tight crop containing only the four large
registration glyphs. Neither removing enhancement, bypassing rectification, nor
preserving aspect ratio recovered the human-readable number. These experiments
do not distinguish the contribution of stylized glyph shape from short-number
sequence distribution; both differ from the synthetic registration training data.
Synthetic validation accuracy does not measure these real-image conditions.

The plate crop is 309 × 94 with an 8% expansion. Row detection finds three regions,
selects the large 275 × 45 rectified registration row (confidence .8904, angle
−.145°), ignores the small header, and suppresses an overlapping duplicate.
The row excludes header and badge. Its outer glyph edges are tight, but the wider
manual crop also fails. Row selection/rectification were therefore left unchanged.

## Checkpoint and input verification

`models/ocr_mixed_v2/best.pt`: epoch 27, 73-character vocabulary, max_length 34.
Stored exact accuracy .98378648, CER .00149731. SHA256:
`e360124715bee5b690cba599b150da7a3f87770382a7861ffd188fb274789b1f`.

Training and recognition use RGB, bilinear stretching to 128 × 32, float32 / 255,
mean (.694, .695, .693), std (.299, .296, .301). An exact tensor-equality test passes.
Training did not use the inference pipeline's extra contrast/unsharp pass. The
normalized CSV-trained models now receive raw row/crop pixels; legacy Latin OCR
keeps its previous input. Aspect-ratio padding was tested only, not deployed.
The 275:45 row is compressed horizontally relative to its height at 128:32;
padding retains shape but reduces glyph height and still fails.

Two-row training uses horizontal concatenation of synthetic halves; inference
rectifies detected rows and may route each independently, otherwise using the
stitched view. These geometry differences are not involved in this one-row case.
NFC and vocabulary filtering preserve scripts; no transliteration was added.
docTR confidence is mean maximum token probability before EOS, not a calibrated
probability that the full registration is correct. Routing scores are separate.

## Frozen mixed-model isolation

| Input | Prediction | Confidence |
|---|---|---:|
| Original image | जे े४४४६ | .7003 |
| Expanded plate | जए यू   यू ४ | .8347 |
| Raw selected row bounding crop | जे३८४६ | .8266 |
| Rectified row | २ू३८४६ | .7965 |
| Old enhanced row | जे३८४४६ | .7013 |
| Manual digits crop (53,33,342,82) | जे ८९६८४८ | .6875 |
| Manual crop with aspect padding | जL ेेे३८९६ | .6608 |
| Rectified row with aspect padding | २L 3 ८९ ६ | .6983 |

Full three-model predictions and saved pixels are in
`outputs/real_debug/isolation/comparison.json` and its sibling PNG files.
The manual box is diagnostic only and is not used by production inference.

## Changes

- `main.py`: default mixed-v2 checkpoint; raw inputs for normalized OCR models;
  per-call debug capture; preserve disagreement review status; ignore identical
  predictions when checking model disagreement; prevent a header-containing
  fallback from overriding an available registration-row result.
- `src/recognizer.py`: checkpoint metadata and one shared RGB resize method for
  actual tensors and exact debug pixel exports.
- `src/row_detector.py`: expose polygons, bounds, angles, and existing rectified
  region images for diagnostics; selection and geometry algorithms unchanged.
- `scripts/diagnose_real_ocr.py`: reproducible frozen-model crop comparisons.
- `tests/test_real_ocr_inference.py`: preprocessing equality, checkpoint defaults,
  Unicode, header filtering, deduplication, debug capture and disagreement tests.
- This report and debug artifacts. No training or detector implementation changes.

Before: Latin `AP9GVR2966`, Devanagari `२द्द३६`, mixed `जे३८४४६`;
final `२द्द३६`, .8612, REVIEW_REQUIRED.
After: Latin `AP9GVR2966`, Devanagari `२द्द३६`, mixed `२ू३८४६`;
final `२ू३८४६`, .7965, REVIEW_REQUIRED. The final number is still incorrect.

40 tests pass. The requested MPS run failed backend initialization in the tool
environment; all real inference comparisons were completed on CPU.

## Next step

Collect verified real registration-row crops, including stylized Devanagari
digits, short digit-only rows, mixed scripts and conventional plates. Preserve
visible text exactly; split by physical plate/source to prevent leakage. Keep a
held-out real test set and retain synthetic/Latin examples for regression coverage.
Evaluate that set before deciding the scope of a separate future fine-tuning run.
One image cannot establish how widespread this generalization failure is.

Repeat inference from the project root:

```sh
.venv/bin/python main.py --image "WhatsApp Image 2026-09-03 at 17.56.53.jpeg" --device mps --mixed-ocr-weights models/ocr_mixed_v2/best.pt --verbose --save-debug --output-dir outputs/real_debug
```

Use `--device cpu` if MPS cannot initialize. Debug artifacts from this task are
under `outputs/real_debug/after/debug/WhatsApp Image 2026-09-03 at 17.56.53/`.
