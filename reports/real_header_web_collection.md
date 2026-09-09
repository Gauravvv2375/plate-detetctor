# Real header web collection — 2026-09-08

## Outcome: target not met

Collected **9 unique, unverified source candidates**, not the requested 100.
44 queries yielded 308 archived image-result appearances representing 205 distinct
image URLs. Earlier queries without complete raw archives are named in the inventory;
these counts describe the recorded candidate inventory, not every image on every web page.
113 image files were downloaded during collection. Final candidate disposition:

| Disposition | Count |
|---|---:|
| Retained source candidates | 9 |
| Unsuitable (including search-stage rejections) | 128 |
| Duplicate URLs/photos excluded | 13 |
| Unavailable source/image URLs | 55 |

The 128 rejection count is not a subset of 113 downloads: obvious products/non-Indian
sources were also rejected before download. URL canonicalization, SHA-256, DCT
perceptual hashes and visual comparisons were used. Similarity alone did not decide
acceptance. The already-present scooter photograph was excluded from the new batch.
Access-denied responses and certificate errors were not bypassed. Search breadth did
not produce the requested diversity: this batch consists of cars, motorcycles and a
scooter, not a balanced commercial-vehicle/night-time dataset.

## Retained candidates and extraction

| Source ID | Content | Automatic header outcome |
|---|---|---|
| 000014 | Oblique car, small Marathi plate text | Failed |
| 000016 | Close car plate, small additional Devanagari word | Accepted crop, 124×42 |
| 000032 | Car plate with Devanagari side text | Failed: Latin registration prefix incorrectly proposed as header; excluded |
| 000036 | Old car plate with small Devanagari/RTO line | Accepted crop, 131×15 |
| 000042 | Motorcycle, small Marathi line above digits | Failed |
| 000045 | Scooter, overhead perspective, Marathi line | Failed |
| 000058 | Motorcycle, orange plate, small Devanagari text | Failed |
| 000065 | Ambassador, Devanagari RTO prefix above number | Failed; header-like prefix candidate, genuinely part of two-row registration |
| 000175 | Official placard directly above registration | Failed; adjacent placard, not text on the same plate substrate |

The source 000058 news composite shows a vehicle and a detail of the same plate; it
counts as ONE source. Sources 000065 and 000175 are explicitly marked edge cases
for human suitability review, not claimed to be ordinary decorative headers.

The existing PlateDetector, RowDetector and tighten_header were used on CPU without
inference code changes, hard-coded coordinates, or OCR-generated labels. Three header
regions were proposed; one was rejected after visual inspection. Two crops are in
unlabeled/. Seven sources have HEADER_EXTRACTION_FAILED and blank crop paths in the
review queue, so no fabricated crop is substituted. Registration-row classification
was not modified to force extraction.

## Files

All collection data is under `data/real_header_ocr_dataset/`:

- `source_images/`: 9 retained original downloads, bytes match recorded SHA-256.
- `source_manifest.csv`: source URLs, source-page URLs, discovery query and suitability notes.
- `review_queue.csv`: 9 rows; 2 real crop paths, 7 explicit extraction failures.
- `unlabeled/header_000016_p1_r1.png`, `unlabeled/header_000036_p1_r1.png`: new unverified crops.
- `representative_samples.jpg`: 10 panels, **9 unique sources plus one crop view**.
- `collection_inventory.json`, `collection_additional.json`: final search inventories.
- `collection_download_audit.json`, `collection_final_audit.json`: complete candidate dispositions.
- `collection_similarity_audit.json`: perceptual hashes and proposed near-duplicate pairs.
- `collection_extraction_audit.json`: detected plate/row geometry and crop metadata.
- `collection_visual_decisions.json`: assistant visual-suitability decisions, NOT label verification.
- `collection_summary.json`: machine-readable counts and safety hashes.
- `collection_staging/`: rejected/unselected downloads retained for audit, NEVER a training source.
- `extraction_staging/`: raw extraction proposals, including the rejected Latin prefix.

One rejected staging download (candidate 000078) was no longer present at final
audit; it is not referenced by either usable manifest. A newly created extra copy of
source 000061 was moved back to staging after full-size inspection showed that its
plate text was Latin and the Devanagari was on a separate body sticker. It was not
deleted. No existing source image, crop, dataset or checkpoint was removed.

Collection-only utilities added:

- `scripts/collect_public_header_images.py`
- `scripts/review_public_header_collection.py`
- `scripts/audit_public_header_similarity.py`
- `scripts/extract_public_header_collection.py`
- `scripts/finalize_public_header_collection.py`

The finalizer uses exclusive manifest creation and will refuse to overwrite these
manifests on a rerun. The downloader/extractor have resumable inventories. Subsequent
collection batches need an intentional manifest merge, not another finalizer run.

## Verification and safety

- Every new manifest/review row is NEEDS_REVIEW.
- Candidate transcriptions deliberately remain blank. No AI transcription is ground truth.
- `labels.csv` still has only its original `image,text` header: **zero new labels**.
- labels.csv SHA-256: `307a65d79c905883cd34d2898ad00347239e63e30f24f4d7c8bacd6ad89b29f5`.
- `charset.txt` and `dataset_info.json` were not updated by this collection.
- **No model training was started.** Existing training/inference code was not changed.
- The plate, row, Latin, Devanagari and mixed-v2 best.pt hashes match the pre-collection
  hashes; exact values are in collection_summary.json. No model directory was written.
- All **58 existing unittest tests passed**. Their synthetic fixtures use temporary
  directories; no existing mixed dataset was regenerated.
- Manifest checks passed: all 9 source paths exist, both crop paths exist, no duplicate
  retained source hashes, all statuses unverified, no training labels appended.

Public availability is not a verified reuse license. Source permissions remain to be
reviewed before training or redistribution. This is a small review seed, not a
production-ready dataset; the 100-source collection objective remains unmet.
