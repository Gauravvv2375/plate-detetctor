"""Final evaluation. Input: held-out test splits and trained weights. Processing: Ultralytics metrics and exact/CER OCR scoring. Output: JSON metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image

from scripts._training import filtered_detection_test_yaml
from scripts.train_ocr import edit_distance
from src.preprocess import load_image
from src.recognizer import PARSeqRecognizer
from src.utils import PROJECT_ROOT, choose_device, clean_plate_text, read_jsonl, resolve_manifest_path


def evaluate_ocr(manifest: Path, recognizer: PARSeqRecognizer, limit: int | None = None) -> dict:
    exact = edits = characters = evaluated = missing = 0
    for record in read_jsonl(manifest):
        if limit is not None and evaluated >= limit:
            break
        path = resolve_manifest_path(str(record.get("path", "")), manifest)
        if not path.is_file():
            missing += 1
            continue
        truth = clean_plate_text(str(record.get("text", "")))
        prediction = recognizer.recognize(load_image(path)).text
        exact += prediction == truth
        edits += edit_distance(prediction, truth)
        characters += len(truth)
        evaluated += 1
    return {"records_evaluated": evaluated, "missing": missing, "exact_accuracy": exact / max(1, evaluated), "cer": edits / max(1, characters)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", type=Path, default=PROJECT_ROOT / "models/plate_detector/best.pt")
    parser.add_argument("--ocr-weights", type=Path, default=PROJECT_ROOT / "models/ocr/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--train-sample", type=int, default=0, help="Optional OCR train sample for overfit-gap diagnosis")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/evaluation.json")
    args = parser.parse_args()
    from ultralytics import YOLO

    device = choose_device(args.device)
    detector = YOLO(str(args.det_weights))
    test_yaml, accepted, excluded = filtered_detection_test_yaml(PROJECT_ROOT / "data/det")
    result = detector.val(data=str(test_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=device, verbose=False)
    metrics = {
        "plate_detection_test": {"images_evaluated": accepted, "images_excluded": excluded, "precision": float(result.box.mp), "recall": float(result.box.mr), "map50": float(result.box.map50), "map50_95": float(result.box.map)},
    }
    recognizer = PARSeqRecognizer(args.ocr_weights, device)
    metrics["ocr_test"] = evaluate_ocr(PROJECT_ROOT / "data/ocr_stitched/test.jsonl", recognizer)
    if args.train_sample > 0:
        metrics["ocr_train_sample"] = evaluate_ocr(PROJECT_ROOT / "data/ocr_stitched/train.jsonl", recognizer, args.train_sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
