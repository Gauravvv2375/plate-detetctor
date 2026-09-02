"""Make metadata portable. Input: YAML/JSONL with foreign absolute paths. Processing: safe metadata rewrite. Output: relative YAML and *.portable.jsonl files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import PROJECT_ROOT, read_jsonl, resolve_manifest_path


def make_manifest_portable(manifest: Path) -> tuple[int, int, Path]:
    output = manifest.with_name(f"{manifest.stem}.portable.jsonl")
    valid = missing = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in read_jsonl(manifest):
            repaired = resolve_manifest_path(str(record.get("path", "")), manifest)
            if repaired.is_file():
                record["path"] = repaired.relative_to(PROJECT_ROOT).as_posix()
                valid += 1
            else:
                missing += 1
            if "orig" in record:
                original = resolve_manifest_path(str(record["orig"]), manifest)
                if original.exists():
                    record["orig"] = original.relative_to(PROJECT_ROOT).as_posix()
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return valid, missing, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create non-destructive, portable OCR manifest copies.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    for directory in (args.data / "ocr", args.data / "ocr_stitched"):
        for split in ("train", "val", "test"):
            manifest = directory / f"{split}.jsonl"
            if manifest.is_file():
                valid, missing, output = make_manifest_portable(manifest)
                print(f"{output}: valid={valid} missing={missing}")


if __name__ == "__main__":
    main()
