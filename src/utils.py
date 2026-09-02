"""Shared helpers. Input: paths/text. Processing: validation and normalization. Output: safe paths/text/device names."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PLATE_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def clean_plate_text(text: str) -> str:
    """Uppercase OCR output and retain only the dataset's A-Z/0-9 alphabet."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def choose_device(requested: str = "auto") -> str:
    """Select CUDA, then Apple MPS, then CPU without hard-coding one platform."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error


def resolve_manifest_path(stored: str, manifest: Path) -> Path:
    """Resolve current paths and repair old /.../data/<relative> paths in memory."""
    candidate = Path(stored).expanduser()
    parts = candidate.parts
    # Repair known foreign absolute dataset roots first. Besides being portable,
    # this avoids thousands of slow filesystem probes under a nonexistent /home.
    if candidate.is_absolute() and "data" in parts:
        relative = Path(*parts[parts.index("data") + 1 :])
        repaired = PROJECT_ROOT / "data" / relative
        if repaired.exists():
            return repaired.resolve()
    if not candidate.is_absolute():
        project_relative = PROJECT_ROOT / candidate
        if project_relative.exists():
            return project_relative.resolve()
    if candidate.exists():
        return candidate.resolve()
    repaired = manifest.parent / candidate.name
    return repaired.resolve()


def batched(values: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
