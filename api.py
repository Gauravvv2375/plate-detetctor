"""HTTP API for ANPR inference."""

from __future__ import annotations

import logging
import os
import resource
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError


def _startup_memory() -> tuple[float, float]:
    """Return current and peak RSS in MiB without adding a runtime dependency."""
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        values: dict[str, float] = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmHWM"}:
                values[key] = float(value.split()[0]) / 1024
        return values.get("VmRSS", 0.0), values.get("VmHWM", 0.0)
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB (handled above through /proc).
    peak_mib = peak / (1024 * 1024)
    return peak_mib, peak_mib


def _startup_step(message: str) -> None:
    current, peak = _startup_memory()
    print(
        f"ANPR STARTUP: {message}; RSS={current:.1f} MiB; PEAK_RSS={peak:.1f} MiB",
        flush=True,
    )


# Railway's build image intentionally excludes checkpoints. Materialize and
# verify them before importing modules that initialize production models.
from scripts.materialize_production_models import materialize_required_models

_startup_step("process initialized")
materialize_required_models()
_startup_step("production checkpoints verified")

from main import infer_all
from src.detector import PlateDetector
from src.header_ocr import load_header_recognizer
from src.recognizer import PARSeqRecognizer
from src.row_detector import RowDetector
from src.utils import PROJECT_ROOT, choose_device


logger = logging.getLogger(__name__)
load_dotenv(PROJECT_ROOT / ".env")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://192.168.137.1").split(",")
    if origin.strip()
]
cors_origin_regex = os.getenv("CORS_ALLOWED_ORIGIN_REGEX") or None
device = choose_device(os.getenv("DEVICE", "auto"))

app = FastAPI(title="ANPR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


def _load_models() -> tuple[
    PlateDetector,
    RowDetector,
    PARSeqRecognizer,
    PARSeqRecognizer | None,
    PARSeqRecognizer | None,
    object | None,
]:
    detector = PlateDetector(PROJECT_ROOT / "models/plate_detector/best.pt", device, 0.30, 640)
    _startup_step("plate detector loaded")
    row_detector = RowDetector(PROJECT_ROOT / "models/row_detector/best.pt", device)
    _startup_step("row detector loaded")
    recognizer = PARSeqRecognizer(PROJECT_ROOT / "models/ocr/best.pt", device)
    _startup_step("Latin OCR loaded")
    devanagari_path = PROJECT_ROOT / "models/ocr_devanagari/best.pt"
    devanagari_recognizer = (
        PARSeqRecognizer(devanagari_path, device)
        if devanagari_path.is_file()
        and devanagari_path.resolve() != Path(recognizer.checkpoint_path)
        else None
    )
    _startup_step("Devanagari OCR loaded")
    mixed_path = PROJECT_ROOT / "models/ocr_mixed_v2/best.pt"
    mixed_recognizer = PARSeqRecognizer(mixed_path, device) if mixed_path.is_file() else None
    _startup_step("Mixed OCR v2 loaded")
    header_recognizer = load_header_recognizer(PROJECT_ROOT / "models/ocr_header_real_v1/best.pt", device)
    return (
        detector,
        row_detector,
        recognizer,
        devanagari_recognizer,
        mixed_recognizer,
        header_recognizer,
    )


(
    detector,
    row_detector,
    recognizer,
    devanagari_recognizer,
    mixed_recognizer,
    header_recognizer,
) = _load_models()
_startup_step("FastAPI application ready")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
            temporary_file.write(await file.read())
            temporary_file.flush()
            results = infer_all(
                Path(temporary_file.name),
                detector,
                row_detector,
                recognizer,
                0.80,
                detector_fallback=True,
                detector_tile_fallback=True,
                devanagari_recognizer=devanagari_recognizer,
                mixed_recognizer=mixed_recognizer,
                header_recognizer=header_recognizer,
            )
        return {"results": [result.as_dict() for result in results]}
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a decodable image") from error
    except Exception as error:
        logger.exception("ANPR prediction failed")
        raise HTTPException(status_code=500, detail=str(error)) from error
