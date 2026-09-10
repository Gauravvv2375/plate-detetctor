#!/usr/bin/env python3
"""Materialize and verify production Git LFS checkpoints during deployment."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFS_SIGNATURE = b"version https://git-lfs.github.com/spec/v1\n"
REQUIRED_MODELS = {
    "models/plate_detector/best.pt": (
        "abfc74a3e0ff42ec56023048067764af001d4f9a6ccd76663832e237fab38751",
        19_179_866,
    ),
    "models/row_detector/best.pt": (
        "ee0f665c2d8ace9eaf5e67404b6cfe4a283232bada55d7f734347de2ed614913",
        19_815_299,
    ),
    "models/ocr/best.pt": (
        "169aa19cf7989f37fc593c325909a35891926edd49b697775090bea665db24a3",
        285_530_469,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_SIGNATURE)) == LFS_SIGNATURE


def _download_model(
    relative_path: str, destination: Path, expected_hash: str, expected_size: int
) -> None:
    owner = os.environ.get("RAILWAY_GIT_REPO_OWNER")
    repository = os.environ.get("RAILWAY_GIT_REPO_NAME")
    revision = os.environ.get("RAILWAY_GIT_COMMIT_SHA")
    if not all((owner, repository, revision)):
        raise RuntimeError(
            "Required model is still a Git LFS pointer; binary checkpoint was not "
            "materialized. Railway Git repository variables are unavailable."
        )

    encoded_owner = urllib.parse.quote(owner, safe="")
    encoded_repository = urllib.parse.quote(repository, safe="")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(relative_path)
    url = (
        f"https://media.githubusercontent.com/media/{encoded_owner}/{encoded_repository}/"
        f"{encoded_revision}/{encoded_path}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(url, headers={"User-Agent": "railway-model-materializer"})
            digest = hashlib.sha256()
            downloaded_size = 0
            with urllib.request.urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
                    downloaded_size += len(chunk)
        if downloaded_size != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError(f"Downloaded Git LFS model failed verification: {relative_path}")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_required_models() -> None:
    for relative_path, (expected_hash, expected_size) in REQUIRED_MODELS.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"Required production model is missing: {relative_path}")
        if _is_lfs_pointer(path):
            print(f"Materializing Git LFS checkpoint: {relative_path}", flush=True)
            _download_model(relative_path, path, expected_hash, expected_size)
        if _is_lfs_pointer(path):
            raise RuntimeError(
                f"Required model is still a Git LFS pointer; binary checkpoint was not "
                f"materialized: {relative_path}"
            )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Required model has the wrong size: {relative_path} "
                f"({actual_size} != {expected_size})"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Required model failed SHA-256 verification: {relative_path}")
        print(f"Verified production checkpoint: {relative_path}", flush=True)


if __name__ == "__main__":
    materialize_required_models()
