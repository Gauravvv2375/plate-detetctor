from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import materialize_production_models as materializer


class ModelMaterializationTests(unittest.TestCase):
    PAYLOAD = b"verified-production-checkpoint"
    RELATIVE_PATH = "models/example/best.pt"
    EXPECTED = (hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD))

    def _configuration(self, root: Path):
        return patch.multiple(
            materializer,
            PROJECT_ROOT=root,
            REQUIRED_MODELS={self.RELATIVE_PATH: self.EXPECTED},
        )

    def test_valid_binary_is_verified_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / self.RELATIVE_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes(self.PAYLOAD)

            with self._configuration(root), patch.object(
                materializer, "_download_model"
            ) as download:
                materializer.materialize_required_models()

            download.assert_not_called()

    def test_missing_binary_is_downloaded_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def download(_relative, destination, _hash, _size):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(self.PAYLOAD)

            with self._configuration(root), patch.object(
                materializer, "_download_model", side_effect=download
            ) as mocked_download:
                materializer.materialize_required_models()

            mocked_download.assert_called_once()
            self.assertEqual((root / self.RELATIVE_PATH).read_bytes(), self.PAYLOAD)

    def test_wrong_downloaded_hash_fails_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def download(_relative, destination, _hash, _size):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"x" * len(self.PAYLOAD))

            with self._configuration(root), patch.object(
                materializer, "_download_model", side_effect=download
            ):
                with self.assertRaisesRegex(RuntimeError, "failed SHA-256"):
                    materializer.materialize_required_models()


if __name__ == "__main__":
    unittest.main()
