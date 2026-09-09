from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_mixed_ocr_dataset import FIXED_EXAMPLES, generate
from scripts.train_ocr import load_datasets


class MixedOCRDatasetTests(unittest.TestCase):
    def test_generator_writes_integral_grouped_unicode_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate(argparse.Namespace(output=root, count=20, seed=123, val_fraction=0.20, smoke=True))

            expected = {
                "images", "labels.csv", "labels.txt", "charset.txt", "dataset_info.json", "integrity_report.json"
            }
            self.assertTrue(expected <= {path.name for path in root.iterdir()})
            with (root / "labels.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            labels = [row["text"] for row in rows]
            self.assertEqual(labels[: len(FIXED_EXAMPLES)], [item[0] for item in FIXED_EXAMPLES])
            self.assertEqual(set((root / "charset.txt").read_text(encoding="utf-8").splitlines()), set("".join(labels)))
            self.assertEqual(json.loads((root / "integrity_report.json").read_text())["status"], "PASS")

            train, validation, vocabulary, max_length, dataset_format = load_datasets(root, 0.20, 42)
            self.assertEqual(dataset_format, "csv")
            self.assertFalse({record.group_id for record in train} & {record.group_id for record in validation})
            self.assertEqual(set(vocabulary), set("".join(labels)))
            self.assertGreaterEqual(max_length, max(map(len, labels)) + 2)


if __name__ == "__main__":
    unittest.main()
