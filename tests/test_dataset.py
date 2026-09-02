from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from carno_router.dataset import group_time_split, load_examples


ROOT = Path(__file__).resolve().parents[1]


class DatasetTest(unittest.TestCase):
    def test_group_time_split_has_no_overlap(self) -> None:
        rows = load_examples(ROOT / "tests" / "fixtures" / "synthetic_battles.csv")
        split = group_time_split(rows)
        groups = [
            {row.group_id for row in split.train},
            {row.group_id for row in split.calibration},
            {row.group_id for row in split.test},
        ]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(split.receipt["feature_columns"], ["prompt"])

    def test_answer_column_is_rejected(self) -> None:
        source = ROOT / "tests" / "fixtures" / "synthetic_battles.csv"
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0]) + ["strong_answer"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaky.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row, "strong_answer": "leaked response"})
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_examples(path)

    def test_duplicate_identifier_is_rejected(self) -> None:
        source = ROOT / "tests" / "fixtures" / "synthetic_battles.csv"
        with source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        rows[1]["example_id"] = rows[0]["example_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "duplicate example_id"):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()
