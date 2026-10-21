#!/usr/bin/env python3
"""Safety tests for isolated NPZ staging conversion; no recording is converted."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_config import load_config  # noqa: E402
from prepare_experiment_npz import (  # noqa: E402
    training_union,
    validate_separate_output,
)


class PrepareNpzTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_training_union_is_exactly_nineteen_maps(self):
        expected = (
            "E01", "E02", "E03", "E07", "E09", "E11", "E12", "E13", "E14",
            "E15", "E17", "E18", "E21", "E22", "E25", "E26", "E27", "E28", "E30",
        )
        self.assertEqual(training_union(self.config), expected)
        self.assertEqual(len(training_union(self.config)), 19)

    def test_test_only_maps_are_not_in_conversion_union(self):
        test_only = {"E04", "E05", "E06", "E08", "E10", "E16", "E19", "E20", "E23", "E24", "E29"}
        self.assertFalse(test_only & set(training_union(self.config)))

    def test_same_source_and_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "disjoint trees"):
                validate_separate_output(root, root)

    def test_nested_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "disjoint trees"):
                validate_separate_output(root, root / "staging")

    def test_nonempty_staging_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "staging"
            source.mkdir()
            output.mkdir()
            (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "new or empty"):
                validate_separate_output(source, output)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "do not overwrite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
