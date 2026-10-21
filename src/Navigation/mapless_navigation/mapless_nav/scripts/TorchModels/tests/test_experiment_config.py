#!/usr/bin/env python3
"""Dependency-free integrity tests for the fixed experimental scenario splits."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_config import (  # noqa: E402
    EXPECTED_SPLITS,
    SCENARIOS,
    load_config,
    validate_config,
)


class ExperimentSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_complete_scenario_universe(self):
        self.assertEqual(len(SCENARIOS), 30)
        self.assertEqual(tuple(self.config["dataset_scenarios"]), SCENARIOS)

    def test_exact_frozen_splits(self):
        for name, expected in EXPECTED_SPLITS.items():
            exp = self.config["experiments"][name]
            self.assertEqual(tuple(exp["train_scenarios"]), expected["train_scenarios"])
            self.assertEqual(tuple(exp["test_scenarios"]), expected["test_scenarios"])
            self.assertEqual(len(exp["train_scenarios"]), 8 if name == "A" else 17)
            self.assertEqual(len(exp["test_scenarios"]), 22 if name == "A" else 13)

    def test_no_intersection_and_full_coverage(self):
        universe = set(SCENARIOS)
        for exp in self.config["experiments"].values():
            train = set(exp["train_scenarios"])
            test = set(exp["test_scenarios"])
            self.assertFalse(train & test)
            self.assertEqual(train | test, universe)

    def test_required_legacy_scenarios_are_fixed(self):
        for exp in self.config["experiments"].values():
            self.assertTrue({"E01", "E02", "E03"}.issubset(exp["train_scenarios"]))
            self.assertTrue({"E04", "E05", "E06"}.issubset(exp["test_scenarios"]))

    def test_validation_is_episode_level(self):
        self.assertEqual(self.config["validation"]["unit"], "episode")
        self.assertGreater(self.config["validation"]["fraction"], 0)
        self.assertLess(self.config["validation"]["fraction"], 1)

    def test_duplicate_scenario_is_rejected(self):
        broken = copy.deepcopy(self.config)
        broken["experiments"]["A"]["train_scenarios"].append("E01")
        with self.assertRaisesRegex(ValueError, "duplicate TRAIN"):
            validate_config(broken)

    def test_train_test_intersection_is_rejected(self):
        broken = copy.deepcopy(self.config)
        broken["experiments"]["A"]["test_scenarios"][0] = "E01"
        with self.assertRaisesRegex(ValueError, "intersection"):
            validate_config(broken)

    def test_missing_scenario_is_rejected(self):
        broken = copy.deepcopy(self.config)
        broken["experiments"]["B"]["test_scenarios"].remove("E29")
        with self.assertRaisesRegex(ValueError, "cover E01-E30"):
            validate_config(broken)

    def test_unspecified_train_change_is_rejected(self):
        broken = copy.deepcopy(self.config)
        broken["experiments"]["A"]["train_scenarios"][3] = "E13"
        with self.assertRaisesRegex(ValueError, "TRAIN differs"):
            validate_config(broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
