#!/usr/bin/env python3
"""Unit tests for the read-only ROS2 recording audit."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_dataset import audit_run, audit_dataset, write_outputs  # noqa: E402


CSV_HEADER = [
    "timestamp", "episode_id", "is_episode_start", "d", "theta", "vx", "vy", "wz",
    "pose_x", "pose_y", "pose_yaw", "goal_x", "goal_y", "lom_file",
    "collision_flag", "stuck_flag",
]


def build_run(root: Path) -> Path:
    run = root / "E01" / "run_test"
    (run / "lom").mkdir(parents=True)
    (run / "lom" / "0000001.png").write_bytes(b"placeholder-not-decoded-by-audit")
    rows = [
        [1.0, 1, 1, 1.2, 0.2, 0.2, 0.0, 0.3, 0, 0, 0, 1, 1, "0000001.png", 0, 0],
        [1.2, 1, 0, 1.0, 0.1, 0.1, 0.0, 0.4, 0, 0, 0, 1, 1, "missing.png", 1, 0],
        [2.0, 2, 1, 0.8, -0.2, 0.2, 0.1, -0.3, 0, 0, 0, 1, 1, "0000001.png", 0, 1],
    ]
    with (run / "data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    (run / "episode_tries.json").write_text(json.dumps({
        "scenario": "E01",
        "entries": [
            {"episode_id": 1, "tries": 2, "class": "straight"},
            {"episode_id": 2, "tries": 1, "class": "complex"},
        ],
    }), encoding="utf-8")
    (run / "descriptor.yaml").write_text(
        "num_samples: 2\nnum_episodes: 3\nn_collision_events: 1\nn_stuck_events: 1\n",
        encoding="utf-8",
    )
    return run


class DatasetAuditTests(unittest.TestCase):
    def test_counts_labels_retries_and_missing_lom(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = build_run(Path(tmp))
            report = audit_run(run)
            self.assertEqual(report["samples"], 3)
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["episodes_by_class"]["straight"], 1)
            self.assertEqual(report["episodes_by_class"]["complex"], 1)
            self.assertEqual(report["samples_by_class"]["straight"], 2)
            self.assertEqual(report["samples_by_class"]["complex"], 1)
            self.assertEqual(report["retry_entries"], 1)
            self.assertEqual(report["retry_count_extra_tries"], 1)
            self.assertEqual(report["missing_lom_references"], 1)
            self.assertEqual(report["collision_samples"], 1)
            self.assertEqual(report["episodes_with_collision"], 1)
            self.assertEqual(report["stuck_samples"], 1)
            self.assertEqual(report["episodes_with_stuck"], 1)
            self.assertEqual(len(report["descriptor_mismatches"]), 1)
            self.assertEqual(report["descriptor_mismatches"][0]["field"], "num_samples")
            self.assertEqual(report["descriptor_episode_count_comparison"]["declared_num_episodes"], 3)
            self.assertEqual(report["descriptor_episode_count_comparison"]["observed_episode_ids_with_samples"], 2)
            self.assertFalse(report["converted_npz_present"])

    def test_failed_path_pool_is_separate_from_episode_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_run(root)
            (root / "E01" / "failed_paths.json").write_text(json.dumps({
                "scenario": "E01",
                "total_target": 10,
                "total_recorded": 8,
                "entries": [
                    {"path_id": 1, "attempts": 3, "class": "semi_complex", "last_status": "collision"},
                    {"path_id": 2, "attempts": 2, "class": "complex", "last_status": "failed"},
                ],
            }), encoding="utf-8")
            report = audit_dataset(root)
            e01 = report["scenarios"][0]
            self.assertEqual(e01["failed_paths"]["entries"], 2)
            self.assertEqual(e01["failed_paths"]["attempts_summed"], 5)
            self.assertEqual(e01["failed_paths"]["by_status"]["collision"], 1)
            self.assertEqual(report["totals"]["failed_path_entries_reported"], 2)
            self.assertEqual(report["totals"]["failed_path_attempts_summed"], 5)

    def test_missing_scenarios_are_reported_without_writing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_run(root)
            report = audit_dataset(root)
            self.assertEqual(report["totals"]["scenario_directories_present"], 1)
            self.assertEqual(report["totals"]["runs"], 1)
            self.assertEqual(len(report["scenarios"]), 30)
            self.assertFalse((root / "E02").exists())

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = audit_dataset(root)
            output = root / "audit_out"
            write_outputs(report, output)
            self.assertTrue((output / "audit_summary.json").is_file())
            with self.assertRaises(FileExistsError):
                write_outputs(report, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
