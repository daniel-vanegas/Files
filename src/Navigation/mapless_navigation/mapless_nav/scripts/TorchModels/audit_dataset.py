#!/usr/bin/env python3
"""Read-only audit of ROS 2 dataset recordings before conversion/training.

The script never edits recording files or generates trajectories. It scans
CSV, episode_tries.json, descriptors and referenced LOM file names, then
writes a machine-readable audit to a separate output directory if requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SCENARIOS = tuple(f"E{i:02d}" for i in range(1, 31))
CLASSES = ("straight", "semi_complex", "complex", "v_complex")
REQUIRED_COLUMNS = {
    "episode_id", "is_episode_start", "d", "theta", "vx", "vy", "wz",
    "pose_x", "pose_y", "pose_yaw", "goal_x", "goal_y", "timestamp", "lom_file",
}


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Audit one run folder without loading the full LOM image content."""
    csv_path = run_dir / "data.csv"
    tries_path = run_dir / "episode_tries.json"
    descriptor_path = run_dir / "descriptor.yaml"
    npz_path = run_dir.with_suffix(".npz")

    result: dict[str, Any] = {
        "run": str(run_dir),
        "csv_present": csv_path.is_file(),
        "episode_tries_present": tries_path.is_file(),
        "descriptor_present": descriptor_path.is_file(),
        "converted_npz_present": npz_path.is_file(),
        "samples": 0,
        "episodes": 0,
        "episode_sample_counts": {},
        "episode_classes": {},
        "episodes_by_class": {c: 0 for c in CLASSES},
        "samples_by_class": {c: 0 for c in CLASSES},
        "retry_entries": 0,
        "retry_count_extra_tries": 0,
        "missing_class_episodes": [],
        "unknown_class_entries": [],
        "duplicate_episode_ids_in_sidecar": [],
        "sidecar_episode_ids_without_samples": [],
        "sample_episode_ids_without_sidecar": [],
        "missing_lom_references": 0,
        "empty_lom_name_rows": 0,
        "duplicate_sample_keys": 0,
        "bad_numeric_rows": 0,
        "collision_samples": 0,
        "stuck_samples": 0,
        "episodes_with_collision": 0,
        "episodes_with_stuck": 0,
        "csv_missing_columns": [],
        "descriptor_mismatches": [],
        "errors": [],
    }

    if not csv_path.is_file():
        result["errors"].append("missing data.csv")
        return result

    try:
        sidecar: dict[int, dict[str, Any]] = {}
        duplicate_ids: set[int] = set()
        if tries_path.is_file():
            with tries_path.open("r", encoding="utf-8") as stream:
                sidecar_obj = json.load(stream)
            for item in sidecar_obj.get("entries", []):
                eid = _int(item.get("episode_id"))
                if eid is None or eid < 0:
                    result["errors"].append(f"invalid sidecar episode_id: {item!r}")
                    continue
                if eid in sidecar:
                    duplicate_ids.add(eid)
                sidecar[eid] = item
                label = str(item.get("class", "unknown"))
                if label not in CLASSES:
                    result["unknown_class_entries"].append({"episode_id": eid, "class": label})
                tries = max(1, _int(item.get("tries"), 1) or 1)
                if tries > 1:
                    result["retry_entries"] += 1
                    result["retry_count_extra_tries"] += tries - 1
        else:
            sidecar = {}
        result["duplicate_episode_ids_in_sidecar"] = sorted(duplicate_ids)

        episodes_samples: Counter[int] = Counter()
        episodes_class: dict[int, str] = {}
        episode_collision: set[int] = set()
        episode_stuck: set[int] = set()
        sample_keys: set[tuple[str, str, str]] = set()
        class_samples: Counter[str] = Counter()
        observed_columns: set[str] = set()
        missing_lom = 0
        empty_lom = 0
        bad_numeric = 0
        collision_samples = 0
        stuck_samples = 0
        duplicate_sample_keys = 0
        missing_numeric_required = ("episode_id", "timestamp", "vx", "vy", "wz", "d", "theta")

        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            observed_columns = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_COLUMNS - observed_columns)
            result["csv_missing_columns"] = missing_columns
            if missing_columns:
                result["errors"].append("CSV missing required columns")

            for row in reader:
                result["samples"] += 1
                eid = _int(row.get("episode_id"))
                if eid is None or eid < 0:
                    bad_numeric += 1
                    continue

                episodes_samples[eid] += 1
                for col in missing_numeric_required:
                    val = _int(row.get(col)) if col == "episode_id" else _float(row.get(col))
                    if val is None:
                        bad_numeric += 1
                        break

                lom_name = (row.get("lom_file") or "").strip()
                if not lom_name:
                    empty_lom += 1
                    missing_lom += 1
                elif not (run_dir / "lom" / lom_name).is_file():
                    missing_lom += 1

                key = (str(eid), (row.get("timestamp") or "").strip(), lom_name)
                if key in sample_keys:
                    duplicate_sample_keys += 1
                sample_keys.add(key)

                if _int(row.get("collision_flag"), 0) == 1:
                    collision_samples += 1
                    episode_collision.add(eid)
                if _int(row.get("stuck_flag"), 0) == 1:
                    stuck_samples += 1
                    episode_stuck.add(eid)

                item = sidecar.get(eid)
                label = str(item.get("class", "unknown")) if item else "unknown"
                episodes_class[eid] = label
                class_samples[label] += 1

        all_episode_ids = set(episodes_samples)
        sidecar_ids = set(sidecar)
        result["episodes"] = len(all_episode_ids)
        result["episode_sample_counts"] = {str(k): v for k, v in sorted(episodes_samples.items())}
        result["episode_classes"] = {str(k): episodes_class.get(k, "unknown") for k in sorted(all_episode_ids)}
        result["samples_by_class"] = {c: int(class_samples[c]) for c in CLASSES}
        result["samples_by_class"]["unknown"] = int(class_samples["unknown"])
        result["episodes_by_class"] = {
            c: sum(1 for eid in all_episode_ids if str(sidecar.get(eid, {}).get("class", "unknown")) == c)
            for c in CLASSES
        }
        result["episodes_by_class"]["unknown"] = sum(
            1 for eid in all_episode_ids if eid not in sidecar or str(sidecar[eid].get("class", "unknown")) not in CLASSES
        )
        result["missing_class_episodes"] = sorted(
            eid for eid in all_episode_ids
            if eid not in sidecar or str(sidecar[eid].get("class", "unknown")) not in CLASSES
        )
        result["sidecar_episode_ids_without_samples"] = sorted(sidecar_ids - all_episode_ids)
        result["sample_episode_ids_without_sidecar"] = sorted(all_episode_ids - sidecar_ids)
        result["missing_lom_references"] = missing_lom
        result["empty_lom_name_rows"] = empty_lom
        result["duplicate_sample_keys"] = duplicate_sample_keys
        result["bad_numeric_rows"] = bad_numeric
        result["collision_samples"] = collision_samples
        result["stuck_samples"] = stuck_samples
        result["episodes_with_collision"] = len(episode_collision)
        result["episodes_with_stuck"] = len(episode_stuck)

        descriptor_text = descriptor_path.read_text(encoding="utf-8") if descriptor_path.is_file() else ""
        descriptor: dict[str, str] = {}
        for line in descriptor_text.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                descriptor[k.strip()] = v.strip()
        # num_samples is a direct row count and should match data.csv.
        declared_samples = _int(descriptor.get("num_samples"))
        if declared_samples is not None and declared_samples != result["samples"]:
            result["descriptor_mismatches"].append({
                "field": "num_samples",
                "declared": declared_samples,
                "observed_csv_rows": result["samples"],
            })

        # num_episodes can reflect recorder/sampler attempts, whereas this
        # audit counts episode IDs that actually emitted at least one CSV row.
        # Preserve both quantities without automatically treating that
        # semantic difference as data corruption.
        declared_episodes = _int(descriptor.get("num_episodes"))
        result["descriptor_episode_count_comparison"] = {
            "declared_num_episodes": declared_episodes,
            "observed_episode_ids_with_samples": result["episodes"],
            "difference_declared_minus_observed": (
                declared_episodes - result["episodes"]
                if declared_episodes is not None else None
            ),
            "interpretation_note": (
                "May include recorded attempts without emitted samples; compare "
                "with episode_tries.json and failed-path reports before calling it an error."
            ),
        }

        result["descriptor_declared_samples"] = declared_samples
        result["descriptor_declared_episodes"] = declared_episodes
        result["descriptor_collision_events"] = _int(descriptor.get("n_collision_events"))
        result["descriptor_stuck_events"] = _int(descriptor.get("n_stuck_events"))
        result["descriptor_episodes_with_collision"] = _int(descriptor.get("n_episodes_with_collision"))
        result["descriptor_episodes_with_stuck"] = _int(descriptor.get("n_episodes_with_stuck"))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")

    return result


def audit_dataset(data_root: Path) -> dict[str, Any]:
    scenario_reports = []
    for scenario in EXPECTED_SCENARIOS:
        scenario_dir = data_root / scenario
        if not scenario_dir.is_dir():
            scenario_reports.append({
                "scenario": scenario,
                "directory_present": False,
                "runs": [],
                "run_count": 0,
                "errors": ["missing scenario directory"],
            })
            continue

        run_dirs = sorted(
            p for p in scenario_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        )
        runs = [audit_run(p) for p in run_dirs]

        # failed_paths.json describes path-pool routes that exhausted retries
        # before being recorded; this is distinct from per-episode tries in
        # each run's episode_tries.json.
        failed_paths_path = scenario_dir / "failed_paths.json"
        failed_paths = {
            "file_present": failed_paths_path.is_file(),
            "entries": 0,
            "attempts_summed": 0,
            "by_status": {},
            "by_class": {},
            "total_target": None,
            "total_recorded": None,
            "error": None,
        }
        if failed_paths_path.is_file():
            try:
                failure_doc = json.loads(failed_paths_path.read_text(encoding="utf-8"))
                entries = failure_doc.get("entries", [])
                by_status = Counter(str(entry.get("last_status", "unknown")) for entry in entries)
                by_class = Counter(str(entry.get("class", "unknown")) for entry in entries)
                failed_paths.update({
                    "entries": len(entries),
                    "attempts_summed": sum(max(0, _int(entry.get("attempts"), 0) or 0) for entry in entries),
                    "by_status": dict(by_status),
                    "by_class": dict(by_class),
                    "total_target": _int(failure_doc.get("total_target")),
                    "total_recorded": _int(failure_doc.get("total_recorded")),
                })
            except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
                failed_paths["error"] = f"{type(exc).__name__}: {exc}"

        scenario_reports.append({
            "scenario": scenario,
            "directory_present": True,
            "failed_paths": failed_paths,
            "runs": runs,
            "run_count": len(runs),
            "samples": sum(r.get("samples", 0) for r in runs),
            "episodes": sum(r.get("episodes", 0) for r in runs),
            "episodes_by_class": {
                c: sum(r.get("episodes_by_class", {}).get(c, 0) for r in runs)
                for c in (*CLASSES, "unknown")
            },
            "samples_by_class": {
                c: sum(r.get("samples_by_class", {}).get(c, 0) for r in runs)
                for c in (*CLASSES, "unknown")
            },
            "missing_lom_references": sum(r.get("missing_lom_references", 0) for r in runs),
            "missing_npz_runs": sum(not r.get("converted_npz_present", False) for r in runs),
            "error_count": sum(len(r.get("errors", [])) for r in runs),
        })

    totals = {
        "scenario_directories_present": sum(s.get("directory_present", False) for s in scenario_reports),
        "scenarios_with_runs": sum(s.get("run_count", 0) > 0 for s in scenario_reports),
        "runs": sum(s.get("run_count", 0) for s in scenario_reports),
        "samples": sum(s.get("samples", 0) for s in scenario_reports),
        "episodes": sum(s.get("episodes", 0) for s in scenario_reports),
        "missing_npz_runs": sum(s.get("missing_npz_runs", 0) for s in scenario_reports),
        "missing_lom_references": sum(s.get("missing_lom_references", 0) for s in scenario_reports),
        "failed_path_entries_reported": sum(s.get("failed_paths", {}).get("entries", 0) for s in scenario_reports),
        "failed_path_attempts_summed": sum(s.get("failed_paths", {}).get("attempts_summed", 0) for s in scenario_reports),
        "errors": sum(s.get("error_count", 0) for s in scenario_reports),
        "episodes_by_class": {
            c: sum(s.get("episodes_by_class", {}).get(c, 0) for s in scenario_reports)
            for c in (*CLASSES, "unknown")
        },
        "samples_by_class": {
            c: sum(s.get("samples_by_class", {}).get(c, 0) for s in scenario_reports)
            for c in (*CLASSES, "unknown")
        },
    }
    return {
        "audit_version": 1,
        "data_root": str(data_root.resolve()),
        "read_only": True,
        "expected_scenarios": list(EXPECTED_SCENARIOS),
        "totals": totals,
        "scenarios": scenario_reports,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty audit output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "audit_by_scenario.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "scenario", "directory_present", "run_count", "samples", "episodes",
            "straight_episodes", "semi_complex_episodes", "complex_episodes",
            "v_complex_episodes", "unknown_episodes", "missing_lom_references",
            "missing_npz_runs", "failed_paths_file_present", "failed_paths_reported",
            "failed_path_attempts_summed", "failed_collision_paths", "error_count",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in report["scenarios"]:
            ep = item.get("episodes_by_class", {})
            writer.writerow({
                "scenario": item["scenario"],
                "directory_present": item.get("directory_present", False),
                "run_count": item.get("run_count", 0),
                "samples": item.get("samples", 0),
                "episodes": item.get("episodes", 0),
                "straight_episodes": ep.get("straight", 0),
                "semi_complex_episodes": ep.get("semi_complex", 0),
                "complex_episodes": ep.get("complex", 0),
                "v_complex_episodes": ep.get("v_complex", 0),
                "unknown_episodes": ep.get("unknown", 0),
                "missing_lom_references": item.get("missing_lom_references", 0),
                "missing_npz_runs": item.get("missing_npz_runs", 0),
                "failed_paths_file_present": item.get("failed_paths", {}).get("file_present", False),
                "failed_paths_reported": item.get("failed_paths", {}).get("entries", 0),
                "failed_path_attempts_summed": item.get("failed_paths", {}).get("attempts_summed", 0),
                "failed_collision_paths": item.get("failed_paths", {}).get("by_status", {}).get("collision", 0),
                "error_count": item.get("error_count", len(item.get("errors", []))),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2] / "dataset" / "data"
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional separate output folder; must not already contain files.",
    )
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error(f"data root does not exist: {args.data_root}")

    report = audit_dataset(args.data_root)
    print(json.dumps({"data_root": report["data_root"], "totals": report["totals"]}, indent=2))
    if args.output_dir:
        write_outputs(report, args.output_dir.expanduser().resolve())
        print(f"Audit files written to: {args.output_dir.resolve()}")
    return 0 if report["totals"]["scenario_directories_present"] == 30 else 2


if __name__ == "__main__":
    raise SystemExit(main())
