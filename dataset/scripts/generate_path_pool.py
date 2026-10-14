#!/usr/bin/env python3
"""Compatibility CLI for the production mapless path-pool generator.

All route-generation logic lives in path_generator.py. This thin wrapper keeps
--scenario/--num-paths/--output-dir compatible with batch collection scripts and
installs validated outputs as data/<scenario>/path_pool.*.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Installed in <repo>/src/Navigation/mapless_navigation/dataset/scripts.
# Keep --help usable when this file is inspected outside the repository.
REPO_ROOT = SCRIPT_DIR.parents[4] if len(SCRIPT_DIR.parents) > 4 else SCRIPT_DIR
DEFAULT_MAPS = REPO_ROOT / "src/Hardware/justina_description/webots_env/environments/maps"
DEFAULT_DATA = REPO_ROOT / "src/Navigation/mapless_navigation/dataset/data"
CLASSES = ("straight", "semi_complex", "complex", "v_complex")


def stable_seed(scenario: str) -> int:
    suffix = scenario.removeprefix("E")
    if suffix.isdigit():
        return 462326533 + int(suffix)
    return 1001 + sum((i + 1) * ord(c) for i, c in enumerate(scenario))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a validated production path pool using path_generator.py.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--num-paths", type=int, default=200,
                        help="approximate total; rounded up to equal quotas for four classes")
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA),
                        help="dataset root; output goes to <root>/<scenario>/")
    parser.add_argument("--maps-root", default=str(DEFAULT_MAPS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=25000)
    parser.add_argument("--timeout-seconds", type=float, default=900.0,
                        help="generation wall-time limit; 0 disables timeout")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.add_argument("--hard-clearance-m", type=float, default=0.35)
    parser.add_argument("--min-goal-distance-m", type=float, default=2.0)
    parser.add_argument("--max-goal-distance-m", type=float, default=3.0)
    parser.add_argument("--max-path-length-m", type=float, default=3.0)
    parser.add_argument("--v-max-goal-distance-m", type=float, default=3.6)
    parser.add_argument("--v-max-path-length-m", type=float, default=4.2)
    parser.add_argument("--max-length-ratio", type=float, default=1.18)
    parser.add_argument("--plan-cell-factor", type=int, default=5)
    args = parser.parse_args()

    if args.num_paths < 4 or args.max_attempts < 1 or args.timeout_seconds < 0:
        parser.error("--num-paths must be >= 4, --max-attempts positive, and timeout >= 0")
    if args.progress_interval_seconds <= 0:
        parser.error("--progress-interval-seconds must be positive")

    per_class = int(math.ceil(args.num_paths / 4.0))
    seed = args.seed if args.seed is not None else stable_seed(args.scenario)
    data_root = Path(args.output_dir).expanduser().resolve()
    maps_root = Path(args.maps_root).expanduser().resolve()
    generator = SCRIPT_DIR / "path_generator.py"
    if not generator.is_file():
        print(f"ERROR: production engine not found: {generator}", file=sys.stderr)
        return 2
    if not maps_root.is_dir():
        print(f"ERROR: maps root not found: {maps_root}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix=f"mapless_{args.scenario}_pool_") as temp_name:
        stage_root = Path(temp_name)
        run_name = f"production_seed_{seed}"
        command = [
            sys.executable, str(generator), "--scenario", args.scenario,
            "--maps-root", str(maps_root), "--output-dir", str(stage_root),
            "--run-name", run_name, "--per-class", str(per_class),
            "--max-attempts", str(args.max_attempts), "--seed", str(seed),
            "--hard-clearance-m", str(args.hard_clearance_m),
            "--min-goal-distance-m", str(args.min_goal_distance_m),
            "--max-goal-distance-m", str(args.max_goal_distance_m),
            "--max-path-length-m", str(args.max_path_length_m),
            "--v-max-goal-distance-m", str(args.v_max_goal_distance_m),
            "--v-max-path-length-m", str(args.v_max_path_length_m),
            "--max-length-ratio", str(args.max_length_ratio),
            "--plan-cell-factor", str(args.plan_cell_factor),
            "--timeout-seconds", str(args.timeout_seconds),
            "--progress-interval-seconds", str(args.progress_interval_seconds),
        ]
        print(f"Production engine: {generator}")
        print(f"Scenario: {args.scenario}; target={per_class * 4} routes ({per_class}/class); seed={seed}")
        print(f"Timeout: {args.timeout_seconds:g}s; progress interval: {args.progress_interval_seconds:g}s")
        try:
            proc = subprocess.run(command, check=False)
        except OSError as exc:
            print(f"ERROR launching production engine: {exc}", file=sys.stderr)
            return 2

        stage_run = stage_root / args.scenario / run_name
        pool_file = stage_run / "path_pool.json"
        poses_file = stage_run / "path_pool_poses.json"
        if not pool_file.is_file() or not poses_file.is_file():
            print("ERROR: engine produced no pool/poses pair; staged output was not installed.", file=sys.stderr)
            return proc.returncode or 2
        try:
            pool = json.loads(pool_file.read_text(encoding="utf-8"))
            poses = json.loads(poses_file.read_text(encoding="utf-8"))
            paths = pool.get("paths", [])
            counts = Counter(p.get("class") for p in paths)
            ids = {str(p.get("id")) for p in paths}
            if pool.get("pool_version") != 5 or not paths or not ids.issubset(poses.keys()):
                raise ValueError("invalid pool schema, empty paths, or missing pose arrays")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"ERROR validating staged output: {exc}", file=sys.stderr)
            return 2

        dest = data_root / args.scenario
        dest.mkdir(parents=True, exist_ok=True)
        for item in stage_run.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)

        complete = all(counts.get(name, 0) >= per_class for name in CLASSES)
        print(f"Installed production pool: {dest / 'path_pool.json'}")
        print("Class counts: " + ", ".join(f"{name}={counts.get(name, 0)}/{per_class}" for name in CLASSES))
        print(f"Accepted routes: {len(paths)}; quota_complete={complete}; engine_exit={proc.returncode}")
        if not complete:
            print("WARNING: valid partial pool retained; target quota was not reached.", file=sys.stderr)
        return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
