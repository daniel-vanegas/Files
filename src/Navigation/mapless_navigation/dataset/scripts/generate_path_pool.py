#!/usr/bin/env python3
"""Generate validated path pools and incremental collection state."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4] if len(SCRIPT_DIR.parents) > 4 else SCRIPT_DIR
DEFAULT_MAPS = REPO_ROOT / "src/Hardware/justina_description/webots_env/environments/maps"
DEFAULT_DATA = REPO_ROOT / "src/Navigation/mapless_navigation/dataset/data"
CLASSES = ("straight", "semi_complex", "complex", "v_complex")


def stable_seed(scenario: str) -> int:
    suffix = scenario.removeprefix("E")
    if suffix.isdigit(): return 462326533 + int(suffix)
    return 1001 + sum((i + 1) * ord(c) for i, c in enumerate(scenario))


def resolve_targets(parser, args):
    if args.st_sem_cmp_vc is not None:
        if args.n_goals is not None or args.per_class is not None or args.balance_class:
            parser.error("--st_sem_cmp_vc cannot be combined with --n-goals, --per-class, or --balance-class")
        parts = args.st_sem_cmp_vc.split("_")
        if len(parts) != 4: parser.error("--st_sem_cmp_vc requires four quotas, e.g. 10_10_15_20")
        try: values = [int(part) for part in parts]
        except ValueError: parser.error("--st_sem_cmp_vc quotas must be integers")
        if any(value < 1 for value in values): parser.error("--st_sem_cmp_vc quotas must be positive")
        return dict(zip(CLASSES, values))
    if args.per_class is not None:
        if args.n_goals is not None or args.balance_class: parser.error("--per-class cannot be combined with --n-goals or --balance-class")
        if args.per_class < 1: parser.error("--per-class must be positive")
        return {name: args.per_class for name in CLASSES}
    total = 200 if args.n_goals is None else args.n_goals
    if total < 4: parser.error("--n-goals must be at least 4")
    base, remainder = divmod(total, 4)
    return {name: int(base + (i < remainder)) for i, name in enumerate(CLASSES)}


def _anchors_from_csv(csv_path):
    anchors = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                eid = int(float(row["episode_id"]))
            except (KeyError, ValueError):
                continue
            if eid in anchors:
                continue
            try:
                anchors[eid] = (
                    (float(row["pose_x"]), float(row["pose_y"])),
                    (float(row["goal_x"]), float(row["goal_y"])),
                )
            except (KeyError, ValueError):
                continue
    return anchors


def _match_committed_ids(scenario_dir, pool_entries, tol):
    committed = set()
    if not pool_entries:
        return committed
    for run_dir in sorted(scenario_dir.glob("run_*_classic_nav")):
        csv_path = run_dir / "data.csv"
        if not csv_path.exists():
            continue
        anchors = _anchors_from_csv(csv_path)
        for pose, goal in anchors.values():
            best_id = None
            best_score = float("inf")
            for p in pool_entries:
                score = math.dist(pose, p["start"]) + math.dist(goal, p["goal"])
                if score < best_score:
                    best_score = score
                    best_id = p["id"]
            if best_id is not None and best_score <= tol:
                committed.add(best_id)
    return committed


def _build_filtered_quota(live_paths, want):
    counts = Counter(p["class"] for p in live_paths)
    target = min(want, len(live_paths))
    base, rem = divmod(target, len(CLASSES))
    quota = {}
    shortfall = 0
    for i, c in enumerate(CLASSES):
        w = base + (1 if i < rem else 0)
        a = counts.get(c, 0)
        if a < w:
            quota[c] = a
            shortfall += w - a
        else:
            quota[c] = w
    if shortfall > 0:
        for c in CLASSES:
            spare = counts.get(c, 0) - quota[c]
            if spare <= 0: continue
            add = min(spare, shortfall)
            quota[c] += add
            shortfall -= add
            if shortfall == 0: break
    return quota


def _stage_generation(scenario, need, args):
    seed = args.seed if args.seed is not None else random.randint(100000, 999999)
    generator = SCRIPT_DIR / "path_generator.py"
    maps_root = Path(args.maps_root).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix=f"mapless_{scenario}_inc_") as tmp:
        stage_root = Path(tmp)
        run_name = f"incremental_seed_{seed}"
        command = [sys.executable, str(generator),
                   "--scenario", scenario, "--maps-root", str(maps_root),
                   "--output-dir", str(stage_root), "--run-name", run_name,
                   "--seed", str(seed),
                   "--hard-clearance-m", str(args.hard_clearance_m),
                   "--min-goal-distance-m", str(args.min_goal_distance_m),
                   "--max-goal-distance-m", str(args.max_goal_distance_m),
                   "--max-path-length-m", str(args.max_path_length_m),
                   "--v-max-goal-distance-m", str(args.v_max_goal_distance_m),
                   "--v-max-path-length-m", str(args.v_max_path_length_m),
                   "--max-length-ratio", str(args.max_length_ratio),
                   "--plan-cell-factor", str(args.plan_cell_factor),
                   "--progress-interval-seconds", str(args.progress_interval_seconds),
                   "--n-goals", str(need), "--balance-class"]
        if args.max_attempts is not None: command.extend(["--max-attempts", str(args.max_attempts)])
        if args.timeout_seconds is not None: command.extend(["--timeout-seconds", str(args.timeout_seconds)])
        process = subprocess.run(command, check=False)
        stage_run = stage_root / scenario / run_name
        pool_file = stage_run / "path_pool.json"
        poses_file = stage_run / "path_pool_poses.json"
        if not pool_file.is_file() or not poses_file.is_file():
            return None, None
        return json.loads(pool_file.read_text()), json.loads(poses_file.read_text())


def run_incremental(args) -> int:
    scenario_dir = Path(args.output_dir).expanduser().resolve() / args.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    history_file = scenario_dir / "path_pool_history.json"
    history_poses_file = scenario_dir / "path_pool_history_poses.json"
    active_pool = scenario_dir / "path_pool.json"
    active_poses = scenario_dir / "path_pool_poses.json"

    if history_file.exists():
        history = json.loads(history_file.read_text())
        history_poses = json.loads(history_poses_file.read_text()) if history_poses_file.exists() else {}
    elif active_pool.exists():
        history = json.loads(active_pool.read_text())
        history_poses = json.loads(active_poses.read_text()) if active_poses.exists() else {}
        print(f"initialized history from existing pool ({len(history.get('paths', []))} paths)")
    else:
        history = {"pool_version": 5, "scenario": args.scenario,
                   "classification_scope": "navigation_task_and_geometry",
                   "paths": [], "num_paths": 0}
        history_poses = {}

    entries = history.get("paths", [])
    pool_ids = {p["id"] for p in entries}

    committed_ids = _match_committed_ids(scenario_dir, entries, args.match_tol) & pool_ids

    failed_ids = set()
    failed_path = scenario_dir / "failed_paths.json"
    if failed_path.exists():
        try:
            for e in json.loads(failed_path.read_text()).get("entries", []):
                pid = e.get("path_id")
                if pid in pool_ids:
                    failed_ids.add(pid)
        except Exception:
            pass
    failed_ids -= committed_ids

    live_ids = pool_ids - committed_ids - failed_ids
    gap = args.target

    print(f"state: committed={len(committed_ids)} failed={len(failed_ids)} "
          f"live={len(live_ids)} add={gap} total_after={len(committed_ids) + gap}")

    if gap <= 0:
        filtered = dict(history)
        filtered["paths"] = []
        filtered["num_paths"] = 0
        filtered["quota"] = {c: 0 for c in CLASSES}
        active_pool.write_text(json.dumps(filtered, indent=2))
        active_poses.write_text("{}")
        history_file.write_text(json.dumps(history, indent=2))
        history_poses_file.write_text(json.dumps(history_poses, indent=1))
        state_file = scenario_dir / "incremental_state.json"
        state_file.write_text(json.dumps({
            "committed_before": len(committed_ids),
            "add": 0,
            "total_after": len(committed_ids),
            "target_add": args.target,
        }, indent=2))
        print("paths_to_record=0")
        return 0

    if len(live_ids) < gap:
        need = gap - len(live_ids)
        print(f"generating {need} new paths (live={len(live_ids)} gap={gap})")
        new_pool, new_poses = _stage_generation(args.scenario, need, args)
        if new_pool is not None:
            max_id = max(pool_ids) if pool_ids else -1
            for p in new_pool.get("paths", []):
                old_id = p["id"]
                max_id += 1
                p["id"] = max_id
                history["paths"].append(p)
                if str(old_id) in new_poses:
                    history_poses[str(max_id)] = new_poses[str(old_id)]
            history["num_paths"] = len(history["paths"])
            pool_ids = {p["id"] for p in history["paths"]}
            live_ids = pool_ids - committed_ids - failed_ids
            print(f"merged {len(new_pool.get('paths', []))} new paths, live={len(live_ids)}")
        else:
            print("generation failed, using existing live paths")

    want = min(gap, len(live_ids))
    if want <= 0:
        filtered = dict(history)
        filtered["paths"] = []
        filtered["num_paths"] = 0
        filtered["quota"] = {c: 0 for c in CLASSES}
        active_pool.write_text(json.dumps(filtered, indent=2))
        active_poses.write_text("{}")
        history_file.write_text(json.dumps(history, indent=2))
        history_poses_file.write_text(json.dumps(history_poses, indent=1))
        print("paths_to_record=0")
        return 0

    live_paths_all = [p for p in history["paths"] if p["id"] in live_ids]
    random.shuffle(live_paths_all)
    quota = _build_filtered_quota(live_paths_all, want)
    selected_paths = []
    class_budget = dict(quota)
    for p in live_paths_all:
        c = p["class"]
        if class_budget.get(c, 0) <= 0:
            continue
        selected_paths.append(p)
        class_budget[c] -= 1
        if len(selected_paths) >= sum(quota.values()):
            break

    selected_poses = {str(p["id"]): history_poses[str(p["id"])]
                      for p in selected_paths if str(p["id"]) in history_poses}
    filtered = dict(history)
    filtered["paths"] = selected_paths
    filtered["num_paths"] = len(selected_paths)
    filtered["quota"] = quota

    active_pool.write_text(json.dumps(filtered, indent=2))
    active_poses.write_text(json.dumps(selected_poses, indent=1))
    history_file.write_text(json.dumps(history, indent=2))
    history_poses_file.write_text(json.dumps(history_poses, indent=1))

    state_file = scenario_dir / "incremental_state.json"
    state_file.write_text(json.dumps({
        "committed_before": len(committed_ids),
        "add": want,
        "total_after": len(committed_ids) + want,
        "target_add": args.target,
    }, indent=2))

    print(f"filtered pool: {len(selected_paths)} paths to record "
          f"(quota={quota})")
    print(f"paths_to_record={sum(quota.values())}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a validated mapless path pool using path_generator.py.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--n-goals", "--num-paths", dest="n_goals", type=int, default=None)
    parser.add_argument("--st_sem_cmp_vc", "--st-sem-cmp-vc", dest="st_sem_cmp_vc", default=None)
    parser.add_argument("--balance-class", action="store_true")
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--incremental", action="store_true",
                        help="Incremental mode: compute state, generate missing, filter pool.")
    parser.add_argument("--target", type=int, default=None,
                        help="Total committed goals target (required with --incremental)")
    parser.add_argument("--match-tol", type=float, default=0.30,
                        help="Matching tolerance for committed detection (incremental)")
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--maps-root", default=str(DEFAULT_MAPS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
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

    if args.incremental:
        if args.target is None or args.target < 1:
            parser.error("--incremental requires --target N (N >= 1)")
        return run_incremental(args)

    targets = resolve_targets(parser, args)
    if args.max_attempts is not None and args.max_attempts < 1: parser.error("--max-attempts must be positive")
    if args.timeout_seconds is not None and args.timeout_seconds < 0: parser.error("--timeout-seconds must be >= 0")
    if args.progress_interval_seconds <= 0: parser.error("--progress-interval-seconds must be positive")
    if args.hard_clearance_m <= 0 or args.max_length_ratio < 1: parser.error("hard clearance must be positive and length ratio >= 1")
    if not (0 < args.min_goal_distance_m <= args.max_goal_distance_m <= args.v_max_goal_distance_m): parser.error("inconsistent goal distance limits")
    if not 1 <= args.plan_cell_factor <= 20: parser.error("--plan-cell-factor must be between 1 and 20")
    seed = args.seed if args.seed is not None else stable_seed(args.scenario)
    output_root = Path(args.output_dir).expanduser().resolve()
    maps_root = Path(args.maps_root).expanduser().resolve()
    generator = SCRIPT_DIR / "path_generator.py"
    if not generator.is_file(): print(f"ERROR: path generator not found: {generator}", file=sys.stderr); return 2
    if not maps_root.is_dir(): print(f"ERROR: maps root not found: {maps_root}", file=sys.stderr); return 2
    with tempfile.TemporaryDirectory(prefix=f"mapless_{args.scenario}_pool_") as tmp:
        stage_root = Path(tmp); run_name = f"generation_seed_{seed}"
        command = [sys.executable, str(generator), "--scenario", args.scenario, "--maps-root", str(maps_root),
                   "--output-dir", str(stage_root), "--run-name", run_name, "--seed", str(seed),
                   "--hard-clearance-m", str(args.hard_clearance_m),
                   "--min-goal-distance-m", str(args.min_goal_distance_m),
                   "--max-goal-distance-m", str(args.max_goal_distance_m),
                   "--max-path-length-m", str(args.max_path_length_m),
                   "--v-max-goal-distance-m", str(args.v_max_goal_distance_m),
                   "--v-max-path-length-m", str(args.v_max_path_length_m),
                   "--max-length-ratio", str(args.max_length_ratio),
                   "--plan-cell-factor", str(args.plan_cell_factor),
                   "--progress-interval-seconds", str(args.progress_interval_seconds)]
        if args.st_sem_cmp_vc is not None: command.extend(["--st_sem_cmp_vc", args.st_sem_cmp_vc])
        elif args.per_class is not None: command.extend(["--per-class", str(args.per_class)])
        else:
            command.extend(["--n-goals", str(sum(targets.values()))])
            if args.balance_class: command.append("--balance-class")
        if args.max_attempts is not None: command.extend(["--max-attempts", str(args.max_attempts)])
        if args.timeout_seconds is not None: command.extend(["--timeout-seconds", str(args.timeout_seconds)])
        print(f"Generator: {generator}")
        print(f"Scenario: {args.scenario}; targets: " + ", ".join(f"{name}={targets[name]}" for name in CLASSES) + f"; total={sum(targets.values())}; seed={seed}")
        print("Limits: " + (f"max_attempts={args.max_attempts}" if args.max_attempts is not None else "max_attempts=unlimited") + ", " + (f"timeout_seconds={args.timeout_seconds:g}" if args.timeout_seconds is not None and args.timeout_seconds > 0 else "timeout=disabled"))
        try: process = subprocess.run(command, check=False)
        except OSError as exc: print(f"ERROR launching path generator: {exc}", file=sys.stderr); return 2
        stage_run = stage_root / args.scenario / run_name
        pool_file, poses_file = stage_run / "path_pool.json", stage_run / "path_pool_poses.json"
        if not pool_file.is_file() or not poses_file.is_file():
            print("ERROR: generator produced no pool/poses pair; staged output was not installed.", file=sys.stderr); return process.returncode or 2
        try:
            pool = json.loads(pool_file.read_text(encoding="utf-8")); poses = json.loads(poses_file.read_text(encoding="utf-8"))
            paths = pool.get("paths", []); counts = Counter(path.get("class") for path in paths)
            ids = {str(path.get("id")) for path in paths}
            if pool.get("pool_version") != 5 or not ids.issubset(poses.keys()): raise ValueError("invalid pool schema or missing pose arrays")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"ERROR validating staged output: {exc}", file=sys.stderr); return 2
        destination = output_root / args.scenario / run_name
        destination.mkdir(parents=True, exist_ok=True)
        for item in stage_run.iterdir():
            if item.is_file(): shutil.copy2(item, destination / item.name)
        actual_targets = pool.get("quota", targets)
        complete = all(counts.get(name, 0) >= actual_targets.get(name, targets[name]) for name in CLASSES)
        print(f"Installed path pool: {destination / 'path_pool.json'}")
        print("Class counts: " + ", ".join(f"{name}={counts.get(name,0)}/{actual_targets.get(name,targets[name])}" for name in CLASSES))
        print(f"Accepted routes: {len(paths)}; quota_complete={complete}; generator_exit={process.returncode}")
        if not complete:
            deficits = {name:max(0,actual_targets.get(name,targets[name])-counts.get(name,0)) for name in CLASSES}
            print(f"WARNING: valid partial pool retained; class deficits: {deficits}", file=sys.stderr)
        return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())