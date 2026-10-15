#!/usr/bin/env python3
"""CLI wrapper that installs a validated path pool into data/<scenario>/."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, tempfile
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

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a validated mapless path pool using path_generator.py.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--n-goals", "--num-paths", dest="n_goals", type=int, default=None)
    parser.add_argument("--st_sem_cmp_vc", "--st-sem-cmp-vc", dest="st_sem_cmp_vc", default=None)
    parser.add_argument("--balance-class", action="store_true")
    parser.add_argument("--per-class", type=int, default=None)
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
