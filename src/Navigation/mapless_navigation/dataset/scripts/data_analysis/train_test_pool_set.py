#!/usr/bin/env python3
"""Build fixed task sets from existing path pools.

Reads the pool at data/<scenario>/default_pool/path_pool.json
(falls back to data/<scenario>/path_pool.json if default_pool is absent)
and writes deterministic task lists to data/pool_sets/.

Selection rule:
    Order paths of each class by metrics.path_length_m.
    For n_per_class=1: pick the median (quantile 50%).
    For n_per_class=2: pick quantiles 33% and 67%.

Outputs:
    data/pool_sets/E08_filter_tasks.json     (1 goal per class)
    data/pool_sets/E08_final_tasks.json      (2 goals per class)
    data/pool_sets/E16_final_tasks.json
    data/pool_sets/E20_final_tasks.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CLASSES = ("straight", "semi_complex", "complex", "v_complex")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_pool(scenario_dir):
    """Return the pool path. Prefer default_pool (raw generator output);
    fall back to the flat copy installed by collect_dataset_batch_classic_nav.sh."""
    candidates = [
        scenario_dir / "default_pool" / "path_pool.json",
        scenario_dir / "path_pool.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def pick_quantile(paths, q):
    """Pick a single path at the given quantile by path_length_m."""
    ordered = sorted(paths, key=lambda p: float(p["metrics"]["path_length_m"]))
    idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


def build_tasks_for_class(paths, class_name, n_per_class):
    class_paths = [p for p in paths if p.get("class") == class_name]
    if len(class_paths) < n_per_class:
        raise ValueError(
            f"Not enough {class_name} paths: need {n_per_class}, have {len(class_paths)}"
        )
    if n_per_class == 1:
        chosen = [pick_quantile(class_paths, 0.50)]
    elif n_per_class == 2:
        chosen = [pick_quantile(class_paths, 0.33), pick_quantile(class_paths, 0.67)]
    else:
        raise ValueError(f"Unsupported n_per_class: {n_per_class}")

    tasks = []
    for path in chosen:
        start = [float(path["start"][0]), float(path["start"][1])]
        goal = [float(path["goal"][0]), float(path["goal"][1])]
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        tasks.append({
            "id": int(path["id"]),
            "route_class": class_name,
            "start": start,
            "goal": goal,
            "start_yaw_rad": float(yaw),
            "generator_start_yaw_rad": float(path.get("start_yaw_rad", 0.0)),
            "task_type": str(path.get("task_type", "unknown")),
            "direct_line_safe": bool(path.get("direct_line_safe", False)),
            "path_length_m": float(path["metrics"]["path_length_m"]),
            "min_clearance_m": float(path["metrics"].get("min_clearance_m", 0.0)),
        })
    return tasks


def build_task_set(scenario, pool_path, n_per_class, seed):
    pool = json.loads(pool_path.read_text())
    if pool.get("pool_version") != 5:
        raise ValueError(f"{pool_path}: unsupported pool_version {pool.get('pool_version')}")
    paths = pool.get("paths", [])
    if not paths:
        raise ValueError(f"{pool_path}: pool contains no paths")

    tasks = []
    for class_name in CLASSES:
        tasks.extend(build_tasks_for_class(paths, class_name, n_per_class))

    return {
        "name": scenario + "_" + ("filter" if n_per_class == 1 else "final"),
        "scenario": scenario,
        "seed": seed,
        "selection_rule": (
            "quantile_50_per_class_by_path_length" if n_per_class == 1
            else "quantiles_33_67_per_class_by_path_length"
        ),
        "source_pool": str(pool_path),
        "source_pool_sha256": sha256_file(pool_path),
        "n_per_class": n_per_class,
        "n_tasks": len(tasks),
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "pool_sets",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--force", action="store_true",
        help="Rewrite task sets even if the source pool hash matches",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(f"data-root not found: {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"data-root:  {args.data_root}")
    print(f"output-dir: {args.output_dir}")

    plans = [
        ("E08", "filter", 1),
        ("E08", "final", 2),
        ("E16", "final", 2),
        ("E20", "final", 2),
    ]

    for scenario, level, n_per_class in plans:
        scenario_dir = args.data_root / scenario
        if not scenario_dir.is_dir():
            print(f"[skip] scenario dir missing: {scenario_dir}")
            continue

        pool_path = find_pool(scenario_dir)
        if pool_path is None:
            print(f"[skip] no pool found in {scenario_dir}")
            continue
        print(f"[source] {pool_path}")

        out_path = args.output_dir / f"{scenario}_{level}_tasks.json"
        pool_sha = sha256_file(pool_path)

        if out_path.exists() and not args.force:
            try:
                existing = json.loads(out_path.read_text())
                if (existing.get("source_pool_sha256") == pool_sha
                        and existing.get("n_per_class") == n_per_class):
                    print(f"[keep]   {out_path.name} (pool hash matches)")
                    continue
            except Exception:
                pass

        task_set = build_task_set(scenario, pool_path, n_per_class, args.seed)
        out_path.write_text(json.dumps(task_set, indent=2))
        print(f"[write]  {out_path.name} ({task_set['n_tasks']} tasks)")

    print("done")


if __name__ == "__main__":
    main()