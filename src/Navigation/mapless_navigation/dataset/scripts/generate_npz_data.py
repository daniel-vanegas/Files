#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import yaml
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))
_DEFAULT_PLOT_OUT = os.path.join(_DEFAULT_DATA_ROOT, "npz_balance.png")
CLASS_NAMES = ("straight", "semi_complex", "complex", "v_complex")

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    import data_analysis
except ImportError:
    try:
        import data_analiysis as data_analysis
    except Exception as exc:
        print(f"DEBUG: Could not import data_analysis because: {exc}", file=sys.stderr)
        data_analysis = None

def _load_run(run_dir):
    csv_path = os.path.join(run_dir, "data.csv")

    if not os.path.isfile(csv_path):
        print(f"  missing: {csv_path}", file=sys.stderr)
        return None

    samples = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "episode_id", "is_episode_start", "d", "theta", "vx", "vy", "wz",
            "pose_x", "pose_y", "pose_yaw", "goal_x", "goal_y", "timestamp", "lom_file"
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing CSV columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            lom_path = os.path.join(run_dir, "lom", row["lom_file"])

            if not os.path.isfile(lom_path):
                print(f"  missing lom: {lom_path}", file=sys.stderr)
                continue

            try:
                lom = np.array(Image.open(lom_path).convert("L"), dtype=np.uint8)

                sample = {
                    "features": {
                        "occ_grid": lom,
                        "target": np.array([float(row["d"]), float(row["theta"])], dtype=np.float32),
                    },
                    "labels": np.array([float(row["vx"]), float(row["vy"]), float(row["wz"])], dtype=np.float32),
                    "meta": {
                        "pose": np.array([float(row["pose_x"]), float(row["pose_y"]), float(row["pose_yaw"])], dtype=np.float32),
                        "goal_xy": np.array([float(row["goal_x"]), float(row["goal_y"])], dtype=np.float32),
                        "timestamp": float(row["timestamp"]),
                        "episode_id": int(row["episode_id"]),
                        "is_episode_start": int(row["is_episode_start"]),
                        "collision_flag": int(row.get("collision_flag", 0)),
                        "stuck_flag": int(row.get("stuck_flag", 0)),
                    },
                }

            except (ValueError, OSError) as exc:
                print(f"  invalid sample at row {row_idx}: {exc}", file=sys.stderr)
                continue

            samples.append(sample)

    return samples


def _load_episode_classes(run_dir):
    """Read episode -> class from episode_tries.json, or {} if missing."""
    tries_path = os.path.join(run_dir, "episode_tries.json")
    if not os.path.isfile(tries_path):
        return {}
    try:
        with open(tries_path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    classes = {}
    for entry in data.get("entries", []):
        eid = int(entry.get("episode_id", -1))
        if eid < 0:
            continue
        classes[eid] = str(entry.get("class", "unknown"))
    return classes


def _parse_st_sem_cmp_vc(text):
    if text is None:
        return None
    parts = text.split("_")
    if len(parts) != 4:
        raise ValueError(
            "--st-sem-cmp-vc requires four quotas: straight_semi_complex_complex_v_complex"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--st-sem-cmp-vc quotas must be integers") from exc
    if any(v < 0 for v in values):
        raise ValueError("--st-sem-cmp-vc quotas must be non-negative")
    return dict(zip(CLASS_NAMES, values))


def _select_episodes(episode_classes, goals_per_scene, class_quotas, shuffle, seed):
    """
    Return a set of episode_ids to keep.
    Priority: class_quotas > goals_per_scene > all.
    """
    all_ids = sorted(episode_classes.keys())

    if class_quotas is not None:
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for eid, cls in episode_classes.items():
            by_class[cls].append(eid)
        selected = set()
        for cls in CLASS_NAMES:
            ids = sorted(by_class.get(cls, []))
            if shuffle:
                rng.shuffle(ids)
            quota = class_quotas.get(cls, 0)
            selected.update(ids[:quota])
        return selected

    if goals_per_scene is not None:
        ids = list(all_ids)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(ids)
        return set(ids[:max(0, goals_per_scene)])

    return set(all_ids)


def _convert_run(run_dir, filter_cfg=None):
    if filter_cfg is None:
        filter_cfg = {"active": False}

    print(f"  reading: {run_dir}")

    samples = _load_run(run_dir)

    if samples is None or len(samples) == 0:
        print("  no samples", file=sys.stderr)
        return None

    if filter_cfg["active"]:
        episode_classes = _load_episode_classes(run_dir)
        if not episode_classes:
            if filter_cfg.get("class_quotas") is not None:
                print("  ERROR: class filter requested but episode_tries.json "
                      "has no entries; re-record or drop --st-sem-cmp-vc",
                      file=sys.stderr)
                return None
            print("  no episode_tries.json; using all episodes", file=sys.stderr)
        else:
            keep = _select_episodes(
                episode_classes,
                filter_cfg.get("goals_per_scene"),
                filter_cfg.get("class_quotas"),
                bool(filter_cfg.get("shuffle")),
                int(filter_cfg.get("seed", 42)),
            )
            if not keep:
                if filter_cfg.get("class_quotas") is not None:
                    recognized = sum(1 for c in episode_classes.values()
                                     if c in CLASS_NAMES)
                    if recognized == 0:
                        print("  ERROR: no episodes with known class; "
                              "this run predates class logging",
                              file=sys.stderr)
                        return None
                print("  filter produced empty set; keeping all", file=sys.stderr)
            else:
                before = len(samples)
                samples = [s for s in samples if s["meta"]["episode_id"] in keep]
                print(f"  filter: kept {len(samples)}/{before} samples "
                      f"({len(keep)} episodes selected)")

    if not samples:
        print("  no samples after filter", file=sys.stderr)
        return None

    data = np.array(samples, dtype=object)
    out_path = run_dir.rstrip("/") + ".npz"

    np.savez_compressed(out_path, data=data)
    print(f"  wrote {out_path} ({len(samples)} samples)")

    desc_src = os.path.join(run_dir, "descriptor.yaml")
    desc_dst = run_dir.rstrip("/") + ".descriptor.yaml"

    if os.path.isfile(desc_src):
        with open(desc_src, "r") as f:
            desc = yaml.safe_load(f) or {}

        desc["num_samples"] = len(samples)
        if filter_cfg["active"]:
            desc["filter_applied"] = True
            desc["filter_goals_per_scene"] = filter_cfg.get("goals_per_scene")
            desc["filter_class_quotas"] = filter_cfg.get("class_quotas")
            desc["filter_shuffle"] = filter_cfg.get("shuffle")
            desc["filter_seed"] = filter_cfg.get("seed")

        with open(desc_dst, "w") as f:
            yaml.safe_dump(desc, f, sort_keys=False)

    return out_path


def _has_data_csv(path):
    return os.path.isfile(os.path.join(path, "data.csv"))


def _iter_child_dirs(path, prefix=None):
    for name in sorted(os.listdir(path)):
        child = os.path.join(path, name)

        if not os.path.isdir(child):
            continue

        if prefix is not None and not name.startswith(prefix):
            continue

        yield child


def _resolve_path(arg):
    """
    Resolve the optional input path.

    No argument:
        <package>/dataset/data

    Relative path:
        interpreted relative to dataset/data

    Absolute path:
        accepted explicitly and used as-is.
    """
    if arg is None:
        return _DEFAULT_DATA_ROOT

    if os.path.isabs(arg):
        return os.path.normpath(arg)

    return os.path.normpath(os.path.join(_DEFAULT_DATA_ROOT, arg))


def _find_runs(root, scenario_filter=None):
    """
    Find runs under data/. If scenario_filter is set, only those scenarios
    are traversed. scenario_filter is a set of scenario names like {"E01","E07"}.
    """
    if _has_data_csv(root):
        yield root
        return

    for child in _iter_child_dirs(root, prefix="run_"):
        if _has_data_csv(child):
            yield child

    for scenario in _iter_child_dirs(root):
        if scenario_filter is not None and os.path.basename(scenario) not in scenario_filter:
            continue
        for run in _iter_child_dirs(scenario, prefix="run_"):
            if _has_data_csv(run):
                yield run


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Convert run_* folders to .npz files.\n\n"
            "Path handling:\n"
            "  no argument -> use <package>/dataset/data/\n"
            "  relative path -> resolved under dataset/data/\n"
            "  absolute path -> used as-is\n\n"
            "The path can point to:\n"
            "  a single run_* folder\n"
            "  a scenario folder\n"
            "  the complete data root"
        ),
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="subfolder of data/ or absolute path",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None, metavar="E##",
        help="Only process these scenarios (e.g. --scenarios E01 E07)",
    )
    parser.add_argument(
        "--goals-per-scene", type=int, default=None, metavar="N",
        help="Max goals (episodes) per scenario. Fewer available -> take all. "
             "Default: keep all episodes.",
    )
    parser.add_argument(
        "--st-sem-cmp-vc", type=str, default=None, metavar="ST_SEMI_CMP_VC",
        help="Per-class quotas per scenario, e.g. 10_20_50_50. "
             "Overrides --goals-per-scene.",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle episodes before taking the first N.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for --shuffle. Default: 42",
    )
    parser.add_argument(
        "--plot-out", default=_DEFAULT_PLOT_OUT, metavar="PATH",
        help=f"Save class balance stacked-bar chart to this path (default: {_DEFAULT_PLOT_OUT})",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip class balance plot generation",
    )

    args = parser.parse_args()
    target = _resolve_path(args.path)

    if not os.path.isdir(target):
        print(f"not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    try:
        class_quotas = _parse_st_sem_cmp_vc(args.st_sem_cmp_vc)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if class_quotas is not None and args.goals_per_scene is not None:
        print("ERROR: --st-sem-cmp-vc and --goals-per-scene are mutually exclusive",
              file=sys.stderr)
        sys.exit(2)

    scenario_filter = set(args.scenarios) if args.scenarios else None

    filter_active = (
        class_quotas is not None
        or args.goals_per_scene is not None
    )
    filter_cfg = {
        "active": filter_active,
        "goals_per_scene": args.goals_per_scene,
        "class_quotas": class_quotas,
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
    }

    runs = list(_find_runs(target, scenario_filter=scenario_filter))

    if not runs:
        print(f"no run_* folders with data.csv under {target}", file=sys.stderr)
        sys.exit(1)

    print(f"data root: {_DEFAULT_DATA_ROOT}")
    print(f"target:    {target}")
    print(f"runs:      {len(runs)}")
    if scenario_filter is not None:
        print(f"scenarios: {sorted(scenario_filter)}")
    if filter_active:
        print(f"filter:    goals_per_scene={args.goals_per_scene} "
              f"class_quotas={class_quotas} shuffle={args.shuffle} seed={args.seed}")
    print()

    ok = 0

    for run in runs:
        try:
            rel = os.path.relpath(run, _DEFAULT_DATA_ROOT)
        except ValueError:
            rel = run

        print(f"[{rel}]")

        try:
            if _convert_run(run, filter_cfg):
                ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    print()
    print(f"done: {ok}/{len(runs)}")

    # Generate class balance plot by default unless --no-plot is specified
    if not args.no_plot and args.plot_out:
        if data_analysis is None:
            print("ERROR: data_analysis module could not be imported; skipping plot generation.", file=sys.stderr)
        else:
            try:
                counts = data_analysis.collect_class_counts_by_scenario(
                    data_root=_DEFAULT_DATA_ROOT,
                    scenarios=args.scenarios,
                    source="auto"
                )
                if counts:
                    out_img = data_analysis.plot_class_balance(counts, args.plot_out)
                    if out_img:
                        print(f"class balance plot saved to: {out_img}")
                else:
                    print("WARN: no class counts found for plotting", file=sys.stderr)
            except Exception as exc:
                print(f"ERROR generating class balance plot: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()