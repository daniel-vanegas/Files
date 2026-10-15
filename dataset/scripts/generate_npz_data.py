#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import numpy as np
import yaml
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))


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


def _convert_run(run_dir):
    print(f"  reading: {run_dir}")

    samples = _load_run(run_dir)

    if samples is None or len(samples) == 0:
        print("  no samples", file=sys.stderr)
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


def _find_runs(root):
    """
    Find runs under:

        data/
          E01/
            run_x/
          E02/
            run_y/

    It also accepts:

        data/E01/run_x
        data/run_x
    """
    if _has_data_csv(root):
        yield root
        return

    # Direct run_* children.
    for child in _iter_child_dirs(root, prefix="run_"):
        if _has_data_csv(child):
            yield child

    # Scenario/run hierarchy.
    for scenario in _iter_child_dirs(root):
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

    args = parser.parse_args()
    target = _resolve_path(args.path)

    if not os.path.isdir(target):
        print(f"not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    runs = list(_find_runs(target))

    if not runs:
        print(f"no run_* folders with data.csv under {target}", file=sys.stderr)
        sys.exit(1)

    print(f"data root: {_DEFAULT_DATA_ROOT}")
    print(f"target:    {target}")
    print(f"runs:      {len(runs)}")
    print()

    ok = 0

    for run in runs:
        try:
            rel = os.path.relpath(run, _DEFAULT_DATA_ROOT)
        except ValueError:
            rel = run

        print(f"[{rel}]")

        try:
            if _convert_run(run):
                ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    print()
    print(f"done: {ok}/{len(runs)}")


if __name__ == "__main__":
    main()