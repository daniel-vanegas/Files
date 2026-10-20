#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))


def resolve_path(arg):
    """
    Resolve an optional dataset path.

    No argument:
        <package>/dataset/data/

    Relative path:
        resolved relative to dataset/data/

    Absolute path:
        used as provided.
    """
    if arg is None:
        return _DEFAULT_DATA_ROOT

    if os.path.isabs(arg):
        return os.path.normpath(arg)

    return os.path.normpath(os.path.join(_DEFAULT_DATA_ROOT, arg))


def find_scenarios(data_root):
    """Find scenario directories such as E01, E02, ..."""
    scenarios = []

    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)

        if not os.path.isdir(path):
            continue

        if not name.startswith("E"):
            continue

        scenarios.append(name)

    return scenarios


def find_npz_files(scenario_dir):
    """Find converted run_*.npz files in one scenario."""
    files = []

    for name in sorted(os.listdir(scenario_dir)):
        if not name.startswith("run_"):
            continue

        if not name.endswith(".npz"):
            continue

        files.append(os.path.join(scenario_dir, name))

    return files


def load_run(npz_path):
    """Load one converted run."""
    with np.load(npz_path, allow_pickle=True) as archive:
        if "data" not in archive:
            raise RuntimeError(f"NPZ has no 'data': {npz_path}")

        return archive["data"]


def get_episode_ids(npz_files, data_root):
    """
    Return all complete episodes contained in the NPZ files.

    The run path is kept together with episode_id because
    episode_id can restart at 1 in another run.
    """
    episodes = []

    for npz_path in npz_files:
        data = load_run(npz_path)
        ids = sorted({int(sample["meta"]["episode_id"]) for sample in data})
        run_relpath = os.path.relpath(npz_path, data_root)

        for episode_id in ids:
            episodes.append({
                "run": run_relpath,
                "episode_id": episode_id,
            })

    return episodes


def split_scenarios(scenarios, train_spl, seed):
    """
    Split complete scenarios.

    Validation B is made from complete scenarios so that
    no frame from a Validation B scenario enters TRAIN.
    """
    if not 0.0 < train_spl < 1.0:
        raise ValueError("train_spl must be between 0 and 1.")

    if len(scenarios) < 2:
        raise ValueError("At least two scenarios are required.")

    rng = random.Random(seed)

    shuffled = list(scenarios)
    rng.shuffle(shuffled)

    n_train = int(round(len(shuffled) * train_spl))

    # Always leave at least one scenario for Validation B.
    n_train = max(1, min(len(shuffled) - 1, n_train))

    train = sorted(shuffled[:n_train])
    validation_b = sorted(shuffled[n_train:])

    return train, validation_b


def split_train_episodes(data_root, train_scenarios, validation_a_spl, seed):
    """
    Split complete episodes/goals inside TRAIN scenarios.

    Validation A therefore contains goals/episodes that were
    never used for training, while the scenario itself remains
    known to the model.
    """
    if not 0.0 < validation_a_spl < 1.0:
        raise ValueError("validation_a_spl must be between 0 and 1.")

    rng = random.Random(seed)

    train_episodes = []
    validation_a_episodes = []

    for scenario in train_scenarios:
        scenario_dir = os.path.join(data_root, scenario)
        npz_files = find_npz_files(scenario_dir)

        if not npz_files:
            raise RuntimeError(
                f"No .npz files found in {scenario_dir}. Run generate_npz_data.py first."
            )

        episodes = get_episode_ids(npz_files, data_root)

        if len(episodes) < 2:
            raise RuntimeError(
                f"Scenario {scenario} has fewer than two episodes; cannot create Validation A."
            )

        shuffled = episodes.copy()
        rng.shuffle(shuffled)

        n_validation = int(round(len(shuffled) * validation_a_spl))

        # Keep at least one episode in both sets.
        n_validation = max(1, min(len(shuffled) - 1, n_validation))

        validation_part = shuffled[:n_validation]
        train_part = shuffled[n_validation:]

        for episode in train_part:
            train_episodes.append({
                "scenario": scenario,
                "run": episode["run"],
                "episode_id": episode["episode_id"],
            })

        for episode in validation_part:
            validation_a_episodes.append({
                "scenario": scenario,
                "run": episode["run"],
                "episode_id": episode["episode_id"],
            })

    return train_episodes, validation_a_episodes


def write_manifest(
    output_path,
    data_root,
    all_scenarios,
    train_scenarios,
    validation_b_scenarios,
    train_episodes,
    validation_a_episodes,
    train_spl,
    validation_a_spl,
    seed,
):
    """Write the complete reproducible split."""
    manifest = {
        "format_version": 1,
        "data_root": os.path.abspath(data_root),
        "seed": seed,
        "scenario_split": {
            "train_spl": train_spl,
            "validation_b_spl": 1.0 - train_spl,
        },
        "validation_a": {
            "validation_a_spl": validation_a_spl,
            "unit": "episode",
            "description": "New goals/episodes inside known TRAIN scenarios.",
        },
        "validation_b": {
            "unit": "scenario",
            "description": "Complete scenarios never seen during training.",
        },
        "all_scenarios": all_scenarios,
        "train_scenarios": train_scenarios,
        "validation_b_scenarios": validation_b_scenarios,
        "train_episodes": train_episodes,
        "validation_a_episodes": validation_a_episodes,
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote: {output_path}")


def print_summary(
    all_scenarios,
    train_scenarios,
    validation_b_scenarios,
    train_episodes,
    validation_a_episodes,
):
    print()
    print("========== DATASET SPLIT ==========")
    print(f"scenarios total:        {len(all_scenarios)}")
    print(f"TRAIN scenarios:        {len(train_scenarios)}")
    print(f"VALIDATION B scenarios: {len(validation_b_scenarios)}")

    print()
    print("TRAIN scenarios:")
    print("  " + " ".join(train_scenarios))

    print()
    print("VALIDATION B scenarios:")
    print("  " + " ".join(validation_b_scenarios))

    print()
    print(f"TRAIN episodes:          {len(train_episodes)}")
    print(f"VALIDATION A episodes:   {len(validation_a_episodes)}")

    print()
    print("Validation A:")
    print("  complete goals/episodes inside TRAIN scenarios")

    print()
    print("Validation B:")
    print("  complete scenarios never seen during training")

    print()
    print("====================================")


def main():
    parser = argparse.ArgumentParser(
        description="Create a reproducible TRAIN / Validation A / Validation B split."
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional dataset path. Relative paths are resolved under dataset/data/.",
    )
    parser.add_argument(
        "--train-spl",
        type=float,
        default=0.75,
        help="Fraction of scenarios used for TRAIN. Default: 0.75",
    )
    parser.add_argument(
        "--validation-a-spl",
        type=float,
        default=0.25,
        help="Fraction of TRAIN episodes used for Validation A. Default: 0.25",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path. Default: dataset/data/dataset_split.json",
    )

    args = parser.parse_args()
    data_root = resolve_path(args.path)

    if not os.path.isdir(data_root):
        print(f"not a directory: {data_root}", file=sys.stderr)
        sys.exit(1)

    scenarios = find_scenarios(data_root)

    if not scenarios:
        print(f"no scenarios found under {data_root}", file=sys.stderr)
        sys.exit(1)

    train_scenarios, validation_b_scenarios = split_scenarios(
        scenarios, args.train_spl, args.seed
    )

    train_episodes, validation_a_episodes = split_train_episodes(
        data_root, train_scenarios, args.validation_a_spl, args.seed + 1
    )

    if args.output is None:
        output_path = os.path.join(data_root, "dataset_split.json")
    else:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(data_root, output_path)

    write_manifest(
        output_path=output_path,
        data_root=data_root,
        all_scenarios=scenarios,
        train_scenarios=train_scenarios,
        validation_b_scenarios=validation_b_scenarios,
        train_episodes=train_episodes,
        validation_a_episodes=validation_a_episodes,
        train_spl=args.train_spl,
        validation_a_spl=args.validation_a_spl,
        seed=args.seed,
    )

    print_summary(
        scenarios,
        train_scenarios,
        validation_b_scenarios,
        train_episodes,
        validation_a_episodes,
    )


if __name__ == "__main__":
    main()