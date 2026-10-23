#!/usr/bin/env python3
"""Shared data loading/preprocessing for legacy ROS 1 and ROS 2 datasets."""

import hashlib
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Episode:
    scenario: str
    run_path: str
    episode_id: int
    samples: list[dict]


@dataclass
class DatasetSplit:
    train_scenarios: list[str]
    validation_b_scenarios: list[str]
    train_episodes: list[Episode]
    validation_a_episodes: list[Episode]
    validation_b_episodes: list[Episode]


def scenario_from_run_path(run_path: str) -> str:
    """
    Infer the scenario name from:

        .../data/E02/run_x.npz

    Returns:
        E02
    """
    run_dir = os.path.dirname(run_path)
    scenario_dir = os.path.dirname(run_dir)
    return os.path.basename(scenario_dir)


def discover_npz_files(data_root: str) -> dict[str, list[str]]:
    """
    Discover all .npz files and group them by scenario.

    Expected layout:

        data/
          E01/
            run_x.npz
          E02/
            run_y.npz
    """
    scenarios = {}

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    for name in sorted(os.listdir(data_root)):
        scenario_path = os.path.join(data_root, name)

        if not os.path.isdir(scenario_path):
            continue

        if not name.startswith("E"):
            continue

        npz_files = []

        for filename in sorted(os.listdir(scenario_path)):
            if not filename.endswith(".npz"):
                continue

            npz_files.append(os.path.join(scenario_path, filename))

        if npz_files:
            scenarios[name] = npz_files

    return scenarios


def load_npz(path: str) -> list[dict]:
    """
    Load a converted run.

    allow_pickle=True is required because the current
    generator stores Python dictionaries in an object array.
    """
    with np.load(path, allow_pickle=True) as archive:
        if "data" not in archive:
            raise ValueError(f'{path}: missing "data" array')

        data = archive["data"]
        samples = []

        for item in data:
            if isinstance(item, np.ndarray):
                item = item.item()

            samples.append(item)

    return samples


def load_episodes(scenario_files: dict[str, list[str]]) -> list[Episode]:
    """
    Convert run files into episode objects.

    Episodes are grouped by episode_id.

    This is important because temporal sequences must never
    cross an episode/goal boundary.
    """
    episodes = []

    for scenario in sorted(scenario_files):
        for run_path in scenario_files[scenario]:
            samples = load_npz(run_path)
            grouped = {}

            for sample in samples:
                episode_id = int(sample["meta"]["episode_id"])
                grouped.setdefault(episode_id, []).append(sample)

            for episode_id in sorted(grouped):
                episode_samples = grouped[episode_id]
                episode_samples.sort(key=lambda s: float(s["meta"]["timestamp"]))

                episodes.append(
                    Episode(
                        scenario=scenario,
                        run_path=run_path,
                        episode_id=episode_id,
                        samples=episode_samples,
                    )
                )

    return episodes


def collect_episodes(data_root: str) -> list[Episode]:
    """
    Helper function to discover and load all episodes directly from data root.
    """
    scenario_files = discover_npz_files(data_root)
    return load_episodes(scenario_files)


def split_scenarios(
    scenarios: Sequence[str],
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split scenarios into TRAIN and Validation B.

    The split happens at scenario level, never at frame level.

    Example:
        30 scenarios
        train_split=0.75
        -> 22 train
        -> 8 validation B
    """
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    scenarios = sorted(set(scenarios))

    if len(scenarios) < 2:
        raise ValueError("At least two scenarios are required")

    rng = random.Random(seed)

    shuffled = list(scenarios)
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * train_split)

    # Ensure both sets are non-empty.
    n_train = max(1, min(n_train, len(shuffled) - 1))

    train = sorted(shuffled[:n_train])
    validation_b = sorted(shuffled[n_train:])

    return train, validation_b


def split_validation_a(
    train_episodes: Sequence[Episode],
    validation_a_split: float = 0.25,
    seed: int = 42,
) -> tuple[list[Episode], list[Episode]]:
    """
    Split TRAIN scenarios into:

        TRAIN episodes
        Validation A episodes

    Validation A contains goals/episodes that were never
    used during training.

    Episodes are kept intact.
    """
    if not 0.0 < validation_a_split < 1.0:
        raise ValueError("validation_a_split must be between 0 and 1")

    by_scenario = {}

    for episode in train_episodes:
        by_scenario.setdefault(episode.scenario, []).append(episode)

    rng = random.Random(seed)

    train = []
    validation_a = []

    for scenario in sorted(by_scenario):
        episodes = list(by_scenario[scenario])
        episodes.sort(key=lambda ep: ep.episode_id)

        if len(episodes) < 2:
            # If a scenario contains only one episode,
            # keep it in training rather than removing the
            # entire scenario from the training dataset.
            train.extend(episodes)
            continue

        rng.shuffle(episodes)

        n_validation = int(len(episodes) * validation_a_split)
        n_validation = max(1, min(n_validation, len(episodes) - 1))

        validation_a.extend(episodes[:n_validation])
        train.extend(episodes[n_validation:])

    return train, validation_a


def make_dataset_split(
    data_root: str,
    train_split: float = 0.8,
    validation_a_split: float = 0.25,
    seed: int = 42,
) -> DatasetSplit:
    """
    Build the complete experimental split.

    A:
        new goals inside TRAIN scenarios

    B:
        completely unknown scenarios
    """
    scenario_files = discover_npz_files(data_root)

    if not scenario_files:
        raise RuntimeError(f"No .npz files found under {data_root}")

    all_scenarios = sorted(scenario_files.keys())

    train_scenarios, validation_b_scenarios = split_scenarios(
        all_scenarios,
        train_split=train_split,
        seed=seed,
    )

    all_episodes = load_episodes(scenario_files)

    train_candidate_episodes = [
        episode for episode in all_episodes if episode.scenario in train_scenarios
    ]

    validation_b_episodes = [
        episode for episode in all_episodes if episode.scenario in validation_b_scenarios
    ]

    train_episodes, validation_a_episodes = split_validation_a(
        train_candidate_episodes,
        validation_a_split=validation_a_split,
        seed=seed,
    )

    return DatasetSplit(
        train_scenarios=train_scenarios,
        validation_b_scenarios=validation_b_scenarios,
        train_episodes=train_episodes,
        validation_a_episodes=validation_a_episodes,
        validation_b_episodes=validation_b_episodes,
    )


def normalize_vx(
    vx: float,
    vmax: float,
    reverse_enable: bool = False,
) -> float:
    """
    Normalize vx.

    reverse_enable=False:
        [0, vmax] -> [0, 1]

    reverse_enable=True:
        [-vmax, vmax] -> [-1, 1]
    """
    if vmax <= 0.0:
        raise ValueError("vmax must be positive")

    value = vx / vmax

    return float(
        np.clip(value, -1.0, 1.0) if reverse_enable else np.clip(value, 0.0, 1.0)
    )


def normalize_symmetric(
    value: float,
    vmax: float,
) -> float:
    """
    Normalize a symmetric velocity:

        [-vmax, vmax] -> [-1, 1]
    """
    if vmax <= 0.0:
        raise ValueError("vmax must be positive")

    return float(np.clip(value / vmax, -1.0, 1.0))


def normalize_target(
    distance: float,
    theta: float,
    distance_max: float | None = None,
) -> tuple[float, float]:
    """
    Normalize the auxiliary target.

    theta is kept in radians.

    distance can optionally be normalized by a known
    maximum distance. If distance_max is None, the raw
    distance is kept.
    """
    if distance_max is not None:
        if distance_max <= 0.0:
            raise ValueError("distance_max must be positive")

        distance = np.clip(distance / distance_max, 0.0, 1.0)

    return float(distance), float(theta)


def build_input_frame(
    sample: dict,
    vmax: tuple[float, float, float],
    reverse_enable: bool = False,
    distance_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one frame.

    Returns:
        input_frame: [82, 80]
        label:       [3]

    Layout:

        rows 0:80
            LOM

        rows 80:82
            target replicated across width

    The replication allows the existing model to receive
    the complete tensor and separate the final two rows
    internally.
    """
    lom = np.asarray(sample["features"]["occ_grid"], dtype=np.float32)

    if lom.ndim != 2:
        raise ValueError(f"Expected 2D LOM, got {lom.shape}")

    height, width = lom.shape

    if height != 80 or width != 80:
        raise ValueError(f"Expected 80x80 LOM, got {height}x{width}")

    # Descriptor says raw_0_100.
    # Convert to [0, 1].
    lom = np.clip(lom / 100.0, 0.0, 1.0)

    distance = float(sample["features"]["target"][0])
    theta = float(sample["features"]["target"][1])

    distance, theta = normalize_target(
        distance,
        theta,
        distance_max=distance_max,
    )

    target = np.empty((2, width), dtype=np.float32)
    target[0, :] = distance
    target[1, :] = theta

    input_frame = np.concatenate([lom, target], axis=0)

    vx, vy, wz = (float(v) for v in sample["labels"])

    vx_norm = normalize_vx(
        vx,
        vmax=vmax[0],
        reverse_enable=reverse_enable,
    )

    vy_norm = normalize_symmetric(vy, vmax=vmax[1])
    wz_norm = normalize_symmetric(wz, vmax=vmax[2])

    label = np.array([vx_norm, vy_norm, wz_norm], dtype=np.float32)

    return input_frame, label


class SequenceDataset(Dataset):
    """
    Dataset of C consecutive frames.
    Each item contains:
        x: [C, 82, 80]
        y: [3]

    A sequence is always contained inside one episode.
    Therefore it can never cross from one goal to another.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        sequence_length: int = 2,
        vmax: tuple[float, float, float] = (0.5, 0.5, 1.0),
        reverse_enable: bool = False,
        distance_max: float | None = None,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")

        self.sequence_length = sequence_length
        self.vmax = vmax
        self.reverse_enable = reverse_enable
        self.distance_max = distance_max

        self.samples = []
        self.sequence_meta = []

        for episode in episodes:
            if len(episode.samples) < sequence_length:
                continue

            frames = []
            labels = []

            for sample in episode.samples:
                frame, label = build_input_frame(
                    sample,
                    vmax=vmax,
                    reverse_enable=reverse_enable,
                    distance_max=distance_max,
                )

                frames.append(frame)
                labels.append(label)

            for start in range(len(frames) - sequence_length + 1):
                end = start + sequence_length

                self.samples.append(
                    (
                        np.stack(frames[start:end], axis=0),
                        labels[end - 1],
                    )
                )

                self.sequence_meta.append(
                    {
                        "scenario": episode.scenario,
                        "run_path": episode.run_path,
                        "episode_id": episode.episode_id,
                        "start_index": start,
                        "end_index": end - 1,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
        )


def summarize_split(split: DatasetSplit) -> str:
    """
    Produce a human-readable split summary.
    """
    lines = [
        "Dataset split",
        "--------------",
        f"TRAIN scenarios:       {len(split.train_scenarios)}",
        f"Validation B scenarios: {len(split.validation_b_scenarios)}",
        f"TRAIN episodes:        {len(split.train_episodes)}",
        f"Validation A episodes: {len(split.validation_a_episodes)}",
        f"Validation B episodes: {len(split.validation_b_episodes)}",
        "",
        f"TRAIN scenarios: {', '.join(split.train_scenarios)}",
        f"Validation B:   {', '.join(split.validation_b_scenarios)}",
    ]

    return "\n".join(lines)


def save_split(
    split: DatasetSplit,
    path: str,
) -> None:
    """
    Save the scenario/episode split so all later model
    experiments can reuse exactly the same split.
    """
    data = {
        "train_scenarios": split.train_scenarios,
        "validation_b_scenarios": split.validation_b_scenarios,
        "train_episodes": [
            {
                "scenario": e.scenario,
                "run_path": e.run_path,
                "episode_id": e.episode_id,
            }
            for e in split.train_episodes
        ],
        "validation_a_episodes": [
            {
                "scenario": e.scenario,
                "run_path": e.run_path,
                "episode_id": e.episode_id,
            }
            for e in split.validation_a_episodes
        ],
        "validation_b_episodes": [
            {
                "scenario": e.scenario,
                "run_path": e.run_path,
                "episode_id": e.episode_id,
            }
            for e in split.validation_b_episodes
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Legacy ROS 1 NPZ loading and preprocessing. Kept here to maintain one shared data utility module.

def unwrap(value: Any) -> Any:
    """Unwrap scalar NumPy objects, without transforming the underlying data."""
    while isinstance(value, np.ndarray) and value.shape == ():
        try:
            value = value.item()
        except Exception:
            break
    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return value
    return value


def is_sample(value: Any) -> bool:
    value = unwrap(value)
    return (
        isinstance(value, dict)
        and isinstance(unwrap(value.get("features")), dict)
        and "labels" in value
        and "occ_grid" in unwrap(value.get("features", {}))
        and "target" in unwrap(value.get("features", {}))
    )


def flatten_samples(value: Any) -> list[dict[str, Any]]:
    """Flatten supported object-array/list containers into individual samples."""
    value = unwrap(value)
    if is_sample(value):
        return [value]
    if isinstance(value, np.ndarray):
        if value.dtype == object or value.ndim > 0:
            result: list[dict[str, Any]] = []
            for item in value:
                result.extend(flatten_samples(item))
            return result
        return []
    if isinstance(value, (list, tuple)):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(flatten_samples(item))
        return result
    return []


def load_samples_from_npz(path: Path) -> list[dict[str, Any]]:
    """Read per-file sample records using common legacy/current NPZ keys.

    Dataset 2025-2 contains one valid NPZ container whose ``data`` array is
    empty (shape ``(0,)``). The legacy flat loader naturally contributes zero
    samples for that file, so treat an empty canonical data array as an empty
    file rather than as a schema error. Non-empty, unrecognized schemas still
    raise an exception to avoid silently discarding data.
    """
    with np.load(path, allow_pickle=True) as archive:
        keys = list(archive.files)
        preferred = [k for k in ("data", "samples", "records", "dataset", "arr_0") if k in keys]
        for key in preferred:
            arr = archive[key]
            samples = flatten_samples(arr)
            if samples:
                return samples

        # This dataset has one file with data.shape == (0,), dtype=float64.
        # Return an empty list only when the canonical data array is empty;
        # any other non-empty but unrecognized schema remains an explicit error.
        if "data" in keys and archive["data"].size == 0:
            return []

        # Some datasets store each field separately instead of storing a `data` object array.
        if {"features", "labels"}.issubset(keys):
            features = flatten_samples(archive["features"])
            labels = flatten_samples(archive["labels"])
            if len(features) == len(labels) and features:
                # This uncommon structure is ambiguous because the nested field arrays
                # may have been flattened; refuse to guess rather than corrupt the data.
                raise ValueError(
                    f"{path}: found separate 'features' and 'labels' arrays but cannot "
                    "safely reconstruct sample dictionaries. Inspect the NPZ structure."
                )

        raise ValueError(
            f"{path}: could not find sample dictionaries. NPZ keys={keys}. "
            "Expected an object array under data/samples/records/dataset/arr_0. "
            "Run inspect_dataset_2025_2.py and adapt the loader to the observed schema."
        )


def load_legacy_flat_dataset(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all .npz files in deterministic path order and flatten their records.

    Legacy train.py used a glob of NPZ files followed by load_data(), then paired
    every two consecutive samples. This loader mirrors that flat-list behavior;
    it does not introduce the newer ROS 2 episode-boundary sequence logic.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    files = sorted(p for p in data_root.rglob("*.npz") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_root}")

    all_samples: list[dict[str, Any]] = []
    file_stats: list[dict[str, Any]] = []
    for path in files:
        samples = load_samples_from_npz(path)
        all_samples.extend(samples)
        file_stats.append({
            # Never serialize the machine-specific absolute path.
            "path": path.relative_to(data_root).as_posix(),
            "relative_path": path.relative_to(data_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "samples_loaded": len(samples),
        })
    return all_samples, file_stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path | None, anchor: Path) -> str | None:
    """Return a portable, non-absolute path for metadata and logs.

    Paths inside TorchModels are stored relative to the script directory.
    For external paths, store only an explicit <external>/<basename> marker,
    avoiding leakage of machine-specific home/workspace paths into artifacts.
    """
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    base = anchor.expanduser().resolve()
    try:
        return resolved.relative_to(base).as_posix() or "."
    except ValueError:
        return f"<external>/{resolved.name}"


def build_legacy_tensors(
    samples: list[dict[str, Any]],
    stack_n_channels: int,
    occ_grid_meters: int,
    normalize_label: bool,
    normalize_data: bool,
    label_normalization: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Mirror the original crop, replicated target rows, non-overlap stacking and labels."""
    if stack_n_channels < 1:
        raise ValueError("stack_n_channels must be >= 1")
    if not samples:
        raise ValueError("No samples loaded from the dataset")

    first_grid = np.asarray(samples[0]["features"]["occ_grid"])
    if first_grid.ndim != 2 or first_grid.shape[0] != first_grid.shape[1]:
        raise ValueError(f"First occ_grid must be square 2D; got {first_grid.shape}")

    matrix_shape = tuple(int(v) for v in first_grid.shape)
    sub_rows = int(20 * occ_grid_meters)
    if sub_rows > matrix_shape[0]:
        raise ValueError(
            f"Legacy crop requires at least {sub_rows}x{sub_rows}, but first occ_grid is {matrix_shape}. "
            "The original script's fallback branch would produce inconsistent tensor shapes."
        )

    img_center = matrix_shape[0] // 2
    min_indx = img_center - sub_rows // 2
    max_indx = img_center + sub_rows // 2

    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    expected_samples = 0
    skipped_invalid: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        try:
            features = sample["features"]
            occ_grid = np.asarray(features["occ_grid"], dtype=np.float32)
            target = np.asarray(features["target"], dtype=np.float32).reshape(-1)
            label = np.asarray(sample["labels"], dtype=np.float32).reshape(-1)
            if occ_grid.shape != matrix_shape:
                raise ValueError(f"occ_grid shape {occ_grid.shape}, expected {matrix_shape}")
            if target.size < 2:
                raise ValueError(f"target must contain distance and theta; shape={target.shape}")
            if label.size != 3:
                raise ValueError(f"labels must have 3 values [vx, vy, wz]; got {label.shape}")

            # Exact crop used by ROS 1 train.py: bottom sub_rows and centered columns.
            grid_crop = occ_grid[matrix_shape[0] - sub_rows :, min_indx:max_indx]
            if grid_crop.shape != (sub_rows, sub_rows):
                raise ValueError(f"cropped occ_grid shape {grid_crop.shape}, expected {(sub_rows, sub_rows)}")

            # Replicate target features across columns, producing (sub_rows + 2, sub_rows).
            target_rows = np.array(
                [np.ones(sub_rows, dtype=np.float32) * target[0],
                 np.ones(sub_rows, dtype=np.float32) * target[1]],
                dtype=np.float32,
            )
            frame = np.vstack((grid_crop, target_rows)).astype(np.float32, copy=False)
            frames.append(frame)
            labels.append(label)
            expected_samples += 1
        except Exception as exc:
            skipped_invalid.append({"sample_index": idx, "error": str(exc)})
            raise ValueError(f"Invalid sample at flattened index {idx}: {exc}") from exc

    # Stack non-overlapping groups of C frames exactly like the original script.
    num_groups = len(frames) // stack_n_channels
    usable = num_groups * stack_n_channels
    if usable == 0:
        raise ValueError(
            f"Only {len(frames)} samples loaded; at least {stack_n_channels} are needed to form an input."
        )
    X = np.stack(
        [np.stack(frames[i : i + stack_n_channels], axis=0) for i in range(0, usable, stack_n_channels)],
        axis=0,
    ).astype(np.float32, copy=False)
    # The label comes from the last frame of each stacked group.
    Y = np.stack([labels[i + stack_n_channels - 1] for i in range(0, usable, stack_n_channels)], axis=0).astype(np.float32)

    original_label_min = Y.min(axis=0).astype(float).tolist()
    original_label_max = Y.max(axis=0).astype(float).tolist()
    if normalize_label:
        if label_normalization == "legacy":
            # Preserve original script behavior—even though its comments misidentify column 1.
            for col, dest_range in ((1, (-1.0, 1.0)), (0, (0.0, 1.0))):
                span = float(np.ptp(Y[:, col]))
                if span == 0.0:
                    Y[:, col] = 0.0
                else:
                    scaled = (Y[:, col] - np.min(Y[:, col])) / span
                    Y[:, col] = scaled * (dest_range[1] - dest_range[0]) + dest_range[0]
        elif label_normalization == "vx_wz":
            # Optional semantically intended alternative; not used for historical replica.
            for col, dest_range in ((0, (0.0, 1.0)), (2, (-1.0, 1.0))):
                span = float(np.ptp(Y[:, col]))
                if span == 0.0:
                    Y[:, col] = 0.0
                else:
                    scaled = (Y[:, col] - np.min(Y[:, col])) / span
                    Y[:, col] = scaled * (dest_range[1] - dest_range[0]) + dest_range[0]
        elif label_normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown label_normalization: {label_normalization}")

    if normalize_data:
        # Original script divided the occupancy-grid rows by 100, not the target rows.
        X[:, :, :-2, :] /= 100.0

    stats = {
        "num_input_samples_loaded": len(samples),
        "num_samples_used_for_pairs": usable,
        "num_samples_dropped_as_unpaired_tail": len(frames) - usable,
        "num_training_examples_after_stacking": int(X.shape[0]),
        "input_occ_grid_shape": list(matrix_shape),
        "cropped_occ_grid_shape": [sub_rows, sub_rows],
        "input_tensor_shape": list(X.shape),
        "label_tensor_shape": list(Y.shape),
        "label_min_before_normalization": original_label_min,
        "label_max_before_normalization": original_label_max,
        "normalization": "legacy columns 0 and 1" if normalize_label and label_normalization == "legacy" else label_normalization,
        "invalid_samples": skipped_invalid,
    }
    return X, Y, stats


