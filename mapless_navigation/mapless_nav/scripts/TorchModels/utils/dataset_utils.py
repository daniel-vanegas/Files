#!/usr/bin/env python3

import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass

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
    train_split: float = 0.75,
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
    train_split: float = 0.75,
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
        sequence_length: int = 4,
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