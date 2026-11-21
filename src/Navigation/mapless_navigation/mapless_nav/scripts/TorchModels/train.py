#!/usr/bin/env python3
"""Single configurable trainer for legacy ROS 1 and ROS 2 mapless datasets.

The default architecture and hyperparameters preserve the historical ROS 1
setup for BOTH data formats. Dataset parsing and split rules remain format-aware:
legacy NPZ files use the original flat/non-overlapping sample pairing, while ROS 2
recordings keep each temporal sequence inside an episode and support scenario
holdouts for generalization experiments.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_TOOLS_DIR = SCRIPT_DIR.parent / "experiment_tools"
EXPERIMENTS_ROOT = (SCRIPT_DIR.parents[5] if len(SCRIPT_DIR.parents) > 5 else SCRIPT_DIR.parent) / "experiments"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if str(EXPERIMENT_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_TOOLS_DIR))

from dataset_utils import (
    SequenceDataset,
    EpisodeChunkDataset,
    EpisodeOrderSampler,
    build_input_frame,
    build_legacy_tensors,
    discover_npz_files,
    load_episodes,
    load_legacy_flat_dataset,
    load_samples_from_npz,
    split_scenarios,
    split_validation_a,
)
from models import RNN, TemporalPolicy, GoalConditionedConvLSTM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_names(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return sorted({item.strip() for item in value.split(",") if item.strip()})


def detect_data_format(root: Path) -> str:
    files = sorted(root.rglob("*.npz"))
    for path in files:
        try:
            samples = load_samples_from_npz(path)
        except Exception:
            continue
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            meta = sample.get("meta")
            if isinstance(meta, dict) and "episode_id" in meta and "timestamp" in meta:
                return "ros2_episode"
            features = sample.get("features")
            if isinstance(features, dict) and "occ_grid" in features and "target" in features and "labels" in sample:
                return "ros1_legacy"
    raise RuntimeError(
        f"Could not infer dataset format under {root}. Set --data-format ros1_legacy "
        "or --data-format ros2_episode explicitly and inspect the NPZ schema."
    )


def resolve_defaults(args, data_format: str) -> None:
    """Resolve one historical default set regardless of dataset source."""
    # Original ROS 1 architecture is the common default for both ROS 1 and ROS 2
    # data. Native ROS 2 model layouts remain configurable for explicit experiments
    # and compatibility with checkpoints trained by the previous ROS 2 model.
    args.sequence_length = args.sequence_length if args.sequence_length is not None else (8 if args.model_type == "convlstm" else 2)
    args.hidden_size = args.hidden_size if args.hidden_size is not None else 350
    args.num_layers = args.num_layers if args.num_layers is not None else (1 if args.model_type == "convlstm" else 4)
    args.batch_size = args.batch_size if args.batch_size is not None else (1 if args.model_type == "convlstm" else 3)
    args.epochs = args.epochs if args.epochs is not None else 20
    args.learning_rate = args.learning_rate if args.learning_rate is not None else 0.0006
    args.train_split = args.train_split if args.train_split is not None else 0.8
    args.patience = args.patience if args.patience is not None else 5
    args.normalize_label = args.normalize_label if args.normalize_label is not None else True
    args.normalize_data = args.normalize_data if args.normalize_data is not None else False
    args.label_normalization = args.label_normalization or "legacy"
    args.vmax = tuple(args.vmax) if args.vmax else (0.5, 0.5, 1.0)


# Image size for each variant after the slice is applied (rows, cols).
VARIANT_SHAPES = {
    "4x4": (80, 80),
    "5x4": (100, 80),
    "7x6": (140, 120),
}


def make_model(args):
    img_h, img_w = VARIANT_SHAPES[args.variant]
    if args.model_type == "convlstm":
        return GoalConditionedConvLSTM(
            img_h=img_h, img_w=img_w,
            spatial_hidden_channels=args.spatial_hidden_channels,
            num_layers=args.num_layers, dropout_1=args.dropout_1, dropout_2=args.dropout_2,
            encoder_channels=tuple(args.encoder_channels),
            goal_fusion=args.goal_fusion,
        )
    if args.model_type is not None:
        return TemporalPolicy(
            model_type=args.model_type,
            img_h=img_h,
            img_w=img_w,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout_1=args.dropout_1,
            dropout_2=args.dropout_2,
        )
    return RNN(
        channels=args.sequence_length,
        img_h=img_h,
        img_w=img_w,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        architecture=args.architecture,
        dropout_1=args.dropout_1,
        dropout_2=args.dropout_2,
    )


def make_ros1_loaders(args, root: Path, seed: int):
    samples, file_stats = load_legacy_flat_dataset(root)
    X, Y, stats = build_legacy_tensors(
        samples=samples,
        stack_n_channels=args.sequence_length,
        occ_grid_meters=args.occ_grid_meters,
        normalize_label=args.normalize_label,
        normalize_data=args.normalize_data,
        label_normalization=args.label_normalization,
    )
    if len(X) < 2:
        raise RuntimeError(f"Only {len(X)} legacy training examples; at least two are required.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(indices) * args.train_split)
    split = max(1, min(split, len(indices) - 1))
    train_idx, val_idx = indices[:split], indices[split:]
    train_set = TensorDataset(torch.from_numpy(X[train_idx]).float(), torch.from_numpy(Y[train_idx]).float())
    val_set = TensorDataset(torch.from_numpy(X[val_idx]).float(), torch.from_numpy(Y[val_idx]).float())
    return (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False),
        {
            "dataset_format": "ros1_legacy",
            "num_npz_files": len(file_stats),
            "num_raw_samples": len(samples),
            "num_train_examples": len(train_idx),
            "num_validation_examples": len(val_idx),
            "data_stats": stats,
            "files": file_stats,
            "split": "random sample-level split after non-overlapping frame stacking (legacy semantics)",
        },
    )


class ManifestPairs(torch.utils.data.Dataset):
    """Load fixed pairs while keeping original episode indices."""

    def __init__(self, root, entries, args):
        self.samples = [None] * len(entries)
        self.sequence_meta = entries
        grouped = {}
        for index, entry in enumerate(entries):
            grouped.setdefault(entry["run_file"], []).append((index, entry))
        for relative, selected in sorted(grouped.items()):
            path = root / relative
            episodes = load_episodes({Path(relative).parent.name: [str(path)]})
            by_id = {episode.episode_id: episode for episode in episodes}
            for index, entry in selected:
                episode = by_id[entry["episode_id"]]
                start = entry["start_index"]
                pair = episode.samples[start:start + args.sequence_length]
                if len(pair) != args.sequence_length or any(not sample["meta"].get("tlom_consistent", True) for sample in pair):
                    raise ValueError(f"Unavailable pair in {relative}, episode {episode.episode_id}, index {start}")
                frames = [build_input_frame(sample, args.vmax, args.reverse_enable,
                          args.distance_max, args.variant, args.representation) for sample in pair]
                inputs = np.stack([item[0] for item in frames])
                inputs[:, :-2, :] *= args.occupancy_max
                self.samples[index] = (inputs, frames[-1][1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y = self.samples[index]
        if getattr(self, "stateful", False):
            meta = self.sequence_meta[index]
            ep_key = meta.get("episode_key", f"{meta['run_file']}#{meta['episode_id']}")
            return torch.from_numpy(x).float(), torch.from_numpy(y).float(), ep_key
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def make_pair_manifest_loaders(args, root, seed):
    manifest_path = args.pair_manifest.expanduser().resolve()
    spec = json.loads(manifest_path.read_text())
    version = spec.get("version")
    channels = 2 if version == 1 else spec.get("sequence_length")
    if (version not in (1, 2) or spec.get("variant") != args.variant
            or channels != args.sequence_length or args.data_format != "ros2_episode"):
        raise ValueError("The manifest requires ROS 2 data and its saved variant and sequence length")
    if any((args.train_scenarios, args.validation_scenarios, args.test_scenarios, args.datasets_dir)):
        raise ValueError("Use the saved manifest split without additional scenario selectors")
    selected_files = {row["run_file"] for split in ("train", "validation") for row in spec[split]}
    if selected_files != set(spec["sources"]):
        raise ValueError("Every selected NPZ must have a saved source hash")
    for relative, expected in spec["sources"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("A manifest source is outside data-root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Dataset source changed: {relative}")
    train_ids = {(r["run_file"], r["episode_id"]) for r in spec["train"]}
    val_ids = {(r["run_file"], r["episode_id"]) for r in spec["validation"]}
    if train_ids & val_ids or not train_ids or not val_ids:
        raise ValueError("The manifest must have nonempty, disjoint episode partitions")
    for split in ("train", "validation"):
        ids = [(r["run_file"], r["episode_id"], r["start_index"]) for r in spec[split]]
        last_end = {}
        for run, episode, start in sorted(ids):
            key = (run, episode)
            if start < last_end.get(key, 0) or (version == 1 and start % 2):
                raise ValueError("Manifest windows must be distinct and non-overlapping within episodes")
            last_end[key] = start + channels
    if args.stateful:
        spec["train"] = sorted(spec["train"], key=lambda x: (x["run_file"], x["episode_id"], x["start_index"]))
        spec["validation"] = sorted(spec["validation"], key=lambda x: (x["run_file"], x["episode_id"], x["start_index"]))
        
    train_set = ManifestPairs(root, spec["train"], args)
    val_set = ManifestPairs(root, spec["validation"], args)
    train_set.stateful = args.stateful
    val_set.stateful = args.stateful
    
    generator = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    
    bs = 1 if args.stateful else args.batch_size
    shuf = not args.stateful
    
    return (
        DataLoader(train_set, batch_size=bs, shuffle=shuf, generator=generator, pin_memory=pin),
        DataLoader(val_set, batch_size=bs, shuffle=False, pin_memory=pin),
        {"dataset_format": "ros2_episode", "environment_set": spec["environment_set"],
         "train_sequences": len(train_set), "validation_sequences": len(val_set),
         "pair_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
         "split": "fixed episode partitions and fixed original pairs",
         **({"require_full_sequence": True} if version == 2 else {})},
    )


def audit_pair_actions(model, dataset, device, vmax):
    """Compare decoded predictions with recorded actions on the same pairs."""
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=128, shuffle=False):
            x = batch[0]
            output = model(x.to(device))
            predictions.append(output.detach().cpu().numpy())
    predicted = np.concatenate(predictions, axis=0) * np.asarray(vmax)
    demonstrated = np.asarray([row["action"] for row in dataset.sequence_meta], dtype=float)
    categories = np.asarray([row["route_class"] for row in dataset.sequence_meta])
    results = {}
    for category in sorted(set(categories)):
        metrics = {}
        for axis, (name, deadband) in enumerate((("vx", 0.02), ("vy", 0.02), ("wz", 0.05))):
            metrics[name] = {}
            for label, sign in (("positive", 1), ("negative", -1)):
                mask = (categories == category) & (sign * demonstrated[:, axis] > deadband)
                expected = demonstrated[mask, axis]
                actual = predicted[mask, axis]
                metrics[name][label] = {
                    "pairs": int(np.count_nonzero(mask)),
                    "sign_accuracy": float(np.mean(sign * actual > 0)) if len(expected) else None,
                    "median_amplitude_ratio": float(np.median(np.abs(actual / expected))) if len(expected) else None,
                }
        results[category] = metrics
    return results



def make_ros2_convlstm_loaders(args, root: Path, scenario_files, train_names,
                                val_names, test_names, seed):
    """Stream one NPZ at a time into a compressed, episode-safe dataset.

    A seeded hash selects Validation A episodes independently of file order.
    Only training/validation scenarios are opened; test maps remain untouched.
    """
    train_dataset = EpisodeChunkDataset([], sequence_length=args.sequence_length,
                                        variant=args.variant, representation=args.representation)
    val_dataset = EpisodeChunkDataset([], sequence_length=args.sequence_length,
                                      variant=args.variant, representation=args.representation)
    train_count, val_count = 0, 0
    episode_plan = None
    wanted_train, wanted_val = set(), set()
    discovered_train, discovered_val = set(), set()
    source_hashes = {}
    if args.episode_plan:
        plan_path = args.episode_plan.expanduser().resolve(strict=True)
        episode_plan = json.loads(plan_path.read_text())
        if episode_plan.get("version") != 1:
            raise ValueError("Unsupported episode plan version")
        if episode_plan.get("train_scenarios") != sorted(set(train_names)):
            raise ValueError("Episode plan training scenarios differ from CLI")
        if episode_plan.get("reserved_test_scenarios") != sorted(set(test_names)):
            raise ValueError("Episode plan test holdout differs from CLI")
        if val_names:
            raise ValueError("Episode plan requires episode-level validation, not --validation-scenarios")
        def read_keys(rows, label):
            keys = set()
            for row in rows:
                relative = row["run_file"]
                p = Path(relative)
                if p.is_absolute() or ".." in p.parts or len(p.parts) != 2 or p.suffix != ".npz":
                    raise ValueError(f"Unsafe {label} NPZ file path: {relative}")
                if p.parts[0] not in set(train_names):
                    raise ValueError(f"{label} episode outside training scenarios: {relative}")
                key = (relative, int(row["episode_id"]))
                if key in keys:
                    raise ValueError(f"Duplicate {label} episode: {key}")
                keys.add(key)
            return keys
        wanted_train = read_keys(episode_plan["train"], "training")
        wanted_val = read_keys(episode_plan["validation"], "validation")
        if not wanted_train or not wanted_val or wanted_train & wanted_val:
            raise ValueError("Plan has empty or overlapping training/validation episodes")
        if any(Path(rel).parts[0] not in set(episode_plan.get("validation_source_scenarios", []))
               for rel, _ in wanted_val):
            raise ValueError("Validation includes scenarios outside the fixed validation source")
        source_hashes = episode_plan["source_hashes"]
        source_set = {rel for rel, _ in wanted_train | wanted_val}
        if set(source_hashes) != source_set:
            raise ValueError("Plan source hashes must match selected NPZ files exactly")

    def add_records(dataset, episodes):
        if not episodes:
            return
        temporary = EpisodeChunkDataset(
            episodes, sequence_length=args.sequence_length, vmax=args.vmax,
            reverse_enable=args.reverse_enable, distance_max=args.distance_max,
            variant=args.variant, representation=args.representation,
            skip_inconsistent_tlom=True,
            max_frame_gap_seconds=args.max_frame_gap_seconds,
        )
        offset = len(dataset.samples)
        dataset.samples.extend(temporary.samples)
        dataset.sequence_meta.extend(temporary.sequence_meta)
        dataset.episode_groups.extend([[i + offset for i in group]
                                       for group in temporary.episode_groups])

    for scenario in sorted(set(train_names) | set(val_names or [])):
        for file in scenario_files[scenario]:
            # Each NPZ is released before the next is loaded. This prevents
            # the uncompressed 30-scenario recording from occupying RAM at once.
            if not Path(file).resolve().is_relative_to(root.resolve()):
                raise ValueError(f"NPZ path is outside data root: {file}")
            relative = Path(file).relative_to(root).as_posix()
            if episode_plan is not None:
                if relative not in source_hashes:
                    continue
                digest = hashlib.sha256()
                with Path(file).open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        digest.update(block)
                if digest.hexdigest() != source_hashes[relative]:
                    raise ValueError(f"Source NPZ changed since episode plan creation: {relative}")
            episodes = load_episodes({scenario: [file]})
            selected_train, selected_val = [], []
            if episode_plan is not None:
                for episode in episodes:
                    key = (relative, int(episode.episode_id))
                    if key in wanted_train:
                        selected_train.append(episode)
                        discovered_train.add(key)
                    elif key in wanted_val:
                        selected_val.append(episode)
                        discovered_val.add(key)
            elif scenario in set(val_names or []):
                selected_val = episodes
            elif val_names:
                selected_train = episodes
            else:
                scored = sorted(episodes, key=lambda ep: hashlib.sha256(
                    f"{seed}|{scenario}|{Path(file).relative_to(root)}|{ep.episode_id}".encode()
                ).digest())
                if len(scored) >= 2:
                    nval = max(1, min(round(len(scored) * args.validation_a_split), len(scored) - 1))
                    validation_ids = {ep.episode_id for ep in scored[:nval]}
                else:
                    validation_ids = ({scored[0].episode_id} if scored and
                        int.from_bytes(hashlib.sha256(
                            f"{seed}|{scenario}|{Path(file).relative_to(root)}|{scored[0].episode_id}".encode()
                        ).digest()[:8], 'big') / 2**64 < args.validation_a_split else set())
                for episode in episodes:
                    (selected_val if episode.episode_id in validation_ids else selected_train).append(episode)
            train_count += len(selected_train)
            val_count += len(selected_val)
            add_records(train_dataset, selected_train)
            add_records(val_dataset, selected_val)
            del selected_train, selected_val, episodes

    if episode_plan is not None:
        if discovered_train != wanted_train or discovered_val != wanted_val:
            raise ValueError(f"Episode plan references missing training/validation episodes: "
                             f"train={sorted(wanted_train-discovered_train)[:5]}, "
                             f"validation={sorted(wanted_val-discovered_val)[:5]}")
    if not train_dataset or not val_dataset:
        raise RuntimeError("ConvLSTM needs nonempty training and validation episodes/chunks; "
                           "use more episodes or explicit validation scenarios")
    if episode_plan is not None:
        method = 'frozen episode selection plan; common core validation; test scenarios untouched'
    elif val_names:
        method = 'explicit scenario validation; test scenarios untouched'
    else:
        method = 'seeded episode-level Validation A, per-run minimum where possible; test scenarios untouched'
    use_pin = torch.cuda.is_available()
    return (
        DataLoader(train_dataset, batch_size=1,
                   sampler=EpisodeOrderSampler(train_dataset, seed, shuffle=True),
                   num_workers=0, pin_memory=use_pin),
        DataLoader(val_dataset, batch_size=1,
                   sampler=EpisodeOrderSampler(val_dataset, seed, shuffle=False),
                   num_workers=0, pin_memory=use_pin),
        {"dataset_format": "ros2_episode", "training_unit": "chronological_episode_chunks",
         "train_scenarios": sorted(set(train_names)),
         "validation_scenarios": sorted(set(val_names or [])),
         "test_scenarios_reserved": sorted(set(test_names)),
         "train_episodes": train_count, "validation_episodes": val_count,
         "train_sequences": len(train_dataset), "validation_sequences": len(val_dataset),
         "max_frame_gap_seconds": args.max_frame_gap_seconds,
         "split": method, "stateful": True,
         "episode_plan_sha256": hashlib.sha256(args.episode_plan.expanduser().read_bytes()).hexdigest() if episode_plan is not None else None,
         "episode_plan_budget": episode_plan.get("budget") if episode_plan is not None else None,
         "train_valid_frames_inventory": sum(int(row.get("valid_frames", 0)) for row in episode_plan["train"]) if episode_plan is not None else None,
         "validation_valid_frames_inventory": sum(int(row.get("valid_frames", 0)) for row in episode_plan["validation"]) if episode_plan is not None else None,
         "source_loading": "one_npz_at_a_time_lossless_compression"},
    )

def make_ros2_loaders(args, root: Path, seed: int):
    scenario_files = discover_npz_files(str(root))
    if not scenario_files:
        raise RuntimeError(
            f"No ROS 2 scenario folders found under {root}; expected data/E01/*.npz, data/E02/*.npz, ..."
        )
    all_scenarios = sorted(scenario_files)
    # Do not load reserved evaluation maps into the training process.
    train_names = parse_csv_names(args.train_scenarios)
    val_names = parse_csv_names(args.validation_scenarios)
    test_names = parse_csv_names(args.test_scenarios) or []
    known = set(all_scenarios)
    for label, names in (("train", train_names), ("validation", val_names), ("test", test_names)):
        if names is not None:
            # Explicitly reserved holdout maps need not have recordings yet.
            # Their NPZ files must not be opened by a training campaign.
            available = {f"E{i:02d}" for i in range(1, 31)} if label == "test" and args.episode_plan else known
            unknown = sorted(set(names) - available)
            if unknown:
                raise ValueError(f"Unknown {label} scenarios: {unknown}; available: {sorted(available)}")

    if train_names is None:
        eligible = sorted(known - set(test_names))
        train_names, auto_val_names = split_scenarios(eligible, args.train_split, seed)
        # By default, scenario-level holdout is reserved for validation B.
        if val_names is None:
            val_names = auto_val_names
    elif val_names is None:
        # Explicit train maps: split episodes within known train maps for validation A.
        val_names = []

    overlap = (set(train_names) & set(val_names or [])) | (set(train_names) & set(test_names)) | (set(val_names or []) & set(test_names))
    if overlap:
        raise ValueError(f"Train/validation/test scenario lists overlap: {sorted(overlap)}")

    if args.model_type == "convlstm":
        return make_ros2_convlstm_loaders(args, root, scenario_files, train_names,
                                         val_names, test_names, seed)

    allowed = set(train_names) | set(val_names or [])
    all_episodes = load_episodes({name: files for name, files in scenario_files.items()
                                  if name in allowed})
    if not all_episodes:
        raise RuntimeError("No train/validation episodes found in selected NPZ files")
    train_candidates = [e for e in all_episodes if e.scenario in set(train_names)]
    if val_names:
        val_episodes = [e for e in all_episodes if e.scenario in set(val_names)]
        train_episodes = train_candidates
        split_description = "explicit/scenario-level train-validation split; test scenarios untouched"
    else:
        train_episodes, val_episodes = split_validation_a(
            train_candidates,
            validation_a_split=args.validation_a_split,
            seed=seed,
        )
        split_description = "episodes split within explicit train scenarios (Validation A); test scenarios untouched"

    train_dataset = SequenceDataset(
        train_episodes,
        sequence_length=args.sequence_length,
        vmax=args.vmax,
        reverse_enable=args.reverse_enable,
        distance_max=args.distance_max,
        variant=args.variant,
        representation=args.representation,
        skip_inconsistent_tlom=True,
    )
    val_dataset = SequenceDataset(
        val_episodes,
        sequence_length=args.sequence_length,
        vmax=args.vmax,
        reverse_enable=args.reverse_enable,
        distance_max=args.distance_max,
        variant=args.variant,
        representation=args.representation,
        skip_inconsistent_tlom=True,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("ROS 2 train dataset contains zero sequences; check sequence length and train scenarios.")
    if len(val_dataset) == 0:
        raise RuntimeError(
            "ROS 2 validation dataset contains zero sequences. Supply --validation-scenarios or "
            "ensure each train scenario has enough episodes for Validation A."
        )
    use_pin = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)
    # Paired T4/T5 tests can use an identical example budget.
    full_train_size, full_val_size = len(train_dataset), len(val_dataset)
    rng = np.random.default_rng(seed)
    if args.train_budget is not None:
        if args.train_budget > full_train_size:
            raise ValueError(f"train-budget={args.train_budget} exceeds {full_train_size} available examples")
        chosen = sorted(rng.choice(full_train_size, args.train_budget, replace=False).tolist())
        train_dataset = Subset(train_dataset, chosen)
    if args.validation_budget is not None:
        if args.validation_budget > full_val_size:
            raise ValueError(f"validation-budget={args.validation_budget} exceeds {full_val_size} available examples")
        chosen = sorted(rng.choice(full_val_size, args.validation_budget, replace=False).tolist())
        val_dataset = Subset(val_dataset, chosen)
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                   pin_memory=use_pin, generator=generator),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin),
        {
            "full_train_sequences": full_train_size,
            "full_validation_sequences": full_val_size,
            "train_budget": args.train_budget,
            "validation_budget": args.validation_budget,
            "dataset_format": "ros2_episode",
            "all_scenarios": all_scenarios,
            "train_scenarios": sorted(set(train_names)),
            "validation_scenarios": sorted(set(val_names or [])),
            "test_scenarios_reserved": sorted(set(test_names)),
            "train_episodes": len(train_episodes),
            "validation_episodes": len(val_episodes),
            "train_sequences": len(train_dataset),
            "validation_sequences": len(val_dataset),
            "split": split_description,
        },
    )


def make_manifest_loaders(args, seed: int):
    from experiment_control import load_frozen_selection, prepare_frozen_pairs

    if (args.architecture, args.sequence_length, args.hidden_size, args.num_layers) != ("ros1_legacy", 2, 350, 4):
        raise ValueError("Prepared datasets require the reference RNN with C=2, hidden size 350 and four layers")
    if (args.normalize_data or not args.normalize_label or args.label_normalization != "legacy"
            or args.distance_max is not None or args.occ_grid_meters != 4
            or any((args.train_scenarios, args.validation_scenarios, args.test_scenarios))):
        raise ValueError("Prepared datasets keep raw LOM values, raw target features, legacy label normalization and the saved partition")
    selection = load_frozen_selection(args.datasets_dir, args.environment_set, args.dataset,
                                      args.data_root, args.catalog_dir)
    print("Manifest checks passed. Loading the selected observations and training normalization pool.", flush=True)
    train_set, val_set, preprocessing = prepare_frozen_pairs(selection)
    generator = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    info = {"dataset_format": "ros2_episode", "environment_set": args.environment_set, "dataset": args.dataset,
            "datasets_sha256": selection["datasets_sha256"], "manifest_sha256": selection["manifest_hashes"],
            "catalog_sha256": selection["metadata"]["catalog_sha256"],
            "split_seed": selection["metadata"]["split_seed"], "selection_seed": selection["metadata"]["selection_seed"],
            "train_sequences": len(train_set), "validation_sequences": len(val_set),
            "split": "frozen episode partition; disjoint pairs; shuffle training pairs across episodes",
            "normalization_source": "complete D0 training pool; shared by recipes of the same environment set",
            "preprocessing": preprocessing}
    return (DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
                       num_workers=0, pin_memory=pin),
            DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin),
            info, selection["root"])


def check_inference_contract(model, config: dict, dataset, device) -> dict:
    node_path = SCRIPT_DIR.parent / "RNN_mapless.py"
    module_spec = importlib.util.spec_from_file_location("mapless_inference_check", node_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    state = model.state_dict()
    architecture = module.detect_architecture(state, {"architecture": config["architecture"]})
    inference, settings = module.make_model(architecture, state, config)
    inference.to(device).eval()
    training_mode = model.training
    model.eval()
    try:
        expected, _ = dataset[0]
        actual = np.stack([module.prepare_observation(np.rot90(np.fliplr(grid), -1), target[0], target[1], settings)
                           for grid, target, _ in dataset.raw_pair(0)])
        np.testing.assert_array_equal(actual, expected)
        x = torch.as_tensor(expected[None], dtype=torch.float32, device=device)
        with torch.inference_mode():
            trained = model(x).detach().cpu().numpy()
            deployed = inference(x).detach().cpu().numpy()
        if not np.isfinite(trained).all() or not np.isfinite(deployed).all():
            raise ValueError("Model compatibility check produced non-finite predictions")
        np.testing.assert_allclose(trained, deployed, atol=1e-6, rtol=1e-5)
        decoded = module.decode_action(deployed[0], settings)
        spec = config["preprocessing"]
        minimum, maximum = np.asarray(spec["label_min"]), np.asarray(spec["label_max"])
        reference = deployed[0].astype(float)
        reference[0] = minimum[0] + reference[0] * (maximum[0] - minimum[0])
        reference[1] = minimum[1] + (reference[1] + 1) * 0.5 * (maximum[1] - minimum[1])
        np.testing.assert_allclose(decoded, reference, atol=1e-6, rtol=1e-5)
    finally:
        model.train(training_mode)
    if tuple(state["conv1.weight"].shape) != (2, 2, 3, 3) or tuple(state["rnn_cell_list.0.x2h.weight"].shape) != (350, 736):
        raise ValueError("Checkpoint does not have the reference convolution and recurrent input sizes")
    return {"status": "passed", "input_shape": list(actual.shape), "input_arrays_equal": True,
            "prediction_max_absolute_difference": float(np.max(np.abs(trained - deployed))),
            "strict_checkpoint_load": True, "action_denormalization": "passed",
            "architecture": architecture, "hidden_size": 350, "num_layers": 4,
            "recurrent_input_size": 736, "scope": "offline model and preprocessing compatibility"}


def run_epoch(model, loader, optimizer, device, legacy_batch_mean=False, action_weights=(1.0, 1.0, 1.0), stateful=False):
    training = optimizer is not None
    model.train(training)
    batch_losses = []
    total, count = 0.0, 0
    context = torch.enable_grad if training else torch.no_grad
    
    hx = None
    last_ep = None
    
    with context():
        for batch in loader:
            if stateful:
                x, y, ep_keys = batch
            else:
                x, y = batch
                
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                
            if stateful:
                current_ep = ep_keys[0]
                if last_ep is not None and current_ep != last_ep:
                    hx = None
                last_ep = current_ep
                
                pred, hx = model(x, hx, return_state=True)
                if hx is not None:
                    hx = [h.detach() for h in hx]
            else:
                pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            mse = torch.nn.functional.mse_loss(pred, y)
            if training and tuple(action_weights) != (1.0, 1.0, 1.0):
                weights = pred.new_tensor(action_weights)
                loss = ((pred - y).square() * weights).sum(dim=-1).mean() / weights.sum()
            else:
                loss = mse
            if training:
                loss.backward()
                if not legacy_batch_mean:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            error = float(mse.item())
            batch_losses.append(error)
            total += error * len(x)
            count += len(x)
    if not count:
        return float("nan")
    # The old ROS 1 trainer averaged per-batch scalar losses equally; retain that convention.
    return float(np.mean(batch_losses)) if legacy_batch_mean else total / count



def run_convlstm_epoch(model, loader, optimizer, device, action_weights=(1.0, 1.0, 1.0)):
    """Truncated BPTT, masked per-frame supervision, reset on segment changes."""
    training = optimizer is not None
    model.train(training)
    hx, last_key = None, None
    squared_total, valid_frames = 0.0, 0.0
    weights = torch.tensor(action_weights, dtype=torch.float32, device=device)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for x, y, mask, keys in loader:
            if len(keys) != 1:
                raise ValueError("Stateful ConvLSTM requires batch size one")
            if keys[0] != last_key:
                hx = None
            last_key = keys[0]
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions, hx = model(x, hx, return_state=True, return_sequence=True)
            # Never backpropagate through a previous optimizer step.
            hx = [(h.detach(), c.detach()) for (h, c) in hx]
            per_step = (predictions - y).square()
            error = (per_step.mean(dim=-1) * mask).sum()
            loss = ((per_step * weights).sum(dim=-1) / weights.sum() * mask).sum() / mask.sum().clamp_min(1)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            squared_total += float(error.item())
            valid_frames += float(mask.sum().item())
    if not valid_frames:
        raise RuntimeError("No valid frames in ConvLSTM epoch")
    return squared_total / valid_frames

def save_curve(
    history: list[dict],
    out: Path,
    best_epoch: int | None = None,
    best_val: float | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable; skipped learning curve: {exc}")
        return

    epochs = [r["epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["validation_loss"] for r in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="validation")

    # Mark the epoch whose checkpoint was saved as the best model.
    if best_epoch is not None and best_epoch in epochs:
        best_index = epochs.index(best_epoch)
        marker_val = val_losses[best_index]
        ax.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.2,
            label=f"best model: epoch {best_epoch}",
        )
        ax.scatter([best_epoch], [marker_val], s=55, zorder=5, label="best validation MSE")
        if best_val is not None and np.isfinite(best_val):
            ax.annotate(
                f"best: epoch {best_epoch}\nval MSE={best_val:.7f}",
                xy=(best_epoch, marker_val),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Mapless RNN training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--pair-manifest", type=Path, help="Fixed original pairs and episode partitions")
    parser.add_argument("--episode-plan", type=Path, default=None,
                        help="Frozen episode-level train/validation selections for stateful ConvLSTM campaigns")
    parser.add_argument("--action-weights", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        metavar=("VX", "VY", "WZ"), help="Training loss weights; validation keeps ordinary MSE")
    parser.add_argument("--data-format", choices=("auto", "ros1_legacy", "ros2_episode"), default="auto")
    parser.add_argument("--architecture", choices=("ros1_legacy", "ros2"), default="ros2",
                        help="Old checkpoint layout; new tests use --model-type and ignore this layout.")
    parser.add_argument("--model-type", choices=(*TemporalPolicy.MODEL_TYPES, "convlstm"), default=None,
                        help="Train a new frame-wise CNN+RNN/GRU/LSTM (no ros1_legacy).")
    parser.add_argument("--spatial-hidden-channels", type=int, default=32,
                        help="ConvLSTM spatial hidden channels (default: 32).")
    parser.add_argument("--goal-fusion", choices=("dual", "mixed"), default="dual",
                        help="ConvLSTM goal channels: dual broadcast or one mixed half-map.")
    parser.add_argument("--encoder-channels", nargs=2, type=int, default=(8, 16),
                         metavar=("FIRST", "SECOND"),
                         help="ConvLSTM CNN filter counts. Default 8 16; ablation 32 8.")
    parser.add_argument("--max-frame-gap-seconds", type=float, default=1.0,
                        help="Split temporal segments when ROS timestamps differ by more than this value.")
    parser.add_argument("--train-budget", type=int, default=None,
                        help="Optional exact example budget for independent ROS2 training (pilot only).")
    parser.add_argument("--validation-budget", type=int, default=None,
                        help="Optional exact validation example budget for ROS2 pilot.")
    parser.add_argument("--train-split", type=float, default=None, help="Scenario split fraction for ROS2; sample split fraction for legacy ROS1.")
    parser.add_argument("--validation-a-split", type=float, default=0.25)
    parser.add_argument("--train-scenarios", default=None, help="Comma-separated ROS2 scenario IDs used for training, e.g. E01,E02,E03.")
    parser.add_argument("--validation-scenarios", default=None, help="Optional comma-separated ROS2 scenarios for validation.")
    parser.add_argument("--test-scenarios", default=None, help="ROS2 scenarios reserved and excluded from training/validation.")
    parser.add_argument("--seed", type=int, default=None, help="If omitted, generate a random seed; specify --seed for reproducibility.")
    parser.add_argument("--sequence-length", "--C", dest="sequence_length", type=int, default=None)
    parser.add_argument("--occ-grid-meters", type=int, default=4)
    parser.add_argument(
        "--variant", choices=("4x4", "5x4", "7x6"), default="4x4",
        help="LOM slice: 4x4 (X 0..4, Y -2..2), 5x4 (X -1..4, Y -2..2), "
             "or 7x6 (X -1..6, Y -3..3).",
    )
    parser.add_argument(
        "--representation", choices=("lom", "tlom"), default="lom",
        help="Use the instantaneous LOM or the temporal T-LOM.",
    )
    parser.add_argument("--occupancy-max", type=int, choices=(1, 100), default=1,
                        help="Occupied cell value for training with --pair-manifest.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stateful", action="store_true", help="Keep recurrent state between batches.")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout-1", type=float, default=0.20, help="Dropout after first dense layer.")
    parser.add_argument("--dropout-2", type=float, default=0.15, help="Dropout after second dense layer.")
    parser.add_argument("--vmax", type=float, nargs=3, default=(0.5, 0.5, 1.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--reverse-enable", action="store_true")
    parser.add_argument("--distance-max", type=float, default=None)
    parser.add_argument("--normalize-label", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalize-data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--label-normalization", choices=("legacy", "vx_wz", "none"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--datasets-dir", type=Path, help="Directory containing datasets.json and frozen manifests")
    parser.add_argument("--environment-set", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--catalog-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Check manifest, model and inference compatibility without training")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.datasets_dir:
        if not args.environment_set or not args.dataset or args.data_format not in ("auto", "ros2_episode"):
            parser.error("--datasets-dir requires --environment-set, --dataset and ROS 2 episode data")
        prepared = json.loads((args.datasets_dir.expanduser() / "datasets.json").read_text())
        data_root = (args.data_root or Path(prepared["data_root"])).expanduser().resolve()
    elif args.check_only or args.environment_set or args.dataset or args.catalog_dir:
        parser.error("Manifest options and --check-only require --datasets-dir")
    elif args.data_root is not None:
        data_root = args.data_root.expanduser().resolve()
    else:
        ros2_default = (script_dir / "../../../dataset/data").resolve()
        ros1_default = (script_dir / "Ros1Data" / "Dataset 2025-2").resolve()

        if args.data_format == "ros1_legacy":
            candidates = [ros1_default, ros2_default]
        else:
            candidates = [ros2_default, ros1_default]

        existing = [candidate for candidate in candidates if candidate.is_dir()]
        if not existing:
            parser.error(
                "No default dataset directory found. Pass --data-root to a ROS 1 dataset "
                "or a ROS 2 scenario-data directory."
            )
        data_root = existing[0]
    if not data_root.is_dir():
        parser.error(f"Dataset directory does not exist: {data_root}; pass --data-root explicitly")

    data_format = "ros2_episode" if args.datasets_dir else detect_data_format(data_root) if args.data_format == "auto" else args.data_format
    resolve_defaults(args, data_format)
    if args.model_type is not None:
        if args.architecture != "ros2" or data_format != "ros2_episode":
            parser.error("New temporal models require --architecture ros2 and --data-format ros2_episode")
        if args.datasets_dir:
            parser.error("--datasets-dir reference manifests are only for historical RNN; use scenario selectors or a pair manifest")
        if args.model_type == "convlstm":
            if args.pair_manifest or args.train_budget or args.validation_budget:
                parser.error("ConvLSTM needs full episodes; pair manifests and random example budgets are not supported")
            if args.batch_size != 1:
                parser.error("ConvLSTM stateful training requires --batch-size 1")
            args.stateful = True
        elif args.stateful:
            parser.error("Stateful mode for RNN/GRU/LSTM is not supported in this trainer")
    if args.episode_plan and (args.model_type != "convlstm" or args.pair_manifest or args.datasets_dir
                              or not args.train_scenarios or not args.test_scenarios or args.validation_scenarios):
        parser.error("--episode-plan requires ConvLSTM, explicit train/test scenarios and episode-level validation")
    if args.stateful and args.model_type != "convlstm" and not args.pair_manifest:
        parser.error("--stateful currently requires --pair-manifest with ordered episode entries")
    if (args.train_budget is not None or args.validation_budget is not None) and (args.pair_manifest or args.datasets_dir):
        parser.error("--train-budget/--validation-budget are currently supported only with ROS2 scenario selectors")
    if any(v is not None and v < 1 for v in (args.train_budget, args.validation_budget)):
        parser.error("Example budgets must be positive")
    if args.sequence_length < 1 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("sequence length, batch size, epochs and patience must be positive")
    if args.hidden_size < 1 or args.num_layers < 1 or args.spatial_hidden_channels < 1:
        parser.error("hidden size, spatial hidden channels and number of recurrent layers must be positive")
    if any(v < 1 for v in args.encoder_channels):
        parser.error("--encoder-channels must contain two positive integers")
    if args.model_type != "convlstm" and args.goal_fusion != "dual":
        parser.error("--goal-fusion applies only to ConvLSTM")
    if args.model_type != "convlstm" and tuple(args.encoder_channels) != (8, 16):
        parser.error("--encoder-channels applies only to --model-type convlstm")
    if not np.isfinite(args.max_frame_gap_seconds) or args.max_frame_gap_seconds <= 0:
        parser.error("--max-frame-gap-seconds must be finite and positive")
    if any(not np.isfinite(value) or value <= 0 for value in args.action_weights):
        parser.error("action weights must be finite and positive")
    if args.occupancy_max != 1 and not args.pair_manifest:
        parser.error("--occupancy-max 100 requires --pair-manifest")
    if args.pair_manifest and args.data_root is None:
        parser.error("--pair-manifest requires --data-root")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay cannot be negative")
    if not (0.0 <= args.dropout_1 < 1.0 and 0.0 <= args.dropout_2 < 1.0):
        parser.error("dropout rates must be in [0, 1)")
    if any(float(value) <= 0.0 for value in args.vmax):
        parser.error("all --vmax values must be positive")
    if not 0.0 < args.train_split < 1.0:
        parser.error("--train-split must be between 0 and 1")
    if not 0.0 < args.validation_a_split < 1.0:
        parser.error("--validation-a-split must be between 0 and 1")

    seed = int(args.seed) if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    if not 0 <= seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    print(f"Seed: {seed} ({'provided' if args.seed is not None else 'randomly generated'})")
    seed_everything(seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (EXPERIMENTS_ROOT / f"unified_{data_format}_{stamp}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    if args.pair_manifest:
        train_loader, val_loader, split_info = make_pair_manifest_loaders(args, data_root, seed)
        default_checkpoint = "best_model.pth"
    elif args.datasets_dir:
        train_loader, val_loader, split_info, data_root = make_manifest_loaders(args, seed)
        default_checkpoint = "best_model.pth"
    elif data_format == "ros1_legacy":
        train_loader, val_loader, split_info = make_ros1_loaders(args, data_root, seed)
        default_checkpoint = "best_model.pth"
    else:
        train_loader, val_loader, split_info = make_ros2_loaders(args, data_root, seed)
        default_checkpoint = "best_model.pth"
    checkpoint_path = out_dir / (args.checkpoint_name or default_checkpoint)

    model = make_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, min(5, args.patience)))
    history = []
    best_gap = float("inf")
    interrupted = False
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    config = {
        "format_version": 2,
        "architecture": "temporal_" + args.model_type if args.model_type is not None else args.architecture,
        "model_family": ("spatiotemporal_v1" if args.model_type == "convlstm"
                         else "temporal_v1" if args.model_type is not None else "historical"),
        "model_type": args.model_type,
        "temporal_mode": ("stateful" if args.model_type == "convlstm" else
                          "independent" if args.model_type is not None else
                          ("stateful" if args.stateful else "independent")),
        "spatial_hidden_channels": args.spatial_hidden_channels if args.model_type == "convlstm" else None,
        "encoder_channels": list(args.encoder_channels) if args.model_type == "convlstm" else None,
        "target_fusion": ("spatial_mixed_half_map" if args.goal_fusion == "mixed" else "spatial_before_convlstm")
                         if args.model_type == "convlstm" else None,
        "goal_fusion": args.goal_fusion if args.model_type == "convlstm" else None,
        "frame_supervision": "every_valid_step" if args.model_type == "convlstm" else "last_step",
        "max_frame_gap_seconds": args.max_frame_gap_seconds if args.model_type == "convlstm" else None,
        "data_format": data_format,
        "variant": args.variant,
        "representation": args.representation,
        "img_h": VARIANT_SHAPES[args.variant][0],
        "img_w": VARIANT_SHAPES[args.variant][1],
        "sequence_length": args.sequence_length,
        "stack_n_channels": args.sequence_length if data_format == "ros1_legacy" else None,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "vmax": list(args.vmax),
        "reverse_enable": args.reverse_enable,
        "distance_max": args.distance_max,
        "normalize_data": args.normalize_data if data_format == "ros1_legacy" else None,
        "normalize_occupancy": bool(args.normalize_data) if data_format == "ros1_legacy" else True,
        "label_normalization": args.label_normalization if data_format == "ros1_legacy" else "vmax",
        "normalize_label": bool(args.normalize_label) if data_format == "ros1_legacy" else True,
        "dropout_1": args.dropout_1,
        "dropout_2": args.dropout_2,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "learning_rate_initial": args.learning_rate,
        "weight_decay": args.weight_decay,
        "action_weights": list(args.action_weights),
        "train_split": args.train_split,
        "patience": args.patience,
        "seed": seed,
        "device": str(device),
        "data_root": data_root.name,
        "split": split_info,
    }
    if args.model_type == "convlstm":
        config["require_full_sequence"] = False  # One observation per online update.
    elif split_info.get("require_full_sequence") or args.model_type is not None:
        config["require_full_sequence"] = True
    if args.pair_manifest and args.occupancy_max == 100:
        config.update(normalize_occupancy=False, occupancy_max=100.0)
    config["campaign_checkpoint_policy"] = "min_validation_and_min_absolute_gap"
    if args.datasets_dir:
        from experiment_control import file_hash
        config.update(format_version=3, preprocessing=split_info["preprocessing"],
                      run_mode="check_only" if args.check_only else "training",
                      normalize_data=False, normalize_occupancy=False, occupancy_max=100.0,
                      label_normalization="legacy_training_pool_minmax", normalize_label=True, stack_n_channels=2,
                      versions={"torch": str(torch.__version__), "numpy": np.__version__,
                                "python": sys.version, "cuda": torch.version.cuda},
                      source_sha256={p.name: file_hash(p) for p in [Path(__file__), SCRIPT_DIR / "models.py",
                                    SCRIPT_DIR / "experiment_control.py", SCRIPT_DIR.parent / "RNN_mapless.py"]})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Data format: {data_format} | architecture: {config['architecture']}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Dataset: {data_root}")
    print(f"Train batches: {len(train_loader)} | validation batches: {len(val_loader)}")
    print(f"Device: {device} | output: {out_dir}")

    if args.datasets_dir:
        preflight = check_inference_contract(model, config, train_loader.dataset, device)
        (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
        print("Preflight passed: model loading, input orientation, raw LOM scale and action denormalization.", flush=True)
        if args.check_only:
            print(f"Checks: {out_dir / 'preflight.json'}")
            return 0
        seed_everything(seed)

    started = time.monotonic()
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.monotonic()
            if args.model_type == "convlstm":
                train_loss = run_convlstm_epoch(model, train_loader, optimizer, device,
                                               action_weights=args.action_weights)
                val_loss = run_convlstm_epoch(model, val_loader, None, device)
            else:
                train_loss = run_epoch(model, train_loader, optimizer, device,
                                       legacy_batch_mean=(data_format == "ros1_legacy" or bool(args.datasets_dir)),
                                       action_weights=args.action_weights, stateful=args.stateful)
                val_loss = run_epoch(model, val_loader, None, device,
                                     legacy_batch_mean=(data_format == "ros1_legacy" or bool(args.datasets_dir)),
                                     stateful=args.stateful)
            scheduler.step(val_loss)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.monotonic() - epoch_start,
            }
            history.append(row)
            print(f"epoch {epoch:03d}/{args.epochs} train={train_loss:.7f} val={val_loss:.7f} lr={row['learning_rate']:.3g} t={row['epoch_seconds']:.2f}s")
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise ValueError("Training produced a nonfinite loss")
            gap = abs(val_loss - train_loss)
            row["absolute_generalization_gap"] = gap
            if gap < best_gap:
                best_gap = gap
                torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "train_loss": train_loss,
                    "validation_loss": val_loss, "absolute_generalization_gap": gap,
                    "selection_criterion": "minimum_absolute_train_validation_gap",
                    "architecture": config["architecture"], "data_format": data_format,
                    "config": config}, out_dir / "best_gap.pth")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            if np.isfinite(val_loss) and val_loss < best_val:
                previous_best = best_val
                best_val, best_epoch, stale_epochs = val_loss, epoch, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "architecture": config["architecture"],
                    "data_format": data_format,
                    "config": config,
                }, checkpoint_path)
                if np.isfinite(previous_best):
                    print(
                        f"  -> Saving best model at epoch {epoch}: "
                        f"validation MSE={val_loss:.7f} "
                        f"(previous best={previous_best:.7f})"
                    )
                else:
                    print(
                        f"  -> Saving best model at epoch {epoch}: "
                        f"validation MSE={val_loss:.7f} (initial best)"
                    )
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted; artifacts from completed epochs will be preserved.")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_curve(history, out_dir / "learning_curves.png", best_epoch=best_epoch, best_val=best_val)
    print(f"Best epoch: {best_epoch}; best validation MSE: {best_val:.7f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Elapsed: {time.monotonic() - started:.1f} s")
    if args.pair_manifest and best_epoch > 0 and not interrupted:
        audits = {}
        for path in (checkpoint_path, out_dir / "best_gap.pth"):
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            audits[path.name] = {"epoch": checkpoint["epoch"],
                                 "by_class": audit_pair_actions(model, val_loader.dataset, device, args.vmax)}
        (out_dir / "action_metrics.json").write_text(json.dumps(audits, indent=2, allow_nan=False) + "\n")
    if args.datasets_dir and best_epoch > 0:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checked = check_inference_contract(model, checkpoint["config"], train_loader.dataset, device)
        checked.update(checkpoint=checkpoint_path.name, epoch=checkpoint["epoch"])
        (out_dir / "checkpoint_check.json").write_text(json.dumps(checked, indent=2))
        print(f"Checkpoint reload passed: {out_dir / 'checkpoint_check.json'}")
    if args.datasets_dir and best_epoch > 0:
        checks = {}
        for saved in (checkpoint_path, out_dir / "best_gap.pth"):
            try:
                item = torch.load(saved, map_location=device, weights_only=False)
            except TypeError:
                item = torch.load(saved, map_location=device)
            model.load_state_dict(item["model_state_dict"], strict=True)
            checked = check_inference_contract(model, item["config"], train_loader.dataset, device)
            checked.update(checkpoint=saved.name, epoch=item["epoch"])
            checks[saved.name] = checked
        (out_dir / "checkpoint_checks.json").write_text(json.dumps(checks, indent=2))
    return 130 if interrupted else 0 if best_epoch > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
