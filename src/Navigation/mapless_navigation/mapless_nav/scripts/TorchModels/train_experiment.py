#!/usr/bin/env python3
"""Train one frozen map-level experiment (A or B) with the current TorchModels RNN.

Only configured TRAIN scenario NPZ files are deserialized. Internal validation
holds out complete episodes from TRAIN scenarios. TEST scenarios are never
used for optimization, validation, model selection, normalization fitting, or
checkpoint selection. Their maps are intended for a later Webots evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import get_experiment, load_config  # noqa: E402
from utils.dataset_utils import (  # noqa: E402
    SequenceDataset,
    discover_npz_files,
    load_episodes,
    split_validation_a,
)
from utils.models import RNN  # noqa: E402


CLASS_NAMES = ("straight", "semi_complex", "complex", "v_complex")
METRIC_NAMES = ("vx", "vy", "wz")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def episode_key(episode: Any) -> tuple[str, str, int]:
    """Use run path as well as episode ID because IDs can restart per run."""
    return (episode.scenario, str(Path(episode.run_path).resolve()), int(episode.episode_id))


def load_episode_classes(episodes: list[Any]) -> dict[tuple[str, str, int], str]:
    """Associate each loaded NPZ episode to its sidecar class label."""
    sidecars: dict[str, dict[int, str]] = {}
    mapping: dict[tuple[str, str, int], str] = {}

    for episode in episodes:
        run_path = str(Path(episode.run_path).resolve())
        raw_run_dir = Path(run_path).with_suffix("")
        sidecar_path = raw_run_dir / "episode_tries.json"
        if run_path not in sidecars:
            if not sidecar_path.is_file():
                raise RuntimeError(
                    f"Missing episode class sidecar for {run_path}: {sidecar_path}"
                )
            try:
                parsed = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Cannot parse {sidecar_path}: {exc}") from exc

            run_classes: dict[int, str] = {}
            duplicates: set[int] = set()
            for item in parsed.get("entries", []):
                try:
                    eid = int(item["episode_id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid episode ID in sidecar {sidecar_path}"
                    ) from exc
                if eid in run_classes:
                    duplicates.add(eid)
                run_classes[eid] = str(item.get("class", "unknown"))
            if duplicates:
                raise RuntimeError(
                    f"Duplicate episode IDs in {sidecar_path}: {sorted(duplicates)[:10]}"
                )
            sidecars[run_path] = run_classes

        classes = sidecars[run_path]
        key = episode_key(episode)
        label = classes.get(int(episode.episode_id))
        if label not in CLASS_NAMES:
            raise RuntimeError(
                f"Episode has missing/unknown class label: {key}, class={label!r}"
            )
        mapping[key] = label

    return mapping


def validate_loaded_episodes(episodes: list[Any]) -> dict[str, Any]:
    """Fail closed on invalid LOM, targets, velocity labels, or episode metadata."""
    if not episodes:
        raise RuntimeError("No episodes were loaded from configured TRAIN scenarios")

    sample_count = 0
    short_episodes = 0
    for episode in episodes:
        if not episode.samples:
            raise RuntimeError(f"Empty episode: {episode_key(episode)}")
        if len(episode.samples) < 2:
            short_episodes += 1
        last_timestamp = None
        for index, sample in enumerate(episode.samples):
            try:
                occ = np.asarray(sample["features"]["occ_grid"])
                target = np.asarray(sample["features"]["target"], dtype=np.float64)
                label = np.asarray(sample["labels"], dtype=np.float64)
                meta = sample["meta"]
                sample_eid = int(meta["episode_id"])
                timestamp = float(meta["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid sample schema in {episode_key(episode)}, sample {index}: {exc}"
                ) from exc

            if occ.shape != (80, 80):
                raise RuntimeError(
                    f"Expected LOM shape (80, 80), got {occ.shape} in {episode_key(episode)}"
                )
            if target.shape != (2,) or not np.isfinite(target).all():
                raise RuntimeError(
                    f"Invalid target (expected 2 finite values) in {episode_key(episode)}"
                )
            if label.shape != (3,) or not np.isfinite(label).all():
                raise RuntimeError(
                    f"Invalid velocity label (expected [vx, vy, wz]) in {episode_key(episode)}"
                )
            if sample_eid != int(episode.episode_id):
                raise RuntimeError(
                    f"Sample episode_id={sample_eid} does not match group "
                    f"{episode_key(episode)}"
                )
            if not np.isfinite(timestamp):
                raise RuntimeError(f"Non-finite timestamp in {episode_key(episode)}")
            if last_timestamp is not None and timestamp < last_timestamp:
                raise RuntimeError(f"Non-monotonic timestamps in {episode_key(episode)}")
            last_timestamp = timestamp
            sample_count += 1

    return {
        "episodes": len(episodes),
        "samples": sample_count,
        "episodes_with_fewer_than_two_samples": short_episodes,
    }


class ClassAwareSequences(Dataset):
    """Decorate SequenceDataset with the episode class for stratified metrics."""

    def __init__(self, base: SequenceDataset, class_map: dict[tuple[str, str, int], str]):
        self.base = base
        self.class_ids: list[int] = []
        for meta in base.sequence_meta:
            key = (
                str(meta["scenario"]),
                str(Path(meta["run_path"]).resolve()),
                int(meta["episode_id"]),
            )
            label = class_map.get(key)
            if label not in CLASS_NAMES:
                raise RuntimeError(f"Sequence refers to an episode without a valid class: {key}")
            self.class_ids.append(CLASS_NAMES.index(label))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, y = self.base[index]
        return x, y, self.class_ids[index]


def _metrics_from_errors(
    loss_sum: float,
    sample_count: int,
    absolute_errors: np.ndarray,
    squared_errors: np.ndarray,
    class_counts: np.ndarray,
    class_absolute: np.ndarray,
    class_squared: np.ndarray,
) -> dict[str, Any]:
    if sample_count == 0:
        return {
            "loss": None, "sequences": 0, "mae_normalized": None,
            "rmse_normalized": None, "mae_velocity_units": None,
            "rmse_velocity_units": None, "by_class": {},
        }

    mae = (absolute_errors / sample_count).tolist()
    rmse = np.sqrt(squared_errors / sample_count).tolist()
    by_class = {}
    for index, name in enumerate(CLASS_NAMES):
        n = int(class_counts[index])
        by_class[name] = {
            "sequences": n,
            "mae_normalized": (class_absolute[index] / n).tolist() if n else None,
            "rmse_normalized": np.sqrt(class_squared[index] / n).tolist() if n else None,
        }
    return {
        "loss": loss_sum / sample_count,
        "sequences": sample_count,
        "mae_normalized": mae,
        "rmse_normalized": rmse,
        "by_class": by_class,
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    abs_error = np.zeros(3, dtype=np.float64)
    squared_error = np.zeros(3, dtype=np.float64)
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    class_abs = np.zeros((len(CLASS_NAMES), 3), dtype=np.float64)
    class_squared = np.zeros((len(CLASS_NAMES), 3), dtype=np.float64)

    for x, y, class_ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model(x)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            loss = torch.nn.functional.mse_loss(prediction, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        error = (prediction.detach() - y).to(torch.float64)
        abs_np = torch.abs(error).cpu().numpy()
        sq_np = (error * error).cpu().numpy()
        class_np = class_ids.cpu().numpy().astype(np.int64)
        n = int(x.shape[0])
        loss_sum += float(loss.item()) * n
        sample_count += n
        abs_error += abs_np.sum(axis=0)
        squared_error += sq_np.sum(axis=0)
        for class_id in range(len(CLASS_NAMES)):
            mask = class_np == class_id
            if np.any(mask):
                class_counts[class_id] += int(mask.sum())
                class_abs[class_id] += abs_np[mask].sum(axis=0)
                class_squared[class_id] += sq_np[mask].sum(axis=0)

    return _metrics_from_errors(
        loss_sum, sample_count, abs_error, squared_error,
        class_counts, class_abs, class_squared,
    )


def scale_metrics(metrics: dict[str, Any], vmax: tuple[float, float, float]) -> dict[str, Any]:
    """Add physical-unit MAE/RMSE by linear inverse scaling from normalized labels."""
    if metrics.get("mae_normalized") is None:
        metrics["mae_velocity_units"] = None
        metrics["rmse_velocity_units"] = None
        return metrics
    metrics["mae_velocity_units"] = {
        name: float(metrics["mae_normalized"][i] * vmax[i])
        for i, name in enumerate(METRIC_NAMES)
    }
    metrics["rmse_velocity_units"] = {
        name: float(metrics["rmse_normalized"][i] * vmax[i])
        for i, name in enumerate(METRIC_NAMES)
    }
    for data in metrics["by_class"].values():
        if data["mae_normalized"] is None:
            data["mae_velocity_units"] = None
            data["rmse_velocity_units"] = None
        else:
            data["mae_velocity_units"] = {
                name: float(data["mae_normalized"][i] * vmax[i])
                for i, name in enumerate(METRIC_NAMES)
            }
            data["rmse_velocity_units"] = {
                name: float(data["rmse_normalized"][i] * vmax[i])
                for i, name in enumerate(METRIC_NAMES)
            }
    return metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_counts_by_class(dataset: ClassAwareSequences) -> dict[str, int]:
    counts = Counter(CLASS_NAMES[class_id] for class_id in dataset.class_ids)
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES}


def save_learning_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train"]["loss"] for row in history]
    validation_losses = [row["validation"]["loss"] for row in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.plot(epochs, validation_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized labels)")
    plt.title("Mapless navigation - frozen experiment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("A", "B"))
    parser.add_argument("--experiment-config", type=Path, default=None)
    parser.add_argument(
        "--data-root", type=Path,
        default=SCRIPT_DIR / "../../../dataset/data",
        help="ROS 2 dataset root; defaults to mapless_navigation/dataset/data.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-split", type=float, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--vmax", type=str, default=None, help="Comma-separated vx,vy,wz velocity limits.")
    parser.add_argument("--reverse-enable", action="store_true", default=None)
    parser.add_argument("--distance-max", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.experiment_config)
    experiment = get_experiment(config, args.experiment)
    baseline = config["baseline_training"]
    experiment_id = experiment["id"]

    seed = int(args.seed if args.seed is not None else baseline.get("seed", config["validation"]["seed"]))
    validation_fraction = float(
        args.validation_split if args.validation_split is not None
        else config["validation"]["fraction"]
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation split must be strictly between 0 and 1")

    sequence_length = int(args.sequence_length if args.sequence_length is not None else baseline["sequence_length"])
    batch_size = int(args.batch_size if args.batch_size is not None else baseline["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else baseline["epochs"])
    learning_rate = float(args.learning_rate if args.learning_rate is not None else baseline["learning_rate"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else baseline.get("weight_decay", 0.0))
    hidden_size = int(args.hidden_size if args.hidden_size is not None else baseline["hidden_size"])
    num_layers = int(args.num_layers if args.num_layers is not None else baseline["num_layers"])
    reverse_enable = bool(args.reverse_enable if args.reverse_enable is not None else baseline["reverse_enable"])
    distance_max = args.distance_max if args.distance_max is not None else baseline.get("distance_max")

    if args.vmax is None:
        vmax = tuple(float(v) for v in baseline["vmax"])
    else:
        vmax = tuple(float(v.strip()) for v in args.vmax.split(","))
        if len(vmax) != 3 or any(v <= 0 for v in vmax):
            raise ValueError("--vmax must contain three positive comma-separated values")

    if sequence_length < 1 or batch_size < 1 or epochs < 1 or learning_rate <= 0:
        raise ValueError("sequence-length, batch-size and epochs must be positive; learning-rate must be > 0")

    set_seed(seed)
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {data_root}")

    train_scenarios = list(experiment["train_scenarios"])
    test_scenarios = list(experiment["test_scenarios"])
    expected_scenario_dirs = sorted(set(train_scenarios + test_scenarios))
    missing_dirs = [s for s in expected_scenario_dirs if not (data_root / s).is_dir()]
    if missing_dirs:
        raise RuntimeError(f"Configured experiment scenario directories are missing: {missing_dirs}")

    # Discovery reads directory listings only. The subsequent loader is given
    # TRAIN paths exclusively; TEST file contents are never deserialized.
    discovered = discover_npz_files(str(data_root))
    missing_train_npz = [s for s in train_scenarios if s not in discovered or not discovered[s]]
    if missing_train_npz:
        raise RuntimeError(
            f"Experiment {args.experiment} is blocked: no converted NPZ files for TRAIN "
            f"scenarios {missing_train_npz}. Convert the existing recordings first; "
            "this script does not generate or alter dataset files."
        )

    train_file_map = {s: discovered[s] for s in train_scenarios}
    episodes = load_episodes(train_file_map)
    if not episodes:
        raise RuntimeError("No TRAIN episodes were decoded from the configured NPZ files")
    loaded_scenarios = {ep.scenario for ep in episodes}
    missing_loaded_scenarios = sorted(set(train_scenarios) - loaded_scenarios)
    if missing_loaded_scenarios:
        raise RuntimeError(
            f"Configured TRAIN scenarios contain no valid episodes: {missing_loaded_scenarios}"
        )
    if loaded_scenarios & set(test_scenarios):
        raise RuntimeError("Leakage guard: TEST scenario appeared among decoded TRAIN episodes")

    integrity = validate_loaded_episodes(episodes)
    class_map = load_episode_classes(episodes)
    episode_classes = Counter(class_map.values())
    missing_classes = sorted(set(CLASS_NAMES) - set(episode_classes))
    if missing_classes:
        raise RuntimeError(
            f"TRAIN NPZ episodes are missing required navigation classes: {missing_classes}"
        )

    train_episodes, validation_episodes = split_validation_a(
        episodes, validation_a_split=validation_fraction, seed=seed
    )
    if not train_episodes or not validation_episodes:
        raise RuntimeError("Internal episode-level split produced an empty TRAIN or VALIDATION group")
    train_keys = {episode_key(ep) for ep in train_episodes}
    validation_keys = {episode_key(ep) for ep in validation_episodes}
    if train_keys & validation_keys:
        raise RuntimeError("Leakage guard: episode appears in both TRAIN and VALIDATION")
    if any(ep.scenario not in set(train_scenarios) for ep in (*train_episodes, *validation_episodes)):
        raise RuntimeError("Leakage guard: non-TRAIN map entered TRAIN or VALIDATION")

    train_base = SequenceDataset(
        train_episodes,
        sequence_length=sequence_length,
        vmax=vmax,
        reverse_enable=reverse_enable,
        distance_max=distance_max,
    )
    validation_base = SequenceDataset(
        validation_episodes,
        sequence_length=sequence_length,
        vmax=vmax,
        reverse_enable=reverse_enable,
        distance_max=distance_max,
    )
    train_dataset = ClassAwareSequences(train_base, class_map)
    validation_dataset = ClassAwareSequences(validation_base, class_map)
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise RuntimeError(
            "No effective temporal sequences in TRAIN or VALIDATION; "
            "check episode lengths and sequence_length."
        )
    train_sequence_classes = sequence_counts_by_class(train_dataset)
    validation_sequence_classes = sequence_counts_by_class(validation_dataset)
    absent_train_classes = [c for c, n in train_sequence_classes.items() if n == 0]
    if absent_train_classes:
        raise RuntimeError(
            f"No temporal training sequences for required classes: {absent_train_classes}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()))
        else "cpu"
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (SCRIPT_DIR / "experiments" / experiment_id / f"seed_{seed}").resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment output {output_dir}. "
            "Choose a different --output-dir or seed."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_npz_paths = sorted(Path(p).resolve() for paths in train_file_map.values() for p in paths)
    source_file_manifest = [
        {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in selected_npz_paths
    ]
    output_config = {
        "experiment": args.experiment,
        "experiment_id": experiment_id,
        "seed": seed,
        "data_root": str(data_root),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "test_scenarios_loaded": False,
        "test_npz_content_opened": False,
        "internal_validation": {
            "unit": "episode",
            "fraction": validation_fraction,
            "seed": seed,
        },
        "model_class": "RNN",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vmax": list(vmax),
        "reverse_enable": reverse_enable,
        "distance_max": distance_max,
        "device": str(device),
        "normalization": baseline["normalization"],
        "source_npz_files": source_file_manifest,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "decoded_episode_count": integrity["episodes"],
        "decoded_sample_count": integrity["samples"],
        "train_sequence_count": len(train_dataset),
        "validation_sequence_count": len(validation_dataset),
        "train_episodes_by_class": {
            c: sum(class_map[episode_key(ep)] == c for ep in train_episodes)
            for c in CLASS_NAMES
        },
        "validation_episodes_by_class": {
            c: sum(class_map[episode_key(ep)] == c for ep in validation_episodes)
            for c in CLASS_NAMES
        },
        "train_sequences_by_class": train_sequence_classes,
        "validation_sequences_by_class": validation_sequence_classes,
        "split_config_path": config.get("_config_path"),
        "trainer_notes": (
            "Current TorchModels model/preprocessing used. Historical ROS1 artifacts are reference only. "
            "No TEST scenario NPZ samples are loaded or used for model selection."
        ),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(output_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = RNN(
        channels=sequence_length, img_size=80,
        hidden_size=hidden_size, num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"Experiment: {args.experiment} ({experiment_id})")
    print(f"TRAIN scenarios ({len(train_scenarios)}): {' '.join(train_scenarios)}")
    print(f"TEST scenarios ({len(test_scenarios)}): {' '.join(test_scenarios)}")
    print(f"TRAIN/VALIDATION episodes: {len(train_episodes)}/{len(validation_episodes)}")
    print(f"TRAIN/VALIDATION sequences: {len(train_dataset)}/{len(validation_dataset)}")
    print(f"TRAIN sequence class counts: {train_sequence_classes}")
    print(f"VALIDATION sequence class counts: {validation_sequence_classes}")
    print(f"Device: {device}; source NPZ files fingerprinted: {len(source_file_manifest)}")
    print("TEST scenarios are excluded from training and internal validation.")

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    consecutive_increases = 0
    patience = int(baseline.get("early_stopping_patience", 5))
    checkpoint_path = output_dir / "best_model.pth"
    start_total = time.time()

    for epoch in range(1, epochs + 1):
        start_epoch = time.time()
        train_metrics = scale_metrics(run_epoch(model, train_loader, optimizer, device), vmax)
        validation_metrics = scale_metrics(run_epoch(model, validation_loader, None, device), vmax)
        if validation_metrics["loss"] is None or not np.isfinite(validation_metrics["loss"]):
            raise RuntimeError("Validation loss is not finite; stopping without declaring a valid run")

        scheduler.step(validation_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": current_lr,
            "epoch_time_sec": time.time() - start_epoch,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} | train_MSE={train_metrics['loss']:.7f} | "
            f"val_MSE={validation_metrics['loss']:.7f} | "
            f"val_MAE(vx,vy,wz)={validation_metrics['mae_velocity_units']} | "
            f"lr={current_lr:.3g}"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    "config": {
                        "sequence_length": sequence_length,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "vmax": vmax,
                        "reverse_enable": reverse_enable,
                        "distance_max": distance_max,
                        "experiment": args.experiment,
                        "experiment_id": experiment_id,
                        "seed": seed,
                    },
                    "experiment_manifest": str(output_dir / "experiment_manifest.json"),
                },
                checkpoint_path,
            )

        if len(history) > 1 and validation_metrics["loss"] > history[-2]["validation"]["loss"]:
            consecutive_increases += 1
        else:
            consecutive_increases = 0
        if consecutive_increases >= patience:
            print(f"Early stopping at epoch {epoch}: validation loss increased {patience} consecutive times.")
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    save_learning_curves(history, output_dir / "learning_curves.png")
    summary = {
        "experiment": args.experiment,
        "experiment_id": experiment_id,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_mse_normalized": best_validation_loss,
        "checkpoint": str(checkpoint_path),
        "history": str(output_dir / "history.json"),
        "manifest": str(output_dir / "experiment_manifest.json"),
        "total_seconds": time.time() - start_total,
        "test_evaluation": "not run; use the held-out maps in Webots",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
