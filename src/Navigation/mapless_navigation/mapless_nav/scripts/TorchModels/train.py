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
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_utils import (
    SequenceDataset,
    build_legacy_tensors,
    discover_npz_files,
    load_episodes,
    load_legacy_flat_dataset,
    load_samples_from_npz,
    split_scenarios,
    split_validation_a,
)
from models import RNN


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
    args.sequence_length = args.sequence_length if args.sequence_length is not None else 2
    args.hidden_size = args.hidden_size if args.hidden_size is not None else 350
    args.num_layers = args.num_layers if args.num_layers is not None else 4
    args.batch_size = args.batch_size if args.batch_size is not None else 3
    args.epochs = args.epochs if args.epochs is not None else 20
    args.learning_rate = args.learning_rate if args.learning_rate is not None else 0.0006
    args.train_split = args.train_split if args.train_split is not None else 0.8
    args.patience = args.patience if args.patience is not None else 5
    args.normalize_label = args.normalize_label if args.normalize_label is not None else True
    args.normalize_data = args.normalize_data if args.normalize_data is not None else False
    args.label_normalization = args.label_normalization or "legacy"
    args.vmax = tuple(args.vmax) if args.vmax else (0.5, 0.5, 1.0)


def make_model(args):
    return RNN(
        channels=args.sequence_length,
        img_size=80,
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


def make_ros2_loaders(args, root: Path, seed: int):
    scenario_files = discover_npz_files(str(root))
    if not scenario_files:
        raise RuntimeError(
            f"No ROS 2 scenario folders found under {root}; expected data/E01/*.npz, data/E02/*.npz, ..."
        )
    all_scenarios = sorted(scenario_files)
    all_episodes = load_episodes(scenario_files)
    if not all_episodes:
        raise RuntimeError("No ROS 2 episodes found in NPZ files.")

    train_names = parse_csv_names(args.train_scenarios)
    val_names = parse_csv_names(args.validation_scenarios)
    test_names = parse_csv_names(args.test_scenarios) or []
    known = set(all_scenarios)
    for label, names in (("train", train_names), ("validation", val_names), ("test", test_names)):
        if names is not None:
            unknown = sorted(set(names) - known)
            if unknown:
                raise ValueError(f"Unknown {label} scenarios: {unknown}; available: {all_scenarios}")

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
    )
    val_dataset = SequenceDataset(
        val_episodes,
        sequence_length=args.sequence_length,
        vmax=args.vmax,
        reverse_enable=args.reverse_enable,
        distance_max=args.distance_max,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("ROS 2 train dataset contains zero sequences; check sequence length and train scenarios.")
    if len(val_dataset) == 0:
        raise RuntimeError(
            "ROS 2 validation dataset contains zero sequences. Supply --validation-scenarios or "
            "ensure each train scenario has enough episodes for Validation A."
        )
    use_pin = torch.cuda.is_available()
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_pin),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin),
        {
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
        actual = np.stack([module.prepare_observation(np.rot90(grid, -1), target[0], target[1], settings)
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


def run_epoch(model, loader, optimizer, device, legacy_batch_mean=False):
    training = optimizer is not None
    model.train(training)
    batch_losses = []
    total, count = 0.0, 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            if pred.ndim == 1:
                pred = pred.unsqueeze(0)
            loss = torch.nn.functional.mse_loss(pred, y)
            if training:
                loss.backward()
                if not legacy_batch_mean:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_losses.append(float(loss.item()))
            total += float(loss.item()) * len(x)
            count += len(x)
    if not count:
        return float("nan")
    # The old ROS 1 trainer averaged per-batch scalar losses equally; retain that convention.
    return float(np.mean(batch_losses)) if legacy_batch_mean else total / count


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
    parser.add_argument("--data-format", choices=("auto", "ros1_legacy", "ros2_episode"), default="auto")
    parser.add_argument("--architecture", choices=("ros1_legacy", "ros2"), default="ros1_legacy", help="Model layout; defaults to the historical ROS 1 layout for both data formats.")
    parser.add_argument("--train-split", type=float, default=None, help="Scenario split fraction for ROS2; sample split fraction for legacy ROS1.")
    parser.add_argument("--validation-a-split", type=float, default=0.25)
    parser.add_argument("--train-scenarios", default=None, help="Comma-separated ROS2 scenario IDs used for training, e.g. E01,E02,E03.")
    parser.add_argument("--validation-scenarios", default=None, help="Optional comma-separated ROS2 scenarios for validation.")
    parser.add_argument("--test-scenarios", default=None, help="ROS2 scenarios reserved and excluded from training/validation.")
    parser.add_argument("--seed", type=int, default=None, help="If omitted, generate a random seed; specify --seed for reproducibility.")
    parser.add_argument("--sequence-length", "--C", dest="sequence_length", type=int, default=None)
    parser.add_argument("--occ-grid-meters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
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
    if args.sequence_length < 1 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("sequence length, batch size, epochs and patience must be positive")
    if args.hidden_size < 1 or args.num_layers < 1:
        parser.error("hidden size and number of recurrent layers must be positive")
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
    out_dir = (args.output_dir.expanduser().resolve() if args.output_dir else (script_dir / "experiments" / f"unified_{data_format}_{stamp}").resolve())
    out_dir.mkdir(parents=True, exist_ok=False)

    if args.datasets_dir:
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
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    config = {
        "format_version": 2,
        "architecture": args.architecture,
        "data_format": data_format,
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
        "train_split": args.train_split,
        "patience": args.patience,
        "seed": seed,
        "device": str(device),
        "data_root": data_root.name,
        "split": split_info,
    }
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
    print(f"Data format: {data_format} | architecture: {args.architecture}")
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
            train_loss = run_epoch(model, train_loader, optimizer, device, legacy_batch_mean=(data_format == "ros1_legacy" or bool(args.datasets_dir)))
            val_loss = run_epoch(model, val_loader, None, device, legacy_batch_mean=(data_format == "ros1_legacy" or bool(args.datasets_dir)))
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
            if np.isfinite(val_loss) and val_loss < best_val:
                previous_best = best_val
                best_val, best_epoch, stale_epochs = val_loss, epoch, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "architecture": args.architecture,
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
        print("\nTraining interrupted; artifacts from completed epochs will be preserved.")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_curve(history, out_dir / "learning_curves.png", best_epoch=best_epoch, best_val=best_val)
    print(f"Best epoch: {best_epoch}; best validation MSE: {best_val:.7f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Elapsed: {time.monotonic() - started:.1f} s")
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
    return 0 if best_epoch > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
