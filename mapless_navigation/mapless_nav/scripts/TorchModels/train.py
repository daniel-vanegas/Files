#!/usr/bin/env python3

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.dataset_utils import (
    SequenceDataset,
    discover_npz_files,
    load_episodes,
)


def set_seed(seed):
    # Set seed for random generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_vmax(value):
    # Split string limits into numbers
    values = [float(v.strip()) for v in value.split(",")]

    if len(values) != 3:
        raise argparse.ArgumentTypeError("vmax must be vx,vy,wz")

    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("all vmax values must be > 0")

    return tuple(values)


def load_model_class(model_class_name):
    # Import model class from utils/models.py
    this_dir = os.path.dirname(os.path.abspath(__file__))

    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    try:
        module = importlib.import_module("utils.models")
    except ImportError as exc:
        raise RuntimeError(f"Could not import utils/models.py from {this_dir}.") from exc

    if not hasattr(module, model_class_name):
        raise RuntimeError(f"utils/models.py does not contain model class '{model_class_name}'.")

    return getattr(module, model_class_name)


def create_model(model_class_name, sequence_length, hidden_size, num_layers, dropout):
    # Create model instance
    model_class = load_model_class(model_class_name)

    try:
        model = model_class(
            channels=sequence_length,
            img_size=80,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
    except TypeError as exc:
        raise RuntimeError(
            f"Could not instantiate {model_class_name} with the expected constructor. "
            f"The model must accept channels, img_size, hidden_size and num_layers.\n"
            f"Error: {exc}"
        ) from exc

    return model


def compute_loss(prediction, target):
    # Calculate mean squared error
    return torch.nn.functional.mse_loss(prediction, target)


def run_epoch(model, loader, optimizer, device, training, current_epoch, total_epochs, epoch_start_time):
    # Run one epoch (train or validation)
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    last_update_time = time.time()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = model(x)
            if isinstance(prediction, tuple):
                prediction = prediction[0]

            loss = compute_loss(prediction, y)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Refresh console output every ~1 second during training
        current_time = time.time()
        if training and (current_time - last_update_time >= 1.0):
            elapsed = current_time - epoch_start_time
            current_loss = total_loss / total_samples
            log_line = (
                f"epoch {current_epoch}/{total_epochs} | "
                f"train={current_loss:.7f} | "
                f"val=... | "
                f"t[s]={elapsed:.2f}"
            )
            sys.stdout.write(f"\r{log_line.ljust(80)}")
            sys.stdout.flush()
            last_update_time = current_time

    if total_samples == 0:
        return float("nan")

    return total_loss / total_samples


def save_learning_curves(history, output_path):
    # Save loss plot image
    if not history:
        print("WARNING: No training history available. Skipping learning curves.")
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    validation_losses = [row["validation_loss"] for row in history]

    best_index = int(np.nanargmin(validation_losses))
    best_epoch = epochs[best_index]
    best_val_loss = validation_losses[best_index]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label="Train", color="tab:blue", linewidth=2)
    plt.plot(epochs, validation_losses, label="Validation", color="tab:orange", linewidth=2)

    plt.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Best epoch ({best_epoch})",
    )

    plt.scatter([best_epoch], [best_val_loss], color="tab:orange", zorder=5)

    plt.annotate(
        f"Best Val: {best_val_loss:.6f}",
        xy=(best_epoch, best_val_loss),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Mapless Navigation - Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Learning curves: {output_path}")


def format_time(seconds):
    # Convert seconds to string HH:MM:SS.ss
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"
    return f"{mins:02d}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Train mapless navigation model splitting episodes directly across all scenarios."
    )

    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root path.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.75,
        help="Fraction of all episodes used for training. Default: 0.75",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--sequence-length",
        "--C",
        dest="sequence_length",
        type=int,
        default=4,
        help="Sequence frame count.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model-class", default="RNN", help="Model class name from utils/models.py."
    )
    parser.add_argument(
        "--vmax",
        type=parse_vmax,
        default=(0.5, 0.5, 1.0),
        help="Velocity limits.",
    )
    parser.add_argument(
        "--reverse-enable",
        action="store_true",
        help="Normalize velocity in both directions.",
    )
    parser.add_argument(
        "--distance-max",
        type=float,
        default=None,
        help="Max distance parameter.",
    )
    parser.add_argument("--output-dir", default="experiments")

    args = parser.parse_args()

    # Get seed from argument or generate a random one
    if args.seed is None:
        args.seed = random.randint(0, 100000)
        print(f"No seed specified. Generated random seed: {args.seed}")
    else:
        print(f"Using specified seed: {args.seed}")

    set_seed(args.seed)

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    if args.data_root is None:
        data_root = (script_dir / "../../../dataset/data").resolve()
    else:
        data_root = Path(args.data_root).expanduser().resolve()

    if not data_root.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {data_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for files
    scenario_files = discover_npz_files(str(data_root))
    if not scenario_files:
        raise RuntimeError(f"No .npz files found under {data_root}")

    all_scenarios = sorted(scenario_files.keys())
    all_episodes = load_episodes(scenario_files)

    if not all_episodes:
        raise RuntimeError(f"No episodes found in {data_root}")

    # Shuffle episodes
    random.Random(args.seed).shuffle(all_episodes)

    # Split dataset
    num_total = len(all_episodes)
    num_train = int(num_total * args.train_split)

    train_episodes = all_episodes[:num_train]
    val_episodes = all_episodes[num_train:]

    print()
    print("Dataset Split Summary (All scenarios included)")
    print("---------------------------------------------")
    print(f"Total scenarios found: {len(all_scenarios)} ({', '.join(all_scenarios)})")
    print(f"Total episodes found:  {num_total}")
    print(f"TRAIN episodes ({args.train_split*100:.0f}%):    {len(train_episodes)}")
    print(f"VAL episodes ({(1.0 - args.train_split)*100:.0f}%):      {len(val_episodes)}")
    print()

    # Build dataset loaders
    train_dataset = SequenceDataset(
        train_episodes,
        sequence_length=args.sequence_length,
        vmax=args.vmax,
        reverse_enable=args.reverse_enable,
        distance_max=args.distance_max,
    )

    validation_dataset = SequenceDataset(
        val_episodes,
        sequence_length=args.sequence_length,
        vmax=args.vmax,
        reverse_enable=args.reverse_enable,
        distance_max=args.distance_max,
    )

    print(f"TRAIN sequence count:      {len(train_dataset)}")
    print(f"VALIDATION sequence count: {len(validation_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError("TRAIN dataset contains no sequences. Try reducing C.")

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=cuda_available,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cuda_available,
    )

    print()
    if device.type == "cuda":
        print("Device: CUDA")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available.")
        print("Falling back to CPU.")

    # Init model and optimizer
    model = create_model(
        model_class_name=args.model_class,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_validation_loss = float("inf")
    best_epoch = -1
    history = []

    checkpoint_path = output_dir / "best_model.pth"

    print()
    print("Training")
    print("--------")

    start_total_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            # Train step with dynamic line update
            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                training=True,
                current_epoch=epoch,
                total_epochs=args.epochs,
                epoch_start_time=epoch_start_time,
            )

            # Validation step
            validation_loss = run_epoch(
                model=model,
                loader=validation_loader,
                optimizer=None,
                device=device,
                training=False,
                current_epoch=epoch,
                total_epochs=args.epochs,
                epoch_start_time=epoch_start_time,
            )

            scheduler.step(validation_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start_time

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_lr,
                "epoch_time_sec": epoch_time,
            }
            history.append(row)

            # Final string for completed epoch
            log_line = (
                f"epoch {epoch}/{args.epochs} | "
                f"train={train_loss:.7f} | "
                f"val={validation_loss:.7f} | "
                f"t[s]={epoch_time:.2f}"
            )
            sys.stdout.write(f"\r{log_line.ljust(80)}\n")
            sys.stdout.flush()

            # Save best checkpoint
            if np.isfinite(validation_loss) and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "config": vars(args),
                        "all_scenarios": all_scenarios,
                    },
                    checkpoint_path,
                )

    except KeyboardInterrupt:
        # Stop training on Ctrl+C without breaking script
        print("\n[EARLY STOP] Ctrl+C detected. Stopping training...")

    total_elapsed = time.time() - start_total_time

    # Save output files
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    learning_curves_path = output_dir / "learning_curves.png"
    save_learning_curves(history, learning_curves_path)

    print()
    print("Training finished.")
    print(f"Total time: {format_time(total_elapsed)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_validation_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Learning curves: {learning_curves_path}")


if __name__ == "__main__":
    main()