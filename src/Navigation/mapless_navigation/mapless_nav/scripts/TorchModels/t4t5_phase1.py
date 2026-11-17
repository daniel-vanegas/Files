#!/usr/bin/env python3
"""Read-only project audit and controlled T4/T5 architecture pilots.

This is a phase-1 tool: pilots train on independent C=2 pairs only.
A pilot MSE is NOT evidence of navigation success or OOD generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TRAIN_CORE = ("E02", "E03", "E04", "E08", "E13", "E17", "E18", "E21", "E25", "E30")
TRAIN_EXPANSION = ("E09", "E14", "E15", "E16", "E27")
TRAIN_EXTENDED = tuple(sorted(TRAIN_CORE + TRAIN_EXPANSION))
EVAL_REPRESENTATIVE = ("E01", "E05", "E06", "E07", "E10", "E11", "E12", "E19", "E20", "E22", "E26", "E29")
EVAL_DIAGNOSTIC = ("E23", "E24", "E28")
EVAL_HOLDOUT = tuple(sorted(EVAL_REPRESENTATIVE + EVAL_DIAGNOSTIC))
ALL = tuple(f"E{i:02d}" for i in range(1, 31))
MODEL_TYPES = ("rnn", "gru", "lstm")


def repository_paths(workspace: Path):
    scripts = workspace / "src/Navigation/mapless_navigation/mapless_nav/scripts"
    models = scripts / "TorchModels"
    data = workspace / "src/Navigation/mapless_navigation/dataset/data"
    return scripts, models, data


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_split():
    core = set(TRAIN_CORE)
    ext = set(TRAIN_EXTENDED)
    held = set(EVAL_HOLDOUT)
    if not (len(core) == 10 and len(ext) == 15 and len(held) == 15
            and core.issubset(ext) and not ext.intersection(held)
            and ext.union(held) == set(ALL)):
        raise RuntimeError("Invalid fixed T4/T5/evaluation split")


def audit(workspace: Path, data_root: Path, output: Path | None):
    validate_split()
    scripts, models_dir, default_data = repository_paths(workspace)
    report = {"phase": "read_only_project_audit", "utc": now(),
              "workspace": str(workspace), "data_root": str(data_root),
              "training_scenarios": {"T4": list(TRAIN_CORE), "T5": list(TRAIN_EXTENDED)},
              "evaluation_holdout": list(EVAL_HOLDOUT),
              "files": {}, "data": {}, "warnings": [], "errors": []}
    files = [models_dir / "models.py", models_dir / "train.py",
             scripts / "RNN_mapless.py", models_dir / "dataset_utils.py"]
    for path in files:
        exists = path.is_file()
        report["files"][str(path.relative_to(workspace))] = {
            "exists": exists, "sha256": sha256(path) if exists else None}
        if not exists:
            report["errors"].append(f"Required source missing: {path}")
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                           "device_count": torch.cuda.device_count()}
    except ImportError as exc:
        report["errors"].append(f"PyTorch not installed: {exc}")
    try:
        import rclpy  # noqa: F401
        report["ros2_python_available"] = True
    except ImportError:
        report["ros2_python_available"] = False
        report["warnings"].append("ROS 2 Python unavailable in this interpreter (not required for smoke tests).")
    if not data_root.is_dir():
        report["errors"].append(f"Data directory missing: {data_root}")
    else:
        for name in ALL:
            root = data_root / name
            direct = sorted(root.glob("*.npz")) if root.is_dir() else []
            nested = sorted(root.rglob("*.npz")) if root.is_dir() else []
            report["data"][name] = {"exists": root.is_dir(), "direct_npz": len(direct),
                                     "nested_npz": len(nested)}
            if name in TRAIN_EXTENDED and len(direct) == 0:
                report["warnings"].append(f"{name}: no direct NPZ in data/{name}; trainer cannot use it yet")
            if name in EVAL_HOLDOUT and nested:
                # These files can exist; they are explicitly excluded from trainer loaders.
                pass
    report["status"] = "ERROR" if report["errors"] else "OK_WITH_WARNINGS" if report["warnings"] else "OK"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 2 if report["errors"] else 0


def smoke(workspace: Path, output: Path | None):
    import numpy as np
    import torch
    scripts, models_dir, _ = repository_paths(workspace)
    sys.path.insert(0, str(models_dir))
    from models import TemporalPolicy
    module_spec = importlib.util.spec_from_file_location("mapless_inference_smoke", scripts / "RNN_mapless.py")
    infer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(infer)
    report = {"phase": "synthetic_model_and_inference_smoke", "utc": now(),
              "models": {}, "errors": []}
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    for model_type in MODEL_TYPES:
        try:
            model = TemporalPolicy(model_type, img_h=100, img_w=80, hidden_size=32, num_layers=2)
            model.train()
            grid = torch.randn(2, 2, 102, 80)
            grid[:, :, :-2, :] = torch.randint(0, 2, (2, 2, 100, 80)).float()
            target = torch.tensor([[1.5, -0.3], [2.1, 0.5]])
            grid[:, :, -2:, :] = target[:, None, :, None]
            y, hx = model(grid, return_state=True)
            assert y.shape == (2, 3) and torch.isfinite(y).all()
            y.square().mean().backward()
            assert model.encoder[0].weight.grad is not None
            assert torch.isfinite(model.encoder[0].weight.grad).all()
            config = {"model_family": "temporal_v1", "model_type": model_type,
                      "architecture": "temporal_" + model_type, "sequence_length": 2,
                      "img_h": 100, "img_w": 80, "hidden_size": 32, "num_layers": 2,
                      "variant": "5x4", "representation": "tlom", "data_format": "ros2_episode",
                      "normalize_occupancy": True, "vmax": [0.5, 0.5, 1.0],
                      "temporal_mode": "independent"}
            checkpoint = {"model_state_dict": model.state_dict(),
                          "architecture": config["architecture"], "config": config}
            state = infer.get_state_dict(checkpoint)
            kind = infer.detect_architecture(state, checkpoint)
            reconstructed, settings = infer.make_model(kind, state, config)
            reconstructed.eval()
            model.eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(grid), reconstructed(grid))
                # A full window is processed without carrying state between calls.
                torch.testing.assert_close(reconstructed(grid), reconstructed(grid))
            # Canonical ROS occupancy grid input is 7x6, 120 x 140 after ROS transposition.
            ros_grid = np.zeros((120, 140), dtype=np.float32)
            ros_grid[40:60, 50:60] = 100.0
            obs = infer.prepare_observation(ros_grid, 1.2, 0.3, settings)
            assert obs.shape == (102, 80)
            assert set(np.unique(obs[:-2])).issubset({0.0, 1.0})
            decoded = infer.decode_action(y[0].detach().numpy(), settings)
            assert np.isfinite(decoded).all()
            # LSTM returns both hidden state and cell state.
            assert (isinstance(hx, tuple)) == (model_type == "lstm")
            report["models"][model_type] = {
                "status": "PASS", "parameters": sum(x.numel() for x in model.parameters()),
                "output_shape": list(y.shape), "input_shape": list(obs.shape),
                "strict_checkpoint_load": True, "forward_backward": True,
                "preprocessing_input_range": [float(obs[:-2].min()), float(obs[:-2].max())]}
        except Exception as exc:
            report["models"][model_type] = {"status": "FAIL", "error": repr(exc)}
            report["errors"].append(f"{model_type}: {exc}")
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if not report["errors"] else 2


def pilot(workspace: Path, data_root: Path, args):
    validate_split()
    scripts, models_dir, _ = repository_paths(workspace)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists, choose another: {output}")
    output.mkdir(parents=True)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    choices = {"T4": TRAIN_CORE, "T5": TRAIN_EXTENDED}
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    names = tuple(d.strip().upper() for d in args.datasets.split(",") if d.strip())
    if not models or not names or any(m not in MODEL_TYPES for m in models) or any(d not in choices for d in names):
        raise ValueError("Allowed models: rnn,gru,lstm; datasets: T4,T5")
    manifest = {"phase": "pilot_not_final_evaluation", "utc": now(), "seed": seed,
                "data_root": str(data_root), "holdout": list(EVAL_HOLDOUT),
                "training_budget_per_run": args.train_budget,
                "validation_budget_per_run": args.validation_budget,
                "epochs": args.epochs, "models": list(models), "datasets": list(names),
                "runs": []}
    try:
        for dataset in names:
            for model in models:
                run_dir = output / f"{dataset}_{model}"
                command = [sys.executable, str(models_dir / "train.py"),
                           "--data-format", "ros2_episode", "--data-root", str(data_root),
                           "--architecture", "ros2", "--model-type", model,
                           "--variant", "5x4", "--representation", "tlom",
                           "--train-scenarios", ",".join(choices[dataset]),
                           "--test-scenarios", ",".join(EVAL_HOLDOUT),
                           "--sequence-length", "2", "--seed", str(seed),
                           "--epochs", str(args.epochs), "--patience", str(args.patience),
                           "--batch-size", str(args.batch_size),
                           "--hidden-size", str(args.hidden_size),
                           "--num-layers", str(args.num_layers),
                           "--train-budget", str(args.train_budget),
                           "--validation-budget", str(args.validation_budget),
                           "--output-dir", str(run_dir), "--device", args.device]
                entry = {"dataset": dataset, "model": model, "command": command, "output_dir": str(run_dir)}
                manifest["runs"].append(entry)
                (output / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"\nRUN {dataset} {model} seed={seed}", flush=True)
                log = output / f"{dataset}_{model}.log"
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(workspace))
                entry["returncode"] = process.returncode
                entry["log"] = str(log)
                (output / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"RESULT {dataset}/{model} returncode={process.returncode} log={log}", flush=True)
                if process.returncode and not args.continue_on_error:
                    break
            if manifest["runs"][-1].get("returncode") and not args.continue_on_error:
                break
    except KeyboardInterrupt:
        manifest["interrupted"] = True
    finally:
        (output / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if all(run.get("returncode") == 0 for run in manifest["runs"]) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "pilot"))
    parser.add_argument("--workspace", type=Path, default=Path.home() / "GIT/mapless")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="JSON report for audit/smoke")
    parser.add_argument("--output-dir", type=Path, default=None, help="Unique directory for pilot outputs")
    parser.add_argument("--models", default="rnn,gru,lstm")
    parser.add_argument("--datasets", default="T4,T5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--train-budget", type=int, default=128)
    parser.add_argument("--validation-budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    _, _, default_data = repository_paths(workspace)
    data_root = (args.data_root or default_data).expanduser().resolve()
    if args.command == "audit":
        return audit(workspace, data_root, args.output)
    if args.command == "smoke":
        return smoke(workspace, args.output)
    if not data_root.is_dir():
        parser.error(f"No data root: {data_root}")
    if min(args.epochs, args.patience, args.batch_size, args.hidden_size,
           args.num_layers, args.train_budget, args.validation_budget) < 1:
        parser.error("Pilot numeric options must be positive")
    if args.output_dir is None:
        root = workspace / "experiments" / "t4t5_phase1"
        args.output_dir = root / datetime.now().strftime("pilot_%Y%m%d_%H%M%S_%f")
    return pilot(workspace, data_root, args)


if __name__ == "__main__":
    raise SystemExit(main())
