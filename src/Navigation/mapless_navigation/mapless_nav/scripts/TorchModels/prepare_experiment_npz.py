#!/usr/bin/env python3
"""Safely convert existing ROS2 recordings into a separate NPZ staging tree.

This is a preparation step for the frozen A/B trainer. It does not write into
or modify the source dataset. It converts only the union of scenarios used for
TRAIN in either experiment; all E01-E30 directories are created in staging so
the trainer can verify the full scenario universe. For any one experiment, the
trainer still deserializes NPZ sample content only from that experiment's TRAIN
scenarios.

Requires the same runtime packages as generate_npz_data.py: NumPy, PyYAML and
Pillow. The source CSV, LOM, descriptor and episode_tries.json remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_SCRIPTS_DIR = (
    SCRIPT_DIR.parents[2] / "dataset" / "scripts"
)
if str(DATASET_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_SCRIPTS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_config import EXPECTED_SPLITS, SCENARIOS, load_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_union(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the fixed union of A/B TRAIN maps, not the TEST-only maps."""
    train = set()
    for name in ("A", "B"):
        train.update(config["experiments"][name]["train_scenarios"])
    return tuple(s for s in SCENARIOS if s in train)


def validate_separate_output(source_root: Path, output_root: Path) -> None:
    """Refuse in-place conversion, nested paths and non-empty output trees."""
    source = source_root.resolve()
    output = output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError(
            f"Source and output roots must be disjoint trees: source={source}, output={output}"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            f"Staging output must be a new or empty directory: {output}. "
            "No files were changed."
        )


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def read_episode_classes(sidecar_path: Path) -> dict[int, str]:
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Missing episode class sidecar: {sidecar_path}")
    obj = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mapping: dict[int, str] = {}
    for item in obj.get("entries", []):
        try:
            episode_id = int(item["episode_id"])
            label = str(item["class"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class entry in {sidecar_path}: {item!r}") from exc
        if episode_id in mapping:
            raise ValueError(f"Duplicate episode_id={episode_id} in {sidecar_path}")
        if label not in ("straight", "semi_complex", "complex", "v_complex"):
            raise ValueError(
                f"Unknown class {label!r} for episode {episode_id} in {sidecar_path}"
            )
        mapping[episode_id] = label
    return mapping


def convert_run(source_run: Path, staging_scenario_dir: Path, converter: Any) -> dict[str, Any]:
    """Convert one raw run and stage its class sidecar beside the NPZ stem."""
    csv_path = source_run / "data.csv"
    sidecar_path = source_run / "episode_tries.json"
    descriptor_path = source_run / "descriptor.yaml"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")
    if not (source_run / "lom").is_dir():
        raise FileNotFoundError(f"Missing LOM directory: {source_run / 'lom'}")
    classes = read_episode_classes(sidecar_path)
    expected_rows = count_csv_rows(csv_path)
    if expected_rows <= 0:
        raise ValueError(f"No recorded samples in {csv_path}")

    # _load_run is the conversion implementation used by the project's existing
    # generator. It reads CSV/LOM and returns the canonical nested NPZ schema.
    samples = converter._load_run(str(source_run))
    if samples is None:
        raise RuntimeError(f"Could not load source run: {source_run}")
    if len(samples) != expected_rows:
        raise RuntimeError(
            f"Conversion would drop samples in {source_run.name}: "
            f"CSV rows={expected_rows}, decoded={len(samples)}. "
            "No staged NPZ was written for this run."
        )

    observed_ids = {int(sample["meta"]["episode_id"]) for sample in samples}
    missing_labels = sorted(observed_ids - set(classes))
    if missing_labels:
        raise ValueError(
            f"episode_tries.json lacks labels for recorded episodes in {source_run}: "
            f"{missing_labels[:20]}"
        )

    # The loader expects data/E##/<run-name>.npz, and class lookup expects
    # data/E##/<run-name>/episode_tries.json. Create both without writing to source.
    stage_sidecar_dir = staging_scenario_dir / source_run.name
    stage_sidecar_dir.mkdir(parents=True, exist_ok=True)
    stage_npz = staging_scenario_dir / f"{source_run.name}.npz"
    if stage_npz.exists():
        raise FileExistsError(f"Refusing to overwrite staged NPZ: {stage_npz}")

    # Import NumPy only here, so config/safety tests can run with stdlib alone.
    import numpy as np

    np.savez_compressed(stage_npz, data=np.asarray(samples, dtype=object))
    shutil.copy2(sidecar_path, stage_sidecar_dir / "episode_tries.json")
    if descriptor_path.is_file():
        shutil.copy2(descriptor_path, stage_sidecar_dir / "descriptor.yaml")

    class_episode_counts = {
        name: sum(classes[eid] == name for eid in observed_ids)
        for name in ("straight", "semi_complex", "complex", "v_complex")
    }
    return {
        "source_run": str(source_run.resolve()),
        "source_csv_sha256": sha256_file(csv_path),
        "source_sidecar_sha256": sha256_file(sidecar_path),
        "source_descriptor_sha256": sha256_file(descriptor_path) if descriptor_path.is_file() else None,
        "source_csv_rows": expected_rows,
        "converted_samples": len(samples),
        "episode_ids_in_npz": len(observed_ids),
        "class_episode_counts": class_episode_counts,
        "staged_npz": str(stage_npz.resolve()),
        "staged_npz_bytes": stage_npz.stat().st_size,
        "staged_npz_sha256": sha256_file(stage_npz),
        "staged_class_sidecar": str((stage_sidecar_dir / "episode_tries.json").resolve()),
    }


def prepare(
    source_root: Path,
    output_root: Path,
    scenarios: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Convert selected source runs to an isolated dataset root."""
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")
    validate_separate_output(source_root, output_root)

    config = load_config()
    selected_scenarios = scenarios or training_union(config)
    if not selected_scenarios:
        raise ValueError("No scenarios selected")
    invalid = sorted(set(selected_scenarios) - set(SCENARIOS))
    if invalid:
        raise ValueError(f"Unknown scenario IDs: {invalid}")
    # For the experimental staging pipeline, keep the conversion target limited
    # to the fixed TRAIN union. That guarantees TEST-only maps have no NPZ here.
    allowed_union = set(training_union(config))
    unexpected = sorted(set(selected_scenarios) - allowed_union)
    if unexpected:
        raise ValueError(
            "Refusing to convert scenarios not used for TRAIN in either experiment: "
            f"{unexpected}. TEST-only map recordings stay outside the staging NPZ tree."
        )

    # Check all inputs before creating output to fail early and preserve atomicity.
    run_plan: list[tuple[str, Path]] = []
    for scenario in selected_scenarios:
        scenario_dir = source_root / scenario
        if not scenario_dir.is_dir():
            raise FileNotFoundError(f"Missing scenario directory: {scenario_dir}")
        runs = sorted(
            p for p in scenario_dir.iterdir()
            if p.is_dir() and p.name.startswith("run_") and (p / "data.csv").is_file()
        )
        if not runs:
            raise FileNotFoundError(f"No run_*/data.csv found under {scenario_dir}")
        for run in runs:
            for required in (run / "data.csv", run / "episode_tries.json", run / "lom"):
                if not required.exists():
                    raise FileNotFoundError(f"Missing required source artifact: {required}")
            run_plan.append((scenario, run))

    # All 30 directories exist so the training runner can verify split coverage.
    output_root.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        (output_root / scenario).mkdir(parents=True, exist_ok=True)

    # Use the project's existing row/LOM decoder, but bypass its writing routine:
    # _convert_run writes beside source_run and is therefore not called.
    converter = importlib.import_module("generate_npz_data")
    run_reports = []
    try:
        for scenario, source_run in run_plan:
            print(f"[{scenario}] converting to staging: {source_run.name}")
            run_reports.append(
                {"scenario": scenario, **convert_run(source_run, output_root / scenario, converter)}
            )
    except Exception:
        # Preserve evidence of a partial staging conversion rather than risk
        # silently reusing it. The user can inspect/delete the separate staging
        # directory explicitly after reviewing the failure.
        raise

    scenario_summary = {}
    for scenario in selected_scenarios:
        these = [r for r in run_reports if r["scenario"] == scenario]
        class_counts = {
            c: sum(r["class_episode_counts"].get(c, 0) for r in these)
            for c in ("straight", "semi_complex", "complex", "v_complex")
        }
        scenario_summary[scenario] = {
            "run_count": len(these),
            "samples": sum(r["converted_samples"] for r in these),
            "episode_ids_with_samples_summed_across_runs": sum(r["episode_ids_in_npz"] for r in these),
            "episodes_by_class_summed_across_runs": class_counts,
        }

    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "staging_root": str(output_root),
        "source_modified": False,
        "experiment_ids": {
            name: config["experiments"][name]["id"] for name in ("A", "B")
        },
        "configured_scenario_universe": list(SCENARIOS),
        "converted_scenarios": list(selected_scenarios),
        "created_empty_scenario_dirs": [s for s in SCENARIOS if s not in selected_scenarios],
        "test_only_scenarios_converted": [],
        "run_count": len(run_reports),
        "sample_count": sum(r["converted_samples"] for r in run_reports),
        "runs": run_reports,
        "scenario_summary": scenario_summary,
        "note": (
            "ROS2 raw recordings converted into a separate staging root using the existing "
            "generate_npz_data.py _load_run decoder. Source files were read only. "
            "No class quotas, subsampling, trajectory generation, or balancing were applied."
        ),
    }
    manifest_path = output_root / "conversion_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite conversion manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=SCRIPT_DIR.parents[2] / "dataset" / "data",
        help="Original ROS2 recordings root; read-only.",
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Separate, new/empty staging root; the script refuses to overwrite files.",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Optional subset of the fixed union of A/B TRAIN maps.",
    )
    args = parser.parse_args()
    try:
        manifest = prepare(args.data_root, args.output_root, tuple(args.scenarios) if args.scenarios else None)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "staging_root": manifest["staging_root"],
        "converted_scenarios": len(manifest["converted_scenarios"]),
        "runs": manifest["run_count"],
        "samples": manifest["sample_count"],
        "manifest": str(Path(manifest["staging_root"]) / "conversion_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
