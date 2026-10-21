#!/usr/bin/env python3
"""Frozen map-level splits for Justina mapless-navigation experiments.

This module intentionally uses only the Python standard library so its
integrity checks can run without ROS 2, PyTorch, or Webots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCENARIOS = tuple(f"E{i:02d}" for i in range(1, 31))
FIXED_TRAIN = ("E01", "E02", "E03")
FIXED_TEST = ("E04", "E05", "E06")

EXPECTED_SPLITS = {
    "A": {
        "id": "mapless_train8_test22",
        "train_scenarios": (
            "E01", "E02", "E03", "E12", "E17", "E22", "E27", "E30",
        ),
        "test_scenarios": (
            "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11",
            "E13", "E14", "E15", "E16", "E18", "E19", "E20", "E21",
            "E23", "E24", "E25", "E26", "E28", "E29",
        ),
    },
    "B": {
        "id": "mapless_train17_test13",
        "train_scenarios": (
            "E01", "E02", "E03", "E07", "E09", "E11", "E13", "E14",
            "E15", "E17", "E18", "E21", "E22", "E25", "E26", "E28",
            "E30",
        ),
        "test_scenarios": (
            "E04", "E05", "E06", "E08", "E10", "E12", "E16", "E19",
            "E20", "E23", "E24", "E27", "E29",
        ),
    },
}


def default_config_path() -> Path:
    """Return the committed split configuration path."""
    return Path(__file__).resolve().parent / "experiments" / "experiment_splits_v1.json"


def validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError if a split deviates from the frozen scientific design."""
    observed_scenarios = config.get("dataset_scenarios")
    if not isinstance(observed_scenarios, list):
        raise ValueError("dataset_scenarios must be a JSON array")
    if len(observed_scenarios) != 30 or len(set(observed_scenarios)) != 30:
        raise ValueError("dataset_scenarios must contain 30 unique scenario IDs")
    if tuple(observed_scenarios) != SCENARIOS:
        raise ValueError("dataset_scenarios must be exactly E01 through E30 in order")

    constraints = config.get("fixed_constraints", {})
    if tuple(constraints.get("train", ())) != FIXED_TRAIN:
        raise ValueError("E01, E02, E03 must remain in TRAIN")
    if tuple(constraints.get("test", ())) != FIXED_TEST:
        raise ValueError("E04, E05, E06 must remain in TEST")

    experiments = config.get("experiments")
    if not isinstance(experiments, dict) or set(experiments) != {"A", "B"}:
        raise ValueError("experiments must contain exactly A and B")

    universe = set(SCENARIOS)
    for name, expected in EXPECTED_SPLITS.items():
        item = experiments[name]
        train = item.get("train_scenarios")
        test = item.get("test_scenarios")
        if not isinstance(train, list) or not isinstance(test, list):
            raise ValueError(f"Experiment {name}: TRAIN and TEST must be arrays")
        if len(train) != len(set(train)):
            raise ValueError(f"Experiment {name}: duplicate TRAIN scenario")
        if len(test) != len(set(test)):
            raise ValueError(f"Experiment {name}: duplicate TEST scenario")
        if set(train) & set(test):
            raise ValueError(f"Experiment {name}: TRAIN/TEST intersection is not empty")
        if set(train) | set(test) != universe:
            missing = sorted(universe - (set(train) | set(test)))
            unexpected = sorted((set(train) | set(test)) - universe)
            raise ValueError(
                f"Experiment {name}: TRAIN+TEST must cover E01-E30; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if tuple(train) != expected["train_scenarios"]:
            raise ValueError(f"Experiment {name}: TRAIN differs from frozen design")
        if tuple(test) != expected["test_scenarios"]:
            raise ValueError(f"Experiment {name}: TEST differs from frozen design")
        if item.get("id") != expected["id"]:
            raise ValueError(f"Experiment {name}: unexpected experiment ID")
        if len(train) != (8 if name == "A" else 17):
            raise ValueError(f"Experiment {name}: incorrect TRAIN scenario count")
        if len(test) != (22 if name == "A" else 13):
            raise ValueError(f"Experiment {name}: incorrect TEST scenario count")

    validation = config.get("validation", {})
    fraction = validation.get("fraction")
    if not isinstance(fraction, (int, float)) or not 0.0 < fraction < 1.0:
        raise ValueError("validation.fraction must be between 0 and 1")
    if validation.get("unit") != "episode":
        raise ValueError("Internal validation must be split by episode")
    if not isinstance(validation.get("seed"), int):
        raise ValueError("validation.seed must be an integer")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the frozen configuration from JSON."""
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def get_experiment(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a validated experiment definition by key (A or B)."""
    validate_config(config)
    key = name.upper()
    if key not in EXPECTED_SPLITS:
        raise ValueError(f"Unknown experiment {name!r}; choose A or B")
    return config["experiments"][key]

def shared_test_scenarios(config: dict[str, Any]) -> list[str]:
    """Return the common unseen maps for paired A-versus-B comparisons.

    The full A and B test sets intentionally differ in size. Comparing their
    raw aggregate means directly would mix the training-size effect with
    differences in which maps were tested. Their intersection is a paired
    comparison set unseen by both experiment models.
    """
    validate_config(config)
    test_a = set(config["experiments"]["A"]["test_scenarios"])
    test_b = set(config["experiments"]["B"]["test_scenarios"])
    return [scenario for scenario in SCENARIOS if scenario in test_a & test_b]

