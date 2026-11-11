#!/usr/bin/env python3
"""Run paired 5x4 navigation experiments with recorded reference trajectories."""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent.parent
SCRIPTS = DATASET.parent / "mapless_nav/scripts"
TRAINER = SCRIPTS / "TorchModels/train.py"
EVALUATOR = SCRIPTS / "evaluate_navigation.py"
CLASSES = ("straight", "semi_complex", "complex", "v_complex")
ENVIRONMENTS = {
    "T1": ["E01", "E02", "E03", "E12", "E17", "E22", "E27", "E30"],
    "T2": ["E01", "E02", "E03", "E07", "E09", "E11", "E12", "E13", "E14", "E15",
           "E17", "E18", "E21", "E22", "E25", "E26", "E28", "E30"],
}
FINAL_SCENARIOS = ["E04", "E05", "E06", "E10", "E19", "E23", "E24", "E29"]
STAGES = ("baseline", "dataset", "learning_rate", "dropout", "loss", "confirm")
SCORE_WEIGHTS = {"path_weight": 0.40, "arrival_weight": 0.30, "collision_weight": 0.15,
                 "time_weight": 0.10, "length_weight": 0.05}
DEMONSTRATION_TOLERANCE_M = 0.50
ORIGINAL_SCALE_CODE = {
    "navigation_campaign.py": "daad828567af14796185da713cb90eaa85acaeadd0c7a36ebf98eb8bf6798b13",
    "train.py": "1f182d7b42aebd74876b0a81ddbc968341932d46afb8dd7b8d23f189f303bfd6",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=None if compact else 2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluator():
    sys.path.insert(0, str(SCRIPTS))
    import evaluate_navigation
    return evaluate_navigation


def resample_path(points, spacing=0.05, maximum=256):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("A path needs finite XY coordinates")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
    points = points[keep]
    if len(points) == 1:
        return points, 0.0
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = min(maximum, max(2, int(math.ceil(distances[-1] / spacing)) + 1))
    samples = np.linspace(0.0, distances[-1], count)
    return np.column_stack([np.interp(samples, distances, points[:, axis]) for axis in (0, 1)]), float(distances[-1])


def discrete_frechet(first, second):
    previous = np.full(len(second), np.inf)
    for i, point in enumerate(first):
        current = np.full(len(second), np.inf)
        for j, distance in enumerate(np.linalg.norm(second - point, axis=1)):
            if i == 0 and j == 0:
                current[j] = distance
            elif i == 0:
                current[j] = max(distance, current[j - 1])
            elif j == 0:
                current[j] = max(distance, previous[j])
            else:
                current[j] = max(distance, min(previous[j], previous[j - 1], current[j - 1]))
        previous = current
    return float(previous[-1])


def geometry_class(poses):
    points, length = resample_path(np.asarray(poses)[:, :2], spacing=0.10, maximum=500)
    if length < 0.01:
        return "straight"
    efficiency = float(np.linalg.norm(points[-1] - points[0])) / length
    direction = np.diff(points, axis=0)
    headings = np.unwrap(np.arctan2(direction[:, 1], direction[:, 0]))
    turns = int(np.count_nonzero(np.abs(np.diff(headings)) >= math.pi / 4))
    if efficiency > 0.95:
        return "straight"
    if turns >= 3:
        return "v_complex"
    if turns >= 2 or efficiency < 0.80:
        return "complex"
    return "semi_complex"


def load_run(path):
    with np.load(path, allow_pickle=True) as archive:
        samples = archive["data"].tolist()
    grouped = collections.defaultdict(list)
    for sample in samples:
        grouped[int(sample["meta"]["episode_id"])].append(sample)
    for group in grouped.values():
        group.sort(key=lambda sample: float(sample["meta"]["timestamp"]))
    sidecar = path.with_suffix("") / "episode_tries.json"
    labels = {}
    if sidecar.is_file():
        for entry in read_json(sidecar).get("entries", []):
            if entry.get("class") in CLASSES:
                labels[int(entry["episode_id"])] = entry["class"]
    return grouped, labels


def sign_label(value, deadband):
    if value > deadband:
        return "left"
    if value < -deadband:
        return "right"
    return "neutral"


def balanced_capacity(rows):
    counts = collections.Counter(row["side"] for row in rows)
    return counts["neutral"] + 2 * min(counts["left"], counts["right"])


def draw_balanced(rows, count, rng):
    buckets = {side: [row for row in rows if row["side"] == side] for side in ("left", "right", "neutral")}
    capacity = balanced_capacity(rows)
    if count > capacity:
        raise ValueError("The available left/right support cannot fill this quota")
    side_pairs = min(len(buckets["left"]), len(buckets["right"]))
    neutral = min(len(buckets["neutral"]), max(count - 2 * side_pairs, round(count * len(buckets["neutral"]) / max(1, capacity))))
    half, remainder = divmod(count - neutral, 2)
    extra_left = rng.randrange(2) if remainder else 0
    selected = rng.sample(buckets["left"], half + extra_left)
    selected += rng.sample(buckets["right"], half + remainder - extra_left)
    selected += rng.sample(buckets["neutral"], neutral)
    return selected


def quotas_for(size, weights):
    raw = [size * weight for weight in weights]
    quotas = [int(value) for value in raw]
    for i in sorted(range(4), key=lambda i: raw[i] - quotas[i], reverse=True)[:size - sum(quotas)]:
        quotas[i] += 1
    return dict(zip(CLASSES, quotas))


def initialize(args):
    root = args.campaign_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = random.SystemRandom()
    seeds = rng.sample(range(1, 2**31), args.seeds)
    config = {
        "version": 1, "campaign_dir": str(root), "data_root": str(args.data_root.expanduser().resolve()),
        "environment_sets": {name: ENVIRONMENTS[name] for name in args.environments},
        "representations": ["lom", "tlom"], "seeds": seeds, "split_seed": rng.randrange(1, 2**31),
        "reference_frame": args.reference_frame, "validation_fraction": 0.25,
        "pair_budget": args.pair_budget, "validation_pairs": 4000, "max_pair_gap_s": 0.5,
        "lateral_deadband": 0.02, "reference_goal_tolerance_m": 0.35,
        "demonstration_goal_tolerance_m": DEMONSTRATION_TOLERANCE_M,
        "screen": {"E08": 1}, "confirmation": {"E16": 2, "E20": 2},
        "timeout_s": 60.0, "rviz": True, "infrastructure_retries": 1,
        "recipes": {"natural": None, "moderate": [0.20, 0.20, 0.30, 0.30], "hard": [0.10, 0.15, 0.35, 0.40]},
        "train": {"epochs": 20, "patience": 5, "batch_size": 3, "learning_rate": 0.00015,
                  "weight_decay": 0.0, "dropout_1": 0.20, "dropout_2": 0.15,
                  "action_weights": [1.0, 1.0, 1.0]},
        "metric": {**SCORE_WEIGHTS, "frechet_decay": 3.0,
                   "spatial_step_m": 0.05, "maximum_points": 256},
        "minimum_improvement": 0.005,
    }
    write_json(root / "config.json", config)
    print(f"Created {root / 'config.json'}")
    print(f"Random shared seeds: {seeds}")


class Campaign:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.cfg = read_json(self.root / "config.json")
        if self.cfg["version"] != 1 or Path(self.cfg["campaign_dir"]) != self.root:
            raise ValueError("Campaign metadata does not match this directory")
        self.data = Path(self.cfg["data_root"]).resolve()
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {"lanes": {}, "stages": {}}
        self.prepared = None

    def log(self, event, **values):
        record = {"timestamp": time.time(), "event": event, **values}
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"[{event}] " + " ".join(f"{key}={value}" for key, value in values.items()), flush=True)

    def source_signature(self):
        paths = [Path(__file__), TRAINER, TRAINER.with_name("dataset_utils.py"),
                 TRAINER.with_name("models.py"), EVALUATOR, SCRIPTS / "RNN_mapless.py"]
        return {path.name: file_hash(path) for path in paths}

    def validate_config(self):
        train_scenarios = {scenario for group in self.cfg["environment_sets"].values() for scenario in group}
        evaluation_scenarios = set(self.cfg["screen"]) | set(self.cfg["confirmation"])
        if train_scenarios & evaluation_scenarios or (train_scenarios | evaluation_scenarios) & set(FINAL_SCENARIOS):
            raise ValueError("Training, development and final scenarios must keep their reserved roles")
        if set(self.cfg["screen"]) & set(self.cfg["confirmation"]):
            raise ValueError("Screening and confirmation need different scenarios")
        if not all(self.cfg[phase] and all(type(count) is int and count > 0 for count in self.cfg[phase].values())
                   for phase in ("screen", "confirmation")):
            raise ValueError("Each evaluation phase needs a positive task quota")
        if not 0 < self.cfg["validation_fraction"] < 1 or self.cfg["validation_pairs"] < 1:
            raise ValueError("Validation needs a nonempty episode split and pair budget")
        for name, weights in self.cfg["recipes"].items():
            if weights is not None and (len(weights) != 4 or min(weights) < 0 or not math.isclose(sum(weights), 1.0)):
                raise ValueError(f"Invalid route proportions: {name}")
        seeds = self.cfg["seeds"]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or not 0 < seed < 2**31 for seed in seeds):
            raise ValueError("Training seeds must be distinct recorded integers")
        weights = [self.cfg["metric"][key] for key in SCORE_WEIGHTS]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError("Navigation score weights must be nonnegative and sum to one")
        for key in ("demonstration_goal_tolerance_m", "reference_goal_tolerance_m"):
            if not math.isfinite(self.cfg[key]) or self.cfg[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")

    def prepare(self):
        import yaml

        output = self.root / "prepared.json"
        if output.exists():
            self.load_prepared()
            self.log("prepare_skip", references=len(self.prepared["references"]))
            return
        previous_config = digest(self.cfg)
        metric = self.cfg["metric"]
        if "collision_weight" not in metric:
            metric.update(SCORE_WEIGHTS)
            metric.pop("failed_multiplier", None)
        self.cfg.setdefault("demonstration_goal_tolerance_m", DEMONSTRATION_TOLERANCE_M)
        self.validate_config()
        if digest(self.cfg) != previous_config:
            write_json(self.root / "config.json", self.cfg)
            self.log("config_updated", score_weights={key: metric[key] for key in SCORE_WEIGHTS},
                     demonstration_tolerance_m=self.cfg["demonstration_goal_tolerance_m"])
        ev = evaluator()
        scenarios = sorted({scenario for values in self.cfg["environment_sets"].values() for scenario in values}
                           | set(self.cfg["screen"]) | set(self.cfg["confirmation"]))
        episodes, sources, annotations = [], {}, {}
        diagnostics = collections.Counter()
        for scenario in scenarios:
            paths = sorted((self.data / scenario).glob("*.npz"))
            if not paths:
                raise ValueError(f"No NPZ input for {scenario}")
            for path in paths:
                relative = path.relative_to(self.data).as_posix()
                self.log("index_run", source=relative)
                sources[relative] = file_hash(path)
                grouped, labels = load_run(path)
                sidecar = path.with_suffix("") / "episode_tries.json"
                if sidecar.is_file():
                    annotations[sidecar.relative_to(self.data).as_posix()] = file_hash(sidecar)
                if scenario in set(self.cfg["screen"]) | set(self.cfg["confirmation"]):
                    descriptor = next((candidate for candidate in (path.with_suffix("") / "descriptor.yaml",
                                      path.with_suffix(".descriptor.yaml")) if candidate.is_file()), None)
                    if descriptor is not None:
                        recorded_frame = (yaml.safe_load(descriptor.read_text()) or {}).get("map_frame")
                        if recorded_frame and recorded_frame != self.cfg["reference_frame"]:
                            raise ValueError(f"Reference frame {recorded_frame} differs from configuration in {descriptor}")
                        annotations[descriptor.relative_to(self.data).as_posix()] = file_hash(descriptor)
                    else:
                        diagnostics["reference_runs_without_frame_descriptor"] += 1
                for episode_id, samples in sorted(grouped.items()):
                    if len(samples) < 2:
                        continue
                    poses = np.asarray([sample["meta"]["pose"] for sample in samples], dtype=float)
                    stamps = np.asarray([sample["meta"]["timestamp"] for sample in samples], dtype=float)
                    goal = np.asarray(samples[-1]["meta"]["goal_xy"], dtype=float)
                    if (poses.shape != (len(samples), 3) or not np.isfinite(poses).all()
                            or not np.isfinite(stamps).all() or goal.shape != (2,) or not np.isfinite(goal).all()):
                        raise ValueError(f"Invalid recorded coordinates in {relative}, episode {episode_id}")
                    category = labels.get(episode_id) or geometry_class(poses)
                    diagnostics["class_from_metadata" if episode_id in labels else "class_from_recorded_geometry"] += 1
                    pairs = []
                    for start in range(0, len(samples) - 1, 2):
                        pair = samples[start:start + 2]
                        valid = all(sample["meta"].get("tlom_consistent", True) for sample in pair)
                        gap = stamps[start + 1] - stamps[start]
                        if not valid:
                            diagnostics["inconsistent_pairs_excluded"] += 1
                            continue
                        if not 0 < gap <= self.cfg["max_pair_gap_s"]:
                            diagnostics["time_gap_pairs_excluded"] += 1
                            continue
                        for sample in pair:
                            for representation in ("occ_grid", "tlom"):
                                grid = np.asarray(sample["features"].get(representation))
                                if grid.shape != (140, 120):
                                    raise ValueError(f"Expected both canonical 140x120 maps in {relative}")
                        action = [float(value) for value in pair[-1]["labels"]]
                        if len(action) != 3 or not all(math.isfinite(value) for value in action):
                            raise ValueError(f"Invalid action in {relative}, episode {episode_id}, index {start}")
                        pairs.append({"run_file": relative, "episode_id": episode_id, "start_index": start,
                                      "route_class": category, "side": sign_label(action[1], self.cfg["lateral_deadband"]),
                                      "action": action})
                    episodes.append({"scenario": scenario, "run_file": relative, "episode_id": episode_id,
                                     "route_class": category, "pairs": pairs, "poses": poses.tolist(),
                                     "times": (stamps - stamps[0]).tolist(), "goal": goal.tolist(),
                                     "collision": any(sample["meta"].get("collision_flag", False) for sample in samples)})
                del grouped
        split_rng = random.Random(self.cfg["split_seed"])
        splits = {}
        train_scenarios = {s for values in self.cfg["environment_sets"].values() for s in values}
        for scenario in sorted(train_scenarios):
            candidates = [e for e in episodes if e["scenario"] == scenario and e["pairs"]]
            if len(candidates) < 2:
                raise ValueError(f"At least two usable episodes are required in {scenario}")
            for category in CLASSES:
                group = [e for e in candidates if e["route_class"] == category]
                split_rng.shuffle(group)
                nval = max(1, min(len(group) - 1, int(len(group) * self.cfg["validation_fraction"]))) if len(group) > 1 else 0
                for index, episode in enumerate(group):
                    splits[f"{episode['run_file']}#{episode['episode_id']}"] = "validation" if index < nval else "train"
        pools = {}
        for name, scenarios_in_set in self.cfg["environment_sets"].items():
            pools[name] = {split: [pair for episode in episodes if episode["scenario"] in scenarios_in_set
                                  and splits.get(f"{episode['run_file']}#{episode['episode_id']}") == split
                                  for pair in episode["pairs"]] for split in ("train", "validation")}
        budget = self.cfg["pair_budget"]
        class_weights = {}
        all_capacities = {}
        for name, pool in pools.items():
            capacities = {category: balanced_capacity([r for r in pool["train"] if r["route_class"] == category]) for category in CLASSES}
            all_capacities[name] = capacities
            natural_counts = collections.Counter(row["route_class"] for row in pool["train"])
            if any(natural_counts[category] == 0 for category in CLASSES) or not pool["validation"]:
                raise ValueError(f"{name} needs training episodes in all four route classes and reserved validation episodes")
            class_weights[name] = {}
            for recipe, values in self.cfg["recipes"].items():
                weights = values or [natural_counts[category] / len(pool["train"]) for category in CLASSES]
                class_weights[name][recipe] = weights
                for category, weight in zip(CLASSES, weights):
                    if weight:
                        budget = min(budget, int(capacities[category] / weight))
        while budget > 0 and any(
            any(quota > all_capacities[name][category]
                for category, quota in quotas_for(budget, weights).items())
            for name, recipes in class_weights.items() for weights in recipes.values()
        ):
            budget -= 1
        if budget < 100:
            raise ValueError("Fewer than 100 distinct balanced pairs fit all recipes; inspect class and side coverage")
        prepared = {"version": 1, "config_sha256": digest(self.cfg), "code_sha256": self.source_signature(),
                    "sources": sources, "annotations": annotations, "split": splits, "pair_budget": budget,
                    "class_weights": class_weights, "pools": pools, "references": {}, "suites": {},
                    "diagnostics": dict(diagnostics)}
        all_actions = np.abs(np.asarray([r["action"] for pool in pools.values() for r in pool["train"]]))
        launched = ev.launch_settings()["policy_parameters"]
        standard_limits = {key: float(launched[key]) for key in ("max_vx", "max_vy", "max_wz")}
        teacher_limits = np.maximum(list(standard_limits.values()),
                                    np.minimum([0.5, 0.5, 1.0], np.quantile(all_actions, 0.99, axis=0)))
        prepared["profiles"] = {
            "legacy": {"turn_help": str(bool(launched.get("turn_help", True))).lower(), **standard_limits},
            "direct": {"turn_help": "false", **standard_limits},
            "demonstration": {"turn_help": "false", **dict(zip(("max_vx", "max_vy", "max_wz"), teacher_limits.tolist()))},
        }
        for phase in ("screen", "confirmation"):
            for scenario, per_class in self.cfg[phase].items():
                tasks = []
                for category in CLASSES:
                    candidates = []
                    rejected = collections.Counter()
                    final_errors = []
                    for episode in episodes:
                        if episode["scenario"] != scenario or episode["route_class"] != category:
                            continue
                        points = np.asarray(episode["poses"])
                        final_error = float(np.linalg.norm(points[-1, :2] - episode["goal"]))
                        final_errors.append(final_error)
                        if episode["collision"]:
                            rejected["collision"] += 1
                            continue
                        if final_error > self.cfg["demonstration_goal_tolerance_m"]:
                            rejected["goal_distance"] += 1
                            continue
                        _, length = resample_path(points[:, :2])
                        duration = episode["times"][-1]
                        direct_distance = np.linalg.norm(points[0, :2] - episode["goal"])
                        if direct_distance >= 0.5 and length >= 0.5 and duration > 0:
                            candidates.append((length, episode))
                        else:
                            rejected["short_or_invalid_duration"] += 1
                    candidates.sort(key=lambda item: (item[0], item[1]["run_file"], item[1]["episode_id"]))
                    if len(candidates) < per_class:
                        closest = f"{min(final_errors):.3f} m" if final_errors else "unavailable"
                        raise ValueError(
                            f"Not enough recorded GT episodes in {scenario}/{category}. "
                            f"Eligible {len(candidates)}/{len(final_errors)}, need {per_class}. "
                            f"Rejected {dict(rejected)}. Closest recorded endpoint: {closest}."
                        )
                    selected = set()
                    quantiles = [0.5] if per_class == 1 else np.linspace(1 / (per_class + 1), per_class / (per_class + 1), per_class)
                    for quantile in quantiles:
                        position = int(round(float(quantile) * (len(candidates) - 1)))
                        position = min((i for i in range(len(candidates)) if i not in selected), key=lambda i: abs(i - position))
                        selected.add(position)
                        length, episode = candidates[position]
                        task_id = len(tasks) + 1
                        reference = {"run_file": episode["run_file"], "episode_id": episode["episode_id"],
                                     "route_class": category, "frame": self.cfg["reference_frame"],
                                     "poses": episode["poses"], "times": episode["times"], "goal": episode["goal"],
                                     "length_m": length}
                        reference_key = f"{phase}_{scenario}_{task_id}"
                        prepared["references"][reference_key] = reference
                        poses = np.asarray(episode["poses"])
                        tasks.append({"id": task_id, "route_class": category, "start": poses[0, :2].tolist(),
                                      "start_yaw_rad": float(poses[0, 2]), "goal": episode["goal"],
                                      "generator_start_yaw_rad": float(poses[0, 2]), "direct_line_safe": None,
                                      "task_type": "recorded_demonstration", "behavior_tags": [], "reference_key": reference_key})
                for profile, overrides in prepared["profiles"].items():
                    suite = self.make_suite(ev, scenario, tasks, overrides)
                    suite_key = f"{phase}_{scenario}_{profile}"
                    path = self.root / "suites" / (suite_key + ".json")
                    write_json(path, suite)
                    prepared["suites"][suite_key] = {"path": str(path), "sha256": file_hash(path)}
        metadata = {"phase": "datasets", "schema_version": 1,
                    "config": {"environment_sets": self.cfg["environment_sets"], "final_scenarios": FINAL_SCENARIOS}}
        write_json(self.root / "datasets_manifest/datasets.json", metadata)
        self.prepared = prepared
        for name in pools:
            for seed in self.cfg["seeds"]:
                for recipe in self.cfg["recipes"]:
                    self.write_pairs(name, seed, recipe)
        write_json(output, prepared, compact=True)
        self.log("prepare_done", pair_budget=budget, references=len(prepared["references"]), diagnostics=dict(diagnostics))
        self.report()

    def make_suite(self, ev, scenario, tasks, overrides):
        import yaml
        background = ev.control.load_review_map(ev.HARDWARE / "environments/maps", scenario, "prohibition")
        metadata = yaml.safe_load(Path(background["source"]["yaml"]).read_text())
        offsets = [float(metadata.get(f"world_offset_{axis}", 0)) for axis in ("x", "y", "yaw")]
        world = ev.HARDWARE / "worlds" / (scenario + ".wbt")
        if self.cfg["reference_frame"] == "world":
            tasks = json.loads(json.dumps(tasks))
            for task in tasks:
                task["start"] = ev.to_map(task["start"], offsets).tolist()
                task["goal"] = ev.to_map(task["goal"], offsets).tolist()
                task["start_yaw_rad"] -= offsets[2]
        return {"version": 1, "scenario": scenario, "purpose": "development", "seed": self.cfg["split_seed"],
                "sources": {"world_sha256": file_hash(world), "map_yaml_sha256": background["source"]["yaml_sha256"],
                            "map_image_sha256": background["source"]["image_sha256"],
                            "runtime": ev.evaluation_settings("7x6"),
                            "policy_overrides": {key: str(value) for key, value in overrides.items()}},
                "offsets": offsets, "map_source": background["source"], "heading_policy": "recorded",
                "generator_sha256": digest(tasks), "pool_complete": True, "tasks": tasks,
                "protocol": {"timeout_s": self.cfg["timeout_s"], "goal_tolerance_m": 0.25,
                             "settle_s": 1.5, "start_tolerance_m": 0.08, "start_yaw_tolerance_rad": 0.12,
                             "collision_threshold": 0.5, "collision_source": "world collision_detector TouchSensor",
                             "pose_source": "Webots Supervisor getSelf position and orientation",
                             "input_monitoring": "diagnostic_only", "input_diagnostic_age_wall_s": 2.0,
                             "world_random_seed": self.cfg["split_seed"], "simulation_mode": "realtime"}}

    def load_prepared(self):
        self.prepared = read_json(self.root / "prepared.json")
        if self.prepared["config_sha256"] != digest(self.cfg):
            raise ValueError("The configuration changed after preparation; initialize a separate campaign")
        current_code = self.source_signature()
        if self.prepared["code_sha256"] not in (current_code, {**current_code, **ORIGINAL_SCALE_CODE}):
            raise ValueError("A campaign source changed after preparation; initialize a separate campaign")
        for relative, expected in {**self.prepared["sources"], **self.prepared["annotations"]}.items():
            if file_hash(self.data / relative) != expected:
                raise ValueError(f"A demonstration input changed: {relative}")
        for suite in self.prepared["suites"].values():
            if file_hash(suite["path"]) != suite["sha256"]:
                raise ValueError("A frozen evaluation suite changed")

    def write_validation(self, environment):
        rows = self.prepared["pools"][environment]["validation"]
        rng = random.Random(self.cfg["split_seed"])
        return rng.sample(rows, min(len(rows), self.cfg["validation_pairs"]))

    def write_pairs(self, environment, seed, recipe):
        path = self.root / "pairs" / environment / f"{recipe}_{seed}.json"
        rows = self.prepared["pools"][environment]["train"]
        quotas = quotas_for(self.prepared["pair_budget"], self.prepared["class_weights"][environment][recipe])
        rng = random.Random(seed)
        chosen = []
        for category in CLASSES:
            chosen.extend(draw_balanced([r for r in rows if r["route_class"] == category], quotas[category], rng))
        rng.shuffle(chosen)
        validation = self.write_validation(environment)
        used = {row["run_file"] for row in chosen + validation}
        payload = {"version": 1, "variant": "5x4", "environment_set": environment,
                   "sources": {relative: self.prepared["sources"][relative] for relative in sorted(used)},
                   "train": chosen, "validation": validation, "recipe": recipe,
                   "class_counts": dict(collections.Counter(row["route_class"] for row in chosen)),
                   "side_counts": dict(collections.Counter(row["side"] for row in chosen))}
        if path.exists():
            if read_json(path) != payload:
                raise ValueError(f"A frozen selection changed: {path}")
        else:
            write_json(path, payload, compact=True)
        return path

    def run_command(self, command, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True)
            try:
                with process.stdout:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                return process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                raise

    def new_attempt(self, parent):
        parent.mkdir(parents=True, exist_ok=True)
        number = max([int(path.name.split("_")[-1]) for path in parent.glob("attempt_*")] or [0]) + 1
        attempt = parent / f"attempt_{number:03d}"
        attempt.mkdir()
        return attempt

    def train(self, environment, representation, seed, candidate, scope="training"):
        manifest = self.write_pairs(environment, seed, candidate["recipe"])
        spec = {"environment": environment, "representation": representation, "seed": seed,
                "candidate": candidate, "pairs_sha256": file_hash(manifest)}
        key = f"{environment}_{representation}_{candidate['recipe']}_{seed}_{digest(spec)[:12]}"
        base = self.root / scope / key
        completed = base / "completed.json"
        if completed.exists():
            receipt = read_json(completed)
            if receipt["spec"] != spec or any(file_hash(path) != expected for path, expected in receipt["checkpoints"].items()):
                raise ValueError(f"Completed training files changed: {base}")
            return receipt
        attempt = self.new_attempt(base)
        output = attempt / "output"
        hp = candidate["train"]
        command = [sys.executable, "-u", str(TRAINER), "--data-root", str(self.data),
                   "--data-format", "ros2_episode", "--pair-manifest", str(manifest), "--variant", "5x4",
                   "--representation", representation, "--sequence-length", "2", "--architecture", "ros1_legacy",
                   "--hidden-size", "350", "--num-layers", "4", "--seed", str(seed), "--output-dir", str(output),
                   "--action-weights", *map(str, hp["action_weights"])]
        for name in ("epochs", "patience", "batch_size", "learning_rate", "weight_decay", "dropout_1", "dropout_2"):
            command.extend(("--" + name.replace("_", "-"), str(hp[name])))
        if "occupancy_max" in hp:
            command.extend(("--occupancy-max", str(hp["occupancy_max"])))
        self.log("train_start", key=key, log=str(attempt / "train.log"))
        started = time.monotonic()
        status = self.run_command(command, attempt / "train.log")
        if status:
            self.log("train_failed", key=key, returncode=status, attempt=str(attempt))
            raise RuntimeError(f"Training failed: {attempt / 'train.log'}")
        checkpoints = {str(output / name): file_hash(output / name) for name in ("best_model.pth", "best_gap.pth")}
        history = read_json(output / "history.json")
        if not history:
            raise ValueError("Training produced no epochs")
        epochs = {str(output / "best_model.pth"): min(history, key=lambda row: row["validation_loss"])["epoch"],
                  str(output / "best_gap.pth"): min(history, key=lambda row: row["absolute_generalization_gap"])["epoch"]}
        receipt = {"key": key, "spec": spec, "checkpoints": checkpoints, "epochs": epochs,
                   "history": str(output / "history.json"), "action_metrics": str(output / "action_metrics.json"),
                   "elapsed_s": time.monotonic() - started}
        write_json(completed, receipt)
        self.log("train_done", key=key, epochs=len(history), elapsed_s=round(receipt["elapsed_s"], 1))
        return receipt

    def evaluate(self, receipt, checkpoint, phase, scenario, profile):
        suite_info = self.prepared["suites"][f"{phase}_{scenario}_{profile}"]
        suite = read_json(suite_info["path"])
        checkpoint_hash = receipt["checkpoints"][checkpoint]
        key = digest([checkpoint_hash, suite_info["sha256"], self.cfg["metric"]])[:16]
        base = self.root / "evaluation" / receipt["key"] / f"{phase}_{scenario}_{profile}_{key}"
        completed = base / "metrics.json"
        if completed.exists():
            result = read_json(completed)
            if not (Path(result["output_dir"]) / "trajectory.csv").is_file():
                raise ValueError(f"Missing trajectory for completed evaluation: {completed}")
            return result
        retries = self.cfg["infrastructure_retries"]
        for retry in range(retries + 1):
            attempt = self.new_attempt(base)
            output = attempt / "session"
            command = [sys.executable, "-u", str(EVALUATOR), "--datasets-dir", str(self.root / "datasets_manifest"),
                       "--checkpoint", checkpoint, "--scenario", scenario, "--purpose", "development",
                       "--suite", suite_info["path"], "--limit", str(len(suite["tasks"])),
                       "--close-after-trial", "--preview-seconds", "0.5", "--output-dir", str(output)]
            if not self.cfg["rviz"]:
                command.append("--no-rviz")
            for parameter, value in suite["sources"]["policy_overrides"].items():
                command.extend(("--" + parameter.replace("_", "-"), str(value)))
            self.log("eval_start", key=receipt["key"], scenario=scenario, profile=profile, attempt=str(attempt))
            started = time.monotonic()
            status = self.run_command(command, attempt / "evaluation.log")
            if status == 0:
                try:
                    scored = self.score_session(output, suite, checkpoint_hash)
                except (ValueError, OSError, KeyError) as exc:
                    self.log("eval_invalid", attempt=str(attempt), reason=str(exc))
                    if retry == retries:
                        raise
                    continue
                scored.update(key=receipt["key"], checkpoint=checkpoint, epoch=receipt["epochs"][checkpoint],
                              scenario=scenario, phase=phase, profile=profile, spec=receipt["spec"],
                              output_dir=str(output), elapsed_wall_s=time.monotonic() - started)
                write_json(completed, scored)
                self.log("eval_done", key=receipt["key"], scenario=scenario, profile=profile,
                         score=round(scored["score"], 4), arrivals=scored["arrivals"], collisions=scored["collisions"])
                return scored
            self.log("eval_failed", attempt=str(attempt), returncode=status)
        raise RuntimeError(f"Webots evaluation failed: {base}")

    def score_session(self, output, suite, checkpoint_hash):
        summary = read_json(output / "summary.json")
        tasks = suite["tasks"]
        ids = [task["id"] for task in tasks]
        if (summary["status"] != "completed" or summary["infrastructure_errors"]
                or summary["completed_trials"] != len(ids) or summary["selected_task_ids"] != ids
                or summary["model"]["sha256"] != checkpoint_hash
                or summary["suite_sha256"] != file_hash(output / "suite.json")
                or read_json(output / "suite.json") != suite):
            raise ValueError("The completed session does not match its checkpoint and tasks")
        paths = collections.defaultdict(list)
        commands = collections.defaultdict(list)
        with (output / "trajectory.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                task_id = int(row["task_id"])
                paths[task_id].append([float(row["world_x"]), float(row["world_y"])])
                commands[task_id].append([float(row[key]) for key in ("cmd_vx", "cmd_vy", "cmd_wz")])
        if set(paths) != set(ids) or [trial["task_id"] for trial in summary["results"]] != ids:
            raise ValueError("Trajectory CSV must contain every executed task")
        offsets = suite["offsets"]
        scored = []
        path_rows = []
        for task, trial in zip(tasks, summary["results"]):
            reference = self.prepared["references"][task["reference_key"]]
            gt = np.asarray(reference["poses"])[:, :2]
            if reference["frame"] == "world":
                gt = to_map_points(gt, offsets)
            path = to_map_points(paths[task["id"]], offsets)
            scored.append(self.score_trial(trial, path, gt, reference, commands[task["id"]], suite))
            for source, points in (("reference", gt), ("inference", path)):
                path_rows.extend((task["id"], source, index, x, y) for index, (x, y) in enumerate(points))
        with (output / "comparison_paths.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("task_id", "source", "sample_index", "map_x", "map_y"))
            writer.writerows(path_rows)
        vmax = summary["model"].get("training_config", {}).get("vmax", [0.5, 0.5, 1.0])
        return {"score": statistics.mean(t["score"] for t in scored), "arrivals": sum(t["arrival"] for t in scored),
                "collisions": sum(t["collision"] for t in scored), "trials": scored,
                "frechet_m": statistics.mean(t["frechet_m"] for t in scored),
                "frechet_normalized": statistics.mean(t["frechet_normalized"] for t in scored),
                "command_adjustment": command_adjustment(output / "simulation.log", vmax)}

    def score_trial(self, trial, path, gt, reference, commands, suite):
        metric = self.cfg["metric"]
        spacing, maximum = metric["spatial_step_m"], metric["maximum_points"]
        actual, length = resample_path(path, spacing, maximum)
        expected, ref_length = resample_path(gt, spacing, maximum)
        frechet = discrete_frechet(actual, expected)
        normalized = frechet / max(ref_length, 0.5)
        similarity = math.exp(-metric["frechet_decay"] * normalized)
        arrival = not trial["collision"] and trial["final_goal_distance_m"] <= self.cfg["reference_goal_tolerance_m"]
        length_ratio = length / ref_length
        time_ratio = trial["elapsed_sim_s"] / reference["times"][-1]
        length_score = min(length_ratio, 1 / length_ratio) if arrival and length_ratio > 0 else 0.0
        time_score = min(time_ratio, 1 / time_ratio) if arrival and time_ratio > 0 else 0.0
        no_collision = not trial["collision"]
        score = (metric["path_weight"] * similarity + metric["arrival_weight"] * int(arrival)
                 + metric["collision_weight"] * int(no_collision)
                 + metric["length_weight"] * length_score + metric["time_weight"] * time_score)
        absolute = np.abs(np.asarray(commands))
        limits = np.asarray([float(suite["sources"]["policy_overrides"][name]) for name in ("max_vx", "max_vy", "max_wz")])
        coverage = float(np.mean(np.min(np.linalg.norm(expected[:, None, :] - actual[None, :, :], axis=2), axis=1) <= 0.35))
        return {**trial, "arrival": arrival, "score": score, "frechet_m": frechet,
                "frechet_normalized": normalized, "reference_coverage": coverage,
                "path_similarity": similarity, "no_collision": no_collision,
                "length_similarity": length_score, "time_similarity": time_score,
                "length_ratio": length_ratio, "time_ratio": time_ratio, "reference_length_m": ref_length,
                "reference_time_s": reference["times"][-1], "reference_episode": reference["episode_id"],
                "reference_run": reference["run_file"], "command_abs_p95": np.quantile(absolute, 0.95, axis=0).tolist(),
                "command_at_limit_fraction": np.mean(absolute >= limits * 0.99, axis=0).tolist()}

    def score_model(self, receipt, phase, profile, validation_only=False):
        candidates = []
        seen_epochs = set()
        for checkpoint in receipt["checkpoints"]:
            epoch = receipt["epochs"][checkpoint]
            if epoch in seen_epochs or (validation_only and Path(checkpoint).name != "best_model.pth"):
                continue
            seen_epochs.add(epoch)
            sessions = [self.evaluate(receipt, checkpoint, phase, scenario, profile) for scenario in self.cfg[phase]]
            candidates.append(aggregate_sessions(sessions, checkpoint))
        return max(candidates, key=lambda r: (r["score"], r["arrival_rate"], -r["collision_rate"]))

    def lanes(self):
        return [(name, environment, representation) for environment in self.cfg["environment_sets"]
                for representation in self.cfg["representations"] for name in [f"{environment}_{representation}"]]

    def baseline_candidate(self):
        return {"recipe": "natural", "train": self.cfg["train"]}

    def evaluate_candidate(self, environment, representation, candidate, profile):
        models = []
        for seed in self.cfg["seeds"]:
            receipt = self.train(environment, representation, seed, candidate)
            models.append(self.score_model(receipt, "screen", profile))
        return {"candidate": candidate, "profile": profile, "models": models,
                "by_class": {category: {metric: statistics.mean(m["by_class"][category][metric] for m in models)
                                       for metric in ("score", "arrival_rate", "collision_rate", "frechet_m")}
                             for category in CLASSES},
                **{name: statistics.mean(m[name] for m in models)
                   for name in ("score", "arrival_rate", "collision_rate", "frechet_m", "frechet_normalized")}}

    def choose_profile(self):
        profiles = {}
        seed = self.cfg["seeds"][0]
        for profile in self.prepared["profiles"]:
            scores = []
            for _, environment, representation in self.lanes():
                receipt = self.train(environment, representation, seed, self.baseline_candidate())
                scores.append(self.score_model(receipt, "screen", profile, validation_only=True))
            profiles[profile] = {name: statistics.mean(m[name] for m in scores)
                                 for name in ("score", "arrival_rate", "collision_rate", "frechet_m")}
        best = "legacy"
        for profile, result in profiles.items():
            if self.improves(result, profiles[best]):
                best = profile
        self.state["profile"] = best
        self.state["profile_comparison"] = profiles
        write_json(self.state_path, self.state)
        self.log("profile_selected", profile=best, limits=self.prepared["profiles"][best])

    def improves(self, candidate, baseline):
        if "models" in candidate:
            scores = {model["seed"]: model["score"] for model in baseline["models"]}
            gains = [model["score"] - scores[model["seed"]] for model in candidate["models"]]
            if sum(gain > 0 for gain in gains) <= len(gains) // 2:
                return False
            for category in ("straight", "semi_complex"):
                if candidate["by_class"][category]["arrival_rate"] < baseline["by_class"][category]["arrival_rate"]:
                    return False
        return (candidate["score"] >= baseline["score"] + self.cfg["minimum_improvement"]
                and candidate["collision_rate"] <= baseline["collision_rate"] + 1e-9
                and candidate["arrival_rate"] >= baseline["arrival_rate"] - 1e-9)

    def proposals(self, stage, incumbent):
        if stage == "dataset":
            return [{**incumbent, "recipe": name} for name in self.cfg["recipes"] if name != incumbent["recipe"]]
        hp = incumbent["train"]
        if stage == "learning_rate":
            changes = [{"learning_rate": hp["learning_rate"] * factor} for factor in (0.5, 2.0)]
        elif stage == "dropout":
            changes = [{"dropout_1": hp["dropout_1"] * 0.5, "dropout_2": hp["dropout_2"] * 0.5}]
        elif stage == "loss":
            changes = [{"action_weights": [1.0, 2.0, 1.0]}]
        else:
            raise ValueError(f"Unknown refinement stage: {stage}")
        return [{**incumbent, "train": {**hp, **change}} for change in changes]

    def run_stage(self, stage):
        if stage in self.state["stages"]:
            self.log("stage_skip", stage=stage)
            return
        index = STAGES.index(stage)
        if "confirm" in self.state["stages"]:
            raise ValueError("Confirmation froze this campaign. Create a new campaign for further tuning.")
        if stage == "confirm" and "baseline" not in self.state["stages"]:
            raise ValueError("Complete baseline before confirmation")
        if stage != "confirm" and index and STAGES[index - 1] not in self.state["stages"]:
            raise ValueError(f"Complete {STAGES[index - 1]} before {stage}")
        self.log("stage_start", stage=stage)
        if stage == "baseline" and "profile" not in self.state:
            self.choose_profile()
        profile = self.state["profile"]
        results = {}
        for lane, environment, representation in self.lanes():
            if stage == "confirm":
                incumbent = self.state["lanes"][lane]
                baseline = self.state["stages"]["baseline"][lane]["winner"]
                results[lane] = {}
                for label, selection in (("baseline", baseline), ("winner", incumbent)):
                    sessions = []
                    for model in selection["models"]:
                        receipt = self.train(environment, representation, model["seed"], selection["candidate"])
                        sessions.extend(self.evaluate(receipt, model["checkpoint"], "confirmation", scenario, profile)
                                        for scenario in self.cfg["confirmation"])
                    results[lane][label] = {**aggregate_sessions(sessions), "candidate": selection["candidate"]}
                continue
            if stage == "baseline":
                winner = self.evaluate_candidate(environment, representation, self.baseline_candidate(), profile)
                comparisons = [winner]
            else:
                winner = self.state["lanes"][lane]
                comparisons = [winner]
                for candidate in self.proposals(stage, winner["candidate"]):
                    result = self.evaluate_candidate(environment, representation, candidate, profile)
                    comparisons.append(result)
                    if self.improves(result, winner):
                        winner = result
            results[lane] = {"winner": winner, "comparisons": comparisons}
        if stage != "confirm":
            self.state["lanes"] = {lane: result["winner"] for lane, result in results.items()}
        self.state["stages"][stage] = results
        write_json(self.state_path, self.state)
        self.log("stage_done", stage=stage)
        self.report()

    def pilot(self):
        candidate = self.baseline_candidate()
        candidate = {**candidate, "train": {**candidate["train"], "epochs": 1, "patience": 1}}
        environment = next(iter(self.cfg["environment_sets"]))
        for representation in self.cfg["representations"]:
            receipt = self.train(environment, representation, self.cfg["seeds"][0], candidate, scope="pilot")
            result = self.score_model(receipt, "screen", "direct", validation_only=True)
            self.log("pilot_done", representation=representation, score=round(result["score"], 4))
        self.report()

    def report(self):
        rows = []
        for path in sorted((self.root / "evaluation").glob("*/*/metrics.json")):
            result = read_json(path)
            spec = result["spec"]
            for trial in result["trials"]:
                rows.append({"key": result["key"], "checkpoint": result["checkpoint"], "epoch": result["epoch"],
                             "environment": spec["environment"], "representation": spec["representation"],
                             "seed": spec["seed"], "recipe": spec["candidate"]["recipe"],
                             "phase": result["phase"], "scenario": result["scenario"], "profile": result["profile"],
                             "learning_rate": spec["candidate"]["train"]["learning_rate"],
                             "dropout_1": spec["candidate"]["train"]["dropout_1"],
                             "dropout_2": spec["candidate"]["train"]["dropout_2"],
                             "action_weights": json.dumps(spec["candidate"]["train"]["action_weights"]),
                             "epoch_limit": spec["candidate"]["train"]["epochs"],
                             **{name: trial[name] for name in ("task_id", "route_class", "reason", "arrival", "collision",
                                  "score", "path_similarity", "no_collision", "time_similarity", "length_similarity",
                                  "frechet_m", "frechet_normalized", "reference_coverage", "length_ratio",
                                  "time_ratio", "final_goal_distance_m", "elapsed_sim_s", "reference_episode", "reference_run")},
                             "trajectory": str(Path(result["output_dir"]) / "trajectory.csv"),
                             "comparison_path": str(Path(result["output_dir"]) / "comparison_paths.csv")})
        if rows:
            path = self.root / "trials.csv"
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        prepared = self.prepared
        if prepared is None and (self.root / "prepared.json").exists():
            prepared = read_json(self.root / "prepared.json")
        lines = ["# Experimentos de navegación 5 × 4 m", "", f"Semillas compartidas: {self.cfg['seeds']}", ""]
        if prepared:
            lines.extend([f"Pares por entrenamiento: {prepared['pair_budget']}",
                          f"Trayectorias grabadas de referencia: {len(prepared['references'])}",
                          f"Auditoría: {prepared['diagnostics']}", ""])
            lines.extend(["| Tarea | Clase | Episodio grabado | Longitud (m) | Tiempo (s) |",
                          "|---|---|---:|---:|---:|"])
            for key, ref in sorted(prepared["references"].items()):
                lines.append(f"| {key} | {ref['route_class']} | {ref['episode_id']} | {ref['length_m']:.2f} | {ref['times'][-1]:.2f} |")
            lines.append("")
        if "profile" in self.state:
            lines.extend([f"Perfil seleccionado: {self.state['profile']}", "",
                          "| Perfil | Puntaje | Llegadas | Colisiones | Fréchet (m) |",
                          "|---|---:|---:|---:|---:|"])
            for profile, score in self.state["profile_comparison"].items():
                lines.append(f"| {profile} | {score['score']:.4f} | {score['arrival_rate']:.1%} | "
                             f"{score['collision_rate']:.1%} | {score['frechet_m']:.3f} |")
            lines.append("")
        for stage, lanes in self.state["stages"].items():
            lines.extend([f"**{stage}**", "", "| Grupo | Receta | LR | Puntaje | Llegadas | Colisiones | Fréchet (m) |",
                          "|---|---|---:|---:|---:|---:|---:|"])
            for lane, result in lanes.items():
                entries = [(f"{lane} {label}", result[label]) for label in ("baseline", "winner")] if stage == "confirm" else [(lane, result["winner"])]
                for label, score in entries:
                    candidate = score["candidate"]
                    lines.append(f"| {label} | {candidate['recipe']} | {candidate['train']['learning_rate']:.6g} | "
                                 f"{score['score']:.4f} | {score['arrival_rate']:.1%} | {score['collision_rate']:.1%} | {score['frechet_m']:.3f} |")
            lines.append("")
        lines.extend([f"Pruebas registradas en `trials.csv`: {len(rows)}", ""])
        path = self.root / "report.md"
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines))
        temporary.replace(path)
        print(f"Report: {path}")


def to_map_points(points, offsets):
    x, y, yaw = offsets
    c, s = math.cos(yaw), math.sin(yaw)
    return (np.asarray(points, dtype=float) - [x, y]) @ np.array([[c, -s], [s, c]])


def command_adjustment(path, vmax):
    """Summarize raw and published values sampled by the policy log."""
    if not path.is_file():
        return {"samples": 0}
    pattern = re.compile(r"\braw=\(([^)]+)\)\s*\|\s*cmd=\(([^)]+)\)")
    samples = []
    for raw, command in pattern.findall(path.read_text(errors="replace")):
        try:
            sample = [[float(value) for value in entry.split(",")] for entry in (raw, command)]
        except ValueError:
            continue
        if len(sample[0]) == len(sample[1]) == 3 and np.isfinite(sample).all():
            samples.append(sample)
    if not samples:
        return {"samples": 0}
    values = np.abs(np.asarray(samples))
    expected, published = values[:, 0, :] * np.asarray(vmax), values[:, 1, :]
    ratios = {}
    for i, name in enumerate(("vx", "vy", "wz")):
        mask = expected[:, i] > (0.05 if name == "wz" else 0.02)
        ratios[name] = float(np.median(published[mask, i] / expected[mask, i])) if mask.any() else None
    return {"samples": len(samples), "median_published_to_decoded": ratios}


def aggregate_sessions(sessions, checkpoint=None):
    trials = [trial for session in sessions for trial in session["trials"]]
    if not trials:
        raise ValueError("No completed tasks to score")
    first = sessions[0]
    by_class = {}
    for category in CLASSES:
        group = [trial for trial in trials if trial["route_class"] == category]
        if group:
            by_class[category] = {"trials": len(group), "score": statistics.mean(t["score"] for t in group),
                                  "arrival_rate": statistics.mean(t["arrival"] for t in group),
                                  "collision_rate": statistics.mean(t["collision"] for t in group),
                                  "frechet_m": statistics.mean(t["frechet_m"] for t in group)}
    return {"checkpoint": checkpoint, "seed": first["spec"]["seed"] if checkpoint else None,
            "epoch": first["epoch"] if checkpoint else None, "trials": len(trials),
            "by_class": by_class,
            "score": statistics.mean(t["score"] for t in trials),
            "arrival_rate": statistics.mean(t["arrival"] for t in trials),
            "collision_rate": statistics.mean(t["collision"] for t in trials),
            "frechet_m": statistics.mean(t["frechet_m"] for t in trials),
            "frechet_normalized": statistics.mean(t["frechet_normalized"] for t in trials)}


def confirmed_ranking(original):
    if "confirm" not in original.state["stages"]:
        raise ValueError("Complete the original confirmation before running comparisons")
    profile = original.state["profile"]
    scenarios = set(original.cfg["confirmation"])
    report_source = original.root / "trials.csv"
    groups = collections.defaultdict(list)
    with report_source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["phase"] == "confirmation" and row["profile"] == profile:
                groups[row["key"], row["checkpoint"]].append(row)

    def difficult(rows):
        return [row for row in rows if row["route_class"] in ("complex", "v_complex")]

    def mean_score(rows):
        return statistics.mean(float(row["score"]) for row in rows)

    for rows in groups.values():
        if {row["scenario"] for row in rows} != scenarios or not difficult(rows):
            raise ValueError("Confirmation data must include every scenario and difficult tasks")
        if len({(row["scenario"], row["task_id"]) for row in rows}) != len(rows):
            raise ValueError("Confirmation data contains repeated tasks for a checkpoint")
    ranked = sorted(groups, key=lambda item: (-mean_score(difficult(groups[item])),
                                              -mean_score(groups[item]), item[1]))
    if len(ranked) < 6:
        raise ValueError("Expected at least six confirmed checkpoints")

    return ranked, groups


def repeat_scale(original, limit):
    ranked, groups = confirmed_ranking(original)
    profile = original.state["profile"]
    report_source = original.root / "trials.csv"

    def difficult(rows):
        return [row for row in rows if row["route_class"] in ("complex", "v_complex")]

    def mean_score(rows):
        return statistics.mean(float(row["score"]) for row in rows)

    root = original.root.with_name(original.root.name + "_occ100")
    root.mkdir(exist_ok=True)
    plan = {"source": str(original.root), "trials_sha256": file_hash(report_source),
            "configuration_sha256": digest(original.cfg), "profile": profile,
            "code_sha256": original.source_signature(), "checkpoints": [list(item) for item in ranked[:6]]}
    plan_path = root / "scale_plan.json"
    previous_code = {**plan["code_sha256"], "navigation_campaign.py":
                     "8b5d384ff0d99a25cab6baa0f1962d58712fa2d2e0139b9ef0509d0edd223f90"}
    if plan_path.exists() and read_json(plan_path) not in (plan, {**plan, "code_sha256": previous_code}):
        raise ValueError(f"Scale comparison inputs changed: {plan_path}")
    if not plan_path.exists():
        if any(root.iterdir()):
            raise ValueError(f"Scale comparison directory already exists: {root}")
        write_json(plan_path, plan)
    write_json(root / "config.json", {**original.cfg, "campaign_dir": str(root)})
    branch = Campaign(root)
    branch.prepared = original.prepared
    write_json(root / "datasets_manifest/datasets.json",
               read_json(original.root / "datasets_manifest/datasets.json"))

    def line(rows):
        def count(name):
            return sum(str(row[name]).lower() in ("true", "1") for row in rows)

        n = len(rows)
        frechet = statistics.mean(float(row["frechet_m"]) for row in rows)
        return f"{mean_score(rows):.4f} | {count('arrival')}/{n} | {count('collision')}/{n} | {frechet:.3f}"

    lines = ["# Comparación de escala de ocupación", "",
             "| Candidato | Escala | Época | Rutas | Puntaje | Llegadas | Colisiones | Fréchet (m) |",
             "|---:|---:|---:|---|---:|---:|---:|---:|"]
    for rank, (key, checkpoint) in enumerate(ranked[:limit], start=1):
        old_rows = groups[key, checkpoint]
        saved = read_json(original.root / "training" / key / "completed.json")
        if file_hash(checkpoint) != saved["checkpoints"][checkpoint]:
            raise ValueError(f"The original checkpoint changed: {checkpoint}")
        spec = saved["spec"]
        environment, representation, seed = (spec[name] for name in ("environment", "representation", "seed"))
        candidate = spec["candidate"]
        old_manifest = original.root / "pairs" / environment / f"{candidate['recipe']}_{seed}.json"
        if file_hash(old_manifest) != spec["pairs_sha256"]:
            raise ValueError(f"The original pair manifest changed: {old_manifest}")
        manifest = branch.write_pairs(environment, seed, candidate["recipe"])
        if file_hash(manifest) != spec["pairs_sha256"]:
            raise ValueError(f"Training pairs differ from the original manifest: {manifest}")

        print(f"Candidate {rank}/6: {key} ({Path(checkpoint).name})", flush=True)
        new_candidate = {**candidate, "train": {**candidate["train"], "occupancy_max": 100}}
        new = branch.train(environment, representation, seed, new_candidate)
        new_checkpoint = next(path for path in new["checkpoints"] if Path(path).name == Path(checkpoint).name)
        cfg = read_json(Path(new_checkpoint).with_name("config.json"))
        if cfg.get("normalize_occupancy") is not False or cfg.get("occupancy_max") != 100:
            raise ValueError(f"Checkpoint metadata does not specify raw 0/100: {new_checkpoint}")

        trials = []
        for scenario in original.cfg["confirmation"]:
            session = branch.evaluate(new, new_checkpoint, "confirmation", scenario, profile)
            trials.extend({**trial, "scenario": scenario} for trial in session["trials"])
        identities = {(row["scenario"], str(row["task_id"]), row["route_class"]) for row in trials}
        expected = {(row["scenario"], row["task_id"], row["route_class"]) for row in old_rows}
        if identities != expected or len(trials) != len(old_rows):
            raise ValueError(f"The confirmation tasks differ for {key}")
        lines.extend([
            f"| {rank} | 0/1 | {saved['epochs'][checkpoint]} | Todas | {line(old_rows)} |",
            f"| {rank} | 0/100 | {new['epochs'][new_checkpoint]} | Todas | {line(trials)} |",
            f"| {rank} | 0/1 | {saved['epochs'][checkpoint]} | Difíciles | {line(difficult(old_rows))} |",
            f"| {rank} | 0/100 | {new['epochs'][new_checkpoint]} | Difíciles | {line(difficult(trials))} |",
        ])
        comparison = root / "comparacion_escala.md"
        temporary = comparison.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(comparison)
        branch.report()
    print(f"Scale comparison: {comparison}")


def temporal_inference_copy(receipt, checkpoint):
    import torch

    source = Path(checkpoint)
    destination = source.parent / "temporal_inference"
    destination.mkdir(exist_ok=True)
    model_path = destination / source.name
    provenance = destination / "source.json"
    original_hash = receipt["checkpoints"][checkpoint]
    if provenance.exists():
        saved = read_json(provenance)
        if saved["original_sha256"] != original_hash or saved["sha256"] != file_hash(model_path):
            raise ValueError(f"The temporal inference checkpoint changed: {model_path}")
    else:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        config = payload["config"]
        sidecar = read_json(source.with_name("config.json"))
        if config.get("representation") != "lom" or digest(sidecar) != digest(config):
            raise ValueError(f"Expected matching LOM checkpoint metadata: {source}")
        temporal = {**config, "representation": "tlom"}
        temporary = model_path.with_suffix(".pth.tmp")
        torch.save({**payload, "config": temporal}, temporary)
        temporary.replace(model_path)
        write_json(destination / "config.json", temporal)
        saved = {"original_sha256": original_hash, "sha256": file_hash(model_path),
                 "config_sha256": file_hash(destination / "config.json")}
        write_json(provenance, saved)
    if saved["config_sha256"] != file_hash(destination / "config.json"):
        raise ValueError(f"The temporal inference metadata changed: {destination}")
    path = str(model_path)
    crossed = {**receipt, "key": receipt["key"] + "_infer_tlom",
               "checkpoints": {path: saved["sha256"]},
               "epochs": {path: receipt["epochs"][checkpoint]}}
    return crossed, path


def run_factorial(original, limit):
    ranked, _ = confirmed_ranking(original)
    profile = original.state["profile"]
    root = original.root.with_name(original.root.name + "_factorial")
    root.mkdir(exist_ok=True)
    plan_path = root / "factorial_plan.json"
    provenance = {"source": str(original.root), "profile": profile,
                  "code_sha256": original.source_signature(),
                  "prepared_sha256": file_hash(original.root / "prepared.json"),
                  "trials_sha256": file_hash(original.root / "trials.csv")}
    templates = []
    for rank, (key, checkpoint) in enumerate(ranked[:6], start=1):
        receipt = read_json(original.root / "training" / key / "completed.json")
        if file_hash(checkpoint) != receipt["checkpoints"][checkpoint]:
            raise ValueError(f"The original checkpoint changed: {checkpoint}")
        templates.append({"rank": rank, "source_key": key, "checkpoint": Path(checkpoint).name,
                          "original_sha256": receipt["checkpoints"][checkpoint], "spec": receipt["spec"]})
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["provenance"] != provenance or [x["template"] for x in plan["blocks"]] != templates:
            raise ValueError(f"Factorial experiment inputs changed: {plan_path}")
    else:
        if any(root.iterdir()):
            raise ValueError(f"Factorial experiment directory already exists: {root}")
        reserved = set(original.cfg["seeds"]) | {x["spec"]["seed"] for x in templates}
        rng = random.SystemRandom()
        blocks = []
        for template in templates:
            seed = rng.randrange(1, 2**31)
            while seed in reserved:
                seed = rng.randrange(1, 2**31)
            reserved.add(seed)
            blocks.append({"template": template, "seed": seed})
        plan = {"provenance": provenance, "blocks": blocks}
        write_json(plan_path, plan)
    write_json(root / "config.json", {**original.cfg, "campaign_dir": str(root),
                                    "seeds": [x["seed"] for x in plan["blocks"]]})
    branch = Campaign(root)
    branch.prepared = original.prepared
    write_json(root / "datasets_manifest/datasets.json",
               read_json(original.root / "datasets_manifest/datasets.json"))

    trained = []
    for scale in (100, 1):
        for block in plan["blocks"][:limit]:
            template = block["template"]
            spec = template["spec"]
            seed = block["seed"]
            candidate = spec["candidate"]
            candidate = {**candidate, "train": {**candidate["train"], "occupancy_max": scale}}
            for representation in ("lom", "tlom"):
                print(f"Finalist {template['rank']}: seed={seed} train={representation} scale={scale}", flush=True)
                receipt = branch.train(spec["environment"], representation, seed, candidate)
                checkpoint = next(path for path in receipt["checkpoints"] if Path(path).name == template["checkpoint"])
                config = read_json(Path(checkpoint).with_name("config.json"))
                normalized = config.get("normalize_occupancy")
                if (config.get("representation") != representation or normalized is not (scale == 1)
                        or (scale == 100 and config.get("occupancy_max") != 100)):
                    raise ValueError(f"The checkpoint does not match its training condition: {checkpoint}")
                trained.append((block, scale, representation, receipt, checkpoint))

    rows = []
    report_path = root / "comparacion_factorial.md"
    for crossed in (False, True):
        for block, scale, representation, receipt, checkpoint in trained:
            if crossed and representation != "lom":
                continue
            inference = "tlom" if crossed else representation
            if crossed:
                receipt, checkpoint = temporal_inference_copy(receipt, checkpoint)
            for scenario in original.cfg["confirmation"]:
                session = branch.evaluate(receipt, checkpoint, "confirmation", scenario, profile)
                for trial in session["trials"]:
                    rows.append({"finalist": block["template"]["rank"], "seed": block["seed"], "scale": scale,
                                 "training": representation, "inference": inference, "scenario": scenario,
                                 "checkpoint": checkpoint, "epoch": receipt["epochs"][checkpoint],
                                 **{name: trial[name] for name in ("task_id", "route_class", "score", "arrival",
                                                                 "collision", "frechet_m", "elapsed_sim_s",
                                                                 "final_goal_distance_m")}})
            write_factorial_report(report_path, rows, plan)
    print(f"Factorial comparison: {report_path}")


def write_factorial_report(path, rows, plan):
    csv_path = path.with_name("trials_factorial.csv")
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    conditions = collections.defaultdict(list)
    for row in rows:
        conditions[row["finalist"], row["scale"], row["training"], row["inference"]].append(row)
        conditions[0, row["scale"], row["training"], row["inference"]].append(row)
    lines = ["# Comparación de escala y representación", "",
             "Las semillas se comparten entre las cuatro variantes de cada finalista.", "",
             "| Finalista | Semilla nueva | Conjunto | Receta | Criterio |",
             "|---:|---:|---|---|---|"]
    for block in plan["blocks"]:
        template = block["template"]
        spec = template["spec"]
        lines.append(f"| {template['rank']} | {block['seed']} | {spec['environment']} | "
                     f"{spec['candidate']['recipe']} | {template['checkpoint']} |")
    lines.extend(["", "| Finalista | Escala | Entrena | Infiere | Rutas | Puntaje | Llegadas | Colisiones | Fréchet (m) |",
                  "|---:|---:|---|---|---|---:|---:|---:|---:|"])
    for (rank, scale, train, inference), items in sorted(conditions.items()):
        difficult = [x for x in items if x["route_class"] in ("complex", "v_complex")]
        for name, group in (("Todas", items), ("Difíciles", difficult)):
            if not group:
                continue
            n = len(group)
            score = statistics.mean(x["score"] for x in group)
            frechet = statistics.mean(x["frechet_m"] for x in group)
            arrivals = sum(x["arrival"] for x in group)
            collisions = sum(x["collision"] for x in group)
            label = rank or "Total"
            lines.append(f"| {label} | 0/{scale} | {train} | {inference} | {name} | "
                         f"{score:.4f} | {arrivals}/{n} | {collisions}/{n} | {frechet:.3f} |")
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a campaign and record fresh random seeds")
    init.add_argument("--campaign-dir", type=Path, required=True)
    init.add_argument("--data-root", type=Path, default=DATASET / "data")
    init.add_argument("--seeds", type=int, default=4)
    init.add_argument("--environments", nargs="+", choices=tuple(ENVIRONMENTS), default=list(ENVIRONMENTS))
    init.add_argument("--pair-budget", type=int, default=10000)
    init.add_argument("--reference-frame", choices=("map", "world"), default="map")
    for command in ("prepare", "pilot", "run", "status", "report", "repeat-scale", "factorial"):
        sub = commands.add_parser(command)
        sub.add_argument("--campaign-dir", type=Path, required=True)
        if command in ("repeat-scale", "factorial"):
            sub.add_argument("--limit", type=int, choices=range(1, 7), default=6)
        if command == "run":
            sub.add_argument("--stage", choices=(*STAGES, "all"), default="baseline")
    args = parser.parse_args()
    try:
        if args.command == "init":
            if not 1 <= args.seeds <= 64 or args.pair_budget < 100:
                raise ValueError("Use one to 64 seeds and a pair budget of at least 100")
            initialize(args)
            return
        campaign = Campaign(args.campaign_dir)
        if args.command == "status":
            print(f"Completed stages: {', '.join(campaign.state['stages']) or 'none'}")
            print(f"Control profile: {campaign.state.get('profile', 'pending')}")
            event_log = campaign.root / "events.jsonl"
            if event_log.exists():
                print("\n".join(event_log.read_text().splitlines()[-8:]))
            return
        with (campaign.root / ".campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "report":
                campaign.report()
            elif args.command == "prepare":
                campaign.prepare()
            else:
                missing = [command for command in ("ros2", "webots") if shutil.which(command) is None]
                if missing:
                    raise ValueError(f"Missing executables: {', '.join(missing)}. Source install/setup.bash.")
                campaign.load_prepared()
                if args.command == "pilot":
                    campaign.pilot()
                elif args.command == "repeat-scale":
                    repeat_scale(campaign, args.limit)
                elif args.command == "factorial":
                    run_factorial(campaign, args.limit)
                else:
                    for stage in STAGES if args.stage == "all" else [args.stage]:
                        campaign.run_stage(stage)
    except BlockingIOError:
        parser.exit(2, "ERROR: This campaign is already active\n")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Completed attempts remain available.\n")


if __name__ == "__main__":
    main()
