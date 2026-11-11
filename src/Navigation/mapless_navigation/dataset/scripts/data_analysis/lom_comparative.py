#!/usr/bin/env python3
"""LOM comparative campaign orchestrator.

Runs the full training and evaluation grid over:
    - datasets (conjuntos): T1, T2
    - variants:             4x4, 5x4, 7x6
    - representations:      lom, tlom
    - seeds:                30 fixed seeds from seeds.json

Training uses the same seed across the six variants, so the comparison
isolates the effect of variant (grid geometry) and representation
(instantaneous vs temporal).

Evaluation is split in two stages:
    - Filter stage: all 360 checkpoints evaluated on a fixed task set
      in E08 with 4 goals (one per route class). Results are grouped
      by (dataset, variant, representation); the top-K per group are
      promoted to the finalist stage.
    - Finalist stage: promoted checkpoints are evaluated on the
      fixed task sets listed in the config.

Resume completed runs with the same config and task sets. Incomplete
evaluation attempts are kept beside the new attempt.

Config file (JSON, required) at --config. See the docstring of
campaign_config example in the project wiki.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent.parent
SCRIPTS_ROOT = DATASET_DIR.parent
MAPLESS_SCRIPTS = SCRIPTS_ROOT / "mapless_nav" / "scripts"
EVAL_SCRIPT = MAPLESS_SCRIPTS / "evaluate_navigation.py"
TRAIN_SCRIPT = MAPLESS_SCRIPTS / "TorchModels" / "train.py"
NEAR_GOAL_M = 0.35

sys.path.insert(0, str(MAPLESS_SCRIPTS))
import evaluate_navigation as ev  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def combo_key(conjunto, seed, variant, representation):
    return f"{conjunto}_{variant}_{representation}_seed{seed}"


def combo_spec(conjunto, seed, variant, representation, cfg):
    return {
        "conjunto": conjunto,
        "seed": seed,
        "variant": variant,
        "representation": representation,
        "datasets": cfg["datasets"][conjunto],
        "train": cfg["train"],
    }


class Runner:
    def __init__(self, cfg, dry_run=False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.campaign_dir = Path(cfg["campaign_dir"]).resolve()
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(cfg["data_root"]).resolve()
        self.pool_sets_dir = Path(cfg["pool_sets_dir"]).resolve()
        self.train_dir = self.campaign_dir / "train"
        self.eval_dir = self.campaign_dir / "eval"
        self.task_sets_cache = self.campaign_dir / "task_sets_cache"
        self.logs_dir = self.campaign_dir / "logs"
        self.train_dir.mkdir(exist_ok=True)
        self.eval_dir.mkdir(exist_ok=True)
        self.task_sets_cache.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.suites = {}
        self.manifest_fh = open(self.campaign_dir / "manifest.jsonl", "a")
        self.seeds = self.load_seeds()
        self.datasets_manifest = self.ensure_datasets_manifest()

    def log(self, event, **fields):
        record = {"timestamp": time.time(), "event": event, **fields}
        self.manifest_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.manifest_fh.flush()
        if not self.dry_run:
            summary = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[{event}] {summary}", flush=True)

    def load_seeds(self):
        path = self.campaign_dir / "seeds.json"
        count = int(self.cfg["n_seeds"])
        if count < 1:
            raise ValueError("n_seeds must be positive")
        if path.exists():
            seeds = json.loads(path.read_text())["seeds"]
            if (not isinstance(seeds, list) or len(seeds) != count
                    or any(type(seed) is not int or not 1 <= seed < 2**31 for seed in seeds)
                    or len(set(seeds)) != count):
                raise ValueError(f"Expected {count} distinct seeds in {path}")
            return seeds
        rng = random.Random(self.cfg["seed_generator"])
        seeds = rng.sample(range(1, 2**31), count)
        if not self.dry_run:
            path.write_text(json.dumps({
                "seed_generator": self.cfg["seed_generator"],
                "n_seeds": count,
                "seeds": seeds,
            }, indent=2))
        return seeds

    def ensure_datasets_manifest(self):
        """Write the datasets.json that evaluate_navigation.py expects."""
        manifest_dir = self.campaign_dir / "datasets_manifest"
        manifest_dir.mkdir(exist_ok=True)
        path = manifest_dir / "datasets.json"
        env_sets = {name: spec["train"] for name, spec in self.cfg["datasets"].items()}
        payload = {
            "phase": "datasets",
            "schema_version": 1,
            "config": {
                "environment_sets": env_sets,
                "final_scenarios": ["E04", "E05", "E06", "E10", "E19", "E23", "E24", "E29"],
            },
        }
        if not path.exists() or json.loads(path.read_text()) != payload:
            path.write_text(json.dumps(payload, indent=2))
        return manifest_dir

    def ensure_full_task_set(self, scenario, task_set_name):
        """Use the same tasks and world seed for all models."""
        key = (scenario, task_set_name)
        if key in self.suites:
            return self.suites[key]
        if not task_set_name.startswith(scenario + "_"):
            raise ValueError(f"Task set {task_set_name!r} does not belong to {scenario}")
        task_set_path = self.pool_sets_dir / f"{task_set_name}_tasks.json"
        suite = self.build_full_task_set(scenario, task_set_path)
        cached = self.task_sets_cache / f"{task_set_name}_suite.json"
        if not cached.exists() or json.loads(cached.read_text()) != suite:
            if not self.dry_run:
                ev.control.write_json(cached, suite)
                self.log("task_set_built", scenario=scenario, task_set=task_set_name, path=str(cached))
        self.suites[key] = (cached, suite)
        return cached, suite

    def build_full_task_set(self, scenario, task_set_path):
        """Add runtime metadata to the fixed tasks."""
        import yaml
        world = ev.HARDWARE / "worlds" / (scenario + ".wbt")
        background = ev.control.load_review_map(ev.HARDWARE / "environments/maps", scenario, "prohibition")
        map_metadata = yaml.safe_load(Path(background["source"]["yaml"]).read_text())
        offsets = [float(map_metadata.get(f"world_offset_{a}", 0)) for a in ("x", "y", "yaw")]

        task_set = json.loads(task_set_path.read_text())
        if task_set["scenario"] != scenario:
            raise ValueError(f"Task set {task_set_path} has the wrong scenario")
        seed = int(task_set["seed"])
        settings = ev.evaluation_settings("7x6")
        return {
            "version": 1,
            "scenario": scenario,
            "purpose": "development",
            "seed": seed,
            "sources": {
                "world_sha256": ev.control.file_hash(world),
                "map_yaml_sha256": background["source"]["yaml_sha256"],
                "map_image_sha256": background["source"]["image_sha256"],
                "runtime": settings,
            },
            "offsets": offsets,
            "map_source": background["source"],
            "heading_policy": "toward_goal",
            "generator_sha256": sha256_file(task_set_path),
            "pool_complete": True,
            "tasks": task_set["tasks"],
            "protocol": {
                "timeout_s": 60.0,
                "goal_tolerance_m": 0.25,
                "settle_s": 1.5,
                "start_tolerance_m": 0.08,
                "start_yaw_tolerance_rad": 0.12,
                "collision_threshold": 0.5,
                "collision_source": "world collision_detector TouchSensor",
                "pose_source": "Webots Supervisor getSelf position and orientation",
                "input_monitoring": "diagnostic_only",
                "input_diagnostic_age_wall_s": 2.0,
                "world_random_seed": max(1, int(seed % 2147483647)),
                "simulation_mode": "realtime",
            },
        }

    def run_training(self, spec):
        key = combo_key(spec["conjunto"], spec["seed"], spec["variant"], spec["representation"])
        out_dir = self.train_dir / key
        combo_path = out_dir / "campaign_combo.json"
        expected_hash = sha256_json({"spec": spec, "data_root": str(self.data_root),
                                     "trainer_sha256": self.trainer_sha256})

        if combo_path.exists():
            try:
                existing = json.loads(combo_path.read_text())
                if existing.get("combo_hash") == expected_hash and (out_dir / "best_model.pth").exists():
                    self.log("train_skip", key=key)
                    return out_dir / "best_model.pth"
            except Exception:
                pass

        if self.dry_run:
            self.log("train_dry_run", key=key)
            return out_dir / "best_model.pth"

        # train.py creates out_dir itself with exist_ok=False.
        # Remove any leftover from a previous failed run so it can start clean.
        if out_dir.exists():
            shutil.rmtree(out_dir)

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-root", str(self.data_root),
            "--data-format", "ros2_episode",
            "--variant", spec["variant"],
            "--representation", spec["representation"],
            "--train-scenarios", ",".join(spec["datasets"]["train"]),
            "--validation-scenarios", ",".join(spec["datasets"]["validation"]),
            "--epochs", str(spec["train"]["epochs"]),
            "--patience", str(spec["train"]["patience"]),
            "--sequence-length", str(spec["train"]["sequence_length"]),
            "--hidden-size", str(spec["train"]["hidden_size"]),
            "--num-layers", str(spec["train"]["num_layers"]),
            "--batch-size", str(spec["train"]["batch_size"]),
            "--learning-rate", str(spec["train"]["learning_rate"]),
            "--weight-decay", str(spec["train"]["weight_decay"]),
            "--architecture", spec["train"]["architecture"],
            "--dropout-1", str(spec["train"]["dropout_1"]),
            "--dropout-2", str(spec["train"]["dropout_2"]),
            "--seed", str(spec["seed"]),
            "--output-dir", str(out_dir),
        ]
        log_path = self.logs_dir / f"{key}_train.log"
        self.log("train_start", key=key, log=str(log_path))
        started = time.time()
        with open(log_path, "w") as fh:
            result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        if result.returncode != 0:
            self.log("train_fail", key=key, returncode=result.returncode, elapsed_s=elapsed)
            raise RuntimeError(f"Training failed: {key} (see {log_path})")
        ckpt = out_dir / "best_model.pth"
        if not ckpt.exists():
            self.log("train_fail", key=key, reason="no_checkpoint", elapsed_s=elapsed)
            raise RuntimeError(f"Training produced no checkpoint: {key}")
        combo_path.write_text(json.dumps({"combo_hash": expected_hash, "spec": spec}, indent=2))
        self.log("train_done", key=key, elapsed_s=elapsed, checkpoint=str(ckpt))
        return ckpt

    def run_evaluation(self, spec, ckpt, scenario, task_set_name, level, limit, rviz, close):
        key = combo_key(spec["conjunto"], spec["seed"], spec["variant"], spec["representation"])
        eval_dir = self.eval_dir / key / f"{level}_{scenario}"
        suite_path, suite = self.ensure_full_task_set(scenario, task_set_name)
        expected_ids = [task["id"] for task in suite["tasks"][:limit]]
        checkpoint_hash = sha256_file(ckpt)
        summary = self.completed_evaluation(eval_dir, suite, expected_ids, checkpoint_hash)
        if summary is not None:
            self.log("eval_skip", key=key, scenario=scenario, level=level)
            return summary

        if eval_dir.exists():
            archive = eval_dir.with_name(f"{eval_dir.name}_attempt_{time.time_ns()}")
            eval_dir.rename(archive)
            self.log("eval_preserved", key=key, path=str(archive))

        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--datasets-dir", str(self.datasets_manifest),
            "--checkpoint", str(ckpt),
            "--scenario", scenario,
            "--purpose", "development",
            "--suite", str(suite_path),
            "--limit", str(limit),
            "--output-dir", str(eval_dir),
        ]
        if not rviz:
            cmd.append("--no-rviz")
        if close:
            cmd.append("--close-after-trial")

        log_path = self.logs_dir / f"{key}_{level}_{scenario}.log"
        self.log("eval_start", key=key, scenario=scenario, level=level, log=str(log_path))
        started = time.time()
        with open(log_path, "a") as fh:
            result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        summary = self.completed_evaluation(eval_dir, suite, expected_ids, checkpoint_hash)
        if result.returncode != 0 or summary is None:
            self.log("eval_fail", key=key, scenario=scenario, level=level,
                     returncode=result.returncode, elapsed_s=elapsed)
            raise RuntimeError(f"Evaluation failed: {key} {scenario} {level} (see {log_path})")

        self.log("eval_done", key=key, scenario=scenario, level=level,
                 elapsed_s=elapsed,
                 success_rate=summary.get("success_rate"),
                 collision_rate=summary.get("collision_rate"))
        return summary

    def completed_evaluation(self, eval_dir, suite, expected_ids, checkpoint_hash):
        try:
            summary = json.loads((eval_dir / "summary.json").read_text())
            if (summary["status"] != "completed" or summary["infrastructure_errors"]
                    or summary["completed_trials"] != len(expected_ids)
                    or summary["selected_task_ids"] != expected_ids
                    or summary["model"]["sha256"] != checkpoint_hash
                    or [result["task_id"] for result in summary["results"]] != expected_ids):
                return None
            runtime = json.loads((eval_dir / "runtime.json").read_text())
            if any(runtime.get(key) != value for key, value in self.evaluation_hashes.items()):
                return None
            suite_path = eval_dir / "suite.json"
            if json.loads(suite_path.read_text()) != suite or sha256_file(suite_path) != summary["suite_sha256"]:
                return None
            with (eval_dir / "trajectory.csv").open(newline="") as fh:
                reader = csv.DictReader(fh)
                if not {"task_id", "simulation_time", "world_x", "world_y", "world_yaw"} <= set(reader.fieldnames or []):
                    return None
                if {int(row["task_id"]) for row in reader} != set(expected_ids):
                    return None
        except (OSError, KeyError, TypeError, ValueError, csv.Error):
            return None
        summary["near_goal_rate"] = sum(
            not result["collision"] and result["final_goal_distance_m"] <= NEAR_GOAL_M
            for result in summary["results"]
        ) / len(expected_ids)
        return summary

    def check_evaluation_plan(self, config):
        eval_plan = [
            (config["filter_scenario"], config["filter_task_set"],
             config["filter_limit"], config["filter_rviz"], config["filter_close_after_trial"]),
            *((scenario, config["finalist_task_sets"][scenario], config["finalist_limit"],
               config["finalist_rviz"], config["finalist_close_after_trial"])
              for scenario in config["finalist_scenarios"]),
        ]
        used_scenarios = {
            scenario for dataset in self.cfg["datasets"].values()
            for scenario in (*dataset["train"], *dataset["validation"])
        }
        reserved = set(json.loads((self.datasets_manifest / "datasets.json").read_text())["config"]["final_scenarios"])
        if used_scenarios & reserved:
            raise ValueError(f"Reserved final scenarios in training or validation: {sorted(used_scenarios & reserved)}")
        for scenario, name, limit, rviz, close in eval_plan:
            if scenario in used_scenarios | reserved:
                raise ValueError(f"{scenario} must be held out for development evaluation")
            if rviz and not close:
                raise ValueError(f"Enable close_after_trial for unattended evaluation of {name}")
            _, suite = self.ensure_full_task_set(scenario, name)
            ids = [task["id"] for task in suite["tasks"]]
            if not 1 <= limit <= len(ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid trial count or duplicate task IDs in {name}")

    def write_summary(self, rows):
        path = self.campaign_dir / "summary.csv"
        if not rows:
            return
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.log("summary_written", path=str(path), n_rows=len(rows))

    def run_campaign(self):
        specs = []
        for conjunto in self.cfg["datasets"]:
            for seed in self.seeds:
                for variant in self.cfg["variants"]:
                    for representation in self.cfg["representations"]:
                        specs.append(combo_spec(conjunto, seed, variant, representation, self.cfg))

        eval_cfg = self.cfg["evaluation"]
        self.check_evaluation_plan(eval_cfg)
        if self.dry_run:
            print(f"Planned training runs: {len(specs)} ({len(self.seeds)} seeds)")
            return

        self.trainer_sha256 = sha256_json({
            path.name: sha256_file(path)
            for path in (TRAIN_SCRIPT, TRAIN_SCRIPT.with_name("dataset_utils.py"), TRAIN_SCRIPT.with_name("models.py"))
        })
        self.evaluation_hashes = {
            "evaluator_sha256": sha256_file(EVAL_SCRIPT),
            "policy_source_sha256": sha256_file(MAPLESS_SCRIPTS / "RNN_mapless.py"),
            "model_definition_sha256": sha256_file(TRAIN_SCRIPT.with_name("models.py")),
        }

        self.log("campaign_start", n_combinations=len(specs), campaign_dir=str(self.campaign_dir))

        # Train all model variants.
        checkpoints = {}
        for i, spec in enumerate(specs, 1):
            key = combo_key(spec["conjunto"], spec["seed"], spec["variant"], spec["representation"])
            print(f"\n=== Training {i}/{len(specs)}: {key} ===", flush=True)
            checkpoints[key] = self.run_training(spec)

        # Evaluate every model on the same filter tasks.
        filter_results = {}
        for i, spec in enumerate(specs, 1):
            key = combo_key(spec["conjunto"], spec["seed"], spec["variant"], spec["representation"])
            print(f"\n=== Filter {i}/{len(specs)}: {key} ===", flush=True)
            filter_results[key] = self.run_evaluation(
                spec, checkpoints[key],
                scenario=eval_cfg["filter_scenario"],
                task_set_name=eval_cfg["filter_task_set"],
                level="filter",
                limit=eval_cfg["filter_limit"],
                rviz=eval_cfg["filter_rviz"],
                close=eval_cfg["filter_close_after_trial"],
            )

        # Select finalists within each dataset and input variant.
        finalist_keys = set()
        top_k = int(eval_cfg["finalist_top_k"])
        threshold = float(eval_cfg["finalist_threshold_success"])
        for conjunto in self.cfg["datasets"]:
            for variant in self.cfg["variants"]:
                for representation in self.cfg["representations"]:
                    group = []
                    for spec in specs:
                        if (spec["conjunto"] == conjunto
                                and spec["variant"] == variant
                                and spec["representation"] == representation):
                            key = combo_key(conjunto, spec["seed"], variant, representation)
                            result = filter_results[key]
                            rate = result["near_goal_rate"]
                            if rate >= threshold:
                                group.append((-rate, result["collision_rate"], -result["success_rate"], key))
                    group.sort()
                    finalist_keys.update(row[-1] for row in group[:top_k])

        self.log("finalists_selected",
                 n_finalists=len(finalist_keys),
                 top_k=top_k,
                 threshold=threshold,
                 keys=sorted(finalist_keys))

        # Evaluate finalists on the configured task sets.
        finalist_results = {}
        for key in sorted(finalist_keys):
            spec = next(s for s in specs
                        if combo_key(s["conjunto"], s["seed"], s["variant"], s["representation"]) == key)
            finalist_results[key] = {}
            for scenario in eval_cfg["finalist_scenarios"]:
                print(f"\n=== Finalist {key} {scenario} ===", flush=True)
                summary = self.run_evaluation(
                    spec, checkpoints[key],
                    scenario=scenario,
                    task_set_name=eval_cfg["finalist_task_sets"][scenario],
                    level="final",
                    limit=eval_cfg["finalist_limit"],
                    rviz=eval_cfg["finalist_rviz"],
                    close=eval_cfg["finalist_close_after_trial"],
                )
                finalist_results[key][scenario] = summary

        # Summary
        rows = []
        for spec in specs:
            key = combo_key(spec["conjunto"], spec["seed"], spec["variant"], spec["representation"])
            history = Path(checkpoints[key]).parent / "history.json"
            best_val = None
            if history.exists():
                hist = json.loads(history.read_text())
                if hist:
                    best_val = min(r["validation_loss"] for r in hist)
            filt = filter_results.get(key) or {}
            final = finalist_results.get(key) or {}
            row = {
                "conjunto": spec["conjunto"],
                "seed": spec["seed"],
                "variant": spec["variant"],
                "representation": spec["representation"],
                "key": key,
                "checkpoint": str(checkpoints[key]),
                "best_val_loss": best_val,
                "filter_near_goal_rate": filt.get("near_goal_rate"),
                "filter_success_rate": filt.get("success_rate"),
                "filter_collision_rate": filt.get("collision_rate"),
                "filter_completed": filt.get("completed_trials"),
                "filter_errors": len(filt.get("infrastructure_errors") or []),
                "is_finalist": key in finalist_keys,
            }
            for scenario in eval_cfg["finalist_scenarios"]:
                s = final.get(scenario) or {}
                row[f"final_{scenario}_near_goal_rate"] = s.get("near_goal_rate")
                row[f"final_{scenario}_success_rate"] = s.get("success_rate")
                row[f"final_{scenario}_collision_rate"] = s.get("collision_rate")
            for metric in ("success_rate", "near_goal_rate"):
                rates = [result[metric] for result in final.values()]
                row[f"final_avg_{metric}"] = sum(rates) / len(rates) if rates else None
            rows.append(row)

        self.write_summary(rows)
        self.log("campaign_done", n_rows=len(rows), n_finalists=len(finalist_keys))
        self.manifest_fh.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned combinations without running anything")
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")
    cfg = json.loads(args.config.read_text())
    campaign_dir = Path(cfg["campaign_dir"]).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    with (campaign_dir / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"A campaign is already running in {campaign_dir}")
        Runner(cfg, dry_run=args.dry_run).run_campaign()


if __name__ == "__main__":
    main()
