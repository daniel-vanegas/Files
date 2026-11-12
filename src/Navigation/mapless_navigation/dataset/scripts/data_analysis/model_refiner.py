#!/usr/bin/env python3
"""Refine a mapless policy with independent training seeds and frozen Webots routes."""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import math
import random
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

from navigation_campaign import (
    CLASSES, EVALUATOR, SCRIPTS, TRAINER, Campaign, aggregate_sessions, balanced_capacity,
    digest, evaluator, file_hash, geometry_class, load_run, quotas_for, read_json,
    resample_path, sign_label, write_json,
)

SCENARIOS = tuple(f"E{i:02}" for i in range(1, 31))
RECIPES = {"moderate": [0.2, 0.2, 0.3, 0.3], "difficult": [0.1, 0.2, 0.3, 0.4]}
DESIGNS = [(6000, "moderate")] + [(n, recipe) for n in (4000, 3000, 2500) for recipe in RECIPES]
PROFILE = "demonstration"
REPRESENTATIONS = ("lom", "tlom")
LOM_REFINER_SHA256 = "ee8ead51a1a5a5ade2dcf0e12125b980e12768a28d11fa7d3b0a5d06744e945a"


def matrix_from_csv(path):
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        names = [name.strip().split("_")[0] for name in next(reader)]
        distance = np.array([[float(x) for x in row] for row in reader if row], dtype=float)
    if len(names) != len(SCENARIOS) or set(names) != set(SCENARIOS) or distance.shape != (30, 30):
        raise ValueError("The Wasserstein CSV must contain the 30 scenario names and a 30 by 30 matrix")
    if not np.isfinite(distance).all() or np.min(distance) < 0 or not np.allclose(distance, distance.T, atol=0.0001):
        raise ValueError("The Wasserstein matrix must be finite, nonnegative and symmetric")
    order = [names.index(name) for name in SCENARIOS]
    return distance[np.ix_(order, order)]


def sample_rows(rows, quotas, seed):
    selected = []
    for category, count in quotas.items():
        group = [r for r in rows if r["route_class"] == category]
        if count > balanced_capacity(group):
            raise ValueError(f"{category}: need {count} balanced windows, have {balanced_capacity(group)}")
        buckets = {side: [r for r in group if r["side"] == side] for side in ("left", "right", "neutral")}
        rng = random.Random(f"{seed}:{category}")
        for bucket in buckets.values():
            rng.shuffle(bucket)
        extra = rng.randrange(2)
        pairs = min(len(buckets["left"]), len(buckets["right"]))
        capacity = len(buckets["neutral"]) + 2 * pairs
        neutral = max(count - 2 * pairs, round(count * len(buckets["neutral"]) / max(1, capacity)))
        half, remainder = divmod(count - neutral, 2)
        selected.extend(buckets["left"][:half + remainder * extra])
        selected.extend(buckets["right"][:half + remainder * (1 - extra)])
        selected.extend(buckets["neutral"][:neutral])
    random.Random(seed).shuffle(selected)
    return selected


def enough(rows, size, weights):
    return all(balanced_capacity([r for r in rows if r["route_class"] == cls]) >= count
               for cls, count in quotas_for(size, weights).items())


def select_tn(index, matrix, quotas):
    names = [s for s in SCENARIOS if index[s]["pairs"]["train"] and index[s]["pairs"]["validation"]]
    if len(names) < 2:
        raise ValueError("TN needs at least two eligible training scenarios")
    sides = ("neutral", "left", "right")
    counts = {}
    for name in names:
        counts[name] = collections.Counter((r["route_class"], r["side"]) for r in index[name]["pairs"]["train"])
    columns = {name: matrix[:, SCENARIOS.index(name)] for name in names}
    candidates = [()]
    viable = []
    audit = []
    for size in range(1, min(6, len(names)) + 1):
        expanded = {tuple(sorted((*group, name))) for group in candidates for name in names if name not in group}
        scored = []
        for group in expanded:
            merged = sum((counts[s] for s in group), collections.Counter())
            capacity = {cls: merged[cls, sides[0]] + 2 * min(merged[cls, sides[1]], merged[cls, sides[2]])
                        for cls in CLASSES}
            shortage = sum(max(0, quotas[cls] - capacity[cls]) / max(1, quotas[cls]) for cls in CLASSES)
            support = min(sum(sum(counts[s][cls, side] for side in sides) >= max(1, quotas[cls] // 10)
                              for s in group) for cls in ("complex", "v_complex"))
            nearest = np.minimum.reduce([columns[s] for s in group])
            mean, maximum = float(nearest.mean()), float(nearest.max())
            coverage = mean + 0.25 * maximum
            scored.append((shortage, max(0, 2 - support), coverage, group, mean, maximum, capacity))
        scored.sort()
        candidates = [x[3] for x in scored[:64]]
        feasible = [x for x in scored if x[:2] == (0, 0)]
        if feasible:
            viable.append(feasible[0])
        best = (feasible or scored)[0]
        audit.append({"size": size, "tested": len(scored), "scenarios": list(best[3]),
                      "mean": best[4], "max": best[5], "capacity": best[6],
                      "relative_shortage": best[0], "difficult_support_deficit": best[1]})
    if not viable:
        raise ValueError(f"No TN up to six scenarios meets the quotas and difficult coverage: {audit}")
    best = min(viable, key=lambda x: x[2])
    chosen = next(x for x in viable if x[2] <= 1.10 * best[2] + 1e-12)
    return {"scenarios": list(chosen[3]), "mean": chosen[4], "max": chosen[5],
            "capacity": chosen[6], "quotas": quotas, "candidates": audit}


def index_scenarios(original):
    indexed, hashes = {}, {}
    for scenario in SCENARIOS:
        print(f"Indexing {scenario}", flush=True)
        episodes = []
        for path in sorted((original.data / scenario).glob("*.npz")):
            relative = path.relative_to(original.data).as_posix()
            hashes[relative] = file_hash(path)
            sidecar = path.with_suffix("") / "episode_tries.json"
            if sidecar.is_file():
                hashes[sidecar.relative_to(original.data).as_posix()] = file_hash(sidecar)
            grouped, labels = load_run(path)
            for episode_id, samples in sorted(grouped.items()):
                if len(samples) < 2:
                    continue
                poses = np.asarray([s["meta"]["pose"] for s in samples], dtype=float)
                stamps = np.asarray([s["meta"]["timestamp"] for s in samples], dtype=float)
                goal = np.asarray(samples[-1]["meta"]["goal_xy"], dtype=float)
                if (poses.shape != (len(samples), 3) or goal.shape != (2,)
                        or not all(np.isfinite(a).all() for a in (poses, stamps, goal))):
                    raise ValueError(f"Invalid trajectory in {relative}, episode {episode_id}")
                route_class = labels.get(episode_id) or geometry_class(poses)
                valid = [s["meta"].get("tlom_consistent", True)
                         and all(np.shape(s["features"].get(name)) == (140, 120) for name in ("occ_grid", "tlom"))
                         for s in samples]
                gaps = np.diff(stamps)
                rows = {2: [], 4: []}
                for length in rows:
                    for start in range(0, len(samples) - length + 1, length):
                        end = start + length
                        if not all(valid[start:end]) or not np.all(
                                (gaps[start:end-1] > 0) & (gaps[start:end-1] <= original.cfg["max_pair_gap_s"])):
                            continue
                        action = list(map(float, samples[end-1]["labels"]))
                        if len(action) != 3 or not all(math.isfinite(x) for x in action):
                            raise ValueError(f"Invalid control in {relative}, episode {episode_id}")
                        rows[length].append({"run_file": relative, "episode_id": episode_id, "start_index": start,
                                             "route_class": route_class, "action": action,
                                             "side": sign_label(action[1], original.cfg["lateral_deadband"])})
                reference = None
                direct = float(np.linalg.norm(poses[0, :2] - goal))
                error = float(np.linalg.norm(poses[-1, :2] - goal))
                _, length_m = resample_path(poses[:, :2])
                if (error <= original.cfg["demonstration_goal_tolerance_m"] and direct >= 0.5
                        and length_m >= 0.5 and stamps[-1] > stamps[0]
                        and not any(s["meta"].get("collision_flag", False) for s in samples)):
                    reference = {"run_file": relative, "episode_id": episode_id, "route_class": route_class,
                                 "frame": original.cfg["reference_frame"], "poses": poses.tolist(),
                                 "times": (stamps - stamps[0]).tolist(), "goal": goal.tolist(), "length_m": length_m}
                episodes.append((f"{relative}#{episode_id}", route_class, rows, reference))
        pairs = {name: [] for name in ("train", "validation")}
        anchors = {name: [] for name in pairs}
        refs = {}
        for cls in CLASSES:
            candidates = [e for e in episodes if e[1] == cls and e[2][2]]
            random.Random(f"{original.cfg['split_seed']}:{scenario}:{cls}").shuffle(candidates)
            n = len(candidates)
            nval = min(n - 1, max(1, int(n * original.cfg["validation_fraction"]))) if n > 1 else 0
            for i, (key, _, rows, _) in enumerate(candidates):
                split = original.prepared["split"].get(key, "validation" if i < nval else "train")
                pairs[split].extend(rows[2])
                anchors[split].extend(rows[4])
            available = sorted((e[3] for e in episodes if e[1] == cls and e[3]),
                               key=lambda r: (r["length_m"], r["run_file"], r["episode_id"]))
            refs[cls] = [available[round(q * (len(available) - 1))] for q in (1/3, 2/3)] if len(available) >= 3 else available
        indexed[scenario] = {"pairs": pairs, "anchors": anchors, "refs": refs}
    return indexed, hashes


class Refiner(Campaign):
    def __init__(self, original, root, prepared, sets, final=()):
        root.mkdir(parents=True, exist_ok=True)
        config = {**original.cfg, "campaign_dir": str(root), "environment_sets": sets}
        write_json(root / "config.json", config)
        super().__init__(root)
        self.prepared = prepared
        self.active = None
        self.stage_root = root
        write_json(root / "datasets_manifest/datasets.json",
                   {"phase": "datasets", "schema_version": 1,
                    "config": {"environment_sets": sets, "final_scenarios": list(final)}})

    def train(self, environment, representation, seed, candidate, scope="training"):
        self.active = candidate
        try:
            return super().train(environment, representation, seed, candidate, scope)
        finally:
            self.active = None

    def run_command(self, command, log_path):
        if command[2] == str(TRAINER):
            hp = self.active["train"]
            for flag, key, fallback in (("--sequence-length", "channels", 2),
                                        ("--hidden-size", "hidden_size", 350),
                                        ("--num-layers", "num_layers", 4)):
                command[command.index(flag) + 1] = str(hp.get(key, fallback))
        elif command[2] == str(EVALUATOR):
            suite = read_json(command[command.index("--suite") + 1])
            command[command.index("--purpose") + 1] = suite["purpose"]
        return super().run_command(command, log_path)

    def write_pairs(self, environment, seed, recipe):
        candidate = self.active
        channels = candidate["train"].get("channels", 2)
        common = candidate.get("anchors", False)
        pool = self.prepared["anchors" if common else "pools"][environment]
        weights = self.prepared["class_weights"][environment][recipe]
        sample_seed = candidate["data_seed"]
        selected = sample_rows(pool["train"], quotas_for(candidate["pairs"], weights), sample_seed)
        validation = random.Random(self.cfg["split_seed"]).sample(
            pool["validation"], min(len(pool["validation"]), self.cfg["validation_pairs"]))
        if common:
            selected = [{**row, "start_index": row["start_index"] + 4 - channels} for row in selected]
            validation = [{**row, "start_index": row["start_index"] + 4 - channels} for row in validation]
        used = {r["run_file"] for r in selected + validation}
        counts = collections.Counter((r["run_file"].split("/")[0], r["route_class"], r["side"]) for r in selected)
        manifest = {"version": 2, "sequence_length": channels, "variant": "5x4", "environment_set": environment,
                    "sources": {name: self.prepared["sources"][name] for name in sorted(used)},
                    "train": selected, "validation": validation, "recipe": recipe,
                    "composition": [{"scenario": s, "class": c, "side": d, "count": n}
                                    for (s, c, d), n in sorted(counts.items())]}
        path = self.stage_root / "pairs" / environment / f"{recipe}_{digest(manifest)[:16]}.json"
        if not path.exists():
            write_json(path, manifest, compact=True)
        elif read_json(path) != manifest:
            raise ValueError(f"Frozen manifest changed: {path}")
        return path


def next_seeds(state, path, name, count):
    slots = state.setdefault("seeds", {})
    if name not in slots:
        reserved = set(state["original_seeds"]) | {s for seeds in slots.values() for s in seeds}
        values = []
        rng = random.SystemRandom()
        while len(values) < count:
            value = rng.randrange(1, 2**31)
            if value not in reserved:
                values.append(value)
                reserved.add(value)
        slots[name] = values
        write_json(path, state)
    if len(slots[name]) != count:
        raise ValueError(f"The recorded seed count changed for {name}")
    return slots[name]


def quality(result):
    return (result["score"], -result["collision_rate"], result["arrival_rate"],
            -result["candidate"]["pairs"])


def measure(branch, state, state_path, name, environment, candidate, scenarios, phase, representation="lom"):
    models = []
    seeds = next_seeds(state, state_path, name, state["replicates"])
    for seed in seeds:
        receipt = branch.train(environment, representation, seed, candidate)
        checkpoint = next(p for p in receipt["checkpoints"] if Path(p).name == "best_model.pth")
        sessions = [branch.evaluate(receipt, checkpoint, phase, scenario, PROFILE) for scenario in scenarios]
        models.append(aggregate_sessions(sessions, checkpoint))
    return summarize(environment, candidate, models, representation)


def summarize(environment, candidate, models, representation):
    metrics = ("score", "collision_rate", "arrival_rate", "frechet_m")
    result = {key: statistics.mean(m[key] for m in models) for key in metrics}
    result.update(environment=environment, representation=representation, candidate=candidate, models=models,
                  score_std=statistics.pstdev(m["score"] for m in models))
    result["by_class"] = {cls: {key: statistics.mean(m["by_class"][cls][key] for m in models)
                               for key in metrics} for cls in CLASSES}
    return result


def reconsider(branch, result, phase, scenarios):
    models = []
    representation = result.get("representation", "lom")
    for model in result["models"]:
        checkpoint = model["checkpoint"]
        receipt = branch.train(result["environment"], representation, model["seed"], result["candidate"])
        sessions = [branch.evaluate(receipt, checkpoint, phase, s, PROFILE) for s in scenarios]
        models.append(aggregate_sessions(sessions, checkpoint))
    return summarize(result["environment"], result["candidate"], models, representation)


def stage1(original, root, state, state_path):
    if "stage1" in state:
        return
    prepared = {**original.prepared, "class_weights": {"T1": {
        **RECIPES, "natural": original.prepared["class_weights"]["T1"]["natural"]}}}
    branch = Refiner(original, root / "stage1", prepared,
                     {"T1": original.cfg["environment_sets"]["T1"]})
    results = []
    for size, recipe in [*DESIGNS, (10000, "natural")]:
        weights = prepared["class_weights"]["T1"][recipe]
        if not enough(prepared["pools"]["T1"]["train"], size, weights):
            state.setdefault("unavailable", {})[f"{size}:{recipe}"] = "Insufficient balanced T1 windows"
            write_json(state_path, state)
            print(f"Skipping T1/{size}/{recipe}: insufficient balanced windows", flush=True)
            continue
        candidate = {"recipe": recipe, "pairs": size, "train": original.cfg["train"],
                     "data_seed": original.cfg["split_seed"]}
        results.append(measure(branch, state, state_path, f"stage1:{size}:{recipe}", "T1", candidate,
                               original.cfg["screen"], "screen"))
    reduced = sorted((r for r in results if r["candidate"]["recipe"] != "natural"), key=quality, reverse=True)
    if not reduced:
        raise ValueError("No reduced T1 configuration could be trained")
    confirmation = [reconsider(branch, r, "confirmation", original.cfg["confirmation"])
                    for r in [*reduced[:2], *(r for r in results if r["candidate"]["recipe"] == "natural")]]
    winner = max((r for r in confirmation if r["candidate"]["recipe"] != "natural"), key=quality)
    state["stage1"] = {"screen": results, "confirmation": confirmation, "winner": winner}
    write_json(state_path, state)
    print(f"Stage 1 winner: {winner['candidate']['pairs']} {winner['candidate']['recipe']}", flush=True)


def suite_scenarios(original, index, excluded):
    eligible = {s for s in SCENARIOS if s not in excluded and all(len(index[s]["refs"][c]) >= 2 for c in CLASSES)}
    existing = set(original.cfg["screen"]) | set(original.cfg["confirmation"])
    order = list(dict.fromkeys([*original.cfg["confirmation"], *original.cfg["screen"], *SCENARIOS]))
    final = [s for s in SCENARIOS if s in eligible and s not in existing][-2:]
    development = [s for s in order if s in eligible and s not in final][:2]
    if len(final) != 2 or len(development) != 2:
        raise ValueError("Need two common development scenarios and two unused final scenarios with recorded routes")
    return development, final


def stage2(original, root, state, state_path, matrix_path):
    if "stage2" in state:
        return
    candidate = state["stage1"]["winner"]["candidate"]
    matrix = matrix_from_csv(matrix_path)
    index, hashes = index_scenarios(original)
    selection = select_tn(index, matrix, quotas_for(candidate["pairs"], RECIPES[candidate["recipe"]]))
    sets = {"T1": original.cfg["environment_sets"]["T1"], "TN": selection["scenarios"]}
    development, final = suite_scenarios(original, index, set(sets["T1"]) | set(sets["TN"]))
    prepared = {"sources": {k: v for k, v in hashes.items() if k.endswith(".npz")}, "input_hashes": hashes,
                "references": {}, "suites": {},
                "class_weights": {name: RECIPES for name in sets}}
    for pool, source in (("pools", "pairs"), ("anchors", "anchors")):
        prepared[pool] = {name: {split: [row for scenario in group for row in index[scenario][source][split]]
                                 for split in ("train", "validation")} for name, group in sets.items()}
    for name in sets:
        if not enough(prepared["pools"][name]["train"], candidate["pairs"], RECIPES[candidate["recipe"]]):
            raise ValueError(f"{name} cannot fill the chosen Stage 1 quota")
    import yaml

    ev = evaluator()
    for phase, scenarios in (("development", development), ("final", final)):
        for scenario in scenarios:
            tasks = []
            for cls in CLASSES:
                for reference in index[scenario]["refs"][cls][:2]:
                    source = original.data / reference["run_file"]
                    descriptors = (source.with_suffix("") / "descriptor.yaml", source.with_suffix(".descriptor.yaml"))
                    descriptor = next((p for p in descriptors if p.is_file()), None)
                    if descriptor is not None:
                        frame = (yaml.safe_load(descriptor.read_text()) or {}).get("map_frame")
                        if frame and frame != reference["frame"]:
                            raise ValueError(f"Reference frame differs from {descriptor}")
                        prepared["input_hashes"][descriptor.relative_to(original.data).as_posix()] = file_hash(descriptor)
                    task_id = len(tasks) + 1
                    key = f"{phase}_{scenario}_{task_id}"
                    prepared["references"][key] = reference
                    start = reference["poses"][0]
                    tasks.append({"id": task_id, "route_class": cls, "start": start[:2],
                                  "start_yaw_rad": start[2], "goal": reference["goal"],
                                  "generator_start_yaw_rad": start[2], "direct_line_safe": None,
                                  "task_type": "recorded_demonstration", "behavior_tags": [],
                                  "reference_key": key})
            suite = original.make_suite(ev, scenario, tasks, original.prepared["profiles"][PROFILE])
            suite["purpose"] = phase
            name = f"{phase}_{scenario}_{PROFILE}"
            path = root / "suites" / (name + ".json")
            write_json(path, suite)
            prepared["suites"][name] = {"path": str(path), "sha256": file_hash(path)}
    prepared_path = root / "tn_prepared.json"
    write_json(prepared_path, prepared, compact=True)
    selection.update(sets=sets, development=development, final=final,
                     matrix=str(matrix_path), matrix_sha256=file_hash(matrix_path),
                     prepared_sha256=file_hash(prepared_path),
                     scenario_counts={s: {"train": dict(collections.Counter(r["route_class"] for r in data["pairs"]["train"])),
                                          "validation": dict(collections.Counter(r["route_class"] for r in data["pairs"]["validation"])),
                                          "eligible_references": {c: len(refs) for c, refs in data["refs"].items()}}
                                      for s, data in index.items()})
    state["stage2"] = selection
    write_json(root / "seleccion_tn.json", selection)
    write_json(state_path, state)
    print(f"TN: {', '.join(sets['TN'])}; development: {development}; final: {final}", flush=True)


def wins(candidate, incumbent):
    return (candidate["score"] > incumbent["score"] + 1e-6
            and candidate["collision_rate"] <= incumbent["collision_rate"] + 1e-9
            and candidate["arrival_rate"] + 1e-9 >= incumbent["arrival_rate"])


def candidates_for(candidate, parameter, values):
    return [{**candidate, "train": {**candidate["train"], parameter: value}} for value in values]


def run_round(branch, state, path, name, environment, candidate, proposals, scenarios, representation="lom"):
    if name in state.get("rounds", {}):
        return state["rounds"][name]["winner"]
    measured = [measure(branch, state, path, f"{name}:{i}", environment, config, scenarios, "development", representation)
                for i, config in enumerate([candidate, *proposals])]
    base, *alternatives = measured
    winner = max([base, *(r for r in alternatives if wins(r, base))], key=quality)
    state.setdefault("rounds", {})[name] = {"results": measured, "winner": winner}
    write_json(path, state)
    write_report(path.parent, state)
    print(f"{name}: baseline={base['score']:.4f}, winner={winner['score']:.4f}", flush=True)
    return winner


def stage3(original, root, state, path):
    if "stage3" in state:
        prune_checkpoints(root, state["stage3"])
        return
    tn = state["stage2"]
    prepared_path = root / "tn_prepared.json"
    if file_hash(prepared_path) != tn["prepared_sha256"]:
        raise ValueError("The frozen TN index changed")
    prepared = read_json(prepared_path)
    for relative, expected in prepared["input_hashes"].items():
        if file_hash(original.data / relative) != expected:
            raise ValueError(f"A frozen dataset input changed: {relative}")
    for suite in prepared["suites"].values():
        if file_hash(suite["path"]) != suite["sha256"]:
            raise ValueError(f"A frozen suite changed: {suite['path']}")
    branch = Refiner(original, root / "stage3", prepared, tn["sets"], tn["final"])
    stage1_candidate = state["stage1"]["winner"]["candidate"]
    baseline = {**stage1_candidate, "train": {**stage1_candidate["train"],
                                            "channels": 2, "hidden_size": 350, "num_layers": 4}}
    development = tn["development"]
    conditions = [(env, representation) for env in tn["sets"] for representation in REPRESENTATIONS]
    names = {(env, rep): env if rep == "lom" else f"{env}:tlom" for env, rep in conditions}
    matched = {names[env, rep]: run_round(branch, state, path, f"{names[env, rep]}:matched",
                                         env, baseline, [], development, rep)
               for env, rep in conditions}
    leaders = {}
    for env, rep in conditions:
        name = names[env, rep]
        winner = matched[name]
        lr = baseline["train"]["learning_rate"]
        for parameter, values in (("batch_size", (2, 6)), ("learning_rate", (lr/2, lr*2)),
                                  ("num_layers", (2, 3)), ("hidden_size", (256,)),
                                  ("dropout_1", (0.10,)), ("dropout_2", (0.05,))):
            current = winner["candidate"]
            proposals = candidates_for(current, parameter, values)
            winner = run_round(branch, state, path, f"{name}:{parameter}", env, current, proposals, development, rep)
            if parameter == "learning_rate" and winner["candidate"] != current:
                new_lr = winner["candidate"]["train"][parameter]
                neighbor = math.sqrt(lr * new_lr)
                current = winner["candidate"]
                winner = run_round(branch, state, path, f"{name}:learning_rate_neighbor", env, current,
                                   candidates_for(current, parameter, [neighbor]), development, rep)

        current = winner["candidate"]
        weights = RECIPES[current["recipe"]]
        budgets = sorted({current["pairs"], *(n for n, _ in DESIGNS if n <= current["pairs"])}, reverse=True)
        budget = next((n for n in budgets if enough(prepared["anchors"][env]["train"], n, weights)), None)
        if budget is not None:
            anchored = {**current, "anchors": True, "pairs": budget}
            sequence = run_round(branch, state, path, f"{name}:channels", env, anchored,
                                 candidates_for(anchored, "channels", (3, 4)), development, rep)
            if sequence["candidate"]["train"]["channels"] != current["train"]["channels"]:
                if budget == current["pairs"] or wins(sequence, winner):
                    winner = sequence
        else:
            state.setdefault("unavailable", {})[f"{name}:channels"] = "Insufficient common C=4 anchors even for 2500 windows"
            write_json(path, state)
            print(f"Skipping {name}/channels: insufficient common C=4 anchors", flush=True)

        current = winner["candidate"]
        alternative = next(name for name in RECIPES if name != current["recipe"])
        source = "anchors" if current.get("anchors") else "pools"
        if (winner["by_class"]["v_complex"]["arrival_rate"] < winner["by_class"]["complex"]["arrival_rate"]
                and enough(prepared[source][env]["train"], current["pairs"], RECIPES[alternative])):
            winner = run_round(branch, state, path, f"{name}:route_mix", env, current,
                               [{**current, "recipe": alternative}], development, rep)
        leaders[name] = winner

    head_to_head = {names[env, rep]: run_round(branch, state, path, f"{names[env, rep]}:selection",
                                              env, leaders[names[env, rep]]["candidate"], [], development, rep)
                    for env, rep in conditions}
    reference = head_to_head["T1"]
    selected = max([reference, *(result for result in head_to_head.values() if wins(result, reference))],
                   key=quality)
    best = max(selected["models"], key=lambda m: (m["score"], -m["collision_rate"], m["arrival_rate"]))
    checkpoint = Path(best["checkpoint"])
    receipt = branch.train(selected["environment"], selected.get("representation", "lom"),
                           best["seed"], selected["candidate"])
    final = [branch.evaluate(receipt, str(checkpoint), "final", scenario, PROFILE) for scenario in tn["final"]]
    final_score = aggregate_sessions(final, str(checkpoint))
    destination = save_winner(root, checkpoint, receipt["checkpoints"][str(checkpoint)], selected, best)
    state["stage3"] = {"matched": matched, "head_to_head": head_to_head,
                       "selected": selected, "model": best, "final": final_score,
                       "winner_path": str(destination), "winner_sha256": file_hash(destination)}
    write_json(path, state)
    write_report(root, state)
    prune_checkpoints(root, state["stage3"])


def save_winner(root, checkpoint, expected, selected, best):
    import torch

    destination = root / "ganador" / "best_model.pth"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".pth.tmp")
    shutil.copyfile(checkpoint, temporary)
    if file_hash(temporary) != expected:
        raise ValueError("Winner copy does not match the evaluated checkpoint")
    sys.path.insert(0, str(SCRIPTS))
    import RNN_mapless
    payload = torch.load(temporary, map_location="cpu", weights_only=False)
    config = payload["config"]
    model, settings = RNN_mapless.make_model(config["architecture"], payload["model_state_dict"], config)
    channels = selected["candidate"]["train"]["channels"]
    if (settings["sequence_length"] != channels or not settings["require_full_sequence"]
            or settings["representation"] != selected.get("representation", "lom")
            or not settings["normalize_occupancy"]):
        raise ValueError("The winner's inference settings do not match the selected temporal input")
    grid = np.zeros((120, 140), dtype=np.float32)
    frame = RNN_mapless.prepare_observation(grid, 1.0, 0.0, settings)
    x = torch.from_numpy(np.stack([frame] * channels)[None])
    with torch.inference_mode():
        outputs = model.eval()(x)
    if outputs.numel() != 3 or not torch.isfinite(outputs).all():
        raise ValueError("The winner failed a CPU inference check")
    temporary.replace(destination)
    shutil.copyfile(checkpoint.with_name("config.json"), destination.with_name("config.json"))
    write_json(destination.with_name("provenance.json"),
               {"checkpoint_sha256": expected, "configuration": selected["candidate"],
                "environment": selected["environment"], "representation": selected.get("representation", "lom"),
                "seed": best["seed"], "epoch": best["epoch"],
                "source_checkpoint": str(checkpoint)})
    return destination


def prune_checkpoints(root, result):
    winner = Path(result["winner_path"])
    if file_hash(winner) != result["winner_sha256"] or not winner.with_name("config.json").is_file():
        raise ValueError("The verified winner is unavailable; training checkpoints are retained")
    training = root / "stage3" / "training"
    count = 0
    for pattern in ("*/attempt_*/output/best_model.pth", "*/attempt_*/output/best_gap.pth"):
        for path in training.glob(pattern):
            if path.is_symlink() or not path.resolve().is_relative_to(training.resolve()):
                raise ValueError(f"Checkpoint outside the refiner training directory: {path}")
            path.unlink()
            count += 1
    print(f"Winner: {winner}; removed {count} redundant Stage 3 checkpoint files", flush=True)


def write_report(root, state):
    lines = ["# Refinamiento de modelos de navegación 5 × 4 m", "",
             "Entrenamientos independientes con LOM o T-LOM y ocupación 0/1.", "",
             "| Fase | Conjunto | Entrada | Pares | Mezcla | Puntaje medio ± DE | Llegadas | Colisiones | Fréchet (m) |",
             "|---|---|---|---:|---|---:|---:|---:|---:|"]

    def row(phase, result):
        candidate = result["candidate"]
        representation = "T-LOM" if result.get("representation") == "tlom" else "LOM"
        lines.append(f"| {phase} | {result['environment']} | {representation} | {candidate['pairs']} | {candidate['recipe']} | "
                     f"{result['score']:.4f} ± {result['score_std']:.4f} | {result['arrival_rate']:.1%} | "
                     f"{result['collision_rate']:.1%} | {result['frechet_m']:.3f} |")

    for name, results in state.get("stage1", {}).items():
        if name != "winner":
            for result in results:
                row(f"Etapa 1 / {name}", result)
    for name, round_data in state.get("rounds", {}).items():
        for i, result in enumerate(round_data["results"]):
            row(f"{name} / {i}", result)

    if "stage2" in state:
        tn = state["stage2"]
        lines.extend(["", f"TN incluye {', '.join(tn['sets']['TN'])}.",
                      f"Desarrollo común en {', '.join(tn['development'])}.",
                      f"Prueba reservada en {', '.join(tn['final'])}."])
    if state.get("unavailable"):
        lines.extend(["", "## Condiciones omitidas", ""])
        lines.extend(f"- `{name}` — {reason}" for name, reason in state["unavailable"].items())
    if "stage3" in state:
        result = state["stage3"]
        selected, model, final = result["selected"], result["model"], result["final"]
        lines.extend(["", "## Ganador y prueba reservada", "",
                      f"Modelo en `{result['winner_path']}`.",
                      f"Conjunto {selected['environment']}, entrada {selected.get('representation', 'lom')}, "
                      f"semilla {model['seed']}, época {model['epoch']}.",
                      f"Configuración `{selected['candidate']['train']}`.", "",
                      "| Rutas | Puntaje | Llegadas | Colisiones | Fréchet (m) |",
                      "|---|---:|---:|---:|---:|"])
        for cls, group in [("Todas", final), *final["by_class"].items()]:
            lines.append(f"| {cls} | {group['score']:.4f} | {group['arrival_rate']:.1%} | "
                         f"{group['collision_rate']:.1%} | {group['frechet_m']:.3f} |")
        difficult = selected["by_class"]["v_complex"]
        if difficult["arrival_rate"] < selected["by_class"]["complex"]["arrival_rate"]:
            lines.extend(["", "En desarrollo, v_complex sigue por debajo de complex en llegadas. "
                          "La siguiente iteración debe revisar la mezcla y la cobertura de esas maniobras."])
    lines.extend(["", "Los resultados por clase, las semillas y las decisiones por ronda están en `refiner_state.json`."])
    path = root / "reporte_refinador.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def open_original(path):
    original = Campaign(path)
    original.prepared = read_json(original.root / "prepared.json")
    if digest(original.cfg) != original.prepared["config_sha256"]:
        raise ValueError("The original campaign configuration differs from its prepared data")
    for relative, expected in {**original.prepared["sources"], **original.prepared["annotations"]}.items():
        if file_hash(original.data / relative) != expected:
            raise ValueError(f"The original dataset input changed: {relative}")
    for info in original.prepared["suites"].values():
        if file_hash(info["path"]) != info["sha256"]:
            raise ValueError(f"The original suite changed: {info['path']}")
    return original


def migrate_lom_run(root, original, state, identity, path):
    old_identity = {**identity, "code": {**identity["code"], Path(__file__).name: LOM_REFINER_SHA256}}
    if state["provenance"] != old_identity:
        raise ValueError("Inputs changed; resume with the original options or use a fresh output directory")
    if "stage3" in state:
        raise ValueError("This LOM-only run already selected and pruned its checkpoints; use a fresh output directory")
    if "stage2" in state:
        prepared_path = root / "tn_prepared.json"
        if file_hash(prepared_path) != state["stage2"]["prepared_sha256"]:
            raise ValueError("The frozen TN index changed")
        prepared = read_json(prepared_path)
        windows = collections.defaultdict(set)
        for name, length in (("pools", 2), ("anchors", 4)):
            for group in prepared[name].values():
                for rows in group.values():
                    for row in rows:
                        windows[row["run_file"]].add((row["episode_id"], row["start_index"], length))
        for relative, intervals in sorted(windows.items()):
            path_npz = original.data / relative
            if file_hash(path_npz) != prepared["sources"][relative]:
                raise ValueError(f"A frozen dataset input changed: {relative}")
            episodes, _ = load_run(path_npz)
            for episode, start, length in intervals:
                samples = episodes[episode][start:start + length]
                if len(samples) != length or any(
                        np.shape(s["features"].get("tlom")) != (140, 120)
                        or np.shape(s["features"].get("occ_grid")) != (140, 120) for s in samples):
                    raise ValueError(f"{relative} has incomplete LOM/T-LOM windows; use a fresh output directory")
    state["provenance"] = identity
    write_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True, help="Completed 5x4 reference campaign")
    parser.add_argument("--output-dir", type=Path, required=True)
    default_matrix = (SCRIPTS.parents[3] / "Hardware/justina_description/webots_env/scripts/maps/similarity"
                      / "30env_wasserstein_sinkhorn_mean.csv")
    parser.add_argument("--matrix", type=Path, default=default_matrix)
    parser.add_argument("--stage", choices=("stage1", "prepare-tn", "refine", "all", "status"), default="all")
    parser.add_argument("--replicates", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()
    try:
        root = args.output_dir.expanduser().resolve()
        original_path = args.campaign_dir.expanduser().resolve()
        if root == original_path or root.is_relative_to(original_path):
            raise ValueError("Use an output directory separate from the original campaign")
        path = root / "refiner_state.json"
        if args.stage == "status":
            state = read_json(path)
            print("Completed stages:", ", ".join(k for k in ("stage1", "stage2", "stage3") if k in state))
            print("Recorded seeds:", sum(map(len, state["seeds"].values())))
            print("Completed refinement rounds:", len(state.get("rounds", {})))
            if "stage3" in state:
                print("Winner:", state["stage3"]["winner_path"])
            return
        matrix = args.matrix.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".refiner.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            original = open_original(original_path)
            identity = {"original": str(original.root), "original_prepared": file_hash(original.root / "prepared.json"),
                        "code": {**original.source_signature(), Path(__file__).name: file_hash(__file__)},
                        "matrix": str(matrix), "matrix_sha256": file_hash(matrix)}
            if path.exists():
                state = read_json(path)
                if state["replicates"] != args.replicates:
                    raise ValueError("Use the original number of repetitions when resuming")
                if state["provenance"] != identity:
                    migrate_lom_run(root, original, state, identity, path)
            else:
                if any(p.name != ".refiner.lock" for p in root.iterdir()):
                    raise ValueError(f"Refiner output directory is not empty: {root}")
                reserved = set(original.cfg["seeds"])
                factorial_plan = original.root.with_name(original.root.name + "_factorial") / "factorial_plan.json"
                if factorial_plan.is_file():
                    reserved.update(block["seed"] for block in read_json(factorial_plan)["blocks"])
                state = {"provenance": identity, "original_seeds": sorted(reserved),
                         "seeds": {}, "replicates": args.replicates}
                write_json(path, state)
            if args.stage in ("stage1", "refine", "all"):
                missing = [name for name in ("ros2", "webots") if not shutil.which(name)]
                if missing:
                    raise ValueError(f"Missing {', '.join(missing)}; source install/setup.bash")
            if args.stage in ("stage1", "all"):
                stage1(original, root, state, path)
                write_report(root, state)
            if args.stage in ("prepare-tn", "all"):
                if "stage1" not in state:
                    raise ValueError("Complete Stage 1 first")
                stage2(original, root, state, path, matrix)
                write_report(root, state)
            if args.stage in ("refine", "all"):
                if "stage2" not in state:
                    raise ValueError("Prepare TN first")
                stage3(original, root, state, path)
                write_report(root, state)
            print(f"Report: {root / 'reporte_refinador.md'}")
    except BlockingIOError:
        parser.exit(2, "ERROR: The refiner is already running\n")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Saved training and evaluation results can be resumed.\n")


if __name__ == "__main__":
    main()
