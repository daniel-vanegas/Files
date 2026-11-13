#!/usr/bin/env python3
"""Compare the factorial and refiner winners on a frozen shared Webots suite."""
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

from navigation_campaign import (
    CLASSES, DATASET, EVALUATOR, Campaign, evaluator, file_hash, load_run, read_json, write_json,
)

SCENARIOS = tuple(f"E{i:02}" for i in range(1, 31))


def factorial_winner(path):
    groups = collections.defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["scale"], row["training"], row["inference"]) == ("1", "lom", "lom"):
                groups[row["checkpoint"]].append(row)
    complete = {
        checkpoint: rows for checkpoint, rows in groups.items()
        if len(rows) == 16 and all(
            len({r["task_id"] for r in rows if r["scenario"] == s}) == 8 for s in ("E16", "E20"))
    }
    if not complete:
        raise ValueError("No complete LOM 0/1 checkpoint found in the factorial report")
    return Path(max(complete, key=lambda path: (
        statistics.mean(float(r["score"]) for r in complete[path]),
        -sum(r["collision"].lower() in ("true", "1") for r in complete[path]), path)))


def read_matrix(path):
    with path.open(newline="") as stream:
        names, *rows = list(csv.reader(stream))
    names = [name.strip().split("_", 1)[0] for name in names]
    if len(names) != 30 or set(names) != set(SCENARIOS) or len(rows) != 30 or any(len(r) != 30 for r in rows):
        raise ValueError("Expected the labeled 30 by 30 Wasserstein matrix")
    distances = {name: dict(zip(names, map(float, row))) for name, row in zip(names, rows)}
    for a in names:
        if abs(distances[a][a]) > 0.0001:
            raise ValueError("The Wasserstein diagonal must be zero")
        for b in names:
            value = distances[a][b]
            if not math.isfinite(value) or value < 0 or abs(value - distances[b][a]) > 0.0001:
                raise ValueError("Expected finite, symmetric and nonnegative Wasserstein distances")
    return distances


def select_scenarios(distances, training, excluded, total):
    common = sorted(set.intersection(*(set(s) for s in training.values())))
    union = set.union(*(set(s) for s in training.values()))
    unseen = sorted(set(SCENARIOS) - union - set(excluded))
    if len(common) < 2 or len(unseen) < 3:
        raise ValueError("Need two common training scenarios and three new unseen scenarios")
    nearest = {scenario: {name: min(distances[scenario][s] for s in train)
                          for name, train in training.items()} for scenario in SCENARIOS}
    far = max(unseen, key=lambda s: (min(nearest[s].values()), statistics.mean(nearest[s].values()), s))
    near = sorted((s for s in unseen if s != far),
                  key=lambda s: (max(nearest[s].values()), statistics.mean(nearest[s].values()), s))[:2]
    seen = common[:2]
    base, difficult = ((4, 4, 1, 1), (2, 2, 3, 3)) if total == 50 else ((2, 2, 1, 1), (1, 1, 2, 2))
    roles = [("visto_base", base), ("visto_dificil", difficult),
             ("no_visto_similar", difficult), ("no_visto_similar", difficult), ("no_visto_distante", difficult)]
    return [{"scenario": scenario, "group": group, "distance": nearest[scenario],
             "quotas": dict(zip(CLASSES, quota))}
            for scenario, (group, quota) in zip([*seen, *near, far], roles)]


def model_details(ev, checkpoints, metadata):
    models, training = {}, {}
    for label, path in checkpoints.items():
        model = ev.candidate(path, None)
        config = model["training_config"]
        if (config.get("variant") != "5x4" or config.get("representation") != "lom"
                or config.get("normalize_occupancy") is not True):
            raise ValueError(f"{label} needs its 5x4 LOM 0/1 checkpoint and original config.json")
        name = config.get("split", {}).get("environment_set")
        scenarios = metadata[label]["config"]["environment_sets"].get(name)
        if not scenarios or not set(scenarios) <= set(SCENARIOS):
            raise ValueError(f"The training environments for {label} could not be determined")
        models[label] = {"checkpoint": model["path"], "sha256": model["sha256"], "set": name,
                         "config_sha256": file_hash(Path(model["path"]).with_name("config.json"))}
        training[label] = sorted(scenarios)
    return models, training


def previous_routes(campaign, ev, scenario, offsets, prior_suites):
    references = []
    for path in sorted((campaign.data / scenario).glob("*.npz")):
        episodes, _ = load_run(path)
        for samples in episodes.values():
            if not samples:
                continue
            start = samples[0]["meta"]["pose"][:2]
            goal = samples[-1]["meta"]["goal_xy"]
            if campaign.cfg["reference_frame"] == "world":
                start, goal = ev.to_map(start, offsets), ev.to_map(goal, offsets)
            references.append((start, goal))
    for path in prior_suites:
        if path.name.startswith(scenario + "_") or f"_{scenario}_" in path.name:
            references.extend((t["start"], t["goal"]) for t in read_json(path)["tasks"])
    return references


def repeated_route(start, goal, references):
    return any((math.dist(start, a) <= 0.20 and math.dist(goal, b) <= 0.25)
               or (math.dist(start, b) <= 0.20 and math.dist(goal, a) <= 0.25)
               for a, b in references)


def prepare_suite(campaign, ev, root, item, profile, seed, generator, timeout, prior_suites):
    scenario = item["scenario"]
    frozen = root / "suites" / f"{scenario}.json"
    receipt = frozen.with_suffix(".sha256")
    if frozen.exists() and receipt.exists():
        if receipt.read_text().strip() != file_hash(frozen):
            raise ValueError(f"Frozen suite changed: {frozen}")
        return frozen
    suite = campaign.make_suite(ev, scenario, [], profile)
    prior = previous_routes(campaign, ev, scenario, suite["offsets"], prior_suites)
    quotas = item["quotas"]
    accepted = collections.Counter()
    paths, used = [], list(prior)
    generation = root / "generation" / scenario

    def collect(pool_path, expected_seed=None):
        if not pool_path.is_file():
            return
        pool = read_json(pool_path)
        if (pool.get("pool_version") != 5 or pool.get("scenario") != scenario
                or (expected_seed is not None and pool.get("seed") != expected_seed)):
            raise ValueError(f"Unexpected pool metadata in {pool_path}")
        for route in pool["paths"]:
            cls = route["class"]
            start, goal = route["start"], route["goal"]
            if accepted[cls] >= quotas[cls] or repeated_route(start, goal, used):
                continue
            paths.append(route)
            used.append((start, goal))
            accepted[cls] += 1

    def progress():
        return ", ".join(f"{cls}={accepted[cls]}/{quotas[cls]}" for cls in CLASSES)

    previous = sorted(generation.glob("attempt_*"), key=lambda path: int(path.name.split("_")[-1]))
    pool_relative = Path("pool") / scenario / "benchmark" / "path_pool.json"
    for attempt in previous:
        collect(attempt / pool_relative)
        if accepted == quotas:
            break
    print(f"{scenario} accepted after novelty filtering: {progress()}", flush=True)

    for _ in range(6):
        if accepted == quotas:
            break
        attempt = campaign.new_attempt(generation)
        offset = int(attempt.name.split("_")[-1]) - 1
        run_seed = (seed - 1 + offset * 1009) % 2147483646 + 1
        targets = [max(1, quotas[cls] - accepted[cls] + 2) if accepted[cls] < quotas[cls] else 1
                   for cls in CLASSES]
        command = [sys.executable, "-u", str(generator), "--scenario", scenario,
                   "--maps-root", str(ev.HARDWARE / "environments/maps"), "--output-dir", str(attempt / "pool"),
                   "--run-name", "benchmark", "--seed", str(run_seed), "--timeout-seconds", str(timeout),
                   "--v-max-goal-distance-m", "3.0", "--st-sem-cmp-vc", "_".join(map(str, targets))]
        status = campaign.run_command(command, attempt / "generator.log")
        if status not in (0, 1):
            raise ValueError(f"Route generator exited with {status}; see {attempt / 'generator.log'}")
        collect(attempt / pool_relative, run_seed)
        print(f"{scenario} accepted after novelty filtering: {progress()}", flush=True)
    if accepted != quotas:
        raise ValueError(f"Novel route quotas incomplete for {scenario}: {progress()}. "
                         f"Resume with the same output directory to continue from {generation}")
    tasks = [
        {"id": index, "start": p["start"], "goal": p["goal"],
         "start_yaw_rad": p["start_yaw_rad"], "generator_start_yaw_rad": p["start_yaw_rad"],
         "route_class": p["class"], "task_type": p.get("task_type"),
         "direct_line_safe": p["direct_line_safe"], "behavior_tags": []}
        for index, p in enumerate(sorted(paths, key=lambda p: (CLASSES.index(p["class"]), p["id"])), start=1)
    ]
    suite.update(tasks=tasks, purpose="final", seed=seed, heading_policy="generator",
                 generator_sha256=file_hash(generator))
    suite["protocol"]["world_random_seed"] = seed
    write_json(frozen, suite)
    receipt.write_text(file_hash(frozen) + "\n")
    return frozen


def completed_result(path, model, suite):
    summary = path / "summary.json"
    actual_suite = path / "suite.json"
    if not summary.is_file() or not actual_suite.is_file() or not (path / "trajectory.csv").is_file():
        return None
    data = read_json(summary)
    frozen = read_json(suite)
    expected = [(t["id"], t["route_class"]) for t in frozen["tasks"]]
    if (data["status"] == "completed" and not data["infrastructure_errors"] and data.get("rviz_config")
            and data["model"]["sha256"] == model["sha256"]
            and data["suite_sha256"] == file_hash(actual_suite) and read_json(actual_suite) == frozen
            and [(r["task_id"], r["route_class"]) for r in data["results"]] == expected):
        return data
    return None


def run_model(campaign, root, label, model, suite, datasets):
    data = read_json(suite)
    base = root / "evaluations" / data["scenario"] / label
    for attempt in sorted(base.glob("attempt_*"), reverse=True):
        completed = completed_result(attempt / "session", model, suite)
        if completed:
            return completed
    attempt = campaign.new_attempt(base)
    output = attempt / "session"
    cmd = [sys.executable, "-u", str(EVALUATOR), "--datasets-dir", str(datasets),
           "--checkpoint", model["checkpoint"], "--scenario", data["scenario"], "--purpose", "final",
           "--suite", str(suite), "--limit", str(len(data["tasks"])), "--close-after-trial",
           "--rviz", "--preview-seconds", "3.0", "--output-dir", str(output)]
    for key, value in data["sources"]["policy_overrides"].items():
        cmd.extend(("--" + key.replace("_", "-"), str(value)))
    status = campaign.run_command(cmd, attempt / "evaluation.log")
    result = completed_result(output, model, suite) if status == 0 else None
    if result is None:
        raise ValueError(f"Webots session incomplete for {label}/{data['scenario']}; see {attempt / 'evaluation.log'}")
    return result


def display_set(model, scenarios):
    if model["set"] == "TN" and set(scenarios) == {"E01", "E03", "E16", "E18", "E19"}:
        return "T3"
    return model["set"]


def report(root, plan, rows):
    csv_path = root / "recorridos.csv"
    temp = csv_path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)
    lines = ["# Comparación de los ganadores factorial y refinado", "",
             f"Rutas emparejadas completadas por modelo: {len(rows)//2}/{plan['goals']}.",
             "", "| Modelo | Conjunto | Escenarios de entrenamiento | Checkpoint |",
             "|---|---|---|---|"]
    for label, model in plan["models"].items():
        scenarios = plan["training"][label]
        lines.append(f"| {label} | {display_set(model, scenarios)} | {', '.join(scenarios)} | `{model['checkpoint']}` |")
    lines += ["", "| Escenario | Grupo | Distancia factorial | Distancia refinado | Cuotas |",
              "|---|---|---:|---:|---|"]
    for item in plan["scenarios"]:
        quotas = " / ".join(map(str, item["quotas"].values()))
        lines.append(f"| {item['scenario']} | {item['group']} | {item['distance']['factorial']:.4f} | "
                     f"{item['distance']['refinado']:.4f} | {quotas} |")
    lines += ["", "Cuotas en el orden straight, semi_complex, complex y v_complex. "
              "Las distancias son los mínimos Wasserstein a cada conjunto de entrenamiento.", "",
              "| Grupo | Modelo | Llegadas ≤ 0.25 m | Llegadas aceptables ≤ 0.35 m | Colisiones | "
              "Tiempo mediano de llegada aceptable (s) |",
              "|---|---|---:|---:|---:|---:|"]
    blocks = {
        "Todas": rows,
        "Vistos": [r for r in rows if r["group"].startswith("visto")],
        "No vistos similares": [r for r in rows if r["group"] == "no_visto_similar"],
        "No visto distante": [r for r in rows if r["group"] == "no_visto_distante"],
        **{item["scenario"]: [r for r in rows if r["scenario"] == item["scenario"]]
           for item in plan["scenarios"]},
        **{cls: [r for r in rows if r["route_class"] == cls] for cls in CLASSES},
    }
    for block, group in blocks.items():
        if not group:
            continue
        for label in plan["models"]:
            subset = [r for r in group if r["model"] == label]
            n = len(subset)
            elapsed = [r["elapsed_sim_s"] for r in subset if r["acceptable"]]
            median = f"{statistics.median(elapsed):.2f}" if elapsed else "—"
            lines.append(f"| {block} | {label} | {sum(r['strict'] for r in subset)}/{n} | "
                         f"{sum(r['acceptable'] for r in subset)}/{n} | {sum(r['collision'] for r in subset)}/{n} | {median} |")
    paired = collections.defaultdict(dict)
    for row in rows:
        paired[row["scenario"], row["task_id"]][row["model"]] = row["acceptable"]
    counts = collections.Counter((entry["factorial"], entry["refinado"]) for entry in paired.values())
    lines += ["", "## Llegadas aceptables por ruta emparejada", "",
              f"- Ambos llegaron en {counts[True, True]} rutas.",
              f"- Solo el factorial llegó en {counts[True, False]} rutas.",
              f"- Solo el refinado llegó en {counts[False, True]} rutas.",
              f"- Ninguno llegó en {counts[False, False]} rutas.", "",
              "Las llegadas se cuentan sin colisión. Las rutas son nuevas y se reutilizan entre los dos modelos. "
              "El puntaje compuesto histórico requiere trayectorias de demostración para estas mismas rutas."]
    target = root / "comparacion.md"
    temp = target.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    campaigns = DATASET / "data/campaigns"
    parser.add_argument("--goals", type=int, choices=(30, 50), default=50, help="Shared goals, repeated for each model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, help="Override the selected factorial checkpoint")
    parser.add_argument("--refinado", type=Path, help="Override the refiner winner checkpoint")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generation-timeout", type=int, default=300)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rviz", action="store_true", help="RViz is always enabled")
    args = parser.parse_args()
    try:
        if (args.seed is not None and not 0 < args.seed <= 2147483642) or args.generation_timeout <= 0:
            raise ValueError("Use a positive seed up to 2147483642 and a positive generation timeout")
        if not args.prepare_only and any(not shutil.which(name) for name in ("ros2", "webots")):
            raise ValueError("Source install/setup.bash before running Webots")
        root = args.output_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".comparison.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_comparison(args, campaigns, root)
    except BlockingIOError:
        parser.exit(2, "ERROR: A comparison is already running in this output directory.\n")
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Resume with the same output directory.\n")


def run_comparison(args, campaigns, root):
    base = campaigns / "refinamiento_5x4_v1"
    refined = campaigns / "refinador_5x4_v1"
    campaign = Campaign(base)
    ev = evaluator()
    datasets = {"factorial": base / "datasets_manifest", "refinado": refined / "stage3/datasets_manifest"}
    checkpoints = {
        "factorial": args.factorial or factorial_winner(campaigns / "refinamiento_5x4_v1_factorial/trials_factorial.csv"),
        "refinado": args.refinado or refined / "ganador/best_model.pth",
    }
    metadata = {label: read_json(path / "datasets.json") for label, path in datasets.items()}
    models, training = model_details(ev, checkpoints, metadata)
    matrix = args.matrix or (ev.HARDWARE / "scripts/maps/similarity/30env_wasserstein_sinkhorn_mean.csv")
    state = read_json(refined / "refiner_state.json")
    excluded = (set(campaign.cfg["screen"]) | set(campaign.cfg["confirmation"]) |
                set(state["stage2"]["development"]) | set(state["stage2"]["final"]))
    items = select_scenarios(read_matrix(matrix), training, excluded, args.goals)
    profile = read_json(base / "prepared.json")["profiles"]["demonstration"]
    generator = DATASET / "scripts/path_generator.py"
    if not generator.is_file():
        generator = DATASET / "scripts/data_analysis/path_generator.py"
    plan_path = root / "plan.json"
    saved = read_json(plan_path) if plan_path.exists() else None
    seed = args.seed if args.seed is not None else (saved["seed"] if saved else random.SystemRandom().randrange(1, 2147483643))
    prior_suites = sorted([*(base / "suites").glob("*.json"), *(refined / "suites").glob("*.json")])
    plan = {"goals": args.goals, "seed": seed, "models": models, "training": training, "scenarios": items,
            "excluded_previous_evaluations": sorted(excluded), "profile": profile, "rviz": True,
            "matrix_sha256": file_hash(matrix), "generator_sha256": file_hash(generator),
            "evaluator_sha256": file_hash(EVALUATOR),
            "policy_sha256": file_hash(EVALUATOR.with_name("RNN_mapless.py")),
            "model_definition_sha256": file_hash(EVALUATOR.parent / "TorchModels/models.py"),
            "prior_suites": {str(path): file_hash(path) for path in prior_suites}}
    if saved is not None and {**saved, "rviz": True} != plan:
        raise ValueError("The comparison plan changed; use the original arguments or a new output directory")
    if saved != plan:
        write_json(plan_path, plan)
    print(f"Plan: {plan_path}", flush=True)
    for label, model in models.items():
        print(f"{label}: {model['checkpoint']} ({display_set(model, training[label])})", flush=True)
    for item in items:
        print(f"{item['scenario']} {item['group']} {item['distance']}", flush=True)
    suites = [prepare_suite(campaign, ev, root, item, profile, seed+i, generator, args.generation_timeout, prior_suites)
              for i, item in enumerate(items)]
    if args.prepare_only:
        return
    rows = []
    labels = list(models)
    for i, (item, suite) in enumerate(zip(items, suites)):
        order = labels if i % 2 == 0 else labels[::-1]
        for label in order:
            result = run_model(campaign, root, label, models[label], suite, datasets[label])
            for trial in result["results"]:
                collision = trial["collision"]
                distance = trial["final_goal_distance_m"]
                rows.append({"scenario": item["scenario"], "group": item["group"], "model": label,
                             "task_id": trial["task_id"], "route_class": trial["route_class"],
                             "strict": distance <= 0.25 and not collision,
                             "acceptable": distance <= 0.35 and not collision,
                             "collision": collision, "reason": trial["reason"], "final_goal_distance_m": distance,
                             "elapsed_sim_s": trial["elapsed_sim_s"], "path_length_m": trial["path_length_m"]})
        report(root, plan, rows)
    print(f"Comparison: {root / 'comparacion.md'}")


if __name__ == "__main__":
    main()
