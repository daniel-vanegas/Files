#!/usr/bin/env python3
"""Build reproducible experiment datasets from accepted ROS 2 demonstrations."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import random
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = (SCRIPT_DIR / "../../../dataset/data").resolve()
MAPS_ROOT = SCRIPT_DIR.parents[5] / "src/Hardware/justina_description/webots_env/environments/maps"
ALL_SCENARIOS = [f"E{i:02d}" for i in range(1, 31)]
DEFAULT_CONFIG = {
    "schema_version": 1,
    "environment_sets": {
        "T0": ["E01", "E02", "E03"],
        "T1": ["E01", "E02", "E03", "E12", "E17", "E22", "E27", "E30"],
        "T2": ["E01", "E02", "E03", "E07", "E09", "E11", "E12", "E13", "E14", "E15", "E17", "E18", "E21", "E22", "E25", "E26", "E28", "E30"],
    },
    "simulation_development_scenarios": ["E08", "E16", "E20"],
    "final_scenarios": ["E04", "E05", "E06", "E10", "E19", "E23", "E24", "E29"],
    "validation_episode_fraction": 0.2,
    "sequence_length": 2,
    "sequence_stride": 2,
    "lom_encoding": "raw_0_100",
    "linear_deadband_m_s": 0.02,
    "angular_deadband_rad_s": 0.05,
    "max_episode_fraction_per_behavior": 0.1,
    "min_episodes_per_evasion_side": 2,
    "behaviors": {
        "B0": {"name": "free_advance", "requires_evasion_side": False},
        "B1": {"name": "simple_lateral_avoidance", "requires_evasion_side": True},
        "B2": {"name": "restricted_avoidance", "requires_evasion_side": True},
        "B3": {"name": "frontal_detour", "requires_evasion_side": True},
        "B4": {"name": "recovery", "requires_evasion_side": True},
    },
    "recipes": {
        "D0": {"mode": "all"},
        "D1": {"mode": "available_equal"},
        "D2": {"weights": {"B0": 0.2, "B1": 0.4, "B2": 0.4}},
        "D3": {"weights": {"B0": 0.15, "B1": 0.3, "B2": 0.25, "B3": 0.2, "B4": 0.1}},
    },
}
REVIEW_FIELDS = [
    "segment_id", "scenario", "run_file", "episode_id", "pair_start", "pair_stop",
    "num_pairs", "motion_signature", "vx_mean", "vy_mean", "wz_mean",
    "theta_mean", "front_occupied_fraction", "left_occupied_fraction",
    "right_occupied_fraction", "behavior", "evasion_side", "reviewed", "notes",
]
TEMPORAL_FIELDS = REVIEW_FIELDS + [
    "annotation_schema", "annotation_scope", "maneuver", "context", "phase",
    "review_status", "catalog_sha256", "source_segment_id",
]
MANEUVERS = {
    "goal_progress": "Progreso hacia el objetivo", "lateral_avoidance": "Evasión lateral",
    "restricted_avoidance": "Evasión restringida", "frontal_detour": "Rodeo frontal",
    "door_passage": "Cruce de puerta", "corridor_navigation": "Recorrido de pasillo",
    "undetermined": "Sin determinar",
}
CONTEXTS = {
    "doorway": "Puerta", "corridor": "Pasillo", "open_area": "Espacio abierto",
    "obstacles_both_sides": "Obstáculos a ambos lados", "sparse_visibility": "Obstáculo poco visible",
}
PHASES = {
    "approach": "Aproximación", "alignment": "Alineación", "crossing": "Cruce",
    "avoidance": "Evasión", "recovery": "Recuperación", "progress": "Avance",
    "undetermined": "Sin determinar",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def generated_seed(value: int | None) -> int:
    seed = value if value is not None else random.SystemRandom().randrange(2**32)
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32 - 1")
    return seed


def read_config(path: Path | None) -> dict:
    c = json.loads(path.read_text()) if path else json.loads(json.dumps(DEFAULT_CONFIG))
    if c.get("schema_version") != 1:
        raise ValueError("Unsupported configuration version")
    if (c["sequence_length"], c["sequence_stride"], c["lom_encoding"]) != (2, 2, "raw_0_100"):
        raise ValueError("This stage keeps disjoint C=2 pairs and the raw 0/100 LOM")
    used = set()
    for name, maps in c["environment_sets"].items():
        if not name.isidentifier() or not maps or len(set(maps)) != len(maps) or not set(maps) <= set(ALL_SCENARIOS):
            raise ValueError(f"Invalid environment set: {name}")
        used.update(maps)
    dev, final = c["simulation_development_scenarios"], c["final_scenarios"]
    if (len(set(dev)) != len(dev) or len(set(final)) != len(final)
            or set(dev) & set(final) or used & (set(dev) | set(final))
            or used | set(dev) | set(final) != set(ALL_SCENARIOS)):
        raise ValueError("Training, simulation development and final scenario roles are inconsistent")
    if not 0 < c["validation_episode_fraction"] < 1:
        raise ValueError("The episode validation fraction must be between zero and one")
    if not 0 < c["max_episode_fraction_per_behavior"] <= 1:
        raise ValueError("The episode contribution limit must be between zero and one")
    if type(c["min_episodes_per_evasion_side"]) is not int or c["min_episodes_per_evasion_side"] < 1:
        raise ValueError("At least one episode per side is required")
    if c["linear_deadband_m_s"] <= 0 or c["angular_deadband_rad_s"] <= 0:
        raise ValueError("Motion deadbands must be positive")
    if not c["behaviors"]:
        raise ValueError("At least one behavior is required")
    for name, behavior in c["behaviors"].items():
        if not name.isidentifier() or not isinstance(behavior["requires_evasion_side"], bool):
            raise ValueError(f"Invalid behavior definition: {name}")
    for name, recipe in c["recipes"].items():
        if not name.isidentifier():
            raise ValueError(f"Invalid recipe name: {name}")
        if "weights" in recipe:
            weights = recipe["weights"]
            if not weights or not set(weights) <= set(c["behaviors"]) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
                raise ValueError(f"Invalid behavior weights: {name}")
        elif recipe.get("mode") not in ("all", "available_equal"):
            raise ValueError(f"Unknown recipe mode: {name}")
    return c


def provenance() -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "source_baseline": "Files/main@0eab89f7b7a4abe308dbed9001ff5b1accd321be",
            "controller_sha256": file_hash(Path(__file__)), "numpy": np.__version__}


def new_output(value: Path | None, phase: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = (value or SCRIPT_DIR / "experiments" / f"{phase}_{stamp}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def read_samples(path: Path) -> list[dict]:
    # The project NPZ files contain trusted Python dictionaries.
    with np.load(path, allow_pickle=True) as archive:
        return archive["data"].tolist()


def grouped_indices(samples: list[dict]) -> dict[int, list[int]]:
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[int(sample["meta"]["episode_id"])].append(i)
    return dict(groups)


def pair_signature(samples: list[dict], indices: list[int], c: dict) -> str:
    actions = np.asarray([samples[i]["labels"] for i in indices])
    dead = c["linear_deadband_m_s"]
    y = actions[:, 1]
    if np.all(y > dead):
        return "positive_lateral_command"
    if np.all(y < -dead):
        return "negative_lateral_command"
    if np.any(np.abs(y) > dead):
        return "lateral_transition"
    if np.any(np.abs(actions[:, 2]) > c["angular_deadband_rad_s"]):
        return "rotation_without_lateral_command"
    if np.any(np.abs(actions[:, 0]) > dead):
        return "longitudinal_command"
    return "below_deadbands"


def region_masks() -> dict[str, np.ndarray]:
    # The ROS 2 recorder rotates grid[y, x] counterclockwise.
    row, col = np.indices((80, 80))
    x = (79 - row + 0.5) * 0.05
    y = (col + 0.5) * 0.05 - 2.0
    return {"front": (x >= 0.3) & (x < 1.5) & (np.abs(y) < 0.35),
            "left": (x < 1.5) & (y >= 0.35) & (y < 1.0),
            "right": (x < 1.5) & (y <= -0.35) & (y > -1.0)}


def segments_for_episode(episode: dict, samples: list[dict], c: dict) -> list[dict]:
    indices = episode["source_indices"]
    pairs = [indices[i:i + 2] for i in range(0, len(indices) - 1, 2)]
    signatures = [pair_signature(samples, pair, c) for pair in pairs]
    masks = region_masks()
    segments = []
    start = 0
    while start < len(pairs):
        stop = start + 1
        while stop < len(pairs) and signatures[stop] == signatures[start]:
            stop += 1
        selected = [i for pair in pairs[start:stop] for i in pair]
        actions = np.asarray([samples[i]["labels"] for i in selected], dtype=float)
        occupancies = np.stack([samples[i]["features"]["occ_grid"] for i in selected])
        token = f"{episode['episode_key']}:{start}:{stop}"
        row = {"segment_id": hashlib.sha256(token.encode()).hexdigest()[:20],
               "scenario": episode["scenario"], "run_file": episode["run_file"],
               "episode_id": episode["episode_id"], "pair_start": start, "pair_stop": stop,
               "num_pairs": stop - start, "motion_signature": signatures[start],
               "vx_mean": float(actions[:, 0].mean()), "vy_mean": float(actions[:, 1].mean()),
               "wz_mean": float(actions[:, 2].mean()),
               "theta_mean": float(np.mean([samples[i]["features"]["target"][1] for i in selected])),
               "behavior": "", "evasion_side": "", "reviewed": "", "notes": ""}
        for name, mask in masks.items():
            row[f"{name}_occupied_fraction"] = float((occupancies[:, mask] > 0).mean())
        segments.append(row)
        start = stop
    return segments


def split_episodes(episodes: list[dict], c: dict, seed: int) -> dict:
    by_scenario = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        if e["num_pairs"]:
            by_scenario[e["scenario"]][e["content_sha256"]].append(e)
    roles = {}
    for name, maps in c["environment_sets"].items():
        assignment = {}
        for scenario in maps:
            groups = by_scenario[scenario]
            if len(groups) < 2:
                raise ValueError(f"{scenario}: two distinct episode groups are needed for the split")
            ordered = sorted(groups, key=lambda h: hashlib.sha256(f"{seed}:{scenario}:{h}".encode()).digest())
            nval = max(1, min(len(ordered) - 1, int(len(ordered) * c["validation_episode_fraction"])))
            val = set(ordered[:nval])
            for h, group in groups.items():
                for e in group:
                    assignment[e["episode_key"]] = "validation" if h in val else "train"
        fingerprints = defaultdict(set)
        for e in episodes:
            fingerprints[e["content_sha256"]].add(assignment.get(e["episode_key"], "reserved"))
        if any(len(v) > 1 for v in fingerprints.values()):
            raise ValueError(f"Duplicate episode content crosses partitions in {name}")
        roles[name] = assignment
    return roles


def catalog(root: Path, c: dict, out: Path, seed: int, previews: int,
            maps_root: Path | None = None, map_variant: str = "prohibition",
            map_padding: float = 1.25, lom_view: str = "robot") -> dict:
    root = root.expanduser().resolve()
    files = sorted(root.glob("E*/*.npz"))
    if {p.parent.name for p in files} != set(ALL_SCENARIOS):
        raise ValueError("The configured experiment needs the existing E01 through E30 NPZ files")
    episodes, segments, runs, labels_all = [], [], [], []
    for path in files:
        samples = read_samples(path)
        relative = path.relative_to(root).as_posix()
        groups = grouped_indices(samples)
        labels = np.asarray([s["labels"] for s in samples], dtype=float)
        labels_all.append(labels)
        runs.append({"run_file": relative, "sha256": file_hash(path), "num_samples": len(samples),
                     "num_episodes": len(groups)})
        for eid, indices in groups.items():
            h = hashlib.sha256()
            for i in indices:
                s = samples[i]
                for value in (s["features"]["occ_grid"], s["features"]["target"], s["labels"]):
                    h.update(np.asarray(value, dtype=np.float32).tobytes())
            e = {"episode_key": f"{relative}#{eid}", "scenario": path.parent.name,
                 "run_file": relative, "episode_id": eid, "source_indices": indices,
                 "num_pairs": len(indices) // 2, "content_sha256": h.hexdigest()}
            episodes.append(e)
            segments.extend(segments_for_episode(e, samples, c))
        print(f"catalog {path.parent.name}: {len(samples)} samples, {len(groups)} episodes", flush=True)
        del samples
    roles = split_episodes(episodes, c, seed)
    with (out / "episodes.jsonl").open("w") as stream:
        for e in episodes:
            stream.write(json.dumps(e) + "\n")
    train_keys = {key for assignment in roles.values() for key, role in assignment.items() if role == "train"}
    review_rows = [r for r in segments if f"{r['run_file']}#{r['episode_id']}" in train_keys]
    for filename, rows in (("segments.csv", segments), ("review_template.csv", review_rows)):
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    labels = np.concatenate(labels_all)
    dead = c["linear_deadband_m_s"]
    directional_counts = {"positive_lateral_command": int((labels[:, 1] > dead).sum()),
                          "negative_lateral_command": int((labels[:, 1] < -dead).sum()),
                          "within_lateral_deadband": int((np.abs(labels[:, 1]) <= dead).sum())}
    result = {"schema_version": 1, "phase": "catalog", "config": c, "split_seed": seed,
              "data_root": str(root), "provenance": provenance(), "runs": runs,
              "num_samples": len(labels), "num_episodes": len(episodes),
              "num_pairs": sum(e["num_pairs"] for e in episodes), "num_segments": len(segments),
              "reviewable_training_segments": len(review_rows), "action_counts": directional_counts,
              "motion_segment_counts": dict(Counter(r["motion_signature"] for r in segments)),
              "episode_roles": roles,
              "manifests": {p.name: file_hash(p) for p in (out / "episodes.jsonl", out / "segments.csv", out / "review_template.csv")},
              "data_status": "Accepted demonstrations. No collision or quality filtering was applied.",
              "behavior_status": "Motion signatures are not behavior annotations."}
    write_json(out / "config.json", c)
    if previews:
        try:
            make_review(root, out, episodes, review_rows, seed, previews, c,
                        maps_root=maps_root, map_variant=map_variant, map_padding=map_padding,
                        lom_view=lom_view)
            result["preview_rows_sha256"] = file_hash(out / "preview_rows.csv")
        except ImportError as exc:
            result["preview_status"] = f"Optional plotting dependency unavailable: {exc}"
    write_json(out / "catalog.json", result)
    return result


def select_review_rows(rows: list[dict], seed: int, count: int, minimum_pairs: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if int(row["num_pairs"]) >= minimum_pairs:
            groups[(row["scenario"], row["motion_signature"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    order = sorted(groups)
    rng.shuffle(order)
    for signature in ("positive_lateral_command", "negative_lateral_command"):
        target = len(selected) + count // 3
        while len(selected) < target:
            progress = False
            for key in order:
                if key[1] == signature and groups[key] and len(selected) < target:
                    selected.append(groups[key].pop())
                    progress = True
            if not progress:
                break
    for include_lateral in (False, True):
        while len(selected) < count:
            progress = False
            for key in order:
                lateral = key[1] in ("positive_lateral_command", "negative_lateral_command")
                if groups[key] and (include_lateral or not lateral) and len(selected) < count:
                    selected.append(groups[key].pop())
                    progress = True
            if not progress:
                break
    return selected


def load_review_map(maps_root: Path, scenario: str, variant: str) -> dict:
    """Read the static scenario map for annotation displays."""
    import yaml
    from PIL import Image
    suffix = "" if variant == "standard" else f"_{variant}"
    folders = [maps_root] if maps_root.name == scenario or maps_root.name.startswith(scenario + "_") else []
    folders.extend(p for p in maps_root.glob(f"{scenario}*")
                   if p.is_dir() and (p.name == scenario or p.name.startswith(scenario + "_")))
    paths = sorted({folder / f"map_{folder.name}{suffix}.yaml" for folder in folders
                    if (folder / f"map_{folder.name}{suffix}.yaml").is_file()})
    if len(paths) != 1:
        raise ValueError(f"Expected one {variant} map YAML for {scenario} in {maps_root}; found {len(paths)}")
    yaml_path = paths[0]
    metadata = yaml.safe_load(yaml_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid map metadata: {yaml_path}")
    resolution = float(metadata["resolution"])
    origin = np.asarray(metadata["origin"], dtype=float)
    if resolution <= 0 or not math.isfinite(resolution) or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError(f"Invalid map resolution or origin: {yaml_path}")
    image_path = (yaml_path.parent / metadata["image"]).expanduser().resolve()
    with Image.open(image_path) as image:
        pixels = np.array(image.convert("L"))
    if int(metadata.get("negate", 0)):
        pixels = 255 - pixels
    return {"pixels": pixels, "resolution": resolution, "origin": origin,
            "source": {"scenario": scenario, "variant": variant, "pose_frame": "map",
                       "yaml": str(yaml_path), "yaml_sha256": file_hash(yaml_path),
                       "image": str(image_path), "image_sha256": file_hash(image_path),
                       "resolution": resolution, "origin": origin.tolist(),
                       "width": pixels.shape[1], "height": pixels.shape[0]}}


def map_local_xy(points: np.ndarray, background: dict) -> np.ndarray:
    """Convert map coordinates to the metric axes of the map image."""
    origin = background["origin"]
    dx, dy = (np.asarray(points) - origin[:2]).T
    yaw = origin[2]
    return np.column_stack((dx * math.cos(yaw) + dy * math.sin(yaw),
                            -dx * math.sin(yaw) + dy * math.cos(yaw)))


def draw_review_map(ax, background: dict | None) -> None:
    if background is None:
        ax.text(0.02, 0.98, "Map unavailable", transform=ax.transAxes, va="top", color="crimson")
        return
    from matplotlib.transforms import Affine2D
    image = background["pixels"]
    resolution, origin = background["resolution"], background["origin"]
    # Recorded poses already use the ROS map frame.
    transform = Affine2D().rotate(origin[2]).translate(*origin[:2]) + ax.transData
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper", interpolation="nearest",
              extent=(0, image.shape[1] * resolution, 0, image.shape[0] * resolution),
              transform=transform, zorder=0)


def draw_review_path(ax, background: dict | None, pose: np.ndarray, highlight: np.ndarray,
                     goal: np.ndarray, snapshots: np.ndarray, padding: float, detail: bool) -> None:
    draw_review_map(ax, background)
    ax.plot(pose[:, 0], pose[:, 1], color="#909aa8", linewidth=1.5, label="Episode", zorder=2)
    ax.plot(highlight[:, 0], highlight[:, 1], color="#e06400", linewidth=3, label="Segment", zorder=3)
    ax.scatter(*pose[0, :2], marker="o", color="#047857", s=40, label="Episode start", zorder=5)
    ax.scatter(*goal, marker="*", color="#bd163c", s=120, label="Goal", zorder=5)
    ax.scatter(*highlight[0, :2], marker="o", color="#e06400", edgecolor="white", s=45, zorder=5)
    ax.scatter(*highlight[-1, :2], marker="s", color="#e06400", edgecolor="white", s=40, zorder=5)
    center = snapshots[len(snapshots) // 2]
    if detail:
        ax.plot([center[0], goal[0]], [center[1], goal[1]], color="#bd163c", linewidth=1,
                linestyle="--", label="Goal direction", zorder=1)
    for number, p in enumerate(snapshots, 1):
        dx, dy = 0.35 * math.cos(p[2]), 0.35 * math.sin(p[2])
        ax.annotate("", xy=(p[0] + dx, p[1] + dy), xytext=p[:2],
                    arrowprops={"arrowstyle": "->", "color": "#4c1d95", "lw": 1.5}, zorder=6)
        ax.annotate(str(number), xy=p[:2], xytext=(4, 7), textcoords="offset points",
                    color="#4c1d95", fontsize=9, fontweight="bold", zorder=7)
    if detail:
        yaw = center[2]
        left = center[:2] + 0.5 * np.array([-math.sin(yaw), math.cos(yaw)])
        ax.annotate("Left", xy=left, xytext=center[:2], color="#007d8a", fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": "#007d8a", "lw": 1}, zorder=6)
    points = highlight[:, :2] if detail else np.vstack((pose[:, :2], goal))
    low, high = points.min(axis=0) - padding, points.max(axis=0) + padding
    mid = (low + high) / 2
    half = max(float((high - low).max()), 3.0) / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)


def draw_review_lom(ax, occupancy: np.ndarray, view: str) -> None:
    """Display stored LOM cells using robot axes or array indices."""
    if view == "robot":
        # Stored columns increase toward the robot's left side.
        ax.imshow(occupancy, cmap="gray_r", vmin=0, vmax=100, origin="upper",
                  interpolation="nearest", extent=(-2.0, 2.0, 0.0, 4.0))
        ax.set_xlim(2.0, -2.0)
        ax.set_ylim(0.0, 4.0)
        ax.scatter(0.0, 0.06, marker="^", color="#4c1d95", s=45, clip_on=False)
        ax.set_xlabel("Body y (m): left (+) / right (-)")
        ax.set_ylabel("Body x (m): forward (+)")
    elif view == "array":
        ax.imshow(occupancy, cmap="gray_r", vmin=0, vmax=100, origin="upper", interpolation="nearest")
        ax.set_xlabel("Stored LOM columns")
        ax.set_ylabel("Stored LOM rows")
    else:
        raise ValueError("Unknown LOM view")


def make_review(root: Path, out: Path, episodes: list[dict], rows: list[dict], seed: int,
                count: int, c: dict, minimum_pairs: int = 2, maps_root: Path | None = None,
                map_variant: str = "prohibition", map_padding: float = 1.25,
                preserve_order: bool = False, lom_view: str = "robot") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not math.isfinite(map_padding) or map_padding <= 0:
        raise ValueError("Map padding must be positive and finite")
    if map_variant not in ("prohibition", "nav", "standard"):
        raise ValueError("Unknown map variant")
    if lom_view not in ("robot", "array"):
        raise ValueError("Unknown LOM view")
    selected = list(rows) if preserve_order else select_review_rows(rows, seed, count, minimum_pairs)
    backgrounds = {}
    resolved_maps = (maps_root or MAPS_ROOT).expanduser().resolve()
    if maps_root is not None or resolved_maps.is_dir():
        for scenario in sorted({row["scenario"] for row in selected}):
            backgrounds[scenario] = load_review_map(resolved_maps, scenario, map_variant)
    else:
        print(f"Review maps unavailable: {resolved_maps}. Use --maps-root to locate them.", flush=True)
    with (out / "preview_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    ep_by_key = {e["episode_key"]: e for e in episodes}
    image_dir = out / "review_images"
    image_dir.mkdir()
    cached_path, samples = None, None
    for row in selected:
        if row["run_file"] != cached_path:
            samples = read_samples(root / row["run_file"])
            cached_path = row["run_file"]
        e = ep_by_key[f"{row['run_file']}#{row['episode_id']}"]
        indices = e["source_indices"]
        segment_indices = indices[2 * int(row["pair_start"]):2 * int(row["pair_stop"])]
        pose = np.array([samples[i]["meta"]["pose"] for i in indices])
        highlight = np.array([samples[i]["meta"]["pose"] for i in segment_indices])
        goal = np.asarray(samples[indices[0]]["meta"]["goal_xy"])
        snapshot_ids = [segment_indices[0], segment_indices[len(segment_indices) // 2], segment_indices[-1]]
        snapshots = np.array([samples[i]["meta"]["pose"] for i in snapshot_ids])
        background = backgrounds.get(row["scenario"])
        if background is not None:
            local = map_local_xy(pose[:, :2], background)
            h, w = background["pixels"].shape
            bounds = np.array([w, h]) * background["resolution"]
            if not ((local >= -0.1).all(axis=1) & (local <= bounds + 0.1).all(axis=1)).any():
                raise ValueError(f"Episode {e['episode_key']} lies outside the map image; check the map frame")
        fig = plt.figure(figsize=(13, 9), constrained_layout=True)
        grid = fig.add_gridspec(2, 6, height_ratios=[1.25, 1])
        overview, detail = fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])
        draw_review_path(overview, background, pose, highlight, goal, snapshots, 0.8, False)
        draw_review_path(detail, background, pose, highlight, goal, snapshots, map_padding, True)
        overview.set_title(f"{row['scenario']} episode {row['episode_id']} | Context", fontsize=11)
        detail.set_title("Segment area | Purple arrows: robot heading", fontsize=11)
        for number, i in enumerate(snapshot_ids, 1):
            ax = fig.add_subplot(grid[1, 2 * (number - 1):2 * number])
            s = samples[i]
            draw_review_lom(ax, s["features"]["occ_grid"], lom_view)
            d, theta = s["features"]["target"]
            ax.set_title(f"{number}. Sample {i}\nd={d:.2f}, theta={theta:.2f}", fontsize=10)
        fig.suptitle(f"{row['motion_signature']} | Pairs [{row['pair_start']}, {row['pair_stop']}) | LOM view: {lom_view}", fontsize=11)
        fig.savefig(image_dir / f"{row['segment_id']}.png", dpi=150)
        plt.close(fig)
        print(f"review {row['scenario']} segment {row['segment_id']}", flush=True)
    write_json(out / "review_maps.json", {"pose_frame": "map", "map_padding_m": map_padding,
               "map_variant": map_variant, "lom_view": lom_view, "lom_array_transform": "none",
               "maps": {s: b["source"] for s, b in backgrounds.items()},
               "map_status": "available" if backgrounds else "unavailable"})
    cards = []
    for i, row in enumerate(selected):
        cards.append(f'<article><h3>{html.escape(row["segment_id"])} · {row["scenario"]}</h3>'
                     f'<a href="review_images/{row["segment_id"]}.png" target="_blank" rel="noopener">'
                     f'<img src="review_images/{row["segment_id"]}.png" alt="Path on map and native LOM"></a>'
                     f'<label>Behavior <select id="b{i}"><option value="">Unassigned</option>'
                     + ''.join(f'<option>{b}</option>' for b in c["behaviors"])
                     + f'</select></label> <label>Evasion side <select id="s{i}">'
                     '<option value="">Unassigned</option><option>left</option><option>right</option>'
                     '<option>neutral</option></select></label> '
                     f'<label><input type="checkbox" id="r{i}">Reviewed</label></article>')
    payload = json.dumps(selected).replace("<", "\\u003c")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Behavior review</title>
<style>body{font:16px system-ui;margin:2rem;max-width:1500px;background:#f5f7fa;color:#17263d}article{background:white;padding:1rem;margin:1rem 0;border-radius:10px}img{width:100%;height:auto}select,button{padding:.6rem}header{position:sticky;top:0;background:#f5f7fa;padding:1rem 0}</style>
<header><h1>Behavior review</h1><p>B0: free advance. B1: simple lateral avoidance. B2: restricted avoidance.
B3: frontal detour. B4: recovery. For B4, record the side of the preceding evasion.
Motion signatures and region densities do not establish these behaviors. Inspect the episode and obstacles.
Only mark verified labels. These previews use training episodes.</p>
<p>The map is used only for annotation. Orange: selected segment. Gray: episode path.
Purple arrows: robot heading. The Left arrow is relative to the robot at snapshot 2.
Snapshots 1, 2 and 3 match the LOM frames below. LOM noise and stored arrays are preserved.
LOM_VIEW_DESCRIPTION
Click an image to open it at full size.</p><button onclick="save()">Export reviewed annotations</button></header>
''' + "\n".join(cards) + '<script>const rows=' + payload + ';const fields=' + json.dumps(REVIEW_FIELDS) + ''';
function save(){const output=[fields];rows.forEach((row,i)=>{if(!document.getElementById('r'+i).checked)return;
const behavior=document.getElementById('b'+i).value;const side=document.getElementById('s'+i).value;
if(!behavior||!side)return;output.push(fields.map(k=>k==='behavior'?behavior:k==='evasion_side'?side:k==='reviewed'?'1':row[k]??''));});
const csv=output.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\\n');
const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='reviewed_annotations.csv';a.click();URL.revokeObjectURL(url);}
</script></html>'''
    description = ("Robot view: forward is up and the robot's left side is on the left. "
                   "Only display axes change; this is not the tensor layout used by the current trainer."
                   if lom_view == "robot" else
                   "Array view: columns and rows match the stored NPZ tensor layout. "
                   "This view does not place the robot's left side on the left.")
    (out / "review.html").write_text(document.replace("LOM_VIEW_DESCRIPTION", description))


def read_catalog(directory: Path, relocated_root: Path | None) -> tuple[dict, list[dict], Path]:
    s = json.loads((directory / "catalog.json").read_text())
    if s.get("schema_version") != 1 or s.get("phase") != "catalog":
        raise ValueError("Unsupported catalog")
    config_path = directory / "config.json"
    c = read_config(config_path)
    if c != s["config"]:
        raise ValueError("Catalog configuration changed")
    for name, expected in s["manifests"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Catalog manifest changed: {name}")
    root = (relocated_root or Path(s["data_root"])).expanduser().resolve()
    current = {p.relative_to(root).as_posix() for p in root.glob("E*/*.npz")}
    if current != {r["run_file"] for r in s["runs"]}:
        raise ValueError("Dataset inventory changed since catalog creation")
    for run in s["runs"]:
        if file_hash(root / run["run_file"]) != run["sha256"]:
            raise ValueError(f"Dataset content changed: {run['run_file']}")
    episodes = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    return s, episodes, root


def read_annotations(paths: list[Path], episodes: list[dict], roles: dict, c: dict,
                     catalog_sha256: str | None = None) -> dict:
    by_key = {e["episode_key"]: e for e in episodes}
    permitted = {key for assignment in roles.values() for key, role in assignment.items() if role == "train"}
    annotations, local_values = {}, {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                schema = row.get("annotation_schema", "1").strip() or "1"
                if schema not in ("1", "2"):
                    raise ValueError(f"Unsupported annotation schema: {schema}")
                if schema == "1" and row.get("reviewed", "").strip().lower() not in ("1", "true", "yes"):
                    continue
                key = f"{row['run_file']}#{int(row['episode_id'])}"
                if key not in permitted:
                    raise ValueError(f"Annotations must refer to training episodes: {key}")
                e = by_key[key]
                start, stop = int(row["pair_start"]), int(row["pair_stop"])
                behavior, side = row["behavior"].strip(), row["evasion_side"].strip()
                if not 0 <= start < stop <= e["num_pairs"] or row["scenario"] != e["scenario"]:
                    raise ValueError(f"Invalid annotation interval: {key}")
                if schema == "2":
                    if catalog_sha256 is not None and row.get("catalog_sha256") != catalog_sha256:
                        raise ValueError(f"Annotations belong to a different catalog: {path}")
                    scope, status = row.get("annotation_scope", ""), row.get("review_status", "")
                    if scope not in ("local", "maneuver") or status not in ("pending_review", "reviewed", "ambiguous"):
                        raise ValueError(f"Invalid annotation scope or review status: {key}")
                    if side not in ("left", "right", "neutral", "unknown", "not_applicable"):
                        raise ValueError(f"Invalid evasion side: {key}")
                    if behavior and behavior not in c["behaviors"]:
                        raise ValueError(f"Invalid behavior: {key}")
                    context = sorted(set(filter(None, row.get("context", "").split(";"))))
                    value = {"behavior": "unassigned", "evasion_side": side,
                             "context": context, "phase": row.get("phase", "undetermined"),
                             "maneuver": row.get("maneuver", ""), "review_status": status,
                             "notes": row.get("notes", ""), "annotation_scope": scope}
                    if scope == "maneuver":
                        value.update(pair_start=start, pair_stop=stop)
                        for pair_index in range(start, stop):
                            bucket = annotations.setdefault((key, pair_index), {"behavior": "unassigned", "evasion_side": "unassigned"})
                            items = bucket.setdefault("maneuvers", [])
                            if value not in items:
                                items.append(value)
                        continue
                    if status == "reviewed" and behavior:
                        requires_side = c["behaviors"][behavior]["requires_evasion_side"]
                        if requires_side and side not in ("left", "right", "unknown"):
                            raise ValueError(f"Choose an evasion side or unknown for {behavior}: {key}")
                        if not requires_side and side not in ("neutral", "not_applicable"):
                            raise ValueError(f"The evasion side must be not_applicable for {behavior}: {key}")
                        value.update(behavior=behavior, evasion_side=side if requires_side else "neutral")
                    elif behavior:
                        value["proposed_behavior"] = behavior
                else:
                    if behavior not in c["behaviors"] or side not in ("left", "right", "neutral"):
                        raise ValueError(f"Invalid behavior or side: {key}")
                    requires_side = c["behaviors"][behavior]["requires_evasion_side"]
                    if requires_side == (side == "neutral"):
                        raise ValueError(f"The behavior and evasion side are inconsistent: {key}. Review this legacy row in the temporal viewer.")
                    value = {"behavior": behavior, "evasion_side": side}
                for pair_index in range(start, stop):
                    pair_key = (key, pair_index)
                    if pair_key in local_values and local_values[pair_key] != value:
                        raise ValueError(f"Conflicting annotations: {pair_key}")
                    local_values[pair_key] = value
                    annotations.setdefault(pair_key, {}).update(value)
    return annotations


def pairs_for_role(episodes: list[dict], assignment: dict, role: str, annotations: dict) -> list[dict]:
    rows = []
    for e in episodes:
        if assignment.get(e["episode_key"]) != role:
            continue
        for pair_index in range(e["num_pairs"]):
            indices = e["source_indices"][2 * pair_index:2 * pair_index + 2]
            row = {"scenario": e["scenario"], "run_file": e["run_file"],
                   "episode_id": e["episode_id"], "episode_key": e["episode_key"],
                   "pair_index": pair_index, "source_indices": indices, "label_source_index": indices[-1]}
            row.update(annotations.get((e["episode_key"], pair_index), {"behavior": "unassigned", "evasion_side": "unassigned"}))
            rows.append(row)
    return rows


def diverse_pick(rows: list[dict], number: int, episode_cap: int, rng: random.Random,
                 counts: Counter | None = None) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["scenario"]][row["episode_key"]].append(row)
    for group in groups.values():
        for items in group.values():
            rng.shuffle(items)
    selected = []
    if counts is None:
        counts = Counter()
    scenarios = sorted(groups)
    rng.shuffle(scenarios)
    while len(selected) < number:
        progress = False
        for scenario in scenarios:
            keys = [key for key, items in groups[scenario].items() if items and counts[key] < episode_cap]
            if keys and len(selected) < number:
                least = min(counts[key] for key in keys)
                key = rng.choice(sorted(key for key in keys if counts[key] == least))
                selected.append(groups[scenario][key].pop())
                counts[key] += 1
                progress = True
        if not progress:
            break
    return selected


def joint_direction_pick(rows: list[dict], number: int, episode_cap: int,
                         minimum_episodes: int, rng: random.Random) -> list[dict]:
    half = number // 2
    if number % 2 or half < minimum_episodes:
        return []
    counts = Counter()
    chosen = []
    for side in ("left", "right"):
        chosen.extend(diverse_pick([r for r in rows if r["evasion_side"] == side],
                                  half, episode_cap, rng, counts))
    if len(chosen) == number and all(
            len({r["episode_key"] for r in chosen if r["evasion_side"] == side}) >= minimum_episodes
            for side in ("left", "right")):
        return chosen

    # Joint allocation avoids using all shared episode capacity on one side.
    groups = defaultdict(lambda: defaultdict(list))
    scenarios = defaultdict(list)
    for row in rows:
        groups[row["episode_key"]][row["evasion_side"]].append(row)
    for key, group in groups.items():
        example = next(iter(group.values()))[0]
        scenarios[example["scenario"]].append(key)
        for values in group.values():
            rng.shuffle(values)
    names = sorted(scenarios)
    rng.shuffle(names)
    for values in scenarios.values():
        values.sort()
        rng.shuffle(values)
    ordered = []
    while any(scenarios.values()):
        for name in names:
            if scenarios[name]:
                ordered.append(scenarios[name].pop())

    graph = [[], []]

    def node():
        graph.append([])
        return len(graph) - 1

    def edge(source, target, capacity):
        forward = [target, len(graph[target]), capacity]
        backward = [source, len(graph[source]), 0]
        graph[source].append(forward)
        graph[target].append(backward)
        return forward

    sources = {}
    for side in ("left", "right"):
        reserved, remaining = node(), node()
        edge(0, reserved, minimum_episodes)
        edge(0, remaining, half - minimum_episodes)
        sources[side] = reserved, remaining
    allocations = []
    for key in ordered:
        episode = node()
        edge(episode, 1, episode_cap)
        for side in ("left", "right"):
            values = groups[key].get(side, [])
            if not values:
                continue
            available = node()
            reserved, remaining = sources[side]
            edge(reserved, available, 1)
            edge(remaining, available, len(values))
            link = edge(available, episode, len(values))
            allocations.append((values, link))

    flow = 0
    while flow < number:
        levels = [-1] * len(graph)
        levels[0] = 0
        queue = deque([0])
        while queue:
            source = queue.popleft()
            for target, _, capacity in graph[source]:
                if capacity and levels[target] < 0:
                    levels[target] = levels[source] + 1
                    queue.append(target)
        if levels[1] < 0:
            return []
        positions = [0] * len(graph)

        def send(source, limit):
            if source == 1:
                return limit
            while positions[source] < len(graph[source]):
                link = graph[source][positions[source]]
                target, reverse, capacity = link
                if capacity and levels[target] == levels[source] + 1:
                    sent = send(target, min(limit, capacity))
                    if sent:
                        link[2] -= sent
                        graph[target][reverse][2] += sent
                        return sent
                positions[source] += 1
            return 0

        while flow < number:
            sent = send(0, number - flow)
            if not sent:
                break
            flow += sent
    return [row for values, link in allocations for row in values[:len(values) - link[2]]]


def capacity_profile(values) -> tuple[list[int], list[int]]:
    ordered = sorted(values)
    prefix = [0]
    for value in ordered:
        prefix.append(prefix[-1] + value)
    return ordered, prefix


def capped_capacity(profile: tuple[list[int], list[int]], cap: int) -> int:
    ordered, prefix = profile
    position = bisect_right(ordered, cap)
    return prefix[position] + (len(ordered) - position) * cap


def select_recipe(rows: list[dict], recipe: dict, c: dict, seed: int) -> tuple[list[dict], dict]:
    if recipe.get("mode") == "all":
        return rows, {"status": "ready", "selection": "all accepted training pairs"}
    by_behavior = defaultdict(list)
    for row in rows:
        behavior = row["behavior"]
        if behavior in c["behaviors"] and (not c["behaviors"][behavior]["requires_evasion_side"]
                                            or row["evasion_side"] in ("left", "right")):
            by_behavior[row["behavior"]].append(row)
    if recipe.get("mode") == "available_equal":
        weights = {b: 1.0 for b in sorted(by_behavior)}
        if len(weights) < 2:
            return [], {"status": "needs_annotations", "reason": "At least two reviewed behaviors are needed"}
    else:
        weights = recipe["weights"]
    support, capacities, profiles, deficits = {}, {}, {}, []
    for behavior in weights:
        group = by_behavior[behavior]
        profiles[behavior] = {side: capacity_profile(Counter(
            r["episode_key"] for r in group if side == "all" or r["evasion_side"] == side).values())
            for side in ("all", "left", "right")}
        support[behavior] = {side: {"pairs": sum(r["evasion_side"] == side for r in group),
                                  "episodes": len({r["episode_key"] for r in group if r["evasion_side"] == side}),
                                  "scenarios": len({r["scenario"] for r in group if r["evasion_side"] == side})}
                             for side in ("neutral", "left", "right")}
        if not c["behaviors"][behavior]["requires_evasion_side"]:
            capacities[behavior] = len(group)
            if not group:
                deficits.append(f"{behavior}: no reviewed pairs")
        else:
            capacities[behavior] = 2 * min(support[behavior][s]["pairs"] for s in ("left", "right"))
            for side in ("left", "right"):
                if support[behavior][side]["episodes"] < c["min_episodes_per_evasion_side"]:
                    deficits.append(f"{behavior}/{side}: insufficient distinct reviewed episodes")
    if deficits:
        return [], {"status": "needs_annotations", "deficits": deficits, "support": support}
    total_weight = sum(weights.values())
    weights = {b: w / total_weight for b, w in weights.items()}
    upper = int(min(capacities[b] / w for b, w in weights.items()))
    previous, first_failure = None, None
    for budget in range(upper, 0, -1):
        quotas = {}
        for behavior, weight in weights.items():
            count = int(budget * weight)
            if c["behaviors"][behavior]["requires_evasion_side"]:
                count -= count % 2
            quotas[behavior] = {"target_pairs": count,
                                "episode_cap": max(1, math.ceil(count * c["max_episode_fraction_per_behavior"]))}
        signature = tuple(q["target_pairs"] for q in quotas.values())
        if signature == previous:
            continue
        previous = signature
        failures = []
        for behavior, quota in quotas.items():
            count, cap = quota["target_pairs"], quota["episode_cap"]
            if not count:
                failures.append(f"{behavior}: insufficient pairs for its target share")
            elif capped_capacity(profiles[behavior]["all"], cap) < count:
                failures.append(f"{behavior}: episode contribution limit prevents filling the quota")
            if c["behaviors"][behavior]["requires_evasion_side"]:
                if count // 2 < c["min_episodes_per_evasion_side"]:
                    failures.append(f"{behavior}: quota is too small for distinct episodes on each side")
                for side in ("left", "right"):
                    if capped_capacity(profiles[behavior][side], cap) < count // 2:
                        failures.append(f"{behavior}/{side}: episode contribution limit prevents filling the quota")
        selected, rng = [], random.Random(seed)
        if not failures:
            for behavior, quota in quotas.items():
                count, cap = quota["target_pairs"], quota["episode_cap"]
                group = by_behavior[behavior]
                if c["behaviors"][behavior]["requires_evasion_side"]:
                    chosen = joint_direction_pick(group, count, cap, c["min_episodes_per_evasion_side"], rng)
                else:
                    chosen = diverse_pick(group, count, cap, rng)
                quota["selected_pairs"] = len(chosen)
                if len(chosen) != count:
                    failures.append(f"{behavior}: shared episode capacity or side coverage prevents filling the quota")
                selected.extend(chosen)
        if failures:
            if first_failure is None:
                first_failure = {"deficits": failures, "quotas": quotas}
            if any(not q["target_pairs"] for q in quotas.values()):
                break
            continue
        rng.shuffle(selected)
        return selected, {"status": "ready", "support": support, "quotas": quotas,
                          "requested_proportions": weights,
                          "achieved_proportions": {b: q["selected_pairs"] / len(selected) for b, q in quotas.items()},
                          "sampling_budget": {"upper_bound": upper, "used": budget,
                                              "selected_pairs": len(selected), "reduced": budget < upper},
                          "rounding": "floor each behavior quota, then round directional quotas down to an even number",
                          "sampling": "without replacement"}
    return [], {"status": "insufficient_reviewed_support", "support": support,
                "sampling_budget": {"upper_bound": upper, "used": 0, "selected_pairs": 0},
                **(first_failure or {"deficits": ["No positive quota satisfies the requested proportions"]})}


def coverage(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['behavior']}/{row['evasion_side']}"].append(row)
    return {key: {"pairs": len(items), "episodes": len({r["episode_key"] for r in items}),
                  "scenarios": len({r["scenario"] for r in items})}
            for key, items in sorted(groups.items())}


def annotation_coverage(rows: list[dict], c: dict) -> dict:
    maneuvers = defaultdict(lambda: {"intervals": set(), "pairs": set(), "episodes": set()})
    for row in rows:
        for value in row.get("maneuvers", []):
            key = f"{value['maneuver']}/{value['review_status']}"
            group = maneuvers[key]
            group["intervals"].add((row["episode_key"], json.dumps(value, sort_keys=True)))
            group["pairs"].add((row["episode_key"], row["pair_index"]))
            group["episodes"].add(row["episode_key"])
    classified = [r for r in rows if r["behavior"] in c["behaviors"]]
    unknown = [r for r in classified if c["behaviors"][r["behavior"]]["requires_evasion_side"]
               and r["evasion_side"] not in ("left", "right")]
    return {"training_pairs": len(rows), "reviewed_classified_pairs": len(classified),
            "unassigned_pairs": len(rows) - len(classified), "unknown_direction_pairs": len(unknown),
            "balanced_eligible_pairs": len(classified) - len(unknown), "local_behaviors": coverage(rows),
            "global_maneuvers": {key: {name: len(values) for name, values in group.items()}
                                 for key, group in sorted(maneuvers.items())}}


def verify_partitions(s: dict, episodes: list[dict]) -> dict:
    by_key = {e["episode_key"]: e for e in episodes}
    if len(by_key) != len(episodes):
        raise ValueError("Catalog contains duplicate episode keys")
    source_owners = set()
    for e in episodes:
        indices = e["source_indices"]
        if (e["episode_key"] != f"{e['run_file']}#{e['episode_id']}"
                or e["num_pairs"] != len(indices) // 2
                or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError(f"Invalid catalog episode: {e['episode_key']}")
        for index in indices:
            identity = (e["run_file"], index)
            if identity in source_owners:
                raise ValueError(f"Observation belongs to multiple episodes: {identity}")
            source_owners.add(identity)
    if split_episodes(episodes, s["config"], s["split_seed"]) != s["episode_roles"]:
        raise ValueError("Catalog episode roles do not match its saved split seed")
    return {"status": "passed", "saved_split_reproduced": True,
            "episode_keys_unique": True, "source_observations_unique": True,
            "train_validation_episode_overlap": 0, "train_validation_content_overlap": 0}


def verify_manifest(path: Path, expected: list[dict], by_key: dict, assignment: dict, role: str) -> dict:
    pairs, observations, episodes, scenarios = set(), set(), set(), set()
    count = 0
    with path.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= len(expected) or row != expected[index]:
                raise ValueError(f"Written manifest differs from its selection: {path}")
            key, pair_index = row["episode_key"], row["pair_index"]
            e = by_key.get(key)
            if (e is None or assignment.get(key) != role or type(pair_index) is not int
                    or not 0 <= pair_index < e["num_pairs"]
                    or any(row[name] != e[name] for name in ("scenario", "run_file", "episode_id"))):
                raise ValueError(f"Manifest pair has an invalid episode or role: {path}:{index + 1}")
            source = e["source_indices"][2 * pair_index:2 * pair_index + 2]
            if row["source_indices"] != source or row["label_source_index"] != source[-1]:
                raise ValueError(f"Manifest pair is not a disjoint consecutive pair: {path}:{index + 1}")
            identity = (key, pair_index)
            if identity in pairs or any((e["run_file"], i) in observations for i in source):
                raise ValueError(f"Manifest repeats a pair or observation: {path}:{index + 1}")
            pairs.add(identity)
            observations.update((e["run_file"], i) for i in source)
            episodes.add(key)
            scenarios.add(e["scenario"])
            count += 1
    if count != len(expected):
        raise ValueError(f"Written manifest is incomplete: {path}")
    return {"status": "passed", "pairs": count, "observations": len(observations),
            "episodes": len(episodes), "scenarios": len(scenarios), "role": role,
            "duplicate_pairs": 0, "overlapping_observations": 0,
            "label_source": "second observation", "sha256": file_hash(path)}


def verify_selection(rows: list[dict], state: dict, c: dict) -> dict:
    if state.get("selection") == "all accepted training pairs":
        return {"status": "passed", "direction_balance_enforced": False}
    if len(rows) != sum(q["target_pairs"] for q in state["quotas"].values()):
        raise ValueError("Selection contains pairs outside its behavior quotas")
    for behavior, quota in state["quotas"].items():
        group = [r for r in rows if r["behavior"] == behavior]
        if len(group) != quota["target_pairs"] or len(group) != quota["selected_pairs"]:
            raise ValueError(f"Selected behavior quota changed: {behavior}")
        if max(Counter(r["episode_key"] for r in group).values(), default=0) > quota["episode_cap"]:
            raise ValueError(f"Selected episode exceeds its behavior cap: {behavior}")
        if c["behaviors"][behavior]["requires_evasion_side"]:
            sides = Counter(r["evasion_side"] for r in group)
            if sides != Counter({"left": len(group) // 2, "right": len(group) // 2}):
                raise ValueError(f"Selected evasion sides are not balanced: {behavior}")
            for side in ("left", "right"):
                if len({r["episode_key"] for r in group if r["evasion_side"] == side}) < c["min_episodes_per_evasion_side"]:
                    raise ValueError(f"Selected side has too few distinct episodes: {behavior}/{side}")
    return {"status": "passed", "direction_balance_enforced": True,
            "episode_caps": "passed", "minimum_episodes_per_side": "passed"}


def write_dataset_report(out: Path, result: dict) -> None:
    c = result["config"]
    lines = ["# Preparación de datasets ROS 2", "",
             f"Semilla de partición conservada: `{result['split_seed']}`.", "",
             f"Semilla de selección: `{result['selection_seed']}`.", "",
             f"Catálogo SHA256: `{result['catalog_sha256']}`.", "",
             "Los manifiestos contienen pares de dos observaciones consecutivas, sin solapamiento. "
             "La etiqueta procede de la segunda observación. Los NPZ y la LOM 0/100 se conservan.", "",
             "`ready` indica que el manifiesto pasó las comprobaciones de preparación. "
             "El entrenamiento y la navegación simulada se validan en fases posteriores.", "",
             "| Configuración | Estado | Pares seleccionados | Pares de validación | Fracción del entrenamiento disponible |",
             "|---|---|---:|---:|---:|"]
    for key, value in result["cells"].items():
        lines.append(f"| {key} | {value['status']} | {value['train_pairs']:,} | {value['validation_pairs']:,} "
                     f"| {value['selected_fraction_of_training']:.1%} |")
    lines.extend(["", "## Comprobaciones", "",
                  "La partición guardada coincide con su semilla. No hay episodios ni contenido compartido entre "
                  "entrenamiento y validación. Los manifiestos escritos se leen de nuevo y se comprueban sus "
                  "observaciones, etiquetas, duplicados y roles. Cada conjunto ambiental comparte un único "
                  "manifiesto de validación entre todas sus recetas.", "",
                  f"Límite por episodio y comportamiento: `ceil({c['max_episode_fraction_per_behavior']} × cuota)` pares. "
                  f"Las evasiones seleccionadas requieren al menos {c['min_episodes_per_evasion_side']} episodios "
                  "distintos por lado y el mismo número de pares izquierdos y derechos. "
                  "La selección no repite pares.", "",
                  "Las cuotas se redondean hacia abajo y las direccionales se ajustan a números pares. "
                  "El presupuesto se reduce cuando hace falta para cumplir los límites sin cambiar las proporciones solicitadas.", ""])
    for name, value in result["annotation_coverage"].items():
        lines.extend([f"## Cobertura revisada de {name}", "",
                      f"Pares de entrenamiento disponibles: {value['training_pairs']:,}. "
                      f"Con clase local revisada: {value['reviewed_classified_pairs']:,}. "
                      f"Sin clase local asignada: {value['unassigned_pairs']:,}. "
                      f"Con clase direccional y lado desconocido: {value['unknown_direction_pairs']:,}.", "",
                      "Las anotaciones sin clase, pendientes y con lado desconocido se conservan en D0. "
                      "No completan cuotas de evasión balanceada. Esta cobertura describe las anotaciones "
                      "disponibles, no demuestra ausencia de comportamientos en el resto del dataset.", "",
                      "| Comportamiento/lado | Pares | Episodios | Escenarios |", "|---|---:|---:|---:|"])
        for key, item in value["local_behaviors"].items():
            lines.append(f"| {key} | {item['pairs']:,} | {item['episodes']:,} | {item['scenarios']:,} |")
        if value["global_maneuvers"]:
            lines.extend(["", "Las maniobras globales se conservan como contexto. "
                          "No convierten sus pares en etiquetas locales B3/B4.", "",
                          "| Maniobra/estado | Intervalos | Pares cubiertos | Episodios |", "|---|---:|---:|---:|"])
            for key, item in value["global_maneuvers"].items():
                lines.append(f"| {key} | {item['intervals']} | {item['pairs']} | {item['episodes']} |")
        lines.append("")
    for key, value in result["cells"].items():
        lines.extend([f"## {key}", "", f"Estado: `{value['status']}`.", ""])
        if value.get("sampling_budget"):
            budget = value["sampling_budget"]
            lines.extend([f"Presupuesto nominal inicial: {budget['upper_bound']:,}. "
                          f"Presupuesto nominal utilizado: {budget['used']:,}. "
                          f"Pares seleccionados después del redondeo: {budget['selected_pairs']:,}.", ""])
        if value.get("reason"):
            lines.extend([value["reason"], ""])
        for reason in value.get("deficits", []):
            lines.extend([f"- {reason}", ""])
        if value["status"] == "ready" and value.get("quotas"):
            lines.extend(["| Comportamiento | Pares | Proporción solicitada | Proporción obtenida | Máximo por episodio |",
                          "|---|---:|---:|---:|---:|"])
            for behavior, quota in value["quotas"].items():
                lines.append(f"| {behavior} | {quota['selected_pairs']} "
                             f"| {value['requested_proportions'][behavior]:.1%} "
                             f"| {value['achieved_proportions'][behavior]:.1%} | {quota['episode_cap']} |")
            lines.append("")
    (out / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def build(directory: Path, root: Path | None, paths: list[Path], out: Path, seed: int) -> dict:
    s, episodes, root = read_catalog(directory, root)
    partition_check = verify_partitions(s, episodes)
    by_key = {e["episode_key"]: e for e in episodes}
    annotations = read_annotations(paths, episodes, s["episode_roles"], s["config"], file_hash(directory / "catalog.json"))
    result = {"schema_version": 1, "phase": "datasets", "selection_seed": seed,
              "split_seed": s["split_seed"], "provenance": provenance(), "config": s["config"],
              "catalog_sha256": file_hash(directory / "catalog.json"), "data_root": str(root),
              "annotations": [{"file": p.name, "sha256": file_hash(p)} for p in paths],
              "checks": {"partition": partition_check}, "annotation_coverage": {}, "cells": {}}
    for name, assignment in s["episode_roles"].items():
        env_dir = out / name
        env_dir.mkdir()
        validation = pairs_for_role(episodes, assignment, "validation", {})
        val_path = env_dir / "validation.jsonl"
        with val_path.open("w") as stream:
            for row in validation:
                stream.write(json.dumps(row) + "\n")
        validation_check = verify_manifest(val_path, validation, by_key, assignment, "validation")
        train = pairs_for_role(episodes, assignment, "train", annotations)
        if len(train) != sum(e["num_pairs"] for e in episodes if assignment.get(e["episode_key"]) == "train"):
            raise ValueError(f"Training pool is incomplete: {name}")
        result["annotation_coverage"][name] = annotation_coverage(train, s["config"])
        for dataset, recipe in s["config"]["recipes"].items():
            chosen, status = select_recipe(train, recipe, s["config"], seed)
            cell_dir = env_dir / dataset
            cell_dir.mkdir()
            row = {"environment_set": name, "dataset": dataset, "train_pairs": len(chosen),
                   "validation_pairs": len(validation), "validation_sha256": file_hash(val_path),
                   "source_train_pairs": len(train),
                   "selected_fraction_of_training": len(chosen) / len(train) if train else 0,
                   "checks": {"validation_manifest": validation_check}, "coverage": coverage(chosen), **status}
            if status["status"] == "ready":
                path = cell_dir / "train.jsonl"
                with path.open("w") as stream:
                    for item in chosen:
                        stream.write(json.dumps(item) + "\n")
                row["checks"]["selection"] = verify_selection(chosen, status, s["config"])
                row["checks"]["train_manifest"] = verify_manifest(path, chosen, by_key, assignment, "train")
                row["train_sha256"] = file_hash(path)
            write_json(cell_dir / "selection.json", row)
            result["cells"][f"{name}/{dataset}"] = row
        if {v["validation_sha256"] for v in result["cells"].values() if v["environment_set"] == name} != {validation_check["sha256"]}:
            raise ValueError(f"Validation manifest changed between recipes: {name}")
    result["checks"]["shared_validation_manifests"] = "passed"
    result["report"] = "dataset_report.md"
    write_dataset_report(out, result)
    write_json(out / "datasets.json", result)
    return result


def temporal_review_data(root: Path, episodes: list[dict], rows: list[dict], s: dict,
                         catalog_hash: str, maps_root: Path | None, map_variant: str,
                         annotation_paths: list[Path]) -> dict:
    """Prepare offline episode views without changing policy observations."""
    from PIL import Image
    by_key = {e["episode_key"]: e for e in episodes}
    selected_keys = list(dict.fromkeys(f"{r['run_file']}#{r['episode_id']}" for r in rows))
    by_run = defaultdict(list)
    for key in selected_keys:
        by_run[by_key[key]["run_file"]].append(key)
    output = {}
    masks = region_masks()
    for run_file, keys in by_run.items():
        samples = read_samples(root / run_file)
        for key in keys:
            e = by_key[key]
            frames = []
            for offset, index in enumerate(e["source_indices"]):
                sample = samples[index]
                grid = np.asarray(sample["features"]["occ_grid"], dtype=np.uint8)
                if grid.shape != (80, 80):
                    raise ValueError(f"Expected an 80x80 stored LOM: {key}, sample {index}")
                frames.append({"source_index": index, "episode_offset": offset,
                               "timestamp": float(sample["meta"]["timestamp"]),
                               "pose": np.asarray(sample["meta"]["pose"]).tolist(),
                               "target": np.asarray(sample["features"]["target"]).tolist(),
                               "action": np.asarray(sample["labels"]).tolist(),
                               "grid": base64.b64encode(grid.tobytes(order="C")).decode("ascii"),
                               "fractions": {name: float((grid[mask] > 0).mean()) for name, mask in masks.items()}})
            output[key] = {"key": key, "scenario": e["scenario"], "run_file": run_file,
                           "episode_id": e["episode_id"], "num_pairs": e["num_pairs"],
                           "goal": np.asarray(samples[e["source_indices"][0]]["meta"]["goal_xy"]).tolist(),
                           "frames": frames,
                           "segments": [r for r in rows if f"{r['run_file']}#{r['episode_id']}" == key]}
            print(f"temporal {e['scenario']} episode {e['episode_id']}: {e['num_pairs']} disjoint pairs", flush=True)
        del samples
    maps, sources = {}, {}
    resolved_maps = (maps_root or MAPS_ROOT).expanduser().resolve()
    if maps_root is not None or resolved_maps.is_dir():
        for scenario in sorted({e["scenario"] for e in output.values()}):
            background = load_review_map(resolved_maps, scenario, map_variant)
            buffer = io.BytesIO()
            Image.fromarray(background["pixels"]).save(buffer, format="PNG")
            maps[scenario] = {"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                              "resolution": background["resolution"], "origin": background["origin"].tolist(),
                              "height": background["pixels"].shape[0], "width": background["pixels"].shape[1]}
            sources[scenario] = background["source"]
    references = []
    for path in annotation_paths:
        text = path.read_text(encoding="utf-8-sig")
        references.append({"filename": path.name, "sha256": file_hash(path), "text": text,
                           "rows": list(csv.DictReader(io.StringIO(text)))})
    return {"version": 2, "catalog_sha256": catalog_hash, "split_seed": s["split_seed"],
            "behaviors": s["config"]["behaviors"], "maneuvers": MANEUVERS, "contexts": CONTEXTS,
            "phases": PHASES, "linear_deadband": s["config"]["linear_deadband_m_s"],
            "angular_deadband": s["config"]["angular_deadband_rad_s"], "fields": TEMPORAL_FIELDS,
            "episodes": [output[key] for key in selected_keys], "maps": maps,
            "map_sources": sources, "references": references}


def make_temporal_review(root: Path, out: Path, episodes: list[dict], rows: list[dict], s: dict,
                         catalog_hash: str, maps_root: Path | None, map_variant: str,
                         annotation_paths: list[Path], lom_view: str) -> dict:
    payload = temporal_review_data(root, episodes, rows, s, catalog_hash, maps_root, map_variant, annotation_paths)
    payload["lom_view"] = lom_view
    with (out / "preview_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for i, path in enumerate(annotation_paths, 1):
        (out / f"source_annotations_{i:02d}.csv").write_bytes(path.read_bytes())
    write_json(out / "review_maps.json", {"pose_frame": "map", "lom_view": lom_view,
               "lom_array_transform": "none", "map_variant": map_variant, "maps": payload["map_sources"]})
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (out / "review.html").write_text(TEMPORAL_HTML.replace("__TEMPORAL_DATA__", encoded), encoding="utf-8")
    return {"review_mode": "temporal", "episodes": len(payload["episodes"]),
            "pairs": sum(e["num_pairs"] for e in payload["episodes"]),
            "reference_files": [{"file": p.name, "sha256": file_hash(p)} for p in annotation_paths]}


def review_segments(directory: Path, root: Path | None, ids: list[str] | None, out: Path,
                    reuse_previews: Path | None = None, maps_root: Path | None = None,
                    map_variant: str = "prohibition", map_padding: float = 1.25,
                    lom_view: str = "robot", temporal: bool = False,
                    annotation_paths: list[Path] | None = None) -> dict:
    s, episodes, root = read_catalog(directory, root)
    with (directory / "review_template.csv").open(newline="") as stream:
        available = {r["segment_id"]: r for r in csv.DictReader(stream)}
    reused = None
    if reuse_previews is not None:
        if ids:
            raise ValueError("Choose segment IDs or an existing preview selection")
        source = reuse_previews.expanduser().resolve()
        source = source / "preview_rows.csv" if source.is_dir() else source
        with source.open(newline="") as stream:
            ids = [r["segment_id"] for r in csv.DictReader(stream)]
        reused = {"file": str(source), "sha256": file_hash(source)}
        if source == directory / "preview_rows.csv" and s.get("preview_rows_sha256") != reused["sha256"]:
            raise ValueError("The original preview selection changed")
    requested = list(dict.fromkeys(ids or []))
    if not requested:
        raise ValueError("No segments were selected for review")
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown or non-training segment IDs: {missing}")
    rows = [available[key] for key in requested]
    details = {}
    if temporal:
        details = make_temporal_review(root, out, episodes, rows, s, file_hash(directory / "catalog.json"),
                                       maps_root, map_variant, annotation_paths or [], lom_view)
    else:
        if annotation_paths:
            raise ValueError("Annotation import requires --temporal")
        make_review(root, out, episodes, rows, s["split_seed"], len(rows), s["config"], minimum_pairs=1,
                    maps_root=maps_root, map_variant=map_variant, map_padding=map_padding, preserve_order=True,
                    lom_view=lom_view)
    result = {"schema_version": 1, "phase": "review", "segment_ids": requested,
              "catalog_sha256": file_hash(directory / "catalog.json"),
              "split_seed": s["split_seed"], "provenance": provenance(),
              "preview_rows_sha256": file_hash(out / "preview_rows.csv"),
              "review_maps_sha256": file_hash(out / "review_maps.json"), "reused_previews": reused,
              "review_html_sha256": file_hash(out / "review.html"), **details}
    write_json(out / "review.json", result)
    return result


TEMPORAL_HTML = r'''<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión temporal de navegación</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;margin:0;background:#f3f5f8;color:#182536}header,main{max-width:1660px;margin:auto;padding:16px 22px}h1{font-size:24px;margin:0 0 6px}h2{font-size:17px;margin:0 0 10px}h3{font-size:15px;margin:0 0 8px}p{line-height:1.45;margin:6px 0}button,select,input,textarea{font:inherit}button,select,input[type=number]{padding:7px 9px;border:1px solid #b6c2d0;border-radius:5px;background:white;color:inherit}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button.primary{background:#174e89;color:white;border-color:#174e89}button.danger{color:#aa2638}label{display:block;font-size:13px;margin:7px 0}label select,label input:not([type=checkbox]),label textarea{display:block;width:100%;margin-top:4px}textarea{resize:vertical;min-height:68px;padding:7px;border:1px solid #b6c2d0;border-radius:5px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.row>label{flex:1;min-width:100px}.panel{background:white;padding:13px;border:1px solid #dce2e9;border-radius:8px;margin-bottom:12px}.review-grid{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px}.visuals{display:grid;grid-template-columns:1.1fr 1fr;gap:12px}.loms{display:grid;grid-template-columns:1fr 1fr;gap:8px;align-content:start}.lom{min-width:0}canvas{display:block;max-width:100%;height:auto;background:#e0e4e9}canvas.lom-canvas{image-rendering:pixelated;cursor:zoom-in}#map{width:100%}#timeline{width:100%;height:40px;cursor:pointer}.small{font-size:12px;color:#4c5e73}.data{font:12px ui-monospace,monospace;white-space:pre-wrap;line-height:1.5}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}.controls label{margin-top:0}input[type=range]{width:100%;accent-color:#174e89}.tags{display:grid;grid-template-columns:1fr 1fr;gap:2px}.tags label{margin:3px 0}fieldset{border:1px solid #dce2e9;margin:8px 0;padding:8px}legend{font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:7px;border-bottom:1px solid #e2e7ed;vertical-align:top}th{background:#eaf0f6;position:sticky;top:0}.table-scroll{max-height:330px;overflow:auto}#message{min-height:25px;padding:6px 0;color:#165f45}#message.error{color:#a32436}#autosave{font-size:12px;color:#4c5e73}#references{white-space:pre-wrap;max-height:260px;overflow:auto;font:12px ui-monospace,monospace}.scope-help{background:#edf4fc;padding:8px;font-size:12px;line-height:1.45}.badge{display:inline-block;padding:2px 7px;border-radius:10px;background:#eaf0f6;font-size:12px}.active-row{background:#fff1df}#pair-info{font-weight:600;margin:6px 0}.table-scroll button{padding:3px 6px;font-size:12px}#global-summary{font-size:13px}.hint{padding:8px;background:#fff6df;font-size:13px}.form-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}@media(max-width:1120px){.review-grid{grid-template-columns:1fr}.visuals{grid-template-columns:1fr 1fr}.editor{display:grid;grid-template-columns:1fr 1fr;gap:12px}.editor h2,.editor .form-actions,.editor .scope-help{grid-column:1/-1}}@media(max-width:720px){header,main{padding:12px}.visuals,.controls,.editor{display:block}.loms{margin-top:12px}.review-grid{display:block}}
</style>
<header><h1>Revisión temporal de navegación</h1>
<p>Episodio completo para interpretar la maniobra. Pares consecutivos de dos observaciones para delimitar el comportamiento local.</p>
<div class="row"><button id="import">Importar CSV o restaurar sesión JSON</button><input id="file" type="file" accept=".csv,.json" hidden>
<button id="export-csv" class="primary">Exportar intervalos CSV</button><button id="export-session">Guardar sesión JSON</button><span id="autosave"></span></div>
<div id="message" role="status"></div><div id="provenance" class="small"></div></header>
<main><section class="panel">
<div class="controls"><label>Episodio <select id="episode"></select></label><label>Tramo de referencia <select id="segment"></select></label></div>
<div class="row"><button id="prev">← Par anterior</button><button id="play">Reproducir</button><button id="next">Par siguiente →</button>
<label>Velocidad <select id="speed"><option value="1000">1 par/s</option><option value="500" selected>2 pares/s</option><option value="200">5 pares/s</option></select></label>
<label>Mapa <select id="map-mode"><option value="episode">Episodio completo</option><option value="pair">Área del par (6 m)</option></select></label>
<label>Vista LOM <select id="lom-view"><option value="robot">Robot: frente arriba, izquierda a la izquierda</option><option value="array">Array almacenado en NPZ</option></select></label></div>
<input id="pair" type="range" min="0" value="0" step="1"><div id="pair-info"></div>
<canvas id="timeline" width="1100" height="40" aria-label="Intervalos locales y par activo"></canvas>
<div class="small">Pares numerados desde 0. La selección incluye el primer y el último par indicados. Un frame final sin pareja se muestra solo como contexto.</div>
</section>
<div class="review-grid"><div><section class="panel"><div class="visuals"><div><h2>Trayectoria y obstáculos</h2><canvas id="map" width="600" height="600"></canvas>
<p class="small">Gris: episodio. Naranja: intervalo del formulario. Azul: par activo. Flechas moradas: orientación del robot. Estrella roja: objetivo. El mapa solo sirve para etiquetar.</p></div>
<div><h2>Observaciones del par activo</h2><div class="loms"><div class="lom"><h3 id="frame-title-0"></h3><canvas id="lom-0" class="lom-canvas" width="320" height="320"></canvas><div id="frame-meta-0" class="data"></div></div>
<div class="lom"><h3 id="frame-title-1"></h3><canvas id="lom-1" class="lom-canvas" width="320" height="320"></canvas><div id="frame-meta-1" class="data"></div></div></div>
<p class="small">Valores y ruido originales. Las acciones se expresan en m/s y rad/s. La segunda observación aporta la acción objetivo del par. Pulsa una LOM para ampliarla.</p></div></div></section>
<section class="panel"><h2>Intervalos de este episodio</h2><div class="table-scroll"><table><thead><tr><th>Ámbito</th><th>Pares incluidos</th><th>Etiqueta y contexto</th><th>Lado / estado</th><th>Acciones</th></tr></thead><tbody id="intervals"></tbody></table></div></section>
<section class="panel"><h2>Cobertura local confirmada</h2><label>Filtrar cobertura por contexto <select id="coverage-context"><option value="">Todos los contextos</option></select></label><div id="global-summary"></div><div id="balance"></div><p class="small">Se cuentan pares y episodios distintos. Las maniobras globales, las etiquetas pendientes y las ambiguas no se convierten en clases locales. El lado indeterminado permanece en D0 y no completa cuotas izquierda/derecha de D1–D3.</p></section>
<details class="panel"><summary>Referencias originales importadas</summary><p class="small">Estos registros se conservan separados de las anotaciones de trabajo. Un CSV antiguo se abre como borrador pendiente de revisión.</p><div id="references"></div></details>
</div>
<section class="panel editor"><h2 id="editor-title">Nuevo intervalo</h2><div>
<label>Ámbito <select id="scope"><option value="local">Comportamiento local</option><option value="maneuver">Maniobra completa</option></select></label>
<div class="row"><label>Primer par <input id="start" type="number" min="0" value="0"></label><label>Último par incluido <input id="end" type="number" min="0" value="0"></label></div>
<div class="row"><button id="mark-start">Inicio = par actual</button><button id="mark-end">Fin = par actual</button><button id="whole">Todo el episodio</button></div>
<label>Comportamiento local <select id="behavior"></select></label>
<label>Maniobra <select id="maneuver"></select></label>
<label>Fase <select id="phase"></select></label></div><div>
<fieldset><legend>Contexto (pueden coexistir)</legend><div id="contexts" class="tags"></div></fieldset>
<label>Lado de la evasión <select id="side"><option value="unknown">Indeterminado</option><option value="left">Izquierda</option><option value="right">Derecha</option><option value="not_applicable">No corresponde</option><option value="neutral">Neutral del registro anterior</option></select></label>
<label>Estado <select id="status"><option value="pending_review">Pendiente de revisión</option><option value="reviewed">Revisado</option><option value="ambiguous">Ambiguo</option></select></label>
<label>Observaciones <textarea id="notes" placeholder="Describe lo que sustenta la etiqueta o lo que falta por resolver."></textarea></label></div>
<div class="scope-help" id="scope-help"></div>
<div class="scope-help">El lado describe la evasión de la maniobra. El lado del obstáculo y el signo instantáneo de vy se muestran por separado. Para B4, indica el lado de la evasión anterior. Una realineación tras una puerta puede tener otra explicación.</div>
<div class="form-actions"><button id="save" class="primary">Guardar intervalo</button><button id="new">Nuevo intervalo</button><button id="split">Dividir antes del par actual</button></div>
<p class="small">Los intervalos locales no se solapan. Una maniobra completa puede abarcar varios intervalos locales. Guardar una etiqueta global no la extiende como comportamiento local a todos sus pares.</p>
</section></div></main>
<script>
const DATA=__TEMPORAL_DATA__;
function makeTemporalCore(data){
 const episodes=new Map(data.episodes.map(e=>[e.key,e]));
 const state={annotations:[],references:[]};let serial=0;
 function clone(v){return JSON.parse(JSON.stringify(v));}
 function id(){return 'interval_'+Date.now().toString(36)+'_'+(++serial).toString(36);}
 function validate(a){
  const e=episodes.get(a.key);if(!e)throw Error('El episodio no está en esta revisión: '+a.key);
  if(!Number.isInteger(a.start)||!Number.isInteger(a.stop)||a.start<0||a.start>=a.stop||a.stop>e.num_pairs)throw Error('El intervalo debe respetar los límites de pares del episodio.');
  if(!['local','maneuver'].includes(a.scope)||!['pending_review','reviewed','ambiguous'].includes(a.status))throw Error('Ámbito o estado inválido.');
  if(a.behavior&&!data.behaviors[a.behavior])throw Error('Comportamiento desconocido.');
  if(!['left','right','unknown','not_applicable','neutral'].includes(a.side))throw Error('Lado de evasión inválido.');
  if(!Array.isArray(a.context)||a.context.some(x=>!data.contexts[x]))throw Error('Contexto desconocido.');
  if(a.maneuver&&!data.maneuvers[a.maneuver])throw Error('Maniobra desconocida.');
  if(a.phase&&!data.phases[a.phase])throw Error('Fase desconocida.');
  if(a.scope==='maneuver'&&a.behavior)throw Error('Una maniobra completa no asigna una clase local.');
  if(a.status==='reviewed'&&a.scope==='local'&&a.behavior){
   const directional=data.behaviors[a.behavior].requires_evasion_side;
   if(directional&&!['left','right','unknown'].includes(a.side))throw Error('Para '+a.behavior+' elige izquierda, derecha o indeterminado.');
   if(!directional&&!['neutral','not_applicable'].includes(a.side))throw Error('Para '+a.behavior+' el lado debe ser «No corresponde».');
  }
  if(a.status==='reviewed'&&a.scope==='local'&&!a.behavior&&!a.context.length&&!a.phase)throw Error('Añade un comportamiento, contexto o fase al intervalo revisado.');
  if(a.status==='reviewed'&&a.scope==='maneuver'&&!a.maneuver)throw Error('Elige la maniobra completa.');
  return a;
 }
 function overlaps(a,items=state.annotations){return items.some(b=>b.id!==a.id&&a.scope==='local'&&b.scope==='local'&&a.key===b.key&&a.start<b.stop&&b.start<a.stop);}
 function put(a){a=clone(a);if(!a.id)a.id=id();validate(a);if(overlaps(a))throw Error('Se solapa con otro intervalo local. Edita o divide el intervalo existente.');const index=state.annotations.findIndex(b=>b.id===a.id);if(index<0)state.annotations.push(a);else state.annotations[index]=a;return a;}
 function remove(key){state.annotations=state.annotations.filter(a=>a.id!==key);}
 function split(key,pair){const a=state.annotations.find(a=>a.id===key);if(!a||!Number.isInteger(pair)||pair<=a.start||pair>=a.stop)throw Error('El par de división debe estar dentro del intervalo seleccionado.');const left={...clone(a),stop:pair,status:'pending_review'},right={...clone(a),id:id(),start:pair,status:'pending_review'};validate(left);validate(right);state.annotations=state.annotations.filter(b=>b.id!==key).concat([left,right]);return [left,right];}
 function parseCSV(text){
  text=text.replace(/^\uFEFF/,'');const records=[];let row=[],field='',quoted=false,closed=false;
  for(let i=0;i<text.length;i++){
   const ch=text[i];if(quoted){if(ch==='"'){if(text[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}}else field+=ch;continue;}
   if(ch==='"'){if(field||closed)throw Error('Comillas inválidas en el CSV.');quoted=true;}
   else if(ch===','){row.push(field);field='';closed=false;}
   else if(ch==='\n'||ch==='\r'){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(x=>x!==''))records.push(row);row=[];field='';closed=false;}
   else{if(closed)throw Error('Texto después de un campo CSV entrecomillado.');field+=ch;}
  }
  if(quoted)throw Error('El CSV tiene un campo sin cerrar.');if(field||row.length||closed){row.push(field);if(row.some(x=>x!==''))records.push(row);}
  const fields=records.shift()||[];if(new Set(fields).size!==fields.length)throw Error('Cabeceras CSV repetidas.');
  for(const name of ['run_file','episode_id','pair_start','pair_stop'])if(!fields.includes(name))throw Error('Falta la columna '+name+'.');
  return records.map(values=>{if(values.length!==fields.length)throw Error('Fila CSV con número de columnas distinto.');return Object.fromEntries(fields.map((name,i)=>[name,values[i]]));});
 }
 function importRows(rows,filename,text=''){
  if(state.references.some(r=>r.filename===filename&&r.text===text))return {imported:0,skipped:0,duplicate:true};
  const working=state.annotations.map(clone);let imported=0,skipped=0;
  for(const row of rows){
   const legacy=!row.annotation_schema||row.annotation_schema==='1';if(!legacy&&row.annotation_schema!=='2')throw Error('Versión CSV no compatible.');
   if(!legacy&&row.catalog_sha256!==data.catalog_sha256)throw Error('El CSV pertenece a otro catálogo.');
   const key=row.run_file+'#'+Number(row.episode_id);const e=episodes.get(key);if(!e){skipped++;continue;}
   if(row.scenario&&row.scenario!==e.scenario)throw Error('Escenario incompatible con el episodio.');
   const a={id:id(),key,start:Number(row.pair_start),stop:Number(row.pair_stop),scope:legacy?'local':row.annotation_scope,
    behavior:row.behavior||'',side:row.evasion_side||'unknown',status:legacy?'pending_review':row.review_status,
    maneuver:row.maneuver||'',phase:row.phase||'',context:(row.context||'').split(';').filter(Boolean),notes:row.notes||'',source:row.source_segment_id||row.segment_id||''};
   validate(a);if(overlaps(a,working)){skipped++;continue;}working.push(a);imported++;
  }
  state.annotations=working;state.references.push({filename,text,rows:clone(rows)});return {imported,skipped,duplicate:false};
 }
 function session(){return {version:2,catalog_sha256:data.catalog_sha256,split_seed:data.split_seed,episode_keys:[...episodes.keys()],annotations:clone(state.annotations),references:clone(state.references)};}
 function restore(saved){
  if(saved.version!==2||saved.catalog_sha256!==data.catalog_sha256||saved.split_seed!==data.split_seed)throw Error('La sesión pertenece a otro catálogo o partición.');
  if(!Array.isArray(saved.episode_keys)||[...episodes.keys()].sort().join('|')!==[...saved.episode_keys].sort().join('|'))throw Error('La sesión contiene una selección de episodios distinta.');
  if(!Array.isArray(saved.annotations)||!Array.isArray(saved.references))throw Error('Sesión incompleta.');
  const working=[];const ids=new Set();for(const original of saved.annotations){const a=clone(original);validate(a);if(!a.id||ids.has(a.id)||overlaps(a,working))throw Error('La sesión tiene identificadores repetidos o intervalos solapados.');ids.add(a.id);working.push(a);}
  state.annotations=working;state.references=clone(saved.references);
 }
 function signature(frames){const y=frames.map(f=>f.action[1]),d=data.linear_deadband;if(y.every(v=>v>d))return 'positive_lateral_command';if(y.every(v=>v<-d))return 'negative_lateral_command';if(y.some(v=>Math.abs(v)>d))return 'lateral_transition';if(frames.some(f=>Math.abs(f.action[2])>data.angular_deadband))return 'rotation_without_lateral_command';if(frames.some(f=>Math.abs(f.action[0])>d))return 'longitudinal_command';return 'below_deadbands';}
 function exportRows(){return state.annotations.map(a=>{
  const e=episodes.get(a.key),frames=e.frames.slice(a.start*2,a.stop*2),mean=fn=>frames.reduce((sum,f)=>sum+fn(f),0)/frames.length;
  const signatures=new Set(Array.from({length:a.stop-a.start},(_,i)=>signature(frames.slice(i*2,i*2+2))));
  return {segment_id:a.id,scenario:e.scenario,run_file:e.run_file,episode_id:e.episode_id,pair_start:a.start,pair_stop:a.stop,num_pairs:a.stop-a.start,
   motion_signature:signatures.size===1?[...signatures][0]:'mixed_motion_signatures',vx_mean:mean(f=>f.action[0]),vy_mean:mean(f=>f.action[1]),wz_mean:mean(f=>f.action[2]),theta_mean:mean(f=>f.target[1]),
   front_occupied_fraction:mean(f=>f.fractions.front),left_occupied_fraction:mean(f=>f.fractions.left),right_occupied_fraction:mean(f=>f.fractions.right),
   behavior:a.behavior,evasion_side:a.side,reviewed:a.status==='reviewed'?'1':'0',notes:a.notes,annotation_schema:'2',annotation_scope:a.scope,
   maneuver:a.maneuver,context:[...new Set(a.context)].sort().join(';'),phase:a.phase,review_status:a.status,catalog_sha256:data.catalog_sha256,source_segment_id:a.source};
 });}
 function csv(){const rows=exportRows();return [data.fields,...rows.map(r=>data.fields.map(f=>r[f]??''))].map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n')+'\r\n';}
 function coverage(context=''){
  const groups={};for(const a of state.annotations){if(a.scope!=='local'||a.status!=='reviewed'||!a.behavior||(context&&!a.context.includes(context)))continue;const side=['neutral','not_applicable'].includes(a.side)?'not_applicable':a.side;const key=a.behavior+'/'+side;const g=groups[key]||(groups[key]={behavior:a.behavior,side,pairs:0,episodes:new Set()});g.pairs+=a.stop-a.start;g.episodes.add(a.key);}
  return Object.values(groups).map(g=>({...g,episodes:g.episodes.size})).sort((a,b)=>(a.behavior+'/'+a.side).localeCompare(b.behavior+'/'+b.side));
 }
 return {state,episodes,validate,put,remove,split,parseCSV,importRows,session,restore,exportRows,csv,coverage};
}
globalThis.makeTemporalCore=makeTemporalCore;
if(typeof document!=='undefined'){
 const C=makeTemporalCore(DATA),$=id=>document.getElementById(id);let episode=DATA.episodes[0],pair=0,editing=null,timer=null;
 const gridCache=new Map(),mapImages=new Map(),colors={B0:'#6b8198',B1:'#167d73',B2:'#cc7200',B3:'#7950ac',B4:'#376cb5'};
 const storageKey='mapless-temporal-v2:'+DATA.catalog_sha256+':'+DATA.episodes.map(e=>e.key).sort().join('|');
 const statusNames={pending_review:'Pendiente',reviewed:'Revisado',ambiguous:'Ambiguo'},sideNames={left:'Izquierda',right:'Derecha',unknown:'Indeterminado',not_applicable:'No corresponde',neutral:'Neutral anterior'};
 function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':'';}
 function option(select,value,text){const item=document.createElement('option');item.value=value;item.textContent=text;select.append(item);}
 function persist(){try{localStorage.setItem(storageKey,JSON.stringify(C.session()));$('autosave').textContent='Sesión guardada en este navegador';}catch(error){$('autosave').textContent='Autoguardado no disponible. Guarda la sesión JSON.';}}
 function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 function act(fn){try{fn();}catch(error){message(error.message,true);}}
 function values(){const number=id=>$(id).value.trim()===''?NaN:Number($(id).value);return {id:editing,key:episode.key,start:number('start'),stop:number('end')+1,scope:$('scope').value,
  behavior:$('scope').value==='local'?$('behavior').value:'',side:$('side').value,status:$('status').value,maneuver:$('maneuver').value,
  phase:$('phase').value,context:[...document.querySelectorAll('#contexts input:checked')].map(el=>el.value),notes:$('notes').value,
  source:editing?(C.state.annotations.find(a=>a.id===editing)?.source||''):$('segment').value};}
 function scopeUI(){const global=$('scope').value==='maneuver';$('behavior').disabled=global;$('scope-help').textContent=global?'La maniobra es una anotación global del intervalo. Sus pares pueden contener varios comportamientos locales.':'El comportamiento local se aplica solo a los pares de este intervalo. Puedes dejarlo sin asignar y registrar contexto o incertidumbre.';}
 function form(a=null){editing=a?.id||null;$('editor-title').textContent=editing?'Editar intervalo':'Nuevo intervalo';$('scope').value=a?.scope||'local';$('start').value=a?.start??Math.min(pair,episode.num_pairs-1);$('end').value=(a?.stop??Math.min(pair+1,episode.num_pairs))-1;
  $('behavior').value=a?.behavior||'';$('maneuver').value=a?.maneuver||'';$('phase').value=a?.phase||'';$('side').value=a?.side||'unknown';$('status').value=a?.status||'pending_review';$('notes').value=a?.notes||'';
  for(const el of document.querySelectorAll('#contexts input'))el.checked=Boolean(a?.context.includes(el.value));$('split').disabled=!editing;scopeUI();drawMap();}
 function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='Reproducir';}
 function go(value){pair=Math.max(0,Math.min(Number(value),Number($('pair').max)));$('pair').value=pair;renderFrames();drawMap();drawTimeline();}
 function chooseEpisode(key){stop();episode=C.episodes.get(key);$('episode').value=key;pair=Number(episode.segments[0].pair_start);$('segment').replaceChildren();for(const row of episode.segments)option($('segment'),row.segment_id,row.segment_id+' · pares '+row.pair_start+'–'+(Number(row.pair_stop)-1));
  $('pair').max=episode.num_pairs-1+(episode.frames.length%2);$('start').max=episode.num_pairs-1;$('end').max=episode.num_pairs-1;form();go(pair);renderList();renderReferences();}
 function decoded(frame){const key=episode.key+':'+frame.source_index;if(!gridCache.has(key)){const bytes=Uint8Array.from(atob(frame.grid),c=>c.charCodeAt(0));if(bytes.length!==6400)throw Error('LOM incompleta.');gridCache.set(key,bytes);}return gridCache.get(key);}
 function lomPixel(column,row,view){return [view==='robot'?79-column:column,row];}
 function drawLom(index,frame){const canvas=$('lom-'+index),ctx=canvas.getContext('2d');ctx.clearRect(0,0,320,320);if(!frame){ctx.fillStyle='#e0e4e9';ctx.fillRect(0,0,320,320);$('frame-title-'+index).textContent='Sin segunda observación';$('frame-meta-'+index).textContent='Este frame final no genera un par de entrenamiento.';return;}
  const pixels=decoded(frame),small=document.createElement('canvas');small.width=80;small.height=80;const sc=small.getContext('2d'),image=sc.createImageData(80,80),view=$('lom-view').value;
  for(let row=0;row<80;row++)for(let col=0;col<80;col++){const [x,y]=lomPixel(col,row,view),at=(y*80+x)*4,shade=Math.round(255*(1-pixels[row*80+col]/100));image.data[at]=shade;image.data[at+1]=shade;image.data[at+2]=shade;image.data[at+3]=255;}
  sc.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(small,0,0,320,320);if(view==='robot'){ctx.fillStyle='#64359c';ctx.beginPath();ctx.moveTo(160,308);ctx.lineTo(153,320);ctx.lineTo(167,320);ctx.fill();}
  $('frame-title-'+index).textContent=(index+1)+'. Muestra '+frame.source_index;$('frame-meta-'+index).textContent='t = '+(frame.timestamp-episode.frames[0].timestamp).toFixed(3)+' s\nd = '+frame.target[0].toFixed(3)+' m; θ = '+frame.target[1].toFixed(3)+' rad\nvx = '+frame.action[0].toFixed(4)+'\nvy = '+frame.action[1].toFixed(4)+'\nwz = '+frame.action[2].toFixed(4);
 }
 function renderFrames(){const a=episode.frames[2*pair],b=episode.frames[2*pair+1];drawLom(0,a);drawLom(1,b);$('pair-info').textContent=pair<episode.num_pairs?'Par '+pair+' de '+(episode.num_pairs-1)+' · observaciones del episodio '+(2*pair)+' y '+(2*pair+1)+' · acción objetivo: muestra '+b.source_index:'Última observación del episodio, sin pareja. Solo contexto.';$('prev').disabled=pair===0;$('next').disabled=pair===Number($('pair').max);$('mark-start').disabled=pair>=episode.num_pairs;$('mark-end').disabled=pair>=episode.num_pairs;}
 function drawMap(){
  const canvas=$('map'),ctx=canvas.getContext('2d'),size=600,frame=episode.frames[Math.min(2*pair,episode.frames.length-1)],positions=episode.frames.map(f=>f.pose),points=positions.map(p=>p.slice(0,2)).concat([episode.goal]);
  let cx,cy,span;if($('map-mode').value==='pair'){cx=frame.pose[0];cy=frame.pose[1];span=6;}else{const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),loX=Math.min(...xs)-.9,hiX=Math.max(...xs)+.9,loY=Math.min(...ys)-.9,hiY=Math.max(...ys)+.9;cx=(loX+hiX)/2;cy=(loY+hiY)/2;span=Math.max(hiX-loX,hiY-loY,3);}
  const xmin=cx-span/2,ymax=cy+span/2,scale=size/span,xy=p=>[(p[0]-xmin)*scale,(ymax-p[1])*scale];ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#d8dde4';ctx.fillRect(0,0,size,size);
  const bg=DATA.maps[episode.scenario],img=mapImages.get(episode.scenario);if(bg&&img?.complete&&img.naturalWidth){const [ox,oy,yaw]=bg.origin,r=bg.resolution,c=Math.cos(yaw),s=Math.sin(yaw);ctx.save();ctx.setTransform(scale*r*c,-scale*r*s,scale*r*s,scale*r*c,scale*(ox-r*s*bg.height-xmin),scale*(ymax-oy-r*c*bg.height));ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0);ctx.restore();}
  function path(items,color,width){if(!items.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();items.forEach((p,i)=>{const q=xy(p);if(i)ctx.lineTo(...q);else ctx.moveTo(...q);});ctx.stroke();}
  path(positions,'#8896a6',2);let start=Number($('start').value),stop=Number($('end').value)+1;if(Number.isInteger(start)&&Number.isInteger(stop))path(positions.slice(Math.max(0,start*2),Math.min(stop*2,positions.length)),'#df7600',5);
  const active=positions.slice(pair*2,pair*2+2);path(active,'#086ab9',5);const first=xy(positions[0]);ctx.fillStyle='#07805b';ctx.beginPath();ctx.arc(...first,6,0,2*Math.PI);ctx.fill();const goal=xy(episode.goal);ctx.fillStyle='#bd2146';ctx.font='bold 25px system-ui';ctx.fillText('★',goal[0]-12,goal[1]+9);
  active.forEach((p,i)=>{const q=xy(p),end=xy([p[0]+.38*Math.cos(p[2]),p[1]+.38*Math.sin(p[2])]),angle=Math.atan2(end[1]-q[1],end[0]-q[0]);ctx.strokeStyle='#64359c';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...q);ctx.lineTo(...end);ctx.moveTo(end[0]-8*Math.cos(angle-.45),end[1]-8*Math.sin(angle-.45));ctx.lineTo(...end);ctx.lineTo(end[0]-8*Math.cos(angle+.45),end[1]-8*Math.sin(angle+.45));ctx.stroke();ctx.fillStyle='#64359c';ctx.font='bold 14px system-ui';ctx.fillText(String(i+1),q[0]+6,q[1]-7);});
  ctx.fillStyle='#1c3046';ctx.font='12px system-ui';ctx.fillText('map x, y (m) · '+span.toFixed(1)+' m de ancho',10,size-12);if(!bg)ctx.fillText('Mapa no disponible',10,20);
 }
 function drawTimeline(){const canvas=$('timeline'),ctx=canvas.getContext('2d'),w=canvas.width,n=episode.num_pairs;ctx.clearRect(0,0,w,40);ctx.fillStyle='#e3e9f0';ctx.fillRect(0,5,w,28);for(const a of C.state.annotations){if(a.key!==episode.key||a.scope!=='local')continue;ctx.fillStyle=a.status==='reviewed'?(colors[a.behavior]||'#788b9d'):a.status==='ambiguous'?'#dca55f':'#bbc5d1';ctx.fillRect(a.start/n*w,7,(a.stop-a.start)/n*w,24);}ctx.strokeStyle='#152b48';ctx.lineWidth=2;ctx.strokeRect(Math.min(pair,n-1)/n*w,3,w/n,32);}
 function renderList(){const body=$('intervals');body.replaceChildren();for(const a of C.state.annotations.filter(a=>a.key===episode.key).sort((a,b)=>a.start-b.start||a.scope.localeCompare(b.scope))){const tr=document.createElement('tr');if(a.id===editing)tr.className='active-row';for(const text of [a.scope==='local'?'Local':'Maniobra',a.start+'–'+(a.stop-1)+' ('+(a.stop-a.start)+')',(a.behavior||DATA.maneuvers[a.maneuver]||'Sin clase local')+'\n'+a.context.map(x=>DATA.contexts[x]).join(', ')+(a.phase?' · '+DATA.phases[a.phase]:''),sideNames[a.side]+'\n'+statusNames[a.status]]){const td=document.createElement('td');td.textContent=text;tr.append(td);}const td=document.createElement('td'),edit=document.createElement('button'),del=document.createElement('button');edit.textContent='Editar';edit.onclick=()=>{stop();go(a.start);form(a);renderList();};del.textContent='Eliminar';del.className='danger';del.onclick=()=>{C.remove(a.id);if(editing===a.id)form();changed();};td.append(edit,document.createTextNode(' '),del);tr.append(td);body.append(tr);}if(!body.children.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=5;td.textContent='Todavía no hay intervalos de trabajo en este episodio.';tr.append(td);body.append(tr);}}
 function renderCoverage(){const groups=C.coverage($('coverage-context').value),container=$('balance');container.replaceChildren();const table=document.createElement('table'),head=document.createElement('tr');for(const text of ['Comportamiento','Lado','Pares','Episodios']){const th=document.createElement('th');th.textContent=text;head.append(th);}table.append(head);for(const g of groups){const tr=document.createElement('tr');for(const text of [g.behavior,sideNames[g.side],g.pairs,g.episodes]){const td=document.createElement('td');td.textContent=text;tr.append(td);}table.append(tr);}container.append(table);const confirmed=C.state.annotations.filter(a=>a.status==='reviewed').length,pending=C.state.annotations.filter(a=>a.status==='pending_review').length,ambiguous=C.state.annotations.filter(a=>a.status==='ambiguous').length;$('global-summary').textContent='Intervalos: '+confirmed+' revisados · '+pending+' pendientes · '+ambiguous+' ambiguos. Episodios disponibles: '+DATA.episodes.length+'.';}
 function renderReferences(){const refs=[];for(const source of C.state.references){const rows=source.rows.filter(r=>r.run_file+'#'+Number(r.episode_id)===episode.key);for(const row of rows)refs.push(source.filename+'\n'+JSON.stringify(row,null,2));}$('references').textContent=refs.join('\n\n')||'Sin referencias importadas para este episodio.';}
 function changed(){persist();renderList();renderCoverage();renderReferences();drawTimeline();drawMap();}
 for(const e of DATA.episodes)option($('episode'),e.key,e.scenario+' · episodio '+e.episode_id+' · '+e.num_pairs+' pares');
 const behaviorNames={B0:'Avance libre',B1:'Evasión lateral simple',B2:'Evasión restringida',B3:'Rodeo frontal',B4:'Recuperación'};
 option($('behavior'),'','Sin asignar');for(const [key,value] of Object.entries(DATA.behaviors))option($('behavior'),key,key+' · '+(behaviorNames[key]||value.name.replaceAll('_',' ')));
 for(const [target,vocabulary] of [['maneuver',DATA.maneuvers],['phase',DATA.phases]]){option($(target),'','Sin asignar');for(const [key,text] of Object.entries(vocabulary))option($(target),key,text);}
 for(const [key,text] of Object.entries(DATA.contexts)){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=key;label.append(input,document.createTextNode(' '+text));$('contexts').append(label);option($('coverage-context'),key,text);}
 $('coverage-context').value='';$('coverage-context').onchange=renderCoverage;
 $('lom-view').value=DATA.lom_view;$('provenance').textContent='Semilla de partición: '+DATA.split_seed+' · Catálogo: '+DATA.catalog_sha256.slice(0,16)+' · Las etiquetas no añaden entradas ni memoria a la política.';
 for(const [scenario,bg] of Object.entries(DATA.maps)){const img=new Image();img.onload=()=>{if(episode.scenario===scenario)drawMap();};img.src=bg.image;mapImages.set(scenario,img);}
 $('episode').onchange=()=>chooseEpisode($('episode').value);$('segment').onchange=()=>{const row=episode.segments.find(r=>r.segment_id===$('segment').value);stop();go(Number(row.pair_start));form({key:episode.key,start:Number(row.pair_start),stop:Number(row.pair_stop),scope:'local',behavior:'',side:'unknown',status:'pending_review',maneuver:'',phase:'',context:[],notes:'',source:row.segment_id});};
 $('pair').oninput=()=>{stop();go($('pair').value);};$('prev').onclick=()=>{stop();go(pair-1);};$('next').onclick=()=>{stop();go(pair+1);};
 $('play').onclick=()=>{if(timer!==null){stop();return;}if(pair===Number($('pair').max))go(0);$('play').textContent='Pausar';timer=setInterval(()=>{if(pair>=Number($('pair').max)){stop();return;}go(pair+1);},Number($('speed').value));};$('speed').onchange=stop;
 $('map-mode').onchange=drawMap;$('lom-view').onchange=renderFrames;$('start').oninput=drawMap;$('end').oninput=drawMap;$('scope').onchange=scopeUI;
 $('behavior').onchange=()=>{if($('behavior').value){const directional=DATA.behaviors[$('behavior').value].requires_evasion_side;if(!directional)$('side').value='not_applicable';else if(!['left','right','unknown'].includes($('side').value))$('side').value='unknown';}};
 $('mark-start').onclick=()=>{$('start').value=pair;drawMap();};$('mark-end').onclick=()=>{$('end').value=pair;drawMap();};$('whole').onclick=()=>{$('start').value=0;$('end').value=episode.num_pairs-1;drawMap();};
 $('new').onclick=()=>{form();renderList();};$('save').onclick=()=>act(()=>{const saved=C.put(values());form(saved);changed();message('Intervalo guardado.');});
 $('split').onclick=()=>act(()=>{const parts=C.split(editing,pair);form(parts[0]);changed();message('Intervalo dividido. Revisa y confirma las etiquetas de cada parte.');});
 $('timeline').onclick=event=>{const rect=$('timeline').getBoundingClientRect();stop();go(Math.min(episode.num_pairs-1,Math.floor((event.clientX-rect.left)/rect.width*episode.num_pairs)));};
 $('import').onclick=()=>$('file').click();$('file').onchange=async()=>{const file=$('file').files[0];if(!file)return;try{const text=await file.text();if(file.name.toLowerCase().endsWith('.json')){C.restore(JSON.parse(text));message('Sesión restaurada.');}else{const result=C.importRows(C.parseCSV(text),file.name,text);message(result.duplicate?'Este CSV ya estaba importado.':result.imported+' intervalos importados; '+result.skipped+' registros conservados solo como referencia (episodio no incluido o solapamiento).');}form();changed();}catch(error){message(error.message,true);}finally{$('file').value='';}};
 $('export-csv').onclick=()=>act(()=>download('reviewed_intervals.csv',C.csv(),'text/csv;charset=utf-8'));
 $('export-session').onclick=()=>download('review_session.json',JSON.stringify(C.session(),null,2),'application/json');
 for(let i=0;i<2;i++)$('lom-'+i).onclick=()=>{const canvas=$('lom-'+i),win=window.open('','_blank');if(win){const img=win.document.createElement('img');img.src=canvas.toDataURL();img.style='width:min(90vw,900px);image-rendering:pixelated';win.document.body.append(img);}};
 let restored=false;try{const saved=localStorage.getItem(storageKey);if(saved){C.restore(JSON.parse(saved));restored=true;}}catch(error){message('No se pudo restaurar el autoguardado: '+error.message,true);}
 if(!restored)for(const source of DATA.references){try{C.importRows(source.rows,source.filename,source.text);}catch(error){message('Referencia '+source.filename+': '+error.message,true);}}
 chooseEpisode(episode.key);renderCoverage();if(restored)message('Se restauró la sesión guardada en este navegador.');
 globalThis.temporalReview={core:C,get episode(){return episode;},get pair(){return pair;},go,chooseEpisode,lomPixel,drawMap,changed};
}
</script></html>'''


def validate_policy_spec(spec: dict) -> None:
    expected = {"version": 1, "lom_encoding": "raw_0_100", "lom_layout": "ros2_recorded_ccw",
                "target_encoding": "distance_m_angle_rad", "sequence_length": 2,
                "training_stride": 2, "label_normalization": "legacy_training_pool_minmax"}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported checkpoint preprocessing specification")
    lo, hi = np.asarray(spec["label_min"], dtype=np.float32), np.asarray(spec["label_max"], dtype=np.float32)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi < lo):
        raise ValueError("Invalid checkpoint label normalization bounds")


def policy_frame(grid, target, source: str = "recorded_npz") -> np.ndarray:
    values = np.asarray(grid, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if values.shape != (80, 80) or target.shape != (2,) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("Policy observations require a finite 80x80 LOM and two target values")
    if source == "ros_occupancy_grid":
        values = np.rot90(values)
    elif source != "recorded_npz":
        raise ValueError(f"Unsupported observation source: {source}")
    return np.vstack((values, np.repeat(target[:, None], 80, axis=1))).astype(np.float32, copy=False)


def normalize_policy_actions(actions, spec: dict, inverse: bool = False) -> np.ndarray:
    validate_policy_spec(spec)
    values = np.asarray(actions, dtype=np.float32).copy()
    if values.shape[-1:] != (3,) or not np.isfinite(values).all():
        raise ValueError("Policy actions must contain finite vx, vy and wz values")
    for column, lower, width in ((0, 0.0, 1.0), (1, -1.0, 2.0)):
        minimum = np.float32(spec["label_min"][column])
        span = np.float32(spec["label_max"][column]) - minimum
        if inverse:
            values[..., column] = minimum if span == 0 else (values[..., column] - lower) / width * span + minimum
        else:
            values[..., column] = 0.0 if span == 0 else (values[..., column] - minimum) / span * width + lower
    return values


def load_frozen_selection(directory: Path, environment: str, recipe: str,
                          data_root: Path | None = None, catalog_dir: Path | None = None) -> dict:
    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / "datasets.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("phase") != "datasets":
        raise ValueError("Unsupported prepared datasets metadata")
    catalog_dir = (catalog_dir or directory.parent).expanduser().resolve()
    catalog, episodes, root = read_catalog(catalog_dir, data_root)
    verify_partitions(catalog, episodes)
    if (file_hash(catalog_dir / "catalog.json") != metadata["catalog_sha256"]
            or catalog["config"] != metadata["config"] or catalog["split_seed"] != metadata["split_seed"]):
        raise ValueError("Prepared datasets and catalog refer to different partitions")
    if environment not in catalog["config"]["environment_sets"] or recipe not in catalog["config"]["recipes"]:
        raise ValueError("Unknown environment set or dataset recipe")
    selected = metadata["cells"][f"{environment}/{recipe}"]
    baseline = metadata["cells"].get(f"{environment}/D0", {})
    if selected["status"] != "ready":
        raise ValueError(f"Dataset {environment}/{recipe} is not ready: {selected.get('deficits', selected['status'])}")
    if baseline.get("status") != "ready" or baseline.get("selection") != "all accepted training pairs":
        raise ValueError("The complete D0 training manifest is needed to freeze label normalization")
    by_key = {e["episode_key"]: e for e in episodes}
    assignment = catalog["episode_roles"][environment]
    paths = {"train": directory / environment / recipe / "train.jsonl",
             "validation": directory / environment / "validation.jsonl",
             "normalization": directory / environment / "D0" / "train.jsonl"}
    expected = {"train": selected["train_sha256"], "validation": selected["validation_sha256"],
                "normalization": baseline["train_sha256"]}
    manifests = {}
    for name, path in paths.items():
        if file_hash(path) != expected[name]:
            raise ValueError(f"Prepared manifest changed: {path}")
        manifests[name] = [json.loads(line) for line in path.read_text().splitlines()]
        verify_manifest(path, manifests[name], by_key, assignment, "validation" if name == "validation" else "train")
    if len(manifests["train"]) != selected["train_pairs"] or len(manifests["validation"]) != selected["validation_pairs"]:
        raise ValueError("Prepared manifest counts do not match datasets.json")
    for name, role in (("normalization", "train"), ("validation", "validation")):
        if len(manifests[name]) != sum(e["num_pairs"] for e in episodes if assignment.get(e["episode_key"]) == role):
            raise ValueError(f"The {name} manifest must cover its complete frozen partition")
    verify_selection(manifests["train"], selected, catalog["config"])
    pool = {(r["episode_key"], r["pair_index"]) for r in manifests["normalization"]}
    if any((r["episode_key"], r["pair_index"]) not in pool for r in manifests["train"]):
        raise ValueError("Selected training pairs do not belong to the D0 training pool")
    return {"metadata": metadata, "root": root, "episodes": episodes, "manifests": manifests,
            "manifest_hashes": expected, "datasets_sha256": file_hash(directory / "datasets.json"),
            "environment_set": environment, "dataset": recipe}


class FrozenPairDataset:
    def __init__(self, rows: list[dict], bank: dict, spec: dict):
        self.rows, self.bank, self.spec = rows, bank, spec

    def __len__(self):
        return len(self.rows)

    def raw_pair(self, index: int) -> list[tuple]:
        row = self.rows[index]
        return [self.bank[row["run_file"], i] for i in row["source_indices"]]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        frames = self.raw_pair(index)
        return (np.stack([policy_frame(grid, target) for grid, target, _ in frames]),
                normalize_policy_actions(frames[-1][2], self.spec))


def prepare_frozen_pairs(selection: dict) -> tuple[FrozenPairDataset, FrozenPairDataset, dict]:
    manifests = selection["manifests"]
    needed, fitting, owners = defaultdict(set), defaultdict(set), {}
    for row in manifests["train"] + manifests["validation"]:
        needed[row["run_file"]].update(row["source_indices"])
    for row in manifests["normalization"]:
        fitting[row["run_file"]].add(row["label_source_index"])
    for episode in selection["episodes"]:
        for index in episode["source_indices"]:
            owners[episode["run_file"], index] = episode["episode_id"]
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    bank = {}
    for run in sorted(set(needed) | set(fitting)):
        samples = read_samples(selection["root"] / run)
        for index in sorted(needed[run] | fitting[run]):
            sample = samples[index]
            if int(sample["meta"]["episode_id"]) != owners[run, index]:
                raise ValueError(f"Manifest observation refers to a different episode: {run}:{index}")
            label = np.asarray(sample["labels"], dtype=np.float32).reshape(-1)
            if label.shape != (3,) or not np.isfinite(label).all():
                raise ValueError(f"Invalid action at {run}:{index}")
            if index in fitting[run]:
                minimum, maximum = np.minimum(minimum, label), np.maximum(maximum, label)
            if index in needed[run]:
                features = sample["features"]
                grid = np.asarray(features["occ_grid"])
                target = np.asarray(features["target"], dtype=np.float32).reshape(-1)
                policy_frame(grid, target)
                bank[run, index] = (grid, target, label)
    spec = {"version": 1, "lom_encoding": "raw_0_100", "lom_layout": "ros2_recorded_ccw",
            "target_encoding": "distance_m_angle_rad", "sequence_length": 2, "training_stride": 2,
            "label_normalization": "legacy_training_pool_minmax",
            "label_min": minimum.tolist(), "label_max": maximum.tolist(),
            "normalization_manifest_sha256": selection["manifest_hashes"]["normalization"]}
    validate_policy_spec(spec)
    return (FrozenPairDataset(manifests["train"], bank, spec),
            FrozenPairDataset(manifests["validation"], bank, spec), spec)


def run_training(args, out: Path, seed: int) -> int:
    directory = args.datasets_dir.expanduser().resolve()
    metadata = json.loads((directory / "datasets.json").read_text())
    key = f"{args.environment_set}/{args.dataset}"
    if metadata.get("cells", {}).get(key, {}).get("status") != "ready":
        raise ValueError(f"Dataset is not ready: {key}")
    command = [sys.executable, "-u", str(SCRIPT_DIR / "train.py"), "--datasets-dir", str(directory),
               "--environment-set", args.environment_set, "--dataset", args.dataset,
               "--seed", str(seed), "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--device", args.device, "--output-dir", str(out / "training")]
    if args.data_root:
        command.extend(["--data-root", str(args.data_root.expanduser().resolve())])
    if args.catalog_dir:
        command.extend(["--catalog-dir", str(args.catalog_dir.expanduser().resolve())])
    if args.check_only:
        command.append("--check-only")
    record = {"phase": "training", "seed": seed, "seed_source": "provided" if args.seed is not None else "random",
              "datasets_sha256": file_hash(directory / "datasets.json"), "cell": key,
              "command": command, "provenance": provenance(), "status": "running"}
    write_json(out / "run.json", record)
    with (out / "console.log").open("w") as log:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
        except OSError:
            record["status"] = "failed"
            write_json(out / "run.json", record)
            raise
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            record["status"] = "interrupted"
            write_json(out / "run.json", record)
            raise
        finally:
            process.stdout.close()
    record.update(status="checked" if args.check_only and code == 0 else "completed" if code == 0 else "failed",
                  returncode=code)
    write_json(out / "run.json", record)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write the initial experiment configuration")
    init.add_argument("--output", type=Path, required=True)
    cat = sub.add_parser("catalog", help="Describe accepted data and export training behavior review")
    cat.add_argument("--data-root", type=Path, default=DATA_ROOT)
    cat.add_argument("--config", type=Path)
    cat.add_argument("--seed", type=int)
    cat.add_argument("--previews", type=int, default=36)
    cat.add_argument("--output-dir", type=Path)
    prep = sub.add_parser("build", help="Build D0 through D3 using reviewed behavior annotations")
    prep.add_argument("--catalog-dir", type=Path, required=True)
    prep.add_argument("--data-root", type=Path)
    prep.add_argument("--annotations", type=Path, nargs="*", default=[])
    prep.add_argument("--seed", type=int)
    prep.add_argument("--output-dir", type=Path)
    training = sub.add_parser("train", help="Run the trainer using frozen dataset manifests")
    training.add_argument("--datasets-dir", type=Path, required=True)
    training.add_argument("--environment-set", required=True)
    training.add_argument("--dataset", required=True)
    training.add_argument("--catalog-dir", type=Path)
    training.add_argument("--data-root", type=Path)
    training.add_argument("--seed", type=int)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--check-only", action="store_true")
    training.add_argument("--output-dir", type=Path)
    rev = sub.add_parser("review", help="Review specific training segments without changing the split")
    rev.add_argument("--catalog-dir", type=Path, required=True)
    rev.add_argument("--data-root", type=Path)
    selection = rev.add_mutually_exclusive_group(required=True)
    selection.add_argument("--segment-ids", nargs="+")
    selection.add_argument("--reuse-previews", type=Path, help="Reuse the segment selection of an existing review directory")
    rev.add_argument("--output-dir", type=Path)
    rev.add_argument("--temporal", action="store_true", help="Review full episodes and annotate disjoint pair intervals")
    rev.add_argument("--annotations", type=Path, nargs="*", default=[], help="Import annotation CSV files as references")
    for review_parser in (cat, rev):
        review_parser.add_argument("--maps-root", type=Path, help=f"Scenario maps directory (default: {MAPS_ROOT})")
        review_parser.add_argument("--map-variant", choices=("prohibition", "nav", "standard"), default="prohibition")
        review_parser.add_argument("--map-padding", type=float, default=1.25, help="Context margin around the segment in meters")
        review_parser.add_argument("--lom-view", choices=("robot", "array"), default="robot",
                                   help="Display LOM using robot axes or stored array indices")
    args = parser.parse_args(argv)
    out = None
    try:
        if args.command == "init":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists():
                raise ValueError(f"Output already exists: {args.output}")
            write_json(args.output, DEFAULT_CONFIG)
            return 0
        if args.command in ("review", "build"):
            directory = args.catalog_dir.expanduser().resolve()
            if not (directory / "catalog.json").is_file():
                raise ValueError(f"Catalog metadata not found: {directory / 'catalog.json'}. Choose an existing catalog directory.")
            for path in args.annotations:
                if not path.expanduser().is_file():
                    raise ValueError(f"Annotation CSV not found: {path}")
            args.annotations = [p.expanduser().resolve() for p in args.annotations]
        if args.command == "review":
            out = new_output(args.output_dir, args.command)
            result = review_segments(directory, args.data_root, args.segment_ids, out,
                                     args.reuse_previews, args.maps_root, args.map_variant, args.map_padding,
                                     args.lom_view, args.temporal, args.annotations)
            print(f"Split seed: {result['split_seed']}")
            print(f"Output: {out}")
            return 0
        seed = generated_seed(args.seed)
        out = new_output(args.output_dir, args.command)
        if args.command == "train":
            code = run_training(args, out, seed)
            print(f"Seed: {seed} ({'provided' if args.seed is not None else 'randomly generated'})")
            print(f"Output: {out}")
            return code
        if args.command == "catalog":
            if args.previews < 0:
                raise ValueError("The preview count cannot be negative")
            result = catalog(args.data_root, read_config(args.config), out, seed, args.previews,
                             args.maps_root, args.map_variant, args.map_padding, args.lom_view)
            print(json.dumps({k: result[k] for k in ("num_samples", "num_episodes", "num_pairs", "num_segments", "action_counts")}, indent=2))
        else:
            result = build(directory, args.data_root, args.annotations, out, seed)
            print(json.dumps({k: {f: v[f] for f in ("status", "train_pairs", "validation_pairs")} for k, v in result["cells"].items()}, indent=2))
            print(f"Report: {out / result['report']}")
        print(f"Seed: {seed} ({'provided' if args.seed is not None else 'randomly generated'})")
        print(f"Output: {out}")
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        if out is not None:
            write_json(out / "FAILED.json", {"error": str(exc)})
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
