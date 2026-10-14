#!/usr/bin/env python3
"""Generate safe, situation-oriented mapless navigation path pools.

Pipeline: occupancy map -> safe grid -> any-angle Theta* -> anchored smoothing
-> cubic spline refinement -> dense clearance validation -> task/geometry labels.

This is an OFFLINE dataset tool. It never commands the robot and never replaces
an existing run directory. Outputs are compatible with the split poses format
used by the previous generate_path_pool.py while adding navigation-task metadata.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting should not prevent valid data generation
    plt = None

CLASSES = ("straight", "semi_complex", "complex", "v_complex")
COLORS = {"straight": "#2878B5", "semi_complex": "#4A9D62",
          "complex": "#E39C37", "v_complex": "#C84D4D"}
DEFAULT_MAPS = "src/Hardware/justina_description/webots_env/environments/maps"
DEFAULT_DATA = "src/Navigation/mapless_navigation/dataset/data"


def map_folder(maps_root: Path, scenario: str) -> Path:
    candidates = sorted(p for p in maps_root.iterdir()
                        if p.is_dir() and p.name.startswith(f"{scenario}_"))
    if not candidates:
        raise FileNotFoundError(f"No map folder starting with {scenario}_ under {maps_root}")
    return candidates[0]


def load_map(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as stream:
        meta = yaml.safe_load(stream)
    image_path = (yaml_path.parent / meta["image"]).resolve()
    pixels = np.flipud(np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8))
    negate = int(meta.get("negate", 0))
    occupancy = pixels.astype(float) / 255.0 if negate else (255.0 - pixels.astype(float)) / 255.0
    free = occupancy < float(meta.get("free_thresh", 0.196))
    # Treat a pixel as traversable only when the map explicitly identifies it as free.
    res = float(meta["resolution"])
    origin = meta["origin"]
    ox, oy = float(origin[0]), float(origin[1])
    # Padding with occupied space models the map boundary as non-traversable too.
    padded = np.pad(free, 1, mode="constant", constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * res
    # A half-cell correction is conservative for distances inferred from cell centers.
    clearance = np.maximum(0.0, clearance - 0.5 * res)
    return {"yaml": str(yaml_path.resolve()), "image": image_path, "pixels": pixels,
            "meta": meta, "free": free, "clearance": clearance, "res": res,
            "ox": ox, "oy": oy, "width": pixels.shape[1], "height": pixels.shape[0]}


def make_planning_context(ctx: dict[str, Any], hard_m: float, factor: int = 5) -> dict[str, Any]:
    """Build a conservative coarse grid for search; validate final routes on the original map.

    A coarse cell is traversable only if every underlying map cell satisfies the
    original clearance constraint. This reduces search nodes by factor**2 without
    weakening the final dense safety check.
    """
    factor = max(1, int(factor))
    safe = ctx["free"] & (ctx["clearance"] + 1e-9 >= hard_m)
    h, w = safe.shape
    hp = int(math.ceil(h / factor) * factor)
    wp = int(math.ceil(w / factor) * factor)
    safe_pad = np.pad(safe, ((0, hp - h), (0, wp - w)),
                      mode="constant", constant_values=False)
    clearance_pad = np.pad(ctx["clearance"], ((0, hp-h), (0, wp-w)),
                           mode="constant", constant_values=0.0)
    blocks_safe = safe_pad.reshape(hp // factor, factor, wp // factor, factor)
    blocks_clearance = clearance_pad.reshape(hp // factor, factor, wp // factor, factor)
    coarse_free = np.all(blocks_safe, axis=(1, 3))
    coarse_clearance = np.min(blocks_clearance, axis=(1, 3))
    # Mark non-traversable blocks explicitly, so Theta* and its LOS test agree.
    coarse_clearance = np.where(coarse_free, coarse_clearance, 0.0)
    return {**ctx, "free": coarse_free, "clearance": coarse_clearance,
            "res": ctx["res"] * factor, "width": coarse_free.shape[1],
            "height": coarse_free.shape[0], "planning_factor": factor,
            "is_planning_grid": True}


def prepare_endpoint_cells(ctx: dict[str, Any], hard_m: float, target_class: str,
                           endpoint_band_m: float) -> np.ndarray:
    """Precompute eligible endpoint cells once per class, not once per attempt."""
    eligible = ctx["free"] & (ctx["clearance"] >= hard_m)
    if target_class != "straight":
        eligible &= ctx["clearance"] <= hard_m + endpoint_band_m + 0.25
    return np.argwhere(eligible)


def world_to_grid(ctx: dict[str, Any], x: float, y: float) -> tuple[int, int] | None:
    col = int(math.floor((x - ctx["ox"]) / ctx["res"]))
    row = int(math.floor((y - ctx["oy"]) / ctx["res"]))
    if row < 0 or col < 0 or row >= ctx["height"] or col >= ctx["width"]:
        return None
    return row, col


def grid_to_world(ctx: dict[str, Any], rc: tuple[int, int]) -> tuple[float, float]:
    r, c = rc
    return (ctx["ox"] + (c + 0.5) * ctx["res"],
            ctx["oy"] + (r + 0.5) * ctx["res"])


def bilinear(field: np.ndarray, ctx: dict[str, Any], points: np.ndarray) -> np.ndarray:
    """Sample a grid field in world coordinates; out-of-map points return zero."""
    out = np.zeros(len(points), dtype=float)
    h, w = field.shape
    for i, (x, y) in enumerate(points):
        col = (float(x) - ctx["ox"]) / ctx["res"] - 0.5
        row = (float(y) - ctx["oy"]) / ctx["res"] - 0.5
        if row < 0 or col < 0 or row > h - 1 or col > w - 1:
            out[i] = 0.0
            continue
        r0, c0 = int(math.floor(row)), int(math.floor(col))
        r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
        fr, fc = row - r0, col - c0
        out[i] = ((1-fr)*(1-fc)*field[r0, c0] + (1-fr)*fc*field[r0, c1]
                  + fr*(1-fc)*field[r1, c0] + fr*fc*field[r1, c1])
    return out


def line_points(a: Iterable[float], b: Iterable[float], spacing: float) -> np.ndarray:
    p0, p1 = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dist = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(dist / max(spacing, 1e-3))) + 1)
    return p0[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0)[None, :]


def line_is_safe(ctx: dict[str, Any], a: Iterable[float], b: Iterable[float], hard_m: float) -> bool:
    pts = line_points(a, b, max(ctx["res"] * 0.45, 0.01))
    cells = [world_to_grid(ctx, float(x), float(y)) for x, y in pts]
    if any(cell is None for cell in cells):
        return False
    for cell in cells:
        r, c = cell
        if not ctx["free"][r, c] or ctx["clearance"][r, c] + 1e-9 < hard_m:
            return False
    return True


def theta_star(ctx: dict[str, Any], start_rc: tuple[int, int], goal_rc: tuple[int, int],
               hard_m: float) -> list[list[float]] | None:
    """8-connected any-angle Theta* with line-of-sight relaxation."""
    free = ctx["free"] & (ctx["clearance"] >= hard_m)
    sr, sc = start_rc
    gr, gc = goal_rc
    if not (free[sr, sc] and free[gr, gc]):
        return None
    start, goal = (sr, sc), (gr, gc)
    g: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {start: start}
    queue: list[tuple[float, float, int, int]] = [(math.hypot(gr-sr, gc-sc), 0.0, sr, sc)]
    closed: set[tuple[int, int]] = set()
    h, w = free.shape

    def los(u: tuple[int, int], v: tuple[int, int]) -> bool:
        # Integer Bresenham traversal avoids allocating np.linspace arrays for each
        # neighbor relaxation. The final route is independently checked on the full map.
        r0, c0 = u
        r1, c1 = v
        dc, dr = abs(c1-c0), abs(r1-r0)
        sc = 1 if c0 < c1 else -1
        sr = 1 if r0 < r1 else -1
        err = dc - dr
        while True:
            if r0 < 0 or c0 < 0 or r0 >= h or c0 >= w or not free[r0, c0]:
                return False
            if r0 == r1 and c0 == c1:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c0 += sc
            if e2 < dc:
                err += dc
                r0 += sr

    while queue:
        _, cost, r, c = heapq.heappop(queue)
        here = (r, c)
        if here in closed or cost > g.get(here, float("inf")) + 1e-9:
            continue
        if here == goal:
            chain = [goal]
            while chain[-1] != start:
                prev = parent[chain[-1]]
                if prev == chain[-1]:
                    return None
                chain.append(prev)
            chain.reverse()
            points = [list(grid_to_world(ctx, rc)) for rc in chain]
            # Theta* parent links already have line of sight; remove collinear/visible intermediates.
            return shortcut_path(ctx, points, hard_m)
        closed.add(here)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc):
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and free[nr, nc]):
                    continue
                nxt = (nr, nc)
                par = parent[here]
                if los(par, nxt):
                    candidate_parent = par
                    edge = math.hypot(nr-par[0], nc-par[1])
                    candidate_g = g[par] + edge
                else:
                    candidate_parent = here
                    edge = math.hypot(dr, dc)
                    candidate_g = g[here] + edge
                if candidate_g + 1e-9 < g.get(nxt, float("inf")):
                    g[nxt] = candidate_g
                    parent[nxt] = candidate_parent
                    heuristic = math.hypot(gr-nr, gc-nc)
                    heapq.heappush(queue, (candidate_g + heuristic, candidate_g, nr, nc))
    return None


def shortcut_path(ctx: dict[str, Any], points: list[list[float]], hard_m: float) -> list[list[float]]:
    if len(points) < 3:
        return points
    result = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not line_is_safe(ctx, points[i], points[j], hard_m):
            j -= 1
        result.append(points[j])
        i = j
    return result


def polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-8))
    pts = pts[keep]
    if len(pts) < 2:
        return pts.copy()
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    count = max(2, int(math.ceil(arc[-1] / max(spacing, 1e-3))) + 1)
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack((np.interp(target, arc, pts[:, 0]), np.interp(target, arc, pts[:, 1])))


def anchored_smooth(points: np.ndarray, data_weight: float = 0.12,
                    smooth_weight: float = 0.20, iterations: int = 45) -> np.ndarray:
    """Mild resampled Laplacian smoother; fixed endpoints and an anchor term."""
    anchor = np.asarray(points, dtype=float)
    if len(anchor) < 4:
        return anchor.copy()
    current = anchor.copy()
    for _ in range(iterations):
        updated = current.copy()
        lap = current[:-2] + current[2:] - 2.0 * current[1:-1]
        updated[1:-1] = current[1:-1] + data_weight * (anchor[1:-1] - current[1:-1]) + smooth_weight * lap
        updated[0], updated[-1] = anchor[0], anchor[-1]
        current = updated
    return current


def thin_controls(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points) <= 4:
        return points.copy()
    keep = [points[0]]
    for p in points[1:-1]:
        if np.linalg.norm(p - keep[-1]) >= spacing_m:
            keep.append(p)
    if np.linalg.norm(points[-1] - keep[-1]) > 1e-7:
        keep.append(points[-1])
    if len(keep) < 4:
        # Keep the original sampled control set: short detours may need it.
        return points.copy()
    return np.asarray(keep, dtype=float)


def spline_refine(points: np.ndarray, spacing_m: float = 0.04,
                  control_spacing_m: float = 0.18, sigma_m: float = 0.025) -> np.ndarray | None:
    controls = thin_controls(np.asarray(points, dtype=float), control_spacing_m)
    if len(controls) < 4:
        return None
    try:
        tck, _ = splprep([controls[:, 0], controls[:, 1]], k=3,
                         s=max(1e-8, len(controls) * sigma_m * sigma_m), per=False)
        count = max(4, int(math.ceil(polyline_length(controls) / spacing_m)) + 1)
        u = np.linspace(0.0, 1.0, count)
        x, y = splev(u, tck)
        out = np.column_stack((x, y)).astype(float)
        frac = np.linspace(0.0, 1.0, len(out))[:, None]
        out += (1-frac) * (controls[0] - out[0]) + frac * (controls[-1] - out[-1])
        out[0], out[-1] = controls[0], controls[-1]
        return out if np.all(np.isfinite(out)) else None
    except (ValueError, TypeError, RuntimeError):
        return None


def route_validation(ctx: dict[str, Any], points: np.ndarray, hard_m: float,
                     spacing_m: float = 0.025) -> dict[str, Any]:
    dense = resample(points, spacing_m)
    clearance = bilinear(ctx["clearance"], ctx, dense)
    cells = [world_to_grid(ctx, float(x), float(y)) for x, y in dense]
    oob = sum(c is None for c in cells)
    occupied = 0
    for cell in cells:
        if cell is None:
            occupied += 1
        else:
            r, c = cell
            if not ctx["free"][r, c]:
                occupied += 1
    return {"path_length_m": polyline_length(points),
            "min_clearance_m": float(np.min(clearance)) if len(clearance) else 0.0,
            "mean_clearance_m": float(np.mean(clearance)) if len(clearance) else 0.0,
            "dense_samples": len(dense), "occupied_samples": int(occupied),
            "out_of_bounds_samples": int(oob),
            "map_collision": bool(occupied or oob),
            "clearance_ok": bool(len(clearance) and float(np.min(clearance)) + 1e-7 >= hard_m),
            "safe": bool(len(dense) >= 2 and not occupied and not oob and
                         float(np.min(clearance)) + 1e-7 >= hard_m)}


def smooth_theta_route(ctx: dict[str, Any], raw_points: list[list[float]], hard_m: float,
                       max_length_ratio: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Run anchored smoothing then spline; choose smoothest safe valid candidate."""
    raw = np.asarray(raw_points, dtype=float)
    anchor = resample(raw, 0.05)
    pre_smoothed = anchored_smooth(anchor)
    raw_len = max(polyline_length(raw), 1e-6)
    candidates: list[tuple[str, np.ndarray]] = []
    # Try increasingly regularized cubic splines; weaker values are tried first.
    for sigma in (0.012, 0.020, 0.030, 0.040):
        refined = spline_refine(pre_smoothed, sigma_m=sigma)
        if refined is not None:
            candidates.append((f"theta_smooth_spline_sigma_{sigma:.3f}", refined))
    # If a spline cannot be fitted, retain the anchored smooth candidate as fallback.
    candidates.append(("theta_anchored_smooth_fallback", pre_smoothed))
    candidates.append(("theta_star_fallback", raw))
    accepted: list[tuple[float, str, np.ndarray, dict[str, Any]]] = []
    rejected = Counter()
    base_validation = route_validation(ctx, raw, hard_m)
    for name, cand in candidates:
        cand = np.asarray(cand, dtype=float)
        if len(cand) < 2 or not np.all(np.isfinite(cand)):
            rejected["invalid_geometry"] += 1
            continue
        cand[0], cand[-1] = raw[0], raw[-1]
        metrics = route_validation(ctx, cand, hard_m)
        if not metrics["safe"]:
            rejected["clearance_or_collision"] += 1
            continue
        if metrics["path_length_m"] > raw_len * max_length_ratio + 1e-5:
            rejected["excessive_length_growth"] += 1
            continue
        shape = curvature_metrics(cand)
        # Prefer lower high-frequency curvature variation; mildly prefer a longer safety margin.
        score = shape["curvature_tv_deg_m2"] - 0.08 * metrics["min_clearance_m"]
        accepted.append((score, name, cand.copy(), {**metrics, **shape}))
    if not accepted:
        if base_validation["safe"]:
            return raw, {**base_validation, **curvature_metrics(raw), "smoothing_status": "theta_star_fallback",
                         "smoothing_method": "theta_star", "rejections": dict(rejected)}
        raise RuntimeError("Theta* route failed dense safety validation")
    accepted.sort(key=lambda item: item[0])
    _, method, points, metrics = accepted[0]
    metrics.update({"smoothing_status": "refined" if "spline" in method else "safe_fallback",
                    "smoothing_method": method, "rejections": dict(rejected),
                    "raw_path_length_m": raw_len,
                    "length_ratio_vs_theta_star": metrics["path_length_m"] / raw_len})
    return points, metrics


def curvature_metrics(points: np.ndarray, sample_spacing_m: float = 0.10) -> dict[str, Any]:
    pts = resample(np.asarray(points, dtype=float), sample_spacing_m)
    if len(pts) < 3:
        return {"sinuosity": 1.0, "total_turn_deg": 0.0, "max_turn_deg": 0.0,
                "turn_event_count": 0, "supported_turn_count": 0,
                "turn_sign_changes": 0, "curvature_tv_deg_m2": 0.0}
    vec = np.diff(pts, axis=0)
    seglen = np.linalg.norm(vec, axis=1)
    good = seglen > 1e-8
    headings = np.arctan2(vec[good, 1], vec[good, 0])
    if len(headings) < 2:
        turns = np.zeros(0)
    else:
        dh = np.diff(headings)
        turns = np.degrees(np.arctan2(np.sin(dh), np.cos(dh)))
    # Filter low-level numerical jitter; turns from adjacent 10 cm segments are not events.
    sig = np.where(np.abs(turns) >= 2.5)[0]
    events: list[dict[str, Any]] = []
    for ix in sig:
        position = min(float(ix + 1) * sample_spacing_m, polyline_length(pts))
        signed = float(turns[ix])
        sign = 1 if signed > 0 else -1
        if (events and position - events[-1]["last_s_m"] <= 0.25
                and events[-1]["sign"] == sign):
            events[-1]["angle_deg"] += signed
            events[-1]["last_s_m"] = position
            events[-1]["peak_deg"] = max(events[-1]["peak_deg"], abs(signed))
        else:
            events.append({"s_m": position, "last_s_m": position,
                           "angle_deg": signed, "sign": sign, "peak_deg": abs(signed)})
    events = [e for e in events if abs(e["angle_deg"]) >= 10.0]
    signs = [1 if e["angle_deg"] > 0 else -1 for e in events if abs(e["angle_deg"]) >= 10.0]
    sign_changes = sum(a != b for a, b in zip(signs, signs[1:]))
    total_turn = float(np.sum(np.abs(turns)))
    direct = float(np.linalg.norm(pts[-1] - pts[0]))
    length = polyline_length(pts)
    curv = turns / max(sample_spacing_m, 1e-6)
    tv = float(np.sum(np.abs(np.diff(curv))) / max(length, 1e-6)) if len(curv) > 1 else 0.0
    return {"sinuosity": float(length / max(direct, 1e-6)),
            "total_turn_deg": total_turn,
            "max_turn_deg": float(np.max(np.abs(turns))) if len(turns) else 0.0,
            "turn_event_count": len(events), "supported_turn_count": len(events),
            "turn_sign_changes": int(sign_changes), "curvature_tv_deg_m2": tv}


def angular_metrics(points: np.ndarray) -> dict[str, float]:
    if len(points) < 2:
        return {"start_heading_rad": 0.0, "goal_heading_rad": 0.0}
    first = points[min(3, len(points)-1)] - points[0]
    last = points[-1] - points[max(0, len(points)-4)]
    return {"start_heading_rad": float(math.atan2(first[1], first[0])),
            "goal_heading_rad": float(math.atan2(last[1], last[0]))}


def route_task(target_class: str, direct_clear: bool, metrics: dict[str, Any],
               hard_clearance_m: float) -> str | None:
    """Require each geometry quota to correspond to a real navigation situation."""
    turns = int(metrics["turn_event_count"])
    supported = int(metrics["supported_turn_count"])
    if target_class == "straight":
        return "clear_travel" if direct_clear and turns == 0 else None
    if target_class == "semi_complex":
        # A genuine obstacle-avoidance maneuver, not a slightly curved clear route.
        return "single_obstacle_avoidance" if (not direct_clear and turns >= 1 and supported >= 1) else None
    if target_class == "complex":
        # Multiple supported bends plus a genuinely constrained clearance profile.
        constrained = float(metrics["min_clearance_m"]) <= hard_clearance_m + 0.20
        return "constrained_passage" if (turns >= 2 and supported >= 2 and constrained) else None
    if target_class == "v_complex":
        # Keep a genuinely multi-maneuver route, but do not require every detected
        # bend to be independently obstacle-supported: a spline can split one
        # physical maneuver into several geometric events. At least two distinct
        # obstacle-supported events remain mandatory.
        return "multiple_obstacle_navigation" if (turns >= 3 and supported >= 2) else None
    return None


def geometry_class(metrics: dict[str, Any], direct_clear: bool,
                   obstacle_supported_events: int | None = None) -> str | None:
    turns = int(metrics["turn_event_count"])
    total = float(metrics["total_turn_deg"])
    sinuosity = float(metrics["sinuosity"])
    supported = turns if obstacle_supported_events is None else obstacle_supported_events
    if direct_clear and turns == 0 and total < 15.0 and sinuosity <= 1.025:
        return "straight"
    # Semi-complex must show a real, obstacle-motivated bend; near-straight routes are rejected.
    # More demanding classes must be checked first; otherwise complex routes could
    # be swallowed by the broader semi-complex thresholds.
    # V-complex means multiple supported maneuvers, not a high-frequency zig-zag.
    if turns >= 3 and supported >= 2 and total >= 72.0 and sinuosity >= 1.065:
        return "v_complex"
    if turns >= 2 and supported >= 2 and total >= 48.0 and sinuosity >= 1.055:
        return "complex"
    if (turns >= 1 and supported >= 1 and total >= 18.0 and
            (sinuosity >= 1.04 or total >= 30.0) and turns <= 3 and total < 105.0):
        return "semi_complex"
    return None


def route_supported_turn_count(ctx: dict[str, Any], points: np.ndarray,
                               obstacle_influence_m: float) -> int:
    # Score turns near obstacles using local clearance around each detected bend.
    pts = resample(points, 0.10)
    if len(pts) < 4:
        return 0
    vec = np.diff(pts, axis=0)
    headings = np.arctan2(vec[:, 1], vec[:, 0])
    d = np.diff(headings)
    turns = np.degrees(np.arctan2(np.sin(d), np.cos(d)))
    clearance = bilinear(ctx["clearance"], ctx, pts[1:-1])
    supported_ix = np.where((np.abs(turns) >= 2.5) & (clearance[:len(turns)] <= obstacle_influence_m))[0]
    # Merge samples into spatially distinct same-direction maneuvers.
    events: list[dict[str, float | int]] = []
    for ix in supported_ix:
        sign = 1 if turns[ix] > 0 else -1
        if (events and sign == events[-1]["sign"]
                and (int(ix) - int(events[-1]["last_ix"])) * 0.10 <= 0.25):
            events[-1]["last_ix"] = int(ix)
        else:
            events.append({"sign": sign, "last_ix": int(ix)})
    return len(events)


def sample_endpoints(ctx: dict[str, Any], rng: random.Random, hard_m: float,
                     target_class: str, min_dist: float, max_dist: float,
                     endpoint_band_m: float, eligible_cells: np.ndarray | None = None
                     ) -> tuple[tuple[int, int], tuple[int, int]] | None:
    clearance = ctx["clearance"]
    cells = eligible_cells if eligible_cells is not None else np.argwhere(
        ctx["free"] & (clearance >= hard_m) &
        ((clearance <= hard_m + endpoint_band_m + 0.25) if target_class != "straight" else True))
    if len(cells) < 2:
        return None
    # Bounded endpoint sampling: bias away from easy straight-line pairs for non-straight classes.
    for _ in range(80):
        a = cells[rng.randrange(len(cells))]
        b = cells[rng.randrange(len(cells))]
        if np.array_equal(a, b):
            continue
        pa, pb = grid_to_world(ctx, (int(a[0]), int(a[1]))), grid_to_world(ctx, (int(b[0]), int(b[1])))
        dist = math.dist(pa, pb)
        if dist < min_dist or dist > max_dist:
            continue
        direct_ok = line_is_safe(ctx, pa, pb, hard_m)
        if target_class == "straight" and not direct_ok:
            continue
        if target_class != "straight" and direct_ok and rng.random() < 0.80:
            continue
        return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    return None


def build_complex_hotspots(ctx: dict[str, Any], plan_ctx: dict[str, Any],
                           paths: list[dict[str, Any]], eligible_mask: np.ndarray,
                           obstacle_influence_m: float, radius_m: float = 0.35) -> list[tuple[int, int]]:
    """Find safe planning cells near obstacle-supported bends in accepted complex routes.

    These cells are not copied trajectories: they are seeds for new start/goal
    combinations in the same challenging regions. Neighbors are included to avoid
    repeatedly starting at exactly the same cell.
    """
    base: set[tuple[int, int]] = set()
    for rec in paths:
        if rec.get("class") not in ("complex", "v_complex"):
            continue
        points = resample(np.asarray(rec.get("poses", []), dtype=float), 0.10)
        if len(points) < 5:
            continue
        clearance = bilinear(ctx["clearance"], ctx, points)
        for i in range(2, len(points) - 2):
            if clearance[i] > obstacle_influence_m:
                continue
            left = points[i] - points[i - 2]
            right = points[i + 2] - points[i]
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom < 1e-8:
                continue
            cosine = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
            angle = abs(math.degrees(math.acos(cosine)))
            if angle < 12.0:
                continue
            rc = world_to_grid(plan_ctx, float(points[i, 0]), float(points[i, 1]))
            if rc is not None and eligible_mask[rc[0], rc[1]]:
                base.add(rc)

    # Expand the bend anchors into a small neighborhood, but only within the same
    # safe endpoint mask and only up to the configured metric radius.
    radius_cells = max(0, int(math.ceil(radius_m / plan_ctx["res"])) )
    expanded: set[tuple[int, int]] = set(base)
    for r, c in base:
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if math.hypot(dr, dc) * plan_ctx["res"] > radius_m + 1e-9:
                    continue
                rr, cc = r + dr, c + dc
                if (0 <= rr < eligible_mask.shape[0] and 0 <= cc < eligible_mask.shape[1]
                        and eligible_mask[rr, cc]):
                    expanded.add((rr, cc))
    return sorted(expanded)


def direct_obstacle_features(ctx: dict[str, Any], start: list[float], goal: list[float],
                             hard_m: float, obstacle_influence_m: float) -> tuple[int, float]:
    """Cheap proxy for how many separate obstacle bands lie between two endpoints."""
    pts = line_points(start, goal, 0.08)
    clearance = bilinear(ctx["clearance"], ctx, pts)
    if len(clearance) == 0:
        return 0, 0.0
    mask = clearance < hard_m
    # Count distinct below-clearance runs, rejecting very short pixel-scale touches.
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    clusters = sum(1 for a, b in zip(starts, ends) if (b - a) * 0.08 >= 0.10)
    near_fraction = float(np.mean(clearance <= obstacle_influence_m))
    return int(clusters), near_fraction


def sample_v_complex_bundle(ctx: dict[str, Any], plan_ctx: dict[str, Any],
                            rng: random.Random, eligible_cells: np.ndarray,
                            eligible_mask: np.ndarray, hotspot_cells: list[tuple[int, int]],
                            args: argparse.Namespace,
                            diagnostics: dict[str, int] | None = None) -> tuple[tuple[int, int], tuple[int, int],
                                                               list[list[float]], dict[str, Any]] | None:
    """Exploit complex-route hotspots and sweep multiple goals around one start.

    Many candidate goals are sampled from a metric annulus around a selected start.
    Goal pairs are ranked using the obstacle bands along the direct segment; only a
    few high-potential pairs are planned with Theta*. The strongest safe raw route
    is then sent through the usual spline + full-resolution validation pipeline.
    """
    use_hotspot = bool(hotspot_cells) and rng.random() < args.v_hotspot_probability
    if diagnostics is not None:
        key = "_diag_v_hotspot_start_batches" if use_hotspot else "_diag_v_exploration_start_batches"
        diagnostics[key] = diagnostics.get(key, 0) + 1
    if use_hotspot:
        start_rc = hotspot_cells[rng.randrange(len(hotspot_cells))]
    else:
        a = eligible_cells[rng.randrange(len(eligible_cells))]
        start_rc = (int(a[0]), int(a[1]))
    start = list(grid_to_world(plan_ctx, start_rc))

    # A fixed start with many different valid goals is much more informative than
    # repeatedly throwing away independent random endpoint pairs.
    candidates: dict[tuple[int, int], tuple[float, int, float, list[float]]] = {}
    max_draws = max(120, int(args.v_goal_candidates) * 80)
    goal_draws = 0
    for _ in range(max_draws):
        if len(candidates) >= args.v_goal_candidates:
            break
        radius = rng.uniform(args.min_goal_distance_m, args.v_max_goal_distance_m)
        angle = rng.uniform(-math.pi, math.pi)
        wx = start[0] + radius * math.cos(angle)
        wy = start[1] + radius * math.sin(angle)
        rc = world_to_grid(plan_ctx, wx, wy)
        if rc is None or rc == start_rc or not eligible_mask[rc[0], rc[1]]:
            continue
        goal = list(grid_to_world(plan_ctx, rc))
        dist = math.dist(start, goal)
        if not (args.min_goal_distance_m <= dist <= args.v_max_goal_distance_m):
            continue
        if rc in candidates:
            continue
        # Coarse-grid rejection is conservative and cheap. The full map check is
        # repeated before any selected route is accepted.
        if line_is_safe(plan_ctx, start, goal, args.hard_clearance_m):
            continue
        # A coarse-grid rejection can be conservative; only keep truly obstacle-
        # driven pairs after rechecking the actual map at native resolution.
        if line_is_safe(ctx, start, goal, args.hard_clearance_m):
            continue
        interactions, near_fraction = direct_obstacle_features(
            ctx, start, goal, args.hard_clearance_m, args.obstacle_influence_m)
        # Include a tiny seeded tie-breaker but rank chiefly by separate obstacle bands.
        tie = rng.random() * 1e-3
        candidates[rc] = (interactions + near_fraction * 0.05 + tie, interactions,
                          near_fraction, goal)
        goal_draws += 1
        if diagnostics is not None:
            diagnostics["_diag_v_goal_candidates_evaluated"] = diagnostics.get("_diag_v_goal_candidates_evaluated", 0) + 1

    if not candidates:
        return None

    # Rank by direct-segment interaction potential, then plan only a handful. Prefer
    # different angular sectors so the sweep explores distinct maneuvers from one start.
    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
    selected: list[tuple[tuple[int, int], tuple[float, int, float, list[float]]]] = []
    used_sectors: set[int] = set()
    for item in ranked:
        rc, feat = item
        goal = feat[3]
        bearing = math.atan2(goal[1] - start[1], goal[0] - start[0])
        sector = int(((bearing + math.pi) / (2.0 * math.pi)) * 8.0) % 8
        if sector in used_sectors:
            continue
        selected.append(item)
        used_sectors.add(sector)
        if len(selected) >= args.v_plan_candidates:
            break
    if len(selected) < args.v_plan_candidates:
        selected_keys = {item[0] for item in selected}
        for item in ranked:
            if item[0] in selected_keys:
                continue
            selected.append(item)
            if len(selected) >= args.v_plan_candidates:
                break

    best: tuple[float, tuple[int, int], list[list[float]], dict[str, Any]] | None = None
    plans_tested = 0
    for goal_rc, features in selected:
        goal = features[3]
        # Defensive duplicate of the collection-stage native-resolution check.
        if line_is_safe(ctx, start, goal, args.hard_clearance_m):
            continue
        raw = theta_star(plan_ctx, start_rc, goal_rc, args.hard_clearance_m)
        plans_tested += 1
        if diagnostics is not None:
            diagnostics["_diag_v_theta_candidate_routes_evaluated"] = diagnostics.get("_diag_v_theta_candidate_routes_evaluated", 0) + 1
        if raw is None:
            continue
        raw[0], raw[-1] = start, goal
        raw_array = np.asarray(raw, dtype=float)
        length = polyline_length(raw_array)
        euclidean = math.dist(start, goal)
        if length > args.v_max_path_length_m or length < euclidean - 1e-4:
            continue
        geom = curvature_metrics(raw_array)
        supported = route_supported_turn_count(ctx, raw_array, args.obstacle_influence_m)
        turns = int(geom["turn_event_count"])
        total_turn = float(geom["total_turn_deg"])
        sinuosity = float(geom["sinuosity"])
        interactions = int(features[1])
        # Reward supported maneuvers first; route length and direct-line obstacle
        # interactions are secondary tie-breakers, not objectives on their own.
        score = (12.0 * supported + 5.0 * turns + total_turn / 24.0
                 + 12.0 * max(0.0, sinuosity - 1.0) + 2.5 * interactions
                 - 0.7 * max(0.0, length - euclidean - 0.35))
        promising = (turns >= 3 and supported >= 2 and total_turn >= 65.0
                     and sinuosity >= 1.055)
        if promising:
            score += 20.0
        search_info = {"search_strategy": "complex_hotspot_multi_goal" if use_hotspot else "multi_goal_exploration",
                       "goal_candidates_evaluated": int(goal_draws),
                       "theta_candidate_routes_evaluated": int(plans_tested),
                       "direct_obstacle_bands": interactions,
                       "raw_turn_events": turns, "raw_supported_turns": int(supported),
                       "raw_total_turn_deg": round(total_turn, 3),
                       "raw_sinuosity": round(sinuosity, 5),
                       "hotspot_start": bool(use_hotspot)}
        if best is None or score > best[0]:
            best = (score, goal_rc, raw, search_info)

    if best is None:
        return None
    _, goal_rc, raw, info = best
    info["goal_candidates_evaluated"] = int(goal_draws)
    info["theta_candidate_routes_evaluated"] = int(plans_tested)
    return start_rc, goal_rc, raw, info


def duplicate_candidate(points: np.ndarray, start: list[float], goal: list[float],
                       records: list[dict[str, Any]], same_class: str) -> bool:
    # Reject repeated endpoint pairs and near-identical spatial traces within the same class.
    for record in records:
        if record.get("class") != same_class:
            continue
        rs, rg = record["start"], record["goal"]
        if ((math.dist(start, rs) < 0.28 and math.dist(goal, rg) < 0.28) or
            (math.dist(start, rg) < 0.28 and math.dist(goal, rs) < 0.28)):
            return True
    return False


class ProgressLine:
    """Throttle progress by elapsed wall time; one line in terminals and logs."""
    def __init__(self, max_attempts: int | None, targets: dict[str, int],
                 report_interval_s: float = 5.0):
        self.max_attempts = max_attempts
        self.targets = dict(targets)
        self.target_total = sum(targets.values())
        self.report_interval_s = max(0.25, float(report_interval_s))
        self.interactive = sys.stdout.isatty()
        self.started = time.monotonic()
        self.last_print = 0.0
        self.last_event = "inicio"

    def update(self, attempts: int, current_class: str, accepted: int,
               per_class: dict[str, int], stage: str, force: bool = False) -> None:
        now = time.monotonic()
        elapsed = max(0.001, now - self.started)
        if not force and now - self.last_print < self.report_interval_s:
            return
        self.last_print = now
        rate = attempts / elapsed
        pct = min(100.0, 100.0 * attempts / self.max_attempts) if self.max_attempts else 0.0
        counts = " ".join(f"{name}={per_class[name]}/{self.targets[name]}" for name in CLASSES)
        attempt_limit = str(self.max_attempts) if self.max_attempts is not None else "sin_tope"
        pct_label = f"{pct:5.1f}%" if self.max_attempts else "sin tope"
        compact = (f"{elapsed:.0f}s {attempts}/{attempt_limit} {accepted}/{self.target_total} "
                   + " ".join(f"{name}={per_class[name]}/{self.targets[name]}" for name in CLASSES))
        narrow = (f"{elapsed:.0f}s {attempts}/{attempt_limit} {accepted}/{self.target_total} "
                  + " ".join(f"{name}={per_class[name]}" for name in CLASSES))
        detailed = (f"Avance {pct_label} | intento {attempts}/{attempt_limit} "
                    f"| aceptadas {accepted}/{self.target_total} | clases {counts} "
                    f"| actual {current_class} | etapa: {stage} "
                    f"| último: {self.last_event} | {elapsed:6.1f}s | {rate:5.1f} intentos/s")
        import shutil
        width = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
        line = detailed if width >= len(detailed) else (compact if width >= len(compact) else narrow)
        if not hasattr(self, "last_line_width"):
            self.last_line_width = 0
        padded = line.ljust(max(self.last_line_width, len(line)))
        self.last_line_width = len(padded)
        if self.interactive:
            sys.stdout.write("\r\x1b[2K" + line)
        else:
            sys.stdout.write("\r" + padded)
        sys.stdout.flush()

    def finish(self, attempts: int, current_class: str, accepted: int,
               per_class: dict[str, int], stage: str = "terminado") -> None:
        self.update(attempts, current_class, accepted, per_class, stage, force=True)
        print(flush=True)


def generate(ctx: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    rng = random.Random(args.seed)
    paths: list[dict[str, Any]] = []
    failures: Counter = Counter()
    per_class = {name: 0 for name in CLASSES}
    started = time.monotonic()
    attempts = 0
    progress = ProgressLine(args.max_attempts, args.target_counts, args.progress_interval_seconds)
    target_class = CLASSES[0]
    # Attempt allocation follows relative quota progress; no hidden per-class attempt ceiling.
    class_attempts = {name: 0 for name in CLASSES}
    plan_ctx = make_planning_context(ctx, args.hard_clearance_m, args.plan_cell_factor)
    endpoint_cells = {cls: prepare_endpoint_cells(
        plan_ctx, args.hard_clearance_m, cls, args.endpoint_obstacle_band_m) for cls in CLASSES}
    eligible_mask = np.zeros(plan_ctx["free"].shape, dtype=bool)
    v_cells = endpoint_cells["v_complex"]
    if len(v_cells):
        eligible_mask[v_cells[:, 0], v_cells[:, 1]] = True
    hotspot_cells: list[tuple[int, int]] = []
    hotspot_source_count = -1
    hotspot_announcement_done = False
    failures["_diag_v_goal_candidates_evaluated"] = 0
    failures["_diag_v_theta_candidate_routes_evaluated"] = 0
    failures["_diag_v_hotspot_start_batches"] = 0
    failures["_diag_v_exploration_start_batches"] = 0
    failures["_diag_v_hotspot_refreshes"] = 0
    failures["_diag_v_hotspot_cells"] = 0
    print(f"planning grid: {plan_ctx['width']}x{plan_ctx['height']} "
          f"at {plan_ctx['res']:.3f} m/cell (factor {plan_ctx['planning_factor']}; "
          f"full-resolution safety validation retained)", flush=True)

    while ((args.max_attempts is None or attempts < args.max_attempts)
           and (args.timeout_seconds is None or args.timeout_seconds <= 0
                or time.monotonic() - started < args.timeout_seconds)
           and any(per_class[k] < args.target_counts[k] for k in CLASSES)):
        missing = [k for k in CLASSES if per_class[k] < args.target_counts[k]]
        if not missing:
            break
        target_class = min(missing, key=lambda k: (class_attempts[k] / args.target_counts[k],
                                                   CLASSES.index(k)))
        attempts += 1
        class_attempts[target_class] += 1
        progress.update(attempts, target_class, len(paths), per_class, "muestreando inicio/meta")
        max_goal = args.v_max_goal_distance_m if target_class == "v_complex" else args.max_goal_distance_m
        search_info: dict[str, Any] = {}
        if target_class == "v_complex":
            failures["_diag_v_attempts"] += 1
            source_count = sum(1 for rec in paths if rec.get("class") in ("complex", "v_complex"))
            if source_count != hotspot_source_count:
                hotspot_cells = build_complex_hotspots(
                    ctx, plan_ctx, paths, eligible_mask, args.obstacle_influence_m,
                    args.v_hotspot_radius_m)
                hotspot_source_count = source_count
                failures["_diag_v_hotspot_refreshes"] += 1
                failures["_diag_v_hotspot_cells"] = len(hotspot_cells)
                if hotspot_cells and not hotspot_announcement_done:
                    print("v_complex: búsqueda dirigida activada", flush=True)
                    hotspot_announcement_done = True
            progress.update(attempts, target_class, len(paths), per_class, "explorando metas en zona compleja")
            bundle = sample_v_complex_bundle(ctx, plan_ctx, rng, endpoint_cells[target_class],
                                             eligible_mask, hotspot_cells, args, failures)
            if bundle is None:
                failures["v_complex:no_promising_goal_bundle"] += 1
                progress.last_event = "sin meta candidata prometedora"
                continue
            start_rc, goal_rc, raw, search_info = bundle
            progress.last_event = (f"{search_info.get('goal_candidates_evaluated', 0)} metas; "
                                   f"{search_info.get('theta_candidate_routes_evaluated', 0)} rutas Theta*")
            start = list(grid_to_world(plan_ctx, start_rc))
            goal = list(grid_to_world(plan_ctx, goal_rc))
        else:
            endpoints = sample_endpoints(plan_ctx, rng, args.hard_clearance_m, target_class,
                                         args.min_goal_distance_m, args.max_goal_distance_m,
                                         args.endpoint_obstacle_band_m,
                                         eligible_cells=endpoint_cells[target_class])
            if endpoints is None:
                failures[f"{target_class}:no_endpoint_pair"] += 1
                progress.last_event = "sin pareja inicio/meta"
                continue
            start_rc, goal_rc = endpoints
            start = list(grid_to_world(plan_ctx, start_rc))
            goal = list(grid_to_world(plan_ctx, goal_rc))
            # Easy examples should be genuinely cheap: straight routes need no search.
            direct_actual = line_is_safe(ctx, start, goal, args.hard_clearance_m)
            if target_class == "straight":
                if not direct_actual:
                    failures[f"{target_class}:direct_not_safe"] += 1
                    progress.last_event = "rechazo: línea recta insegura"
                    continue
                raw = [start, goal]
                progress.update(attempts, target_class, len(paths), per_class, "ruta directa (sin Theta*)")
            else:
                if direct_actual:
                    failures[f"{target_class}:direct_line_clear"] += 1
                    progress.last_event = "rechazo: no requiere maniobra"
                    continue
                progress.update(attempts, target_class, len(paths), per_class, "planificando Theta*")
                raw = theta_star(plan_ctx, start_rc, goal_rc, args.hard_clearance_m)
                if raw is None:
                    failures[f"{target_class}:theta_star_no_path"] += 1
                    progress.last_event = "Theta* sin ruta"
                    continue
                # Lock endpoints to the sampled coarse-cell centers.
                raw[0], raw[-1] = start, goal
        euclidean = math.dist(start, goal)
        max_len = args.v_max_path_length_m if target_class == "v_complex" else args.max_path_length_m
        if euclidean < args.min_goal_distance_m or euclidean > max_goal:
            failures[f"{target_class}:goal_distance"] += 1
            progress.last_event = "distancia inicio-meta fuera de límite"
            continue
        progress.update(attempts, target_class, len(paths), per_class, "suavizado + spline")
        try:
            points, smooth_meta = smooth_theta_route(ctx, raw, args.hard_clearance_m,
                                                     args.max_length_ratio)
        except RuntimeError:
            failures[f"{target_class}:unsafe_theta_star"] += 1
            progress.last_event = "suavizado inseguro"
            continue
        progress.update(attempts, target_class, len(paths), per_class, "validando seguridad")
        metrics = route_validation(ctx, points, args.hard_clearance_m)
        if not metrics["safe"]:
            failures[f"{target_class}:final_safety"] += 1
            progress.last_event = "rechazo: colisión/clearance"
            continue
        if metrics["path_length_m"] > max_len:
            failures[f"{target_class}:path_length"] += 1
            progress.last_event = "rechazo: longitud máxima"
            continue
        if metrics["path_length_m"] + 1e-5 < euclidean:
            failures[f"{target_class}:length_inconsistent"] += 1
            progress.last_event = "rechazo: longitud inconsistente"
            continue
        progress.update(attempts, target_class, len(paths), per_class, "clasificando maniobra")
        direct_ok = line_is_safe(ctx, start, goal, args.hard_clearance_m)
        geom = curvature_metrics(points)
        supported = route_supported_turn_count(ctx, points, args.obstacle_influence_m)
        geom["supported_turn_count"] = supported
        detected_class = geometry_class(geom, direct_ok, supported)
        if detected_class != target_class:
            failures[f"{target_class}:class_mismatch_{detected_class or 'unclassified'}"] += 1
            progress.last_event = f"clase no coincide ({detected_class or 'ninguna'})"
            continue
        if duplicate_candidate(points, start, goal, paths, target_class):
            failures[f"{target_class}:duplicate_endpoints"] += 1
            progress.last_event = "ruta duplicada"
            continue
        yaw_start = rng.uniform(-math.pi, math.pi)
        yaw_goal = rng.uniform(-math.pi, math.pi)
        task = route_task(target_class, direct_ok, {**metrics, **geom}, args.hard_clearance_m)
        if task is None:
            failures[f"{target_class}:task_condition_not_met"] += 1
            progress.last_event = "condición de tarea no satisfecha"
            continue
        rec = {"id": len(paths), "class": target_class, "purposeful_class": target_class,
               "task_type": task, "start": [round(x, 4) for x in start],
               "goal": [round(x, 4) for x in goal], "start_yaw_rad": round(yaw_start, 5),
               "goal_yaw_rad": round(yaw_goal, 5), "poses": np.round(points, 4).tolist(),
               "metrics": {**{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                             for k, v in metrics.items()},
                           **{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                              for k, v in geom.items()},
                           **{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                              for k, v in smooth_meta.items() if k not in ("rejections",)}},
               "direct_line_safe": bool(direct_ok),
               "orientation_policy": "independent_uniform_yaw",
               "planner": "theta_star", "smoother": "anchored_smooth+cubic_spline",
               "search": search_info}
        paths.append(rec)
        per_class[target_class] += 1
        progress.last_event = f"ACEPTADA {task}"
        progress.update(attempts, target_class, len(paths), per_class, "ruta aceptada")

    failures["attempts_total"] = attempts
    elapsed = time.monotonic() - started
    failures["elapsed_seconds"] = round(elapsed, 2)
    timeout_hit = bool(args.timeout_seconds is not None and args.timeout_seconds > 0
                       and elapsed >= args.timeout_seconds)
    max_attempts_hit = bool(args.max_attempts is not None and attempts >= args.max_attempts)
    failures["timeout_hit"] = int(timeout_hit)
    failures["max_attempts_hit"] = int(max_attempts_hit)
    missing_counts = [k for k in CLASSES if per_class[k] < args.target_counts[k]]
    if not missing_counts:
        end_stage = "cuotas completas"
    elif timeout_hit and max_attempts_hit:
        end_stage = "timeout e intentos máximos; cuotas pendientes=" + ",".join(missing_counts)
    elif timeout_hit:
        end_stage = "timeout; cuotas pendientes=" + ",".join(missing_counts)
    elif max_attempts_hit:
        end_stage = "límite de intentos; cuotas pendientes=" + ",".join(missing_counts)
    else:
        end_stage = "generación detenida; cuotas pendientes=" + ",".join(missing_counts)
    progress.finish(attempts, target_class, len(paths), per_class, end_stage)
    return paths, failures


def write_outputs(ctx: dict[str, Any], args: argparse.Namespace,
                  paths: list[dict[str, Any]], failures: Counter, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    poses = {}
    csv_rows = []
    for p in paths:
        q = dict(p)
        poses[str(p["id"])] = p["poses"]
        del q["poses"]
        records.append(q)
        row = {"id": p["id"], "class": p["class"], "task_type": p["task_type"],
               "start_x": p["start"][0], "start_y": p["start"][1],
               "goal_x": p["goal"][0], "goal_y": p["goal"][1],
               "start_yaw_rad": p["start_yaw_rad"], "goal_yaw_rad": p["goal_yaw_rad"],
               "direct_line_safe": p["direct_line_safe"], "smoother": p["smoother"],
               "search_strategy": p.get("search", {}).get("search_strategy", "random_pair"),
               "goal_candidates_evaluated": p.get("search", {}).get("goal_candidates_evaluated", 0),
               "theta_candidate_routes_evaluated": p.get("search", {}).get("theta_candidate_routes_evaluated", 0),
               "direct_obstacle_bands": p.get("search", {}).get("direct_obstacle_bands", 0)}
        row.update(p["metrics"])
        csv_rows.append(row)
    counts = Counter(p["class"] for p in paths)
    task_counts = Counter(p["task_type"] for p in paths)
    quota = {c: int(args.target_counts[c]) for c in CLASSES}
    pool = {"pool_version": 5, "scenario": args.scenario,
            "generated_at": datetime.now(UTC).isoformat(), "seed": args.seed,
            "num_paths": len(paths), "source_num_paths": len(paths),
            "constraints": {"min_goal_distance_m": args.min_goal_distance_m,
                            "max_goal_distance_m": args.max_goal_distance_m,
                            "v_max_goal_distance_m": args.v_max_goal_distance_m,
                            "max_path_length_m": args.max_path_length_m,
                            "v_max_path_length_m": args.v_max_path_length_m,
                            "hard_clearance_m": args.hard_clearance_m,
                            "max_length_ratio_vs_theta_star": args.max_length_ratio,
                            "v_goal_candidates": args.v_goal_candidates,
                            "v_plan_candidates": args.v_plan_candidates,
                            "v_hotspot_probability": args.v_hotspot_probability,
                            "v_hotspot_radius_m": args.v_hotspot_radius_m},
            "map_yaml": ctx["yaml"], "classification_scope": "navigation_task_and_geometry",
            "planner": "Theta*", "smoothing_pipeline": ["arc_length_resample", "anchored_laplacian_smoothing", "cubic_spline", "dense_clearance_validation"],
            "quota": quota, "distribution": dict(counts), "task_distribution": dict(task_counts),
            "generation_complete": all(counts[c] >= quota[c] for c in CLASSES),
            "paths": records}
    (run_dir / "path_pool.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
    (run_dir / "path_pool_poses.json").write_text(json.dumps(poses, indent=1), encoding="utf-8")
    fieldnames = sorted({k for row in csv_rows for k in row})
    with (run_dir / "path_pool.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    report = {"scenario": args.scenario, "map_yaml": ctx["yaml"], "seed": args.seed,
              "attempt_budget": args.max_attempts, "attempts_used": failures.get("attempts_total", 0),
              "max_attempts_hit": bool(failures.get("max_attempts_hit", 0)),
              "timeout_seconds": args.timeout_seconds, "timeout_hit": bool(failures.get("timeout_hit", 0)),
              "stop_reason": ("complete" if all(counts[c] >= quota[c] for c in CLASSES)
                              else "timeout_and_max_attempts" if failures.get("timeout_hit", 0) and failures.get("max_attempts_hit", 0)
                              else "timeout" if failures.get("timeout_hit", 0)
                              else "max_attempts" if failures.get("max_attempts_hit", 0)
                              else "stopped_incomplete"),
              "accepted_paths": len(paths), "target_counts": quota,
              "target_total": sum(quota.values()),
              "target_per_class": next(iter(set(quota.values()))) if len(set(quota.values())) == 1 else None,
              "class_counts": dict(counts),
              "class_deficits": {c: max(0, quota[c] - counts.get(c, 0)) for c in CLASSES},
              "task_counts": dict(task_counts),
              "complete": all(counts[c] >= quota[c] for c in CLASSES),
              "clearance_policy_m": args.hard_clearance_m,
              "failure_counts": {k: v for k, v in failures.items()
                                 if k not in ("attempts_total", "elapsed_seconds", "timeout_hit", "max_attempts_hit") and not k.startswith("_diag_")},
              "search_diagnostics": {k.removeprefix("_diag_"): v for k, v in failures.items()
                                     if k.startswith("_diag_")},
              "elapsed_seconds": failures.get("elapsed_seconds", 0),
              "notes": ["Offline dataset generator; no robot commands are sent.",
                        "Initial and goal yaw are sampled independently and stored in radians.",
                        "v_complex uses multi-goal local search around obstacle-supported bends from accepted complex routes when available.",
                        "Inspect per-route plots and narrow passages before downstream use."]}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(report, pool), encoding="utf-8")
    if plt is not None:
        render_class_sheets(ctx, paths, run_dir)
    else:
        print("WARN: matplotlib unavailable; route plots not written.")


def render_report(report: dict[str, Any], pool: dict[str, Any]) -> str:
    timeout_s = report.get("timeout_seconds")
    timeout_label = "disabled" if timeout_s is None or timeout_s <= 0 else f"{timeout_s:.1f} s"
    attempt_limit = report.get("attempt_budget")
    attempt_label = "unlimited" if attempt_limit is None else str(attempt_limit)
    lines = [f"# Path generator report — {report['scenario']}", "",
             f"- Seed: `{report['seed']}`", f"- Accepted: **{report['accepted_paths']}**",
             f"- Attempts used: {report['attempts_used']}",
             f"- Attempt limit: {attempt_label}; reached: {report.get('max_attempts_hit', False)}",
             f"- Time limit: {timeout_label}; timeout hit: {report.get('timeout_hit', False)}",
             f"- Stop reason: **{report.get('stop_reason', 'unknown')}**",
             f"- Complete: **{report['complete']}**", f"- Hard clearance: {report['clearance_policy_m']:.3f} m", "",
             "## Geometry distribution", "", "| Class | Count | Target | Deficit |", "|---|---:|---:|---:|"]
    targets = report.get("target_counts", {})
    for name in CLASSES:
        count = report["class_counts"].get(name, 0)
        target = targets.get(name, 0)
        lines.append(f"| {name} | {count} | {target} | {max(0, target-count)} |")
    lines += ["", "## Situation distribution", "", "| Task | Count |", "|---|---:|"]
    for name, count in sorted(report["task_counts"].items()):
        lines.append(f"| {name} | {count} |")
    lines += ["", "## Rejections", "", "| Reason | Count |", "|---|---:|"]
    for name, count in sorted(report["failure_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} |")
    lines += ["", "## Files", "", "- `path_pool.json`: pool metadata and route attributes.",
              "- `path_pool_poses.json`: route coordinates keyed by ID.",
              "- `path_pool.csv`: flat metrics table.",
              "- `routes_<class>.png`: one route per panel; not an overlay of every route."]
    diagnostics = report.get("search_diagnostics", {})
    if diagnostics:
        lines += ["", "## V-complex search diagnostics", "", "| Counter | Value |", "|---|---:|"]
        for name, value in sorted(diagnostics.items()):
            lines.append(f"| {name} | {value} |")
    return "\n".join(lines) + "\n"


def render_class_sheets(ctx: dict[str, Any], paths: list[dict[str, Any]], out_dir: Path) -> None:
    display = np.asarray(ctx["pixels"])
    extent = (ctx["ox"], ctx["ox"] + ctx["width"]*ctx["res"],
              ctx["oy"], ctx["oy"] + ctx["height"]*ctx["res"])
    for cls in CLASSES:
        group = [p for p in paths if p["class"] == cls]
        if not group:
            continue
        ncols = 2
        nrows = int(math.ceil(len(group) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.7*nrows), dpi=130, squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, p in zip(axes.flat, group):
            pts = np.asarray(p["poses"], dtype=float)
            ax.set_visible(True)
            ax.imshow(display, cmap="gray", origin="lower", interpolation="nearest", extent=extent, vmin=0, vmax=255)
            ax.plot(pts[:, 0], pts[:, 1], color=COLORS[cls], lw=2.1)
            ax.scatter(pts[0, 0], pts[0, 1], marker="o", c="#168b67", edgecolor="black", s=28, zorder=4)
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="x", c="black", s=35, zorder=4)
            pad = max(0.45, 2.0 * ctx["res"])
            ax.set_xlim(max(extent[0], pts[:, 0].min()-pad), min(extent[1], pts[:, 0].max()+pad))
            ax.set_ylim(max(extent[2], pts[:, 1].min()-pad), min(extent[3], pts[:, 1].max()+pad))
            ax.set_aspect("equal", adjustable="box")
            m = p["metrics"]
            ax.set_title(f"ID {p['id']} · L={m['path_length_m']:.2f} m · turns={m['turn_event_count']} · clr={m['min_clearance_m']:.2f} m\n{p['task_type']} · {m.get('smoothing_status','?')}", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"{ctx['meta'].get('image', 'map')} — {cls}: rutas individuales", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out_dir / f"routes_{cls}.png", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", required=True, help="Map prefix, e.g. E01, E02, E03")
    p.add_argument("--maps-root", default=DEFAULT_MAPS)
    p.add_argument("--output-dir", default=DEFAULT_DATA,
                   help="Dataset root; output is placed in <scenario>/path_generator_seed_<seed>/")
    p.add_argument("--run-name", default="", help="Optional explicit new run-folder name")
    p.add_argument("--n-goals", "--num-paths", dest="n_goals", type=int, default=None,
                   help="Total target routes; split evenly across four classes (default total: 200)")
    p.add_argument("--st_sem_cmp_vc", "--st-sem-cmp-vc", dest="st_sem_cmp_vc", default=None,
                   metavar="STRAIGHT_SEMI_COMPLEX_COMPLEX_V_COMPLEX",
                   help="Explicit class quotas separated by underscores, e.g. 10_10_15_20")
    p.add_argument("--balance-class", action="store_true",
                   help="Explicitly select balanced classes; automatic unless quotas are supplied")
    p.add_argument("--per-class", type=int, default=None, help="Legacy option: same target for all classes")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Optional global attempt limit; omitted means unlimited")
    p.add_argument("--progress-interval-seconds", type=float, default=5.0,
                   help="Update one-line progress at this wall-time interval")
    p.add_argument("--timeout-seconds", type=float, default=None,
                   help="Optional wall-time limit; omitted or 0 disables timeout")
    p.add_argument("--seed", type=int, default=462326534)
    p.add_argument("--hard-clearance-m", type=float, default=0.35)
    p.add_argument("--endpoint-obstacle-band-m", type=float, default=0.65)
    p.add_argument("--obstacle-influence-m", type=float, default=0.80)
    p.add_argument("--min-goal-distance-m", type=float, default=2.0)
    p.add_argument("--max-goal-distance-m", type=float, default=3.0)
    p.add_argument("--max-path-length-m", type=float, default=3.0)
    p.add_argument("--v-max-goal-distance-m", type=float, default=3.6)
    p.add_argument("--v-max-path-length-m", type=float, default=4.2)
    p.add_argument("--max-length-ratio", type=float, default=1.18)
    p.add_argument("--v-goal-candidates", type=int, default=10)
    p.add_argument("--v-plan-candidates", type=int, default=3)
    p.add_argument("--v-hotspot-probability", type=float, default=0.80)
    p.add_argument("--v-hotspot-radius-m", type=float, default=0.35)
    p.add_argument("--plan-cell-factor", type=int, default=5)
    p.add_argument("--overwrite-run", action="store_true")
    return p.parse_args()


def configure_targets(args: argparse.Namespace) -> dict[str, int]:
    """Resolve the single source of truth for per-class targets."""
    if args.st_sem_cmp_vc is not None:
        if args.n_goals is not None or args.per_class is not None or args.balance_class:
            raise ValueError("--st_sem_cmp_vc cannot be combined with --n-goals, --per-class, or --balance-class")
        parts = args.st_sem_cmp_vc.split("_")
        if len(parts) != 4:
            raise ValueError("--st_sem_cmp_vc requires four quotas: straight_semi_complex_complex_v_complex")
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError("--st_sem_cmp_vc quotas must be integers") from exc
        if any(value < 1 for value in values):
            raise ValueError("--st_sem_cmp_vc quotas must all be positive")
        targets = dict(zip(CLASSES, values))
    elif args.per_class is not None:
        if args.n_goals is not None or args.balance_class:
            raise ValueError("--per-class cannot be combined with --n-goals or --balance-class")
        if args.per_class < 1:
            raise ValueError("--per-class must be positive")
        targets = {name: args.per_class for name in CLASSES}
    else:
        total = 200 if args.n_goals is None else args.n_goals
        if total < 4:
            raise ValueError("--n-goals must be at least 4")
        base, remainder = divmod(total, len(CLASSES))
        targets = {name: int(base + (index < remainder)) for index, name in enumerate(CLASSES)}
    args.target_counts = targets
    args.target_total = sum(targets.values())
    return targets

def main() -> int:
    args = parse_args()
    try:
        target_counts = configure_targets(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.max_attempts is not None and args.max_attempts < 1:
        print("ERROR: --max-attempts must be positive when supplied", file=sys.stderr)
        return 2
    if (args.timeout_seconds is not None and args.timeout_seconds < 0) or args.progress_interval_seconds <= 0:
        print("ERROR: timeout must be >= 0 and progress interval must be > 0", file=sys.stderr)
        return 2
    if args.hard_clearance_m <= 0 or args.max_length_ratio < 1.0:
        print("ERROR: hard clearance must be positive and length ratio >= 1", file=sys.stderr)
        return 2
    if args.plan_cell_factor < 1 or args.plan_cell_factor > 20:
        print("ERROR: --plan-cell-factor must be between 1 and 20", file=sys.stderr)
        return 2
    if args.v_goal_candidates < 1 or args.v_plan_candidates < 1:
        print("ERROR: v_complex goal/plan candidate counts must be positive", file=sys.stderr)
        return 2
    if not (0.0 <= args.v_hotspot_probability <= 1.0) or args.v_hotspot_radius_m < 0:
        print("ERROR: invalid v_complex hotspot probability/radius", file=sys.stderr)
        return 2
    if not (0 < args.min_goal_distance_m <= args.max_goal_distance_m <= args.v_max_goal_distance_m):
        print("ERROR: inconsistent goal distance limits", file=sys.stderr)
        return 2
    maps_root = Path(args.maps_root).expanduser().resolve()
    try:
        folder = map_folder(maps_root, args.scenario)
        yaml_path = folder / f"map_{folder.name}_prohibition.yaml"
        if not yaml_path.exists():
            # Fall back to the one prohibition YAML in the selected map folder.
            options = sorted(folder.glob("*prohibition*.yaml"))
            if len(options) != 1:
                raise FileNotFoundError(f"Could not resolve a unique prohibition YAML in {folder}")
            yaml_path = options[0]
        ctx = load_map(yaml_path)
    except Exception as exc:
        print(f"ERROR loading map: {exc}", file=sys.stderr)
        return 2
    eligible = ctx["free"] & (ctx["clearance"] >= args.hard_clearance_m)
    n = int(eligible.sum())
    if n < 100:
        print(f"ERROR: only {n} cells satisfy free-space and hard-clearance constraints; check map polarity/margin", file=sys.stderr)
        return 2
    run_name = args.run_name.strip() or f"path_generator_seed_{args.seed}"
    run_dir = Path(args.output_dir).expanduser().resolve() / args.scenario / run_name
    if run_dir.exists() and not args.overwrite_run:
        print(f"ERROR: output directory exists; refusing to overwrite: {run_dir}\nUse a new --run-name or explicitly pass --overwrite-run.", file=sys.stderr)
        return 2
    if run_dir.exists() and args.overwrite_run:
        import shutil
        shutil.rmtree(run_dir)
    print(f"scenario: {args.scenario}\nmap: {yaml_path}\nmap size: {ctx['width']}x{ctx['height']} at {ctx['res']:.3f} m/cell")
    print(f"safe cells: {n}; clearance >= {args.hard_clearance_m:.2f} m")
    print("target quotas: " + ", ".join(f"{cls}={target_counts[cls]}" for cls in CLASSES)
          + f" (total={sum(target_counts.values())})")
    print("limits: " +
          (f"max_attempts={args.max_attempts}" if args.max_attempts is not None else "max_attempts=unlimited") + ", " +
          (f"timeout_seconds={args.timeout_seconds:g}" if args.timeout_seconds is not None and args.timeout_seconds > 0 else "timeout=disabled"))
    paths, failures = generate(ctx, args)
    # Ensure every ID is deterministic and compact after generation.
    for i, pth in enumerate(paths):
        pth["id"] = i
    write_outputs(ctx, args, paths, failures, run_dir)
    counts = Counter(pth["class"] for pth in paths)
    print(f"wrote: {run_dir}")
    print(f"accepted: {len(paths)} / target {args.target_total}; classes={dict(counts)}")
    print(f"attempts={failures.get('attempts_total', 0)} elapsed={failures.get('elapsed_seconds', 0)}s")
    if plt is not None:
        print("individual route sheets:")
        for cls in CLASSES:
            target = run_dir / f"routes_{cls}.png"
            if target.exists():
                print(f"  {target}")
    return 0 if all(counts[c] >= target_counts[c] for c in CLASSES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
