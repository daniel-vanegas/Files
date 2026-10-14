#!/usr/bin/env python3
"""Generate, upgrade and analyze per-scenario path pools.

Writes <output_dir>/<scenario>/path_pool.json and path_pool_poses.json.
"""

import argparse
import csv
import heapq
import json
import math
import os
import random
import sys
import subprocess
import time
from collections import Counter
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

try:
    from skimage.graph import route_through_array
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


FEATURE_NAMES = [
    'sinuosity',
    'max_turn_deg',
    'total_turn_deg',
    'num_turns_30',
    'num_turns_60',
    'turn_sign_changes',
    'min_clearance_m',
    'mean_clearance_m',
    'path_length_m',
]
GEOMETRY_FEATURE_NAMES = [
    'sinuosity', 'max_turn_deg', 'total_turn_deg',
    'num_turns_30', 'num_turns_60', 'turn_sign_changes',
]
POOL_VERSION = 4
CLASSIFIER_NAME = 'geometry_morphology_v2'
CLASS_COLORS = {
    'straight': '#2878B5',
    'semi_complex': '#4A9D62',
    'complex': '#E39C37',
    'v_complex': '#C84D4D',
}

CLASS_NAMES = ['straight', 'semi_complex', 'complex', 'v_complex']


def resolve_default_maps_root():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('justina_description'),
            'webots_env', 'environments', 'maps')
    except Exception:
        return os.path.join(
            'src', 'Hardware', 'justina_description',
            'webots_env', 'environments', 'maps')


def find_map_folder(maps_root, scenario):
    prefix = f'{scenario}_'
    candidates = sorted(
        name for name in os.listdir(maps_root)
        if name.startswith(prefix)
        and os.path.isdir(os.path.join(maps_root, name)))
    if not candidates:
        raise RuntimeError(f'no folder starting with {prefix} in {maps_root}')
    return os.path.join(maps_root, candidates[0])


def load_map(map_yaml):
    with open(map_yaml, 'r') as f:
        meta = yaml.safe_load(f)
    image_path = os.path.join(os.path.dirname(map_yaml), meta['image'])
    img = Image.open(image_path).convert('L')
    arr_top_down = np.array(img, dtype=np.uint8)
    arr = np.flipud(arr_top_down)
    resolution = float(meta['resolution'])
    origin = meta['origin']
    ox, oy = float(origin[0]), float(origin[1])
    free_thresh = float(meta.get('free_thresh', 0.196))
    occupied_thresh = float(meta.get('occupied_thresh', 0.65))
    negate = int(meta.get('negate', 0))
    return arr, resolution, ox, oy, free_thresh, occupied_thresh, negate


def build_free_mask(map_arr, free_thresh, occupied_thresh, negate):
    if negate:
        occupancy = map_arr.astype(np.float64) / 255.0
    else:
        occupancy = (255.0 - map_arr.astype(np.float64)) / 255.0
    return occupancy < free_thresh


def smooth_path(points, weight_data=0.1, weight_smooth=0.9, tolerance=1e-6):
    if len(points) < 3:
        return [list(p) for p in points]
    new_path = [list(p) for p in points]
    tol = tolerance * len(points)
    change = tol + 1.0
    attempts = 0
    while change >= tol and attempts < 1000:
        change = 0.0
        for i in range(1, len(points) - 1):
            old_p = points[i]
            new_p = new_path[i]
            next_p = new_path[i + 1]
            prev_p = new_path[i - 1]
            last_x, last_y = new_p
            new_p[0] += weight_data * (old_p[0] - new_p[0]) + \
                weight_smooth * (next_p[0] + prev_p[0] - 2.0 * new_p[0])
            new_p[1] += weight_data * (old_p[1] - new_p[1]) + \
                weight_smooth * (next_p[1] + prev_p[1] - 2.0 * new_p[1])
            change += abs(new_p[0] - last_x) + abs(new_p[1] - last_y)
        attempts += 1
    return new_path


def chaikin_smooth(points, iterations=4):
    if len(points) < 3:
        return points
    current = [list(p) for p in points]
    for _ in range(iterations):
        if len(current) < 3:
            break
        nxt = [current[0]]
        for i in range(len(current) - 1):
            p0 = current[i]
            p1 = current[i + 1]
            a = [0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]]
            b = [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]
            nxt.append(a)
            nxt.append(b)
        nxt.append(current[-1])
        current = nxt
    return current


def downsample(points, min_dist):
    if len(points) < 2:
        return points
    out = [points[0]]
    for i in range(1, len(points)):
        p = points[i]
        q = out[-1]
        if math.hypot(p[0] - q[0], p[1] - q[1]) >= min_dist:
            out.append(p)
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def world_to_grid(x, y, ox, oy, res):
    col = int((x - ox) / res)
    row = int((y - oy) / res)
    return row, col


def grid_to_world(row, col, ox, oy, res):
    x = ox + (col + 0.5) * res
    y = oy + (row + 0.5) * res
    return x, y


def astar_grid(free, start, goal):
    sr, sc = start
    gr, gc = goal
    if not free[sr, sc] or not free[gr, gc]:
        return None
    h, w = free.shape
    open_list = [(0.0, 0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    closed = set()
    while open_list:
        _, _, r, c = heapq.heappop(open_list)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            return list(reversed(path))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not free[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    if not free[r + dr, c] or not free[r, c + dc]:
                        continue
                if (nr, nc) in closed:
                    continue
                step = 1.4142135 if dr != 0 and dc != 0 else 1.0
                ng = g_score[(r, c)] + step
                if ng < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = (r, c)
                    g_score[(nr, nc)] = ng
                    hv = math.hypot(nr - gr, nc - gc)
                    heapq.heappush(open_list, (ng + hv, ng, nr, nc))
    return None


def astar_cropped(free, sr, sc, gr, gc, margin_cells=300):
    r0 = max(0, min(sr, gr) - margin_cells)
    r1 = min(free.shape[0], max(sr, gr) + margin_cells + 1)
    c0 = max(0, min(sc, gc) - margin_cells)
    c1 = min(free.shape[1], max(sc, gc) + margin_cells + 1)
    sub = free[r0:r1, c0:c1]
    path = astar_grid(sub, (sr - r0, sc - c0), (gr - r0, gc - c0))
    if path is None:
        return None
    return [(r + r0, c + c0) for r, c in path]


def plan_path_stages(free, start_xy, goal_xy, ox, oy, res):
    """Return raw grid A*, smoothed grid path, and final decimated path."""
    sr, sc = world_to_grid(start_xy[0], start_xy[1], ox, oy, res)
    gr, gc = world_to_grid(goal_xy[0], goal_xy[1], ox, oy, res)
    h, w = free.shape
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if not free[sr, sc] or not free[gr, gc]:
        return None

    path_rc = None
    if HAS_SKIMAGE:
        margin = 300
        r0 = max(0, min(sr, gr) - margin)
        r1 = min(h, max(sr, gr) + margin + 1)
        c0 = max(0, min(sc, gc) - margin)
        c1 = min(w, max(sc, gc) + margin + 1)
        sub = free[r0:r1, c0:c1]
        costs = np.where(sub, 1.0, 10000.0).astype(np.float64)
        try:
            idx, _ = route_through_array(
                costs, (sr - r0, sc - c0), (gr - r0, gc - c0),
                fully_connected=False, geometric=True)
            path_rc = [(r + r0, c + c0) for r, c in idx]
        except Exception:
            path_rc = None

    if path_rc is None:
        path_rc = astar_cropped(free, sr, sc, gr, gc, margin_cells=400)
    if not path_rc:
        return None
    if any(not free[r, c] for r, c in path_rc):
        return None

    raw = [list(grid_to_world(r, c, ox, oy, res)) for r, c in path_rc]
    smoothed = smooth_path(raw, 0.1, 0.9, 1e-6)
    decimated = downsample(smoothed, 0.15)
    final = chaikin_smooth(decimated, 2)
    return {'raw': raw, 'smoothed': smoothed, 'final': final}


def plan_path(free, start_xy, goal_xy, ox, oy, res):
    """Compatibility wrapper: preserve the original final-path behavior."""
    stages = plan_path_stages(free, start_xy, goal_xy, ox, oy, res)
    return None if stages is None else stages['final']


def resample_polyline(points, spacing_m=0.05):
    """Resample a polyline at nearly uniform arc-length intervals."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return pts[:1]
    samples = list(np.arange(0.0, total, spacing_m))
    if not samples or total - samples[-1] > 1e-6:
        samples.append(total)
    out = []
    j = 0
    for distance in samples:
        while j < len(seg) - 1 and cumulative[j + 1] < distance:
            j += 1
        fraction = (distance - cumulative[j]) / max(seg[j], 1e-9)
        point = pts[j] + fraction * (pts[j + 1] - pts[j])
        out.append(point.tolist())
    return np.asarray(out, dtype=np.float64)


def compute_geometry_features(path_xy):
    """Compute scale-aware geometry features from uniformly sampled points.

    Turning is measured from tangents separated by 0.25 m, not from every
    arbitrary source vertex, so point density has less influence on labels.
    """
    pts = np.asarray(path_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None
    raw_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    raw_lengths = raw_lengths[raw_lengths > 1e-8]
    if len(raw_lengths) == 0:
        return None
    path_length = float(raw_lengths.sum())
    euclidean = float(np.linalg.norm(pts[-1] - pts[0]))
    if euclidean < 1e-5:
        return None

    step = 0.05
    sampled = resample_polyline(pts, step)
    if len(sampled) < 3:
        return {
            'sinuosity': path_length / euclidean,
            'max_turn_deg': 0.0,
            'total_turn_deg': 0.0,
            'num_turns_30': 0.0,
            'num_turns_60': 0.0,
            'turn_sign_changes': 0.0,
            'path_length_m': path_length,
        }

    distance_window = 0.25
    half_window = max(1, int(round(distance_window / (2.0 * step))))
    headings = []
    valid_positions = []
    for i in range(len(sampled)):
        lo = max(0, i - half_window)
        hi = min(len(sampled) - 1, i + half_window)
        delta = sampled[hi] - sampled[lo]
        if np.linalg.norm(delta) < 1e-9:
            continue
        headings.append(math.atan2(float(delta[1]), float(delta[0])))
        valid_positions.append(i)

    signed_turns = []
    for previous, current in zip(headings[:-1], headings[1:]):
        delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
        deg = math.degrees(delta)
        if abs(deg) >= 1.0:
            signed_turns.append(deg)
        else:
            signed_turns.append(0.0)

    abs_turns = [abs(v) for v in signed_turns]
    total_turn = float(sum(abs_turns))
    window_steps = max(1, int(round(0.30 / step)))
    local_turns = []
    for i in range(len(abs_turns)):
        local_turns.append(float(sum(abs_turns[i:i + window_steps])))
    max_turn = max(local_turns, default=0.0)

    # Count disjoint bend windows rather than counting the same bend repeatedly.
    num_turns_30 = 0
    num_turns_60 = 0
    for i in range(0, len(abs_turns), window_steps):
        window_turn = sum(abs_turns[i:i + window_steps])
        if window_turn >= 30.0:
            num_turns_30 += 1
        if window_turn >= 60.0:
            num_turns_60 += 1

    significant_signs = [1 if v >= 3.0 else -1 if v <= -3.0 else 0 for v in signed_turns]
    significant_signs = [v for v in significant_signs if v]
    sign_changes = sum(
        1 for a, b in zip(significant_signs[:-1], significant_signs[1:]) if a != b)

    return {
        'sinuosity': path_length / euclidean,
        'max_turn_deg': max_turn,
        'total_turn_deg': total_turn,
        'num_turns_30': float(num_turns_30),
        'num_turns_60': float(num_turns_60),
        'turn_sign_changes': float(sign_changes),
        'path_length_m': path_length,
    }


def compute_features(path_xy, clearance_grid, ox, oy, res):
    geometry = compute_geometry_features(path_xy)
    if geometry is None:
        return None
    points = np.asarray(path_xy, dtype=np.float64)
    height, width = clearance_grid.shape
    clearances = []
    # Use denser points so a smoothed segment cannot skip a narrow obstacle.
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        count = max(1, int(math.ceil(length / max(min(res * 0.5, 0.01), 0.002))))
        for j in range(count + 1):
            t = j / count
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            col = int((x - ox) / res)
            row = int((y - oy) / res)
            if not (0 <= row < height and 0 <= col < width):
                return None
            clearances.append(float(clearance_grid[row, col]))
    if not clearances:
        return None
    geometry['min_clearance_m'] = float(min(clearances))
    geometry['mean_clearance_m'] = float(sum(clearances) / len(clearances))
    return geometry


def build_rooms(free, res, ox, oy, room_radius_m, min_area_m2):
    dist_px = distance_transform_edt(free)
    min_dist_px = room_radius_m / res
    traversable = dist_px >= min_dist_px
    labels, n = label(traversable, structure=np.ones((3, 3)))
    min_cells = max(1, int(min_area_m2 / (res * res)))
    sizes = np.bincount(labels.ravel())
    valid = {i for i in range(1, n + 1) if sizes[i] >= min_cells}
    rooms = {}
    ys, xs = np.where(traversable)
    for row, col in zip(ys, xs):
        rid = int(labels[row, col])
        if rid not in valid:
            continue
        mx = ox + (col + 0.5) * res
        my = oy + (row + 0.5) * res
        rooms.setdefault(rid, []).append((mx, my))
    return rooms, dist_px


def geometry_complexity_score(features):
    """Ordered geometric severity score; safety and path length are excluded."""
    sinuosity = max(0.0, float(features.get('sinuosity', 1.0)) - 1.0)
    total_turn = max(0.0, float(features.get('total_turn_deg', 0.0)))
    max_turn = max(0.0, float(features.get('max_turn_deg', 0.0)))
    turns30 = max(0.0, float(features.get('num_turns_30', 0.0)))
    turns60 = max(0.0, float(features.get('num_turns_60', 0.0)))
    inflections = max(0.0, float(features.get('turn_sign_changes', 0.0)))

    # Normalize to interpretable ranges and clamp outliers.
    total_norm = min(total_turn / 180.0, 1.5)
    local_norm = min(max_turn / 75.0, 1.5)
    sinuosity_norm = min(sinuosity / 0.40, 1.5)
    bends_norm = min(turns30 / 4.0, 1.5)
    strong_bends_norm = min(turns60 / 2.0, 1.5)
    inflection_norm = min(inflections / 3.0, 1.5)

    # Cumulative turning and sharp bends rank above a single shallow arc.
    return float(
        0.34 * total_norm
        + 0.26 * local_norm
        + 0.18 * sinuosity_norm
        + 0.10 * bends_norm
        + 0.07 * strong_bends_norm
        + 0.05 * inflection_norm
    )


def classify_one_geometry(features):
    """Classify morphology, keeping broad smooth arcs below sharp/multi-turn paths."""
    sinuosity = float(features.get('sinuosity', 1.0))
    max_turn = float(features.get('max_turn_deg', 0.0))
    total_turn = float(features.get('total_turn_deg', 0.0))
    turns30 = float(features.get('num_turns_30', 0.0))
    turns60 = float(features.get('num_turns_60', 0.0))
    sign_changes = float(features.get('turn_sign_changes', 0.0))

    # V-complex routes have large cumulative turning, like a U-shaped return,
    # or repeated substantial bends. A single sharp corner is not sufficient.
    if (total_turn >= 110.0 and
            (sinuosity >= 1.04 or turns30 >= 2.0 or sign_changes >= 2.0)):
        return 'v_complex'

    # Complex routes have a localized strong bend or multiple distinct bends.
    # Do not use total turning alone here: a broad, smooth arc belongs to semi.
    if (
        max_turn >= 40.0
        or turns60 >= 1.0
        or turns30 >= 2.0
        or (total_turn >= 70.0 and max_turn >= 25.0)
        or (total_turn >= 55.0 and sign_changes >= 2.0)
    ):
        return 'complex'

    # Semi-complex paths are gently curved arcs without a localized sharp bend.
    if total_turn >= 25.0 or sinuosity >= 1.025 or max_turn >= 20.0:
        return 'semi_complex'

    return 'straight'


def classify_by_geometry(features_list, quota=None):
    """Apply fixed geometry rules; no PCA and no per-pool quantile bins."""
    if quota is None:
        quota = [25, 25, 25, 25]
    if len(quota) != 4 or sum(quota) <= 0 or any(float(x) < 0 for x in quota):
        raise ValueError('quota must contain four non-negative values with positive sum')

    labels = [classify_one_geometry(features) for features in features_list]
    scores = [geometry_complexity_score(features) for features in features_list]
    signatures = {}
    for class_name in CLASS_NAMES:
        indices = [i for i, label_name in enumerate(labels) if label_name == class_name]
        subset = [features_list[i] for i in indices]
        class_scores = [scores[i] for i in indices]
        signature = {
            'n': len(subset),
            'complexity_score': (
                [float(min(class_scores)), float(max(class_scores))]
                if class_scores else [None, None]),
        }
        if subset:
            for name in GEOMETRY_FEATURE_NAMES:
                vals = [float(f.get(name, 0.0)) for f in subset]
                signature[name] = [float(np.mean(vals)), float(np.std(vals))]
        signatures[class_name] = signature

    rules = {
        'straight': 'total_turn_deg < 25 AND sinuosity < 1.025 AND max_turn_deg < 20',
        'semi_complex': 'smooth broad curve; no complex/v_complex rule matched; total_turn_deg >= 25 OR sinuosity >= 1.025 OR max_turn_deg >= 20',
        'complex': 'not v_complex AND (max_turn_deg >= 40 OR num_turns_60 >= 1 OR num_turns_30 >= 2 OR (total_turn_deg >= 70 AND max_turn_deg >= 25) OR (total_turn_deg >= 55 AND turn_sign_changes >= 2))',
        'v_complex': 'total_turn_deg >= 110 AND (sinuosity >= 1.04 OR num_turns_30 >= 2 OR turn_sign_changes >= 2)',
    }
    meta = {
        'name': CLASSIFIER_NAME,
        'feature_names': GEOMETRY_FEATURE_NAMES,
        'rules': rules,
        'complexity_score_weights': {
            'total_turn_deg': 0.34,
            'max_turn_deg_over_0_30m_window': 0.26,
            'sinuosity_excess': 0.18,
            'num_turns_30': 0.10,
            'num_turns_60': 0.07,
            'turn_sign_changes': 0.05,
        },
        'sampling_target_percent': dict(zip(CLASS_NAMES, [float(x) for x in quota])),
        'note': ('Class membership uses fixed geometric rules rather than PCA or per-pool bins. '
                 'v_complex requires cumulative turning of at least 110 degrees plus additional '
                 'evidence of curvature. Clearance and path length are hard filters, not class inputs. '
                 'The sampler balances successful episodes; pool inventory counts may differ by class.'),
    }
    return labels, scores, meta, rules, signatures

def validate_pool(pool, quota, min_paths_per_class):
    distribution = pool.get('distribution', {})
    failures = 0
    for name in CLASS_NAMES:
        n = distribution.get(name, 0)
        q = quota.get(name, 0)
        needed = max(min_paths_per_class, 2 * q)
        status = 'OK' if n >= needed else 'FAIL'
        print(f'  {name}: {n} (need >= {needed}) [{status}]')
        if n < needed:
            failures += 1
    return failures



def _pool_file_paths(args):
    data_root = os.path.abspath(args.input_dir or args.output_dir)
    scenario_dir = os.path.join(data_root, args.scenario)
    return scenario_dir, os.path.join(scenario_dir, 'path_pool.json'), os.path.join(scenario_dir, 'path_pool_poses.json')


def upgrade_existing_pool(args):
    """Filter and relabel an existing pool using the same generator classifier."""
    from collections import Counter
    from scipy.ndimage import distance_transform_edt

    scenario_dir, pool_path, poses_path = _pool_file_paths(args)
    if not os.path.isfile(pool_path) or not os.path.isfile(poses_path):
        print(f'ERROR: existing pool files not found for {args.scenario}', file=sys.stderr)
        return 1
    with open(pool_path, 'r') as f:
        source_pool = json.load(f)
    with open(poses_path, 'r') as f:
        source_poses = json.load(f)

    maps_root = os.path.abspath(args.maps_root)
    map_folder = find_map_folder(maps_root, args.scenario)
    folder_name = os.path.basename(map_folder)
    map_yaml = os.path.join(map_folder, f'map_{folder_name}_prohibition.yaml')
    map_arr, res, ox, oy, free_thresh, occ_thresh, negate = load_map(map_yaml)
    free = build_free_mask(map_arr, free_thresh, occ_thresh, negate)
    clearance_grid = distance_transform_edt(free) * res

    kept = []
    kept_poses = {}
    rejects = Counter()
    for original in source_pool.get('paths', []):
        try:
            path_id = int(original['id'])
            start = [float(original['start'][0]), float(original['start'][1])]
            goal = [float(original['goal'][0]), float(original['goal'][1])]
            points = source_poses.get(str(path_id))
            if not points or len(points) < 2:
                rejects['missing_poses'] += 1
                continue
            points = [[float(p[0]), float(p[1])] for p in points]
        except (KeyError, TypeError, ValueError, IndexError):
            rejects['bad_record'] += 1
            continue

        euclidean = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if euclidean < args.min_goal_distance or euclidean > args.max_goal_distance:
            rejects['distance'] += 1
            continue
        if (math.hypot(points[0][0] - start[0], points[0][1] - start[1]) > 0.05 or
                math.hypot(points[-1][0] - goal[0], points[-1][1] - goal[1]) > 0.05):
            rejects['endpoint_mismatch'] += 1
            continue
        features = compute_features(points, clearance_grid, ox, oy, res)
        if features is None:
            rejects['bad_metrics'] += 1
            continue
        if features['min_clearance_m'] < args.min_clearance:
            rejects['clearance'] += 1
            continue
        if features['path_length_m'] > args.max_path_length:
            rejects['path_length'] += 1
            continue
        kept.append({
            'id': path_id,
            'start': start,
            'goal': goal,
            'metrics': {k: round(float(v), 4) for k, v in features.items()},
        })
        kept_poses[str(path_id)] = points

    print(f'[{args.scenario}] upgrade source={len(source_pool.get("paths", []))} '
          f'kept={len(kept)} rejects={dict(rejects)}', flush=True)
    required_count = max(int(args.num_paths), 4 * int(args.min_paths_per_class))
    if len(kept) < required_count:
        print(f'ERROR: filtered pool has {len(kept)} routes; need {required_count}. '
              'Original files were not replaced.', file=sys.stderr)
        return 1

    quota = [int(x) for x in args.quota.split(',')]
    if len(quota) != 4 or sum(quota) <= 0:
        print('ERROR: quota must contain four non-negative values with positive sum', file=sys.stderr)
        return 1
    labels, scores, classifier_meta, boundaries, signatures = classify_by_geometry(
        [p['metrics'] for p in kept], quota)
    for index, record in enumerate(kept):
        record['class'] = labels[index]
        record['score'] = float(scores[index])
    order = {name: index for index, name in enumerate(CLASS_NAMES)}
    kept.sort(key=lambda p: (order[p['class']], p['score']))
    distribution = Counter(p['class'] for p in kept)
    insufficient = [name for name in CLASS_NAMES if distribution.get(name, 0) < args.min_paths_per_class]
    if insufficient:
        print(f'ERROR: not enough routes per class: {dict(distribution)}', file=sys.stderr)
        return 1

    active_ids = {str(p['id']) for p in kept}
    kept_poses = {key: value for key, value in kept_poses.items() if key in active_ids}
    pool = {
        'pool_version': POOL_VERSION,
        'scenario': args.scenario,
        'generated_at': source_pool.get('generated_at', datetime.now().isoformat()),
        'upgraded_at': datetime.now().isoformat(),
        'seed': source_pool.get('seed', args.seed),
        'num_paths': len(kept),
        'source_num_paths': int(source_pool.get('source_num_paths', source_pool.get('num_paths', len(source_pool.get('paths', []))))),
        'classification_scope': 'geometry_only',
        'classifier': CLASSIFIER_NAME,
        'classification_method': 'fixed_geometric_rules',
        'quota': dict(zip(CLASS_NAMES, quota)),
        'constraints': {
            'min_goal_distance_m': args.min_goal_distance,
            'max_goal_distance_m': args.max_goal_distance,
            'max_path_length_m': args.max_path_length,
            'min_clearance_m': args.min_clearance,
        },
        'geometry_classifier': classifier_meta,
        'class_boundaries': boundaries,
        'class_signatures': signatures,
        'distribution': {name: int(distribution.get(name, 0)) for name in CLASS_NAMES},
        'paths': kept,
    }
    tmp_pool = pool_path + '.tmp'
    tmp_poses = poses_path + '.tmp'
    with open(tmp_pool, 'w') as f:
        json.dump(pool, f, indent=1)
    with open(tmp_poses, 'w') as f:
        json.dump(kept_poses, f)
    os.replace(tmp_pool, pool_path)
    os.replace(tmp_poses, poses_path)
    print(f'[{args.scenario}] upgraded pool written: {pool_path}')
    for name in CLASS_NAMES:
        print(f'  {name}: {distribution.get(name, 0)}')
    return 0


def analyze_existing_pool(args):
    """Write CSV, report, and white-background path plots from a saved pool."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f'ERROR: --analyze-existing needs matplotlib: {exc}', file=sys.stderr)
        return 1
    import csv
    from collections import Counter

    _, pool_path, poses_path = _pool_file_paths(args)
    if not os.path.isfile(pool_path) or not os.path.isfile(poses_path):
        print(f'ERROR: pool or poses file missing: {pool_path}', file=sys.stderr)
        return 1
    with open(pool_path) as f:
        pool = json.load(f)
    with open(poses_path) as f:
        poses_map = json.load(f)
    records = pool.get('paths', [])
    valid = []
    errors = []
    for rec in records:
        path_id = rec.get('id')
        points = poses_map.get(str(path_id))
        if not points or len(points) < 2:
            errors.append(f'{path_id}: missing poses')
            continue
        geometry = compute_geometry_features(points)
        if geometry is None:
            errors.append(f'{path_id}: invalid geometry')
            continue
        old_metrics = rec.get('metrics', {})
        for key in ('min_clearance_m', 'mean_clearance_m'):
            if key in old_metrics:
                geometry[key] = float(old_metrics[key])
        start = rec.get('start', points[0])
        goal = rec.get('goal', points[-1])
        euclidean = math.hypot(float(goal[0])-float(start[0]), float(goal[1])-float(start[1]))
        valid.append({
            'record': rec, 'points': np.asarray(points, dtype=float),
            'metrics': geometry, 'euclidean': euclidean,
        })
    if not valid:
        print('ERROR: no valid paths found', file=sys.stderr)
        return 1

    labels, scores, _, class_rules, signatures = classify_by_geometry(
        [item['metrics'] for item in valid], [25, 25, 25, 25])
    for item, label_name, score in zip(valid, labels, scores):
        item['new_class'] = label_name
        item['complexity_score'] = score
    out_dir = Path(pool_path).parent
    rows = []
    for item in valid:
        rec = item['record']; met = item['metrics']; pts = item['points']
        rows.append({
            'id': rec.get('id'), 'original_class': rec.get('class'),
            'geometry_class': item['new_class'], 'complexity_score': round(item['complexity_score'], 6),
            'start_x': float(rec.get('start', pts[0])[0]), 'start_y': float(rec.get('start', pts[0])[1]),
            'goal_x': float(rec.get('goal', pts[-1])[0]), 'goal_y': float(rec.get('goal', pts[-1])[1]),
            'euclidean_distance_m': item['euclidean'],
            **{k: met.get(k, '') for k in FEATURE_NAMES},
            'within_runtime_distance_and_length': (2.0 <= item['euclidean'] <= 3.0 and met['path_length_m'] <= 3.0),
            'within_runtime_filters_using_stored_clearance': (2.0 <= item['euclidean'] <= 3.0 and met['path_length_m'] <= 3.0 and float(met.get('min_clearance_m', 0.0)) >= 0.15),
            'num_poses': len(pts),
        })
    csv_path = out_dir / 'path_pool_analysis.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    class_lines = {c: [] for c in CLASS_NAMES}
    for item in valid:
        class_lines[item['new_class']].append(item['points'])
    fig, ax = plt.subplots(figsize=(10, 12), dpi=180, facecolor='white')
    ax.set_facecolor('white')
    for cls in CLASS_NAMES:
        if class_lines[cls]:
            ax.add_collection(LineCollection(class_lines[cls], colors=CLASS_COLORS[cls], linewidths=0.45, alpha=0.30))
    ax.autoscale_view(); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'{args.scenario}: {len(valid)} paths classified by geometric rules')
    ax.grid(True, color='#DDDDDD', linewidth=0.4, alpha=0.6)
    counts = Counter(labels)
    ax.legend(handles=[Line2D([0], [0], color=CLASS_COLORS[c], lw=2,
                              label=f'{c} ({counts.get(c, 0)})') for c in CLASS_NAMES],
              loc='upper right', framealpha=0.95, facecolor='white')
    fig.tight_layout()
    all_png = out_dir / 'path_pool_all_paths_white.png'
    fig.savefig(all_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), dpi=150, facecolor='white')
    for cls, ax in zip(CLASS_NAMES, axes.flat):
        ax.set_facecolor('white')
        if class_lines[cls]:
            ax.add_collection(LineCollection(class_lines[cls], colors=CLASS_COLORS[cls], linewidths=0.7, alpha=0.45))
        ax.autoscale_view(); ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{cls}: {counts.get(cls, 0)} paths')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.grid(True, color='#DDDDDD', linewidth=0.4, alpha=0.6)
    fig.tight_layout()
    by_class_png = out_dir / 'path_pool_by_class_white.png'
    fig.savefig(by_class_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    filtered = [item for item in valid
                if 2.0 <= item['euclidean'] <= 3.0
                and item['metrics']['path_length_m'] <= 3.0
                and float(item['metrics'].get('min_clearance_m', 0.0)) >= 0.15]
    filtered_png = out_dir / 'path_pool_filtered_geometry_classes_white.png'
    filtered_counts = Counter()
    if filtered:
        filtered_labels, filtered_scores, _, _, _ = classify_by_geometry(
            [item['metrics'] for item in filtered], [25, 25, 25, 25])
        for item, class_name, score in zip(filtered, filtered_labels, filtered_scores):
            item['filtered_class'] = class_name
            item['filtered_score'] = score
            filtered_counts[class_name] += 1
        fig, ax = plt.subplots(figsize=(10, 12), dpi=180, facecolor='white')
        ax.set_facecolor('white')
        for cls in CLASS_NAMES:
            segments = [item['points'] for item in filtered if item.get('filtered_class') == cls]
            if segments:
                ax.add_collection(LineCollection(segments, colors=CLASS_COLORS[cls], linewidths=0.65, alpha=0.42))
        ax.autoscale_view(); ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f'{args.scenario}: filtered routes (n={len(filtered)})\nDistance 2-3 m, path <=3 m, stored clearance >=0.15 m')
        ax.grid(True, color='#DDDDDD', linewidth=0.4, alpha=0.6)
        ax.legend(handles=[Line2D([0], [0], color=CLASS_COLORS[c], lw=2,
                                  label=f'{c} ({filtered_counts.get(c, 0)})') for c in CLASS_NAMES],
                  loc='upper right', framealpha=0.95, facecolor='white')
        fig.tight_layout()
        fig.savefig(filtered_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    report = [
        f'# Path pool analysis: {args.scenario}', '',
        f'- Pool path count: {len(records)}', f'- Valid path count: {len(valid)}',
        f'- Invalid records: {len(errors)}', f'- Pool version: {pool.get("pool_version")}',
        f'- Original labels: {pool.get("distribution", {})}',
        f'- Reclassified by fixed geometry rules: {dict(counts)}',
        f'- Routes passing runtime filters using stored clearance: {len(filtered)}',
        f'- Filtered class counts: {dict(filtered_counts)}', '',
        'Labels use fixed geometric rules. Smooth broad arcs remain semi-complex even when their cumulative turning is moderate/high; complex requires a localized strong bend or multiple distinct bends; v_complex requires at least 110 degrees of cumulative turning plus evidence of a return, sinuosity or repeated bends. Clearance and path length are hard filters and do not determine the class.', '',
        '| Class | Count | Geometry rule | Mean sinuosity | Mean max turn (deg) | Mean total turn (deg) | Mean sign changes |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for cls in CLASS_NAMES:
        subset = [x for x in valid if x['new_class'] == cls]
        vals = [x['metrics'] for x in subset]
        mean = lambda key: float(np.mean([m.get(key, 0.0) for m in vals])) if vals else 0.0
        report.append(f'| {cls} | {len(subset)} | {class_rules[cls]} | {mean("sinuosity"):.3f} | {mean("max_turn_deg"):.1f} | {mean("total_turn_deg"):.1f} | {mean("turn_sign_changes"):.1f} |')
    report += [
        '',
        '## Filtered routes that satisfy the configured distance/length/clearance checks',
        '',
        '| Class | Count | Mean sinuosity | Mean max turn (deg) | Mean total turn (deg) | Mean sign changes |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for cls in CLASS_NAMES:
        subset = [x for x in filtered if x.get('filtered_class') == cls]
        vals = [x['metrics'] for x in subset]
        mean = lambda key: float(np.mean([m.get(key, 0.0) for m in vals])) if vals else 0.0
        report.append(f'| {cls} | {len(subset)} | {mean("sinuosity"):.3f} | {mean("max_turn_deg"):.1f} | {mean("total_turn_deg"):.1f} | {mean("turn_sign_changes"):.1f} |')
    report += ['', '## Output files', '', f'- `{csv_path.name}`', f'- `{all_png.name}`', f'- `{by_class_png.name}`']
    if filtered:
        report.append(f'- `{filtered_png.name}`')
    md_path = out_dir / 'path_pool_analysis.md'
    md_path.write_text('\n'.join(report) + '\n')
    print(f'[{args.scenario}] analysis complete: {len(valid)}/{len(records)} routes')
    print(f'classes by geometric rules: {dict(counts)}')
    print(f'CSV: {csv_path}')
    print(f'All paths image: {all_png}')
    print(f'By-class image: {by_class_png}')
    if filtered:
        print(f'Filtered paths image: {filtered_png} ({dict(filtered_counts)})')
    print(f'Report: {md_path}')
    if errors:
        print(f'Invalid route examples: {errors[:5]}')
    return 0

def _gmm_resample_polyline(points, spacing_m):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-8))
    pts = pts[keep]
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    length = float(cumulative[-1])
    if length <= 1e-5:
        return None
    distances = np.arange(0.0, length, spacing_m)
    if not len(distances) or length - distances[-1] > 1e-6:
        distances = np.append(distances, length)
    else:
        distances[-1] = length
    xs = np.interp(distances, cumulative, pts[:, 0])
    ys = np.interp(distances, cumulative, pts[:, 1])
    return np.column_stack((xs, ys)), length


def _gmm_count_turn_events(turns_deg, threshold_deg, window_m, min_separation_m=0.45):
    selected = np.flatnonzero(np.abs(turns_deg) >= threshold_deg)
    if len(selected) == 0:
        return 0
    max_gap = max(1, int(round(min_separation_m / window_m)))
    count = 1
    previous = int(selected[0])
    for raw in selected[1:]:
        current = int(raw)
        if current - previous > max_gap:
            count += 1
        previous = current
    return count


def compute_gmm_geometry_features(points_xy, old_metrics=None,
                                  spacing_m=0.05, turn_window_m=0.60):
    """Experimental features based on fixed physical spacing and turn windows."""
    old_metrics = old_metrics or {}
    resampled_result = _gmm_resample_polyline(points_xy, spacing_m)
    if resampled_result is None:
        return None
    pts, path_length = resampled_result
    euclidean = float(np.linalg.norm(pts[-1] - pts[0]))
    if euclidean < 1e-5:
        return None

    hop = max(1, int(round(turn_window_m / spacing_m)))
    coarse = pts[::hop]
    if np.linalg.norm(coarse[-1] - pts[-1]) > 1e-5:
        coarse = np.vstack((coarse, pts[-1]))
    if len(coarse) >= 3:
        delta_xy = np.diff(coarse, axis=0)
        headings = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
        signed_turns = []
        for previous, current in zip(headings[:-1], headings[1:]):
            delta = current - previous
            signed_turns.append(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
        signed_turns = np.asarray(signed_turns, dtype=np.float64)
    else:
        signed_turns = np.asarray([], dtype=np.float64)

    absolute_turns = np.abs(signed_turns)
    significant_signs = []
    for turn in signed_turns:
        if abs(turn) >= 4.0:
            sign = 1 if turn > 0 else -1
            if not significant_signs or significant_signs[-1] != sign:
                significant_signs.append(sign)
    sign_changes = max(0, len(significant_signs) - 1)

    return {
        'path_length_m': float(path_length),
        'euclidean_distance_m': euclidean,
        'sinuosity': float(path_length / max(euclidean, 1e-9)),
        'max_turn_deg': float(absolute_turns.max()) if len(absolute_turns) else 0.0,
        'total_turn_deg': float(absolute_turns.sum()) if len(absolute_turns) else 0.0,
        'num_turns_30': float(_gmm_count_turn_events(signed_turns, 30.0, turn_window_m)),
        'num_turns_60': float(_gmm_count_turn_events(signed_turns, 60.0, turn_window_m)),
        'turn_sign_changes': float(sign_changes),
        'min_clearance_m': float(old_metrics.get('min_clearance_m', float('nan'))),
        'mean_clearance_m': float(old_metrics.get('mean_clearance_m', float('nan'))),
    }


def _gmm_draw_background(ax, map_context):
    if map_context is None:
        ax.set_facecolor('white')
        return
    map_arr, resolution, ox, oy = map_context[:4]
    height, width = map_arr.shape
    ax.imshow(
        map_arr, cmap='gray', origin='lower', interpolation='nearest',
        extent=(ox, ox + width * resolution, oy, oy + height * resolution),
        vmin=0, vmax=255, alpha=1.0, zorder=0)


def _manual_review_pick_class(records, count, seed):
    """Select a repeatable mix of low-, medium- and high-confidence routes."""
    rng = random.Random(seed)
    records = list(records)
    confidence = [float(r.get('gmm_confidence', 0.0)) for r in records]
    buckets = [
        [r for r, c in zip(records, confidence) if c < 0.60],
        [r for r, c in zip(records, confidence) if 0.60 <= c < 0.85],
        [r for r, c in zip(records, confidence) if c >= 0.85],
    ]
    quotas = [int(round(count * 0.30)), int(round(count * 0.40)), 0]
    quotas[2] = count - quotas[0] - quotas[1]
    chosen = []
    used_ids = set()
    for bucket, quota in zip(buckets, quotas):
        rng.shuffle(bucket)
        for record in bucket[:quota]:
            if record['id'] not in used_ids:
                chosen.append(record)
                used_ids.add(record['id'])
    if len(chosen) < min(count, len(records)):
        remaining = [r for r in records if r['id'] not in used_ids]
        rng.shuffle(remaining)
        chosen.extend(remaining[:min(count, len(records)) - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def export_manual_review(args, records, candidates, map_context, output_dir,
                         used_features, silhouette, clearance_source, map_yaml_used):
    """Write a seeded, balanced manual-label sample and obstacle-backed plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f'ERROR: --manual-review requires matplotlib: {exc}', file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_labels = output_dir / 'manual_review_labels.csv'
    if existing_labels.exists() and not args.manual_review_overwrite:
        print(f'ERROR: {existing_labels} already exists; choose another --gmm-output-dir or pass --manual-review-overwrite only if you intend to discard its manual labels.', file=sys.stderr)
        return 1
    if map_context is None:
        print('ERROR: manual review requires the scenario prohibition map. Check --maps-root.', file=sys.stderr)
        return 1

    selected = []
    per_class_counts = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        available = [r for r in candidates if r['gmm_class'] == class_name]
        picked = _manual_review_pick_class(
            available, args.manual_per_class, args.manual_seed + class_index * 1009)
        per_class_counts[class_name] = len(picked)
        for record in picked:
            record_copy = dict(record)
            record_copy['manual_sample_id'] = f'M{len(selected) + 1:03d}'
            selected.append(record_copy)

    if not selected:
        print('ERROR: no eligible paths available for manual review.', file=sys.stderr)
        return 1

    csv_path = output_dir / 'manual_review_labels.csv'
    csv_fields = [
        'manual_sample_id', 'path_id', 'predicted_class', 'gmm_confidence',
        'p_straight', 'p_semi_complex', 'p_complex', 'p_v_complex',
        'geometry_rule_class', 'euclidean_distance_m', 'path_length_m',
        'sinuosity', 'max_turn_deg', 'total_turn_deg', 'num_turns_30',
        'num_turns_60', 'turn_sign_changes', 'min_clearance_m',
        'mean_clearance_m', 'manual_class', 'manual_pattern', 'manual_notes',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in selected:
            f = record['gmm_features']
            probabilities = record.get('gmm_probabilities', {})
            row = {
                'manual_sample_id': record['manual_sample_id'],
                'path_id': record['id'],
                'predicted_class': record['gmm_class'],
                'gmm_confidence': round(float(record['gmm_confidence']), 6),
                'geometry_rule_class': record.get('current_rule_class', ''),
                'euclidean_distance_m': round(float(f.get('euclidean_distance_m', 0.0)), 4),
                'path_length_m': round(float(f.get('path_length_m', 0.0)), 4),
                'manual_class': '',
                'manual_pattern': '',
                'manual_notes': '',
            }
            for name, key in [
                ('p_straight', 'straight'), ('p_semi_complex', 'semi_complex'),
                ('p_complex', 'complex'), ('p_v_complex', 'v_complex')]:
                row[name] = round(float(probabilities.get(key, 0.0)), 6)
            for key in [
                'sinuosity', 'max_turn_deg', 'total_turn_deg', 'num_turns_30',
                'num_turns_60', 'turn_sign_changes', 'min_clearance_m', 'mean_clearance_m']:
                value = f.get(key, '')
                row[key] = round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(float(value)) else ''
            writer.writerow(row)

    json_path = output_dir / 'manual_review_samples.json'
    json_samples = []
    for record in selected:
        json_samples.append({
            'manual_sample_id': record['manual_sample_id'],
            'path_id': record['id'],
            'predicted_class': record['gmm_class'],
            'gmm_confidence': float(record['gmm_confidence']),
            'gmm_probabilities': record.get('gmm_probabilities', {}),
            'eligible': bool(record['eligible']),
            'points_xy_m': np.asarray(record['points'], dtype=float).round(4).tolist(),
            'features': {k: (float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None)
                         for k, v in record['gmm_features'].items()},
        })
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'scenario': args.scenario,
            'seed': args.manual_seed,
            'num_per_predicted_class': args.manual_per_class,
            'selected_count': len(selected),
            'map_yaml': map_yaml_used,
            'source_clearance': clearance_source,
            'filter_constraints': {
                'min_goal_distance_m': args.min_goal_distance,
                'max_goal_distance_m': args.max_goal_distance,
                'max_path_length_m': args.max_path_length,
                'min_clearance_m': args.min_clearance,
            },
            'samples': json_samples,
        }, handle, indent=2)

    colors = CLASS_COLORS
    map_arr, resolution, ox, oy = map_context[:4]
    map_height, map_width = map_arr.shape
    map_extent = (ox, ox + map_width * resolution, oy, oy + map_height * resolution)

    # Overview uses the full map and metric coordinates with equal X/Y scale.
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180, facecolor='white')
    _gmm_draw_background(ax, map_context)
    for class_name in CLASS_NAMES:
        subset = [r for r in selected if r['gmm_class'] == class_name]
        for record in subset:
            pts = np.asarray(record['points'], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=colors[class_name], linewidth=1.2, alpha=0.85)
            ax.scatter(pts[0, 0], pts[0, 1], marker='o', s=12, color=colors[class_name], zorder=4)
    ax.set_xlim(map_extent[0], map_extent[1])
    ax.set_ylim(map_extent[2], map_extent[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(False)
    handles = [Line2D([0], [0], color=colors[c], lw=2, label=f'{c} (n={per_class_counts[c]})') for c in CLASS_NAMES]
    ax.legend(handles=handles, loc='upper right', framealpha=0.92)
    ax.set_title(f'{args.scenario}: seeded manual review sample, seed={args.manual_seed}')
    fig.tight_layout()
    overview_path = output_dir / 'manual_review_overview_map.png'
    fig.savefig(overview_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    # Four contact sheets, one per predicted class, each tile cropped in metres around its path.
    for class_name in CLASS_NAMES:
        subset = [r for r in selected if r['gmm_class'] == class_name]
        if not subset:
            continue
        columns = 2
        rows = int(math.ceil(len(subset) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(11, max(8, rows * 3.1)),
                                 dpi=160, facecolor='white', squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, record in zip(axes.flat, subset):
            ax.set_visible(True)
            _gmm_draw_background(ax, map_context)
            pts = np.asarray(record['points'], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=colors[class_name], linewidth=2.1, zorder=5)
            ax.scatter(pts[0, 0], pts[0, 1], color='#1b9e77', edgecolor='black', linewidth=0.4, s=24, zorder=6)
            ax.scatter(pts[-1, 0], pts[-1, 1], color='#222222', marker='x', linewidth=1.4, s=30, zorder=6)
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            span = max(float(xmax - xmin), float(ymax - ymin), 1.2) + 0.8
            cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
            half = span * 0.5
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect('equal', adjustable='box')
            f = record['gmm_features']
            ax.set_title(
                f"{record['manual_sample_id']} | id={record['id']} | conf={record['gmm_confidence']:.2f}\n"
                f"L={f.get('path_length_m', 0):.2f} m, sin={f.get('sinuosity', 0):.2f}, "
                f"turn={f.get('total_turn_deg', 0):.0f} deg, clr={f.get('min_clearance_m', 0):.2f} m",
                fontsize=8)
            ax.set_xlabel('x (m)', fontsize=7)
            ax.set_ylabel('y (m)', fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(f'{args.scenario} manual review: predicted {class_name} (seed={args.manual_seed})', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        sheet_path = output_dir / f'manual_review_{class_name}.png'
        fig.savefig(sheet_path, facecolor='white', bbox_inches='tight')
        plt.close(fig)

    readme_path = output_dir / 'README_manual_review.md'
    readme_path.write_text(
        f'# Manual path classification review: {args.scenario}\n\n'
        f'- Seed: {args.manual_seed}\n'
        f'- Sample selected: {len(selected)} paths, target {args.manual_per_class} per predicted class.\n'
        f'- Map image: `{map_yaml_used}`\n'
        f'- Clearance source: {clearance_source}\n'
        f'- GMM turn window: {args.gmm_turn_window_m:.2f} m\n'
        f'- GMM features: {", ".join(used_features)}\n'
        f'- Silhouette: {silhouette:.4f}\n\n'
        '## How to label\n\n'
        'Open the four `manual_review_<class>.png` sheets. Each panel shows the path ID and metric data over the true scenario map. '
        'The green dot is the start, the black X is the goal, and both axes are in metres with equal X/Y scale. '
        'In `manual_review_labels.csv`, fill `manual_class` with exactly one of `straight`, `semi_complex`, `complex`, `v_complex`. '
        'Use `manual_pattern` for a short visual pattern such as `straight`, `single_arc`, `s_curve`, `obstacle_detour`, `zigzag_return`, '
        'or `narrow_passage`; use `manual_notes` for corrections. Do not edit `predicted_class` or `path_id`. '
        'The CSV and JSON are review-only and do not modify the active pool.\n',
        encoding='utf-8')

    print(f'[{args.scenario}] manual review seed={args.manual_seed} per_class={args.manual_per_class} '
          f'selected={len(selected)} counts={per_class_counts}')
    print(f'CSV labels template: {csv_path}')
    print(f'Overview map: {overview_path}')
    print(f'Contact sheets: {output_dir}/manual_review_<class>.png')
    print(f'JSON samples: {json_path}')
    if any(per_class_counts[name] < args.manual_per_class for name in CLASS_NAMES):
        print('WARNING: at least one predicted class has fewer eligible candidates than requested; all available candidates were selected.')
    return 0


def analyze_gmm_existing_pool(args, manual_review_only=False):
    """Offline GMM comparison. It never changes the active pool or sampler."""
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        print(f'ERROR: --gmm-experiment requires scikit-learn: {exc}', file=sys.stderr)
        return 1
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError as exc:
        print(f'ERROR: --gmm-experiment requires matplotlib: {exc}', file=sys.stderr)
        return 1

    scenario_dir, pool_path, poses_path = _pool_file_paths(args)
    if not os.path.isfile(pool_path) or not os.path.isfile(poses_path):
        print(f'ERROR: pool or poses file missing for {args.scenario}: {pool_path}', file=sys.stderr)
        return 1
    with open(pool_path, 'r', encoding='utf-8') as handle:
        pool = json.load(handle)
    with open(poses_path, 'r', encoding='utf-8') as handle:
        poses_map = json.load(handle)

    map_context = None
    clearance_grid = None
    clearance_source = 'stored pool metrics'
    map_yaml_used = None
    try:
        maps_root = os.path.abspath(args.maps_root)
        folder = find_map_folder(maps_root, args.scenario)
        folder_name = os.path.basename(folder)
        map_yaml_used = os.path.join(folder, f'map_{folder_name}_prohibition.yaml')
        map_arr, map_res, map_ox, map_oy, free_thresh, occ_thresh, negate = load_map(map_yaml_used)
        free = build_free_mask(map_arr, free_thresh, occ_thresh, negate)
        from scipy.ndimage import distance_transform_edt
        clearance_grid = distance_transform_edt(free) * map_res
        display_map = np.where(free, 255, map_arr).astype(np.uint8)
        map_context = (display_map, map_res, map_ox, map_oy)
        clearance_source = 'recomputed from scenario prohibition map'
    except Exception as exc:
        print(f'WARNING: map/clearance recomputation unavailable; using stored clearance: {exc}')

    records = []
    rejected = Counter()
    for rec in pool.get('paths', []):
        try:
            path_id = int(rec['id'])
            raw_points = poses_map.get(str(path_id))
            if not raw_points or len(raw_points) < 2:
                rejected['missing_poses'] += 1
                continue
            points = [[float(p[0]), float(p[1])] for p in raw_points]
            old_metrics = rec.get('metrics', {})
            current_features = compute_geometry_features(points)
            if current_features is None:
                rejected['invalid_current_features'] += 1
                continue
            gmm_features = compute_gmm_geometry_features(
                points, old_metrics, args.gmm_resample_spacing_m, args.gmm_turn_window_m)
            if gmm_features is None:
                rejected['invalid_gmm_features'] += 1
                continue
            start = rec.get('start', points[0])
            goal = rec.get('goal', points[-1])
            euclidean = math.hypot(float(goal[0]) - float(start[0]), float(goal[1]) - float(start[1]))
            gmm_features['euclidean_distance_m'] = euclidean

            # Recompute clearance from the scenario map when possible. Fall back to stored pool values offline.
            if clearance_grid is not None:
                exact = compute_features(points, clearance_grid, map_ox, map_oy, map_res)
                if exact is None:
                    rejected['outside_map'] += 1
                    continue
                for key in ('min_clearance_m', 'mean_clearance_m'):
                    if key in exact:
                        current_features[key] = exact[key]
                        gmm_features[key] = exact[key]
            else:
                for key in ('min_clearance_m', 'mean_clearance_m'):
                    if key in old_metrics:
                        current_features[key] = float(old_metrics[key])
                        gmm_features[key] = float(old_metrics[key])

            clearance = float(gmm_features.get('min_clearance_m', float('nan')))
            eligible = (
                args.min_goal_distance <= euclidean <= args.max_goal_distance
                and gmm_features['path_length_m'] <= args.max_path_length
                and math.isfinite(clearance) and clearance >= args.min_clearance
            )
            records.append({
                'id': path_id, 'start': start, 'goal': goal, 'points': np.asarray(points, dtype=float),
                'original_class': rec.get('class', ''),
                'original_score': rec.get('score', ''),
                'current_features': current_features, 'gmm_features': gmm_features,
                'current_rule_class': classify_one_geometry(current_features),
                'eligible': bool(eligible),
            })
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            rejected['bad_record'] += 1

    candidates = [r for r in records if r['eligible']]
    print(f'[{args.scenario}] GMM experiment: {len(records)}/{len(pool.get("paths", []))} paths loaded; '
          f'eligible={len(candidates)}; clearance={clearance_source}')
    print(f'filters: euclidean=[{args.min_goal_distance:.2f},{args.max_goal_distance:.2f}]m; '
          f'path<={args.max_path_length:.2f}m; clearance>={args.min_clearance:.2f}m')
    if len(candidates) < 40:
        print('ERROR: fewer than 40 eligible routes; cannot compare four GMM components reliably.', file=sys.stderr)
        return 1

    all_feature_names = [
        'sinuosity', 'max_turn_deg', 'total_turn_deg',
        'num_turns_30', 'num_turns_60', 'turn_sign_changes',
    ]
    feature_matrix_all = np.asarray([
        [r['gmm_features'][name] for name in all_feature_names] for r in candidates
    ], dtype=np.float64)
    if not np.isfinite(feature_matrix_all).all():
        print('ERROR: non-finite features in eligible paths.', file=sys.stderr)
        return 1
    feature_std = feature_matrix_all.std(axis=0)
    active_indices = [i for i, value in enumerate(feature_std) if value > 1e-8]
    used_features = [all_feature_names[i] for i in active_indices]
    if len(used_features) < 2:
        print(f'ERROR: insufficient variable geometry features: {used_features}', file=sys.stderr)
        return 1
    if len(used_features) != len(all_feature_names):
        dropped = [name for name in all_feature_names if name not in used_features]
        print(f'[{args.scenario}] GMM ignoring constant features: {dropped}')

    x_candidates = feature_matrix_all[:, active_indices]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_candidates)
    gmm = GaussianMixture(
        n_components=4, covariance_type='full', random_state=args.gmm_seed,
        n_init=5, reg_covar=1e-5, max_iter=500)
    # Limit BLAS threads so this quick analysis stays predictable on desktop systems.
    with threadpool_limits(limits=1):
        components = gmm.fit_predict(x_scaled)
        probabilities = gmm.predict_proba(x_scaled)
        try:
            silhouette = float(silhouette_score(
                x_scaled, components, sample_size=min(1500, len(candidates)),
                random_state=args.gmm_seed)) if len(set(components)) > 1 else float('nan')
        except Exception:
            silhouette = float('nan')

    mean_all = feature_matrix_all.mean(axis=0)
    component_data = []
    for component in range(4):
        centroid_active = scaler.inverse_transform(gmm.means_[component:component + 1])[0]
        centroid_all = {name: float(mean_all[i]) for i, name in enumerate(all_feature_names)}
        for j, original_index in enumerate(active_indices):
            centroid_all[all_feature_names[original_index]] = float(centroid_active[j])
        score = (
            2.0 * max(0.0, centroid_all['sinuosity'] - 1.0)
            + centroid_all['num_turns_30'] / 3.0
            + centroid_all['num_turns_60'] / 1.5
            + centroid_all['max_turn_deg'] / 90.0
            + centroid_all['total_turn_deg'] / 180.0
            + centroid_all['turn_sign_changes'] / 2.0
        )
        component_data.append((float(score), component, centroid_all))
    component_data.sort(key=lambda value: (
        value[0], value[2]['num_turns_60'], value[2]['turn_sign_changes']))
    component_to_class = {component: CLASS_NAMES[index]
                          for index, (_, component, _) in enumerate(component_data)}

    all_matrix = np.asarray([[r['gmm_features'][name] for name in used_features] for r in records], dtype=float)
    with threadpool_limits(limits=1):
        all_probabilities = gmm.predict_proba(scaler.transform(all_matrix))
    for item, probs in zip(records, all_probabilities):
        component = int(np.argmax(probs))
        item['gmm_component'] = component
        item['gmm_class'] = component_to_class[component]
        item['gmm_confidence'] = float(np.max(probs))
        item['gmm_probabilities'] = {
            component_to_class[c]: float(probs[c]) for c in range(4)}

    counts_gmm = Counter(r['gmm_class'] for r in candidates)
    counts_current = Counter(r['current_rule_class'] for r in candidates)
    counts_original = Counter(r['original_class'] for r in candidates)
    counts_confident = Counter(r['gmm_class'] for r in candidates
                               if r['gmm_confidence'] >= args.gmm_min_confidence)
    low_confidence = sum(1 for r in candidates if r['gmm_confidence'] < args.gmm_min_confidence)

    default_analysis_dir = 'analysis_manual_review' if manual_review_only else 'analysis_gmm_experiment'
    output_dir = Path(args.gmm_output_dir) if args.gmm_output_dir else Path(scenario_dir) / default_analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if manual_review_only:
        return export_manual_review(
            args, records, candidates, map_context, output_dir, used_features,
            silhouette, clearance_source, map_yaml_used)

    csv_fields = [
        'id', 'eligible', 'original_class', 'current_geometry_rule_class',
        'gmm_class', 'gmm_component', 'gmm_confidence',
        'euclidean_distance_m', 'current_path_length_m', 'gmm_path_length_m',
        'sinuosity', 'max_turn_deg', 'total_turn_deg', 'num_turns_30',
        'num_turns_60', 'turn_sign_changes', 'min_clearance_m', 'mean_clearance_m',
    ]
    csv_path = output_dir / 'path_pool_gmm_comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for r in records:
            f = r['gmm_features']
            writer.writerow({
                'id': r['id'], 'eligible': int(r['eligible']),
                'original_class': r['original_class'],
                'current_geometry_rule_class': r['current_rule_class'],
                'gmm_class': r['gmm_class'], 'gmm_component': r['gmm_component'],
                'gmm_confidence': f"{r['gmm_confidence']:.6f}",
                'euclidean_distance_m': f"{f['euclidean_distance_m']:.4f}",
                'current_path_length_m': f"{r['current_features'].get('path_length_m', '')}",
                'gmm_path_length_m': f.get('path_length_m', ''),
                **{key: (f[key] if math.isfinite(f.get(key, float('nan'))) else '')
                   for key in ['sinuosity', 'max_turn_deg', 'total_turn_deg',
                               'num_turns_30', 'num_turns_60', 'turn_sign_changes',
                               'min_clearance_m', 'mean_clearance_m']},
            })

    output_json = {
        'scenario': args.scenario,
        'mode': 'offline_gmm_comparison_only',
        'generated_at': datetime.now().isoformat(),
        'source_pool': pool_path, 'source_poses': poses_path,
        'source_pool_modified': False,
        'clearance_source': clearance_source, 'map_yaml': map_yaml_used,
        'filters': {
            'min_goal_distance_m': args.min_goal_distance,
            'max_goal_distance_m': args.max_goal_distance,
            'max_path_length_m': args.max_path_length,
            'min_clearance_m': args.min_clearance,
        },
        'geometry_settings': {
            'resample_spacing_m': args.gmm_resample_spacing_m,
            'turn_window_m': args.gmm_turn_window_m,
            'features_requested': all_feature_names,
            'features_used': used_features,
            'features_dropped_as_constant': [n for n in all_feature_names if n not in used_features],
        },
        'gmm': {
            'n_components': 4, 'covariance_type': 'full', 'random_state': args.gmm_seed,
            'n_init': 5, 'converged': bool(gmm.converged_), 'n_iter': int(gmm.n_iter_),
            'weights': gmm.weights_.tolist(),
            'silhouette_score_sampled': silhouette,
            'silhouette_sample_size': min(1500, len(candidates)),
            'min_confidence': args.gmm_min_confidence,
            'component_to_class': {str(k): v for k, v in component_to_class.items()},
            'components_ordered_by_complexity': [
                {'component': component, 'class': component_to_class[component],
                 'complexity_score': score, 'centroid': centroid}
                for score, component, centroid in component_data],
        },
        'counts': {
            'total_loaded': len(records), 'eligible': len(candidates),
            'original_class': dict(counts_original),
            'current_geometry_rules': dict(counts_current),
            'gmm': dict(counts_gmm), 'gmm_confident': dict(counts_confident),
            'gmm_low_confidence_count': low_confidence,
            'rejects': dict(rejected),
        },
        'paths': [{
            'id': r['id'], 'start': r['start'], 'goal': r['goal'],
            'eligible': r['eligible'], 'original_class': r['original_class'],
            'current_geometry_rule_class': r['current_rule_class'],
            'gmm_class': r['gmm_class'], 'gmm_component': r['gmm_component'],
            'gmm_confidence': r['gmm_confidence'], 'gmm_probabilities': r['gmm_probabilities'],
            'current_metrics': r['current_features'], 'gmm_metrics': r['gmm_features'],
            'num_poses': int(len(r['points'])),
        } for r in records],
    }
    json_path = output_dir / 'path_pool_gmm_experiment.json'
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(output_json, handle, indent=2, allow_nan=False)

    colors = CLASS_COLORS
    def draw_paths(ax, items, class_key, alpha=0.24, linewidth=0.35, only_eligible=False):
        _gmm_draw_background(ax, map_context)
        for cls in CLASS_NAMES:
            for eligible_state in (True, False):
                if only_eligible and not eligible_state:
                    continue
                selected = [r['points'] for r in items
                            if r[class_key] == cls and r['eligible'] == eligible_state]
                if selected:
                    line_alpha = alpha if eligible_state else alpha * 0.35
                    ax.add_collection(LineCollection(
                        selected, colors=colors[cls], linewidths=linewidth,
                        alpha=line_alpha, zorder=2))
        ax.autoscale_view()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.grid(False)

    fig, ax = plt.subplots(figsize=(10, 12), dpi=160, facecolor='white')
    draw_paths(ax, records, 'gmm_class', alpha=0.15, linewidth=0.30)
    ax.set_title(f'{args.scenario}: all {len(records)} paths by experimental GMM labels')
    ax.legend(handles=[Line2D([0], [0], color=colors[c], lw=2,
                              label=f'{c}: {counts_gmm.get(c, 0)} eligible') for c in CLASS_NAMES],
              loc='upper right', framealpha=0.9, facecolor='white')
    fig.tight_layout()
    all_png = output_dir / 'paths_all_gmm.png'
    fig.savefig(all_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), dpi=160, facecolor='white')
    for cls, ax in zip(CLASS_NAMES, axes.flat):
        subset = [r for r in records if r['gmm_class'] == cls and r['eligible']]
        draw_paths(ax, subset, 'gmm_class', alpha=0.35, linewidth=0.55, only_eligible=True)
        ax.set_title(f'{cls}: {len(subset)} eligible paths')
    fig.tight_layout()
    by_class_png = output_dir / 'paths_gmm_eligible_by_class.png'
    fig.savefig(by_class_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=150, facecolor='white')
    for ax, key, title in [
        (axes[0], 'original_class', 'Original pool labels'),
        (axes[1], 'current_rule_class', 'Current geometry rules'),
        (axes[2], 'gmm_class', 'GMM experiment'),
    ]:
        draw_paths(ax, records, key, alpha=0.13, linewidth=0.27)
        ax.set_title(title)
    fig.tight_layout()
    comparison_png = output_dir / 'paths_classifier_comparison.png'
    fig.savefig(comparison_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    # Representative eligible routes close to each GMM centroid are easier to inspect than 5,000 overlays.
    all_active = np.asarray([[r['gmm_features'][n] for n in used_features] for r in records], dtype=float)
    with threadpool_limits(limits=1):
        all_scaled = scaler.transform(all_active)
    eligible_indices_by_class = {
        cls: [i for i, r in enumerate(records) if r['eligible'] and r['gmm_class'] == cls]
        for cls in CLASS_NAMES}
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=180, facecolor='white')
    for cls, ax in zip(CLASS_NAMES, axes.flat):
        _gmm_draw_background(ax, map_context)
        chosen = []
        cls_records = eligible_indices_by_class[cls]
        if cls_records:
            component = next(c for c, name in component_to_class.items() if name == cls)
            dists = [(float(np.linalg.norm(all_scaled[i] - gmm.means_[component])), i) for i in cls_records]
            chosen = [idx for _, idx in sorted(dists)[:min(15, len(dists))]]
        for idx in chosen:
            pts = records[idx]['points']
            ax.plot(pts[:, 0], pts[:, 1], color=colors[cls], linewidth=1.2, alpha=0.82)
            ax.scatter(pts[0, 0], pts[0, 1], color=colors[cls], marker='o', s=12, zorder=3)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=colors[cls], marker='x', s=17, zorder=3)
        ax.autoscale_view(); ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{cls}: {len(chosen)} representative eligible paths')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.tight_layout()
    representatives_png = output_dir / 'paths_gmm_representatives.png'
    fig.savefig(representatives_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    report = [
        f'# GMM experiment: {args.scenario}', '',
        '**Offline only:** source pool, poses and live sampler are not modified.', '',
        f'- Pool paths loaded: {len(records)} / {len(pool.get("paths", []))}',
        f'- Eligible after runtime-like filters: {len(candidates)}',
        f'- Clearance source: {clearance_source}',
        f'- Euclidean range: {args.min_goal_distance:.2f} to {args.max_goal_distance:.2f} m',
        f'- Max path length: {args.max_path_length:.2f} m',
        f'- Minimum clearance: {args.min_clearance:.2f} m',
        f'- Resampling interval: {args.gmm_resample_spacing_m:.2f} m',
        f'- Turn window: {args.gmm_turn_window_m:.2f} m',
        f'- GMM features used: {", ".join(used_features)}',
        f'- Constant features dropped: {", ".join(n for n in all_feature_names if n not in used_features) or "none"}',
        f'- GMM converged: {gmm.converged_}; iterations: {gmm.n_iter_}',
        f'- Silhouette score: {silhouette:.4f}' if math.isfinite(silhouette) else '- Silhouette score unavailable',
        f'- GMM assignments below confidence {args.gmm_min_confidence:.2f}: {low_confidence}', '',
        '## Eligible candidate counts', '',
        '| Class | Original pool label | Current geometry rules | GMM experimental | GMM confident |',
        '|---|---:|---:|---:|---:|',
    ]
    for cls in CLASS_NAMES:
        report.append(f'| {cls} | {counts_original.get(cls, 0)} | {counts_current.get(cls, 0)} | {counts_gmm.get(cls, 0)} | {counts_confident.get(cls, 0)} |')
    report += ['', '## GMM component signatures', '',
               '| Proposed class | Component | Complexity score | Sinuosity | Max turn (deg) | Total turn (deg) | Turns >=30 | Turns >=60 | Sign changes |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for score, component, metrics in component_data:
        report.append(
            f'| {component_to_class[component]} | {component} | {score:.3f} | '
            f'{metrics["sinuosity"]:.3f} | {metrics["max_turn_deg"]:.1f} | '
            f'{metrics["total_turn_deg"]:.1f} | {metrics["num_turns_30"]:.2f} | '
            f'{metrics["num_turns_60"]:.2f} | {metrics["turn_sign_changes"]:.2f} |')
    report += [
        '', '## Interpretation', '',
        '- GMM clusters the eligible geometry feature space; names are assigned by ascending geometric complexity score.',
        '- Clearance is a safety filter only and is not used to make a path geometrically complex.',
        '- If silhouette is below 0.30, treat the four-cluster separation as poor. Do not activate this model in the sampler automatically.',
        '- Class names and candidate availability still require visual approval against the E01/E02/E03 examples.',
        '- The full JSON contains per-path GMM probabilities and confidence; the CSV keeps a row per path.', '',
        '## Outputs', '',
        f'- CSV: `{csv_path.name}`', f'- JSON: `{json_path.name}`',
        f'- All paths: `{all_png.name}`', f'- Eligible paths by class: `{by_class_png.name}`',
        f'- Three-classifier comparison: `{comparison_png.name}`', f'- Representative routes: `{representatives_png.name}`',
    ]
    report_path = output_dir / 'path_pool_gmm_experiment.md'
    report_path.write_text('\n'.join(report) + '\n', encoding='utf-8')

    print('Eligible candidate counts:')
    for cls in CLASS_NAMES:
        print(f'  {cls}: original={counts_original.get(cls, 0)} current_rules={counts_current.get(cls, 0)} '
              f'gmm={counts_gmm.get(cls, 0)}')
    print(f'GMM silhouette={silhouette:.4f}' if math.isfinite(silhouette) else 'GMM silhouette unavailable')
    print(f'analysis directory: {output_dir}')
    print(f'report: {report_path}')
    print(f'representative routes: {representatives_png}')
    if math.isfinite(silhouette) and silhouette < 0.30:
        print('WARNING: silhouette < 0.30; clusters may not represent four distinct geometry groups.')
    if any(counts_confident.get(cls, 0) < 50 for cls in CLASS_NAMES):
        print('WARNING: at least one class has fewer than 50 confident eligible paths; balanced 200-goal capture is not provisioned by this pool.')
    return 0



# Experimental geometry diagnosis and directed candidate generation.
# These modes write to a separate analysis directory and never update the active pool.

def compute_geometry_diagnostic(points_xy, spacing_m=0.05, turn_window_m=0.50):
    """Measure morphology at fixed metric spacing and suppress vertex noise."""
    result = _gmm_resample_polyline(points_xy, spacing_m)
    if result is None:
        return None
    pts, path_length = result
    chord = pts[-1] - pts[0]
    euclidean = float(np.linalg.norm(chord))
    if euclidean <= 1e-6:
        return None
    sinuosity = float(path_length / euclidean)

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        gaussian_filter1d = None

    # Lateral offset describes the route relative to the endpoint chord.
    chord_unit = chord / euclidean
    normal = np.asarray([-chord_unit[1], chord_unit[0]])
    lateral = (pts - pts[0]) @ normal
    lateral_smooth = (gaussian_filter1d(lateral, sigma=3.0, mode='nearest')
                      if gaussian_filter1d is not None and len(lateral) > 10 else lateral.copy())
    max_lateral = float(np.max(np.abs(lateral_smooth)))

    # Count sustained excursions to either side of the chord, not small waviness.
    lateral_threshold = 0.10
    side = np.zeros(len(lateral_smooth), dtype=np.int8)
    side[lateral_smooth >= lateral_threshold] = 1
    side[lateral_smooth <= -lateral_threshold] = -1
    raw_lobes = []
    current_sign = 0
    start_idx = None
    for i, value in enumerate(side):
        sign = int(value)
        if sign != current_sign:
            if current_sign != 0 and start_idx is not None:
                run = lateral_smooth[start_idx:i]
                if len(run) * spacing_m >= 0.20 and float(np.max(np.abs(run))) >= 0.12:
                    raw_lobes.append((current_sign, float(np.max(np.abs(run))), len(run) * spacing_m))
            current_sign = sign
            start_idx = i if sign != 0 else None
        elif sign != 0 and start_idx is None:
            start_idx = i
    if current_sign != 0 and start_idx is not None:
        run = lateral_smooth[start_idx:]
        if len(run) * spacing_m >= 0.20 and float(np.max(np.abs(run))) >= 0.12:
            raw_lobes.append((current_sign, float(np.max(np.abs(run))), len(run) * spacing_m))
    lobe_signs = []
    for sign, amplitude, length in raw_lobes:
        if not lobe_signs or lobe_signs[-1] != sign:
            lobe_signs.append(sign)
    lateral_lobes = len(raw_lobes)
    lateral_sign_changes = max(0, len(lobe_signs) - 1)

    # Estimate tangent every 0.20 m from a chord spanning turn_window_m.
    # Measuring every 5 cm heading delta made grid/smoothing jitter look like turns.
    station_m = 0.20
    station_hop = max(1, int(round(station_m / spacing_m)))
    half_window = max(1, int(round(turn_window_m / (2.0 * spacing_m))))
    indices = list(range(0, len(pts), station_hop))
    if indices[-1] != len(pts) - 1:
        indices.append(len(pts) - 1)
    headings = []
    for i in indices:
        lo = max(0, i - half_window)
        hi = min(len(pts) - 1, i + half_window)
        delta = pts[hi] - pts[lo]
        headings.append(math.atan2(float(delta[1]), float(delta[0])))
    headings = np.unwrap(np.asarray(headings, dtype=np.float64))
    if gaussian_filter1d is not None and len(headings) > 5:
        headings = gaussian_filter1d(headings, sigma=0.75, mode='nearest')

    signed_turns = np.degrees(np.diff(headings))
    signed_turns[np.abs(signed_turns) < 8.0] = 0.0
    abs_turns = np.abs(signed_turns)
    total_turn = float(abs_turns.sum())
    window_steps = max(1, int(round(0.60 / station_m)))
    local_turns = [float(abs_turns[i:i + window_steps].sum())
                   for i in range(len(abs_turns))]
    max_turn_window = max(local_turns, default=0.0)

    # Count contiguous, direction-consistent bends of meaningful magnitude.
    turn_lobes = []
    current_sign = 0
    angle_sum = 0.0
    for value in signed_turns:
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == current_sign and sign != 0:
            angle_sum += abs(float(value))
        else:
            if current_sign != 0 and angle_sum >= 18.0:
                turn_lobes.append((current_sign, angle_sum))
            current_sign = sign
            angle_sum = abs(float(value)) if sign else 0.0
    if current_sign != 0 and angle_sum >= 18.0:
        turn_lobes.append((current_sign, angle_sum))
    turn_sign_changes = max(0, len(turn_lobes) - 1)
    num_turns_30 = sum(1 for _, value in turn_lobes if value >= 30.0)
    num_turns_60 = sum(1 for _, value in turn_lobes if value >= 60.0)

    # Measure reverse progress against the chord axis for U-turn-like paths.
    progress = np.diff(pts, axis=0) @ chord_unit
    reverse_progress = float(np.maximum(-progress - 0.015, 0.0).sum())
    cumulative_progress = np.concatenate(([0.0], np.cumsum(progress)))
    peak_progress = -float('inf')
    max_backtrack = 0.0
    for value in cumulative_progress:
        peak_progress = max(peak_progress, float(value))
        max_backtrack = max(max_backtrack, peak_progress - float(value))
    if max_backtrack < 0.05:
        max_backtrack = 0.0

    return {
        'euclidean_distance_m': euclidean,
        'path_length_m': float(path_length),
        'sinuosity': sinuosity,
        'max_lateral_deviation_m': max_lateral,
        'lateral_lobes': float(lateral_lobes),
        'lateral_sign_changes': float(lateral_sign_changes),
        'max_turn_window_deg': float(max_turn_window),
        'total_turn_deg': total_turn,
        'turn_lobes': float(len(turn_lobes)),
        'num_turns_30': float(num_turns_30),
        'num_turns_60': float(num_turns_60),
        'turn_sign_changes': float(turn_sign_changes),
        'reverse_progress_m': reverse_progress,
        'max_backtrack_m': float(max_backtrack),
    }


def classify_geometry_diagnostic(features):
    """Classify path morphology; obstacle clearance is a separate label."""
    if not features:
        return 'invalid'
    sinuosity = float(features['sinuosity'])
    max_turn = float(features['max_turn_window_deg'])
    total_turn = float(features['total_turn_deg'])
    turns30 = float(features['num_turns_30'])
    turn_lobes = float(features['turn_lobes'])
    lateral_lobes = float(features['lateral_lobes'])
    lateral_changes = float(features['lateral_sign_changes'])
    max_lateral = float(features['max_lateral_deviation_m'])
    backtrack = float(features['max_backtrack_m'])
    turns60 = float(features['num_turns_60'])

    # V-complex means repeated side changes or genuine reverse progress.
    if ((backtrack >= 0.18 and total_turn >= 55.0) or
            (lateral_changes >= 2 and lateral_lobes >= 3 and max_lateral >= 0.12) or
            (turn_lobes >= 3 and turns30 >= 2 and total_turn >= 80.0) or
            (turns60 >= 2 and total_turn >= 100.0)):
        return 'v_complex'

    # Complex means an S curve or multiple distinct meaningful bends.
    if ((lateral_changes >= 1 and lateral_lobes >= 2 and max_lateral >= 0.12) or
            (turn_lobes >= 2 and turns30 >= 2 and max_turn >= 25.0) or
            (max_lateral >= 0.28 and sinuosity >= 1.06 and total_turn >= 25.0)):
        return 'complex'

    # A single broad arc stays semi-complex even if its total heading change is high.
    if max_lateral >= 0.08 or max_turn >= 18.0 or sinuosity >= 1.025 or total_turn >= 25.0:
        return 'semi_complex'
    return 'straight'


def obstacle_interaction_label(features, min_clearance_m):
    """Label clearance separately from geometry morphology."""
    min_clearance_m = float(min_clearance_m) if min_clearance_m is not None else float('nan')
    max_lateral = float(features.get('max_lateral_deviation_m', 0.0)) if features else 0.0
    if not math.isfinite(min_clearance_m):
        return 'unknown'
    if min_clearance_m < 0.20:
        return 'narrow_passage'
    if min_clearance_m < 0.30:
        return 'near_obstacle'
    if max_lateral >= 0.20 and min_clearance_m < 0.45:
        return 'possible_obstacle_detour'
    return 'open_or_clear'


def classify_dataset_diagnostic(features, min_clearance_m):
    """Combine geometry and obstacle proximity for the user's four dataset labels."""
    geometry_class = classify_geometry_diagnostic(features)
    if geometry_class == 'invalid':
        return geometry_class
    try:
        clearance = float(min_clearance_m)
    except (TypeError, ValueError):
        clearance = float('nan')
    if math.isfinite(clearance) and clearance < 0.25:
        has_nontrivial_shape = (
            float(features.get('sinuosity', 1.0)) >= 1.025 or
            float(features.get('max_lateral_deviation_m', 0.0)) >= 0.08 or
            float(features.get('max_turn_window_deg', 0.0)) >= 18.0 or
            float(features.get('total_turn_deg', 0.0)) >= 25.0)
        if has_nontrivial_shape:
            return 'v_complex'
    return geometry_class


def _diagnostic_map_context(args):
    maps_root = os.path.abspath(args.maps_root)
    folder = find_map_folder(maps_root, args.scenario)
    folder_name = os.path.basename(folder)
    yaml_path = os.path.join(folder, f'map_{folder_name}_prohibition.yaml')
    arr, res, ox, oy, free_thresh, occ_thresh, negate = load_map(yaml_path)
    free = build_free_mask(arr, free_thresh, occ_thresh, negate)
    clearance = distance_transform_edt(free) * res
    # Show traversable pixels white and prohibited/occupied pixels in their map shades.
    display = np.where(free, 255, np.minimum(arr, 170)).astype(np.uint8)
    context = {'folder': folder, 'yaml': yaml_path, 'image': display,
               'res': res, 'ox': ox, 'oy': oy, 'free': free, 'clearance': clearance}
    return context


def _draw_metric_map(ax, context):
    image = context['image']
    h, w = image.shape
    ox, oy, res = context['ox'], context['oy'], context['res']
    ax.imshow(image, cmap='gray', origin='lower', interpolation='nearest',
              extent=(ox, ox + w * res, oy, oy + h * res), vmin=0, vmax=255,
              zorder=0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')


def _load_pool_and_poses(args):
    _, pool_path, poses_path = _pool_file_paths(args)
    if not os.path.isfile(pool_path) or not os.path.isfile(poses_path):
        raise FileNotFoundError(f'pool or poses file missing: {pool_path} / {poses_path}')
    with open(pool_path, 'r', encoding='utf-8') as handle:
        pool = json.load(handle)
    with open(poses_path, 'r', encoding='utf-8') as handle:
        poses = json.load(handle)
    return pool, poses, pool_path, poses_path


def _line_hausdorff_approx(a, b, spacing=0.05):
    aa = _gmm_resample_polyline(a, spacing)
    bb = _gmm_resample_polyline(b, spacing)
    if aa is None or bb is None:
        return float('nan')
    pa, _ = aa
    pb, _ = bb
    try:
        from scipy.spatial import cKDTree
        d_ab = cKDTree(pb).query(pa, k=1)[0]
        d_ba = cKDTree(pa).query(pb, k=1)[0]
        return float(max(np.max(d_ab), np.max(d_ba)))
    except ImportError:
        return float('nan')



def compute_dense_clearance(points_xy, context, spacing_m=0.05):
    """Sample path segments at fixed metric spacing to check map safety."""
    resampled = _gmm_resample_polyline(points_xy, spacing_m)
    if resampled is None:
        return None
    points, _ = resampled
    height, width = context['free'].shape
    clearances = []
    out_of_bounds = 0
    occupied_samples = 0
    for x, y in points:
        col = int(math.floor((float(x) - context['ox']) / context['res']))
        row = int(math.floor((float(y) - context['oy']) / context['res']))
        if not (0 <= row < height and 0 <= col < width):
            out_of_bounds += 1
            continue
        if not bool(context['free'][row, col]):
            occupied_samples += 1
        clearances.append(float(context['clearance'][row, col]))
    if not clearances:
        return {
            'dense_min_clearance_m': float('nan'),
            'dense_mean_clearance_m': float('nan'),
            'dense_samples': int(len(points)),
            'out_of_bounds_samples': out_of_bounds,
            'occupied_samples': occupied_samples,
            'map_collision': True,
        }
    return {
        'dense_min_clearance_m': float(min(clearances)),
        'dense_mean_clearance_m': float(sum(clearances) / len(clearances)),
        'dense_samples': int(len(points)),
        'out_of_bounds_samples': out_of_bounds,
        'occupied_samples': occupied_samples,
        'map_collision': bool(out_of_bounds or occupied_samples),
    }

def geometry_audit_existing_pool(args):
    """Compare regenerated raw A* paths against the stored smoothed paths."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f'ERROR: --geometry-audit needs matplotlib: {exc}', file=sys.stderr)
        return 1
    try:
        pool, poses_map, pool_path, poses_path = _load_pool_and_poses(args)
        context = _diagnostic_map_context(args)
    except Exception as exc:
        print(f'ERROR: geometry audit setup failed: {exc}', file=sys.stderr)
        return 1

    candidates = []
    for rec in pool.get('paths', []):
        path_id = int(rec['id'])
        points = poses_map.get(str(path_id))
        if not points or len(points) < 2:
            continue
        candidates.append((rec, np.asarray(points, dtype=np.float64)))
    if not candidates:
        print('ERROR: pool has no routes with poses', file=sys.stderr)
        return 1
    rng = random.Random(args.audit_seed)
    requested_ids = set()
    if args.audit_ids.strip():
        try:
            requested_ids = {int(value.strip()) for value in args.audit_ids.split(',') if value.strip()}
        except ValueError:
            print('ERROR: --audit-ids must be comma-separated integer IDs', file=sys.stderr)
            return 1
    requested = [item for item in candidates if int(item[0]['id']) in requested_ids]
    remaining = [item for item in candidates if int(item[0]['id']) not in requested_ids]
    if args.audit_count == 0:
        candidates = requested + remaining
    else:
        to_pick = max(0, args.audit_count - len(requested))
        if len(remaining) > to_pick:
            remaining = rng.sample(remaining, to_pick)
        candidates = requested + remaining
    requested.sort(key=lambda item: int(item[0]['id']))
    remaining.sort(key=lambda item: int(item[0]['id']))
    candidates = requested + remaining

    out_dir = Path(args.gmm_output_dir) if args.gmm_output_dir else Path(pool_path).parent / 'analysis_geometry_audit'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    example_records = []
    rejected = Counter()
    for rec, stored in candidates:
        start = rec.get('start', stored[0].tolist())
        goal = rec.get('goal', stored[-1].tolist())
        stages = plan_path_stages(context['free'], start, goal,
                                  context['ox'], context['oy'], context['res'])
        if stages is None:
            rejected['raw_astar_failed'] += 1
            raw = None
            regenerated_final = None
        else:
            raw = np.asarray(stages['raw'], dtype=np.float64)
            regenerated_final = np.asarray(stages['final'], dtype=np.float64)

        stored_f = compute_geometry_diagnostic(stored, args.audit_spacing_m, args.audit_turn_window_m)
        raw_f = compute_geometry_diagnostic(raw, args.audit_spacing_m, args.audit_turn_window_m) if raw is not None else None
        regen_f = compute_geometry_diagnostic(regenerated_final, args.audit_spacing_m, args.audit_turn_window_m) if regenerated_final is not None else None
        stored_clearance = compute_dense_clearance(stored, context, args.audit_spacing_m)
        raw_clearance = compute_dense_clearance(raw, context, args.audit_spacing_m) if raw is not None else None
        regen_clearance = compute_dense_clearance(regenerated_final, context, args.audit_spacing_m) if regenerated_final is not None else None
        row = {
            'path_id': int(rec['id']), 'original_class': rec.get('class', ''),
            'stored_path_length_m': stored_f['path_length_m'] if stored_f else '',
            'stored_sinuosity': stored_f['sinuosity'] if stored_f else '',
            'stored_total_turn_deg': stored_f['total_turn_deg'] if stored_f else '',
            'stored_max_turn_window_deg': stored_f['max_turn_window_deg'] if stored_f else '',
            'stored_turn_sign_changes': stored_f['turn_sign_changes'] if stored_f else '',
            'stored_max_backtrack_m': stored_f['max_backtrack_m'] if stored_f else '',
            'stored_geometry_class': classify_geometry_diagnostic(stored_f),
            'stored_min_clearance_m': stored_clearance['dense_min_clearance_m'] if stored_clearance else '',
            'stored_obstacle_interaction': obstacle_interaction_label(stored_f, stored_clearance['dense_min_clearance_m'] if stored_clearance else None),
            'stored_dataset_class': classify_dataset_diagnostic(stored_f, stored_clearance['dense_min_clearance_m'] if stored_clearance else None),
            'raw_astar_path_length_m': raw_f['path_length_m'] if raw_f else '',
            'raw_astar_sinuosity': raw_f['sinuosity'] if raw_f else '',
            'raw_astar_total_turn_deg': raw_f['total_turn_deg'] if raw_f else '',
            'raw_astar_max_turn_window_deg': raw_f['max_turn_window_deg'] if raw_f else '',
            'raw_astar_turn_sign_changes': raw_f['turn_sign_changes'] if raw_f else '',
            'raw_astar_max_backtrack_m': raw_f['max_backtrack_m'] if raw_f else '',
            'raw_astar_geometry_class': classify_geometry_diagnostic(raw_f),
            'raw_astar_min_clearance_m': raw_clearance['dense_min_clearance_m'] if raw_clearance else '',
            'raw_astar_obstacle_interaction': obstacle_interaction_label(raw_f, raw_clearance['dense_min_clearance_m'] if raw_clearance else None),
            'raw_astar_dataset_class': classify_dataset_diagnostic(raw_f, raw_clearance['dense_min_clearance_m'] if raw_clearance else None),
            'regenerated_smoothed_path_length_m': regen_f['path_length_m'] if regen_f else '',
            'regenerated_smoothed_sinuosity': regen_f['sinuosity'] if regen_f else '',
            'regenerated_smoothed_total_turn_deg': regen_f['total_turn_deg'] if regen_f else '',
            'regenerated_smoothed_geometry_class': classify_geometry_diagnostic(regen_f),
            'regenerated_smoothed_min_clearance_m': regen_clearance['dense_min_clearance_m'] if regen_clearance else '',
            'regenerated_smoothed_obstacle_interaction': obstacle_interaction_label(regen_f, regen_clearance['dense_min_clearance_m'] if regen_clearance else None),
            'regenerated_smoothed_dataset_class': classify_dataset_diagnostic(regen_f, regen_clearance['dense_min_clearance_m'] if regen_clearance else None),
            'raw_vs_stored_hausdorff_approx_m': _line_hausdorff_approx(raw, stored) if raw is not None else '',
            'stored_dense_collision': stored_clearance['map_collision'] if stored_clearance else '',
            'stored_dense_samples': stored_clearance['dense_samples'] if stored_clearance else '',
            'raw_dense_collision': raw_clearance['map_collision'] if raw_clearance else '',
            'raw_dense_samples': raw_clearance['dense_samples'] if raw_clearance else '',
            'regenerated_dense_collision': regen_clearance['map_collision'] if regen_clearance else '',
            'regenerated_dense_samples': regen_clearance['dense_samples'] if regen_clearance else '',
            'stored_pose_count': len(stored),
            'raw_astar_pose_count': len(raw) if raw is not None else 0,
        }
        for prefix, feature_set in (
                ('stored', stored_f), ('raw_astar', raw_f),
                ('regenerated_smoothed', regen_f)):
            if feature_set:
                for feature_name, feature_value in feature_set.items():
                    row[f'{prefix}_{feature_name}'] = feature_value
        rows.append(row)
        example_records.append({'record': rec, 'stored': stored, 'raw': raw,
                                'regenerated_final': regenerated_final, 'features': row})

    csv_path = out_dir / 'geometry_audit.csv'
    fieldnames = list(rows[0].keys())
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / 'geometry_audit_paths.json'
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'scenario': args.scenario, 'seed': args.audit_seed,
            'source_pool': pool_path, 'source_poses': poses_path,
            'map_yaml': context['yaml'], 'map_resolution_m': context['res'],
            'sample_count': len(example_records), 'raw_astar_failures': dict(rejected),
            'explicit_audit_ids': sorted(requested_ids),
            'missing_explicit_audit_ids': sorted(requested_ids - {int(x['record']['id']) for x in example_records}),
            'records': [{
                'path_id': int(x['record']['id']), 'original_class': x['record'].get('class', ''),
                'start': x['record'].get('start'), 'goal': x['record'].get('goal'),
                'stored_smoothed_poses': x['stored'].tolist(),
                'raw_astar_poses': x['raw'].tolist() if x['raw'] is not None else None,
                'regenerated_final_poses': x['regenerated_final'].tolist() if x['regenerated_final'] is not None else None,
                'metrics': x['features'],
            } for x in example_records]
        }, handle, indent=2)

    manual_ids_present = bool(requested_ids)
    display_records = ([item for item in example_records if int(item['record']['id']) in requested_ids]
                       if manual_ids_present else example_records[:24])
    if not display_records:
        display_records = example_records[:24]
    n = len(display_records)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows), dpi=150, facecolor='white')
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, display_records):
        _draw_metric_map(ax, context)
        if item['raw'] is not None:
            pts = item['raw']
            ax.plot(pts[:, 0], pts[:, 1], color='#777777', linewidth=0.8, alpha=0.75, linestyle='--', label='raw A*')
        if item['regenerated_final'] is not None:
            pts = item['regenerated_final']
            ax.plot(pts[:, 0], pts[:, 1], color='#2878B5', linewidth=1.3, alpha=0.85, label='smoothed')
        pts = item['stored']
        ax.plot(pts[:, 0], pts[:, 1], color='#C84D4D', linewidth=1.0, alpha=0.85, label='stored pool')
        displayed_points = [item['stored']]
        if item['raw'] is not None:
            displayed_points.append(item['raw'])
        if item['regenerated_final'] is not None:
            displayed_points.append(item['regenerated_final'])
        bounds_points = np.vstack(displayed_points)
        margin = 0.65
        x0 = max(context['ox'], float(bounds_points[:, 0].min()) - margin)
        x1 = min(context['ox'] + context['image'].shape[1] * context['res'],
                 float(bounds_points[:, 0].max()) + margin)
        y0 = max(context['oy'], float(bounds_points[:, 1].min()) - margin)
        y1 = min(context['oy'] + context['image'].shape[0] * context['res'],
                 float(bounds_points[:, 1].max()) + margin)
        # Local crop makes nearby obstacles visible without distorting metric scale.
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(
            f"id={item['record']['id']} pool={item['record'].get('class', '')}\n"
            f"raw={item['features']['raw_astar_geometry_class']} "
            f"stored={item['features']['stored_geometry_class']}\n"
            f"dataset={item['features']['stored_dataset_class']}", fontsize=8)
    for ax in axes[n:]:
        ax.remove()
    if example_records:
        axes[0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{args.scenario}: raw A* vs smoothing vs stored path', fontsize=14)
    fig.tight_layout()
    comparison_png = out_dir / 'geometry_audit_comparison.png'
    fig.savefig(comparison_png, facecolor='white', bbox_inches='tight'); plt.close(fig)

    report_lines = [
        f'# Geometry audit: {args.scenario}', '',
        'Offline diagnostic only; production pool and sampler were not changed.',
        f'- Source pool: `{pool_path}`', f'- Map: `{context["yaml"]}`',
        f'- Paths examined: {len(rows)}', f'- Raw A* failures: {dict(rejected)}',
        f'- Explicit path IDs: {sorted(requested_ids)}',
        f'- Requested IDs missing from pool: {sorted(requested_ids - {int(x["record"]["id"]) for x in example_records})}',
        f'- CSV: `{csv_path.name}`', f'- JSON: `{json_path.name}`',
        f'- Scaled obstacle-mask overlay: `{comparison_png.name}`', '',
        'Raw A* is regenerated for the same stored endpoints. This is a controlled comparison, but it may differ from the original generated route if the path planner/image version changed.',
        'Inspect `raw_astar_*` against `stored_*` to determine whether grid-search geometry or post-processing changes the assigned morphology.',
    ]
    (out_dir / 'geometry_audit_report.md').write_text('\n'.join(report_lines)+'\n', encoding='utf-8')
    print(f'geometry audit complete: {len(rows)} paths')
    print(f'CSV: {csv_path}')
    print(f'plot: {comparison_png}')
    print(f'report: {out_dir / "geometry_audit_report.md"}')
    return 0


def _snap_point_to_clearance(point, valid_cells, ox, oy, res, rng, max_snap_m=0.30):
    row, col = world_to_grid(float(point[0]), float(point[1]), ox, oy, res)
    radius = max(1, int(math.ceil(max_snap_m / res)))
    delta = np.abs(valid_cells - np.asarray([row, col]))
    keep = (delta[:, 0] <= radius) & (delta[:, 1] <= radius)
    nearby = valid_cells[keep]
    if len(nearby) == 0:
        return None
    distance = np.hypot(nearby[:, 0] - row, nearby[:, 1] - col) * res
    nearby = nearby[distance <= max_snap_m]
    distance = distance[distance <= max_snap_m]
    if len(nearby) == 0:
        return None
    order = np.argsort(distance)[:min(25, len(distance))]
    chosen = int(rng.choice(order.tolist()))
    rr, cc = int(nearby[chosen, 0]), int(nearby[chosen, 1])
    return [ox + (cc + 0.5) * res, oy + (rr + 0.5) * res]


def _pattern_waypoints(start, goal, pattern, rng):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return []
    unit = delta / distance
    normal = np.asarray([-unit[1], unit[0]])
    amp = rng.uniform(0.16, 0.32)
    if pattern == 'straight':
        fractions, offsets = [], []
    elif pattern == 'single_arc':
        fractions, offsets = [0.50], [rng.choice([-1.0, 1.0]) * rng.uniform(0.18, 0.35)]
    elif pattern == 's_curve':
        a = rng.uniform(0.18, 0.30)
        fractions, offsets = [0.32, 0.68], [a, -a]
    else:  # u_turn_zigzag: includes a deliberate local backtrack along the chord.
        a = amp
        fractions, offsets = [0.30, 0.56, 0.40, 0.74], [a, a, -a, -a]
    points = []
    for frac, offset in zip(fractions, offsets):
        target = start + unit * (frac * distance) + normal * offset
        points.append(target.tolist())
    return points


def _route_through_waypoints(free, nodes, context):
    segments = []
    for first, second in zip(nodes[:-1], nodes[1:]):
        stages = plan_path_stages(free, first, second,
                                  context['ox'], context['oy'], context['res'])
        if stages is None:
            return None
        segment = stages['final']
        if segments:
            segment = segment[1:]
        segments.extend(segment)
    return segments if len(segments) >= 2 else None


def directed_candidate_probe(args):
    """Test via-point templates for producing genuinely curved/S/U-like routes.

    This is diagnostic only. It never writes path_pool.json and never sends
    goals to ROS. Forced via points here do not imply the live planner will follow
    the same path unless that behavior is implemented separately.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f'ERROR: --directed-probe needs matplotlib: {exc}', file=sys.stderr)
        return 1
    try:
        context = _diagnostic_map_context(args)
    except Exception as exc:
        print(f'ERROR: directed-probe map setup failed: {exc}', file=sys.stderr)
        return 1

    rng = random.Random(args.probe_seed)
    free = context['free']
    res, ox, oy = context['res'], context['ox'], context['oy']
    clearance_grid = context['clearance']
    valid_mask = free & (clearance_grid >= args.min_clearance)
    cells_rc = np.argwhere(valid_mask)
    if len(cells_rc) < 100:
        print('ERROR: fewer than 100 cells satisfy free-space and clearance constraints', file=sys.stderr)
        return 1
    patterns = ['straight', 'single_arc', 's_curve', 'u_turn_zigzag']
    accepted = []
    outcomes = Counter()
    attempts_by_pattern = Counter()
    t0 = time.time()

    for pattern in patterns:
        while (sum(1 for item in accepted if item['requested_pattern'] == pattern) < args.probe_per_pattern
               and attempts_by_pattern[pattern] < args.probe_attempts_per_pattern):
            attempts_by_pattern[pattern] += 1
            s_rc = cells_rc[rng.randrange(len(cells_rc))]
            start = [ox + (int(s_rc[1]) + 0.5) * res, oy + (int(s_rc[0]) + 0.5) * res]
            goal = None
            euclidean = None
            pair_max_distance = args.max_goal_distance
            if pattern in ('s_curve', 'u_turn_zigzag'):
                pair_max_distance = min(args.max_goal_distance, args.min_goal_distance + 0.60)
            for _ in range(60):
                g_rc = cells_rc[rng.randrange(len(cells_rc))]
                candidate = [ox + (int(g_rc[1]) + 0.5) * res, oy + (int(g_rc[0]) + 0.5) * res]
                d = math.hypot(candidate[0] - start[0], candidate[1] - start[1])
                if args.min_goal_distance <= d <= pair_max_distance:
                    goal = candidate
                    euclidean = d
                    break
            if goal is None:
                outcomes[f'{pattern}:distance_pair_missing'] += 1
                continue

            via_targets = _pattern_waypoints(start, goal, pattern, rng)
            via_points = []
            for target in via_targets:
                snapped = _snap_point_to_clearance(target, cells_rc, ox, oy, res, rng, args.probe_waypoint_snap_m)
                if snapped is None:
                    via_points = None
                    break
                via_points.append(snapped)
            if via_points is None:
                outcomes[f'{pattern}:waypoint_not_found'] += 1
                continue

            route = _route_through_waypoints(free, [start] + via_points + [goal], context)
            if route is None:
                outcomes[f'{pattern}:segment_unreachable'] += 1
                continue
            feats = compute_geometry_diagnostic(route, args.audit_spacing_m, args.audit_turn_window_m)
            if feats is None:
                outcomes[f'{pattern}:bad_geometry'] += 1
                continue
            # Sample each segment at fixed spacing to check safety between vertices.
            clearance_metrics = compute_dense_clearance(route, context, args.audit_spacing_m)
            if clearance_metrics is None or clearance_metrics['map_collision']:
                outcomes[f'{pattern}:map_collision'] += 1
                continue
            feats['min_clearance_m'] = float(clearance_metrics['dense_min_clearance_m'])
            feats['mean_clearance_m'] = float(clearance_metrics['dense_mean_clearance_m'])
            feats['dense_samples'] = int(clearance_metrics['dense_samples'])
            feats['obstacle_interaction'] = obstacle_interaction_label(feats, feats['min_clearance_m'])
            if euclidean < args.min_goal_distance or euclidean > args.max_goal_distance:
                outcomes[f'{pattern}:distance'] += 1
                continue
            if feats['path_length_m'] > args.max_path_length:
                outcomes[f'{pattern}:path_too_long'] += 1
                continue
            if feats['min_clearance_m'] < args.min_clearance:
                outcomes[f'{pattern}:clearance'] += 1
                continue

            actual_class = classify_geometry_diagnostic(feats)
            dataset_class = classify_dataset_diagnostic(feats, feats['min_clearance_m'])
            accepted.append({
                'id': len(accepted) + 1, 'requested_pattern': pattern,
                'geometry_class': actual_class, 'dataset_class': dataset_class, 'start': start, 'goal': goal,
                'via_points': via_points, 'poses': route,
                'metrics': feats, 'matched_requested_class': (
                    (pattern == 'straight' and actual_class == 'straight') or
                    (pattern == 'single_arc' and actual_class == 'semi_complex') or
                    (pattern == 's_curve' and actual_class == 'complex') or
                    (pattern == 'u_turn_zigzag' and actual_class == 'v_complex')),
            })
            outcomes[f'{pattern}:accepted_{actual_class}'] += 1

    out_dir = Path(args.gmm_output_dir) if args.gmm_output_dir else Path(args.output_dir) / args.scenario / f'directed_probe_seed_{args.probe_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'directed_candidate_probe.csv'
    fields = ['id', 'requested_pattern', 'geometry_class', 'dataset_class', 'matched_requested_class',
              'euclidean_distance_m', 'path_length_m', 'sinuosity', 'max_turn_window_deg',
              'total_turn_deg', 'num_turns_30', 'num_turns_60', 'turn_sign_changes',
              'max_lateral_deviation_m', 'lateral_lobes', 'lateral_sign_changes',
              'turn_lobes', 'reverse_progress_m', 'max_backtrack_m', 'min_clearance_m', 'mean_clearance_m', 'obstacle_interaction',
              'dense_samples', 'start_x', 'start_y', 'goal_x', 'goal_y', 'via_points']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in accepted:
            f = item['metrics']
            writer.writerow({
                'id': item['id'], 'requested_pattern': item['requested_pattern'],
                'geometry_class': item['geometry_class'], 'dataset_class': item['dataset_class'], 'matched_requested_class': item['matched_requested_class'],
                **{name: f.get(name, '') for name in fields if name in f},
                'start_x': item['start'][0], 'start_y': item['start'][1],
                'goal_x': item['goal'][0], 'goal_y': item['goal'][1],
                'via_points': json.dumps(item['via_points']),
            })
    json_path = out_dir / 'directed_candidate_probe.json'
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'scenario': args.scenario, 'seed': args.probe_seed,
            'map_yaml': context['yaml'], 'constraints': {
                'min_euclidean_m': args.min_goal_distance,
                'max_euclidean_m': args.max_goal_distance,
                'max_path_length_m': args.max_path_length,
                'min_clearance_m': args.min_clearance},
            'target_per_pattern': args.probe_per_pattern,
            'attempts_by_pattern': dict(attempts_by_pattern), 'outcomes': dict(outcomes),
            'paths': accepted,
        }, handle, indent=2)

    colors = {'straight': '#2878B5', 'single_arc': '#4A9D62',
              's_curve': '#E39C37', 'u_turn_zigzag': '#C84D4D'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=160, facecolor='white')
    for pattern, ax in zip(patterns, axes.flat):
        _draw_metric_map(ax, context)
        subset = [item for item in accepted if item['requested_pattern'] == pattern]
        for item in subset:
            pts = np.asarray(item['poses'], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=colors[pattern], linewidth=1.15, alpha=0.78)
            ax.plot(item['start'][0], item['start'][1], 'o', color='#189B72', markersize=3)
            ax.plot(item['goal'][0], item['goal'][1], 'x', color='black', markersize=4)
        matching = sum(bool(item['matched_requested_class']) for item in subset)
        ax.set_title(f'{pattern}: {len(subset)} safe routes; morphology matched {matching}')
    fig.tight_layout()
    plot_path = out_dir / 'directed_candidate_probe_by_pattern.png'
    fig.savefig(plot_path, facecolor='white', bbox_inches='tight'); plt.close(fig)

    report = [f'# Directed candidate probe: {args.scenario}', '',
              '**Diagnostic only. Does not change path_pool.json and does not command the robot.**',
              f'- Map: `{context["yaml"]}`', f'- Seed: {args.probe_seed}',
              f'- Constraints: Euclidean {args.min_goal_distance:.2f}-{args.max_goal_distance:.2f} m; '
              f'path length <= {args.max_path_length:.2f} m; clearance >= {args.min_clearance:.2f} m',
              f'- Elapsed: {time.time()-t0:.1f} s', f'- Accepted paths: {len(accepted)}', '',
              '| Requested shape | Accepted | Classified as intended class |', '|---|---:|---:|']
    mapping = {'straight':'straight', 'single_arc':'semi_complex', 's_curve':'complex', 'u_turn_zigzag':'v_complex'}
    for pattern in patterns:
        subset = [item for item in accepted if item['requested_pattern'] == pattern]
        matched = sum(bool(item['matched_requested_class']) for item in subset)
        report.append(f'| {pattern} | {len(subset)} | {matched} ({mapping[pattern]}) |')
    report += ['', '## Rejection / outcome counts', '']
    report += [f'- `{key}`: {value}' for key, value in sorted(outcomes.items())]
    report += ['', 'A directed via-point path is a geometry feasibility experiment only. The live classic planner still replans from start to goal; it will not follow these mandatory waypoints unless a separate path-following change is made.',
                '', f'- CSV: `{csv_path.name}`', f'- JSON: `{json_path.name}`', f'- Plot: `{plot_path.name}`']
    report_path = out_dir / 'directed_candidate_probe.md'
    report_path.write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(f'directed candidate probe complete: {len(accepted)} routes in {time.time()-t0:.1f}s')
    for pattern in patterns:
        subset = [item for item in accepted if item['requested_pattern'] == pattern]
        matched = sum(bool(item['matched_requested_class']) for item in subset)
        print(f'  {pattern}: {len(subset)} accepted, {matched} classified as intended')
    print(f'CSV: {csv_path}')
    print(f'Plot: {plot_path}')
    print(f'Report: {report_path}')
    return 0


def _weighted_astar_stages(safe_mask, clearance_grid, start_xy, goal_xy,
                           ox, oy, res, clearance_weight=3.0,
                           preferred_clearance_m=0.45,
                           hard_clearance_m=0.35, margin_m=4.0):
    """Plan on the hard-safe mask while penalizing clearance below preference."""
    sr, sc = world_to_grid(start_xy[0], start_xy[1], ox, oy, res)
    gr, gc = world_to_grid(goal_xy[0], goal_xy[1], ox, oy, res)
    h, w = safe_mask.shape
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if not safe_mask[sr, sc] or not safe_mask[gr, gc]:
        return None

    margin = max(10, int(math.ceil(margin_m / res)))
    r0 = max(0, min(sr, gr) - margin)
    r1 = min(h, max(sr, gr) + margin + 1)
    c0 = max(0, min(sc, gc) - margin)
    c1 = min(w, max(sc, gc) + margin + 1)
    walkable = safe_mask[r0:r1, c0:c1]
    clearance = clearance_grid[r0:r1, c0:c1]
    if clearance_weight > 0:
        preference_band = max(preferred_clearance_m - hard_clearance_m, 1e-3)
        shortfall = np.clip((preferred_clearance_m - clearance) / preference_band, 0.0, 1.0)
        penalty = 1.0 + clearance_weight * np.square(shortfall)
    else:
        penalty = np.ones_like(clearance, dtype=np.float64)
    penalty = np.asarray(penalty, dtype=np.float64)
    penalty[~walkable] = np.inf

    start = (sr - r0, sc - c0)
    goal = (gr - r0, gc - c0)
    queue = [(math.hypot(goal[0] - start[0], goal[1] - start[1]), 0.0,
              start[0], start[1])]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    height, width = walkable.shape

    while queue:
        _, g_current, row, col = heapq.heappop(queue)
        node = (row, col)
        if g_current > g_score.get(node, float('inf')) + 1e-9:
            continue
        if node in closed:
            continue
        if node == goal:
            path_rc = [node]
            while path_rc[-1] in came_from:
                path_rc.append(came_from[path_rc[-1]])
            path_rc.reverse()
            path_xy = [list(grid_to_world(r + r0, c + c0, ox, oy, res))
                       for r, c in path_rc]
            smoothed = smooth_path(path_xy, 0.1, 0.9, 1e-6)
            conservative = smooth_path(path_xy, 0.30, 0.25, 1e-5)
            options = [
                chaikin_smooth(downsample(smoothed, 0.12), 1),
                downsample(smoothed, 0.08),
                smoothed,
                chaikin_smooth(downsample(conservative, 0.08), 1),
                conservative,
            ]
            return {'raw': path_xy, 'smoothed': smoothed,
                    'final': options[0], 'safe_options': options}
        closed.add(node)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if not walkable[nr, nc] or not math.isfinite(penalty[nr, nc]):
                    continue
                if dr and dc and (not walkable[row + dr, col] or
                                  not walkable[row, col + dc]):
                    continue
                step = math.sqrt(2.0) if dr and dc else 1.0
                edge_cost = step * 0.5 * (penalty[row, col] + penalty[nr, nc])
                new_g = g_current + edge_cost
                next_node = (nr, nc)
                if new_g >= g_score.get(next_node, float('inf')):
                    continue
                g_score[next_node] = new_g
                came_from[next_node] = node
                heuristic = math.hypot(goal[0] - nr, goal[1] - nc)
                heapq.heappush(queue, (new_g + heuristic, new_g, nr, nc))
    return None


def _path_clearance_metrics(points, context, spacing_m):
    step = min(spacing_m, max(context['res'] * 0.5, 0.002))
    return compute_dense_clearance(points, context, step)


def _direct_line_metrics(start, goal, context, spacing_m):
    return _path_clearance_metrics([start, goal], context, spacing_m)


def _turn_events_near_obstacles(points, context, spacing_m=0.05,
                                turn_window_m=0.50,
                                min_event_turn_deg=18.0,
                                obstacle_influence_m=0.85):
    """Count sustained bends and mark those occurring near obstacles."""
    resampled = _gmm_resample_polyline(points, spacing_m)
    if resampled is None:
        return [], {'supported_turn_count': 0, 'turn_event_count': 0,
                    'supported_turn_fraction': 0.0,
                    'supported_maneuver_count': 0}
    pts, path_length = resampled
    if len(pts) < 5:
        return [], {'supported_turn_count': 0, 'turn_event_count': 0,
                    'supported_turn_fraction': 0.0,
                    'supported_maneuver_count': 0}

    hop = max(1, int(round(0.10 / spacing_m)))
    half_window = max(1, int(round(turn_window_m / (2.0 * spacing_m))))
    indices = list(range(0, len(pts), hop))
    if indices[-1] != len(pts) - 1:
        indices.append(len(pts) - 1)
    headings = []
    station_indices = []
    for idx in indices:
        lo = max(0, idx - half_window)
        hi = min(len(pts) - 1, idx + half_window)
        delta = pts[hi] - pts[lo]
        if np.linalg.norm(delta) < 1e-8:
            continue
        headings.append(math.atan2(float(delta[1]), float(delta[0])))
        station_indices.append(idx)
    if len(headings) < 3:
        return [], {'supported_turn_count': 0, 'turn_event_count': 0,
                    'supported_turn_fraction': 0.0,
                    'supported_maneuver_count': 0}

    signed = []
    positions = []
    for j in range(len(headings) - 1):
        angle = math.atan2(math.sin(headings[j + 1] - headings[j]),
                           math.cos(headings[j + 1] - headings[j]))
        deg = math.degrees(angle)
        signed.append(deg if abs(deg) >= 2.0 else 0.0)
        positions.append(station_indices[j + 1])

    events = []
    current_sign = 0
    angle_sum = 0.0
    indices_in_event = []

    def add_event(sign, angle, event_indices):
        if not sign or angle < min_event_turn_deg or not event_indices:
            return
        center_idx = event_indices[len(event_indices) // 2]
        x, y = pts[center_idx]
        col = int(math.floor((float(x) - context['ox']) / context['res']))
        row = int(math.floor((float(y) - context['oy']) / context['res']))
        h, w = context['clearance'].shape
        clearance = (float(context['clearance'][row, col])
                     if 0 <= row < h and 0 <= col < w else 0.0)
        events.append({
            'sign': int(sign), 'turn_deg': float(angle),
            'x': float(x), 'y': float(y),
            's_m': float(center_idx * spacing_m),
            'clearance_m': clearance,
            'obstacle_supported': clearance <= obstacle_influence_m,
        })

    for value, idx in zip(signed, positions):
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == current_sign and sign != 0:
            angle_sum += abs(float(value))
            indices_in_event.append(idx)
            continue
        add_event(current_sign, angle_sum, indices_in_event)
        current_sign = sign
        angle_sum = abs(float(value)) if sign else 0.0
        indices_in_event = [idx] if sign else []
    add_event(current_sign, angle_sum, indices_in_event)

    supported_events = sorted(
        [event for event in events if event['obstacle_supported']],
        key=lambda event: event['s_m'])
    # Nearby direction changes around one object form one avoidance maneuver.
    # Separate obstacle interactions require a meaningful along-path gap.
    maneuver_count = 0
    last_s = None
    for event in supported_events:
        event_s = float(event['s_m'])
        if last_s is None or event_s - last_s >= 0.85:
            maneuver_count += 1
        last_s = event_s

    supported = len(supported_events)
    metrics = {
        'turn_event_count': len(events),
        'supported_turn_count': supported,
        'supported_turn_fraction': supported / max(len(events), 1),
        'supported_maneuver_count': maneuver_count,
    }
    return events, metrics


def _purposeful_route_label(features, events, direct_metrics,
                            hard_clearance_m=0.35):
    """Label only obstacle-required maneuvers, not arbitrary curvature."""
    direct_clearance = float(direct_metrics['dense_min_clearance_m'])
    direct_collision = bool(direct_metrics['map_collision'])
    line_requires_avoidance = direct_collision or direct_clearance < hard_clearance_m
    supported_events = [event for event in events
                        if event.get('obstacle_supported', False)]
    supported = len(supported_events)
    maneuvers = int(features.get('supported_maneuver_count', 0))
    turn_total = float(features.get('total_turn_deg', 0.0))
    backtrack = float(features.get('max_backtrack_m', 0.0))
    sign_changes = int(features.get('turn_sign_changes', 0))
    lateral_sign_changes = int(features.get('lateral_sign_changes', 0))
    max_lateral = float(features.get('max_lateral_deviation_m', 0.0))
    sinuosity = float(features.get('sinuosity', 1.0))

    # A safe chord is optimal. Never add curvature to make a class quota.
    if not line_requires_avoidance:
        return 'straight', 'direct_line_fits_clearance'

    if supported == 0:
        # A direct line may be unsafe even if an almost straight corridor route is valid.
        if features.get('min_clearance_m', 0.0) >= hard_clearance_m and sinuosity <= 1.04 and turn_total < 30.0:
            return 'semi_complex', 'constrained_corridor_minimal_bend'
        return 'reject', 'detour_turns_not_supported_by_obstacles'

    # A U/return requires real reverse progress, or repeated obstacle-supported
    # maneuvers. Angle accumulation alone is never sufficient for v_complex.
    true_return = backtrack >= 0.28 and turn_total >= 100.0
    repeated_obstacle_detour = (
        maneuvers >= 2 and turn_total >= 115.0 and
        (lateral_sign_changes >= 1 or sign_changes >= 2) and max_lateral >= 0.22)
    multi_obstacle_return = (
        maneuvers >= 2 and turn_total >= 125.0 and
        lateral_sign_changes >= 1 and max_lateral >= 0.25 and backtrack >= 0.10)
    if true_return or repeated_obstacle_detour or multi_obstacle_return:
        return 'v_complex', 'obstacle_required_return_or_multiple_maneuvers'

    # Complex requires multiple obstacle interactions, not a multi-segment curve
    # around one box or minor path noise.
    obstacle_supported_s_curve = (
        supported >= 3 and sign_changes >= 1 and turn_total >= 65.0 and
        max_lateral >= 0.18)
    if ((maneuvers >= 2 and turn_total >= 40.0) or
            obstacle_supported_s_curve or
            (maneuvers >= 2 and sinuosity >= 1.08 and max_lateral >= 0.18)):
        return 'complex', 'multiple_obstacle_supported_maneuvers'

    return 'semi_complex', 'single_obstacle_avoidance_maneuver'


def _shortcut_path_by_clearance(points, context, min_clearance_m, spacing_m=0.05):
    """Remove unnecessary intermediate bends while preserving dense clearance."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        return pts.tolist()
    # Reduce dense A* grid traces before the visibility search; every proposed
    # shortcut is still checked against the map at metric spacing.
    coarse_spacing = max(0.08, float(spacing_m))
    pts = np.asarray(downsample(pts.tolist(), coarse_spacing), dtype=np.float64)
    if len(pts) < 3:
        return pts.tolist()
    output = [pts[0].tolist()]
    i = 0
    while i < len(pts) - 1:
        chosen = i + 1
        for j in range(len(pts) - 1, i + 1, -1):
            metrics = compute_dense_clearance(
                [pts[i].tolist(), pts[j].tolist()], context, spacing_m)
            if (metrics is not None and not metrics['map_collision'] and
                    metrics['dense_min_clearance_m'] >= min_clearance_m):
                chosen = j
                break
        output.append(pts[chosen].tolist())
        i = chosen
    return output

def purposeful_candidate_probe(args):
    """Generate only near-optimal paths whose bends respond to real obstacles.

    This is an offline experiment. It never modifies the production pool and
    never commands the robot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f'ERROR: --purposeful-probe needs matplotlib: {exc}', file=sys.stderr)
        return 1
    try:
        context = _diagnostic_map_context(args)
    except Exception as exc:
        print(f'ERROR: purposeful probe map setup failed: {exc}', file=sys.stderr)
        return 1

    rng = random.Random(args.purposeful_seed)
    free = context['free']
    clearance = context['clearance']
    res, ox, oy = context['res'], context['ox'], context['oy']
    # Inflate obstacles by the robot-center clearance, not only the map's free-cell threshold.
    hard_clearance = max(float(args.min_clearance), float(args.purposeful_hard_clearance_m))
    safe_mask = free & (clearance >= hard_clearance)
    min_clearance = hard_clearance
    cells = np.argwhere(safe_mask)
    near_obstacle_cells = np.argwhere(
        safe_mask & (clearance <= hard_clearance + args.purposeful_endpoint_obstacle_band_m))
    if len(near_obstacle_cells) < 100:
        near_obstacle_cells = cells
    if len(cells) < 100:
        print('ERROR: fewer than 100 safe map cells', file=sys.stderr)
        return 1

    classes = ['straight', 'semi_complex', 'complex', 'v_complex']
    samples = {name: [] for name in classes}
    rejects = Counter()
    total_attempts = 0
    started = time.time()
    # One map-cell spacing is enough for dense validation; sub-cell sampling
    # every 5 mm adds cost without meaningful precision on a 1 cm occupancy grid.
    effective_spacing = min(args.audit_spacing_m, max(res, 0.01))

    while total_attempts < args.purposeful_max_attempts and any(
            len(samples[name]) < args.purposeful_per_class for name in classes):
        total_attempts += 1
        endpoint_pool = cells
        v_quota_missing = len(samples['v_complex']) < args.purposeful_per_class
        favor_v_pair = v_quota_missing and rng.random() < args.purposeful_v_candidate_bias
        if len(samples['straight']) >= args.purposeful_per_class and rng.random() < args.purposeful_endpoint_obstacle_bias:
            endpoint_pool = near_obstacle_cells
        a = endpoint_pool[rng.randrange(len(endpoint_pool))]
        start = [ox + (int(a[1]) + 0.5) * res,
                 oy + (int(a[0]) + 0.5) * res]
        goal = None
        for _ in range(80):
            b = endpoint_pool[rng.randrange(len(endpoint_pool))]
            cand = [ox + (int(b[1]) + 0.5) * res,
                    oy + (int(b[0]) + 0.5) * res]
            distance = math.hypot(cand[0] - start[0], cand[1] - start[1])
            # Expand endpoint distance only while actively searching for v_complex.
            # Other classes keep the regular max_goal_distance limit.
            pair_max_distance = args.max_goal_distance
            if favor_v_pair:
                pair_max_distance = args.purposeful_v_max_goal_distance_m
            if args.min_goal_distance <= distance <= pair_max_distance:
                goal = cand
                break
        if goal is None:
            rejects['no_endpoint_pair'] += 1
            continue

        direct = _direct_line_metrics(start, goal, context, effective_spacing)
        if direct is None or direct.get('map_collision'):
            direct_clearance = 0.0 if direct is None else float(direct['dense_min_clearance_m'])
        else:
            direct_clearance = float(direct['dense_min_clearance_m'])
        direct_collision = direct is None or bool(direct.get('map_collision', True))

        # Preferred clearance is a soft route-cost target. Only the hard footprint
        # clearance determines whether the direct segment is unsafe.
        line_requires_avoidance = (
            direct_collision or direct_clearance < hard_clearance)
        if not line_requires_avoidance and len(samples['straight']) >= args.purposeful_per_class:
            rejects['class_quota_full_straight'] += 1
            continue

        if not line_requires_avoidance:
            # In open space a clear chord is the route. Do not add artificial bends.
            route = [start, goal]
            baseline_length = distance
        else:
            raw_stages = plan_path_stages(safe_mask, start, goal, ox, oy, res)
            if raw_stages is None:
                rejects['no_safe_path'] += 1
                continue
            baseline_raw = raw_stages.get('raw', [])
            if len(baseline_raw) < 2:
                rejects['invalid_raw_path'] += 1
                continue
            shortest_visible = _shortcut_path_by_clearance(
                baseline_raw, context, hard_clearance, effective_spacing)
            baseline_options = [shortest_visible]
            if len(shortest_visible) >= 3:
                lightly_smoothed = smooth_path(shortest_visible, 0.10, 0.15, 1e-5)
                baseline_options.append(lightly_smoothed)
                baseline_options.append(
                    chaikin_smooth(downsample(lightly_smoothed, 0.08), 1))
            baseline_route = None
            baseline_features = None
            for option in baseline_options:
                if len(option) < 2:
                    continue
                cm = _path_clearance_metrics(option, context, effective_spacing)
                if (cm is not None and not cm['map_collision'] and
                        cm['dense_min_clearance_m'] >= hard_clearance):
                    f = compute_geometry_diagnostic(option, args.audit_spacing_m,
                                                    args.purposeful_turn_window_m)
                    if f and (baseline_features is None or
                              f['path_length_m'] < baseline_features['path_length_m']):
                        baseline_route, baseline_features = option, f
            if baseline_route is None:
                rejects['baseline_smoothing_not_safe'] += 1
                continue
            baseline_length = float(baseline_features['path_length_m'])
            weighted_stages = _weighted_astar_stages(
                safe_mask, clearance, start, goal, ox, oy, res,
                clearance_weight=args.purposeful_clearance_weight,
                preferred_clearance_m=args.purposeful_preferred_clearance_m,
                hard_clearance_m=hard_clearance,
                margin_m=max(3.0, args.max_goal_distance + 1.0))
            route = baseline_route
            best_score = float('inf')
            options = [baseline_route]
            if weighted_stages:
                options.extend([weighted_stages.get('raw', []),
                                weighted_stages.get('smoothed', []),
                                weighted_stages.get('final', [])])
            expanded_options = []
            for source in options:
                if len(source) < 2:
                    continue
                expanded_options.append(source)
                expanded_options.append(_shortcut_path_by_clearance(
                    source, context, hard_clearance, effective_spacing))
            for option in expanded_options:
                if len(option) < 2:
                    continue
                cm = _path_clearance_metrics(option, context, effective_spacing)
                if cm is None or cm['map_collision'] or cm['dense_min_clearance_m'] < hard_clearance:
                    continue
                f = compute_geometry_diagnostic(option, args.audit_spacing_m,
                                                args.purposeful_turn_window_m)
                if f is None:
                    continue
                length = float(f['path_length_m'])
                candidate_path_limit = (
                    args.purposeful_v_max_path_length_m
                    if favor_v_pair else args.max_path_length
                )
                if length > candidate_path_limit:
                    continue
                if length > baseline_length * args.purposeful_max_near_optimal_ratio:
                    continue
                # Penalize low clearance, but keep path length dominant.
                sampled_for_risk = _gmm_resample_polyline(option, args.audit_spacing_m)
                if sampled_for_risk is None:
                    continue
                risk_values = []
                preference_band = max(args.purposeful_preferred_clearance_m - hard_clearance, 1e-3)
                for px, py in sampled_for_risk[0]:
                    rr = int(math.floor((float(py) - oy) / res))
                    cc = int(math.floor((float(px) - ox) / res))
                    if 0 <= rr < clearance.shape[0] and 0 <= cc < clearance.shape[1]:
                        shortfall = np.clip(
                            (args.purposeful_preferred_clearance_m - float(clearance[rr, cc])) / preference_band,
                            0.0, 1.0)
                        risk_values.append(float(shortfall * shortfall))
                if not risk_values:
                    continue
                risk_integral = float(np.mean(risk_values)) * length
                score = length + args.purposeful_risk_weight * risk_integral
                if score < best_score:
                    route, best_score = option, score

        features = compute_geometry_diagnostic(
            route, args.audit_spacing_m, args.purposeful_turn_window_m)
        if features is None:
            rejects['invalid_geometry'] += 1
            continue
        candidate_path_limit = (
            args.purposeful_v_max_path_length_m
            if favor_v_pair else args.max_path_length
        )
        if features['path_length_m'] > candidate_path_limit:
            rejects['path_too_long'] += 1
            continue
        route_clearance = _path_clearance_metrics(route, context, effective_spacing)
        if route_clearance is None or route_clearance['map_collision']:
            rejects['smoothed_path_collision'] += 1
            continue
        if route_clearance['dense_min_clearance_m'] < hard_clearance:
            rejects['hard_footprint_clearance'] += 1
            continue
        events, event_metrics = _turn_events_near_obstacles(
            route, context, spacing_m=args.audit_spacing_m,
            turn_window_m=args.purposeful_turn_window_m,
            min_event_turn_deg=args.purposeful_min_event_turn_deg,
            obstacle_influence_m=args.purposeful_obstacle_influence_m)
        features.update(event_metrics)
        features['euclidean_distance_m'] = math.hypot(goal[0]-start[0], goal[1]-start[1])
        features['min_clearance_m'] = float(route_clearance['dense_min_clearance_m'])
        features['mean_clearance_m'] = float(route_clearance['dense_mean_clearance_m'])
        features['direct_line_min_clearance_m'] = direct_clearance
        features['direct_line_collision'] = direct_collision
        features['detour_ratio'] = float(features['path_length_m'] / max(features['euclidean_distance_m'], 1e-6))
        sampled_route = _gmm_resample_polyline(route, args.audit_spacing_m)
        route_clearance_values = [
            float(clearance[int((p[1] - oy) / res), int((p[0] - ox) / res)])
            for p in sampled_route[0]
            if 0 <= int((p[1] - oy) / res) < clearance.shape[0] and
               0 <= int((p[0] - ox) / res) < clearance.shape[1]]
        features['near_obstacle_fraction'] = float(np.mean(
            [value <= args.purposeful_obstacle_influence_m for value in route_clearance_values])) if route_clearance_values else 0.0
        if direct_collision:
            features['obstacle_interaction'] = 'direct_line_blocked'
        elif direct_clearance < args.purposeful_preferred_clearance_m:
            features['obstacle_interaction'] = 'direct_line_too_close'
        elif features['near_obstacle_fraction'] > 0.20:
            features['obstacle_interaction'] = 'near_obstacle'
        else:
            features['obstacle_interaction'] = 'open_space'

        label, reason = _purposeful_route_label(
            features, events, direct,
            hard_clearance_m=hard_clearance)
        features['purposeful_class_reason'] = reason

        if label == 'reject':
            rejects[reason] += 1
            continue
        if label not in samples:
            rejects['unknown_class'] += 1
            continue

        # Enforce class-specific limits only after morphology has been checked.
        # A non-v-complex candidate cannot use the expanded v_complex limits.
        if label == 'v_complex':
            class_max_distance = args.purposeful_v_max_goal_distance_m
            class_max_length = args.purposeful_v_max_path_length_m
        else:
            class_max_distance = args.max_goal_distance
            class_max_length = args.max_path_length
        if features['euclidean_distance_m'] > class_max_distance:
            rejects['class_goal_distance_limit'] += 1
            continue
        if features['path_length_m'] > class_max_length:
            rejects['class_path_length_limit'] += 1
            continue
        if len(samples[label]) >= args.purposeful_per_class:
            rejects[f'class_quota_full_{label}'] += 1
            continue
        # Keep one route per endpoint pair; no artificial waypoints are inserted.
        samples[label].append({
            'id': sum(len(v) for v in samples.values()) + 1,
            'purposeful_class': label,
            'reason': reason,
            'start': [float(start[0]), float(start[1])],
            'goal': [float(goal[0]), float(goal[1])],
            'poses': [[float(p[0]), float(p[1])] for p in route],
            'metrics': features,
            'turn_events': events,
            'near_optimal_baseline_length_m': float(baseline_length),
        })

        if total_attempts % 50 == 0:
            counts = ' '.join(f'{name}={len(samples[name])}/{args.purposeful_per_class}' for name in classes)
            print(f'[{args.scenario}] purposeful probe {total_attempts}/{args.purposeful_max_attempts}: {counts}', flush=True)

    out_dir = Path(args.gmm_output_dir) if args.gmm_output_dir else Path(args.output_dir) / args.scenario / f'purposeful_probe_seed_{args.purposeful_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = [item for name in classes for item in samples[name]]
    csv_path = out_dir / 'purposeful_paths.csv'
    csv_fields = [
        'id', 'purposeful_class', 'reason', 'start_x', 'start_y', 'goal_x', 'goal_y',
        'euclidean_distance_m', 'path_length_m', 'detour_ratio', 'sinuosity',
        'max_turn_deg', 'total_turn_deg', 'turn_event_count', 'supported_turn_count',
        'supported_turn_fraction', 'turn_sign_changes', 'max_lateral_deviation_m',
        'max_backtrack_m', 'min_clearance_m', 'mean_clearance_m',
        'supported_maneuver_count',
        'obstacle_interaction', 'direct_line_min_clearance_m', 'direct_line_collision', 'near_obstacle_fraction',
        'near_optimal_baseline_length_m', 'purposeful_class_reason']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in flat:
            f = item['metrics']
            writer.writerow({
                'id': item['id'], 'purposeful_class': item['purposeful_class'],
                'reason': item['reason'], 'start_x': item['start'][0],
                'start_y': item['start'][1], 'goal_x': item['goal'][0],
                'goal_y': item['goal'][1],
                **{key: f.get(key, '') for key in csv_fields if key in f},
                'near_optimal_baseline_length_m': item['near_optimal_baseline_length_m'],
            })
    json_path = out_dir / 'purposeful_paths.json'
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'scenario': args.scenario, 'seed': args.purposeful_seed,
            'map_yaml': context['yaml'], 'diagnostic_only': True,
            'constraints': {'euclidean_min_m': args.min_goal_distance,
                            'euclidean_max_m': args.max_goal_distance,
                            'path_length_max_m': args.max_path_length,
                            'v_complex_goal_distance_max_m': args.purposeful_v_max_goal_distance_m,
                            'v_complex_path_length_max_m': args.purposeful_v_max_path_length_m,
                            'clearance_min_m': hard_clearance,
                            'hard_footprint_clearance_m': hard_clearance,
                            'preferred_clearance_m': args.purposeful_preferred_clearance_m,
                            'max_near_optimal_ratio': args.purposeful_max_near_optimal_ratio},
            'attempts': total_attempts, 'rejections': dict(rejects),
            'paths': flat,
        }, handle, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150, facecolor='white')
    colors = {'straight': '#2878B5', 'semi_complex': '#4A9D62',
              'complex': '#E39C37', 'v_complex': '#C84D4D'}
    for name, ax in zip(classes, axes.flat):
        _draw_metric_map(ax, context)
        for item in samples[name]:
            pts = np.asarray(item['poses'], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=colors[name], linewidth=1.5, alpha=0.9)
            ax.plot(item['start'][0], item['start'][1], 'o', color='#189B72', markersize=3)
            ax.plot(item['goal'][0], item['goal'][1], 'x', color='black', markersize=4)
        ax.set_title(f'{name}: {len(samples[name])}/{args.purposeful_per_class}')
        ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    plot_path = out_dir / 'purposeful_paths_by_class.png'
    fig.savefig(plot_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    report = [f'# Purposeful path probe: {args.scenario}', '',
              'Diagnostic only. This mode does not modify production pools or command the robot.',
              f'- Map: `{context["yaml"]}`', f'- Seed: {args.purposeful_seed}',
              f'- Attempts: {total_attempts}/{args.purposeful_max_attempts}',
              f'- Elapsed: {time.time()-started:.1f} s',
              f'- Constraints: Euclidean {args.min_goal_distance:.2f}-{args.max_goal_distance:.2f} m; '
              f'length <= {args.max_path_length:.2f} m for straight/semi/complex; '
              f'v_complex euclidean <= {args.purposeful_v_max_goal_distance_m:.2f} m and '
              f'path <= {args.purposeful_v_max_path_length_m:.2f} m; '
              f'hard footprint clearance >= {hard_clearance:.2f} m; '
              f'soft preferred clearance {args.purposeful_preferred_clearance_m:.2f} m; '
              f'near-optimal ratio <= {args.purposeful_max_near_optimal_ratio:.2f}', '',
              '| Class | Accepted | Target |', '|---|---:|---:|']
    for name in classes:
        report.append(f'| {name} | {len(samples[name])} | {args.purposeful_per_class} |')
    report += ['', '## Rejections', '']
    report += [f'- `{name}`: {count}' for name, count in rejects.most_common()]
    report += ['', '## Route design rule', '',
               'A direct route is accepted whenever it satisfies the hard robot-center clearance. Preferred clearance affects the soft route cost, not the class trigger. A* paths are line-of-sight simplified with dense checks; too-close paths are rejected. Complex classes require obstacle-supported maneuvers, and a U/return requires measurable backtracking or multiple obstacle interactions. Free-space zigzags are rejected. The regular maximum distance and path length stay unchanged for straight, semi_complex and complex. Only a candidate proven to be v_complex can use the expanded v_complex distance/path-length limits. The output is a candidate study, not a production path pool.',
               '', f'- CSV: `{csv_path.name}`', f'- JSON: `{json_path.name}`', f'- Plot: `{plot_path.name}`']
    report_path = out_dir / 'purposeful_paths_report.md'
    report_path.write_text('\n'.join(report) + '\n', encoding='utf-8')

    print(f'Purposeful probe complete: {len(flat)} paths in {time.time()-started:.1f}s')
    for name in classes:
        print(f'  {name}: {len(samples[name])}/{args.purposeful_per_class}')
    print(f'Report: {report_path}')
    print(f'Plot: {plot_path}')
    return 0 if all(len(samples[n]) >= args.purposeful_per_class for n in classes) else 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--upgrade-existing', action='store_true',
                        help='filter and relabel an existing scenario pool in place')
    parser.add_argument('--analyze-existing', action='store_true',
                        help='analyze and visualize an existing scenario pool with current geometry rules')
    parser.add_argument('--gmm-experiment', action='store_true',
                        help='compare experimental GMM labels offline without modifying the pool')
    parser.add_argument('--manual-review', action='store_true',
                        help='select a seeded sample from every predicted GMM class for manual labeling')
    parser.add_argument('--geometry-audit', action='store_true',
                        help='compare regenerated raw A* paths, smoothed paths, and stored pool routes offline')
    parser.add_argument('--directed-probe', action='store_true',
                        help='test via-point templates for straight, arc, S-curve, and U-turn/zigzag routes offline')
    parser.add_argument('--purposeful-probe', action='store_true',
                        help='generate near-optimal routes whose turns are caused by obstacles; diagnostic only')
    parser.add_argument('--purposeful-per-class', type=int, default=10)
    parser.add_argument('--purposeful-max-attempts', type=int, default=3000)
    parser.add_argument('--purposeful-seed', type=int, default=20261013)
    parser.add_argument('--purposeful-preferred-clearance-m', type=float, default=0.45,
                        help='soft clearance target used in route cost, not a class trigger')
    parser.add_argument('--purposeful-hard-clearance-m', type=float, default=0.35,
                        help='minimum obstacle clearance for the robot center; set from footprint radius plus margin')
    parser.add_argument('--purposeful-endpoint-obstacle-band-m', type=float, default=0.65,
                        help='sample endpoint cells within this band beyond hard clearance')
    parser.add_argument('--purposeful-endpoint-obstacle-bias', type=float, default=0.75,
                        help='probability of drawing endpoints near obstacles after straight quota is full')
    parser.add_argument('--purposeful-v-candidate-bias', type=float, default=0.70,
                        help='probability of sampling short, obstacle-adjacent endpoint pairs while v_complex is missing')
    parser.add_argument('--purposeful-v-distance-band-m', type=float, default=0.55,
                        help='legacy setting retained for command-line compatibility')
    parser.add_argument('--purposeful-v-max-goal-distance-m', type=float, default=3.6,
                        help='maximum start-goal Euclidean distance only for v_complex candidates')
    parser.add_argument('--purposeful-v-max-path-length-m', type=float, default=4.2,
                        help='maximum travelled path length only for v_complex candidates')
    parser.add_argument('--purposeful-obstacle-influence-m', type=float, default=0.80)
    parser.add_argument('--purposeful-turn-window-m', type=float, default=0.50)
    parser.add_argument('--purposeful-min-event-turn-deg', type=float, default=18.0)
    parser.add_argument('--purposeful-clearance-weight', type=float, default=3.0)
    parser.add_argument('--purposeful-risk-weight', type=float, default=0.60)
    parser.add_argument('--purposeful-max-near-optimal-ratio', type=float, default=1.18)
    parser.add_argument('--audit-count', type=int, default=200,
                        help='number of existing pool paths to inspect; 0 means all paths')
    parser.add_argument('--audit-seed', type=int, default=20261013)
    parser.add_argument('--audit-ids', default='',
                        help='comma-separated pool path IDs to include explicitly in the geometry audit')
    parser.add_argument('--audit-spacing-m', type=float, default=0.05)
    parser.add_argument('--audit-turn-window-m', type=float, default=0.50)
    parser.add_argument('--probe-per-pattern', type=int, default=10)
    parser.add_argument('--probe-attempts-per-pattern', type=int, default=500)
    parser.add_argument('--probe-waypoint-snap-m', type=float, default=0.30)
    parser.add_argument('--probe-seed', type=int, default=20261013)
    parser.add_argument('--manual-per-class', type=int, default=10,
                        help='number of eligible paths per predicted class (default: 10)')
    parser.add_argument('--manual-seed', type=int, default=20261013,
                        help='seed for repeatable manual-review selection')
    parser.add_argument('--manual-review-overwrite', action='store_true',
                        help='allow replacing an existing manual_review_labels.csv (can discard human labels)')
    parser.add_argument('--gmm-turn-window-m', type=float, default=0.60,
                        help='physical path distance used for turn measurements in GMM experiment')
    parser.add_argument('--gmm-resample-spacing-m', type=float, default=0.05,
                        help='arc-length spacing used by experimental GMM features')
    parser.add_argument('--gmm-seed', type=int, default=1001)
    parser.add_argument('--gmm-min-confidence', type=float, default=0.50)
    parser.add_argument('--gmm-output-dir', default='',
                        help='output directory for GMM experiment; default <scenario>/analysis_gmm_experiment')
    parser.add_argument('--input-dir', default=None,
                        help='data root containing <scenario>/path_pool.json; defaults to --output-dir')
    parser.add_argument('--num-paths', type=int, default=3000)
    parser.add_argument('--quota', default='25,25,25,25')
    parser.add_argument('--room-radius', type=float, default=0.45)
    parser.add_argument('--min-room-area', type=float, default=0.5)
    parser.add_argument('--min-goal-distance', type=float, default=2.0)
    parser.add_argument('--max-goal-distance', type=float, default=3.0)
    parser.add_argument('--max-path-length', type=float, default=3.0)
    parser.add_argument('--min-clearance', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--min-paths-per-class', type=int, default=50)
    parser.add_argument('--maps-root', default=resolve_default_maps_root())
    parser.add_argument(
        '--output-dir',
        default='src/Navigation/mapless_navigation/dataset/data')
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--smooth-existing-json', default='',
                        help='run offline smoothing on an existing routes JSON; does not generate or overwrite the pool')
    parser.add_argument('--smooth-output-dir', default='',
                        help='new, non-existing output directory required with --smooth-existing-json')
    parser.add_argument('--smooth-planner', choices=('preserve', 'theta-star'), default='preserve',
                        help='route source passed to smoother.py when --smooth-existing-json is used')
    parser.add_argument('--smooth-target-ids', default='',
                        help='optional comma-separated route IDs for the smoothing stage')
    args = parser.parse_args()

    selected_modes = sum(bool(x) for x in (args.upgrade_existing, args.analyze_existing, args.gmm_experiment, args.manual_review, args.geometry_audit, args.directed_probe, args.purposeful_probe, bool(args.smooth_existing_json)))
    if selected_modes > 1:
        parser.error('analysis/generation modes are mutually exclusive')
    if args.smooth_output_dir and not args.smooth_existing_json:
        parser.error('--smooth-output-dir requires --smooth-existing-json')
    if args.smooth_existing_json:
        if not args.smooth_output_dir:
            parser.error('--smooth-output-dir is required with --smooth-existing-json')
        if not os.path.isfile(args.smooth_existing_json):
            parser.error(f'input JSON not found: {args.smooth_existing_json}')
        if os.path.exists(args.smooth_output_dir):
            parser.error(f'smoothing output directory already exists: {args.smooth_output_dir}')
        smoother_path = Path(__file__).resolve().parent / 'path_smoothing' / 'smoother.py'
        if not smoother_path.is_file():
            parser.error(f'unified smoother module not found: {smoother_path}')
        command = [
            sys.executable, str(smoother_path),
            '--scenario', args.scenario,
            '--input-json', str(Path(args.smooth_existing_json).expanduser().resolve()),
            '--generator', str(Path(__file__).resolve()),
            '--maps-root', str(Path(args.maps_root).expanduser().resolve()),
            '--output-dir', str(Path(args.smooth_output_dir).expanduser().resolve()),
            '--planner', args.smooth_planner,
        ]
        if args.smooth_target_ids.strip():
            command.extend(['--target-ids', args.smooth_target_ids.strip()])
        print('Delegating existing-route smoothing to path_smoothing/smoother.py', flush=True)
        return subprocess.run(command, check=False).returncode
    if args.manual_per_class < 1:
        parser.error('--manual-per-class must be positive')
    if args.manual_review:
        if args.gmm_turn_window_m <= 0 or args.gmm_resample_spacing_m <= 0:
            parser.error('GMM turn window and resampling spacing must be positive')
        return analyze_gmm_existing_pool(args, manual_review_only=True)
    if args.purposeful_probe:
        if (args.purposeful_per_class < 1 or args.purposeful_max_attempts < 1 or
                args.purposeful_preferred_clearance_m < args.min_clearance or
                args.purposeful_hard_clearance_m < args.min_clearance or
                args.purposeful_endpoint_obstacle_band_m < 0 or
                args.purposeful_v_distance_band_m <= 0 or
                args.purposeful_v_max_goal_distance_m < args.max_goal_distance or
                args.purposeful_v_max_path_length_m < args.purposeful_v_max_goal_distance_m or
                not 0.0 <= args.purposeful_v_candidate_bias <= 1.0 or
                not 0.0 <= args.purposeful_endpoint_obstacle_bias <= 1.0 or
                args.purposeful_obstacle_influence_m <= 0 or
                args.purposeful_turn_window_m <= 0 or
                args.purposeful_clearance_weight < 0 or
                args.purposeful_max_near_optimal_ratio < 1.0):
            parser.error('invalid purposeful-probe parameter values')
        return purposeful_candidate_probe(args)
    if args.geometry_audit:
        if args.audit_count < 0 or args.audit_spacing_m <= 0 or args.audit_turn_window_m <= 0:
            parser.error('audit count must be >= 0 and distances must be positive')
        return geometry_audit_existing_pool(args)
    if args.directed_probe:
        if args.probe_per_pattern < 1 or args.probe_attempts_per_pattern < 1:
            parser.error('directed probe limits must be positive')
        if args.probe_waypoint_snap_m <= 0:
            parser.error('probe waypoint snap distance must be positive')
        return directed_candidate_probe(args)
    if args.gmm_experiment:
        if args.gmm_turn_window_m <= 0 or args.gmm_resample_spacing_m <= 0:
            parser.error('GMM turn window and resampling spacing must be positive')
        return analyze_gmm_existing_pool(args)
    if args.analyze_existing:
        return analyze_existing_pool(args)
    if args.upgrade_existing:
        return upgrade_existing_pool(args)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    maps_root = os.path.abspath(args.maps_root)
    map_folder = find_map_folder(maps_root, args.scenario)
    folder_name = os.path.basename(map_folder)
    map_yaml = os.path.join(
        map_folder, f'map_{folder_name}_prohibition.yaml')

    print(f'scenario: {args.scenario}')
    print(f'map folder: {map_folder}')
    print(f'skimage available: {HAS_SKIMAGE}')

    map_arr, res, ox, oy, free_thresh, occ_thresh, negate = load_map(map_yaml)
    free = build_free_mask(map_arr, free_thresh, occ_thresh, negate)

    n_free = int(free.sum())
    print(f'map size: {map_arr.shape[1]}x{map_arr.shape[0]}')
    print(f'pixel range: min={map_arr.min()} max={map_arr.max()} '
          f'mean={map_arr.mean():.1f}')
    print(f'thresholds: free<{free_thresh} occ>{occ_thresh} negate={negate}')
    print(f'path constraints: euclidean=[{args.min_goal_distance:.2f}, {args.max_goal_distance:.2f}] m, '
          f'path_length<={args.max_path_length:.2f} m, min_clearance>={args.min_clearance:.2f} m')
    print(f'free cells: {n_free}')
    if n_free < 100:
        print('ERROR: too few free cells')
        print('hint: check map yaml thresholds or image polarity')
        return 1

    rooms, dist_px = build_rooms(
        free, res, ox, oy, args.room_radius, args.min_room_area)
    print(f'rooms: {len(rooms)}')

    free_cell_list = []
    for cells in rooms.values():
        free_cell_list.extend(cells)

    if not free_cell_list:
        print('ERROR: no free cells after room filtering')
        return 1

    clearance_grid = distance_transform_edt(free) * res

    quota = [int(x) for x in args.quota.split(',')]
    if len(quota) != 4 or sum(quota) == 0:
        print('ERROR: bad quota, expected 4 positive integers')
        return 1

    paths = []
    attempts = 0
    max_attempts = args.num_paths * 10
    t0 = time.time()
    last_report = t0
    report_interval_s = 1.0
    progress_width = 90

    while len(paths) < args.num_paths and attempts < max_attempts:
        attempts += 1

        now = time.time()
        if now - last_report >= report_interval_s:
            elapsed = now - t0
            rate = len(paths) / elapsed if elapsed > 0 else 0.0
            remaining = args.num_paths - len(paths)
            eta = remaining / rate if rate > 0.01 else 0.0
            msg = (f'[{args.scenario}] pool: {len(paths)}/{args.num_paths} '
                   f'paths | {elapsed:.0f}s | eta {eta:.0f}s')
            sys.stdout.write(f'\r{msg:<{progress_width}}')
            sys.stdout.flush()
            last_report = now

        start = rng.choice(free_cell_list)
        goal = None
        for _ in range(50):
            cand = rng.choice(free_cell_list)
            d = math.hypot(cand[0] - start[0], cand[1] - start[1])
            if args.min_goal_distance <= d <= args.max_goal_distance:
                goal = cand
                break
        if goal is None:
            continue

        path = plan_path(free, start, goal, ox, oy, res)
        if path is None:
            continue

        feats = compute_features(path, clearance_grid, ox, oy, res)
        if feats is None or feats['min_clearance_m'] < args.min_clearance:
            continue
        if (feats['path_length_m'] > args.max_path_length or
                feats['path_length_m'] + 1e-6 < math.hypot(goal[0] - start[0], goal[1] - start[1])):
            continue

        paths.append({
            'start': [round(float(start[0]), 3), round(float(start[1]), 3)],
            'goal': [round(float(goal[0]), 3), round(float(goal[1]), 3)],
            'poses': [[round(float(x), 3), round(float(y), 3)]
                      for x, y in path],
            'metrics': {k: round(v, 4) for k, v in feats.items()},
        })

    sys.stdout.write('\r' + ' ' * progress_width + '\r')
    sys.stdout.flush()
    print(f'[{args.scenario}] pool: {len(paths)} paths '
          f'generated in {time.time() - t0:.1f}s', flush=True)
    if len(paths) < args.num_paths:
        print(f'ERROR: requested {args.num_paths} safe paths, but generated {len(paths)}. '
              'Increase the attempt budget or inspect map constraints.')
        return 1

    features_list = [p['metrics'] for p in paths]
    classes, scores, classifier_meta, boundaries, signatures = classify_by_geometry(
        features_list, quota)

    for i, p in enumerate(paths):
        p['id'] = i
        p['class'] = classes[i]
        p['score'] = float(scores[i])

    class_order = {name: i for i, name in enumerate(CLASS_NAMES)}
    paths.sort(key=lambda p: (class_order[p['class']], p['score']))

    distribution = Counter(p['class'] for p in paths)
    quota_dict = dict(zip(CLASS_NAMES, quota))
    under_provisioned = {
        name: int(distribution.get(name, 0))
        for name in CLASS_NAMES
        if distribution.get(name, 0) < max(args.min_paths_per_class, 2 * quota_dict[name])
    }
    if under_provisioned:
        print(f'ERROR: pool class counts are insufficient: {under_provisioned}')
        print('Existing pool files have not been replaced.')
        return 1

    output_dir = os.path.abspath(args.output_dir)
    scenario_dir = os.path.join(output_dir, args.scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    pool_path = os.path.join(scenario_dir, 'path_pool.json')

    poses_by_id = {p['id']: p['poses'] for p in paths}
    for p in paths:
        del p['poses']

    pool = {
        'pool_version': POOL_VERSION,
        'scenario': args.scenario,
        'generated_at': datetime.now().isoformat(),
        'seed': args.seed,
        'num_paths': len(paths),
        'source_num_paths': len(paths),
        'constraints': {
            'min_goal_distance_m': args.min_goal_distance,
            'max_goal_distance_m': args.max_goal_distance,
            'max_path_length_m': args.max_path_length,
            'min_clearance_m': args.min_clearance,
        },
        'classification_scope': 'geometry_only',
        'classifier': CLASSIFIER_NAME,
        'classification_method': 'fixed_geometric_rules',
        'quota': quota_dict,
        'geometry_classifier': classifier_meta,
        'class_boundaries': boundaries,
        'class_signatures': signatures,
        'distribution': dict(distribution),
        'paths': paths,
    }

    poses_path = pool_path.replace('.json', '_poses.json')
    pool_tmp = pool_path + '.tmp'
    poses_tmp = poses_path + '.tmp'
    with open(pool_tmp, 'w') as f:
        json.dump(pool, f, indent=1)
    with open(poses_tmp, 'w') as f:
        json.dump(poses_by_id, f)
    os.replace(pool_tmp, pool_path)
    os.replace(poses_tmp, poses_path)

    print(f'wrote {pool_path}')
    print('distribution:')
    for name in CLASS_NAMES:
        print(f'  {name}: {distribution.get(name, 0)}')

    print('signatures:')
    for name in CLASS_NAMES:
        s = signatures[name]
        print(f'  {name} (n={s["n"]})')
        for fn in FEATURE_NAMES:
            if fn in s:
                m, sd = s[fn]
                print(f'    {fn}: {m:.4f} +/- {sd:.4f}')

    if args.skip_validation:
        return 0

    print('validation:')
    failures = validate_pool(pool, quota_dict, args.min_paths_per_class)
    if failures:
        print(f'FAILED: {failures} classes under-provisioned')
        return 1

    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())