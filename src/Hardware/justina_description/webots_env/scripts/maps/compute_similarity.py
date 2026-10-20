import sys

#!/usr/bin/env python3
"""
Compute pairwise Wasserstein distances between occupancy maps.

This script computes a distance matrix between all environments using
a selected optimal transport metric. It supports three methods:
  - sliced   : Sliced-Wasserstein (fast, approximate)
  - sinkhorn : Entropy-regularized Wasserstein (medium speed, accurate)
  - emd      : Exact Wasserstein-1 (slow, exact)

All parameters can be tuned at the top of this file in the "GLOBAL CONFIGURATION"
section. You can also override them via command-line arguments.

Usage examples:
  ./compute_similarity.py                           # Uses global config
  ./compute_similarity.py --method emd --max_points 800
  ./compute_similarity.py --method sinkhorn --reg 0.001 --max_points 4000
"""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# ============================================================
#  GLOBAL CONFIGURATION
# ============================================================

METHOD = 'sinkhorn'          # 'sliced', 'sinkhorn', or 'emd'
MAX_POINTS = 4000            # Points per map
ALIGNED = True               # Read origin from YAML (validated automatically)
RESOLUTION = 0.01            # m/pixel (only if ALIGNED=True)
SEED = 42                    # For reproducible subsampling
BOOTSTRAP = 1                # Number of runs (1 = no bootstrap)
CROSS_CHECK = False          # Verify Sinkhorn vs EMD on a few pairs
REG = 0.001                  # Entropy reg (scaled internally by global distance)
SINKHORN_ITERS = 200
N_PROJECTIONS = 200

OUTPUT_DIR = None
OUTPUT_PREFIX = 'wasserstein_distances'

# ============================================================
#  END OF GLOBAL CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
WEBOTS_ROOT = SCRIPT_DIR.parent.parent
MAPS_BASE = WEBOTS_ROOT / 'environments' / 'maps'

try:
    import ot
except ImportError:
    print("ERROR: POT library not installed. Run: pip install POT")
    exit(1)


def load_occupied_points(env_name, aligned=True, resolution=RESOLUTION):
    """Load occupied cell coordinates (Nx2) in meters from the map."""
    img_path = MAPS_BASE / env_name / f"map_{env_name}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Map not found for {env_name}")

    grid = np.array(Image.open(img_path))
    h, w = grid.shape
    rows, cols = np.where(grid == 0)

    if len(rows) == 0:
        return np.array([])

    if aligned:
        # Coordinates relative to image center (assuming aligned maps)
        center_x = w / 2.0
        center_y = h / 2.0
        pts_x = (cols - center_x) * resolution
        pts_y = (center_y - rows) * resolution
    else:
        # Read origin and resolution from YAML (non‑aligned maps)
        yaml_path = MAPS_BASE / env_name / f"map_{env_name}.yaml"
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        origin_x, origin_y, _ = metadata['origin']
        res = metadata['resolution']
        pts_x = origin_x + cols * res
        pts_y = origin_y + (h - rows) * res

    return np.column_stack((pts_x, pts_y)).astype(np.float64)


def subsample_points(pts, max_points, rng=None):
    if len(pts) <= max_points:
        return pts
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), max_points, replace=False)
    return pts[idx]


def compute_sliced(pts_a, pts_b, n_projections):
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('inf')
    return ot.sliced_wasserstein_distance(pts_a, pts_b, n_projections=n_projections)

def compute_sinkhorn(pts_a, pts_b, reg, numItermax, global_scale=None):
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('inf')
    M = ot.dist(pts_a, pts_b, metric='euclidean')
    if global_scale is not None:
        M = M / global_scale
        reg_actual = reg
    else:
        reg_actual = reg
    a = np.ones(len(pts_a)) / len(pts_a)
    b = np.ones(len(pts_b)) / len(pts_b)
    # Sinkhorn returns the cost; we want the distance (sqrt for W2 if using squared euclidean)
    # But here metric='euclidean' gives W1 directly if we don't square M.
    return ot.sinkhorn2(a, b, M, reg_actual, numItermax=numItermax)

def compute_emd(pts_a, pts_b, global_scale=None):
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('inf')
    M = ot.dist(pts_a, pts_b, metric='euclidean')
    if global_scale is not None:
        M = M / global_scale
    a = np.ones(len(pts_a)) / len(pts_a)
    b = np.ones(len(pts_b)) / len(pts_b)
    return ot.emd2(a, b, M)

def validate_alignment(envs):
    """Verify that all maps share the same size and origin."""
    sizes = set()
    origins = set()
    for env in envs:
        img_path = MAPS_BASE / env / f"map_{env}.png"
        yaml_path = MAPS_BASE / env / f"map_{env}.yaml"
        if img_path.exists():
            img = Image.open(img_path)
            sizes.add(img.size)
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                y = yaml.safe_load(f)
            origins.add(tuple(y.get("origin", [0, 0, 0])))
            
    if len(sizes) > 1:
        raise SystemExit(f"ERROR: Maps have different sizes: {sizes}")
    if len(origins) > 1:
        print(f"WARNING: Origins vary ({len(origins)} distinct). Consider --aligned false.")
        return False
    return True

def bootstrap_distance(pts_a, pts_b, method, args, global_scale, n_boot, seed, pair_label=""):
    rng = np.random.default_rng(seed)
    vals = []
    t_pair = time.time()
    for k in range(n_boot):
        a = subsample_points(pts_a, args.max_points, rng)
        b = subsample_points(pts_b, args.max_points, rng)
        if method == 'sliced':
            v = compute_sliced(a, b, args.n_projections)
        elif method == 'sinkhorn':
            v = compute_sinkhorn(a, b, args.reg, args.sinkhorn_iters, global_scale)
        else:
            v = compute_emd(a, b, global_scale)
        vals.append(v)
        # Live line: overwritten in-place with \r, no newline until the end
        sys.stdout.write(f"\r    {pair_label} iter {k+1:02d}/{n_boot}  "
                         f"({time.time()-t_pair:5.1f}s)  ")
        sys.stdout.flush()
    # Close the live line with a newline so the next print starts fresh
    sys.stdout.write("\n")
    sys.stdout.flush()
    vals = np.array(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "cv": float(vals.std() / vals.mean()) if vals.mean() > 0 else 0.0
    }

def main():
    parser = argparse.ArgumentParser(description='Compute pairwise Wasserstein distances.')
    parser.add_argument('--method', choices=['sliced', 'sinkhorn', 'emd'], default=METHOD)
    parser.add_argument('--max_points', type=int, default=MAX_POINTS)
    parser.add_argument('--aligned', type=lambda x: x.lower() in ['true', '1', 'yes'], default=ALIGNED)
    parser.add_argument('--resolution', type=float, default=RESOLUTION)
    parser.add_argument('--n_projections', type=int, default=N_PROJECTIONS)
    parser.add_argument('--reg', type=float, default=REG)
    parser.add_argument('--sinkhorn_iters', type=int, default=SINKHORN_ITERS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--bootstrap', type=int, default=BOOTSTRAP, help='Number of runs for CI (1 = no bootstrap)')
    parser.add_argument('--cross_check', action='store_true', help='Compare Sinkhorn with EMD')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--output_prefix', type=str, default=OUTPUT_PREFIX)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "similarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean previous outputs (keep behavioral_results.csv and archive/)
    for old_file in out_dir.glob(f"{args.output_prefix}_{args.method}_*.csv"):
        old_file.unlink()
    for old_file in out_dir.glob(f"{args.output_prefix}_{args.method}_*.json"):
        old_file.unlink()
    print(f"Cleaned old {args.method} outputs in {out_dir}")

    envs = sorted([
        d.name for d in MAPS_BASE.iterdir()
        if d.is_dir() and (d / f"map_{d.name}.png").exists()
    ])
    if not envs:
        print("No maps found.")
        return

    import os as _os
    _filter = _os.environ.get("COMPUTE_SIM_SCENARIOS", "").strip()
    if _filter:
        _wanted = [x.strip() for x in _filter.split(",") if x.strip()]
        _missing = [w for w in _wanted if w not in envs]
        if _missing:
            print(f"ERROR: scenarios not found: {_missing}")
            return
        envs = _wanted
        print(f"Filtered to {len(envs)} scenarios: {envs}")

    if args.aligned:
        print("Validating map alignment...")
        validate_alignment(envs)

    n = len(envs)
    
    # Pre-load points
    print(f"Found {n} environments. Loading and subsampling...")
    pts_dict = {}
    for k, env in enumerate(envs, 1):
        pts = load_occupied_points(env, aligned=args.aligned, resolution=args.resolution)
        pts_dict[env] = pts  # Store FULL points; bootstrap will subsample per iter
        print(f"  loaded {k}/{n}: {env} ({len(pts)} pts)", flush=True)

    # Compute global scale (for normalization)
    all_pts = np.vstack([pts_dict[e] for e in envs if len(pts_dict[e]) > 0])
    rng_scale = np.random.default_rng(args.seed)
    idx1 = rng_scale.integers(0, len(all_pts), 2000)
    idx2 = rng_scale.integers(0, len(all_pts), 2000)
    global_scale = float(np.linalg.norm(all_pts[idx1] - all_pts[idx2], axis=1).max())
    print(f"Global scale (max distance): {global_scale:.4f} m")

    total_pairs = n * (n - 1) // 2
    print(f"Computing {total_pairs} pairwise distances (bootstrap={args.bootstrap})...", flush=True)
    
    # Initialize matrices for bootstrap statistics
    stats = {k: np.zeros((n, n)) for k in ["mean", "std", "ci_low", "ci_high", "cv"]}
    start_time = time.time()
    done = 0

    for i, env_i in enumerate(envs):
        for j in range(i + 1, n):
            env_j = envs[j]
            
            pair_label = f"[{env_i[:6]}<->{env_j[:6]}]"
            res = bootstrap_distance(
                pts_dict[env_i], pts_dict[env_j], args.method, args, 
                global_scale, args.bootstrap, args.seed + done,
                pair_label=pair_label
            )
            
            for k in stats:
                stats[k][i, j] = res[k]
                stats[k][j, i] = res[k]
                
            if args.cross_check and args.bootstrap == 1 and done % 10 == 0:
                # Check EMD vs Sinkhorn for a few pairs
                emd_val = compute_emd(pts_dict[env_i], pts_dict[env_j], global_scale)
                rel_err = abs(res["mean"] - emd_val) / max(emd_val, 1e-9)
                print(f"    [Cross-check] Sinkhorn={res['mean']:.4f}, EMD={emd_val:.4f}, RelErr={rel_err:.1%}")

            done += 1
            if True: 
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total_pairs - done) / rate if rate > 0 else 0.0
                print(f"  [{done:4d}/{total_pairs}] {env_i[:6]} <-> {env_j[:6]}: "
                      f"{res['mean']:.4f} +/- {res['std']:.4f}  (eta={eta:.0f}s)", flush=True)

    # Save outputs
    header = ",".join(envs)
    for k in ["mean", "std", "ci_low", "ci_high"]:
        out_file = out_dir / f"{args.output_prefix}_{args.method}_{k}.csv"
        np.savetxt(out_file, stats[k], delimiter=",", header=header, comments="", fmt="%.4f")
        print(f"Saved {k} matrix to: {out_file}")
        
    # Save metadata sidecar
    meta = {
        "method": args.method,
        "max_points": args.max_points,
        "aligned": args.aligned,
        "resolution": args.resolution,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "global_scale": global_scale,
        "reg": args.reg,
        "timestamp": datetime.datetime.now().isoformat(),
        "envs": envs,
        "stats_summary": {
            "mean_range": [float(stats["mean"][np.triu_indices_from(stats["mean"], k=1)].min()), 
                           float(stats["mean"][np.triu_indices_from(stats["mean"], k=1)].max())],
            "mean_cv_avg": float(stats["cv"][np.triu_indices_from(stats["cv"], k=1)].mean())
        }
    }
    meta_file = out_dir / f"{args.output_prefix}_{args.method}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_file}")

if __name__ == "__main__":
    main()