#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

MAPS_BASE = Path(__file__).resolve().parent.parent.parent / "environments" / "maps"
SIM_DIR = Path(__file__).resolve().parent / "similarity"
CSV_FILE = SIM_DIR / "wasserstein_distances_sinkhorn_mean.csv"
BEHAV_FILE = SIM_DIR / "behavioral_results.csv"

def load_grid(env):
    return np.array(Image.open(MAPS_BASE / env / f"map_{env}.png"))

def occupied_points(grid, resolution=0.01):
    rows, cols = np.where(grid == 0)
    cx, cy = grid.shape[1] / 2.0, grid.shape[0] / 2.0
    return np.column_stack([(cols-cx)*resolution, (cy-rows)*resolution]).astype(np.float64)

def subsample(pts, n=4000, rng=None):
    if len(pts) <= n: return pts
    if rng is None: rng = np.random.default_rng(0)
    return pts[rng.choice(len(pts), n, replace=False)]

def w_dist(a, b, reg=0.001, iters=200):
    M = ot.dist(a, b, metric="euclidean")
    M = M / M.max()
    ua = np.ones(len(a)) / len(a)
    ub = np.ones(len(b)) / len(b)
    return ot.sinkhorn2(ua, ub, M, reg, numItermax=iters)

def check_sanity(pts_dict):
    print("=== Test 1: Sanity (Identity & Symmetry) ===")
    envs = list(pts_dict.keys())
    passed = True
    for e in envs[:3]:
        d = w_dist(pts_dict[e], pts_dict[e])
        if d > 1e-2: passed = False
        print(f"  d({e[:4]}, {e[:4]}) = {d:.2e} (Expected ~0)")
    for i in range(min(3, len(envs))):
        a, b = envs[i], envs[(i+1)%len(envs)]
        d_ab = w_dist(pts_dict[a], pts_dict[b])
        d_ba = w_dist(pts_dict[b], pts_dict[a])
        if abs(d_ab - d_ba) > 1e-2: passed = False
        print(f"  |d({a[:4]},{b[:4]}) - d({b[:4]},{a[:4]})| = {abs(d_ab-d_ba):.2e} (Expected ~0)")
    print(f"  Status: {'PASS' if passed else 'FAIL'}\n")

def check_monotonicity(env, ks=[0, 10, 40, 160]):
    print(f"=== Test 2: Monotonicity (Env {env[:4]}) ===")
    grid0 = load_grid(env)
    pts0 = subsample(occupied_points(grid0))
    results = []
    
    # Perturbation (add obstacles)
    free = (grid0 == 255)
    dist = distance_transform_edt(free)
    candidates = np.argwhere(dist >= 8)
    
    for k in ks:
        rng = np.random.default_rng(42)
        if k > 0 and len(candidates) >= k:
            grid_k = grid0.copy()
            idx = rng.choice(len(candidates), k, replace=False)
            for i in idx:
                grid_k[candidates[i][0], candidates[i][1]] = 0
            pts_k = subsample(occupied_points(grid_k), rng=rng)
            d = w_dist(pts0, pts_k)
        else:
            d = 0.0
        results.append(d)
        print(f"  Added {k:3d} obstacles -> d = {d:.4f}")
        
    mono = all(results[i] <= results[i+1] + 1e-4 for i in range(len(results)-1))
    print(f"  Status: {'PASS' if mono else 'FAIL'}\n")

def check_behavioral():
    print("=== Test 3: Behavioral Correlation ===")
    if not CSV_FILE.exists() or not BEHAV_FILE.exists():
        print("  SKIP: Missing CSV files.\n")
        return
        
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        D = np.array([[float(x) for x in row] for row in reader])
    
    df = pd.read_csv(BEHAV_FILE)
    idx = {e: i for i, e in enumerate(header)}
    trained = df[df["trained_on"]]["scenario"].tolist()
    
    rows = []
    for _, row in df[~df["trained_on"]].iterrows():
        t = row["scenario"]
        d_vals = [D[idx[s], idx[t]] for s in trained]
        rows.append({"scenario": t, "d_min": min(d_vals), "d_mean": np.mean(d_vals), "success": row["success_rate"]})
        print(f"  {t[:4]}: d_mean={np.mean(d_vals):.4f} -> success={row['success_rate']:.2f}")
        
    d_mean = [r["d_mean"] for r in rows]
    succ = [r["success"] for r in rows]
    rho, p = spearmanr(d_mean, succ)
    
    print(f"  Spearman rho = {rho:+.3f} (p={p:.4f})")
    print(f"  Status: {'PASS' if rho < -0.6 and p < 0.1 else 'WEAK'}\n")

if __name__ == "__main__":
    # Prepare data
    envs = ['E01_C2_02', 'E02_C2_03', 'E03_C2_04', 'E04_C2_06']
    print("Cargando mapas...")
    pts_dict = {e: subsample(occupied_points(load_grid(e))) for e in envs}
    
    # Execute tests
    check_sanity(pts_dict)
    check_monotonicity('E01_C2_02')
    check_behavioral()
