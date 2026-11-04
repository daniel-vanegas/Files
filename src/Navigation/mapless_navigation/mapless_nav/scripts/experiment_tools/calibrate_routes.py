#!/usr/bin/env python3
"""Infiere la categoría de ruta a partir de la geometría de cada episodio."""
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CATALOG_DIR = SCRIPT_DIR / "experiments/catalog_20261026T164348_713171Z"
DATA_ROOT = SCRIPT_DIR.parents[2] / "dataset/data"
CSV_PATH = CATALOG_DIR / "annotations/reviewed_intervals.csv"

def route_metrics(poses):
    """Calcula métricas geométricas de una trayectoria."""
    p = np.asarray(poses)[:, :2]
    if len(p) < 2:
        return None
    diffs = np.diff(p, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = float(seg_lengths.sum())
    direct = float(np.linalg.norm(p[-1] - p[0]))
    efficiency = direct / total_length if total_length > 0 else 0.0
    
    # Desviación máxima de la línea recta inicio-fin
    if direct > 0:
        v = (p[-1] - p[0]) / direct
        n = np.array([-v[1], v[0]])
        deviations = np.abs((p - p[0]) @ n)
        max_deviation = float(deviations.max())
    else:
        max_deviation = 0.0
    
    # Cambios de dirección (giros bruscos)
    if len(diffs) > 1:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        ang_diffs = np.diff(angles)
        ang_diffs = np.abs((ang_diffs + np.pi) % (2 * np.pi) - np.pi)
        turns = int((ang_diffs > np.pi / 4).sum())  # >45°
    else:
        turns = 0
    
    return {
        "length_m": total_length,
        "direct_m": direct,
        "efficiency": efficiency,
        "max_deviation_m": max_deviation,
        "sharp_turns": turns,
    }

def load_behavior_labels():
    """Lee las etiquetas B0-B4 existentes por episodio."""
    by_episode = defaultdict(set)
    if not CSV_PATH.exists():
        return by_episode
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row.get("annotation_scope") != "local":
                continue
            if row.get("review_status") != "reviewed":
                continue
            key = f"{row['run_file']}#{int(row['episode_id'])}"
            if row.get("behavior"):
                by_episode[key].add(row["behavior"])
    return by_episode

def main():
    labels = load_behavior_labels()
    episodes = []
    for line in (CATALOG_DIR / "episodes.jsonl").read_text().splitlines():
        ep = json.loads(line)
        episodes.append(ep)
    
    # Agrupar por run_file para cargar cada NPZ una sola vez
    by_run = defaultdict(list)
    for ep in episodes:
        by_run[ep["run_file"]].append(ep)
    
    print(f"Analizando {len(episodes)} episodios en {len(by_run)} archivos...\n")
    
    # Calcular métricas para todos
    results = []
    for run_file, eps in by_run.items():
        samples_path = DATA_ROOT / run_file
        if not samples_path.exists():
            print(f"⚠️ No encontrado: {samples_path}")
            continue
        with np.load(samples_path, allow_pickle=True) as archive:
            samples = archive["data"].tolist()
        for ep in eps:
            indices = ep["source_indices"]
            poses = [samples[i]["meta"]["pose"] for i in indices]
            metrics = route_metrics(poses)
            if metrics is None:
                continue
            metrics["episode_key"] = ep["episode_key"]
            metrics["scenario"] = ep["scenario"]
            metrics["labels"] = labels.get(ep["episode_key"], set())
            results.append(metrics)
    
    # Cruzar con etiquetas existentes
    print("=" * 90)
    print("DISTRIBUCIÓN DE MÉTRICAS POR ETIQUETA DOMINANTE")
    print("=" * 90)
    
    def dominant_label(lbls):
        for b in ["B3", "B2", "B1", "B0", "B4"]:
            if b in lbls:
                return b
        return None
    
    by_label = defaultdict(list)
    for r in results:
        dom = dominant_label(r["labels"])
        if dom:
            by_label[dom].append(r)
    
    for dom in ["B3", "B2", "B1", "B0", "B4"]:
        group = by_label.get(dom, [])
        if not group:
            continue
        effs = [r["efficiency"] for r in group]
        devs = [r["max_deviation_m"] for r in group]
        turns = [r["sharp_turns"] for r in group]
        print(f"\n{dom} ({len(group)} episodios etiquetados):")
        print(f"  efficiency:       media={np.mean(effs):.3f}  p25={np.percentile(effs,25):.3f}  p75={np.percentile(effs,75):.3f}")
        print(f"  max_deviation_m:  media={np.mean(devs):.2f}  p25={np.percentile(devs,25):.2f}  p75={np.percentile(devs,75):.2f}")
        print(f"  sharp_turns:      media={np.mean(turns):.1f}  min={min(turns)}  max={max(turns)}")
    
    # Proponer umbrales y ver la distribución resultante
    print("\n" + "=" * 90)
    print("PROPUESTA DE CLASIFICACIÓN POR GEOMETRÍA")
    print("=" * 90)
    
    def classify(r):
        if r["sharp_turns"] >= 4 or r["efficiency"] < 0.55:
            return "v_complex"
        if r["sharp_turns"] >= 2 or r["efficiency"] < 0.75 or r["max_deviation_m"] > 1.5:
            return "complex"
        if r["sharp_turns"] >= 1 or r["efficiency"] < 0.90 or r["max_deviation_m"] > 0.5:
            return "semi_complex"
        return "straight"
    
    counts = Counter(classify(r) for r in results)
    total = sum(counts.values())
    for cls in ["straight", "semi_complex", "complex", "v_complex"]:
        n = counts[cls]
        print(f"  {cls:15s}: {n:5d} episodios ({n/total*100:5.1f}%)")
    
    # Validación: ¿los episodios con B3 caen en v_complex?
    print("\n" + "=" * 90)
    print("VALIDACIÓN CRUZADA (solo episodios etiquetados)")
    print("=" * 90)
    
    confusion = defaultdict(Counter)
    for r in results:
        dom = dominant_label(r["labels"])
        if not dom:
            continue
        inferred = classify(r)
        confusion[dom][inferred] += 1
    
    print(f"{'Etiqueta':10s} | {'straight':>12s} | {'semi':>12s} | {'complex':>12s} | {'v_complex':>12s}")
    print("-" * 70)
    for dom in ["B0", "B1", "B2", "B3"]:
        if dom not in confusion: continue
        row = confusion[dom]
        total_dom = sum(row.values())
        cells = []
        for cls in ["straight", "semi_complex", "complex", "v_complex"]:
            n = row[cls]
            pct = n/total_dom*100 if total_dom else 0
            cells.append(f"{n:4d} ({pct:4.1f}%)")
        print(f"{dom:10s} | " + " | ".join(f"{c:>12s}" for c in cells))
    
    print("\nInterpretación:")
    print("  - Si B0 cae mayormente en 'straight' → umbrales razonables")
    print("  - Si B3 cae mayormente en 'v_complex' → la heurística detecta rutas difíciles")
    print("  - Si B2 cae mayormente en 'complex' → captura evasiones restringidas")

if __name__ == "__main__":
    main()
