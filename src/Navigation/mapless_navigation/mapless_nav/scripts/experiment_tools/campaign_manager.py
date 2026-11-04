#!/usr/bin/env python3
import json
import random
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
CAMPAIGN_CONFIG = {
    "name": "campaign_20k_routes",
    "environment": "T1",
    "seeds": [42, 43, 44],
    "budget": 20000,
    "distributions": {
        "control": {  # Submuestra natural de D0 (asumiendo la distribución del doc)
            "straight": 0.196,
            "semi_complex": 0.229,
            "complex": 0.254,
            "v_complex": 0.321
        },
        "difficult": {  # Tu propuesta
            "straight": 0.15,
            "semi_complex": 0.15,
            "complex": 0.30,
            "v_complex": 0.40
        }
    }
}

def load_episodes(catalog_dir):
    episodes = []
    with open(catalog_dir / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes

def assign_route_class(ep):
    """Asigna una categoría de ruta a un episodio basado en reglas o metadatos.
       Aquí usamos una heurística basada en la longitud del path o el ID."""
    # Si los NPZ tienen un campo 'route_class' en meta, usarlo aquí.
    # Por ahora, usaremos el hash para simular la asignación (reemplazar con lógica real).
    h = int(ep["content_sha256"][:8], 16)
    if h % 10 < 2: return "straight"
    elif h % 10 < 5: return "semi_complex"
    elif h % 10 < 8: return "complex"
    else: return "v_complex"

def build_manifests(catalog_dir, episodes, env, distribution, budget, seed, out_dir):
    rng = random.Random(seed)
    
    # Filtrar por entorno (T1)
    env_episodes = [e for e in episodes if e["scenario"] in ["E01","E02","E03","E12","E17","E22","E27","E30"]]
    
    # Agrupar por categoría de ruta
    by_class = defaultdict(list)
    for ep in env_episodes:
        cls = assign_route_class(ep)
        by_class[cls].append(ep)
        
    selected_pairs = []
    for cls, weight in distribution.items():
        quota = int(budget * weight)
        class_eps = by_class[cls]
        rng.shuffle(class_eps)
        
        # Recolectar pares de esta clase hasta llegar a la cuota
        pairs_found = 0
        for ep in class_eps:
            if pairs_found >= quota: break
            # Tomar todos los pares de este episodio (o hasta la cuota)
            for i in range(ep["num_pairs"]):
                if pairs_found >= quota: break
                selected_pairs.append({
                    "episode_key": ep["episode_key"],
                    "pair_index": i,
                    "scenario": ep["scenario"],
                    "route_class": cls
                })
                pairs_found += 1
                
    # Escribir manifiesto
    manifest_path = out_dir / f"{env}_campaign_{cls}_seed{seed}.jsonl"
    with open(manifest_path, "w") as f:
        for p in selected_pairs:
            f.write(json.dumps(p) + "\n")
            
    return manifest_path, len(selected_pairs)

def run_campaign():
    SCRIPT_DIR = Path(__file__).resolve().parent
    CATALOG_DIR = SCRIPT_DIR / "experiments/catalog_20261026T164348_713171Z"
    OUT_DIR = SCRIPT_DIR / "experiments/campaigns_20k"
    OUT_DIR.mkdir(exist_ok=True)
    
    episodes = load_episodes(CATALOG_DIR)
    
    for config_name, dist in CAMPAIGN_CONFIG["distributions"].items():
        for seed in CAMPAIGN_CONFIG["seeds"]:
            print(f"\n--- Preparando {config_name} (Semilla {seed}) ---")
            manifest, count = build_manifests(
                CATALOG_DIR, episodes, CAMPAIGN_CONFIG["environment"], 
                dist, CAMPAIGN_CONFIG["budget"], seed, OUT_DIR
            )
            print(f"Generado {manifest.name} con {count} pares.")
            
            # Aquí iría el comando para entrenar (descomentar para ejecutar en cola)
            # cmd = [
            #     "python3", str(SCRIPT_DIR / "experiment_control.py"), "train",
            #     "--datasets-dir", str(CATALOG_DIR / "datasets_20261028_221947"),
            #     "--environment-set", CAMPAIGN_CONFIG["environment"],
            #     "--dataset", "D0", # Usamos D0 pero sobrescribimos el manifiesto
            #     "--seed", str(seed)
            # ]
            # subprocess.run(cmd)

if __name__ == "__main__":
    run_campaign()
