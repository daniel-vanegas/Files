"""data_analysis.py — Shared class-balance analysis for the mapless dataset.

Library API
-----------
collect_class_counts_by_scenario(data_root, scenarios=None, source="auto")
    -> {scenario: {class: count}}

format_class_balance_table(counts)   -> ASCII table string
print_class_balance(counts)          -> writes table to stdout
plot_class_balance(counts, path)     -> saves stacked-bar PNG
check_class_imbalance(counts, min_fraction=0.10) -> [warnings]

Consumers
---------
* generate_npz_data.py : report class balance of what was actually written to .npz
* train.py             : report class balance of the training set

CLI
---
    python3 data_analysis.py                     # auto, saves ./class_balance.png if --plot-out
    python3 data_analysis.py --data-root ../data
    python3 data_analysis.py --scenarios E01 E02
    python3 data_analysis.py --source episode_tries
    python3 data_analysis.py --plot-out /tmp/balance.png
    python3 data_analysis.py --no-plot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

try:
    import yaml
except ImportError:
    yaml = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


CLASS_NAMES = ("straight", "semi_complex", "complex", "v_complex")
CLASS_COLORS = {
    "straight":     "#2878B5",
    "semi_complex": "#4A9D62",
    "complex":      "#E39C37",
    "v_complex":    "#C84D4D",
}

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))


# Discovery

def _scenario_of(path, data_root):
    rel = os.path.relpath(path, data_root)
    if rel == ".":
        return "root"
    return rel.split(os.sep)[0]


def _find_npz(data_root, scenarios=None):
    """Yield run_*.npz files (scenario subdirs or top-level)."""
    if not os.path.isdir(data_root):
        return
    # Top-level (single-scenario layout)
    if scenarios is None or "root" in scenarios:
        for name in sorted(os.listdir(data_root)):
            if name.startswith("run_") and name.endswith(".npz"):
                path = os.path.join(data_root, name)
                if os.path.isfile(path):
                    yield path
    # Scenario subdirectories
    for name in sorted(os.listdir(data_root)):
        scen_path = os.path.join(data_root, name)
        if not os.path.isdir(scen_path):
            continue
        if scenarios is not None and name not in scenarios:
            continue
        for child in sorted(os.listdir(scen_path)):
            if child.startswith("run_") and child.endswith(".npz"):
                yield os.path.join(scen_path, child)


def _find_runs(data_root, scenarios=None):
    """Yield run_* directories that contain data.csv."""
    if not os.path.isdir(data_root):
        return
    if scenarios is None or "root" in scenarios:
        for name in sorted(os.listdir(data_root)):
            run_dir = os.path.join(data_root, name)
            if (os.path.isdir(run_dir) and name.startswith("run_")
                    and os.path.isfile(os.path.join(run_dir, "data.csv"))):
                yield run_dir
    for name in sorted(os.listdir(data_root)):
        scen_path = os.path.join(data_root, name)
        if not os.path.isdir(scen_path):
            continue
        if scenarios is not None and name not in scenarios:
            continue
        for child in sorted(os.listdir(scen_path)):
            run_dir = os.path.join(scen_path, child)
            if (os.path.isdir(run_dir) and child.startswith("run_")
                    and os.path.isfile(os.path.join(run_dir, "data.csv"))):
                yield run_dir


# Class sources

def _load_episode_classes(run_dir):
    """Read episode_tries.json -> {episode_id: class}. Empty on failure."""
    path = os.path.join(run_dir, "episode_tries.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for entry in data.get("entries", []):
        try:
            eid = int(entry.get("episode_id", -1))
        except (TypeError, ValueError):
            continue
        if eid < 0:
            continue
        out[eid] = str(entry.get("class", "unknown"))
    return out


def _descriptor_class_counts(npz_path):
    """Read class_counts from the run_*.descriptor.yaml sidecar, or None."""
    if yaml is None or not npz_path.endswith(".npz"):
        return None
    sidecar = npz_path[:-4] + ".descriptor.yaml"
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar) as f:
            desc = yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return None
    counts = desc.get("class_counts")
    if not isinstance(counts, dict):
        return None
    return {name: int(counts.get(name, 0)) for name in CLASS_NAMES}


def _npz_episode_ids(npz_path):
    """Return set of episode_ids stored in a .npz, or None on failure."""
    if np is None:
        return None
    try:
        with np.load(npz_path, allow_pickle=True) as archive:
            if "data" not in archive:
                return None
            return {int(s["meta"]["episode_id"]) for s in archive["data"]}
    except Exception:
        return None


# Public API

def collect_class_counts_by_scenario(
    data_root=DEFAULT_DATA_ROOT,
    scenarios=None,
    source="auto",
):
    """Aggregate saved-episode class counts per scenario.

    Parameters
    ----------
    data_root : str
        Dataset root (containing E01, E02, ...).
    scenarios : iterable[str] | None
        Restrict to these scenario names. None = all.
    source : str
        "descriptor"     fast path: read run_*.descriptor.yaml sidecars
        "episode_tries"  accurate: parse run_*/episode_tries.json for all runs
        "npz"            accurate: intersect .npz episode_ids with episode_tries.json
        "auto"           descriptor if any sidecar has class_counts, else episode_tries

    Returns
    -------
    dict[str, dict[str, int]]
        {scenario: {class: count}}.
    """
    scenarios_set = set(scenarios) if scenarios else None
    result = defaultdict(lambda: {name: 0 for name in CLASS_NAMES})

    if source not in ("auto", "descriptor", "episode_tries", "npz"):
        raise ValueError(f"unknown source: {source}")

    if source == "auto":
        has_sidecar = any(
            _descriptor_class_counts(p) is not None
            for p in _find_npz(data_root, scenarios_set)
        )
        source = "descriptor" if has_sidecar else "episode_tries"

    if source in ("descriptor", "npz"):
        any_found = False
        for npz_path in _find_npz(data_root, scenarios_set):
            scenario = _scenario_of(npz_path, data_root)
            if source == "descriptor":
                counts = _descriptor_class_counts(npz_path)
            else:
                counts = None
            if counts is None:
                # Exact fallback: intersect .npz ids with episode_tries.json
                run_dir = npz_path[:-4]
                if os.path.isdir(run_dir):
                    episode_classes = _load_episode_classes(run_dir)
                    ids = _npz_episode_ids(npz_path)
                    if ids is not None and episode_classes:
                        counts = {name: 0 for name in CLASS_NAMES}
                        for eid in ids:
                            cls = episode_classes.get(eid)
                            if cls in counts:
                                counts[cls] += 1
            if counts is None:
                continue
            any_found = True
            for name in CLASS_NAMES:
                result[scenario][name] += int(counts.get(name, 0))
        if not any_found and source == "descriptor":
            return collect_class_counts_by_scenario(
                data_root, scenarios, source="episode_tries"
            )
        return dict(result)

    # source == "episode_tries"
    for run_dir in _find_runs(data_root, scenarios_set):
        episode_classes = _load_episode_classes(run_dir)
        if not episode_classes:
            continue
        scenario = _scenario_of(run_dir, data_root)
        for cls in episode_classes.values():
            if cls in CLASS_NAMES:
                result[scenario][cls] += 1
    return dict(result)


def format_class_balance_table(counts, col_width=14):
    """Return the summary table as a string."""
    if not counts:
        return "(no class metadata found)\n"
    scenarios = sorted(counts)
    header = (
        f"{'scenario':<10}"
        + "".join(f"{name:>{col_width}}" for name in CLASS_NAMES)
        + f"{'total':>8}"
    )
    lines = [header, "-" * len(header)]
    grand = {name: 0 for name in CLASS_NAMES}
    for scenario in scenarios:
        row_counts = counts[scenario]
        total = sum(row_counts.values())
        lines.append(
            f"{scenario:<10}"
            + "".join(f"{row_counts.get(n, 0):>{col_width}}" for n in CLASS_NAMES)
            + f"{total:>8}"
        )
        for name in CLASS_NAMES:
            grand[name] += row_counts.get(name, 0)
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<10}"
        + "".join(f"{grand[n]:>{col_width}}" for n in CLASS_NAMES)
        + f"{sum(grand.values()):>8}"
    )
    return "\n".join(lines) + "\n"


def print_class_balance(counts, file=None):
    """Print the summary table (defaults to stdout)."""
    (file or sys.stdout).write(format_class_balance_table(counts))
    (file or sys.stdout).flush()


def plot_class_balance(counts, output_path):
    """Save a grouped bar chart with 4 bars per scenario, split into rows of 10 scenarios."""
    if plt is None or np is None:
        print("WARN: matplotlib or numpy unavailable; skipping class balance chart",
              file=sys.stderr)
        return None
    scenarios = sorted(counts)
    if not scenarios:
        return None
    
    # Dividir los escenarios en bloques de 10 por nivel (fila)
    chunk_size = 10
    scenario_chunks = [scenarios[i:i + chunk_size] for i in range(0, len(scenarios), chunk_size)]
    n_rows = len(scenario_chunks)
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), dpi=130)
    if n_rows == 1:
        axes = [axes]
    else:
        axes = list(axes)
    
    num_classes = len(CLASS_NAMES)
    bar_width = 0.18  # Ancho de cada barra por clase
    
    for row_idx, chunk in enumerate(scenario_chunks):
        ax = axes[row_idx]
        x = np.arange(len(chunk))
        
        for c_idx, cls in enumerate(CLASS_NAMES):
            # Desplazamiento horizontal para cada barra de clase dentro del escenario
            offsets = x + (c_idx - num_classes / 2 + 0.5) * bar_width
            values = [counts[s].get(cls, 0) for s in chunk]
            
            bars = ax.bar(offsets, values, width=bar_width, label=cls if row_idx == 0 else "",
                          color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.5)
            
            # Indicar la cantidad exacta encima de cada barra
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.annotate(str(int(val)),
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # Desplazamiento vertical de 3 puntos
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax.set_xticks(x)
        ax.set_xticklabels(chunk, fontsize=9)
        ax.set_ylabel("Episodes", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        
        # Ajustar el límite superior del eje Y para que quepan las etiquetas de texto rotadas
        max_val = max([counts[s].get(cls, 0) for s in chunk for cls in CLASS_NAMES] + [1])
        ax.set_ylim(0, max_val * 1.30)

    axes[0].set_title("Class balance per scenario (Grouped by class)", fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Scenario", fontsize=10)
    
    # Leyenda unificada en la esquina superior del primer nivel
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=4)
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output_path


def check_class_imbalance(counts, min_fraction=0.10):
    """Return warnings for underrepresented classes."""
    totals = {name: 0 for name in CLASS_NAMES}
    grand_total = 0
    for scenario_counts in counts.values():
        for name, c in scenario_counts.items():
            if name in totals:
                totals[name] += int(c)
                grand_total += int(c)
    if grand_total == 0:
        return []
    return [
        f"class '{name}' is underrepresented: "
        f"{totals[name]}/{grand_total} ({totals[name] / grand_total:.1%})"
        for name in CLASS_NAMES
        if totals[name] / grand_total < min_fraction
    ]


# CLI

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_root", nargs="?", default=None,
                   help=f"Dataset root (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--scenarios", nargs="+", default=None, metavar="E##",
                   help="Restrict to these scenarios (default: all)")
    p.add_argument("--source", choices=("auto", "descriptor", "episode_tries", "npz"),
                   default="auto", help="Where to read class counts from")
    p.add_argument("--plot-out", default=None, metavar="PATH",
                   help="Save stacked-bar chart to this path")
    p.add_argument("--no-plot", action="store_true", help="Skip the chart")
    p.add_argument("--no-table", action="store_true", help="Skip the table")
    args = p.parse_args(argv)

    data_root = args.data_root or DEFAULT_DATA_ROOT
    if not os.path.isdir(data_root):
        print(f"not a directory: {data_root}", file=sys.stderr)
        return 1

    counts = collect_class_counts_by_scenario(
        data_root, args.scenarios, source=args.source)
    if not counts:
        print("no class metadata found", file=sys.stderr)
        return 1

    if not args.no_table:
        print_class_balance(counts)

    if not args.no_plot and args.plot_out:
        plot_class_balance(counts, args.plot_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())