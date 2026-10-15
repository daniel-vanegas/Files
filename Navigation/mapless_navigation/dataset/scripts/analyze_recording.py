import csv
import sys
from collections import Counter
from pathlib import Path


def analyze_recording_logs(base_dir):
  base_path = Path(base_dir)

  print("=== REPORTE DE DIAGNÓSTICO DE GRABACIÓN ===")

  # 1. Analizar carpetas de ejecución (run_*)
  run_dirs = sorted(list(base_path.glob("**/run_*")))
  if run_dirs:
    print(
        f"\n[1] Inspeccionando carpetas de ejecución ({len(run_dirs)}"
        " encontradas):"
    )
    print(
        f"{'Escenario / Carpeta de Ejecución':<55} | {'Frames (lom)':<12} |"
        f" {'Registros CSV'}"
    )
    print("-" * 85)

    for run_dir in run_dirs:
      rel_path = f"{run_dir.parent.name}/{run_dir.name}"
      lom_dir = run_dir / "lom"
      frames = len(list(lom_dir.glob("*.png"))) if lom_dir.exists() else 0

      data_csv = run_dir / "data.csv"
      rows = 0
      if data_csv.exists():
        try:
          with open(data_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = sum(1 for _ in reader) - 1  # Excluir cabecera
        except Exception:
          rows = "Error al leer"

      print(f"{rel_path:<55} | {frames:<12} | {rows}")

  # 2. Buscar palabras clave de fallo en los logs generales (default_pool.log)
  log_files = sorted(list(base_path.glob("**/default_pool.log")))
  if log_files:
    print("\n[2] Analizando eventos críticos en 'default_pool.log':")
    keywords = ["collision", "stuck", "fail", "error", "abort", "timeout"]

    for log_file in log_files:
      scenario = log_file.parent.name
      counts = Counter()
      with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
          lower_line = line.lower()
          for kw in keywords:
            if kw in lower_line:
              counts[kw] += 1

      if counts:
        summary = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        print(f"   - Escenario {scenario} -> {summary}")
      else:
        print(f"   - Escenario {scenario} -> Sin incidencias registradas en log.")


if __name__ == "__main__":
  target = sys.argv[1] if len(sys.argv) > 1 else "."
  analyze_recording_logs(target)