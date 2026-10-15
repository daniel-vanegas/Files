import re
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path


def parse_log_stream(file_stream):
  """Analiza un flujo de datos de log para extraer fallos y estadísticas."""
  failure_counts = Counter()
  accepted_goals = 0
  total_goals = 0

  timeout_pattern = re.compile(r"timeout|timed out", re.IGNORECASE)
  sampler_incomplete_pattern = re.compile(r"sampler_incomplete", re.IGNORECASE)
  collision_pattern = re.compile(r"collision", re.IGNORECASE)
  stuck_pattern = re.compile(r"stuck", re.IGNORECASE)
  accepted_pattern = re.compile(
      r"accepted:\s*(\d+)\s*/\s*target\s*(\d+)", re.IGNORECASE
  )

  for line in file_stream:
    if isinstance(line, bytes):
      line = line.decode("utf-8", errors="ignore")

    if timeout_pattern.search(line):
      failure_counts["timeout"] += 1
    if sampler_incomplete_pattern.search(line):
      failure_counts["sampler_incomplete"] += 1
    if collision_pattern.search(line):
      failure_counts["collision"] += 1
    if stuck_pattern.search(line):
      failure_counts["stuck"] += 1

    match = accepted_pattern.search(line)
    if match:
      accepted_goals = int(match.group(1))
      total_goals = int(match.group(2))

  return {
      "failures": dict(failure_counts),
      "accepted_goals": accepted_goals,
      "total_goals": total_goals,
  }


def summarize_logs(target_path_str):
  target_path = Path(target_path_str)

  if not target_path.exists():
    print(f"Error: La ruta '{target_path_str}' no existe.")
    return

  results = []

  # Caso 1: Archivo ZIP
  if target_path.suffix.lower() == ".zip" or zipfile.is_zipfile(target_path):
    with zipfile.ZipFile(target_path, "r") as zf:
      log_files = [
          name for name in zf.namelist() if name.endswith("default_pool.log")
      ]
      if not log_files:
        print("No se encontraron archivos 'default_pool.log' dentro del ZIP.")
        return
      for log_name in sorted(log_files):
        with zf.open(log_name, "r") as f:
          res = parse_log_stream(f)
          results.append((log_name, res))

  # Caso 2: Archivo TAR / TAR.GZ / TGZ
  elif tarfile.is_tarfile(target_path):
    with tarfile.open(target_path, "r:*") as tf:
      log_files = [
          member
          for member in tf.getmembers()
          if member.name.endswith("default_pool.log")
      ]
      if not log_files:
        print("No se encontraron archivos 'default_pool.log' dentro del TAR.")
        return
      for member in sorted(log_files, key=lambda m: m.name):
        f = tf.extractfile(member)
        if f:
          res = parse_log_stream(f)
          results.append((member.name, res))
          f.close()

  # Caso 3: Directorio normal
  elif target_path.is_dir():
    log_files = sorted(list(target_path.glob("**/default_pool.log")))
    if not log_files:
      print(f"No se encontraron archivos default_pool.log en '{target_path_str}'.")
      return
    for log_file in log_files:
      with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        res = parse_log_stream(f)
        results.append((str(log_file), res))
  else:
    print("Formato no soportado o archivo no válido.")
    return

  # Imprimir la tabla resumen
  print(
      f"\n{'Ruta en el comprimido / Directorio':<60} | {'Aceptados/Obj':<15} |"
      f" {'Resumen de Fallos'}"
  )
  print("-" * 115)

  for name, result in results:
    acc = result["accepted_goals"]
    tot = result["total_goals"]
    acc_str = f"{acc} / {tot}" if tot > 0 else "N/A"
    failures = result["failures"]
    fail_str = (
        ", ".join([f"{k}: {v}" for k, v in failures.items()])
        if failures
        else "Ninguno detectado"
    )
    print(f"{name:<60} | {acc_str:<15} | {fail_str}")


if __name__ == "__main__":
  target = sys.argv[1] if len(sys.argv) > 1 else input("Ruta del archivo o carpeta: ")
  summarize_logs(target)