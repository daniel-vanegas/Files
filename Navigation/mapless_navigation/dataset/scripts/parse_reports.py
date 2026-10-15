import json
import sys
from pathlib import Path


def parse_report_file(report_path):
  """Carga y extrae la información del archivo report.json."""
  try:
    with open(report_path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception as e:
    return {"error": str(e)}


def summarize_reports(base_dir):
  base_path = Path(base_dir)
  # Busca todos los report.json dentro de cualquier carpeta default_pool
  report_files = sorted(list(base_path.glob("**/default_pool/report.json")))

  if not report_files:
    print(f"No se encontraron archivos report.json bajo '{base_dir}'.")
    return

  print(f"Se encontraron {len(report_files)} reportes.\n")
  print("=" * 60)

  for report_file in report_files:
    # Obtiene el nombre del escenario (ej. E01, E03, E05)
    scenario_name = report_file.parent.parent.name
    print(f"📁 Escenario: {scenario_name}")
    print(f"   Ruta: {report_file}")

    data = parse_report_file(report_file)

    if "error" in data:
      print(f"   ❌ Error al leer el archivo: {data['error']}\n")
      print("-" * 60)
      continue

    # Muestra las claves y valores principales del JSON del reporte
    for key, value in data.items():
      # Si el valor es un diccionario o lista grande, puedes formatearlo o mostrarlo resumido
      if isinstance(value, (dict, list)):
        print(f"   - {key}: ({len(value)} elementos)")
      else:
        print(f"   - {key}: {value}")

    print("-" * 60)


if __name__ == "__main__":
  target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
  summarize_reports(target_dir)