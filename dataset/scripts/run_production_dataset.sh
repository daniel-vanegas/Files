#!/usr/bin/env bash
set -Eeuo pipefail

# Production reset + route-pool generation. This intentionally deletes ALL of
# dataset/data, including prior experiment outputs and recorded episodes, then
# creates fresh route pools. It never starts Webots or a ROS launch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
DATA_ROOT="$DATASET_DIR/data"
EXPECTED_DATA_ROOT="$REPO_ROOT/src/Navigation/mapless_navigation/dataset/data"
MAPS_ROOT="$REPO_ROOT/src/Hardware/justina_description/webots_env/environments/maps"
GENERATOR="$SCRIPT_DIR/generate_path_pool.py"

PER_CLASS="${PER_CLASS:-50}"
NUM_PATHS="${NUM_PATHS:-$((PER_CLASS * 4))}"
TARGET_PER_CLASS=$(((NUM_PATHS + 3) / 4))
MAX_ATTEMPTS="${MAX_ATTEMPTS:-25000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-5}"
SEED_BASE="${SEED_BASE:-2026101400}"
SCENARIOS_TEXT="${SCENARIOS:-E01 E02 E03}"
PURGE_DATA=0
PRUNE_EXPERIMENTS=0
ASSUME_YES=0

usage() {
  cat <<EOF
Usage: $0 --purge-data [--prune-experiments] [--yes]

Required:
  --purge-data         permanently removes the complete dataset/data directory.
Options:
  --prune-experiments  remove obsolete candidate/audit scripts after production checks.
  --yes                skip the interactive confirmation (dangerous).
Environment variables:
  SCENARIOS="E01 E02 E03" PER_CLASS=50 MAX_ATTEMPTS=25000
  TIMEOUT_SECONDS=900 PROGRESS_INTERVAL_SECONDS=5 SEED_BASE=2026101400

This only generates route pools. Recorded navigation episodes must be recollected
later using the appropriate ROS/Webots collection launch.
EOF
}
while (($#)); do
  case "$1" in
    --purge-data) PURGE_DATA=1; shift ;;
    --prune-experiments) PRUNE_EXPERIMENTS=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if (( PURGE_DATA != 1 )); then
  echo "ERROR: this workflow requires explicit --purge-data." >&2
  usage
  exit 2
fi
if [[ -L "$DATA_ROOT" ]]; then
  echo "ERROR: refusing to purge a symlink: $DATA_ROOT" >&2
  exit 2
fi
actual_data="$(realpath -m "$DATA_ROOT")"
expected_data="$(realpath -m "$EXPECTED_DATA_ROOT")"
if [[ "$actual_data" != "$expected_data" || "$DATA_ROOT" != "$EXPECTED_DATA_ROOT" ]]; then
  echo "ERROR: safety check failed; target was not the expected dataset/data directory." >&2
  echo "Target:   $DATA_ROOT" >&2
  echo "Expected: $EXPECTED_DATA_ROOT" >&2
  exit 2
fi
[[ -f "$GENERATOR" ]] || { echo "ERROR: production wrapper not found: $GENERATOR" >&2; exit 2; }
[[ -f "$SCRIPT_DIR/path_generator.py" ]] || { echo "ERROR: production engine not found: $SCRIPT_DIR/path_generator.py" >&2; exit 2; }
[[ -d "$MAPS_ROOT" ]] || { echo "ERROR: maps root not found: $MAPS_ROOT" >&2; exit 2; }
python3 -c 'import numpy, yaml, PIL, scipy' || {
  echo "ERROR: Python dependencies missing. Need numpy, PyYAML, Pillow, and SciPy." >&2; exit 2;
}

read -r -a SCENARIOS <<< "$SCENARIOS_TEXT"
if ((${#SCENARIOS[@]} == 0)); then echo "ERROR: SCENARIOS is empty" >&2; exit 2; fi
for scenario in "${SCENARIOS[@]}"; do
  if [[ ! "$scenario" =~ ^E[0-9]{2}$ ]]; then
    echo "ERROR: invalid scenario name: $scenario (expected E01 format)" >&2; exit 2
  fi
  map_dir="$(find "$MAPS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${scenario}_*" -print -quit)"
  if [[ -z "$map_dir" ]]; then echo "ERROR: no map directory for $scenario under $MAPS_ROOT" >&2; exit 2; fi
done

cat <<EOF
PRODUCTION DATA RESET
Repository : $REPO_ROOT
Data to be deleted permanently: $DATA_ROOT
Scenarios : ${SCENARIOS[*]}
Target    : $NUM_PATHS routes/scenario (~$TARGET_PER_CLASS/class)
Timeout   : ${TIMEOUT_SECONDS}s/scenario
Progress  : every ${PROGRESS_INTERVAL_SECONDS}s, single-line display
Prune old experimental scripts: $PRUNE_EXPERIMENTS

This removes all existing files under dataset/data, including logs, experiments,
path pools, and recorded navigation episode datasets. Only route pools and their
reports/plots will be regenerated now; episode recordings require a later run.
EOF
if (( ASSUME_YES == 0 )); then
  if [[ ! -t 0 ]]; then echo "ERROR: non-interactive run; rerun with --yes after reviewing the deletion scope." >&2; exit 2; fi
  printf 'Type PURGE to permanently delete data and continue: '
  read -r confirmation
  [[ "$confirmation" == "PURGE" ]] || { echo "Cancelled; no files changed."; exit 1; }
fi

rm -rf -- "$DATA_ROOT"
mkdir -p "$DATA_ROOT"
echo "Purged and recreated: $DATA_ROOT"

if (( PRUNE_EXPERIMENTS == 1 )); then
  for rel in \
    "scripts/path_generator copy.py" \
    "scripts/run_candidate_batch_200.sh" \
    "scripts/run_candidate_batch_timeout.sh"; do
    [[ -f "$DATASET_DIR/$rel" ]] && rm -f -- "$DATASET_DIR/$rel"
  done
  [[ ! -d "$SCRIPT_DIR/path_smoothing" ]] || rm -rf -- "$SCRIPT_DIR/path_smoothing"
  # The pool wrapper is still used by the batch collection launchers, so it is
  # retained as a small compatibility CLI; generation itself is in path_generator.py.
  for collector in "$SCRIPT_DIR"/collect_dataset_batch_*.sh; do
    [[ -f "$collector" ]] || continue
    sed -i -E 's/^POOL_NUM_PATHS=3000$/POOL_NUM_PATHS=200/' "$collector"
  done
  echo "Removed obsolete exploratory scripts; retained production and ROS dataset tools."
fi

overall_status=0
for i in "${!SCENARIOS[@]}"; do
  scenario="${SCENARIOS[$i]}"
  seed=$((SEED_BASE + i + 1))
  scenario_dir="$DATA_ROOT/$scenario"
  mkdir -p "$scenario_dir"
  log_file="$scenario_dir/generation.log"
  echo
  echo "=============================================================="
  echo "[$scenario] production seed=$seed"
  echo "=============================================================="
  set +e
  python3 "$GENERATOR" \
    --scenario "$scenario" \
    --num-paths "$NUM_PATHS" \
    --max-attempts "$MAX_ATTEMPTS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
    --seed "$seed" \
    --maps-root "$MAPS_ROOT" \
    --output-dir "$DATA_ROOT" \
    2>&1 | tee "$log_file"
  statuses=("${PIPESTATUS[@]}")
  set -e
  gen_status="${statuses[0]}"

  if [[ ! -s "$scenario_dir/path_pool.json" || ! -s "$scenario_dir/path_pool_poses.json" ]]; then
    echo "[$scenario] FAIL: missing pool or poses; see $log_file" >&2
    overall_status=1
    continue
  fi
  check_status=0
  python3 - "$scenario_dir/path_pool.json" "$scenario_dir/path_pool_poses.json" "$TARGET_PER_CLASS" <<'PY' || check_status=$?
import collections, json, sys
pool_path, poses_path, target_s = sys.argv[1:]
target = int(target_s)
with open(pool_path, encoding="utf-8") as f: pool = json.load(f)
with open(poses_path, encoding="utf-8") as f: poses = json.load(f)
report_path = str(pool_path).replace("path_pool.json", "report.json")
try:
    with open(report_path, encoding="utf-8") as f: report = json.load(f)
except (OSError, json.JSONDecodeError):
    report = {}
classes = ("straight", "semi_complex", "complex", "v_complex")
paths = pool.get("paths", [])
counts = collections.Counter(p.get("class") for p in paths)
ids = {str(p.get("id")) for p in paths}
if pool.get("pool_version") != 5 or not paths or not ids.issubset(poses.keys()):
    print("ERROR: malformed pool or missing poses", file=sys.stderr); sys.exit(2)
for c in classes: print(f"  {c}: {counts.get(c, 0)}/{target}")
print(f"  TOTAL: {len(paths)}/{target*4}")
complete = all(counts.get(c, 0) >= target for c in classes)
print(f"  COMPLETE: {complete}; timeout_hit={report.get('timeout_hit', 'unknown')}")
sys.exit(0 if complete else 1)
PY
  if (( check_status == 0 )); then
    echo "[$scenario] PASS: target quota complete (generator exit=$gen_status)."
  else
    echo "[$scenario] INCOMPLETE: valid partial pool retained; inspect report and consider training-time balancing."
    overall_status=1
  fi
done

python3 - "$DATA_ROOT" "$TARGET_PER_CLASS" "$TIMEOUT_SECONDS" "$PROGRESS_INTERVAL_SECONDS" "${SCENARIOS[@]}" <<'PY'
import collections, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); target = int(sys.argv[2]); timeout = float(sys.argv[3]); interval = float(sys.argv[4]); scenarios = sys.argv[5:]
classes = ("straight", "semi_complex", "complex", "v_complex")
manifest = {"generator": "path_generator.py", "pool_version": 5,
            "target_per_class": target, "target_routes_per_scenario": target*4,
            "timeout_seconds_per_scenario": timeout, "progress_interval_seconds": interval,
            "scenarios": {}}
for scenario in scenarios:
    p = root / scenario / "path_pool.json"
    if not p.is_file():
        manifest["scenarios"][scenario] = {"status":"missing_pool"}; continue
    pool = json.loads(p.read_text(encoding="utf-8")); counts = collections.Counter(r.get("class") for r in pool.get("paths", []))
    complete = all(counts.get(c,0) >= target for c in classes)
    report_file = root / scenario / "report.json"
    try: report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): report = {}
    manifest["scenarios"][scenario] = {"status":"complete" if complete else "incomplete",
       "route_count":len(pool.get("paths",[])), "class_counts":{c:counts.get(c,0) for c in classes},
       "seed":pool.get("seed"), "generation_complete":pool.get("generation_complete"),
       "timeout_hit":report.get("timeout_hit"), "elapsed_seconds":report.get("elapsed_seconds")}
(root / "production_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\nManifest: {root / 'production_manifest.json'}")
PY

if (( overall_status == 0 )); then
  echo "Production generation finished: all scenario quotas complete."
else
  echo "Production generation finished with incomplete quotas. Partial validated pools remain available; overall exit=1." >&2
fi
exit "$overall_status"
