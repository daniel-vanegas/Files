#!/usr/bin/env bash
# Unified path-pool generation + Webots/ROS 2 recording, one scenario at a time.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
DATA_ROOT="$DATASET_DIR/data"
EXPECTED_DATA_ROOT="$REPO_ROOT/src/Navigation/mapless_navigation/dataset/data"
MAPS_ROOT="$REPO_ROOT/src/Hardware/justina_description/webots_env/environments/maps"
GENERATOR="$SCRIPT_DIR/path_generator.py"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$DATASET_DIR/logs/collection_$RUN_TIMESTAMP}"
SCENARIOS_TEXT="E01 E02 E03"
N_GOALS=100
CLASS_QUOTAS=""
SEED_BASE=2026101400
TIMEOUT_SECONDS=""
MAX_ATTEMPTS=""
PROGRESS_INTERVAL_SECONDS=5
STALL_TIMEOUT=600
GOAL_POSITION_TOLERANCE_M=0.40
HARD_CLEARANCE_M=0.35
ENDPOINT_OBSTACLE_BAND_M=0.65
OBSTACLE_INFLUENCE_M=0.80
MIN_GOAL_DISTANCE_M=2.0
MAX_GOAL_DISTANCE_M=3.0
MAX_PATH_LENGTH_M=3.0
V_MAX_GOAL_DISTANCE_M=3.6
V_MAX_PATH_LENGTH_M=4.2
MAX_LENGTH_RATIO=1.18
V_GOAL_CANDIDATES=10
V_PLAN_CANDIDATES=3
V_HOTSPOT_PROBABILITY=0.80
V_HOTSPOT_RADIUS_M=0.35
PLAN_CELL_FACTOR=5
PURGE_DATA=0
ASSUME_YES=0
EXTRA_ARGS=()
PKG=mapless_dataset
LAUNCH_FILE=record_single_world_classic_nav.launch.xml
GRACEFUL_STOP=0
CURRENT_PGID=""

usage() {
  cat <<EOF
Usage: $0 [generation/recording options] [-- extra ros2 launch arguments]

Scenario and class targets:
  --scenarios "E01 E02 ..."       scenarios to process (default: E01 E02 E03)
  --n-goals N, --num-goals N       balanced routes per scenario (default: 100)
  --st_sem_cmp_vc S_S_C_V          explicit quotas: straight_semi_complex_complex_v_complex
  --seed-base N, --base-seed N     scenario seed = base + numeric scenario suffix
  --maps-root DIR                  map directory root

Generation limits and geometry:
  --timeout-seconds SEC            generation wall-time limit per scenario; 0 disables
  --max-attempts N                 maximum attempts per scenario; omitted is unlimited
  --progress-interval-seconds SEC  progress update interval (default: 5)
  --hard-clearance-m M             minimum clearance (default: 0.35)
  --endpoint-obstacle-band-m M     endpoint obstacle band (default: 0.65)
  --obstacle-influence-m M          obstacle influence distance (default: 0.80)
  --min-goal-distance-m M           minimum goal distance (default: 2.0)
  --max-goal-distance-m M           maximum goal distance (default: 3.0)
  --max-path-length-m M             ordinary route length cap (default: 3.0)
  --v-max-goal-distance-m M         v-complex goal distance cap (default: 3.6)
  --v-max-path-length-m M            v-complex route length cap (default: 4.2)
  --max-length-ratio R               route/Euclidean distance cap (default: 1.18)
  --v-goal-candidates N              v-complex candidates per start (default: 10)
  --v-plan-candidates N               plans per v-complex start (default: 3)
  --v-hotspot-probability P           bend hotspot sampling probability (default: 0.80)
  --v-hotspot-radius-m M               bend hotspot radius (default: 0.35)
  --plan-cell-factor N                 planning-grid factor (default: 5)

Recording and safety:
  --goal-position-tolerance-m M       sampler/controller endpoint tolerance (default: 0.40)
  --stall-timeout SEC                 stop recording if log stalls (default: 600; 0 disables)
  --log-dir DIR                      logs/status output directory
  --purge-data                        delete all of DATA_ROOT before running
  --yes                               required with --purge-data to skip prompt
  -h, --help                          show this help

This is the sole classic-navigation orchestrator. It calls path_generator.py directly,
installs a pool only after quota/schema validation, then launches recording. Arguments
after -- are passed to ros2 launch, e.g. -- world_offset_x:=0.0 use_path_pool:=true.
EOF
}

while (($#)); do
  case "$1" in
    --scenarios|--worlds) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; SCENARIOS_TEXT="$2"; shift 2 ;;
    --n-goals|--num-goals) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; N_GOALS="$2"; shift 2 ;;
    --st_sem_cmp_vc|--st-sem-cmp-vc) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; CLASS_QUOTAS="$2"; shift 2 ;;
    --seed-base|--base-seed) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; SEED_BASE="$2"; shift 2 ;;
    --timeout-seconds) (($# >= 2)) || { echo "ERROR: --timeout-seconds needs a value" >&2; exit 2; }; TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-attempts) (($# >= 2)) || { echo "ERROR: --max-attempts needs a value" >&2; exit 2; }; MAX_ATTEMPTS="$2"; shift 2 ;;
    --progress-interval-seconds) (($# >= 2)) || { echo "ERROR: --progress-interval-seconds needs a value" >&2; exit 2; }; PROGRESS_INTERVAL_SECONDS="$2"; shift 2 ;;
    --maps-root) (($# >= 2)) || { echo "ERROR: --maps-root needs a value" >&2; exit 2; }; MAPS_ROOT="$2"; shift 2 ;;
    --hard-clearance-m) (($# >= 2)) || { echo "ERROR: --hard-clearance-m needs a value" >&2; exit 2; }; HARD_CLEARANCE_M="$2"; shift 2 ;;
    --endpoint-obstacle-band-m) (($# >= 2)) || { echo "ERROR: --endpoint-obstacle-band-m needs a value" >&2; exit 2; }; ENDPOINT_OBSTACLE_BAND_M="$2"; shift 2 ;;
    --obstacle-influence-m) (($# >= 2)) || { echo "ERROR: --obstacle-influence-m needs a value" >&2; exit 2; }; OBSTACLE_INFLUENCE_M="$2"; shift 2 ;;
    --min-goal-distance-m) (($# >= 2)) || { echo "ERROR: --min-goal-distance-m needs a value" >&2; exit 2; }; MIN_GOAL_DISTANCE_M="$2"; shift 2 ;;
    --max-goal-distance-m) (($# >= 2)) || { echo "ERROR: --max-goal-distance-m needs a value" >&2; exit 2; }; MAX_GOAL_DISTANCE_M="$2"; shift 2 ;;
    --max-path-length-m) (($# >= 2)) || { echo "ERROR: --max-path-length-m needs a value" >&2; exit 2; }; MAX_PATH_LENGTH_M="$2"; shift 2 ;;
    --v-max-goal-distance-m) (($# >= 2)) || { echo "ERROR: --v-max-goal-distance-m needs a value" >&2; exit 2; }; V_MAX_GOAL_DISTANCE_M="$2"; shift 2 ;;
    --v-max-path-length-m) (($# >= 2)) || { echo "ERROR: --v-max-path-length-m needs a value" >&2; exit 2; }; V_MAX_PATH_LENGTH_M="$2"; shift 2 ;;
    --max-length-ratio) (($# >= 2)) || { echo "ERROR: --max-length-ratio needs a value" >&2; exit 2; }; MAX_LENGTH_RATIO="$2"; shift 2 ;;
    --v-goal-candidates) (($# >= 2)) || { echo "ERROR: --v-goal-candidates needs a value" >&2; exit 2; }; V_GOAL_CANDIDATES="$2"; shift 2 ;;
    --v-plan-candidates) (($# >= 2)) || { echo "ERROR: --v-plan-candidates needs a value" >&2; exit 2; }; V_PLAN_CANDIDATES="$2"; shift 2 ;;
    --v-hotspot-probability) (($# >= 2)) || { echo "ERROR: --v-hotspot-probability needs a value" >&2; exit 2; }; V_HOTSPOT_PROBABILITY="$2"; shift 2 ;;
    --v-hotspot-radius-m) (($# >= 2)) || { echo "ERROR: --v-hotspot-radius-m needs a value" >&2; exit 2; }; V_HOTSPOT_RADIUS_M="$2"; shift 2 ;;
    --plan-cell-factor) (($# >= 2)) || { echo "ERROR: --plan-cell-factor needs a value" >&2; exit 2; }; PLAN_CELL_FACTOR="$2"; shift 2 ;;
    --goal-position-tolerance-m) (($# >= 2)) || { echo "ERROR: --goal-position-tolerance-m needs a value" >&2; exit 2; }; GOAL_POSITION_TOLERANCE_M="$2"; shift 2 ;;
    --stall-timeout) (($# >= 2)) || { echo "ERROR: --stall-timeout needs a value" >&2; exit 2; }; STALL_TIMEOUT="$2"; shift 2 ;;
    --purge-data) PURGE_DATA=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    --log-dir) (($# >= 2)) || { echo "ERROR: --log-dir needs a value" >&2; exit 2; }; LOG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$GENERATOR" ]] || { echo "ERROR: path generator not found: $GENERATOR" >&2; exit 2; }
[[ -d "$MAPS_ROOT" ]] || { echo "ERROR: maps root not found: $MAPS_ROOT" >&2; exit 2; }
[[ ! -L "$DATA_ROOT" ]] || { echo "ERROR: refusing to purge or write through a data-root symlink: $DATA_ROOT" >&2; exit 2; }
actual_data="$(realpath -m "$DATA_ROOT")"
expected_data="$(realpath -m "$EXPECTED_DATA_ROOT")"
[[ "$actual_data" == "$expected_data" && "$DATA_ROOT" == "$EXPECTED_DATA_ROOT" ]] || { echo "ERROR: data-root safety check failed: $DATA_ROOT" >&2; exit 2; }
python3 -c 'import numpy, yaml, PIL, scipy' || { echo "ERROR: missing Python dependencies: numpy, PyYAML, Pillow, SciPy" >&2; exit 2; }
[[ "$N_GOALS" =~ ^[0-9]+$ ]] && ((N_GOALS >= 4)) || { echo "ERROR: --n-goals must be an integer >= 4" >&2; exit 2; }
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || { echo "ERROR: --seed-base must be a non-negative integer" >&2; exit 2; }
[[ "$STALL_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "ERROR: --stall-timeout must be a non-negative integer" >&2; exit 2; }
python3 - "$GOAL_POSITION_TOLERANCE_M" <<'PYTOL' || { echo "ERROR: --goal-position-tolerance-m must be finite and >= 0.20 m" >&2; exit 2; }
import math,sys
x=float(sys.argv[1])
raise SystemExit(0 if math.isfinite(x) and x >= 0.20 else 1)
PYTOL
[[ "$V_GOAL_CANDIDATES" =~ ^[0-9]+$ ]] && ((V_GOAL_CANDIDATES > 0)) || { echo "ERROR: --v-goal-candidates must be a positive integer" >&2; exit 2; }
[[ "$V_PLAN_CANDIDATES" =~ ^[0-9]+$ ]] && ((V_PLAN_CANDIDATES > 0)) || { echo "ERROR: --v-plan-candidates must be a positive integer" >&2; exit 2; }
[[ "$PLAN_CELL_FACTOR" =~ ^[0-9]+$ ]] && ((PLAN_CELL_FACTOR >= 1 && PLAN_CELL_FACTOR <= 20)) || { echo "ERROR: --plan-cell-factor must be an integer in [1,20]" >&2; exit 2; }
if [[ -n "$TIMEOUT_SECONDS" ]]; then
  python3 - "$TIMEOUT_SECONDS" <<'PY' || { echo "ERROR: --timeout-seconds must be non-negative" >&2; exit 2; }
import math,sys
x=float(sys.argv[1])
raise SystemExit(0 if math.isfinite(x) and x>=0 else 1)
PY
fi
if [[ -n "$MAX_ATTEMPTS" ]]; then
  [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] && ((MAX_ATTEMPTS > 0)) || { echo "ERROR: --max-attempts must be a positive integer" >&2; exit 2; }
fi
if ((ASSUME_YES && ! PURGE_DATA)); then echo "ERROR: --yes is only valid with --purge-data" >&2; exit 2; fi
if ((PURGE_DATA && ! ASSUME_YES)); then
  [[ -t 0 ]] || { echo "ERROR: non-interactive purge requires --yes" >&2; exit 2; }
  printf 'This permanently deletes %s. Type PURGE to continue: ' "$DATA_ROOT"
  read -r confirmation
  [[ "$confirmation" == PURGE ]] || { echo "Cancelled; no files changed"; exit 1; }
  ASSUME_YES=1
fi

read -r -a SCENARIOS <<< "$SCENARIOS_TEXT"
((${#SCENARIOS[@]})) || { echo "ERROR: empty scenario list" >&2; exit 2; }
declare -A SEEN=()
for scenario in "${SCENARIOS[@]}"; do
  [[ "$scenario" =~ ^E[0-9]{2}$ ]] || { echo "ERROR: invalid scenario '$scenario' (expected E01)" >&2; exit 2; }
  [[ -z "${SEEN[$scenario]+x}" ]] || { echo "ERROR: duplicate scenario '$scenario'" >&2; exit 2; }
  SEEN[$scenario]=1
  map_dir="$(find "$MAPS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${scenario}_*" -print -quit)"
  [[ -n "$map_dir" ]] || { echo "ERROR: no map directory for $scenario under $MAPS_ROOT" >&2; exit 2; }
done

if [[ -n "$CLASS_QUOTAS" ]]; then
  python3 - "$CLASS_QUOTAS" <<'PY' || { echo "ERROR: quotas must be four positive underscore-separated integers" >&2; exit 2; }
import sys
p=sys.argv[1].split('_')
if len(p)!=4 or any(not x.isdigit() or int(x)<1 for x in p): raise SystemExit(1)
PY
  N_GOALS="$(python3 - "$CLASS_QUOTAS" <<'PY'
import sys
print(sum(map(int,sys.argv[1].split('_'))))
PY
)"
fi

if ((PURGE_DATA)); then
  echo "Purging requested dataset directory: $DATA_ROOT"
  rm -rf -- "$DATA_ROOT"
fi
mkdir -p "$DATA_ROOT" "$LOG_DIR"
echo "Collection root: $DATA_ROOT"
echo "Map root: $MAPS_ROOT"
echo "Scenarios: ${SCENARIOS[*]}"
echo "Goals per scenario: $N_GOALS"
echo "Class quotas: ${CLASS_QUOTAS:-balanced total}"
echo "Generation timeout per scenario: ${TIMEOUT_SECONDS:-disabled}"
echo "Generation max attempts: ${MAX_ATTEMPTS:-unlimited}"
echo "Recording stall timeout: ${STALL_TIMEOUT}s (0 disables)"
echo "Goal endpoint tolerance: ${GOAL_POSITION_TOLERANCE_M}m (sampler and controller aligned)"
echo "Logs: $LOG_DIR"
: > "$LOG_DIR/generation_batch.log"

# A new manifest is written before generation so a stale pool cannot be accepted after a failure.
manifest_path="$DATA_ROOT/generation_manifest.json"
python3 - "$manifest_path" "$N_GOALS" "$CLASS_QUOTAS" "$MAX_ATTEMPTS" "$TIMEOUT_SECONDS" "$PROGRESS_INTERVAL_SECONDS" "$SEED_BASE" "${SCENARIOS[@]}" <<'PY'
import json, pathlib, sys
path=pathlib.Path(sys.argv[1]); total=int(sys.argv[2]); quota=sys.argv[3]
max_attempts=sys.argv[4]; timeout=sys.argv[5]; interval=float(sys.argv[6]); seed_base=int(sys.argv[7]); scenarios=sys.argv[8:]
items={}
for sc in scenarios:
    items[sc]={"status":"pending", "seed":seed_base+int(sc[1:]), "run_directory":None}
manifest={"status":"running", "generator":"path_generator.py", "pool_version":5,
 "target_mode":"balanced_total" if not quota else "explicit_class_quotas",
 "n_goals":total if not quota else None, "explicit_quotas":quota or None,
 "max_attempts_per_scenario":int(max_attempts) if max_attempts else None,
 "timeout_seconds_per_scenario":float(timeout) if timeout else None,
 "progress_interval_seconds":interval, "scenarios":items}
path.write_text(json.dumps(manifest, indent=2)+"\n")
PY

declare -A RUN_NAMES=()
declare -A RUN_READY=()
gen_status=0
for scenario in "${SCENARIOS[@]}"; do
  scenario_number=$((10#${scenario#E}))
  seed=$((SEED_BASE + scenario_number))
  base_run_name="generation_seed_$seed"
  run_name="$base_run_name"
  if [[ -e "$DATA_ROOT/$scenario/$run_name" ]]; then
    run_name="${base_run_name}_run_$(date +%Y%m%d_%H%M%S)_$$"
    suffix=1
    while [[ -e "$DATA_ROOT/$scenario/$run_name" ]]; do
      run_name="${base_run_name}_run_$(date +%Y%m%d_%H%M%S)_$$_$suffix"
      ((suffix+=1))
    done
  fi
  RUN_NAMES["$scenario"]="$run_name"
  RUN_READY["$scenario"]=0
  scenario_dir="$DATA_ROOT/$scenario"
  run_dir="$scenario_dir/$run_name"
  log="$scenario_dir/${run_name}.log"
  mkdir -p "$scenario_dir"

  cmd=(python3 "$GENERATOR"
    --scenario "$scenario" --seed "$seed" --run-name "$run_name"
    --maps-root "$MAPS_ROOT" --output-dir "$DATA_ROOT"
    --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS"
    --hard-clearance-m "$HARD_CLEARANCE_M"
    --endpoint-obstacle-band-m "$ENDPOINT_OBSTACLE_BAND_M"
    --obstacle-influence-m "$OBSTACLE_INFLUENCE_M"
    --min-goal-distance-m "$MIN_GOAL_DISTANCE_M"
    --max-goal-distance-m "$MAX_GOAL_DISTANCE_M"
    --max-path-length-m "$MAX_PATH_LENGTH_M"
    --v-max-goal-distance-m "$V_MAX_GOAL_DISTANCE_M"
    --v-max-path-length-m "$V_MAX_PATH_LENGTH_M"
    --max-length-ratio "$MAX_LENGTH_RATIO"
    --v-goal-candidates "$V_GOAL_CANDIDATES"
    --v-plan-candidates "$V_PLAN_CANDIDATES"
    --v-hotspot-probability "$V_HOTSPOT_PROBABILITY"
    --v-hotspot-radius-m "$V_HOTSPOT_RADIUS_M"
    --plan-cell-factor "$PLAN_CELL_FACTOR")
  if [[ -n "$CLASS_QUOTAS" ]]; then
    cmd+=(--st_sem_cmp_vc "$CLASS_QUOTAS")
  else
    cmd+=(--n-goals "$N_GOALS" --balance-class)
  fi
  [[ -z "$MAX_ATTEMPTS" ]] || cmd+=(--max-attempts "$MAX_ATTEMPTS")
  [[ -z "$TIMEOUT_SECONDS" ]] || cmd+=(--timeout-seconds "$TIMEOUT_SECONDS")

  echo "[$scenario] generating pool; seed=$seed log=$log"
  set +e
  "${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/generation_batch.log" "$log"
  statuses=("${PIPESTATUS[@]}")
  set -e
  scenario_gen_status="${statuses[0]:-1}"
  if ((scenario_gen_status != 0)); then
    echo "[$scenario] generation failed (exit=$scenario_gen_status); scenario will not be recorded." >&2
    gen_status=1
    continue
  fi

  pool_file="$run_dir/path_pool.json"
  poses_file="$run_dir/path_pool_poses.json"
  if [[ ! -s "$pool_file" || ! -s "$poses_file" ]]; then
    echo "[$scenario] generation failed: pool/poses files missing." >&2
    gen_status=1
    continue
  fi
  validation=0
  python3 - "$pool_file" "$poses_file" "$scenario" "$seed" "$N_GOALS" "$CLASS_QUOTAS" <<'PY' || validation=$?
import collections, json, pathlib, sys
pool_path, poses_path = map(pathlib.Path, sys.argv[1:3])
scenario, seed, total, quota_text = sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
try:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    poses = json.loads(poses_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"ERROR: invalid pool JSON: {exc}", file=sys.stderr); raise SystemExit(2)
classes = ("straight", "semi_complex", "complex", "v_complex")
paths = pool.get("paths", [])
counts = collections.Counter(p.get("class") for p in paths)
targets = pool.get("quota", {})
ids = {str(p.get("id")) for p in paths}
if pool.get("pool_version") != 5 or pool.get("scenario") != scenario or int(pool.get("seed", -1)) != seed:
    print("ERROR: pool version/scenario/seed mismatch", file=sys.stderr); raise SystemExit(2)
if not pool.get("generation_complete") or not ids.issubset(poses):
    print("ERROR: generation incomplete or pool poses missing", file=sys.stderr); raise SystemExit(2)
if any(c not in targets or int(targets[c]) < 1 for c in classes):
    print("ERROR: invalid class quotas", file=sys.stderr); raise SystemExit(2)
expected = sum(map(int, quota_text.split("_"))) if quota_text else total
if sum(int(targets[c]) for c in classes) != expected:
    print("ERROR: pool quota total differs from requested goal count", file=sys.stderr); raise SystemExit(2)
for c in classes:
    print(f"  {c}: {counts.get(c, 0)}/{targets[c]}")
complete = all(counts.get(c, 0) >= int(targets[c]) for c in classes)
print(f"  TOTAL: {len(paths)}/{expected}; complete={complete}")
raise SystemExit(0 if complete else 1)
PY
  if ((validation != 0)); then
    echo "[$scenario] pool failed validation; scenario will not be recorded." >&2
    gen_status=1
    continue
  fi
  install -m 0644 "$pool_file" "$scenario_dir/path_pool.json"
  install -m 0644 "$poses_file" "$scenario_dir/path_pool_poses.json"
  RUN_READY["$scenario"]=1
  echo "[$scenario] pool ready for recording: $scenario_dir/path_pool*.json"
done

# Finalize the manifest with exactly the run directories generated by this invocation.
manifest_args=()
for scenario in "${SCENARIOS[@]}"; do manifest_args+=("$scenario=${RUN_NAMES[$scenario]:-}:${RUN_READY[$scenario]:-0}"); done
python3 - "$DATA_ROOT" "$N_GOALS" "$CLASS_QUOTAS" "$MAX_ATTEMPTS" "$TIMEOUT_SECONDS" "$PROGRESS_INTERVAL_SECONDS" "$SEED_BASE" "${SCENARIOS[@]}" -- "${manifest_args[@]}" <<'PY'
import collections, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); total=int(sys.argv[2]); quota_text=sys.argv[3]
ma=sys.argv[4]; timeout=sys.argv[5]; interval=float(sys.argv[6]); seed_base=int(sys.argv[7])
sep=sys.argv.index("--", 8); scenarios=sys.argv[8:sep]
run_info={}
for item in sys.argv[sep+1:]:
    if "=" in item:
        key, rest = item.split("=", 1)
        name, _, ready = rest.rpartition(":")
        run_info[key]=(name, ready == "1")
classes=("straight", "semi_complex", "complex", "v_complex")
manifest={"status":"running", "generator":"path_generator.py", "pool_version":5,
 "target_mode":"balanced_total" if not quota_text else "explicit_class_quotas",
 "n_goals":total if not quota_text else None,"explicit_quotas":quota_text or None,
 "max_attempts_per_scenario":int(ma) if ma else None,
 "timeout_seconds_per_scenario":float(timeout) if timeout else None,
 "progress_interval_seconds":interval,"scenarios":{}}
all_complete=True
for sc in scenarios:
    expected_seed=seed_base+int(sc[1:]); info=run_info.get(sc); name=info[0] if info else ""; ready=bool(info and info[1])
    if not name:
        manifest["scenarios"][sc]={"status":"missing_pool","seed":expected_seed,"run_directory":None}; all_complete=False; continue
    d=root/sc/name; pp=d/"path_pool.json"; rp=d/"report.json"
    try:
        pool=json.loads(pp.read_text(encoding="utf-8")); paths=pool.get("paths",[])
        counts=collections.Counter(p.get("class") for p in paths); targets=pool.get("quota",{})
        report=json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else {}
        quotas_complete=all(counts.get(c,0)>=int(targets.get(c,1)) for c in classes)
        complete=(ready and bool(pool.get("generation_complete")) and pool.get("scenario")==sc
                  and int(pool.get("seed",-1))==expected_seed and quotas_complete)
        status="complete" if complete else "incomplete"
        manifest["scenarios"][sc]={"status":status,"route_count":len(paths),
          "target_counts":{c:targets.get(c) for c in classes},
          "class_counts":{c:counts.get(c,0) for c in classes},
          "class_deficits":{c:max(0,int(targets.get(c,0))-counts.get(c,0)) for c in classes},
          "seed":pool.get("seed"),"run_directory":str(d.relative_to(root)),
          "generation_complete":pool.get("generation_complete"),"stop_reason":report.get("stop_reason"),
          "max_attempts_hit":report.get("max_attempts_hit"),"timeout_hit":report.get("timeout_hit"),
          "elapsed_seconds":report.get("elapsed_seconds")}
        if not complete: all_complete=False
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        manifest["scenarios"][sc]={"status":"invalid_pool","seed":expected_seed,
          "run_directory":str(d.relative_to(root)),"error":str(exc)}; all_complete=False
manifest["status"]="complete" if all_complete else "incomplete"
path=root/"generation_manifest.json"; path.write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Manifest: {path}")
PY
if ((gen_status == 0)); then
  echo "Dataset generation finished: all quotas complete."
else
  echo "Dataset generation finished with incomplete pools; only validated complete scenarios may be recorded." >&2
fi

if ! ros2 pkg prefix "$PKG" >/dev/null 2>&1; then
  echo "ERROR: ROS 2 package '$PKG' not found. Source install/setup.bash and build the workspace." >&2
  exit 2
fi

cleanup_current() {
  if [[ -n "${CURRENT_PGID:-}" ]]; then
    kill -INT -- "-$CURRENT_PGID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 -- "-$CURRENT_PGID" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$CURRENT_PGID" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$CURRENT_PGID" 2>/dev/null || true
    CURRENT_PGID=""
  fi
}
on_signal() { GRACEFUL_STOP=1; cleanup_current; }
trap on_signal INT TERM
trap cleanup_current EXIT

FAILED=0
for scenario in "${SCENARIOS[@]}"; do
  ((GRACEFUL_STOP == 0)) || break
  scenario_number=$((10#${scenario#E}))
  seed=$((SEED_BASE + scenario_number))
  manifest_json="$DATA_ROOT/generation_manifest.json"
  pool_json="$DATA_ROOT/$scenario/path_pool.json"
  poses_json="$DATA_ROOT/$scenario/path_pool_poses.json"
  log="$LOG_DIR/${scenario}_recording.log"
  status_file="$LOG_DIR/${scenario}.status"

  # Only record a pool confirmed complete by this batch's freshly written manifest.
  # This blocks a stale flat pool from being recorded when a repeat generation fails.
  pool_rel_dir="$(python3 - "$manifest_json" "$scenario" "$seed" <<'PY' 2>/dev/null || true
import json,pathlib,sys
manifest_path=pathlib.Path(sys.argv[1]); scenario=sys.argv[2]; expected_seed=int(sys.argv[3])
try:
    manifest=json.loads(manifest_path.read_text())
    item=manifest.get("scenarios",{}).get(scenario,{})
    run_directory=str(item.get("run_directory") or "")
    rel=pathlib.PurePosixPath(run_directory)
    if (item.get("status") != "complete" or int(item.get("seed",-1)) != expected_seed
            or not run_directory or rel.is_absolute() or ".." in rel.parts
            or not rel.parts or rel.parts[0] != scenario):
        raise SystemExit(1)
    print(run_directory)
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
PY
)"
  if [[ -z "$pool_rel_dir" ]]; then
    echo "[$scenario] SKIP: no complete pool for seed $seed in the current generation manifest"
    printf 'scenario=%s\nseed=%s\nreason=generation_not_complete_for_current_seed\n' "$scenario" "$seed" > "$status_file"
    FAILED=1
    continue
  fi
  pool_dir="$DATA_ROOT/$pool_rel_dir"

  if [[ ! -s "$pool_json" || ! -s "$poses_json" || ! -s "$pool_dir/path_pool.json" || ! -s "$pool_dir/path_pool_poses.json" ]]; then
    echo "[$scenario] SKIP: missing active pool or generated-run files"
    printf 'scenario=%s\nseed=%s\nreason=missing_or_incomplete_pool\npool_dir=%s\n' "$scenario" "$seed" "$pool_dir" > "$status_file"
    FAILED=1
    continue
  fi
  # The flat files consumed by the sampler must match this batch's validated run.
  if ! cmp -s "$pool_dir/path_pool.json" "$pool_json" || ! cmp -s "$pool_dir/path_pool_poses.json" "$poses_json"; then
    echo "[$scenario] SKIP: active pool differs from the current validated generation"
    printf 'scenario=%s\nseed=%s\nreason=active_pool_not_current_generation\npool_dir=%s\n' "$scenario" "$seed" "$pool_dir" > "$status_file"
    FAILED=1
    continue
  fi

  # Validate sampler contract as a last defense against mismatched or stale files.
  if ! python3 - "$pool_json" "$poses_json" "$scenario" "$seed" "$N_GOALS" <<'PY'
import collections,json,sys
pool=json.load(open(sys.argv[1])); poses=json.load(open(sys.argv[2]))
scenario=sys.argv[3]; expected_seed=int(sys.argv[4]); expected=int(sys.argv[5])
if pool.get('scenario') != scenario or pool.get('pool_version') != 5 or int(pool.get('seed', -1)) != expected_seed: raise SystemExit(1)
paths=pool.get('paths',[]); quota=pool.get('quota',{})
classes=('straight','semi_complex','complex','v_complex')
counts=collections.Counter(p.get('class') for p in paths)
if sum(int(quota.get(c,0)) for c in classes) != expected: raise SystemExit(1)
if not pool.get('generation_complete'): raise SystemExit(1)
if any(counts[c] < int(quota[c]) for c in classes): raise SystemExit(1)
if any(str(p.get('id')) not in poses for p in paths): raise SystemExit(1)
PY
  then
    echo "[$scenario] SKIP: pool failed collection preflight validation"
    printf 'scenario=%s\nseed=%s\nreason=pool_preflight_failed\npool_dir=%s\n' "$scenario" "$seed" "$pool_dir" > "$status_file"
    FAILED=1
    continue
  fi

  echo "[$scenario] recording start (seed=$seed, goals=$N_GOALS)"
  setsid ros2 launch "$PKG" "$LAUNCH_FILE" \
    world:="$scenario" scenario:="$scenario" num_goals:="$N_GOALS" \
    seed:="$seed" output_dir:="$DATA_ROOT" \
    goal_position_tolerance_m:="$GOAL_POSITION_TOLERANCE_M" "${EXTRA_ARGS[@]}" >"$log" 2>&1 &
  pid=$!
  CURRENT_PGID="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  sleep 3
  start_ts=$(date +%s)
  reason="launch_exit"
  if ! kill -0 "$pid" 2>/dev/null; then
    reason="launch_failed"
  else
    while kill -0 "$pid" 2>/dev/null; do
      ((GRACEFUL_STOP == 0)) || { reason="interrupted"; break; }
      if grep -q "All requested goals collected" "$log" 2>/dev/null; then
        reason="sampler_complete"; sleep 3; break
      fi
      if grep -Eq "PATH_POOL_ERROR|Sampler stopped before target|Sampler terminated incomplete|Incomplete dataset|ERROR: No /map received|ERROR: Room build failed|No traversable cells found|Sampler cannot start goal search|stage=map_wait_failed" "$log" 2>/dev/null; then
        reason="sampler_incomplete"; break
      fi
      if ((STALL_TIMEOUT > 0)) && [[ -f "$log" ]]; then
        now=$(date +%s); mtime=$(stat -c %Y "$log")
        if ((now-mtime > STALL_TIMEOUT)); then reason="stalled"; break; fi
      fi
      sleep 2
    done
  fi
  end_ts=$(date +%s)
  cleanup_current
  wait "$pid" 2>/dev/null || true
  printf 'scenario=%s\nseed=%s\nnum_goals=%s\nduration_s=%s\nreason=%s\npool_dir=%s\nlog=%s\n' \
    "$scenario" "$seed" "$N_GOALS" "$((end_ts-start_ts))" "$reason" "$pool_dir" "$log" > "$status_file"
  echo "[$scenario] recording ended: $reason"
  if [[ "$reason" != "sampler_complete" ]]; then FAILED=1; tail -n 25 "$log" || true; fi
done

if ((FAILED == 0 && gen_status == 0)); then
  echo "Collection complete for all requested scenarios."
  exit 0
fi
echo "Collection finished with failures; inspect $LOG_DIR and data/generation_manifest.json." >&2
exit 1
