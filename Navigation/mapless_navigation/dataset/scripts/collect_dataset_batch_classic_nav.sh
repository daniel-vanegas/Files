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
SEED_BASE=""
EXPLICIT_SEED=""
INCREMENTAL=0
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
  --seed N, --seed-base N          explicit seed or base seed
  --incremental                    reuse existing pools and append/complete missing quotas if needed
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
EOF
}

while (($#)); do
  case "$1" in
    --scenarios|--worlds) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; SCENARIOS_TEXT="$2"; shift 2 ;;
    --n-goals|--num-goals) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; N_GOALS="$2"; shift 2 ;;
    --st_sem_cmp_vc|--st-sem-cmp-vc) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; CLASS_QUOTAS="$2"; shift 2 ;;
    --seed|--seed-base|--base-seed) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; EXPLICIT_SEED="$2"; shift 2 ;;
    --incremental) INCREMENTAL=1; shift ;;
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
[[ "$STALL_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "ERROR: --stall-timeout must be a non-negative integer" >&2; exit 2; }

if ((PURGE_DATA)); then
  echo "Purging requested dataset directory: $DATA_ROOT"
  rm -rf -- "$DATA_ROOT"
fi
mkdir -p "$DATA_ROOT" "$LOG_DIR"
echo "Collection root: $DATA_ROOT"
echo "Map root: $MAPS_ROOT"
echo "Scenarios: $SCENARIOS_TEXT"
echo "Goals per scenario: $N_GOALS"
echo "Class quotas: ${CLASS_QUOTAS:-balanced total}"
echo "Incremental mode: $INCREMENTAL"
echo "Logs: $LOG_DIR"
: > "$LOG_DIR/generation_batch.log"

manifest_path="$DATA_ROOT/generation_manifest.json"

read -r -a SCENARIOS <<< "$SCENARIOS_TEXT"
((${#SCENARIOS[@]})) || { echo "ERROR: empty scenario list" >&2; exit 2; }
declare -A SEEN=()
for scenario in "${SCENARIOS[@]}"; do
  [[ "$scenario" =~ ^E[0-9]{2}$ ]] || { echo "ERROR: invalid scenario '$scenario'" >&2; exit 2; }
  [[ -z "${SEEN[$scenario]+x}" ]] || { echo "ERROR: duplicate scenario '$scenario'" >&2; exit 2; }
  SEEN[$scenario]=1
  map_dir="$(find "$MAPS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${scenario}_*" -print -quit)"
  [[ -n "$map_dir" ]] || { echo "ERROR: no map directory for $scenario under $MAPS_ROOT" >&2; exit 2; }
done

declare -A RUN_NAMES=()
declare -A RUN_READY=()
gen_status=0

for scenario in "${SCENARIOS[@]}"; do
  scenario_number=$((10#${scenario#E}))
  
  # Seed and run name logic
  if [[ -n "$EXPLICIT_SEED" ]]; then
    seed=$((EXPLICIT_SEED + scenario_number))
    run_name="generation_seed_$seed"
  else
    seed=$((RANDOM % 900000 + 100000))
    run_name="default_pool"
  fi

  RUN_NAMES["$scenario"]="$run_name"
  RUN_READY["$scenario"]=0
  scenario_dir="$DATA_ROOT/$scenario"
  run_dir="$scenario_dir/$run_name"
  log="$scenario_dir/${run_name}.log"
  mkdir -p "$scenario_dir"

  pool_file="$scenario_dir/path_pool.json"
  poses_file="$scenario_dir/path_pool_poses.json"

  # Non-incremental mode: substitute/overwrite existing folder
  if ((!INCREMENTAL)); then
    rm -rf "$run_dir"
    rm -f "$pool_file" "$poses_file"
  else
    # Incremental mode: check if existing pool satisfies quota
    if [[ -s "$pool_file" && -s "$poses_file" ]]; then
      echo "[$scenario] Incremental mode: checking existing pool..."
      has_enough=$(python3 - "$pool_file" "$N_GOALS" "$CLASS_QUOTAS" <<'PY'
import json, sys, collections
try:
    pool = json.load(open(sys.argv[1]))
    total = int(sys.argv[2])
    quota_text = sys.argv[3]
    classes = ("straight", "semi_complex", "complex", "v_complex")
    paths = pool.get("paths", [])
    counts = collections.Counter(p.get("class") for p in paths)
    targets = pool.get("quota", {})
    expected = sum(map(int, quota_text.split("_"))) if quota_text else total
    complete = all(counts.get(c, 0) >= int(targets.get(c, 0)) for c in classes) if targets else (len(paths) >= expected)
    print("1" if complete else "0")
except Exception:
    print("0")
PY
)
      if [[ "$has_enough" == "1" ]]; then
        echo "[$scenario] Existing pool already satisfies quota. Reusing without regeneration."
        RUN_READY["$scenario"]=1
        continue
      fi
    fi
  fi

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

  gen_run_pool="$run_dir/path_pool.json"
  gen_run_poses="$run_dir/path_pool_poses.json"
  
  if [[ ! -s "$gen_run_pool" || ! -s "$gen_run_poses" ]]; then
    echo "[$scenario] generation output missing." >&2
    gen_status=1
    continue
  fi

  # Validation: Allow partial pools if at least 1 path was generated (Does not collapse program)
  validation=0
  python3 - "$gen_run_pool" "$gen_run_poses" <<'PY' || validation=$?
import json, sys
try:
    pool = json.load(open(sys.argv[1]))
    poses = json.load(open(sys.argv[2]))
    paths = pool.get("paths", [])
    if len(paths) == 0 or not poses:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY

  if ((validation != 0)); then
    echo "[$scenario] pool validation failed (no paths found)." >&2
    gen_status=1
    continue
  fi

  # Install active pool files to scenario root for recorder
  install -m 0644 "$gen_run_pool" "$scenario_dir/path_pool.json"
  install -m 0644 "$gen_run_poses" "$scenario_dir/path_pool_poses.json"
  RUN_READY["$scenario"]=1
  echo "[$scenario] pool ready (partial or complete): $scenario_dir/path_pool*.json"
done

if ! ros2 pkg prefix "$PKG" >/dev/null 2>&1; then
  echo "ERROR: ROS 2 package '$PKG' not found." >&2
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
  pool_json="$DATA_ROOT/$scenario/path_pool.json"
  poses_json="$DATA_ROOT/$scenario/path_pool_poses.json"
  log="$LOG_DIR/${scenario}_recording.log"
  status_file="$LOG_DIR/${scenario}.status"

  if [[ ! -s "$pool_json" || ! -s "$poses_json" ]]; then
    echo "[$scenario] SKIP: missing active pool for recording"
    printf 'scenario=%s\nreason=missing_pool\n' "$scenario" > "$status_file"
    FAILED=1
    continue
  fi

  actual_goals=$(python3 -c "import json, collections, sys
try:
    with open('$pool_json', 'r') as f:
        d = json.load(f)
    paths = d.get('paths', [])
    if not paths:
        print(0)
        sys.exit(0)
    # Ajustar la cuota interna al conteo real para evitar errores de dataset incompleto en ROS 2
    counts = collections.Counter(p.get('class') for p in paths)
    d['quota'] = dict(counts)
    with open('$pool_json', 'w') as f:
        json.dump(d, f, indent=2)
    print(len(paths))
except Exception:
    print(0)
")

  if ((actual_goals <= 0)); then
    echo "[$scenario] SKIP: pool has zero paths"
    printf 'scenario=%s\nreason=empty_pool\n' "$scenario" > "$status_file"
    FAILED=1
    continue
  fi

  echo "[$scenario] recording start (partial goals=$actual_goals)"
  setsid ros2 launch "$PKG" "$LAUNCH_FILE" \
    world:="$scenario" scenario:="$scenario" num_goals:="$actual_goals" \
    output_dir:="$DATA_ROOT" \
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
  printf 'scenario=%s\nduration_s=%s\nreason=%s\nlog=%s\n' \
    "$scenario" "$((end_ts-start_ts))" "$reason" "$log" > "$status_file"
  echo "[$scenario] recording ended: $reason"
  if [[ "$reason" != "sampler_complete" && "$reason" != "sampler_incomplete" ]]; then FAILED=1; tail -n 25 "$log" || true; fi
done

if ((FAILED == 0)); then
  echo "Collection and recording complete for all scenarios."
  exit 0
fi
echo "Collection finished with minor issues; inspect $LOG_DIR." >&2
exit 1