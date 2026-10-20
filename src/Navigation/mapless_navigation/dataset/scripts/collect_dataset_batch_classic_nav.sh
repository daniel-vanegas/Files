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
LOG_DIR="${LOG_DIR:-$DATASET_DIR/data/logs/run_$RUN_TIMESTAMP}"
SCENARIOS_TEXT="E01 E02 E03"
N_GOALS=100
CLASS_QUOTAS=""
SEED_BASE=""
EXPLICIT_SEED=""
INCREMENTAL=0
SKIP_GENERATION=0
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

# Number of times to try bringing the Webots world up before giving up.
LAUNCH_RETRIES="${LAUNCH_RETRIES:-2}"
# Seconds to wait for the recorder node to log its ready marker.
WORLD_READY_TIMEOUT_S="${WORLD_READY_TIMEOUT_S:-60}"

usage() {
  cat <<EOF
Usage: $0 [generation/recording options] [-- extra ros2 launch arguments]

Scenario and class targets:
  --scenarios "E01 E02 ..."       scenarios to process (default: E01 E02 E03)
  --n-goals N, --num-goals N       balanced routes per scenario (default: 100)
  --st_sem_cmp_vc S_S_C_V          explicit quotas: straight_semi_complex_complex_v_complex
  --seed N, --seed-base N          explicit seed or base seed
  --incremental                    reuse existing pools and append/complete missing quotas if needed
  --record-only                    skip pool generation, run recorder with existing pools
  --replace                        delete DATA_ROOT before running (alias for --purge-data)
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
    --record-only|--skip-generation) SKIP_GENERATION=1; shift ;;
    --replace) PURGE_DATA=1; shift ;;
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

  if ((INCREMENTAL)); then
    # Incremental mode handles generation and filtering inside Phase 2.
    # Phase 1 only ensures the scenario directory exists.
    echo "[$scenario] Incremental: deferring pool prep to Phase 2."
    RUN_READY["$scenario"]=1
    continue
  fi

  NEED_OVERWRITE=0

  if ((SKIP_GENERATION)); then
    if [[ -s "$pool_file" && -s "$poses_file" ]]; then
      echo "[$scenario] Record-only: using existing pool."
      RUN_READY["$scenario"]=1
      continue
    else
      echo "[$scenario] Record-only: no pool found, skipping." >&2
      continue
    fi
  fi

  rm -rf "$run_dir"
  rm -f "$pool_file" "$poses_file"
  rm -f "$scenario_dir/path_pool_history.json" "$scenario_dir/path_pool_history_poses.json"

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
  (( NEED_OVERWRITE == 1 )) && cmd+=(--overwrite-run)

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

# Wait until the recorder node logs its ready marker, the launch crashes,
# or the timeout is reached. Returns 0 ready, 1 crashed, 2 timeout.
wait_for_recorder_ready() {
  local log_file="$1"
  local timeout_s="$2"
  local pid="$3"
  local start_t
  start_t=$(date +%s)
  while true; do
    if grep -q "Recorder ready" "$log_file" 2>/dev/null; then
      return 0
    fi
    if grep -Eq "process has died|Failed to launch|Traceback \(most recent" "$log_file" 2>/dev/null; then
      return 1
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    local now_t
    now_t=$(date +%s)
    if (( now_t - start_t >= timeout_s )); then
      return 2
    fi
    sleep 1
  done
}

# Count sample rows written by recorder run directories created in this launch.
# Takes the list of run dirs that already existed before the launch so that
# incremental mode does not count stale data.
count_recorded_samples() {
  local scen="$1"
  local before_list="$2"
  local run_parent="$DATA_ROOT/$scen"
  local total=0
  local d rows
  shopt -s nullglob
  for d in "$run_parent"/run_*_classic_nav; do
    [[ -d "$d" ]] || continue
    if [[ -n "$before_list" ]] && printf '%s\n' "$before_list" | grep -qxF "$d"; then
      continue
    fi
    if [[ -f "$d/data.csv" ]]; then
      rows=$(wc -l < "$d/data.csv" 2>/dev/null || echo 0)
      if (( rows > 1 )); then
        total=$((total + rows - 1))
      fi
    fi
  done
  shopt -u nullglob
  echo "$total"
}

# Count committed episodes from the recorder's episode_tries.json sidecar.
# Each committed episode contributes one entry, independent of how many
# samples that episode wrote to disk.
count_committed_goals() {
  local scen="$1"
  local before_list="$2"
  local run_parent="$DATA_ROOT/$scen"
  local total=0
  local d n
  shopt -s nullglob
  for d in "$run_parent"/run_*_classic_nav; do
    [[ -d "$d" ]] || continue
    if [[ -n "$before_list" ]] && printf '%s\n' "$before_list" | grep -qxF "$d"; then
      continue
    fi
    if [[ -f "$d/episode_tries.json" ]]; then
      n=$(python3 -c "import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(len(d.get('entries', [])))
except Exception:
    print(0)
" "$d/episode_tries.json" 2>/dev/null || echo 0)
      total=$((total + n))
    fi
  done
  shopt -u nullglob
  echo "$total"
}

FAILED=0
for scenario in "${SCENARIOS[@]}"; do
  ((GRACEFUL_STOP == 0)) || break
  scenario_number=$((10#${scenario#E}))
  pool_json="$DATA_ROOT/$scenario/path_pool.json"
  poses_json="$DATA_ROOT/$scenario/path_pool_poses.json"
  status_file="$LOG_DIR/${scenario}.status"

  if [[ ! -s "$pool_json" || ! -s "$poses_json" ]]; then
    echo "[$scenario] SKIP: missing active pool for recording"
    printf 'scenario=%s\nreason=missing_pool\n' "$scenario" > "$status_file"
    FAILED=1
    continue
  fi

  if ((INCREMENTAL)); then
    inc_cmd=(python3 "$SCRIPT_DIR/generate_path_pool.py"
      --scenario "$scenario"
      --incremental
      --target "$N_GOALS"
      --output-dir "$DATA_ROOT"
      --maps-root "$MAPS_ROOT"
      --hard-clearance-m "$HARD_CLEARANCE_M"
      --min-goal-distance-m "$MIN_GOAL_DISTANCE_M"
      --max-goal-distance-m "$MAX_GOAL_DISTANCE_M"
      --max-path-length-m "$MAX_PATH_LENGTH_M"
      --v-max-goal-distance-m "$V_MAX_GOAL_DISTANCE_M"
      --v-max-path-length-m "$V_MAX_PATH_LENGTH_M"
      --max-length-ratio "$MAX_LENGTH_RATIO"
      --plan-cell-factor "$PLAN_CELL_FACTOR"
      --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS")
    [[ -z "$EXPLICIT_SEED" ]] || inc_cmd+=(--seed "$EXPLICIT_SEED")
    [[ -z "$MAX_ATTEMPTS" ]] || inc_cmd+=(--max-attempts "$MAX_ATTEMPTS")
    [[ -z "$TIMEOUT_SECONDS" ]] || inc_cmd+=(--timeout-seconds "$TIMEOUT_SECONDS")

    echo "[$scenario] incremental pool prep:"
    inc_out=$("${inc_cmd[@]}" 2>&1)
    echo "$inc_out" | tee -a "$LOG_DIR/generation_batch.log"
    paths_to_record=$(echo "$inc_out" | grep '^paths_to_record=' | tail -1 | cut -d= -f2)
    actual_goals=${paths_to_record:-0}
    if ((actual_goals <= 0)); then
      echo "[$scenario] Incremental: no paths to record (add count is 0 or pool empty)."
      printf 'scenario=%s\nreason=incremental_skip\n' "$scenario" > "$status_file"
      continue
    fi
    echo "[$scenario] Incremental: recording $actual_goals paths."
  else

    actual_goals=$(python3 -c "import json, collections, sys
try:
    with open('$pool_json', 'r') as f:
        d = json.load(f)
    paths = d.get('paths', [])
    if not paths:
        print(0)
        sys.exit(0)
    counts = collections.Counter(p.get('class') for p in paths)
    skip_gen = '$SKIP_GENERATION' == '1'
    cap = int('$N_GOALS') if skip_gen else 0
    if cap > 0 and cap < len(paths):
        classes = ('straight', 'semi_complex', 'complex', 'v_complex')
        base, rem = divmod(cap, len(classes))
        quota = {}
        for i, c in enumerate(classes):
            want = base + (1 if i < rem else 0)
            quota[c] = min(want, counts.get(c, 0))
        shortfall = cap - sum(quota.values())
        while shortfall > 0:
            progressed = False
            for c in classes:
                if quota[c] < counts.get(c, 0):
                    quota[c] += 1
                    shortfall -= 1
                    progressed = True
                    if shortfall == 0:
                        break
            if not progressed:
                break
        d['quota'] = quota
        total = sum(quota.values())
    else:
        d['quota'] = dict(counts)
        total = len(paths)
    with open('$pool_json', 'w') as f:
        json.dump(d, f, indent=2)
    print(total)
except Exception:
    print(0)
")
  fi

  if ((actual_goals <= 0)); then
    echo "[$scenario] SKIP: pool has zero paths"
    printf 'scenario=%s\nreason=empty_pool\n' "$scenario" > "$status_file"
    FAILED=1
    continue
  fi

  echo "[$scenario] recording start (partial goals=$actual_goals)"

  # Snapshot existing recorder run dirs so we can tell which ones are new.
  shopt -s nullglob
  before_dirs_arr=("$DATA_ROOT/$scenario"/run_*_classic_nav)
  shopt -u nullglob
  before_dirs=""
  if ((${#before_dirs_arr[@]})); then
    before_dirs="$(printf '%s\n' "${before_dirs_arr[@]}")"
  fi

  launch_ok=0
  reason="world_start_failed"
  last_log=""
  pid=""
  total_start_ts=$(date +%s)
  total_end_ts=$total_start_ts

  for attempt in $(seq 1 "$LAUNCH_RETRIES"); do
    ((GRACEFUL_STOP == 0)) || { reason="interrupted"; break; }
    log="$LOG_DIR/${scenario}_recording_attempt${attempt}.log"
    last_log="$log"
    echo "[$scenario] launch attempt $attempt/$LAUNCH_RETRIES (log=$log)"

    setsid ros2 launch "$PKG" "$LAUNCH_FILE" \
      world:="$scenario" scenario:="$scenario" num_goals:="$actual_goals" \
      output_dir:="$DATA_ROOT" \
      goal_position_tolerance_m:="$GOAL_POSITION_TOLERANCE_M" "${EXTRA_ARGS[@]}" >"$log" 2>&1 &
    pid=$!
    CURRENT_PGID="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"

    ready_rc=0
    wait_for_recorder_ready "$log" "$WORLD_READY_TIMEOUT_S" "$pid" || ready_rc=$?

    if (( ready_rc != 0 )); then
      echo "[$scenario] recorder not ready (rc=$ready_rc); retrying"
      cleanup_current
      wait "$pid" 2>/dev/null || true
      pid=""
      continue
    fi

    launch_ok=1
    reason="launch_exit"

    while kill -0 "$pid" 2>/dev/null; do
      ((GRACEFUL_STOP == 0)) || { reason="interrupted"; break; }
      if grep -q "All requested goals collected" "$log" 2>/dev/null; then
        reason="sampler_complete"
        sleep 3
        break
      fi
      if grep -Eq "PATH_POOL_ERROR|Sampler stopped before target|Sampler terminated incomplete|Incomplete dataset|ERROR: No /map received|ERROR: Room build failed|No traversable cells found|Sampler cannot start goal search|stage=map_wait_failed" "$log" 2>/dev/null; then
        reason="sampler_incomplete"
        break
      fi
      if ((STALL_TIMEOUT > 0)) && [[ -f "$log" ]]; then
        now=$(date +%s); mtime=$(stat -c %Y "$log")
        if ((now-mtime > STALL_TIMEOUT)); then reason="stalled"; break; fi
      fi
      sleep 2
    done
    break
  done

  total_end_ts=$(date +%s)
  cleanup_current
  if [[ -n "$pid" ]]; then
    wait "$pid" 2>/dev/null || true
  fi

  n_samples="$(count_recorded_samples "$scenario" "$before_dirs")"
  n_goals="$(count_committed_goals "$scenario" "$before_dirs")"

  # Refine the reason based on whether samples were written.
  if (( launch_ok == 0 )); then
    reason="world_start_failed"
  elif (( n_samples == 0 )); then
    if grep -q "PATH_POOL_ERROR" "$last_log" 2>/dev/null; then
      reason="pool_incompatible_no_data"
    else
      reason="recording_failed_no_data"
    fi
  fi

  # Store the log path relative to data/logs/ when it lives there.
  # Falls back to the given path if --log-dir points somewhere else.
  log_for_status="$last_log"
  if [[ -n "$last_log" ]]; then
    logs_base_real="$(realpath -m "$DATASET_DIR/data/logs")"
    last_log_real="$(realpath -m "$last_log")"
    if [[ "$last_log_real" == "$logs_base_real"/* ]]; then
      log_for_status="${last_log_real#"$logs_base_real"/}"
    fi
  fi

  printf 'scenario=%s\nduration_s=%s\nreason=%s\nsamples=%s\ngoals=%s\ngoals_target=%s\nlog=%s\n' \
    "$scenario" "$((total_end_ts-total_start_ts))" "$reason" "$n_samples" \
    "$n_goals" "$actual_goals" "$log_for_status" > "$status_file"
  echo "[$scenario] recording ended: $reason (samples=$n_samples) (goals saved=$n_goals/$actual_goals)"
  if [[ "$reason" != "sampler_complete" && "$reason" != "sampler_incomplete" ]]; then
    FAILED=1
    tail -n 25 "$last_log" || true
  fi
done

if ((FAILED == 0)); then
  echo "Collection and recording complete for all scenarios."
  exit 0
fi
echo "Collection finished with minor issues; inspect $LOG_DIR." >&2
exit 1