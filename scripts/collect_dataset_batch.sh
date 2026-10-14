#!/usr/bin/env bash
# Unified path-pool generation + Webots/ROS 2 recording, one scenario at a time.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$DATASET_DIR/data"
GENERATOR="$SCRIPT_DIR/generate_dataset.sh"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$DATASET_DIR/logs/collection_$RUN_TIMESTAMP}"
SCENARIOS_TEXT="E01 E02 E03"
N_GOALS=100
CLASS_QUOTAS=""
SEED_BASE=1001
TIMEOUT_SECONDS=""
MAX_ATTEMPTS=""
PROGRESS_INTERVAL_SECONDS=5
STALL_TIMEOUT=600
PURGE_DATA=0
ASSUME_YES=0
EXTRA_ARGS=()
PKG=mapless_dataset
LAUNCH_FILE=record_single_world_classic_nav.launch.xml
GRACEFUL_STOP=0
CURRENT_PGID=""

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra ros2 launch arguments]

Generation (applied independently to each scenario):
  --scenarios "E01 E02 ..."   scenarios to process (default: E01 E02 E03)
  --n-goals N                 balanced routes/goals per scenario (default: 100)
  --st_sem_cmp_vc S_S_C_V     explicit class quotas, e.g. 20_70_70_40
  --seed-base N               seed base; scenario E## uses base + ## (default: 1001)
  --timeout-seconds SEC       generation timeout PER SCENARIO; does not time-limit recording
  --max-attempts N             optional generator attempt cap PER SCENARIO
  --progress-interval-seconds N generator progress interval (default: 5)

Data and recording:
  --purge-data                delete DATA_ROOT before generating any routes
  --yes                       required with --purge-data; skips confirmation
  --stall-timeout SEC          stop a simulation if its log is unchanged (default: 600; 0 disables)
  --log-dir DIR                log/status directory
  -h, --help                   show this help

Arguments after -- are passed to ros2 launch. The generator timeout only limits route
generation; simulation recording continues until goals complete or stall/interruption.
EOF
}

while (($#)); do
  case "$1" in
    --scenarios) (($# >= 2)) || { echo "ERROR: --scenarios needs a value" >&2; exit 2; }; SCENARIOS_TEXT="$2"; shift 2 ;;
    --n-goals) (($# >= 2)) || { echo "ERROR: --n-goals needs a value" >&2; exit 2; }; N_GOALS="$2"; shift 2 ;;
    --st_sem_cmp_vc|--st-sem-cmp-vc) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; CLASS_QUOTAS="$2"; shift 2 ;;
    --seed-base) (($# >= 2)) || { echo "ERROR: --seed-base needs a value" >&2; exit 2; }; SEED_BASE="$2"; shift 2 ;;
    --timeout-seconds) (($# >= 2)) || { echo "ERROR: --timeout-seconds needs a value" >&2; exit 2; }; TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-attempts) (($# >= 2)) || { echo "ERROR: --max-attempts needs a value" >&2; exit 2; }; MAX_ATTEMPTS="$2"; shift 2 ;;
    --progress-interval-seconds) (($# >= 2)) || { echo "ERROR: --progress-interval-seconds needs a value" >&2; exit 2; }; PROGRESS_INTERVAL_SECONDS="$2"; shift 2 ;;
    --stall-timeout) (($# >= 2)) || { echo "ERROR: --stall-timeout needs a value" >&2; exit 2; }; STALL_TIMEOUT="$2"; shift 2 ;;
    --purge-data) PURGE_DATA=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    --log-dir) (($# >= 2)) || { echo "ERROR: --log-dir needs a value" >&2; exit 2; }; LOG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$GENERATOR" ]] || { echo "ERROR: generator not executable: $GENERATOR" >&2; exit 2; }
[[ "$N_GOALS" =~ ^[0-9]+$ ]] && ((N_GOALS >= 4)) || { echo "ERROR: --n-goals must be an integer >= 4" >&2; exit 2; }
[[ "$SEED_BASE" =~ ^[0-9]+$ ]] || { echo "ERROR: --seed-base must be a non-negative integer" >&2; exit 2; }
[[ "$STALL_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "ERROR: --stall-timeout must be a non-negative integer" >&2; exit 2; }
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

mkdir -p "$LOG_DIR"
echo "Collection root: $DATA_ROOT"
echo "Scenarios: ${SCENARIOS[*]}"
echo "Goals per scenario: $N_GOALS"
echo "Class quotas: ${CLASS_QUOTAS:-balanced total}"
echo "Generation timeout per scenario: ${TIMEOUT_SECONDS:-disabled}"
echo "Recording stall timeout: ${STALL_TIMEOUT}s (0 disables)"
echo "Logs: $LOG_DIR"

# Build one generator invocation so generation_manifest.json describes this whole batch.
gen_args=(--scenarios "${SCENARIOS[*]}" --seed-base "$SEED_BASE" --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS")
if [[ -n "$CLASS_QUOTAS" ]]; then gen_args+=(--st_sem_cmp_vc "$CLASS_QUOTAS"); else gen_args+=(--n-goals "$N_GOALS"); fi
[[ -z "$TIMEOUT_SECONDS" ]] || gen_args+=(--timeout-seconds "$TIMEOUT_SECONDS")
[[ -z "$MAX_ATTEMPTS" ]] || gen_args+=(--max-attempts "$MAX_ATTEMPTS")
if ((PURGE_DATA)); then gen_args+=(--purge-data --yes); fi

set +e
"$GENERATOR" "${gen_args[@]}" 2>&1 | tee "$LOG_DIR/generation_batch.log"
gen_status=${PIPESTATUS[0]}
set -e
if ((gen_status != 0)); then
  echo "WARNING: some route pools may be incomplete; only validated complete pools will be recorded." >&2
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
  pool_dir="$DATA_ROOT/$scenario/generation_seed_$seed"
  pool_json="$DATA_ROOT/$scenario/path_pool.json"
  poses_json="$DATA_ROOT/$scenario/path_pool_poses.json"
  log="$LOG_DIR/${scenario}_recording.log"
  status_file="$LOG_DIR/${scenario}.status"

  if [[ ! -s "$pool_json" || ! -s "$poses_json" ]]; then
    echo "[$scenario] SKIP: no validated pool at $pool_json"
    printf 'scenario=%s\nseed=%s\nreason=missing_or_incomplete_pool\n' "$scenario" "$seed" > "$status_file"
    FAILED=1
    continue
  fi
  # Protect against a stale flat pool left by an older generation attempt.
  if ! python3 - "$pool_json" "$poses_json" "$scenario" "$seed" "$N_GOALS" <<'PY'
import collections,json,sys
pool=json.load(open(sys.argv[1])); poses=json.load(open(sys.argv[2]))
scenario=sys.argv[3]; expected_seed=int(sys.argv[4]); expected=int(sys.argv[5])
if pool.get('scenario') != scenario or pool.get('pool_version') != 5 or int(pool.get('seed', -1)) != expected_seed: raise SystemExit(1)
paths=pool.get('paths',[]); quota=pool.get('quota',{})
counts=collections.Counter(p.get('class') for p in paths)
if sum(int(quota.get(c,0)) for c in ('straight','semi_complex','complex','v_complex')) != expected: raise SystemExit(1)
if not pool.get('generation_complete'): raise SystemExit(1)
if any(counts[c] < int(quota[c]) for c in ('straight','semi_complex','complex','v_complex')): raise SystemExit(1)
if any(str(p.get('id')) not in poses for p in paths): raise SystemExit(1)
PY
  then
    echo "[$scenario] SKIP: pool failed collection preflight validation"
    printf 'scenario=%s\nseed=%s\nreason=pool_preflight_failed\n' "$scenario" "$seed" > "$status_file"
    FAILED=1
    continue
  fi

  echo "[$scenario] recording start (seed=$seed, goals=$N_GOALS)"
  setsid ros2 launch "$PKG" "$LAUNCH_FILE" \
    world:="$scenario" scenario:="$scenario" num_goals:="$N_GOALS" \
    seed:="$seed" output_dir:="$DATA_ROOT" "${EXTRA_ARGS[@]}" >"$log" 2>&1 &
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
