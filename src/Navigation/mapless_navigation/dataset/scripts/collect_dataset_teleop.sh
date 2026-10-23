#!/usr/bin/env bash
# Standalone teleop dataset collection: prepare pool, then record interactively.
# Does not touch collect_dataset_batch_classic_nav.sh.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
DATA_ROOT="$DATASET_DIR/data"
EXPECTED_DATA_ROOT="$REPO_ROOT/src/Navigation/mapless_navigation/dataset/data"
MAPS_ROOT="$REPO_ROOT/src/Hardware/justina_description/webots_env/environments/maps"
GENERATOR="$SCRIPT_DIR/path_generator.py"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$DATASET_DIR/data/logs/teleop_$RUN_TIMESTAMP}"

# Defaults
SCENARIOS_TEXT="E01"
N_GOALS=4
CLASS_QUOTAS=""
EXPLICIT_SEED=""
RECORD_ONLY=0
PURGE_DATA=0
ASSUME_YES=0
ENDPOINT_TOL_M=0.30
STALL_TIMEOUT=600
MAX_ATTEMPTS=""
TIMEOUT_SECONDS=""
PROGRESS_INTERVAL_SECONDS=5

# Geometry for path_generator.py
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

# Recording
GOAL_POSITION_TOLERANCE_M=0.40
EXTRA_ARGS=()

PKG=mapless_dataset
LAUNCH_FILE=record_single_world_teleop.launch.xml
GRACEFUL_STOP=0
CURRENT_PGID=""
LAUNCH_RETRIES="${LAUNCH_RETRIES:-2}"
WORLD_READY_TIMEOUT_S="${WORLD_READY_TIMEOUT_S:-60}"

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra ros2 launch arguments]

Scenarios:
  --scenarios "E01 E02 ..."    scenarios (default: E01)
  --n-goals N                  balanced routes per scenario (default: 4)
  --st_sem_cmp_vc S_S_C_V      explicit quotas (overrides --n-goals)
  --seed N                     base seed for pool generation
  --record-only                skip pool prep, use existing pool as-is
  --replace                    delete DATA_ROOT before running (needs --yes)
  --purge-data                 delete DATA_ROOT before running (needs --yes)
  --yes                        confirm --purge-data/--replace
  --log-dir DIR                output dir for logs/status

Pool filter:
  --endpoint-tolerance-m M     start/goal match tolerance (default: 0.30)

Generation limits:
  --timeout-seconds SEC        generator wall-time limit (0=off)
  --max-attempts N             generator attempt limit (default: unlimited)
  --progress-interval-seconds  generator progress rate (default: 5)

Recording safety:
  --goal-position-tolerance-m  sampler/controller tolerance (default: 0.40)
  --stall-timeout SEC          stop if log stalls (default: 600; 0=off)

Geometry passed to path_generator.py:
  --hard-clearance-m, --endpoint-obstacle-band-m, --obstacle-influence-m,
  --min-goal-distance-m, --max-goal-distance-m, --max-path-length-m,
  --v-max-goal-distance-m, --v-max-path-length-m, --max-length-ratio,
  --v-goal-candidates, --v-plan-candidates, --v-hotspot-probability,
  --v-hotspot-radius-m, --plan-cell-factor

  -h, --help                   show this help
EOF
}

while (($#)); do
  case "$1" in
    --scenarios|--worlds) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; SCENARIOS_TEXT="$2"; shift 2 ;;
    --n-goals|--num-goals) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; N_GOALS="$2"; shift 2 ;;
    --st_sem_cmp_vc|--st-sem-cmp-vc) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; CLASS_QUOTAS="$2"; shift 2 ;;
    --seed|--seed-base) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; EXPLICIT_SEED="$2"; shift 2 ;;
    --record-only) RECORD_ONLY=1; shift ;;
    --replace|--purge-data) PURGE_DATA=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    --endpoint-tolerance-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; ENDPOINT_TOL_M="$2"; shift 2 ;;
    --stall-timeout) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; STALL_TIMEOUT="$2"; shift 2 ;;
    --timeout-seconds) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-attempts) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; MAX_ATTEMPTS="$2"; shift 2 ;;
    --progress-interval-seconds) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; PROGRESS_INTERVAL_SECONDS="$2"; shift 2 ;;
    --goal-position-tolerance-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; GOAL_POSITION_TOLERANCE_M="$2"; shift 2 ;;
    --hard-clearance-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; HARD_CLEARANCE_M="$2"; shift 2 ;;
    --endpoint-obstacle-band-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; ENDPOINT_OBSTACLE_BAND_M="$2"; shift 2 ;;
    --obstacle-influence-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; OBSTACLE_INFLUENCE_M="$2"; shift 2 ;;
    --min-goal-distance-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; MIN_GOAL_DISTANCE_M="$2"; shift 2 ;;
    --max-goal-distance-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; MAX_GOAL_DISTANCE_M="$2"; shift 2 ;;
    --max-path-length-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; MAX_PATH_LENGTH_M="$2"; shift 2 ;;
    --v-max-goal-distance-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; V_MAX_GOAL_DISTANCE_M="$2"; shift 2 ;;
    --v-max-path-length-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; V_MAX_PATH_LENGTH_M="$2"; shift 2 ;;
    --max-length-ratio) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; MAX_LENGTH_RATIO="$2"; shift 2 ;;
    --v-goal-candidates) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; V_GOAL_CANDIDATES="$2"; shift 2 ;;
    --v-plan-candidates) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; V_PLAN_CANDIDATES="$2"; shift 2 ;;
    --v-hotspot-probability) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; V_HOTSPOT_PROBABILITY="$2"; shift 2 ;;
    --v-hotspot-radius-m) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; V_HOTSPOT_RADIUS_M="$2"; shift 2 ;;
    --plan-cell-factor) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; PLAN_CELL_FACTOR="$2"; shift 2 ;;
    --log-dir) (($# >= 2)) || { echo "ERROR: $1 needs a value" >&2; exit 2; }; LOG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Validate numeric
[[ "$N_GOALS" =~ ^[0-9]+$ ]] && ((N_GOALS >= 1)) \
  || { echo "ERROR: --n-goals must be an integer >= 1" >&2; exit 2; }
[[ "$STALL_TIMEOUT" =~ ^[0-9]+$ ]] \
  || { echo "ERROR: --stall-timeout must be a non-negative integer" >&2; exit 2; }

# Validate quotas and derive N_GOALS if explicit quotas given
if [[ -n "$CLASS_QUOTAS" ]]; then
  IFS='_' read -ra _Q <<< "$CLASS_QUOTAS"
  if (( ${#_Q[@]} != 4 )); then
    echo "ERROR: --st_sem_cmp_vc needs 4 values a_b_c_d" >&2; exit 2
  fi
  _SUM=0
  for _q in "${_Q[@]}"; do
    [[ "$_q" =~ ^[0-9]+$ ]] || { echo "ERROR: quota '$_q' not an integer" >&2; exit 2; }
    _SUM=$((_SUM + _q))
  done
  if ((_SUM < 1)); then
    echo "ERROR: --st_sem_cmp_vc sum must be >= 1" >&2; exit 2
  fi
  N_GOALS="$_SUM"
fi

# Filesystem checks
[[ -f "$GENERATOR" ]] || { echo "ERROR: path_generator.py not found: $GENERATOR" >&2; exit 2; }
[[ -d "$MAPS_ROOT" ]] || { echo "ERROR: maps root not found: $MAPS_ROOT" >&2; exit 2; }
[[ ! -L "$DATA_ROOT" ]] || { echo "ERROR: data root is a symlink: $DATA_ROOT" >&2; exit 2; }
_actual="$(realpath -m "$DATA_ROOT")"
_expected="$(realpath -m "$EXPECTED_DATA_ROOT")"
[[ "$_actual" == "$_expected" && "$DATA_ROOT" == "$EXPECTED_DATA_ROOT" ]] \
  || { echo "ERROR: data-root safety check failed: $DATA_ROOT" >&2; exit 2; }
python3 -c 'import numpy, yaml, PIL, scipy' \
  || { echo "ERROR: missing numpy/PyYAML/Pillow/SciPy" >&2; exit 2; }

# Purge
if ((PURGE_DATA)); then
  if ((ASSUME_YES == 0)); then
    if [[ -t 0 ]]; then
      read -r -p "Type PURGE to delete $DATA_ROOT: " _confirm
      [[ "$_confirm" == "PURGE" ]] || { echo "Aborted."; exit 1; }
    else
      echo "ERROR: --purge-data/--replace requires --yes in non-interactive mode" >&2
      exit 2
    fi
  fi
  echo "Purging: $DATA_ROOT"
  rm -rf -- "$DATA_ROOT"
fi

mkdir -p "$DATA_ROOT" "$LOG_DIR"
echo "Collection root: $DATA_ROOT"
echo "Map root: $MAPS_ROOT"
echo "Scenarios: $SCENARIOS_TEXT"
echo "Goals per scenario: $N_GOALS"
echo "Class quotas: ${CLASS_QUOTAS:-balanced}"
echo "Endpoint tolerance: ${ENDPOINT_TOL_M} m"
echo "Logs: $LOG_DIR"
: > "$LOG_DIR/teleop_batch.log"

# Validate scenario list
read -r -a SCENARIOS <<< "$SCENARIOS_TEXT"
((${#SCENARIOS[@]})) || { echo "ERROR: empty scenario list" >&2; exit 2; }
declare -A _SEEN=()
for _sc in "${SCENARIOS[@]}"; do
  [[ "$_sc" =~ ^E[0-9]{2}$ ]] || { echo "ERROR: invalid scenario '$_sc'" >&2; exit 2; }
  [[ -z "${_SEEN[$_sc]+x}" ]] || { echo "ERROR: duplicate scenario '$_sc'" >&2; exit 2; }
  _SEEN[$_sc]=1
  _md="$(find "$MAPS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${_sc}_*" -print -quit)"
  [[ -n "$_md" ]] || { echo "ERROR: no map dir for $_sc under $MAPS_ROOT" >&2; exit 2; }
done


# Phase 1: pool preparation (filter used endpoints, regen if deficits)

declare -A RUN_READY=()
gen_status=0

for _sc in "${SCENARIOS[@]}"; do
  _scnum=$((10#${_sc#E}))
  _sdir="$DATA_ROOT/$_sc"
  _pool="$_sdir/path_pool.json"
  _poses="$_sdir/path_pool_poses.json"
  mkdir -p "$_sdir"

  if ((RECORD_ONLY)); then
    if [[ -s "$_pool" && -s "$_poses" ]]; then
      echo "[$_sc] record-only: using existing pool"
      RUN_READY[$_sc]=1
    else
      echo "[$_sc] record-only: pool missing, skipping" >&2
    fi
    continue
  fi

  if [[ -n "$EXPLICIT_SEED" ]]; then
    _seed=$((EXPLICIT_SEED + _scnum))
  else
    _seed=$((RANDOM % 900000 + 100000))
  fi

  _ma_arg="$MAX_ATTEMPTS"
  _to_arg="$TIMEOUT_SECONDS"

  echo "[$_sc] preparing pool (filter+regen if needed)"
  _frc=0
  python3 - \
    "$_sdir" "$ENDPOINT_TOL_M" "$N_GOALS" "$CLASS_QUOTAS" \
    "$GENERATOR" "$_sc" "$MAPS_ROOT" "$DATA_ROOT" "$_seed" \
    "$HARD_CLEARANCE_M" "$ENDPOINT_OBSTACLE_BAND_M" \
    "$OBSTACLE_INFLUENCE_M" "$MIN_GOAL_DISTANCE_M" \
    "$MAX_GOAL_DISTANCE_M" "$MAX_PATH_LENGTH_M" \
    "$V_MAX_GOAL_DISTANCE_M" "$V_MAX_PATH_LENGTH_M" \
    "$MAX_LENGTH_RATIO" "$V_GOAL_CANDIDATES" "$V_PLAN_CANDIDATES" \
    "$V_HOTSPOT_PROBABILITY" "$V_HOTSPOT_RADIUS_M" \
    "$PLAN_CELL_FACTOR" "$PROGRESS_INTERVAL_SECONDS" \
    "$_ma_arg" "$_to_arg" \
    <<'PYEOF' || _frc=$?
import json, math, shutil, subprocess, sys
from pathlib import Path
from collections import Counter
import csv as _csv

(scen_dir, tol, n_goals, quotas_str, path_gen, scenario, maps_root,
 data_root, seed, hard_clearance_m, endpoint_band_m, obstacle_influence_m,
 min_goal_d, max_goal_d, max_path_len, v_max_goal_d, v_max_path_len,
 max_len_ratio, v_goal_cand, v_plan_cand, v_hotspot_prob, v_hotspot_rad,
 plan_cell_factor, progress_s, max_attempts, timeout_s) = sys.argv[1:28]

scen_dir = Path(scen_dir)
tol = float(tol)
n_goals = int(n_goals)
seed = int(seed)
CLASSES = ("straight", "semi_complex", "complex", "v_complex")

def parse_quotas(s, total):
    if not s:
        base, rem = divmod(total, 4)
        return {c: base + (1 if i < rem else 0)
                for i, c in enumerate(CLASSES)}
    parts = s.split("_")
    if len(parts) != 4:
        raise SystemExit(f"bad quotas: {s}")
    return {c: int(p) for c, p in zip(CLASSES, parts)}

quotas = parse_quotas(quotas_str, n_goals)

# Collect used endpoints from prior sessions
used = []
meta = scen_dir / "episode_metadata.jsonl"
if meta.is_file():
    for line in meta.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        s, g = e.get("start"), e.get("goal")
        if s and g and len(s) >= 2 and len(g) >= 2:
            used.append(((float(s[0]), float(s[1])),
                         (float(g[0]), float(g[1]))))

failed = scen_dir / "failed_paths.json"
if failed.is_file():
    try:
        data = json.loads(failed.read_text())
    except Exception:
        data = {}
    for e in data.get("entries", []):
        s, g = e.get("start"), e.get("goal")
        if s and g and len(s) >= 2 and len(g) >= 2:
            used.append(((float(s[0]), float(s[1])),
                         (float(g[0]), float(g[1]))))

# Endpoints from prior recorder CSVs (classic_nav, nav2, human)
for run_dir in sorted(scen_dir.glob("run_*")):
    csv_path = run_dir / "data.csv"
    if not csv_path.is_file():
        continue
    try:
        with csv_path.open("r", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    if int(row.get("is_episode_start", "0")) != 1:
                        continue
                    sx = float(row["pose_x"]); sy = float(row["pose_y"])
                    gx = float(row["goal_x"]); gy = float(row["goal_y"])
                except (KeyError, ValueError, TypeError):
                    continue
                used.append(((sx, sy), (gx, gy)))
    except OSError:
        continue

print(f"[prep] used endpoints (prior sessions): {len(used)}", flush=True)

def filter_pool():
    pf = scen_dir / "path_pool.json"
    pp = scen_dir / "path_pool_poses.json"
    if not (pf.is_file() and pp.is_file()):
        return 0, 0, {c: 0 for c in CLASSES}
    pool = json.loads(pf.read_text())
    poses = json.loads(pp.read_text())
    kept, removed = [], 0
    for p in pool.get("paths", []):
        s, g = p.get("start"), p.get("goal")
        if not s or not g:
            removed += 1
            continue
        ps = (float(s[0]), float(s[1]))
        pg = (float(g[0]), float(g[1]))
        dup = any(
            math.hypot(ps[0]-us[0], ps[1]-us[1]) < tol and
            math.hypot(pg[0]-ug[0], pg[1]-ug[1]) < tol
            for us, ug in used)
        if dup:
            removed += 1
            continue
        kept.append(p)
    pool["paths"] = kept
    pool["num_paths"] = len(kept)
    new_poses = {str(p["id"]): poses[str(p["id"])]
                 for p in kept if str(p["id"]) in poses}
    pf.write_text(json.dumps(pool, indent=2))
    pp.write_text(json.dumps(new_poses, indent=1))
    counts = Counter(p.get("class") for p in kept)
    return removed, len(kept), counts

removed, kept, counts = filter_pool()
print(f"[prep] pool filtered: removed={removed} kept={kept} "
      f"per_class={dict(counts)}", flush=True)

deficits = {c: max(0, quotas[c] - counts.get(c, 0)) for c in CLASSES}
if sum(deficits.values()) == 0:
    print(f"[prep] quotas covered: {quotas}", flush=True)
    sys.exit(0)

new_quotas = {c: max(1, quotas[c] + deficits[c] + 1) for c in CLASSES}
mix = "_".join(str(new_quotas[c]) for c in CLASSES)
run_name = f"teleop_pool_seed_{seed}"
print(f"[prep] deficits={deficits} -> regenerating mix={mix} seed={seed}",
      flush=True)

cmd = [sys.executable, path_gen,
       "--scenario", scenario,
       "--seed", str(seed),
       "--run-name", run_name,
       "--maps-root", maps_root,
       "--output-dir", data_root,
       "--progress-interval-seconds", progress_s,
       "--hard-clearance-m", hard_clearance_m,
       "--endpoint-obstacle-band-m", endpoint_band_m,
       "--obstacle-influence-m", obstacle_influence_m,
       "--min-goal-distance-m", min_goal_d,
       "--max-goal-distance-m", max_goal_d,
       "--max-path-length-m", max_path_len,
       "--v-max-goal-distance-m", v_max_goal_d,
       "--v-max-path-length-m", v_max_path_len,
       "--max-length-ratio", max_len_ratio,
       "--v-goal-candidates", v_goal_cand,
       "--v-plan-candidates", v_plan_cand,
       "--v-hotspot-probability", v_hotspot_prob,
       "--v-hotspot-radius-m", v_hotspot_rad,
       "--plan-cell-factor", plan_cell_factor,
       "--st_sem_cmp_vc", mix,
       "--overwrite-run"]
if max_attempts:
    cmd += ["--max-attempts", max_attempts]
if timeout_s:
    cmd += ["--timeout-seconds", timeout_s]

rc = subprocess.run(cmd).returncode
if rc != 0:
    print(f"[prep] generator failed rc={rc}", file=sys.stderr)
    sys.exit(3)

run_dir = scen_dir / run_name
gp = run_dir / "path_pool.json"
gpp = run_dir / "path_pool_poses.json"
if not (gp.is_file() and gpp.is_file()):
    print(f"[prep] generator outputs missing in {run_dir}", file=sys.stderr)
    sys.exit(3)

shutil.copy2(gp, scen_dir / "path_pool.json")
shutil.copy2(gpp, scen_dir / "path_pool_poses.json")

removed2, kept2, counts2 = filter_pool()
deficits2 = {c: max(0, quotas[c] - counts2.get(c, 0)) for c in CLASSES}
print(f"[prep] after regen: removed={removed2} kept={kept2} "
      f"deficits={deficits2}", flush=True)

if sum(deficits2.values()) > 0:
    print(f"[prep] WARNING still short after regen: {deficits2}. "
          f"Recording will run with available paths.",
          file=sys.stderr)

sys.exit(0)
PYEOF

  if ((_frc != 0)); then
    echo "[$_sc] pool prep failed rc=$_frc" >&2
    gen_status=1
    continue
  fi

  if [[ -s "$_pool" && -s "$_poses" ]]; then
    RUN_READY[$_sc]=1
    echo "[$_sc] pool ready: $_pool"
  else
    echo "[$_sc] pool missing after prep" >&2
    gen_status=1
  fi
done

if ! ros2 pkg prefix "$PKG" >/dev/null 2>&1; then
  echo "ERROR: ROS 2 package '$PKG' not found." >&2
  exit 2
fi


# Helpers

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

wait_for_recorder_ready() {
  local log_file="$1"
  local timeout_s="$2"
  local pid="$3"
  local t0
  t0=$(date +%s)
  while true; do
    if grep -q "Recorder ready" "$log_file" 2>/dev/null; then
      return 0
    fi
    if grep -Eq "process has died|Failed to launch|Traceback \(most recent" \
        "$log_file" 2>/dev/null; then
      return 1
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    if (( $(date +%s) - t0 >= timeout_s )); then
      return 2
    fi
    sleep 1
  done
}

count_recorded_samples() {
  local scen="$1"
  local before_list="$2"
  local parent="$DATA_ROOT/$scen"
  local total=0 d rows
  shopt -s nullglob
  for d in "$parent"/run_*; do
    [[ -d "$d" ]] || continue
    if [[ -n "$before_list" ]] \
        && printf '%s\n' "$before_list" | grep -qxF "$d"; then
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

count_committed_goals() {
  local scen="$1"
  local before_list="$2"
  local parent="$DATA_ROOT/$scen"
  local total=0 d n
  shopt -s nullglob
  for d in "$parent"/run_*; do
    [[ -d "$d" ]] || continue
    if [[ -n "$before_list" ]] \
        && printf '%s\n' "$before_list" | grep -qxF "$d"; then
      continue
    fi
    if [[ -f "$d/episode_tries.json" ]]; then
      n=$(python3 -c "import json,sys
try:
    print(len(json.load(open(sys.argv[1])).get('entries', [])))
except Exception:
    print(0)
" "$d/episode_tries.json" 2>/dev/null || echo 0)
      total=$((total + n))
    fi
  done
  shopt -u nullglob
  echo "$total"
}


# Phase 2: recording

FAILED=0
for _sc in "${SCENARIOS[@]}"; do
  ((GRACEFUL_STOP == 0)) || break
  _sdir="$DATA_ROOT/$_sc"
  _pool="$_sdir/path_pool.json"
  _poses="$_sdir/path_pool_poses.json"
  _status="$LOG_DIR/${_sc}.status"

  if [[ ! -s "$_pool" || ! -s "$_poses" ]]; then
    echo "[$_sc] SKIP: missing pool"
    printf 'scenario=%s\nreason=missing_pool\n' "$_sc" > "$_status"
    FAILED=1
    continue
  fi

  # Snapshot existing run dirs to isolate new ones
  shopt -s nullglob
  _before_arr=("$_sdir"/run_*)
  shopt -u nullglob
  _before=""
  ((${#_before_arr[@]})) \
    && _before="$(printf '%s\n' "${_before_arr[@]}")"

  _launch_ok=0
  _reason="world_start_failed"
  _last_log=""
  _pid=""
  _t_start=$(date +%s)
  _t_end=$_t_start

  for _attempt in $(seq 1 "$LAUNCH_RETRIES"); do
    ((GRACEFUL_STOP == 0)) || { _reason="interrupted"; break; }
    _log="$LOG_DIR/${_sc}_attempt${_attempt}.log"
    _last_log="$_log"
    echo "[$_sc] launch attempt $_attempt/$LAUNCH_RETRIES (log=$_log)"

    _targs=(auto_generate_pool:=false)
    if [[ -n "$CLASS_QUOTAS" ]]; then
      _targs+=(class_mix:="$CLASS_QUOTAS")
    fi

    setsid ros2 launch "$PKG" "$LAUNCH_FILE" \
      world:="$_sc" scenario:="$_sc" num_goals:="$N_GOALS" \
      output_dir:="$DATA_ROOT" \
      goal_position_tolerance_m:="$GOAL_POSITION_TOLERANCE_M" \
      "${_targs[@]}" "${EXTRA_ARGS[@]}" >"$_log" 2>&1 &
    _pid=$!
    CURRENT_PGID="$(ps -o pgid= -p "$_pid" 2>/dev/null | tr -d ' ' || true)"

    _ready=0
    wait_for_recorder_ready "$_log" "$WORLD_READY_TIMEOUT_S" "$_pid" || _ready=$?
    if (( _ready != 0 )); then
      echo "[$_sc] recorder not ready (rc=$_ready); retrying"
      cleanup_current
      wait "$_pid" 2>/dev/null || true
      _pid=""
      continue
    fi

    _launch_ok=1
    _reason="launch_exit"

    while kill -0 "$_pid" 2>/dev/null; do
      ((GRACEFUL_STOP == 0)) || { _reason="interrupted"; break; }
      if grep -qE "Stopping: all quotas met|Stopping: pool exhausted|Session complete:|All quotas met or pools exhausted" \
          "$_log" 2>/dev/null; then
        _reason="teleop_complete"
        sleep 3
        break
      fi
      if grep -Eq "PATH_POOL_ERROR|ERROR: No /map received|ERROR: Room build failed|No traversable cells found|stage=map_wait_failed" \
          "$_log" 2>/dev/null; then
        _reason="teleop_incomplete"
        break
      fi
      if ((STALL_TIMEOUT > 0)) && [[ -f "$_log" ]]; then
        _now=$(date +%s)
        _mtime=$(stat -c %Y "$_log")
        if (( _now - _mtime > STALL_TIMEOUT )); then
          _reason="stalled"
          break
        fi
      fi
      sleep 2
    done
    break
  done

  # Re-classify: on_exit="shutdown" may kill the launch before the in-loop
  # grep sees the completion message.
  if [[ "$_reason" == "launch_exit" && -n "$_last_log" ]] \
      && grep -qE "Stopping:|Session complete:" "$_last_log" 2>/dev/null; then
    _reason="teleop_complete"
  fi

  _t_end=$(date +%s)
  cleanup_current
  [[ -n "$_pid" ]] && wait "$_pid" 2>/dev/null || true

  _n_samples="$(count_recorded_samples "$_sc" "$_before")"
  _n_goals="$(count_committed_goals "$_sc" "$_before")"

  if (( _launch_ok == 0 )); then
    _reason="world_start_failed"
  elif (( _n_samples == 0 )); then
    _reason="recording_failed_no_data"
  fi

  _log_for_status="$_last_log"
  if [[ -n "$_last_log" ]]; then
    _base="$(realpath -m "$DATASET_DIR/data/logs")"
    _real="$(realpath -m "$_last_log")"
    [[ "$_real" == "$_base"/* ]] \
      && _log_for_status="${_real#"$_base"/}"
  fi

  printf 'scenario=%s\nduration_s=%s\nreason=%s\nsamples=%s\ngoals=%s\ngoals_target=%s\nlog=%s\n' \
    "$_sc" "$((_t_end - _t_start))" "$_reason" "$_n_samples" \
    "$_n_goals" "$N_GOALS" "$_log_for_status" > "$_status"
  echo "[$_sc] recording ended: $_reason (samples=$_n_samples) (goals saved=$_n_goals/$N_GOALS)"
  if [[ "$_reason" != "teleop_complete" && "$_reason" != "teleop_incomplete" ]]; then
    FAILED=1
    tail -n 25 "$_last_log" || true
  fi
done

if ((FAILED == 0)); then
  echo "Teleop collection complete for all scenarios."
  exit 0
fi
echo "Teleop collection finished with issues; inspect $LOG_DIR." >&2
exit 1