#!/usr/bin/env bash
# Batch recorder for the classic navigation backend.
# Generates the path pool before launching each world, then records.
# Sequential worlds, cleanup between runs.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${SCRIPT_DIR}/../data"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

WORLDS=(
    E01 E02 E03 E04 E05 E06 E07 E08 E09 E10
    E11 E12 E13 E14 E15 E16 E17 E18 E19 E20
    E21 E22 E23 E24 E25 E26 E27 E28 E29 E30
)
NUM_GOALS=100
BASE_SEED=1001
EXPLICIT_SEEDS=()
PKG=mapless_dataset
LAUNCH_FILE=record_single_world_classic_nav.launch.xml
LOG_DIR="${DATA_ROOT}/batch_logs/${RUN_TIMESTAMP}"
PER_GOAL_TIMEOUT=60
STALL_TIMEOUT=600
CLEANUP_WAIT=3
EXTRA_ARGS=()

POOL_SCRIPT="${SCRIPT_DIR}/generate_path_pool.py"
POOL_NUM_PATHS=200
SKIP_POOL_GEN=0
POOL_FORCE=0
POOL_CONTINUE_ON_FAIL=0

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra launch args]

Options:
  --num-goals N         goals per scenario (default: $NUM_GOALS)
  --base-seed N         seed for first world; each next uses base+N
  --seeds "S1 S2 ..."   explicit seed per world, must match --worlds count
  --worlds "E01 E02"    space separated list
  --log-dir DIR         where to write logs
  --stall-timeout N     kill a world if the log has no writes for N s
                        (default: $STALL_TIMEOUT). Set 0 to disable.
  --skip-pool-gen       do not generate path pools; use existing ones
  --pool-force          regenerate path pools even if they already exist
  --pool-num-paths N    number of candidate paths per scenario
                        (default: $POOL_NUM_PATHS)
  --pool-continue-on-fail
                        if pool generation fails, skip the world instead
                        of aborting the batch
  -h, --help            this message

Any args after -- are passed straight to ros2 launch.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-goals) NUM_GOALS="$2"; shift 2 ;;
        --base-seed) BASE_SEED="$2"; shift 2 ;;
        --seeds)     IFS=' ' read -r -a EXPLICIT_SEEDS <<< "$2"; shift 2 ;;
        --worlds)    IFS=' ' read -r -a WORLDS <<< "$2"; shift 2 ;;
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --stall-timeout) STALL_TIMEOUT="$2"; shift 2 ;;
        --skip-pool-gen) SKIP_POOL_GEN=1; shift ;;
        --pool-force) POOL_FORCE=1; shift ;;
        --pool-num-paths) POOL_NUM_PATHS="$2"; shift 2 ;;
        --pool-continue-on-fail) POOL_CONTINUE_ON_FAIL=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        --)          shift; EXTRA_ARGS=("$@"); break ;;
        *)           echo "unknown arg: $1"; usage; exit 1 ;;
    esac
done

if [[ ${#EXPLICIT_SEEDS[@]} -gt 0 && ${#EXPLICIT_SEEDS[@]} -ne ${#WORLDS[@]} ]]; then
    echo "--seeds has ${#EXPLICIT_SEEDS[@]} values but --worlds has ${#WORLDS[@]}"
    exit 1
fi

if ! ros2 pkg prefix "$PKG" >/dev/null 2>&1; then
    echo "package $PKG not found. Run: source install/setup.bash"
    exit 1
fi

if [[ "$SKIP_POOL_GEN" -eq 0 && ! -f "$POOL_SCRIPT" ]]; then
    echo "pool generator not found: $POOL_SCRIPT"
    exit 1
fi

GRACEFUL_STOP=0

handle_signal() {
    echo ""
    echo "=== signal received, stopping batch ==="
    GRACEFUL_STOP=1
}

trap handle_signal SIGINT SIGTERM SIGQUIT
trap 'cleanup' EXIT

mkdir -p "$LOG_DIR"

cleanup() {
    pkill -INT  -f "ros2 launch $PKG" 2>/dev/null || true
    sleep 1
    pkill -TERM -f webots                    2>/dev/null || true
    pkill -TERM -f justina_controller        2>/dev/null || true
    pkill -TERM -f random_goal_sampler       2>/dev/null || true
    pkill -TERM -f dataset_recorder          2>/dev/null || true
    pkill -TERM -f simple_move_node          2>/dev/null || true
    pkill -TERM -f path_planner_node         2>/dev/null || true
    pkill -TERM -f obs_detect_node           2>/dev/null || true
    pkill -TERM -f map_augmenter_node        2>/dev/null || true
    pkill -TERM -f depth_to_pointcloud       2>/dev/null || true
    pkill -TERM -f local_occ_grid_node       2>/dev/null || true
    pkill -TERM -f get_goal_point            2>/dev/null || true
    pkill -TERM -f prohibition_map_publisher 2>/dev/null || true
    pkill -TERM -f rviz2                     2>/dev/null || true
    pkill -TERM -f "nav2_amcl"               2>/dev/null || true
    pkill -TERM -f "nav2_map_server"         2>/dev/null || true
    pkill -TERM -f "nav2_lifecycle"          2>/dev/null || true
    sleep 2
    pkill -9 -f webots                    2>/dev/null || true
    pkill -9 -f justina_controller        2>/dev/null || true
    pkill -9 -f random_goal_sampler       2>/dev/null || true
    pkill -9 -f dataset_recorder          2>/dev/null || true
    pkill -9 -f simple_move_node          2>/dev/null || true
    pkill -9 -f path_planner_node         2>/dev/null || true
    pkill -9 -f obs_detect_node           2>/dev/null || true
    pkill -9 -f map_augmenter_node        2>/dev/null || true
    pkill -9 -f depth_to_pointcloud       2>/dev/null || true
    pkill -9 -f local_occ_grid_node       2>/dev/null || true
    pkill -9 -f get_goal_point            2>/dev/null || true
    pkill -9 -f prohibition_map_publisher 2>/dev/null || true
    pkill -9 -f rviz2                     2>/dev/null || true
    pkill -9 -f "nav2_amcl"               2>/dev/null || true
    pkill -9 -f "nav2_map_server"         2>/dev/null || true
    pkill -9 -f "nav2_lifecycle"          2>/dev/null || true
    sleep "$CLEANUP_WAIT"

    local leftover
    leftover=$(pgrep -f "webots|justina_controller|random_goal_sampler|dataset_recorder|simple_move_node|path_planner_node|obs_detect_node|map_augmenter_node|rviz2|nav2_amcl|nav2_map_server" | wc -l)
    if [[ "$leftover" -gt 0 ]]; then
        echo "  [cleanup] warning: $leftover process(es) still alive"
    fi
}

extract_counter() {
    grep -o "$2[0-9]*" "$1" 2>/dev/null | tail -1 | cut -d= -f2
}

pool_is_current() {
    local pool_json="$1"
    local poses_json="$2"
    local requested_count="$3"
    local num_goals="$4"

    python3 - "$pool_json" "$poses_json" "$requested_count" "$num_goals" <<'PYPOOLCHECK'
import json, math, sys
pool_path, poses_path = sys.argv[1], sys.argv[2]
requested_count, num_goals = int(sys.argv[3]), int(sys.argv[4])
try:
    with open(pool_path) as f:
        pool = json.load(f)
    with open(poses_path) as f:
        poses = json.load(f)
    constraints = pool.get('constraints', {})
    classes = ('straight', 'semi_complex', 'complex', 'v_complex')
    distribution = pool.get('distribution', {})
    quota = pool.get('quota', {})
    minimum_class = max(50, 4 * ((num_goals + 3) // 4))
    balanced_quota = all(int(quota.get(c, 0)) == 25 for c in classes)
    usable = int(pool.get('num_paths', 0))
    enough_source = usable >= requested_count
    current = (
        int(pool.get('pool_version', 0)) >= 4
        and pool.get('classifier') == 'geometry_morphology_v2'
        and pool.get('classification_scope') == 'geometry_only'
        and balanced_quota
        and enough_source
        and len(poses) >= usable
        and all(int(distribution.get(c, 0)) >= minimum_class for c in classes)
        and math.isclose(float(constraints.get('min_goal_distance_m', -1)), 2.0, abs_tol=1e-6)
        and math.isclose(float(constraints.get('max_goal_distance_m', -1)), 3.0, abs_tol=1e-6)
        and math.isclose(float(constraints.get('max_path_length_m', -1)), 3.0, abs_tol=1e-6)
        and math.isclose(float(constraints.get('min_clearance_m', -1)), 0.15, abs_tol=1e-6)
    )
    sys.exit(0 if current else 1)
except Exception:
    sys.exit(1)
PYPOOLCHECK
}

ensure_path_pool() {
    local world="$1"
    local pool_json="${DATA_ROOT}/${world}/path_pool.json"
    local pool_poses="${DATA_ROOT}/${world}/path_pool_poses.json"
    local pool_log="${LOG_DIR}/${world}_pool.log"
    local min_paths_per_class=$(( 4 * ((NUM_GOALS + 3) / 4) ))
    if (( min_paths_per_class < 50 )); then min_paths_per_class=50; fi

    if [[ "$SKIP_POOL_GEN" -eq 1 ]]; then
        if [[ ! -f "$pool_json" || ! -f "$pool_poses" ]]; then
            echo "[$world] pool: missing JSON or poses file while --skip-pool-gen is set"
            return 1
        fi
        if pool_is_current "$pool_json" "$pool_poses" "$POOL_NUM_PATHS" "$NUM_GOALS"; then
            echo "[$world] pool: skipped generation; compatible pool verified"
            return 0
        fi
        echo "[$world] pool: --skip-pool-gen requires a version-4 geometric-morphology pool with enough routes"
        return 1
    fi

    if [[ -f "$pool_json" && -f "$pool_poses" && "$POOL_FORCE" -eq 0 ]]; then
        if pool_is_current "$pool_json" "$pool_poses" "$POOL_NUM_PATHS" "$NUM_GOALS"; then
            echo "[$world] pool: compatible pool exists, reusing"
            return 0
        fi

        echo "[$world] pool: existing pool is old/incompatible; trying in-place upgrade first"
        if python3 "$POOL_SCRIPT" \
            --upgrade-existing \
            --scenario "$world" \
            --input-dir "$DATA_ROOT" \
            --num-paths "$POOL_NUM_PATHS" \
            --min-goal-distance 2.0 \
            --max-goal-distance 3.0 \
            --max-path-length 3.0 \
            --min-clearance 0.15 \
            --min-paths-per-class "$min_paths_per_class" \
            --quota "25,25,25,25" \
            2>&1 | tee "$pool_log"; then
            if pool_is_current "$pool_json" "$pool_poses" "$POOL_NUM_PATHS" "$NUM_GOALS"; then
                echo "[$world] pool: upgrade OK, reusing filtered paths"
                return 0
            fi
        fi
        echo "[$world] pool: upgrade could not provide enough valid routes; generating a new pool"
    fi

    echo "[$world] pool: generating (target: $POOL_NUM_PATHS safe paths)"
    local start_ts end_ts dur
    start_ts=$(date +%s)

    if python3 "$POOL_SCRIPT" \
        --scenario "$world" \
        --num-paths "$POOL_NUM_PATHS" \
        --min-goal-distance 2.0 \
        --max-goal-distance 3.0 \
        --max-path-length 3.0 \
        --min-clearance 0.15 \
        --min-paths-per-class "$min_paths_per_class" \
        --quota "25,25,25,25" \
        --output-dir "$DATA_ROOT" \
        2>&1 | tee -a "$pool_log"
    then
        if ! pool_is_current "$pool_json" "$pool_poses" "$POOL_NUM_PATHS" "$NUM_GOALS"; then
            echo "[$world] pool: generator exited but pool validation failed"
            return 1
        fi
        end_ts=$(date +%s)
        dur=$((end_ts - start_ts))
        echo "[$world] pool: OK (${dur}s)"
        return 0
    fi

    echo "[$world] pool: FAILED (see $pool_log)"
    return 1
}
run_world() {
    local world="$1"
    local seed="$2"
    local log="$LOG_DIR/${world}.log"
    local status="$LOG_DIR/${world}.status"

    if ! ensure_path_pool "$world"; then
        {
            echo "world=$world"
            echo "seed=$seed"
            echo "num_goals=$NUM_GOALS"
            echo "duration_s=0"
            echo "reason=pool_generation_failed"
            echo "committed=0"
            echo "collision_events=0"
            echo "stuck_events=0"
            echo "Data root: $DATA_ROOT"
        } > "$status"
        if [[ "$POOL_CONTINUE_ON_FAIL" -eq 1 ]]; then
            echo "[$world] pool failed, skipping world and counting it as failed"
            return 1
        fi
        echo "[$world] pool failed, aborting world"
        return 2
    fi

    local alive
    alive=$(pgrep -f "webots|random_goal_sampler|dataset_recorder|simple_move_node|path_planner_node|obs_detect_node|map_augmenter_node|rviz2" | wc -l)
    if [[ "$alive" -gt 0 ]]; then
        echo "[$world] zombies detected before launch, cleaning first"
        cleanup
    fi

    echo "==============================================="
    echo "[$world] start  seed=$seed  goals=$NUM_GOALS"
    echo "==============================================="

    setsid ros2 launch "$PKG" "$LAUNCH_FILE" \
        world:="$world" \
        scenario:="$world" \
        num_goals:="$NUM_GOALS" \
        seed:="$seed" \
        output_dir:="$DATA_ROOT" \
        "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    local pid=$!
    local pgid
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')

    sleep 3
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$world] launch died within 3s. tail of log:"
        tail -20 "$log"
        cleanup
        return 2
    fi

    local start_ts
    start_ts=$(date +%s)
    local reason="finished"

    while kill -0 "$pid" 2>/dev/null; do
        if [[ "$GRACEFUL_STOP" -eq 1 ]]; then
            reason="interrupted"
            break
        fi

        if grep -q "All requested goals collected" "$log" 2>/dev/null; then
            reason="sampler_complete"
            sleep 5
            break
        fi

        if grep -Eq "PATH_POOL_ERROR|Sampler stopped before target|Sampler terminated incomplete|Incomplete dataset|ERROR: No /map received|ERROR: Room build failed|No traversable cells found|Sampler cannot start goal search|stage=map_wait_failed" "$log" 2>/dev/null; then
            reason="sampler_incomplete"
            echo "[$world] sampler stopped before collecting all requested goals"
            tail -35 "$log"
            sleep 3
            break
        fi

        if [[ "$STALL_TIMEOUT" -gt 0 && -f "$log" ]]; then
            local log_mtime now_ts idle
            log_mtime=$(stat -c %Y "$log")
            now_ts=$(date +%s)
            idle=$((now_ts - log_mtime))
            if (( idle > STALL_TIMEOUT )); then
                reason="stalled"
                echo "[$world] no log activity for ${idle}s, killing"
                break
            fi
        fi

        sleep 2
    done

    if [[ -n "${pgid:-}" ]]; then
        kill -INT -- -"$pgid" 2>/dev/null || true
    else
        kill -INT "$pid" 2>/dev/null || true
    fi
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done

    if kill -0 "$pid" 2>/dev/null; then
        if [[ -n "${pgid:-}" ]]; then
            kill -TERM -- -"$pgid" 2>/dev/null || true
        else
            kill -TERM "$pid" 2>/dev/null || true
        fi
        sleep 2
        if [[ -n "${pgid:-}" ]]; then
            kill -KILL -- -"$pgid" 2>/dev/null || true
        else
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi

    wait "$pid" 2>/dev/null || true

    local end_ts dur
    end_ts=$(date +%s)
    dur=$((end_ts - start_ts))

    local committed coll stuck
    committed=$(extract_counter "$log" "Done. committed=")
    coll=$(extract_counter "$log" "collision_events=")
    stuck=$(extract_counter "$log" "stuck_events=")

    {
        echo "world=$world"
        echo "seed=$seed"
        echo "num_goals=$NUM_GOALS"
        echo "duration_s=$dur"
        echo "reason=$reason"
        echo "committed=${committed:-unknown}"
        echo "collision_events=${coll:-unknown}"
        echo "stuck_events=${stuck:-unknown}"
        echo "Data root: $DATA_ROOT"
    } > "$status"

    echo "[$world] done  ${dur}s  reason=$reason  committed=${committed:-?}"

    cleanup
    if [[ "$reason" != "sampler_complete" ]]; then
        echo "[$world] FAILED: sampler did not report complete dataset (reason=$reason)"
        return 1
    fi
    return 0
}

if [[ ${#EXPLICIT_SEEDS[@]} -gt 0 ]]; then
    echo "Batch start (classic): ${#WORLDS[@]} worlds, goals=$NUM_GOALS, explicit seeds"
else
    echo "Batch start (classic): ${#WORLDS[@]} worlds, goals=$NUM_GOALS, base_seed=$BASE_SEED"
fi
echo "Stall timeout: ${STALL_TIMEOUT}s (0 = disabled)"
echo "Pool generation: $([ "$SKIP_POOL_GEN" -eq 1 ] && echo "disabled" || echo "enabled (num_paths=$POOL_NUM_PATHS)")"
echo "Logs: $LOG_DIR"
cleanup

FAILED=0
idx=0
for world in "${WORLDS[@]}"; do
    if [[ "$GRACEFUL_STOP" -eq 1 ]]; then
        echo "batch interrupted, skipping remaining worlds"
        break
    fi
    if [[ ${#EXPLICIT_SEEDS[@]} -gt 0 ]]; then
        seed="${EXPLICIT_SEEDS[$idx]}"
    else
        seed=$((BASE_SEED + idx))
    fi
    if ! run_world "$world" "$seed"; then
        echo "[$world] FAILED"
        FAILED=$((FAILED + 1))
    fi
    idx=$((idx + 1))
done

cleanup

echo "==============================================="
echo "Batch done. Failed: $FAILED / ${#WORLDS[@]}"
echo "Logs: $LOG_DIR"
echo "==============================================="

exit $FAILED