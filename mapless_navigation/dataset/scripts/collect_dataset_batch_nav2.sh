#!/usr/bin/env bash
# Batch recorder for the Nav2 backend. Sequential worlds, cleanup between runs.

set -u

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
LAUNCH_FILE=record_single_world_nav2.launch.xml
LOG_DIR="${DATA_ROOT}/batch_logs/${RUN_TIMESTAMP}"
PER_GOAL_TIMEOUT=60
STALL_TIMEOUT=600
CLEANUP_WAIT=3
EXTRA_ARGS=()

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
    pkill -TERM -f webots             2>/dev/null || true
    pkill -TERM -f justina_controller 2>/dev/null || true
    pkill -TERM -f random_goal_sampler 2>/dev/null || true
    pkill -TERM -f dataset_recorder   2>/dev/null || true
    pkill -TERM -f rviz2              2>/dev/null || true
    pkill -TERM -f velocity_smoother  2>/dev/null || true
    pkill -TERM -f "nav2_amcl"        2>/dev/null || true
    pkill -TERM -f "nav2_planner"     2>/dev/null || true
    pkill -TERM -f "nav2_controller"  2>/dev/null || true
    pkill -TERM -f "nav2_bt_navigator" 2>/dev/null || true
    pkill -TERM -f "nav2_behaviors"   2>/dev/null || true
    pkill -TERM -f "nav2_smoother"    2>/dev/null || true
    pkill -TERM -f "nav2_waypoint"    2>/dev/null || true
    pkill -TERM -f "nav2_map_server"  2>/dev/null || true
    pkill -TERM -f "nav2_lifecycle"   2>/dev/null || true
    sleep 2
    pkill -9 -f webots             2>/dev/null || true
    pkill -9 -f justina_controller 2>/dev/null || true
    pkill -9 -f random_goal_sampler 2>/dev/null || true
    pkill -9 -f dataset_recorder   2>/dev/null || true
    pkill -9 -f rviz2              2>/dev/null || true
    pkill -9 -f velocity_smoother  2>/dev/null || true
    pkill -9 -f "nav2_amcl"        2>/dev/null || true
    pkill -9 -f "nav2_planner"     2>/dev/null || true
    pkill -9 -f "nav2_controller"  2>/dev/null || true
    pkill -9 -f "nav2_bt_navigator" 2>/dev/null || true
    pkill -9 -f "nav2_behaviors"   2>/dev/null || true
    pkill -9 -f "nav2_smoother"    2>/dev/null || true
    pkill -9 -f "nav2_waypoint"    2>/dev/null || true
    pkill -9 -f "nav2_map_server"  2>/dev/null || true
    pkill -9 -f "nav2_lifecycle"   2>/dev/null || true
    sleep "$CLEANUP_WAIT"

    local leftover
    leftover=$(pgrep -f "webots|justina_controller|random_goal_sampler|dataset_recorder|rviz2|velocity_smoother|nav2_" | wc -l)
    if [[ "$leftover" -gt 0 ]]; then
        echo "  [cleanup] warning: $leftover process(es) still alive"
    fi
}

extract_counter() {
    grep -o "$2[0-9]*" "$1" 2>/dev/null | tail -1 | cut -d= -f2
}

run_world() {
    local world="$1"
    local seed="$2"
    local log="$LOG_DIR/${world}.log"
    local status="$LOG_DIR/${world}.status"

    local alive
    alive=$(pgrep -f "webots|random_goal_sampler|dataset_recorder|rviz2|velocity_smoother|nav2_" | wc -l)
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

        if grep -q "Reached max_goals" "$log" 2>/dev/null; then
            reason="recorder_done"
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
}

if [[ ${#EXPLICIT_SEEDS[@]} -gt 0 ]]; then
    echo "Batch start (nav2): ${#WORLDS[@]} worlds, goals=$NUM_GOALS, explicit seeds"
else
    echo "Batch start (nav2): ${#WORLDS[@]} worlds, goals=$NUM_GOALS, base_seed=$BASE_SEED"
fi
echo "Stall timeout: ${STALL_TIMEOUT}s (0 = disabled)"
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