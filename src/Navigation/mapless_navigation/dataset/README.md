# Dataset Pipeline

Offline data collection for mapless navigation. Generates path pools, records
Webots simulations with multiple teachers (Classic Navigation, Nav2,
teleoperation), converts runs to NPZ, and produces reproducible
train/validation splits.

## Requirements

- ROS 2 (Jazzy or newer)
- Webots
- Nav2 (only for the Nav2 teacher pipeline)
- Python 3.12 with: numpy, PyYAML, Pillow, SciPy, matplotlib
- Compiled package: `colcon build --packages-select mapless_dataset`

Run all commands from the repository root:

    cd ~/GIT/mapless
    source install/setup.bash

## Directory layout

    src/Navigation/mapless_navigation/dataset/
    ├── CMakeLists.txt
    ├── package.xml
    ├── README.md
    ├── config/
    │   └── nav2_params_justina.yaml       Nav2 parameters for Justina
    ├── launch/
    │   ├── record_single_world_classic_nav.launch.xml
    │   ├── record_single_world_nav2.launch.xml
    │   ├── record_single_world_teleop.launch.xml
    │   ├── record_teleop_obstacles.launch.xml
    │   └── nav2_bringup_justina.launch.xml
    ├── scripts/
    │   ├── collect_dataset_batch_classic_nav.sh
    │   ├── collect_dataset_batch_nav2.sh
    │   ├── generate_path_pool.py
    │   ├── path_generator.py
    │   ├── random_goal_sampler.py
    │   ├── dataset_obstacles_control.py
    │   ├── teleop_dataset_recorder.py
    │   ├── Nav2_map_relay.py
    │   ├── prohibition_map_publisher.py
    │   ├── generate_npz_data.py
    │   ├── make_dataset_split.py
    │   ├── data_analiysis.py
    │   └── README.md
    ├── src/
    │   └── dataset_recorder.cpp
    └── data/
        ├── E01/
        │   ├── path_pool.json              active pool used by the sampler
        │   ├── path_pool_poses.json        dense poses per pool path
        │   ├── default_pool/               generator run output
        │   │   ├── path_pool.json
        │   │   ├── path_pool_poses.json
        │   │   ├── path_pool.csv
        │   │   ├── report.json
        │   │   ├── report.md
        │   │   └── routes_<class>.png
        │   └── run_YYYYMMDD_HHMMSS_<teacher>/
        │       ├── data.csv
        │       ├── descriptor.yaml
        │       ├── episode_tries.json
        │       └── lom/
        │           └── NNNNNNN.png
        ├── E02/ ... E30/
        └── logs/
            └── run_YYYYMMDD_HHMMSS/
                ├── E01.status
                ├── E01_recording_attempt1.log
                └── generation_batch.log

## Teachers and pipelines

Three independent teacher pipelines share the same recorder, data format and
NPZ converter. They differ in how the robot is driven and in how the pool is
consumed.

| Teacher | Batch script | Launch file | Notes |
|---|---|---|---|
| `classic_nav` | `collect_dataset_batch_classic_nav.sh` | `record_single_world_classic_nav.launch.xml` | A* planner + potential fields + localization |
| `nav2` | `collect_dataset_batch_nav2.sh` | `record_single_world_nav2.launch.xml` | Nav2 stack with `nav2_params_justina.yaml` |
| `teleop` | none (manual) | `record_single_world_teleop.launch.xml` | Human operator via joystick, no pool required |

`teleop` has an additional variant `record_teleop_obstacles.launch.xml` for
teleoperated sessions with dynamically placed obstacles (uses
`dataset_obstacles_control.py` to spawn and move obstacles during the run).

## Workflow

Full pipeline from zero for the classic_nav teacher:

    # 1. Collect pools and recordings for a set of scenarios
    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
        --scenarios "E01 E02 E03" \
        --n-goals 200 \
        --max-attempts 5000 \
        --timeout-seconds 900 \
        --seed 42

    # 2. Convert all runs to NPZ
    python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py

    # 3. Split into train / validation A / validation B
    python3 src/Navigation/mapless_navigation/dataset/scripts/make_dataset_split.py

For the Nav2 teacher, replace step 1 with:

    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_nav2.sh \
        --scenarios "E01 E02 E03" \
        --n-goals 200 \
        --seed 42

For teleoperation, run the launch file directly and drive the robot with a
joystick. No pool is needed; episodes are labelled with class `unknown`
unless `backfill_classes.py` is run later.

## Collection scripts

### Classic Navigation

    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh [options]

| Flag | Meaning |
|---|---|
| `--scenarios "E01 E02"` | Scenarios to process |
| `--n-goals N` | Target goals per scenario |
| `--st_sem_cmp_vc A_B_C_D` | Per-class quotas (straight_semi_complex_complex_v_complex) |
| `--max-attempts N` | Generation attempt limit per scenario |
| `--timeout-seconds N` | Generation wall-time limit per scenario |
| `--seed N` | Base seed; each scenario uses N + scenario number |
| `--incremental` | Reuse existing pools, regenerate only if quota missing |
| `--record-only` | Skip generation, record with existing pools |
| `--purge-data` | Delete data/ before running; **no confirmation prompt** |
| `--replace` | Alias for --purge-data (also destructive, no prompt) |
| `--log-dir DIR` | Override default log directory |

### Nav2

    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_nav2.sh [options]

Shares most options with the classic_nav batch script. The Nav2 parameters
are loaded from `config/nav2_params_justina.yaml`; the map is relayed by
`Nav2_map_relay.py` from the Webots world. Additional launch arguments
(`--`) are forwarded to `ros2 launch`.

Example with a custom parameter file:

    bash collect_dataset_batch_nav2.sh \
        --scenarios "E04" \
        --n-goals 50 \
        -- -p nav2_params:=/abs/path/to/nav2_params.yaml

### Teleoperation

Teleop has no batch script because goals and trajectories are not
precomputed. Launch the recording session and drive with a joystick:

    ros2 launch mapless_dataset record_single_world_teleop.launch.xml \
        scenario:=E01 world:=E01

To add dynamically moving obstacles during a teleop session:

    ros2 launch mapless_dataset record_teleop_obstacles.launch.xml \
        scenario:=E01 world:=E01

## Path generation (classic_nav and nav2)

Two scripts participate in pool generation:

- **`path_generator.py`** — main generator, used by the collection
  script in the normal (non-incremental) path.
- **`generate_path_pool.py`** — used only by `--incremental` mode.

The main generator can also be run standalone:

    python3 src/Navigation/mapless_navigation/dataset/scripts/path_generator.py \
        --scenario E01 \
        --n-goals 200 \
        --seed 42

Options:

| Flag | Meaning |
|---|---|
| `--scenario E##` | Required, map prefix |
| `--n-goals N` | Total target paths, split evenly by class |
| `--st_sem_cmp_vc A_B_C_D` | Explicit per-class quotas |
| `--max-attempts N` | Attempt limit |
| `--timeout-seconds N` | Wall-time limit |
| `--seed N` | Deterministic seed |
| `--hard-clearance-m M` | Minimum clearance (default 0.35) |
| `--plan-cell-factor N` | Coarse grid factor (default 5) |

Output is installed under `data/<scenario>/generation_seed_<seed>/` (with
`--seed`) or `data/<scenario>/default_pool/` (without). The collection
script copies it to `data/<scenario>/path_pool.json` for the sampler to
consume.

`prohibition_map_publisher.py` publishes the per-scenario prohibition map
used by the planner to reject goals inside forbidden regions.

## Sampler, recorder and teleop helpers

`random_goal_sampler.py` and `dataset_recorder` run inside the ROS 2 launch
files. They are not invoked directly. The sampler:

1. Loads the active pool from `data/<scenario>/path_pool.json`
2. For each path: teleports the robot to `start`, navigates along `poses`
3. Publishes episode id, tries counter, and class for each path
4. On success, publishes goal_success=true; on failure, retries up to
   `path_max_attempts` before writing the path to `failed_paths.json`

`teleop_dataset_recorder.py` replaces the sampler for teleop sessions. It
subscribes to the same topics as `dataset_recorder`, but triggers an episode
on a joystick button press instead of a pool path. Class labels default to
`unknown` and can be backfilled later.

`dataset_obstacles_control.py` spawns and animates obstacles in the Webots
world during a recording session. It is used only by
`record_teleop_obstacles.launch.xml`.

The recorder (C++ node `dataset_recorder`) subscribes to:

| Topic | Type | Purpose |
|---|---|---|
| `/mapless/local_occ_grid` | OccupancyGrid | 80x80 crop, saved as PNG |
| `/hardware/base/cmd_vel` | Twist | Recorded as vx, vy, wz |
| `/NN_goal` | Float32MultiArray | d and theta relative to goal |
| `/mapless_nav/recording_goal` | PointStamped | Goal in map frame |
| `/mapless_nav/goal_active` | Bool | Starts and stops episode buffer |
| `/mapless_nav/episode_id` | Int32 | Groups samples of one path |
| `/mapless_nav/episode_tries` | Int32 | Retry counter within episode |
| `/mapless_nav/episode_class` | String | Class label (straight, ...) |
| `/mapless_nav/goal_success` | Bool | Commits or discards buffer |
| `/navigation/collision` | Bool | Sets collision flag |

## Launch files

| File | Purpose |
|---|---|
| `record_single_world_classic_nav.launch.xml` | Full classic_nav recording session |
| `record_single_world_nav2.launch.xml` | Full Nav2 recording session |
| `record_single_world_teleop.launch.xml` | Teleop recording session (joystick) |
| `record_teleop_obstacles.launch.xml` | Teleop with dynamically spawned obstacles |
| `nav2_bringup_justina.launch.xml` | Nav2 stack bringup (included by the Nav2 recording launch) |

## Data formats

### data.csv

One row per recorded sample:

| Column | Type | Meaning |
|---|---|---|
| timestamp | float | Sim time, seconds |
| episode_id | int | Unique per path attempt |
| is_episode_start | int | 1 on first sample of an episode |
| d | float | Distance to goal, meters |
| theta | float | Bearing to goal, radians |
| vx, vy, wz | float | Velocity command |
| pose_x, pose_y, pose_yaw | float | Robot pose in map frame |
| goal_x, goal_y | float | Goal in map frame |
| lom_file | string | PNG filename under lom/ |
| collision_flag | int | 0 or 1 |
| stuck_flag | int | 0 or 1 |

Note: `is_episode_start` is set correctly for all teachers, including
teleop (the recorder triggers it whenever `episode_id` or `episode_tries`
changes). Very old runs produced before the class-logging change may have
`is_episode_start=0` everywhere; those must be inspected individually.

### episode_tries.json

One entry per committed episode:

    {
      "scenario": "E01",
      "teacher": "classic_nav",
      "entries": [
        {"episode_id": 1, "class": "straight", "tries": 1},
        {"episode_id": 2, "class": "semi_complex", "tries": 2}
      ]
    }

`teacher` is one of `classic_nav`, `nav2`, `teleop`.

### descriptor.yaml

Run metadata: scenario, teacher, record_hz, map_frame, base_frame,
lom_encoding, stuck_* thresholds. After conversion, `num_samples` is added.

### .status file

One per scenario, under logs/run_*/:

    scenario=E01
    duration_s=4285
    reason=sampler_complete
    samples=8342
    goals=199
    goals_target=199
    saved_classes=straight=50 semi_complex=50 complex=50 v_complex=49
    log=/mapless/.../E01_recording_attempt1.log

`reason` values:

| Value | Meaning |
|---|---|
| sampler_complete | All requested goals recorded |
| sampler_incomplete | Sampler stopped before reaching target |
| world_start_failed | Webots or launch failed to start |
| recording_failed_no_data | Recorder produced no samples |
| pool_incompatible_no_data | Pool rejected by sampler |
| stalled | No log activity for stall timeout |
| missing_pool | No pool file found |
| empty_pool | Pool has zero paths |

### NPZ output

Generated by `generate_npz_data.py`. One file per run directory:

    data/E01/run_YYYYMMDD_HHMMSS_classic_nav.npz

Structure:

    import numpy as np
    with np.load("data/E01/run_YYYYMMDD_HHMMSS_classic_nav.npz", allow_pickle=True) as a:
        for sample in a["data"]:
            lom   = sample["features"]["occ_grid"]   # 80x80 uint8
            target= sample["features"]["target"]     # [d, theta]
            cmd   = sample["labels"]                 # [vx, vy, wz]
            meta  = sample["meta"]                   # pose, goal_xy, timestamp,
                                                     # episode_id, is_episode_start,
                                                     # collision_flag, stuck_flag

## NPZ conversion

Convert all runs under data/:

    python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py

Convert specific scenarios only:

    python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py \
        --scenarios E01 E03

Limit episodes per scenario (keeps the first N by episode_id):

    python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py \
        --goals-per-scene 50

Random sample before cutting:

    python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py \
        --goals-per-scene 50 --shuffle --seed 42

Per-class quotas (requires class in episode_tries.json):

    python3 src/Navigation/mapless_navigation/dataset/scripts/generate_npz_data.py \
        --st-sem-cmp-vc 20_20_20_20

Default behavior with no filters: keep every episode of every run.

## Splitting

Produce `data/dataset_split.json`:

    python3 src/Navigation/mapless_navigation/dataset/scripts/make_dataset_split.py

Options:

| Flag | Meaning |
|---|---|
| `--train-spl X` | Fraction of scenarios in train (default 0.75) |
| `--validation-a-spl X` | Fraction of train episodes held out per scenario (default 0.25) |
| `--seed N` | Random seed (default 42) |

The split separates:

- TRAIN scenarios: used for training
- VALIDATION A: held-out episodes within train scenarios
- VALIDATION B: complete scenarios never seen during training

`dataset_split.json` records train_scenarios, validation_b_scenarios,
train_episodes, and validation_a_episodes. Each episode entry contains
scenario, run (relative path), and episode_id.

Splits are reproducible with the same seed and the same set of scenarios.
Adding a scenario changes the assignment.

## Analysis tools

`data_analiysis.py` (sic) inspects one or more runs and produces summary
statistics: episodes per class, samples per episode, velocity distribution,
collision and stuck counts, and per-scenario coverage. Use it after a
recording campaign to detect classes with too few samples or scenarios that
failed silently.

    python3 src/Navigation/mapless_navigation/dataset/scripts/data_analiysis.py \
        --data-root data \
        --output reports/analysis_$(date +%Y%m%d).md

The script tolerates the `unknown` class (teleop runs) and reports it
separately so it does not skew per-class statistics.

## Backfilling class labels

For runs produced before the class logging change (and for teleop runs),
`episode_tries.json` may have no `class` field. Fill it from the pool:

    python3 src/Navigation/mapless_navigation/dataset/scripts/backfill_classes.py \
        --data-root data \
        --backup

The script matches each episode to a pool path by comparing `goal_x, goal_y`
from the CSV with the goal of each path in `path_pool.json`. Pose at the
first sample is used to break ties between paths with the same goal.

Options:

| Flag | Meaning |
|---|---|
| `--data-root DIR` | Path to data/ (default src/Navigation/.../data) |
| `--tol M` | Max pose drift before flagging ambiguous (default 0.50) |
| `--dry-run` | Show what would change without writing |
| `--backup` | Save .bak next to each modified file |

Also writes `saved_classes=...` lines into every .status file for each
scenario processed.

Recommended sequence:

    # Dry run first
    python3 backfill_classes.py --data-root data --dry-run

    # Check the max_score per run and unmatched counts
    # max_score above 0.30 suggests a wrong pool or missing anchors

    # Apply with backups
    python3 backfill_classes.py --data-root data --backup

Revert a scenario if needed:

    cp data/E01/run_*/episode_tries.json.bak data/E01/run_*/episode_tries.json

## Common tasks

Check what scenarios exist:

    ls -d data/E*/ | xargs -n1 basename

List status reason per scenario:

    grep -H '^reason=' data/logs/run_*/E*.status

Find scenarios that failed recording:

    grep -l 'reason=world_start_failed\|reason=recording_failed_no_data' \
        data/logs/run_*/E*.status

Re-record a single scenario with a fresh pool (classic_nav):

    rm -rf data/E07
    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
        --scenarios "E07" \
        --n-goals 200 \
        --max-attempts 5000 \
        --timeout-seconds 900 \
        --seed 42

Re-record only (keep the existing pool):

    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_classic_nav.sh \
        --scenarios "E07" \
        --record-only

Re-record with Nav2:

    bash src/Navigation/mapless_navigation/dataset/scripts/collect_dataset_batch_nav2.sh \
        --scenarios "E07" \
        --n-goals 200 \
        --seed 42

## Troubleshooting

Pool generation exits with no paths:
  Check that `map_<scenario>_prohibition.yaml` exists in the maps folder.
  Check that `hard-clearance-m` is not too high for the map. E01 with 0.35
  works; some small maps need 0.25.

Sampler exits with PATH_POOL_ERROR quota_mismatch:
  The active pool's quota in `path_pool.json` must sum to `num_goals` given
  to the sampler. The collection script rewrites the quota to match the
  actual path count; standalone use of the sampler needs manual adjustment.

Sampler exits with PATH_POOL_ERROR insufficient_class_paths:
  The pool has fewer paths in a class than its quota. Regenerate with more
  attempts or a longer timeout, or relax the class quota.

Recording produces zero samples:
  Open the corresponding `E##_recording_attempt1.log` under logs/run_*/.
  Look for launch failures, Webots crashes, or missing topics. Common cause
  is Webots not starting within WORLD_READY_TIMEOUT_S (default 60).

`generate_npz_data.py` fails with "no run_* folders":
  Pass an explicit path argument (`data/`), or run from the repository root
  so the default `src/Navigation/.../data` resolves correctly. Relative
  paths are resolved under dataset/data, not the current directory.

`make_dataset_split.py` fails with "fewer than two episodes":
  Every scenario in the train set needs at least two committed episodes.
  Remove or re-record scenarios that produced fewer.

`episode_tries.json` has only `tries` and no `class`:
  This run predates class logging. Run `backfill_classes.py` to fill the
  class field from the pool.

`data.csv` has `is_episode_start=0` everywhere:
  Very old runs produced before the class-logging change. `backfill_classes.py`
  does not need this field; it uses the first occurrence of each `episode_id`
  in the CSV. Teleop runs **do** set it correctly; only pre-change runs are
  affected.

Nav2 recording produces no goals:
  Verify that `nav2_params_justina.yaml` matches the robot frame names and
  that `Nav2_map_relay.py` is publishing `/map`. Check the Nav2 launch log
  under logs/run_*/. The relay node must be alive before the sampler starts.

Obstacles do not move during teleop_obstacles:
  Confirm `dataset_obstacles_control.py` is running and that the scenario
  name passed to the launch file matches a key in its obstacle config.

## Corrections and additional notes (v2)

These supplement the earlier sections, based on reviewing the actual
implementation in `dataset_recorder.cpp`, `random_goal_sampler.py`,
`teleop_dataset_recorder.py` and `collect_dataset_batch_classic_nav.sh`.

### Generator script naming

The main generator is **`path_generator.py`**. `generate_path_pool.py` is
used only by `--incremental` mode.

### `--purge-data` has no confirmation prompt

`--purge-data` (and its alias `--replace`) deletes `data/` without asking.
The variable `generation_manifest.json` appears in the collection script
but the file is **not** actually produced; use `logs/run_*/E##.status`
instead.

### Pool folder naming

Without `--seed`: `data/<scenario>/default_pool/`. With `--seed N`:
`data/<scenario>/generation_seed_<N + scenario_number>/`. Both are copied
to `path_pool.json` / `path_pool_poses.json` at the scenario root.

### Pool version check

The sampler accepts `pool_version >= 5` with
`classification_scope == "navigation_task_and_geometry"`, or legacy
`pool_version >= 4` with `classifier == "geometry_morphology_v2"`.
Otherwise: `PATH_POOL_ERROR incompatible_pool_version`. The sampler also
verifies `pool['scenario'] == <current scenario>`.

### `failed_paths.json` is accumulative

Merged by `path_id` on every write. Reset manually with
`rm -f data/<scenario>/failed_paths.json`.

### Episode ID and retries

One `episode_id` per path. Retries increment only `episode_tries`. The
recorder discards its buffer on each `episode_id`/`episode_tries` change,
so only the last try reaches `data.csv`. Every committed episode in
`data.csv` therefore comes from a single trajectory.

### Recorder behavior details

- First `stabilization_time_sec` (default 3 s) are not recorded.
- Static samples (command magnitude < `min_cmd_magnitude`, default 0.01)
  are dropped, except the first sample of each episode.
- `collision_flag` and `stuck_flag` back-mark the previous
  `*_lookback_frames` (default 5) rows; a flag means the event occurred
  at or within N ticks **after** that sample.
- On shutdown (Ctrl+C or `max_goals`), uncommitted samples are discarded.

### `descriptor.yaml` fields

Header: scenario, teacher, record_hz, map_frame, base_frame, max_goals,
lom_width, lom_height, lom_resolution, lom_encoding, min_cmd_magnitude,
collision_topic, stuck_* thresholds, lookback frames.

Summary (appended at shutdown by the recorder, not by the NPZ converter):
num_samples, num_discarded_static, num_episodes, n_collision_frames,
n_collision_events, n_stuck_events, n_episodes_with_collision,
n_episodes_with_stuck.

### Teleop workflow

Buttons (default indices): `B` (1) toggles recording, `X` (2) emergency
teleport, `Y` (3) new pair. State machine:
`IDLE -> WAITING_FOR_START -> RECORDING`. The episode opens when the
robot reaches the start marker (`start_reached_radius_m`, default 0.5 m)
and closes on goal (`goal_reached_radius_m`, default 0.30 m), timeout
(`goal_timeout_s`, default 180 s), or idle (`idle_timeout_s`, 6 s).

### Teleop orphan cleanup warning

`teleop_dataset_recorder.py` kills any process matching `ORPHAN_PATTERNS`
on exit (webots, justina_controller, dataset_recorder, rviz2, nav2_amcl,
nav2_map_server, nav2_lifecycle_manager, joy_node, local_occ_grid_node,
depth_to_pointcloud, ...). Other instances on the same machine will be
terminated.

### `debugger` package required

`debugger` (Python: `debugger.utils.DebugClient`; C++:
`debugger/debug_client.hpp`) must be on the workspace. Missing package ->
`ModuleNotFoundError: No module named 'debugger'`.

### Zero-quota classes

Valid. `insufficient_class_paths` only triggers when a **positive** quota
cannot be satisfied.

### `auto_world_offset` YAML requirement

Both the sampler and the teleop recorder read `world_offset_x/y/yaw` from
the YAML next to the prohibition PNG. Missing YAML + `auto_world_offset=true`
(the launch default) causes a `RuntimeError` at startup:
`auto_world_offset is enabled but YAML not found`.

### Status `reason` values

`sampler_complete`, `sampler_incomplete`, `world_start_failed`,
`recording_failed_no_data`, `pool_incompatible_no_data`, `stalled`,
`missing_pool`, `empty_pool`, `incremental_skip`, `launch_exit`,
`interrupted`.
