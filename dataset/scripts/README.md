# Dataset production workflow

This is the production path-generation workflow for `mapless_navigation/dataset`.
There is one route-generation engine: `path_generator.py`. `generate_path_pool.py`
is only a compatibility command-line wrapper for the existing collection scripts.
Experimental candidate/audit/smoothing scripts are not part of production.

## Default production run

From the repository at `~/GIT/mapless`, replace these files under
`src/Navigation/mapless_navigation/dataset/scripts/` with the files in this package:

- `path_generator.py`
- `generate_path_pool.py`
- `run_production_dataset.sh`
- `README.md`

Then run:

```bash
cd ~/GIT/mapless/src/Navigation/mapless_navigation/dataset/scripts
chmod +x path_generator.py generate_path_pool.py run_production_dataset.sh
./run_production_dataset.sh --purge-data --prune-experiments
```

The command asks you to type `PURGE`. Do not use `--yes` unless you have reviewed
and intentionally accepted the deletion.

## What the production run does

- **Purges all of `dataset/data/`** before generating anything. This permanently
  removes prior route pools, experiment outputs, batch logs, and recorded navigation
  episodes stored there. No backup is made by this script.
- By default generates E01, E02 and E03; set `SCENARIOS="E01 E02 E03"` to select
  them explicitly, or change the variable to another whitespace-separated scenario list.
- Targets 200 valid routes per scenario: 50 each for `straight`, `semi_complex`,
  `complex` and `v_complex`.
- Stops searching each scenario after 900 seconds by default, or 25,000 attempts,
  whichever comes first. A timeout keeps valid partial paths and reports the shortfall.
- Updates progress every 5 seconds using a single terminal line. Counts use full class names
  (`straight`, `semi_complex`, `complex`, `v_complex`) rather than S/M/C/V abbreviations.
  Narrow terminals prioritize elapsed time, attempts, accepted count, and class counts to
  avoid wrapping the progress line.
- Validates the pool/poses JSON pair and the class counts. It continues to the next
  scenario if one scenario is incomplete; final exit status is nonzero if any class quota
  was not reached.
- Does not launch ROS, Webots, or the navigation simulator.

Example with custom settings:

```bash
PER_CLASS=35 TIMEOUT_SECONDS=900 MAX_ATTEMPTS=25000 \
PROGRESS_INTERVAL_SECONDS=5 SCENARIOS="E01 E02 E03" \
./run_production_dataset.sh --purge-data --prune-experiments
```

`PER_CLASS` determines the default class quota; `NUM_PATHS` can override the total directly
and is rounded up to equal per-class quotas. Prefer leaving `NUM_PATHS` unset and setting
`PER_CLASS` instead.

## Production files

Each scenario is stored directly in `data/<scenario>/`:

- `path_pool.json` — version-5 pool metadata, class labels and route metrics.
- `path_pool_poses.json` — route points keyed by ID.
- `path_pool.csv` — route metric table.
- `report.json` and `report.md` — generation summary and rejection diagnostics.
- `routes_<class>.png` — individual route panels for visual review.
- `generation.log` — generation output for the scenario.

`data/production_manifest.json` records the target and actual counts for all scenarios.

## Partial results and training balance

The generator saves validated partial routes at timeout, including when `v_complex` is
slower to find. Do not silently duplicate routes to fill a class. The training pipeline
should sample or weight examples by class and record the effective per-class counts.
If a class has zero examples, that scenario is incomplete for four-class training and
needs a longer generation run or a targeted fix before training.

## Script cleanup

`--prune-experiments` removes only the obsolete local experiment scripts:

- `scripts/path_generator copy.py`
- `scripts/run_candidate_batch_200.sh`
- `scripts/run_candidate_batch_timeout.sh` (if present)
- `scripts/path_smoothing/`

The old `generate_path_pool.py` body is replaced by the compatibility wrapper in this
package because both collection launchers call it. The launchers and ROS data-collection
scripts remain. Collection defaults for `POOL_NUM_PATHS=3000` are changed to 200 when
those settings are present.

The separate `dataset/archive/` directory is intentionally not touched.

## Recorded episodes are separate

Route pools are map-level candidates. Deleting `data/` also deletes already recorded
`data.csv`/sensor episodes and batch logs. This workflow does not recreate sensor or
navigation recordings; those must be collected later through the selected ROS/Webots
recording launch after pool generation and route review.
