# Dataset generation workflow

The route-pool engine is `path_generator.py`; `generate_path_pool.py` is the compatibility wrapper used by collection scripts. This generates map-level route pools only and does not launch ROS or Webots.

## Balanced generation

```bash
cd ~/GIT/mapless/src/Navigation/mapless_navigation/dataset/scripts
chmod +x path_generator.py generate_path_pool.py generate_dataset.sh
./generate_dataset.sh --n-goals 202 --scenarios "E01 E02 E03"
```

`--n-goals` is the total number of routes per scenario and is split as evenly as possible among `straight`, `semi_complex`, `complex`, and `v_complex`. Remainders are assigned in that order, so 202 means quotas 51, 51, 50, and 50. Without an explicit total or quotas, the default total is 200. `--balance-class` explicitly selects the balanced mode; it is otherwise the default.

## Explicit quotas

```bash
./generate_dataset.sh --st_sem_cmp_vc=10_10_15_20 --scenarios "E01 E02 E03"
```

This sets 10, 10, 15, and 20 routes per class (55 per scenario). Do not combine explicit quotas with `--n-goals` or `--balance-class`. One target map drives sampling, stopping conditions, progress, pool metadata, report, manifest, and validation. Legacy `--per-class N` remains supported in the Python CLIs.

## Optional limits

No attempt cap or wall-clock timeout is active by default. Search continues until all targets are met unless you specify a limit.

```bash
# Stop on either 900 seconds or 25000 attempts, whichever comes first.
./generate_dataset.sh --n-goals 202 --timeout-seconds 900 --max-attempts 25000
```

Either limit can be used alone. Validated partial paths remain available when a limit is reached; the report gives the actual stop reason and deficits by class. Routes are never duplicated to fill quotas.

Progress is printed every five seconds. The v-complex startup status is now `v_complex: búsqueda dirigida activada`; detailed diagnostics remain in `report.json`.

## Data safety

The launcher does not delete `dataset/data/` by default. It updates the route-pool/report outputs for selected scenarios while leaving unrelated files untouched. To deliberately delete the full directory, pass `--purge-data`; an interactive confirmation is required. In non-interactive use, `--yes` skips the prompt.

```bash
# Preserve existing unrelated files.
./generate_dataset.sh

# Explicit class quotas.
./generate_dataset.sh --st_sem_cmp_vc=10_10_15_20

# Deliberate full reset.
./generate_dataset.sh --purge-data --n-goals 202
```

The script checks maps and Python dependencies, validates the pool/poses JSON pair and class targets, and continues to later scenarios if one remains incomplete. The overall exit code is nonzero if any quota is missed.

## Generated files

Each scenario is stored under `data/<scenario>/generation_seed_<seed>/`: `path_pool.json`, `path_pool_poses.json`, `path_pool.csv`, `report.json`, `report.md`, `routes_<class>.png`, and `generation.log`. `data/generation_manifest.json` records targets, optional limits, class counts, deficits, and stop reasons.

Route pools are map-level candidates. Navigation episodes and sensor data must be collected separately via the appropriate ROS/Webots collection launch. Review route plots and narrow passages before downstream use.
