# Mapless navigation training experiments (ROS 2 / Webots)

This folder is the entry point for the frozen map-level experiments. The ROS 1
material under `Ros1Data/` is reference-only; the training runner reads the
current ROS 2 NPZ files under `dataset/data/E##/`.

## 1. Read-only dataset audit

From the workspace root:

```bash
cd ~/GIT/mapless
python3 src/Navigation/mapless_navigation/mapless_nav/scripts/TorchModels/audit_dataset.py \
  --data-root src/Navigation/mapless_navigation/dataset/data \
  --output-dir /tmp/mapless_audit_20261021
```

The audit reads `data.csv`, `episode_tries.json`, `descriptor.yaml`, and LOM
file references. It does not generate, delete, rename, convert, or rewrite
recordings. It reports counts of rows, episodes, classes, retry attempts,
collision/stuck flags, missing LOM references, descriptor disagreements, and
whether the companion converted NPZ exists. The output folder must be new or
empty. Results are written outside the dataset.

## 2. Run tests without training or ROS 2

```bash
python3 -m unittest discover \
  -s src/Navigation/mapless_navigation/mapless_nav/scripts/TorchModels/tests \
  -v
```

The split/config and audit tests use Python's standard library; the training
runner itself needs the project's installed PyTorch, NumPy, and Matplotlib
dependencies.

## 3. Fixed training splits

The authoritative configuration is
`experiment_splits_v1.json`. The module `experiment_config.py` rejects
unexpected changes to the exact scenario membership, requires all 30 scenarios
to be partitioned once, and protects E01-E03 in TRAIN and E04-E06 in TEST.

- Experiment A: TRAIN E01, E02, E03, E12, E17, E22, E27, E30; TEST is the
  remaining 22 scenarios.
- Experiment B: TRAIN E01, E02, E03, E07, E09, E11, E13, E14, E15, E17, E18,
  E21, E22, E25, E26, E28, E30; TEST is the remaining 13 scenarios.

The baseline uses the current `TorchModels/utils/models.py` RNN and the
current ROS 2 preprocessing. Initial parameters are based on the historical
ROS 1 training record (sequence length 2, hidden size 350, 4 recurrent layers,
Adam with LR 0.0006, batch size 3, maximum 20 epochs). This is a controlled
ROS 2 experiment informed by the historic baseline, not a claim of bit-exact
ROS 1 replay. Normalization remains the current TorchModels behavior:
`vx` is nonnegative when reverse is disabled, `vy` and `wz` are signed, and
the occupancy grid is scaled by 1/100 and clipped to [0, 1].

## 4. Prepare an isolated NPZ staging dataset

The recorded workspace contains CSV and LOM data but may not contain converted NPZ files. Do not call `generate_npz_data.py` on the source tree until its write behavior and existing artifacts have been checked: it writes `.npz` and `.descriptor.yaml` beside each source run and can overwrite them.

Use the staging converter to read existing CSV/LOM files and write converted NPZ files to a new directory outside the source dataset. It converts only the union of maps used for TRAIN in A or B; maps that are TEST-only for both experiments get empty directories and no NPZ files. For either run, the trainer loads NPZ *contents* only from that experiment’s TRAIN maps.

```bash
STAGE="/tmp/mapless-npz-stage-$(date +%Y%m%d_%H%M%S)"
python3 src/Navigation/mapless_navigation/mapless_nav/scripts/TorchModels/prepare_experiment_npz.py \\
  --data-root src/Navigation/mapless_navigation/dataset/data \\
  --output-root "$STAGE"
```

The script refuses to use an output directory that is inside, contains, or equals the source tree, or that already contains files. It writes `conversion_manifest.json` with source CSV/sidecar/descriptor hashes, converted NPZ hashes, sample counts and per-class episode counts. It does not regenerate paths, rebalance classes, change labels, or modify source recordings. Conversion fails if a sample would be dropped by the existing CSV/LOM decoder or if an observed episode has no class label.

Review the output summary and manifest. Expected staging coverage is 19 converted TRAIN-union scenarios and 11 TEST-only scenarios without NPZ files. Stop and investigate if counts differ from the source audit or if the command exits with an error. Keep the staging root path for the training commands below.

## 5. Run an experiment

Run only after conversion is complete and the manifest has been reviewed. The trainer intentionally fails rather than converting data or silently skipping a configured TRAIN scenario.

```bash
python3 src/Navigation/mapless_navigation/mapless_nav/scripts/TorchModels/train_experiment.py --experiment A --data-root "$STAGE"
python3 src/Navigation/mapless_navigation/mapless_nav/scripts/TorchModels/train_experiment.py --experiment B --data-root "$STAGE"
```

Both runs default to seed 42 and use a 25% validation split by complete
episodes *within the configured TRAIN scenarios*. Test scenario NPZ contents are never deserialized by this runner. The test maps remain reserved for the final Webots generalization evaluation. Temporal windows are constructed lazily from each episode using the same `build_input_frame()` preprocessing as `dataset_utils.py`, avoiding the memory cost of caching every overlapping float32 window.

Each run gets an independent output folder under
`TorchModels/experiments/<experiment-id>/seed_<seed>/`, containing:

- `experiment_manifest.json`: split, episode identities, source NPZ SHA-256
  fingerprints, class counts, parameters, and effective sequence counts.
- `history.json`: train/validation loss, normalized MAE/RMSE, velocity-unit
  MAE/RMSE, and results by episode class at each epoch.
- `best_model.pth`: best checkpoint selected solely by internal validation
  loss.
- `learning_curves.png` and `summary.json`.

A run refuses to use a non-empty output directory. To intentionally run a new
seed or configuration, use a different seed and/or `--output-dir`. No
TEST-map navigation evaluation is inferred from the training losses; Webots
evaluation must be recorded as a separate phase.

## 6. Destructive options

Do not use `--purge-data` or `--replace` for these experiments.
