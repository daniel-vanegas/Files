# Webots Integration for Mapless Navigation

This package provides the simulation environment for the Justina robot using Webots. It includes tools to generate occupancy grid maps from CAD meshes, compute structural similarity between environments, and launch randomised test scenarios.

Lidar robot is located at 0.235 cm in Z

## Table of Contents

- [Creating a New Environment](#creating-a-new-environment)
- [Launching the Simulation](#launching-the-simulation)
- [Map Generation Pipeline](#map-generation-pipeline)
  - [1. Generating Occupancy Grids (maps)](#1-generating-occupancy-grids)
  - [2. Computing Structural Similarity](#2-computing-structural-similarity)
---

## Creating a New Environment

To add a new randomized environment for the robot, follow these steps:

1. Create the mesh folder

    Inside `webots_env/environments/mesh/`, create a new folder named after your environment (e.g., E29_Custom).

    Place two DAE files:

    `E29_Custom.dae` – the full 3D scene (walls, furniture).

    `ground_E29_Custom.dae` – a planar mesh representing the navigable floor (used for map generation).

2. Generate the occupancy map

    Run `generate_maps.py` to create the map:
    ```bash
    ./generate_maps.py E29_Custom
    ```

    The map will be saved in `environments/maps/`E29_Custom/.

3. Set up furniture and objects (optional)

    The script `set_wbt_environments_furniture.py` can place objects inside the world based on a template (SetEnvironment.wbt). Edit its parameters to specify the number and types of objects.
    ```bash
    ./set_wbt_environments_furniture.py --env E29_Custom --objects 5
    ```

    This generates a new `.wbt` world file in `worlds/`.

###  Edit environment in Webots GUI

Open the generated world in Webots. `Pause the simulation` and `reset time`, and manually adjust object positions or add/remove items.

Save the world file back to (`source` directory) as `src/Hardware/justina_description/webots_env/worlds/`

- Update the ROS package
    ```bash
    colcon build --packages-select justina_description
    ```

Launch the new world
```bash
roslaunch justina_description justina_mapless_wbt.launch.xml world:=E29_Custom.wbt

## Launching the Simulation

To start the simulation with the Justina robot in a specific world, use the provided launch files in the `webots_env` folder, example:
```bash
roslaunch justina_description justina_mapless_wbt.launch.xml
```

By default, this loads the world defined in the launch file. You can change the world by modifying the world argument:
```bash
roslaunch justina_description justina_mapless_wbt.launch.xml world:=E01.wbt
```
`Note`: The launch file loads worlds from the installed package directory. `After editing` a world file in the source directory, you must `reinstall the package` (colcon build) for the changes to take effect.

## Map Generation Pipeline

The environment maps are generated from DAE meshes (exported from Blender). The pipeline consists of three steps:
### 1. Generating Occupancy Grids (maps)

- The script `generate_maps.py` converts the `ground_*.dae` mesh (the navigable floor polygon) into a ROS‑compatible occupancy grid map (PNG + YAML).

- Location: `webots_env/scripts/maps/generate_maps.py`
- Usage:
    ```bash
    ./generate_maps.py [env_name] [resolution] [inner_thickness] [outer_thickness] [--align true|false]
    ```
    | Parameter | Default | Description |
    |---|---|---|
    | `env_name` | (all) | Process a specific environment folder; if omitted, all are processed. |
    | `resolution` | `0.01 m` | Map resolution in meters per pixel. |
    | `inner_thickness` | `0.05 m` | Thickness of walls eroded from free space (ensures robot clearance). |
    | `outer_thickness` | `0.10 m` | Additional wall thickness added as occupied cells. |
    | `--align` | `true` | If true, maps are centered by the centroid of walls and share a common origin (recommended for similarity analysis). If false, default origins are preserved. |

    Example:
    ```bash
    # Generate all maps with alignment (default)
    ./generate_maps.py

    # Generate only E1 with custom resolution and no alignment
    ./generate_maps.py E1_C2_02 0.02 --align false
    ```
    Output: Maps are saved in `webots_env/environments/maps/<env_name>/` as `map_<env_name>.png` and `map_<env_name>.yaml`.

    ### Aligning Maps for Comparison

    [Do step 1](#1-generating-occupancy-grids) with `--align true` to ensure all maps share the same coordinate frame. This is crucial for meaningful structural similarity comparisons.

    This ensures that all maps share the same coordinate frame, so that distances computed later reflect structural differences only, not absolute positions.

### 2. Computing Structural Similarity

The script `compute_similarity.py` computes the Wasserstein‑1 distance (Earth Mover's Distance) between all pairs of environments using their occupancy maps. This distance quantifies how similar the obstacle distributions are.

- Location: `webots_env/scripts/maps/compute_similarity.py`

- Global Configuration `(edit at the top of the script)`:
    ```bash
    METHOD = 'sinkhorn'          # 'sliced', 'sinkhorn', or 'emd'
    MAX_POINTS = 4000            # Points subsampled per map (higher = more accurate)
    ALIGNED = True               # Must match the alignment used when generating maps
    RESOLUTION = 0.01            # Map resolution (m/pixel)
    REG = 0.001                  # Entropy regularisation (sinkhorn)
    SINKHORN_ITERS = 200         # Max Sinkhorn iterations
    N_PROJECTIONS = 200          # Number of projections (sliced method)
    ```

- Command-line overrides are also available:
    ```bash
    ./compute_similarity.py --method emd --max_points 800
    ```

    `Output`: A CSV file `wasserstein_distances_<method>.csv` containing the pairwise distance matrix. The first row and column list environment names.

    `Interpretation`: `Smaller distances` indicate more similar environments. You can correlate these distances with experimental success rates to predict generalization performance.

