# Map generation scripts usage

This folder contains the tools used to generate, transform, and compare occupancy grid maps for the Justina robot inside Webots.

These scripts form the core of the map generation pipeline, converting CAD meshes (DAE) into ROS-compatible maps (`PNG` + `YAML`), handling furniture placement, creating new maps, and computing structural similarities between environments.

---

## Table of Contents

- [Overview](#overview)
- [Scripts](#scripts)
  - [1. `map_generator.py`](#1-map_generatorpy)
  - [2. `set_wbt_environments_furniture.py`](#2-set_wbt_environments_furniturepy)
  - [3. `compute_similarity.py`](#3-compute_similaritypy)
  - [4. `wbt_to_dae.py` (in `misc/`)](#4-wbt_to_daepy-in-misc)
- [Map Generation Pipeline](#map-generation-pipeline)
---

## Overview

The robot's LiDAR is mounted at **0.2387 meters** above the floor. This height directly affects what the sensor can "see". The scripts here allow you to generate:

1. **Base maps** – The raw free-space grid from the ground mesh (plus wall thickness).
2. **Prohibition maps** – Maps that mark the *full volume* of furniture, even if the sensor can't see it. Used to prevent the robot from physically driving through objects.
3. **Detection maps** – Maps that replicate what the LiDAR *actually detects* at its specific height (e.g., table legs but not table tops).

---

## Scripts

### 1. `map_generator.py`

**Purpose:**
Converts the environment DAE meshes into ROS-compatible occupancy grid maps (`PNG` + `YAML`).

The script uses three types of DAE files:

- `ground_<env_name>.dae` — defines the navigable/free-space floor polygon.
- `<env_name>.dae` — defines the physical walls. All triangle faces from all meshes are projected vertically onto the XY plane and added as black occupied areas.
- `obstacles_<env_name>.dae` — optional navigation mesh. Its triangles represent free navigation space; areas inside the ground but not represented by these triangles become obstacles.

The resulting regular map preserves the floor geometry, projects the real wall geometry over it, and keeps areas outside the floor as gray/unknown.

`Location`: scripts/maps/generate_maps.py

`Usage`:
```bash
./generate_maps.py [env_name] [resolution] [--feature-grow METERS] [--outer-wall-thickness METERS] [--margin METERS] [--align true|false]
```

| Parameter | Default | Description |
 |---|---|---|
 | `env_name` | (all) | Process a specific environment folder; if omitted, all environments containing `ground_*.dae` are processed. |
 | `resolution` | `0.01 m` | Map resolution in meters per pixel. |
 | `inner_thickness` | `0.05 m` | hickness used when processing the inner side of the walls. |
 | `outer_thickness` | `0.10 m` | dditional wall thickness used when generating the wall ring. |
 | `--align` | `true` | If true, maps are centered using the projected wall centroid and placed on a common canvas size. Recommended for similarity analysis. If false, each map keeps its own dimensions and origin. |
 | `--world-file` | `automatic` | Webots .wbt file used to generate the prohibition map. If omitted, the script attempts to find the appropriate world file automatically. |

### Input files

For an environment such as E01_C2_02, the directory should contain:
```bash
environments/mesh/E01_C2_02/
├── ground_E01_C2_02.dae
├── E01_C2_02.dae
└── obstacles_E01_C2_02.dae
```
The files have the following roles:

- `ground_<env_name>.dae` — Defines the floor/free-space geometry.
- `<env_name>.dae` — Defines the physical walls.
- `obstacles_<env_name>.dae` — Optional navigation mesh. Areas represented by this mesh are considered navigable.
- `<object>_BB.dae` — Bounding-box meshes used by the prohibition map. These are loaded from models/BioRoboticsLab/meshes_bounding/.
- `<scene>.wbt` — Webots scene containing the furniture names, translations, and rotations.


### Map generation

The script first generates an environment-specific map from the ground and wall meshes.

If an `obstacles_<env_name>.dae` file is available, a navigation map is generated as well.

When `--align true` is enabled, all processed environments are placed on a common canvas:

1. Each environment is rasterized independently.
2. The largest required canvas dimensions are calculated.
3. The centroid of the wall pixels is calculated for each environment.
4. Each environment is shifted so that its wall centroid matches the center of the common canvas.
5. Unused areas of the common canvas are filled with gray (128 / unknown).

This keeps the standard and navigation maps from different environments in the same coordinate frame, which is required for meaningful similarity comparisons.

### Prohibition map

When a Webots world file is available, `map_generator.py` also generates a prohibition map.

The script:

1. Parses the furniture Solid nodes from the `.wbt` file.
2. Extracts each object's translation, rotation, and name.
3. Finds the corresponding `_BB.dae` bounding-box mesh.
4. Applies the Webots axis-angle rotation around the mesh local origin.
5. Applies the Webots translation.
6. Projects the transformed bounding box onto the map's XY plane.
7. Converts the resulting world coordinates into the same aligned map coordinate system.
8. Marks the furniture footprint as occupied.

The prohibition map therefore uses the same global canvas, resolution, origin, and alignment as the standard and navigation maps.

This is important because furniture coordinates come directly from the Webots scene, while the map coordinates originate from the DAE environment meshes. The script keeps both representations in the same coordinate frame rather than applying an independent alignment to the prohibition map.

**Output**

The standard map is always generated:
```bash
environments/maps/<env_name>/
├── map_<env_name>.png
└── map_<env_name>.yaml
```

If obstacles_<env_name>.dae exists, the navigation map is also generated:
```bash
environments/maps/<env_name>/
├── map_<env_name>.png
├── map_<env_name>.yaml
├── map_<env_name>_nav.png
└── map_<env_name>_nav.yaml
```
If a valid Webots world file and matching BB meshes are available, the prohibition map is generated:
```bash
environments/maps/<env_name>/
├── map_<env_name>.png
├── map_<env_name>.yaml
├── map_<env_name>_nav.png
├── map_<env_name>_nav.yaml
├── map_<env_name>_prohibition.png
└── map_<env_name>_prohibition.yaml
```

**Examples**

Generate all environments with the default configuration:
```bash
./map_generator.py
```
Generate one environment:
```bash
./map_generator.py E01_C2_02
```
Generate one environment at 2 cm resolution:
```bash
./map_generator.py E01_C2_02 0.02
```
Generate an environment using a specific Webots scene:
```bash
./map_generator.py E01_C2_02 --world-file E01.wbt
```
Generate without centroid alignment:
```bash
./map_generator.py E01_C2_02 --align false
```
Generate with custom wall thickness:
```bash
./map_generator.py E01_C2_02 0.01 0.05 0.10 --align true
```
**Note**: `--align true` is recommended when the generated maps will be compared using compute_similarity.py. The standard, navigation, and prohibition maps generated for the same environment share the same canvas and coordinate system.


### 2. set_wbt_environments_furniture.py

- `Purpose`:

    Places furniture inside an environment's free space, based on the previously generated map. It uses oriented bounding boxes (OBB) to ensure objects fit, avoids placing objects near walls, and generates a complete Webots world file.

- `Location`: scripts/maps/set_wbt_environments_furniture.py

- `Usage`:
    ```bash
    ./set_wbt_environments_furniture.py <env_name> <num_items> <output_world_name> [furniture_list]
    ```

- `Example`:
    ```bash
    ./set_wbt_environments_furniture.py E01_C2_02 6 scene.wbt "Table_02.dae,Desk_01.dae"
    ```
- `Output`:

    A .wbt world file in worlds/ with placed furniture.


### 3. compute_similarity.py

`Purpose`:
Computes the Wasserstein‑1 distance (Earth Mover's Distance) between all pairs of environments using their occupancy maps. This quantifies how similar the obstacle distributions are, which can be used to predict generalization performance.

`Location`: scripts/maps/compute_similarity.py

`Usage`:
```bash
./compute_similarity.py [--method emd|sinkhorn|sliced] [--max_points 800]
```

`Global Configuration (edit at the top of the script)`:
```bash
METHOD = 'sinkhorn'          # 'sliced', 'sinkhorn', or 'emd'
MAX_POINTS = 4000            # Points subsampled per map
ALIGNED = True               # Must match alignment used in generate_maps.py
RESOLUTION = 0.01            # Map resolution (m/pixel)
REG = 0.001                  # Entropy regularisation (sinkhorn)
SINKHORN_ITERS = 200         # Max Sinkhorn iterations
N_PROJECTIONS = 200          # Number of projections (sliced method)
```

`Output`:
A CSV file `wasserstein_distances_<method>.csv` containing the pairwise distance matrix.

- `Interpretation`: Smaller distances indicate more similar environments.

### 4. wbt_to_dae.py (in misc/)

`Purpose`:
Exports a whole Webots scene (walls, floor, furniture) into a single .dae file with all transforms applied. This is useful for complex boolean operations or 2D projections in Blender.

`Location`: webots_env/misc/wbt_to_dae.py

`Usage`:
```bash
./wbt_to_dae.py <world_file> [output_filename]
```

`Example`:
```bash
./wbt_to_dae.py E01.wbt E01_full_scene.dae
```
---
## Map Generation Pipeline

The typical pipeline for a new environment is:

1. Create the environment mesh
    Place `ground_<env>.dae` and `<env>.dae` in `environments/mesh/<env>/`.

2. Generate the base map
    ```bash
    ./generate_maps.py <env> --align true
    ```

3. Place furniture in the world (optional)
    ```bash
    ./set_wbt_environments_furniture.py <env> 5 scene.wbt
    ```

4. Generate the Prohibition map (using the BB meshes)
    ```bash
    ./generate_prohibition_maps.py <env> scene.wbt _prohibition
    ```

 5. Generate the Detection map (optional, for realistic sensor sim)
    ```bash
    ./generate_occupancy_maps.py <env> scene.wbt
    ```

6. Compute similarity between environments
    ```bash
    ./compute_similarity.py --method sinkhorn
    ```

