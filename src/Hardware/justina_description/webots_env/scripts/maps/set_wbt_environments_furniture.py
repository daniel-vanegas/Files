#!/usr/bin/env python3
"""
Place furniture inside a selected environment's free space, using its map.
- Uses oriented bounding boxes (OBB) for accurate footprints.
- Clearance margin to prevent door blocking.
- Pixel-accurate free-space check.
- Optionally select specific furniture items (fourth arg).
- Generates a Webots world file based on SetEnvironment.wbt template.
- Furniture is selectable and movable in Webots (isPickable TRUE, no locked).
- Samples without replacement to minimize repeats.
Usage: ./set_wbt_environments_furniture.py [env_name] [num_items] [output_world_name] [furniture_list]
Example: ./set_wbt_environments_furniture.py C2_02_E1 6 scene.wbt "Table_02.dae,Desk_01.dae"
"""

import sys
import yaml
import random
import trimesh
import numpy as np
from PIL import Image
from pathlib import Path
from shapely.geometry import Polygon, Point, box
from shapely.affinity import translate, rotate
from scipy import ndimage

# Path detection
SCRIPT_DIR = Path(__file__).resolve().parent


def find_webots_env_root(start_dir):
    current = start_dir
    for _ in range(5):
        if (current / 'scripts').exists() and (current / 'environments').exists():
            return current
        if current.name == 'scripts':
            return current.parent
        current = current.parent
    if (start_dir / 'environments').exists():
        return start_dir
    raise FileNotFoundError("Could not locate webots_env root")


WEBOTS_ROOT = find_webots_env_root(SCRIPT_DIR)
ENV_MESH_BASE = WEBOTS_ROOT / 'environments' / 'mesh'
MAPS_BASE = WEBOTS_ROOT / 'environments' / 'maps'
MODELS_BASE = WEBOTS_ROOT / 'models' / 'BioRoboticsLab' / 'meshes'
WORLDS_BASE = WEBOTS_ROOT / 'worlds'
TEMPLATE_WORLD = WORLDS_BASE / 'SetEnvironment.wbt'

# Furniture list
EXCLUDED_FURNITURE = {
    'Bike.dae', 'Door.dae', 'Lab_Floor.dae', 'Lab_Mesh_B.dae', 'Lab_Mesh.dae',
}
ALL_MESHES = [f for f in MODELS_BASE.glob(
    "*.dae") if f.name not in EXCLUDED_FURNITURE and "_Collider" not in f.name]
FURNITURE_NAMES = [f.name for f in ALL_MESHES]

# Table/desk names (can be placed anywhere)
TABLE_KEYWORDS = ['Table_', 'Desk_']


def is_table(name):
    return any(kw in name for kw in TABLE_KEYWORDS)


def load_map(env_name):
    map_dir = MAPS_BASE / env_name
    if not map_dir.exists():
        raise FileNotFoundError(f"No map folder for {env_name} in {MAPS_BASE}")
    png_path = map_dir / f"map_{env_name}.png"
    yaml_path = map_dir / f"map_{env_name}.yaml"
    if not png_path.exists() or not yaml_path.exists():
        raise FileNotFoundError(f"Map files missing for {env_name}")
    img = Image.open(png_path)
    grid = np.array(img, dtype=np.uint8)
    with open(yaml_path) as f:
        meta = yaml.unsafe_load(f)
    resolution = meta['resolution']
    origin = meta['origin']
    if isinstance(origin, np.ndarray):
        origin = origin.tolist()
    return grid, resolution, origin


def get_free_space_mask(grid):
    mask = (grid == 255).astype(np.uint8)
    rows, cols = np.where(mask == 1)
    free_pixels = list(zip(rows, cols))
    return mask, free_pixels


def distance_from_walls(mask):
    walls = (1 - mask).astype(np.uint8)
    return ndimage.distance_transform_edt(walls)


def pixel_to_world(pixel, origin, resolution, image_shape):
    row, col = pixel
    x = origin[0] + col * resolution
    y = origin[1] + (image_shape[0] - 1 - row) * resolution
    return x, y


def world_to_pixel(x, y, origin, resolution, image_shape):
    col = int(round((x - origin[0]) / resolution))
    row = int(round((image_shape[0] - 1 - (y - origin[1]) / resolution)))
    return row, col


def get_furniture_obb(dae_path):
    """Return a Shapely Polygon of the oriented bounding box (XY projection)."""
    mesh = trimesh.load(dae_path)
    if mesh is None:
        raise ValueError(f"Could not load {dae_path}")
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g,
                                                                  trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No Trimesh in scene")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Not a Trimesh")
    obb = mesh.bounding_box_oriented
    verts = obb.vertices[:, :2]
    from shapely.geometry import MultiPoint
    poly = MultiPoint(verts).convex_hull
    if poly.geom_type != 'Polygon':
        raise ValueError("OBB did not produce a polygon")
    return poly


def points_in_polygon(polygon, step, origin, resolution, image_shape):
    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if polygon.contains(Point(x, y)):
                row, col = world_to_pixel(
                    x, y, origin, resolution, image_shape)
                if 0 <= row < image_shape[0] and 0 <= col < image_shape[1]:
                    pts.append((row, col))
            y += step
        x += step
    return pts


def place_furniture(mask, free_pixels, dist_transform, furniture_list, num_items, origin, resolution,
                    wall_bias_max_dist_m=0.5, clearance_m=0.3, step=0.02, max_attempts=2000):
    free_set = set(free_pixels)
    random.shuffle(furniture_list)

    furniture_data = {}
    for name in furniture_list:
        try:
            obb_poly = get_furniture_obb(MODELS_BASE / name)
            minx, miny, maxx, maxy = obb_poly.bounds
            w = maxx - minx
            d = maxy - miny
            furniture_data[name] = (obb_poly, w, d)
        except Exception as e:
            print(f"  Skipping {name} (OBB error: {e})")

    if not furniture_data:
        print("No valid furniture data.")
        return []

    placed_geoms = []
    placed_items = []
    usage_count = {name: 0 for name in furniture_data.keys()}

    attempts = 0
    max_total_attempts = num_items * max_attempts * 2
    consecutive_failures = 0
    max_consecutive_failures = 300
    pool_round = 1

    # Initial pool: all items with usage < 2
    available_names = [name for name in furniture_data.keys()
                       if usage_count[name] < 2]
    random.shuffle(available_names)

    while len(placed_items) < num_items and free_set and attempts < max_total_attempts:
        attempts += 1

        # If we've tried all items in the current pool without enough placements,
        # rebuild the pool from items with usage < 2 that are not yet placed
        if not available_names:
            # Rebuild from items with usage < 2 and not yet placed in this round
            placed_names = [item[0] for item in placed_items]
            available_names = [name for name in furniture_data.keys(
            ) if usage_count[name] < 2 and name not in placed_names]
            if not available_names:
                print("  No more distinct furniture types available (each used twice).")
                break
            random.shuffle(available_names)
            pool_round += 1
            print(f"  Reshuffling pool (round {pool_round}, allow 2nd copy)")

        # Take the next item from the pool (without removing it permanently)
        # take first, will be reinserted if fails
        mesh_name = available_names.pop(0)

        obb_poly, w, d = furniture_data[mesh_name]

        if not free_set:
            break

        center_pixel = random.choice(list(free_set))
        row, col = center_pixel
        x, y = pixel_to_world((row, col), origin, resolution, mask.shape)

        yaw = random.choice([0, np.pi/2, np.pi, 3*np.pi/2])

        centroid = obb_poly.centroid
        poly_rot = rotate(obb_poly, np.degrees(
            yaw), origin=(centroid.x, centroid.y))
        poly_final = translate(
            poly_rot, xoff=x - centroid.x, yoff=y - centroid.y)

        clearance_poly = poly_final.buffer(
            clearance_m, cap_style=3, join_style=2)
        pts = points_in_polygon(clearance_poly, step,
                                origin, resolution, mask.shape)
        if not pts:
            # Failed: reinsert at a random position
            available_names.insert(random.randint(
                0, len(available_names)), mesh_name)
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                print("  Too many consecutive failures – no more space likely.")
                break
            continue

        if not all(p in free_set for p in pts):
            available_names.insert(random.randint(
                0, len(available_names)), mesh_name)
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                print("  Too many consecutive failures – no more space likely.")
                break
            continue

        dist = dist_transform[row, col] * resolution
        if not is_table(mesh_name) and dist > wall_bias_max_dist_m:
            available_names.insert(random.randint(
                0, len(available_names)), mesh_name)
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                print("  Too many consecutive failures – no more space likely.")
                break
            continue

        overlap = False
        for (other_poly, ox, oy, oyaw) in placed_geoms:
            if poly_final.intersects(other_poly):
                overlap = True
                break
        if overlap:
            available_names.insert(random.randint(
                0, len(available_names)), mesh_name)
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                print("  Too many consecutive failures – no more space likely.")
                break
            continue

        # Place the item
        placed_geoms.append((poly_final, x, y, yaw))
        placed_items.append((mesh_name, x, y, yaw))
        usage_count[mesh_name] += 1
        free_set.difference_update(pts)
        consecutive_failures = 0
        # Item is already removed from available_names (we popped it), so we don't reinsert

        # Print progress
        print(
            f"  Placed {len(placed_items)}/{num_items} {mesh_name} (copy #{usage_count[mesh_name]}) at ({x:.2f}, {y:.2f})")

    if len(placed_items) < num_items:
        print(
            f"  Warning: only placed {len(placed_items)} out of {num_items} items.")

    return placed_items


def generate_world(env_name, placed_items, output_filename):
    if not TEMPLATE_WORLD.exists():
        print(f"Template world not found: {TEMPLATE_WORLD}")
        return
    with open(TEMPLATE_WORLD, 'r') as f:
        lines = f.readlines()

    start = None
    brace_count = 0
    for i, line in enumerate(lines):
        if "DEF BioRoboticsLab_Mesh Group" in line:
            start = i
            break
    if start is None:
        print("Could not find DEF BioRoboticsLab_Mesh Group in template.")
        return
    for i in range(start, len(lines)):
        brace_count += lines[i].count('{') - lines[i].count('}')
        if brace_count == 0:
            end = i
            break
    else:
        print("Could not find matching closing brace.")
        return

    children = []
    # Ground (locked)
    children.append("""    Solid {
      translation 0 0 -0.01
      children [
        Shape {
          appearance PBRAppearance {
            baseColor 0.752941 0.74902 0.737255
            roughness 1
            metalness 0
          }
          geometry Plane {
            size 50 50
          }
          isPickable FALSE
        }
      ]
      name "ground"
      boundingObject Plane {
        size 50 50
      }
      locked TRUE
    }""")
    # lab_floor removed
    # lab_walls (locked)
    env_mesh_path = f"../environments/mesh/{env_name}/{env_name}.dae"
    children.append(f"""    Solid {{
      children [
        CadShape {{
          url [
            "{env_mesh_path}"
          ]
          isPickable FALSE
        }}
      ]
      name "lab_walls"
      boundingObject Mesh {{
        url [
          "{env_mesh_path}"
        ]
      }}
      locked TRUE
    }}""")

    # Furniture: selectable and movable (isPickable TRUE, no locked)
    for (mesh_name, x, y, yaw) in placed_items:
        mesh_path = f"../models/BioRoboticsLab/meshes/{mesh_name}"
        collider_path = MODELS_BASE / \
            mesh_name.replace('.dae', '_Collider.dae')
        collider_url = f"../models/BioRoboticsLab/meshes/{collider_path.name}" if collider_path.exists(
        ) else mesh_path
        solid = f"""    Solid {{
            translation {x:.4f} {y:.4f} 0
            rotation 0 0 1 {yaw:.4f}
            name "{mesh_name.replace('.dae', '')}"
            children [
                CadShape {{
                url [
                    "{mesh_path}"
                ]
                isPickable TRUE
                }}
            ]
            boundingObject Mesh {{
                url [
                "{collider_url}"
                ]
            }}
            rotationStep 0.001
            }}"""
        children.append(solid)

    group_content = "DEF BioRoboticsLab_Mesh Group {\n  children [\n" + ",\n".join(
        children) + "\n  ]\n}"
    new_lines = lines[:start] + [group_content + "\n"] + lines[end+1:]

    output_path = WORLDS_BASE / output_filename
    with open(output_path, 'w') as f:
        f.writelines(new_lines)
    print(f"World file generated: {output_path}")


def main():
    env_name = None
    num_items = 5
    output_name = None
    furniture_selection = None

    if len(sys.argv) > 1:
        env_name = sys.argv[1]
        if len(sys.argv) > 2:
            num_items = int(sys.argv[2])
        if len(sys.argv) > 3:
            output_name = sys.argv[3]
        if len(sys.argv) > 4:
            furniture_selection = sys.argv[4]
    else:
        env_name = input("Enter environment name (e.g., E5_C2_01): ").strip()
        num_items = int(input("Number of furniture items to place: ") or "5")
        output_name = input(
            "Output world filename (e.g., scene_E5_C2_01.wbt): ").strip()
        if not output_name:
            output_name = f"scene_{env_name}.wbt"
        sel = input(
            "Optional: comma-separated list of furniture names to use (leave blank for all): ").strip()
        if sel:
            furniture_selection = sel

    if not output_name.endswith('.wbt'):
        output_name += '.wbt'

    print(f"Environment: {env_name}")
    print(f"Number of items: {num_items}")
    print(f"Output world: {output_name}")

    try:
        grid, resolution, origin = load_map(env_name)
        print(
            f"Loaded map for {env_name} (resolution {resolution}, origin {origin})")
    except Exception as e:
        print(f"Error loading map: {e}")
        sys.exit(1)

    mask, free_pixels = get_free_space_mask(grid)
    print(f"Free space pixels: {len(free_pixels)}")
    if len(free_pixels) == 0:
        print("No free space pixels found.")
        sys.exit(1)

    try:
        dist_transform = distance_from_walls(mask)
    except Exception as e:
        print(f"Error computing distance transform: {e}. Fallback to no bias.")
        dist_transform = np.zeros(mask.shape, dtype=np.float32)

    furniture_list = FURNITURE_NAMES
    if furniture_selection:
        selected = [name.strip()
                    for name in furniture_selection.split(',') if name.strip()]
        filtered = [name for name in furniture_list if name in selected]
        if filtered:
            furniture_list = filtered
            print(f"Using selected furniture: {', '.join(furniture_list)}")
        else:
            print(f"Warning: none of the selected items found. Using full list.")

    print(f"Available furniture: {len(furniture_list)} items")

    placed = place_furniture(
        mask, free_pixels, dist_transform, furniture_list, num_items, origin, resolution)
    print(f"Placed {len(placed)} items.")

    if not placed:
        print("No items placed.")
        sys.exit(0)

    generate_world(env_name, placed, output_name)


if __name__ == "__main__":
    main()
