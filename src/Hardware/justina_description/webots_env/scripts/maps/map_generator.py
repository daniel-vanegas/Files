#!/usr/bin/env python3
"""
Generate occupancy maps from DAE meshes and Webots scenes.

Generated maps:
1. Standard map: ground + walls.
2. Prohibition map: standard map + furniture bounding boxes (BB, convex hull).
3. Navigation map: standard map + furniture footprints (FP, exact 2D shape).

Both furniture maps read positions from the Webots world file. The prohibition
map uses meshes_bounding and takes the convex hull of each mesh. The navigation
map uses meshes_footprint and preserves every projected piece separately
(for example, the four legs of a table).

Usage:
    ./map_generator.py [env_name] [resolution] [inner_thickness] [outer_thickness]
                       [--align true|false] [--world-file <path>]
                       [--obstacle-inflation <meters>]
"""

import sys
import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SCRIPT_DIR = Path(__file__).resolve().parent


def find_webots_env_root(start_dir):
    current = start_dir
    for _ in range(5):
        if (current / "scripts").exists() and (current / "environments").exists():
            return current
        if current.name == "scripts":
            return current.parent
        current = current.parent

    if (start_dir / "environments").exists():
        return start_dir

    raise FileNotFoundError("Could not locate webots_env root")


WEBOTS_ROOT = find_webots_env_root(SCRIPT_DIR)
ENV_BASE = WEBOTS_ROOT / "environments" / "mesh"
MAPS_BASE = WEBOTS_ROOT / "environments" / "maps"
BB_MODELS_BASE = WEBOTS_ROOT / "models" / "BioRoboticsLab" / "meshes_bounding"
FP_MODELS_BASE = WEBOTS_ROOT / "models" / "BioRoboticsLab" / "meshes_footprint"
WORLDS_BASE = WEBOTS_ROOT / "worlds"

MAPS_BASE.mkdir(exist_ok=True)

if not ENV_BASE.exists():
    raise FileNotFoundError(f"Mesh directory not found: {ENV_BASE}")

print(f"Webots root:      {WEBOTS_ROOT}")
print(f"BB models folder: {BB_MODELS_BASE}")
print(f"FP models folder: {FP_MODELS_BASE}")


def find_environment_files(env_folder, env_name):
    """Find the ground and wall DAE files."""
    ground_file = env_folder / f"ground_{env_name}.dae"
    if not ground_file.exists():
        candidates = sorted(env_folder.glob("ground_*.dae"))
        ground_file = candidates[0] if candidates else None

    wall_file = env_folder / f"{env_name}.dae"
    if not wall_file.exists():
        wall_file = None

    return ground_file, wall_file


def load_dae(path):
    """Load a DAE file as one Trimesh object."""
    if path is None:
        return None

    print(f"    Loading: {path.name}")
    loaded = trimesh.load(path)

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
        print(f"      vertices: {len(mesh.vertices)}")
        print(f"      faces:    {len(mesh.faces)}")
        return mesh

    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            try:
                transform, geometry_name = loaded.graph[node_name]
            except Exception as error:
                print(f"      WARNING: cannot read node {node_name}: {error}")
                continue

            geometry = loaded.geometry.get(geometry_name)
            if not isinstance(geometry, trimesh.Trimesh):
                continue

            mesh = geometry.copy()
            try:
                mesh.apply_transform(transform)
            except Exception as error:
                print(
                    f"      WARNING: transform failed for {node_name}: {error}")
                continue

            meshes.append(mesh)

        if not meshes:
            for geometry in loaded.geometry.values():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry.copy())

        if not meshes:
            raise ValueError(f"No Trimesh geometry found in {path}")

        mesh = trimesh.util.concatenate(meshes)
        print(f"      components: {len(meshes)}")
        print(f"      vertices:   {len(mesh.vertices)}")
        print(f"      faces:      {len(mesh.faces)}")
        return mesh

    raise TypeError(f"Unsupported DAE type: {type(loaded)}")


def compute_bounds(ground_mesh, margin):
    """Compute the map bounds from the ground mesh."""
    vertices = np.asarray(ground_mesh.vertices)
    xy = vertices[:, :2]

    minx = float(np.min(xy[:, 0]))
    miny = float(np.min(xy[:, 1]))
    maxx = float(np.max(xy[:, 0]))
    maxy = float(np.max(xy[:, 1]))

    minx -= margin
    miny -= margin
    maxx += margin
    maxy += margin

    return minx, miny, maxx, maxy


def bounds_to_shape(bounds, resolution):
    """Convert world bounds into image height and width."""
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    return height, width


def world_to_source_pixel(x, y, bounds, resolution):
    """Convert world coordinates to the original, unaligned DAE raster."""
    minx, miny, maxx, maxy = bounds
    col = (x - minx) / resolution
    row = (maxy - y) / resolution
    return col, row


def rasterize_mesh(mesh, resolution, bounds, fill_value=255):
    """Project all mesh triangles onto the XY plane. Z is ignored."""
    height, width = bounds_to_shape(bounds, resolution)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    projected_faces = 0
    skipped_faces = 0

    for face in faces:
        if len(face) != 3:
            skipped_faces += 1
            continue

        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])

        if (i0 < 0 or i1 < 0 or i2 < 0 or
                i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices)):
            skipped_faces += 1
            continue

        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]

        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            skipped_faces += 1
            continue

        xy0 = world_to_source_pixel(p0[0], p0[1], bounds, resolution)
        xy1 = world_to_source_pixel(p1[0], p1[1], bounds, resolution)
        xy2 = world_to_source_pixel(p2[0], p2[1], bounds, resolution)

        draw.polygon([xy0, xy1, xy2], fill=fill_value)
        projected_faces += 1

    result = np.asarray(image, dtype=np.uint8)

    print(f"      faces projected: {projected_faces}")
    if skipped_faces:
        print(f"      faces skipped:   {skipped_faces}")

    return result


def create_ground_mask(ground_mesh, resolution, bounds):
    """Rasterize the ground into a boolean free-space mask."""
    print("    Rasterizing ground...")
    grid = rasterize_mesh(ground_mesh, resolution, bounds, fill_value=255)
    mask = grid == 255
    print(f"      free pixels: {np.count_nonzero(mask)}")
    return mask


def project_wall_faces(wall_mesh, resolution, bounds):
    """Project wall geometry onto the XY plane."""
    height, width = bounds_to_shape(bounds, resolution)

    if wall_mesh is None:
        print("    No wall mesh found.")
        return np.zeros((height, width), dtype=bool)

    print("    Projecting wall faces...")
    wall_grid = rasterize_mesh(wall_mesh, resolution, bounds, fill_value=255)
    wall_mask = wall_grid == 255
    print(f"      projected wall pixels: {np.count_nonzero(wall_mask)}")
    return wall_mask


def create_ground_map(ground_mask, wall_mask):
    """Create the regular occupancy map."""
    result = np.full(ground_mask.shape, 128, dtype=np.uint8)
    result[ground_mask] = 255
    result[wall_mask] = 0
    return result


def compute_wall_centroid(wall_mask, ground_mask):
    """Calculate the centroid of the wall pixels."""
    wall_pixels = np.argwhere(wall_mask)
    if len(wall_pixels) > 0:
        centroid = np.mean(wall_pixels, axis=0)
        return float(centroid[0]), float(centroid[1])

    print("    WARNING: no wall pixels found, using ground centroid.")
    ground_pixels = np.argwhere(ground_mask)
    if len(ground_pixels) > 0:
        centroid = np.mean(ground_pixels, axis=0)
        return float(centroid[0]), float(centroid[1])

    return ground_mask.shape[0] / 2.0, ground_mask.shape[1] / 2.0


def align_map(grid, offset_h, offset_w, canvas_shape):
    """Copy an original environment map onto the common global canvas."""
    canvas_h, canvas_w = canvas_shape
    source_h, source_w = grid.shape

    canvas = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)

    src_h_start = max(0, -offset_h)
    src_h_end = min(source_h, canvas_h - offset_h)
    src_w_start = max(0, -offset_w)
    src_w_end = min(source_w, canvas_w - offset_w)

    dst_h_start = max(0, offset_h)
    dst_h_end = min(canvas_h, offset_h + source_h)
    dst_w_start = max(0, offset_w)
    dst_w_end = min(canvas_w, offset_w + source_w)

    if (src_h_end <= src_h_start or src_w_end <= src_w_start or
            dst_h_end <= dst_h_start or dst_w_end <= dst_w_start):
        return canvas

    canvas[dst_h_start:dst_h_end, dst_w_start:dst_w_end] = grid[
        src_h_start:src_h_end, src_w_start:src_w_end
    ]
    return canvas


def process_environment(env_name, resolution, margin):
    """Load and rasterize the standard map for one environment."""
    env_folder = ENV_BASE / env_name
    if not env_folder.exists():
        print(f"ERROR: environment directory not found: {env_folder}")
        return None

    ground_file, wall_file = find_environment_files(env_folder, env_name)
    if ground_file is None:
        print(f"ERROR: ground_{env_name}.dae not found.")
        return None

    print(f"\nPROCESSING: {env_name}")
    print("  Files:")
    print(f"    Ground: {ground_file.name}")
    print(f"    Walls:  {wall_file.name if wall_file else 'NOT FOUND'}")

    try:
        ground_mesh = load_dae(ground_file)
        wall_mesh = load_dae(wall_file) if wall_file is not None else None
    except Exception as error:
        print(f"ERROR loading environment: {error}")
        return None

    bounds = compute_bounds(ground_mesh, margin)
    height, width = bounds_to_shape(bounds, resolution)

    print("\n  Map:")
    print(f"    Size: {width} x {height} pixels")
    print(f"    X: {bounds[0]:.4f} -> {bounds[2]:.4f}")
    print(f"    Y: {bounds[1]:.4f} -> {bounds[3]:.4f}")

    ground_mask = create_ground_mask(ground_mesh, resolution, bounds)
    wall_mask = project_wall_faces(wall_mesh, resolution, bounds)
    ground_map = create_ground_map(ground_mask, wall_mask)

    cy, cx = compute_wall_centroid(wall_mask, ground_mask)
    print(f"\n  Alignment centroid: cy={cy:.3f}, cx={cx:.3f}")

    return {
        "env": env_name,
        "ground_map": ground_map,
        "ground_mask": ground_mask,
        "wall_mask": wall_mask,
        "bounds": bounds,
        "shape": ground_map.shape,
        "centroid": (cy, cx),
    }


def parse_wbt_furniture(world_path):
    """Extract Solid objects from the Webots world before the robot."""
    with open(world_path, "r") as file:
        lines = file.readlines()

    robot_index = len(lines)
    for i, line in enumerate(lines):
        if "DEF Justina Robot" in line:
            robot_index = i
            break

    furniture = []
    current_solid = None
    brace_count = 0

    for line in lines[:robot_index]:
        if "Solid {" in line and current_solid is None:
            current_solid = {"translation": None,
                             "rotation": None, "name": None}
            brace_count = line.count("{") - line.count("}")
            continue

        if current_solid is None:
            continue

        if "translation" in line:
            parts = line.split()
            if len(parts) >= 4:
                current_solid["translation"] = (
                    float(parts[1]), float(parts[2]), float(parts[3])
                )
        elif "rotation" in line:
            parts = line.split()
            if len(parts) >= 5:
                current_solid["rotation"] = (
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])
                )
        elif "name" in line:
            parts = line.split('"')
            if len(parts) > 1:
                current_solid["name"] = parts[1]

        brace_count += line.count("{") - line.count("}")

        if brace_count == 0:
            name = current_solid["name"]
            if name is not None and name not in ["ground", "lab_walls"]:
                furniture.append(current_solid)
            current_solid = None

    return furniture


def _normalize(name):
    """Lowercase and remove non-alphanumerics, for fuzzy comparisons."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_suffixed_mesh(models_base, item_name, suffix, verbose=True):
    """
    Find a mesh with the given suffix inside models_base.

    Tries, in order:
      1. Exact {base}_{suffix}.dae
      2. Exact {item_name}_{suffix}.dae
      3. Case-insensitive exact match
      4. Normalized match (ignores underscores, dashes, case)
      5. Fuzzy match via difflib
    """
    if not models_base.exists():
        if verbose:
            print(f"      WARNING: folder does not exist: {models_base}")
        return None

    base_name = re.sub(r"_\d+$", "", item_name)
    candidates = [
        f"{base_name}_{suffix}.dae",
        f"{item_name}_{suffix}.dae",
    ]

    for name in candidates:
        path = models_base / name
        if path.exists():
            return path

    existing = list(models_base.glob("*.dae"))
    lower_map = {f.name.lower(): f for f in existing}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    norm_map = {_normalize(f.stem): f for f in existing}
    for name in candidates:
        key = _normalize(Path(name).stem)
        if key in norm_map:
            return norm_map[key]

    target = _normalize(candidates[0])
    all_keys = list(norm_map.keys())
    closest = difflib.get_close_matches(target, all_keys, n=1, cutoff=0.75)
    if closest:
        return norm_map[closest[0]]

    return None


def find_bb_mesh(item_name):
    """Find the bounding-box mesh for a Webots object."""
    return _find_suffixed_mesh(BB_MODELS_BASE, item_name, "BB")


def find_fp_mesh(item_name):
    """Find the footprint mesh for a Webots object."""
    return _find_suffixed_mesh(FP_MODELS_BASE, item_name, "FP")


def apply_webots_transform_to_points(vertices, translation, rotation):
    """Apply the Webots Solid transformation (rotation, then translation)."""
    vertices = np.asarray(vertices, dtype=float)
    transformed = vertices.copy()

    if rotation is not None:
        axis = np.asarray(rotation[:3], dtype=float)
        angle = float(rotation[3])
        axis_norm = np.linalg.norm(axis)

        if axis_norm > 1e-12:
            axis = axis / axis_norm
            x, y, z = axis

            c = np.cos(angle)
            s = np.sin(angle)
            t = 1.0 - c

            rotation_matrix = np.array([
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ])

            transformed = transformed @ rotation_matrix.T

    if translation is not None:
        transformed += np.asarray(translation, dtype=float)

    return transformed


def _load_mesh_geometry(mesh_path):
    """Load a mesh file and return a single Trimesh object."""
    mesh = trimesh.load(mesh_path)
    if mesh is None:
        raise ValueError(f"Could not load {mesh_path}")

    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values(
        ) if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(geometries)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Invalid mesh type for {mesh_path}")

    return mesh


def project_mesh_convex_hull(mesh_path, translation, rotation, inflation=0.0):
    """Project the mesh as its convex hull (used for BB)."""
    mesh = _load_mesh_geometry(mesh_path)

    transformed_vertices = apply_webots_transform_to_points(
        mesh.vertices, translation, rotation
    )
    vertices_2d = transformed_vertices[:, :2]

    if len(vertices_2d) < 3:
        return None

    polygon = MultiPoint(
        [(float(v[0]), float(v[1])) for v in vertices_2d]
    ).convex_hull

    if polygon.is_empty:
        return None

    if inflation > 0:
        polygon = polygon.buffer(inflation, cap_style=3, join_style=2)

    return polygon


def project_mesh_exact_footprint(mesh_path, translation, rotation, inflation=0.0,
                                 min_triangle_area=1e-6):
    """
    Project the mesh keeping every separate piece.

    Each triangle is projected to XY and kept as an independent polygon.
    After the union, the result is forced back into a MultiPolygon so that
    pieces touching only at a vertex do not get merged into a single blob.

    A minimum triangle area filter drops degenerate triangles that can
    appear in DAE files exported from CAD software.
    """
    mesh = _load_mesh_geometry(mesh_path)

    transformed_vertices = apply_webots_transform_to_points(
        mesh.vertices, translation, rotation
    )

    pieces = []
    for face in mesh.faces:
        if len(face) != 3:
            continue

        try:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        except (ValueError, IndexError):
            continue

        if (i0 < 0 or i1 < 0 or i2 < 0 or
                i0 >= len(transformed_vertices) or
                i1 >= len(transformed_vertices) or
                i2 >= len(transformed_vertices)):
            continue

        p0 = transformed_vertices[i0]
        p1 = transformed_vertices[i1]
        p2 = transformed_vertices[i2]

        if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            continue

        try:
            tri = Polygon([
                (float(p0[0]), float(p0[1])),
                (float(p1[0]), float(p1[1])),
                (float(p2[0]), float(p2[1])),
            ])
        except Exception:
            continue

        if tri.is_empty or tri.area < min_triangle_area:
            continue

        if not tri.is_valid:
            tri = tri.buffer(0)

        if tri.is_empty:
            continue

        pieces.append(tri)

    if not pieces:
        return None

    footprint = unary_union(pieces)

    if not footprint.is_valid:
        footprint = footprint.buffer(0)

    if footprint.is_empty:
        return None

    # If we ended up with a single Polygon, keep it as-is.
    # If pieces are disconnected, they will already be a MultiPolygon.
    # A tiny negative buffer followed by a positive buffer separates
    # pieces that touch only at a vertex, which is what we want for
    # tables with thin legs.
    if footprint.geom_type == "Polygon":
        footprint = footprint.buffer(0)

    if inflation > 0:
        footprint = footprint.buffer(inflation, cap_style=3, join_style=2)

    return footprint


def world_to_aligned_pixel(x, y, bounds, resolution, offset_h, offset_w):
    """Convert a world coordinate into the final aligned canvas."""
    minx, miny, maxx, maxy = bounds
    source_col = (x - minx) / resolution
    source_row = (maxy - y) / resolution

    col = source_col + offset_w
    row = source_row + offset_h

    return col, row


def _ring_to_pixels(ring, bounds, resolution, offset_h, offset_w):
    """Convert a shapely ring to pixel coordinates."""
    coords = []
    for x, y in ring.coords:
        col, row = world_to_aligned_pixel(
            x, y, bounds, resolution, offset_h, offset_w)
        coords.append((int(round(col)), int(round(row))))
    return coords


def draw_polygon(grid, polygon, bounds, resolution, offset_h, offset_w, base_grid=None):
    """
    Draw a polygon or multipolygon on the aligned map.

    Each exterior ring becomes occupied (0). Each interior ring is restored
    from base_grid so that holes are preserved (for example, the empty area
    under a table).
    """
    if base_grid is None:
        base_grid = grid.copy()

    image = Image.fromarray(grid)
    draw = ImageDraw.Draw(image)

    if polygon.geom_type == "MultiPolygon":
        polys = list(polygon.geoms)
    else:
        polys = [polygon]

    for poly in polys:
        if poly.geom_type != "Polygon":
            continue

        exterior = _ring_to_pixels(
            poly.exterior, bounds, resolution, offset_h, offset_w)
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=0)

    result = np.asarray(image).copy()
    base_arr = np.asarray(base_grid)

    for poly in polys:
        if poly.geom_type != "Polygon":
            continue

        for interior in poly.interiors:
            hole = _ring_to_pixels(
                interior, bounds, resolution, offset_h, offset_w)
            if len(hole) < 3:
                continue

            hole_mask_img = Image.new("L", image.size, 0)
            hole_draw = ImageDraw.Draw(hole_mask_img)
            hole_draw.polygon(hole, fill=1)
            hole_mask = np.array(hole_mask_img, dtype=bool)

            result[hole_mask] = base_arr[hole_mask]

    return result


def resolve_world_file(env_name, requested_world_file):
    """Find the WBT file for an environment."""
    candidates = []

    if requested_world_file:
        candidates.append(Path(requested_world_file))
        candidates.append(WORLDS_BASE / requested_world_file)

    prefix = env_name.split("_")[0]
    candidates.append(WORLDS_BASE / f"{prefix}.wbt")
    candidates.append(WORLDS_BASE / f"{env_name}.wbt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def save_map(grid, origin_x, origin_y, resolution, env_name, suffix="",
             world_offset_x=0.0, world_offset_y=0.0, world_offset_yaw=0.0):
    """Save PNG and YAML map files."""
    env_map_dir = MAPS_BASE / env_name
    env_map_dir.mkdir(exist_ok=True)

    image_name = f"map_{env_name}{suffix}.png"
    yaml_name = f"map_{env_name}{suffix}.yaml"

    image_path = env_map_dir / image_name
    yaml_path = env_map_dir / yaml_name

    Image.fromarray(grid).save(image_path)

    metadata = {
        "image": image_name,
        "resolution": float(resolution),
        "origin": [float(origin_x), float(origin_y), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "world_offset_x": float(world_offset_x),
        "world_offset_y": float(world_offset_y),
        "world_offset_yaw": float(world_offset_yaw),
    }

    with open(yaml_path, "w") as file:
        yaml.dump(metadata, file, sort_keys=False)

    print(f"    Saved: {image_path}")
    print(f"    Saved: {yaml_path}")


def count_pieces(polygon):
    """Return (num_polygons, num_holes) for a shapely geometry."""
    if polygon is None:
        return 0, 0

    if polygon.geom_type == "MultiPolygon":
        polys = list(polygon.geoms)
    elif polygon.geom_type == "Polygon":
        polys = [polygon]
    else:
        return 0, 0

    holes = sum(len(p.interiors) for p in polys if p.geom_type == "Polygon")
    return len(polys), holes


def build_furniture_map(base_grid, furniture, finder, projector, bounds, resolution,
                        offset_h, offset_w, inflation=0.0, label="", verbose=True):
    """
    Draw every furniture footprint on top of base_grid.
    finder returns the mesh path for a given item name.
    projector is either convex hull or exact footprint.
    """
    grid = base_grid.copy()
    matched = 0
    missing = 0
    total_pieces = 0
    total_holes = 0

    for item in furniture:
        item_name = item["name"]
        mesh_path = finder(item_name)

        if mesh_path is None:
            missing += 1
            if verbose:
                print(f"    Skipping {item_name}: no {label} mesh found.")
            continue

        try:
            polygon = projector(
                mesh_path,
                item["translation"],
                item["rotation"],
                inflation=inflation,
            )

            if polygon is None:
                print(f"    Skipping {item_name}: empty polygon.")
                continue

            n_pieces, n_holes = count_pieces(polygon)
            total_pieces += n_pieces
            total_holes += n_holes

            if verbose:
                print(
                    f"    [{label}] {item_name} -> {mesh_path.name} "
                    f"(pieces={n_pieces}, holes={n_holes})"
                )

            grid = draw_polygon(
                grid, polygon, bounds, resolution, offset_h, offset_w,
                base_grid=base_grid,
            )
            matched += 1

        except Exception as error:
            print(f"    ERROR processing {item_name}: {error}")

    if verbose:
        print(
            f"    {label}: matched={matched}, missing={missing}, "
            f"pieces={total_pieces}, holes={total_holes}"
        )

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Generate standard, prohibition and navigation occupancy maps."
    )
    parser.add_argument("env_name", nargs="?")
    parser.add_argument("resolution", nargs="?", type=float, default=0.01)
    parser.add_argument("inner_thickness", nargs="?", type=float, default=0.05,
                        help="Reserved for compatibility.")
    parser.add_argument("outer_thickness", nargs="?", type=float, default=0.05,
                        help="Reserved for compatibility.")
    parser.add_argument("--align", type=lambda value: value.lower() in ["true", "1", "yes"],
                        default=True)
    parser.add_argument("--world-file", type=str, default=None)
    parser.add_argument("--margin", type=float, default=0.50)
    parser.add_argument("--obstacle-inflation", type=float, default=0.012,
                        help="Inflate FP obstacles in the navigation map, in meters.")

    args = parser.parse_args()

    resolution = args.resolution
    margin = args.margin
    obstacle_inflation = args.obstacle_inflation

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero")
    if margin < 0:
        raise ValueError("margin must be greater than or equal to zero")

    if args.env_name:
        env_list = [args.env_name]
    else:
        env_list = []
        for directory in ENV_BASE.iterdir():
            if directory.is_dir() and any(directory.glob("ground_*.dae")):
                env_list.append(directory.name)
        env_list.sort()

    if not env_list:
        print("No environment folders found.")
        sys.exit(1)

    print("DAE / WEBOTS MAP GENERATOR")
    print(f"  Resolution: {resolution} m/pixel")
    print(f"  Margin:     {margin} m")
    print(f"  Alignment:  {args.align}")
    print(f"  Obstacle inflation (nav): {obstacle_inflation} m")

    processed = []
    global_max_h = 0
    global_max_w = 0

    for env_name in env_list:
        data = process_environment(env_name, resolution, margin)
        if data is None:
            continue

        processed.append(data)
        height, width = data["shape"]
        global_max_h = max(global_max_h, height)
        global_max_w = max(global_max_w, width)

    if not processed:
        print("No maps generated.")
        sys.exit(1)

    if args.align:
        canvas_shape = (global_max_h, global_max_w)
        target_h = global_max_h // 2
        target_w = global_max_w // 2
        print(f"\nCOMMON CANVAS: {global_max_w}x{global_max_h}")
    else:
        canvas_shape = None
        target_h = 0
        target_w = 0

    for data in processed:
        env_name = data["env"]
        cy, cx = data["centroid"]

        if args.align:
            offset_h = target_h - int(round(cy))
            offset_w = target_w - int(round(cx))

            ground_map = align_map(
                data["ground_map"], offset_h, offset_w, canvas_shape)

            origin_x = -global_max_w * resolution / 2.0
            origin_y = -global_max_h * resolution / 2.0

            minx, miny, maxx, maxy = data["bounds"]
            canvas_h = canvas_shape[0]
            world_offset_x = minx - origin_x - offset_w * resolution
            world_offset_y = (maxy - origin_y
                              - (canvas_h - 1) * resolution
                              + offset_h * resolution)

            print(f"\nSAVING {env_name}")
            print(f"  Wall centroid: cx={cx:.3f}, cy={cy:.3f}")
            print(f"  Alignment offset: X={offset_w}px, Y={offset_h}px")
            print(f"  World offset: X={world_offset_x:.3f}, "
                  f"Y={world_offset_y:.3f}")
        else:
            offset_h = 0
            offset_w = 0
            ground_map = data["ground_map"]
            bounds = data["bounds"]
            origin_x = bounds[0]
            origin_y = bounds[1]
            world_offset_x = 0.0
            world_offset_y = 0.0

        save_map(ground_map, origin_x, origin_y, resolution, env_name,
                 world_offset_x=world_offset_x,
                 world_offset_y=world_offset_y)

        world_file = resolve_world_file(env_name, args.world_file)
        if world_file is None:
            print(
                f"  No world file found for {env_name}; skipping furniture maps.")
            continue

        print(f"\n  Reading furniture from {world_file.name}...")
        furniture = parse_wbt_furniture(world_file)
        print(f"  Furniture objects found: {len(furniture)}")

        print("\n  Generating prohibition map (BB, convex hull)...")
        prohibition_grid = build_furniture_map(
            ground_map,
            furniture,
            finder=find_bb_mesh,
            projector=project_mesh_convex_hull,
            bounds=data["bounds"],
            resolution=resolution,
            offset_h=offset_h,
            offset_w=offset_w,
            inflation=0.0,
            label="BB",
        )
        save_map(prohibition_grid, origin_x, origin_y,
                 resolution, env_name, "_prohibition",
                 world_offset_x=world_offset_x,
                 world_offset_y=world_offset_y)

        print("\n  Generating navigation map (FP, exact footprint)...")
        nav_grid = build_furniture_map(
            ground_map,
            furniture,
            finder=find_fp_mesh,
            projector=project_mesh_exact_footprint,
            bounds=data["bounds"],
            resolution=resolution,
            offset_h=offset_h,
            offset_w=offset_w,
            inflation=obstacle_inflation,
            label="FP",
        )
        save_map(nav_grid, origin_x, origin_y, resolution, env_name, "_nav",
                 world_offset_x=world_offset_x,
                 world_offset_y=world_offset_y)

    print("\nDONE")


if __name__ == "__main__":
    main()
