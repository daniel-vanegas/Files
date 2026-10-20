#!/usr/bin/env python3
"""
Extract the static environment from a Webots .wbt world and export it as a Collada .dae file.

The robot is intentionally ignored. Only objects inside BioRoboticsLab_Mesh are exported.
The ground Plane is also ignored.

Usage:
    ./wbt_to_dae.py E01.wbt -o E01.dae
    ./wbt_to_dae.py ../worlds/E01.wbt -o E01.dae

If the WBT is not in the current directory, the script checks ../worlds/ automatically.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh


def parse_vector(text: str) -> np.ndarray:
    """Parse a space-separated vector."""
    return np.fromstring(text.strip(), sep=" ")


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert a Webots axis-angle rotation to a 4x4 matrix."""
    length = np.linalg.norm(axis)
    if length == 0:
        return np.eye(4)

    x, y, z = axis / length
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def make_transform(translation: np.ndarray | None, rotation: np.ndarray | None) -> np.ndarray:
    """Create a Webots transformation matrix."""
    transform = np.eye(4)
    if rotation is not None:
        transform = axis_angle_to_matrix(rotation[:3], rotation[3])
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def extract_block(text: str, start: int) -> str:
    """Extract a complete VRML node starting at its declaration."""
    opening_brace = text.find("{", start)
    if opening_brace == -1:
        raise RuntimeError("Could not find the opening brace of a node.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(opening_brace, len(text)):
        char = text[index]
        if char == '"' and not escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False

    raise RuntimeError("Could not find the end of the VRML node.")


def get_brace_depth(text: str, position: int) -> int:
    """Return the VRML brace depth at a given position."""
    depth = 0
    in_string = False
    escaped = False

    for char in text[:position]:
        if char == '"' and not escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False

    return depth


def find_environment_block(world_text: str) -> str:
    """Find the BioRoboticsLab_Mesh group."""
    match = re.search(r"DEF\s+BioRoboticsLab_Mesh\s+Group\s*\{", world_text)
    if not match:
        raise RuntimeError("DEF BioRoboticsLab_Mesh Group was not found.")
    return extract_block(world_text, match.start())


def find_top_level_solids(group_text: str) -> list[str]:
    """
    Find Solid nodes directly contained by BioRoboticsLab_Mesh.

    This is limited to direct children so the robot hierarchy is never traversed.
    """
    solids = []
    for match in re.finditer(r"\bSolid\s*\{", group_text):
        depth = get_brace_depth(group_text, match.start())
        if depth != 1:  # The Group opens at depth 1
            continue
        solids.append(extract_block(group_text, match.start()))
    return solids


def extract_name(node_text: str) -> str:
    """Read the name of a Webots node."""
    match = re.search(r'\bname\s+"([^"]+)"', node_text)
    return match.group(1) if match else "environment_object"


def extract_transform(node_text: str) -> np.ndarray:
    """Read translation and rotation from a Webots Solid."""
    translation = None
    rotation = None

    match = re.search(r"\btranslation\s+([^\n\r]+)", node_text)
    if match:
        values = parse_vector(match.group(1))
        if len(values) >= 3:
            translation = values[:3]

    match = re.search(r"\brotation\s+([^\n\r]+)", node_text)
    if match:
        values = parse_vector(match.group(1))
        if len(values) >= 4:
            rotation = values[:4]

    return make_transform(translation, rotation)


def extract_visual_mesh_path(node_text: str) -> str | None:
    """Get the visual CadShape mesh path. Collision meshes are ignored."""
    match = re.search(r"CadShape\s*\{\s*url\s*\[\s*\"([^\"]+)\"", node_text, re.DOTALL)
    return match.group(1) if match else None


def load_dae_meshes(path: Path) -> list[trimesh.Trimesh]:
    """
    Load all geometries from a DAE file.

    A DAE can contain multiple geometries, so each one is returned separately.
    """
    loaded = trimesh.load(path, force="scene", process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    if not isinstance(loaded, trimesh.Scene):
        raise RuntimeError(f"Unsupported mesh type in '{path}'.")

    meshes = []
    for geometry_name, geometry in loaded.geometry.items():
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        transform, _ = loaded.graph.get(geometry_name)
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def apply_transform(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    """Apply the Webots Solid transform to a mesh."""
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def calculate_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Get vertex normals.

    If trimesh cannot provide valid normals, calculate them from the faces.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    try:
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if len(normals) == len(vertices) and np.isfinite(normals).all():
            return normals
    except Exception:
        pass

    normals = np.zeros_like(vertices)
    for face in faces:
        a, b, c = face
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        length = np.linalg.norm(normal)
        if length > 0:
            normal /= length
        normals[a] += normal
        normals[b] += normal
        normals[c] += normal

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    return normals


def format_float(value: float) -> str:
    """Format a float for Collada."""
    return f"{float(value):.9g}"


def add_source(mesh_element: ET.Element, source_id: str, values: np.ndarray) -> None:
    """Add a float source to a Collada mesh."""
    values = np.asarray(values, dtype=float)
    source = ET.SubElement(mesh_element, "source", {"id": source_id})
    flat = values.reshape(-1)

    float_array = ET.SubElement(source, "float_array", {"id": f"{source_id}-array", "count": str(len(flat))})
    float_array.text = " ".join(format_float(v) for v in flat)

    technique = ET.SubElement(source, "technique_common")
    accessor = ET.SubElement(technique, "accessor", {"source": f"#{source_id}-array", "count": str(len(values)), "stride": "3"})
    for name in ("X", "Y", "Z"):
        ET.SubElement(accessor, "param", {"name": name, "type": "float"})


def add_geometry(library_geometries: ET.Element, mesh: trimesh.Trimesh, geometry_id: str, name: str) -> None:
    """Write one mesh as a Collada geometry."""
    geometry = ET.SubElement(library_geometries, "geometry", {"id": geometry_id, "name": name})
    mesh_element = ET.SubElement(geometry, "mesh")

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = calculate_vertex_normals(mesh)

    add_source(mesh_element, f"{geometry_id}-positions", vertices)
    add_source(mesh_element, f"{geometry_id}-normals", normals)

    vertices_element = ET.SubElement(mesh_element, "vertices", {"id": f"{geometry_id}-vertices"})
    ET.SubElement(vertices_element, "input", {"semantic": "POSITION", "source": f"#{geometry_id}-positions"})

    triangles = ET.SubElement(mesh_element, "triangles", {"count": str(len(faces)), "material": "material0"})
    ET.SubElement(triangles, "input", {"semantic": "VERTEX", "source": f"#{geometry_id}-vertices", "offset": "0"})
    ET.SubElement(triangles, "input", {"semantic": "NORMAL", "source": f"#{geometry_id}-normals", "offset": "1"})

    indices = []
    for face in faces:
        for vertex_index in face:
            indices.append(str(int(vertex_index)))
            indices.append(str(int(vertex_index)))

    p = ET.SubElement(triangles, "p")
    p.text = " ".join(indices)


def add_material(library_effects: ET.Element, library_materials: ET.Element) -> None:
    """Create a simple default material."""
    effect = ET.SubElement(library_effects, "effect", {"id": "environment_effect"})
    profile = ET.SubElement(effect, "profile_COMMON")
    technique = ET.SubElement(profile, "technique", {"sid": "common"})
    phong = ET.SubElement(technique, "phong")
    diffuse = ET.SubElement(phong, "diffuse")
    color = ET.SubElement(diffuse, "color")
    color.text = "0.8 0.8 0.8 1"

    material = ET.SubElement(library_materials, "material", {"id": "environment_material", "name": "environment_material"})
    ET.SubElement(material, "instance_effect", {"url": "#environment_effect"})


def create_collada(meshes: list[tuple[str, trimesh.Trimesh]], output_path: Path) -> None:
    """Write all extracted meshes to a Collada 1.4.1 file."""
    root = ET.Element("COLLADA", {"xmlns": "http://www.collada.org/2005/11/COLLADASchema", "version": "1.4.1"})

    asset = ET.SubElement(root, "asset")
    contributor = ET.SubElement(asset, "contributor")
    author = ET.SubElement(contributor, "author")
    author.text = "wbt_to_dae.py"

    created = ET.SubElement(asset, "created")
    created.text = "2026-09-07"
    modified = ET.SubElement(asset, "modified")
    modified.text = "2026-09-07"

    up_axis = ET.SubElement(asset, "up_axis")
    up_axis.text = "Z_UP"

    library_effects = ET.SubElement(root, "library_effects")
    library_materials = ET.SubElement(root, "library_materials")
    add_material(library_effects, library_materials)

    library_geometries = ET.SubElement(root, "library_geometries")
    geometry_entries = []

    for index, (name, mesh) in enumerate(meshes):
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        geometry_id = f"{safe_name}_{index}_geometry"
        add_geometry(library_geometries, mesh, geometry_id, name)
        geometry_entries.append((name, geometry_id))

    library_visual_scenes = ET.SubElement(root, "library_visual_scenes")
    visual_scene = ET.SubElement(library_visual_scenes, "visual_scene", {"id": "EnvironmentScene", "name": "EnvironmentScene"})

    for index, (name, geometry_id) in enumerate(geometry_entries):
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        node = ET.SubElement(visual_scene, "node", {"id": f"{safe_name}_{index}", "name": name})
        instance_geometry = ET.SubElement(node, "instance_geometry", {"url": f"#{geometry_id}"})
        bind_material = ET.SubElement(instance_geometry, "bind_material")
        technique_common = ET.SubElement(bind_material, "technique_common")
        ET.SubElement(technique_common, "instance_material", {"symbol": "material0", "target": "#environment_material"})

    scene = ET.SubElement(root, "scene")
    ET.SubElement(scene, "instance_visual_scene", {"url": "#EnvironmentScene"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def resolve_world_path(requested_path: Path) -> Path:
    """
    Resolve the WBT path.

    First try the path relative to the current directory.
    If that fails, try ../worlds/ from the script directory.
    """
    candidate = requested_path.resolve() if requested_path.is_absolute() else (Path.cwd() / requested_path).resolve()
    if candidate.exists():
        return candidate

    fallback = (Path(__file__).resolve().parent / ".." / "worlds" / requested_path.name).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"World file does not exist.\nTried:\n  {candidate}\n  {fallback}")


def collect_environment(world_path: Path) -> list[tuple[str, trimesh.Trimesh]]:
    """Extract and transform all static environment meshes."""
    world_text = world_path.read_text(encoding="utf-8")
    environment_text = find_environment_block(world_text)
    solids = find_top_level_solids(environment_text)

    if not solids:
        raise RuntimeError("No Solid objects were found inside BioRoboticsLab_Mesh.")

    world_directory = world_path.parent
    result = []

    for solid in solids:
        name = extract_name(solid)
        mesh_reference = extract_visual_mesh_path(solid)

        # Ground is a Plane and has no CadShape, so it's skipped.
        if mesh_reference is None:
            print(f"Skipping '{name}': no visual CadShape.")
            continue

        mesh_path = (world_directory / mesh_reference).resolve()
        print(f"Object: {name}")
        print(f"  Mesh: {mesh_reference}")
        print(f"  Resolved: {mesh_path}")

        if not mesh_path.exists():
            print("  WARNING: mesh does not exist, skipping.\n")
            continue

        transform = extract_transform(solid)

        try:
            meshes = load_dae_meshes(mesh_path)
        except Exception as exc:
            print(f"  ERROR loading mesh: {exc}\n")
            continue

        for index, mesh in enumerate(meshes):
            transformed = apply_transform(mesh, transform)
            output_name = name if len(meshes) == 1 else f"{name}_{index}"
            result.append((output_name, transformed))

        print(f"  Added geometries: {len(meshes)}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract static Webots environment objects and export them as Collada.")
    parser.add_argument("world", type=Path, help="Input .wbt file.")
    parser.add_argument("output_positional", type=Path, nargs="?", default=None, help="Output .dae file.")
    parser.add_argument("-o", "--output", type=Path, dest="output_option", default=None, help="Output .dae file.")

    args = parser.parse_args()

    if args.output_positional is not None and args.output_option is not None:
        parser.error("Specify the output either as a positional argument or with -o, not both.")

    world_path = resolve_world_path(args.world)

    if args.output_option is not None:
        output_path = args.output_option
    elif args.output_positional is not None:
        output_path = args.output_positional
    else:
        output_path = Path.cwd() / f"{world_path.stem}_objects.dae"

    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()

    print("\nWebots WBT to DAE\n=================\n")
    print(f"Input world:\n  {world_path}\n")
    print(f"Output:\n  {output_path}\n")
    print("Ground: disabled")
    print("Robot: disabled\n")

    meshes = collect_environment(world_path)

    if not meshes:
        raise RuntimeError("No environment meshes were found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Total geometries: {len(meshes)}\n")
    print(f"Writing Collada:\n  {output_path}")

    create_collada(meshes, output_path)

    print("\nDone.")
    print(f"Created: {output_path}\n")


if __name__ == "__main__":
    main()