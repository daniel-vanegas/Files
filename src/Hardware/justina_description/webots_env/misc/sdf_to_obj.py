#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
import os
import sys

def euler_matrix(roll, pitch, yaw):
    """Return 4x4 transformation matrix from extrinsic XYZ Euler angles."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    return T

def parse_pose(pose_str):
    """Convert SDF pose string to 4x4 transform matrix."""
    vals = list(map(float, pose_str.split()))
    if len(vals) != 6:
        return np.eye(4)
    x, y, z, roll, pitch, yaw = vals
    T = euler_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T

def create_geometry_from_element(geom_elem):
    """Create a trimesh geometry from a <geometry> element."""
    # Box
    box = geom_elem.find('box')
    if box is not None:
        size_elem = box.find('size')
        if size_elem is not None:
            size = list(map(float, size_elem.text.split()))
            if len(size) == 3:
                return trimesh.creation.box(extents=size)
    # Cylinder
    cyl = geom_elem.find('cylinder')
    if cyl is not None:
        radius_elem = cyl.find('radius')
        length_elem = cyl.find('length')
        if radius_elem is not None and length_elem is not None:
            radius = float(radius_elem.text)
            length = float(length_elem.text)
            return trimesh.creation.cylinder(radius=radius, height=length, segments=32)
    # Sphere
    sphere = geom_elem.find('sphere')
    if sphere is not None:
        radius_elem = sphere.find('radius')
        if radius_elem is not None:
            radius = float(radius_elem.text)
            return trimesh.creation.icosphere(radius=radius, subdivisions=2)
    # Mesh
    mesh = geom_elem.find('mesh')
    if mesh is not None:
        uri_elem = mesh.find('uri')
        if uri_elem is not None:
            uri = uri_elem.text.strip()
            # Convert model to relative path
            if uri.startswith('model://'):
                rel_path = uri[8:]
                file_path = os.path.join(os.getcwd(), *rel_path.split('/'))
            else:
                file_path = uri
            if os.path.exists(file_path):
                try:
                    mesh_obj = trimesh.load(file_path, force='mesh')
                    if mesh_obj is not None:
                        return mesh_obj
                except Exception as e:
                    print(f"Failed to load mesh {file_path}: {e}")
            else:
                print(f"Mesh file not found: {file_path}")
    return None

def main():
    sdf_file = 'model-1_4.sdf'
    if not os.path.exists(sdf_file):
        print(f"Error: {sdf_file} not found.")
        sys.exit(1)

    tree = ET.parse(sdf_file)
    root = tree.getroot()

    # Get model name for output
    model_elem = root.find('model')
    model_name = model_elem.get('name', 'model') if model_elem is not None else 'model'

    # Find the link (assume first link)
    link_elem = root.find('.//link')
    if link_elem is None:
        print("No <link> found.")
        return

    # Link pose (default identity)
    link_pose_elem = link_elem.find('pose')
    link_pose = link_pose_elem.text if link_pose_elem is not None else "0 0 0 0 0 0"
    T_link = parse_pose(link_pose)

    # Gather all visual geometries
    visuals = link_elem.findall('visual')
    if not visuals:
        print("No <visual> elements found.")
        return

    meshes = []
    for visual in visuals:
        name = visual.get('name', 'unnamed')
        # Visual pose relative to link
        vis_pose_elem = visual.find('pose')
        vis_pose = vis_pose_elem.text if vis_pose_elem is not None else "0 0 0 0 0 0"
        T_vis = parse_pose(vis_pose)
        T_combined = T_link @ T_vis

        geom_elem = visual.find('geometry')
        if geom_elem is None:
            continue

        mesh_obj = create_geometry_from_element(geom_elem)
        if mesh_obj is None:
            print(f"Skipping {name}: unsupported or missing geometry")
            continue

        # Apply combined transform
        mesh_obj.apply_transform(T_combined)
        mesh_obj.visual.face_colors = [200, 180, 150, 255]
        meshes.append(mesh_obj)
        print(f"Added {name}")

    if not meshes:
        print("No geometries created.")
        return

    # Combine all into a scene
    scene = trimesh.Scene(meshes)
    output_file = f"{model_name}.obj"
    scene.export(output_file)
    print(f"Exported to {output_file}")

if __name__ == "__main__":
    main()