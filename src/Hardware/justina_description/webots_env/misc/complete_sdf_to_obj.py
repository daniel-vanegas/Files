#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import trimesh
import numpy as np
import sys


def parse_pose(pose_str):
    if pose_str is None:
        return np.eye(4)
    vals = list(map(float, pose_str.split()))
    if len(vals) != 6:
        return np.eye(4)
    x, y, z, roll, pitch, yaw = vals
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x, y, z]
    return T


def create_mesh_from_geometry(geom_elem, pose_str):
    box = geom_elem.find('box')
    if box is not None:
        size = list(map(float, box.find('size').text.split()))
        mesh = trimesh.creation.box(extents=size)
    else:
        cyl = geom_elem.find('cylinder')
        if cyl is not None:
            radius = float(cyl.find('radius').text)
            length = float(cyl.find('length').text)
            mesh = trimesh.creation.cylinder(radius=radius, height=length, segments=32)
        else:
            sphere = geom_elem.find('sphere')
            if sphere is not None:
                radius = float(sphere.find('radius').text)
                mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
            else:
                return None
    T = parse_pose(pose_str)
    mesh.apply_transform(T)
    mesh.visual.face_colors = [200, 180, 150, 255]   # light wood
    return mesh


def main():
    # Input SDF file
    if len(sys.argv) > 1:
        sdf_file = sys.argv[1]
    else:
        sdf_file = 'model-1_4.sdf'
    try:
        tree = ET.parse(sdf_file)
    except FileNotFoundError:
        print(f"Error: file '{sdf_file}' not found.")
        sys.exit(1)
    root = tree.getroot()

    # Find the first <model>
    model = root.find('.//model')
    if model is None:
        print("No <model> found.")
        sys.exit(1)

    # Gather all links
    links = {link.get('name'): link for link in model.findall('link')}
    link_names = set(links.keys())

    # Collect link poses (relative to model frame)
    link_poses = {}
    for name, link in links.items():
        pose_elem = link.find('pose')
        link_poses[name] = parse_pose(pose_elem.text if pose_elem is not None else "0 0 0 0 0 0")

    # Collect joints
    joint_parent = {}
    joint_pose = {}
    for joint in model.findall('joint'):
        child = joint.find('child').text
        parent = joint.find('parent').text
        pose_elem = joint.find('pose')
        jpose = parse_pose(pose_elem.text if pose_elem is not None else "0 0 0 0 0 0")
        joint_parent[child] = parent
        joint_pose[child] = jpose

    # Determine root links: those with no parent, or whose parent is not a link (e.g., "world")
    child_links = set(joint_parent.keys())
    root_links = set()
    for link in link_names:
        if link not in child_links:
            root_links.add(link)
        else:
            parent = joint_parent[link]
            if parent not in link_names:
                root_links.add(link)

    if not root_links:
        print("No root link found. Using 'body' as root.")
        root_links = {'body'}

    # Choose a root (prefer 'body' if present)
    root_name = 'body' if 'body' in root_links else next(iter(root_links))

    # Compute absolute transform for each link
    abs_transforms = {}
    def compute_abs(link_name, parent_abs):
        if link_name == root_name:
            # Root: its own pose relative to model frame
            abs_transforms[link_name] = link_poses[link_name]
            return
        parent = joint_parent[link_name]
        # If parent is not a link, we already handled it as root:
        if parent not in link_names:
            abs_transforms[link_name] = link_poses[link_name]
            return
        # Ensure parent transform is computed
        if parent not in abs_transforms:
            compute_abs(parent, np.eye(4))
        parent_abs = abs_transforms[parent]
        # Child transform = parent_abs * joint_pose (ignore child's own pose)
        abs_transforms[link_name] = parent_abs @ joint_pose[link_name]

    # Compute for all links
    for link in link_names:
        if link not in abs_transforms:
            compute_abs(link, np.eye(4))

    # Collect all visual meshes
    meshes = []
    for link in model.findall('link'):
        link_name = link.get('name')
        if link_name not in abs_transforms:
            print(f"Warning: link '{link_name}' has no absolute transform, skipping.")
            continue
        link_transform = abs_transforms[link_name]
        for visual in link.findall('visual'):
            geom = visual.find('geometry')
            if geom is None:
                continue
            pose_elem = visual.find('pose')
            pose_str = pose_elem.text if pose_elem is not None else "0 0 0 0 0 0"
            mesh = create_mesh_from_geometry(geom, pose_str)
            if mesh is not None:
                mesh.apply_transform(link_transform)
                meshes.append(mesh)
                print(f"Added {visual.get('name')} from link {link_name}")

    if not meshes:
        print("No visual geometries found.")
        return

    # Combine and export
    scene = trimesh.Scene(meshes)
    outfile = 'sideboard_complete.obj'
    scene.export(outfile)
    print(f"Exported to {outfile}")

if __name__ == '__main__':
    main()