"""Render an animated mesh sequence to video with pyrender."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """Camera-to-world pose looking from eye to target (OpenGL convention)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, np.cross(right, forward), -forward, eye
    return pose


def render_animation(tpose_mesh, all_frame_vertices, output_path, fps=10, resolution=720,
                     elevation_deg=20.0, yfov=np.pi / 6):
    """
    Render (T, V, 3) vertices with the T-pose mesh's faces and texture. The
    world is Y up; a fixed camera looks from +Z, elevation_deg above the
    horizon, and frames the whole sequence over a grey floor at the lowest vertex.
    """
    all_vertices = all_frame_vertices.reshape(-1, 3)
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    center = (lower + upper) / 2.0
    radius = np.linalg.norm(upper - lower) / 2.0
    distance = 0.9 * radius / np.sin(yfov / 2.0)
    elevation = np.radians(elevation_deg)
    eye = center + distance * np.array([0.0, np.sin(elevation), np.cos(elevation)])
    camera_pose = look_at(eye, center)

    margin = 0.3 * np.ptp(all_vertices[:, [0, 2]], axis=0).max() + 1e-3
    floor = trimesh.creation.box(extents=[upper[0] - lower[0] + 2 * margin, 1e-3, upper[2] - lower[2] + 2 * margin])
    floor.apply_translation([center[0], lower[1] - 5e-4, center[2]])
    floor_mesh = pyrender.Mesh.from_trimesh(floor, smooth=False, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.75, 0.75, 0.75, 1.0], metallicFactor=0.0, roughnessFactor=1.0))

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(floor_mesh)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(intensity=2.5), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(intensity=1.5), pose=look_at(center + np.array([0.0, distance, 0.0]),
                                                                      center, up=(0.0, 0.0, -1.0)))

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    mesh_node = None
    try:
        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for vertices in tqdm(all_frame_vertices, desc="Rendering video"):
                frame_mesh = trimesh.Trimesh(vertices=vertices, faces=tpose_mesh.faces, process=False)
                frame_mesh.visual = tpose_mesh.visual
                if mesh_node is not None:
                    scene.remove_node(mesh_node)
                mesh_node = scene.add(pyrender.Mesh.from_trimesh(frame_mesh))
                color, _ = renderer.render(scene)
                writer.append_data(color)
    finally:
        renderer.delete()


def main():
    parser = argparse.ArgumentParser(description="Render animation.mp4 for an animate_mesh.py output folder.")
    parser.add_argument("--animation-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=720)
    args = parser.parse_args()
    from mesh_animation.animate import load_tpose_mesh
    vertices = np.load(args.animation_dir / "animated_mesh.npz")["vertices"]
    tpose_mesh = load_tpose_mesh(args.animation_dir / "tpose_mesh" / "base.obj")
    render_animation(tpose_mesh, vertices, args.animation_dir / "animation.mp4", fps=args.fps, resolution=args.resolution)
    print(f"Saved {args.animation_dir / 'animation.mp4'}")


if __name__ == "__main__":
    main()
