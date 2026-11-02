#!/usr/bin/env python3
"""Interactive viser viewer for a reconstructed joint sequence.

Usage:
    python view_motion.py --joints-path OUT/joints_global.npy --sample-dir CLIP_DIR [--load-mesh]
    python view_motion.py --animation-dir ANIMATION_DIR

Open the printed URL in a browser. The viewer plays the skeleton next to the
input RGB frames; --load-mesh also shows the per-frame SMAL meshes written by
run_reconstruction.py --save-mesh, and --animation-dir shows a mesh animated by
animate_mesh.py together with the joints it follows.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import trimesh
import viser
import viser.transforms as tf

from smal_models.custom_smal import SMAL_PARENTS

SMAL_EDGES = [(i, p) for i, p in enumerate(SMAL_PARENTS) if p != -1]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a joint sequence with viser.")
    parser.add_argument("--joints-path", type=Path, default=None,
                        help="joints_global.npy, shape (T, 35, 3). Optional with --animation-dir.")
    parser.add_argument("--animation-dir", type=Path, default=None,
                        help="Output folder of animate_mesh.py (contains animated_mesh.npz).")
    parser.add_argument("--sample-dir", type=Path, default=None,
                        help="Folder with the clip's *_rgb.png frames; defaults to the joints file's folder.")
    parser.add_argument("--load-mesh", action="store_true",
                        help="Also show *_mesh_animer_global.obj meshes from the joints file's folder.")
    parser.add_argument("--fps", type=float, default=10, help="Playback frame rate.")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.joints_path is None and args.animation_dir is None:
        parser.error("Provide --joints-path, --animation-dir, or both.")
    return args


def edge_points(joints_frame):
    return np.array([[joints_frame[i], joints_frame[j]] for i, j in SMAL_EDGES])


def plot_image(image):
    fig = px.imshow(image)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


def main():
    args = parse_args()
    animation = None
    if args.animation_dir is not None:
        animation = dict(np.load(args.animation_dir / "animated_mesh.npz"))
    if args.joints_path is not None:
        joints = np.load(args.joints_path)
    else:
        joints = animation["joints"]
    if joints.shape[1] == 34:
        joints = np.concatenate([joints[:, [0], :], joints], axis=1)
    if animation is not None:
        joints = joints[:len(animation["vertices"])]
    num_frames, num_joints = joints.shape[:2]

    images = []
    sample_dir = args.sample_dir or (args.joints_path.parent if args.joints_path is not None else None)
    if sample_dir is not None:
        img_paths = sorted(sample_dir.glob("*_rgb.png"))
        if img_paths and len(img_paths) != num_frames:
            print(f"Warning: found {len(img_paths)} images in {sample_dir} but {num_frames} frames of joints")
        images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in img_paths]

    meshes = []
    if args.load_mesh and args.joints_path is not None:
        mesh_files = sorted(args.joints_path.parent.glob("*_mesh_animer_global.obj"))
        if len(mesh_files) == num_frames:
            meshes = [trimesh.load(str(f), process=False) for f in mesh_files]
            print(f"Loaded {len(meshes)} mesh frames.")
        else:
            print(f"Found {len(mesh_files)} meshes for {num_frames} frames; skipping meshes.")

    # Rotate from the Y-up reconstruction frame to viser's Z-up frame.
    rot = tf.SO3.exp(np.array([0, 0, np.pi / 2])) @ tf.SO3.exp(np.array([np.pi / 2, 0, 0]))
    joints = np.stack([rot.apply(frame) for frame in joints])

    server = viser.ViserServer(port=args.port)
    axis_labels = [
        server.scene.add_label(name=f"axis_{name}_label", text=f"+{name.upper()}", position=pos, visible=False)
        for name, pos in (("x", (0.6, 0, 0)), ("y", (0, 0.6, 0)), ("z", (0, 0, 0.6)))
    ]
    joint_handle = server.scene.add_point_cloud(
        name="joints", points=joints[0], colors=np.array([[0.2, 0.7, 1.0]] * num_joints),
        point_size=0.03, point_shape="circle",
    )
    edge_handle = server.scene.add_line_segments(
        name="skeleton_edges", points=edge_points(joints[0]), colors=(0.0, 1.0, 0.0), line_width=5,
    )
    mesh_handle = None
    if meshes:
        mesh_handle = server.scene.add_mesh_simple(
            name="smal_mesh", vertices=np.asarray(meshes[0].vertices, dtype=np.float32),
            faces=np.asarray(meshes[0].faces, dtype=np.int32), color=(0.6, 0.8, 1.0), wxyz=rot.wxyz,
        )
    animated_handle = None
    if animation is not None:
        animated_handle = server.scene.add_mesh_simple(
            name="animated_mesh", vertices=animation["vertices"][0], faces=animation["faces"].astype(np.int32),
            color=(0.8, 0.7, 0.6), wxyz=rot.wxyz,
        )
    joint_labels = [
        server.scene.add_label(name=f"joint_indices/joint_{i}", text=str(i), position=joints[0, i], visible=False)
        for i in range(num_joints)
    ]

    frame_slider = server.gui.add_slider(label="Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    play_button = server.gui.add_checkbox("Playing", True)
    show_indices = server.gui.add_checkbox("Show Joint Indices", False)
    show_axis_labels = server.gui.add_checkbox("Show Axis Labels", False)
    image_plot = server.gui.add_plotly(figure=plot_image(images[0]), aspect=1.0) if images else None

    print("Use the slider or the Playing checkbox in the GUI. Press Ctrl+C to exit.")
    frame, shown_frame, last_update = 0, None, time.time()
    try:
        while True:
            if play_button.value:
                if time.time() - last_update > 1.0 / args.fps:
                    frame = (frame + 1) % num_frames
                    frame_slider.value = frame
                    last_update = time.time()
            else:
                frame = frame_slider.value

            joint_handle.points = joints[frame]
            edge_handle.points = edge_points(joints[frame])
            for i, label in enumerate(joint_labels):
                label.position = joints[frame, i]
                label.visible = show_indices.value
            for label in axis_labels:
                label.visible = show_axis_labels.value
            if mesh_handle is not None:
                mesh_handle.vertices = np.asarray(meshes[frame].vertices, dtype=np.float32)
            if animated_handle is not None:
                animated_handle.vertices = animation["vertices"][frame]
            if image_plot is not None and frame < len(images) and frame != shown_frame:
                image_plot.figure = plot_image(images[frame])
                shown_frame = frame
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting visualization.")


if __name__ == "__main__":
    main()
