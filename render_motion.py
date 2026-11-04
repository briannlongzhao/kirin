#!/usr/bin/env python3
"""Render a reconstructed joint sequence to a skeleton video and an overlay image.

Usage:
    python render_motion.py --joints-path OUT/joints_global.npy [--fps 10]

Writes motion.mp4, motion_overlay.png and color_palette.png next to the joints
file, or to --output-dir.
"""

import argparse
from pathlib import Path

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from smal.smal import SMAL_PARENTS

SMAL_EDGES = [(i, p) for i, p in enumerate(SMAL_PARENTS) if p != -1]
JOINT_COLOR = np.array([0.2, 0.7, 1.0])
EDGE_COLOR = np.array([0.0, 1.0, 0.0])


def parse_args():
    parser = argparse.ArgumentParser(description="Render a joint sequence to video and overlay image.")
    parser.add_argument("--joints-path", type=Path, required=True, help="joints_global.npy, shape (T, 35, 3).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the joints file's folder.")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate (clips are sampled at 10 fps).")
    parser.add_argument("--resolution", type=int, default=720, help="Video height and width in pixels.")
    parser.add_argument("--num-overlay-frames", type=int, default=4, help="Number of poses in the overlay image.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only render the first N frames.")
    parser.add_argument("--no-ground-to-floor", action="store_true",
                        help="Keep original heights instead of moving the lowest joint to the floor.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer")
    return args


def load_joints(joints_path, max_frames=None):
    """Load joints and convert from Y-up to the Z-up frame used for plotting."""
    joints = np.load(joints_path)
    if max_frames is not None:
        joints = joints[:max_frames]
    if joints.shape[1] == 34:
        joints = np.concatenate([joints[:, [0], :], joints], axis=1)
    rot = Rotation.from_rotvec([0, 0, np.pi / 2]) * Rotation.from_rotvec([np.pi / 2, 0, 0])
    return rot.apply(joints.reshape(-1, 3)).reshape(joints.shape)


def ground_joints_to_floor(joints, floor_z=0.0):
    """Shift all joints so the global minimum Z is at floor_z."""
    joints[:, :, 2] = joints[:, :, 2] - joints[:, :, 2].min() + floor_z
    return joints


def get_axis_limits(joints, margin=0.3):
    """Fixed limits over the whole sequence, with the floor at the lowest joint."""
    all_joints = joints.reshape(-1, 3)
    coord_min = all_joints.min(axis=0) - margin
    coord_max = all_joints.max(axis=0) + margin
    coord_min[2] = all_joints[:, 2].min()
    return [(coord_min[i], coord_max[i]) for i in range(3)]


def get_floor_trajectory(joints, floor_z):
    """Root joint trajectory projected onto the floor."""
    trajectory = joints[:, 0, :].copy()
    trajectory[:, 2] = floor_z
    return trajectory


def new_axes(resolution, axis_limits):
    dpi = 100
    fig = plt.figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=0)
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_zlim(axis_limits[2])
    xlim, ylim, zlim = axis_limits
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 20), np.linspace(ylim[0], ylim[1], 20))
    ax.plot_surface(xx, yy, np.ones_like(xx) * zlim[0], alpha=0.3, color="gray")
    return fig, ax


def draw_skeleton(ax, joints_frame, joint_color, edge_color):
    ax.scatter(joints_frame[:, 0], joints_frame[:, 1], joints_frame[:, 2], c=[joint_color], s=40, alpha=0.8)
    for i, j in SMAL_EDGES:
        points = joints_frame[[i, j]]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=edge_color, linewidth=2.5, alpha=0.9)


def draw_trajectory(ax, trajectory):
    """Draw the floor trajectory with an arrow showing the latest direction."""
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="red", linewidth=2, alpha=0.6)
    if len(trajectory) >= 2:
        direction = trajectory[-1] - trajectory[-2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm * 0.25
            ax.quiver(*trajectory[-1], *direction, color="red", arrow_length_ratio=0.6, linewidth=2.5, alpha=0.8)


def figure_to_image(fig, ax):
    ax.set_axis_off()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def create_video(joints, output_path, fps=10, resolution=720):
    axis_limits = get_axis_limits(joints)
    trajectory = get_floor_trajectory(joints, axis_limits[2][0])
    print(f"Rendering {len(joints)} frames to {output_path}")
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame_idx, joints_frame in enumerate(joints):
            fig, ax = new_axes(resolution, axis_limits)
            draw_skeleton(ax, joints_frame, JOINT_COLOR, EDGE_COLOR)
            draw_trajectory(ax, trajectory[:frame_idx + 1])
            writer.append_data(figure_to_image(fig, ax))


def create_overlay_image(joints, output_path, num_frames=4, resolution=720):
    """Draw evenly sampled poses on one stage, colored from blue (start) to green (end)."""
    axis_limits = get_axis_limits(joints)
    frame_indices = np.linspace(0, len(joints) - 1, min(num_frames, len(joints)), dtype=int)
    colors = [JOINT_COLOR + (EDGE_COLOR - JOINT_COLOR) * t for t in np.linspace(0, 1, len(frame_indices))]
    fig, ax = new_axes(resolution, axis_limits)
    for color, frame_idx in zip(colors, frame_indices):
        draw_skeleton(ax, joints[frame_idx], color, color)
    draw_trajectory(ax, get_floor_trajectory(joints, axis_limits[2][0]))
    cv2.imwrite(str(output_path), cv2.cvtColor(figure_to_image(fig, ax), cv2.COLOR_RGB2BGR))
    print(f"Overlay image saved to {output_path}")


def create_color_palette_image(output_path, resolution=720):
    """Blue-to-green legend for the overlay image, labeled 0 (start) and T (end)."""
    width, height = max(120, resolution // 5), resolution
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)
    gradient = (1.0 - t)[:, None] * JOINT_COLOR[None, :] + t[:, None] * EDGE_COLOR[None, :]
    palette = (np.tile(gradient[:, None, :], (1, width, 1)) * 255.0).clip(0, 255).astype(np.uint8)
    palette = cv2.cvtColor(palette, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.8, resolution / 380.0)
    thickness = max(3, int(round(resolution / 240.0)))
    for text, top in (("0", True), ("T", False)):
        size, _ = cv2.getTextSize(text, font, scale, thickness)
        x = (width - size[0]) // 2
        y = max(18, int(round(resolution * 0.035))) + size[1] if top else height - max(14, int(round(resolution * 0.03)))
        cv2.putText(palette, text, (x, y), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(palette, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.imwrite(str(output_path), palette)
    print(f"Color palette image saved to {output_path}")


def render_motion(joints_path, output_dir=None, fps=10, resolution=720, num_overlay_frames=4,
                  max_frames=None, ground_to_floor=True):
    joints_path = Path(joints_path)
    output_dir = Path(output_dir) if output_dir is not None else joints_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    joints = load_joints(joints_path, max_frames=max_frames)
    if ground_to_floor:
        joints = ground_joints_to_floor(joints)
    create_video(joints, output_dir / "motion.mp4", fps=fps, resolution=resolution)
    create_overlay_image(joints, output_dir / "motion_overlay.png", num_frames=num_overlay_frames, resolution=resolution)
    create_color_palette_image(output_dir / "color_palette.png", resolution=resolution)


def main():
    args = parse_args()
    if not args.joints_path.is_file():
        raise SystemExit(f"--joints-path does not exist: {args.joints_path}")
    render_motion(
        args.joints_path, output_dir=args.output_dir, fps=args.fps, resolution=args.resolution,
        num_overlay_frames=args.num_overlay_frames, max_frames=args.max_frames,
        ground_to_floor=not args.no_ground_to_floor,
    )


if __name__ == "__main__":
    main()
