#!/usr/bin/env python3
"""Animate an image-generated animal mesh with a joint sequence.

Steps: generate a T-pose mesh from the image with Rodin, fit SMAL to the mesh,
transfer SMAL skinning weights, fit SMAL to every frame of the joint sequence,
and deform the mesh with linear blend skinning. Writes per-frame OBJ meshes,
animated_mesh.npz and a rendered video.
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from mesh_animation.animate import animate_mesh_from_smal_params, export_obj_sequence, load_tpose_mesh
from mesh_animation.fit_smal_to_joints import fit_smal_to_joint_sequence
from mesh_animation.fit_smal_to_mesh import fit_smal_to_mesh
from mesh_animation.rodin import apply_mask, generate_mesh
from mesh_animation.skinning import skinning
from smal.smal import SMAL_PARENTS, load_smal_model

# Tail joints after the first tail joint are not driven by the motion.
DISABLED_JOINTS = [26, 27, 28, 29, 30, 31]


def parse_args():
    parser = argparse.ArgumentParser(description="Animate a mesh generated from an image with a joint sequence.")
    parser.add_argument("--joints-path", type=Path, required=True,
                        help="Joint sequence .npy of shape (T, 35, 3) or (T, 34, 3), Y up.")
    parser.add_argument("--image-path", type=Path, default=None,
                        help="Image of the animal used to generate the T-pose mesh.")
    parser.add_argument("--mask-path", type=Path, default=None,
                        help="Optional animal mask; the background is made transparent before mesh generation.")
    parser.add_argument("--tpose-mesh", type=Path, default=None,
                        help="Existing Rodin base.obj to use instead of generating one; its folder is copied.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <joints folder>/animated_mesh.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only animate the first N frames.")
    parser.add_argument("--fit-num-iterations", type=int, default=150,
                        help="SMAL-to-joints iterations per frame (10x for the first frame).")
    parser.add_argument("--fit-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10, help="Frame rate of the rendered video.")
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--no-render", action="store_true", help="Skip rendering animation.mp4.")
    args = parser.parse_args()
    if args.image_path is None and args.tpose_mesh is None:
        parser.error("Provide --image-path to generate a T-pose mesh, or --tpose-mesh to reuse one.")
    for flag, path in (("--joints-path", args.joints_path), ("--image-path", args.image_path),
                       ("--mask-path", args.mask_path), ("--tpose-mesh", args.tpose_mesh)):
        if path is not None and not path.is_file():
            parser.error(f"{flag} does not exist: {path}")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    return args


def load_joints(joints_path, max_frames=None):
    joints = np.load(joints_path)
    if joints.ndim != 3 or joints.shape[-1] != 3 or joints.shape[1] not in (34, 35):
        raise ValueError(f"Expected joints of shape (T, 35, 3) or (T, 34, 3), got {joints.shape}")
    if max_frames is not None:
        joints = joints[:max_frames]
    if joints.shape[1] == 34:
        joints = np.concatenate([joints[:, [0], :], joints], axis=1)
    return joints


def prepare_tpose_mesh(args, tpose_dir):
    """Return the T-pose mesh path in tpose_dir, generating or copying it if needed."""
    mesh_path = tpose_dir / "base.obj"
    if mesh_path.is_file():
        print(f"Using existing T-pose mesh: {mesh_path}")
        return mesh_path
    tpose_dir.mkdir(parents=True, exist_ok=True)
    if args.tpose_mesh is not None:
        for f in args.tpose_mesh.parent.iterdir():
            if f.is_file() and f.name != "smal_params.pth":
                shutil.copy2(f, tpose_dir / f.name)
        if args.tpose_mesh.name != "base.obj":
            shutil.copy2(args.tpose_mesh, mesh_path)
        print(f"Copied T-pose mesh from {args.tpose_mesh.parent}")
        return mesh_path
    image_path = args.image_path
    if args.mask_path is not None:
        image_path = tpose_dir / "input_rgba.png"
        apply_mask(args.image_path, args.mask_path, image_path)
    print(f"Generating T-pose mesh with Rodin from {image_path}")
    generate_mesh(str(image_path), str(tpose_dir))
    return mesh_path


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir or args.joints_path.parent / "animated_mesh"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smal_model = load_smal_model().to(device)

    joints = load_joints(args.joints_path, args.max_frames)
    print(f"Loaded {len(joints)} frames from {args.joints_path}")

    tpose_dir = output_dir / "tpose_mesh"
    tpose_mesh = load_tpose_mesh(prepare_tpose_mesh(args, tpose_dir))
    print(f"T-pose mesh: {len(tpose_mesh.vertices)} vertices, {len(tpose_mesh.faces)} faces")

    tpose_params_path = tpose_dir / "smal_params.pth"
    if tpose_params_path.is_file():
        print(f"Using existing SMAL fit: {tpose_params_path}")
        tpose_params = torch.load(tpose_params_path)
    else:
        tpose_params = fit_smal_to_mesh(tpose_mesh, smal_model, smal_params_path=str(tpose_params_path),
                                        lr=1e-2, lr_pose=1e-4)

    weights = skinning(tpose_mesh, smal_model, tpose_params, SMAL_PARENTS, disabled_joints=DISABLED_JOINTS)

    frame_params = fit_smal_to_joint_sequence(
        joints, tpose_params["betas"], smal_model, num_iterations=args.fit_num_iterations, lr=args.fit_lr,
        output_dir=str(output_dir / "smal_fit_params"), disabled_joints=DISABLED_JOINTS,
    )
    vertices = animate_mesh_from_smal_params(tpose_mesh, weights, smal_model, tpose_params, frame_params)

    export_obj_sequence(tpose_mesh, vertices, output_dir / "meshes")
    np.savez_compressed(output_dir / "animated_mesh.npz", vertices=vertices.astype(np.float32),
                        faces=np.asarray(tpose_mesh.faces), joints=joints.astype(np.float32))
    print(f"Saved {len(vertices)} meshes to {output_dir / 'meshes'} and {output_dir / 'animated_mesh.npz'}")

    if not args.no_render:
        from mesh_animation.render import render_animation
        render_animation(tpose_mesh, vertices, output_dir / "animation.mp4", fps=args.fps, resolution=args.resolution)
        print(f"Saved {output_dir / 'animation.mp4'}")


if __name__ == "__main__":
    main()
