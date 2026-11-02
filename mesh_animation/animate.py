"""Deform a skinned T-pose mesh with per-frame SMAL joint transforms."""

import os

import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.exchange.obj import export_obj
from tqdm import tqdm


def load_tpose_mesh(mesh_path):
    """Load the generated mesh without merging vertices, with its diffuse texture if available."""
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    texture_path = os.path.join(os.path.dirname(str(mesh_path)), "texture_diffuse.png")
    uv = getattr(mesh.visual, "uv", None)
    if uv is not None and os.path.isfile(texture_path):
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(texture_path))
    return mesh


def apply_linear_blend_skinning(rest_vertices, weights, joint_transforms):
    """Deform (V, 3) vertices with (V, J) weights and (J, 4, 4) joint transforms."""
    n_vertices, n_joints = rest_vertices.shape[0], joint_transforms.shape[0]
    T = np.matmul(weights, joint_transforms.reshape(n_joints, 16)).reshape(n_vertices, 4, 4)
    v_homo = np.column_stack([rest_vertices, np.ones((n_vertices, 1))])
    return np.matmul(T, v_homo[:, :, np.newaxis])[:, :3, 0]


@torch.no_grad()
def animate_mesh_from_smal_params(tpose_mesh, skinning_weights, smal_model, smal_params_tpose, smal_params_frames):
    """
    Move the T-pose mesh to the canonical SMAL pose with the inverse T-pose
    joint transforms, then to each frame with that frame's joint transforms.
    Returns a (T, V, 3) vertex array.
    """
    dtype = smal_model.J_regressor.dtype
    device = smal_model.J_regressor.device
    betas = smal_params_tpose["betas"].to(device=device, dtype=dtype)

    tpose_output = smal_model(
        betas=betas,
        pose=smal_params_tpose["pose"].to(device=device, dtype=dtype),
        global_orient=smal_params_tpose["global_orient"].to(device=device, dtype=dtype),
        transl=smal_params_tpose["transl"].to(device=device, dtype=dtype),
        pose2rot=False,
    )
    canonical_to_tpose = tpose_output.joints_transform_matrix.squeeze(0).cpu().numpy()
    tpose_to_canonical = np.stack([np.linalg.inv(transform) for transform in canonical_to_tpose])
    tpose_translation = smal_params_tpose["transl"].squeeze().cpu().numpy()
    canonical_vertices = apply_linear_blend_skinning(
        np.asarray(tpose_mesh.vertices) - tpose_translation, skinning_weights, tpose_to_canonical)

    all_frame_vertices = []
    for i in tqdm(range(len(smal_params_frames["frame_ids"])), desc="Animating mesh"):
        frame_output = smal_model(
            betas=betas,
            pose=smal_params_frames["pose"][i].to(device=device, dtype=dtype),
            global_orient=smal_params_frames["global_orient"][i].to(device=device, dtype=dtype),
            transl=smal_params_frames["transl"][i].to(device=device, dtype=dtype),
            pose2rot=False,
        )
        canonical_to_frame = frame_output.joints_transform_matrix.squeeze(0).cpu().numpy()
        frame_vertices = apply_linear_blend_skinning(canonical_vertices, skinning_weights, canonical_to_frame)
        all_frame_vertices.append(frame_vertices + smal_params_frames["transl"][i].squeeze().cpu().numpy())
    return np.stack(all_frame_vertices)


def export_obj_sequence(tpose_mesh, all_frame_vertices, output_dir):
    """Write NNNNNN_mesh_animated.obj per frame, sharing one material and texture."""
    os.makedirs(output_dir, exist_ok=True)
    texture_written = False
    for frame_idx, vertices in enumerate(tqdm(all_frame_vertices, desc="Writing meshes")):
        frame_mesh = trimesh.Trimesh(vertices=vertices, faces=tpose_mesh.faces, process=False)
        frame_mesh.visual = tpose_mesh.visual
        text, texture = export_obj(frame_mesh, return_texture=True, mtl_name="material.mtl")
        with open(os.path.join(output_dir, f"{frame_idx:06d}_mesh_animated.obj"), "w") as f:
            f.write(text)
        if not texture_written:
            for name, data in texture.items():
                with open(os.path.join(output_dir, name), "wb") as f:
                    f.write(data)
            texture_written = True
