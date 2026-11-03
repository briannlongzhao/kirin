"""Fit SMAL parameters to a sequence of target joint positions."""

import os

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from tqdm import tqdm

# Joint indices used to estimate the body frame: leg base joints and feet.
LEFT_REAR_BASE, RIGHT_REAR_BASE, LEFT_FRONT_BASE, RIGHT_FRONT_BASE = 16, 20, 6, 10
LEFT_REAR_FOOT, RIGHT_REAR_FOOT, LEFT_FRONT_FOOT, RIGHT_FRONT_FOOT = 19, 23, 9, 13

# Leg joints whose side bending is regularized; the pose tensor excludes the root joint.
LEG_JOINT_IDS = [6, 7, 8, 10, 11, 12, 16, 17, 18, 20, 21, 22]
LEG_POSE_INDICES = [j - 1 for j in LEG_JOINT_IDS if j > 0]
LEG_SIDE_BEND_WEIGHT = 1
LEG_FLEXION_AXIS = "y"


def compute_leg_side_bending_loss(pose_mats, flexion_axis=LEG_FLEXION_AXIS):
    """Penalize the rotation-matrix entries that are zero for a rotation about flexion_axis only."""
    leg_mats = pose_mats[torch.as_tensor(LEG_POSE_INDICES, device=pose_mats.device, dtype=torch.long)]
    if flexion_axis == "x":
        off_axis = [(0, 1), (0, 2), (1, 0), (2, 0)]
    elif flexion_axis == "y":
        off_axis = [(0, 1), (1, 0), (1, 2), (2, 1)]
    else:
        off_axis = [(0, 2), (1, 2), (2, 0), (2, 1)]
    (a, b), (c, d), (e, f), (g, h) = off_axis
    return (leg_mats[:, a, b]**2 + leg_mats[:, c, d]**2 + leg_mats[:, e, f]**2 + leg_mats[:, g, h]**2).mean()


def compute_body_frame(joints):
    """Body frame (forward, up, right) from the leg joints of a (35, 3) joint array."""
    forward = ((joints[LEFT_FRONT_FOOT] - joints[LEFT_REAR_FOOT])
               + (joints[RIGHT_FRONT_FOOT] - joints[RIGHT_REAR_FOOT])) / 2.0
    forward_norm = np.linalg.norm(forward)
    forward = np.array([1.0, 0.0, 0.0]) if forward_norm < 1e-6 else forward / forward_norm

    # Up is opposite to the mean base-to-foot leg direction.
    avg_leg_direction = (
        (joints[LEFT_REAR_FOOT] - joints[LEFT_REAR_BASE]) + (joints[RIGHT_REAR_FOOT] - joints[RIGHT_REAR_BASE])
        + (joints[LEFT_FRONT_FOOT] - joints[LEFT_FRONT_BASE]) + (joints[RIGHT_FRONT_FOOT] - joints[RIGHT_FRONT_BASE])
    ) / 4.0
    up = -avg_leg_direction
    up_norm = np.linalg.norm(up)
    up = np.array([0.0, 0.0, 1.0]) if up_norm < 1e-6 else up / up_norm

    up = up - np.dot(up, forward) * forward
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        up = up - np.dot(up, forward) * forward
        up = up / np.linalg.norm(up)
    else:
        up = up / up_norm

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    return {"forward": forward, "up": up, "right": right}


def compute_rotation_matrix_from_frames(source_frame, target_frame):
    """Rotation that maps the source body frame onto the target body frame."""
    R_source = np.column_stack([source_frame["right"], source_frame["up"], source_frame["forward"]])
    R_target = np.column_stack([target_frame["right"], target_frame["up"], target_frame["forward"]])
    return R_target @ R_source.T


def _as_tensor(x, device, dtype):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x.to(device=device, dtype=dtype)


def fit_smal_to_joints(target_joints, smal_model, betas_initial=None, num_iterations=100, lr=1e-2,
                       lr_betas=None, lr_pose=None, lr_transl=None, lr_global_orient=None,
                       disabled_joints=None, initial_params=None):
    """
    Fit SMAL parameters to one frame of 35 target joints.

    Without initial_params, the SMAL rest pose is first rotated and translated
    onto the target body frame. With initial_params (the previous frame's
    result), optimization starts from that solution, shifted to the target root.
    """
    device = smal_model.J_regressor.device
    dtype = smal_model.J_regressor.dtype
    target_joints = _as_tensor(target_joints, device, dtype)
    if target_joints.ndim == 2:
        target_joints = target_joints.unsqueeze(0)
    n_joints = target_joints.shape[1]
    if n_joints != 35:
        raise ValueError(f"Expected 35 joints but received {n_joints} joints.")

    if initial_params is not None and initial_params.get("betas") is not None:
        betas_init = _as_tensor(initial_params["betas"], device, dtype)
    elif betas_initial is not None:
        betas_init = _as_tensor(betas_initial, device, dtype)
    else:
        betas_init = torch.zeros(1, smal_model.num_betas, device=device)
    if betas_init.dim() == 1:
        betas_init = betas_init.unsqueeze(0)
    betas = torch.nn.Parameter(betas_init)

    identity_pose = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(1, 34, -1, -1)
    zero_transl = torch.zeros(1, 3, device=device, dtype=dtype)
    alignment_rotation = alignment_translation = None
    if initial_params is None:
        with torch.no_grad():
            initial_smal_joints = smal_model(
                betas=betas, pose=identity_pose, global_orient=torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3),
                transl=zero_transl, pose2rot=False,
            ).joints.squeeze(0).cpu().numpy()
        target_joints_np = target_joints.squeeze(0).cpu().numpy()
        alignment_rotation = compute_rotation_matrix_from_frames(
            compute_body_frame(initial_smal_joints), compute_body_frame(target_joints_np))
        with torch.no_grad():
            aligned_global_orient = torch.from_numpy(alignment_rotation).float().to(device=device, dtype=dtype)[None, None]
            aligned_smal_joints = smal_model(
                betas=betas, pose=identity_pose, global_orient=aligned_global_orient,
                transl=zero_transl, pose2rot=False,
            ).joints.squeeze(0).cpu().numpy()
        alignment_translation = target_joints_np[0] - aligned_smal_joints[0]

    identity_6d = torch.tensor([1., 0., 0., 0., 1., 0.], device=device, dtype=dtype)
    pose_init = identity_6d.expand(34, -1).clone()
    if initial_params is not None and initial_params.get("pose") is not None:
        pose_init = matrix_to_rotation_6d(_as_tensor(initial_params["pose"], device, dtype).squeeze(0)).clone()
    pose = torch.nn.Parameter(pose_init)

    global_orient_init = identity_6d.clone()
    if initial_params is not None and initial_params.get("global_orient") is not None:
        global_orient_mats = _as_tensor(initial_params["global_orient"], device, dtype)
        global_orient_init = matrix_to_rotation_6d(global_orient_mats.view(-1, 3, 3))[0].clone()
    elif alignment_rotation is not None:
        alignment_rotation_tensor = torch.from_numpy(alignment_rotation).float().to(device=device, dtype=dtype)
        global_orient_init = matrix_to_rotation_6d(alignment_rotation_tensor.unsqueeze(0))[0].clone()
    global_orient = torch.nn.Parameter(global_orient_init)

    if initial_params is not None and initial_params.get("transl") is not None:
        transl_tensor = _as_tensor(initial_params["transl"], device, dtype).clone()
    elif alignment_translation is not None:
        transl_tensor = torch.from_numpy(alignment_translation).float().to(device=device, dtype=dtype).unsqueeze(0)
    else:
        transl_tensor = target_joints.mean(dim=1, keepdim=True).squeeze(1).clone()

    # Warm start: move the previous solution so its root matches this frame's root.
    if initial_params is not None:
        with torch.no_grad():
            init_root = smal_model(
                betas=betas,
                pose=rotation_6d_to_matrix(pose_init).unsqueeze(0),
                global_orient=rotation_6d_to_matrix(global_orient_init).unsqueeze(0).unsqueeze(0),
                transl=transl_tensor,
                pose2rot=False,
            ).joints[:, 0, :]
            transl_tensor = transl_tensor + (target_joints[:, 0, :] - init_root)
    transl = torch.nn.Parameter(transl_tensor)

    optimizer = torch.optim.Adam([
        {"params": [betas], "lr": lr_betas if lr_betas is not None else lr},
        {"params": [pose], "lr": lr_pose if lr_pose is not None else lr},
        {"params": [global_orient], "lr": lr_global_orient if lr_global_orient is not None else lr},
        {"params": [transl], "lr": lr_transl if lr_transl is not None else lr},
    ])

    joint_mask = torch.ones(n_joints, dtype=torch.bool, device=device)
    for joint_idx in disabled_joints or []:
        if 0 <= joint_idx < n_joints:
            joint_mask[joint_idx] = False

    for _ in range(num_iterations):
        optimizer.zero_grad()
        pose_mats = rotation_6d_to_matrix(pose)
        smal_output = smal_model(
            betas=betas,
            pose=pose_mats.unsqueeze(0),
            global_orient=rotation_6d_to_matrix(global_orient).unsqueeze(0).unsqueeze(0),
            transl=transl,
            pose2rot=False,
        )
        pred_joints = smal_output.joints
        target_joints_for_loss = target_joints
        if disabled_joints:
            pred_joints = pred_joints[:, joint_mask, :]
            target_joints_for_loss = target_joints[:, joint_mask, :]
        joint_loss = torch.nn.functional.mse_loss(pred_joints, target_joints_for_loss)
        loss = joint_loss + LEG_SIDE_BEND_WEIGHT * compute_leg_side_bending_loss(pose_mats)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_pose_mats = rotation_6d_to_matrix(pose)
        final_output = smal_model(
            betas=betas.to(device),
            pose=final_pose_mats[None].to(device),
            global_orient=rotation_6d_to_matrix(global_orient)[None, None].to(device),
            transl=transl.to(device),
            pose2rot=False,
        )
        params = {
            "betas": betas.detach().cpu(),
            "pose": final_pose_mats[None].detach().cpu(),
            "global_orient": rotation_6d_to_matrix(global_orient)[None, None].detach().cpu(),
            "transl": transl.detach().cpu(),
            "joints": final_output.joints.detach().cpu(),
        }
    return params, joint_loss.item()


def fit_smal_to_joint_sequence(joint_sequence, betas_initial, smal_model, num_iterations=100, lr=1e-3,
                               lr_betas=None, lr_transl=None, lr_pose=None, lr_global_orient=None,
                               output_dir=None, disabled_joints=None):
    """
    Fit SMAL to every frame of a (T, 35, 3) joint sequence. The first frame
    runs 10x num_iterations from the aligned rest pose; each later frame starts
    from the previous frame's solution. Per-frame parameters are saved to
    output_dir/NNNNNNNN_smal_params.pth when output_dir is given.
    """
    if len(joint_sequence) == 0:
        raise ValueError("joint_sequence is empty; expected at least one frame.")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    all_params = []
    prev_params = None
    progress = tqdm(range(len(joint_sequence)), desc="Fitting SMAL to joints")
    for frame_idx in progress:
        frame_params, joint_loss = fit_smal_to_joints(
            target_joints=joint_sequence[frame_idx],
            smal_model=smal_model,
            betas_initial=betas_initial,
            num_iterations=num_iterations if prev_params is not None else num_iterations * 10,
            lr=lr, lr_betas=lr_betas, lr_transl=lr_transl, lr_pose=lr_pose, lr_global_orient=lr_global_orient,
            disabled_joints=disabled_joints,
            initial_params=prev_params,
        )
        progress.set_postfix(joint_loss=f"{joint_loss:.2e}")
        all_params.append(frame_params)
        prev_params = frame_params
        if output_dir is not None:
            torch.save(frame_params, os.path.join(output_dir, f"{frame_idx:08d}_smal_params.pth"))

    return {
        "betas": torch.stack([p["betas"] for p in all_params], dim=0),
        "pose": torch.stack([p["pose"] for p in all_params], dim=0),
        "global_orient": torch.stack([p["global_orient"] for p in all_params], dim=0),
        "transl": torch.stack([p["transl"] for p in all_params], dim=0),
        "frame_ids": list(range(len(all_params))),
    }
