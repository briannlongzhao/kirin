import os
import cv2
import json
import torch
import imageio
import trimesh
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from einops import repeat
from random import shuffle

from AniMer.amr.utils import recursive_to
from AniMer.amr.utils.renderer import Renderer, cam_crop_to_full
from AniMer.amr.utils.geometry import perspective_projection
from AniMer.amr.datasets.vitdet_dataset import ViTDetDataset
from smal_models.custom_amr import load_amr
from smal_models.custom_smal import SMAL_TO_VIT_POSE
from utils import get_all_sequence_dirs, draw_keypoints

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='AMR demo code')
    parser.add_argument(
        '--checkpoint_path', type=str,help='Path to pretrained model checkpoint',
        required=True
    )
    parser.add_argument(
        '--data_dir', type=str, help='Dataset root (batch mode)',
        default=None
    )
    parser.add_argument(
        '--sequence_dir', type=str, help='Process this one clip directory',
        default=None
    )
    parser.add_argument(
        '--out_folder', type=str, help='Output folder to save rendered results',
        default=None
    )
    parser.add_argument(
        '--save_mesh', action='store_true',
        help='If set, save meshes to disk also'
        )
    parser.add_argument(
        "--save_video", action='store_true',
        help='If set, save videos to disk also'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, 
        help='Batch size for inference/fitting'
    )
    parser.add_argument(
        '--num_iters', type=int, default=0,
        help='Number of optimization iterations'
    )
    parser.add_argument(
        "--center_keypoint", type=int, default=-1,
        help='Index of the keypoint to center the pose around, -1 to disable'
    )
    parser.add_argument(
        "--global_motion", action="store_true",
        help="If set, apply global motion from tracker result"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help='Learning rate for all parameters (beta, pose, camera, translation)'
    )
    parser.add_argument(
        "--cam_anchor_weight", type=float, default=0.0,
        help='Weight for camera anchor loss (keeps camera close to initial values)'
    )
    parser.add_argument(
        "--proj_loss_weight", type=float, default=1.0,
        help='Weight for the 2D projection loss'
    )
    parser.add_argument(
        "--smooth_loss_weight", type=float, default=0.1,
        help='Weight for the smoothness loss'
    )
    parser.add_argument(
        "--restrict_side_bending", action="store_true",
        help='If set, restrict body side-bending (roll) during optimization'
    )
    parser.add_argument(
        "--restrict_twist", action="store_true",
        help='If set, restrict leg twist (yaw and roll) during optimization'
    )
    parser.add_argument(
        "--uncropped_results", action="store_true",
        help='If set, use uncropped results from tracker'
    )
    parser.add_argument(
        "--tracker_dir", type=str, default=None,
        help='Folder containing tracker_results*.npz; defaults to the output folder'
    )
    args = parser.parse_args()
    if bool(args.data_dir) == bool(args.sequence_dir):
        parser.error("Provide exactly one of --data_dir or --sequence_dir.")
    return args


def so3_geodesic_sq(R1, R2, eps=1e-6):
    # R_rel: (..., 3, 3)
    R_rel = torch.matmul(R1.transpose(-1, -2), R2)
    # cos(theta) = (trace(R) - 1) / 2
    cos = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)) - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    theta = torch.arccos(cos)
    return theta ** 2

def rotation_smoothness_loss(rot6d_seq, reduce='mean'):
    """
    rot6d_seq: shape [T, ..., 6]
    """
    R = rotation_6d_to_matrix(rot6d_seq)          # [T, ..., 3, 3]
    loss_t = so3_geodesic_sq(R[:-1], R[1:])  # [T-1, ...]
    if reduce == 'mean':
        return loss_t.mean()
    elif reduce == 'sum':
        return loss_t.sum()
    else:
        return loss_t

def rotation_accel_smoothness_loss(rot6d_seq, reduce='mean'):
    R = rotation_6d_to_matrix(rot6d_seq)                   # [T, ..., 3, 3]
    dR_t = torch.matmul(R[:-1].transpose(-1, -2), R[1:])   # [T-1, ..., 3, 3]
    loss_t = so3_geodesic_sq(dR_t[:-1], dR_t[1:])          # [T-2, ...]
    if reduce == 'mean':
        return loss_t.mean()
    elif reduce == 'sum':
        return loss_t.sum()
    else:
        return loss_t

def normalize_translation(world_joints, global_translations):
    """
    Normalize global translations so that:
    1) Root joint at frame 0 is centered horizontally (X=0, Z=0)
    2) Each frame is lifted so its lowest joint is on the ground plane (y=0)
    
    Note: After all transformations, world coordinates are in SMAL convention:
    X: right, Y: up (vertical), Z: backward
    
    Args:
        world_joints: np.ndarray of shape [T, num_joints, 3] - joints in world frame
        global_translations: np.ndarray of shape [T, 3] - global translation for each frame
    
    Returns:
        normalized_translations: np.ndarray of shape [T, 3] - adjusted translations
    """
    # 1) Center root joint horizontally (X and Z axes) at frame 0
    root_position_frame0 = world_joints[0, 0, :]  # Assuming joint 0 is root
    # Only shift X and Z to zero, keep Y as-is for now
    horizontal_offset = np.array([-root_position_frame0[0], 0, -root_position_frame0[2]])
    
    # Apply horizontal offset to all translations
    adjusted_translations = global_translations + horizontal_offset
    
    # Recompute world joints with adjusted translations
    adjusted_world_joints = world_joints + horizontal_offset[np.newaxis, np.newaxis, :]
    
    # 2) Lift each frame individually so its lowest joint is on ground plane (y=0)
    # In SMAL world convention, Y is the vertical axis (up)
    y_lifts = []
    for t in range(len(adjusted_translations)):
        min_y_frame = adjusted_world_joints[t, :, 1].min()  # Find lowest Y in this frame
        y_lift = -min_y_frame  # Amount to lift this frame to bring lowest point to y=0
        adjusted_translations[t, 1] += y_lift  # Apply y-lift to this frame's translation
        adjusted_world_joints[t, :, 1] += y_lift  # Also update world joints for correct debug output
        y_lifts.append(y_lift)
    
    print(f"Normalization applied:")
    print(f"  - Horizontal offset (X, Z): ({horizontal_offset[0]:.3f}, {horizontal_offset[2]:.3f})")
    print(f"  - Root at frame 0 after full normalization: X={adjusted_world_joints[0, 0, 0]:.3f}, Y={adjusted_world_joints[0, 0, 1]:.3f}, Z={adjusted_world_joints[0, 0, 2]:.3f}")
    print(f"  - Per-frame Y-lifts (first 5): {y_lifts[:5]}")
    print(f"  - Frame 0: lowest joint Y after lift = {adjusted_world_joints[0, :, 1].min():.3f} (should be ~0)")
    
    return adjusted_translations

def smooth_translation(translations, mode='gaussian', window_size=5, sigma=1.0):
    """
    Smooth global translation trajectories over time.
    
    Args:
        translations (np.ndarray or torch.Tensor): Translation vectors of shape [T, 3]
        mode (str): Smoothing method:
            - 'gaussian': Gaussian kernel smoothing
            - 'moving_average': Uniform window averaging
            - 'median': Median filter (best for outliers)
            - 'two_stage': Median filter + Gaussian (recommended for outliers)
            - 'none': No smoothing
        window_size (int): Size of the smoothing window (must be odd)
        sigma (float): Standard deviation for Gaussian smoothing
    
    Returns:
        np.ndarray or torch.Tensor: Smoothed translations with same shape and type as input
    """
    if mode == 'none':
        return translations
    
    is_torch = isinstance(translations, torch.Tensor)
    device = translations.device if is_torch else None
    dtype = translations.dtype if is_torch else None
    
    # Convert to numpy for processing
    if is_torch:
        trans_np = translations.detach().cpu().numpy()
    else:
        trans_np = translations.copy()
    
    T = trans_np.shape[0]
    smoothed = np.zeros_like(trans_np)
    
    if mode == 'gaussian':
        # Create Gaussian kernel
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        half_window = window_size // 2
        
        # Generate Gaussian weights
        x = np.arange(-half_window, half_window + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Apply convolution for each dimension
        for dim in range(3):
            # Pad with edge values
            padded = np.pad(trans_np[:, dim], half_window, mode='edge')
            # Convolve
            smoothed[:, dim] = np.convolve(padded, kernel, mode='valid')
    
    elif mode == 'moving_average':
        # Simple uniform moving average
        half_window = window_size // 2
        
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            smoothed[t] = trans_np[start:end].mean(axis=0)
    
    elif mode == 'median':
        # Median filter - excellent for removing outliers
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        half_window = window_size // 2
        
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            # Take median across the window for each dimension
            smoothed[t] = np.median(trans_np[start:end], axis=0)
    
    elif mode == 'two_stage':
        # Two-stage: Median filter to remove outliers, then Gaussian to smooth
        # Stage 1: Median filter with smaller window
        median_window = max(3, window_size // 2)
        if median_window % 2 == 0:
            median_window += 1
        half_median = median_window // 2
        
        stage1 = np.zeros_like(trans_np)
        for t in range(T):
            start = max(0, t - half_median)
            end = min(T, t + half_median + 1)
            stage1[t] = np.median(trans_np[start:end], axis=0)
        
        # Stage 2: Gaussian smoothing
        if window_size % 2 == 0:
            window_size += 1
        half_window = window_size // 2
        
        x = np.arange(-half_window, half_window + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        for dim in range(3):
            padded = np.pad(stage1[:, dim], half_window, mode='edge')
            smoothed[:, dim] = np.convolve(padded, kernel, mode='valid')
    
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}. Use 'gaussian', 'moving_average', 'median', 'two_stage', or 'none'")
    
    # Convert back to original type
    if is_torch:
        return torch.from_numpy(smoothed).to(device=device, dtype=dtype)
    else:
        return smoothed

def optimize_parameters(
    smal_model, smal_cfg, betas, poses, cams, kp2ds, img_size, masks,
    num_iters, lr, proj_loss_weight, smooth_loss_weight, cam_anchor_weight,
    restrict_side_bending=False, restrict_twist=False, image_paths=None, images=None, renderer=None, device=None, save_video=False,
    out_folder=None
):
    """
    Optimize SMAL parameters.
    
    Args:
        restrict_side_bending: If True, keep body side-bending (roll) close to initial prediction.
        restrict_twist: If True, lock leg twist (yaw and roll) to initial values, allow only pitch.
    """
    num_frames = betas.shape[0]
    assert poses.shape[0] == cams.shape[0] == kp2ds.shape[0]
    mean_beta = betas.mean(dim=0, keepdim=True).clone().detach().requires_grad_(True)
    initial_pose_mats = poses.clone().detach()
    poses = matrix_to_rotation_6d(poses).clone().detach().requires_grad_(True)
    initial_cams = cams.clone().detach()  # Store initial camera values for anchor loss
    cams = cams.clone().detach().requires_grad_(True)  # Make camera parameters learnable
    transl = torch.zeros(num_frames, 3, device=cams.device).requires_grad_(True)  # Per-frame translation
    optimizer = torch.optim.Adam([
        {'params': mean_beta, 'lr': lr},
        {'params': poses, 'lr': lr},
        {'params': cams, 'lr': lr},
        {'params': transl, 'lr': lr}
    ])
    optimize_progress_frames = []
    for iter in tqdm(range(num_iters), desc="Optimizing"):
        optimizer.zero_grad()
        total_loss = 0.0
        smoothness_loss = 0.0
        proj_loss = 0.0

        for i in range(num_frames):
            # Forward pass through SMAL model
            smal_out = smal_model.smal(
                betas=mean_beta,
                global_orient=rotation_6d_to_matrix(poses[i, 0:1])[None],
                pose=rotation_6d_to_matrix(poses[i, 1:])[None],
                transl=transl[i][None],
                pose2rot=False,
            )
            verts = smal_out.vertices
            kp3d = smal_out.joints
            kp3d = torch.cat([
                kp3d,
                verts[:, None, 1863], # end_of_nose
                verts[:, None, 26], # chin
                verts[:, None, 2124], # right ear tip
                verts[:, None, 150], # left ear tip
                verts[:, None, 1097], # left eye
                verts[:, None, 3055], # right eye
                ], dim = 1) 
            # Project 3D keypoints to 2D
            focal_length = torch.tensor([smal_cfg.EXTRA.FOCAL_LENGTH, smal_cfg.EXTRA.FOCAL_LENGTH], dtype=torch.float32)
            focal_length = focal_length / smal_cfg.MODEL.IMAGE_SIZE

            # Simple perspective projection
            kp2d_proj = perspective_projection(kp3d, cams[i][None], focal_length[None])
            kp2d_proj = (kp2d_proj + 0.5) * img_size

            # 2D projection loss
            gt_kp2d = kp2ds[i][None]    

            # Build index arrays for mapping
            smal_indices = list(SMAL_TO_VIT_POSE.keys())
            vitpose_indices = [SMAL_TO_VIT_POSE[idx] for idx in smal_indices]

            # Select corresponding keypoints from kp2d_proj
            kp2d_proj_mapped = kp2d_proj[:, smal_indices]
            gt_kp2d_mapped = gt_kp2d[:, vitpose_indices]
            gt_kp2d_coords = gt_kp2d_mapped[..., :2]
            gt_kp2d_conf = gt_kp2d_mapped[..., 2:3]

            gt_kp2d_mask = (
                (gt_kp2d_coords[..., 0] >= 0) & (gt_kp2d_coords[..., 0] < img_size) &
                (gt_kp2d_coords[..., 1] >= 0) & (gt_kp2d_coords[..., 1] < img_size)
            ).float().unsqueeze(-1)
            if masks[i] is not None:
                mask_tensor = torch.tensor(masks[i], device=gt_kp2d_coords.device)
                x_coords = gt_kp2d_coords[..., 0].long().clamp(0, img_size - 1)
                y_coords = gt_kp2d_coords[..., 1].long().clamp(0, img_size - 1)
                visibility_mask = mask_tensor[y_coords, x_coords].unsqueeze(-1)
                gt_kp2d_mask = gt_kp2d_mask * visibility_mask
            # Clamp both projected and GT coords to image range
            # kp2d_proj_mapped = kp2d_proj_mapped.clamp(0, img_size - 1)
            # gt_kp2d_coords = gt_kp2d_coords.clamp(0, img_size - 1)

            # Masked, confidence-weighted L1 loss
            weights = gt_kp2d_conf * gt_kp2d_mask
            diff = torch.abs(kp2d_proj_mapped - gt_kp2d_coords) * weights
            proj_loss += diff.sum() / (weights.sum() + 1e-6)

            # Save progress frame (first frame only)
            if save_video and i == 0 and images is not None and renderer is not None:
                with torch.no_grad():
                    verts_np = smal_out.vertices[0].detach().cpu().numpy()
                    if isinstance(cams[i], np.ndarray):
                        cam_translation = cams[i]
                    else:
                        cam_translation = cams[i].detach().cpu().numpy()
                    progress_img = renderer(
                        vertices=verts_np + cam_translation,
                        camera_translation=np.zeros_like(cam_translation),
                        image=images[i],
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                    )
                    kp3d_progress = smal_out.joints
                    kp3d_progress = torch.cat([
                        kp3d_progress,
                        smal_out.vertices[:, None, 1863],
                        smal_out.vertices[:, None, 26],
                        smal_out.vertices[:, None, 2124],
                        smal_out.vertices[:, None, 150],
                        smal_out.vertices[:, None, 1097],
                        smal_out.vertices[:, None, 3055],
                    ], dim=1)
                    # Add translation offset to keypoints to match mesh rendering
                    kp3d_progress = kp3d_progress + transl[i:i+1].unsqueeze(1)
                    focal_length = torch.tensor(
                        [smal_cfg.EXTRA.FOCAL_LENGTH, smal_cfg.EXTRA.FOCAL_LENGTH],
                        dtype=torch.float32,
                        device=kp3d_progress.device,
                    )
                    focal_length = focal_length / smal_cfg.MODEL.IMAGE_SIZE
                    kp2d_progress = perspective_projection(
                        kp3d_progress, translation=cams[i][None], focal_length=focal_length[None]
                    )
                    kp2d_progress_np = kp2d_progress[0].detach().cpu().numpy()
                    kp2d_progress_mapped = kp2d_progress_np[list(SMAL_TO_VIT_POSE.keys())]
                    progress_img = (progress_img[:, :, ::-1] * 255).astype(np.uint8)
                    progress_img = draw_keypoints(progress_img, kp2d_progress_mapped)
                    gt_kp2d = kp2ds[i][:, :-1].detach().cpu().numpy()
                    image_size = int(img_size)
                    gt_kp2d_norm = gt_kp2d / image_size - 0.5
                    progress_img = draw_keypoints(progress_img, gt_kp2d_norm, circle_color=(0, 255, 0))
                    optimize_progress_frames.append(progress_img)

            # Smoothness loss (temporal difference)
            # if i > 0:
            #     smoothness_loss += torch.nn.functional.mse_loss(poses[i], poses[i-1])

        proj_loss = proj_loss / num_frames
        # smoothness_loss = smoothness_loss / (num_frames - 1)
        smoothness_loss = rotation_smoothness_loss(poses) + 0.2 * rotation_accel_smoothness_loss(poses)
        
        # Camera anchor loss - keep cameras close to initial values
        cam_anchor_loss = ((cams - initial_cams) ** 2).mean()
        
        # Translation smoothness loss
        transl_smooth_loss = ((transl[1:] - transl[:-1]) ** 2).mean()
        
        # Compute total loss
        total_loss = proj_loss_weight * proj_loss + smooth_loss_weight * (smoothness_loss + transl_smooth_loss) + cam_anchor_weight * cam_anchor_loss
        
        total_loss.backward()
        optimizer.step()
        
        if iter % 10 == 0:
            print(f'Iter: {iter} Loss: {total_loss.item():.4f} Proj: {proj_loss.item():.4f} Smooth: {smoothness_loss.item():.4f} TranslSmooth: {transl_smooth_loss.item():.4f} CamAnchor: {cam_anchor_loss.item():.4f}')
        
        # Apply rotation restrictions
        if restrict_side_bending or restrict_twist:
            with torch.no_grad():
                # Body joints (exclude global)
                body_poses_mat = rotation_6d_to_matrix(poses[:, 1:])
                initial_body_poses_mat = initial_pose_mats[:, 1:].to(body_poses_mat.device)
                restricted_mats = body_poses_mat.clone()

                # Leg joint indices in SMAL (0-indexed after excluding global joint)
                # SMAL joints: 8, 9, 10, 12, 13, 14, 18, 19, 20, 22, 23, 24
                # After excluding global (joint 0), they become: 7, 8, 9, 11, 12, 13, 17, 18, 19, 21, 22, 23
                leg_joint_indices = [7, 8, 9, 11, 12, 13, 17, 18, 19, 21, 22, 23]
                leg_joint_indices = [idx for idx in leg_joint_indices if idx < body_poses_mat.shape[1]]

                # Extract Euler angles from rotation matrix (ZYX convention)
                # R = Rz(yaw) * Ry(pitch) * Rx(roll)
                pitch = torch.atan2(
                    body_poses_mat[..., 0, 2],
                    torch.sqrt(body_poses_mat[..., 0, 0]**2 + body_poses_mat[..., 0, 1]**2),
                )
                yaw = torch.atan2(-body_poses_mat[..., 0, 1], body_poses_mat[..., 0, 0])
                roll_init = torch.atan2(
                    -initial_body_poses_mat[..., 1, 2],
                    initial_body_poses_mat[..., 2, 2],
                )
                yaw_init = torch.atan2(-initial_body_poses_mat[..., 0, 1], initial_body_poses_mat[..., 0, 0])

                # Restrict side-bending (roll): lock roll to initial value for ALL joints
                if restrict_side_bending:
                    cos_pitch = torch.cos(pitch)
                    sin_pitch = torch.sin(pitch)
                    cos_yaw = torch.cos(yaw)
                    sin_yaw = torch.sin(yaw)
                    cos_roll = torch.cos(roll_init)
                    sin_roll = torch.sin(roll_init)

                    # Reconstruct with initial roll, but allow pitch and yaw for all joints
                    all_body_mats = torch.zeros_like(restricted_mats)
                    all_body_mats[..., 0, 0] = cos_pitch * cos_yaw
                    all_body_mats[..., 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
                    all_body_mats[..., 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
                    all_body_mats[..., 1, 0] = cos_pitch * sin_yaw
                    all_body_mats[..., 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
                    all_body_mats[..., 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
                    all_body_mats[..., 2, 0] = -sin_pitch
                    all_body_mats[..., 2, 1] = cos_pitch * sin_roll
                    all_body_mats[..., 2, 2] = cos_pitch * cos_roll
                    restricted_mats = all_body_mats

                # Restrict twist (yaw and roll): allow only pitch for LEG joints only
                if restrict_twist and leg_joint_indices:
                    leg_pitch = pitch[:, leg_joint_indices]
                    leg_cos_pitch = torch.cos(leg_pitch)
                    leg_sin_pitch = torch.sin(leg_pitch)
                    leg_cos_yaw = torch.cos(yaw_init[:, leg_joint_indices])
                    leg_sin_yaw = torch.sin(yaw_init[:, leg_joint_indices])
                    leg_cos_roll = torch.cos(roll_init[:, leg_joint_indices])
                    leg_sin_roll = torch.sin(roll_init[:, leg_joint_indices])

                    # Reconstruct with initial yaw and roll, allow only pitch for leg joints
                    leg_mats = torch.zeros_like(restricted_mats[:, leg_joint_indices])
                    leg_mats[..., 0, 0] = leg_cos_pitch * leg_cos_yaw
                    leg_mats[..., 0, 1] = leg_cos_yaw * leg_sin_pitch * leg_sin_roll - leg_sin_yaw * leg_cos_roll
                    leg_mats[..., 0, 2] = leg_cos_yaw * leg_sin_pitch * leg_cos_roll + leg_sin_yaw * leg_sin_roll
                    leg_mats[..., 1, 0] = leg_cos_pitch * leg_sin_yaw
                    leg_mats[..., 1, 1] = leg_sin_yaw * leg_sin_pitch * leg_sin_roll + leg_cos_yaw * leg_cos_roll
                    leg_mats[..., 1, 2] = leg_sin_yaw * leg_sin_pitch * leg_cos_roll - leg_cos_yaw * leg_sin_roll
                    leg_mats[..., 2, 0] = -leg_sin_pitch
                    leg_mats[..., 2, 1] = leg_cos_pitch * leg_sin_roll
                    leg_mats[..., 2, 2] = leg_cos_pitch * leg_cos_roll
                    restricted_mats[:, leg_joint_indices] = leg_mats

                poses[:, 1:] = matrix_to_rotation_6d(restricted_mats)
    
    print("Optimization finished.")
    poses = rotation_6d_to_matrix(poses)
    
    # Return optimized translation along with beta and poses
    optimized_transl = transl.detach()
    
    # Save videos inside optimize function
    if save_video and image_paths is not None and images is not None and renderer is not None:
        optimized_images = []
        rotation_matrix = torch.from_numpy(
            trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
        ).to(device=poses.device, dtype=poses.dtype)
        
        for i, image_path in enumerate(image_paths):
            # Use global_orient without 180 rotation for video display (used as-is)
            smal_output_video = smal_model.smal(
                betas=mean_beta,
                global_orient=poses[i:i+1, 0:1],  # [1, 1, 3, 3]
                pose=poses[i:i+1, 1:],  # [1, num_joints-1, 3, 3]
                transl=transl[i:i+1],  # Use optimized per-frame translation
                pose2rot=False,
            )
            final_verts_video = smal_output_video['vertices'][0].detach().cpu().numpy()
            # Use vertex translation and zero camera translation (matches previous working code)
            if isinstance(cams[i], np.ndarray):
                cam_translation = cams[i]
            else:
                cam_translation = cams[i].detach().cpu().numpy()
            final_img = renderer(
                vertices=final_verts_video + cam_translation,
                camera_translation=np.zeros_like(cam_translation),
                image=images[i],
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            # Project 3D keypoints to 2D for overlay
            kp3d = smal_output_video.joints
            kp3d = torch.cat([
                kp3d,
                smal_output_video.vertices[:, None, 1863], # end_of_nose
                smal_output_video.vertices[:, None, 26],   # chin
                smal_output_video.vertices[:, None, 2124], # right ear tip
                smal_output_video.vertices[:, None, 150],  # left ear tip
                smal_output_video.vertices[:, None, 1097], # left eye
                smal_output_video.vertices[:, None, 3055], # right eye
            ], dim=1)
            # Add translation offset to keypoints to match mesh rendering
            kp3d = kp3d + transl[i:i+1].unsqueeze(1)
            focal_length = torch.tensor([smal_cfg.EXTRA.FOCAL_LENGTH, smal_cfg.EXTRA.FOCAL_LENGTH], dtype=torch.float32).to(device)
            focal_length = focal_length / smal_cfg.MODEL.IMAGE_SIZE

            # Project keypoints using the original camera translation tensor
            kp2d_proj = perspective_projection(
                kp3d, translation=cams[i][None], focal_length=focal_length[None]
            )

            kp2d_proj_np = kp2d_proj[0].detach().cpu().numpy()
            kp2d_proj_mapped = kp2d_proj_np[list(SMAL_TO_VIT_POSE.keys())]

            final_img = (final_img[:, :, ::-1] * 255).astype(np.uint8)
            
            # Draw keypoints using draw_keypoints in normalized space
            final_img = draw_keypoints(final_img, kp2d_proj_mapped)
            gt_kp2d = kp2ds[i][:, :-1].detach().cpu().numpy()
            image_size = int(img_size)
            gt_kp2d_norm = gt_kp2d / image_size - 0.5
            final_img = draw_keypoints(final_img, gt_kp2d_norm, circle_color=(0, 255, 0))
            
            optimized_images.append(final_img)
        
        # Save optimized overlay video
        if len(optimized_images) > 0:
            with imageio.get_writer(os.path.join(out_folder, 'rgb_optimized_overlayed.mp4'), fps=10) as writer:
                for frame in optimized_images:
                    writer.append_data(frame)
            print(f"Saved optimized overlay video to {os.path.join(out_folder, 'rgb_optimized_overlayed.mp4')}")
        
        # Save optimization progress video
        if len(optimize_progress_frames) > 0:
            with imageio.get_writer(os.path.join(out_folder, 'rgb_optimized_progress.mp4'), fps=10) as writer:
                for frame in optimize_progress_frames:
                    writer.append_data(frame)
            print(f"Saved optimization progress video to {os.path.join(out_folder, 'rgb_optimized_progress.mp4')}")
    
    return mean_beta, poses, optimized_transl


def compute_global_motion_uncropped(image_paths, cams, device, smal_model, betas, poses, tracker_results_path):
    """
    Compute global motion from tracked points in uncropped video.
    
    Args:
        image_paths: List of image file paths
        cams: Tensor of camera offsets [T, 3] from AMR predictions (not used in this mode)
        device: torch device
        smal_model: SMAL model to compute 3D joints
        betas: Tensor [T, num_betas] shape parameters
        poses: Tensor [T, num_joints, 3, 3] pose matrices
    
    Returns:
        dict with keys:
            - global_transforms: Tensor [T, 4, 4] identity transforms (no camera rotation)
            - R_cam0: Tensor [3, 3] identity rotation
            - translations: np.ndarray [T, 3] animal translations from tracked points
            - rotations: list of [3, 3] identity rotation matrices
            - use_uncropped: True (flag for downstream processing)
    """
    sequence_dir = os.path.dirname(image_paths[0])
    
    # Load uncropped tracker results
    assert os.path.exists(tracker_results_path), f"Uncropped tracker results not found at {tracker_results_path}"
    tracker_results = np.load(tracker_results_path)
    
    # Extract 3D object-point coords (T, N, 3)
    # IMPORTANT: coords are already in METERS (metric units), not pixels!
    # They are world coordinates computed from depth (in meters) and camera intrinsics
    coords = tracker_results['coords_object']  # Shape: [T, N, 3], units: METERS
    intrinsics = tracker_results['intrinsics']  # Shape: [T, 3, 3]
    video_frame_size = tracker_results['video_frame_size']  # (h, w) of RESIZED video used for tracking
    h_resized, w_resized = video_frame_size
    
    # Load metadata to get crop dimensions in ORIGINAL video resolution
    metadata_files = sorted(glob(os.path.join(sequence_dir, '*metadata.json')))
    assert len(metadata_files) > 0, f"No metadata file found in {sequence_dir}"
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    crop_box_xyxy = metadata['crop_box_xyxy']  # [x1, y1, x2, y2] in ORIGINAL video resolution
    video_frame_height = metadata['video_frame_height']  # Original video height
    video_frame_width = metadata['video_frame_width']    # Original video width
    
    x1, y1, x2, y2 = crop_box_xyxy
    crop_width_original = x2 - x1   # Width in pixels (ORIGINAL video scale)
    crop_height_original = y2 - y1  # Height in pixels (ORIGINAL video scale)
    
    # Scale crop dimensions to RESIZED video coordinates (to match intrinsics)
    scale_h = h_resized / video_frame_height
    scale_w = w_resized / video_frame_width
    
    crop_width_resized = crop_width_original * scale_w
    crop_height_resized = crop_height_original * scale_h
    
    print(f"\n=== Using Uncropped Tracker Results ===")
    print(f"Tracked points shape: {coords.shape}")
    print(f"Coords are in METERS (metric world coordinates)")
    print(f"Original video: {video_frame_width:.0f}x{video_frame_height:.0f} pixels")
    print(f"Resized video (tracker): {w_resized:.0f}x{h_resized:.0f} pixels")
    print(f"Crop in original video: {crop_width_original:.0f}x{crop_height_original:.0f} pixels")
    print(f"Crop in resized video: {crop_width_resized:.1f}x{crop_height_resized:.1f} pixels")
    
    # Compute head-to-tail distance from 3D SMAL joints (average over all frames)
    head_to_tail_distances = []
    for i in range(len(betas)):
        smal_out = smal_model.smal(
            betas=betas[i][None],
            global_orient=poses[i, 0:1][None],
        pose=poses[i, 1:][None],
        pose2rot=False,
    )
    joints_3d = smal_out.joints[0].detach().cpu().numpy()  # [num_joints, 3]
    head_joint = joints_3d[32]  # Head is 32nd joint (0-indexed)
    tail_joint = joints_3d[26]  # Tail is 26th joint (0-indexed)
    distance = np.linalg.norm(head_joint - tail_joint)
    head_to_tail_distances.append(distance)
    
    smal_head_to_tail = np.mean(head_to_tail_distances)
    print(f"SMAL head-to-tail distance (avg): {smal_head_to_tail:.4f} SMAL units")
    
    # Estimate real animal size in meters using crop dimensions and depth
    # Using pinhole camera model: real_size = (pixel_size / focal_length) * depth
    
    # Get average depth of tracked points (Z coordinate in camera space)
    # Transform world coords back to camera space to get depth
    extrinsics = tracker_results['extrinsics']  # Shape: [T, 4, 4], world-to-camera
    
    # For frame 0, compute depth of tracked points
    w2c = extrinsics[0]  # World-to-camera transform
    coords_cam = (w2c[:3, :3] @ coords[0].T).T + w2c[:3, 3]  # Transform to camera space
    depths = coords_cam[:, 2]  # Z coordinate = depth in meters
    avg_depth = np.mean(depths)
    
    print(f"\nDepth analysis:")
    print(f"  Tracked points depth range: {depths.min():.3f} to {depths.max():.3f} meters")
    print(f"  Average depth: {avg_depth:.3f} meters")
    
    # Get focal lengths from intrinsics (frame 0)
    # NOTE: These intrinsics are for the RESIZED video!
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    
    # Compute real-world size of the crop (which contains the full animal)
    # Using RESIZED crop dimensions (which match the intrinsics scale)
    # real_width = (crop_width_resized / fx) * depth
    # real_height = (crop_height_resized / fy) * depth
    real_width_meters = (crop_width_resized / fx) * avg_depth
    real_height_meters = (crop_height_resized / fy) * avg_depth
    
    # Average as estimate of animal body size
    real_animal_size_meters = (real_width_meters + real_height_meters) / 2
    
    print(f"\nReal animal size in video (using crop dimensions + depth):")
    print(f"  Focal length (for resized video): fx={fx:.1f}, fy={fy:.1f}")
    print(f"  Crop width (resized): {crop_width_resized:.1f} pixels → {real_width_meters:.3f} meters")
    print(f"  Crop height (resized): {crop_height_resized:.1f} pixels → {real_height_meters:.3f} meters")
    print(f"  Average animal size: {real_animal_size_meters:.3f} meters")
    
    # Compute scale factor: converts meters to SMAL units
    # scale_factor = SMAL_size / metric_size
    scale_factor = smal_head_to_tail / real_animal_size_meters
    
    print(f"\nScale factor calculation:")
    print(f"  scale_factor = smal_head_to_tail / real_animal_size")
    print(f"  scale_factor = {smal_head_to_tail:.4f} / {real_animal_size_meters:.3f}")
    print(f"  scale_factor = {scale_factor:.4f} (meters → SMAL units)")
    
    # Compute center of mass for each frame and get relative translations
    coms = np.mean(coords, axis=1)  # Shape: [T, 3], units: METERS
    com_0 = coms[0]
    global_translations = coms - com_0  # Relative motion from frame 0, units: METERS
    
    # Scale translations from meters to SMAL units
    global_translations *= scale_factor
    
    # Apply 180-degree rotation around X-axis to match SMAL coordinate system
    # The SMAL model is rotated 180° around X-axis, so we need to rotate translations too
    # Rotation matrix for 180° around X: [1, 0, 0; 0, -1, 0; 0, 0, -1]
    # This flips Y and Z axes: [x, y, z] -> [x, -y, -z]
    rotation_matrix_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
    global_translations = (rotation_matrix_180 @ global_translations.T).T
    
    print(f"\nGlobal translations:")
    print(f"  Before rotation and scaling (meters): {np.linalg.norm((coms - com_0)[-1]):.4f} m")
    print(f"  After rotation and scaling (SMAL units): {np.linalg.norm(global_translations[-1]):.4f} units")
    print(f"  First 5 frames (SMAL units, rotated to SMAL coords):")
    for i in range(min(5, len(global_translations))):
        print(f"    Frame {i}: [{global_translations[i, 0]:.4f}, {global_translations[i, 1]:.4f}, {global_translations[i, 2]:.4f}]")
    
    # Create dummy rotations (identity) since we're not using camera extrinsics
    all_rotations = [np.eye(3) for _ in range(len(coords))]
    R_cam0_tensor = torch.eye(3).to(device)
    
    return {
        'global_transforms': torch.eye(4).unsqueeze(0).repeat(len(coords), 1, 1).to(device),
        'R_cam0': R_cam0_tensor,
        'translations': global_translations,
        'rotations': all_rotations,
        'use_uncropped': True  # Flag to skip camera rotations in process_global_translation
    }


def compute_global_motion_cropped(image_paths, cams, device, smal_model, betas, poses, tracker_results_path):
    """
    Compute global motion from camera extrinsics in cropped video.
    
    Args:
        image_paths: List of image file paths
        cams: Tensor of camera offsets [T, 3] from AMR predictions
        device: torch device
        smal_model: SMAL model to compute 3D joints (not used in this mode)
        betas: Tensor [T, num_betas] shape parameters (not used in this mode)
        poses: Tensor [T, num_joints, 3, 3] pose matrices (not used in this mode)
    
    Returns:
        dict with keys:
            - global_transforms: Tensor [T, 4, 4] camera-to-world transforms
            - R_cam0: Tensor [3, 3] rotation of camera at frame 0
            - translations: np.ndarray [T, 3] animal translations
            - rotations: list of [3, 3] camera rotation matrices
            - use_uncropped: False (flag for downstream processing)
    """
    # Load cropped tracker results
    assert os.path.exists(tracker_results_path), f"Tracker results not found at {tracker_results_path}"
    tracker_results = np.load(tracker_results_path)
    extrinsics = tracker_results['extrinsics']  # Shape: [T, 4, 4], world-to-camera
    
    # Conversion matrix from OpenCV to SMAL convention
    opencv_to_smal = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)
    
    print("\n=== Coordinate System Conversion ===")
    print("Tracker data: OpenCV convention (X:right, Y:down, Z:forward)")
    print("SMAL data: Graphics convention (X:right, Y:up, Z:backward)")
    print("Conversion: Y→-Y, Z→-Z")
    
    # extrinsics is world-to-camera transform in OpenCV convention
    # Convert to camera-to-world in SMAL convention
    T_cam_to_world_opencv = [np.linalg.inv(extrinsics[i]) for i in range(len(extrinsics))]
    T_cam_to_world_smal = [opencv_to_smal @ T @ np.linalg.inv(opencv_to_smal) for T in T_cam_to_world_opencv]
    
    # Store absolute camera positions and rotations (in SMAL convention)
    global_transforms = np.stack(T_cam_to_world_smal, axis=0)  # [T, 4, 4]
    global_transforms = torch.from_numpy(global_transforms).float().to(device)
    
    # Compute animal position at frame 0
    T_cam0 = global_transforms[0].cpu().numpy()
    R_cam0 = T_cam0[:3, :3]
    t_cam0 = T_cam0[:3, 3]
    cam_offset_0 = cams[0].cpu().numpy()
    animal_in_cam_frame_0 = -cam_offset_0
    animal_position_world_0 = R_cam0 @ animal_in_cam_frame_0 + t_cam0
    
    R_cam0_tensor = torch.from_numpy(R_cam0).float().to(device)
    animal_origin_offset = animal_position_world_0
    
    # Extract camera positions
    cam_positions = global_transforms[:, :3, 3].cpu().numpy()
    cam_pos_0 = cam_positions[0]
    cam_deltas = cam_positions - cam_pos_0
    
    max_displacement = np.linalg.norm(cam_deltas, axis=1).max()
    print(f"Applying global motion from camera. Max displacement: {max_displacement:.3f}")
    print(f"Animal origin offset (frame 0 position): {animal_position_world_0}")
    print(f"Camera positions (first 5): {cam_positions[:5]}")
    print(f"Camera deltas from frame 0 (first 5): {cam_deltas[:5]}")
    
    # Precompute all animal translations
    all_animal_translations = []
    all_rotations = []
    for i in range(len(image_paths)):
        T_cam = global_transforms[i].cpu().numpy()
        R_cam = T_cam[:3, :3]
        t_cam = T_cam[:3, 3]
        cam_offset = cams[i].cpu().numpy()
        animal_in_cam_frame = -cam_offset
        animal_position_world = R_cam @ animal_in_cam_frame + t_cam
        animal_position_relative = animal_position_world - animal_origin_offset
        all_animal_translations.append(animal_position_relative)
        all_rotations.append(R_cam)
    
    all_animal_translations = np.stack(all_animal_translations)
    
    return {
        'global_transforms': global_transforms,
        'R_cam0': R_cam0_tensor,
        'translations': all_animal_translations,
        'rotations': all_rotations,
        'use_uncropped': False  # Flag for camera-based mode
    }


def compute_global_motion(image_paths, cams, device, smal_model, betas, poses, tracker_results_path, use_uncropped_results=False):
    """
    Wrapper function to compute global motion using either camera or tracked points.
    
    Args:
        image_paths: List of image file paths
        cams: Tensor of camera offsets [T, 3] from AMR predictions
        device: torch device
        smal_model: SMAL model to compute 3D joints
        betas: Tensor [T, num_betas] shape parameters
        poses: Tensor [T, num_joints, 3, 3] pose matrices
        use_uncropped_results: If True, use tracked points; if False, use camera extrinsics
    
    Returns:
        dict with motion data (see individual functions for details)
    """
    if use_uncropped_results:
        return compute_global_motion_uncropped(image_paths, cams, device, smal_model, betas, poses, tracker_results_path)
    else:
        return compute_global_motion_cropped(image_paths, cams, device, smal_model, betas, poses, tracker_results_path)


def regularize_global_translation_direction(
    smal_model, betas, poses, translations, device, rotations=None, R_cam0_tensor=None,
    use_uncropped=False, max_angle_degrees=30.0, withers_joint=15,
):
    """Limit sideways frame-to-frame motion relative to the animal's heading.

    The heading is the root-to-withers vector projected onto the ground (XZ) plane.
    Each frame's displacement is ``translation[t] - translation[t - 1]``. For a
    step outside the allowed heading cone, its lateral component is removed rather
    than rotated. Signed forward/backward motion and the vertical component are
    preserved; a purely sideways step therefore becomes zero horizontal motion
    instead of creating artificial forward movement. Corrected steps are
    integrated from frame 0.
    """
    regularized = translations.copy()
    rotation_matrix_180 = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0]
    )[:3, :3]
    print(f"\nRegularizing global-motion direction (maximum heading difference: {max_angle_degrees:.1f} degrees)")

    for i in range(len(regularized)):
        smal_out = smal_model.smal(
            betas=betas[i][None],
            global_orient=poses[i, 0:1][None],
            pose=poses[i, 1:][None],
            pose2rot=False,
        )
        joints = smal_out.joints[0].detach().cpu().numpy()
        joints = (rotation_matrix_180 @ joints.T).T

        if not use_uncropped:
            R_rel = rotations[i] @ R_cam0_tensor.cpu().numpy().T
            joints = (R_rel @ joints.T).T

        # Compare directions only on the ground plane: X (right) and Z (back).
        if i == 0:
            print("Frame 0: no previous frame; keeping initial translation unchanged")
            continue

        heading_xz = joints[withers_joint, [0, 2]] - joints[0, [0, 2]]
        # Measure every step from the original tracker trajectory. Integrating
        # below preserves all prior corrections while avoiding drift from using
        # an already-corrected step as the next measurement.
        raw_step = translations[i] - translations[i - 1]
        step_xz = raw_step[[0, 2]]
        heading_norm = np.linalg.norm(heading_xz)
        step_norm = np.linalg.norm(step_xz)

        if heading_norm < 1e-8 or step_norm < 1e-8:
            regularized[i] = regularized[i - 1] + raw_step
            print(f"Frame {i}: heading/motion-step angle undefined (near-zero horizontal vector); no regularization")
            continue

        heading_dir = heading_xz / heading_norm
        step_dir = step_xz / step_norm
        dot = np.clip(np.dot(heading_dir, step_dir), -1.0, 1.0)
        angle_degrees = np.degrees(np.arccos(dot))
        print(f"Frame {i}: motion-step-to-heading difference = {angle_degrees:.2f} degrees")

        # Split the horizontal step into heading-parallel and lateral components.
        # This retains the sign of real forward/backward movement rather than
        # rotating a backward/noisy step into an artificial forward step.
        parallel_magnitude = np.dot(step_xz, heading_dir)
        parallel_step = parallel_magnitude * heading_dir
        lateral_step = step_xz - parallel_step
        lateral_magnitude = np.linalg.norm(lateral_step)

        if angle_degrees > max_angle_degrees:
            # Do not rotate noisy side/backward motion into the forward direction.
            # Retaining only the signed heading component eliminates lateral drift:
            # forward remains forward, backward remains backward, and a pure
            # sideways step has no horizontal displacement.
            corrected_step = raw_step.copy()
            corrected_step[[0, 2]] = parallel_step
            regularized[i] = regularized[i - 1] + corrected_step
            print(
                f"  Removed lateral-motion component: {lateral_magnitude:.4f} "
                f"→ 0.0000; preserved signed heading and vertical components"
            )
        else:
            regularized[i] = regularized[i - 1] + raw_step

    return regularized


def process_global_translation(
    smal_model, betas, poses, motion_data, device,
    normalize=True, smooth=True, smooth_mode='two_stage', window_size=10, sigma=2
):
    """
    Process global translations: normalize and smooth.
    
    Args:
        smal_model: SMAL model
        betas: Tensor [T, num_betas] shape parameters
        poses: Tensor [T, num_joints, 3, 3] pose matrices
        motion_data: dict from compute_global_motion with 'translations', 'rotations', 'R_cam0'
        device: torch device
        normalize: Whether to normalize translations
        smooth: Whether to smooth translations
        smooth_mode: Smoothing mode ('gaussian', 'two_stage', etc.)
        window_size: Window size for smoothing
        sigma: Sigma for Gaussian smoothing
    
    Returns:
        np.ndarray [T, 3] processed translations
    """
    translations = motion_data['translations'].copy()
    use_uncropped = motion_data.get('use_uncropped', False)
    
    if use_uncropped:
        # For uncropped mode: skip all camera rotations, just use pure translations
        print("\nProcessing translations from tracked points (no camera rotation applied)")

        # Correct raw tracker motion steps before any normalization or temporal
        # filtering, so smoothing operates on the heading-constrained trajectory.
        translations = regularize_global_translation_direction(
            smal_model, betas, poses, translations, device,
            use_uncropped=True,
        )
        
        if normalize:
            # Normalize without applying camera rotations
            # Just center root horizontally and vertically
            all_world_joints = []
            for i in range(len(translations)):
                smal_out = smal_model.smal(
                    betas=betas[i][None],
                    global_orient=poses[i, 0:1][None],
                    pose=poses[i, 1:][None],
                    pose2rot=False,
                )
                joints_local = smal_out.joints[0].detach().cpu().numpy()
                
                # Apply 180-degree rotation
                rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
                joints_rotated = (rotation_matrix @ joints_local.T).T
                
                # NO camera rotation applied - use identity orientation
                joints_oriented = joints_rotated
                
                # Apply global translation to get world coordinates
                joints_world = joints_oriented + translations[i]
                all_world_joints.append(joints_world)
            
            all_world_joints = np.stack(all_world_joints)
            translations = normalize_translation(all_world_joints, translations)
            print(f"Normalized translations (from tracked points only)")

        if smooth:
            translations = smooth_translation(translations, mode=smooth_mode, window_size=window_size, sigma=sigma)
            print(f"Smoothed {len(translations)} translations with mode={smooth_mode}")
            
            # Re-normalize after smoothing to ensure frame 0 stays centered
            if normalize:
                smoothed_world_joints = []
                for i in range(len(translations)):
                    smal_out = smal_model.smal(
                        betas=betas[i][None],
                        global_orient=poses[i, 0:1][None],
                        pose=poses[i, 1:][None],
                        pose2rot=False,
                    )
                    joints_local = smal_out.joints[0].detach().cpu().numpy()
                    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
                    joints_rotated = (rotation_matrix @ joints_local.T).T
                    joints_oriented = joints_rotated  # NO camera rotation
                    joints_world = joints_oriented + translations[i]
                    smoothed_world_joints.append(joints_world)
                
                smoothed_world_joints = np.stack(smoothed_world_joints)
                translations = normalize_translation(smoothed_world_joints, translations)
                print(f"Re-normalized after smoothing")
        
        return translations
    
    # Original camera-based mode
    rotations = motion_data['rotations']
    R_cam0_tensor = motion_data['R_cam0']

    # Correct raw tracker motion steps before any normalization or temporal
    # filtering, so smoothing operates on the heading-constrained trajectory.
    translations = regularize_global_translation_direction(
        smal_model, betas, poses, translations, device,
        rotations=rotations, R_cam0_tensor=R_cam0_tensor,
        use_uncropped=False,
    )
    
    if normalize:
        # Compute world-frame joints for all frames
        all_world_joints = []
        for i in range(len(translations)):
            smal_out = smal_model.smal(
                betas=betas[i][None],
                global_orient=poses[i, 0:1][None],
                pose=poses[i, 1:][None],
                pose2rot=False,
            )
            joints_local = smal_out.joints[0].detach().cpu().numpy()
            
            # Apply 180-degree rotation
            rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
            joints_rotated = (rotation_matrix @ joints_local.T).T
            
            # Apply camera rotation
            R_cam = rotations[i]
            R_cam_torch = torch.from_numpy(R_cam).float().to(device)
            R_rel = R_cam_torch.cpu().numpy() @ R_cam0_tensor.cpu().numpy().T
            joints_oriented = (R_rel @ joints_rotated.T).T
            
            # Apply global translation to get world coordinates
            joints_world = joints_oriented + translations[i]
            all_world_joints.append(joints_world)
        
        all_world_joints = np.stack(all_world_joints)
        translations = normalize_translation(all_world_joints, translations)
        print(f"Normalized translations applied")

    if smooth:
        translations = smooth_translation(translations, mode=smooth_mode, window_size=window_size, sigma=sigma)
        print(f"Smoothed {len(translations)} translations with mode={smooth_mode}")
        
        # Re-normalize after smoothing to ensure frame 0 stays centered
        if normalize:
            # Recompute world joints with smoothed translations
            smoothed_world_joints = []
            for i in range(len(translations)):
                smal_out = smal_model.smal(
                    betas=betas[i][None],
                    global_orient=poses[i, 0:1][None],
                    pose=poses[i, 1:][None],
                    pose2rot=False,
                )
                joints_local = smal_out.joints[0].detach().cpu().numpy()
                rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
                joints_rotated = (rotation_matrix @ joints_local.T).T
                R_cam = rotations[i]
                R_cam_torch = torch.from_numpy(R_cam).float().to(device)
                R_rel = R_cam_torch.cpu().numpy() @ R_cam0_tensor.cpu().numpy().T
                joints_oriented = (R_rel @ joints_rotated.T).T
                joints_world = joints_oriented + translations[i]
                smoothed_world_joints.append(joints_world)
            
            smoothed_world_joints = np.stack(smoothed_world_joints)
            translations = normalize_translation(smoothed_world_joints, translations)
            print(f"Re-normalized after smoothing to ensure frame 0 root is centered")
    
    return translations


def subsample_and_interpolate(data, sample_every=5, is_rotation=False):
    """
    Subsample data at keyframes and interpolate between them for smoothing.
    
    Args:
        data: np.ndarray or torch.Tensor
            - For translations: shape [T, 3]
            - For rotations: shape [T, 3, 3] (rotation matrices)
        sample_every: int, subsample every N frames as keyframes
        is_rotation: bool, whether data contains rotation matrices
    
    Returns:
        smoothed data with same shape and type as input
    """
    is_torch = isinstance(data, torch.Tensor)
    device = data.device if is_torch else None
    dtype = data.dtype if is_torch else None
    
    # Convert to numpy
    if is_torch:
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data.copy()
    
    T = len(data_np)
    
    # Create keyframe indices (including first and last)
    keyframe_indices = list(range(0, T, sample_every))
    if keyframe_indices[-1] != T - 1:
        keyframe_indices.append(T - 1)
    
    if is_rotation:
        # For rotations, use SLERP via quaternions
        from scipy.spatial.transform import Rotation as R
        
        # Convert rotation matrices to quaternions
        keyframe_quats = []
        for idx in keyframe_indices:
            rot = R.from_matrix(data_np[idx])
            keyframe_quats.append(rot.as_quat())
        
        # Interpolate between keyframes
        smoothed = np.zeros_like(data_np)
        for i in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i + 1]
            
            # Create SLERP interpolator between keyframes
            key_rots = R.from_quat([keyframe_quats[i], keyframe_quats[i + 1]])
            key_times = [0, end_idx - start_idx]
            
            # Interpolate for all frames in this segment
            for j in range(start_idx, end_idx + 1):
                t = j - start_idx
                slerp = R.from_quat(key_rots.as_quat())
                if t == 0:
                    smoothed[j] = data_np[start_idx]
                elif t == end_idx - start_idx:
                    smoothed[j] = data_np[end_idx]
                else:
                    # Linear interpolation in quaternion space (SLERP)
                    alpha = t / (end_idx - start_idx)
                    q1 = keyframe_quats[i]
                    q2 = keyframe_quats[i + 1]
                    
                    # SLERP formula
                    dot = np.dot(q1, q2)
                    if dot < 0:
                        q2 = -q2
                        dot = -dot
                    
                    if dot > 0.9995:
                        # Use linear interpolation for very close quaternions
                        q_interp = q1 + alpha * (q2 - q1)
                        q_interp = q_interp / np.linalg.norm(q_interp)
                    else:
                        theta = np.arccos(np.clip(dot, -1, 1))
                        q_interp = (np.sin((1 - alpha) * theta) * q1 + np.sin(alpha * theta) * q2) / np.sin(theta)
                    
                    smoothed[j] = R.from_quat(q_interp).as_matrix()
    else:
        # For translations, use linear interpolation
        smoothed = np.zeros_like(data_np)
        for i in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i + 1]
            
            # Linear interpolation between keyframes
            for j in range(start_idx, end_idx + 1):
                alpha = (j - start_idx) / (end_idx - start_idx)
                smoothed[j] = (1 - alpha) * data_np[start_idx] + alpha * data_np[end_idx]
    
    # Convert back to original type
    if is_torch:
        return torch.from_numpy(smoothed).to(device=device, dtype=dtype)
    else:
        return smoothed


@torch.no_grad()
def save_results(
    smal_model, smal_cfg, betas, poses, cams, full_cams, kp2ds, images, image_paths, device, image_size, out_folder,
    renderer=None, motion_data=None, processed_translations=None, center_keypoint=None, 
    save_video=False, save_mesh=False, save_smal_params=True, subsample_smooth=True, subsample_every=3
):
    """
    Save final SMAL results with optional global motion.
    
    Args:
        motion_data: dict from compute_global_motion (if applying global motion)
        processed_translations: np.ndarray [T, 3] from process_global_translation (if applying global motion)
        subsample_smooth: bool, if True apply subsampled interpolation smoothing
        subsample_every: int, subsample every N frames for smoothing
    """
    # Extract motion parameters if provided
    global_transforms = motion_data['global_transforms'] if motion_data else None
    R_cam0_tensor = motion_data['R_cam0'] if motion_data else None
    
    # Apply subsample smoothing if requested
    if subsample_smooth:
        print(f"\nApplying subsample smoothing (every {subsample_every} frames)...")
        
        # Smooth SMAL poses (global_orient and body poses separately)
        # Global orient: [T, 3, 3]
        global_orient_matrices = poses[:, 0, :, :]  # [T, 3, 3]
        smoothed_global_orient = subsample_and_interpolate(
            global_orient_matrices, sample_every=subsample_every, is_rotation=True
        )
        
        # Body poses: [T, num_joints-1, 3, 3]
        body_poses = poses[:, 1:, :, :]
        smoothed_body_poses = []
        for joint_idx in range(body_poses.shape[1]):
            joint_rotations = body_poses[:, joint_idx, :, :]  # [T, 3, 3]
            smoothed_joint = subsample_and_interpolate(
                joint_rotations, sample_every=subsample_every, is_rotation=True
            )
            smoothed_body_poses.append(smoothed_joint[:, None, :, :])
        smoothed_body_poses = torch.cat(smoothed_body_poses, dim=1)
        
        # Reconstruct full poses
        poses = torch.cat([smoothed_global_orient[:, None, :, :], smoothed_body_poses], dim=1)
        
        # Smooth processed translations if available
        if processed_translations is not None:
            processed_translations = subsample_and_interpolate(
                processed_translations, sample_every=subsample_every, is_rotation=False
            )
        
        print(f"Smoothing complete.")

    all_joints = []
    
    for i, image_path in enumerate(image_paths):
        smal_out = smal_model.smal(
            betas=betas[i][None],
            global_orient=poses[i, 0:1][None],
            pose=poses[i, 1:][None],
            pose2rot=False,
        )
        verts = smal_out.vertices[0].detach().cpu().numpy()

        # Apply 180 rotation transformations
        rotation_matrix = torch.from_numpy(
            trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])[:3, :3]
        ).to(device=poses.device, dtype=poses.dtype)      
        global_orient_rotated = rotation_matrix @ poses[i, 0:1][None]

        # Compute the translation and orientation for this frame (for saving parameters, not for video)
        if processed_translations is not None and global_transforms is not None:
            # Apply global motion using processed translations
            animal_position_relative = processed_translations[i]
            frame_translation = torch.from_numpy(animal_position_relative).float().to(device)
            
            # Check if using uncropped results (no camera rotation)
            use_uncropped = motion_data.get('use_uncropped', False) if motion_data else False
            
            if use_uncropped:
                # No camera rotation applied - just use processed translation
                # global_orient_rotated already has 180-degree rotation, don't apply camera rotation
                if i < 5:  # Debug: print first 5 frames
                    print(f"Frame {i}: processed_translation={animal_position_relative} (uncropped, no camera rotation)")
            else:
                # Original camera-based approach: apply relative camera rotation
                T_cam = global_transforms[i].cpu().numpy()
                R_cam = T_cam[:3, :3] if len(T_cam.shape) == 2 else T_cam[:3, :3]
                R_cam_torch = torch.from_numpy(R_cam).float().to(device) if isinstance(R_cam, np.ndarray) else R_cam
                R_rel = R_cam_torch @ R_cam0_tensor.T
                global_orient_rotated = R_rel @ global_orient_rotated
                
                if i < 5:  # Debug: print first 5 frames
                    print(f"Frame {i}: processed_translation={animal_position_relative} (camera-based)")
            
        elif center_keypoint is not None and center_keypoint >= 0:
            # Center a specific keypoint
            smal_output_rotated = smal_model.smal(
                betas=betas[i][None],
                global_orient=global_orient_rotated,
                pose=poses[i, 1:][None],
                transl=None,
                pose2rot=False,
            )
            rotated_keypoints = smal_output_rotated['joints'][0].detach().cpu().numpy()
            root_position = rotated_keypoints[center_keypoint]
            frame_translation = torch.from_numpy(-root_position).to(smal_model.device)
        else:
            # No translation
            frame_translation = torch.zeros(3).to(smal_model.device)
        
        # Final forward pass with correct parameters
        final_betas = betas[i][None]
        final_global_orient = global_orient_rotated
        final_pose = poses[i, 1:][None]
        final_transl = frame_translation[None]
        smal_output_final = smal_model.smal(
            betas=final_betas,
            global_orient=final_global_orient,
            pose=final_pose,
            transl=final_transl,
            pose2rot=False,
        )

        # Save smal parameters
        if save_smal_params:
            out_name = os.path.basename(image_path).replace('_rgb.png', '_smal_params_global.pth')
            torch.save({
                'betas': final_betas.cpu(),
                'global_orient': final_global_orient.cpu(),
                'pose': final_pose.cpu(),
                'transl': final_transl.cpu(),
                'joints': smal_output_final.joints.cpu(),
                'cam': cams[i].cpu(),
            }, os.path.join(out_folder, out_name))

        # Save final mesh per frame
        if save_mesh:
            final_verts = smal_output_final['vertices'][0].squeeze().detach().cpu().numpy()
            out_mesh_name = os.path.basename(image_path).replace('_rgb.png', '_mesh_animer_global.obj')
            mesh = trimesh.Trimesh(vertices=final_verts, faces=smal_model.smal.faces.detach().cpu().numpy())
            mesh.export(os.path.join(out_folder, out_mesh_name))

        # Collect all joints from final forward pass
        final_joints = smal_output_final['joints'][0].squeeze().detach().cpu().numpy()
        all_joints.append(final_joints)
            
    # Save all joints as a single file
    all_joints = np.stack(all_joints, axis=0)
    assert all_joints.shape[0] == len(image_paths)
    assert all_joints.shape[1] == 35
    np.save(os.path.join(out_folder, 'joints_global.npy'), all_joints)
    print(f"Saved to {os.path.join(out_folder, 'joints_global.npy')}")


def process_sequence(args, model, model_cfg, device, renderer, sequence_dir):
    """
    Process a single sequence directory.
    
    Args:
        args: Parsed command line arguments
        model: SMAL/AMR model
        model_cfg: Model configuration
        device: torch device
        renderer: Mesh renderer
        sequence_dir: Path to sequence directory
    """
    out_folder = args.out_folder if args.out_folder is not None else sequence_dir
    joints_path = os.path.join(out_folder, 'joints_global.npy')
    if os.path.exists(joints_path):
        print(f"Skipping {sequence_dir} as {joints_path} already exists")
        return
    
    tracker_dir = args.tracker_dir if args.tracker_dir is not None else out_folder
    if args.uncropped_results:
        tracker_results_path = os.path.join(tracker_dir, 'tracker_results_uncropped.npz')
    else:
        tracker_results_path = os.path.join(tracker_dir, 'tracker_results.npz')
    if args.global_motion and not os.path.exists(tracker_results_path):
        print(f"Tracker results not found at {tracker_results_path}, skipping {sequence_dir}")
        return
        
    print(f"Processing {sequence_dir}")
    os.makedirs(out_folder, exist_ok=True)
    img_paths = sorted(Path(sequence_dir).glob('*rgb.png'))
    all_original_renders = []
    all_original_betas = []
    all_original_poses = []
    all_original_cams = []
    all_original_fullcams = []
    all_original_images = []
    all_original_masks = []
    all_gt_kp2ds = []

    # Get all camera prediction, rendered images before optimization
    for i, img_path in enumerate(tqdm(img_paths, desc="Processing images")):
        img_cv2 = cv2.imread(str(img_path))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        boxes = np.array([[0, 0, img_cv2.shape[1], img_cv2.shape[0]]])
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        data_img = torch.tensor(dataset[0]['img'], dtype=torch.float32).to(device)
        all_original_images.append(data_img)
        mask_path = str(img_path).replace('_rgb.png', '_mask.png')
        if os.path.exists(mask_path):
            mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            mask_cv2 = (mask_cv2 > 127).astype(np.float32)
            all_original_masks.append(mask_cv2)
        else:
            all_original_masks.append(None)
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            pred_keypoints_2d = out['pred_keypoints_2d']
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            all_original_fullcams.append(pred_cam_t_full)
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                final_img = renderer(
                    vertices=out['pred_vertices'][n].detach().cpu().numpy(),
                    camera_translation=out['pred_cam_t'][n].detach().cpu().numpy(),
                    image=batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                final_img = (final_img[:, :, ::-1] * 255).astype(np.uint8)
                final_img = draw_keypoints(final_img, pred_keypoints_2d.detach().squeeze().cpu().numpy())
                
                all_original_renders.append(final_img)
                all_original_betas.append(out['pred_smal_params']['betas'][n].detach())
                all_original_cams.append(out['pred_cam_t'][n].detach().cpu().numpy())
                all_gt_kp2ds.append(np.loadtxt(str(img_path).replace('_rgb.png', '_keypoint.txt')))
                all_original_poses.append(torch.cat([
                    out['pred_smal_params']['global_orient'][n].detach(), 
                    out['pred_smal_params']['pose'][n].detach()
                ], dim=0))

    all_original_betas = torch.stack(all_original_betas, dim=0).to(model.device)
    all_original_poses = torch.stack(all_original_poses, dim=0).to(model.device)
    all_original_cams = torch.tensor(all_original_cams).to(model.device)
    all_original_fullcams = torch.tensor(all_original_fullcams).to(model.device)
    all_gt_kp2ds = torch.tensor(all_gt_kp2ds).to(model.device)

    # Initialize with original predictions (will be overwritten if optimization runs)
    mean_beta = all_original_betas.mean(dim=0, keepdim=True)
    poses = all_original_poses
    
    # Step 1: Optimize SMAL parameters (optional - can be commented out to skip optimization)
    opt_result = optimize_parameters(
        smal_model=model, smal_cfg=model_cfg, betas=all_original_betas, poses=all_original_poses,
        kp2ds=all_gt_kp2ds, cams=all_original_cams, img_size=int(img_size.max().item()),
        masks=all_original_masks,
        num_iters=args.num_iters, lr=args.lr,
        proj_loss_weight=args.proj_loss_weight, smooth_loss_weight=args.smooth_loss_weight,
        cam_anchor_weight=args.cam_anchor_weight,
        restrict_side_bending=args.restrict_side_bending,
        restrict_twist=args.restrict_twist,
        image_paths=img_paths, images=all_original_images, renderer=renderer, 
        device=model.device, save_video=args.save_video, out_folder=out_folder
    )
    
    # Unpack optimization result
    mean_beta, poses, optimized_transl = opt_result
    print(f"Optimized translation range: {optimized_transl.min(dim=0).values} to {optimized_transl.max(dim=0).values}")

    betas = repeat(mean_beta, "1 d -> n d", n=len(img_paths))
    
    # Step 2: Compute and process global motion (if enabled)
    motion_data = None
    processed_translations = None
    
    if args.global_motion:
        # Fix SMAL camera to frame 0 to remove jitter before applying tracker extrinsics
        fixed_cams = all_original_cams[0:1].repeat(len(img_paths), 1)
        motion_data = compute_global_motion(
            image_paths=img_paths,
            cams=fixed_cams,
            device=model.device,
            smal_model=model,
            betas=betas,
            poses=poses,
            tracker_results_path=tracker_results_path,
            use_uncropped_results=args.uncropped_results
        )
        processed_translations = process_global_translation(
            smal_model=model,
            betas=betas,
            poses=poses,
            motion_data=motion_data,
            device=model.device,
            normalize=True,
            smooth=True,
            smooth_mode='two_stage',
            window_size=10,
            sigma=2
        )
    
    # Step 3: Save results
    save_results(
        smal_model=model, smal_cfg=model_cfg, betas=betas, poses=poses, kp2ds=all_gt_kp2ds, out_folder=out_folder,
        cams=all_original_cams, full_cams=all_original_fullcams, images=all_original_images, image_paths=img_paths, 
        image_size=int(img_size.max().item()), renderer=renderer, 
        motion_data=motion_data, processed_translations=processed_translations,
        center_keypoint=args.center_keypoint, save_video=args.save_video, save_mesh=args.save_mesh, device=model.device
    )


if __name__ == '__main__':
    args = parse_args()
    model, model_cfg = load_amr(args.checkpoint_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    renderer = Renderer(model_cfg, faces=model.smal.faces)
    if args.sequence_dir:
        sequence_dirs = [args.sequence_dir]
    else:
        sequence_dirs = get_all_sequence_dirs(args.data_dir)
        shuffle(sequence_dirs)

    for sequence_dir in tqdm(sequence_dirs):

        # if os.path.basename(sequence_dir) != "98L2bpqyvYg_009_001":
        #     print(f"Skipping {sequence_dir} as it is not the target sequence")
        #     continue

        try:
            process_sequence(args, model, model_cfg, device, renderer, sequence_dir)
        except Exception as e:
            print(f"Error processing {sequence_dir}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
