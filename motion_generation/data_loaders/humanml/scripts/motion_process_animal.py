from os.path import join as pjoin
import sys
import os
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)

from data_loaders.humanml.common.skeleton import Skeleton
import numpy as np
import os
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *

import torch
from tqdm import tqdm
from data_loaders.humanml_utils import HML_JOINT_NAMES, HML_EE_JOINT_NAMES

import random
from copy import copy, deepcopy
from sklearn.model_selection import train_test_split

joints_file_name = "joints_global.npy"
new_joints_file_name = "new_joints_global.npy"
vecs_file_name = "new_joint_vecs_global.npy"

# Configuration for your dataset (adjust these parameters as needed)
# Lower legs
l_idx1, l_idx2 = 8, 9
# Four feet indices: front_left, front_right, rear_left, rear_right
fid_fl, fid_fr, fid_rl, fid_rr = 9, 13, 19, 23  # Adjust these to your skeleton
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [20, 16, 10, 6]
# l_hip, r_hip
# r_hip, l_hip = 11, 16
joints_num = 34

feet_threshold = 0.005


def remove_global_translation(positions):
    """
    Remove global translation by fixing root joint at first frame's y-position with x=0, z=0.
    
    Steps:
    1. Calculate the y-coordinate where motion should be fixed (first frame's root y)
    2. Move all frames so root is at (0, 0, 0)
    3. Move all frames to the target y-axis coordinate
    
    Args:
        positions: (n_frames, n_joints, 3) array
    
    Returns:
        positions: Modified array with root joint fixed at (0, y_target, 0) for all frames
    """
    # Step 1: Calculate the target y-coordinate from first frame's root
    target_y = positions[0, 0, 1]  # y-coordinate of first frame's root
    
    # Step 2: Move all frames so their root is at (0, 0, 0)
    # Subtract each frame's root position from all joints in that frame (vectorized)
    root_positions = positions[:, 0:1, :]  # (n_frames, 1, 3) - broadcast-friendly shape
    positions = positions - root_positions  # Now root is at (0, 0, 0) for all frames
    
    # Step 3: Move all frames to the target y-axis coordinate
    # Add [0, target_y, 0] to all joints in all frames
    positions[:, :, 1] += target_y  # Now root is at (0, target_y, 0) for all frames
    
    return positions

# positions (batch, joint_num, 3)
def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    
    # debug: save new joints:
    # tosave = (np.concatenate([new_joints[:,[0],:], new_joints], axis=1))
    # np.save("temp_new_joints.npy", tosave)

    return new_joints


def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_fl, fid_fr, fid_rl, fid_rr):
    global_positions = positions.copy()
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor = thres  # Single threshold value for each foot
        
        # Front left foot
        feet_fl_x = (positions[1:, fid_fl, 0] - positions[:-1, fid_fl, 0]) ** 2
        feet_fl_y = (positions[1:, fid_fl, 1] - positions[:-1, fid_fl, 1]) ** 2
        feet_fl_z = (positions[1:, fid_fl, 2] - positions[:-1, fid_fl, 2]) ** 2
        feet_fl = ((feet_fl_x + feet_fl_y + feet_fl_z) < velfactor).astype(float)
        
        # Front right foot
        feet_fr_x = (positions[1:, fid_fr, 0] - positions[:-1, fid_fr, 0]) ** 2
        feet_fr_y = (positions[1:, fid_fr, 1] - positions[:-1, fid_fr, 1]) ** 2
        feet_fr_z = (positions[1:, fid_fr, 2] - positions[:-1, fid_fr, 2]) ** 2
        feet_fr = ((feet_fr_x + feet_fr_y + feet_fr_z) < velfactor).astype(float)
        
        # Rear left foot
        feet_rl_x = (positions[1:, fid_rl, 0] - positions[:-1, fid_rl, 0]) ** 2
        feet_rl_y = (positions[1:, fid_rl, 1] - positions[:-1, fid_rl, 1]) ** 2
        feet_rl_z = (positions[1:, fid_rl, 2] - positions[:-1, fid_rl, 2]) ** 2
        feet_rl = ((feet_rl_x + feet_rl_y + feet_rl_z) < velfactor).astype(float)
        
        # Rear right foot
        feet_rr_x = (positions[1:, fid_rr, 0] - positions[:-1, fid_rr, 0]) ** 2
        feet_rr_y = (positions[1:, fid_rr, 1] - positions[:-1, fid_rr, 1]) ** 2
        feet_rr_z = (positions[1:, fid_rr, 2] - positions[:-1, fid_rr, 2]) ** 2
        feet_rr = ((feet_rr_x + feet_rr_y + feet_rr_z) < velfactor).astype(float)
        
        return feet_fl, feet_fr, feet_rl, feet_rr

    #
    feet_fl, feet_fr, feet_rl, feet_rr = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_fl[..., np.newaxis], feet_fr[..., np.newaxis], feet_rl[..., np.newaxis], feet_rr[..., np.newaxis]], axis=-1)

    return data


def align_to_ground_plane(positions, foot_indices):
    original_shape = positions.shape
    n_frames, n_joints, _ = original_shape 
    foot_points = positions[:, foot_indices, :].reshape(-1, 3)
    if foot_points.shape[0] < 3:
        return positions    
    centroid = foot_points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(foot_points - centroid)
    normal = vv[2]  # Normal vector of the fitted plane    
    if normal[1] < 0:
        normal = -normal
    target = np.array([0, 1, 0])    
    dot_product = np.dot(normal, target)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues
    if np.abs(dot_product - 1.0) < 1e-6:
        # Already aligned
        return positions    
    axis = np.cross(normal, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6:
        axis = axis / axis_norm
        angle = np.arccos(dot_product)        
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)        
        positions_flat = positions.reshape(-1, 3)
        positions_rotated = np.dot(positions_flat - centroid, R.T) + centroid
        positions = positions_rotated.reshape(original_shape)        
        foot_heights = positions[:, foot_indices, 1]
        min_foot_height = foot_heights.min()
        positions[:, :, 1] -= min_foot_height
    return positions



def debug_save_joints(joints):
    tosave = (np.concatenate([joints[:, [0], :], joints], axis=1))
    np.save("temp_joints.npy", tosave)


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    # floor_height = positions.min(axis=0).min(axis=0)[1]
    # positions[:, :, 1] -= floor_height
    positions = align_to_ground_plane(positions, [fid_fl, fid_fr, fid_rl, fid_rr])
    # debug
    # print(floor_height)
    # debug_save_joints(positions)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    # debug
    # debug_save_joints(positions)

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    # debug
    # debug_save_joints(positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor = thres  # Single threshold value for each foot
        
        # Front left foot
        feet_fl_x = (positions[1:, fid_fl, 0] - positions[:-1, fid_fl, 0]) ** 2
        feet_fl_y = (positions[1:, fid_fl, 1] - positions[:-1, fid_fl, 1]) ** 2
        feet_fl_z = (positions[1:, fid_fl, 2] - positions[:-1, fid_fl, 2]) ** 2
        feet_fl = ((feet_fl_x + feet_fl_y + feet_fl_z) < velfactor).astype(float)
        
        # Front right foot
        feet_fr_x = (positions[1:, fid_fr, 0] - positions[:-1, fid_fr, 0]) ** 2
        feet_fr_y = (positions[1:, fid_fr, 1] - positions[:-1, fid_fr, 1]) ** 2
        feet_fr_z = (positions[1:, fid_fr, 2] - positions[:-1, fid_fr, 2]) ** 2
        feet_fr = ((feet_fr_x + feet_fr_y + feet_fr_z) < velfactor).astype(float)
        
        # Rear left foot
        feet_rl_x = (positions[1:, fid_rl, 0] - positions[:-1, fid_rl, 0]) ** 2
        feet_rl_y = (positions[1:, fid_rl, 1] - positions[:-1, fid_rl, 1]) ** 2
        feet_rl_z = (positions[1:, fid_rl, 2] - positions[:-1, fid_rl, 2]) ** 2
        feet_rl = ((feet_rl_x + feet_rl_y + feet_rl_z) < velfactor).astype(float)
        
        # Rear right foot
        feet_rr_x = (positions[1:, fid_rr, 0] - positions[:-1, fid_rr, 0]) ** 2
        feet_rr_y = (positions[1:, fid_rr, 1] - positions[:-1, fid_rr, 1]) ** 2
        feet_rr_z = (positions[1:, fid_rr, 2] - positions[:-1, fid_rr, 2]) ** 2
        feet_rr = ((feet_rr_x + feet_rr_y + feet_rr_z) < velfactor).astype(float)
        
        return feet_fl, feet_fr, feet_rl, feet_rr
    #
    feet_fl, feet_fr, feet_rl, feet_rr = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_fl[..., np.newaxis], feet_fr[..., np.newaxis], feet_rl[..., np.newaxis], feet_rr[..., np.newaxis]], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation dataset
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_root_rot_heading_ang(joints):
    
    '''Get Forward Direction'''
    face_joint_idx = [2, 1, 17, 16]
    # l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx  # Note the bugfix
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = torch.nn.functional.normalize(across, dim=1)
    # print(across1.shape, across2.shape)

    # forward (batch_size, 3)
    forward = torch.cross(torch.tensor([[[0], [1], [0]]], dtype=across.dtype, device=across.device), across, axis=1)
    forward = torch.nn.functional.normalize(forward, dim=1)

    return torch.atan2(forward[:, 0], forward[:, 2])[:, None]

def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions

def recover_rot(data):
    # dataset [bs, seqlen, 263/251] HumanML/KIT
    joints_num = 22 if data.shape[-1] == 263 else 21
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_pos_pad = torch.cat([r_pos, torch.zeros_like(r_pos)], dim=-1).unsqueeze(-2)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    cont6d_params = torch.cat([cont6d_params, r_pos_pad], dim=-2)
    return cont6d_params


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions



def traj_global2vel(traj_positions, traj_rot):

    # traj_positions [bs, 2 (x,z), seqlen]
    # traj_positions [bs, 1 (z+, rad), seqlen]
    # return first 3 hml enries [bs, 3, seqlen-1]

    # skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # # (seq_len, joints_num, 4)
    # quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    bs, _, seqlen = traj_positions.shape
    traj_positions = traj_positions.permute(0, 2, 1)
    euler = torch.zeros([bs, 3, seqlen], dtype=traj_rot.dtype, device=traj_rot.device)
    euler[:, 1:2] = traj_rot
    euler = euler.permute(0, 2, 1).contiguous()
    traj_rot_quat = euler2quat(euler, 'yxz', deg=False)

    # '''Quaternion to continuous 6D'''
    # cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # # (seq_len, 4)
    r_rot = traj_rot_quat.clone()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = torch.zeros_like(euler[:, 1:, :])
    velocity[:, :, [0,2]] = (traj_positions[:, 1:, :] - traj_positions[:, :-1, :]).clone()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot(r_rot[:, 1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul(r_rot[:, 1:].contiguous(), qinv(r_rot[:, :-1]))
    # (seq_len, joints_num, 4)

    r_velocity = torch.arcsin(r_velocity[:, :, 2:3])
    l_velocity = velocity[:, :, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = torch.cat([r_velocity, l_velocity], axis=-1).permute(0, 2, 1)[:, :, None]

    return root_data

def get_target_location(motion, mean, std, lengths, joints_num, all_goal_joint_names, target_joint_names, is_heading):
    assert (lengths == lengths[0]).all(), 'currently supporting only fixed length'
    batch_size = motion.shape[0]
    extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']  # todo: fix hardcoded indexing that assumes traj and heading are last      
   
    # output tensor
    target_loc = torch.zeros((batch_size, len(extended_goal_joint_names), 3, lengths[0]), dtype=motion.dtype, device=motion.device)  #  n_samples x (n_target_joints+1) x 3 x n_frames

    # hml to abs loc (all joints, not only the requested ones)
    joints_loc = hml_to_abs_loc(motion, mean, std, joints_num)
    pelvis_loc = HML_JOINT_NAMES.index('pelvis')  
    joints_loc = torch.concat([joints_loc, joints_loc[:, pelvis_loc:pelvis_loc+1]], dim=1)  # concatenate the pelvis location to be used for traj 
    
    # joint names to indices
    HML_JOINT_NAMES_w_traj = HML_JOINT_NAMES + ['traj']
    for sample_idx in range(batch_size):
        req_joint_idx_in = [HML_JOINT_NAMES_w_traj.index(name) for name in target_joint_names[sample_idx]]
        req_joint_idx_out = [extended_goal_joint_names.index(name) for name in target_joint_names[sample_idx]]    
    
        target_loc[sample_idx, req_joint_idx_out] = joints_loc[sample_idx, req_joint_idx_in]  # assign joints loc to output tensor
    
    target_loc[:, -2, 1] = 0   # zero the y axis for the trajectory
        
    # last entry is the heading
    heading = recover_root_rot_heading_ang(joints_loc)
    target_loc[:, -1:, 0][is_heading] = heading[is_heading]
    
    return target_loc[..., -1]  # return last frame only


def hml_to_abs_loc(motion, mean, std, joints_num):
    # hml to abs loc (all joints, not only the requested ones)
    unnormed_motion = (motion * std + mean).permute(0, 2, 3, 1).float()
    joints_loc = recover_from_ric(unnormed_motion, joints_num)
    joints_loc = joints_loc.view(-1, *joints_loc.shape[2:]).permute(0, 2, 3, 1)  # n_samples x n_joints x 3 x n_frames
    return joints_loc


def sample_goal(batch_size, device, force_joints=None):
    if force_joints is None:
        choices = np.array(['None', 'traj', 'pelvis'] + HML_EE_JOINT_NAMES)  # todo: fix hardcoded 'pelvis' ('traj' is ok because it's our convention)  
        none_prob = 0.5  # todo: maybe convert to an argument
        probabilities = torch.ones(len(choices)) * (1-none_prob) / (len(choices)  -1)
        probabilities[0] = none_prob  # None's probability 
        assert probabilities.sum() - 1 < 1e-6, 'probabilities should sum to 1'
        max_goal_joints_per_sample = 2
        # target_cond_idx = torch.randint(low=0, high=len(choices), size=(batch_size,max_goal_joints_per_sample))
        target_cond_idx = torch.multinomial(probabilities, max_goal_joints_per_sample * batch_size, replacement=True).view(batch_size, max_goal_joints_per_sample)    
        names = choices[target_cond_idx]
        names = np.array([np.unique(name) for name in names])
        names = np.array([np.delete(name, np.argwhere(name=='None')) for name in names])
        is_heading = torch.bernoulli(torch.ones(batch_size, device=device) * .5).to(bool)
    else:
        options = get_allowed_joint_options(force_joints)
        names = [copy(random.choice(options)) for _ in range(batch_size)]
        is_heading = torch.zeros(batch_size, device=device).to(bool)
        for i, n in enumerate(names):
            if 'heading' in n:
                is_heading[i] = True
                del n[n.index('heading')]
    return names, is_heading

def get_allowed_joint_options(config_name):
    if config_name == 'DIMP_FULL':
        return [['pelvis', 'heading'], ['pelvis', 'head'], ['traj', 'heading'], ['right_wrist', 'heading'], ['left_wrist', 'heading'], ['right_foot', 'heading'], ['left_foot', 'heading']]
    elif config_name == 'DIMP_FINAL':
        return [['pelvis', 'heading'], ['traj', 'heading'], ['right_wrist', 'heading'], ['left_wrist', 'heading'], ['right_foot', 'heading'], ['left_foot', 'heading'], []]
    elif config_name == 'DIMP_SLIM':
        return [['pelvis', 'heading'], ['pelvis', 'head'], ['traj', 'heading'], ['left_wrist', 'heading'], ['left_foot', 'heading']]
    elif config_name == 'DIMP_BENCH':
        return [['pelvis', 'heading'], ['pelvis', 'head']]
    elif config_name == 'PURE_T2M':
        return [[]]
    else:
        return [config_name.split(',')]
    

def validate_recovery(positions, feature_data, joints_num):
    """
    Simple validation: recover joints from features and calculate error.
    
    Args:
        positions: Original processed joint positions (n_frames, n_joints, 3)
        feature_data: Extracted feature vectors (n_frames-1, feature_dim)
        joints_num: Number of joints
    
    Returns:
        recovered_joints: Recovered joint positions (n_frames-1, n_joints, 3)
        error: Maximum absolute error between original and recovered
    """
    # Recover joints from feature vectors
    recovered_joints = recover_from_ric(torch.from_numpy(feature_data).unsqueeze(0).float(), joints_num)
    recovered_joints = recovered_joints.squeeze(0).numpy()
    
    # Calculate error (compare with original[:-1] since features have n_frames-1)
    error = np.max(np.abs(recovered_joints - positions[:-1]))
    
    return recovered_joints, error


def get_all_files(data_dir, file_name):
    all_files = []
    for root, _, files in tqdm(os.walk(data_dir)):
        for f in files:
            if f == file_name:
                all_files.append(os.path.join(root, f))
                print(f"Found {len(all_files)} {file_name}", end='\r', flush=True)
                break
    return all_files


def mean_variance(data, save_dir, joints_num):
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
    return Mean, Std


def joints_to_vecs(
        joints, 
        feet_thre=feet_threshold, 
        n_raw_offsets=t2m_animal_unified_raw_offsets, 
        kinematic_chain=t2m_animal_unified_kinematic_chain, 
        face_joint_indx=[19, 15, 11, 7], fid_fl=10, fid_fr=14, fid_rl=18, fid_rr=19):
    # default values are for unified animal skeleton
    # joints: (seq_len, joints_num, 3)
    # vecs: (seq_len, dim_vecs)
    n_raw_offsets = torch.from_numpy(n_raw_offsets)
    vecs = extract_features(joints, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_fl, fid_fr, fid_rl, fid_rr)
    return vecs

def vecs_to_joints(vecs, joints_num):
    # vecs: (seq_len, dim_vecs)
    # joints: (seq_len, joints_num, 3)
    joints = recover_from_ric(torch.from_numpy(vecs).unsqueeze(0).float(), joints_num)
    joints = joints.squeeze(0).numpy()
    return joints
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert reconstructed joints (joints_global.npy) of every clip "
                                                 "into motion features (new_joint_vecs_global.npy).")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing one folder per clip.")
    parser.add_argument("--test_dir", default=None,
                        help="Test-set root; its clip ids are written to test.txt and left out of train.txt. "
                             "Without it, 1%% of the clips are split off at random.")
    cli = parser.parse_args()
    data_dir = cli.data_dir
    all_joints_files = get_all_files(data_dir=data_dir, file_name=joints_file_name)

    n_raw_offsets = torch.from_numpy(t2m_animal_raw_offsets)
    kinematic_chain = t2m_animal_kinematic_chain

    # Get target skeleton from first file
    if len(all_joints_files) > 0:
        example_data = np.load(all_joints_files[0])[:, 1:]  # Skip joint 0
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
        # (joints_num, 3)
        tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

        # debug: save target offsets
        # to_save = torch.cat([tgt_offsets[0].unsqueeze(0), tgt_offsets], dim=0).unsqueeze(0)
        # np.save('temp_tgt_offset.npy', to_save.numpy())

        frame_num = 0
        success_count = 0
        all_data = []
        all_ids = []

        print(f"Processing {len(all_joints_files)} joint files...")
        
        for joint_file_path in tqdm(all_joints_files):
            # Load joint data
            source_data = np.load(joint_file_path)[:, 1:]  # Skip joint 0
            assert source_data.ndim == 3 and source_data.shape[1] == joints_num

            if np.any(np.isnan(source_data)):
                print(f"NaN detected in source data for {joint_file_path}")
                continue
            
            # Process to feature vectors
            data, ground_positions, positions, l_velocity = process_file(source_data, feet_thre=feet_threshold)
            
            # Check for NaN values in data
            if np.isnan(data).any():
                print(f"NaN detected for {joint_file_path}")
                continue
            
            # Validate recovery
            recovered_joints, error = validate_recovery(positions, data, joints_num)
            # print(f"Recovery error: {error:.6f} for {joint_file_path}")

            all_data.append(data)
            all_ids.append(os.path.basename(os.path.dirname(joint_file_path)))
            
            # debug recovered joints
            # debug_save_joints(recovered_joints)

            vecs_file_path = joint_file_path.replace(joints_file_name, vecs_file_name)
            joints_file_path = joint_file_path.replace(joints_file_name, new_joints_file_name)

            # Save feature vectors
            np.save(vecs_file_path, data)
            # Save processed joint positions (recovered)
            np.save(joints_file_path, recovered_joints)
            
            frame_num += data.shape[0]
            success_count += 1
                
        print(f'Successfully processed {success_count}/{len(all_joints_files)} files')
        print(f'Total frames: {frame_num}')

        data = np.concatenate(all_data, axis=0)
        mean, std = mean_variance(data, data_dir, joints_num)
        np.save(pjoin(data_dir, 'Mean.npy'), mean)
        np.save(pjoin(data_dir, 'Std.npy'), std)
        if cli.test_dir is not None:
            test_ids = sorted({os.path.basename(os.path.dirname(f))
                               for name in ("texts_gemini.txt", "captions.txt") for f in get_all_files(cli.test_dir, name)})
            train_ids = [i for i in all_ids if i not in set(test_ids)]
        else:
            train_ids, test_ids = train_test_split(all_ids, test_size=0.01, random_state=42)
        with open(pjoin(data_dir, 'train.txt'), 'w') as f:
            for item in train_ids:
                f.write("%s\n" % item)
        with open(pjoin(data_dir, 'test.txt'), 'w') as f:
            for item in test_ids:
                f.write("%s\n" % item)
        print(f"Mean, Std, train test split saved to {data_dir}")
    else:
        print("No joint files found!")


