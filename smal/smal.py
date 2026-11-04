import os
import torch
import pickle
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple
from smplx.lbs import blend_shapes, vertices2joints, batch_rigid_transform, batch_rodrigues

from AniMer.amr.models.smal_warapper import SMALOutput, SMALLayer

SMAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "AniMer/data/smal/my_smpl_00781_4_all.pkl")

# SMAL joint hierarchy - parent of each joint
SMAL_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 11, 12, 13, 6, 15, 0,
          17, 18, 19, 0, 21, 22, 23, 0, 25, 26, 27, 28, 29, 30, 16, 16, 16]

SMPLH_PARENTS = [-1,          0,          0,          0,          1,
                 2,          3,          4,          5,          6,
                 7,          8,          9,          9,          9,
                12,         13,         14,         16,         17,
                18,         19,         20,         22,         23,
                20,         25,         26,         20,         28,
                29,         20,         31,         32,         20,
                34,         35,         21,         37,         38,
                21,         40,         41,         21,         43,
                44,         21,         46,         47,         21,
                49,         50]

ANIMO_PARENTS = [-1, 0, 1, 2, 3, 3, 3, 1, 1, 0, 9, 10, 7, 12, 13, 14, 8, 16, 17, 18, 0, 20, 21, 22, 23, 0, 25, 26, 27, 28]

# Joint names for reference
SMAL_JOINT_NAMES = [
    'root', 'pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head', 
    'left_shoulder', 'left_elbow', 'left_wrist', 'left_paw',
    'right_shoulder', 'right_elbow', 'right_wrist', 'right_paw',
    'left_hip', 'left_knee', 'left_ankle', 'left_foot',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot',
    'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail6', 'tail7',
    'jaw', 'nose', 'left_ear', 'right_ear'
]

SMAL_JOINTS = {
    10: "LF feet", 9: "LF middle", 8: "LF top",
    20: "LR feet", 19: "LR middle", 18: "LR top",
    14: "RF feet", 13: "RF middle", 12: "RF top",
    24: "RR feet", 23: "RR middle", 22: "RR top",
    15: "withers", 28: "tail middle",
    25: "tail start", 31: "tail end",
    33: "left ear base", 34: "right ear base",
    # smalify extra
    35: "nose", 36: "chin",
    38: "left ear tip", 37: "right ear tip",
    39: "left eye", 40: "right eye",
}

VITPOSE_JOINTS = {
    0: "left eye", 1: "right eye", 2: "nose",
    3: "withers", 4: "tail start",
    5: "LF top", 6: "LF middle", 7: "LF feet",
    8: "RF top", 9: "RF middle", 10: "RF feet",
    11: "LR top", 12: "LR middle", 13: "LR feet",
    14: "RR top", 15: "RR middle", 16: "RR feet",
}


SMAL_TO_VIT_POSE = {
    39: 0,  # left eye
    40: 1,  # right eye
    35: 2,  # nose
    15: 3,  # withers
    25: 4,  # tail start
    8: 5,   # LF top
    9: 6,   # LF middle
    10: 7,  # LF feet
    12: 8,  # RF top
    13: 9,  # RF middle
    14: 10, # RF feet
    18: 11, # LR top
    19: 12, # LR middle
    20: 13, # LR feet
    22: 14, # RR top
    23: 15, # RR middle
    24: 16, # RR feet
}

def load_smal_model(smal_data_path=None):
    """Load SMAL model for full parameter animation."""
    if smal_data_path is None:
        smal_data_path = SMAL_DATA_PATH    
    if not os.path.exists(smal_data_path):
        raise FileNotFoundError(f"SMAL data file not found: {smal_data_path}")
    with open(smal_data_path, 'rb') as f:
        smal_data = pickle.load(f, encoding='latin1')
    smal_model = CustomSMALLayer(num_betas=41, **smal_data)
    return smal_model


@dataclass
class CustomSMALOutput(SMALOutput):
    joints_transform_matrix: Optional[Tensor] = None


class CustomSMALLayer(SMALLayer):
    def __init__(self, num_betas, **kwargs):
        super().__init__(num_betas=num_betas, **kwargs)

    @staticmethod
    def lbs(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        posedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        lbs_weights: Tensor,
        pose2rot: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Custom lbs with A transform matrix output
        """
        batch_size = max(betas.shape[0], pose.shape[0])
        device, dtype = betas.device, betas.dtype

        # Add shape contribution
        v_shaped = v_template + blend_shapes(betas, shapedirs)

        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
                [batch_size, -1, 3, 3])

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(
                pose_feature, posedirs).view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                        posedirs).view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped
        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        verts = v_homo[:, :, :3, 0]

        return verts, J_transformed, A

    def forward(
            self,
            betas: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            pose: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts: bool = True,
            return_full_pose: bool = False,
            **kwargs):
        """
        Args:
            betas: [batch_size, 10]
            global_orient: [batch_size, 1, 3, 3]
            pose: [batch_size, num_joints, 3, 3]
            transl: [batch_size, num_joints, 3]
            return_verts:
            return_full_pose:
            **kwargs:
        Returns:
        """
        device, dtype = betas.device, betas.dtype
        if global_orient is None:
            batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        else:
            batch_size = global_orient.shape[0]
        if pose is None:
            pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 34, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, pose], dim=1)
        vertices, joints, joints_transform_matrix = self.lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = CustomSMALOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            pose=pose,
            transl=transl,
            full_pose=full_pose if return_full_pose else None,
            joints_transform_matrix=joints_transform_matrix
        )
        return output
