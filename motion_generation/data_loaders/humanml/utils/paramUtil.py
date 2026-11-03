import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]


t2m_animal_kinematic_chain = [
    [0, 1, 2, 3, 4, 5, 14, 15, 31],  # root -> spine -> head -> nose
    [0, 24, 25, 26, 27, 28, 29, 30],  # root -> tail
    [0, 20, 21, 22, 23],  # root -> rear right leg
    [0, 16, 17, 18, 19],  # root -> rear left leg
    [5, 10, 11, 12, 13],  # whitter -> front right leg
    [5, 6, 7, 8, 9],  # whitter -> front left leg
    [15, 33],  # head -> right ear
    [15, 32],  # head -> left ear
]

t2m_animal_unified_kinematic_chain = [
    [0, 1, 2, 3, 4],  # root -> spine -> head -> nose
    [0, 23, 24, 25],  # root -> tail
    [0, 19, 20, 21, 22],  # root -> rear right leg
    [0, 15, 16, 17, 18],  # root -> rear left leg
    [1, 7, 8, 9, 10],  # whitter -> front left leg
    [1, 11, 12, 13, 14],  # whitter -> front right leg
    [3, 5],  # head -> right ear
    [3, 6],  # head -> left ear
]

t2m_animal_raw_offsets = np.array([
    [0, 0, 0],    # 0
    [0, 0, 1],    # 1
    [0, -1, 1],   # 2
    [0, 0, 1],    # 3
    [0, 0, 1],    # 4
    [0, 1, 1],    # 5
    [1, -1, 0],   # 6
    [0, -1, 0],   # 7
    [0, -1, 0],   # 8
    [0, -1, 0],   # 9
    [-1, -1, 0],  # 10
    [0, -1, 0],   # 11
    [0, -1, 0],   # 12
    [0, -1, 0],   # 13
    [0, 0, 1],    # 14
    [0, 0, 1],    # 15
    [1, 0, 0],    # 16
    [0, -1, 0],   # 17
    [0, -1, 0],   # 18
    [0, -1, 0],   # 19
    [-1, 0, 0],   # 20
    [0, -1, 0],   # 21
    [0, -1, 0],   # 22
    [0, -1, 0],   # 23
    [0, 0, -1],   # 24
    [0, 0, -1],   # 25
    [0, 0, -1],   # 26
    [0, 0, -1],   # 27
    [0, 0, -1],   # 28
    [0, 0, -1],   # 29
    [0, 0, -1],   # 30
    [0, 0, 1],    # 31
    [1, 1, 0],    # 32
    [-1, 1, 0],   # 33
])

t2m_animal_unified_raw_offsets = np.array([
    [0, 0, 0],    # 0
    [0, 0, 1],    # 1
    [0, 0, 1],    # 2
    [0, 0, 1],    # 3
    [0, 0, 1],    # 4
    [-1, 0, 0],   # 5
    [1, 0, 0],    # 6
    [1, 0, 0],    # 7
    [0, -1, 0],   # 8
    [0, -1, 0],   # 9
    [0, -1, 0],   # 10
    [-1, 0, 0],   # 11
    [0, -1, 0],   # 12
    [0, -1, 0],   # 13
    [0, -1, 0],   # 14
    [1, 0, 0],    # 15
    [0, -1, 0],   # 16
    [0, -1, 0],   # 17
    [0, -1, 0],   # 18
    [-1, 0, 0],   # 19
    [0, -1, 0],   # 20
    [0, -1, 0],   # 21
    [0, -1, 0],   # 22
    [0, 0, -1],   # 23
    [0, 0, -1],   # 24
    [0, 0, -1],   # 25
])

kit_tgt_skel_id = '03950'


t2m_tgt_skel_id = '000021'


animo_to_unified_joint_map = {
    # root -> spine -> head
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    # LF leg
    7: 7, 13: 8, 14: 9, 15: 10,
    # RF leg
    8: 11, 17: 12, 18: 13, 19: 14,
    # LR leg
    20: 15, 22: 16, 23: 17, 24: 18,
    # RR leg
    25: 19, 27: 20, 28: 21, 29: 22,
    # tail
    9: 23, 10: 24, 11: 25
}

# 35 joints
smal_to_unified_joint_map = {
    # root -> spine -> head
    0: 0, 6: 1, 15: 2, 16: 3, 32: 4, 34: 5, 33: 6,
    # LF leg
    7: 7, 8: 8, 9: 9, 10: 10,
    # RF leg
    11: 11, 12: 12, 13: 13, 14: 14,
    # LR leg
    17: 15, 18: 16, 19: 17, 20: 18,
    # RR leg
    21: 19, 22: 20, 23: 21, 24: 22,
    # tail
    25: 23, 26: 24, 27: 25,
}

unified_parents = [-1, 0, 1, 2, 3, 3, 3, 1, 7, 8, 9,  1, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21, 0, 23, 24]


def unify_animo_joints(animo_joints):
    assert animo_joints.shape[1] == 30
    unified_joints = np.zeros((animo_joints.shape[0], 26, 3))
    for animo_id, unified_id in animo_to_unified_joint_map.items():
        unified_joints[:, unified_id, :] = animo_joints[:, animo_id, :]
    return unified_joints

def unify_smal_joints(smal_joints):
    assert smal_joints.shape[1] == 35 or smal_joints.shape[1] == 34
    if smal_joints.shape[1] == 34:
        smal_joints = np.concatenate([smal_joints[:, :1, :], smal_joints], axis=1)
    unified_joints = np.zeros((smal_joints.shape[0], 26, 3))
    for smal_id, unified_id in smal_to_unified_joint_map.items():
        unified_joints[:, unified_id, :] = smal_joints[:, smal_id, :]
    return unified_joints