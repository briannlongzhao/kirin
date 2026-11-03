import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from random import shuffle
from einops import rearrange
from random import choice
from pathlib import Path
from quaternion import qrot, qinv, quaternion_to_cont6d


video_caption_prompts = [
    "Describe the {}'s motion in a single short sentence.",
    "The {} is doing what kind of motion in the video? Answer in one short sentence.",
    "Use simple sentences to describe the {}'s motion and global movement in the video.",
    "What is the main action of the {} in the video? Be specific about **speed, style, and sequence of actions**.",
    "Describe the {}'s motion in this video. Focus on the detailed motion.",
    "Write a short caption of the {} motion in this video that describes the **motion type** and **global movement path**.",
]


action_labels = [
    "keeping still",
    "walking",
    "eating",
    "swimming",
    "drinking",
    "running",
    "shaking head",
    "jumping",
    "attacking",
    "flapping tail",
    "shaking",
    "climbing",
    "turning around",
    "grooming",
    "struggling",
    "digging",
    "sitting",
    "lying down",
    "rearing",
    "dancing",
    "neck raising",
    "face dipping",
    "back kicking",
    "resting",
    "leaning",
    "rearing",
    "rolling",
    "barking"
]


def get_video_caption_examples(data_dir, max_examples=None):
    print("Getting video caption examples from", data_dir)
    captions = []
    count = 0
    max_examples = 1e10 if max_examples is None else max_examples
    if "omnimotiongpt" in str(data_dir).lower():
        for path in Path(data_dir).rglob("*.txt"):
            with open(path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if len(line.strip()) > 0:
                    captions.append(line.strip().lower())
                    count += 1
                    if count >= max_examples:
                        break
            if count >= max_examples:
                break
    elif "animo" in str(data_dir).lower():
        for path in Path(data_dir).rglob("*.txt"):
            with open(path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if len(line.strip()) > 0:
                    captions.append(line.split('#')[2].strip().lower())
                    count += 1
                    if count >= max_examples:
                        break
            if count >= max_examples:
                break
    else:
        raise NotImplementedError
    return captions

# Optional examples are configured externally, never through machine-specific paths.
video_caption_examples = []
for _caption_dir in filter(None, os.environ.get("KIRIN_CAPTION_EXAMPLE_DIRS", "").split(os.pathsep)):
    video_caption_examples.extend(get_video_caption_examples(_caption_dir, max_examples=100))


def get_all_sequence_dirs(data_dir, image_suffix="rgb.png", sorted=False, use_cache=True):
    cache_path = os.path.join(data_dir, 'all_sequence_dirs.txt')
    if use_cache and os.path.exists(cache_path):
        print(f'Loading all sequence dirs from cache file [{cache_path}]...')
        with open(cache_path, 'r') as f:
            result = [line.strip() for line in f.readlines()]
        return result
    result = set()
    for root, dirs, files in tqdm(os.walk(data_dir, followlinks=True), desc="Getting all sequence dirs"):
        for f in files:
            if f.endswith(image_suffix):
                result.add(root)
                break
    result = list(result)
    if sorted:
        result.sort()
    else:
        shuffle(result)
    if use_cache:
        with open(cache_path, 'w') as f:
            for r in result:
                f.write(r + '\n')
        print(f'Saved all sequence dirs to cache file [{cache_path}]...')    
    return result


def draw_keypoints(image_np, keypoints, circle_color=(0, 0, 255), text_color=(0, 255, 255), radius=3):
    assert isinstance(image_np, np.ndarray), "Expected image to be a numpy array"
    assert image_np.ndim == 3 and image_np.shape[-1] == 3, "Expected image to be (h,w,3)"
    assert keypoints.ndim == 2 and keypoints.shape[-1] == 2, "Expected keypoints to be (k,2)"
    image_size = image_np.shape[0]
    keypoints = (keypoints + 0.5) * image_size
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(image_np, center=(x, y), radius=radius, color=circle_color, thickness=-1)
        cv2.putText(
            image_np, text=str(i), org=(x + radius + 1, y - radius - 1),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=text_color, thickness=1, lineType=cv2.LINE_AA
        )
    return image_np


def process_caption(caption):
    import spacy
    caption = caption.lower().strip()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(caption)
    data = caption + '#'
    for token in doc:
        if token.pos_ in ["NOUN", "VERB"] and (token.text != "left"):
            data += f"{token.lemma_}/{token.pos_} "
        elif not token.text.isalpha():
            continue
        else:
            data += f"{token.text}/{token.pos_} "
    data = data.strip()
    data += "#0.0#0.0"
    return data



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

t2m_animal_raw_offsets = [
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
]


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