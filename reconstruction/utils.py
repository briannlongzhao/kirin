import os
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm


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
