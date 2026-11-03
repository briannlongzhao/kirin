"""Compute the DINOv3 image feature of every clip used for image conditioning.

For each clip folder under --data_dir, up to 32 random *_rgb.png frames are
encoded with DINOv3 ViT-B/16 and their [CLS] features are averaged into
mean_feature_dinov3.npy of shape (1, 768).
"""

import argparse
import os
from pathlib import Path
from random import sample, shuffle

import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

from data_loaders.humanml.data.dataset import get_all_sequence_dirs

SAVE_NAME = "mean_feature_dinov3.npy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Dataset root containing one folder per clip.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_num_images", type=int, default=32)
    args = parser.parse_args()

    feature_extractor = pipeline(
        model="facebook/dinov3-vitb16-pretrain-lvd1689m",
        task="image-feature-extraction",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=args.batch_size,
    )
    sequence_dirs = get_all_sequence_dirs(args.data_dir)
    shuffle(sequence_dirs)
    for seq_dir in tqdm(sequence_dirs):
        if os.path.exists(os.path.join(seq_dir, SAVE_NAME)):
            continue
        image_paths = sorted(Path(seq_dir).glob("*rgb.png"))
        image_paths = sample(image_paths, min(len(image_paths), args.max_num_images))
        features = feature_extractor([str(p) for p in image_paths])
        mean_feature = np.array(features)[..., 0, :].mean(axis=0)
        assert mean_feature.shape == (1, 768)
        np.save(os.path.join(seq_dir, SAVE_NAME), mean_feature)


if __name__ == "__main__":
    main()
