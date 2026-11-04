"""
Run SpaTrackerv2 on all video data, save results npz to same directory as video data.
Use SpaTrack2 conda environment
"""

import argparse
import os
import sys
import traceback
from contextlib import contextmanager

# Set PyTorch CUDA allocator to avoid fragmentation BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import cv2
import decord
import json
from glob import glob
from random import shuffle
import torchvision.transforms as T
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


try:
    import yt_dlp
except ImportError:  # only needed to download full-frame videos from YouTube
    yt_dlp = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SpaTrackerV2"))
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from models.SpaTrackV2.models.predictor import Predictor
from models.SpaTrackV2.models.utils import get_points_on_a_grid
from models.SpaTrackV2.models.tracker3D.spatrack_modules.utils import depth_to_points_colmap
from models.SpaTrackV2.models.vggt4track.utils.loss import compute_loss


@contextmanager
def patched_functional_dtype_match():
    """
    Temporarily patch torch.nn.functional linear/conv2d to cast floating inputs
    to weight dtype when they mismatch.
    """
    original_linear = torch.nn.functional.linear
    original_conv2d = torch.nn.functional.conv2d

    def _patched_linear(input, weight, bias=None):
        if input.is_floating_point() and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return original_linear(input, weight, bias)

    def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.is_floating_point() and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return original_conv2d(input, weight, bias, stride, padding, dilation, groups)

    torch.nn.functional.linear = _patched_linear
    torch.nn.functional.conv2d = _patched_conv2d
    try:
        yield
    finally:
        torch.nn.functional.linear = original_linear
        torch.nn.functional.conv2d = original_conv2d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset root (batch mode).")
    parser.add_argument("--sequence_dir", type=str, default=None,
                        help="Process this one clip directory.")
    parser.add_argument("--rgb_video", type=str, default=None,
                        help="Cropped RGB video path for the selected clip.")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Clip-level *_info.json metadata for the selected clip.")
    parser.add_argument("--uncropped_video", type=str, default=None,
                        help="Direct full-frame clip path; avoids lookup/download.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory for tracker results in --sequence_dir mode; defaults to --sequence_dir.")
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--sliding_window", action="store_true", help="Use sliding window for long videos to avoid OOM")
    parser.add_argument("--uncropped", action="store_true", 
                        help="Whether the input videos are already cropped around the animal, if set will skip the focal length scaling and find uncropped vidoe")
    parser.add_argument("--uncropped_video_dir", type=str, 
                        default=None
    )
    args = parser.parse_args()
    if args.sequence_dir:
        if args.data_dir:
            parser.error("Use either --sequence_dir or --data_dir.")
        if not args.rgb_video or not args.metadata:
            parser.error("--sequence_dir mode requires --rgb_video and --metadata.")
    elif not args.data_dir:
        parser.error("Provide --sequence_dir for one clip or --data_dir for batch mode.")
    if args.uncropped and not (args.uncropped_video or args.uncropped_video_dir):
        parser.error("--uncropped requires --uncropped_video or --uncropped_video_dir.")
    return args


def get_all_sequence_dirs(data_dir, image_suffix="info.json", sorted=False, use_cache=True):
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


def download_video(video_id, save_path):
    """Download video from YouTube using yt-dlp, avoiding AV1 codec"""
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not installed. Please install it to download videos.")

    url = "https://youtube.com/watch?v=" + video_id
    ydl_opts = {
        "outtmpl": save_path,
        "format": "bestvideo[vcodec!*=av01][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "noplaylist": True,
        "retries": 3,
        "quiet": False,
        "no_warnings": True,
        # Do not load authentication data from a cookie file shipped with source.
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Video downloaded to {save_path}")


def cut_video(video_path, start_time, duration, output_path, trim_end_seconds=0):
    """Cut video using ffmpeg based on start time and duration
    
    Args:
        video_path: path to input video
        start_time: start time in seconds
        duration: clip duration in seconds
        output_path: path to save output video
        trim_end_seconds: seconds to trim from the end (default: 1/30s for 2 frames at 60fps)
    """
    end_time = start_time + duration - trim_end_seconds
    # Use re-encoding for precise frame-accurate cuts to avoid frozen frames at the start
    # -ss before -i for fast seeking, then -ss 0 -t duration for accurate cut with re-encoding
    command = f"ffmpeg -y -ss {start_time:.3f} -i {video_path} -t {duration - trim_end_seconds:.3f} -c:v libx264 -preset fast -crf 18 -c:a aac {output_path}"
    print(f"Cutting video: {command}")
    os.system(command)
    print(f"Video cut and saved to {output_path}")


def is_valid_video(video_path, min_size_kb=1):
    """Check if a video file is valid
    
    Args:
        video_path: path to video file
        min_size_kb: minimum file size in KB
    
    Returns:
        tuple: (is_valid: bool, reason: str or None)
    """
    # Check if file exists
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    # Check file size
    file_size_kb = os.path.getsize(video_path) / 1024
    if file_size_kb < min_size_kb:
        return False, f"File size {file_size_kb:.1f}KB < {min_size_kb}KB (likely corrupted/incomplete)"
    
    # Check if video can be read and has frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames < 2:
        return False, f"Video has < 2 frames (has {total_frames})"
    
    return True, None


def get_uncropped_video(sequence_dir, uncropped_video_dir, use_cache=True):
    video_id = os.path.basename(sequence_dir)
    clip_id = video_id[:-4]
    category = os.path.basename(os.path.dirname(os.path.dirname(sequence_dir)))
    uncropped_video_path = os.path.join(uncropped_video_dir, category, video_id[:11], f"{clip_id}.mp4")

    # Check if uncropped video exists and is valid
    is_valid, reason = is_valid_video(uncropped_video_path)
    
    if not is_valid:
        if os.path.exists(uncropped_video_path):
            print(f"Invalid video found: {reason}. Deleting and re-downloading...")
            os.remove(uncropped_video_path)
        else:
            print(f"Uncropped video not found. Downloading from YouTube...")

        # Load info.json to get video_id and clip timing
        info_json_path = os.path.join(sequence_dir, f"{video_id}_info.json")
        assert os.path.exists(info_json_path), f"info.json not found at {info_json_path}"

        with open(info_json_path, "r") as f:
            info_data = json.load(f)

        yt_video_id = info_data.get("video_id", video_id[:11])
        clip_start_time = info_data.get("clip_start_time")
        clip_duration = info_data.get("duration")

        assert clip_start_time is not None and clip_duration is not None, \
            f"clip_start_time and duration not found in {info_json_path}"

        # Download the original video
        temp_raw_path = os.path.join(uncropped_video_dir, f"{yt_video_id}_raw.mp4")
        download_video(yt_video_id, temp_raw_path)

        # Cut the video based on clip timing
        os.makedirs(os.path.dirname(uncropped_video_path), exist_ok=True)
        cut_video(temp_raw_path, clip_start_time, clip_duration, uncropped_video_path)

        # Clean up raw video only
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
            print(f"Cleaned up temporary raw video: {temp_raw_path}")

    return align_uncropped_video(
        uncropped_video_path, sequence_dir,
        os.path.join(sequence_dir, f"{video_id}_info.json"),
        os.path.join(sequence_dir, f"{video_id}_uncropped.mp4"),
    )


def align_uncropped_video(uncropped_video_path, sequence_dir, info_json_path, temp_video_path):
    """Resample a full-frame clip to the tracked frames listed in the per-frame metadata."""
    all_meta_data_paths = sorted(glob(os.path.join(sequence_dir, "*_metadata.json")))
    all_frame_ids = []
    for meta_data_path in all_meta_data_paths:
        with open(meta_data_path, "r") as f:
            metadata = json.load(f)
        all_frame_ids.append(metadata["clip_frame_id"])
    with open(info_json_path, "r") as f:
        metadata = json.load(f)
        cropped_fps = metadata["fps"]

    cap = cv2.VideoCapture(uncropped_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open uncropped video: {uncropped_video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if original_fps <= 0:
        raise ValueError(f"Invalid FPS ({original_fps}) for uncropped video: {uncropped_video_path}")
    if cropped_fps <= 0:
        raise ValueError(f"Invalid cropped FPS ({cropped_fps}) in {info_json_path}")
    
    # Step 1: Downsample entire uncropped video from original_fps to cropped_fps
    # clip_frame_id values are indices into this downsampled video
    frame_interval = original_fps / cropped_fps
    
    print(f"Video info: {total_frames} frames at {original_fps} fps")
    print(f"Downsampling from {original_fps} fps to {cropped_fps} fps (interval: {frame_interval})")
    
    # Read all frames sequentially (more reliable than seeking)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    print(f"Read {len(all_frames)} frames from video")
    if len(all_frames) == 0:
        raise ValueError(f"No frames could be read from uncropped video: {uncropped_video_path}")
    
    # Resample to cropped_fps. This handles both downsampling and cases where
    # the uncropped source FPS is lower than the cropped metadata FPS.
    target_frame_count = max(1, int(round(len(all_frames) * cropped_fps / original_fps)))
    downsampled_frames = []
    for frame_id in range(target_frame_count):
        source_idx = int(round(frame_id * original_fps / cropped_fps))
        source_idx = min(source_idx, len(all_frames) - 1)
        downsampled_frames.append(all_frames[source_idx])
    
    print(f"Downsampled to {len(downsampled_frames)} frames")
    del all_frames  # Free memory

    # Step 1.5: Detect and remove shot changes from the end
    # Check from end backwards for large pixel differences indicating shot changes
    num_frames_to_check = 3
    shot_change_threshold = 30.0
    
    if len(downsampled_frames) >= num_frames_to_check:
        frames_to_keep = len(downsampled_frames)
        
        # Check from the last num_frames_to_check frames backwards
        for offset in range(num_frames_to_check, 0, -1):
            if frames_to_keep < len(downsampled_frames):
                # Already found a shot change
                break
            
            idx = len(downsampled_frames) - offset
            if idx > 0:
                prev_frame = downsampled_frames[idx - 1].astype(np.float32)
                curr_frame = downsampled_frames[idx].astype(np.float32)
                
                # Calculate mean absolute pixel difference
                pixel_diff = np.abs(prev_frame - curr_frame).mean()
                
                if pixel_diff > shot_change_threshold:
                    frames_to_keep = idx
                    print(f"Shot change detected at frame {idx}: "
                          f"pixel diff = {pixel_diff:.2f}, keeping first {frames_to_keep} frames")
                    break
        
        # Drop frames after the shot change
        if frames_to_keep < len(downsampled_frames):
            downsampled_frames = downsampled_frames[:frames_to_keep]
    
    # Handle frame count mismatch by duplicating last frame (before extracting specific frame IDs)
    max_frame_id = max(all_frame_ids)
    if len(downsampled_frames) < max_frame_id + 1:
        diff = max_frame_id + 1 - len(downsampled_frames)
        print(f"Frame count mismatch: have {len(downsampled_frames)}, need {max_frame_id + 1}. "
              f"Duplicating last frame {diff} times")
        for _ in range(diff):
            downsampled_frames.append(downsampled_frames[-1])
    
    # Step 2: Extract frames at specific indices from all_frame_ids
    # clip_frame_id values are direct indices into the downsampled video
    clipped_frames = [downsampled_frames[frame_id] for frame_id in all_frame_ids]
    
    # Step 3: Save as new video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, cropped_fps, (width, height))
    
    for frame in clipped_frames:
        out.write(frame)
    out.release()
    
    return temp_video_path



def scale_focal_length(f_list, data_dir):
    all_meta_data_paths = sorted(glob(os.path.join(data_dir, "*_metadata.json"), recursive=True))
    all_crop_sizes = []
    assert len(all_meta_data_paths) == len(f_list), f"Mismatch: {len(all_meta_data_paths)} metadata files vs {len(f_list)} focal lengths"
    
    # Load all crop sizes
    for all_meta_data_path in all_meta_data_paths:
        with open(all_meta_data_path, "r") as f:
            metadata = json.load(f)
        crop_width = metadata["crop_box_xyxy"][2] - metadata["crop_box_xyxy"][0]
        all_crop_sizes.append(crop_width)
    
    # Reference crop size from first frame
    ref_crop_size = all_crop_sizes[0]
    
    # Scale focal lengths: f_scaled = f_ref * (crop_size_i / crop_size_ref)
    # When crop is larger, focal length should be proportionally larger
    scaled_f_list = np.array(f_list).copy()
    for i in range(len(scaled_f_list)):
        scale_factor = all_crop_sizes[i] / ref_crop_size
        scaled_f_list[i] = f_list[0] * scale_factor
    
    return scaled_f_list
    

def smooth_camera_trajectory(c2w_traj, keyframe_interval=5):
    """
    Smooth camera trajectory by selecting keyframes and interpolating between them.
    
    Args:
        c2w_traj: (T, 4, 4) camera-to-world transformation matrices
        keyframe_interval: interval for selecting keyframes (e.g., every 5 frames)
    
    Returns:
        Smoothed c2w_traj with interpolated translations
    """
    from scipy.interpolate import interp1d
    
    T = c2w_traj.shape[0]
    c2w_traj_smooth = c2w_traj.copy()
    
    # Select keyframe indices: every keyframe_interval frames + ensure last frame
    keyframe_indices = list(range(0, T, keyframe_interval))
    if keyframe_indices[-1] != T - 1:
        keyframe_indices.append(T - 1)
    keyframe_indices = np.array(keyframe_indices)
    
    # Extract translation components at keyframes
    keyframe_translations = c2w_traj[keyframe_indices, :3, 3]
    
    # Create interpolation function for each translation component (X, Y, Z)
    interp_func = interp1d(keyframe_indices, keyframe_translations, 
                          axis=0, kind='cubic', fill_value='extrapolate')
    
    # Interpolate translations for all frames
    all_frame_indices = np.arange(T)
    translations_smooth = interp_func(all_frame_indices)
    
    # Apply smoothed translations back
    c2w_traj_smooth[:, :3, 3] = translations_smooth
    
    return c2w_traj_smooth


def estimate_ground_plane_normal_ransac(points_3d, num_iters=200, distance_thresh=None, seed=0):
    """
    Estimate ground plane normal from 3D points using RANSAC + SVD refinement.
    Args:
        points_3d: (N, 3) numpy array
    Returns:
        normal: (3,) unit vector, or None if estimation fails
    """
    pts = np.asarray(points_3d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None

    if distance_thresh is None:
        bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        distance_thresh = max(1e-4, 0.01 * bbox_diag)

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1

    for _ in range(num_iters):
        idx = rng.choice(pts.shape[0], size=3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        dists = np.abs((pts - p0) @ n)
        inliers = dists < distance_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return None

    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return normal.astype(np.float32)


def rotation_matrix_align_vector_to_vector(src_vec, dst_vec):
    """
    Compute rotation matrix R so that R @ src_vec ~= dst_vec.
    """
    a = src_vec / (np.linalg.norm(src_vec) + 1e-12)
    b = dst_vec / (np.linalg.norm(dst_vec) + 1e-12)
    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        # 180-degree turn around any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if np.abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = axis - np.dot(axis, a) * a
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        v = axis
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ], dtype=np.float32)
        return (np.eye(3, dtype=np.float32) + 2.0 * (vx @ vx)).astype(np.float32)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return R.astype(np.float32)


def estimate_per_frame_translation_from_ground(coords_ground):
    """
    Estimate robust per-frame translation to keep ground points stable over time.
    coords_ground: (T, N, 3), expected same point indexing across frames.
    Returns:
        translations: (T, 3), translation to ADD to each frame.
    """
    T = coords_ground.shape[0]
    translations = np.zeros((T, 3), dtype=np.float32)
    ref = coords_ground[0]

    for t in range(1, T):
        cur = coords_ground[t]
        valid = np.isfinite(ref).all(axis=1) & np.isfinite(cur).all(axis=1)
        if valid.sum() < 5:
            continue

        delta = cur[valid] - ref[valid]  # observed drift from frame 0
        med = np.median(delta, axis=0)

        # Robust inlier filtering around median translation
        residual = np.linalg.norm(delta - med[None, :], axis=1)
        med_res = np.median(residual)
        mad = np.median(np.abs(residual - med_res))
        sigma = 1.4826 * mad
        thresh = max(1e-4, 3.0 * sigma)
        inliers = residual <= thresh

        if inliers.sum() >= 3:
            trans = -np.median(delta[inliers], axis=0)
        else:
            trans = -med
        translations[t] = trans.astype(np.float32)

    return translations


def smooth_frame_translations(translations, median_window=5, mean_window=9, ema_alpha=0.25):
    """
    Temporal smoothing for per-frame translations.
    Applies median filter -> moving average -> EMA.
    """
    x = translations.astype(np.float32).copy()
    T = x.shape[0]

    def _odd(k):
        return k if k % 2 == 1 else k + 1

    median_window = max(1, _odd(median_window))
    mean_window = max(1, _odd(mean_window))

    # Median filter (robust against spikes)
    if median_window > 1 and T > 1:
        pad = median_window // 2
        x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        x_med = np.zeros_like(x)
        for t in range(T):
            x_med[t] = np.median(x_pad[t:t + median_window], axis=0)
        x = x_med

    # Moving average (low-pass smoothing)
    if mean_window > 1 and T > 1:
        pad = mean_window // 2
        x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        x_mean = np.zeros_like(x)
        for t in range(T):
            x_mean[t] = x_pad[t:t + mean_window].mean(axis=0)
        x = x_mean

    # EMA for additional temporal smoothness
    if T > 1:
        ema = np.zeros_like(x)
        ema[0] = x[0]
        for t in range(1, T):
            ema[t] = ema_alpha * x[t] + (1.0 - ema_alpha) * ema[t - 1]
        x = ema

    # Keep frame 0 anchored
    x = x - x[0]
    return x.astype(np.float32)


def smooth_3d_points_temporal(points, median_window=3, mean_window=7, preserve_first=True):
    """
    Temporal smoothing directly on 3D trajectories.
    points: (T, N, 3)
    """
    x = points.astype(np.float32).copy()
    T = x.shape[0]

    def _odd(k):
        return k if k % 2 == 1 else k + 1

    median_window = max(1, _odd(median_window))
    mean_window = max(1, _odd(mean_window))

    # Median filter along time for robust outlier suppression.
    if median_window > 1 and T > 1:
        pad = median_window // 2
        x_pad = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        x_med = np.zeros_like(x)
        for t in range(T):
            x_med[t] = np.median(x_pad[t:t + median_window], axis=0)
        x = x_med

    # Moving average for smooth temporal trajectories.
    if mean_window > 1 and T > 1:
        pad = mean_window // 2
        x_pad = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        x_mean = np.zeros_like(x)
        for t in range(T):
            x_mean[t] = x_pad[t:t + mean_window].mean(axis=0)
        x = x_mean

    if preserve_first and T > 0:
        x[0] = points[0]

    return x.astype(np.float32)


def align_windows_with_se3(window1_results, window2_results, overlap_start_w1, overlap_end_w1, 
                           overlap_start_w2, overlap_end_w2, intrs_overlap):
    """
    Align window 2 to window 1's coordinate frame using SE(3) transformation.
    
    Returns:
        transform_w2_to_w1: 4x4 matrix to transform window 2's extrinsics to window 1's frame
        depth_scale: scalar to scale window 2's depth to match window 1
    """
    from scipy.spatial.transform import Rotation
    from scipy.optimize import least_squares
    
    # Extract overlapping frames
    extrs_w1 = window1_results['extrinsic'][overlap_start_w1:overlap_end_w1]  # World-to-camera
    extrs_w2 = window2_results['extrinsic'][overlap_start_w2:overlap_end_w2]
    depth_w1 = window1_results['depth_map'][overlap_start_w1:overlap_end_w1]
    depth_w2 = window2_results['depth_map'][overlap_start_w2:overlap_end_w2]
    conf_w1 = window1_results['depth_conf'][overlap_start_w1:overlap_end_w1]
    conf_w2 = window2_results['depth_conf'][overlap_start_w2:overlap_end_w2]
    
    num_overlap = extrs_w1.shape[0]
    if num_overlap < 2:
        # Not enough overlap, return identity
        return np.eye(4), 1.0
    
    # Compute depth scale factor
    mask_common = (conf_w1 > 0.5) & (conf_w2 > 0.5)
    if mask_common.sum() > 100:
        depth_scale = np.median(depth_w1[mask_common] / (depth_w2[mask_common] + 1e-6))
    else:
        depth_scale = 1.0
    
    # Compute relative transformation using camera poses
    # Window 1 and Window 2 predict poses in their own coordinate frames
    # We want T such that: extrs_w1[i] ≈ T @ extrs_w2[i]
    # This means: c2w_w1[i] ≈ c2w_w2[i] @ T^-1
    
    # Convert to camera-to-world for easier reasoning
    c2w_w1 = np.linalg.inv(extrs_w1)  # (N, 4, 4)
    c2w_w2 = np.linalg.inv(extrs_w2)
    
    # Use first frame alignment as initial guess
    # T transforms from window2's world to window1's world
    # c2w_w1[0] = T @ c2w_w2[0]
    # T = c2w_w1[0] @ inv(c2w_w2[0])
    T_init = c2w_w1[0] @ np.linalg.inv(c2w_w2[0])
    
    # Refine using all overlapping frames
    # Minimize: sum_i || c2w_w1[i] - T @ c2w_w2[i] ||^2
    def residual_fn(params):
        # params: [tx, ty, tz, rx, ry, rz] (translation + rotation axis-angle)
        t = params[:3]
        r = params[3:6]
        R = Rotation.from_rotvec(r).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        residuals = []
        for i in range(num_overlap):
            c2w_transformed = T @ c2w_w2[i]
            # Compare translation and rotation
            t_diff = c2w_w1[i][:3, 3] - c2w_transformed[:3, 3]
            R_diff = c2w_w1[i][:3, :3] @ c2w_transformed[:3, :3].T
            r_diff = Rotation.from_matrix(R_diff).as_rotvec()
            residuals.extend(t_diff)
            residuals.extend(r_diff * 0.1)  # Scale rotation error
        return np.array(residuals)
    
    # Initial parameters from T_init
    init_params = np.concatenate([
        T_init[:3, 3],
        Rotation.from_matrix(T_init[:3, :3]).as_rotvec()
    ])
    
    # Optimize
    result = least_squares(residual_fn, init_params, method='lm', max_nfev=100)
    
    # Build final transformation
    t_opt = result.x[:3]
    r_opt = result.x[3:6]
    R_opt = Rotation.from_rotvec(r_opt).as_matrix()
    T_w2_to_w1 = np.eye(4)
    T_w2_to_w1[:3, :3] = R_opt
    T_w2_to_w1[:3, 3] = t_opt
    
    return T_w2_to_w1, depth_scale


def merge_window_results(all_window_results, window_size, stride, total_frames):
    """
    Merge results from overlapping windows with SE(3) alignment for extrinsics.
    """
    # Initialize output arrays
    first_window = all_window_results[0]
    H, W = first_window['depth_map'].shape[1:]
    
    merged_extrinsic = np.zeros((total_frames, 4, 4))
    merged_intrinsic = np.zeros((total_frames, 3, 3))
    merged_depth_map = np.zeros((total_frames, H, W))
    merged_depth_conf = np.zeros((total_frames, H, W))
    weight_accumulator = np.zeros((total_frames, H, W))
    
    for window_idx, window_results in enumerate(all_window_results):
        start_frame = window_idx * stride
        end_frame = min(start_frame + window_size, total_frames)
        window_frames = end_frame - start_frame
        
        # Deep copy to avoid modifying original
        window_extrinsic = window_results['extrinsic'][:window_frames].copy()
        window_intrinsic = window_results['intrinsic'][:window_frames].copy()
        window_depth = window_results['depth_map'][:window_frames].copy()
        window_conf = window_results['depth_conf'][:window_frames].copy()
        
        # Align with merged results if this is not the first window
        if window_idx > 0:
            # Find overlap region with merged results
            prev_window_end = min((window_idx - 1) * stride + window_size, total_frames)
            
            # Overlap in global frame
            overlap_start_global = start_frame
            overlap_end_global = min(prev_window_end, end_frame)
            overlap_size = overlap_end_global - overlap_start_global
            
            # Overlap in current window's local indices
            overlap_start_w2 = 0
            overlap_end_w2 = overlap_size
            
            if overlap_size >= 2:  # Need at least 2 frames for reliable alignment
                # Extract and normalize merged results for overlapping region
                merged_extrs_overlap = merged_extrinsic[overlap_start_global:overlap_end_global].copy()
                merged_depth_overlap = merged_depth_map[overlap_start_global:overlap_end_global].copy()
                merged_conf_overlap = merged_depth_conf[overlap_start_global:overlap_end_global].copy()
                weights_overlap = weight_accumulator[overlap_start_global:overlap_end_global].copy()
                
                # Normalize by weights
                weights_overlap = np.maximum(weights_overlap, 1e-6)
                merged_depth_overlap /= weights_overlap
                merged_conf_overlap /= weights_overlap
                merged_extrs_overlap /= weights_overlap[..., 0:1, 0:1]
                
                # CRITICAL: Apply SVD to fix non-orthogonal rotations before alignment
                # Averaged rotation matrices are not orthogonal, which breaks align_windows_with_se3
                for i in range(overlap_size):
                    R = merged_extrs_overlap[i, :3, :3]
                    U, S, Vt = np.linalg.svd(R)
                    R_ortho = U @ Vt
                    if np.linalg.det(R_ortho) < 0:
                        Vt[-1, :] *= -1
                        R_ortho = U @ Vt
                    merged_extrs_overlap[i, :3, :3] = R_ortho
                
                merged_window_for_alignment = {
                    'extrinsic': merged_extrs_overlap,
                    'intrinsic': window_intrinsic[:overlap_size],  # Use current window's intrinsics
                    'depth_map': merged_depth_overlap,
                    'depth_conf': merged_conf_overlap
                }
                
                current_window_for_alignment = {
                    'extrinsic': window_extrinsic[:overlap_size],
                    'intrinsic': window_intrinsic[:overlap_size],
                    'depth_map': window_depth[:overlap_size],
                    'depth_conf': window_conf[:overlap_size]
                }
                
                # Compute transformation from current window to global frame
                T_w2_to_global, depth_scale = align_windows_with_se3(
                    merged_window_for_alignment, current_window_for_alignment,
                    0, overlap_size,
                    0, overlap_size,
                    window_intrinsic[:overlap_size]
                )
                
                # Transform all extrinsics in current window to global frame
                T_inv = np.linalg.inv(T_w2_to_global)
                for i in range(window_frames):
                    window_extrinsic[i] = window_extrinsic[i] @ T_inv
                
                # Scale depth
                window_depth *= depth_scale
                
                print(f"Window {window_idx}: depth_scale={depth_scale:.3f}, T_trans=[{T_w2_to_global[0,3]:.2f}, {T_w2_to_global[1,3]:.2f}, {T_w2_to_global[2,3]:.2f}]")
        
        # Compute blending weights (higher weight at center, lower at edges)
        weights = np.ones((window_frames, H, W))
        if window_idx > 0:  # Fade in from start
            fade_in = min(window_size - stride, window_frames)
            for i in range(fade_in):
                alpha = (i + 1) / (fade_in + 1)  # Smooth fade
                weights[i] = alpha
        if window_idx < len(all_window_results) - 1:  # Fade out at end
            fade_out = min(window_size - stride, window_frames)
            for i in range(fade_out):
                alpha = 1.0 - ((i + 1) / (fade_out + 1))
                weights[-(i+1)] = alpha
        
        # Accumulate weighted results
        merged_extrinsic[start_frame:end_frame] += window_extrinsic * weights[..., 0:1, 0:1]
        merged_intrinsic[start_frame:end_frame] += window_intrinsic * weights[..., 0:1, 0:1]
        merged_depth_map[start_frame:end_frame] += window_depth * weights
        merged_depth_conf[start_frame:end_frame] += window_conf * weights
        weight_accumulator[start_frame:end_frame] += weights
    
    # Normalize by accumulated weights
    weight_accumulator = np.maximum(weight_accumulator, 1e-6)
    merged_depth_map /= weight_accumulator
    merged_depth_conf /= weight_accumulator
    merged_extrinsic /= weight_accumulator[..., 0:1, 0:1]
    merged_intrinsic /= weight_accumulator[..., 0:1, 0:1]
    
    # Apply SVD to project averaged rotations to closest valid rotation matrices
    # This is necessary because weighted averaging of rotation matrices produces non-orthogonal results
    # SVD finds the closest orthogonal matrix, ensuring valid camera poses
    for t in range(total_frames):
        R = merged_extrinsic[t, :3, :3]
        U, S, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        # Ensure proper rotation (det = 1, not reflection with det = -1)
        if np.linalg.det(R_ortho) < 0:
            Vt[-1, :] *= -1
            R_ortho = U @ Vt
        merged_extrinsic[t, :3, :3] = R_ortho
    
    return merged_extrinsic, merged_intrinsic, merged_depth_map, merged_depth_conf


def get_ground_mask_with_sam3(video_path, frame_h, frame_w, sam3_processor, prompt="ground"):
    """
    Use SAM3 text prompting on the first frame to estimate a ground mask.
    Returns a boolean array of shape (frame_h, frame_w).
    """
    cap = cv2.VideoCapture(video_path)
    ok, first_frame_bgr = cap.read()
    cap.release()
    if not ok or first_frame_bgr is None:
        raise ValueError(f"Failed to read first frame from video for SAM3: {video_path}")

    from PIL import Image

    # SAM3 expects RGB image input.
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(first_frame_rgb)

    with patched_functional_dtype_match():
        inference_state = sam3_processor.set_image(image)
        output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    scores = output.get("scores", None)

    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim == 2:
        best_mask = masks_np
    elif masks_np.ndim >= 3:
        # Handle (N,H,W) and batched outputs like (B,N,H,W) by flattening candidates.
        masks_candidates = masks_np.reshape(-1, masks_np.shape[-2], masks_np.shape[-1])
        if len(masks_candidates) == 0:
            print(f"SAM3 returned no mask candidates for prompt='{prompt}'; using empty ground mask")
            return np.zeros((frame_h, frame_w), dtype=bool)
        if scores is not None:
            if isinstance(scores, torch.Tensor):
                scores_np = scores.detach().cpu().numpy()
            else:
                scores_np = np.asarray(scores)
            scores_flat = scores_np.reshape(-1)
            if len(scores_flat) == len(masks_candidates) and len(scores_flat) > 0:
                best_idx = int(np.argmax(scores_flat))
            else:
                best_idx = 0
        else:
            best_idx = 0
        best_mask = masks_candidates[best_idx]
    else:
        raise ValueError(f"Unexpected SAM3 mask shape: {masks_np.shape}")

    if best_mask.dtype != np.bool_:
        best_mask = best_mask > 0.5

    if best_mask.shape != (frame_h, frame_w):
        best_mask = cv2.resize(best_mask.astype(np.uint8), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST) > 0

    return best_mask


def inference_video(args, model_front, model_offline, video_path, results_path, 
                   sequence_dir=None, sliding_window=False, window_size=10, stride=3,
                   sam3_processor=None):
    video_reader = decord.VideoReader(video_path)
    
    if not sliding_window:
        # Original code path when sliding_window=False - load entire video
        video_tensor_raw = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
        video_tensor_raw = video_tensor_raw[::args.fps].float()
        
        # Preprocess video BEFORE passing to model (like inference.py)
        video_tensor = preprocess_image(video_tensor_raw)[None]
        del video_tensor_raw
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if args.uncropped:
                        predictions = model_front(
                            video_tensor.cuda() / 255.0
                        )
                    else:
                        f_list = np.array([1000*518/256] * video_tensor.shape[1])
                        f_list = scale_focal_length(f_list, sequence_dir or os.path.dirname(video_path))
                        predictions = model_front(
                            video_tensor.cuda() / 255.0, 
                            fx_prev=f_list, fy_prev=f_list
                        )
                    
                    # Move results to CPU immediately
                    extrinsic = predictions["poses_pred"].cpu()
                    intrinsic = predictions["intrs"].cpu()
                    depth_map = predictions["points_map"][..., 2].cpu()
                    depth_conf = predictions["unc_metric"].cpu()
                    
                    # Delete predictions
                    del predictions
                    torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM in model_front forward pass")
            raise
    else:
        # Sliding window approach - load and process windows on-demand
        print(f"Using sliding window: window_size={window_size}, stride={stride}")
        
        # Get total frames and build frame indices based on fps
        total_video_frames = len(video_reader)
        frame_indices = list(range(0, total_video_frames, args.fps))
        total_frames = len(frame_indices)
        
        print(f"Total frames after fps sampling: {total_frames}")
        
        f_list_full = np.array([1000*518/256] * total_frames)
        f_list_full = scale_focal_length(f_list_full, sequence_dir or os.path.dirname(video_path))
        
        all_window_results = []
        
        # Process video in sliding windows
        for start_idx in range(0, total_frames, stride):
            end_idx = min(start_idx + window_size, total_frames)
            print(f"Processing window: frames {start_idx} to {end_idx}")
            
            # Clear CUDA cache before processing each window
            torch.cuda.empty_cache()
            
            # Load only the frames needed for this window
            window_frame_indices = frame_indices[start_idx:end_idx]
            video_window_raw = torch.from_numpy(video_reader.get_batch(window_frame_indices).asnumpy()).permute(0, 3, 1, 2).float()
            
            # Preprocess BEFORE passing to model
            video_window = preprocess_image(video_window_raw)[None]
            del video_window_raw
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if args.uncropped:
                        predictions = model_front(
                            video_window.cuda() / 255.0
                        )
                    else:
                        f_list_window = f_list_full[start_idx:end_idx]
                        predictions = model_front(
                            video_window.cuda() / 255.0,
                            fx_prev=f_list_window,
                            fy_prev=f_list_window
                        )
                    
                    # Extract results and move to CPU immediately
                    window_extrinsic = predictions["poses_pred"].squeeze().cpu().numpy()
                    window_intrinsic = predictions["intrs"].squeeze().cpu().numpy()
                    window_depth_map = predictions["points_map"][..., 2].squeeze().cpu().numpy()
                    window_depth_conf = predictions["unc_metric"].squeeze().cpu().numpy()
            
            # Free GPU memory
            del predictions, video_window
            torch.cuda.empty_cache()
            
            all_window_results.append({
                'extrinsic': window_extrinsic,
                'intrinsic': window_intrinsic,
                'depth_map': window_depth_map,
                'depth_conf': window_depth_conf,
            })
            
            # Break if we've reached the end
            if end_idx >= total_frames:
                break
        
        # Merge all window results with ICP-based alignment
        print("Merging window results with alignment...")
        extrinsic_merged, intrinsic_merged, depth_map_merged, depth_conf_merged = \
            merge_window_results(all_window_results, window_size, stride, total_frames)
        
        # Convert back to tensors (keep on CPU to save GPU memory)
        extrinsic = torch.from_numpy(extrinsic_merged).unsqueeze(0)
        intrinsic = torch.from_numpy(intrinsic_merged).unsqueeze(0)
        depth_map = torch.from_numpy(depth_map_merged).unsqueeze(0)
        depth_conf = torch.from_numpy(depth_conf_merged).unsqueeze(0)
        
        # Now load full video for offline model (preprocess it)
        print("Loading full video for offline model...")
        video_tensor_full = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy()).permute(0, 3, 1, 2).float()
        video_tensor = preprocess_image(video_tensor_full)[None]
        del video_tensor_full  # Free memory
        torch.cuda.empty_cache()

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    #NOTE: 20% of the depth is not reliable
    # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    data_npz_load = {}

    model_offline.spatrack.track_num = args.vo_points
    
    model_offline.eval()
    model_offline.to("cuda")

    if video_tensor is  None:
        cap = cv2.VideoCapture(video_path)
        frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(args.grid_size, (frame_H, frame_W), device="cpu")
    
    # Find and apply mask to filter grid points to only track animal
    assert sequence_dir is not None, "sequence_dir must be provided for mask-based filtering"
    mask_files = sorted(glob(os.path.join(sequence_dir, "*_mask.png")))
    assert len(mask_files) > 0, f"No mask file (*_mask.png) found in {sequence_dir}"
    mask_path = mask_files[0]
    assert os.path.exists(mask_path), f"Mask file does not exist: {mask_path}"
    
    mask = cv2.imread(mask_path)
    assert mask is not None, f"Failed to load mask from {mask_path}"
    
    # If using uncropped video, transform cropped mask to uncropped coordinates
    if args.uncropped:
        # Load metadata to get crop box
        metadata_files = sorted(glob(os.path.join(sequence_dir, "*_metadata.json")))
        assert len(metadata_files) > 0, f"No metadata file (*_metadata.json) found in {sequence_dir} (required for uncropped mode)"
        with open(metadata_files[0], "r") as f:
            metadata = json.load(f)
            
            crop_box = metadata["crop_box_xyxy"]  # [x1, y1, x2, y2]
            assert "video_frame_height" in metadata and "video_frame_width" in metadata, \
                f"Metadata missing video_frame_height/video_frame_width: {metadata_files[0]}"
            uncropped_h = metadata["video_frame_height"]
            uncropped_w = metadata["video_frame_width"]
            assert uncropped_h is not None and uncropped_w is not None, \
                f"Metadata has empty video_frame_height/video_frame_width: {metadata_files[0]}"

            target_h = int(frame_H) if frame_H is not None else int(uncropped_h)
            target_w = int(frame_W) if frame_W is not None else int(uncropped_w)
            scale_x = target_w / float(uncropped_w)
            scale_y = target_h / float(uncropped_h)
            if scale_x != 1.0 or scale_y != 1.0:
                print(
                    f"Scaling crop_box from metadata size {uncropped_w}x{uncropped_h} "
                    f"to input video size {target_w}x{target_h}"
                )
                crop_box = [
                    crop_box[0] * scale_x,
                    crop_box[1] * scale_y,
                    crop_box[2] * scale_x,
                    crop_box[3] * scale_y,
                ]

            if any(v < 0 for v in crop_box):
                print(f"Warning: crop_box has negative values: {crop_box}")
            
            # Calculate actual crop dimensions from crop_box coordinates
            # Ensure we convert to int and validate
            x1, y1, x2, y2 = [int(round(v)) for v in crop_box]
            assert x2 > x1 and y2 > y1, \
                f"Invalid crop box after scaling: {crop_box}"
            actual_crop_w = x2 - x1
            actual_crop_h = y2 - y1
            
            # Debug: print crop box info
            print(f"Crop box info: crop_box_xyxy={crop_box}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            print(f"Calculated dimensions: width={actual_crop_w}, height={actual_crop_h}")
            print(f"Uncropped video dimensions: {target_w}x{target_h}")
            print(f"Mask dimensions before resize: {mask.shape}")
            
            # Validate crop dimensions
            assert actual_crop_w > 0 and actual_crop_h > 0, \
                f"Invalid crop box: crop_box_xyxy={crop_box}, width={actual_crop_w}, height={actual_crop_h}"
            
            # Resize mask to match actual crop box dimensions
            mask_resized = cv2.resize(mask, (actual_crop_w, actual_crop_h))
            
            # Create full-size mask for uncropped video
            full_mask = np.zeros((target_h, target_w, 3), dtype=mask.dtype)
            
            # Compute intersection between crop box and actual frame bounds
            # Crop box can extend outside frame to maintain square aspect ratio
            frame_x1 = max(0, x1)
            frame_y1 = max(0, y1)
            frame_x2 = min(target_w, x2)
            frame_y2 = min(target_h, y2)
            
            # Compute corresponding region in resized mask
            mask_x1 = frame_x1 - x1  # Offset from crop box origin
            mask_y1 = frame_y1 - y1
            mask_x2 = mask_x1 + (frame_x2 - frame_x1)
            mask_y2 = mask_y1 + (frame_y2 - frame_y1)
            
            # Place only the valid portion of mask into frame
            if frame_x2 > frame_x1 and frame_y2 > frame_y1:
                full_mask[frame_y1:frame_y2, frame_x1:frame_x2] = mask_resized[mask_y1:mask_y2, mask_x1:mask_x2]
                print(f"Placed mask: frame[{frame_y1}:{frame_y2}, {frame_x1}:{frame_x2}] = mask[{mask_y1}:{mask_y2}, {mask_x1}:{mask_x2}]")
            else:
                print(f"ERROR: Crop box completely outside frame bounds!")
                print(f"  Crop box: [{x1}, {y1}, {x2}, {y2}], Frame size: {target_w}x{target_h}")
                print(f"  Frame intersection: [{frame_x1}, {frame_y1}, {frame_x2}, {frame_y2}]")
                raise ValueError(f"Crop box is completely outside frame bounds. Cannot create mask.")
            
            mask = full_mask
            print(f"Transformed mask from cropped ({actual_crop_w}x{actual_crop_h}) to uncropped ({target_w}x{target_h})")
    
    # Resize mask to match video dimensions
    mask = cv2.resize(mask, (frame_W, frame_H))
    # Convert to binary mask (any non-zero pixel)
    mask = mask.sum(axis=-1) > 0
    
    # Verify mask has non-zero pixels
    num_mask_pixels = mask.sum()
    if num_mask_pixels == 0:
        raise ValueError(f"Mask is completely empty (all zeros)! Cannot extract grid points. "
                        f"Mask shape: {mask.shape}, video size: {frame_W}x{frame_H}")
    
    # Build ground mask with SAM3 from the first frame, and enforce disjointness with object mask.
    assert sam3_processor is not None, "sam3_processor must be provided"
    ground_mask = get_ground_mask_with_sam3(video_path, frame_H, frame_W, sam3_processor, prompt="ground")
    overlap_pixels = np.logical_and(ground_mask, mask).sum()
    if overlap_pixels > 0:
        print(f"Removing {overlap_pixels} overlapping ground/object pixels from ground mask")
    ground_mask = np.logical_and(ground_mask, np.logical_not(mask))

    # Sample masks at grid point locations.
    # Keep all points for tracking, and only use masks to define subset indices.
    grid_pts_int = grid_pts[0].long()
    mask_values = np.asarray(mask[grid_pts_int[..., 1], grid_pts_int[..., 0]], dtype=bool)
    # These IDs describe the input query grid only. The tracker can return its
    # output columns in a different order, so do not use these IDs to slice the
    # returned 2D/3D tracks. Reclassify returned columns below from their frame-0
    # 2D locations.
    query_object_point_indices = np.flatnonzero(mask_values).astype(np.int64)
    ground_mask_values = np.asarray(ground_mask[grid_pts_int[..., 1], grid_pts_int[..., 0]], dtype=bool)
    query_ground_point_indices = np.flatnonzero(ground_mask_values).astype(np.int64)
    overlap_point_indices = np.intersect1d(query_object_point_indices, query_ground_point_indices)
    if len(overlap_point_indices) > 0:
        print(f"Removing {len(overlap_point_indices)} overlapping points from ground set")
        query_ground_point_indices = np.setdiff1d(query_ground_point_indices, query_object_point_indices)
    print(f"Input-query object mask points: {len(query_object_point_indices)} / {len(mask_values)}")
    print(f"Input-query ground mask points: {len(query_ground_point_indices)} / {len(ground_mask_values)}")
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    
    # Final validation before passing to model
    if query_xyt.shape[0] == 0:
        raise ValueError(f"No query points remaining after all processing! "
                        f"grid_pts shape: {grid_pts.shape}, query_xyt shape: {query_xyt.shape}")
    
    unc_metric_np = np.asarray(unc_metric)
    if unc_metric_np.ndim == 2:
        first_unc_metric = unc_metric_np
    elif unc_metric_np.ndim == 3:
        first_unc_metric = unc_metric_np[0]
    else:
        raise ValueError(f"Unexpected unc_metric shape: {unc_metric_np.shape}")
    query_xy = np.rint(query_xyt[:, 1:3]).astype(np.int64)
    query_xy[:, 0] = np.clip(query_xy[:, 0], 0, frame_W - 1)
    query_xy[:, 1] = np.clip(query_xy[:, 1], 0, frame_H - 1)
    reliable_query_count = int((first_unc_metric[query_xy[:, 1], query_xy[:, 0]] > 0.5).sum())
    full_point_for_queries = reliable_query_count == 0
    print(f"Reliable query points by depth confidence: {reliable_query_count} / {query_xyt.shape[0]}")
    if full_point_for_queries:
        print("No reliable query points found; disabling uncertainty pruning for offline tracking")

    print(f"intrinsic input to offline model: {intrs[:5]}")

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model_offline.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=full_point_for_queries, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2)
        print(f"intrinsic output from offline model: {intrs[:5]}")
        
        # # Smooth camera trajectory to reduce Z-axis jitter
        c2w_traj_np = c2w_traj.cpu().numpy()
        c2w_traj_smooth = smooth_camera_trajectory(c2w_traj_np, keyframe_interval=5)
        c2w_traj = torch.from_numpy(c2w_traj_smooth).to(c2w_traj.device)
        print("Applied camera trajectory smoothing")
        
        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is 336
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))

        # Preserve the exact 2D tracker output and its query IDs.  ``coords`` is
        # a 3D, world-space product that can receive later ground alignment, so
        # it is not a reliable replacement for direct image-space tracks when
        # debugging object-mask selection.
        # Classify the *returned tracker columns* by where they actually land in
        # frame 0. SpaTracker can permute output columns relative to query_xyt;
        # slicing coords with input-grid IDs would then mix object/background
        # tracks. ``track2d_pred`` is now in the compact saved-video resolution,
        # while masks remain in the original tracker-input resolution.
        first_track_xy = track2d_pred[0, :, :2].detach().cpu().numpy() / scale
        first_track_xy = np.rint(first_track_xy).astype(np.int64)
        first_track_xy[:, 0] = np.clip(first_track_xy[:, 0], 0, frame_W - 1)
        first_track_xy[:, 1] = np.clip(first_track_xy[:, 1], 0, frame_H - 1)
        object_point_indices = np.flatnonzero(
            mask[first_track_xy[:, 1], first_track_xy[:, 0]]
        ).astype(np.int64)
        ground_point_indices = np.flatnonzero(
            ground_mask[first_track_xy[:, 1], first_track_xy[:, 0]]
        ).astype(np.int64)
        overlap_point_indices = np.intersect1d(object_point_indices, ground_point_indices)
        if len(overlap_point_indices) > 0:
            ground_point_indices = np.setdiff1d(ground_point_indices, object_point_indices)
        print(
            f"Returned-track object mask points: {len(object_point_indices)} / {len(first_track_xy)}; "
            f"ground points: {len(ground_point_indices)}"
        )

        # save as the tapip3d format   
        coords_world = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        c2w_traj_np = c2w_traj.detach().cpu().numpy().astype(np.float32)
        if len(ground_point_indices) > 0:
            coords_ground = coords_world[:, ground_point_indices, :]
            # Use frame-0 ground normal as reference vertical direction. Each frame is
            # rotated to align its normal with reference (or opposite, whichever is closer).
            reference_normal = estimate_ground_plane_normal_ransac(coords_ground[0], seed=0)
            if reference_normal is None:
                for t in range(1, coords_ground.shape[0]):
                    reference_normal = estimate_ground_plane_normal_ransac(coords_ground[t], seed=t)
                    if reference_normal is not None:
                        print(f"Frame 0 normal unavailable; using frame {t} as reference")
                        break

            if reference_normal is None:
                print("No valid ground normal found in any frame; skipping tilt correction")
            else:
                reference_normal = reference_normal / (np.linalg.norm(reference_normal) + 1e-12)
                print(
                    f"Reference vertical normal=[{reference_normal[0]:.6f}, {reference_normal[1]:.6f}, {reference_normal[2]:.6f}]"
                )

                for t in range(coords_ground.shape[0]):
                    normal_t = estimate_ground_plane_normal_ransac(coords_ground[t], seed=t)
                    if normal_t is None:
                        print(f"Frame {t}: ground normal estimation failed; using identity correction")
                        continue

                    normal_t = normal_t / (np.linalg.norm(normal_t) + 1e-12)
                    target_vertical = reference_normal if np.dot(normal_t, reference_normal) >= 0 else -reference_normal
                    R_t = rotation_matrix_align_vector_to_vector(normal_t, target_vertical)

                    coords_world[t] = (coords_world[t] @ R_t.T).astype(np.float32)
                    c2w_traj_np[t, :3, :3] = R_t @ c2w_traj_np[t, :3, :3]
                    c2w_traj_np[t, :3, 3] = R_t @ c2w_traj_np[t, :3, 3]

                    print(
                        f"Frame {t}: normal=[{normal_t[0]:.6f}, {normal_t[1]:.6f}, {normal_t[2]:.6f}] "
                        f"-> target=[{target_vertical[0]:.6f}, {target_vertical[1]:.6f}, {target_vertical[2]:.6f}]"
                    )

            # Robust per-frame translation to reduce jitter from residual camera drift.
            coords_ground_after_rot = coords_world[:, ground_point_indices, :]
            frame_translations = estimate_per_frame_translation_from_ground(coords_ground_after_rot)
            for t in range(coords_world.shape[0]):
                coords_world[t] = coords_world[t] + frame_translations[t]
                c2w_traj_np[t, :3, 3] = c2w_traj_np[t, :3, 3] + frame_translations[t]
                if np.linalg.norm(frame_translations[t]) > 0:
                    print(
                        f"Frame {t}: translation correction = "
                        f"[{frame_translations[t,0]:.6f}, {frame_translations[t,1]:.6f}, {frame_translations[t,2]:.6f}]"
                    )

            # Compute smoothing signal from ground points only, then apply to all points/camera.
            coords_ground_before_smooth = coords_world[:, ground_point_indices, :]
            coords_ground_smoothed = smooth_3d_points_temporal(
                coords_ground_before_smooth, median_window=3, mean_window=7, preserve_first=True
            )
            delta_ground = coords_ground_smoothed - coords_ground_before_smooth
            frame_smooth_offsets = np.zeros((coords_world.shape[0], 3), dtype=np.float32)
            for t in range(coords_world.shape[0]):
                valid = np.isfinite(delta_ground[t]).all(axis=1)
                if valid.sum() == 0:
                    continue
                frame_smooth_offsets[t] = np.median(delta_ground[t, valid], axis=0).astype(np.float32)

            for t in range(coords_world.shape[0]):
                coords_world[t] = coords_world[t] + frame_smooth_offsets[t]
                c2w_traj_np[t, :3, 3] = c2w_traj_np[t, :3, 3] + frame_smooth_offsets[t]
        else:
            print("No ground mask points found; skipping ground-based post alignment")

        extrinsics_np = np.linalg.inv(c2w_traj_np).astype(np.float32)
        data_npz_load["coords"] = coords_world
        data_npz_load["coords_object"] = coords_world[:, object_point_indices, :]
        data_npz_load["coords_ground"] = coords_world[:, ground_point_indices, :]
        # Tiny but necessary to label the object subset within the full ``coords``
        # grid. Reconstruction uses coords_object directly; visualization cannot
        # safely recover these IDs from coordinates because tracker columns can
        # be permuted or duplicated.
        data_npz_load["object_point_indices"] = object_point_indices
        data_npz_load["extrinsics"] = extrinsics_np
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        data_npz_load["video_frame_size"] = video_tensor.shape[2:]

        # Comment out saving large/unnecessary data to reduce file size
        # depth_save = point_map[:,2,...]
        # depth_save[conf_depth<0.5] = 0
        # data_npz_load["depths"] = depth_save.cpu().numpy()
        # data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        # data_npz_load["visibs"] = vis_pred.cpu().numpy()
        # data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        
        np.savez(results_path, **data_npz_load)
        print(f"Results saved to {results_path}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {results_path}[/bold yellow]")

    

class CustomVGGT4Track(VGGT4Track):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        annots = {},
        fx_prev = None,
        fy_prev = None,
        **kwargs):

        # If without batch dimension, add it
        B, T, C, H, W = images.shape
        images_proc = preprocess_image(images.view(B*T, C, H, W).clone())
        images_proc = rearrange(images_proc, '(b t) c h w -> b t c h w', b=B, t=T)
        _, _, _, H_proc, W_proc = images_proc.shape

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images_proc)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images_proc, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["unc_metric"] = depth_conf.view(B*T, H_proc, W_proc)

        predictions["images"] = (images)*255.0
                                                         
        # output the camera pose
        predictions["poses_pred"] = torch.eye(4)[None].repeat(T, 1, 1)[None]
        predictions["poses_pred"][:,:,:3,:4], predictions["intrs"] = pose_encoding_to_extri_intri(predictions["pose_enc_list"][-1],
                                                                                                                     images_proc.shape[-2:])
        predictions["poses_pred"] = torch.inverse(predictions["poses_pred"])

        if fx_prev is not None:
            # Compute per-frame scale factors without averaging
            fx_tensor = torch.from_numpy(fx_prev).to(predictions["intrs"].device)
            scale_x_per_frame = fx_tensor / predictions["intrs"][0, :fx_prev.shape[0], 0, 0]
            scale_x_per_frame = scale_x_per_frame * (W_proc / W)
            # Apply per-frame scaling
            predictions["intrs"][0, :fx_prev.shape[0], 0, 0] = fx_tensor * (W_proc / W)
            
        if fy_prev is not None:
            # Compute per-frame scale factors without averaging
            fy_tensor = torch.from_numpy(fy_prev).to(predictions["intrs"].device)
            scale_y_per_frame = fy_tensor / predictions["intrs"][0, :fy_prev.shape[0], 1, 1]
            scale_y_per_frame = scale_y_per_frame * (H_proc / H)
            # Apply per-frame scaling
            predictions["intrs"][0, :fy_prev.shape[0], 1, 1] = fy_tensor * (H_proc / H)

        # get the points map
        points_map = depth_to_points_colmap(depth.view(B*T, H_proc, W_proc), predictions["intrs"].view(B*T, 3, 3))
        predictions["points_map"] = points_map
        #NOTE: resize back
        predictions["points_map"] = F.interpolate(points_map.permute(0,3,1,2),
                                                         size=(H, W), mode='bilinear', align_corners=True).permute(0,2,3,1)

        predictions["unc_metric"] = F.interpolate(predictions["unc_metric"][:,None],
                                                         size=(H, W), mode='bilinear', align_corners=True)[:,0]
        predictions["intrs"][..., :1, :] *= W/W_proc 
        predictions["intrs"][..., 1:2, :] *= H/H_proc 

        if self.training:
            loss = compute_loss(predictions, annots)
            predictions["loss"] = loss
                                                                                   
        return predictions
        


if __name__ == "__main__":
    args = parse_args()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    if args.sequence_dir:
        sequence_dir = os.path.abspath(args.sequence_dir)
        video_path = os.path.abspath(args.rgb_video)
        metadata_path = os.path.abspath(args.metadata)
        if not os.path.isdir(sequence_dir):
            raise FileNotFoundError(f"Clip directory does not exist: {sequence_dir}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Cropped RGB video does not exist: {video_path}")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Clip metadata does not exist: {metadata_path}")
        all_videos = [video_path]
    else:
        sequence_dirs = get_all_sequence_dirs(args.data_dir)
        all_videos = []
        for sequence_dir in sequence_dirs:
            video_id = os.path.basename(sequence_dir)
            all_videos.append(os.path.join(sequence_dir, f"{video_id}_rgb.mp4"))

    shuffle(all_videos)
    
    vggt4track_model_front = CustomVGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model_front.eval()
    vggt4track_model_front = vggt4track_model_front.to("cuda")
    vggt4track_model_offline = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    sam3_model = build_sam3_image_model()
    sam3_processor = Sam3Processor(sam3_model)
    print(f"Running SpaTrackerv2 on {len(all_videos)} videos...")

    for video_path in tqdm(all_videos):
        sequence_dir = os.path.abspath(args.sequence_dir) if args.sequence_dir else os.path.dirname(video_path)
        out_dir = os.path.abspath(args.out_dir) if args.sequence_dir and args.out_dir else sequence_dir
        os.makedirs(out_dir, exist_ok=True)

        if args.uncropped:
            result_path = os.path.join(out_dir, "tracker_results_uncropped.npz")
        else:
            result_path = os.path.join(out_dir, "tracker_results.npz")
        
        # DEBUG
        # if "0k2kPGczdzs_017_002" not in video_path:
        #     continue

        if os.path.exists(result_path):
            print(f"Results already exist for {video_path}, skipping...")
            continue

        print(f"Processing {video_path}...")
        
        # Find first mask file in the selected sample directory
        temp_video_path = None
        try:
            if args.uncropped:
                if args.uncropped_video:
                    uncropped_video_path = os.path.abspath(args.uncropped_video)
                    if not os.path.isfile(uncropped_video_path):
                        raise FileNotFoundError(f"Uncropped video does not exist: {uncropped_video_path}")
                    # Keep only the frames that were cropped for this clip, as in batch mode.
                    temp_video_path = align_uncropped_video(
                        uncropped_video_path, sequence_dir, os.path.abspath(args.metadata),
                        os.path.join(out_dir, f"{os.path.basename(sequence_dir)}_uncropped.mp4"),
                    )
                else:
                    temp_video_path = get_uncropped_video(sequence_dir, args.uncropped_video_dir)
                inference_video(args, vggt4track_model_front, vggt4track_model_offline, temp_video_path, result_path,
                              sequence_dir=sequence_dir, sliding_window=args.sliding_window,
                              sam3_processor=sam3_processor)
            else:
                inference_video(args, vggt4track_model_front, vggt4track_model_offline, video_path, result_path, 
                              sequence_dir=sequence_dir, sliding_window=args.sliding_window,
                              sam3_processor=sam3_processor)
        except Exception as e:
            if yt_dlp is not None and isinstance(e, yt_dlp.utils.DownloadError):
                print(f"Skipping {video_path}: YouTube download failed")
            else:
                print(f"Error processing {video_path}:")
                traceback.print_exc()
            # Batch mode can continue past a bad sequence. Single-sample mode
            # must fail so the wrapper never proceeds as if tracking succeeded.
            if args.sequence_dir:
                raise
        finally:
            if args.uncropped and temp_video_path is not None and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
