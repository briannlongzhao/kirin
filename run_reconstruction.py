#!/usr/bin/env python3
"""Reconstruct 3D animal motion for one clip with AniMer, optionally adding
global motion estimated by SpaTrackerV2."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D motion (joints_global.npy) for one clip."
    )
    parser.add_argument("--rgb-video", type=Path, required=True,
                        help="Square-cropped *_rgb.mp4 for this clip.")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Clip-level *_info.json metadata.")
    parser.add_argument("--checkpoint-path", type=Path, required=True,
                        help="AniMer checkpoint file (AniMer/data/AniMer/checkpoints/checkpoint.ckpt).")
    parser.add_argument("--sample-dir", type=Path, default=None,
                        help="Directory containing this clip's *_rgb.png, *_mask.png, *_keypoint.txt, and per-frame "
                             "*_metadata.json files. Defaults to the RGB video's parent. It is only read.")
    parser.add_argument("--out-folder", type=Path, default=None,
                        help="Output directory for joints, SMAL parameters and tracker results. Defaults to --sample-dir.")
    parser.add_argument("--global-motion", choices=("none", "cropped", "uncropped"), default="none",
                        help="none: camera-space motion without SpaTrackerV2. cropped/uncropped: estimate global "
                             "translation with SpaTrackerV2 on the square-cropped or the full-frame video.")
    parser.add_argument("--uncropped-video", type=Path, default=None,
                        help="Full-frame video of the clip, required for --global-motion uncropped.")
    parser.add_argument("--rerun-tracker", action="store_true",
                        help="Run SpaTrackerV2 even if tracker results already exist in --out-folder.")
    parser.add_argument("--tracker-vo-points", type=int, default=756)
    parser.add_argument("--tracker-grid-size", type=int, default=10)
    parser.add_argument("--tracker-fps", type=int, default=1)
    parser.add_argument("--sliding-window", action=argparse.BooleanOptionalAction, default=None,
                        help="Sliding-window tracker inference. Defaults to on for uncropped and off for cropped tracking.")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-mesh", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="Also render motion.mp4 and motion_overlay.png from the reconstructed joints.")
    parser.add_argument("--num-iters", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proj-loss-weight", type=float, default=1.0)
    parser.add_argument("--smooth-loss-weight", type=float, default=0.1)
    parser.add_argument("--cam-anchor-weight", type=float, default=0.0)
    parser.add_argument("--center-keypoint", type=int, default=-1)
    parser.add_argument("--restrict-side-bending", action="store_true")
    parser.add_argument("--restrict-twist", action="store_true")
    args = parser.parse_args()

    if args.global_motion == "uncropped" and args.uncropped_video is None:
        parser.error("--uncropped-video is required for --global-motion uncropped.")
    if args.sliding_window is None:
        args.sliding_window = args.global_motion == "uncropped"
    args.rgb_video = args.rgb_video.expanduser().absolute()
    args.metadata = args.metadata.expanduser().absolute()
    args.checkpoint_path = args.checkpoint_path.expanduser().absolute()
    args.sample_dir = (args.sample_dir or args.rgb_video.parent).expanduser().absolute()
    args.out_folder = (args.out_folder or args.sample_dir).expanduser().absolute()
    if args.uncropped_video is not None:
        args.uncropped_video = args.uncropped_video.expanduser().absolute()
    for flag, path in (("--rgb-video", args.rgb_video), ("--metadata", args.metadata),
                       ("--checkpoint-path", args.checkpoint_path), ("--uncropped-video", args.uncropped_video)):
        if path is not None and not path.is_file():
            parser.error(f"{flag} does not exist: {path}")
    if not args.sample_dir.is_dir():
        parser.error(f"--sample-dir does not exist: {args.sample_dir}")
    frames = list(args.sample_dir.glob("*_rgb.png"))
    if not frames:
        parser.error(f"No *_rgb.png frame images found in --sample-dir: {args.sample_dir}")
    required = ["*_mask.png", "*_keypoint.txt"]
    if args.global_motion != "none":
        required.append("*_metadata.json")
    for pattern in required:
        found = len(list(args.sample_dir.glob(pattern)))
        if found != len(frames):
            parser.error(f"Expected {len(frames)} {pattern} files in --sample-dir, found {found}")
    return args


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent
    args.out_folder.mkdir(parents=True, exist_ok=True)
    uncropped = args.global_motion == "uncropped"

    if args.global_motion != "none":
        result_name = "tracker_results_uncropped.npz" if uncropped else "tracker_results.npz"
        result_path = args.out_folder / result_name
        if args.rerun_tracker and result_path.exists():
            result_path.unlink()
        if result_path.exists():
            print(f"Using existing tracker results: {result_path}", flush=True)
        else:
            track = [
                sys.executable, "-m", "reconstruction.tracker",
                "--sequence_dir", str(args.sample_dir),
                "--rgb_video", str(args.rgb_video),
                "--metadata", str(args.metadata),
                "--out_dir", str(args.out_folder),
                "--vo_points", str(args.tracker_vo_points),
                "--grid_size", str(args.tracker_grid_size),
                "--fps", str(args.tracker_fps),
            ]
            if args.sliding_window:
                track.append("--sliding_window")
            if uncropped:
                track += ["--uncropped", "--uncropped_video", str(args.uncropped_video)]
            print("Running SpaTrackerV2:", " ".join(track), flush=True)
            subprocess.run(track, cwd=repo, check=True)

    reconstruct = [
        sys.executable, "-m", "reconstruction.animer",
        "--sequence_dir", str(args.sample_dir),
        "--out_folder", str(args.out_folder),
        "--checkpoint_path", str(args.checkpoint_path),
        "--num_iters", str(args.num_iters),
        "--lr", str(args.lr),
        "--proj_loss_weight", str(args.proj_loss_weight),
        "--smooth_loss_weight", str(args.smooth_loss_weight),
        "--cam_anchor_weight", str(args.cam_anchor_weight),
        "--center_keypoint", str(args.center_keypoint),
    ]
    if args.global_motion != "none":
        reconstruct.append("--global_motion")
        if uncropped:
            reconstruct.append("--uncropped_results")
    if args.save_video:
        reconstruct.append("--save_video")
    if args.save_mesh:
        reconstruct.append("--save_mesh")
    if args.restrict_side_bending:
        reconstruct.append("--restrict_side_bending")
    if args.restrict_twist:
        reconstruct.append("--restrict_twist")

    print("Running AniMer reconstruction:", " ".join(reconstruct), flush=True)
    subprocess.run(reconstruct, cwd=repo, check=True)

    joints_path = args.out_folder / "joints_global.npy"
    if not joints_path.is_file():
        sys.exit(f"Reconstruction did not produce {joints_path}")
    if args.render:
        render = [sys.executable, str(repo / "render_motion.py"), "--joints-path", str(joints_path)]
        print("Rendering motion:", " ".join(render), flush=True)
        subprocess.run(render, cwd=repo, check=True)
    print(f"Done: {joints_path}")


if __name__ == "__main__":
    main()
