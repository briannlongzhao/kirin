# Kirin

Code for *Kirin: Animal Motion Generation from In-the-Wild Video*. The
repository has three parts that share one environment:

- **Motion reconstruction** (`run_reconstruction.py`): 3D motion of an animal
  clip as a SMAL joint sequence, with AniMer for pose and optionally
  SpaTrackerV2 (with SAM3) for global translation.
- **Motion generation** (`motion_generation/`): the text- and image-conditioned
  diffusion model, its training on AiM3D, and pretrained models.
- **Mesh animation** (`animate_mesh.py`): a mesh generated from one image with
  Rodin, animated with a reconstructed or generated joint sequence.

## Installation

Clone this repository with its AniMer and SpaTrackerV2 submodules. Then apply
the patch that replaces an upstream interactive debugger stop in SpaTrackerV2
with a rotation projection:

    git clone --recurse-submodules git@github.com:briannlongzhao/kirin.git
    cd kirin
    git apply patches/spatrack-weighted-procrustes.patch

Create the environment (Python 3.10, PyTorch 2.4.1, CUDA 12.4):

    conda env create -f environment.yml
    conda activate kirin

Install Chumpy, PyTorch3D, the AniMer submodule and OpenAI CLIP (used by the
CLIP-based generation baselines). PyTorch3D is built from source, so a CUDA 12.4 toolkit (`nvcc`) must be available, for example through
`CUDA_HOME=/usr/local/cuda-12.4`:

    python -m pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git@51d5afd92a8ded3637553be8cef41f328a1c863a"
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@e3d3a67a89907476bd5b63289f9669bd427ae550"
    python -m pip install --no-deps -e AniMer
    python -m pip install --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"

Finally, install the OpenCV build used to produce the released results and the
viser viewer. Video decoding differs between OpenCV versions and changes
full-frame tracking results. Both packages are installed without dependencies
because their declared requirements would replace this OpenCV build and NumPy
1.26.4, which Chumpy and SAM3 require:

    python -m pip install --no-deps opencv-python-headless==4.13.0.90 viser==1.0.4
    python -m pip install msgspec nodeenv "websockets>=13.1,<16" yourdfpy

## Model weights

AniMer: download the `AniMer` checkpoint folder from the
[AniMer release](https://drive.google.com/drive/folders/1xYCJyxZeSYPI6RXnHqa_lDbjnIAs8hVi?usp=sharing)
into `AniMer/data/`, so that the following files exist next to the SMAL model
files already in `AniMer/data/smal/`:

    AniMer/data/AniMer/checkpoints/checkpoint.ckpt
    AniMer/data/AniMer/.hydra/config.yaml

SpaTrackerV2 and SAM3 weights are downloaded from Hugging Face on first use.
SAM3 is gated: request access at https://huggingface.co/facebook/sam3 and log
in with `hf auth login` before running with global motion.

## Input data for one clip

Each clip directory (for example `track_3.0.0/horse/043zINOeztc_002/043zINOeztc_002_001/`)
contains:

- `CLIP_ID_rgb.mp4`: the square-cropped RGB video
- `CLIP_ID_info.json`: clip-level metadata, including the clip frame rate
- per frame `NNNNNNNN_rgb.png`, `NNNNNNNN_mask.png`, and `NNNNNNNN_keypoint.txt`
- per frame `NNNNNNNN_metadata.json`, with the crop box and frame index in the
  full-frame video (needed for global motion)

For global motion from the full-frame video, also provide the full-frame clip
video. Its frames are resampled to the clip frame rate and matched to the
per-frame `clip_frame_id` values before tracking.

The clip directory is only read. All outputs go to `--out-folder`, which
defaults to the clip directory.

## Run reconstruction

Choose the global motion source with `--global-motion`:

- `none` (default): camera-space motion from AniMer only; SpaTrackerV2 is not used.
- `cropped`: global translation from SpaTrackerV2 run on the square-cropped video.
- `uncropped`: global translation from SpaTrackerV2 run on the full-frame video.

Camera-space motion:

    python run_reconstruction.py \
      --rgb-video /path/to/CLIP_ID/CLIP_ID_rgb.mp4 \
      --metadata /path/to/CLIP_ID/CLIP_ID_info.json \
      --checkpoint-path AniMer/data/AniMer/checkpoints/checkpoint.ckpt \
      --out-folder /path/to/output/CLIP_ID

Global motion from the square-cropped video:

    python run_reconstruction.py \
      --rgb-video /path/to/CLIP_ID/CLIP_ID_rgb.mp4 \
      --metadata /path/to/CLIP_ID/CLIP_ID_info.json \
      --checkpoint-path AniMer/data/AniMer/checkpoints/checkpoint.ckpt \
      --out-folder /path/to/output/CLIP_ID \
      --global-motion cropped

Global motion from the full-frame video:

    python run_reconstruction.py \
      --rgb-video /path/to/CLIP_ID/CLIP_ID_rgb.mp4 \
      --metadata /path/to/CLIP_ID/CLIP_ID_info.json \
      --uncropped-video /path/to/full_frame/CLIP.mp4 \
      --checkpoint-path AniMer/data/AniMer/checkpoints/checkpoint.ckpt \
      --out-folder /path/to/output/CLIP_ID \
      --global-motion uncropped

SpaTrackerV2 runs automatically when `tracker_results.npz` (cropped) or
`tracker_results_uncropped.npz` (uncropped) is missing from `--out-folder`, and
existing results are reused. Pass `--rerun-tracker` to recompute them. Tracking
uses a 10x10 query grid and 756 visual-odometry points; full-frame tracking
uses sliding-window inference by default (`--no-sliding-window` disables it).
If the frames are stored apart from the video, pass `--sample-dir`.

## Outputs

In `--out-folder`:

- `joints_global.npy`: float array of shape `(T, 35, 3)`, the 35 SMAL joints per
  frame, Y up. With `--global-motion none` the joints keep the AniMer
  orientation relative to the camera, without translation. With global motion,
  the frame-0 root is at X = Z = 0 and each frame is lifted so its lowest joint
  is at Y = 0.
- `NNNNNNNN_smal_params_global.pth`: per-frame SMAL parameters (`betas`,
  `global_orient`, `pose`, `transl`, `joints`, `cam`).
- `tracker_results*.npz`: SpaTrackerV2 results, when global motion is used.

`--save-mesh` also writes per-frame meshes (`*_mesh_animer_global.obj`). A
reconstruction is skipped when `joints_global.npy` already exists in the output
folder.

## Visualization

Render a skeleton video (`motion.mp4`), an overlay of evenly spaced poses
(`motion_overlay.png`, blue to green over time) and its color legend
(`color_palette.png`) next to the joints file:

    python render_motion.py --joints-path /path/to/output/CLIP_ID/joints_global.npy

Adding `--render` to `run_reconstruction.py` does the same right after
reconstruction. The red line is the root trajectory on the floor.

Browse a reconstruction interactively with viser, next to the input frames:

    python view_motion.py \
      --joints-path /path/to/output/CLIP_ID/joints_global.npy \
      --sample-dir /path/to/CLIP_ID

Open the printed URL (port 8080 by default). The GUI has a frame slider,
playback, and toggles for joint indices and axis labels. With meshes saved by
`run_reconstruction.py --save-mesh`, pass `--load-mesh` to show them.

## Animate a mesh

`animate_mesh.py` animates an animal mesh generated from a single image with a
joint sequence, for example a reconstructed `joints_global.npy` (35 joints) or
a generated motion (34 joints, without the duplicated root). It

1. generates a textured T-pose mesh from the image with
   [Rodin](https://hyper3d.ai) (Hyper3D API),
2. fits SMAL shape and pose to the mesh (Chamfer, edge, and foot-anchor losses),
3. transfers SMAL skinning weights to the mesh from its 10 nearest SMAL vertices,
4. fits SMAL to every frame of the joint sequence, starting each frame from
   the previous one, and
5. deforms the mesh with linear blend skinning.

Rodin needs an API key in the environment:

    export RODIN_API_KEY=...
    python animate_mesh.py \
      --joints-path /path/to/output/CLIP_ID/joints_global.npy \
      --image-path /path/to/CLIP_ID/NNNNNNNN_rgb.png \
      --mask-path /path/to/CLIP_ID/NNNNNNNN_mask.png \
      --output-dir /path/to/animation

The mask is optional; with it, the background is made transparent before the
image is sent to Rodin. To reuse a mesh generated earlier, pass its `base.obj`
with `--tpose-mesh` instead of `--image-path`; its folder (textures included)
is copied. `--max-frames` limits the number of frames and `--fps` sets the
frame rate of the rendered video (reconstructed clips are 10 fps). Mesh fitting
uses random surface samples on the GPU, so repeated runs give slightly
different T-pose fits; an existing `tpose_mesh/smal_params.pth` in the output
folder is reused.

Outputs in `--output-dir`:

- `meshes/NNNNNN_mesh_animated.obj`: one textured mesh per frame, sharing
  `material.mtl` and `material_0.png`
- `animated_mesh.npz`: `vertices` `(T, V, 3)`, `faces` `(F, 3)`, and the
  `joints` `(T, 35, 3)` the mesh follows
- `animation.mp4`: textured rendering from a fixed camera (`--no-render` skips it)
- `tpose_mesh/`: the Rodin mesh, its textures, and the fitted SMAL parameters
  (`smal_params.pth`)
- `smal_fit_params/NNNNNNNN_smal_params.pth`: per-frame SMAL parameters

Re-render the video of an existing output folder:

    python -m mesh_animation.render --animation-dir /path/to/animation --fps 10

View the animated mesh with the joints it follows:

    python view_motion.py --animation-dir /path/to/animation

## Motion generation

`motion_generation/` contains the Kirin motion generator, an MDM-based
diffusion model conditioned on text (DistilBERT) and, optionally, an image
(DINOv3 [CLS] feature). Run all commands from `motion_generation/`.

### Data layout

The AiM3D dataset can be downloaded from
https://download.cs.stanford.edu/viscam/AiM/AiM_full/index.html (one archive per
animal category, plus `Mean.npy`, `Std.npy`, `train.txt` and `test.txt`).

The test set used for evaluation (230 clips, 10 per category, with the
ground-truth motions in `animal_test.npy`) is downloaded separately into
`dataset/AnimalMotion_test/`:

    wget https://download.cs.stanford.edu/viscam/AiM/AiM_preview.zip
    unzip AiM_preview.zip && mkdir -p dataset && mv data dataset/AnimalMotion_test

Expected layout:

    motion_generation/
      dataset/AnimalMotion/        training clips, CATEGORY/VIDEO_ID/CLIP_ID/, plus Mean.npy, Std.npy, train.txt
      dataset/AnimalMotion_test/   test clips, CATEGORY/CLIP_ID/
      animal_unified/              evaluator weights and statistics
      glove/                       GloVe word vectors used by the evaluator (our_vab_*)

Each clip folder holds motion features (`new_joint_vecs_global.npy`, or
`new_joint_vecs.npy` in the test set, `(T, 407)`), six captions
(`texts_gemini.txt`, or `captions.txt` in the test set), and
`mean_feature_dinov3.npy` (image feature, `(1, 768)`).

### Prepare data

To build a dataset from reconstructions (`joints_global.npy` per clip, see
above), compute the motion features, normalization statistics and split. With
`--test_dir`, the test clips' ids go to `test.txt` and are left out of
`train.txt`:

    python -m data_loaders.humanml.scripts.motion_process_animal \
      --data_dir dataset/AnimalMotion --test_dir dataset/AnimalMotion_test

Image features (mean DINOv3 [CLS] feature over up to 32 random frames) and
Gemini captions (six per clip; needs `GEMINI_API_KEY` and a file of example
captions that sets the caption style):

    python -m prepare.extract_image_features --data_dir dataset/AnimalMotion
    python -m prepare.generate_captions --data_dir dataset/AnimalMotion --example_captions examples.txt

### Train

Text- and image-conditioned model:

    python -m train.train_mdm --save_dir save/text_image --dataset animal \
      --arch trans_dec --text_encoder_type bert --emb_policy concat \
      --image_condition --cond_mask_prob 0.2 --cond_img_prob 0.1 \
      --mask_frames --use_ema --grad_clip_norm 0.5 --diffusion_steps 50 \
      --batch_size 512 --num_steps 150000 --use_cache --train_platform_type NoPlatform

Text-only model:

    python -m train.train_mdm --save_dir save/text --dataset animal \
      --arch trans_dec --text_encoder_type bert --emb_policy concat \
      --diffusion_steps 50 --batch_size 128 --num_steps 600000 --train_platform_type NoPlatform

Checkpoints (`model*.pt`, with EMA weights when `--use_ema` is set) and
`args.json` are written to `--save_dir`. `--use_cache` stores the loaded
training set as `dataset/AnimalMotion/animal_train.npy` on the first run; delete
it after changing `train.txt`. `--seed N` makes a run repeatable, and
`--train_platform_type WandBPlatform` logs to Weights & Biases (`pip install wandb`).

The evaluator (text and motion encoders on the 26-joint unified skeleton) is
trained with the following command and saved to `animal_unified/text_mot_match/`:

    python train_evaluator.py --name text_mot_match --dataset_name animal --batch_size 8 --gpu_id 0

### Inference

Download the pretrained models from
[Hugging Face](https://huggingface.co/briannlongzhao/kirin) into `save/`:

    hf download briannlongzhao/kirin --include "kirin_*/*" --local-dir save

This gives `save/kirin_text_image/` (text- and image-conditioned) and
`save/kirin_text/` (text only). Each folder holds the weights (`model.pt`), the
model configuration (`args.json`) and the normalization statistics
(`Mean.npy`, `Std.npy`) that generation uses. Image inputs are encoded with
DINOv3, a gated model: request access at
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and log in with
`hf auth login`.

From a text prompt and an image of the animal:

    python generate.py --model_path save/kirin_text_image/model.pt \
      --text_prompt "a horse rears up on its hind legs" \
      --image_path /path/to/horse.png --output_dir out/horse

From an image only:

    python generate.py --model_path save/kirin_text_image/model.pt \
      --no_text --image_path /path/to/horse.png --output_dir out/horse_image

From a text prompt only:

    python generate.py --model_path save/kirin_text/model.pt \
      --text_prompt "a horse rears up on its hind legs" --output_dir out/horse_text

Each motion is saved as `sample_NN.npy`, a `(T, 34, 3)` sequence of the SMAL
joints without the duplicated root (Y up, 10 fps), and all motions together in
`results_MODEL_NAME.npy`. The joint files can be passed to `render_motion.py`,
`view_motion.py` and `animate_mesh.py`. `--num_samples` and `--num_repetitions`
set how many motions are sampled (`num_samples * num_repetitions`),
`--guidance_param` the classifier-free guidance scale (default 2.5),
`--motion_length L` the length as `20 * L` frames (at most 196), and `--seed N`
makes a run repeatable.

### Evaluate

Download the evaluator and word vectors, generate motions for every caption of
every test clip (saved as `results_NAME.npy` in each clip folder), then compute
R-precision, FID, MM-Dist, Diversity and MultiModality:

    hf download briannlongzhao/kirin --include "animal_unified/*" --local-dir .
    bash prepare/download_glove.sh
    python generate.py --model_path save/kirin_text_image/model.pt --image_condition \
      --test_data_dir dataset/AnimalMotion_test --test_dataset animal \
      --num_samples 2 --num_repetitions 5 --motion_length 20 --save_name kirin_text_image
    python evaluate.py --dataset animal --use_cache \
      --results_files kirin_text_image:results_kirin_text_image.npy --log_dir eval_logs

Add `--no_text` to `generate.py` for image-only generation, and drop
`--image_condition` for `kirin_text`. Metrics are averaged over 10 replications
and written to `eval_logs/`.
