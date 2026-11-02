# Kirin

Code for *Kirin: Animal Motion Generation from In-the-Wild Video*:
3D motion reconstruction from video, motion generation, and mesh animation.

## Installation

    git clone --recurse-submodules git@github.com:briannlongzhao/kirin.git
    cd kirin
    git apply patches/spatrack-weighted-procrustes.patch

    conda env create -f environment.yml
    conda activate kirin

    # PyTorch3D builds from source and needs CUDA 12.4 (e.g. CUDA_HOME=/usr/local/cuda-12.4)
    python -m pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git@51d5afd92a8ded3637553be8cef41f328a1c863a"
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@e3d3a67a89907476bd5b63289f9669bd427ae550"
    python -m pip install --no-deps -e AniMer
    python -m pip install --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
    python -m pip install --no-deps opencv-python-headless==4.13.0.90 viser==1.0.4
    python -m pip install msgspec nodeenv "websockets>=13.1,<16" yourdfpy

Download the [AniMer checkpoint folder](https://drive.google.com/drive/folders/1xYCJyxZeSYPI6RXnHqa_lDbjnIAs8hVi?usp=sharing)
into `AniMer/data/`. SAM3 and DINOv3 are gated on Hugging Face: request access to
[facebook/sam3](https://huggingface.co/facebook/sam3) and
[facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
then run `hf auth login`.

## Motion reconstruction

    python run_reconstruction.py \
      --rgb-video CLIP_ID/CLIP_ID_rgb.mp4 \
      --metadata CLIP_ID/CLIP_ID_info.json \
      --checkpoint-path AniMer/data/AniMer/checkpoints/checkpoint.ckpt \
      --out-folder output/CLIP_ID \
      --global-motion uncropped --uncropped-video FULL_FRAME.mp4

`--global-motion` is `none` (camera space), `cropped`, or `uncropped` (needs
`--uncropped-video`). The motion is saved as `joints_global.npy`.

## Visualization

    python render_motion.py --joints-path output/CLIP_ID/joints_global.npy
    python view_motion.py --joints-path output/CLIP_ID/joints_global.npy --sample-dir CLIP_ID

## Mesh animation

    export RODIN_API_KEY=...
    python animate_mesh.py \
      --joints-path output/CLIP_ID/joints_global.npy \
      --image-path IMAGE.png --output-dir animation

Use `--tpose-mesh base.obj` instead of `--image-path` to reuse an existing mesh.
View the result with `python view_motion.py --animation-dir animation`.

## Motion generation

Run from `motion_generation/`.

### Data

The AiM3D dataset is available at
https://download.cs.stanford.edu/viscam/AiM/AiM_full/index.html; place it in
`dataset/AnimalMotion/`. Download the test set for evaluation:

    wget https://download.cs.stanford.edu/viscam/AiM/AiM_preview.zip
    unzip AiM_preview.zip && mkdir -p dataset && mv data dataset/AnimalMotion_test

### Train

    # text + image
    python -m train.train_mdm --save_dir save/text_image --dataset animal \
      --arch trans_dec --text_encoder_type bert --emb_policy concat \
      --image_condition --cond_mask_prob 0.2 --cond_img_prob 0.1 \
      --mask_frames --use_ema --grad_clip_norm 0.5 --diffusion_steps 50 \
      --batch_size 512 --num_steps 150000 --use_cache --train_platform_type NoPlatform

    # text only
    python -m train.train_mdm --save_dir save/text --dataset animal \
      --arch trans_dec --text_encoder_type bert --emb_policy concat \
      --diffusion_steps 50 --batch_size 128 --num_steps 600000 --train_platform_type NoPlatform

### Inference

Download the pretrained models from [Hugging Face](https://huggingface.co/briannlongzhao/kirin):

    hf download briannlongzhao/kirin --include "kirin_*/*" --local-dir save

    # text + image
    python generate.py --model_path save/kirin_text_image/model.pt \
      --text_prompt "a horse rears up on its hind legs" --image_path horse.png --output_dir out/horse
    # image only
    python generate.py --model_path save/kirin_text_image/model.pt \
      --no_text --image_path horse.png --output_dir out/horse_image
    # text only
    python generate.py --model_path save/kirin_text/model.pt \
      --text_prompt "a horse rears up on its hind legs" --output_dir out/horse_text

### Evaluate

    hf download briannlongzhao/kirin --include "animal_unified/*" --local-dir .
    bash prepare/download_glove.sh
    python generate.py --model_path save/kirin_text_image/model.pt --image_condition \
      --test_data_dir dataset/AnimalMotion_test --test_dataset animal \
      --num_samples 2 --num_repetitions 5 --motion_length 20 --save_name kirin_text_image
    python evaluate.py --dataset animal --use_cache \
      --results_files kirin_text_image:results_kirin_text_image.npy --log_dir eval_logs

For `kirin_text`, use its model path and drop `--image_condition`.
