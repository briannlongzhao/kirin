# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric, get_target_location, sample_goal
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from tqdm import tqdm
from data_loaders.tensors import collate
from moviepy.editor import clips_array
import re
from data_loaders.humanml.data.dataset import get_all_sequence_dirs, caption_file



def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, no_dir=False):
    
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]
    
    for sample_i in range(0,n_samples,num_samples_in_out_file):
        last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
        if no_dir and n_samples <= num_samples_in_out_file:
            all_sample_save_path = out_path
        else:
            all_sample_save_path = os.path.join(out_path, all_sample_save_file)
            print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important
 

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template



def is_substr_in_list(substr, list_of_strs):
    return np.char.find(list_of_strs, substr) != -1  # [substr in string for string in list_of_strs]



def generate(args, model, text_prompt, image_feature=None):  # Single text prompt generation
    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, [text_prompt] * args.num_samples)]
    _, model_kwargs = collate(collate_args)

    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
    init_image = None    
    
    all_motions = []
    all_lengths = []
    all_text = []
    all_motion_vecs = []

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    
    if 'text' in model_kwargs['y'].keys():
        # encoding once instead of each iteration saves lots of time
        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
    
    if args.dynamic_text_path != '':
        # Rearange the text to match the autoregressive sampling - each prompt fits to a single prediction
        # Which is 2 seconds of motion by default
        model_kwargs['y']['text'] = [model_kwargs['y']['text']] * args.num_samples
        if args.text_encoder_type == 'bert':
            model_kwargs['y']['text_embed'] = (model_kwargs['y']['text_embed'][0].unsqueeze(0).repeat(args.num_samples, 1, 1, 1), 
                                               model_kwargs['y']['text_embed'][1].unsqueeze(0).repeat(args.num_samples, 1, 1))
        else:
            raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
        
    if args.image_condition:
        assert image_feature is not None, "Image feature must be provided for image-conditioned generation!"
        model_kwargs['y']['feature'] = torch.as_tensor(image_feature, device=dist_util.dev()).repeat(args.num_samples, 1, 1)
    
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        sample = sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if data is not None:
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        else:
            sample = (sample.cpu().permute(0, 2, 3, 1) * std + mean).float()
        all_motion_vecs.append(sample.squeeze().cpu().numpy())
        sample = recover_from_ric(sample, njoints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        _len = model_kwargs['y']['lengths'].cpu().numpy()
        if 'prefix' in model_kwargs['y'].keys():
            _len[:] = sample.shape[-1]
        all_lengths.append(_len)

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_motion_vecs = np.concatenate(all_motion_vecs, axis=0)

    return all_motions, all_lengths, all_text, all_motion_vecs



def extract_image_feature(image_path):
    """DINOv3 [CLS] feature (1, 768) of one image, matching the per-clip training features."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"--image_path does not exist: {image_path}")
    from transformers import pipeline
    extractor = pipeline(model="facebook/dinov3-vitb16-pretrain-lvd1689m", task="image-feature-extraction",
                         device=0 if torch.cuda.is_available() else -1)
    return np.array(extractor([image_path]))[..., 0, :].reshape(1, 768)


def get_model_name(args, save_name=None):
    if save_name is None:
        model_name = os.path.basename(os.path.dirname(args.model_path))
        model_name = re.sub(r'[^A-Za-z0-9_-]+', '_', model_name)
        model_name = re.sub(r'_+', '_', model_name).strip('_')
    else:
        model_name = save_name
    return model_name


def get_npy_path(args, out_path, save_name=None):
    model_name = get_model_name(args, save_name=save_name)
    return os.path.join(out_path, f'results_{model_name}.npy')


def save(args, motions, lengths, texts, motion_vecs, out_path, npy_path, save_video=False, max_vis_samples=3, save_name=None):
    os.makedirs(out_path, exist_ok=True)
    model_name = get_model_name(args, save_name=save_name)
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motions': motions, 'motion_vecs': motion_vecs, 'texts': texts, 'lengths': lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})

    if not save_video:
        return

    print(f"saving video to [{out_path}]...")
    skeleton = paramUtil.t2m_animal_kinematic_chain
    
    # Build a mapping from text -> list of motion indices (preserve original behavior)
    text_to_indices = {}
    for idx, txt in enumerate(texts):
        key = txt if not isinstance(txt, list) else ' '.join(txt)
        text_to_indices.setdefault(key, []).append(idx)

    # For each unique text, randomly pick up to max_vis_samples motions and save one combined video
    rng = np.random.RandomState()
    max_length = int(max(lengths)) if len(lengths) > 0 else n_frames
    for text_key, indices in text_to_indices.items():
        if len(indices) == 0:
            continue
        k = min(len(indices), max_vis_samples)
        chosen = rng.choice(indices, size=k, replace=False)

        clips = []
        for idx in chosen:
            # Follow original code's motion formatting: transpose from (njoints, feats, seqlen) -> (seqlen, njoints, feats)
            motion = motions[idx]
            # motion = motion.transpose(2, 0, 1)[:max_length]

            length = int(lengths[idx])
            if motion.shape[0] > length:
                motion[length:-1] = motion[length-1]

            # sanitize filename: allow only alphanumerics, underscore and hyphen
            safe_text = re.sub(r'[^A-Za-z0-9_-]+', '_', text_key)
            safe_text = re.sub(r'_+', '_', safe_text).strip('_')
            if safe_text == '':
                safe_text = 'text'
            # cap filename base to 100 chars
            if len(safe_text) > 100:
                safe_text = safe_text[:100]
            base_name = f"{safe_text}_{model_name}"
            save_file = f"{base_name}.mp4"
            animation_save_path = os.path.join(out_path, save_file)
            gt_frames = np.arange(args.context_len) if args.context_len > 0 and not args.autoregressive else []
            clip = plot_3d_motion(animation_save_path, skeleton, motion, dataset="animal", title=text_key, fps=fps, gt_frames=gt_frames)
            clips.append(clip)

        # Combine and write the row video
        try:
            row = clips_array([clips])
            row.duration = max_length / fps
            safe_text = re.sub(r'[^A-Za-z0-9_-]+', '_', text_key)
            safe_text = re.sub(r'_+', '_', safe_text).strip('_')
            if safe_text == '':
                safe_text = 'text'
                model_name = 'model'
            base_name = f"{safe_text}_{model_name}"
            if len(base_name) > 100:
                base_name = base_name[:100]
            out_file = os.path.join(out_path, f"{base_name}.mp4")
            print(f"saving text video [{out_file}] with {len(clips)} clips")
            row.write_videofile(out_file, fps=fps, threads=4, logger=None)
        finally:
            for c in clips:
                try:
                    c.close()
                except Exception:
                    pass

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    args = generate_args()
    fixseed(args.seed)
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196
    fps = 20
    njoints = 34
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = False
    if args.text_prompt is not None and args.text_prompt != '':
        assert args.test_data_dir is None, "Cannot use both text prompt and test data directory!"
        is_using_data = False
    if args.test_data_dir is not None:
        assert args.text_prompt is None or args.text_prompt == '', "Cannot use both text prompt and test data directory!"
        is_using_data = True
    if args.context_len > 0:
        is_using_data = True  # For prefix completion, we need to sample a prefix
    dist_util.setup_dist(args.device)
    

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    data = None
    # Use the normalization statistics stored with the checkpoint when available.
    stats_dir = os.path.dirname(args.model_path)
    if not os.path.isfile(os.path.join(stats_dir, "Mean.npy")):
        stats_dir = "dataset/AnimalMotion"
    mean = np.load(os.path.join(stats_dir, "Mean.npy"))
    std = np.load(os.path.join(stats_dir, "Std.npy"))
    print(f"Using Mean.npy/Std.npy from [{stats_dir}]")
    total_num_samples = args.num_samples * args.num_repetitions
    # args.cond_mask_prob = 0
    # args.cond_img_prob = 0

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    args.image_condition = bool(args.image_condition or args.image_path is not None or args.image_feature_path is not None)
    if args.no_image:
        assert not args.no_text, "--no_text and --no_image cannot be combined"
        args.image_condition = False
    model.cond_mode = "text" if not args.image_condition else model.cond_mode

    sample_fn = diffusion.p_sample_loop
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    image_feature = None
    all_motions, all_lengths, all_text = None, None, None
    if not is_using_data:
        print(args.image_condition)
        if args.no_text or (args.image_condition and (args.text_prompt is None or args.text_prompt == '')):
            # Keep a placeholder token so image-only generation works with text-conditioned checkpoints.
            args.text_prompt = ' '
        if args.image_condition:
            assert args.image_path or args.image_feature_path, "Either image path or image feature path must be provided for image-conditioned generation!"
            if args.image_path:
                image_feature = extract_image_feature(args.image_path)
            else:
                image_feature = np.load(args.image_feature_path)
        all_motions, all_lengths, all_texts, all_motion_vecs = generate(args, model, args.text_prompt, image_feature=image_feature)
        out_path = args.output_dir
        if out_path == '':
            out_path = os.path.join(os.path.dirname(args.model_path), 'samples_{}_{}'.format(name, niter) + ('' if args.seed is None else f'_seed{args.seed}'))
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
        all_motions = np.transpose(all_motions, (0, 3, 1, 2))
        npy_path = get_npy_path(args, out_path)
        save(args, all_motions, all_lengths, all_texts, all_motion_vecs, out_path, npy_path, save_video=False)
        for i, (joints, length) in enumerate(zip(all_motions, all_lengths)):
            np.save(os.path.join(out_path, f'sample_{i:02d}.npy'), joints[:int(length)])
        print(f'Saved {len(all_motions)} joint sequences (T, {njoints}, 3) to [{out_path}]/sample_NN.npy')
    elif args.test_dataset.lower() == "animal":
        all_sequence_dirs = get_all_sequence_dirs(args.test_data_dir)
        np.random.shuffle(all_sequence_dirs)
        for seq_dir in tqdm(all_sequence_dirs, total=len(all_sequence_dirs)):
            npy_path = get_npy_path(args, seq_dir, save_name=args.save_name)
            if os.path.exists(npy_path):
                print(f'Results file already exists [{npy_path}]. Skipping sequence directory [{seq_dir}]...')
                continue
            print(f'Processing sequence directory [{seq_dir}]...')
            all_text_input = []
            if args.no_text:
                # Image-only mode for sequence generation: one placeholder prompt per sequence.
                all_text_input = [' ']
            else:
                text_file = caption_file(seq_dir)
                with open(text_file, 'r') as f:
                    text_lines = [line.strip() for line in f.readlines()]
                for line in text_lines:
                    all_text_input.append(line.split('#')[0].strip())
            image_feature = None
            if args.image_condition:
                # Load image feature
                image_feature = np.load(os.path.join(seq_dir, "mean_feature_dinov3.npy"))
            all_motions, all_lengths, all_texts, all_motion_vecs = [], [], [], []
            for text in all_text_input:
                print(f'Generating samples for text prompt: [{text}]')
                motions, lengths, texts, motion_vecs = generate(args, model, text, image_feature=image_feature)
                all_motions.append(motions)
                all_lengths.append(lengths)
                all_texts.extend(texts)
                all_motion_vecs.append(motion_vecs)
            all_motions = np.concatenate(all_motions, axis=0)
            all_motions = np.transpose(all_motions, (0, 3, 1, 2))
            all_lengths = np.concatenate(all_lengths, axis=0)
            all_motion_vecs = np.concatenate(all_motion_vecs, axis=0)
            save(args, all_motions, all_lengths, all_texts, all_motion_vecs, seq_dir, npy_path, save_video=False, max_vis_samples=1, save_name=args.save_name)
