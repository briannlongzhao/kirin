import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from random import shuffle

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.scripts.motion_process_animal import joints_to_vecs, vecs_to_joints, remove_global_translation
from data_loaders.humanml.utils.paramUtil import unify_smal_joints, unify_animo_joints

# import spacy

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def caption_file(seq_dir):
    """A clip's caption file: texts_gemini.txt, or captions.txt in the AiM releases."""
    path = os.path.join(seq_dir, "texts_gemini.txt")
    return path if os.path.exists(path) else os.path.join(seq_dir, "captions.txt")


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


'''For use of training text-2-motion generative model'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        if self.opt.fixed_len > 0:
            self.max_length = self.opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []

        _split = os.path.basename(split_file).replace('.txt', '')
        _name =''
        # cache_path = os.path.join(opt.meta_dir, self.opt.dataset_name + '_' + _split + _name + '.npy')
        cache_path = os.path.join(opt.cache_dir, 'dataset', self.opt.dataset_name + '_' + _split + _name + '.npy')
        if opt.use_cache and os.path.exists(cache_path):
            print(f'Loading motions from cache file [{cache_path}]...')
            _cache = np.load(cache_path, allow_pickle=True)[None][0]
            name_list, length_list, data_dict = _cache['name_list'], _cache['length_list'], _cache['data_dict']
            # name_list = name_list[:15]; length_list = length_list[:15]
            # data_dict = {key: data_dict[key] for key in name_list}
        else:
            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 200):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                           'length': len(n_motion),
                                                           'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    pass

            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            print(f'Saving motions to cache file [{cache_path}]...')
            np.save(cache_path, {
                'name_list': name_list,
                'length_list': length_list,
                'data_dict': data_dict})

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print(f"Dataset size: {len(self.data_dict)}")

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def process_text(self, sentence):
        if not hasattr(self, 'nlp'):
            self.nlp = spacy.load('en_core_web_sm')
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a valid caption.
        # Some generated sets (e.g. no-text variants) may contain empty captions/tokens,
        # which would otherwise cause an infinite retry loop here.
        valid_text_list = [
            t for t in text_list
            if t is not None and t.get("caption", "").strip() != '' and len(t.get("tokens", [])) > 1
        ]
        if len(valid_text_list) > 0:
            text_data = random.choice(valid_text_list)
            caption, tokens = text_data['caption'], text_data['tokens']
        else:
            caption = 'unknown motion'
            tokens = ['unk/OTHER']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            if '/' not in token:
                print(key)
                print(text_data)
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        
        original_length = None
        if hasattr(self.opt, 'fixed_len') and self.opt.fixed_len > 0:
            # Crop fixed_len
            original_length = m_length
            m_length = self.opt.fixed_len
        
        idx = random.randint(0, len(motion) - m_length)
        if hasattr(self.opt, 'disable_offset_aug') and self.opt.disable_offset_aug:
            idx = random.randint(0, self.opt.unit_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)

        length = (original_length, m_length) if hasattr(self.opt, 'fixed_len') and self.opt.fixed_len > 0 else m_length

        return word_embeddings, pos_one_hots, caption, sent_len, motion, length, '_'.join(tokens)


class Text2MotionDatasetAnimal(Text2MotionDatasetV2):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        if hasattr(self.opt, 'fixed_len') and self.opt.fixed_len > 0:
            self.max_length = self.opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40

        data_dict = {}
        id_list = []
        if os.path.exists(split_file):
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        

        _split = os.path.basename(split_file).replace('.txt', '')
        _name =''
        # cache_path = os.path.join(opt.meta_dir, self.opt.dataset_name + '_' + _split + _name + '.npy')
        cache_path = os.path.join(opt.data_root, self.opt.dataset_name + '_' + _split + _name + '.npy')
        if hasattr(opt, 'use_cache') and opt.use_cache and os.path.exists(cache_path):
            print(f'Loading motions from cache file [{cache_path}]...')
            _cache = np.load(cache_path, allow_pickle=True)[None][0]
            name_list, length_list, data_dict = _cache['name_list'], _cache['length_list'], _cache['data_dict']
        else:
            count = 0
            all_sequence_dirs = get_all_sequence_dirs(opt.data_root, sorted=True)
            all_motion_paths = [os.path.join(d, "new_joint_vecs_global.npy") for d in all_sequence_dirs]
            all_text_paths = [caption_file(d) for d in all_sequence_dirs]
            for motion_path, text_path in tqdm(zip(all_motion_paths, all_text_paths), total=len(all_motion_paths), desc="Loading data"):
                try:
                    name = os.path.basename(os.path.dirname(text_path))
                    if len(id_list) > 0 and name not in id_list:
                        continue
                    motion = np.load(motion_path)
                    if len(motion) >= 200:
                        motion = motion[:200]
                    name = os.path.basename(os.path.dirname(motion_path))
                    text_data = []
                    with cs.open(text_path) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            text_data.append(text_dict)
                    action_file = os.path.join(os.path.dirname(text_path), "actions.txt")
                    actions = []
                    with cs.open(action_file) as f:
                        for line in f.readlines():
                            actions.append(line.strip())
                    if opt.image_condition:
                        feature_path = os.path.join(os.path.dirname(text_path), "mean_feature_dinov3.npy")
                        feature = np.load(feature_path)
                    else:
                        feature = None
                    
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data,
                        'action': actions,
                        'feature': feature
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e

                # count += 1
                # if count >= 100:
                #     break

            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            if hasattr(opt, 'use_cache') and opt.use_cache:
                print(f'Saving motions to cache file [{cache_path}]...')
                np.save(cache_path, {
                    'name_list': name_list,
                    'length_list': length_list,
                    'data_dict': data_dict})

        if hasattr(opt, 'unify_joints') and opt.unify_joints:
            print("Unifying joints...")
            new_data_dict = {}
            for name, data in data_dict.items():
                motion = data['motion']
                motion_joints = vecs_to_joints(motion, joints_num=34)
                motion_joints = unify_smal_joints(motion_joints)
                motion = joints_to_vecs(motion_joints)
                new_data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': data['text'],
                    'action': data['action'],
                    'feature': data.get('feature', None)
                }
            data_dict = new_data_dict
            all_motions = np.concatenate([data['motion'] for data in data_dict.values()], axis=0)
            if os.path.exists(opt.unify_joints_mean):
                mean = np.load(opt.unify_joints_mean)
            else:
                mean = np.mean(all_motions, axis=0)
                np.save(opt.unify_joints_mean, mean)
            if os.path.exists(opt.unify_joints_std):
                std = np.load(opt.unify_joints_std)
            else:
                std = np.std(all_motions, axis=0)
                np.save(opt.unify_joints_std, std)


        self.mean = mean
        self.std = std
        self.std[self.std == 0] = 1.0  # To avoid division by zero
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        # self.balance_dataset()
        self.reset_max_len(self.max_length)
        print(f"Dataset size: {len(self.data_dict)}")

    def __getitem__(self, item):
        word_embeddings, pos_one_hots, caption, sent_len, motion, length, tokens = super().__getitem__(item)
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]
        image_feature = data.get('feature', None)
        if image_feature is not None:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, length, tokens, image_feature
        return word_embeddings, pos_one_hots, caption, sent_len, motion, length, tokens

    def balance_dataset(self):
        # Ensure name_list is mutable
        if not isinstance(self.name_list, list):
            self.name_list = list(self.name_list)

        # 1) Compute current action counts (only count first action in list)
        action_to_names = {}
        for name in self.name_list:
            action_list = self.data_dict[name].get('action', [])
            # Only use the first action if it's a list, otherwise treat as string
            if isinstance(action_list, list) and len(action_list) > 0:
                action = action_list[0].strip()
            elif isinstance(action_list, str):
                action = action_list.strip()
            else:
                action = ''
            if action:
                action_to_names.setdefault(action, []).append(name)
        
        if len(action_to_names) == 0:
            print('Action distribution: {}')
            return
        
        action_to_num = {action: len(names) for action, names in action_to_names.items()}
        print(f"Action distribution before balancing: {action_to_num}")

        # 2) Compute target count (use whichever is higher: mean or median)
        counts = list(action_to_num.values())
        mean_count = int(np.mean(counts))
        median_count = int(np.median(counts))
        target_count = max(mean_count, median_count)
        if target_count < 1:
            target_count = 1
        print(f"Mean action count: {mean_count}, Median action count: {median_count}, Using: {target_count}")

        # 3) Downsample actions above the target
        for action, names in list(action_to_names.items()):
            if len(names) > target_count:
                # randomly select which entries to keep
                keep_names = random.sample(names, target_count)
                remove_names = [n for n in names if n not in keep_names]
                for n in remove_names:
                    self.data_dict.pop(n, None)
                    try:
                        self.name_list.remove(n)
                    except ValueError:
                        pass
                action_to_names[action] = keep_names
        
        # 4) Upsample actions below the target
        for action, names in list(action_to_names.items()):
            if len(names) < target_count:
                need = target_count - len(names)
                # Track how many times each entry has been upsampled
                upsample_count = {name: 0 for name in names}
                
                for _ in range(need):
                    # Choose entry with minimum upsample count
                    min_count = min(upsample_count.values())
                    candidates = [name for name, count in upsample_count.items() if count == min_count]
                    orig = random.choice(candidates)
                    upsample_count[orig] += 1
                    
                    # Extract the last number in the string and increment it
                    import re
                    # Find all numbers in the string
                    numbers = re.findall(r'\d+', orig)
                    if numbers:
                        # Get the last number
                        last_num = numbers[-1]
                        last_num_int = int(last_num)
                        # Find the position of the last number
                        last_pos = orig.rfind(last_num)
                        # Increment and create new name
                        new_num = str(last_num_int + 1).zfill(len(last_num))
                        new_name = orig[:last_pos] + new_num + orig[last_pos + len(last_num):]
                    else:
                        # No numbers found, append _001
                        new_name = f"{orig}_001"
                    
                    # Ensure uniqueness
                    while new_name in self.data_dict:
                        # If collision, keep incrementing
                        numbers = re.findall(r'\d+', new_name)
                        if numbers:
                            last_num = numbers[-1]
                            last_num_int = int(last_num)
                            last_pos = new_name.rfind(last_num)
                            new_num = str(last_num_int + 1).zfill(len(last_num))
                            new_name = new_name[:last_pos] + new_num + new_name[last_pos + len(last_num):]
                        else:
                            new_name = f"{new_name}_001"
                    
                    d = self.data_dict[orig]
                    self.data_dict[new_name] = {
                        'motion': d['motion'], 
                        'length': d['length'], 
                        'text': d.get('text', []), 
                        'action': d.get('action', []),
                        'feature': d.get('feature', None)
                    }
                    self.name_list.append(new_name)
                    action_to_names[action].append(new_name)
        
        # 5) Update length_arr and report final distribution
        self.length_arr = np.array([self.data_dict[n]['length'] for n in self.name_list]) if len(self.name_list) > 0 else np.array([])
        
        action_to_num = {}
        for name in self.name_list:
            action_list = self.data_dict[name].get('action', [])
            # Only use the first action if it's a list, otherwise treat as string
            if isinstance(action_list, list) and len(action_list) > 0:
                action = action_list[0].strip()
            elif isinstance(action_list, str):
                action = action_list.strip()
            else:
                action = ''
            if action:
                action_to_num[action] = action_to_num.get(action, 0) + 1
        print(f"Final action distribution: {action_to_num}")


class Text2MotionDatasetAnimalGTEval(Text2MotionDatasetAnimal):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        if hasattr(self.opt, 'fixed_len') and self.opt.fixed_len > 0:
            self.max_length = self.opt.fixed_len
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40

        length_list = []
        new_name_list = []
        data_dict = {}

        _split = os.path.basename(split_file).replace('.txt', '')
        _name =''
        cache_path = os.path.join(opt.data_root, self.opt.dataset_name + '_' + _split + _name + '.npy')
        if hasattr(opt, 'use_cache') and opt.use_cache and os.path.exists(cache_path):
            print(f'Loading motions from cache file [{cache_path}]...')
            _cache = np.load(cache_path, allow_pickle=True)[None][0]
            name_list, length_list, data_dict = _cache['name_list'], _cache['length_list'], _cache['data_dict']
        else:
            all_sequence_dirs = get_all_sequence_dirs(opt.data_root, sorted=True)
            all_motion_paths = [os.path.join(d, "new_joint_vecs_global.npy") for d in all_sequence_dirs]
            all_text_paths = [caption_file(d) for d in all_sequence_dirs]
            for motion_path, text_path in tqdm(zip(all_motion_paths, all_text_paths), total=len(all_motion_paths), desc="Loading data"):
                try:
                    name = os.path.basename(os.path.dirname(text_path))
                    motion = np.load(motion_path)
                    if len(motion) > 200:
                        motion = motion[:200]
                    name = os.path.basename(os.path.dirname(motion_path))
                    text_data = []
                    with cs.open(text_path) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            text_data.append(text_dict)
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except Exception as e:
                    print(e)
                    pass

            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            if hasattr(opt, 'use_cache') and opt.use_cache:
                print(f'Saving motions to cache file [{cache_path}]...')
                np.save(cache_path, {
                    'name_list': name_list,
                    'length_list': length_list,
                    'data_dict': data_dict})
                    
        # duplicate data for each text
        new_length_list = []
        new_name_list = []
        new_data_dict = {}
        for name, data in data_dict.items():
            motion = data['motion']
            length = data['length']
            text_list = data['text']
            for i, text_data in enumerate(text_list):
                new_name = name + f'_{i}'
                new_data_dict[new_name] = {
                    'motion': motion,
                    'length': length,
                    'text': [text_data]
                }
                new_name_list.append(new_name)
                new_length_list.append(length)
        data_dict = new_data_dict
        name_list = new_name_list
        length_list = new_length_list

        if hasattr(opt, 'unify_joints') and opt.unify_joints:
            new_data_dict = {}
            for name, data in data_dict.items():
                motion = data['motion']
                motion_joints = vecs_to_joints(motion, joints_num=34)
                motion_joints = unify_smal_joints(motion_joints)
                motion = joints_to_vecs(motion_joints)
                new_data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': data['text']
                }
            data_dict = new_data_dict
            all_motions = np.concatenate([data['motion'] for data in data_dict.values()], axis=0)
            if os.path.exists(opt.unify_joints_mean):
                mean = np.load(opt.unify_joints_mean)
            else:
                mean = np.mean(all_motions, axis=0)
                np.save(opt.unify_joints_mean, mean)
            if os.path.exists(opt.unify_joints_std):
                std = np.load(opt.unify_joints_std)
            else:
                std = np.std(all_motions, axis=0)
                np.save(opt.unify_joints_std, std)

        self.mean = mean
        self.std = std
        self.std[self.std == 0] = 1.0  # To avoid division by zero
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        print(f"Dataset size: {len(self.data_dict)}")


class Text2MotionDatasetAnimalGenEval(Text2MotionDatasetAnimal):
    # Load data from result.npy in each folder
    def __init__(self, opt, mean, std, split_file, w_vectorizer, results_file=None, max_samples=None, sequence_names=None):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        if hasattr(self.opt, 'fixed_len') and self.opt.fixed_len > 0:
            self.max_length = self.opt.fixed_len
        self.pointer = 0
        self.nlp = spacy.load('en_core_web_sm')
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40

        data_dict = {}
        id_list = []

        self.name_list = []
        length_list = []

        _split = os.path.basename(split_file).replace('.txt', '')
        _name = ''
        all_sequence_dirs = get_all_sequence_dirs(opt.data_root, sorted=True)
        if sequence_names is not None:
            sequence_name_set = set(sequence_names)
            all_sequence_dirs = [d for d in all_sequence_dirs if os.path.basename(d) in sequence_name_set]
        all_results_paths = [os.path.join(d, results_file) for d in all_sequence_dirs]

        count = 0
        total_loaded = 0

        for results_path in tqdm(all_results_paths, total=len(all_results_paths), desc="Loading data"):
            count += 1
            data = np.load(results_path, allow_pickle=True).item()
            for i, (motion, text, length) in enumerate(zip(data['motion_vecs'], data['texts'], data['lengths'])):
                if max_samples is not None and total_loaded >= max_samples:
                    break
                name = os.path.basename(os.path.dirname(results_path)) + f'_{i}'
                text_data = {}
                text_data['caption'] = text
                tokens = []
                word_list, pos_list = self.process_text(text.strip())
                tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
                text_data['tokens'] = tokens
                text_data = [text_data]
                data_dict[name] = {
                    'motion': motion,
                    'length': length,
                    'text': text_data
                }
                self.name_list.append(name)
                total_loaded += 1
            if max_samples is not None and total_loaded >= max_samples:
                print(f'Limiting to {max_samples} generated samples for evaluation')
                break
            # if count >= 33:
            #     break

        if hasattr(opt, 'unify_joints') and opt.unify_joints:
            new_data_dict = {}
            for name, data in data_dict.items():
                motion = data['motion']
                
                if "animo" in results_file:
                    motion_joints = vecs_to_joints(motion, joints_num=30)
                    motion_joints = unify_animo_joints(motion_joints)
                else:
                    motion_joints = vecs_to_joints(motion, joints_num=34)
                    motion_joints = unify_smal_joints(motion_joints)
                # motion_joints = remove_global_translation(motion_joints)
                motion = joints_to_vecs(motion_joints)
                new_data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': data['text']
                }
            data_dict = new_data_dict
            all_motions = np.concatenate([data['motion'] for data in data_dict.values()], axis=0)
            if os.path.exists(opt.unify_joints_mean):
                mean = np.load(opt.unify_joints_mean)
            else:
                mean = np.mean(all_motions, axis=0)
                np.save(opt.unify_joints_mean, mean)
            if os.path.exists(opt.unify_joints_std):
                std = np.load(opt.unify_joints_std)
            else:
                std = np.std(all_motions, axis=0)
                np.save(opt.unify_joints_std, std)

        self.mean = mean
        self.std = std
        self.std[self.std == 0] = 1.0  # To avoid division by zero
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.reset_max_len(self.max_length)
        print(f"Dataset size: {len(self.data_dict)}")
        

class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


class TextOnlyAnimalDataset(TextOnlyDataset):
    def __init__(self, opt, mean, std, split_file):
        super(TextOnlyAnimalDataset, self).__init__(opt, mean, std, split_file)
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        all_sequence_dirs = get_all_sequence_dirs(opt.data_root, sorted=True)
        all_text_paths = [caption_file(d) for d in all_sequence_dirs]
        for text_path in tqdm(all_text_paths, total=len(all_text_paths), desc="Loading text data"):
            try:
                name = os.path.basename(os.path.dirname(text_path))
                if name not in id_list:
                    continue
                text_data = []
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        text_data.append(text_dict)
                data_dict[name] = {'text': text_data}
                new_name_list.append(name)
            except Exception as e:
                print(e)
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

# A wrapper class for t2m original dataset for MDM purposes
class AnimalMotion(data.Dataset):
    def __init__(self, mode, datapath='./dataset/animal_opt.txt', split="train", name="animal", **kwargs):
        self.mode = mode

        self.dataset_name = name
        self.dataname = name

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = kwargs.get('abs_path', '.')
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = kwargs.get('device', None)
        opt = get_opt(dataset_opt_path, device)
        # opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.cache_dir = kwargs.get('cache_path', '.')
        if opt.motion_dir is not None:
            opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        if opt.text_dir is not None:
            opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = pjoin(abs_base_path, './dataset')
        opt.use_cache = kwargs.get('use_cache', True)
        opt.fixed_len = kwargs.get('fixed_len', 0)
        opt.image_condition = kwargs.get('image_condition', False)
        if opt.fixed_len > 0:
            opt.max_motion_length = opt.fixed_len
        is_autoregressive = kwargs.get('autoregressive', False)
        opt.disable_offset_aug = is_autoregressive and (opt.fixed_len > 0) and (mode == 'eval')  # for autoregressive evaluation, use the start of the motion and not something from the middle
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        # Training statistics; evaluation modes use the evaluator's unified statistics instead.
        stats_dir = "dataset/AnimalMotion"
        self.mean = np.load(pjoin(stats_dir, "Mean.npy")) if os.path.exists(pjoin(stats_dir, "Mean.npy")) else None
        self.std = np.load(pjoin(stats_dir, "Std.npy")) if os.path.exists(pjoin(stats_dir, "Std.npy")) else None

        # if mode == 'gt':
        #     # used by T2M models (including evaluators)
        #     self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
        #     self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        # elif mode in ['train', 'eval', 'text_only']:
        #     # used by our models
        #     self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        #     self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        # if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            # self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            # self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')

        if mode in ["gt", "eval"]:
            self.opt.unify_joints = True
            self.opt.unify_joints_mean = "./animal_unified/mean_unified.npy"
            self.opt.unify_joints_std = "./animal_unified/std_unified.npy"
            self.opt.data_root = str(self.opt.data_root) + "_test"
            self.split_file = pjoin(opt.data_root, os.path.basename(self.split_file))
            self.w_vectorizer = WordVectorizer(pjoin(opt.cache_dir, 'glove'), 'our_vab')

        if mode == 'text_only':
            self.t2m_dataset = TextOnlyAnimalDataset(self.opt, self.mean, self.std, self.split_file)
        elif mode == "eval":
            self.t2m_dataset = Text2MotionDatasetAnimalGenEval(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, results_file=kwargs.get('results_file'))
        elif mode == "gt":
            self.t2m_dataset = Text2MotionDatasetAnimalGTEval(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(opt.cache_dir, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetAnimal(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer)
            self.num_actions = 1 # dummy placeholder

        if self.mean is not None:
            self.mean_gpu = torch.tensor(self.mean).to(device)[None, :, None, None]
            self.std_gpu = torch.tensor(self.std).to(device)[None, :, None, None]

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

