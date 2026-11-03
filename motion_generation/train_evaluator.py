import os

from os.path import join as pjoin
import torch
import argparse

from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import TextMotionMatchTrainer
from data_loaders.humanml.data.dataset import collate_fn, Text2MotionDatasetAnimal
from data_loaders.humanml.scripts.motion_process import *
from torch.utils.data import DataLoader
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer, POS_enumerator


class TrainTexMotMatchOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1,
                                 help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='animal', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--decomp_name', type=str, default="Decomp_SP001_SM001_H512", help='Name of this trial')

        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--dim_movement_enc_hidden', type=int, default=512,
                                 help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_movement_latent', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--dim_text_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_motion_hidden', type=int, default=1024, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--dim_coemb_hidden', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Training iterations')
        self.parser.add_argument('--estimator_mod', type=str, default='bigru')
        self.parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')
        self.parser.add_argument('--negative_margin', type=float, default=10.0)

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Layers of GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='Training iterations')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of printing training progress')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of printing training progress')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        args = vars(self.opt)
        return self.opt


def build_models(opt):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    # if not opt.is_continue:
    #    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
    #                            map_location=opt.device)
    #    movement_enc.load_state_dict(checkpoint['movement_enc'])
    return text_enc, motion_enc, movement_enc


if __name__ == '__main__':
    parser = TrainTexMotMatchOptions()
    opt = parser.parse()

    opt.data_root = "dataset/AnimalMotion"
    save_root = "./animal_unified"
   
    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(save_root, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    opt.eval_dir = pjoin(opt.save_root, 'eval')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)


    # opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    # opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 34
    opt.max_motion_length = 196
    dim_pose = 311  # features of the 26-joint unified skeleton
    num_classes = 200 // opt.unit_length
    meta_root = pjoin(save_root, 'Comp_v6_KLD01', 'meta')


    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)

    mean = np.load("dataset/AnimalMotion/Mean.npy")
    std = np.load("dataset/AnimalMotion/Std.npy")

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'test.txt')

    text_encoder, motion_encoder, movement_encoder = build_models(opt)

    pc_text_enc = sum(param.numel() for param in text_encoder.parameters())
    print(text_encoder)
    print("Total parameters of text encoder: {}".format(pc_text_enc))
    pc_motion_enc = sum(param.numel() for param in motion_encoder.parameters())
    print(motion_encoder)
    print("Total parameters of motion encoder: {}".format(pc_motion_enc))
    print("Total parameters: {}".format(pc_motion_enc + pc_text_enc))


    trainer = TextMotionMatchTrainer(opt, text_encoder, motion_encoder, movement_encoder)

    opt.image_condition = False
    opt.use_cache = True
    opt.unify_joints = True
    opt.unify_joints_mean = "./animal_unified/mean_unified.npy"
    opt.unify_joints_std = "./animal_unified/std_unified.npy"
    train_dataset = Text2MotionDatasetAnimal(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = Text2MotionDatasetAnimal(opt, mean, std, val_split_file, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    trainer.train(train_loader, val_loader)