import os
from os import path
import glob
import random

from data import common
from data.meta_learner import preprocessing
from data import random_kernel_generator as rkg

import numpy as np
import math, imageio
import data.util as util

import torch
import torch.utils.data as data


class Vimeo(data.Dataset):
    """Vimeo trainset class
    """
    def __init__(self, opt, train=True, **kwargs):
        super().__init__()
        self.data_root = opt['datasets']['train']['data_root']
        self.data_root = path.join(self.data_root, 'vimeo_septuplet')
        self.opt = opt
        self.scale = opt['scale']
        self.nframes = opt['datasets']['train']['N_frames']
        self.train = train
        if train:
            meta = "F:/vimeo_septuplet/sep_trainlist.txt"
            with open(meta, 'r') as f:
                self.img_list = sorted(f.read().splitlines())
        else:
            # meta = "F:/vimeo_septuplet/sep_testlist.txt"
            self.valopt = opt['datasets']['val']
            self.GT_root = self.valopt['dataroot_GT']
            self.half_N_frames = self.valopt['N_frames'] // 2
            self.cache_data = self.valopt['cache_data']
            self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
            #### Generate data info and cache data
            
            self.imgs_LQ, self.imgs_GT = {}, {}
            img_type = 'img'
            subfolders_GT = util.glob_file_list(self.GT_root)

            for subfolder_GT in subfolders_GT:
                subfolder_name = path.basename(subfolder_GT)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_GT)

                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT, img_type)
                    
        sigma_x = float(opt['sigma_x'])
        sigma_y = float(opt['sigma_y'])
        theta = float(opt['theta'])
        kernel_size = int(opt['kernel_size'])
        gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
        self.kernel_gen = rkg.Degradation(kernel_size, self.scale, **gen_kwargs)

    def __getitem__(self, index):

        if self.train:
            idx = index
            folder = self.img_list[index]
            name_hr = path.join(self.data_root, 'sequences', self.img_list[index])
            names_hr = sorted(glob.glob(path.join(name_hr, '*.png')))
            names = [path.splitext(path.basename(f))[0] for f in names_hr]
            names = [path.join(self.img_list[index], name) for name in names]
            
            imgs_GT = [imageio.imread(f) for f in names_hr]
            imgs_GT = np.stack(imgs_GT, axis=-1)
            start_frame = random.randint(0, 7-self.nframes)
            imgs_GT = imgs_GT[..., start_frame:start_frame+self.nframes]
            imgs_GT = preprocessing.np2tensor(imgs_GT)
            
            imgs_GT = preprocessing.crop_border(imgs_GT, border=[4, 4])
            imgs_GT = preprocessing.common_crop(imgs_GT, patch_size=self.opt['datasets']['train']['patch_size']*4)
            imgs_GT = util.augment(imgs_GT, self.opt['use_flip'], self.opt['use_rot'])
            imgs_LR, kernel = self.kernel_gen.apply(imgs_GT)
            imgs_LR = imgs_LR.mul(255).clamp(0, 255).round().div(255)
        else:
            folder = self.data_info['folder'][index]
            idx, max_idx = self.data_info['idx'][index].split('/')
            idx, max_idx = int(idx), int(max_idx)
            border = self.data_info['border'][index]
            select_idx = util.index_generation(idx, max_idx, self.valopt['N_frames'],
                                            padding=self.valopt['padding'])
            imgs_GT = self.imgs_GT[folder].index_select(0, torch.LongTensor(select_idx))
            imgs_LR, kernel = self.kernel_gen.apply(imgs_GT)
            imgs_LR = imgs_LR.mul(255).clamp(0, 255).round().div(255)

        # include random noise for each frame
        return {
            'LQs': imgs_LR,
            'kernel' : kernel,
            'GT': imgs_GT,
            'folder': folder,
            'idx': idx
        }

    def __len__(self):
        if self.train:
            return len(self.img_list)
        else:
            return len(self.data_info['path_GT'])
        
