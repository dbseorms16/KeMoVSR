import os.path as osp
import torch
import torch.utils.data as data
from data import util
import numpy as np
import math
from data import random_kernel_generator as rkg
from data.meta_learner import preprocessing


class video_test_dataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt, **kwargs):
        super(video_test_dataset, self).__init__()
        self.scale = opt['scale']
        self.kernel_size = opt['kernel_size']
        self.model_name = opt['model_name']
        idx = opt['idx'] if 'idx' in kwargs else None
        self.opt = opt
        self.cache_data = True
        self.half_N_frames = opt['datasets']['train']['N_frames'] // 2
        self.name = opt['name']
        self.dataopt = opt['datasets']['train']
        self.GT_root, self.LQ_root = self.dataopt['dataroot_GT'], self.dataopt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        self.train = kwargs['train']
        
        img_type = 'img'
        
        subfolders_GT = util.glob_file_list(self.GT_root)
        self.subfolders_GT = subfolders_GT
        for subfolder_GT in subfolders_GT:
            subfolder_name = osp.basename(subfolder_GT)
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

            # if self.cache_data:
            #     self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT, img_type)

                # Generate kernel

        sigma_x = float(opt['sigma_x'])
        sigma_y = float(opt['sigma_y'])
        theta = float(opt['theta'])
        gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
        self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale, **gen_kwargs)
        self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        select_idx = util.index_generation(idx, max_idx, self.opt['datasets']['train']['N_frames'],
                                           padding='new_info')

        imgs_GT = util.read_img_seq(self.subfolders_GT[int(folder)], 'img')
        
        imgs_GT = imgs_GT.index_select(0, torch.LongTensor(select_idx))
        '''
        for i in range(imgs_GT.shape[0]):
            imgs_LR_slice = self.kernel_gen.apply(imgs_GT[i])
            imgs_LR.append(imgs_LR_slice)

        imgs_LR = torch.stack(imgs_LR, dim=0)
        '''
        imgs_LR, kernel = self.kernel_gen.apply(imgs_GT)
        imgs_LR = imgs_LR.mul(255).clamp(0, 255).round().div(255)
        
        if self.train:
            imgs_GT, imgs_LR = preprocessing.crop(imgs_GT, imgs_LR, patch_size=self.opt['datasets']['train']['patch_size'])
            imgs_GT, imgs_LR = preprocessing.augment(imgs_GT, imgs_LR )

        
        return {
            'LQs': imgs_LR,
            'kernel' : kernel,
            'GT': imgs_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
