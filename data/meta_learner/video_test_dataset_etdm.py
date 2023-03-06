import os.path as osp
import torch
import torch.utils.data as data
from data import util
import numpy as np
import math
from data import random_kernel_generator as rkg
from data.meta_learner import preprocessing
import cv2
import pickle

class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt, **kwargs):
        super(VideoTestDataset, self).__init__()
        self.scale = kwargs['scale']
        self.kernel_size = kwargs['kernel_size']
        self.model_name = kwargs['model_name']
        idx = kwargs['idx'] if 'idx' in kwargs else None
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        if idx is None:
            self.name = opt['name']
            self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        else:
            self.name = opt['name'].split('+')[idx]
            self.GT_root, self.LQ_root = opt['dataroot_GT'].split('+')[idx], opt['dataroot_LQ'].split('+')[idx]

        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        if self.name.lower() in ['vid4', 'reds', 'mm522']:
            if self.name.lower() == 'vid4':
                img_type = 'img'
                subfolders_GT = util.glob_file_list(self.GT_root)
            elif self.name.lower() == 'reds':
                img_type = 'img'
                list_hr_seq = util.glob_file_list(self.GT_root)
                subfolders_GT = [k for k in list_hr_seq if
                                   k.find('000') >= 0 or k.find('011') >= 0 or k.find('015') >= 0 or k.find('020') >= 0]
            else:
                img_type = 'img'
                subfolders_GT = util.glob_file_list(self.GT_root)

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

                if self.cache_data:
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT, img_type)
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

                # Generate kernel

        if opt['degradation_mode'] == 'set':
            sigma_x = float(opt['sigma_x'])
            sigma_y = float(opt['sigma_y'])
            theta = float(opt['theta'])
            gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
            self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale, **gen_kwargs)
            self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]

        elif opt['degradation_mode'] == 'preset':
            self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale)
            if self.name.lower() == 'vid4':
                self.kernel_dict = np.load('F:/DynaVSR-master/pretrained_models/Vid4Gauss.npy')
            elif self.name.lower() == 'reds':
                self.kernel_dict = np.load('F:/DynaVSR-master/pretrained_models/REDSGauss.npy')
            else:
                raise NotImplementedError()

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                           padding=self.opt['padding'])
        imgs_GT_org = self.imgs_GT[folder].index_select(0, torch.LongTensor(select_idx))
        imgs_GT = imgs_GT_org.permute(0,2,3,1)
        
        target = np.lib.pad(imgs_GT, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        # imgs_GT = preprocessing.np2tensor(imgs_GT)
        # target = imgs_GT.permute(2,0,3,1)
        
        t = target.shape[0]
        h = target.shape[1]
        w = target.shape[2]
        c = target.shape[3]
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        target = preprocessing.np2tensor(target) # Tensor, [CT',H',W']
            
        target = target.view(c,t,h,w)
        LR, extra_dataset = make_etdm_dataset(target)
        SLR, S_extra_dataset = make_etdm_dataset(LR, replication=False)
        LR = LR.permute(1,0,2,3)
        SLR = SLR.permute(1,0,2,3)
        
        return {
            'SuperLQs': SLR,
            'extra_data': extra_dataset,
            'S_extra_dataset': S_extra_dataset,
            'LQs': LR,
            'GT': imgs_GT_org,
            'ref': imgs_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])


def make_etdm_dataset(target, scale=4, replication=True):
    LR = DUF_downsample(target, scale) # [c,t,h,w]
    if replication:
        LR = torch.cat((LR[:,1:2,:,:], LR, LR[:,-2:-1,:,:]), dim=1)
    GT = torch.cat((target[:,1:2,:,:], target, target[:,-2:-1,:,:]), dim=1)

    C,T,H,W = LR.shape
    ref = LR.permute(1,2,3,0).numpy() #T,H,W,3
    LR_Y = [cv2.cvtColor(ref[i,:,:,:], cv2.COLOR_RGB2YCR_CB)[:,:,:1] for i in range(T)]
    tensor_left_HV = []
    tensor_left_LV = []
    tensor_right_HV = []
    tensor_right_LV = []
    for i in range(T-2):
        left = LR_Y[i]
        middel = LR_Y[i+1]
        right = LR_Y[i+2]
        left_diff = abs(middel - left)
        eps_left = left_diff.mean()
        mask_left_HV = (left_diff > eps_left) * 1.0

        mask_left_LV = 1 - mask_left_HV
        tensor_left_HV.append(mask_left_HV)
        tensor_left_LV.append(mask_left_LV)

        right_diff = abs(middel - right)
        eps_right = right_diff.mean()
        mask_right_HV = (right_diff > eps_right) * 1.0

        mask_right_LV = 1 - mask_right_HV
        tensor_right_HV.append(mask_right_HV)
        tensor_right_LV.append(mask_right_LV)

    tensor_left_HV = np.asarray(tensor_left_HV).astype(np.float32)
    tensor_left_LV = np.asarray(tensor_left_LV).astype(np.float32)
    tensor_right_HV = np.asarray(tensor_right_HV).astype(np.float32)
    tensor_right_LV = np.asarray(tensor_right_LV).astype(np.float32)

    T,H,W,C = tensor_right_HV.shape

    tensor_left_HV = tensor_left_HV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]
    tensor_left_LV = tensor_left_LV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]
    tensor_right_HV = tensor_right_HV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]
    tensor_right_LV = tensor_right_LV.transpose(1,2,3,0).reshape(H, W, -1) # numpy, [H',W',CT]

    tensor_left_HV = preprocessing.np2tensor(tensor_left_HV)
    tensor_left_LV = preprocessing.np2tensor(tensor_left_LV)
    tensor_right_HV = preprocessing.np2tensor(tensor_right_HV)
    tensor_right_LV = preprocessing.np2tensor(tensor_right_LV)
    tensor_left_HV = tensor_left_HV.view(C,T,H,W) # Tensor, [C,T,H,W]
    tensor_left_LV = tensor_left_LV.view(C,T,H,W) # Tensor, [C,T,H,W]
    tensor_right_HV = tensor_right_HV.view(C,T,H,W) # Tensor, [C,T,H,W]
    tensor_right_LV = tensor_right_LV.view(C,T,H,W) # Tensor, [C,T,H,W]
    
    return LR, [tensor_left_HV, tensor_left_LV, tensor_right_HV, tensor_right_LV]


# Gaussian kernel for downsampling
import scipy.ndimage.filters as fi
import numpy as np
import torch.nn.functional as F
import torch 

def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code
    Args:
        x (Tensor, [C, T, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = gkern(13, 0.8)  # 13 and 0.8 for x2
    elif scale == 3:
        h = gkern(13, 1.2)  # 13 and 1.2 for x3
    elif scale == 4:
        h = gkern(13, 1.6)  # 13 and 1.6 for x4
    else:
        print('Invalid upscaling factor: {} (Must be one of 2, 3, 4)'.format(R))
        exit(1)

    C, T, H, W = x.size()
    x = x.contiguous().view(-1, 1, H, W) # depth convolution (channel-wise convolution)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0

    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)

    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')
    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    # please keep the operation same as training.
    # if  downsample to 32 on training time, use the below code.
    x = x[:, :, 2:-2, 2:-2]
    # if downsample to 28 on training time, use the below code.
    #x = x[:,:,scale:-scale,scale:-scale]
    x = x.view(C, T, x.size(2), x.size(3))
    return x


