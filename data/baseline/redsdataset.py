'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass
from data.meta_learner import preprocessing
from data import random_kernel_generator as rkg
logger = logging.getLogger('base')
import imageio

class redsdataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt, train=True, **kwargs):
        super(redsdataset, self).__init__()
        self.opt = opt
        opt_d = opt['datasets']['train']
        self.opt_d = opt_d
        self.train = train
        
        # temporal augmentation
        self.interval_list = opt_d['interval_list']
        self.random_reverse = opt_d['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt_d['interval_list']), self.random_reverse))

        self.half_N_frames = opt_d['N_frames'] // 2
        self.GT_root, self.LQ_root = opt_d['dataroot_GT'], opt_d['dataroot_LQ']
        self.data_type = opt_d['img_type']
        self.LR_input = False  # low resolution inputs
        #### directly load image keys
        if self.data_type == 'img':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt_d['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt_d['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt_d['cache_keys']))
            self.paths_GT = pickle.load(open(opt_d['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove the REDS4 for testing
        # self.paths_GT = [
        #     v for v in self.paths_GT if v.split('_')[0] not in ['000', '011', '015', '020']
        # ]
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))
        
        sigma_x = float(opt['sigma_x'])
        sigma_y = float(opt['sigma_y'])
        theta = float(opt['theta'])
        gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
        self.kernel_gen = rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.opt['scale'], **gen_kwargs)
        # self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]
        self.kernelparam = {'theta': gen_kwargs['theta'], 'sigma': [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1]]}


    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt_d['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt_d['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        self.scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        
        name = key.split('/')
        name_a = name[-2]
        name_b = name[-1].split('.')[0]
        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        while (center_frame_idx + self.half_N_frames * interval >
                99) or (center_frame_idx - self.half_N_frames * interval < 0):
            center_frame_idx = random.randint(0, 99)
        # get the neighbor list
        neighbor_list = list(
            range(center_frame_idx - self.half_N_frames * interval,
                    center_frame_idx + self.half_N_frames * interval + 1, interval))
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()
        name_b = '{:08d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt_d['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))

        # #### get the GT image (as the center frame)
        # if self.data_type == 'lmdb':
        #     img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))
        # else:
        # img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))

        #         #### get the GT image (as the center frame)
        # if self.data_type == 'mc':
        #     img_GT = self._read_img_mc_BGR(self.GT_root, name_a, name_b)
        #     img_GT = img_GT.astype(np.float32) / 255.
        # elif self.data_type == 'lmdb':
        #     img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))
        # else:
        #     img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))
        
        
        img_GTs = []
        for v in neighbor_list:
            img_GT = imageio.imread(osp.join(self.GT_root, name_a, '{:08d}'.format(v) + '.png'))
            img_GTs.append(img_GT)
            
        img_GTs = np.stack(img_GTs, axis=-1)
        
        # gen_kwargs = preprocessing.set_kernel_params()
        # self.kernel_gen = rkg.Degradation(self.opt['datasets']['train']['kernel_size'], self.opt['scale'], **gen_kwargs)
        # self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]
        img_GTs = preprocessing.np2tensor(img_GTs)

        if self.train:
            img_GTs = preprocessing.crop_fix(img_GTs, scale = self.scale , patch_size=self.opt_d['patch_size'])
            img_GTs = preprocessing.augment(img_GTs)

        img_LQs, _ = self.kernel_gen.apply(img_GTs)
        img_LQs = img_LQs.mul(255).clamp(0, 255).round().div(255)
        # seq_superlr, _ = self.kernel_gen.apply(seq_lr)

        # #### get LQ images
        # LQ_size_tuple = (3, 180, 320) if self.LR_input else (3, 720, 1280)
        # img_LQ_l = []
        # for v in neighbor_list:
        #     img_LQ_path = osp.join(self.LQ_root, name_a, '{:08d}.png'.format(v))
        #     if self.data_type == 'mc':
        #         if self.LR_input:
        #             img_LQ = self._read_img_mc(img_LQ_path)
        #         else:
        #             img_LQ = self._read_img_mc_BGR(self.LQ_root, name_a, '{:08d}'.format(v))
        #         img_LQ = img_LQ.astype(np.float32) / 255.
        #     elif self.data_type == 'lmdb':
        #         img_LQ = util.read_img(self.LQ_env, '{}_{:08d}'.format(name_a, v), LQ_size_tuple)
        #     else:
        #         img_LQ = util.read_img(None, img_LQ_path)
        #     img_LQ_l.append(img_LQ)

        # if self.opt['phase'] == 'train':
        #     C, H, W = LQ_size_tuple  # LQ size
        #     # randomly crop
        #     if self.LR_input:
        #         LQ_size = GT_size // scale
        #         rnd_h = random.randint(0, max(0, H - LQ_size))
        #         rnd_w = random.randint(0, max(0, W - LQ_size))
        #         img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
        #         rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        #         img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
        #     else:
        #         rnd_h = random.randint(0, max(0, H - GT_size))
        #         rnd_w = random.randint(0, max(0, W - GT_size))
        #         img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
        #         img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

        #     # augmentation - flip, rotate
        #     img_LQ_l.append(img_GT)
        #     rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
        #     img_LQ_l = rlt[0:-1]
        #     img_GT = rlt[-1]

        # # stack LQ images to NHWC, N is the frame number
        # img_LQs = np.stack(img_LQ_l, axis=0)
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_GT = img_GT[:, :, [2, 1, 0]]
        # img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        # img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
        #                                                              (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GTs, 'key': key, 'kernelparam': self.kernelparam}

    def __len__(self):
        return len(self.paths_GT)
