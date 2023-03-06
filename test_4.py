import os
import math
import argparse
import random
import logging
import imageio
import time
import pandas as pd
from copy import deepcopy
import numpy as np

import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data.baseline import loader, create_dataloader, create_dataset
from models import create_model
import matplotlib.pyplot as plt


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='temp')
    parser.add_argument('--degradation_type', type=str, default=None)
    parser.add_argument('--sigma_x', type=float, default=None)
    parser.add_argument('--sigma_y', type=float, default=None)
    parser.add_argument('--theta', type=float, default=None)
    args = parser.parse_args()
    if args.exp_name == 'temp':
        opt = option.parse(args.opt, is_train=False)
    else:
        opt = option.parse(args.opt, is_train=False, exp_name=args.exp_name)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.degradation_type is not None:
        if args.degradation_type == 'preset':
            opt['datasets']['val']['degradation_mode'] = args.degradation_type
        else:
            opt['datasets']['val']['degradation_type'] = args.degradation_type
    if args.sigma_x is not None:
        opt['datasets']['val']['sigma_x'] = args.sigma_x
    if args.sigma_y is not None:
        opt['datasets']['val']['sigma_y'] = args.sigma_y
    if args.theta is not None:
        opt['datasets']['val']['theta'] = args.theta
    
    if 'degradation_mode' not in opt['datasets']['val'].keys():
        degradation_name = ''
    elif opt['datasets']['val']['degradation_mode'] == 'set':
        degradation_name = '_' + str(opt['datasets']['val']['degradation_type'])\
                  + '_' + str(opt['datasets']['val']['sigma_x']) \
                  + '_' + str(opt['datasets']['val']['sigma_y'])\
                  + '_' + str(opt['datasets']['val']['theta'])
    else:
        degradation_name = '_' + opt['datasets']['val']['degradation_mode']
    folder_name = opt['name'] + '_' + degradation_name

    if args.exp_name != 'temp':
        folder_name = args.exp_name

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch

    #### create model
    modelcp = create_model(opt)
    modelcp.load_modulated_network(opt['path']['bicubic_G'], modelcp.netG)

    center_idx = (opt['datasets']['val']['N_frames']) // 2
    with_GT = False if opt['datasets']['val']['mode'] == 'demo' else True

    pd_log = pd.DataFrame(columns=['PSNR_Bicubic', 'SSIM_Bicubic'])
    # Single GPU
    # PSNR_rlt: psnr_init, psnr_before, psnr_after

    dd = 1
    
    sigma_xs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    sigma_ys = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    thetas = [0, math.pi /4]
    for theta in thetas:
        for x in sigma_xs:
            for y in sigma_ys:
                for phase, dataset_opt in opt['datasets'].items():
                    
                    dataset_opt['sigma_x'] = x            
                    dataset_opt['sigma_y'] = y     
                    dataset_opt['theta'] = theta     
                    if phase == 'train':
                        pass
                    elif phase == 'val':
                        if '+' in opt['datasets']['val']['name']:
                            raise NotImplementedError('Do not use + signs in test mode')
                        else:
                            
                            val_set = create_dataset(dataset_opt, scale=opt['scale'],
                                                    kernel_size=opt['datasets']['train']['kernel_size'],
                                                    model_name=opt['network_E']['which_model_E'])
                            val_loader = create_dataloader(val_set, dataset_opt, opt, None)

                    else:
                        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
            
                for val_data in val_loader:

                    meta_train_data = {}
                    meta_test_data = {}

                    # Make SuperLR seq using estimation model
                    meta_test_data['LQs'] = val_data['LQs'][0:1].unsqueeze(1)
                    meta_test_data['GT'] = val_data['GT'][0:1] if with_GT else None
                            
                h_ms = np.arange(0, 1.05, 0.05)
                v_ms = np.arange(0, 1.05, 0.05)
                h_psnr = 0
                
                last_h_m = 0
                last_v_m = 0
                # print(chs)
                # st = time.time()
                f = open('./sigma.txt', 'a')
                
                for h_m in h_ms:
                    for v_m in v_ms:
                        h_m = round(h_m, 3)
                        v_m = round(v_m, 3)
                        total_psnr = 0
                        
                        orgh_ms = []
                        orgv_ms = []
                        orgh_ms.append(h_m)
                        orgv_ms.append(v_m)
                        
                        meta_test_data['h_m'] = torch.tensor(orgh_ms, dtype=torch.float).cuda()
                        meta_test_data['v_m'] = torch.tensor(orgv_ms, dtype=torch.float).cuda()
                        modelcp.feed_data(meta_test_data, need_GT=with_GT)
                        modelcp.test()

                        if with_GT:
                            model_start_visuals = modelcp.get_current_visuals(need_GT=True)
                            hr_image = util.tensor2img(model_start_visuals['GT'], mode='rgb')
                            start_image = util.tensor2img(model_start_visuals['rlt'], mode='rgb')
                            # Bic_LQs = util.tensor2img(Bic_LQs[center_idx], mode='rgb')
                        total_psnr += util.calculate_psnr(start_image, hr_image)
                            
                            # total_psnr += util.calculate_psnr(start_image, hr_image)
                            # print(total_psnr)
                        if h_psnr < total_psnr:
                            h_psnr = total_psnr
                            last_h_m = h_m
                            last_v_m = v_m
                f.write('sigx={} ,sigy={}, theta={} h_m={}, v_m={}, h_psnr={:2f} \n '.format(x,y, theta, last_h_m, last_v_m, h_psnr))
                f.close()
                            # print('sigx=',x,'sigy=',y, last_h_m, last_v_m)
                # et = time.time()
                # f.write('sigx=',x,'sigy=',y, last_h_m, last_v_m,' ', total_psnr,"\\")
                
                print('sigx=',x,'sigy=',y, 'theta', theta, last_h_m, last_v_m, h_psnr)
            
                # psnr_rlt[0][folder].append(util.calculate_psnr(start_image, hr_image))
                # ssim_rlt[0][folder].append(util.calculate_ssim(start_image, hr_image))

            # if with_GT:
            #     psnr_rlt_avg = {}
            #     psnr_total_avg = 0.
            #     # Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
            #     for k, v in psnr_rlt[0].items():
            #         psnr_rlt_avg[k] = sum(v) / len(v)
            #         psnr_total_avg += psnr_rlt_avg[k]
            #     psnr_total_avg /= len(psnr_rlt[0])
            #     log_s = '# Validation # Bic PSNR: {:.4f}:'.format(psnr_total_avg)
            #     for k, v in psnr_rlt_avg.items():
            #         log_s += ' {}: {:.4f}'.format(k, v)
            #     print(log_s)

            #     ssim_rlt_avg = {}
            #     ssim_total_avg = 0.
            #     # Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
            #     for k, v in ssim_rlt[0].items():
            #         ssim_rlt_avg[k] = sum(v) / len(v)
            #         ssim_total_avg += ssim_rlt_avg[k]
            #     ssim_total_avg /= len(ssim_rlt[0])
            #     log_s = '# Validation # Bicubic SSIM: {:.4e}:'.format(ssim_total_avg)
            #     for k, v in ssim_rlt_avg.items():
            #         log_s += ' {}: {:.4e}'.format(k, v)
            #     print(log_s)

    print('End of evaluation.')

if __name__ == '__main__':
    main()
