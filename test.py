import os
import math
import argparse
import random
import logging
import imageio
import time
import pandas as pd
from copy import deepcopy

import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data.baseline import loader, create_dataloader, create_dataset
from models import create_model


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
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            pass
        elif phase == 'val':
            if '+' in opt['datasets']['val']['name']:
                raise NotImplementedError('Do not use + signs in test mode')
            else:
                val_set = create_dataset(dataset_opt, scale=opt['scale'],
                                         kernel_size=opt['datasets']['train']['kernel_size'],
                                         model_name=opt['network_E']['which_model_E'])
                # val_set = loader.get_dataset(opt, train=False)
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)

            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    modelcp = create_model(opt)

    center_idx = (opt['datasets']['val']['N_frames']) // 2
    lr_alpha = opt['train']['maml']['lr_alpha']
    update_step = opt['train']['maml']['adapt_iter']
    with_GT = False if opt['datasets']['val']['mode'] == 'demo' else True

    pd_log = pd.DataFrame(columns=['PSNR_Bicubic', 'PSNR_Ours', 'SSIM_Bicubic', 'SSIM_Ours'])

    # Single GPU
    # PSNR_rlt: psnr_init, psnr_before, psnr_after
    psnr_rlt = [{}, {}]
    # SSIM_rlt: ssim_init, ssim_after
    ssim_rlt = [{}, {}]

    pbar = util.ProgressBar(len(val_set))
    for val_data in val_loader:
        folder = val_data['folder'][0]
        idx_d = int(val_data['idx'][0].split('/')[0])
        if 'name' in val_data.keys():
            name = val_data['name'][0][center_idx][0]
        else:
            name = folder

        train_folder = os.path.join('../test_results', folder_name, name)
        maml_train_folder = os.path.join(train_folder, 'DynaVSR')

        if not os.path.exists(train_folder):
            os.makedirs(train_folder, exist_ok=False)
        if not os.path.exists(maml_train_folder):
            os.mkdir(maml_train_folder)

        for i in range(len(psnr_rlt)):
            if psnr_rlt[i].get(folder, None) is None:
                psnr_rlt[i][folder] = []
        for i in range(len(ssim_rlt)):
            if ssim_rlt[i].get(folder, None) is None:
                ssim_rlt[i][folder] = []
        
        meta_train_data = {}
        meta_test_data = {}

        # Make SuperLR seq using estimation model
        meta_train_data['GT'] = val_data['LQs'][:, center_idx]
        meta_test_data['LQs'] = val_data['LQs'][0:1]
        meta_test_data['GT'] = val_data['GT'][0:1, center_idx] if with_GT else None
        # Check whether the batch size of each validation data is 1
        assert val_data['LQs'].size(0) == 1

        if opt['network_G']['which_model_G'] == 'TOF':
            LQs = meta_test_data['LQs']
            B, T, C, H, W = LQs.shape
            LQs = LQs.reshape(B*T, C, H, W)
            Bic_LQs = F.interpolate(LQs, scale_factor=opt['scale'], mode='bicubic', align_corners=True)
            meta_test_data['LQs'] = Bic_LQs.reshape(B, T, C, H*opt['scale'], W*opt['scale'])
        
        ## Before start testing
        # Bicubic Model Results
        
        # max_sigx = 2.0
        # start_x = 0.4
        
        # h_coff = (max_sigx - start_x) / start_x
        h_coff = 1
        modelcp.load_network(opt['path']['VSR_G'], modelcp.netG)
        modelcp.feed_data(meta_test_data, need_GT=with_GT)
        modelcp.test()

        if with_GT:
            model_start_visuals = modelcp.get_current_visuals(need_GT=True)
            hr_image = util.tensor2img(model_start_visuals['GT'], mode='rgb')
            start_image = util.tensor2img(model_start_visuals['rlt'], mode='rgb')
            psnr_rlt[0][folder].append(util.calculate_psnr(start_image, hr_image))
            ssim_rlt[0][folder].append(util.calculate_ssim(start_image, hr_image))

        # modelcp.netG = deepcopy(model.netG)

        # Inner Loop Update
        st = time.time()
            

        et = time.time()
        update_time = et - st

        model_update_visuals = modelcp.get_current_visuals(need_GT=False)
        update_image = util.tensor2img(model_update_visuals['rlt'], mode='rgb')
        # Save and calculate final image
        print(os.path.join(maml_train_folder, '{:08d}.png'.format(idx_d)))
        imageio.imwrite(os.path.join(maml_train_folder, '{:08d}.png'.format(idx_d)), update_image)

        if with_GT:
            psnr_rlt[1][folder].append(util.calculate_psnr(update_image, hr_image))
            ssim_rlt[1][folder].append(util.calculate_ssim(update_image, hr_image))

            name_df = '{}/{:08d}'.format(folder, idx_d)
            if name_df in pd_log.index:
                pd_log.at[name_df, 'PSNR_Bicubic'] = psnr_rlt[0][folder][-1]
                pd_log.at[name_df, 'PSNR_Ours'] = psnr_rlt[1][folder][-1]
                pd_log.at[name_df, 'SSIM_Bicubic'] = ssim_rlt[0][folder][-1]
                pd_log.at[name_df, 'SSIM_Ours'] = ssim_rlt[1][folder][-1]
            else:
                pd_log.loc[name_df] = [psnr_rlt[0][folder][-1],
                                    psnr_rlt[1][folder][-1],
                                    ssim_rlt[0][folder][-1], ssim_rlt[1][folder][-1]]

            pd_log.to_csv(os.path.join('../test_results', folder_name, 'psnr_update.csv'))

            pbar.update('Test {} - {}: I: {:.3f}/{:.4f} \tF+: {:.3f}/{:.4f} \tTime: {:.3f}s'
                            .format(folder, idx_d,
                                    psnr_rlt[0][folder][-1], ssim_rlt[0][folder][-1],
                                    psnr_rlt[1][folder][-1], ssim_rlt[1][folder][-1],
                                    update_time
                                    ))
        else:
            pbar.update()

    if with_GT:
        psnr_rlt_avg = {}
        psnr_total_avg = 0.
        # Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
        for k, v in psnr_rlt[0].items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]
        psnr_total_avg /= len(psnr_rlt[0])
        log_s = '# Validation # Bic PSNR: {:.4f}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        print(log_s)

        psnr_rlt_avg = {}
        psnr_total_avg = 0.
        # Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
        for k, v in psnr_rlt[1].items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]
        psnr_total_avg /= len(psnr_rlt[1])
        log_s = '# Validation # PSNR: {:.4f}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        print(log_s)

        ssim_rlt_avg = {}
        ssim_total_avg = 0.
        # Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
        for k, v in ssim_rlt[0].items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]
        ssim_total_avg /= len(ssim_rlt[0])
        log_s = '# Validation # Bicubic SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        print(log_s)

        ssim_rlt_avg = {}
        ssim_total_avg = 0.
        # Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
        for k, v in ssim_rlt[1].items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]
        ssim_total_avg /= len(ssim_rlt[1])
        log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        print(log_s)

    print('End of evaluation.')

if __name__ == '__main__':
    main()
