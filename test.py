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

            print('Number of val images in [{:s}]: {:d}, [x={}, y={}, t={}]'.format(dataset_opt['name'], len(val_set),\
                                                                                        opt['sigma_x'], opt['sigma_y'], opt['theta'],  
                ))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    modelcp = create_model(opt)
    modelcp_m = create_model(opt, ada=True)

    center_idx = (opt['datasets']['val']['N_frames']) // 2
    with_GT = False if opt['datasets']['val']['mode'] == 'demo' else True

    pd_log = pd.DataFrame(columns=['PSNR_Bicubic', 'SSIM_Bicubic'])

    # Single GPU
    # PSNR_rlt: psnr_init, psnr_before, psnr_after
    psnr_rlt = [{}, {}]
    # SSIM_rlt: ssim_init, ssim_after
    ssim_rlt = [{}, {}]

    pbar = util.ProgressBar(len(val_set))
    dd = 1
    for val_data in val_loader:
        folder = val_data['folder'][0]
        idx_d = int(val_data['idx'][0].split('/')[0])
        # idx_d = int(val_data['idx'])
        
        # idx_d = int(val_data['idx'][0].split('/')[0])
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
        # bic_lq = meta_test_data['LQs'][0][center_idx].unsqueeze(0)
        # Bic_LQs = F.interpolate(bic_lq, scale_factor=4, mode='bicubic', align_corners=True)
        # max_sigx = 2.0
        # start_x = 0.4
        # h_coff = (max_sigx - start_x) / start_x
        modelcp.load_modulated_network(opt['path']['bicubic_G'], modelcp.netG)
        # st = time.time()
        orgh_ms = []
        orgv_ms = []
        LQs = meta_test_data['LQs']
        B, T, C, H, W = LQs.shape
        thetas = val_data['kernelparam']['theta']
        sigxs =  val_data['kernelparam']['sigma'][0]
        sigys =  val_data['kernelparam']['sigma'][1]
        # for i in range(B):
        #     h_m = 0
        #     v_m = 0.00
        #     orgh_ms.append(h_m)
        #     orgv_ms.append(v_m)
        sigma_x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        sigma_y = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        h_ms = [0,0, 0.0, 0.03, 0.15, 0.23, 0.31, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.75, 0.81, 0.86, 0.90, 0.94, 0.98, 1.0]
        v_ms = [0.0, 0.01, 0.04, 0.09, 0.15, 0.20, 0.27, 0.34, 0.41, 0.49, 0.55, 0.68, 0.69, 0.8, 0.8, 0.86, 0.92, 0.96, 1.0]

        # 0.7 30.2928
        # 0.68 =
        # 0.67 = 
        # 0.66 = 
        # 0.7
        # 0.65 = 31.2213
        # 0.6 = 31.6982
        # 0.55 = 31.8119
        for i in range(B):
            theta = thetas[i]
            sigx = sigxs[i]
            x_index = sigma_x.index(sigx)
            sigy = sigys[i]
            y_index = sigma_y.index(sigy)

            # if abs(theta) > 0 and abs(sigx - sigy) > 0.2:
            #     sigx *= 0.7
            #     sigy *= 0.7
                
            # h_m = sigx * 0.55
            # v_m = sigy * 0.55
            
            # if sigx < 0.6:
            #     h_m = 0.1
            # if sigy < 0.6:
            #     v_m = 0.1
            
            h_m = h_ms[x_index]
            v_m = v_ms[y_index]
            orgh_ms.append(0)
            # v_m = 0.4
            orgv_ms.append(0)
            
        meta_test_data['h_m'] = torch.tensor(orgh_ms, dtype=torch.float).cuda()
        meta_test_data['v_m'] = torch.tensor(orgv_ms, dtype=torch.float).cuda()
        modelcp.feed_data(meta_test_data, need_GT=with_GT)
        modelcp.test()
        # et = time.time()
        
        # print(meta_test_data['LQs'][:, center_idx].size())
        # Bic_LQs = F.interpolate(meta_test_data['LQs'][:, center_idx], scale_factor=opt['scale'], mode='bicubic', align_corners=True)

        if with_GT:
            model_start_visuals = modelcp.get_current_visuals(need_GT=True)
            hr_image = util.tensor2img(model_start_visuals['GT'], mode='rgb')
            start_image = util.tensor2img(model_start_visuals['rlt'][center_idx], mode='rgb')
            # Bic_LQs = util.tensor2img(Bic_LQs[center_idx], mode='rgb')
            
        imageio.imwrite(os.path.join(maml_train_folder, 'start_{}.png'.format(idx_d)), start_image)
        # imageio.imwrite(os.path.join(maml_train_folder, 'bicubic_{}.png'.format(idx_d+1)), Bic_LQs)
        # Bic_LQs = F.interpolate(meta_test_data['LQs'][:, center_idx], scale_factor=opt['scale'], mode='bicubic', align_corners=True)
        # Bic_LQs = util.tensor2img(Bic_LQs, mode='rgb')
        
        # LQs = meta_test_data['LQs'][:, center_idx]
        # LQs = util.tensor2img(LQs, mode='rgb')
        
        # samplingresults = util.tensor2img(val_data['test_LR'][:, center_idx], mode='rgb')
        # imageio.imwrite(os.path.join(maml_train_folder, 'samplingresults_{}.png'.format(idx_d)), samplingresults)
        # imageio.imwrite(os.path.join(maml_train_folder, 'LR_{}.png'.format(idx_d)), LQs)
        # modelcp.netG = deepcopy(model.netG)
        print(start_image.shape)
        print(start_image.shape)
        print(start_image.shape)
        print(start_image.hr_image)
        psnr_rlt[0][folder].append(util.calculate_psnr(start_image, hr_image))
        ssim_rlt[0][folder].append(util.calculate_ssim(start_image, hr_image))

        # modelcp_m.load_modulated_network(opt['path']['bicubic_G'], modelcp_m.netG)

        # thetas = val_data['kernelparam']['theta']
        # sigxs =  val_data['kernelparam']['sigma'][0]
        # sigys =  val_data['kernelparam']['sigma'][1]
        # val_h_ms = []
        # val_v_ms = []
        # LQs = val_data['LQs']
        # B, T, C, H, W = LQs.shape
        
        
        # sigma_x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        # sigma_y = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        # h_ms = [0,0, 0.0, 0.03, 0.15, 0.23, 0.31, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.75, 0.81, 0.86, 0.90, 0.94, 0.98, 1.0]
        # v_ms = [0.0, 0.0, 0.04, 0.16, 0.27, 0.35, 0.43, 0.51, 0.58, 0.64, 0.69, 0.74, 0.79, 0.83, 0.88, 0.92, 0.96, 0.99, 1.0]
        
        # for i in range(B):
        #     theta = thetas[i]
        #     sigx = sigxs[i]
        #     x_index = sigma_x.index(sigx)
        #     sigy = sigys[i]
        #     y_index = sigma_y.index(sigy)

            
            
        #     # if abs(theta) > 0 and abs(sigx - sigy) > 0.2:
        #     #     sigx *= 0.7
        #     #     sigy *= 0.7
                
        #     # h_m = sigx * 0.55
        #     # v_m = sigy * 0.55
            
        #     # if sigx < 0.6:
        #     #     h_m = 0.1
        #     # if sigy < 0.6:
        #     #     v_m = 0.1
            
        #     h_m = h_ms[x_index]
        #     v_m = v_ms[y_index]
            
        # meta_test_data['h_m'] = torch.tensor(val_h_ms, dtype=torch.float).cuda()
        # meta_test_data['v_m'] = torch.tensor(val_v_ms, dtype=torch.float).cuda()     

        st = time.time()
        
        # modelcp_m.feed_data(meta_test_data, need_GT=with_GT)
        # modelcp_m.test()
        et = time.time()
        update_time = et - st
        
        # if with_GT:
        #     modulated_image = modelcp_m.get_current_visuals(need_GT=True)
        #     hr_image = util.tensor2img(modulated_image['GT'], mode='rgb')
        #     modulated_image = modulated_image['rlt'][center_idx]
        #     update_image = util.tensor2img(modulated_image, mode='rgb')
        #     # print(six, siy, the)
        #     # print(sigx[0].item(), sigy[0].item(), theta[0].item(), float(util.calculate_psnr(update_image, hr_image)-util.calculate_psnr(start_image, hr_image)))
        #     # print(h_m, v_m)
        #     psnr_rlt[1][folder].append(util.calculate_psnr(update_image, hr_image))
        #     ssim_rlt[1][folder].append(util.calculate_ssim(update_image, hr_image))
            
        # imageio.imwrite(os.path.join(maml_train_folder, '{:08d}.png'.format(idx_d)), update_image)
        
        # Bic_LQs = util.tensor2img(Bic_LQs, mode='rgb')
        # bic_lq = util.tensor2img(bic_lq, mode='rgb')
        
        # Save and calculate final image
        # imageio.imwrite(os.path.join(maml_train_folder, '{:08d}_kernel.png'.format(idx_d)), kernel)
        # plt.clf()
        # plt.axis('off')
        # ax = plt.subplot(111)
        # im = ax.imshow(kernel, vmin=kernel.min(), vmax=kernel.max())
        # # plt.colorbar(im, ax=ax)
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.axis('off'), plt.xticks([]), plt.yticks([])
        # plt.tight_layout()
        # fig = plt.gcf()
        # fig.set_size_inches(1,1)
        # plt.savefig(os.path.join(maml_train_folder, '{:08d}_kernel.png'.format(idx_d)),  bbox_inches='tight', pad_inches=0)

        # # plt.show()
        # imageio.imwrite(os.path.join(maml_train_folder, '{:08d}.png'.format(idx_d)), update_image)
        # imageio.imwrite(os.path.join(maml_train_folder, 'LR_{:08d}.png'.format(idx_d)), bic_lq)

        if with_GT:
            name_df = '{}/{:08d}'.format(folder, idx_d)
            if name_df in pd_log.index:
                pd_log.at[name_df, 'PSNR_Bicubic'] = psnr_rlt[0][folder][-1]
                # pd_log.at[name_df, 'PSNR_Ours'] = psnr_rlt[1][folder][-1]
                pd_log.at[name_df, 'SSIM_Bicubic'] = ssim_rlt[0][folder][-1]
                # pd_log.at[name_df, 'SSIM_Ours'] = ssim_rlt[1][folder][-1]
            else:
                pd_log.loc[name_df] = [psnr_rlt[0][folder][-1],
                                    # psnr_rlt[1][folder][-1],
                                    ssim_rlt[0][folder][-1]]

            pd_log.to_csv(os.path.join('../test_results', folder_name, 'psnr_update.csv'))

            # pbar.update('Test {} - {}: I: {:.3f}/{:.4f} \tF+: {:.3f}/{:.4f} \tTime: {:.3f}s'
            #                 .format(folder, idx_d,
            #                         psnr_rlt[0][folder][-1], ssim_rlt[0][folder][-1],
            #                         psnr_rlt[1][folder][-1], ssim_rlt[1][folder][-1],
            #                         update_time
            #                         ))
            pbar.update('Test {} - {}: I: {:.3f}/{:.4f} \tTime: {:.3f}s'
                .format(folder, idx_d,
                        psnr_rlt[0][folder][-1], ssim_rlt[0][folder][-1],
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

        # psnr_rlt_avg = {}
        # psnr_total_avg = 0.
        # # Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
        # for k, v in psnr_rlt[1].items():
        #     psnr_rlt_avg[k] = sum(v) / len(v)
        #     psnr_total_avg += psnr_rlt_avg[k]
        # psnr_total_avg /= len(psnr_rlt[1])
        # log_s = '# Validation # PSNR: {:.4f}:'.format(psnr_total_avg)
        # for k, v in psnr_rlt_avg.items():
        #     log_s += ' {}: {:.4f}'.format(k, v)
        # print(log_s)

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

        # ssim_rlt_avg = {}
        # ssim_total_avg = 0.
        # # Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
        # for k, v in ssim_rlt[1].items():
        #     ssim_rlt_avg[k] = sum(v) / len(v)
        #     ssim_total_avg += ssim_rlt_avg[k]
        # ssim_total_avg /= len(ssim_rlt[1])
        # log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
        # for k, v in ssim_rlt_avg.items():
        #     log_s += ' {}: {:.4e}'.format(k, v)
        # print(log_s)

    print('End of evaluation.')

if __name__ == '__main__':
    main()
