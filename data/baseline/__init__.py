"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt, **kwargs):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'LQ':
        from data.baseline.LQ_dataset import LQDataset as D
    elif mode == 'LQGT':
        from data.baseline.LQGT_dataset import LQGTDataset as D
    # datasets for video restoration
    elif mode == 'REDS':
        from data.baseline.redsdataset import redsdataset as D
    elif mode == 'Vimeo90K':
        from data.baseline.Vimeo90K_dataset import Vimeo90KDataset as D
    elif mode == 'video_test':
        from data.baseline.VideoTestDataset import VideoTestDataset as D
    elif mode == 'video_test_etdm':
        from data.meta_learner.video_test_dataset_etdm import VideoTestDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt, **kwargs)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
