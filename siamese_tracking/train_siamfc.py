# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and  Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: SiamFC training script
# ------------------------------------------------------------------------------

import _init_paths
import os
import shutil
import time
import math
import pprint
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

import models.models as models
from utils.utils import create_logger, print_speed, load_pretrain, restore_from, save_model
from dataset.siamfc import SiamFCDataset
from core.config import config, update_config
from core.function import siamfc_train

eps = 1e-5
def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train SiamFC')
    # general
    parser.add_argument('--cfg', required=True, type=str, default='experiments/train/SiamFC.yaml', help='yaml configure file name')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')

    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def get_optimizer(cfg, trainable_params):
    """
    get optimizer
    """

    optimizer = torch.optim.SGD(trainable_params, cfg.SIAMFC.TRAIN.LR,
                    momentum=cfg.SIAMFC.TRAIN.MOMENTUM,
                    weight_decay=cfg.SIAMFC.TRAIN.WEIGHT_DECAY)

    return optimizer


def lr_decay(cfg, optimizer):
    if cfg.SIAMFC.TRAIN.LR_POLICY == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.8685)
    elif cfg.SIAMFC.TRAIN.LR_POLICY == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif cfg.SIAMFC.TRAIN.LR_POLICY == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    elif cfg.SIAMFC.TRAIN.LR_POLICY == 'log':
        scheduler = np.logspace(math.log10(cfg.SIAMFC.TRAIN.LR), math.log10(cfg.SIAMFC.TRAIN.LR_END), cfg.SIAMFC.TRAIN.END_EPOCH)
    else:
        raise ValueError('unsupported learing rate scheduler')

    return scheduler


def pretrain_zoo():
    GDriveIDs = dict()
    GDriveIDs['SiamFCRes22'] = "1kgYJdydU7Wm6oj9-tGA5EFc6Io2V7rPT"
    GDriveIDs['SiamFCIncep22'] = "1FxbQOSsG51Wau6-MUzsteoald3Y14xJ4"
    GDriveIDs['SiamFCNext22'] = "1sURid92u4hEHR4Ev0wrQPAw8GZtLmB5n"
    return GDriveIDs


def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'SIAMFC', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # auto-download train model from GoogleDrive
    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')
    try:
        DRIVEID = pretrain_zoo()

        if not os.path.exists('./pretrain/{}'.format(config.SIAMFC.TRAIN.PRETRAIN)):
            os.system(
                'wget --no-check-certificate \'https://drive.google.com/uc?export=download&id={0}\' -O ./pretrain/{1}'
                .format(DRIVEID[config.SIAMFC.TRAIN.MODEL], config.SIAMFC.TRAIN.PRETRAIN))
    except:
        print('auto-download pretrained model fail, please download it and put it in pretrain directory')


    # [*] gpus parallel and model prepare
    # prepare
    model = models.__dict__[config.SIAMFC.TRAIN.MODEL]()  # build model
    model = load_pretrain(model, './pretrain/{}'.format(config.SIAMFC.TRAIN.PRETRAIN))  # load pretrain
    trainable_params = check_trainable(model, logger)           # print trainable params info
    optimizer = get_optimizer(config, trainable_params)         # optimizer
    lr_scheduler = lr_decay(config, optimizer)      # learning rate decay scheduler

    if config.SIAMFC.TRAIN.RESUME and config.SIAMFC.TRAIN.START_EPOCH != 0:   # resume
        model.features.unfix((config.SIAMFC.TRAIN.START_EPOCH - 1) / config.SIAMFC.TRAIN.END_EPOCH)
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, config.SIAMFC.TRAIN.RESUME)

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    logger.info('model prepare done')

    # [*] train

    for epoch in range(config.SIAMFC.TRAIN.START_EPOCH, config.SIAMFC.TRAIN.END_EPOCH):
        # build dataloader, benefit to tracking
        train_set = SiamFCDataset(config)
        train_loader = DataLoader(train_set, batch_size=config.SIAMFC.TRAIN.BATCH * gpu_num, num_workers=config.WORKERS,
                                  pin_memory=True, sampler=None)

        if config.SIAMFC.TRAIN.LR_POLICY == 'log':
            curLR = lr_scheduler[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = curLR
        else:
            lr_scheduler.step()

        model, writer_dict = siamfc_train(train_loader, model, optimizer, epoch + 1, curLR, config, writer_dict, logger)

        # save model
        save_model(model, epoch, optimizer, config.SIAMFC.TRAIN.MODEL, config, isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()




