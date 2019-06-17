# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: SiamRPN training script
# Reference: SiamRPN [Li]
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
from easydict import EasyDict as edict

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

import models.models as models
from utils.utils import create_logger, print_speed, load_pretrain, restore_from, save_model
from dataset.siamrpn import SiamRPNDataset
from core.config import config, update_config
from core.function import siamrpn_train


eps = 1e-5
def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train SiamRPN')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/train/SiamRPN.yaml', help='yaml configure file name')

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

    optimizer = torch.optim.SGD(trainable_params, cfg.SIAMRPN.TRAIN.LR,
                    momentum=cfg.SIAMRPN.TRAIN.MOMENTUM,
                    weight_decay=cfg.SIAMRPN.TRAIN.WEIGHT_DECAY)

    return optimizer


def pretrain_zoo():
    GDriveIDs = dict()
    GDriveIDs['SiamRPNRes22'] = "1kgYJdydU7Wm6oj9-tGA5EFc6Io2V7rPT"
    return GDriveIDs


def lr_decay(cfg, optimizer):
    if cfg.SIAMRPN.TRAIN.LR_POLICY == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=0.8685)
    elif cfg.SIAMRPN.TRAIN.LR_POLICY == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif cfg.SIAMRPN.TRAIN.LR_POLICY == 'Reduce':
        scheduler = ReduceLROnPlateau(optimizer, patience=5)
    elif cfg.SIAMRPN.TRAIN.LR_POLICY == 'log':
        scheduler = np.logspace(math.log10(cfg.SIAMRPN.TRAIN.LR), math.log10(cfg.SIAMRPN.TRAIN.LR_END), cfg.SIAMRPN.TRAIN.END_EPOCH)
    else:
        raise ValueError('unsupported learing rate scheduler')

    return scheduler


def get_lr(optimizer):
    """
    get current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'SIAMRPN', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # [*] gpus parallel and model prepare
    # prepare pretrained model -- download from google drive
    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')

    # auto-download train model from GoogleDrive
    try:
        DRIVEID = pretrain_zoo()

        if not os.path.exists('./pretrain/{}'.format(config.SIAMRPN.TRAIN.PRETRAIN)):
            os.system(
                'wget --no-check-certificate \'https://drive.google.com/uc?export=download&id={0}\' -O ./pretrain/{1}'
                    .format(DRIVEID[config.SIAMRPN.TRAIN.MODEL], config.SIAMRPN.TRAIN.PRETRAIN))
    except:
        print('auto-download pretrained model fail, please download it and put it in pretrain directory')


    # define model
    anchor_nums = len(config.SIAMRPN.TRAIN.ANCHORS_RATIOS) * len(config.SIAMRPN.TRAIN.ANCHORS_SCALES)
    model = models.__dict__[config.SIAMRPN.TRAIN.MODEL](anchors_nums=anchor_nums, cls_type=config.SIAMRPN.TRAIN.CLS_TYPE)  # build model
    print(model)
    model = load_pretrain(model, './pretrain/{0}'.format(config.SIAMRPN.TRAIN.PRETRAIN))    # load pretrain
    trainable_params = check_trainable(model, logger)    # print trainable params info
    optimizer = get_optimizer(config, trainable_params)  # optimizer
    lr_scheduler = lr_decay(config, optimizer)  # learning rate decay scheduler

    if config.SIAMRPN.TRAIN.RESUME and config.SIAMRPN.TRAIN.START_EPOCH != 0:  # resume
        model.features.unfix((config.SIAMRPN.TRAIN.START_EPOCH - 1) / config.SIAMRPN.TRAIN.END_EPOCH)
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, config.SIAMRPN.TRAIN.RESUME)

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    logger.info('model prepare done')

    # [*] train

    for epoch in range(config.SIAMRPN.TRAIN.START_EPOCH, config.SIAMRPN.TRAIN.END_EPOCH):
        # build dataloader, benefit to tracking
        train_set = SiamRPNDataset(config)
        train_loader = DataLoader(train_set, batch_size=config.SIAMRPN.TRAIN.BATCH * gpu_num, num_workers=config.WORKERS,
                                  pin_memory=True, sampler=None)

        if config.SIAMRPN.TRAIN.LR_POLICY == 'log':
            curLR = lr_scheduler[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = curLR
        else:
            lr_scheduler.step()
            curLR = get_lr(optimizer)

        model, writer_dict = siamrpn_train(train_loader, model, optimizer, epoch + 1, curLR, config, writer_dict,
                                           logger, cls_type = config.SIAMRPN.TRAIN.CLS_TYPE)

        # save model
        save_model(model, epoch, optimizer, config.SIAMRPN.TRAIN.MODEL, config, isbest=False)


    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
