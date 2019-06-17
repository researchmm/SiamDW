# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng
# Email:  houwen.peng@microsoft.com
# Details: This script provides epoch-train for siamfc
# ------------------------------------------------------------------------------

import math
import time
import torch

from torch.autograd import Variable
from utils.utils import print_speed


def siamfc_train(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger):
    # unfix for FREEZE-OUT method
    model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.cuda()

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = Variable(input[0]).cuda()
        search = Variable(input[1]).cuda()
        label_cls = Variable(label_cls).cuda()

        loss = model(template, search, label_cls)
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.data[0]):
            optimizer.step()

        # record loss
        loss = loss.data[0]
        losses.update(loss, template.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t Loss:{loss.avg:.5f}'.format(
                epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time, loss=losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg, cfg.SIAMFC.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

def siamrpn_train(train_loader, model,  optimizer, epoch, cur_lr, cfg, writer_dict, logger, cls_type='thicker'):
    model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.cuda()

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        template = Variable(input[0]).cuda()
        search = Variable(input[1]).cuda()

        if cls_type == 'thinner':
            label_cls = Variable(input[2].type(torch.FloatTensor)).cuda()
        else:
            label_cls = Variable(input[2]).cuda()
        label_reg = Variable(input[3]).cuda()
        label_reg_weight = Variable(input[4]).cuda()
        sum_weight = Variable(input[5].type(torch.FloatTensor)).cuda()

        cls_loss, loc_loss = model(template, search, label_cls, label_reg, label_reg_weight, sum_weight)
        cls_loss = torch.mean(cls_loss)
        loc_loss = torch.mean(loc_loss)
        loss = cfg.SIAMRPN.TRAIN.CLS_WEIGHT * cls_loss + cfg.SIAMRPN.TRAIN.REG_WEIGHT * loc_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.data[0]):
            optimizer.step()

        # record loss
        cls_loss = cls_loss.data[0]
        loc_loss = loc_loss.data[0]
        loss = loss.data[0]
        losses.update(loss, template.size(0))
        cls_losses.update(cls_loss, template.size(0))
        reg_losses.update(loc_loss, template.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr : {lr:.7f} \t Batch Time: {batch_time.avg:.3f} \t Data Time:{data_time.avg:.3f} '
                        '\tCLS_Loss:{cls_loss.avg:.5f}\tREG_Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time,
                data_time=data_time, cls_loss=cls_losses, reg_loss=reg_losses, loss=losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg, cfg.SIAMRPN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict


def unfix_more(model, optimizer, epoch, cfg, cur_lr, logger):
    if model.module.features.unfix(epoch / cfg.SIAMFC.TRAIN.END_EPOCH):
        logger.info('unfix part model.')
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, cur_lr,
                                    momentum=cfg.SIAMFC.TRAIN.MOMENTUM,
                                    weight_decay=cfg.SIAMFC.TRAIN.WEIGHT_DECAY)

        logger.info('trainable params:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    else:
        logger.info('no unfix in this epoch.')

    return model, optimizer


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
