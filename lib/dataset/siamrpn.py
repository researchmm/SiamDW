# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: siamrpn dataset generator
# Reference: SiamRPN [Li]
# ------------------------------------------------------------------------------

from __future__ import division

import os
import cv2
import json
import math
import random
import numpy as np
import torchvision.transforms as transforms

from os.path import join
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from scipy.ndimage.filters import gaussian_filter
import sys
sys.path.append('../')
from utils.utils import *
from core.config import config
from .module import  SingleData

sample_random = random.Random()
# sample_random.seed(123456)

eps = 1e-7


class SiamRPNDataset(Dataset):
    def __init__(self, cfg):
        super(SiamRPNDataset, self).__init__()
        # pair information
        self.template_size = cfg.SIAMRPN.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.SIAMRPN.TRAIN.SEARCH_SIZE
        self.score_size = (self.search_size - self.template_size) // cfg.SIAMRPN.TRAIN.STRIDE + 1  # from cross-correlation

        # anchors information
        self.thr_high = cfg.SIAMRPN.TRAIN.ANCHORS_THR_HIGH
        self.thr_low = cfg.SIAMRPN.TRAIN.ANCHORS_THR_LOW
        self.pos_keep = cfg.SIAMRPN.TRAIN.ANCHORS_POS_KEEP   # kept positive anchors to calc loss
        self.all_keep = cfg.SIAMRPN.TRAIN.ANCHORS_ALL_KEEP   # kept anchors to calc loss
        self.stride = cfg.SIAMRPN.TRAIN.STRIDE
        self.anchor_nums = len(cfg.SIAMRPN.TRAIN.ANCHORS_RATIOS) * len(config.SIAMRPN.TRAIN.ANCHORS_SCALES)
        self._naive_anchors(cfg)   # return self.anchors_naive [anchor_num, 4]
        self._pair_anchors(center=self.search_size//2, score_size=self.score_size)

        # aug information
        self.color = cfg.SIAMRPN.DATASET.COLOR
        self.flip = cfg.SIAMRPN.DATASET.FLIP
        self.rotation = cfg.SIAMRPN.DATASET.ROTATION
        self.blur = cfg.SIAMRPN.DATASET.BLUR
        self.shift = cfg.SIAMRPN.DATASET.SHIFT
        self.scale = cfg.SIAMRPN.DATASET.SCALE
    

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
        )

        # train data information
        print('train datas: {}'.format(cfg.SIAMRPN.TRAIN.WHICH_USE))
        self.train_datas = []    # all train dataset
        start = 0
        self.num = 0
        for data_name in cfg.SIAMRPN.TRAIN.WHICH_USE:
            dataset = SingleData(cfg, data_name, start)
            self.train_datas.append(dataset)
            start += dataset.num         # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        # assert abs(self.num - cfg.SIAMRPN.TRAIN.PAIRS) < eps, 'given pairs is not equal to sum of all dataset'

        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        # choose a dataset
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        template, search = dataset._get_pairs(index)

        # read images
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # transform original bbox to cropped image
        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, _, _ = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)

        # get label for regression
        cls, delta, delta_weight = self._anchor_target(bbox,  pos_keep=self.pos_keep, all_keep=self.all_keep, thr_high=self.thr_high, thr_low=self.thr_low)
        sum_weight = self._dynamic_label([self.score_size, self.score_size], dag_param['shift'], 'balanced')
        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        return template, search, cls, delta, delta_weight, sum_weight, np.array(bbox, np.float32)

    # ------------------------------------
    # function groups for selecting pairs
    # ------------------------------------
    def _python2round(self, f):
        """
        use python2 round in python3 verison.
        """
        if round(f + 1) - round(f) != 1:
            return f + abs(f) / f * 0.5
        return round(f)

    def _shuffle(self):
        """
        random shuffel
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0


    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size):
        """
        data augmentation for input pairs
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
        param.scale = ((1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)  # other data augmentation
        return image, bbox, param


    # ------------------------------------
    # function for anchors and labels
    # ------------------------------------
    
    def _pair_anchors(self, center, score_size):
        """
        anchors corresponding to pairs
        :param center: center of search image
        :param score_size: output score size after cross-correlation
        :return: anchors not corresponding to ground truth
        """
        a0x = center - score_size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors_naive + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_nums, 1, 1), [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, score_size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, score_size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        zero = np.zeros((self.anchor_nums, score_size, score_size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.anchorsPairs = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])

    def _naive_anchors(self, cfg):
        """
        anchors corresponding to score map
        """
        self.anchors_naive = np.zeros((self.anchor_nums, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in cfg.SIAMRPN.TRAIN.ANCHORS_RATIOS:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in cfg.SIAMRPN.TRAIN.ANCHORS_SCALES:
                w = ws * s
                h = hs * s
                self.anchors_naive[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1


    def _anchor_target(self, target, pos_keep=16, all_keep=64, thr_high=0.6, thr_low=0.3):
        cls = np.zeros((self.anchor_nums, self.score_size, self.score_size), dtype=np.int64)
        cls[...] = -1  # -1 ignore 0 negative 1 positive
        delta = np.zeros((4, self.anchor_nums, self.score_size, self.score_size), dtype=np.float32)
        delta_weight = np.zeros((self.anchor_nums, self.score_size, self.score_size), dtype=np.float32)

        tcx, tcy, tw, th = corner2center(target)

        anchor_box = self.anchorsPairs[0]
        anchor_center = self.anchorsPairs[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]


        # delta
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / (w + eps) + eps)
        delta[3] = np.log(th / (h + eps) + eps)

        # IoU
        overlap = IoU([x1, y1, x2, y2], target)
        pos = np.where(overlap > thr_high)
        neg = np.where(overlap < thr_low)

        pos, pos_num = self._select(pos, pos_keep)
        neg, neg_num = self._select(neg, all_keep - pos_num)

        cls[pos] = 1
        w_temp = 1. / (pos_num + 1e-6)  # fix bugs here
        delta_weight[pos] = w_temp

        cls[neg] = 0

        return cls, delta, delta_weight

    def _select(self, position, keep_num=16):
        """
        select pos and neg anchors to balance loss
        """
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num


    def _dynamic_label(self, fixedLabelSize, c_shift, labelWeight='balanced', rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        if labelWeight == 'balanced':
            d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)
        else:
            logger.error('TODO or unknown')

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        #　the real shift is -param['shifts']
        sz_x = sz // 2 + round(-c_shift[0]) // 8   # 8 is strides
        sz_y = sz // 2 + round(-c_shift[1]) // 8

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                        np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                        np.ones_like(y),
                        np.where(dist_to_center < rNeg,
                            0.5 * np.ones_like(y),
                            np.zeros_like(y)))

        return label




