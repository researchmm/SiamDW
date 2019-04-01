# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang and Houwen Peng
# Email: houwen.peng@microsoft.com
# Details: siamfc dataset generator
# ------------------------------------------------------------------------------


from __future__ import division

import cv2
import json
import torch
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils.utils import *
from core.config import config

sample_random = random.Random()
# sample_random.seed(123456)


class SiamFCDataset(Dataset):
    def __init__(self, cfg):
        super(SiamFCDataset, self).__init__()
        # pair information
        self.template_size = cfg.SIAMFC.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.SIAMFC.TRAIN.SEARCH_SIZE
        self.size = (self.search_size - self.template_size) // cfg.SIAMFC.TRAIN.STRIDE + 1   # from cross-correlation

        # aug information
        self.color = cfg.SIAMFC.DATASET.COLOR
        self.flip = cfg.SIAMFC.DATASET.FLIP
        self.rotation = cfg.SIAMFC.DATASET.ROTATION
        self.blur = cfg.SIAMFC.DATASET.BLUR
        self.shift = cfg.SIAMFC.DATASET.SHIFT
        self.scale = cfg.SIAMFC.DATASET.SCALE

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
        )

        # train data information
        if cfg.SIAMFC.TRAIN.WHICH_USE == 'VID':
            self.anno = cfg.SIAMFC.DATASET.VID.ANNOTATION
            self.num_use = cfg.SIAMFC.TRAIN.PAIRS
            self.root = cfg.SIAMFC.DATASET.VID.PATH
        elif cfg.SIAMFC.TRAIN.WHICH_USE == 'GOT10K':
            self.anno = cfg.SIAMFC.DATASET.GOT10K.ANNOTATION
            self.num_use = cfg.SIAMFC.TRAIN.PAIRS
            self.root = cfg.SIAMFC.DATASET.GOT10K.PATH
        else:
            raise ValueError('not supported training dataset')

        self.labels = json.load(open(self.anno, 'r'))
        self.videos = list(self.labels.keys())
        self.num = len(self.videos)   # video number
        self.frame_range = 100
        self.pick = self._shuffle()

    def __len__(self):
        return self.num_use

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        index = self.pick[index]
        template, search = self._get_pairs(index)

        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        template, _, _ = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)

        out_label = self._dynamic_label([self.size, self.size], dag_param.shift)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        return template, search, out_label, np.array(bbox, np.float32)  # self.label 15*15/17*17

    # ------------------------------------
    # function groups for selecting pairs
    # ------------------------------------
    def _shuffle(self):
        """
        shuffel to get random pairs index
        """
        lists = list(range(0, self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_image_anno(self, video, track, frame):
        """
        get image and annotation
        """
        frame = "{:06d}".format(frame)
        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]

        return image_path, image_anno

    def _get_pairs(self, index):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            frames = track_info['frames']
        except:
            frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames)-1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

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

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """
        draw_image = image.copy()
        x1, y1, x2, y2 = map(lambda x:int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0,255,0))
        cv2.circle(draw_image, (int(round(x1 + x2)/2), int(round(y1 + y2) /2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2)/2), int(round(y1 + y2) /2)), (int(round(x1 + x2)/2) - 3, int(round(y1 + y2) /2) -3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.imwrite(name, draw_image)

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

        param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)   # shift
        param.scale = ((1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)   # shift and scale

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)        # other data augmentation
        return image, bbox, param

    # ------------------------------------
    # function for creating training label
    # ------------------------------------
    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        # the real shift is -param['shifts']
        sz_x = sz // 2 + round(-c_shift[0]) // 8  # 8 is strides
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


