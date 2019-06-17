# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: extern modules used by SiamFC and SiamRPN
# Reference: SiamRPN [Li]
# ------------------------------------------------------------------------------

import os
import json
import random
from os.path import join

sample_random = random.Random()
class SingleData(object):
    """
    for training with multi dataset
    """
    def __init__(self, cfg, data_name, start):
        self.data_name = data_name
        self.start = start

        info = cfg.SIAMRPN.DATASET[data_name]
        self.frame_range = info.RANGE
        self.num_use = info.USE
        self.root = info.PATH

        with open(info.ANNOTATION) as fin:
            self.labels = json.load(fin)
            self._clean()
            self.num = len(self.labels)    # video number

        self._shuffle()

    def _clean(self):
        """
        remove empty videos/frames/annos in dataset
        """
        # no frames
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))

        for video, track in to_del:
            del self.labels[video][track]

        # no track/annos
        to_del = []

        if self.data_name == 'YTB':
            to_del.append('train/1/YyE0clBPamU')  # This video has no bounding box.

        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)

        for video in to_del:
            del self.labels[video]

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
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
        template_frame = frames[template_frame]
        search_frame = random.choice(search_range)

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)
