# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Clean testing scripts for SiamFC
# New: support GENE tuning
# ------------------------------------------------------------------------------

import _init_paths
import os
import cv2
import random
import argparse
import numpy as np
import matlab.engine
import models.models as models

from os.path import exists, join
from torch.autograd import Variable
from tracker.siamfc import SiamFC
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

# for GENE tuning
from core.eval_otb import eval_auc_tune
eng = matlab.engine.start_matlab()  # for test eao in vot-toolkit
                                    # start-up slowly TODO: speed up


def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', dest='arch', default='SiamFCIncep22', help='backbone architecture')
    parser.add_argument('--resume', required=True, type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2017', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    args = parser.parse_args()

    return args


def track(tracker, net, video, args):
    start_frame, toc = 0, 0

    # save result to evaluate
    if args.epoch_test:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('result', args.dataset, args.arch + suffix)
    else:
        tracker_path = os.path.join('result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)   # align with training

        tic = cv2.getTickCount()

        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        else:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    args = parse_args()

    # prepare model
    net = models.__dict__[args.arch]()
    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    # prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    tracker = SiamFC(info)

    # tracking all videos in benchmark
    for video in video_keys:
        track(tracker, net, dataset[video], args)


# --------------------------------------------------------------------------
# The next few functions are utilized for tuning hyper-parameters
# You should choose a validation dataset (eg. validation videos in VID)
# --------------------------------------------------------------------------
def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # scale_step, scale_penalty, scale_lr, window_influence

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                 '_step_{:.4f}'.format(hp['scale_step']) +
                                 '_penalty_s_{:.4f}'.format(hp['scale_penalty']) +
                                 '_w_influence_{:.4f}'.format(hp['w_influence']) +
                                 '_scale_lr_{:.4f}'.format(hp['scale_lr'])).replace('.', '_'))  # no .

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        return tracker_path

    start_frame, lost_times, toc = 0, 0, 0

    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net, hp=hp)  # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(location)

    with open(result_path, "w") as fin:
        for x in regions:
            p_bbox = x.copy()
            fin.write(
                ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    return tracker_path


def auc_otb(tracker, net, config):
    """
    get AUC for OTB benchmark
    """
    dataset = load_dataset(config['benchmark'])
    video_keys = list(dataset.keys()).copy()
    random.shuffle(video_keys)

    for video in video_keys:
        result_path = track_tune(tracker, net, dataset[video], config)

    auc = eval_auc_tune(result_path, config['benchmark'])

    return auc


def eao_vot(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())
    results = []
    for video in video_keys:
        video_result = track_tune(tracker, net, dataset[video], config)
        results.append(video_result)

    channel = config['benchmark'].split('VOT')[-1]

    eng.cd('./lib/core')
    eao = eng.get_eao(results, channel)

    return eao


if __name__ == '__main__':
    main()

