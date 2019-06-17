# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Clean testing scripts for SiamRPN
# New: support GENE and TPE tuning
# ------------------------------------------------------------------------------

import _init_paths
import os
import cv2
import random
import argparse
import numpy as np
import matlab.engine
from os.path import exists, join
import models.models as models
from tracker.siamrpn import SiamRPN
from torch.autograd import Variable
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

eng = matlab.engine.start_matlab()

def parse_args():
    """
    args for rpn testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamRPN Tracking Test')
    parser.add_argument('--arch', dest='arch', default='SiamRPNIncep22', help='backbone architecture')
    parser.add_argument('--resume', required=True, type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2017', help='dataset test')
    parser.add_argument('--anchor_nums', default=5, type=int, help='anchor numbers')
    parser.add_argument('--cls_type', default="thicker", type=str, help='cls/loss type, thicker or thinner or else you defined')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    args = parser.parse_args()

    return args


def track(tracker, net, video, args):
    start_frame, lost_times, toc = 0, 0, 0

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
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    else:
        result_path = join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return 0    # for mult-gputesting

    regions = []  # result and states[1 init / 2 lost / 0 skip]
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        tic = cv2.getTickCount()

        if f == start_frame:  # init
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = tracker.init(im, target_pos, target_sz, net)  # init tracker
            regions.append(1 if 'VOT' in args.dataset else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()

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
                fin.write(','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(video['name'], toc, f / toc, lost_times))

    return lost_times


def main():
    args = parse_args()
    total_lost = 0

    # prepare model
    net = models.__dict__[args.arch](anchors_nums=args.anchor_nums, cls_type=args.cls_type)
    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    # prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # prepare tracker
    info = edict()
    info.arch = args.arch
    info.cls_type = args.cls_type
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    tracker = SiamRPN(info)

    for video in video_keys:
        total_lost += track(tracker, net, dataset[video], args)
    print('Total Lost: {:d}'.format(total_lost))


# ------------------------------------------------------------
# The next few functions are utilized for tuning
# Only VOT is supported
# About 1000 - 3000 group is needed
# ------------------------------------------------------------
def track_tune(tracker, net, video, config):
    arch = config['arch']
    benchmark_name = config['benchmark']
    resume = config['resume']
    hp = config['hp']  # penalty_k, scale_lr, window_influence, adaptive size (for vot2017 or later)

    tracker_path = join('test', (benchmark_name + resume.split('/')[-1].split('.')[0] +
                                 '_small_size_{:.4f}'.format(hp['small_sz']) +
                                 '_big_size_{:.4f}'.format(hp['big_sz']) +
                                 '_penalty_k_{:.4f}'.format(hp['penalty_k']) +
                                 '_w_influence_{:.4f}'.format(hp['window_influence']) +
                                 '_scale_lr_{:.4f}'.format(hp['lr'])).replace('.', '_'))  # no .

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in benchmark_name:
        baseline_path = join(tracker_path, 'baseline')
        video_path = join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = join(video_path, video['name'] + '_001.txt')
    else:
        raise ValueError('Only VOT is supported')

    # occ for parallel running
    if not os.path.exists(result_path):
        fin = open(result_path, 'w')
        fin.close()
    else:
        if benchmark_name.startswith('VOT'):
            return 0
        else:
            raise ValueError('Only VOT is supported')


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
            regions.append([float(1)] if 'VOT' in benchmark_name else gt[f])
        elif f > start_frame:  # tracking
            state = tracker.track(state, im)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in benchmark_name else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append([float(2)])
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append([float(0)])

    # save results for OTB
    if benchmark_name.startswith('VOT'):
        return regions
    else:
        raise ValueError('Only VOT is supported')



def eao_vot_rpn(tracker, net, config):
    dataset = load_dataset(config['benchmark'])
    video_keys = sorted(list(dataset.keys()).copy())
    results = []
    for video in video_keys:
        video_result = track_tune(tracker, net, dataset[video], config)
        results.append(video_result)

    year = config['benchmark'][-4:]  # need a str, instead of a int
    eng.cd('./lib/core')
    eao = eng.get_eao(results, year)

    return eao




if __name__ == '__main__':
    main()

