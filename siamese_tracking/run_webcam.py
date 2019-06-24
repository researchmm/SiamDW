# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: zhangzhipeng2017@ia.ac.cn
# Detail: Tracking through webcam
# Reference: ATOM code
# ------------------------------------------------------------------------------

import _init_paths
import os
import cv2
import random
import argparse
import numpy as np

import models.models as models

from os.path import exists, join
from torch.autograd import Variable
from tracker.siamfc import SiamFC
from tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou


def parse_args():
    """
    args for fc testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    parser.add_argument('--arch', default='SiamRPNRes22', type=str, help='backbone architecture')
    parser.add_argument('--resume', default='snapshot/CIResNet22RPN.model', type=str, help='pretrained model')
    args = parser.parse_args()

    return args


def track_webcam(tracker, model):
    """Run tracker with webcam."""

    class UIControl:
        def __init__(self):
            self.mode = 'init'  # init, select, track
            self.target_tl = (-1, -1)
            self.target_br = (-1, -1)
            self.mode_switch = False

        def mouse_callback(self, event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self.mode == 'init':
                self.target_tl = (x, y)
                self.target_br = (x, y)
                self.mode = 'select'
                self.mode_switch = True
            elif event == cv2.EVENT_MOUSEMOVE and self.mode == 'select':
                self.target_br = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN and self.mode == 'select':
                self.target_br = (x, y)
                self.mode = 'track'
                self.mode_switch = True

        def get_tl(self):
            return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

        def get_br(self):
            return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

        def get_bb(self):
            tl = self.get_tl()
            br = self.get_br()

            bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
            return bb   # [lx, ly, w, h]

    ui_control = UIControl()
    cap = cv2.VideoCapture(0)
    display_name = 'SiamDW on webcam'
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)
    cv2.setMouseCallback(display_name, ui_control.mouse_callback)


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_disp = frame.copy()

        if ui_control.mode == 'track' and ui_control.mode_switch:
            ui_control.mode_switch = False
            lx, ly, w, h = ui_control.get_bb()
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])
            state = tracker.init(frame_disp, target_pos, target_sz, model)  # init tracker


        # Draw box
        if ui_control.mode == 'select':
            cv2.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
        elif ui_control.mode == 'track':
            state = tracker.track(state, frame_disp)  # track
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(
                location[1] + location[3])

            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Put text
        font_color = (0, 0, 0)
        if ui_control.mode == 'init' or ui_control.mode == 'select':
            cv2.putText(frame_disp, 'Select target', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv2.putText(frame_disp, 'Press q to quit', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
        elif ui_control.mode == 'track':
            cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
        # Display the resulting frame
        cv2.imshow(display_name, frame_disp)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            ui_control.mode = 'init'

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # prepare model (SiamRPN or SiamFC)

    # prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = 'NOUSE'
    info.epoch_test = True
    info.cls_type = 'thinner'

    if 'FC' in args.arch:
        net = models.__dict__[args.arch]()
        tracker = SiamFC(info)
    else:
        net = models.__dict__[args.arch](anchors_nums=5, cls_type='thinner')
        tracker = SiamRPN(info)

    print('[*] ======= Track video with {} ======='.format(args.arch))

    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()


    track_webcam(tracker, net)

if __name__ == '__main__':
    main()