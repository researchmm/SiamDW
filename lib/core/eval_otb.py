import sys
import json
import os
import glob
from os.path import join, realpath, dirname
import numpy as np

OTB2013 = ['carDark', 'car4', 'david', 'david2', 'sylvester', 'trellis', 'fish', 'mhyang', 'soccer', 'matrix',
           'ironman', 'deer', 'skating1', 'shaking', 'singer1', 'singer2', 'coke', 'bolt', 'boy', 'dudek',
           'crossing', 'couple', 'football1', 'jogging_1', 'jogging_2', 'doll', 'girl', 'walking2', 'walking',
           'fleetface', 'freeman1', 'freeman3', 'freeman4', 'david3', 'jumping', 'carScale', 'skiing', 'dog1',
           'suv', 'motorRolling', 'mountainBike', 'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2',
           'basketball', 'football', 'subway', 'tiger1', 'tiger2']

OTB2015 = ['carDark', 'car4', 'david', 'david2', 'sylvester', 'trellis', 'fish', 'mhyang', 'soccer', 'matrix',
           'ironman', 'deer', 'skating1', 'shaking', 'singer1', 'singer2', 'coke', 'bolt', 'boy', 'dudek',
           'crossing', 'couple', 'football1', 'jogging_1', 'jogging_2', 'doll', 'girl', 'walking2', 'walking',
           'fleetface', 'freeman1', 'freeman3', 'freeman4', 'david3', 'jumping', 'carScale', 'skiing', 'dog1',
           'suv', 'motorRolling', 'mountainBike', 'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2',
           'basketball', 'football', 'subway', 'tiger1', 'tiger2', 'clifBar', 'biker', 'bird1', 'blurBody',
           'blurCar2', 'blurFace', 'blurOwl', 'box', 'car1', 'crowds', 'diving', 'dragonBaby', 'human3', 'human4_2',
           'human6', 'human9', 'jump', 'panda', 'redTeam', 'skating2_1', 'skating2_2', 'surfer', 'bird2',
           'blurCar1', 'blurCar3', 'blurCar4', 'board', 'bolt2', 'car2', 'car24', 'coupon', 'dancer', 'dancer2',
           'dog', 'girl2', 'gym', 'human2', 'human5', 'human7', 'human8', 'kiteSurf', 'man', 'rubik', 'skater',
           'skater2', 'toy', 'trans', 'twinnings', 'vase']


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success


def get_result_bb(arch, seq):
    result_path = join(arch, seq + '.txt')
    temp = np.loadtxt(result_path, delimiter=',').astype(np.float)
    return np.array(temp)


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def eval_auc(dataset='OTB2015', result_path='./test/', tracker_reg='S*', start=0, end=1e6):
    list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3

    trackers = glob.glob(join(result_path, dataset, tracker_reg))
    trackers = trackers[start:min(end, len(trackers))]

    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    # thresholds_error = np.arange(0, 51, 1)

    success_overlap = np.zeros((n_seq, len(trackers), len(thresholds_overlap)))
    # success_error = np.zeros((n_seq, len(trackers), len(thresholds_error)))
    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        for j in range(len(trackers)):
            tracker = trackers[j]
            print('{:d} processing:{} tracker: {}'.format(i, seq, tracker))
            bb = get_result_bb(tracker, seq)
            center = convert_bb_to_center(bb)
            success_overlap[i][j] = compute_success_overlap(gt_rect, bb)
            # success_error[i][j] = compute_success_error(gt_center, center)

    print('Success Overlap')

    if 'OTB2015' == dataset:
        OTB2013_id = []
        for i in range(n_seq):
            if seqs[i] in OTB2013:
                OTB2013_id.append(i)
        max_auc_OTB2013 = 0.
        max_name_OTB2013 = ''
        for i in range(len(trackers)):
            auc = success_overlap[OTB2013_id, i, :].mean()
            if auc > max_auc_OTB2013:
                max_auc_OTB2013 = auc
                max_name_OTB2013 = trackers[i]
            # print('%s(%.4f)' % (trackers[i], auc))

        max_auc = 0.
        max_name = ''
        for i in range(len(trackers)):
            auc = success_overlap[:, i, :].mean()
            if auc > max_auc:
                max_auc = auc
                max_name = trackers[i]
            print('%s(%.4f)' % (trackers[i], auc))
        print('\nOTB2015 Best: %s(%.4f)' % (max_name, max_auc))
    else:
        max_auc = 0.
        max_name = ''
        for i in range(len(trackers)):
            auc = success_overlap[:, i, :].mean()
            if auc > max_auc:
                max_auc = auc
                max_name = trackers[i]
            print('%s(%.4f)' % (trackers[i], auc))

        print('\n%s Best: %s(%.4f)' % (dataset, max_name, max_auc))


# ------------------------------------------------------------------------------------------------------------------
# This function is used for eval performance on validation dataset
# This example provides auc on a dataset, you can modify it to your needs according to your validation dataset (eg, iou)
# If you want to use auc(this demo) to eval performance, please generate a json file for your validation dataset as OTB format
# ------------------------------------------------------------------------------------------------------------------
def eval_auc_tune(result_path, dataset='OTB2015'):
    list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3
    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((n_seq, 1, len(thresholds_overlap)))

    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        bb = get_result_bb(result_path, seq)
        center = convert_bb_to_center(bb)
        success_overlap[i][0] = compute_success_overlap(gt_rect, bb)

    auc = success_overlap[:, 0, :].mean()
    return auc



if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('python ./lib/core/eval_otb.py OTB2013 ./result SiamFC* 0 1')
        exit()
    dataset = sys.argv[1]
    result_path = sys.argv[2]
    tracker_reg = sys.argv[3]
    start = int(sys.argv[4])
    end = int(sys.argv[5])
    eval_auc(dataset, result_path, tracker_reg, start, end)
