# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: SiamRPN onekey script
# ------------------------------------------------------------------------------

import _init_paths
import os
import yaml
import argparse
from os.path import exists
from utils.utils import load_yaml, extract_logs

def parse_args():
    """
    args for onekey.
    """
    parser = argparse.ArgumentParser(description='Train SiamRPN with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='experiments/train/SiamRPN.yaml', help='yaml configure file name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # train - test - tune information
    info = yaml.load(open(args.cfg, 'r').read())
    info = info['SIAMRPN']
    trainINFO = info['TRAIN']
    testINFO = info['TEST']
    tuneINFO = info['TUNE']

    # epoch training -- train 50 or more epochs
    if trainINFO['ISTRUE']:
        print('==> train phase')
        print('python ./siamese_tracking/train_siamrpn.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee logs/siamrpn_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

        if not exists('logs'):
            os.makedirs('logs')

        os.system('python ./siamese_tracking/train_siamrpn.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee logs/siamrpn_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

    # epoch testing -- test 30-50 epochs (or more)
    if testINFO['ISTRUE']:
        print('==> test phase')
        print('mpiexec -n {0} python ./siamese_tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                  --threads {0} --dataset {5} --anchor_nums {6} 2>&1 | tee logs/siamrpn_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          (len(info['GPUS']) + 1) // 2, testINFO['DATA'], len(trainINFO['ANCHORS_RATIOS']) * len(trainINFO['ANCHORS_SCALES'])))

        if not exists('logs'):
            os.makedirs('logs')

        os.system('mpiexec -n {0} python ./siamese_tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                  --threads {0} --dataset {5} --anchor_nums {6} 2>&1 | tee logs/siamrpn_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          (len(info['GPUS']) + 1) // 2, testINFO['DATA'], len(trainINFO['ANCHORS_RATIOS']) * len(trainINFO['ANCHORS_SCALES'])))
        if 'VOT' in testINFO['DATA']:
            os.system('python ./lib/core/eval_vot.py {0} ./result 2>&1 | tee logs/siamrpn_eval_epochs.log'.format(testINFO['DATA']))
        else:
            raise ValueError('not supported')

    # tuning -- with TPE
    if tuneINFO['ISTRUE']:

        if 'VOT' in testINFO['DATA']:   # for vot real-time and baseline
            resume = extract_logs('logs/siamrpn_eval_epochs.log', 'VOT')
        else:
            raise ValueError('not supported now')

        print('==> tune phase')
        print('python -u ./siamese_tracking/tune_tpe.py --arch {0} --resume {1} --dataset {2} --gpu_nums {3}  --anchor_nums {4} \
                  2>&1 | tee logs/tpe_tune_rpn.log'.format(trainINFO['MODEL'], 'snapshot/'+ resume, tuneINFO['DATA'], (len(info['GPUS']) + 1) // 2, len(trainINFO['ANCHORS_RATIOS']) * len(trainINFO['ANCHORS_SCALES'])))

        if not exists('logs'):
            os.makedirs('logs')
        os.system('python -u ./siamese_tracking/tune_tpe.py --arch {0} --resume {1} --dataset {2} --gpu_nums {3}  --anchor_nums {4} \
                  2>&1 | tee logs/tpe_tune_rpn.log'.format(trainINFO['MODEL'], 'snapshot/'+ resume, tuneINFO['DATA'], (len(info['GPUS']) + 1) // 2, len(trainINFO['ANCHORS_RATIOS']) * len(trainINFO['ANCHORS_SCALES'])))


if __name__ == '__main__':
    main()