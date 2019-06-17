
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
    parser = argparse.ArgumentParser(description='Train SiamFC with onekey')
    # for train
    parser.add_argument('--cfg', type=str, default='experiments/train/SiamFC.yaml', help='yaml configure file name')

    # for

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # train - test - tune information
    info = yaml.load(open(args.cfg, 'r').read())
    info = info['SIAMFC']
    trainINFO = info['TRAIN']
    testINFO = info['TEST']
    tuneINFO = info['TUNE']
    dataINFO = info['DATASET']

    # epoch training -- train 50 or more epochs
    if trainINFO['ISTRUE']:
        print('==> train phase')
        print('python ./siamese_tracking/train_siamfc.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee logs/siamfc_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

        if not exists('logs'):
            os.makedirs('logs')

        os.system('python ./siamese_tracking/train_siamfc.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee logs/siamrpn_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

    # epoch testing -- test 30-50 epochs (or more)
    if testINFO['ISTRUE']:
        print('==> test phase')
        print('mpiexec -n {0} python ./siamese_tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                  --threads {0} --dataset {5}  2>&1 | tee logs/siamfc_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          (len(info['GPUS']) + 1) // 2, testINFO['DATA']))

        if not exists('logs'):
            os.makedirs('logs')

        os.system('mpiexec -n {0} python ./siamese_tracking/test_epochs.py --arch {1} --start_epoch {2} --end_epoch {3} --gpu_nums={4} \
                  --threads {0} --dataset {5}   2>&1 | tee logs/siamfc_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          (len(info['GPUS']) + 1) // 2, testINFO['DATA']))

        # test on vot or otb benchmark
        if 'OTB' in testINFO['DATA']:
            os.system('python ./lib/core/eval_otb.py {0} ./result SiamFC* 0 100 2>&1 | tee logs/siamfc_eval_epochs.log'.format(testINFO['DATA']))
        elif 'VOT' in testINFO['DATA']:
            os.system('python ./lib/core/eval_vot.py {0} ./result 2>&1 | tee logs/siamfc_eval_epochs.log'.format(testINFO['DATA']))
        else:
            raise ValueError('not supported')


    # tuning -- with TPE
    if tuneINFO['ISTRUE']:

        if 'OTB' in testINFO['DATA']:
            pass
        elif 'VOT' in testINFO['DATA'] and 'LT' not in testINFO['DATA']:   # for vot real-time and baseline
            resume = extract_logs('logs/siamfc_eval_epochs.log', 'VOT')
        elif 'LT' in testINFO['DATA']:                                     # for vot longterm
            pass
        else:
            raise ValueError('not supported now')

        print('==> tune phase')
        print('python -u ./siamese_tracking/tune_tpe.py --arch {0} --resume {1} --dataset {2} --gpu_nums {3} \
                  2>&1 | tee logs/tpe_tune_fc.log'.format(trainINFO['MODEL'], 'snapshot/'+ resume, tuneINFO['DATA'], (len(info['GPUS']) + 1) // 2))

        if not exists('logs'):
            os.makedirs('logs')
        os.system('python -u ./siamese_tracking/tune_tpe.py --arch {0} --resume {1} --dataset {2} --gpu_nums {3}\
                  2>&1 | tee logs/tpe_tune_fc.log'.format(trainINFO['MODEL'], 'snapshot/'+ resume, tuneINFO['DATA'], (len(info['GPUS']) + 1) // 2))


if __name__ == '__main__':
    main()