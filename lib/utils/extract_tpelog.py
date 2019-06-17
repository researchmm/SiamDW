# -*- coding:utf-8 -*-
# ! ./usr/bin/env python
# __author__ = 'zzp'

import shutil
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Analysis siamfc tune results')
parser.add_argument('--path', default='logs/gene_adjust_rpn.log', help='tune result path')
parser.add_argument('--dataset', default='VOT2018', help='test dataset')
parser.add_argument('--save_path', default='logs', help='log file save path')


def collect_results(args):
    if not args.path.endswith('txt'):
        name = args.path.split('.')[0]
        name = name + '.txt'
        shutil.copy(args.path, name)
        args.path = name
    fin = open(args.path, 'r')
    lines = fin.readlines()
    penalty_k = []
    scale_lr = []
    wi = []
    sz = []
    bz = []
    eao = []
    count = 0 # total numbers

    for line in lines:
        if not line.startswith('penalty_k'):
            pass
        else:
     #       print(line)
            count += 1
            temp0, temp1, temp2, temp3, temp4, temp5 = line.split(',')
            penalty_k.append(float(temp0.split(': ')[-1]))
            scale_lr.append(float(temp1.split(': ')[-1]))
            wi.append(float(temp2.split(': ')[-1]))
            sz.append(float(temp3.split(': ')[-1]))
            bz.append(float(temp4.split(': ')[-1]))
            eao.append(float(temp5.split(': ')[-1]))

    # find max
    eao = np.array(eao)
    max_idx = np.argmax(eao)
    max_eao = eao[max_idx]
    print('{} params group  have been tested'.format(count))
    print('penalty_k: {:.4f}, scale_lr: {:.4f}, wi: {:.4f}, small_sz: {}, big_sz: {}, auc: {}'.format(penalty_k[max_idx], scale_lr[max_idx], wi[max_idx], sz[max_idx], bz[max_idx], max_eao))


if __name__ == '__main__':
    args = parser.parse_args()
    collect_results(args)
