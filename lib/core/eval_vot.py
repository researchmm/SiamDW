# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Details: This script provides a VOT test toolkit in python
# ------------------------------------------------------------------------------
import sys
import glob
import matlab.engine
eng = matlab.engine.start_matlab()  # for test eao in vot-toolkit
eng.cd('./lib/core')

import os
from os import listdir
from os.path import join


absPath = os.path.abspath('.')
eval_path = os.path.join(absPath, 'lib/core/')

def eval_vot(dataset, result_path):
    trackers = listdir(join(result_path, dataset))

    for tracker in trackers:
        base_path = join(result_path, dataset, tracker, 'baseline')
        eao = eval_eao(base_path, dataset)

        print('[*] tracker: {0} : EAO: {1}'.format(tracker, eao))
        eng.cd(eval_path)


def eval_eao(base_path, dataset):
    """
    start matlab engin and test eao in vot toolkit
    """
    results = []
    videos = sorted(listdir(base_path))  # must sorted!!

    for video in videos:
        video_re = []
        path_v = join(base_path, video, '{}_001.txt'.format(video))
        fin = open(path_v).readlines()
        
        for line in fin:
            line = eval(line)  # tuple
            if isinstance(line, float) or isinstance(line, int):
                line = [float(line)]   # have to be float
            else:
                line = list(line)
            video_re.append(line)
        results.append(video_re)

    year = dataset.split('VOT')[-1]
    eao = eng.get_eao(results, year)

    return eao


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python ./lib/core/eval_vot.py VOT2017 ./result')
        exit()
    dataset = sys.argv[1]
    result_path = sys.argv[2]
    eval_vot(dataset, result_path)
