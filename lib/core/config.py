# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email:  houwen.peng@microsoft.com
# Details: This script provides configs for siamfc and siamrpn
# ------------------------------------------------------------------------------

import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------config for general parameters------
config.GPUS = "0,1,2,3"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'logs'
config.CHECKPOINT_DIR = 'snapshot'

# #-----————- config for siamfc ------------
config.SIAMFC = edict()
config.SIAMFC.TRAIN = edict()
config.SIAMFC.TEST = edict()
config.SIAMFC.TUNE = edict()
config.SIAMFC.DATASET = edict()
config.SIAMFC.DATASET.VID = edict()       # paper utlized but not recommended
config.SIAMFC.DATASET.GOT10K = edict()    # not utlized in paper but recommended, better performance and more stable

# augmentation
config.SIAMFC.DATASET.SHIFT = 4
config.SIAMFC.DATASET.SCALE = 0.05
config.SIAMFC.DATASET.COLOR = 1
config.SIAMFC.DATASET.FLIP = 0
config.SIAMFC.DATASET.BLUR = 0
config.SIAMFC.DATASET.ROTATION = 0

# vid
config.SIAMFC.DATASET.VID.PATH = '/home/zhbli/Dataset/data2/vid/crop511'
config.SIAMFC.DATASET.VID.ANNOTATION = '/home/zhbli/Dataset/data2/vid/train.json'

# got10k
config.SIAMFC.DATASET.GOT10K.PATH = '/home/zhbli/Dataset/data3/got10k/crop511'
config.SIAMFC.DATASET.GOT10K.ANNOTATION = '/home/zhbli/Dataset/data3/got10k/train.json'

# train
config.SIAMFC.TRAIN.MODEL = "SiamFCIncep22"
config.SIAMFC.TRAIN.RESUME = False
config.SIAMFC.TRAIN.START_EPOCH = 0
config.SIAMFC.TRAIN.END_EPOCH = 50
config.SIAMFC.TRAIN.TEMPLATE_SIZE = 127
config.SIAMFC.TRAIN.SEARCH_SIZE = 255
config.SIAMFC.TRAIN.STRIDE = 8
config.SIAMFC.TRAIN.BATCH = 32
config.SIAMFC.TRAIN.PAIRS = 200000
config.SIAMFC.TRAIN.PRETRAIN = 'resnet23_inlayer.model'
config.SIAMFC.TRAIN.LR_POLICY = 'log'
config.SIAMFC.TRAIN.LR = 0.001
config.SIAMFC.TRAIN.LR_END = 0.0000001
config.SIAMFC.TRAIN.MOMENTUM = 0.9
config.SIAMFC.TRAIN.WEIGHT_DECAY = 0.0001
config.SIAMFC.TRAIN.WHICH_USE = 'GOT10K'  # VID or 'GOT10K'

# test
config.SIAMFC.TEST.MODEL = config.SIAMFC.TRAIN.MODEL
config.SIAMFC.TEST.DATA = 'VOT2015'
config.SIAMFC.TEST.START_EPOCH = 30
config.SIAMFC.TEST.END_EPOCH = 50

# tune
config.SIAMFC.TUNE.MODEL = config.SIAMFC.TRAIN.MODEL
config.SIAMFC.TUNE.DATA = 'VOT2015'
config.SIAMFC.TUNE.METHOD = 'GENE'  # 'GENE' or 'RAY'



def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST', 'TUNE']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'COCO', 'DET', 'YTB']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    config[model_name][k][vk][vvk] = vvv
    else:
        config[k] = v   # gpu et.


def update_config(config_file):
    """
    ADD new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['SIAMFC', 'SIAMRPN']:
            raise ValueError('please edit config.py to support new model')

        model_config = exp_config[model_name]  # siamfc or siamrpn
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k=SIAMFC or SIAMRPN
            else:
                raise ValueError("{} not exist in config.py".format(k))
