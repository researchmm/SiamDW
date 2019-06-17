from __future__ import absolute_import
import _init_paths
import os
import argparse
import numpy as np

import models.models as models
from utils.utils import load_pretrain
from test_siamfc import auc_otb, eao_vot
from test_siamrpn import eao_vot_rpn
from tracker.siamfc import SiamFC
from tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch
from hyperopt import hp


parser = argparse.ArgumentParser(description='tuning for both SiamFC and SiamRPN (works well on VOT)')
parser.add_argument('--arch', dest='arch', default='SiamFCRes22', help='architecture of model')
parser.add_argument('--resume', default='', type=str, required=True, help='resumed model')
parser.add_argument('--gpu_nums', default=4, type=int, help='gpu numbers')
parser.add_argument('--anchor_nums', default=5, type=int,  help='anchor numbers for rpn')
parser.add_argument('--cls_type', default="thicker", type=str,  help='cls/loss type, thicker or thinner or else you defined')
parser.add_argument('--dataset', default='VOT2015', type=str, help='dataset')

args = parser.parse_args()

print('==> TPE works well with both SiamFC and SiamRPN')
print('==> However TPE is slower than GENE')

# prepare tracker -- rpn waited
info = edict()
info.arch = args.arch
info.dataset = args.dataset
info.epoch_test = False
info.cls_type = args.cls_type

# create model
if 'SiamFC' in args.arch:
    model = models.__dict__[args.arch]()
    tracker = SiamFC(info)
elif 'SiamRPN' in args.arch:
    model = models.__dict__[args.arch](anchors_nums=args.anchor_nums, cls_type=args.cls_type)
    tracker = SiamRPN(info)
else:
    raise ValueError('not supported other model now')

model = load_pretrain(model, args.resume)
model.eval()
model = model.cuda()
print('pretrained model has been loaded')


# fitness function
def fitness(config, reporter):
    # different params for SiamFC and SiamRPN
    if 'SiamFC' in args.arch:
        scale_step = config["scale_step"]
        scale_penalty = config["scale_penalty"]
        scale_lr = config["scale_lr"]
        w_influence = config["w_influence"]
        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['scale_step'] = scale_step
        model_config['hp']['scale_penalty'] = scale_penalty
        model_config['hp']['w_influence'] = w_influence
        model_config['hp']['scale_lr'] = scale_lr
    elif 'SiamRPN' in args.arch:
        penalty_k = config["penalty_k"]
        scale_lr = config["scale_lr"]
        window_influence = config["window_influence"]
        small_sz = config["small_sz"]
        big_sz = config["big_sz"]

        model_config = dict()
        model_config['benchmark'] = args.dataset
        model_config['arch'] = args.arch
        model_config['resume'] = args.resume
        model_config['hp'] = dict()
        model_config['hp']['penalty_k'] = penalty_k
        model_config['hp']['window_influence'] = window_influence
        model_config['hp']['lr'] = scale_lr
        model_config['hp']['small_sz'] = small_sz
        model_config['hp']['big_sz'] = big_sz

    # OTB and SiamFC
    if args.dataset.startswith('OTB') and 'SiamFC' in args.arch:
        auc = auc_otb(tracker, model, model_config)
        print("scale_step: {0}, scale_lr: {1}, scale_penalty: {2}, window_influence: {3}, auc: {4}".format(scale_step, scale_lr, scale_penalty, w_influence, auc.item()))
        reporter(AUC=auc)

    # VOT and SiamFC
    if args.dataset.startswith('VOT') and 'SiamFC' in args.arch:
        eao = eao_vot(tracker, model, model_config)
        print("scale_step: {0}, scale_lr: {1}, scale_penalty: {2}, window_influence: {3}, eao: {4}".format(scale_step, scale_lr, scale_penalty, w_influence, eao))
        reporter(EAO=eao)

    # VOT and SiamRPN
    if args.dataset.startswith('VOT') and 'SiamRPN' in args.arch:
        eao = eao_vot_rpn(tracker, model, model_config)
        print("penalty_k: {0}, scale_lr: {1}, window_influence: {2}, small_sz: {3}, big_sz: {4}, eao: {5}".format(penalty_k, scale_lr, window_influence, small_sz, big_sz, eao))
        reporter(EAO=eao)


if __name__ == "__main__":
    # the resources you computer have, object_store_memory is shm
    ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 8, redirect_output=True, object_store_memory=30000000000)
    tune.register_trainable("fitness", fitness)

    # define search space for SiamFC or SiamRPN
    if 'SiamFC' in args.arch:
        params = {
            "scale_step": hp.quniform('scale_step', 1.0, 1.2, 0.0001),
            "scale_penalty": hp.quniform('scale_penalty', 0.95, 1.0, 0.0001),
            "w_influence": hp.quniform('w_influence', 0.05, 0.7, 0.0001),
            "scale_lr": hp.quniform("scale_lr", 0.15, 0.7, 0.0001),
        }

    if 'SiamRPN' in args.arch:
        params = {
                "penalty_k": hp.quniform('penalty_k', 0.001, 0.6, 0.001),
                "scale_lr": hp.quniform('scale_lr', 0.1, 0.8, 0.001),
                "window_influence": hp.quniform('window_influence', 0.05, 0.65, 0.001),
                "small_sz": hp.choice("small_sz", [255]),
                "big_sz": hp.choice("big_sz", [255]),
                # "small_sz": hp.choice("small_sz", [255, 271]),
                # "big_sz": hp.choice("big_sz", [255, 271, 287]),
                }

    tune_spec = {
        "zp_tune": {
            "run": "fitness",
            "trial_resources": {
                "cpu": 1,  # single task cpu num
                "gpu": 0.5,  # single task gpu num
            },
            "num_samples": 10000,  # sample hyperparameters times
            "local_dir": './TPE_results'
        }
    }

    # stop condition for VOT and OTB
    if args.dataset.startswith('VOT'):
        stop = {
            "EAO": 0.50,  # if EAO >= 0.50, this procedures will stop
            # "timesteps_total": 100, # iteration times
        }
        tune_spec['zp_tune']['stop'] = stop

        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="EAO",
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*2 + 1, reward_attr="EAO") # max_concurrent: the max running task

    elif args.dataset.startswith('OTB'):
        stop = {
            # "timesteps_total": 100, # iteration times
            "AUC": 0.80
        }
        tune_spec['zp_tune']['stop'] = stop
        scheduler = AsyncHyperBandScheduler(
            # time_attr="timesteps_total",
            reward_attr="AUC",
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*2 + 1, reward_attr="AUC")  #
    else:
        raise ValueError("not support other dataset now")

    tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)



