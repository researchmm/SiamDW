# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Writtern by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: SiamFC tuning script
# Choose a validation dataset according to your needs
# ------------------------------------------------------------------------------
from __future__ import absolute_import
import _init_paths
import os
import time
import numpy as np
import argparse
import models.models as models
from easydict import EasyDict as edict
from utils.utils import load_pretrain
from tracker.siamfc import SiamFC
from test_siamfc import auc_otb, eao_vot

from mpi4py import MPI
from gaft import GAEngine
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.components import Population, DecimalIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation


parser = argparse.ArgumentParser(description='parameters for SiamFC tracker')
parser.add_argument('--arch', dest='arch', default='SiamFCRes23',
                    help='architecture of pretrained model')
parser.add_argument('--resume', default='', type=str, required=True,
                    help='resumed checkpoint')
parser.add_argument('--gpu_nums', default=4, type=int, help='gpu numbers')
parser.add_argument('--dataset', default='OTB2013', type=str, metavar='DATASET', help='dataset')
args = parser.parse_args()

print('==> GENE works not well with SiamRPN')

# Distribute task on multi-gpu
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name() # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
time.sleep(rank * 5)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))

# define population
indv_template = DecimalIndividual(ranges=[(1.0, 1.2), (0.15, 0.7), (0.9, 1.0), (0.05, 0.65)], eps=0.0001)
population = Population(indv_template=indv_template, size=100)  # zzp: Population size
population.init()  # Initialize population with individuals.

# Create genetic operators
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine to run optimization
engine = GAEngine(population=population, selection=selection, \
                  crossover=crossover, mutation=mutation, \
                  analysis=[FitnessStore, ConsoleOutput])

# create model
net = models.__dict__[args.arch]()
net = load_pretrain(net, args.resume)
net.eval()
net = net.cuda()
print('==> pretrained model has been loaded')

# prepare tracker
info = edict()
info.arch = args.arch
info.dataset = args.dataset
info.epoch_test = False
tracker = SiamFC(info)


# Define fitness function.
@engine.fitness_register
def fitness(indv):
    scale_step, scale_lr, scale_penalty, window_influence = indv.solution

    # add params top config
    config = dict()
    config['benchmark'] = args.dataset
    config['arch'] = args.arch
    config['resume'] = args.resume
    config['hp'] = dict()
    config['hp']['scale_step'] = scale_step
    config['hp']['scale_penalty'] = scale_penalty
    config['hp']['scale_lr'] = scale_lr
    config['hp']['w_influence'] = window_influence

    if args.dataset.startswith('OTB'):
        auc = auc_otb(tracker, net, config)
        print("scale_step: {0}, scale_lr: {1}, scale_penalty: {2}, window_influence: {3}, auc: {4}".format(scale_step,
                                                                                                           scale_lr,
                                                                                                           scale_penalty,
                                                                                                           window_influence,
                                                                                                           auc.item()))
        return auc.item()

    elif args.dataset.startswith('VOT'):
        eao = eao_vot(tracker, net, config)
        print("scale_step: {0}, scale_lr: {1}, scale_penalty: {2}, window_influence: {3}, eao: {4}".format(scale_step,
                                                                                                           scale_lr,
                                                                                                           scale_penalty,
                                                                                                           window_influence,
                                                                                                           eao))
        return eao


if __name__ == "__main__":
    engine.run(ng=100)




