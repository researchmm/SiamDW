# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: This script provides CIR backbones proposed in CVPR2019 paper
# Main Results: see readme.md
# ------------------------------------------------------------------------------

import torch.nn as nn
from .modules import Bottleneck_CI, Bottleneck_BIG_CI, ResNet, Inception, InceptionM, ResNeXt


eps = 1e-5
class ResNet22(nn.Module):
    """
    FAT: fix all at first (for siamrpn)
    """
    def __init__(self):
        super(ResNet22, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        self.feature_size = 512

    def forward(self, x):
        x = self.features(x)
        return x


class Incep22(nn.Module):
    def __init__(self):
        super(Incep22, self).__init__()
        self.features = Inception(InceptionM, [3, 4])
        self.feature_size = 640

    def forward(self, x):
        x = self.features(x)
        return x


class ResNeXt22(nn.Module):
    def __init__(self):
        super(ResNeXt22, self).__init__()
        self.features = ResNeXt(num_blocks=[3, 4], cardinality=32, bottleneck_width=4)
        self.feature_size = 512

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet22W(nn.Module):
    """
    ResNet22W: double 3*3 layer (only) channels in residual blob
    """
    def __init__(self):
        super(ResNet22W, self).__init__()
        self.features = ResNet(Bottleneck_BIG_CI, [3, 4], [True, False], [False, True], firstchannels=64, channels=[64, 128])
        self.feature_size = 512

    def forward(self, x):
        x = self.features(x)

        return x

