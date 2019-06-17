# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: zhangzhipeng2017@ia.ac.cn
# Details: This script provides CIR backbones proposed in CVPR2019 paper
# Main Results: see readme.md
# ------------------------------------------------------------------------------

import torch.nn as nn
from .modules import Bottleneck_CI, Bottleneck_BIG_CI, ResNet, Inception, InceptionM, ResNeXt


eps = 1e-5
class ResNet22(nn.Module):
    """
    default: unfix gradually (lr: 1r-2 ~ 1e-5)
    optional: unfix all at first with small lr (lr: 1e-7 ~ 1e-3)
    """
    def __init__(self):
        super(ResNet22, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        self.feature_size = 512
        self.train_num = 0
        self.unfix(0.0)

    def forward(self, x):
        x = self.features(x)
        return x

    def unfix(self, ratio):
        """
        unfix gradually as paper said
        """
        if abs(ratio - 0.0) < eps:
            self.train_num = 2  # epoch0 1*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.1) < eps:
            self.train_num = 3  # epoch5 2*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.2) < eps:
            self.train_num = 4  # epoch10 3*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.3) < eps:
            self.train_num = 6  # epoch15 4*[1,3,1]  stride2pool makes stage2 have a more index
            self.unlock()
            return True
        elif abs(ratio - 0.5) < eps:
            self.train_num = 7  # epoch25 5*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.6) < eps:
            self.train_num = 8  # epoch30 6*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.7) < eps:
            self.train_num = 9  # epoch35 7*[1,3,1]
            self.unlock()
            return True

        return False

    def unlock(self):
        for p in self.parameters():
            p.requires_grad = False

        for i in range(1, self.train_num):  # zzp pay attention here
            if i <= 5:
                m = self.features.layer2[-i]
            elif i <= 8:
                m = self.features.layer1[-(i - 5)]
            else:
                m = self.features

            for p in m.parameters():
                p.requires_grad = True
        self.eval()
        self.train()

    def train(self, mode=True):
        self.training = mode
        if mode == False:
            super(ResNet22, self).train(False)
        else:
            for i in range(self.train_num):  # zzp pay attention here
                if i <= 5:
                    m = self.features.layer2[-i]
                elif i <= 8:
                    m = self.features.layer1[-(i - 5)]
                else:
                    m = self.features
                m.train(mode)

        return self


# --------------------------------------------------------------
# The next few backbones donot set "unfix gradually" as default
# You can modify refering to ResNet22
# ResNet22W with GOT10K is highly recommended
# --------------------------------------------------------------

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

