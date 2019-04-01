# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Main Results: see readme.md
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SiamRPN_(nn.Module):
    def __init__(self, anchors_nums=None):
        super(SiamRPN_, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking
        self.anchor_nums = anchors_nums

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_cls, pred_reg = self.connect_model(template_feature, search_feature)
        return pred_cls, pred_reg

    def template(self, z):
        self.zf = self.feature_extractor(z)

    def track(self, x):
        xf = self.feature_extractor(x)
        pred_cls, pred_reg = self.connector(self.zf, xf)
        return pred_cls, pred_reg







