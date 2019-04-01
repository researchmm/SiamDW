# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Main Results: see readme.md
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable


class SiamFC_(nn.Module):
    def __init__(self):
        super(SiamFC_, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking
        self.criterion = nn.BCEWithLogitsLoss()

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0: return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  # the same as tf version

    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def template(self, z):
        self.zf = self.feature_extractor(z)

    def track(self, x):
        xf = self.feature_extractor(x)
        score = self.connector(self.zf, xf)
        return score

    def forward(self, template, search, label=None):
        zf = self.feature_extractor(template)
        xf = self.feature_extractor(search)
        score = self.connector(zf, xf)
        if self.training:
            return self._weighted_BCE(score, label)
        else:
            raise ValueError('forward is only used for training.')





