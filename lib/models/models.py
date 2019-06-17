# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: This script provides main models proposed in CVPR2019 paper
#    1) SiamFCRes22: SiamFC with CIResNet-22 backbone
#    2) SiamFCRes22: SiamFC with CIResIncep-22 backbone
#    3) SiamFCNext22:SiamFC with CIResNext-22 backbone
#    4) SiamFCRes22W:Double 3*3 in the residual blob of CIResNet-22
# Main Results: see readme.md
# ------------------------------------------------------------------------------

from .siamfc import SiamFC_
from .siamrpn import SiamRPN_
from .connect import Corr_Up, RPN_Up
from .backbones import ResNet22, Incep22, ResNeXt22, ResNet22W


class SiamFCRes22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        self.connect_model = Corr_Up()


class SiamFCIncep22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCIncep22, self).__init__(**kwargs)
        self.features = Incep22()
        self.connect_model = Corr_Up()


class SiamFCNext22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCNext22, self).__init__(**kwargs)
        self.features = ResNeXt22()
        self.connect_model = Corr_Up()


class SiamFCRes22W(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22W, self).__init__(**kwargs)
        self.features = ResNet22W()
        self.connect_model = Corr_Up()


class SiamRPNRes22(SiamRPN_):
    def __init__(self, **kwargs):
        super(SiamRPNRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        inchannels = self.features.feature_size

        if self.cls_type == 'thinner': outchannels = 256
        elif self.cls_type == 'thicker': outchannels = 512
        else: raise ValueError('not implemented loss/cls type')

        self.connect_model = RPN_Up(anchor_nums=self.anchor_nums,
                                    inchannels=inchannels,
                                    outchannels=outchannels,
                                    cls_type = self.cls_type)


