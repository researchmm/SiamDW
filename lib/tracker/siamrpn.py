# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# siamrpn class
# ------------------------------------------------------------------------------
import torch
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F
from utils.utils import load_yaml, get_subwindow_tracking, python2round, generate_anchor


class SiamRPN(object):
    """
    modified from VOT18 released model
    """
    def __init__(self, info):
        super(SiamRPN, self).__init__()
        self.info = info   # model and benchmark info

    def init(self, im, target_pos, target_sz, model, hp=None):
        state = dict()
        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]
        p = RPNConfig()

        # single test
        if not hp and not self.info.epoch_test:
            prefix = [x for x in ['OTB', 'VOT'] if x in self.info.dataset]
            cfg = load_yaml('./experiments/test/{0}/{1}.yaml'.format(prefix[0], self.info.arch))
            cfg_benchmark = cfg[self.info.dataset]
            p.update(cfg_benchmark)
            p.renew()

        # for vot17 or vot18: from siamrpn released
        if '2017' in self.info.dataset:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = 287
                p.renew()
            else:
                p.instance_size = 271
                p.renew()

        # param tune
        if hp:
            p.update(hp)
            p.renew()

            # for small object (from DaSiamRPN released)
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = hp['big_sz']
                p.renew()
            else:
                p.instance_size = hp['small_sz']
                p.renew()


        net = model
        p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = python2round(np.sqrt(wc_z * hc_z))

        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

        z = Variable(z_crop.unsqueeze(0))
        net.template(z.cuda())

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.expand_dims(window, axis=0)           # [1,17,17]
        window = np.repeat(window, p.anchor_num, axis=0)  # [5,17,17]

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        return state

    def update(self, net, x_crop, target_pos, target_sz, window, scale_z, p):
        score, delta = net.track(x_crop)

        b, c, s, s = delta.size()
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1, s, s).data.cpu().numpy()  # [4,5,17,17]
        if self.info.cls_type == 'thinner':
            score = torch.sigmoid(score).squeeze().cpu().data.numpy()  # [5,17,17]
        elif self.info.cls_type == 'thicker':
            score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1, s, s), dim=0).squeeze().data[1, ...].cpu().numpy()  # [5,17,17]

        delta[0, ...] = delta[0, ...] * p.anchor[2, ...] + p.anchor[0, ...]
        delta[1, ...] = delta[1, ...] * p.anchor[3, ...] + p.anchor[1, ...]
        delta[2, ...] = np.exp(delta[2, ...]) * p.anchor[2, ...]
        delta[3, ...] = np.exp(delta[3, ...]) * p.anchor[3, ...]

        # size penalty
        s_c = self.change(self.sz(delta[2, ...], delta[3, ...]) / (self.sz_wh(target_sz)))  # scale penalty  [5,17,17]
        r_c = self.change((target_sz[0] / target_sz[1]) / (delta[2, ...] / delta[3, ...]))  # ratio penalty  [5,17,17]

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)  # [5,17,17]
        pscore = penalty * score  # [5, 17, 17]

        # window float
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence  # [5, 17, 17]
        a_max, r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        target = delta[:, a_max, r_max, c_max] / scale_z  # [4,1]

        target_sz = target_sz / scale_z
        lr = penalty[a_max, r_max, c_max] * score[a_max, r_max, c_max] * p.lr  # lr for OTB

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        return target_pos, target_sz, score[a_max, r_max, c_max]

    def track(self, state, im):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans).unsqueeze(0))

        target_pos, target_sz, score = self.update(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['score'] = score
        return state

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class RPNConfig(object):
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1
    context_amount = 0.5
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1
        self.anchor_num = len(self.ratios) * len(self.scales)



