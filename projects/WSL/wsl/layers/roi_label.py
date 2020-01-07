# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from wsl import _C


class _ROILabel(Function):
    @staticmethod
    def forward(ctx, S, U, L, CW, P):
        RL, RW, P_ = _C.roi_label_forward(S, U, L, CW, P)
        ctx.mark_non_differentiable(RL)
        ctx.mark_non_differentiable(RW)
        ctx.mark_non_differentiable(P)
        return RL, RW, P

    @staticmethod
    @once_differentiable
    def backward(ctx, GRL, GRW, GP):
        return None, None, None, None, None


roi_label = _ROILabel.apply


class ROILabel(nn.Module):
    def __init__(
        self, fg_threshold, bg_thresh_hi, bg_thresh_lo, num_pos, num_neg, top_k, debug_info, display
    ):
        super(ROILabel, self).__init__()

        self.P = nn.Parameter(torch.zeros(14, dtype=torch.float), requires_grad=False)

        self.P[0] = fg_threshold
        self.P[1] = bg_thresh_hi
        self.P[2] = bg_thresh_lo

        self.P[3] = num_pos
        self.P[4] = num_neg
        self.P[5] = top_k

        self.P[6] = debug_info
        self.P[7] = random.randint(1, 9999)
        self.P[8] = display

        self.P[9] = 0

        self.P[10] = 0
        self.P[11] = 0

        self.P[12] = 0
        self.P[13] = 0

    def forward(self, S, U, L, CW):
        P = self.P.cpu()
        RL, RW, P = roi_label(S, U, L, CW, P)
        self.P.copy_(P)

        return RL, RW

    def __repr__(self):
        P = self.P.cpu().numpy()
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "fg_threshold=" + str(P[0])
        tmpstr += " bg_thresh_hi=" + str(P[1])
        tmpstr += " bg_thresh_lo=" + str(P[2])
        tmpstr += " num_pos=" + str(P[3])
        tmpstr += " num_neg=" + str(P[4])
        tmpstr += " top_k=" + str(P[5])
        tmpstr += " debug_info=" + str(P[6])
        tmpstr += " uuid=" + str(P[7])
        tmpstr += " display=" + str(P[8])
        tmpstr += " cur_iter=" + str(P[9])
        tmpstr += " acc_fg_rois=" + str(P[10])
        tmpstr += " acc_bg_rois=" + str(P[11])
        tmpstr += " acc_fg_weight=" + str(P[12])
        tmpstr += " acc_bg_weight=" + str(P[13])
        tmpstr += ")"
        return tmpstr
