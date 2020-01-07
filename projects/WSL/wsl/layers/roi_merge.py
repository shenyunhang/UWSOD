# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from wsl import _C


class _ROIMerge(Function):
    @staticmethod
    def forward(ctx, S, J, C, D, P):
        MC, MD, I, IC, P_ = _C.roi_merge_forward(S, J, C, D, P)
        ctx.save_for_backward(C, D, I, IC)
        ctx.mark_non_differentiable(I)
        ctx.mark_non_differentiable(IC)
        ctx.mark_non_differentiable(P)
        return MC, MD, P

    @staticmethod
    @once_differentiable
    def backward(ctx, GMC, GMD, GP):
        (C, D, I, IC) = ctx.saved_tensors
        GC, GD = _C.roi_merge_backward(C, D, GMC, GMD, I, IC)
        return None, None, GC, GD, None


roi_merge = _ROIMerge.apply


class ROIMerge(nn.Module):
    def __init__(self, debug_info, display, max_epoch, size_epoch):
        super(ROIMerge, self).__init__()

        self.P = nn.Parameter(torch.zeros(8, dtype=torch.int), requires_grad=False)
        self.P[0] = debug_info
        self.P[1] = display

        self.P[2] = 0
        self.P[3] = max_epoch
        self.P[4] = size_epoch

        self.P[5] = 0
        self.P[6] = 0
        self.P[7] = 0

    def forward(self, S, J, C, D):
        P = self.P.cpu()
        MC, MD, P = roi_merge(S, J, C, D, P)
        self.P.copy_(P)

        return MC, MD

    def __repr__(self):
        P = self.P.cpu().numpy()
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "debug_info=" + str(P[0])
        tmpstr += " display=" + str(P[1])
        tmpstr += " cur_iter=" + str(P[2])
        tmpstr += " max_epoch=" + str(P[3])
        tmpstr += " size_epoch=" + str(P[4])
        tmpstr += " acc_num_top_id=" + str(P[5])
        tmpstr += " acc_max_clique=" + str(P[6])
        tmpstr += " acc_min_clique=" + str(P[7])
        tmpstr += ")"
        return tmpstr
