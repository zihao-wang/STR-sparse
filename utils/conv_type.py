from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args
import numpy as np

DenseConv = nn.Conv2d

def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x)*activation(torch.abs(x)-f(s))

def initialize_sInit():

    if parser_args.sInit_type == "constant":
        return parser_args.sInit_value*torch.ones([1, 1])


class VanillaConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, t=1e-3):
        nonzero = self.weight.gt(t).sum()
        total = self.weight.numel()
        return nonzero, total, t


class STRConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu

        if parser_args.sparse_function == 'sigmoid':
            self.f = torch.sigmoid
            self.sparseThreshold = nn.Parameter(initialize_sInit())
        else:
            self.sparseThreshold = nn.Parameter(initialize_sInit())

    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return temp.sum(), temp.numel(), f(self.sparseThreshold).item()


class SparedConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.red_weight = nn.Parameter(self.weight.clone())

    def forward(self, x):
        sparse_weight = self.red_weight * self.weight
        x = F.conv2d(
            x, sparse_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, t=1e-3):
        sparse_weight = self.red_weight * self.weight
        nonzero = sparse_weight.gt(t).sum()
        total = sparse_weight.numel()
        return nonzero, total, t


class ChooseEdges(autograd.Function):
    @staticmethod
    def forward(ctx, weight, prune_rate):
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(prune_rate * weight.numel())
        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class DNWConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print(f"=> Setting prune rate to {prune_rate}")

    def forward(self, x):
        w = ChooseEdges.apply(self.weight, self.prune_rate)

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x

def GMPChooseEdges(weight, prune_rate):
    output = weight.clone()
    _, idx = weight.flatten().abs().sort()
    p = int(prune_rate * weight.numel())
    # flat_oup and output access the same memory.
    flat_oup = output.flatten()
    flat_oup[idx[:p]] = 0
    return output

class GMPConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        self.curr_prune_rate = 0.0
        print(f"=> Setting prune rate to {prune_rate}")

    def set_curr_prune_rate(self, curr_prune_rate):
        self.curr_prune_rate = curr_prune_rate

    def forward(self, x):
        w = GMPChooseEdges(self.weight, self.curr_prune_rate)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x
