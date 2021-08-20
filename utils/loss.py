import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
from pdb import set_trace as bkpt

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1, _ = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent

def adentropy(F1, feat, iteration, args, eta=1.0):
    lamda = args.lamda
    out_t1, _ = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=1)
    uc, _ = torch.max(out_t1, dim=1)
    weights = 1 - uc
    if iteration < args.start_weighted_entropy: 
        loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    else:
        loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

def adentropy_old(F1, feat, lamda, eta=1.0):
    out_t1, _ = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent
