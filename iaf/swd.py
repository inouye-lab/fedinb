import torch
import torch.nn as nn
import wandb
from copy import deepcopy
import time

def SlicedWassersteinDistance(theta, x_dict,n_classes, perdim=True):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    When M=2, this is |x-y|^2/2, but this should not affect the gradient
    '''
    bary = 0
    for i, x in enumerate(x_dict.values()):
        bary += torch.sort(theta @ x.T, dim=-1)[0]
    bary = bary / n_classes

    SWD = 0
    for i, x in enumerate(x_dict.values()):
        x1 = torch.sort(theta @ x.T, dim=-1)[0]
        if perdim:
            SWD += torch.mean(torch.abs(x1 - bary) ** 2)
        else:
            SWD += torch.mean(torch.abs(x1 - bary) ** 2, dim=-1)
    return SWD

def FedSlicedWassersteinDistance(theta, x_dict,n_classes, perdim=True):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    When M=2, this is |x-y|^2/2, but this should not affect the gradient
    '''
    bary = 0
    for i, x in enumerate(x_dict.values()):
        bary += torch.sort(theta @ x.T, dim=-1)[0].detach()
    bary = bary / n_classes

    SWD = 0
    for i, x in enumerate(x_dict.values()):
        x1 = torch.sort(theta @ x.T, dim=-1)[0]
        if perdim:
            SWD += torch.mean(torch.abs(x1 - bary) ** 2)
        else:
            SWD += torch.mean(torch.abs(x1 - bary) ** 2, dim=-1)
    return SWD


def maxSWDdirection(X, y, K=None, maxiter=200, Npercentile=None, p=2, eps=1e-6, weight=None, vis_swd=False, is_fed=False):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    Designed for calculating distance for k>2
    '''
    # unifrom weight if no weight is assigned
    classes = torch.unique(y)
    n_classes = len(classes)
    if weight is None:
        weight = torch.ones((1, n_classes)) / n_classes
    assert weight.shape[0] == 1

    X_list = dict()
    for t in classes:
        X_list[t] = X[y==t]

    d = X.shape[1]
    if K is None:
        K = d

    # initialize orthonormal projection matrix w/theta
    # algorithm from https://arxiv.org/pdf/math-ph/0609050.pdf
    wi = torch.randn(d, K, device=X.device)
    Q, R = torch.qr(wi)
    L = torch.sign(torch.diag(R))
    w = (Q * L).T

    lr = 0.1
    down_fac = 0.5
    up_fac = 1.5
    c = 0.5

    # algorithm from http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
    # note that here w = X.T
    # use backtracking line search
    w1 = w.clone()
    w.requires_grad_(True)

    if is_fed:
        loss = -FedSlicedWassersteinDistance(w, X_list, n_classes)
    else:
        loss = -SlicedWassersteinDistance(w, X_list, n_classes)
    loss1 = loss
    loss_list = []
    loss_list.append(loss)

    nums = 0
    for iter in range(maxiter):
        GT = torch.autograd.grad(loss, w)[0]
        w.requires_grad_(False)
        WT = w.T @ GT - GT.T @ w
        e = - w @ WT  # dw/dlr
        m = torch.sum(GT * e)  # dloss/dlr
        lr /= down_fac

        while loss1 > loss + c * m * lr:
            lr *= down_fac
            if 2 * K < d:
                UT = torch.cat((GT, w), dim=0).double()
                V = torch.cat((w.T, -GT.T), dim=1).double()
                w1 = (w.double() - lr * w.double() @ V @ torch.pinverse(
                    torch.eye(2 * K, dtype=torch.double, device=X.device)
                    + lr / 2 * UT @ V) @ UT).to(torch.get_default_dtype())
            else:
                w1 = (w.double() @ (
                            torch.eye(d, dtype=torch.double, device=X.device)
                            - lr / 2 * WT.double()) @ torch.pinverse(
                    torch.eye(d, dtype=torch.double, device=X.device)
                    + lr / 2 * WT.double())).to(torch.get_default_dtype())
            w1.requires_grad_(True)

            if is_fed:
                loss1 = - FedSlicedWassersteinDistance(w1, X_list, n_classes)

            else:
                loss1 = - SlicedWassersteinDistance(w1, X_list, n_classes)

            loss_list.append(loss1)
            nums +=1

        if torch.max(torch.abs(w1 - w)) < eps:
            w = w1
            break

        lr *= up_fac
        loss = loss1
        w = w1


    iter += 1
    extra_iter = nums - iter
    # print(iter)
    # print(extra_iter)
    # wandb.log({'iters': nums})
    # wandb.log({'loss_list': loss_list})

    if is_fed:
        KSWD = FedSlicedWassersteinDistance(w, X_list, n_classes, perdim=False)
    else:
        KSWD = SlicedWassersteinDistance(w, X_list, n_classes, perdim=False)
    if vis_swd:
        print(f'After {iter} iterations, current max-K-SW is {torch.mean(KSWD)}')
    return w.T, KSWD ** (1 / p), iter, extra_iter


