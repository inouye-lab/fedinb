import torch
import torch.nn as nn
from copy import deepcopy
from .naive_bary import Histogram
import wandb
import time

import torch.distributions as distribution


def QHistSlicedWassersteinDistance(theta, x, y, swd_bins, perdim=True):
    '''
    Modified from https://github.com/biweidai/SIG_GIS/blob/master/SlicedWasserstein.py
    When M=2, this is |x-y|^2/2, but this should not affect the gradient
    '''
    slices = x @ theta.T

    # Step 1: Compute quantiles
    q = torch.linspace(0, 1, steps=swd_bins + 1).to(x.device)
    raw_quantiles = torch.stack([
        # torch.quantile(slices[y==yy], q[1:-1], dim=0) # Ignore min and max
        torch.quantile(slices[y == yy, :], q, dim=0)  # Allow min and max for this case
        for yy in torch.unique(y)  # "classes" is domains right?
    ])
    # Shape (n_classes/domains, n_quantiles, n_dir)
    # print('raw_quantiles', raw_quantiles.shape)

    # Compute bary quantiles
    bary_quantiles = raw_quantiles.mean(dim=0).to(x.device) # Shape (n_quantiles, n_dir)
    assert torch.all(torch.sort(bary_quantiles, dim=0)[0] == bary_quantiles), 'Should be sorted already'
    # print('bary_quantiles', bary_quantiles.shape)

    # Create barycenter from interpolated quantiles
    hist_bary = Histogram(bin_edges=bary_quantiles.T, counts=torch.ones(swd_bins).to(x.device))

    # Get corresponding icdf values for each sample
    #  NOTE: we are not sampling but getting n_sample quantiles via icdf.
    #  Also, these will already be sorted because sample_q is sorted.
    n_samples_per_class = torch.sum(y == y[0])
    sample_q = torch.linspace(0, 1, steps=n_samples_per_class).to(x.device)
    bary_values = hist_bary.icdf(sample_q.expand(bary_quantiles.shape[1], -1).T)
    bary_values = bary_values.T
    assert torch.all(torch.sort(bary_values, dim=1)[0] == bary_values), 'Should be sorted already'
    # print('bary_values', bary_values.shape)

    # Now use these sorted bary_values in the SWD calculations
    SWD = 0
    for yy in torch.unique(y):
        x1 = torch.sort(theta @ x[y == yy, :].T, dim=-1)[0]
        if perdim:
            SWD += torch.mean(torch.abs(x1 - bary_values) ** 2)
        else:
            SWD += torch.mean(torch.abs(x1 - bary_values) ** 2, dim=-1)
    return SWD


def QHistmaxSWDdirection(X, y, K=None, maxiter=200, Npercentile=None, p=2, eps=1e-6, weight=None, vis_swd=False,
                        is_fed=False, swd_bins='auto'):
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
        X_list[t] = X[y == t]

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

    loss = -QHistSlicedWassersteinDistance(w, X, y,swd_bins=swd_bins)
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

            loss1 = - QHistSlicedWassersteinDistance(w1, X, y,swd_bins=swd_bins)

            #wandb.log({'loss': loss1})
            loss_list.append(loss1)
            nums += 1

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

    KSWD = QHistSlicedWassersteinDistance(w, X, y, perdim=False, swd_bins=swd_bins)
    if vis_swd:
        print(f'After {iter} iterations, current max-K-SW is {torch.mean(KSWD)}')
    return w.T, KSWD ** (1 / p), iter, extra_iter

class VarbinwidthBary(nn.Module):

    def __init__(self, hist_bins='auto'):
        super().__init__()

        # determine the number of bins
        self.hist_bins = hist_bins

    def compute_bary(self, x, y):
        x = self._normalize(x, y)
        x = self._varbary(x, y)
        x = self._denormalize(x)
        sorted_x, _ = torch.sort(x, dim=0)
        return sorted_x

    def _normalize(self, x, y):

        assert len(x.shape) == 2 and len(y.shape) == 1, 'Check the dimension of input!'
        classes = torch.unique(y)
        self.classes = classes

        fitted_cdf = dict()
        xuni = torch.zeros_like(x).to(x.device)

        for i, yy in enumerate(classes):

            x_ = x[y == yy, :]

            mean = torch.mean(x_, dim=0)
            std = torch.std(x_, dim=0)
            normal = distribution.normal.Normal(mean, std)
            x_ = normal.cdf(x_)
            xuni[y == yy, :] = torch.clamp(x_, min=1e-6, max=1 - 1e-6)

            if i == 0:
                cum_mean = mean
                cum_std = std
            else:
                cum_mean += mean
                cum_std += std

        self.mean = cum_mean / len(classes)
        self.std = cum_std / len(classes)

        return xuni

    def _varbary(self, x, y):

        nd = x.shape[1]
        nsamples = int(x.shape[0] / len(self.classes))

        bins = self.hist_bins
        if bins == 'auto':
            bins = int(torch.round(torch.sqrt(torch.Tensor([x.shape[0]]))))
            self.hist_bins = bins

        assert torch.max(x) <= 1 and torch.min(x) >= 0, 'check normalization'

        q = torch.linspace(0, 1, steps=bins + 1).to(x.device)

        for i, yy in enumerate(self.classes):
            x_ = x[y == yy, :]
            cur_quantiles = torch.quantile(x_, q[1:-1], dim=0)
            cur_quantiles = torch.cat((torch.zeros(1, nd).to(x.device),
                                       cur_quantiles,
                                       torch.ones(1, nd).to(x.device)))
            if i == 0:
                bin_edges = cur_quantiles
            else:
                bin_edges += cur_quantiles
        bin_edges = bin_edges / len(self.classes)
        counts = torch.ones_like(bin_edges)[:-1, :]
        bin_edges = bin_edges.to(x.device)
        hist = Histogram(bin_edges.T, counts=counts.T)

        us = torch.rand(nsamples, nd).to(x.device)
        x_ = hist.icdf(us)
        return x_

    def _denormalize(self, x):

        normal = distribution.normal.Normal(self.mean, self.std)
        xreal = normal.icdf(x)
        return xreal








