import torch
import torch.nn as nn

from .naive_bary import NaiveBary
from .swd import maxSWDdirection
from .swd_hist import QHistmaxSWDdirection
import torch.distributions as distribution



class StandardGaussianTransformer(nn.Module):
    # First normalize the data using
    def __init__(self):
        super().__init__()

    def fit(self, x):
        mean = torch.zeros(x.shape[1]).to(x.device)
        std = torch.ones(x.shape[1]).to(x.device)
        self.normal = distribution.normal.Normal(mean, std)
        return self

    def forward(self, x):
        x = self.normal.cdf(x)
        #return torch.clamp(x, min=1e-6, max=1 - 1e-6)
        return x

    def inverse(self, x):
        x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
        x = self.normal.icdf(x)
        return x

class GaussianInverseCDF(nn.Module):
    '''
    Independent inverse Gaussian CDF transformer applied coordinate-wise.
    From [0,1] to [-inf,inf]
    '''

    def __init__(self):
        super().__init__()
        self.fitted_cdf = None

    def fit(self, X, y):
        classes = torch.unique(y)
        fitted_cdf = dict()
        for i, yy in enumerate(classes):
            gt = StandardGaussianTransformer()
            gt.fit(X[y==yy,:])
            fitted_cdf[int(yy)] = gt

        self.fitted_cdf = fitted_cdf
        return self


    def forward(self, x, y):
        z = torch.zeros_like(x)
        classes = torch.unique(y)
        for yy in classes:
            z[y==yy] = self.fitted_cdf[int(yy)].inverse(x[y==yy])
        return z

    def inverse(self, x, y):
        z = torch.zeros_like(x)
        classes = torch.unique(y)
        for yy in classes:
            z[y==yy] = self.fitted_cdf[int(yy)](x[y==yy])
        return z


class GaussianCDF(nn.Module):
    '''
    Independent Gaussian CDF transformer applied coordinate-wise.
    From [-inf,inf] to [0,1]
    '''

    def __init__(self):
        super().__init__()
        self.fitted_cdf = None

    def fit(self, X, y):
        classes = torch.unique(y)
        fitted_cdf = dict()
        for i, yy in enumerate(classes):
            gt = StandardGaussianTransformer()
            gt.fit(X[y == yy, :])
            fitted_cdf[int(yy)] = gt

        self.fitted_cdf = fitted_cdf

        return self

    def forward(self, x, y):
        z = torch.zeros_like(x)
        classes = torch.unique(y)
        for yy in classes:
            z[y == yy] = self.fitted_cdf[int(yy)](x[y == yy])
        return z

    def inverse(self, x, y):
        z = torch.zeros_like(x)
        classes = torch.unique(y)
        for yy in classes:
            z[y == yy] = self.fitted_cdf[int(yy)].inverse(x[y == yy])
        return z

class IterAlignFlow(nn.Module):
    """
    Iterative Alignment Flows (IAF)
    """

    def __init__(self):

        super().__init__()
        self.layer = nn.ModuleList([])
        self.num_layer = 0

    def forward(self, X, y):
        for i in range(len(self.layer)):
            X = self.layer[i](X, y)
        return X

    def inverse(self, X, y):
        for i in reversed(range(len(self.layer))):
            X = self.layer[i].inverse(X, y)
        return X

    # Add iters as needed
    def add_layer(self, layer, idx=None):
        if not idx:
            self.layer.append(layer)
        else:
            self.layer.insert(idx,layer)
        self.num_layer += 1
        return self

    # From [0,1] to [-inf,inf]
    def initialize(self, X, y):
        cdf = GaussianInverseCDF()
        cdf.fit(X, y)
        z = cdf(X, y)
        self.layer.append(cdf)
        self.num_layer += 1
        return z

    # From [-inf,inf] to [0,1]
    def end(self, X, y):
        cdf = GaussianCDF()
        cdf.fit(X, y)
        z = cdf(X, y)
        self.layer.append(cdf)
        self.num_layer += 1
        return z



class INB(nn.Module):
    """
    Iterative Naive Barycenter
    Building blocks for IAF
    Each instance is one layer/iteration in IAF
    """

    def __init__(self, d, K, bary_type='nb', device=torch.device('cpu')):
        super().__init__()
        self.d = d  # dimension of original data
        self.K = K

        # initialize w
        wi = torch.randn(self.d, self.K)
        Q, R = torch.qr(wi)
        L = torch.sign(torch.diag(R))
        wT = (Q * L)
        self.wT = wT

        self.bary_type = bary_type
        self.bary = None
        self.n_theta = dict()  # keep track of the number of samples used to fit theta
        self.n_t = dict()  # keep track of the number of samples used to fit t

    def fit_wT(self, X, y, K=30, MSWD_p=2, MSWD_max_iter=200, is_fed=False, vis_swd=False, swd_bins='auto', trans_hist=False,quantile=False):
        # find the projection matrix
        # modified from https://github.com/biweidai/SIG_GIS/blob/master/SIT.py
        if trans_hist:
            wT, SWD, swd_iters, swd_extra_iters = QHistmaxSWDdirection(X, y, K=K,
                                                                       maxiter=MSWD_max_iter,
                                                                       p=MSWD_p,
                                                                      is_fed=is_fed,
                                                                      vis_swd=vis_swd,
                                                                       swd_bins=swd_bins)
        else:
            wT, SWD, swd_iters, swd_extra_iters = maxSWDdirection(X, y, K=K, maxiter=MSWD_max_iter, p=MSWD_p, is_fed=is_fed,
                                                              vis_swd=vis_swd)
        self.wT = self.wT.to(wT.device)
        self.swd_iters = swd_iters
        self.swd_extra_iters = swd_extra_iters
        with torch.no_grad():
            SWD, indices = torch.sort(SWD, descending=True)
            wT = wT[:, indices]
            self.wT[:] = torch.qr(wT)[0]

        return self

    def fit_wT_rand(self, X, y, K=16):
        # for random projection compared to mSWD

        wi = torch.randn(X.shape[1], K, device=X.device)
        with torch.no_grad():
            self.wT[:] = torch.qr(wi)[0]
        return self

    def fit_bary(self, X, y, hist_bins='auto', n_grid=100, bound_eps='auto', weight=None):
        # fit the specified destructor

        if self.bary_type == 'nb':
            cd = NaiveBary(hist_bins=hist_bins, n_grid=n_grid, bound_eps=bound_eps)

        # fit the destrutor after the projection
        Xm = X @ self.wT

        cd.fit(Xm, y)
        #Xmm = cd(Xm, y)

        self.bary = cd

        K = Xm.shape[1]
        n_domains = len(torch.unique(y))
        self.nb_params = (cd.hist_bins + 2) * n_domains * K + (cd.n_grid + 2) * K * n_domains

        return self

    def transform(self, X, y, mode='forward'):

        Xm = X @ self.wT
        remaining = X - Xm @ self.wT.T

        if mode == 'forward':
            z = self.bary(Xm, y)
        elif mode == 'inverse':
            z = self.bary.inverse(Xm, y)

        X = remaining + z @ self.wT.T

        return X

    def forward(self, X, y):
        return self.transform(X, y, mode='forward')

    def inverse(self, X, y):
        return self.transform(X, y, mode='inverse')


def add_one_layer(model,
                  X,
                  y,
                  K,
                  n_samples_theta=None,
                  bary_type='nb',
                  rand=False,
                  is_fed=False,
                  hist_bins='auto',
                  n_grid=100,
                  bound_eps='auto',
                  weight=None,
                  vis_swd=False,
                  use_all_data=True,
                  max_swd_iters=200,
                  swd_bins='auto',
                  trans_hist=False,
                  quantile=False):
    """
    Add one layer to IAF
    """

    d = X.shape[1]

    if bary_type == 'nb':
        layer = INB(d, K, 'nb')

    classes = torch.unique(y)  # return labels in ascending order
    n_classes = len(classes)

    X_list = []
    y_list = []
    for t in classes:
        X_list.append(X[y == t])
        y_list.append(y[y == t])

    if not n_samples_theta:
        # if the number of samples used to fit theta is not specified, then use the half of samples
        # which is computed from the distribution with least samples

        min_n_samples = torch.inf
        for i in range(n_classes):
            cur_n_samples = len(X_list[i])
            min_n_samples = min(min_n_samples, cur_n_samples)

        if use_all_data:
            n_samples_theta = min_n_samples
        else:
            n_samples_theta = int(torch.floor(min_n_samples / 2))
        layer.n_theta['theta'] = n_samples_theta

    n_samples_t = n_samples_theta

    # prepare data for fitting theta and t
    x_theta = []
    y_theta = []
    x_t = []
    y_t = []

    for i in range(n_classes):
        layer.n_t[classes[i]] = len(X_list[i]) - n_samples_theta
        x_theta.append(X_list[i][:n_samples_theta])
        y_theta.append(y_list[i][:n_samples_theta])
        x_t.append(X_list[i][n_samples_t:])
        y_t.append(y_list[i][n_samples_t:])

    x_theta = torch.cat(x_theta)
    y_theta = torch.cat(y_theta)
    x_t = torch.cat(x_t)
    y_t = torch.cat(y_t)


    if rand:
        layer.fit_wT_rand(x_theta, y_theta, K=K)
    else:
        layer.fit_wT(x_theta, y_theta, K=K, MSWD_p=2, MSWD_max_iter=max_swd_iters, is_fed=is_fed, vis_swd=vis_swd,
                     swd_bins=swd_bins, trans_hist=trans_hist, quantile = quantile)

    if use_all_data:
        layer.fit_bary(X, y, hist_bins=hist_bins, n_grid=n_grid, bound_eps=bound_eps, weight=weight)
    else:
        layer.fit_bary(x_t, y_t, hist_bins=hist_bins, n_grid=n_grid, bound_eps=bound_eps, weight=weight)

    z = layer(X, y)

    model.add_layer(layer)

    return model, z
