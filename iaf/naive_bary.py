

import torch
import contextlib
from numbers import Number
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

import torch.distributions as distribution
import torch.nn as nn


class NaiveBary(nn.Module):
    def __init__(self, hist_bins='auto', n_grid=100, bound_eps='auto', fixed_bin_width=True):
        super().__init__()
        self.hist_bins = hist_bins
        self.n_grid = n_grid
        self.bound_eps = bound_eps
        self.fixed_bin_width = fixed_bin_width

    def _fit_cdf(self, X, y):

        assert len(X.shape) == 2 and len(y.shape) == 1, 'Check the dimension of input!'
        classes = torch.unique(y)

        # determine the number of bins
        bins = self.hist_bins
        if bins == 'auto':
            bins = int(torch.round(torch.sqrt(torch.Tensor([X.shape[0]]))))
            self.hist_bins = bins

        #        if self.fixed_bin_width:

        fitted_cdf = dict()
        for i, yy in enumerate(classes):
            ht = HistogramTransformer(bins)
            ht.fit(X[y == yy, :])
            fitted_cdf[int(yy)] = ht

        self.fitted_cdf = fitted_cdf
        return self

    def _fit_inv_bary(self, X, y):

        classes = torch.unique(y)
        n_features = X.shape[1]
        self.n_features = n_features
        device = X.device
        X = None  # Only need X for n_features

        if self.bound_eps == 'auto':
            bound_eps = 1 / self.n_grid
        else:
            bound_eps = self.bound_eps

        u_bary = torch.linspace(-0.5, 0.5, self.n_grid) * (1 - bound_eps) + 0.5
        U_bary = torch.outer(u_bary, torch.ones(n_features))
        U_bary = U_bary.to(device)

        X_query_per_class = torch.cat([
            self.fitted_cdf[int(yy)].inverse(U_bary).unsqueeze(0)
            for yy in classes
        ])
        X_bary = torch.mean(X_query_per_class, dim=0)

        bary_normal = GaussianTransformer()
        bary_normal.fit(X_bary)
        X_bary = bary_normal(X_bary)
        self.bary_normal = bary_normal

        # plt.plot(X_bary)

        bin_edges = []
        hists = []
        for xb, ub in zip(X_bary.T, U_bary.T):  # each feature independently
            hists.append(
                torch.cat([torch.Tensor([bound_eps / 2]).to(device), torch.diff(ub),
                           torch.Tensor([bound_eps / 2]).to(device)]).unsqueeze(0))
            bin_edges.append(torch.cat([torch.Tensor([0]).to(device), xb, torch.Tensor([1]).to(device)]).unsqueeze(0))
            # bin_edges.append(torch.cat([torch.min(xb).unsqueeze(0),xb,torch.max(xb).unsqueeze(0)]).unsqueeze(0))
        bin_edges = torch.cat(bin_edges, dim=0)

        hists = torch.cat(hists, dim=0)
        ht = NoGHistogramTransformer(self.n_grid, alpha=0)  # this does not really matter in this case
        ht.fit(X_bary, bin_edges=bin_edges, counts=hists)
        self.fitted_inv_cdf = ht

    def fit(self, x, y):

        assert len(y.shape) == 1, 'check y shape'

        self._fit_cdf(x, y)
        self._fit_inv_bary(x, y)

    def forward(self, x, y):
        z = torch.zeros_like(x)
        classes = torch.unique(y)
        for yy in classes:
            z[y == yy] = self.bary_normal.inverse(
                self.fitted_inv_cdf.inverse(
                    self.fitted_cdf[int(yy)](x[y == yy])))
        return z

    def inverse(self, x, y):
        z = torch.zeros_like(x)
        classes = torch.unique(y)
        for yy in classes:
            z[y == yy] = self.fitted_cdf[int(yy)].inverse(
                self.fitted_inv_cdf(
                    self.bary_normal(x[y == yy])))
        return z


class HistogramTransformer(nn.Module):
    # First normalize the data using
    def __init__(self, hist_bins, alpha=1e-6):
        super().__init__()
        self.hist_bins = hist_bins
        self.alpha = alpha

    def fit(self, x, bin_edges=None, counts=None):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        self.normal = distribution.normal.Normal(mean, std)
        x = self.normal.cdf(x)

        if bin_edges is None and counts is None:
            bin_edges, counts = self._get_hists(x)
        elif bin_edges is not None and counts is not None:
            pass
        else:
            raise Exception('Double check input!')
        hist_rv = Histogram(bin_edges, counts)
        self.hist_rv = hist_rv
        return self

    def forward(self, x):
        x = self.normal.cdf(x)
        x = self.hist_rv.cdf(x)
        return torch.clamp(x, min=1e-5, max=1 - 1e-5)

    def inverse(self, x):
        x = self.hist_rv.icdf(x)
        x = torch.clamp(x, min=1e-5, max=1 - 1e-5)
        x = self.normal.icdf(x)
        return x

    def _get_hists(self, x):
        ndir = x.shape[1]
        bin_edges = []
        counts = []
        for i in range(ndir):
            # torch.histogram does not support cuda
            # hist = torch.histogram(x[:, i], bins=self.hist_bins, range=[0, 1])
            # bin_edges.append(hist.bin_edges.unsqueeze(0))
            # cur_counts = hist.hist.unsqueeze(0)
            # cur_counts += self.alpha
            # counts.append(cur_counts)

            cur_counts = torch.histc(x[:, i], bins=self.hist_bins, min=0, max=1).unsqueeze(0)
            cur_counts += self.alpha
            counts.append(cur_counts)
            cur_bin_edges = torch.linspace(0, 1, self.hist_bins + 1).to(x.device)
            bin_edges.append(cur_bin_edges.unsqueeze(0))
        return torch.cat(bin_edges, dim=0), torch.cat(counts, dim=0)


class NoGHistogramTransformer(nn.Module):
    # First normalize the data using
    def __init__(self, hist_bins, alpha=1e-6):
        super().__init__()
        self.hist_bins = hist_bins
        self.alpha = alpha

    def fit(self, x, bin_edges=None, counts=None):

        if bin_edges is None and counts is None:
            bin_edges, counts = self._get_hists(x)
        elif bin_edges is not None and counts is not None:
            pass
        else:
            raise Exception('Double check input!')
        hist_rv = Histogram(bin_edges, counts)
        self.hist_rv = hist_rv
        return self

    def forward(self, x):
        x = self.hist_rv.cdf(x)
        return torch.clamp(x, min=1e-5, max=1 - 1e-5)

    def inverse(self, x):
        x = self.hist_rv.icdf(x)
        return x

    def _get_hists(self, x):
        ndir = x.shape[1]
        bin_edges = []
        counts = []
        for i in range(ndir):
            # hist = torch.histogram(x[:, i], bins=self.hist_bins)
            # bin_edges.append(hist.bin_edges.unsqueeze(0))
            # cur_counts = hist.hist.unsqueeze(0)
            # cur_counts += self.alpha
            # counts.append(cur_counts)
            cur_counts = torch.histc(x[:, i], bins=self.hist_bins).unsqueeze(0)
            cur_counts += self.alpha
            counts.append(cur_counts)
            cur_bin_edges = torch.linspace(torch.min(x[:, i]), torch.max(x[:, i]), self.hist_bins + 1).to(x.device)
            bin_edges.append(cur_bin_edges.unsqueeze(0))
        return torch.cat(bin_edges, dim=0), torch.cat(counts, dim=0)


class GaussianTransformer(nn.Module):
    # First normalize the data using
    def __init__(self):
        super().__init__()

    def fit(self, x):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        self.normal = distribution.normal.Normal(mean, std)
        return self

    def forward(self, x):
        x = self.normal.cdf(x)
        return torch.clamp(x, min=1e-6, max=1 - 1e-6)

    def inverse(self, x):
        x = self.normal.icdf(x)
        return x

class Histogram(Distribution):
    r"""
    Generates random samples from histogram within the half-open interval
    ``[low, high)``.

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """
    arg_constraints = {}

    # arg_constraints = {'low': constraints.dependent(is_discrete=False, event_dim=0),
    #                   'high': constraints.dependent(is_discrete=False, event_dim=0)}
    # has_rsample = True

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def stddev(self):
        raise NotImplementedError()

    @property
    def variance(self):
        raise NotImplementedError()

    def __init__(self, bin_edges, counts=None, density=None, validate_args=None):
        # bin_edges could be of shape (B1,B2,...) x n_bins, where K is the number of bins
        # For example (n_directions, n_classes, n_components, n_bins) where first 3 are batch shape
        # Barycenter would be reduction over n_components (2nd to last dimension)
        # Inverse would expand dimension by broadcasting over n_components
        #  (expanding on second-to-last dimension)

        bin_widths = bin_edges[..., 1:] - bin_edges[..., :-1]
        if counts is not None and density is None:
            assert bin_edges.shape[-1] == 1 + counts.shape[-1], \
                'Last dimension of bin_edges must be +1 more than last dimension of counts'
            # Pad by 0 to match bin_edges
            # counts = torch.nn.functional.pad(counts, (1, 0), value=0, mode='constant')
            # self.bin_edges, counts = broadcast_all(bin_edges, counts)
        elif counts is None and density is not None:
            assert bin_edges.shape[-1] == 1 + density.shape[-1], \
                'Last dimension of bin_edges must be +1 more than last dimension of counts'
            # Pad by 0 to match bin_edges
            # density = torch.nn.functional.pad(density, (1, 0), value=0, mode='constant')
            # self.bin_edges, density = broadcast_all(bin_edges, density)
            # Compute bin probabilities
            counts = density * bin_widths  # Just uniform density across interval
        else:
            raise ValueError('Either density or counts must be set but not both')

        bin_prob = counts / torch.sum(counts, dim=-1, keepdims=True)

        # Get cumulative sum (i.e., the y values for the piecewise linear function)
        cum_prob = torch.cumsum(bin_prob, dim=-1)
        assert torch.all(torch.isclose(cum_prob[..., -1], torch.ones_like(cum_prob[..., -1]))), \
            'Last value of cumsum should be close to 1'
        cum_prob = cum_prob.clamp(0, 1)
        # Add zero so that it is the same size as bin_edges
        cum_prob = torch.nn.functional.pad(cum_prob, (1, 0), value=0, mode='constant')
        assert cum_prob.shape[-1] == bin_edges.shape[-1], \
            'cum_prob and bin_edges (i.e., the x and y for linear interpolation) should be the same length'

        # Broadcast sizes as necessary for final parameters
        self.bin_edges, self.cum_prob = broadcast_all(bin_edges, cum_prob)

        batch_shape = self.bin_edges.size()[:-1]
        super(Histogram, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args:
            if not torch.all(torch.greater_equal(bin_widths, 0)):
                raise ValueError('Bin edges must be in sorted order')
            if counts is not None and not torch.all(torch.greater_equal(counts, 0)):
                raise ValueError('Counts must be >= 0')
            if density is not None and not torch.all(torch.greater_equal(density, 0)):
                raise ValueError('Density values must be >= 0')

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Histogram, _instance)
        batch_shape = torch.Size(batch_shape)
        # new.low = self.low.expand(batch_shape)
        # new.high = self.high.expand(batch_shape)
        # TODO expand other parameters and recompute hidden variables
        raise NotImplementedError()

        super(Histogram, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.bin_edges[..., 0], self.bin_edges[..., -1])

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()
        # shape = self._extended_shape(sample_shape)
        # rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        # return self.low + rand * (self.high - self.low)

    def _expand_value(self, value):
        # Broadcast to size (B, n_query) where B is all the batch dimensions
        #  e.g., (B, n_directions, n_classes, n_components)
        value = value.expand(-1, *self.bin_edges.shape[:-1])
        return value

    def log_prob(self, value):
        # Run cdf and use autograd to get density
        if self._validate_args:
            self._validate_sample(value)
        value = self._expand_value(value)

        value = value.clone()
        value.requires_grad_(True)
        cdf = self.cdf(value)
        pdf = torch.autograd.grad(cdf, value, torch.ones_like(cdf), retain_graph=True)[0]
        return torch.log(pdf)

    def _get_interp_inputs(self, value):

        # Change view of input to be D-N arrays where D is the like a batch size
        #  and N is the number of bins for vectorizing this call
        n_bins = self.bin_edges.shape[-1]
        x = self.bin_edges.reshape(-1, n_bins)
        y = self.cum_prob.reshape(-1, n_bins)
        # Should be prod(batch_shape) x batch_size so first collapse and then transpose
        xnew = value.reshape(-1, torch.prod(torch.tensor(self._batch_shape))).t()
        return x, y, xnew

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = self._expand_value(value)
        x, y, xnew = self._get_interp_inputs(value)
        out = Interp1d()(x, y, xnew)
        return out.t().reshape(value.shape)  # Transpose and return to correct shape

    def icdf(self, value):
        if self._validate_args:
            assert torch.all(torch.greater_equal(value, 0)) and torch.all(torch.less_equal(value, 1))
        value = self._expand_value(value)
        x, y, ynew = self._get_interp_inputs(value)
        out = Interp1d()(y, x, ynew)
        return out.t().reshape(value.shape)  # Transpose and return to correct shape

    def entropy(self):
        raise NotImplementedError()

class Interp1d(torch.autograd.Function):
    # Copied from https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py
    #
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.

        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)