"""Helper functions: seeding, ergodic standard deviation, truncated prior, and measurement error."""

import random

import numpy as np
import torch
from torch.distributions import constraints


# %% Seeding
def seed_all(seed):
    """Seed torch (CPU + all CUDA), numpy, and python random. No-op if None."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# %% Ergodic standard deviation
def ergodic_sigma(rho, sigma):
    return (sigma) / (1.0 - rho**2) ** 0.5


# %% Fisher-Burmeister NCP function
def fisher_burmeister(a, b, w=1.0):
    """Fisher-Burmeister nonlinear complementarity function.

    ``FB(a, b) = a + w*b - sqrt(a**2 + (w*b)**2)``. The root ``FB(a, b) = 0`` is
    equivalent to the complementarity conditions ``a >= 0``, ``b >= 0``, ``a*b = 0``.
    Used to encode the household borrowing constraint as a smooth residual.
    """
    b = w * b
    return a + b - (a**2 + b**2) ** 0.5


class TruncatedDistribution:
    """Truncated version of a base distribution with inverse-CDF sampling."""
    def __init__(self, base_dist, low, high):
        self.base_dist = base_dist
        self._lower_bound = torch.tensor(low, dtype=torch.float32)
        self._upper_bound = torch.tensor(high, dtype=torch.float32)

    def sample(self, shape=torch.Size([])):
        cdf_low = self.base_dist.cdf(self._lower_bound)
        cdf_high = self.base_dist.cdf(self._upper_bound)
        u = torch.rand(shape) * (cdf_high - cdf_low) + cdf_low
        # cdf(bound) can under/overflow to exactly 0/1 in float32, where icdf returns +-inf;
        # the exact inverse-CDF value there is the bound itself, so clamp to it.
        return self.base_dist.icdf(u).clamp(self._lower_bound, self._upper_bound)

    def log_prob(self, x):
        in_bounds = (x >= self._lower_bound) & (x <= self._upper_bound)
        norm = self.base_dist.cdf(self._upper_bound) - self.base_dist.cdf(self._lower_bound)
        logp = self.base_dist.log_prob(x) - torch.log(norm)
        logp = torch.where(in_bounds, logp, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype))
        return logp

    def prob(self, x):
        in_bounds = (x >= self._lower_bound) & (x <= self._upper_bound)
        norm = self.base_dist.cdf(self._upper_bound) - self.base_dist.cdf(self._lower_bound)
        p = self.base_dist.log_prob(x).exp() / norm
        p = torch.where(in_bounds, p, torch.tensor(0.0, device=x.device, dtype=x.dtype))
        return p

    @property
    def support(self):
        return constraints.interval(self._lower_bound, self._upper_bound)

    @property
    def low(self):
        return self._lower_bound

    @property
    def high(self):
        return self._upper_bound

    def __str__(self):
        return f"Truncated({self.base_dist}, low: {self._lower_bound.item():.6g}, high: {self._upper_bound.item():.6g})"

    def __repr__(self):
        return self.__str__()


# %% Measurement error
def covariance_matrix(data):
    """Compute diagonal covariance matrix from dict of time series."""
    R = torch.zeros((len(data), len(data)))
    for i, (key, value) in enumerate(data.items()):
        R[i, i] = value.var()
    return R


def add_measurement_error(data, error_share=0.1, seed=None, me_shocks=None):
    """Add Gaussian measurement error to simulation data.

    If ``me_shocks`` is given (dict of standardized innovations per observable),
    use it instead of fresh ``torch.randn`` draws.

    Returns:
        (data_with_error, R) tuple where R is the error covariance matrix.
    """
    if seed is not None and me_shocks is None:
        torch.manual_seed(seed)

    R = covariance_matrix(data) * error_share

    me = {}
    for i, (key, value) in enumerate(data.items()):
        if me_shocks is not None:
            eps = torch.as_tensor(me_shocks[key], dtype=value.dtype, device=value.device)
            me[key] = eps.view_as(value) * torch.sqrt(R[i, i])
        else:
            me[key] = torch.randn_like(value) * torch.sqrt(R[i, i])

    data_me = {}
    for key, value in data.items():
        data_me[key] = value + me[key]

    return data_me, R
