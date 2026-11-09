"""Data structures for model elements: parameters, states, shocks, priors, and ranges."""

import torch
from torch.distributions import constraints
from typing import Dict


# %% Element base class
class Element:
    """Base container that stores named tensors as attributes.

    Supports dict-like access (keys/values/items/in), attribute access (par.sigma),
    and common tensor operations (cat, to, clone, etc.).
    """
    def __init__(self, data: Dict) -> None:
        for key, value in data.items():
            if isinstance(value, float):
                value = torch.tensor([value])
            setattr(self, key, value)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, Element):
            return NotImplemented
        if self.keys() != other.keys():
            return False
        return all(torch.equal(self[k], other[k]) for k in self.keys())

    def __str__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                lines.append(f"  {key}: {value.item():.6g}" if value.numel() == 1
                             else f"  {key}: tensor{tuple(value.shape)}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def update(self, data: Dict):
        for key, value in data.items():
            setattr(self, key, value)

    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def items(self):
        return list(self.__dict__.items())

    # --- Tensor operations ---

    def cat(self, keys=None):
        if keys is None:
            keys = self.keys()
        return torch.cat([self[key] for key in keys], dim=-1)

    def view(self, *shape):
        return self.__class__({key: value.view(*shape) for key, value in self.__dict__.items()})

    def unsqueeze(self, dim):
        for key, value in self.__dict__.items():
            setattr(self, key, value.unsqueeze(dim))
        return self

    def expand_dim(self, dim, size):
        new = {}
        for key, value in self.__dict__.items():
            expanded = value.expand(value.shape[:dim] + (size,) + value.shape[dim + 1 :])
            new[key] = expanded
        return self.__class__(new)

    def to(self, device):
        for key, value in self.__dict__.items():
            if callable(getattr(value, "to", None)):
                setattr(self, key, value.to(device))
        return self

    # --- Gradient operations (in place, mainly for interactive debugging) ---

    def requires_grad_(self, requires_grad=True):
        for value in self.__dict__.values():
            value.requires_grad_(requires_grad)
        return self

    def zero_grad(self):
        for value in self.__dict__.values():
            value.grad = None
        return self

    def detach_(self):
        for value in self.__dict__.values():
            value.detach_()
        return self

    def clone(self):
        return self.__class__({key: value.clone() for key, value in self.__dict__.items()})

    def reindex(self, idx):
        """Return a new container with all tensors indexed along dim 0 by *idx*."""
        return self.__class__({key: value[idx, ...] for key, value in self.__dict__.items()})

    @property
    def device(self):
        for value in self.__dict__.values():
            if hasattr(value, "device"):
                return value.device
        return None


# %% Parameters
class Parameters(Element):
    """Named parameter container with expand support."""

    def expand(self, shape):
        return Parameters({key: value.expand(shape) for key, value in self.__dict__.items()})


# %% Ranges
class Ranges(Element):
    """Parameter ranges for training: maps each parameter to a sampling distribution or Identity."""
    def __init__(self, par_dict: Dict, priors_dict: Dict) -> None:
        for key, value in par_dict.items():
            if key in priors_dict:
                value = priors_dict[key]
            else:
                value = Identity(value)
            setattr(self, key, value)

    def keys_estimable(self):
        return [key for key, value in self.__dict__.items() if not isinstance(value, Identity)]

    def limits(self):
        return {key: value.support for key, value in self.__dict__.items()}

    def limits_estimable(self):
        return {key: value.support for key, value in self.__dict__.items() if not isinstance(value, Identity)}

    def low_tensor(self, keys=None):
        if keys is None:
            keys = self.keys()
        return torch.tensor([self[key].low for key in keys])

    def high_tensor(self, keys=None):
        if keys is None:
            keys = self.keys()
        return torch.tensor([self[key].high for key in keys])

    def sample(self, shape, device="cpu"):
        par_draw = {key: value.sample(shape).to(device) for key, value in self.__dict__.items()}
        return Parameters(par_draw)


# %% Identity
class Identity:
    """Non-estimable parameter: always returns the same fixed value.

    Used by Ranges to mark parameters that are held constant during estimation.
    """
    def __init__(self, value) -> None:
        self.value = torch.tensor(value)

    def __str__(self):
        return f"Identity({self.value.item()})"

    def sample(self, shape):
        return self.value.expand(shape)

    @property
    def support(self):
        return constraints.interval(torch.tensor(-1.0), torch.tensor(1.0))

    @property
    def low(self):
        return self.support.lower_bound

    @property
    def high(self):
        return self.support.upper_bound


# %% Fixed
class Fixed:
    """Fixed parameter value (used by Priors for non-estimated parameters)."""
    def __init__(self, value):
        self.value = torch.as_tensor(value)

    def __str__(self):
        return f"Fixed({self.value.item()})"

    def sample(self, shape):
        return self.value.expand(shape)

    @property
    def low(self):
        return self.value

    @property
    def high(self):
        return self.value

    @property
    def support(self):
        return constraints.interval(self.value, self.value)


# %% Priors
class Priors(Element):
    """Prior distributions for Bayesian estimation (RWMH).

    Maps each parameter to a prior distribution or Fixed.
    Provides joint log-prior evaluation over a parameter tensor.
    """
    def __init__(self, par_dict: Dict, priors_dict: Dict) -> None:
        for key, value in par_dict.items():
            if key in priors_dict:
                value = priors_dict[key]
            else:
                value = Fixed(value)
            setattr(self, key, value)

    def sample(self, shape, device="cpu"):
        par_draw = {key: value.sample(shape).to(device) for key, value in self.__dict__.items()}
        return Parameters(par_draw)

    def log_prob(self, param_tensor):
        """Compute the joint log prior for a parameter tensor.

        Ignores parameters without a log_prob method (e.g., Fixed).
        Assumes param_tensor ordering matches self.keys().

        Args:
            param_tensor: 1D tensor of parameter values (same order as self.keys()).

        Returns:
            Scalar log prior probability.
        """
        logp = 0.0
        for i, name in enumerate(self.keys()):
            prior_dist = getattr(self, name)
            if hasattr(prior_dist, "log_prob"):
                logp += prior_dist.log_prob(param_tensor[i])
        return logp


# %% State
class State(Element):
    """Named state container with sort and permute support."""

    def sort(self, key, dim=-1, descending=False):
        """Return a new State sorted along dim by the values of key."""
        _, idx = torch.sort(self.get(key), dim=dim, descending=descending, stable=True)
        sorted_dict = {}
        for key, value in self.__dict__.items():
            sorted_dict[key] = torch.gather(value, dim=dim, index=idx)
        return self.__class__(sorted_dict)

    def permute(self, dims):
        """Return a new State with all tensors permuted according to dims."""
        permuted_dict = {}
        for key, value in self.__dict__.items():
            permuted_dict[key] = value.permute(dims).contiguous()
        return self.__class__(permuted_dict)


# %% Shocks
class Shocks(Element):
    """Named shock distribution container with sampling support."""
    def sample(self, shape, antithetic=False, device="cpu"):
        if antithetic:
            assert shape[0] % 2 == 0, "antithetic sampling needs an even leading dimension"
            shape_antithetic = (shape[0] // 2, *shape[1:])
            shock_draw = {}
            for key, value in self.__dict__.items():
                sample = value.sample(shape_antithetic) - value.mean
                shock_draw[key] = torch.cat([value.mean + sample, value.mean - sample], dim=0).to(device)
        else:
            shock_draw = {}
            for key, value in self.__dict__.items():
                shock_draw[key] = value.sample(shape).to(device)
        return Element(shock_draw)
