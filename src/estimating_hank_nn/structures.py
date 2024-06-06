import torch
from torch.distributions import constraints
from typing import Dict


# %% Base class for the elements of the model
class Element(object):
    def __init__(self, dict: Dict) -> None:
        for key, value in dict.items():
            if isinstance(value, float):
                value = torch.tensor([value])
            setattr(self, key, value)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        output = ""
        for key, value in self.__dict__.items():
            output += f"{key}: {value}\n"
        return output

    def update(self, dict: Dict):
        for key, value in dict.items():
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

    def cat(self):
        return torch.cat(self.values(), dim=-1)

    def to(self, device):
        for key, value in self.__dict__.items():
            setattr(self, key, value.to(device))
        return self

    def requires_grad(self, requires_grad=True):
        for value in self.__dict__.values():
            value.requires_grad = requires_grad
        return self

    def zero_grad(self):
        for value in self.__dict__.values():
            value.grad = None
        return self


# %% Class for the parameters
class Parameters(Element):
    def expand(self, shape):
        return Parameters({key: value.expand(shape) for key, value in self.__dict__.items()})


# %% Identity class for fixed parameters
class Identity(object):
    def __init__(self, value) -> None:
        self.value = torch.tensor(value)

    def __str__(self):
        return f"Identity({str(self.value.item())})"

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


# %% Class for the priors
class Ranges(Element):
    def __init__(self, par_dict: Dict, priors_dict: Dict) -> None:
        for key, value in par_dict.items():
            if key in priors_dict:
                value = priors_dict[key]
            else:
                value = Identity(value)
            setattr(self, key, value)

    def cat(self):
        pass

    def limits(self):
        return {key: value.support for key, value in self.__dict__.items()}

    def low_tensor(self):
        return torch.tensor([value.low for value in self.__dict__.values()])

    def high_tensor(self):
        return torch.tensor([value.high for value in self.__dict__.values()])

    def sample(self, shape, device="cpu"):
        par_draw = {key: value.sample(shape).to(device) for key, value in self.__dict__.items()}
        return Parameters(par_draw)


# %% Class for the state
class State(Element):
    pass


# %% Class for the shocks
class Shocks(Element):
    def sample(self, shape, antithetic=False, device="cpu"):
        if antithetic:
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
