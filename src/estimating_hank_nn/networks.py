"""Neural network layers: input normalization and output scaling."""

import torch


# %% NormalizeLayer
class NormalizeLayer(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound):
        super(NormalizeLayer, self).__init__()

        # Register the lower bound and upper bound as buffers
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

    def forward(self, x):
        return 2 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1


# %% ScaleLayer
class ScaleLayer(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound):
        super(ScaleLayer, self).__init__()

        # Register the lower bound and upper bound as buffers
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

    def forward(self, x):
        # Scale the output x [-1, 1] to the range [lower_bound, upper_bound]
        ub = self.upper_bound
        lb = self.lower_bound

        return torch.where(ub == lb, x, (x + 1) * (ub - lb) / 2 + lb)
