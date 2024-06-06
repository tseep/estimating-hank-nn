import torch


# %% Layer to normalize the inputs
class NormalizeLayer(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound):
        super(NormalizeLayer, self).__init__()

        # Register the lower bound and upper bound as buffers
        self.register_buffer("lower_bound", lower_bound)
        self.register_buffer("upper_bound", upper_bound)

    def forward(self, x):
        return 2 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1
