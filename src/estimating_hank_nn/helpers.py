import torch


# %% Function to count parameters of a neural network and visualize the architecture
def count_parameters(model):
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


# %% Function to calculate the ergodic standard deviation of the AR(1) shock process
def ergodic_sigma(rho, sigma):
    return (sigma) / (1.0 - rho**2) ** 0.5
