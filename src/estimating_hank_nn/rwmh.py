"""Random-walk Metropolis-Hastings using the surrogate likelihood network."""

import torch
from tqdm import trange

from .helpers import seed_all


# %% RWMH
def rwmh(pf, priors, steps, param_start, cov_estimable, device="cpu", tune=0.3, seed=None, print_fn=print):
    """Random Walk Metropolis-Hastings using the surrogate likelihood network.

    Only estimable parameters (those with log_prob in priors) are proposed;
    fixed parameters stay at their starting values.

    Args:
        pf: ParticleFilter with a trained surrogate network.
        priors: Priors object mapping each parameter to a distribution or Fixed.
        steps: Total MCMC steps (caller handles burn-in slicing).
        param_start: 1D tensor of starting values for ALL parameters.
        cov_estimable: Covariance matrix for estimable parameters only.
        device: Computation device.
        tune: Scaling factor for proposal covariance.
        seed: Random seed.
        print_fn: Logging function.

    Returns:
        draws: (steps, n_estimable) tensor of parameter draws.
        posterior_lls: (steps,) tensor of posterior log-likelihoods.
    """
    seed_all(seed)

    # Classify parameters: estimable (have a prior with log_prob) vs fixed.
    all_keys = list(priors.keys())
    estimable_keys = [k for k in all_keys if hasattr(getattr(priors, k), "log_prob")]
    estimable_idx = [all_keys.index(k) for k in estimable_keys]
    n_est = len(estimable_keys)

    if pf.network is not None:
        pf.network.to(device)

    # MCMC state is the full parameter vector; fixed slots stay at their
    # `param_start` values, only `estimable_idx` slots are proposed each step.
    param_old = param_start.clone().to(device)
    cov_estimable = cov_estimable.to(device)

    # Initial posterior.
    with torch.no_grad():
        model_ll_old = pf.network(param_old).squeeze()
    prior_ll_old = priors.log_prob(param_old)
    posterior_ll_old = model_ll_old + prior_ll_old
    assert torch.isfinite(posterior_ll_old), f"non-finite initial posterior ({posterior_ll_old.item()})"

    draws = torch.empty(steps, n_est)
    posterior_lls = torch.empty(steps)
    accept_count = 0

    pbar = trange(steps)
    for step in pbar:
        # Propose new estimable values; fixed slots are copied from param_old.
        proposal_dist = torch.distributions.MultivariateNormal(
            param_old[estimable_idx], tune * cov_estimable
        )
        param_new = param_old.clone()
        param_new[estimable_idx] = proposal_dist.sample()

        # Posterior at the proposal.
        with torch.no_grad():
            model_ll_new = pf.network(param_new).squeeze()
        prior_ll_new = priors.log_prob(param_new)
        posterior_ll_new = model_ll_new + prior_ll_new

        # Metropolis-Hastings acceptance.
        accept_prob = torch.exp(posterior_ll_new - posterior_ll_old)
        accept_ratio = 0.0 if torch.isnan(accept_prob) else accept_prob.item()   # NaN rejects
        if torch.rand(1).item() < min(1.0, accept_ratio):
            param_old = param_new
            posterior_ll_old = posterior_ll_new
            accept_count += 1

        draws[step] = param_old[estimable_idx].cpu()
        posterior_lls[step] = posterior_ll_old.cpu()

        if (step + 1) % 100 == 0:
            rate = accept_count / (step + 1) * 100
            pbar.set_postfix({"accept%": f"{rate:.1f}"}, refresh=False)

    rate = accept_count / steps * 100
    print_fn(f"RWMH acceptance rate: {rate:.1f}%")

    return draws, posterior_lls
