"""Diagnostics for the one-asset Krusell-Smith model: simulate the trained networks and return
plot-ready arrays. Companion plotting lives in ``ksone_plots``."""

import numpy as np
import torch
from tqdm import trange

from estimating_hank_nn.structures import State
from estimating_hank_nn.helpers import seed_all


# --- Policy diagnostics ---

@torch.no_grad()
def policy_over_state(model, s_values, n=100, batch=100, burn=1000, device="cpu", seed=0):
    """DSS consumption and next-period capital over the capital grid, one curve per log
    productivity in ``s_values``, evaluated against a simulated DSS panel and averaged over the
    batch. Returns ``{k_grid, s_grid, c, k_next}``."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    model.dss_steps(batch=batch, device=device, steps=burn)

    state_dist, state_agg, par, ss = model.state_dist, model.state_agg, model.par_draw, model.ss
    k_grid = torch.linspace(model.par.k_bar.item() - 0.1, model.par.k_down.item() + 0.1, n, device=device)

    # Factor prices implied by the simulated distribution, as in dss_step.
    A_dss = model.dss_policy(par) * ss.A
    Kl = state_dist.k.mean(dim=-2, keepdim=True)
    N = ss.N * torch.exp(state_dist.s).mean(dim=-2, keepdim=True)
    _, r, w = model.factor_prices(A_dss, Kl, N, par)

    c_curves, k_curves = [], []
    for s_val in s_values:
        # The capital grid enters as the agent dimension and broadcasts against the batch.
        state_ind = State({"k": k_grid.view(1, n, 1), "s": torch.full((1, n, 1), float(s_val), device=device)})
        c_ind = model.dss_hh_policy(state_ind, state_dist, state_agg, par) * ss.C
        k_next = (1 + r) * state_ind.k + w * torch.exp(state_ind.s) * ss.N - c_ind
        c_curves.append(c_ind.mean(dim=0).flatten().cpu().numpy())
        k_curves.append(k_next.mean(dim=0).flatten().cpu().numpy())

    return {"k_grid": k_grid.cpu().numpy(), "s_grid": np.asarray(s_values),
            "c": np.asarray(c_curves), "k_next": np.asarray(k_curves), "A_dss": A_dss.mean().item()}


@torch.no_grad()
def policy_by_income(model, percentiles=(0.1, 0.5, 0.9), n=100, k_max=20.0, batch=100, burn=1000,
                     income_band=0.1, n_bins=40, device="cpu", seed=0):
    """DSS consumption and next-period capital over the capital grid up to ``k_max`` at percentiles
    of the simulated log productivity, each paired with the capital density of the households
    within ``income_band`` (in rank) of that productivity level. Returns
    ``{k_grid, centers, edges, percentiles, s_values, groups}`` with one ``{c, k_next, density}``
    per percentile."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    model.dss_steps(batch=batch, device=device, steps=burn)

    state_dist, state_agg, par, ss = model.state_dist, model.state_agg, model.par_draw, model.ss
    s_all, k_all = state_dist.s.flatten(), state_dist.k.flatten()
    k_low = model.par.k_bar.item() - 0.1
    k_grid = torch.linspace(k_low, k_max, n, device=device)
    edges = np.linspace(0.0, k_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Factor prices implied by the simulated distribution, as in dss_step.
    A_dss = model.dss_policy(par) * ss.A
    Kl = state_dist.k.mean(dim=-2, keepdim=True)
    N = ss.N * torch.exp(state_dist.s).mean(dim=-2, keepdim=True)
    _, r, w = model.factor_prices(A_dss, Kl, N, par)

    s_values, groups = [], []
    for q in percentiles:
        s_val = float(s_all.quantile(q))
        state_ind = State({"k": k_grid.view(1, n, 1), "s": torch.full((1, n, 1), s_val, device=device)})
        c_ind = model.dss_hh_policy(state_ind, state_dist, state_agg, par) * ss.C
        k_next = (1 + r) * state_ind.k + w * torch.exp(state_ind.s) * ss.N - c_ind
        s_lo = float(s_all.quantile(max(0.0, q - income_band)))
        s_hi = float(s_all.quantile(min(1.0, q + income_band)))
        cond = k_all[(s_all >= s_lo) & (s_all <= s_hi)].clamp(min=0.0).cpu().numpy()
        s_values.append(s_val)
        groups.append({"c": c_ind.mean(dim=0).flatten().cpu().numpy(),
                       "k_next": k_next.mean(dim=0).flatten().cpu().numpy(),
                       "density": np.histogram(cond, bins=edges, density=True)[0]})

    return {"k_grid": k_grid.cpu().numpy(), "centers": centers, "edges": edges,
            "percentiles": list(percentiles), "s_values": s_values, "groups": groups}


@torch.no_grad()
def asset_histogram(model, which="full", batch=1000, burn=1000, amax=40.0, n_bins=50, device="cpu", seed=0):
    """Simulated cross-sectional capital distribution over ``[0, amax]`` without aggregate shocks;
    capital below zero is clamped to 0 and capital above ``amax`` is left out."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    if which == "full":
        model.steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=0.0)
    else:
        model.dss_steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=0.0)

    k = torch.clamp(model.state_dist.k.cpu().flatten(), min=0.0).numpy()
    density, edges = np.histogram(k[k <= amax], bins=n_bins, range=(0.0, amax), density=True)
    return {"edges": edges, "density": density, "cdf": np.cumsum(density * np.diff(edges))}


# --- Accuracy diagnostics ---

@torch.no_grad()
def residual_accuracy(model, which="full", batch=100, mc=1000, burn=1000, sim=1000, device="cpu", seed=0):
    """Absolute equilibrium residuals along a simulated path, one array per residual. Every
    economy and household is its own sample."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    if which == "full":
        step_fn, residual_fn, scale_agg = model.step, model.residuals, 1.0
    else:
        step_fn, residual_fn, scale_agg = model.dss_step, model.dss_residuals, 0.0

    for _ in trange(burn, desc="Burn-in (errors)"):
        e = model.draw_shocks(batch=batch, mc=None, antithetic=False, device=device)
        model.set_state(step_fn(e, scale_ind=1.0, scale_agg=scale_agg))

    names = model.RESIDUAL_NAMES[which]
    errors = {name: [] for name in names}
    for _ in trange(sim, desc="Residual errors"):
        e = model.draw_shocks(batch=batch, mc=mc, antithetic=False, device=device)
        residuals = residual_fn(e=e, mc=mc, detach_next=True, scale_ind=1.0, scale_agg=scale_agg)
        for name, r in zip(names, residuals):
            errors[name].append(r.abs().cpu().numpy().reshape(-1))
        e_step = model.draw_shocks(batch=batch, mc=None, antithetic=False, device=device)
        model.set_state(step_fn(e_step, scale_ind=1.0, scale_agg=scale_agg))

    return {name: np.stack(errs) for name, errs in errors.items()}


# --- Impulse responses ---

@torch.no_grad()
def irf(model, shock_name="A_scale", shock_value=-1.0, batch=1000, burn=1000, sim=100, device="cpu", seed=0):
    """Impulse response to a one-time ``shock_name`` innovation with common random numbers: after
    a burn-in without aggregate shocks, two copies of the economy share the idiosyncratic draws,
    one receives the innovation, and the response is the difference of their paths averaged over
    the batch. Period 0 is the period before the shock moves productivity."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)

    for _ in trange(burn, desc="Burn-in (IRF)"):
        e = model.draw_shocks(batch=batch, mc=None, antithetic=False, device=device)
        for k in e["shocks_agg"].keys():
            e["shocks_agg"][k].zero_()
        model.set_state(model.step(e, scale_ind=1.0, scale_agg=0.0))

    state_shock = {"state_ind": model.state_ind, "state_dist": model.state_dist, "state_agg": model.state_agg}
    state_base = dict(state_shock)
    shock_seq = [shock_value] + [0.0] * (sim - 1)

    paths = None
    for i in trange(sim, desc="IRF"):
        e = model.draw_shocks(batch=batch, mc=None, antithetic=False, device=device)
        for k in e["shocks_agg"].keys():
            e["shocks_agg"][k].zero_()

        e["shocks_agg"][shock_name] = torch.full_like(e["shocks_agg"][shock_name], shock_seq[i])
        model.set_state(state_shock)
        state_shock = model.step(e, scale_ind=1.0, scale_agg=1.0)
        sim_shock = dict(model.sim)

        e["shocks_agg"][shock_name].zero_()
        model.set_state(state_base)
        state_base = model.step(e, scale_ind=1.0, scale_agg=1.0)
        sim_base = dict(model.sim)

        if paths is None:
            paths = {k: [] for k in sim_shock}
        for k in paths:
            paths[k].append((sim_shock[k] - sim_base[k]).mean().item())

    return {k: np.asarray(v) for k, v in paths.items()}


# --- Summary numbers for the log ---

def mean_log10_errors(errors, log_min=-8, log_max=0):
    """Mean of the log10 absolute residuals, clipped as in the error histograms, per residual."""
    return {name: float(np.clip(np.log10(np.abs(e) + 1e-12), log_min, log_max).mean()) for name, e in errors.items()}
