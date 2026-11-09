"""Diagnostics for the two-asset Krusell-Smith model: simulate the trained networks and return
plot-ready arrays. Companion plotting lives in ``kstwo_plots``."""

import numpy as np
import torch
from tqdm import trange

from estimating_hank_nn.structures import State
from estimating_hank_nn.helpers import seed_all


# --- Policy diagnostics ---

@torch.no_grad()
def policy_over_state(model, s_values, n=100, batch=100, burn=1000, device="cpu", seed=0):
    """DSS consumption, next-period capital and bond choice over the capital grid for a household
    that holds the bond supply ``B``, one curve per log productivity in ``s_values``, evaluated
    against a simulated DSS panel and averaged over the batch. Returns
    ``{k_grid, s_grid, c, k_next, b_next}``."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    model.dss_steps(batch=batch, device=device, steps=burn)

    state_dist, state_agg, par, ss = model.state_dist, model.state_agg, model.par_draw, model.ss
    k_grid = torch.linspace(model.par.k_bar.item() - 0.1, model.par.k_down.item() + 0.1, n, device=device)

    # Prices implied by the simulated distribution, as in dss_step.
    A_dss_scale, Q_dss = model.dss_policy(par)
    Kl = state_dist.k.mean(dim=-2, keepdim=True)
    N = ss.N * torch.exp(state_dist.s).mean(dim=-2, keepdim=True)
    _, r, w = model.factor_prices(A_dss_scale * ss.A, Kl, N, par)
    T = (1 - Q_dss) * par.B

    c_curves, k_curves, b_curves = [], [], []
    for s_val in s_values:
        # The capital grid enters as the agent dimension and broadcasts against the batch.
        state_ind = State({"k": k_grid.view(1, n, 1),
                           "b": torch.full((1, n, 1), model.par.B.item(), device=device),
                           "s": torch.full((1, n, 1), float(s_val), device=device)})
        c_scale, b_scale = model.dss_hh_policy(state_ind, state_dist, state_agg, par)
        c_ind, b_ind = c_scale * ss.C, b_scale * par.B
        k_next = (1 + r) * state_ind.k + state_ind.b + w * torch.exp(state_ind.s) * ss.N - Q_dss * b_ind - T - c_ind
        c_curves.append(c_ind.mean(dim=0).flatten().cpu().numpy())
        k_curves.append(k_next.mean(dim=0).flatten().cpu().numpy())
        b_curves.append(b_ind.mean(dim=0).flatten().cpu().numpy())

    return {"k_grid": k_grid.cpu().numpy(), "s_grid": np.asarray(s_values), "c": np.asarray(c_curves),
            "k_next": np.asarray(k_curves), "b_next": np.asarray(b_curves),
            "A_dss": (A_dss_scale * ss.A).mean().item(), "Q_dss": Q_dss.mean().item()}


@torch.no_grad()
def policy_surface(model, s_quantile=0.5, n_k=40, n_b=40, batch=100, burn=1000, device="cpu", seed=0):
    """DSS consumption over capital and bonds at the ``s_quantile`` quantile of simulated log
    productivity, evaluated against a simulated DSS panel and averaged over the batch. Returns
    ``{k_grid, b_grid, c, s_val, s_quantile}`` with ``c`` of shape (n_k, n_b)."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    model.dss_steps(batch=batch, device=device, steps=burn)

    state_dist, state_agg, par, ss = model.state_dist, model.state_agg, model.par_draw, model.ss
    s_val = torch.quantile(state_dist.s.flatten(), s_quantile).item()
    k_grid = torch.linspace(model.par.k_bar.item() - 0.1, model.par.k_down.item() + 0.1, n_k)
    b_grid = torch.linspace(model.par.b_bar.item() - 0.1, model.par.b_down.item() + 0.1, n_b)

    c_surface = np.zeros((n_k, n_b))
    for i, k_val in enumerate(k_grid):
        # The bond grid enters as the agent dimension and broadcasts against the batch.
        state_ind = State({
            "k": torch.full((1, n_b, 1), k_val.item(), device=device),
            "b": b_grid.to(device).view(1, n_b, 1),
            "s": torch.full((1, n_b, 1), s_val, device=device),
        })
        c_scale, _ = model.dss_hh_policy(state_ind, state_dist, state_agg, par)
        c_surface[i, :] = (c_scale * ss.C).mean(dim=0).squeeze().cpu().numpy()

    return {"k_grid": k_grid.numpy(), "b_grid": b_grid.numpy(), "c": c_surface,
            "s_val": s_val, "s_quantile": s_quantile}


@torch.no_grad()
def asset_histogram(model, which="full", batch=1000, burn=1000, amax=40.0, bmax=2.0, n_bins=50,
                    device="cpu", seed=0):
    """Simulated cross-sectional capital and bond distributions over ``[0, amax]`` and
    ``[0, bmax]`` without aggregate shocks; holdings below zero are clamped to 0 and holdings
    above the range are left out. ``share_above`` is the share of holdings above the caps
    ``k_down`` / ``b_down``."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    if which == "full":
        model.steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=0.0)
    else:
        model.dss_steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=0.0)

    state_dist = model.state_dist
    share_above = {"k": (state_dist.k > model.par.k_down.item()).float().mean().item(),
                   "b": (state_dist.b > model.par.b_down.item()).float().mean().item()}

    k = torch.clamp(state_dist.k.cpu().flatten(), min=0.0).numpy()
    b = torch.clamp(state_dist.b.cpu().flatten(), min=0.0).numpy()
    density_k, edges_k = np.histogram(k[k <= amax], bins=n_bins, range=(0.0, amax), density=True)
    density_b, edges_b = np.histogram(b[b <= bmax], bins=n_bins, range=(0.0, bmax), density=True)
    return {"edges_k": edges_k, "density_k": density_k, "edges_b": edges_b, "density_b": density_b,
            "share_above": share_above}


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
