"""Diagnostics for the one-asset HANK model: simulate the trained networks and return plot-ready
arrays. Each function takes the ``OneAssetHANK`` instance and never trains or alters parameters
(``asset_choice`` additionally reuses the model's ``_bond_tax`` helper, to stay identical to the
budget block in ``residuals``). Companion plotting lives in ``hankmp_plots``.

Analysis contract (what the model must expose): ``to``, ``initialize``, ``steps``/``dss_steps``,
``step``, ``draw_shocks``, the policies ``hh_policy``/``dss_hh_policy``/``agg_policy``/``dss_policy``,
``residuals``/``dss_residuals``; attributes ``state_ind``/``state_dist``/``state_agg``,
``par``/``par_draw``, ``ss``, ``range``, ``RESIDUAL_NAMES``.
"""

import numpy as np
import torch

from estimating_hank_nn.structures import State
from estimating_hank_nn.helpers import seed_all


# --- Shared simulation + curve helpers ---

def _simulate_panel(model, which, batch, burn, device):
    """Simulate the ``which`` panel; return (state_dist, state_agg, par, ss, hh_policy, W)."""
    model.to(device)
    model.initialize(batch=batch, device=device)
    if which == "full":
        model.steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=1.0)
        hh_policy = model.hh_policy
        _, W = model.agg_policy(state_dist=model.state_dist, state_agg=model.state_agg, par=model.par_draw)
        W = W.mean()
    else:
        model.dss_steps(batch=batch, device=device, steps=burn, scale_ind=1.0)
        hh_policy = model.dss_hh_policy
        W = model.ss.W.mean()
    return model.state_dist, model.state_agg, model.par_draw, model.ss, hh_policy, W


def _hh_curves(hh_policy, ctx, W, s_values, b_grid, device):
    """Labor/consumption curves over ``b_grid``, one row per income state in ``s_values``."""
    state_dist, state_agg, par, ss = ctx
    n_curves, c_curves = [], []
    for s_val in s_values:
        n_row, c_row = [], []
        for b_val in b_grid:
            state_ind = State({
                "b": torch.full((1, 1, 1), float(b_val), device=device),
                "s": torch.full((1, 1, 1), float(s_val), device=device),
            })
            n_ind = hh_policy(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
            c_ind = ((W * float(np.exp(s_val))) / (ss.chi * n_ind ** par.eta)) ** (1 / par.gamma)
            n_row.append(n_ind.mean().item())
            c_row.append(c_ind.mean().item())
        n_curves.append(n_row)
        c_curves.append(c_row)
    return np.asarray(n_curves), np.asarray(c_curves)


# --- Policy diagnostics ---

@torch.no_grad()
def policy_over_par(model, n=100, device="cpu"):
    """DSS predictions (Rn_dss, Y_dss) as each estimable parameter sweeps its training range, the
    others held at base. Returns ``{key: {"grid", "Rn_dss", "Y_dss"}}``."""
    model.to(device)
    out = {}
    for key in model.range.keys_estimable():
        r = model.range.get(key)
        grid = torch.linspace(float(r.low), float(r.high), n, device=device)
        par = model.par.view(1, 1, 1).expand([n, 1, 1]).to(device)
        par.set(key, grid.view(n, 1, 1))
        Rn_dss, Y_dss = model.dss_policy(par=par)
        out[key] = {"grid": grid.cpu().numpy(),
                    "Rn_dss": Rn_dss.reshape(-1).cpu().numpy(),
                    "Y_dss": Y_dss.reshape(-1).cpu().numpy()}
    return out


@torch.no_grad()
def policy_over_state(model, s_values, which="dss", n=100, b_high=50.0, batch=100, burn=1000, device="cpu", seed=0):
    """Labor/consumption policy over assets, one curve per income state in ``s_values`` (evaluated
    against a simulated panel). Returns ``{b_grid, s_grid, n, c}``."""
    seed_all(seed)
    *ctx, hh_policy, W = _simulate_panel(model, which, batch, burn, device)
    b_grid = torch.linspace(float(ctx[2].b_min.mean()) - 0.1, b_high, n)
    n_curves, c_curves = _hh_curves(hh_policy, ctx, W, s_values, b_grid, device)
    return {"b_grid": b_grid.cpu().numpy(), "s_grid": np.asarray(s_values), "n": n_curves, "c": c_curves}


@torch.no_grad()
def policy_by_income(model, which="dss", percentiles=(0.1, 0.5, 0.9), n=100, b_high=50.0,
                     batch=1000, burn=1000, income_band=0.1, n_bins=30, device="cpu", seed=0):
    """Labor/consumption curves over assets at income-state percentiles, each paired with the asset
    density of agents within +/- ``income_band`` (rank) of that income level."""
    seed_all(seed)
    state_dist, state_agg, par, ss, hh_policy, W = _simulate_panel(model, which, batch, burn, device)
    ctx = (state_dist, state_agg, par, ss)
    s_all, b_all = state_dist.s.flatten(), state_dist.b.flatten()

    b_low = float(par.b_min.mean()) - 0.1
    b_grid = torch.linspace(b_low, b_high, n)
    edges = np.linspace(b_low, b_high, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    s_values = [float(s_all.quantile(q)) for q in percentiles]
    n_curves, c_curves = _hh_curves(hh_policy, ctx, W, s_values, b_grid, device)

    groups = []
    for i, q in enumerate(percentiles):
        s_lo = float(s_all.quantile(max(0.0, q - income_band)))
        s_hi = float(s_all.quantile(min(1.0, q + income_band)))
        cond = b_all[(s_all >= s_lo) & (s_all <= s_hi)].cpu().numpy()
        density = np.histogram(cond, bins=edges, density=True)[0]
        groups.append({"n": n_curves[i], "c": c_curves[i], "density": density})

    return {"b_grid": b_grid.cpu().numpy(), "centers": centers, "edges": edges,
            "percentiles": list(percentiles), "s_values": s_values, "groups": groups}


@torch.no_grad()
def asset_histogram(model, which="full", batch=2000, burn=500, amax=50.0, n_bins=50, device="cpu",
                    seed=0):
    """Simulated cross-sectional asset distribution over ``[0, amax]``.

    Assets below zero (the soft borrowing constraint lets agents dip slightly under the bound) are
    clamped to 0 so the constrained mass lands in the first bin, comparable with the SSJ mass point
    at the bound; the pre-clamp share is reported in ``tail_mass``. Density uses the full simulated
    sample as its denominator; the CDF is evaluated at the histogram edges.
    """
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    if which == "full":
        model.steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=1.0)
    else:
        model.dss_steps(batch=batch, device=device, steps=burn, scale_ind=1.0)

    b = model.state_dist.b.flatten().cpu().numpy()
    tail_mass = {"below_zero": np.mean(b < 0.0), "above_amax": np.mean(b > amax)}
    b = np.maximum(b, 0.0)
    counts, edges = np.histogram(b, bins=n_bins, range=(0.0, amax))
    mass = counts / b.size
    density = mass / np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf = np.searchsorted(np.sort(b), edges, side="right") / b.size
    return {"centers": centers, "edges": edges, "density": density, "cdf": cdf,
            "tail_mass": tail_mass}


# --- Accuracy diagnostics ---

@torch.no_grad()
def asset_choice(model):
    """Current-period asset choice b' for the state_ind households, mirroring the current-period
    block of ``OneAssetHANK.residuals`` (same states and networks, so it matches the b_ind used
    there). Full model only."""
    par, ss = model.par_draw, model.ss
    state_ind, state_dist, state_agg = model.state_ind, model.state_dist, model.state_agg
    n_ind = model.hh_policy(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
    n_dist = model.hh_policy(state_ind=state_dist, state_dist=state_dist, state_agg=state_agg, par=par)
    Pi, W = model.agg_policy(state_dist=state_dist, state_agg=state_agg, par=par)
    Rn_lag = state_agg.Rn_lag
    NE = torch.mean(torch.exp(state_dist.s) * n_dist, dim=-2, keepdim=True)
    Y = torch.exp(state_agg.z) * NE
    T = model._bond_tax(par.B, Rn_lag, Pi)
    tau = T / torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
    c_ind = ((W * torch.exp(state_ind.s)) / (ss.chi * n_ind ** par.eta)) ** (1 / par.gamma)
    div = Y - W * NE - 0.5 * par.varphi * torch.log(Pi) ** 2 * Y
    wealth_ind = (W * torch.exp(state_ind.s) * n_ind + Rn_lag / Pi * state_ind.b
                  + div * torch.exp(state_ind.s) - tau * torch.exp(state_ind.s))
    return wealth_ind - c_ind


@torch.no_grad()
def residual_accuracy(model, which="full", batch=10, mc=50, burn=200, draws=50,
                      device="cpu", adjusted_residuals=False, collect_asset_choice=False, seed=0):
    """Absolute equilibrium residuals collected over simulated states, one array per residual.
    ``collect_asset_choice`` (full model only) adds a ``b_choice`` array aligned with ``res_fb``,
    for conditioning the household Euler errors on unconstrained households."""
    seed_all(seed)
    model.to(device)
    model.initialize(batch=batch, device=device)
    residual_fn = model.residuals if which == "full" else model.dss_residuals
    scale_agg = 1.0 if which == "full" else 0.0
    if which == "full":
        model.steps(batch=batch, device=device, steps=burn, scale_ind=1.0, scale_agg=scale_agg)
    else:
        model.dss_steps(batch=batch, device=device, steps=burn, scale_ind=1.0)

    names = model.RESIDUAL_NAMES[which]
    collected = {name: [] for name in names}
    if collect_asset_choice:
        collected["b_choice"] = []
    for _ in range(draws):
        if collect_asset_choice:
            collected["b_choice"].append(asset_choice(model).flatten().cpu())
        e = model.draw_shocks(batch=batch, mc=mc, antithetic=True, device=device)
        residuals = residual_fn(e=e, mc=mc, scale_ind=1.0, scale_agg=scale_agg, adjusted_residuals=adjusted_residuals)
        for name, r in zip(names, residuals):
            collected[name].append(r.abs().flatten().cpu())
        if which == "full":
            model.steps(batch=batch, device=device, steps=1, scale_ind=1.0, scale_agg=scale_agg)
        else:
            model.dss_steps(batch=batch, device=device, steps=1, scale_ind=1.0)

    return {name: torch.cat(parts).numpy() for name, parts in collected.items()}


# --- Impulse responses ---

@torch.no_grad()
def _burn_in_zero_agg(model, burn, batch, device):
    """Simulate ``burn`` steps with all aggregate innovations set to zero."""
    for _ in range(burn):
        e = model.draw_shocks(batch, mc=None, antithetic=False, device=device)
        _, _, e_agg = e
        for k in e_agg.keys():
            e_agg[k].zero_()
        model.step(e, scale_ind=1.0, scale_agg=0.0)


@torch.no_grad()
def _draw_sim_shocks(model, batch, sim, device):
    """Pre-draw a fixed idiosyncratic shock sequence (aggregate innovations zeroed) for CRN."""
    sequence = []
    for _ in range(sim):
        e = model.draw_shocks(batch, mc=None, antithetic=False, device=device)
        _, _, e_agg = e
        for k in e_agg.keys():
            e_agg[k].zero_()
        sequence.append(e)
    return sequence


@torch.no_grad()
def _run_sim(model, start_state, shock_seq, shock_name, mps_path):
    """Replay ``shock_seq`` from ``start_state``, overriding the aggregate ``shock_name`` innovation
    with ``mps_path[t]`` each step (idiosyncratic shocks stay common). Returns {obs: (sim, batch)}."""
    s_ind, s_dist, s_agg = start_state
    model.state_ind = s_ind.clone()
    model.state_dist = s_dist.clone()
    model.state_agg = s_agg.clone()
    recorded = None
    for t, e in enumerate(shock_seq):
        _, _, e_agg = e
        e_agg[shock_name] = torch.full_like(e_agg[shock_name], float(mps_path[t]))
        raw = model.step(e, scale_ind=1.0, scale_agg=1.0)
        if recorded is None:
            recorded = {k: [] for k in raw.keys()}
        for k in recorded:
            recorded[k].append(raw[k].squeeze(-1).squeeze(-1).detach().to("cpu"))
    return {k: torch.stack(v) for k, v in recorded.items()}


@torch.no_grad()
def irf(model, shock_name="mps", shock_value=-1.0, batch=10_000, burn=1000, sim=100,
        device="cpu", rho_mps=None, seed=0, method="crn"):
    """Impulse response to a one-time ``shock_name`` innovation. ``method="crn"`` differences a
    shocked vs unshocked replay sharing idiosyncratic shocks (low-variance deviation IRF);
    ``"level"`` returns the shocked path. Returns {observable: (sim, batch)}."""
    seed_all(seed)
    if rho_mps is not None:
        model.par.set("rho_mps", torch.tensor(float(rho_mps)))
    model.initialize(batch=batch, device=device)
    _burn_in_zero_agg(model, burn, batch, device)

    start_state = (model.state_ind.clone(), model.state_dist.clone(), model.state_agg.clone())
    shock_seq = _draw_sim_shocks(model, batch, sim, device)
    shock_path = [shock_value] + [0.0] * (sim - 1)

    shocked = _run_sim(model, start_state, shock_seq, shock_name, shock_path)
    if method == "crn":
        baseline = _run_sim(model, start_state, shock_seq, shock_name, [0.0] * sim)
        return {k: shocked[k] - baseline[k] for k in shocked}
    return shocked
