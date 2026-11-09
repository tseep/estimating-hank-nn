"""Policy plots adapted from the paper's analytical replication example.

The same parameter panels and policy surfaces support the additional interest-rate
output and lagged-rate state. Generic loss and likelihood plots come from the package.
"""

from copy import deepcopy

import torch
import matplotlib.pyplot as plt

from estimating_hank_nn.plot_style import style_fig_ax, make_subplot_grid, panel_size, N_COLS, LW_SINGLE
from estimating_hank_nn.structures import State


def policy_over_par(model, policy_fn, shock_std=0.0, par_name="beta", n=100):
    """Evaluate policy function over a 1D parameter grid.

    ``shock_std`` plays the role of eps in the zeta transition (-1 gives positive zeta).

    Returns:
        Dictionary with the grid and one entry per policy output.
    """
    par_range = model.ranges.get(par_name)
    lower_bound = par_range.support.lower_bound
    upper_bound = par_range.support.upper_bound
    grid = torch.linspace(lower_bound, upper_bound, n).view(-1, 1)

    par = deepcopy(model.par)
    par.to("cpu")

    values = {key: [] for key in model.POLICY_NAMES}
    for g in grid:
        par.set(par_name, g)
        ss = model.steady_state(par=par)
        loading = par.sigma_a * par.sigma * (par.rho_a - 1) * ss.omega
        state = State({"zeta": (shock_std * loading).view(1)})

        if "mp" in model.STATE_NAMES:
            state.mp = torch.zeros_like(state.zeta)
        if "IL" in model.STATE_NAMES:
            state.IL = torch.zeros_like(state.zeta)
        with torch.no_grad():
            policies = policy_fn(state=state, par=par)
        for key, value in zip(model.POLICY_NAMES, policies):
            values[key].append(value)

    return {"grid": grid, **{key: torch.cat(series, dim=0) for key, series in values.items()}}


def policy_over_par_list(model, policy_fn, shock_std, par_list, n=100):
    """Evaluate policy over a list of parameters."""
    results = {}
    for par_name in par_list:
        results[par_name] = policy_over_par(model, policy_fn, shock_std, par_name=par_name, n=n)
    return results


def plot_par_list(model, nn_policy_fn, analytical_policy_fn=None, shock_std=-1.0, policy="Pi",
                  par_list=None, par_labels=None, n=100, n_cols=N_COLS, center_last_row=True,
                  model_label="Neural network", y_space=0.001, panel_titles=True):
    """Plot model vs analytical policy over each parameter."""
    if par_list is None:
        par_list = model.par.keys()
    n_par = len(par_list)

    fig, axes = make_subplot_grid(n_par, n_cols, panel_size(n_cols), center_last_row=center_last_row)

    ana = (policy_over_par_list(model, analytical_policy_fn, shock_std, par_list=par_list, n=n)
           if analytical_policy_fn is not None else None)
    num = policy_over_par_list(model, nn_policy_fn, shock_std, par_list=par_list, n=n)

    policy_latex = {"X": r"\hat{X}", "Pi": r"\hat{\Pi}", "I": r"\hat{I}"}[policy]
    title = rf"Policy function ${policy_latex}_t$ conditioned on "

    for i, key in enumerate(par_list):
        a = axes[i]

        a.plot(num[key]["grid"], num[key][policy], label=model_label, linewidth=LW_SINGLE, color="C0")
        if ana is not None:
            a.plot(ana[key]["grid"], ana[key][policy], label="Analytical",
                   linewidth=LW_SINGLE, linestyle="dashed", color="C1")

        if par_labels and key in par_labels:
            ylabel, latex = par_labels[key]
        else:
            ylabel, latex = key, key

        if panel_titles:
            a.set_title(f"{title}{latex}")
        a.set_xlabel(f"{ylabel} {latex}")

        curves = [num[key][policy]]
        if ana is not None:
            curves.append(ana[key][policy])
        ymin = min(curve.min().item() for curve in curves)
        ymax = max(curve.max().item() for curve in curves)
        xmin = num[key]["grid"].min().item()
        xmax = num[key]["grid"].max().item()
        a.set_ylim(bottom=ymin - y_space, top=ymax + y_space)
        a.set_xlim(left=xmin, right=xmax)
        a.yaxis.set_major_locator(plt.MaxNLocator(4))

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3),
               frameon=True, edgecolor="none", framealpha=1)
    fig, _ = style_fig_ax(fig, axes, xminor=5, yminor=4)
    return fig, axes


def plot_3d_policy_par(model, policy_fn, plot="Pi", par_name="beta", par_labels=None, n=100):
    """Plot the policy surface in 3D over one parameter and the state grid.

    ``plot`` selects a policy output. The lagged rate, when present, is held at zero.
    """
    par = deepcopy(model.par)
    par.to("cpu")

    loading = par.sigma_a * par.sigma * (par.rho_a - 1) * model.steady_state(par).omega
    std = loading.abs() / (1 - par.rho_a.square()).sqrt()
    grid_state = torch.linspace(-3 * std.item(), 3 * std.item(), n)

    par_range = model.ranges.get(par_name)
    grid_par = torch.linspace(par_range.support.lower_bound, par_range.support.upper_bound, n)

    # Evaluate policy over (par, state) grid: one policy call per par value,
    # batched across the state grid.
    out_idx = model.POLICY_NAMES.index(plot)
    surface = torch.empty(n, n)
    state_batch = State({"zeta": grid_state.view(-1, 1)})
    if "IL" in model.STATE_NAMES:
        state_batch.IL = torch.zeros_like(state_batch.zeta)
    if "mp" in model.STATE_NAMES:
        state_batch.mp = torch.zeros_like(state_batch.zeta)
    for i, p_val in enumerate(grid_par):
        par.set(par_name, p_val.view(1))
        with torch.no_grad():
            results = policy_fn(state=state_batch, par=par)
        surface[i, :] = results[out_idx].flatten()

    # Evaluate the cross-section at the reference value, even if it is between grid points.
    cross_val = getattr(model.par, par_name).item()
    par.set(par_name, torch.tensor([cross_val]))
    with torch.no_grad():
        cross_policy = policy_fn(state=state_batch, par=par)[out_idx].flatten()

    xlabel, par_latex = par_labels[par_name]
    title = {"X": r"Output gap ($\hat{X}_t$)",
             "Pi": r"Inflation ($\hat{\Pi}_t$)",
             "I": r"Nominal interest rate ($\hat{I}_t$)"}[plot]

    X_mesh, Y_mesh = torch.meshgrid(grid_par, grid_state, indexing="ij")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_proj_type("ortho")
    ax.plot_surface(X_mesh, Y_mesh, surface, cmap="viridis_r", edgecolor="none")
    line = ax.plot(torch.full_like(grid_state, cross_val), grid_state, cross_policy,
                   color="black", linewidth=LW_SINGLE, linestyle="--", zorder=10,
                   label=f"At {par_latex}={cross_val:.2f}")[0]

    ax.set_title(title)
    ax.set_xlabel(f"{xlabel} ({par_latex})", labelpad=2)
    ax.set_ylabel(r"State $\zeta$", labelpad=2)
    ax.set_zlabel({"X": r"$\hat{X}_t$", "Pi": r"$\hat{\Pi}_t$", "I": r"$\hat{I}_t$"}[plot], labelpad=2)
    ax.tick_params(axis="both", which="major", pad=0)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

    fig.legend(handles=[line], loc="lower center", bbox_to_anchor=(0.5, 0.02),
               frameon=True, edgecolor="none", framealpha=1, ncol=1)
    return fig, ax
