"""Plotting for the one-asset Krusell-Smith model: arrays from ``ksone_analysis`` and the SSJ
reference in, ``(fig, ax)`` out, no model calls."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from estimating_hank_nn.plot_style import style_fig_ax, make_subplot_grid, DUO_SIZE, MULTI_SIZE, LW_MULTI

# Variables shown in the IRF plot, as NN observable -> SSJ reference column.
IRF_MAPPING = {"A": "Z", "Y": "Y", "C": "C", "K": "K", "w": "w", "r": "r"}

VAR_LABELS = {"A": "TFP shock", "Y": "Output", "C": "Consumption", "K": "Capital", "w": "Wage",
              "r": "Interest rate"}

EULER_TITLES = {"res_fb": "Euler equation", "res_y": "Output", "res_k": "Capital",
                "res_k_next": "Capital (next)"}


# %% Household policy functions (NN vs SSJ)

def plot_hh_policy(curves, reference, labels=("NN method", "SSJ method")):
    """NN household policies (solid, one line per income state) vs the SSJ reference (dashed):
    consumption on the left panel, next-period capital on the right. ``labels`` is the
    (NN, reference) pair."""
    nn_label, ssj_label = labels
    fig, ax = make_subplot_grid(2, 2, DUO_SIZE)
    for i, (kind, ref_key, ylabel) in enumerate(
            [("c", "hh_c", "Consumption"), ("k_next", "hh_a", "Capital")]):
        nn = curves[kind]
        ref = reference[ref_key]
        for idx in range(nn.shape[0]):
            ax[i].plot(curves["k_grid"], nn[idx], color=f"C{idx}", linewidth=LW_MULTI,
                       label=nn_label if idx == 0 else None)
            ax[i].plot(ref["a_grid"], ref[f"e{idx}"], color="black", linestyle="--",
                       alpha=0.5, linewidth=LW_MULTI,
                       label=ssj_label if idx == 0 else None)
        ax[i].set_xlabel("Capital")
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(f"{ylabel} policy function")
    # The capital policy is close to the 45-degree line; the low end shows the borrowing limit.
    ax[1].set_xlim(0.0, 5.0)
    ax[1].set_ylim(0.0, 6.0)
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


# %% Household policies at productivity percentiles

def plot_hh_policy_percentiles(policies, label="NN method (DSS)"):
    """Next-period capital (top row) and consumption (bottom row) over capital at the productivity
    percentiles of ``policies`` (from ``ksone_analysis.policy_by_income``), with the conditional
    capital density shaded behind on a twin axis (labelled on the right column)."""
    titles = [f"{int(round(q * 100))}th percentile" for q in policies["percentiles"]]
    fig, axes = plt.subplots(2, len(titles), figsize=(MULTI_SIZE[0] * len(titles), MULTI_SIZE[1] * 2),
                             layout="constrained")
    fig.get_layout_engine().set(hspace=0.18, wspace=0.06)
    twins = [[a.twinx() for a in row] for row in axes]
    width = np.diff(policies["edges"])
    for k, (title, group) in enumerate(zip(titles, policies["groups"])):
        axes[0][k].plot(policies["k_grid"], group["k_next"], color="C0", linewidth=LW_MULTI)
        axes[1][k].plot(policies["k_grid"], group["c"], color="C0", linewidth=LW_MULTI)
        for row in (0, 1):
            twins[row][k].bar(policies["centers"], group["density"], width=width, color="C0", alpha=0.15)
            if k < len(titles) - 1:
                twins[row][k].set_yticklabels([])
            else:
                twins[row][k].set_ylabel("Density")
        axes[0][k].set_title(title)
        axes[1][k].set_xlabel("Capital")
    axes[0][0].set_ylabel("Capital")
    axes[1][0].set_ylabel("Consumption")
    handles = [Line2D([0], [0], color="C0", linewidth=LW_MULTI, label=f"{label}, policy"),
               Patch(facecolor="C0", alpha=0.3, label=f"{label}, distribution")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, [a for row in axes for a in row])


# %% Asset distribution (NN vs SSJ)

def plot_asset_distribution(hist, reference, colors=("C0", "C3"),
                            labels=("NN method", "SSJ method")):
    """NN simulated capital distribution vs the SSJ steady state: density histogram (left) and CDF
    (right) over common bins, both normalized on the plotted range. ``colors`` / ``labels`` are
    the (NN, reference) pairs."""
    nn_color, ssj_color = colors
    nn_label, ssj_label = labels
    a_grid = reference["asset_dist"]["a_grid"]
    mass = reference["asset_dist"]["mass"]
    edges = hist["edges"]
    width = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ssj_density = np.histogram(a_grid, bins=edges, weights=mass, density=True)[0]

    fig, axes = plt.subplots(1, 2, figsize=(DUO_SIZE[0] * 2, DUO_SIZE[1]), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    ax = list(axes)
    ax[0].bar(centers, hist["density"], width=width, color=nn_color, alpha=0.55,
              edgecolor="black", linewidth=0.4, label=nn_label)
    ax[0].bar(centers, ssj_density, width=width, color=ssj_color, alpha=0.55,
              edgecolor="black", linewidth=0.4, label=ssj_label)
    ax[0].set_xlabel("Capital")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Asset distribution")

    ax[1].plot(edges[1:], hist["cdf"], color=nn_color, linewidth=LW_MULTI, label=nn_label)
    ax[1].plot(edges[1:], np.cumsum(ssj_density * width), color="black", linestyle="--",
               linewidth=LW_MULTI, label=ssj_label)
    ax[1].set_xlabel("Capital")
    ax[1].set_ylabel("Cumulative share")
    ax[1].set_title("Cumulative asset distributions")

    # A legend per panel, so each shows its own mark type: fills left, lines right.
    for a in ax:
        handles, labels_leg = a.get_legend_handles_labels()
        a.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                 frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


# %% Euler-equation errors

def plot_euler_errors(errors, names, log_min=-8, log_max=0):
    """Density histograms of log10 absolute residuals, one panel per residual in ``names``. The
    x-axis is the log10 residual clipped to [log_min, log_max]; a dashed line marks the mean."""
    fig, ax = make_subplot_grid(len(names), 2, DUO_SIZE)
    for i, name in enumerate(names):
        log_err = np.clip(np.log10(np.abs(errors[name]).flatten() + 1e-12), log_min, log_max)
        color = f"C{i}"
        ax[i].hist(log_err, bins=50, color=color, edgecolor=color, alpha=0.25, density=True)
        ax[i].axvline(log_err.mean(), color=color, linestyle="--", linewidth=LW_MULTI)
        ax[i].set_xlim(log_min, log_max)
        ax[i].set_title(EULER_TITLES.get(name, name))
        ax[i].set_xlabel(r"$\log_{10}$ error")
        ax[i].set_ylabel("Density")
    handles = [Patch(facecolor="C0", edgecolor="C0", alpha=0.25, label="Distribution"),
               Line2D([0], [0], color="C0", linestyle="--", linewidth=LW_MULTI, label="Mean")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


# %% Impulse responses (NN vs SSJ)

def plot_irf_comparison(nn_irf, reference, horizon=40, n_cols=3):
    """NN IRF vs the SSJ linear deviation IRF, one panel per variable in ``IRF_MAPPING``."""
    ssj = reference["irf_linear_dev"]
    fig, ax = make_subplot_grid(len(IRF_MAPPING), n_cols, (MULTI_SIZE[0], MULTI_SIZE[1] + 0.5))
    for i, (nn_key, ssj_col) in enumerate(IRF_MAPPING.items()):
        # The NN path starts one period before the innovation moves productivity, so NN index 1
        # is the impact quarter, aligned with SSJ index 0.
        path = nn_irf[nn_key][1:horizon + 2]
        ax[i].plot(np.arange(path.shape[0]), path, color="C0", linewidth=LW_MULTI, label="NN method")
        ax[i].plot(ssj["h"][:horizon + 1], ssj[ssj_col][:horizon + 1], color="C1", linestyle="--",
                   linewidth=LW_MULTI, label="SSJ method")
        ax[i].set_title(VAR_LABELS[nn_key])
        ax[i].xaxis.set_major_locator(MultipleLocator(10))
        ax[i].yaxis.set_major_locator(MaxNLocator(5))
        ax[i].set_xlabel("Quarters")
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)
