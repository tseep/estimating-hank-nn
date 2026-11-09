"""Plotting for the two-asset Krusell-Smith model: arrays from ``kstwo_analysis`` and the SSJ
reference in, ``(fig, ax)`` out, no model calls."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from estimating_hank_nn.plot_style import style_fig_ax, make_subplot_grid, DUO_SIZE, MULTI_SIZE, LW_MULTI

# Variables shown in the IRF plot, as NN observable -> SSJ reference column.
IRF_MAPPING = {"A": "Z", "Y": "Y", "C": "C", "K": "K", "w": "w", "r": "r", "Q": "Q"}

VAR_LABELS = {"A": "TFP shock", "Y": "Output", "C": "Consumption", "K": "Capital", "w": "Wage",
              "r": "Interest rate", "Q": "Bond price"}

EULER_TITLES = {"res_fb_k": "FOC capital", "res_fb_b": "FOC bond", "res_y": "Output",
                "res_k": "Capital", "res_k_next": "Capital (next)", "res_b": "Bond market clearing"}


# %% Household policy functions (NN vs SSJ)

def plot_hh_policy(curves, reference, labels=("NN method", "SSJ method")):
    """NN household policies (solid, one line per income state) vs the SSJ reference (dashed) over
    capital, for a household that holds the bond supply: consumption, next-period capital and
    bonds. ``labels`` is the (NN, reference) pair."""
    nn_label, ssj_label = labels
    fig, ax = make_subplot_grid(3, 3, (MULTI_SIZE[0], MULTI_SIZE[1] + 0.5))
    for i, (kind, ref_key, ylabel) in enumerate(
            [("c", "hh_c", "Consumption"), ("k_next", "hh_k", "Capital"), ("b_next", "hh_b", "Bonds")]):
        nn = curves[kind]
        ref = reference[ref_key]
        for idx in range(nn.shape[0]):
            ax[i].plot(curves["k_grid"], nn[idx], color=f"C{idx}", linewidth=LW_MULTI,
                       label=nn_label if idx == 0 else None)
            ax[i].plot(ref["k_grid"], ref[f"e{idx}"], color="black", linestyle="--",
                       alpha=0.5, linewidth=LW_MULTI,
                       label=ssj_label if idx == 0 else None)
        ax[i].set_xlabel("Capital")
        ax[i].set_title(f"{ylabel} policy function")
    # The capital policy is close to the 45-degree line; the low end shows the borrowing limit.
    ax[1].set_xlim(0.0, 5.0)
    ax[1].set_ylim(0.0, 6.0)
    ax[2].set_xlim(0.0, 5.0)
    ax[2].set_ylim(0.0, 2.0)
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


def plot_policy_surface(surface, zlim=None):
    """3D consumption surface over capital and bonds at a fixed productivity quantile. ``zlim``
    gives several surfaces the same consumption axis."""
    fig = plt.figure(figsize=(4.2, 3.5))
    ax = fig.add_subplot(projection="3d")
    K_mesh, B_mesh = np.meshgrid(surface["k_grid"], surface["b_grid"], indexing="ij")

    # The light end of Blues is cut off: the low-consumption corner, where the borrowing
    # constraint binds, would be near-white.
    cmap_blues = ListedColormap(plt.cm.Blues(np.linspace(0.35, 1.0, 256)))
    ax.plot_surface(K_mesh, B_mesh, surface["c"], cmap=cmap_blues, edgecolor="none", antialiased=True)

    ax.set_xlabel("Capital")
    ax.set_ylabel("Bonds")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Consumption", rotation=90, labelpad=2)
    ax.set_title("Consumption policy function (DSS)", y=0.97)
    if zlim is not None:
        ax.set_zlim(*zlim)

    # plot_surface returns a collection that a legend cannot show, so a proxy patch stands in.
    proxy = Patch(facecolor=cmap_blues(0.7), edgecolor="none")
    ax.legend([proxy], [f"Productivity: {int(round(surface['s_quantile'] * 100))}th pct"],
              loc="lower right", bbox_to_anchor=(0.98, 0.16), fontsize="small")

    ax.view_init(elev=25, azim=-120)
    # tight and constrained layouts do not handle 3D axes, so the margins are explicit.
    ax.set_box_aspect(None, zoom=1.05)
    fig.subplots_adjust(left=0.06, right=1.0, top=1.0, bottom=0.06)
    return fig, ax


# %% Asset distributions

def plot_asset_distribution(hist, reference, labels=("NN method", "SSJ method")):
    """NN simulated capital (left) and bond (right) distributions vs the SSJ steady state, as
    density histograms over common bins. ``labels`` is the (NN, reference) pair."""
    nn_label, ssj_label = labels
    ssj = reference["asset_dist"]
    fig, axes = plt.subplots(1, 2, figsize=(DUO_SIZE[0] * 2, DUO_SIZE[1]), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    ax = list(axes)
    for a, asset, name, title, color in [(ax[0], "k", "Capital", "Capital distribution", "C0"),
                                         (ax[1], "b", "Bonds", "Bond distribution", "C1")]:
        edges = hist[f"edges_{asset}"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        ssj_density = np.histogram(ssj[asset], bins=edges, weights=ssj["mass"], density=True)[0]
        a.bar(centers, hist[f"density_{asset}"], width=np.diff(edges), color=color,
              alpha=0.55, edgecolor="black", linewidth=0.4, label=nn_label)
        a.bar(centers, ssj_density, width=np.diff(edges), color="C3",
              alpha=0.55, edgecolor="black", linewidth=0.4, label=ssj_label)
        a.set_xlabel(name)
        a.set_ylabel("Density")
        a.set_title(title)
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, edgecolor="none", framealpha=1)
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
