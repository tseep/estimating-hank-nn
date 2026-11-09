"""Pure plotting for the one-asset HANK replication.

Every function here takes already-computed arrays/dicts (from ``hankmp_analysis`` —
``policy_over_state`` / ``residual_accuracy`` / ``asset_histogram`` / ``irf``) plus the loaded
SSJ reference (pandas DataFrames), and returns a ``(fig, ax)`` pair. There are no model calls or
simulation here. The caller saves figures with ``estimating_hank_nn.plot_style.save_figure``.

Styling follows the shared common style (``estimating_hank_nn.plot_style``):
constrained-layout grids via ``make_subplot_grid``, the ``*_SIZE`` constants, the ``LW_*``
line widths, and a single legend centered below the panels (the asset-distribution figure is the
one exception: it carries a legend per panel).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from estimating_hank_nn.plot_style import (
    style_fig_ax, make_subplot_grid,
    DUO_SIZE, MULTI_SIZE, LW_MULTI,
)

# Economic variables shown in the IRF plots, as NN observable -> SSJ reference column. This
# subset (output, real rate, inflation, wage) drives both the NN-vs-SSJ and the reference IRFs.
IRF_MAPPING = {"Y": "Y", "Rr": "Rr", "Pi": "Pi", "W": "w"}

# The published Figure 7 panel set and order: shock, output, interest rate / inflation, wage, tax.
IRF_MAPPING_PAPER = {"mps": "rstar", "Y": "Y", "Rr": "Rr", "Pi": "Pi", "W": "w", "T": "Tax"}

# Descriptive panel titles for the economic variables. Keyed by both the NN observable
# (e.g. "W") and the SSJ column (e.g. "w") so either side can look a label up.
VAR_LABELS = {
    "Y": "Output",
    "C": "Consumption",
    "NE": "Labor",
    "Rr": "Interest rate",
    "Rn": "Nominal rate",
    "Pi": "Inflation",
    "w": "Wage",
    "W": "Wage",
    "T": "Tax",
    "Tax": "Tax",
    "mps": "Monetary policy shock",
    "rstar": "Monetary policy shock",
}

# Descriptive names for the equilibrium residuals, keyed by the residual code names in
# ``OneAssetHANK.RESIDUAL_NAMES`` (full uses res_nkpc, dss uses res_y).
RESIDUAL_LABELS = {
    "res_fb": "Euler equation",
    "res_nkpc": "NKPC",
    "res_y": "Output",
    "res_bond": "Bond",
    "res_product": "Product",
    "res_bond_next": "Bond (next)",
    "res_product_next": "Product (next)",
}

# Euler-error panel titles (Figure 15 wording — only the four current-period conditions).
EULER_TITLES = {"res_fb": "Euler equation", "res_nkpc": "NKPC", "res_y": "Output",
                "res_bond": "Bond market", "res_product": "Product market"}


# %% Deterministic steady-state values (NN vs SSJ)

DSS_VALUE_LABELS = {"Rn_dss": "Steady-state interest rate", "Y_dss": "Steady-state output"}


def plot_dss_values(sweep, true_values, par_labels=None, outputs=("Rn_dss", "Y_dss"), min_rel_pad=0.002):
    """NN deterministic-steady-state predictions as each estimable parameter is swept across its
    training range. One panel per (parameter, output): the NN curve (solid) against the constant
    SSJ steady-state level (dashed). The DSS does not depend on these monetary-shock parameters, so
    a well-trained network is flat and on target. ``sweep`` is ``hankmp_analysis.policy_over_par``
    output (``{param: {"grid", "Rn_dss", "Y_dss"}}``); ``true_values`` maps each output to its SSJ
    level; ``par_labels`` optionally relabels the x-axes. The y-range is centred on the target with a
    half-window floored at ``min_rel_pad`` of the level, so the network's ~1e-5 noise floor reads as
    flat (not auto-zoomed to look jagged) while a genuine level offset stays visible."""
    par_labels = par_labels or {}
    panels = [(p, o) for p in sweep for o in outputs]
    fig, ax = make_subplot_grid(len(panels), len(outputs), DUO_SIZE)
    for i, (p, o) in enumerate(panels):
        series = np.asarray(sweep[p][o], dtype=float)
        target = true_values[o]
        ax[i].plot(sweep[p]["grid"], series, color="C0", linewidth=LW_MULTI, label="NN method")
        ax[i].axhline(target, color="C1", linestyle="--", linewidth=LW_MULTI, label="SSJ target")
        half = 1.3 * max(abs(series - target).max(), min_rel_pad * abs(target))
        ax[i].set_ylim(target - half, target + half)
        ax[i].ticklabel_format(axis="y", useOffset=False)  # plain tick labels, no "+1.004" offset
        ax[i].set_xlabel(par_labels.get(p, p))
        ax[i].set_title(DSS_VALUE_LABELS.get(o, o))
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 2), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


# %% Household policy functions (NN vs SSJ)

def plot_hh_policy(curves, reference, labels=("NN method", "SSJ method")):
    """NN household policies (solid, one line per income state) vs the SSJ reference
    (dashed): labor on the left panel, consumption on the right. ``labels`` is the
    (NN, reference) pair."""
    nn_label, ssj_label = labels
    fig, ax = make_subplot_grid(2, 2, DUO_SIZE)
    for i, (kind, ref_key, ylabel) in enumerate(
            [("n", "hh_n", "Labor supply"), ("c", "hh_c", "Consumption")]):
        nn = curves[kind]
        ref = reference[ref_key]
        for idx in range(nn.shape[0]):
            ax[i].plot(curves["b_grid"], nn[idx], color=f"C{idx}", linewidth=LW_MULTI,
                       label=nn_label if idx == 0 else None)
            ax[i].plot(ref["a_grid"], ref[f"e{idx}"], color="black", linestyle="--",
                       alpha=0.5, linewidth=LW_MULTI,
                       label=ssj_label if idx == 0 else None)
        ax[i].set_xlabel("Assets")
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(f"{ylabel} policy function")
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


# %% Asset distribution (NN vs SSJ)

def plot_asset_distribution(hist, reference, colors=("C0", "C3"),
                            labels=("NN method", "SSJ method")):
    """NN simulated asset distribution vs the SSJ steady state: density histogram (left) and
    unconditional CDF (right), over common bins. ``colors`` / ``labels`` are the (NN, reference)
    pairs. Both CDFs are cumulative mass through each bin, plotted at its right edge: at the
    borrowing limit itself SSJ holds an atom while the NN's constrained mass sits just above it,
    so comparing there reads a smear against an atom rather than a real difference."""
    nn_color, ssj_color = colors
    nn_label, ssj_label = labels
    a_grid = reference["asset_dist"]["a_grid"]
    mass = reference["asset_dist"]["mass"]
    width = np.diff(hist["edges"])
    bin_right = hist["edges"][1:]
    ssj_density = np.histogram(a_grid, bins=hist["edges"], weights=mass)[0] / width
    ssj_cdf = np.array([mass[a_grid <= edge].sum() for edge in bin_right])

    # Own figure rather than make_subplot_grid: the panels carry their own legends (below), which
    # only centre on the plot box when anchored in axes coordinates.
    fig, axes = plt.subplots(1, 2, figsize=(DUO_SIZE[0] * 2, DUO_SIZE[1]), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    ax = list(axes)
    ax[0].bar(hist["centers"], hist["density"], width=width, color=nn_color, alpha=0.55,
              edgecolor="black", linewidth=0.4, label=nn_label)
    ax[0].bar(hist["centers"], ssj_density, width=width, color=ssj_color, alpha=0.55,
              edgecolor="black", linewidth=0.4, label=ssj_label)
    ax[0].set_xlabel("Assets")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Asset distribution")

    ax[1].plot(bin_right, hist["cdf"][1:], color=nn_color, linewidth=LW_MULTI, label=nn_label)
    ax[1].plot(bin_right, ssj_cdf, color="black", linestyle="--", linewidth=LW_MULTI,
               label=ssj_label)
    ax[1].set_xlabel("Assets")
    ax[1].set_ylabel("Cumulative share")
    ax[1].set_title("Cumulative asset distributions")

    # A legend per panel, so each shows its own mark type: fills left, lines right.
    for a in ax:
        handles, labels_leg = a.get_legend_handles_labels()
        a.legend(handles, labels_leg, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                 frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


# %% Ladder overlays (agent-count / MC-draw sensitivity)
# Both functions read the frames that main_hankmp.py saves to results/ for each run
# (asset_hist_full.csv, hh_policy_income_full.csv, hh_policy_income_density_full.csv), so
# main_hankmp_comparisons.py can overlay the D*/MC* ladders without reloading any model.

def plot_asset_distribution_ladder(hists):
    """Overlaid simulated asset distributions for a ladder of runs: density histogram (left)
    and CDF (right), one color per run. ``hists`` maps display label -> histogram frame
    (columns ``center``, ``width``, ``density``, ``cdf_right``)."""
    fig, axes = plt.subplots(1, 2, figsize=(DUO_SIZE[0] * 2, DUO_SIZE[1]), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    ax = list(axes)
    for i, (label, df) in enumerate(hists.items()):
        ax[0].bar(df["center"], df["density"], width=df["width"], color=f"C{i}", alpha=0.3,
                  edgecolor="black", linewidth=0.4, label=label)
        ax[1].plot(df["center"] + df["width"] / 2, df["cdf_right"], color=f"C{i}",
                   linewidth=LW_MULTI, label=label)
    ax[0].set_xlabel("Assets")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Asset distribution")
    ax[1].set_xlabel("Assets")
    ax[1].set_ylabel("Cumulative share")
    ax[1].set_title("Cumulative asset distributions")
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 4), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


def plot_hh_policy_ladder(runs):
    """Labor (top row) and consumption (bottom row) policies at the 10th/50th/90th income
    percentiles for a pair of ladder runs — first run solid, second dashed — with each run's
    conditional asset density shaded behind on a twin axis (labelled on the right column).
    ``runs`` maps display label -> (curves, densities) frame pair."""
    linestyles = ["-", "--", "-.", ":"]
    pcts = [("p10", "10th percentile"), ("p50", "50th percentile"), ("p90", "90th percentile")]
    fig, axes = plt.subplots(2, 3, figsize=(MULTI_SIZE[0] * 3, MULTI_SIZE[1] * 2),
                             layout="constrained")
    fig.get_layout_engine().set(hspace=0.18, wspace=0.06)
    twins = [[a.twinx() for a in row] for row in axes]
    for j, (label, (curves, dens)) in enumerate(runs.items()):
        for k, (pct, title) in enumerate(pcts):
            axes[0][k].plot(curves["b_grid"], curves[f"n_{pct}"], color=f"C{j}",
                            linestyle=linestyles[j], linewidth=LW_MULTI)
            axes[1][k].plot(curves["b_grid"], curves[f"c_{pct}"], color=f"C{j}",
                            linestyle=linestyles[j], linewidth=LW_MULTI)
            for row in (0, 1):
                twins[row][k].bar(dens["center"], dens[f"d_{pct}"], width=dens["width"],
                                  color=f"C{j}", alpha=0.15)
    for k, (pct, title) in enumerate(pcts):
        axes[0][k].set_title(title)
        axes[1][k].set_xlabel("Assets")
        for row in (0, 1):
            if k < len(pcts) - 1:
                twins[row][k].set_yticklabels([])
            else:
                twins[row][k].set_ylabel("Density")
    axes[0][0].set_ylabel("Labor supply")
    axes[1][0].set_ylabel("Consumption")
    labels = list(runs)
    handles = [Line2D([0], [0], color=f"C{j}", linestyle=linestyles[j], linewidth=LW_MULTI,
                      label=f"{lab}, policy") for j, lab in enumerate(labels)]
    handles += [Patch(facecolor=f"C{j}", alpha=0.3, label=f"{lab}, distribution")
                for j, lab in enumerate(labels)]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, [a for row in axes for a in row])


# %% Euler-equation errors

def euler_error_histograms(errors, log_min=-8, log_max=0, n_bins=50):
    """Histograms of the log10 absolute residuals of the four current-period equilibrium conditions
    (Euler/FB, NKPC, bond market, product market); next-period residuals are omitted. The log10
    residual is clipped to [log_min, log_max]. Returns {name: {edges, counts, mean}}."""
    hist = {}
    for name in errors:
        if name.endswith("_next"):
            continue
        log_err = np.clip(np.log10(np.abs(errors[name]) + 1e-12), log_min, log_max)
        counts, edges = np.histogram(log_err, bins=n_bins)
        hist[name] = {"edges": edges, "counts": counts, "mean": log_err.mean()}
    return hist


def plot_euler_errors(hist, log_min=-8, log_max=0):
    """2x2 density histograms from ``euler_error_histograms``; a dashed line marks the mean."""
    names = list(hist)
    colors = ["C0", "C1", "C2", "C3"]  # blue, red, black (gray via alpha), orange — Euler/NKPC/bond/product
    fig, ax = make_subplot_grid(len(names), 2, DUO_SIZE)
    for i, name in enumerate(names):
        edges, counts = hist[name]["edges"], hist[name]["counts"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = colors[i % len(colors)]
        ax[i].hist(centers, bins=edges, weights=counts, color=color, edgecolor=color, alpha=0.25, density=True)
        ax[i].axvline(hist[name]["mean"], color=color, linestyle="--", linewidth=LW_MULTI)
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

def plot_irf_comparison(nn_irf, reference, mapping=None, horizon=20, nn_ss=None, ssj_ss=None,
                        scale_nn=1.0, nn_label="NN method", nn_color="C0", xlabel="Quarters", ylabel=None,
                        n_cols=2, ssj_kinds=("nonlinear", "linear"), panel_size=None):
    """NN IRF vs the SSJ deviation IRFs, one panel per mapped variable.

    ``nn_irf`` is ``hankmp_analysis.irf`` output (CRN deviations), {obs: (sim, batch)}; each panel plots
    the mean over the batch of economies. ``ssj_kinds`` picks the overlaid SSJ lines: with both,
    nonlinear is dashed and linear dotted; a linear-only figure draws linear dashed instead (the
    published Figure 7 look). NN is ``nn_color`` solid; ``nn_label`` sets its legend, ``xlabel`` /
    ``ylabel`` the axes (``None`` = no axis label). ``scale_nn`` rescales the NN response.

    Pass ``nn_ss`` (the run's stochastic steady state, keyed by NN observable) and ``ssj_ss``
    (``reference["ss"]``, keyed by SSJ column) to plot in levels: each method is shifted to its own
    steady state, so the long-run NN-vs-SSJ gap is the SSS-DSS wedge. Default (both None) plots
    deviations.
    """
    mapping = mapping or IRF_MAPPING
    ssj_nl, ssj_lin = reference["irf_nonlinear_dev"], reference["irf_linear_dev"]
    items = [(nn_key, ssj_col) for nn_key, ssj_col in mapping.items()
             if nn_key in nn_irf and ssj_col in ssj_nl.columns]
    lin_color, lin_ls = ("C2", ":") if "nonlinear" in ssj_kinds else ("C1", "--")
    lin_label = "SSJ linear" if "nonlinear" in ssj_kinds else "SSJ method"
    if panel_size is None:
        panel_size = DUO_SIZE if n_cols == 2 else MULTI_SIZE

    fig, ax = make_subplot_grid(len(items), n_cols, panel_size)
    for i, (nn_key, ssj_col) in enumerate(items):
        nn_shift = nn_ss[nn_key] if nn_ss is not None else 0.0
        ssj_shift = ssj_ss[ssj_col] if ssj_ss is not None else 0.0
        # Advance-then-observe: the mps innovation at step 0 enters that step's obs, so NN index 0
        # is the contemporaneous response, aligned with SSJ index 0.
        path = scale_nn * nn_irf[nn_key].numpy()[:horizon + 1]  # quarters 0..horizon, so the axis runs to horizon
        mean = nn_shift + path.mean(axis=1)
        h = np.arange(mean.shape[0])
        hs = np.arange(min(horizon + 1, ssj_nl[ssj_col].shape[0]))
        ax[i].plot(h, mean, color=nn_color, linewidth=LW_MULTI, label=nn_label)
        if "nonlinear" in ssj_kinds:
            ax[i].plot(hs, ssj_shift + ssj_nl[ssj_col][:horizon + 1], color="C1", linestyle="--", linewidth=LW_MULTI, label="SSJ nonlinear")
        if "linear" in ssj_kinds:
            ax[i].plot(hs, ssj_shift + ssj_lin[ssj_col][:horizon + 1], color=lin_color, linestyle=lin_ls, linewidth=LW_MULTI, label=lin_label)
        ax[i].set_title(VAR_LABELS.get(nn_key, nn_key))
        ax[i].xaxis.set_major_locator(MultipleLocator(5))  # integer quarters: 0, 5, 10, ...
        ax[i].yaxis.set_major_locator(MaxNLocator(5))
        if xlabel:
            ax[i].set_xlabel(xlabel)
        if ylabel:
            ax[i].set_ylabel(ylabel)
    handles, labels_leg = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)


def plot_irf_curves(curves, panels, levels=False, horizon=20, ylabel=None, legend_cols=3, n_cols=3,
                    panel_size=None, draw_order=None):
    """Overlay impulse responses read from saved results, one panel per ``panels`` entry
    (column -> title). Each curve is a dict ``{"label", "color", "ls", "dev", "ss", "scale"}``
    with an optional ``"lw"``: ``dev`` is a deviation frame and ``ss`` its steady-state levels,
    used when ``levels`` is true. ``draw_order`` fixes the painting order, bottom to top; by
    default the first curve is drawn on top."""
    if ylabel is None:
        ylabel = "Level" if levels else "Deviation"
    if panel_size is None:
        panel_size = MULTI_SIZE if n_cols >= 3 else DUO_SIZE
    fig, ax = make_subplot_grid(len(panels), n_cols, panel_size)
    for i, (var, title) in enumerate(panels.items()):
        for c in (draw_order if draw_order is not None else list(reversed(curves))):
            y = c["scale"] * c["dev"][var].to_numpy()[:horizon + 1]
            if levels:
                y = c["ss"][var] + y
            ax[i].plot(np.arange(len(y)), y, color=c["color"], linestyle=c["ls"],
                       linewidth=c.get("lw", LW_MULTI))
        ax[i].set_title(title)
        ax[i].set_xlabel("Quarters")
        ax[i].set_ylabel(ylabel)
        ax[i].xaxis.set_major_locator(MultipleLocator(5))
    handles = [Line2D([], [], color=c["color"], linestyle=c["ls"], linewidth=c.get("lw", LW_MULTI),
                      label=c["label"]) for c in curves]
    fig.legend(handles, [h.get_label() for h in handles], loc="outside lower center",
               ncol=min(len(handles), legend_cols), frameon=True, edgecolor="none", framealpha=1)
    return style_fig_ax(fig, ax)
