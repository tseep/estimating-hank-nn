"""Plotting functions for model training, diagnostics, policy comparisons, and likelihoods.

All functions accept data via dicts/arrays and are model-agnostic.
"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_style import style_fig_ax, make_subplot_grid, panel_size, N_COLS, PLOT_SIZE, MULTI_SIZE, DUO_SIZE, LW_SINGLE, LW_MULTI


# %% Training loss plots

def _smooth(series, spec=None):
    """Smooth a 1-D series. ``spec`` is ``None`` (raw) or ``(method, window)``:
    ``"boxcar"`` (centred rolling mean — flattens spikes into plateaus), ``"ema"`` (debiased
    exponential moving average, ``span=window`` — causal, can look saw-toothed), ``"gaussian"``
    (centred Gaussian-weighted rolling mean, ``sigma=window/6`` — symmetric, smoothest), or
    ``"median"`` (centred rolling median — robust, removes spikes). All return the input length.

    A third element ``(method, window, "expanding")`` replaces the mirrored left edge with an
    expanding window, so the curve starts at the first raw value instead of the median of the
    first half-window. Use it when the opening transient is itself the point (training loss
    collapsing from its initial level); the default mirrors at both ends."""
    x = np.asarray(series, dtype=float)
    if not spec:
        return x
    method, window, *rest = spec
    expanding_start = bool(rest) and rest[0] == "expanding"
    window = int(window)
    if window <= 1:
        return x
    if method == "ema":
        beta = 1.0 - 2.0 / (window + 1.0)          # span-equivalent decay
        out = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = beta * acc + (1.0 - beta) * v
            out[i] = acc / (1.0 - beta ** (i + 1))  # debias
        return out
    pad = window // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="reflect"), window)[:x.size]
    if method == "median":
        out = np.median(windows, axis=1)
    elif method == "boxcar":
        out = windows.mean(axis=1)
    elif method == "gaussian":
        offsets = np.arange(window) - (window - 1) / 2.0
        weights = np.exp(-0.5 * (offsets / (window / 6.0)) ** 2)
        out = windows @ (weights / weights.sum())
    else:
        raise ValueError(f"unknown smooth method {method!r}; use 'boxcar', 'ema', 'gaussian', or 'median'")
    if expanding_start:
        reduce = {"median": np.median, "boxcar": np.mean, "gaussian": np.mean}[method]
        for i in range(min(pad, x.size)):
            out[i] = reduce(x[:i + 1])
    return out


def plot_avg_loss(loss_dict, ylim=None, smooth=None, fig=None, ax=None, phases=None):
    """Plot the average training loss over iterations. ``phases`` is an optional list of
    shaded regions, each ``{"start", "end", "color", "alpha", "label"}``, with a legend below."""
    created = fig is None and ax is None
    if created:
        fig, ax = make_subplot_grid(1, 1, PLOT_SIZE)
        ax = ax[0]

    # Average over the per-residual components (the res_* keys).
    r_keys = [k for k in loss_dict if k.startswith("res_")]
    total_avg = np.mean([np.asarray(loss_dict[k], dtype=float) for k in r_keys], axis=0)

    ax.plot(loss_dict["iteration"], _smooth(total_avg, smooth), color="C0", linewidth=LW_SINGLE)
    for ph in phases or []:
        ax.axvspan(ph["start"], ph["end"], color=ph["color"], alpha=ph["alpha"], label=ph["label"])
    ax.set_title("Average loss")
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel("Mean squared error")
    ax.set_xlabel("Iteration")

    if phases and created:
        handles, labels_leg = ax.get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc="outside lower center",
                   ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)

    fig, _ = style_fig_ax(fig, ax)
    return fig, ax


def plot_loss(loss_dict, residual_names=None, ylim=None, smooth=None, n_cols=N_COLS, phases=None,
              panel_h=None, avg_h=None):
    """Average loss spanning the top, then the per-residual losses in an ``n_cols``-column grid below,
    all on a log scale. ``smooth`` is ``(method, window)`` (see ``_smooth``) or ``None`` for raw.

    Args:
        residual_names: Display names for the per-equation panels (e.g. ["NKPC", "Bond Euler"]).
            Defaults to the ``res_*`` keys.
        phases: Optional shaded regions ``{"start", "end", "color", "alpha", "label"}``, drawn on
            every panel with a legend on the average panel.
        panel_h: Optional height override (inches) for the per-equation panels.
        avg_h: Optional height override (inches) for the average-loss row; defaults to ``panel_h``.
    """
    r_keys = [k for k in loss_dict if k.startswith("res_")]
    if residual_names is None:
        residual_names = r_keys

    total_avg = np.mean([np.asarray(loss_dict[k], dtype=float) for k in r_keys], axis=0)
    x = loss_dict["iteration"]

    # Mosaic: the average spans the full-width top row; per-equation panels fill the rows below.
    mosaic = [["avg"] * n_cols]
    for i in range(0, len(r_keys), n_cols):
        row = r_keys[i:i + n_cols]
        mosaic.append(row + ["."] * (n_cols - len(row)))
    pw, ph = panel_size(n_cols)
    if panel_h is not None:
        ph = panel_h
    top_h = avg_h if avg_h is not None else ph
    n_res_rows = len(mosaic) - 1
    fig, axd = plt.subplot_mosaic(mosaic, layout="constrained",
                                  figsize=(pw * n_cols, top_h + ph * n_res_rows),
                                  height_ratios=[top_h] + [ph] * n_res_rows)
    fig.get_layout_engine().set(hspace=0.18, wspace=0.06)

    axd["avg"].plot(x, _smooth(total_avg, smooth), linewidth=LW_MULTI)
    axd["avg"].set_title("Average loss")
    axd["avg"].set_yscale("log")
    axd["avg"].set_ylabel("Mean squared error")
    if ylim is not None:
        axd["avg"].set_ylim(ylim)
    for ph in phases or []:
        axd["avg"].axvspan(ph["start"], ph["end"], color=ph["color"], alpha=ph["alpha"], label=ph["label"])
    if phases:
        handles, labels_leg = axd["avg"].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc="outside lower center",
                   ncol=min(len(labels_leg), 3), frameon=True, edgecolor="none", framealpha=1)

    for k, (key, name) in enumerate(zip(r_keys, residual_names)):
        axd[key].plot(x, _smooth(loss_dict[key], smooth), linewidth=LW_MULTI)
        axd[key].set_title(f"{name} residual")
        axd[key].set_yscale("log")
        if k % n_cols == 0:
            axd[key].set_ylabel("Mean squared error")
        if k >= len(r_keys) - n_cols:
            axd[key].set_xlabel("Iteration")
        if ylim is not None:
            axd[key].set_ylim(ylim)
        for ph in phases or []:
            axd[key].axvspan(ph["start"], ph["end"], color=ph["color"], alpha=ph["alpha"])

    axes = list(axd.values())
    fig, _ = style_fig_ax(fig, axes)
    return fig, axes


# %% Particle filter / likelihood plots

def plot_pf_surrogate_loss(loss_dict, fig=None, ax=None):
    """Plot training and validation loss for the PF surrogate network (two panels)."""
    if fig is None and ax is None:
        fig, ax = make_subplot_grid(2, 2, DUO_SIZE)

    x = loss_dict.get("iteration")
    ax[0].plot(x, loss_dict["training_loss"], color="C0", linewidth=LW_MULTI)
    ax[0].set_title("Total loss — Training sample")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Mean squared error")
    ax[0].set_yscale("log")

    ax[1].plot(x, loss_dict["validation_loss"], color="C1", linewidth=LW_MULTI)
    ax[1].set_title("Total loss — Validation sample")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Mean squared error")
    ax[1].set_yscale("log")

    fig, _ = style_fig_ax(fig, ax)
    return fig, ax


def plot_ess(diagnostics, threshold=0.5, fig=None, ax=None):
    """Plot the particle filter's effective sample size (as a fraction of P) over the filter path.

    ``diagnostics`` is the dict stored on ``ParticleFilter.diagnostics`` after a ``filter`` run
    (keys ``ess`` and ``P``). A low ESS/P means the weights concentrate on few particles (degeneracy /
    impoverishment) — the metric to watch under the ZLB nonlinearity; the dashed ``threshold`` line is a
    visual guide. Accepts torch tensors or numpy arrays for ``ess``.
    """
    if fig is None and ax is None:
        fig, ax = make_subplot_grid(1, 1, PLOT_SIZE)
        ax = ax[0]

    ess = diagnostics["ess"]
    ess = ess.detach().cpu().numpy() if hasattr(ess, "detach") else np.asarray(ess)
    frac = ess / int(diagnostics["P"])
    step = np.arange(1, frac.size + 1)

    ax.plot(step, frac, color="C0", linewidth=LW_SINGLE)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=LW_MULTI, label=f"{threshold:g} threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Particle filter effective sample size")
    ax.set_xlabel("Filter step")
    ax.set_ylabel("ESS / P")
    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               frameon=True, edgecolor="none", framealpha=1)

    fig, _ = style_fig_ax(fig, ax)
    return fig, ax


def _posterior_keys(result):
    """Return (key_main, key_full, key_random, grid_random_key, kind_label) for a result dict."""
    has_posterior = "posterior_surrogate" in result
    if has_posterior:
        return "posterior_surrogate", "posterior_full", "posterior_random", "grid_random", "posterior"
    return "ll_grid_surrogate", "ll_grid_full", "ll_grid_random", "grid_random", "likelihood"


def plot_posterior_single(result, par_name, true_value=None, full=True, random=True,
                          par_labels=None, y_buffer=25, n_xticks=9, fig=None, ax=None):
    """Plot likelihood or posterior for a single parameter.

    Uses posterior_* keys if present (i.e. `add_log_prior` was applied),
    else ll_grid_* keys. Axis labels adapt accordingly.
    """
    if fig is None and ax is None:
        fig, ax = make_subplot_grid(1, 1, PLOT_SIZE)
        ax = ax[0]

    key_main, key_full, key_random, grid_random_key, kind = _posterior_keys(result)

    ax.plot(result["grid"], result[key_main], color="C0", linewidth=LW_SINGLE, label="Neural Network")

    if full and key_full in result:
        ax.plot(result["grid"], result[key_full], linestyle="dashed", color="C1", linewidth=LW_SINGLE, label="Particle Filter (full)")

    if random and key_random in result:
        ax.scatter(result[grid_random_key], result[key_random], marker="o", s=25, alpha=0.75,
                   color="C3", label="Particle Filter", zorder=1)

    if true_value is not None:
        ax.axvline(x=true_value, color="C2", linestyle="dotted", label="True value")

    if par_labels and par_name in par_labels:
        ylabel, latex = par_labels[par_name]
    else:
        ylabel, latex = par_name, par_name
    if kind == "likelihood":
        ax.set_title(rf"Log likelihood $\log \mathcal{{L}}_{{\bar{{\Theta}}}}$ conditioned on {latex}")
    else:
        ax.set_title(f"Log {kind} conditioned on {latex}")
    ax.set_ylabel(f"Log {kind}")
    ax.set_xlabel(f"{ylabel} ({latex})")
    # n_xticks evenly spanning the bounds — pick a count that divides the range into round steps.
    x_lo, x_hi = float(min(result["grid"])), float(max(result["grid"]))
    ax.set_xticks([x_lo + i * (x_hi - x_lo) / (n_xticks - 1) for i in range(n_xticks)])

    # Range over the curves only (as in the paper's figures) — the scatter dots don't drive the axis.
    all_ll = list(result[key_main])
    if full and key_full in result:
        all_ll.extend(list(result[key_full]))
    ax.set_ylim([min(all_ll) - y_buffer, max(all_ll) + y_buffer])
    ax.set_xlim([min(result["grid"]), max(result["grid"])])

    handles, labels_leg = ax.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3),
               frameon=True, edgecolor="none", framealpha=1)

    fig, _ = style_fig_ax(fig, [ax])
    return fig, ax


def plot_posterior_parameters(results, true_values=None, ml_values=None, par_labels=None,
                              y_buffer=None, ylim=None, n_cols=N_COLS, center_last_row=True,
                              panel_titles=True, main_label="Neural Network", comparison=None,
                              comparison_label="Time Iteration", comparison_color="C1",
                              comparison_style="dashed"):
    """Plot likelihood or posterior for all parameters in a grid.

    Uses posterior_* keys if present, else ll_grid_*. Axis labels adapt. At
    most one of the full/random PF overlays is expected to be set per result.

    Args:
        results: Dict of par_name -> compute_likelihood (+ optional add_log_prior) output.
        true_values: Dict mapping par_name -> true value (for vertical lines).
        par_labels: Dict mapping par_name -> (display_name, latex).
        y_buffer: Fixed y-padding in log-points per panel; None (default) pads each
            panel by 5% of its own range, so every panel's axis fits its curve.
        ylim: Common y-limits for all panels (overrides y_buffer).
        n_cols: Number of columns in the grid.
        center_last_row: Center subplots when the last row is incomplete.
        comparison: Optional dict par_name -> {"grid": array, "posterior": array}
            overlaying a reference curve (e.g. time-iteration result).
        comparison_label: Legend label for the comparison line.
    """
    n_par = len(results)
    fig, axes = make_subplot_grid(n_par, n_cols, panel_size(n_cols), center_last_row=center_last_row)

    for i, (par_name, result) in enumerate(results.items()):
        a = axes[i]
        key_main, key_full, key_random, grid_random_key, kind = _posterior_keys(result)

        a.plot(result["grid"], result[key_main], color="C0", label=main_label, linewidth=LW_MULTI)
        if ml_values and par_name in ml_values:
            a.axvline(x=ml_values[par_name], color="C0", linestyle="dashed", linewidth=LW_MULTI,
                      label="Maximum likelihood")
        if result.get(key_full) is not None:
            a.plot(result["grid"], result[key_full], color="C1", linestyle="dashed",
                   label="Particle Filter", linewidth=LW_MULTI)
        if result.get(key_random) is not None:
            a.scatter(result[grid_random_key], result[key_random], marker="o", s=25, alpha=0.75,
                      color="C3", label="Particle Filter", zorder=1)
        if comparison and par_name in comparison:
            a.plot(comparison[par_name]["grid"], comparison[par_name]["posterior"],
                   color=comparison_color, linestyle=comparison_style, label=comparison_label,
                   linewidth=LW_MULTI)
        if true_values and par_name in true_values:
            a.axvline(x=true_values[par_name], color="C2", linestyle="dotted", label="True value")

        if ylim is not None:
            a.set_ylim(ylim)
        else:
            # Range over the curves only (as in the paper's figures) — the scatter dots don't drive the axis.
            all_ll = list(result[key_main])
            if result.get(key_full) is not None:
                all_ll.extend(list(result[key_full]))
            if comparison and par_name in comparison:
                all_ll.extend(list(comparison[par_name]["posterior"]))
            pad = y_buffer if y_buffer is not None else (0.05 * (max(all_ll) - min(all_ll)) or 1.0)
            a.set_ylim([min(all_ll) - pad, max(all_ll) + pad])
            # Integer ticks keep the labels short; sub-log-point decimals carry no information.
            if max(all_ll) - min(all_ll) > 2:
                a.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        a.autoscale(enable=True, axis="x", tight=True)

        if par_labels and par_name in par_labels:
            ylabel, latex = par_labels[par_name]
        else:
            ylabel, latex = par_name, par_name
        if panel_titles and kind == "likelihood":
            a.set_title(rf"Log Likelihood $\log \mathcal{{L}}_{{\bar{{\Theta}}}}$ conditioned on {latex}")
        elif panel_titles:
            a.set_title(f"Log {kind} conditioned on {latex}")
        a.set_xlabel(f"{ylabel} ({latex})")
        if i % n_cols == 0:
            a.set_ylabel(f"Log {kind}")
        a.xaxis.set_major_locator(plt.MaxNLocator(5))

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="outside lower center",
               ncol=min(len(labels_leg), 3),
               frameon=True, edgecolor="none", framealpha=1)

    fig, _ = style_fig_ax(fig, axes)
    return fig, axes
