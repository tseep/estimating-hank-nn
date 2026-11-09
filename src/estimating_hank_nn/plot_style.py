"""Plot styling: colors, figure constants, rcParams, and axis formatting."""

import builtins

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator

_interactive = hasattr(builtins, "__IPYTHON__")


# %% Save figure
def save_figure(fig, file, close=True, show_plot=True, format_list=["pdf", "png", "jpg"], dpi=300):
    for format in format_list:
        # Raster (png/jpg) on white; pdf transparent. jpg = low compression.
        if format in ("jpg", "jpeg"):
            save_kwargs = {"facecolor": "white", "pil_kwargs": {"quality": 95, "subsampling": 0}}
        elif format == "pdf":
            save_kwargs = {"transparent": True}
        else:
            save_kwargs = {"facecolor": "white"}
        fig.savefig(file + "." + format, format=format, dpi=dpi, **save_kwargs)
    if close is True:
        if show_plot is True and _interactive:
            plt.show()
        plt.close(fig)
    return


# %% Colors
colors = {
    "blue": "#0000FF",
    "red": "#FF0000",
    "black": "#000000",
    "orange": "#dd8453",
    "green": "#55a868",
    "purple": "#8172b2",
    "brown": "#a65628",
    "pink": "#f781bf",
}


# %% Figure and line constants
N_COLS = 2                     # default columns in multi-panel grids (override per call / in main_*)
PLOT_SIZE = (8, 4)             # single full-width plot (w, h)
MULTI_SIZE = (8 / 3, 2)        # per-panel for 3+-column grids (8/3 × 3 = 8" at 3 cols; 4+ cols run wider)
DUO_SIZE = (4, 3)              # per-panel for 2-column grids (4 × 2 = 8")
STACK_SIZE = (8, 2)            # full-width stacked row (w, h)

LW_SINGLE = 2.0   # line width for single-panel plots
LW_MULTI = 1.5    # line width for multi-panel plots


def panel_size(n_cols):
    """Per-panel (width, height) for an ``n_cols`` multi-panel grid: the 2-column config (``DUO_SIZE``)
    at ``n_cols == 2``, the 3-column config (``MULTI_SIZE``) at 3 or more. So 2- and 3-column grids are
    both 8" wide; 4+ columns keep the 3-col panel width and run wider."""
    return DUO_SIZE if n_cols == 2 else MULTI_SIZE


# %% RC params and axis styling
def set_rc_params(**kwargs):
    """Set matplotlib rcParams for consistent plot styling."""
    mpl.rcParams["font.size"] = 9
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(colors.values()))

    for key, value in kwargs.items():
        mpl.rcParams[key] = value
    return


def make_subplot_grid(n_items, n_cols, subplot_size, center_last_row=True):
    """Create a figure with a grid of subplots.

    Uses constrained_layout for automatic spacing (handles legends, labels).
    Incomplete last rows are centered via a narrower subfigure.

    Args:
        n_items: Number of subplots.
        n_cols: Number of columns.
        subplot_size: (width, height) per subplot in inches.
        center_last_row: Center subplots when the last row is incomplete.

    Returns:
        fig, list of axes (length n_items).
    """
    n_rows = (n_items + n_cols - 1) // n_cols
    remainder = n_items % n_cols
    figsize = (subplot_size[0] * n_cols, subplot_size[1] * n_rows)

    if not center_last_row or remainder == 0:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize,
                               layout="constrained")
        fig.get_layout_engine().set(hspace=0.18, wspace=0.06)
        axes_flat = np.atleast_1d(ax).flatten()
        for i in range(n_items, n_rows * n_cols):
            fig.delaxes(axes_flat[i])
        return fig, list(axes_flat[:n_items])

    # Centered last row: doubled gridspec positions one subfigure per row.
    # Last-row subfigure spans (remainder/n_cols) of the width, centered.
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(hspace=0.18, wspace=0.06)
    outer = fig.add_gridspec(n_rows, 2 * n_cols)

    axes = []
    for r in range(n_rows):
        if r < n_rows - 1:
            subfig = fig.add_subfigure(outer[r, :])
            row_axes = subfig.subplots(1, n_cols)
        else:
            offset = n_cols - remainder
            subfig = fig.add_subfigure(outer[r, offset:2 * n_cols - offset])
            row_axes = subfig.subplots(1, remainder)
        axes.extend(np.atleast_1d(row_axes))
    return fig, list(axes)


def style_fig_ax(fig, ax, xminor=5, yminor=0):
    """Apply grid and minor tick styling to axes."""
    ax = np.array(ax)
    for a in ax.flatten():
        a.minorticks_on()
        a.grid(which="major", alpha=0.3)
        a.grid(which="minor", linestyle=":", linewidth=0.5, color="gray", alpha=0.3)
        if xminor > 0:
            a.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        else:
            a.xaxis.set_minor_locator(NullLocator())

        if yminor > 0:
            a.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        else:
            a.yaxis.set_minor_locator(NullLocator())

    return fig, ax
