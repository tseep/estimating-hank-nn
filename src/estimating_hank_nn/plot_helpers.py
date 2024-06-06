import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator


def save_figure(fig, file, close=True, show_plot=True, format_list=["pdf", "png"], dpi=500):
    for format in format_list:
        fig.savefig(file + "." + format, format=format, dpi=dpi)
    if close is True:
        if show_plot is True:
            plt.show()
        plt.close(fig)
    return


colors = {"blue": "blue", "red": "red", "black": "black", "orange": "#dd8453"}


def set_rc_params(**kwargs):
    # Font
    mpl.rcParams["font.size"] = 9

    # Ticks
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7

    # Pallette
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(colors.values()))

    for key, value in kwargs.items():
        mpl.rcParams[key] = value
    return


# Function to apply some ax styles that are not in rcParams
def style_fig_ax(fig, ax, xminor=5, yminor=0):
    ax = np.array(ax)
    for a in ax.flatten():
        a.minorticks_on()  # Enable minor ticks
        a.grid(which="major", alpha=0.5)
        a.grid(which="minor", alpha=0.2)
        if xminor > 0:
            a.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        else:
            a.xaxis.set_minor_locator(NullLocator())

        if yminor > 0:
            a.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        else:
            a.yaxis.set_minor_locator(NullLocator())

    return fig, ax
