import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from .utils_data_processing import exclude_arrests_from_series_at_ecdysis
from .utils_plotting import build_legend
from .utils_plotting import get_colors


def plot_heterogeneity_at_ecdysis(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    remove_hatch=True,
    legend=None,
    colors=None,
    x_axis_label=None,
    y_axis_label=None,
    exclude_arrests: bool = False,
):
    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        if remove_hatch:
            values = values[:, 1:]
        if exclude_arrests:
            values = exclude_arrests_from_series_at_ecdysis(values)
        cvs = []
        for j in range(values.shape[1]):
            values_at_ecdysis = values[:, j]
            cv = np.nanstd(values_at_ecdysis) / np.nanmean(values_at_ecdysis)
            cvs.append(cv)
        label = build_legend(condition_dict, legend)
        plt.plot(cvs, label=label, marker="o", color=color_palette[i])
    # replace the ticks by [L1, L2, L3, L4]
    if remove_hatch:
        plt.xticks(range(4), ["M1", "M2", "M3", "M4"])
    else:
        plt.xticks(range(5), ["Hatch", "M1", "M2", "M3", "M4"])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig


def plot_heterogeneity_rescaled_data(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    smooth: bool = False,
    remove_hatch=True,
    legend=None,
    colors=None,
    x_axis_label=None,
    y_axis_label=None,
    exclude_arrests: bool = False,
):
    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        cvs = np.nanstd(values, axis=0) / np.nanmean(values, axis=0)
        label = build_legend(condition_dict, legend)
        if smooth:
            cvs = medfilt(cvs, 7)
            # cvs = savgol_filter(cvs, 15, 3)
        plt.plot(cvs, label=label, color=color_palette[i])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig
