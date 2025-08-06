import matplotlib.pyplot as plt
import numpy as np
from towbintools.data_analysis import rescale_and_aggregate
from towbintools.data_analysis.time_series import (
    smooth_series_classified,
)

from .utils_plotting import build_legend
from .utils_plotting import get_colors
from .utils_plotting import set_scale


def plot_aggregated_series(
    conditions_struct,
    series_column,
    conditions_to_plot,
    experiment_time=True,
    aggregation="mean",
    n_points=100,
    time_step=10,
    log_scale=True,
    colors=None,
    legend=None,
    y_axis_label=None,
):
    color_palette = get_colors(conditions_to_plot, colors)

    def plot_single_series(column: str):
        for i, condition_id in enumerate(conditions_to_plot):
            condition_dict = conditions_struct[condition_id]
            if experiment_time:
                time = condition_dict["experiment_time_hours"]
                larval_stage_durations = condition_dict[
                    "larval_stage_durations_experiment_time_hours"
                ]
            else:
                time = condition_dict["time"]
                larval_stage_durations = condition_dict[
                    "larval_stage_durations_time_step"
                ]
            # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
            worm_type_key = [
                key for key in condition_dict.keys() if "worm_type" in key
            ][0]
            rescaled_time, aggregated_series, _, ste_series = rescale_and_aggregate(
                condition_dict[column],
                time,
                condition_dict["ecdysis_index"],
                larval_stage_durations,
                condition_dict[worm_type_key],
                aggregation=aggregation,
                n_points=n_points,
            )
            ci_lower = aggregated_series - 1.96 * ste_series
            ci_upper = aggregated_series + 1.96 * ste_series
            if not experiment_time:
                rescaled_time = rescaled_time * time_step / 60
            label = build_legend(condition_dict, legend)
            plt.plot(
                rescaled_time, aggregated_series, color=color_palette[i], label=label
            )
            plt.fill_between(
                rescaled_time, ci_lower, ci_upper, color=color_palette[i], alpha=0.2
            )

    if isinstance(series_column, list):
        for column in series_column:
            plot_single_series(column)
    else:
        plot_single_series(series_column)
    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("time (h)")
    plt.yscale("log" if log_scale else "linear")
    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(series_column)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_growth_curves_individuals(
    conditions_struct,
    column,
    conditions_to_plot,
    share_y_axis,
    log_scale=True,
    figsize=None,
    legend=None,
    y_axis_label=None,
    cut_after=None,
):
    if figsize is None:
        figsize = (len(conditions_to_plot) * 8, 10)
    fig, ax = plt.subplots(
        1, len(conditions_to_plot), figsize=figsize, sharey=share_y_axis
    )
    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]
        # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
        worm_type_key = [key for key in condition_dict.keys() if "worm_type" in key][0]
        for j in range(len(condition_dict[column])):
            time = condition_dict["experiment_time"][j] / 3600
            data = condition_dict[column][j]
            worm_type = condition_dict[worm_type_key][j]
            hatch = condition_dict["ecdysis_time_step"][j][0]
            hatch_experiment_time = (
                condition_dict["ecdysis_experiment_time"][j][0] / 3600
            )
            if not np.isnan(hatch):
                hatch = int(hatch)
                if cut_after is not None:
                    indexes_to_cut = np.where(time > cut_after)[0]
                    if len(indexes_to_cut) > 0:
                        data = data[: indexes_to_cut[0] + 1]
                        time = time[: indexes_to_cut[0] + 1]
                        worm_type = worm_type[: indexes_to_cut[0] + 1]

                time = time[hatch:]
                time = time - hatch_experiment_time
                data = data[hatch:]
                worm_type = worm_type[hatch:]
                filtered_data = smooth_series_classified(
                    data,
                    time,
                    worm_type,
                )
                label = build_legend(condition_dict, legend)
                try:
                    ax[i].plot(time, filtered_data)
                    set_scale(ax[i], log_scale)
                except TypeError:
                    ax.plot(time, filtered_data)
                    set_scale(ax, log_scale)
        try:
            ax[i].title.set_text(label)
        except TypeError:
            ax.title.set_text(label)
    # Set labels
    if y_axis_label is not None:
        try:
            ax[0].set_ylabel(y_axis_label)
            ax[0].set_xlabel("Time (h)")
        except TypeError:
            ax.set_ylabel(y_axis_label)
            ax.set_xlabel("Time (h)")
    else:
        try:
            ax[0].set_ylabel(column)
            ax[0].set_xlabel("Time (h)")
        except TypeError:
            ax.set_ylabel(column)
            ax.set_xlabel("Time (h)")
    fig = plt.gcf()
    plt.show()
    return fig
