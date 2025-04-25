from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from .utils_data_processing import rescale_without_flattening
from .utils_plotting import build_legend
from .utils_plotting import get_colors


def _setup_figure(
    df,
    conditions_struct,
    conditions_to_plot,
    legend,
    color_palette,
    figsize,
    titles,
    share_y_axis,
):
    # Determine figure size
    if figsize is None:
        figsize = (6 * df["Event"].nunique(), 10)
    if titles is not None and len(titles) != df["Event"].nunique():
        print("Number of titles does not match the number of ecdysis events.")
        titles = None

    fig, ax = plt.subplots(
        1,
        df["Event"].nunique(),
        figsize=(figsize[0] + 3, figsize[1]),
        sharey=False,
    )

    return fig, ax


def _annotate_significance(
    df, conditions_to_plot, column, boxplot, significance_pairs, event_index
):
    if significance_pairs is None:
        pairs = list(combinations(df["Condition"].unique(), 2))
    else:
        pairs = significance_pairs
    annotator = Annotator(
        ax=boxplot,
        pairs=pairs,
        data=df[df["Event"] == event_index],
        x="Condition",
        order=conditions_to_plot,
        y=column,
    )
    annotator.configure(
        test="Mann-Whitney", text_format="star", loc="inside", verbose=False
    )
    annotator.apply_and_annotate()


def _plot_boxplot(
    df,
    conditions_to_plot,
    column,
    color_palette,
    ax,
    titles,
    share_y_axis,
    plot_significance,
    significance_pairs,
):
    y_min, y_max = [], []
    for event_index in range(df["Event"].nunique()):
        if share_y_axis:
            if event_index > 0:
                ax[event_index].tick_params(
                    axis="y", which="both", left=False, labelleft=False
                )

        boxplot = sns.boxplot(
            data=df[df["Event"] == event_index],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue_order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            showfliers=False,
            ax=ax[event_index],
            dodge=False,
            linewidth=2,
            legend="full",
        )

        sns.stripplot(
            data=df[df["Event"] == event_index],
            x="Condition",
            order=conditions_to_plot,
            y=column,
            ax=ax[event_index],
            alpha=0.5,
            color="black",
            dodge=True,
        )

        ax[event_index].set_xlabel("")
        # Hide y-axis labels and ticks for all subplots except the first one
        if event_index > 0:
            ax[event_index].set_ylabel("")

        if titles is not None:
            ax[event_index].set_title(titles[event_index])

        # remove ticks
        ax[event_index].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            _annotate_significance(
                df, conditions_to_plot, column, boxplot, significance_pairs, event_index
            )

        min_y, max_y = ax[event_index].get_ylim()
        y_min.append(min_y)
        y_max.append(max_y)

    return y_min, y_max


def _set_all_y_limits(ax, y_min, y_max):
    global_min = min(y_min)
    global_max = max(y_max)
    range_padding = (global_max - global_min) * 0.05  # 5% padding
    global_min = global_min - range_padding
    global_max = global_max + range_padding
    for i in range(len(ax)):
        ax[i].set_ylim(global_min, global_max)


def _set_labels_and_legend(
    ax,
    fig,
    conditions_struct,
    conditions_to_plot,
    column,
    y_axis_label,
    legend,
):
    # Set y label for the first plot
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    # Add legend to the right of the subplots
    legend_labels = [
        build_legend(conditions_struct[condition_id], legend)
        for condition_id in conditions_to_plot
    ]

    legend_handles = ax[0].get_legend_handles_labels()[0]

    # Remove the legend from all subplots
    for i in range(len(ax)):
        ax[i].legend_.remove()

    # Place legend to the right of the subplots
    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.9, 0.5),
        loc="center left",
        title=None,
        frameon=True,
    )


def boxplot_at_molt(
    conditions_struct,
    column,
    conditions_to_plot,
    remove_hatch=False,
    log_scale: bool = True,
    figsize: tuple = None,
    colors=None,
    plot_significance: bool = False,
    significance_pairs=None,
    legend=None,
    y_axis_label=None,
    titles=None,
    share_y_axis: bool = False,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        range_start = 1 if remove_hatch else 0
        for j in range(range_start, data.shape[1]):
            for value in data[:, j]:
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Event": j,
                        column: np.log(value) if log_scale else value,
                    }
                )
    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        conditions_struct,
        conditions_to_plot,
        legend,
        color_palette,
        figsize,
        titles,
        share_y_axis,
    )

    y_min, y_max = _plot_boxplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)
        # set the figure to sharey
        for i in range(len(ax)):
            ax[i].sharey(ax[0])

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig


def boxplot_larval_stage(
    conditions_struct,
    column,
    conditions_to_plot,
    aggregation: str = "mean",
    n_points: int = 100,
    fraction: tuple[float, float] = (0.2, 0.8),
    log_scale: bool = True,
    figsize: tuple = None,
    colors=None,
    plot_significance: bool = False,
    significance_pairs=None,
    significance_position="inside",
    legend=None,
    y_axis_label=None,
    titles=None,
    share_y_axis: bool = False,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    if "rescaled" not in column:
        rescaled_column = column + "_rescaled"
        conditions_struct = rescale_without_flattening(
            conditions_struct, column, rescaled_column, aggregation, n_points
        )
        column = rescaled_column

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for i in range(data.shape[1]):
            data_of_stage = data[:, i]
            data_of_stage = data_of_stage[
                :,
                int(fraction[0] * data_of_stage.shape[1]) : int(
                    fraction[1] * data_of_stage.shape[1]
                ),
            ]

            data_of_stage = np.log(data_of_stage) if log_scale else data_of_stage
            if aggregation == "mean":
                aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
            elif aggregation == "median":
                aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)

            for j in range(aggregated_data_of_stage.shape[0]):
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Event": i,
                        column: aggregated_data_of_stage[j],
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        conditions_struct,
        conditions_to_plot,
        legend,
        color_palette,
        figsize,
        titles,
        share_y_axis,
    )

    y_min, y_max = _plot_boxplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig
