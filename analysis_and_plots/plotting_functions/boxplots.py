from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator

from .utils_data_processing import rescale_without_flattening
from .utils_plotting import build_legend


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
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors
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
                        "Molt": j,
                        column: np.log(value) if log_scale else value,
                    }
                )
    df = pd.DataFrame(data_list)

    # Determine figure size
    if figsize is None:
        figsize = (6 * df["Molt"].nunique(), 10)
    if titles is not None and len(titles) != df["Molt"].nunique():
        print("Number of titles does not match the number of ecdysis events.")
        titles = None

    # Create figure with extra space on the right for legend
    fig, ax = plt.subplots(
        1,
        df["Molt"].nunique(),
        figsize=(figsize[0] + 3, figsize[1]),
        sharey=share_y_axis,
    )

    # Create a dummy plot to get proper legend handles
    dummy_ax = fig.add_axes([0, 0, 0, 0])
    for i, condition in enumerate(conditions_to_plot):
        dummy_ax.boxplot(
            [],
            [],
            patch_artist=True,
            label=build_legend(conditions_struct[condition], legend),
        )
        for j, patch in enumerate(dummy_ax.patches):
            patch.set_facecolor(color_palette[j])
    dummy_ax.set_visible(False)

    for i in range(df["Molt"].nunique()):
        if share_y_axis:
            if i > 0:
                ax[i].tick_params(axis="y", which="both", left=False, labelleft=False)

        boxplot = sns.boxplot(
            data=df[df["Molt"] == i],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            showfliers=False,
            ax=ax[i],
            dodge=False,
            linewidth=2,
            legend=False,
        )

        sns.stripplot(
            data=df[df["Molt"] == i],
            x="Condition",
            order=conditions_to_plot,
            y=column,
            ax=ax[i],
            alpha=0.5,
            color="black",
            dodge=True,
        )

        ax[i].set_xlabel("")
        # Hide y-axis labels and ticks for all subplots except the first one
        if i > 0:
            ax[i].set_ylabel("")

        if titles is not None:
            ax[i].set_title(titles[i])

        # remove ticks
        ax[i].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            if significance_pairs is None:
                pairs = list(combinations(df["Condition"].unique(), 2))
            else:
                pairs = significance_pairs
            annotator = Annotator(
                ax=boxplot,
                pairs=pairs,
                data=df[df["Molt"] == i],
                x="Condition",
                order=conditions_to_plot,
                y=column,
            )
            annotator.configure(
                test="Mann-Whitney", text_format="star", loc="inside", verbose=False
            )
            annotator.apply_and_annotate()

        y_min, y_max = ax[i].get_ylim()

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
    legend_handles = dummy_ax.get_legend_handles_labels()[0]

    # Place legend to the right of the subplots
    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.9, 0.5),
        loc="center left",
        title=None,
        frameon=True,
    )

    if share_y_axis:
        global_min = y_min
        global_max = y_max
        range_padding = (global_max - global_min) * 0.05  # 5% padding
        global_min = global_min - range_padding
        global_max = global_max + range_padding
        for i in range(df["Molt"].nunique()):
            ax[i].set_ylim(global_min, global_max)

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
    fraction: float = 0.8,
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
    new_column = column + "_rescaled"
    struct = rescale_without_flattening(
        conditions_struct, column, new_column, aggregation, n_points
    )
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors
    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = struct[condition_id]
        data = condition_dict[new_column]
        for i in range(data.shape[1]):
            data_of_stage = data[:, i]
            data_of_stage = data_of_stage[:, 0 : int(fraction * data_of_stage.shape[1])]

            data_of_stage = np.log(data_of_stage) if log_scale else data_of_stage
            if aggregation == "mean":
                aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
            elif aggregation == "median":
                aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)

            for j in range(aggregated_data_of_stage.shape[0]):
                data_list.append(
                    {
                        "Condition": condition_id,
                        "LarvalStage": i,
                        column: aggregated_data_of_stage[j],
                    }
                )

    df = pd.DataFrame(data_list)

    # Determine figure size
    if figsize is None:
        figsize = (6 * df["LarvalStage"].nunique(), 10)
    if titles is not None and len(titles) != df["LarvalStage"].nunique():
        print("Number of titles does not match the number of ecdysis events.")
        titles = None

    # Create figure with extra space on the right for legend
    fig, ax = plt.subplots(
        1,
        df["LarvalStage"].nunique(),
        figsize=(figsize[0] + 3, figsize[1]),
        sharey=share_y_axis,
    )

    # Create a dummy plot to get proper legend handles
    dummy_ax = fig.add_axes([0, 0, 0, 0])
    for i, condition in enumerate(conditions_to_plot):
        dummy_ax.boxplot(
            [], [], patch_artist=True, label=build_legend(struct[condition], legend)
        )
        for j, patch in enumerate(dummy_ax.patches):
            patch.set_facecolor(color_palette[j])
    dummy_ax.set_visible(False)

    for i in range(df["LarvalStage"].nunique()):
        if share_y_axis:
            if i > 0:
                ax[i].tick_params(axis="y", which="both", left=False, labelleft=False)

        boxplot = sns.boxplot(
            data=df[df["LarvalStage"] == i],
            x="Condition",
            y=column,
            hue="Condition",
            order=conditions_to_plot,
            palette=color_palette,
            showfliers=False,
            ax=ax[i],
            dodge=False,
            linewidth=2,
            legend=False,
        )

        sns.stripplot(
            data=df[df["LarvalStage"] == i],
            x="Condition",
            order=conditions_to_plot,
            y=column,
            ax=ax[i],
            alpha=0.5,
            color="black",
            dodge=True,
        )

        ax[i].set_xlabel("")
        # Hide y-axis labels and ticks for all subplots except the first one
        if i > 0:
            ax[i].set_ylabel("")

        if titles is not None:
            ax[i].set_title(titles[i])

        # remove ticks
        ax[i].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            if significance_pairs is None:
                pairs = list(combinations(df["Condition"].unique(), 2))
            else:
                pairs = significance_pairs
            annotator = Annotator(
                ax=boxplot,
                pairs=pairs,
                data=df[df["LarvalStage"] == i],
                x="Condition",
                order=conditions_to_plot,
                y=column,
            )
            annotator.configure(
                test="Mann-Whitney",
                text_format="star",
                loc=significance_position,
                verbose=False,
            )
            annotator.apply_and_annotate()

        y_min, y_max = ax[i].get_ylim()

    # Set y label for the first plot
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    # Add legend to the right of the subplots
    legend_labels = [
        build_legend(struct[condition_id], legend)
        for condition_id in conditions_to_plot
    ]
    legend_handles = dummy_ax.get_legend_handles_labels()[0]

    # Place legend to the right of the subplots
    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.9, 0.5),
        loc="center left",
        title=None,
        frameon=True,
    )

    if share_y_axis:
        global_min = y_min
        global_max = y_max
        range_padding = (global_max - global_min) * 0.05  # 5% padding
        global_min = global_min - range_padding
        global_max = global_max + range_padding
        for i in range(df["LarvalStage"].nunique()):
            ax[i].set_ylim(global_min, global_max)

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig
