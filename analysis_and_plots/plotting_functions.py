import os
import shutil
from collections import defaultdict
from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import starbars
import statsmodels.api as sm
import yaml
from scipy.interpolate import make_interp_spline
from scipy.stats import mannwhitneyu
from towbintools.data_analysis import (
    aggregate_interpolated_series,
    compute_growth_rate_per_larval_stage,
    compute_larval_stage_duration,
    compute_series_at_time_classified,
    correct_series_with_classification,
    filter_series_with_classification,
    rescale_and_aggregate,
    rescale_series,
)

from microfilm.microplot import microshow

from towbintools.foundation.file_handling import get_dir_filemap
from typing import Dict, List, Tuple, Any
from towbintools.foundation.image_handling import read_tiff_file
from tifffile import imwrite


# BUILDING THE PLOTTING STRUCTURE


def build_conditions(config):
    conditions = []
    condition_id = 0

    for condition in config["conditions"]:
        condition = {
            key: [val] if not isinstance(val, list) else val
            for key, val in condition.items()
        }

        lengths = set(len(val) for val in condition.values())
        if len(lengths) > 2 or (len(lengths) == 2 and 1 not in lengths):
            raise ValueError(
                "All lists in the condition must have the same length or be of length 1."
            )

        max_length = max(lengths)
        for i in range(max_length):
            condition_dict = {
                key: val[0] if len(val) == 1 else val[i]
                for key, val in condition.items()
            }
            condition_dict["condition_id"] = condition_id
            conditions.append(condition_dict)
            condition_id += 1

    return conditions


def add_conditions_to_filemap(experiment_filemap, conditions, config):
    for condition in conditions:
        if "point_range" in condition.keys():
            point_range = condition["point_range"]

            # check if point range is a list of lists
            if isinstance(point_range[0], list):
                for pr in point_range:
                    # Get all the rows that are in the point range
                    condition_rows = experiment_filemap[
                        experiment_filemap["Point"].between(pr[0], pr[1])
                    ]
                    # Remove the point range from the condition
                    conditions_to_add = {
                        key: val
                        for key, val in condition.items()
                        if key != "point_range"
                    }
                    for key, val in conditions_to_add.items():
                        # Directly fill the rows with the value for the new or existing column
                        experiment_filemap.loc[condition_rows.index, key] = val
            else:
                # Get all the rows that are in the point range
                condition_rows = experiment_filemap[
                    experiment_filemap["Point"].between(point_range[0], point_range[1])
                ]
                # Remove the point range from the condition
                conditions_to_add = {
                    key: val for key, val in condition.items() if key != "point_range"
                }
                for key, val in conditions_to_add.items():
                    # Directly fill the rows with the value for the new or existing column
                    experiment_filemap.loc[condition_rows.index, key] = val

        elif "pad" in condition.keys():
            pad = condition["pad"]
            # Get all the rows that are in the pad
            condition_rows = experiment_filemap[experiment_filemap["Pad"] == pad]
            # Remove the pad from the condition
            conditions_to_add = {
                key: val for key, val in condition.items() if key != "pad"
            }
            for key, val in conditions_to_add.items():
                # Directly fill the rows with the value for the new or existing column
                experiment_filemap.loc[condition_rows.index, key] = val

        else:
            print(
                "Condition does not contain 'point_range' or 'pad' key, impossible to add condition to filemap, skipping."
            )
    return experiment_filemap


def get_ecdysis_and_durations(filemap):
    all_ecdysis_time_step = []
    all_durations_time_step = []

    all_ecdysis_experiment_time = []
    all_durations_experiment_time = []

    for point in filemap["Point"].unique():
        point_df = filemap[filemap["Point"] == point]
        point_ecdysis = point_df[["HatchTime", "M1", "M2", "M3", "M4"]].iloc[0]
        larval_stage_durations = list(
            compute_larval_stage_duration(point_ecdysis).values()
        )

        point_ecdysis = point_ecdysis.to_numpy()
        all_ecdysis_time_step.append(point_ecdysis)
        all_durations_time_step.append(larval_stage_durations)

        ecdysis_experiment_time = []
        for ecdys in point_ecdysis:
            if np.isnan(ecdys):
                ecdysis_experiment_time.append(np.nan)
            else:
                ecdys_experiment_time = point_df[point_df["Time"] == ecdys][
                    "ExperimentTime"
                ].iloc[0]
                ecdysis_experiment_time.append(ecdys_experiment_time)

        all_ecdysis_experiment_time.append(ecdysis_experiment_time)

        durations_experiment_time = []
        for i in range(len(ecdysis_experiment_time) - 1):
            start = ecdysis_experiment_time[i]
            end = ecdysis_experiment_time[i + 1]
            duration_experiment_time = end - start
            durations_experiment_time.append(duration_experiment_time)

        all_durations_experiment_time.append(durations_experiment_time)

    return (
        np.array(all_ecdysis_time_step),
        np.array(all_durations_time_step),
        np.array(all_ecdysis_experiment_time),
        np.array(all_durations_experiment_time),
    )


def separate_column_by_point(filemap, column):
    separated_column = []
    for point in filemap["Point"].unique():
        point_df = filemap[filemap["Point"] == point]
        separated_column.append(point_df[column].values)
    return np.array(separated_column)


def build_plotting_struct(
    experiment_dir, filemap_path, config_path, organ_channels={"body": 2, "pharynx": 1}
):

    experiment_filemap = pd.read_csv(filemap_path)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        file.close()

    conditions = build_conditions(config)
    conditions_keys = list(conditions[0].keys())

    # remove 'point_range' and 'pad' from the conditions keys if they are present
    if "point_range" in conditions_keys:
        conditions_keys.remove("point_range")
    if "pad" in conditions_keys:
        conditions_keys.remove("pad")

    experiment_filemap = add_conditions_to_filemap(
        experiment_filemap, conditions, config
    )

    experiment_filemap.columns

    # if ExperimentTime is not present in the filemap, add it
    if "ExperimentTime" not in experiment_filemap.columns:
        experiment_filemap["ExperimentTime"] = np.nan

    # remove rows where condition_id is NaN
    experiment_filemap = experiment_filemap[
        ~experiment_filemap["condition_id"].isnull()
    ]

    conditions_struct = []

    for condition_id in experiment_filemap["condition_id"].unique():
        condition_df = experiment_filemap[
            experiment_filemap["condition_id"] == condition_id
        ]
        condition_dict = {}
        for key in conditions_keys:
            condition_dict[key] = condition_df[key].iloc[0]

        (
            ecdysis_time_step,
            larval_stage_durations_time_step,
            ecdysis_experiment_time,
            larval_stage_durations_experiment_time,
        ) = get_ecdysis_and_durations(condition_df)
        condition_dict["condition_id"] = int(condition_dict["condition_id"])
        condition_dict["ecdysis_time_step"] = ecdysis_time_step
        condition_dict["larval_stage_durations_time_step"] = (
            larval_stage_durations_time_step
        )
        condition_dict["ecdysis_experiment_time"] = ecdysis_experiment_time
        condition_dict["larval_stage_durations_experiment_time"] = (
            larval_stage_durations_experiment_time
        )
        condition_dict["experiment"] = np.array([experiment_dir]*condition_df['Point'].nunique())[:, np.newaxis]
        condition_dict["point"] = np.unique(condition_df["Point"].values)[:, np.newaxis]

        worm_type_column = [col for col in condition_df.columns if "worm_type" in col][
            0
        ]
        worm_types = separate_column_by_point(condition_df, worm_type_column)

        for organ in organ_channels.keys():
            organ_channel = organ_channels[organ]
            organ_channel = f"ch{organ_channel}"
            organ_columns = [
                col for col in condition_df.columns if col.startswith(organ_channel)
            ]
            organ_columns = [col for col in organ_columns if not ("_at_" in col)]
            renamed_organ_columns = [
                col.replace(organ_channel, organ) for col in organ_columns
            ]

            for organ_column, renamed_organ_column in zip(
                organ_columns, renamed_organ_columns
            ):
                condition_dict[renamed_organ_column] = separate_column_by_point(
                    condition_df, organ_column
                )

            # remove any column with worm_type in it
            renamed_organ_columns = [
                col for col in renamed_organ_columns if not ("worm_type" in col)
            ]
            for column in renamed_organ_columns:
                condition_dict[f"{column}_at_ecdysis"] = np.stack(
                    [
                        compute_series_at_time_classified(
                            condition_dict[column][i],
                            worm_types[i],
                            ecdysis_time_step[i],
                        )
                        for i in range(len(ecdysis_time_step))
                    ]
                )

        condition_dict["time"] = separate_column_by_point(condition_df, "Time").astype(
            float
        )
        condition_dict["experiment_time"] = separate_column_by_point(
            condition_df, "ExperimentTime"
        ).astype(float)

        conditions_struct.append(condition_dict)

    conditions_info = [
        {key: condition[key] for key in conditions_keys}
        for condition in conditions_struct
    ]

    # sort the conditions and conditions_info by condition_id
    conditions_struct = sorted(conditions_struct, key=lambda x: x["condition_id"])
    conditions_info = sorted(conditions_info, key=lambda x: x["condition_id"])

    return conditions_struct, conditions_info


def remove_unwanted_info(conditions_info):
    for condition in conditions_info:
        if "description" in condition.keys():
            condition.pop("description")
        if "condition_id" in condition.keys():
            condition.pop("condition_id")
    return conditions_info

def combine_experiments(filemap_paths, config_paths, experiment_dirs=None, organ_channels=[{"body": 2, "pharynx": 1}]):
    all_conditions_struct = []
    condition_info_merge_list = []
    conditions_info_keys = set()
    condition_id_counter = 0

    if isinstance(organ_channels, dict):
        organ_channels = [organ_channels]
        
    if len(organ_channels) == 1:
        organ_channels = organ_channels * len(filemap_paths)
    elif len(organ_channels) != len(filemap_paths):
        raise ValueError(
            "Number of organ channels must be equal to the number of experiments."
        )

    # Process each experiment
    for i, (filemap_path, config_path, organ_channel) in enumerate(zip(filemap_paths, config_paths, organ_channels)):
        experiment_dir = (
            experiment_dirs[i] if experiment_dirs else os.path.dirname(filemap_path)
        )
        conditions_struct, conditions_info = build_plotting_struct(
            experiment_dir, filemap_path, config_path, organ_channels=organ_channel,
        )

        # Process conditions for this experiment
        for condition in conditions_struct:
            condition["condition_id"] = condition_id_counter
            condition_id_counter += 1
            all_conditions_struct.append(condition)

        # Process condition info
        experiment_conditions_info = remove_unwanted_info(conditions_info)
        condition_info_merge_list.extend(experiment_conditions_info)
        conditions_info_keys.update(
            *[condition.keys() for condition in experiment_conditions_info]
        )

    # Merge conditions based on their info
    condition_dict = defaultdict(list)
    for i, condition_info in enumerate(condition_info_merge_list):
        key = frozenset(condition_info.items())
        condition_dict[key].append(i)

    merged_conditions_struct = []
    for indices in condition_dict.values():
        base_condition = all_conditions_struct[indices[0]]
        for idx in indices[1:]:
            for key, value in all_conditions_struct[idx].items():
                if key not in conditions_info_keys:
                    if isinstance(value, np.ndarray):
                        if value.shape[1] > base_condition[key].shape[1]:
                            base_condition[key] = np.pad(
                                base_condition[key],
                                (
                                    (0, 0),
                                    (0, value.shape[1] - base_condition[key].shape[1]),
                                ),
                                mode="constant",
                                constant_values=np.nan,
                            )
                        elif value.shape[1] < base_condition[key].shape[1]:
                            value = np.pad(
                                value,
                                (
                                    (0, 0),
                                    (0, base_condition[key].shape[1] - value.shape[1]),
                                ),
                                mode="constant",
                                constant_values=np.nan,
                            )
                    try:
                        base_condition[key] = np.concatenate(
                            (base_condition[key], value), axis=0
                        )
                    except ValueError as e:
                        print(f"Could not concatenate {key}: {e}")

        merged_conditions_struct.append(base_condition)

    # # Sort and reassign condition IDs
    # merged_conditions_struct.sort(key=lambda x: x['condition_id'])
    for i, condition in enumerate(merged_conditions_struct):
        condition["condition_id"] = i

    return merged_conditions_struct


# PLOTTING FUNCTIONS


def build_legend(single_condition_dict, legend):
    if legend is None:
        return f'Condition {int(single_condition_dict["condition_id"])}'
    else:
        legend_string = ""
        for i, (key, value) in enumerate(legend.items()):
            if value:
                legend_string += f"{single_condition_dict[key]} {value}"
            else:
                legend_string += f"{single_condition_dict[key]}"
            if i < len(legend) - 1:
                legend_string += ", "
        return legend_string


def set_scale(ax, log_scale):
    if isinstance(log_scale, bool):
        ax.set_yscale("log" if log_scale else "linear")
    elif isinstance(log_scale, tuple):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")
    elif isinstance(log_scale, list):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")


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
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors

    def plot_single_series(column: str):
        for i, condition_id in enumerate(conditions_to_plot):
            condition_dict = conditions_struct[condition_id]
            if experiment_time:
                time = condition_dict["experiment_time"]
                larval_stage_durations = condition_dict[
                    "larval_stage_durations_experiment_time"
                ]
            else:
                time = condition_dict["time"]
                larval_stage_durations = condition_dict[
                    "larval_stage_durations_time_step"
                ]

            rescaled_time, aggregated_series, _, ste_series = rescale_and_aggregate(
                condition_dict[column],
                time,
                condition_dict["ecdysis_time_step"],
                larval_stage_durations,
                condition_dict["body_seg_str_worm_type"],
                aggregation=aggregation,
            )

            ci_lower = aggregated_series - 1.96 * ste_series
            ci_upper = aggregated_series + 1.96 * ste_series

            if experiment_time:
                rescaled_time = rescaled_time / 3600
            else:
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

    plt.xlabel("Time (h)")
    plt.yscale("log" if log_scale else "linear")

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(series_column)

    plt.show()


def plot_correlation(
    conditions_struct,
    column_one,
    column_two,
    conditions_to_plot,
    log_scale=True,
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        _, aggregated_series_one, _, _ = rescale_and_aggregate(
            condition_dict[column_one],
            condition_dict["time"],
            condition_dict["ecdysis_time_step"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict["body_seg_str_worm_type"],
            aggregation="mean",
        )

        _, aggregated_series_two, _, _ = rescale_and_aggregate(
            condition_dict[column_two],
            condition_dict["time"],
            condition_dict["ecdysis_time_step"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict["body_seg_str_worm_type"],
            aggregation="mean",
        )

        # sort the values
        order = np.argsort(aggregated_series_one)
        aggregated_series_one = aggregated_series_one[order]
        aggregated_series_two = aggregated_series_two[order]

        label = build_legend(condition_dict, legend)

        plt.plot(
            aggregated_series_one,
            aggregated_series_two,
            color=color_palette[i],
            label=label,
        )

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()


def plot_correlation_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    conditions_to_plot,
    remove_hatch=True,
    log_scale=True,
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors

    for i, condition_id in enumerate(conditions_to_plot):

        condition_dict = conditions_struct[condition_id]

        if remove_hatch:
            column_one_values = condition_dict[column_one][:, 1:]
            column_two_values = condition_dict[column_two][:, 1:]
        else:
            column_one_values = condition_dict[column_one]
            column_two_values = condition_dict[column_two]

        x = np.nanmean(column_one_values, axis=0)
        x_std = np.nanstd(column_one_values, axis=0)
        x_ste = x_std / np.sqrt(np.sum(np.isfinite(column_one_values), axis=0))

        y = np.nanmean(column_two_values, axis=0)
        y_std = np.nanstd(column_two_values, axis=0)
        y_ste = y_std / np.sqrt(np.sum(np.isfinite(column_two_values), axis=0))

        label = build_legend(condition_dict, legend)
        plt.errorbar(
            x,
            y,
            xerr=x_std,
            yerr=y_std,
            fmt="o",
            color=color_palette[i],
            label=label,
            capsize=3,
        )
        plt.plot(x, y, color=color_palette[i])

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()


def boxplot_at_molt(
    conditions_struct,
    column,
    conditions_to_plot,
    log_scale: bool = True,
    figsize: tuple = None,
    color_palette="colorblind",
    plot_significance: bool = False,
    legend=None,
    y_axis_label=None,
    titles=None,
):

    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for j in range(data.shape[1]):
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
        figsize = (5 * df["Molt"].nunique(), 6)

    if titles is not None and len(titles) != df["Molt"].nunique():
        print("Number of titles does not match the number of ecdysis events.")
        titles = None

    # Create plot
    fig, ax = plt.subplots(1, df["Molt"].nunique(), figsize=figsize)
    for i in range(df["Molt"].nunique()):
        sns.boxplot(
            data=df[df["Molt"] == i],
            x="Condition",
            y=column,
            hue="Condition",
            palette=color_palette,
            showfliers=False,
            ax=ax[i],
            dodge=False,
        )

        handles, labels = ax[i].get_legend_handles_labels()
        new_label_list = [
            build_legend(conditions_struct[condition_id], legend)
            for condition_id in conditions_to_plot
        ]
        ax[i].legend(handles, new_label_list)

        ylims = ax[i].get_ylim()
        sns.stripplot(
            data=df[df["Molt"] == i],
            x="Condition",
            y=column,
            ax=ax[i],
            alpha=0.5,
            color="black",
            dodge=True,
        )
        ax[i].set_ylim(ylims)

        ax[i].set_xlabel("")
        ax[i].set_ylabel("")
        if titles is not None:
            ax[i].set_title(titles[i])
        # remove ticks
        ax[i].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            pairs = list(combinations(df["Condition"].unique(), 2))
            p_values = []

            bars = []
            for pair in pairs:
                data1 = df[(df["Condition"] == pair[0]) & (df["Molt"] == i)][
                    column
                ].dropna()
                data2 = df[(df["Condition"] == pair[1]) & (df["Molt"] == i)][
                    column
                ].dropna()
                if len(data1) == 0 or len(data2) == 0:
                    continue
                p_value = mannwhitneyu(data1, data2).pvalue

                # convert condition id to condition index
                bar = [
                    conditions_to_plot.index(pair[0]),
                    conditions_to_plot.index(pair[1]),
                    p_value,
                ]
                bars.append(bar)

            starbars.draw_annotation(bars, ax=ax[i])
    # Set y label for the first plot
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    # remove x label

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_growth_curves_individuals(
    conditions_struct,
    column,
    conditions_to_plot,
    log_scale=True,
    color_palette="colorblind",
    figsize=None,
    legend=None,
    y_axis_label=None,
):
    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))

    if figsize is None:
        figsize = (len(conditions_to_plot) * 8, 10)

    fig, ax = plt.subplots(1, len(conditions_to_plot), figsize=figsize)

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        for j in range(len(condition_dict[column])):
            time = condition_dict["time"][j]
            data = condition_dict[column][j]
            worm_type = condition_dict["body_seg_str_worm_type"][j]

            filtered_data = filter_series_with_classification(data, worm_type)

            label = build_legend(condition_dict, legend)

            ax[i].title.set_text(label)
            ax[i].plot(time, filtered_data)

        set_scale(ax[i], log_scale)

    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    plt.show()


def get_proportion_model(
    series_one, series_two, worm_type, x_axis_label=None, y_axis_label=None
):
    assert len(series_one) == len(
        series_two
    ), "The two series must have the same length."

    series_one = np.array(series_one).flatten()
    series_two = np.array(series_two).flatten()
    worm_type = np.array(worm_type).flatten()

    series_one = filter_series_with_classification(series_one, worm_type)
    series_two = filter_series_with_classification(series_two, worm_type)

    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one) & ~np.isnan(series_two)
    series_one = series_one[correct_indices]
    series_two = series_two[correct_indices]

    # log transform the data
    series_one = np.log(series_one)
    series_two = np.log(series_two)

    # for duplicate values, take the mean
    unique_series_one = np.unique(series_one)
    unique_series_two = np.array(
        [np.mean(series_two[series_one == value]) for value in unique_series_one]
    )

    series_one = unique_series_one
    series_two = unique_series_two

    plt.scatter(series_one, series_two)

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel("column one")

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel("column two")

    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(series_two, series_one, frac=1.0 / 3)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # plt.scatter(lowess_x, lowess_y, color='red')
    # plt.show()

    # interpolate the loess curve
    model = make_interp_spline(lowess_x, lowess_y, k=3)

    x = np.linspace(min(series_one), max(series_one), 100)
    y = model(x)

    plt.plot(x, y, color="red")
    plt.show()

    return model


def get_deviation_from_model(
    series_one, series_two, time, ecdysis, model, worm_type, n_points=100
):

    _, rescaled_series_one = rescale_series(
        series_one, time, ecdysis, worm_type, n_points=n_points
    )
    _, rescaled_series_two = rescale_series(
        series_two, time, ecdysis, worm_type, n_points=n_points
    )

    # log transform the data
    rescaled_series_one = np.log(rescaled_series_one)
    rescaled_series_two = np.log(rescaled_series_two)

    expected_series_two = model(rescaled_series_one)

    log_residuals = rescaled_series_two - expected_series_two
    residuals = np.exp(log_residuals)

    aggregated_series_one = np.full((4, n_points), np.nan)
    aggregated_residuals = np.full((4, n_points), np.nan)
    std_residuals = np.full((4, n_points), np.nan)
    ste_residuals = np.full((4, n_points), np.nan)

    for i in range(4):
        aggregated_residuals[i, :] = np.nanmean(residuals[:, i, :], axis=0)
        std_residuals[i, :] = np.nanstd(residuals[:, i, :], axis=0)
        ste_residuals[i, :] = std_residuals[i, :] / np.sqrt(
            np.sum(np.isfinite(residuals[:, i, :]), axis=0)
        )

        aggregated_series_one[i, :] = np.nanmean(
            np.exp(rescaled_series_one[:, i, :]), axis=0
        )

    aggregated_series_one = aggregated_series_one.flatten()

    aggregated_residuals = aggregated_residuals.flatten()
    std_residuals = std_residuals.flatten()
    ste_residuals = ste_residuals.flatten()

    return aggregated_series_one, aggregated_residuals, std_residuals, ste_residuals


def plot_deviation_from_model(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    log_scale=(True, False),
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {column_two}"
    )

    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model(
        control_condition[column_one],
        control_condition[column_two],
        control_condition["body_seg_str_worm_type"],
        x_axis_label=xlbl,
        y_axis_label=ylbl,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        ecdysis = condition["ecdysis_time_step"]
        time = condition["time"]
        body_data, pharynx_data = condition[column_one], condition[column_two]
        x, residuals, std_residuals, ste_residuals = get_deviation_from_model(
            body_data,
            pharynx_data,
            time,
            ecdysis,
            control_model,
            condition["body_seg_str_worm_type"],
        )

        label = build_legend(condition, legend)
        plt.plot(x, residuals, label=label, color=color_palette[i])
        plt.fill_between(
            x,
            residuals - 1.96 * std_residuals,
            residuals + 1.96 * std_residuals,
            color=color_palette[i],
            alpha=0.2,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()


def exclude_arrests_from_series(series_at_ecdysis):
    filtered_series = np.full(series_at_ecdysis.shape, np.nan)
    # keep only a value at one ecdys event if the next one is not nan
    if series_at_ecdysis.shape[0] == 1 or len(series_at_ecdysis.shape) == 1:
        for i in range(len(series_at_ecdysis)):
            if i == len(series_at_ecdysis) - 1:
                filtered_series[i] = series_at_ecdysis[i]
            elif not np.isnan(series_at_ecdysis[i + 1]):
                filtered_series[i] = series_at_ecdysis[i]
        return filtered_series
    else:
        for i in range(series_at_ecdysis.shape[0]):
            for j in range(series_at_ecdysis.shape[1]):
                if j == series_at_ecdysis.shape[1] - 1:
                    filtered_series[i, j] = series_at_ecdysis[i, j]
                elif not np.isnan(series_at_ecdysis[i, j + 1]):
                    filtered_series[i, j] = series_at_ecdysis[i, j]
        return filtered_series


def get_proportion_model_ecdysis(
    series_one_at_ecdysis,
    series_two_at_ecdysis,
    remove_hatch=True,
    x_axis_label=None,
    y_axis_label=None,
    exclude_arrests=False,
):
    assert len(series_one_at_ecdysis) == len(
        series_two_at_ecdysis
    ), "The two series must have the same length."

    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series(series_one_at_ecdysis)
        series_two_at_ecdysis = exclude_arrests_from_series(series_two_at_ecdysis)

    series_one_at_ecdysis = np.array(series_one_at_ecdysis).flatten()
    series_two_at_ecdysis = np.array(series_two_at_ecdysis).flatten()
    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one_at_ecdysis) & ~np.isnan(
        series_two_at_ecdysis
    )
    series_one_at_ecdysis = series_one_at_ecdysis[correct_indices]
    series_two_at_ecdysis = series_two_at_ecdysis[correct_indices]

    # log transform the data
    series_one_at_ecdysis = np.log(series_one_at_ecdysis)
    series_two_at_ecdysis = np.log(series_two_at_ecdysis)

    plt.scatter(series_one_at_ecdysis, series_two_at_ecdysis)

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel("column one")

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel("column two")

    fit = np.polyfit(series_one_at_ecdysis, series_two_at_ecdysis, 3)
    model = np.poly1d(fit)

    plt.plot(
        np.sort(series_one_at_ecdysis),
        model(np.sort(series_one_at_ecdysis)),
        color="red",
    )
    plt.show()

    return model


def get_deviation_from_model_at_ecdysis(
    series_one_at_ecdysis,
    series_two_at_ecdysis,
    model,
    remove_hatch=True,
    exclude_arrests=False,
):
    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series(series_one_at_ecdysis)
        series_two_at_ecdysis = exclude_arrests_from_series(series_two_at_ecdysis)

    # remove elements that are nan in one of the two arrays
    for i in range(series_one_at_ecdysis.shape[1]):
        nan_mask = np.isnan(series_one_at_ecdysis[:, i]) | np.isnan(series_two_at_ecdysis[:, i])
        series_one_at_ecdysis[:, i][nan_mask] = np.nan
        series_two_at_ecdysis[:, i][nan_mask] = np.nan

    # log transform the data
    series_one_at_ecdysis = np.log(series_one_at_ecdysis)
    series_two_at_ecdysis = np.log(series_two_at_ecdysis)

    expected_series_two = model(series_one_at_ecdysis)

    log_residuals = series_two_at_ecdysis - expected_series_two
    residuals = np.exp(log_residuals)

    y = np.nanmean(residuals, axis=0)
    y_err = np.nanstd(residuals, axis=0) / np.sqrt(len(residuals))
    x = np.nanmean(np.exp(series_one_at_ecdysis), axis=0)

    return x, y, y_err


def get_deviation_percentage_from_model_at_ecdysis(
    series_one_at_ecdysis,
    series_two_at_ecdysis,
    model,
    remove_hatch=True,
    exclude_arrests=False,
):
    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series(series_one_at_ecdysis)
        series_two_at_ecdysis = exclude_arrests_from_series(series_two_at_ecdysis)

    # remove elements that are nan in one of the two arrays
    for i in range(series_one_at_ecdysis.shape[1]):
        nan_mask = np.isnan(series_one_at_ecdysis[:, i]) | np.isnan(series_two_at_ecdysis[:, i])
        series_one_at_ecdysis[:, i][nan_mask] = np.nan
        series_two_at_ecdysis[:, i][nan_mask] = np.nan

    # Apply the model to the log-transformed series_one to get expected values
    expected_series_two = np.exp(model(np.log(series_one_at_ecdysis)))

    # Calculate percentage deviation using real values
    percentage_deviation = (
        (series_two_at_ecdysis - expected_series_two) / expected_series_two * 100
    )

    y = np.nanmean(percentage_deviation, axis=0)
    y_err = np.nanstd(percentage_deviation, axis=0) / np.sqrt(
        np.sum(~np.isnan(percentage_deviation), axis=0)
    )
    x = np.nanmean(series_one_at_ecdysis, axis=0)

    return x, y, y_err


def plot_deviation_from_model_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    remove_hatch=True,
    log_scale=(True, False),
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    percentage=True,
    exclude_arrests=False,
):

    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {column_two}"
    )

    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model_ecdysis(
        control_condition[column_one],
        control_condition[column_two],
        remove_hatch,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        exclude_arrests=exclude_arrests,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        body_data, pharynx_data = condition[column_one], condition[column_two]

        if percentage:
            x, y, y_err = get_deviation_percentage_from_model_at_ecdysis(
                body_data, pharynx_data, control_model, remove_hatch, exclude_arrests
            )
        else:
            x, y, y_err = get_deviation_from_model_at_ecdysis(
                body_data, pharynx_data, control_model, remove_hatch, exclude_arrests
            )

        label = build_legend(condition, legend)
        plt.plot(x, y, label=label, color=color_palette[i], marker="o")
        plt.errorbar(x, y, yerr=y_err, color=color_palette[i], fmt="o", capsize=3)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()    

def plot_normalized_proportions(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    aggregation="mean",
    log_scale=(True, False),
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))
    control_condition = conditions_struct[control_condition_id]
    control_column_one, control_column_two = (
        control_condition[column_one],
        control_condition[column_two],
    )

    aggregation_function = np.nanmean
    control_proportion = aggregation_function(
        control_column_two / control_column_one, axis=0
    )

    x_axis_label = x_axis_label if x_axis_label is not None else column_one
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"normalized {column_two} to {column_one} ratio"
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        condition_column_one = condition[column_one]
        condition_column_two = condition[column_two]

        proportion = condition_column_two / condition_column_one
        normalized_proportion = proportion / control_proportion

        y = aggregation_function(normalized_proportion, axis=0)
        y_err = np.nanstd(normalized_proportion, axis=0) / np.sqrt(
            len(normalized_proportion)
        )
        x = aggregation_function(condition_column_one, axis=0)

        label = build_legend(condition, legend)

        plt.plot(x, y, label=label, color=color_palette[i])
        plt.errorbar(x, y, yerr=y_err, fmt="o", capsize=3, color=color_palette[i])

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()


def process_series_data(
    series_one: np.ndarray,
    series_two: np.ndarray,
    point: np.ndarray,
    ecdysis: np.ndarray,
    remove_hatch: bool = True,
    exclude_arrests: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process and clean the input series data.
    """
    if remove_hatch:
        series_one = series_one[:, 1:]
        series_two = series_two[:, 1:]
        ecdysis = ecdysis[:, 1:]
    if exclude_arrests:
        series_one = exclude_arrests_from_series(series_one)
        series_two = exclude_arrests_from_series(series_two)

    # Stack point horizontally to match series shape
    point = np.hstack([point for _ in range(series_one.shape[1])]).astype(float)

    # Remove nan elements
    for i in range(series_one.shape[1]):
        nan_mask = np.isnan(series_one[:, i]) | np.isnan(series_two[:, i])
        series_one[:, i][nan_mask] = np.nan
        series_two[:, i][nan_mask] = np.nan
        point[:, i][nan_mask] = np.nan

    return series_one, series_two, point, ecdysis

def process_single_series_data(
    series: np.ndarray,
    point: np.ndarray,
    ecdysis: np.ndarray,
    remove_hatch: bool = True,
    exclude_arrests: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process and clean the input single series data.
    """
    if remove_hatch:
        series = series[:, 1:]
        ecdysis = ecdysis[:, 1:]
    if exclude_arrests:
        series = exclude_arrests_from_series(series)

    # Stack point horizontally to match series shape
    point = np.hstack([point for _ in range(series.shape[1])]).astype(float)

    # Remove nan elements
    for i in range(series.shape[1]):
        nan_mask = np.isnan(series[:, i])
        series[:, i][nan_mask] = np.nan
        point[:, i][nan_mask] = np.nan

    return series, point, ecdysis

def filter_non_worm_data(
    data: np.ndarray,
    worm_type: np.ndarray,
    ecdysis: np.ndarray
) -> np.ndarray:
    """
    Filter out non-worm data points.
    """
    filtered_data = data.copy()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            try:
                if ~(np.isnan(data[i][j])) and worm_type[i][int(ecdysis[i][j])] != 'worm':
                    filtered_data[i][j] = np.nan
            except ValueError:
                filtered_data[i][j] = np.nan
    return filtered_data

def setup_image_filemaps(
    experiment: np.ndarray,
    img_dir_list: List[str]
) -> Dict[str, Any]:
    """
    Set up file mappings for image directories.
    """
    unique_experiment = np.unique(experiment)
    filemaps = {}
    
    for exp in unique_experiment:
        exp = exp.split('analysis')[0]
        for img_dir in img_dir_list:
            img_dir = os.path.join(exp, img_dir)
            filemap = get_dir_filemap(img_dir)
            filemaps[img_dir] = filemap
            
    return filemaps

def display_image(
    img_path: str,
    dpi: int = 200,
    cmap: str = 'viridis',
    backup_dir: str = None,
    backup_file_name: str = None,
) -> None:
    """
    Display an image with the specified parameters.
    """
    img = read_tiff_file(img_path)

    if backup_dir is not None:
        if backup_file_name is not None:
            shutil.copy(img_path, os.path.join(backup_dir, backup_file_name))
        else:
            shutil.copy(img_path, backup_dir)
    height, width = img.shape[-2:]
    
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    plt.imshow(img, interpolation='none', aspect='equal', cmap=cmap)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    plt.show()

def display_image_overlay(
    img_paths: List[str],
    dpi: int = 200,
    cmap: List[str] = ['viridis'],
    backup_dir: str = None,
    backup_file_name: str = None
):
    """
    Display an overlay of multiple images.
    """
    img_list = [read_tiff_file(img_path) for img_path in img_paths]

    if len(cmap) == 1:
        cmap = cmap * len(img_list)

    stacked_img = np.stack(img_list, axis=0)

    if backup_dir is not None:
        if backup_file_name is not None:
            imwrite(os.path.join(backup_dir, backup_file_name), stacked_img)
        else:
            img_path = img_paths[0]
            imwrite(os.path.join(backup_dir, os.path.basename(img_path)), stacked_img)

    # add an empty channel to the image
    print(stacked_img.shape)

    height, width = stacked_img.shape[-2:]
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    microim = microshow(images = stacked_img, fig_scaling = 5, cmaps = cmap, ax = ax, proj_type='max', dpi = dpi,)

def display_sample_images(
    experiment: str,
    point: int,
    ecdysis: int,
    img_dir_list: List[str],
    filemaps: Dict[str, Any],
    dpi: int,
    overlay: bool = True,
    cmap: List[str] = ['viridis'],
    backup_dir: str = None
) -> None:
    """
    Display sample images for a given experiment, point, and ecdysis time.
    """
    # Remove analysis suffix from experiment name if present
    experiment_base = experiment.split('analysis')[0]
    
    if isinstance(cmap, str):
        cmap = [cmap] * len(img_dir_list)

    if overlay:
        img_paths = []
        for img_dir in img_dir_list:
            img_dir_path = os.path.join(experiment_base, img_dir)
            filemap = filemaps[img_dir_path]
            
            # Find matching image path
            matching_rows = filemap[
                (filemap['Point'] == point) & 
                (filemap['Time'] == ecdysis)
            ]
            
            if len(matching_rows) > 0:
                img_path = matching_rows['ImagePath'].values[0]
                img_paths.append(img_path)
        
        display_image_overlay(img_paths, dpi, cmap, backup_dir)
    else:
        for img_dir in img_dir_list:
            img_dir_path = os.path.join(experiment_base, img_dir)
            filemap = filemaps[img_dir_path]
            
            # Find matching image path
            matching_rows = filemap[
                (filemap['Point'] == point) & 
                (filemap['Time'] == ecdysis)
            ]
            
            if len(matching_rows) > 0:
                img_path = matching_rows['ImagePath'].values[0]
                display_image(img_path, dpi, cmap=cmap[0], backup_dir=backup_dir)
        

def get_most_average_deviations_at_ecdysis(
    conditions_struct: Dict,
    column_one: str,
    column_two: str,
    img_dir_list: List[str],
    control_condition_id: str,
    conditions_to_plot: List[str],
    remove_hatch: bool = True,
    exclude_arrests: bool = False,
    dpi: int = 200,
    nb_per_condition: int = 1,
    overlay: bool = True,
    cmap: List[str] = ['viridis'],
    backup_dir: str = None,
    backup_name = None,
) -> None:
    """
    Calculate and display the most average deviations at ecdysis.
    """
    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model_ecdysis(
        control_condition[column_one],
        control_condition[column_two],
        remove_hatch,
        x_axis_label=column_one,
        y_axis_label=column_two,
        exclude_arrests=exclude_arrests,
    )

    for condition_id in conditions_to_plot:
        condition = conditions_struct[condition_id]
        series_one, series_two, point, experiment, ecdysis, worm_type = [
            condition[key] for key in [column_one, column_two, 'point', 'experiment', 'ecdysis_time_step', 'body_seg_str_worm_type']
        ]

        filemaps = setup_image_filemaps(experiment, img_dir_list)
        
        series_one, series_two, point, ecdysis = process_series_data(
            series_one, series_two, point, ecdysis, remove_hatch, exclude_arrests
        )

        # Calculate expected values and deviations
        expected_series_two = np.exp(control_model(np.log(series_one)))
        percentage_deviation = ((series_two - expected_series_two) / expected_series_two * 100)
        percentage_deviation = filter_non_worm_data(percentage_deviation, worm_type, ecdysis)
        
        y = np.nanmean(percentage_deviation, axis=0)

        for i in range(percentage_deviation.shape[1]):
            deviation_molt = percentage_deviation[:, i]
            mean_deviation = y[i]
            sorted_idx = np.argsort(np.abs(deviation_molt - mean_deviation))
            valid_idx = sorted_idx[~np.isnan(deviation_molt[sorted_idx])][:nb_per_condition]
            
            for idx in valid_idx:
                display_sample_images(experiment[idx][0], int(point[idx][i]), 
                                   int(ecdysis[idx][i]), img_dir_list, filemaps, dpi, overlay=overlay, cmap=cmap, backup_dir=backup_dir)

def get_most_average_proportions_at_ecdysis(
    conditions_struct: Dict,
    column_one: str,
    column_two: str,
    img_dir_list: List[str],
    conditions_to_plot: List[str],
    remove_hatch: bool = True,
    exclude_arrests: bool = False,
    dpi: int = 200,
    nb_per_condition: int = 1,
    overlay: bool = True,
    cmap: List[str] = ['viridis'],
    backup_dir: str = None,
    backup_name = None,
) -> None:
    """
    Calculate and display the most average proportions at ecdysis.
    """
    for condition_id in conditions_to_plot:
        condition = conditions_struct[condition_id]
        series_one, series_two, point, experiment, ecdysis, worm_type = [
            condition[key] for key in [column_one, column_two, 'point', 'experiment', 'ecdysis_time_step', 'body_seg_str_worm_type']
        ]

        filemaps = setup_image_filemaps(experiment, img_dir_list)
        
        series_one, series_two, point, ecdysis = process_series_data(
            series_one, series_two, point, ecdysis, remove_hatch, exclude_arrests
        )

        series_one = filter_non_worm_data(series_one, worm_type, ecdysis)
        series_two = filter_non_worm_data(series_two, worm_type, ecdysis)

        series_one_mean = np.nanmean(series_one, axis=0)
        series_two_mean = np.nanmean(series_two, axis=0)

        for i in range(series_one.shape[1]):
            series_one_molt = series_one[:, i]
            series_two_molt = series_two[:, i]

            series_one_mean_molt = series_one_mean[i]
            series_two_mean_molt = series_two_mean[i]

            distance_series_one = np.abs(series_one_molt - series_one_mean_molt)/series_one_mean_molt
            distance_series_two = np.abs(series_two_molt - series_two_mean_molt)/series_two_mean_molt

            distance_score = distance_series_one + distance_series_two
    
            sorted_idx = np.argsort(distance_score)
            valid_idx = sorted_idx[:nb_per_condition]
            
            for idx in valid_idx:
                display_sample_images(
                    experiment[idx][0],
                    int(point[idx][i]),
                    int(ecdysis[idx][i]),
                    img_dir_list,
                    filemaps,
                    dpi,
                    overlay=overlay,
                    cmap=cmap,
                )

def get_most_average_size_at_ecdysis(
    conditions_struct: Dict,
    column : str,
    img_dir_list: List[str],
    conditions_to_plot: List[str],
    remove_hatch: bool = True,
    exclude_arrests: bool = False,
    dpi: int = 200,
    nb_per_condition: int = 1,
    overlay: bool = True,
    cmap: List[str] = ['viridis'],
    backup_dir: str = None,
    backup_name = None,
) -> None:
    """
    Calculate and display the most average sizes at ecdysis.
    """
    for condition_id in conditions_to_plot:
        condition = conditions_struct[condition_id]
        series, point, experiment, ecdysis, worm_type = [
            condition[key] for key in [column, 'point', 'experiment', 'ecdysis_time_step', 'body_seg_str_worm_type']
        ]

        filemaps = setup_image_filemaps(experiment, img_dir_list)
        
        series, point, ecdysis = process_single_series_data(
            series, point, ecdysis, remove_hatch, exclude_arrests
        )

        series = filter_non_worm_data(series, worm_type, ecdysis)

        series_mean = np.nanmean(series, axis=0)

        for i in range(series.shape[1]):
            series_molt = series[:, i]
            series_mean_molt = series_mean[i]

            distance_score = np.abs(series_molt - series_mean_molt)
    
            sorted_idx = np.argsort(distance_score)
            valid_idx = sorted_idx[:nb_per_condition]
            
            for idx in valid_idx:
                display_sample_images(
                    experiment[idx][0],
                    int(point[idx][i]),
                    int(ecdysis[idx][i]),
                    img_dir_list,
                    filemaps,
                    dpi,
                    overlay=overlay,
                    cmap=cmap,
                )
