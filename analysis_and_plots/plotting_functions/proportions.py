import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
from towbintools.data_analysis import rescale_and_aggregate

from .utils_data_processing import exclude_arrests_from_series_at_ecdysis
from .utils_plotting import build_legend
from .utils_plotting import get_colors
from .utils_plotting import set_scale


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
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
        worm_type_key = [key for key in condition_dict.keys() if "worm_type" in key][0]
        _, aggregated_series_one, _, _ = rescale_and_aggregate(
            condition_dict[column_one],
            condition_dict["time"],
            condition_dict["ecdysis_index"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict[worm_type_key],
            aggregation="mean",
        )

        _, aggregated_series_two, _, _ = rescale_and_aggregate(
            condition_dict[column_two],
            condition_dict["time"],
            condition_dict["ecdysis_index"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict[worm_type_key],
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

    fig = plt.gcf()
    plt.show()

    return fig


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
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

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

        y = np.nanmean(column_two_values, axis=0)
        y_std = np.nanstd(column_two_values, axis=0)

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

    fig = plt.gcf()
    plt.show()

    return fig


def get_proportion_model(
    rescaled_series_one, rescaled_series_two, x_axis_label=None, y_axis_label=None
):
    assert len(rescaled_series_one) == len(
        rescaled_series_two
    ), "The two series must have the same length."

    series_one = np.array(rescaled_series_one).flatten()
    series_two = np.array(rescaled_series_two).flatten()

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
    lowess = sm.nonparametric.lowess(series_two, series_one, frac=0.1)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # interpolate the loess curve
    model = make_interp_spline(lowess_x, lowess_y, k=1)

    x = np.linspace(min(series_one), max(series_one), 500)
    y = model(x)

    plt.plot(x, y, color="red", linewidth=2)
    plt.show()

    return model


def get_deviation_from_model(rescaled_series_one, rescaled_series_two, model):
    # log transform the data

    expected_series_two = np.exp(model(np.log(rescaled_series_one)))

    # log_residuals = rescaled_series_two - expected_series_two
    # residuals = np.exp(log_residuals)
    percentage_deviation = (
        (rescaled_series_two - expected_series_two) / expected_series_two * 100
    )

    mean_series_one = np.nanmean(rescaled_series_one, axis=0)
    # mean_residuals = np.nanmean(residuals, axis=0)
    # std_residuals = np.nanstd(residuals, axis=0)
    # ste_residuals = std_residuals / np.sqrt(np.sum(np.isfinite(residuals), axis=0))

    mean_residuals = np.nanmean(percentage_deviation, axis=0)
    std_residuals = np.nanstd(percentage_deviation, axis=0)
    ste_residuals = std_residuals / np.sqrt(
        np.sum(np.isfinite(percentage_deviation), axis=0)
    )

    return mean_series_one, mean_residuals, std_residuals, ste_residuals


def plot_deviation_from_model(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    colors=None,
    log_scale=(True, False),
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

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
        x_axis_label=xlbl,
        y_axis_label=ylbl,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        body_data, pharynx_data = condition[column_one], condition[column_two]
        x, residuals, std_residuals, ste_residuals = get_deviation_from_model(
            body_data,
            pharynx_data,
            control_model,
        )

        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        residuals = residuals[sorted_indices]
        ste_residuals = ste_residuals[sorted_indices]

        label = build_legend(condition, legend)
        plt.plot(x, residuals, label=label, color=color_palette[i])
        plt.fill_between(
            x,
            residuals - 1.96 * ste_residuals,
            residuals + 1.96 * ste_residuals,
            color=color_palette[i],
            alpha=0.2,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def get_proportion_model_ecdysis(
    series_one_at_ecdysis,
    series_two_at_ecdysis,
    remove_hatch=True,
    x_axis_label=None,
    y_axis_label=None,
    exclude_arrests=False,
    poly_degree=3,
    plot_model=True,
):
    assert len(series_one_at_ecdysis) == len(
        series_two_at_ecdysis
    ), "The two series must have the same length."

    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
            series_one_at_ecdysis
        )
        series_two_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
            series_two_at_ecdysis
        )

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

    fit = np.polyfit(series_one_at_ecdysis, series_two_at_ecdysis, poly_degree)
    model = np.poly1d(fit)

    if plot_model:
        plt.scatter(series_one_at_ecdysis, series_two_at_ecdysis)

        if x_axis_label is not None:
            plt.xlabel(x_axis_label)
        else:
            plt.xlabel("column one")

        if y_axis_label is not None:
            plt.ylabel(y_axis_label)
        else:
            plt.ylabel("column two")

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
        series_one_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
            series_one_at_ecdysis
        )
        series_two_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
            series_two_at_ecdysis
        )

    # remove elements that are nan in one of the two arrays
    for i in range(series_one_at_ecdysis.shape[1]):
        nan_mask = np.isnan(series_one_at_ecdysis[:, i]) | np.isnan(
            series_two_at_ecdysis[:, i]
        )
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
        series_one_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
            series_one_at_ecdysis
        )
        series_two_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
            series_two_at_ecdysis
        )

    # remove elements that are nan in one of the two arrays
    for i in range(series_one_at_ecdysis.shape[1]):
        nan_mask = np.isnan(series_one_at_ecdysis[:, i]) | np.isnan(
            series_two_at_ecdysis[:, i]
        )
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
    poly_degree=3,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

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
        poly_degree=poly_degree,
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

    fig = plt.gcf()
    plt.show()

    return fig


def plot_model_comparison_at_ecdysis(
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
    poly_degree=3,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    xlbl = column_one

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {column_two}"
    )

    # control_condition = conditions_struct[control_condition_id]
    # control_model = get_proportion_model_ecdysis(
    #     control_condition[column_one],
    #     control_condition[column_two],
    #     remove_hatch,
    #     exclude_arrests=exclude_arrests,
    #     poly_degree=poly_degree,
    #     plot_model=False,
    # )

    column_one_minimums = []
    column_one_maximums = []
    column_two_minimums = []
    column_two_maximums = []

    models = {}

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        model = get_proportion_model_ecdysis(
            condition[column_one],
            condition[column_two],
            remove_hatch,
            exclude_arrests=exclude_arrests,
            poly_degree=poly_degree,
            plot_model=False,
        )

        column_one_values = np.log(condition[column_one])
        column_two_values = np.log(condition[column_two])

        column_one_minimums.append(np.nanmin(column_one_values))
        column_one_maximums.append(np.nanmax(column_one_values))
        column_two_minimums.append(np.nanmin(column_two_values))
        column_two_maximums.append(np.nanmax(column_two_values))

        models[condition_id] = model

    column_one_minimum = np.nanmin(column_one_minimums)
    column_one_maximum = np.nanmax(column_one_maximums)

    x = np.linspace(column_one_minimum, column_one_maximum, 100)
    # control_y = np.exp(control_model(x))

    for i, condition_id in enumerate(conditions_to_plot):
        plt.plot(
            np.exp(x),
            np.exp(models[condition_id](x)),
            color=color_palette[i],
            label=build_legend(conditions_struct[condition_id], legend),
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def plot_normalized_proportions_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    colors=None,
    aggregation="mean",
    log_scale=(True, False),
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )
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

    fig = plt.gcf()
    plt.show()

    return fig
