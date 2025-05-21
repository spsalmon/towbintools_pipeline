import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
from towbintools.data_analysis import rescale_and_aggregate

from .utils_plotting import build_legend
from .utils_plotting import get_colors
from .utils_plotting import set_scale

# from .utils_data_processing import exclude_arrests_from_series_at_ecdysis

# MODEL BUILDING


def _get_continuous_proportion_model(
    rescaled_series_one,
    rescaled_series_two,
    x_axis_label=None,
    y_axis_label=None,
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


def _get_proportion_model(
    series_one_values,
    series_two_values,
    x_axis_label=None,
    y_axis_label=None,
    poly_degree=2,
    plot_model=True,
):
    assert len(series_one_values) == len(
        series_two_values
    ), "The two series must have the same length."

    series_one_values = np.array(series_one_values).flatten()
    series_two_values = np.array(series_two_values).flatten()
    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one_values) & ~np.isnan(series_two_values)
    series_one_values = series_one_values[correct_indices]
    series_two_values = series_two_values[correct_indices]

    # log transform the data
    series_one_values = np.log(series_one_values)
    series_two_values = np.log(series_two_values)

    fit = np.polyfit(series_one_values, series_two_values, poly_degree)
    model = np.poly1d(fit)

    if plot_model:
        plt.scatter(series_one_values, series_two_values)

        if x_axis_label is not None:
            plt.xlabel(x_axis_label)
        else:
            plt.xlabel("column one")

        if y_axis_label is not None:
            plt.ylabel(y_axis_label)
        else:
            plt.ylabel("column two")

        x = np.linspace(np.nanmin(series_one_values), np.nanmax(series_one_values), 100)
        y = model(x)
        plt.plot(
            x,
            y,
            color="red",
        )
        plt.show()

    return model


# COMPUTE DEVIATION FROM MODEL


def get_deviation_from_model(
    series_one_values, series_two_values, model, percentage=True
):
    if percentage:
        expected_series_two = np.exp(model(np.log(series_one_values)))
        # Calculate percentage deviation using real values
        deviation = (
            (series_two_values - expected_series_two) / expected_series_two * 100
        )

    else:
        log_expected_series_two = model(np.log(series_one_values))
        deviation = np.log(series_two_values) - log_expected_series_two

    return deviation


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


def plot_continuous_deviation_from_model(
    conditions_struct,
    rescaled_column_one,
    rescaled_column_two,
    control_condition_id,
    conditions_to_plot,
    deviation_as_percentage=True,
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

    xlbl = rescaled_column_one
    ylbl = rescaled_column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {rescaled_column_two}"
    )

    control_condition = conditions_struct[control_condition_id]

    control_model = _get_continuous_proportion_model(
        control_condition[rescaled_column_one],
        control_condition[rescaled_column_two],
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        plot_model=True,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        rescaled_column_one_values, rescaled_column_two_values = (
            condition[rescaled_column_one],
            condition[rescaled_column_two],
        )
        residuals = get_deviation_from_model(
            rescaled_column_one_values,
            rescaled_column_two_values,
            control_model,
            percentage=deviation_as_percentage,
        )

        sorted_indices = np.argsort(rescaled_column_one_values)
        rescaled_column_one_values = rescaled_column_one_values[sorted_indices]
        residuals = residuals[sorted_indices]

        average_column_one_values = np.nanmean(rescaled_column_one_values, axis=0)
        average_residuals = np.nanmean(residuals, axis=0)
        ste_residuals = np.nanstd(residuals, axis=0) / np.sqrt(
            np.sum(~np.isnan(residuals), axis=0)
        )

        label = build_legend(condition, legend)
        plt.plot(
            average_column_one_values,
            average_residuals,
            label=label,
            color=color_palette[i],
        )
        plt.fill_between(
            average_column_one_values,
            average_residuals - 1.96 * ste_residuals,
            average_residuals + 1.96 * ste_residuals,
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


def plot_deviation_from_model_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    remove_hatch=False,
    deviation_as_percentage=True,
    log_scale=(True, False),
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    poly_degree=2,
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

    column_one_values = control_condition[column_one]
    column_two_values = control_condition[column_two]

    if remove_hatch:
        column_one_values = column_one_values[:, 1:]
        column_two_values = column_two_values[:, 1:]

    control_model = _get_proportion_model(
        column_one_values,
        column_two_values,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        poly_degree=poly_degree,
        plot_model=True,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]
        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            percentage=deviation_as_percentage,
        )

        mean_column_one_values = np.nanmean(column_one_values, axis=0)
        mean_deviations = np.nanmean(deviations, axis=0)
        # std_deviations = np.nanstd(deviations, axis=0)
        ste_deviations = np.nanstd(deviations, axis=0) / np.sqrt(
            np.sum(~np.isnan(deviations), axis=0)
        )

        label = build_legend(condition, legend)
        plt.plot(
            mean_column_one_values,
            mean_deviations,
            label=label,
            color=color_palette[i],
            marker="o",
        )
        plt.errorbar(
            mean_column_one_values,
            mean_deviations,
            yerr=ste_deviations,
            color=color_palette[i],
            fmt="o",
            capsize=3,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def plot_deviation_from_model_development_percentage(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    percentages,
    deviation_as_percentage=True,
    log_scale=(True, False),
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    poly_degree=3,
):
    """
    Plot the percentage deviation from a model at specified development percentages for multiple conditions.

    Args:
        conditions_struct (dict): Dictionary of condition data.
        column_one (str): Key for the first variable.
        column_two (str): Key for the second variable.
        control_condition_id (str): Key for the control condition.
        conditions_to_plot (list): List of condition keys to plot.
        percentages (np.ndarray): Array of percentages (0-1) at which to sample.
        colors (list, optional): List of colors for plotting.
        legend (list, optional): Legend labels.
        x_axis_label (str, optional): Label for x-axis.
        y_axis_label (str, optional): Label for y-axis.
        poly_degree (int, optional): Degree of polynomial for model fitting.
    """
    color_palette = get_colors(conditions_to_plot, colors)

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {column_two}"
    )

    control_condition = conditions_struct[control_condition_id]
    column_one_values = control_condition[column_one]
    column_two_values = control_condition[column_two]

    indices = np.clip(
        (percentages * column_one_values.shape[1]).astype(int),
        0,
        column_one_values.shape[1] - 1,
    ).astype(int)

    control_one_values = column_one_values[:, indices]
    control_two_values = column_two_values[:, indices]

    # Fit the model on the control condition
    control_model = _get_proportion_model(
        control_one_values,
        control_two_values,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        poly_degree=poly_degree,
        plot_model=True,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        column_one_values = column_one_values[:, indices]
        column_two_values = column_two_values[:, indices]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            percentage=deviation_as_percentage,
        )

        mean_column_one_values = np.nanmean(column_one_values, axis=0)
        mean_deviations = np.nanmean(deviations, axis=0)
        # std_deviations = np.nanstd(deviations, axis=0)
        ste_deviations = np.nanstd(deviations, axis=0) / np.sqrt(
            np.sum(~np.isnan(deviations), axis=0)
        )

        label = build_legend(condition, legend)
        plt.plot(
            mean_column_one_values,
            mean_deviations,
            label=label,
            color=color_palette[i],
            marker="o",
        )
        plt.errorbar(
            mean_column_one_values,
            mean_deviations,
            yerr=ste_deviations,
            fmt="o",
            capsize=3,
            color=color_palette[i],
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    set_scale(plt.gca(), log_scale)
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

    models = {}
    xs = {}

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        model = _get_proportion_model(
            condition[column_one],
            condition[column_two],
            remove_hatch,
            exclude_arrests=exclude_arrests,
            poly_degree=poly_degree,
            plot_model=False,
        )

        column_one_values = np.log(condition[column_one])

        x = np.linspace(np.nanmin(column_one_values), np.nanmax(column_one_values), 100)

        models[condition_id] = model
        xs[condition_id] = x

    # determine the overlap of all the x values
    x_min = np.nanmax(
        [np.nanmin(xs[condition_id]) for condition_id in conditions_to_plot]
    )
    x_max = np.nanmin(
        [np.nanmax(xs[condition_id]) for condition_id in conditions_to_plot]
    )

    x = np.linspace(x_min, x_max, 100)
    control_values = np.exp(models[control_condition_id](x))

    for i, condition_id in enumerate(conditions_to_plot):
        plt.plot(
            np.exp(x),
            (np.exp(models[condition_id](x)) / control_values - 1) * 100,
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
