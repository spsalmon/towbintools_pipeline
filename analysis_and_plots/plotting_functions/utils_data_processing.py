import numpy as np
from towbintools.data_analysis import rescale_series
from towbintools.data_analysis.growth_rate import (
    compute_instantaneous_growth_rate_classified,
)


def combine_series(
    conditions_struct, series_one, series_two, operation, new_series_name
):
    for condition in conditions_struct:
        series_one_values = condition[series_one]
        series_two_values = condition[series_two]

        if operation == "add":
            new_series_values = np.add(series_one_values, series_two_values)
        elif operation == "subtract":
            new_series_values = series_one_values - series_two_values
        elif operation == "multiply":
            new_series_values = series_one_values * series_two_values
        elif operation == "divide":
            new_series_values = np.divide(series_one_values, series_two_values)
        condition[new_series_name] = new_series_values
    return conditions_struct


def transform_series(conditions_struct, series, operation, new_series_name):
    for conditions in conditions_struct:
        series_values = conditions[series]

        if operation == "log":
            new_series_values = np.log(series_values)
        elif operation == "exp":
            new_series_values = np.exp(series_values)
        elif operation == "sqrt":
            new_series_values = np.sqrt(series_values)
        conditions[new_series_name] = new_series_values

    return conditions_struct


def compute_growth_rate(
    conditions_struct,
    series_name,
    gr_series_name,
    experiment_time=True,
    lmbda=0.0075,
    order=2,
    medfilt_window=5,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
        worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
        worm_type = condition[worm_type_key]

        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        growth_rate = []
        for i in range(series_values.shape[0]):
            gr = compute_instantaneous_growth_rate_classified(
                series_values[i],
                time[i],
                worm_type[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )
            growth_rate.append(gr)

        growth_rate = np.array(growth_rate)

        condition[gr_series_name] = growth_rate

    return conditions_struct


def rescale(
    conditions_struct,
    series_name,
    rescaled_series_name,
    experiment_time=True,
    n_points=100,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
        worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
        worm_type = condition[worm_type_key]
        ecdysis = condition["ecdysis_index"]

        if experiment_time:
            time = condition["experiment_time"]
        else:
            time = condition["time"]

        _, rescaled_series = rescale_series(
            series_values, time, ecdysis, worm_type, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        # reshape into (n_worms, 4*n_points)

        rescaled_series = rescaled_series.reshape(rescaled_series.shape[0], -1)

        condition[rescaled_series_name] = rescaled_series

    return conditions_struct


def rescale_without_flattening(
    conditions_struct,
    series_name,
    rescaled_series_name,
    experiment_time=True,
    n_points=100,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
        worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
        worm_type = condition[worm_type_key]
        ecdysis = condition["ecdysis_index"]

        if experiment_time:
            time = condition["experiment_time"]
        else:
            time = condition["time"]

        _, rescaled_series = rescale_series(
            series_values, time, ecdysis, worm_type, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        condition[rescaled_series_name] = rescaled_series

    return conditions_struct


def exclude_arrests_from_series_at_ecdysis(series_at_ecdysis):
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
