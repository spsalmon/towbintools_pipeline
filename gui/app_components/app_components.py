import os
import re

import numpy as np
import polars as pl
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.foundation import image_handling
from towbintools.foundation.worm_features import get_features_to_compute_at_molt

# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly.graph_objs as go
# import scipy.io as sio
# from shiny import App
# from shiny import reactive
# from shiny import render
# from shiny import ui
# from shinywidgets import output_widget
# from shinywidgets import render_widget

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()

KEY_CONVERSION_MAP = {
    "vol": "volume",
    "len": "length",
    "strClass": "worm_type",
    "ecdys": "ecdysis",
}

ECDYSIS_COLUMNS = ["HatchTime", "M1", "M2", "M3", "M4"]


def get_backup_path(filemap_folder, filemap_name):
    # check if the filemap is already annotated
    match = re.search(r"annotated_v(/d+)", filemap_name)
    if not match:
        iteration = 1
    else:
        iteration = int(match.group(1))

    filemap_save_path = f"{filemap_name}_v{iteration}.csv"
    while os.path.exists(os.path.join(filemap_folder, filemap_save_path)):
        iteration += 1
        filemap_save_path = f"{filemap_name}_v{iteration}.csv"

    filemap_save_path = os.path.join(filemap_folder, filemap_save_path)
    return filemap_save_path


def open_filemap(filemap_path, open_annotated=True):
    filemap_folder = os.path.dirname(filemap_path)
    filemap_name = os.path.basename(filemap_path)
    filemap_name = filemap_name.split(".")[0]

    if "annotated" not in filemap_name and open_annotated:
        filemap_save_path = f"{filemap_name}_annotated.csv"
        filemap_save_path = os.path.join(filemap_folder, filemap_save_path)

        if os.path.exists(filemap_save_path):
            print(f"Annotated filemap already exists at {filemap_save_path}")
            print("Opening the existing filemap instead ...")
            filemap = pl.read_csv(filemap_save_path, infer_schema_length=10000)
            filemap_path = filemap_save_path
            filemap_name = os.path.basename(filemap_path)
            filemap_name = filemap_name.split(".")[0]

            # backup the filemap
            backup_path = get_backup_path(filemap_folder, filemap_name)
            filemap.write_csv(backup_path)
    else:
        # backup the filemap
        filemap = pl.read_csv(filemap_path, infer_schema_length=10000)
        backup_path = get_backup_path(filemap_folder, filemap_name)
        filemap.write_csv(backup_path)
        filemap_save_path = filemap_path

    return filemap, filemap_save_path


def infer_n_channels(filemap):
    first_image_path = filemap.select(pl.col("raw")).to_numpy().squeeze()[0]
    first_image = image_handling.read_tiff_file(first_image_path)

    if first_image.ndim == 3:
        n_channels = first_image.shape[0]
    elif first_image.ndim == 4:
        n_channels = first_image.shape[1]
    elif first_image.ndim == 2:
        n_channels = 1
    else:
        raise ValueError("Unknown number of channels")

    return n_channels


def populate_column_choices(filemap):
    usual_columns = [
        "Time",
        "ExperimentTime",
        "Point",
        "raw",
        "HatchTime",
        "M1",
        "M2",
        "M3",
        "M4",
    ]

    usual_columns.extend(
        [column for column in filemap.columns if "worm_type" in column]
    )

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        usual_columns.extend(
            [column for column in filemap.columns if feature in column]
        )

    feature_columns = []
    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        feature_columns.extend(
            [
                column
                for column in filemap.columns
                if feature in column and "_at_" not in column
            ]
        )

    custom_columns = [
        column for column in filemap.columns if column not in usual_columns
    ]

    # add None to the list of custom columns
    custom_columns_choices = ["None"] + custom_columns

    try:
        worm_type_column = [
            column for column in filemap.columns if "worm_type" in column
        ][0]
    except IndexError:
        print("No worm_type column found in the filemap, creating a placeholder.")
        worm_type_column = "placeholder_worm_type"
        filemap["placeholder_worm_type"] = "worm"

    default_plotted_column = [
        column
        for column in filemap.columns
        if "volume" in column and "_at_" not in column
    ][0]

    segmentation_columns = [
        column for column in filemap.columns if "seg" in column and "str" not in column
    ]

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        segmentation_columns = [
            column for column in segmentation_columns if feature not in column
        ]

    overlay_segmentation_choices = ["None"] + segmentation_columns

    return (
        feature_columns,
        custom_columns_choices,
        worm_type_column,
        default_plotted_column,
        overlay_segmentation_choices,
    )


def save_filemap(filemap, filemap_path):
    print("Saving filemap ...")
    filemap.write_csv(filemap_path)
    print("Filemap saved !")


def separate_column_by_point(filemap, column):
    points = (
        filemap.select(pl.col("Point").unique(maintain_order=True).sort())
        .to_numpy()
        .flatten()
    )

    filemap_points = filemap.select(pl.col("Point"), pl.col(column))
    point_dataframes = filemap_points.partition_by("Point", maintain_order=True)

    sample = point_dataframes[0].select(pl.col(column)).head(1).item()
    is_string = isinstance(sample, str) or (
        hasattr(sample, "dtype") and np.issubdtype(sample.dtype, np.str_)
    )

    max_height = max(point_df.height for point_df in point_dataframes)
    if is_string:
        result = np.full((len(points), max_height), "error", dtype=object)
    else:
        result = np.full((len(points), max_height), np.nan)

    for i, point_df in enumerate(point_dataframes):
        point_column = point_df.select(pl.col(column)).to_numpy().squeeze()
        result[i, : len(point_column)] = point_column
    return result


def get_time_and_ecdysis(filemap):
    ecdysis_df = filemap.select(pl.col(ECDYSIS_COLUMNS + ["Point"]))
    ecdysis_time = (
        (
            ecdysis_df.group_by("Point", maintain_order=True)
            .agg(pl.col(ECDYSIS_COLUMNS).first())
            .select(pl.col(ECDYSIS_COLUMNS))
        )
        .to_numpy()
        .squeeze()
    )

    time = separate_column_by_point(filemap, "Time").astype(float)
    experiment_time = separate_column_by_point(filemap, "ExperimentTime").astype(float)
    # convert experiment_time to hours
    experiment_time = experiment_time / 3600

    ecdysis_index = []
    ecdysis_experiment_time = []

    for i in range(len(ecdysis_time)):
        time_of_point = time[i]
        experiment_time_of_point = experiment_time[i]
        ecdysis_of_point = ecdysis_time[i]
        ecdysis_index.append(
            [
                float(np.where(time_of_point == ecdysis)[0][0])
                if ecdysis in time_of_point
                else np.nan
                for ecdysis in ecdysis_of_point
            ]
        )
        ecdysis_experiment_time.append(
            [
                experiment_time_of_point[int(index)] if not np.isnan(index) else np.nan
                for index in ecdysis_index[i]
            ]
        )

    ecdysis_index = np.array(ecdysis_index)
    ecdysis_experiment_time = np.array(ecdysis_experiment_time)

    return time, experiment_time, ecdysis_time, ecdysis_index, ecdysis_experiment_time


def build_single_values_df(filemap):
    columns = filemap.columns

    columns_to_keep = ["Point"]
    columns_to_keep.extend([col for col in columns if "_at_" in col])
    columns_to_keep.extend(ECDYSIS_COLUMNS)

    single_values_df = filemap.select(pl.col(columns_to_keep))

    columns_to_keep.remove("Point")

    print(single_values_df.columns)
    single_values_df = single_values_df.group_by("Point", maintain_order=True).agg(
        pl.col(columns_to_keep).first()
    )

    return single_values_df


def process_feature_at_molt_columns(
    filemap, feature_columns, worm_type_column, recompute_features_at_molt=False
):
    columns = filemap.columns

    for ecdys in ECDYSIS_COLUMNS:
        if ecdys not in filemap.columns:
            filemap.with_columns(pl.lit(np.nan).alias(ecdys))

    (
        time,
        experiment_time,
        ecdysis_time,
        ecdysis_index,
        ecdysis_experiment_time,
    ) = get_time_and_ecdysis(filemap)

    worm_types = separate_column_by_point(filemap, worm_type_column)

    unique_points = (
        filemap.select(pl.col("Point")).unique(maintain_order=True).to_numpy().squeeze()
    )

    for feature_column in feature_columns:
        series = separate_column_by_point(filemap, feature_column)
        feature_at_ecdysis_columns = [
            f"{feature_column}_at_{ecdys}" for ecdys in ECDYSIS_COLUMNS
        ]
        for column in feature_at_ecdysis_columns:
            if column not in columns:
                filemap.with_columns(pl.lit(np.nan).alias(column))

        series_at_ecdysis = _get_values_at_molt(filemap, feature_column)

        new_series_at_ecdysis = _compute_series_at_molt(
            series,
            series_at_ecdysis,
            worm_types,
            ecdysis_experiment_time,
            ecdysis_index,
            experiment_time,
            time,
            recompute_values_at_molt=recompute_features_at_molt,
        )

        # check if the new series is different from the old one
        if not np.allclose(series_at_ecdysis, new_series_at_ecdysis, equal_nan=True):
            series_at_ecdysis = new_series_at_ecdysis

        else:
            print(f"{feature_column} at ecdysis is already computed, skipping ...")
            continue

        # For each column, create an expression that handles all points
        updated_df = pl.DataFrame(
            {
                "Point": unique_points,
                **{
                    feature_at_ecdysis_columns[j]: [
                        series_at_ecdysis[i][j] for i in range(len(unique_points))
                    ]
                    for j in range(len(feature_at_ecdysis_columns))
                },
            }
        )

        filemap = filemap.drop(feature_at_ecdysis_columns)
        filemap = filemap.join(updated_df, on="Point", how="left")

    return filemap


def _get_values_at_molt(filemap, column):
    ecdysis = ["HatchTime", "M1", "M2", "M3", "M4"]
    columns_at_ecdysis = [f"{column}_at_{e}" for e in ecdysis]
    column_list = ["Point"] + columns_at_ecdysis
    filemap = filemap.select(pl.col(column_list))

    values_at_ecdysis = (
        (
            filemap.group_by("Point", maintain_order=True)
            .agg(pl.col(columns_at_ecdysis).first())
            .drop("Point")
            .cast(pl.Float64)
        )
        .to_numpy()
        .squeeze()
    )

    return values_at_ecdysis


def _compute_series_at_molt(
    series,
    series_at_ecdysis,
    worm_types,
    ecdysis_experiment_time,
    ecdysis_index,
    experiment_time,
    time,
    recompute_values_at_molt=False,
):
    new_series_at_ecdysis = series_at_ecdysis.copy()

    nan_indexes_values_mask = np.isnan(series_at_ecdysis)

    if (~np.isnan(experiment_time)).any():
        time = experiment_time
        ecdysis = ecdysis_experiment_time
    else:
        time = time
        ecdysis = ecdysis_index

    non_nan_indexes_ecdysis_mask = np.invert(np.isnan(ecdysis))

    if recompute_values_at_molt:
        values_to_recompute_mask = non_nan_indexes_ecdysis_mask
    else:
        values_to_recompute_mask = (
            nan_indexes_values_mask & non_nan_indexes_ecdysis_mask
        )

    for i in range(len(values_to_recompute_mask)):
        mask = values_to_recompute_mask[i]
        idx_values_to_recompute = np.where(mask)[0]

        if len(idx_values_to_recompute) == 0:
            continue

        ecdys = ecdysis[i][idx_values_to_recompute]

        recomputed_values = compute_series_at_time_classified(
            series[i],
            worm_types[i],
            ecdys,
            series_time=time[i],
        )

        new_series_at_ecdysis[i][idx_values_to_recompute] = recomputed_values

    return new_series_at_ecdysis


def set_marker_shape(
    times_of_point,
    selected_time_index,
    worm_types,
    hatch_time,
    m1,
    m2,
    m3,
    m4,
    custom_annotations: list = [],
):
    # fill the gaps in the times_of_point list
    times_of_point_filled = np.arange(
        np.min(times_of_point), np.max(times_of_point) + 1
    )

    worm_types_filled = [""] * len(times_of_point_filled)
    for time, worm_type in zip(times_of_point, worm_types):
        worm_types_filled[time] = worm_type

    symbols = []
    for worm_type in worm_types_filled:
        if worm_type == "egg":
            symbol = "square-open"
        elif worm_type == "worm":
            symbol = "circle-open"
        elif worm_type != "":
            symbol = "triangle-up-open"
        else:
            symbol = ""
        symbols.append(symbol)

    # create a list full of "blue"
    sizes = [4] * len(symbols)
    colors = ["black"] * len(symbols)

    for custom_annotation in custom_annotations:
        if np.isfinite(custom_annotation):
            symbols[int(custom_annotation)] = "circle"
            sizes[int(custom_annotation)] = 8
            colors[int(custom_annotation)] = "pink"

    if np.isfinite(hatch_time):
        symbols[int(hatch_time)] = "square"
        sizes[int(hatch_time)] = 8
        colors[int(hatch_time)] = "red"

    if np.isfinite(m1):
        symbols[int(m1)] = "circle"
        sizes[int(m1)] = 8
        colors[int(m1)] = "orange"

    if np.isfinite(m2):
        symbols[int(m2)] = "circle"
        sizes[int(m2)] = 8
        colors[int(m2)] = "yellow"

    if np.isfinite(m3):
        symbols[int(m3)] = "circle"
        sizes[int(m3)] = 8
        colors[int(m3)] = "green"

    if np.isfinite(m4):
        symbols[int(m4)] = "circle"
        sizes[int(m4)] = 8
        colors[int(m4)] = "blue"

    widths = [1] * len(symbols)
    widths[int(selected_time_index)] = 4

    # find the index of all empty symbols
    symbols, sizes, colors, widths = zip(
        *[
            (symbol, size, color, width)
            for symbol, size, color, width in zip(symbols, sizes, colors, widths)
            if symbol != ""
        ]
    )

    markers = dict(symbol=symbols, size=sizes, color=colors, line=dict(width=widths))
    return markers


def get_points_for_value_at_molts(
    hatch,
    m1,
    m2,
    m3,
    m4,
    value_at_hatch,
    value_at_m1,
    value_at_m2,
    value_at_m3,
    value_at_m4,
):
    ecdys_list = [hatch, m1, m2, m3, m4]
    value_at_ecdys_list = [
        value_at_hatch,
        value_at_m1,
        value_at_m2,
        value_at_m3,
        value_at_m4,
    ]
    symbols = ["cross", "cross", "cross", "cross", "cross"]
    colors = ["red", "orange", "yellow", "green", "blue"]
    sizes = [8, 8, 8, 8, 8]
    widths = [4, 4, 4, 4, 4]

    # Use numpy to handle NaN values efficiently
    ecdys_array = np.array(ecdys_list)
    value_at_ecdys_array = np.array(value_at_ecdys_list)
    valid_mask = np.isfinite(ecdys_array) & np.isfinite(value_at_ecdys_array)

    # Filter arrays using the mask
    ecdys_filtered = ecdys_array[valid_mask]
    value_at_ecdys_filtered = value_at_ecdys_array[valid_mask]
    symbols_filtered = np.array(symbols)[valid_mask]
    colors_filtered = np.array(colors)[valid_mask]
    sizes_filtered = np.array(sizes)[valid_mask]
    widths_filtered = np.array(widths)[valid_mask]

    return (
        ecdys_filtered,
        value_at_ecdys_filtered,
        symbols_filtered,
        colors_filtered,
        sizes_filtered,
        widths_filtered,
    )
