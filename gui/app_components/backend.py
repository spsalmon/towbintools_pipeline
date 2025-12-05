import os
import re

import numpy as np
import polars as pl
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.foundation import image_handling
from towbintools.foundation.worm_features import get_features_to_compute_at_molt

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()

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
    filemap_name = os.path.basename(filemap_path).split(".")[0]

    annotated_name = f"{filemap_name}_annotated.csv"
    annotated_path = os.path.join(filemap_folder, annotated_name)

    # If we want to open the annotated version and it's not already annotated
    if (
        open_annotated
        and os.path.exists(annotated_path)
        and ("annotated" not in filemap_name)
    ):
        print(f"Annotated filemap already exists at {annotated_path}")
        print("Opening the existing annotated filemap instead ...")
        filemap_path = annotated_path
        filemap_save_path = annotated_path
        filemap_name = os.path.basename(filemap_path).split(".")[0]
    elif "annotated" not in filemap_name:
        filemap_save_path = annotated_path
    elif "annotated" in filemap_name:
        filemap_save_path = filemap_path

    # Read the filemap (either original or annotated)
    filemap = pl.read_csv(
        filemap_path,
        infer_schema_length=10000,
        null_values=["np.nan", "[nan]", "", "NaN", "nan", "NA", "N/A"],
    )

    # Backup the filemap
    backup_path = get_backup_path(filemap_folder, filemap_name)
    filemap.write_csv(backup_path)

    return filemap, filemap_save_path


def check_use_experiment_time(filemap):
    """
    Check if the filemap contains the 'ExperimentTime' column and if this column contains valid data.
    """
    if "ExperimentTime" in filemap.columns:
        experiment_time = (
            filemap.select(pl.col("ExperimentTime")).to_numpy().squeeze().astype(float)
        )
        if np.any(np.isfinite(experiment_time)):
            return True
        else:
            return False
    else:
        return False


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
        "Arrest",
        "Ignore",
        "Death",
        "Dead",
    ]

    usual_columns.extend(
        [column for column in filemap.columns if "worm_type" in column]
    )

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        usual_columns.extend(
            [column for column in filemap.columns if feature in column]
        )

    usual_columns.extend([column for column in filemap.columns if "analysis" in column])

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
        filemap = filemap.with_columns(pl.lit("worm").alias(worm_type_column))

    try:
        default_plotted_column = [
            column
            for column in filemap.columns
            if "volume" in column and "_at_" not in column
        ][0]
    except IndexError:
        try:
            default_plotted_column = feature_columns[0]
        except IndexError:
            default_plotted_column = "placeholder_feature"
            filemap = filemap.with_columns(pl.lit(np.nan).alias(default_plotted_column))
            print("No feature column found in the filemap, creating a placeholder.")

    segmentation_columns = [
        column for column in filemap.columns if "seg" in column and "str" not in column
    ]

    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        segmentation_columns = [
            column for column in segmentation_columns if feature not in column
        ]

    overlay_segmentation_choices = ["None"] + segmentation_columns

    return (
        filemap,
        feature_columns,
        custom_columns_choices,
        worm_type_column,
        default_plotted_column,
        overlay_segmentation_choices,
    )


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

    if "ExperimentTime" not in filemap.columns:
        filemap = filemap.with_columns(pl.lit(np.nan).alias("ExperimentTime"))

    experiment_time = separate_column_by_point(filemap, "ExperimentTime").astype(float)

    if np.ndim(ecdysis_time) < 2:
        ecdysis_time = ecdysis_time[np.newaxis, :]
    if np.ndim(time) < 2:
        time = time[np.newaxis, :]
    if np.ndim(experiment_time) < 2:
        experiment_time = experiment_time[np.newaxis, :]

    # convert experiment_time to hours
    experiment_time = experiment_time / 3600

    ecdysis_index = []

    for i in range(len(ecdysis_time)):
        time_of_point = time[i]
        ecdysis_of_point = ecdysis_time[i]

        ecdysis_index.append(
            [
                (
                    float(np.where(time_of_point == ecdysis)[0][0])
                    if ecdysis in time_of_point
                    else np.nan
                )
                for ecdysis in ecdysis_of_point
            ]
        )

    ecdysis_index = np.array(ecdysis_index)

    return time, experiment_time, ecdysis_index


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

    single_values_df = single_values_df.with_columns(
        [pl.col(col).cast(pl.Float64) for col in columns_to_keep]
    )

    return single_values_df


def process_feature_at_molt_columns(
    filemap, feature_columns, worm_type_column, recompute_features_at_molt=False
):
    columns = filemap.columns

    for ecdys in ECDYSIS_COLUMNS:
        if ecdys not in filemap.columns:
            filemap = filemap.with_columns(pl.lit(np.nan).alias(ecdys))

    (
        time,
        experiment_time,
        ecdysis_index,
    ) = get_time_and_ecdysis(filemap)

    worm_types = separate_column_by_point(filemap, worm_type_column)

    unique_points = (
        filemap.select(pl.col("Point")).unique(maintain_order=True).to_numpy().squeeze()
    )

    if unique_points.ndim == 0:
        unique_points = np.array([unique_points])

    for feature_column in feature_columns:
        # convert the feature column to float
        filemap = filemap.with_columns(pl.col(feature_column).cast(pl.Float64))
        series = separate_column_by_point(filemap, feature_column)
        feature_at_ecdysis_columns = [
            f"{feature_column}_at_{ecdys}" for ecdys in ECDYSIS_COLUMNS
        ]
        for column in feature_at_ecdysis_columns:
            if column not in columns:
                filemap = filemap.with_columns(pl.lit(np.nan).alias(column))

        series_at_ecdysis = _get_values_at_molt(filemap, feature_column)

        new_series_at_ecdysis = _compute_series_at_molt(
            series,
            series_at_ecdysis,
            worm_types,
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


def update_molt_and_ecdysis_columns(
    point_filemap,
    single_values_df,
    ecdys_event,
    new_time,
    new_time_index,
    worm_type_column,
    experiment_time=True,
):
    single_values_df = single_values_df.with_columns(
        pl.lit(new_time).alias(ecdys_event)
    )

    worm_type_values = (
        point_filemap.select(pl.col(worm_type_column)).to_numpy().squeeze()
    )
    if experiment_time:
        time = (
            point_filemap.select(pl.col("ExperimentTime")).to_numpy().squeeze() / 3600
        ).astype(float)
    else:
        time = point_filemap.select(pl.col("Time")).to_numpy().squeeze().astype(float)

    value_at_ecdys_columns = [
        column for column in point_filemap.columns if f"_at_{ecdys_event}" in column
    ]

    value_columns = [
        re.sub(r"_at_.*$", "", column) for column in value_at_ecdys_columns
    ]

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        if np.isnan(new_time_index):
            new_value_at_ecdys = np.nan
        else:
            series = (
                point_filemap.select(pl.col(value_column)).to_numpy().squeeze().copy()
            )
            new_value_at_ecdys = compute_series_at_time_classified(
                series,
                time[int(new_time_index)],
                time,
                worm_type_values,
            )

        print(
            f"Old value {value_at_ecdys_column}: {single_values_df.select(pl.col(value_at_ecdys_column)).to_numpy().squeeze()}"
        )

        single_values_df = single_values_df.with_columns(
            pl.lit(new_value_at_ecdys).alias(value_at_ecdys_column)
        )

        print(
            f"New value {value_at_ecdys_column}: {single_values_df.select(pl.col(value_at_ecdys_column)).to_numpy().squeeze()}"
        )

    return single_values_df


def correct_ecdysis_columns(point_filemap, single_values_df, ecdys_event, time_index):
    value_at_ecdys_columns = [
        column for column in point_filemap.columns if f"_at_{ecdys_event}" in column
    ]

    value_columns = [
        re.sub(r"_at_.*$", "", column) for column in value_at_ecdys_columns
    ]

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        new_value_at_ecdys = (
            point_filemap.select(pl.col(value_column))
            .to_numpy()
            .squeeze()
            .copy()[int(time_index)]
        )
        single_values_df = single_values_df.with_columns(
            pl.lit(new_value_at_ecdys).alias(value_at_ecdys_column)
        )

    return single_values_df


def _compute_series_at_molt(
    series,
    series_at_ecdysis,
    worm_types,
    ecdysis_index,
    experiment_time,
    time,
    recompute_values_at_molt=False,
):
    if series_at_ecdysis.ndim < 2:
        series_at_ecdysis = series_at_ecdysis[np.newaxis, :]

    new_series_at_ecdysis = series_at_ecdysis.copy()

    nan_indexes_values_mask = np.isnan(series_at_ecdysis)

    if (~np.isnan(experiment_time)).any():
        time = experiment_time
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

        ecdys = ecdysis[i][idx_values_to_recompute].astype(int)

        recomputed_values = compute_series_at_time_classified(
            series[i],
            time[i][ecdys],
            time[i],
            worm_types[i],
        )

        print(new_series_at_ecdysis[i])
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
    symbols = []
    for worm_type in worm_types:
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
            sizes[int(custom_annotation)] = 12
            colors[int(custom_annotation)] = "pink"

    if np.isfinite(hatch_time):
        try:
            hatch_index = np.where(times_of_point == hatch_time)[0][0]
            symbols[hatch_index] = "square"
            sizes[hatch_index] = 8
            colors[hatch_index] = "red"
        except IndexError:
            print(f"Hatch time {hatch_time} not in list of times")

    if np.isfinite(m1) and m1 in times_of_point:
        try:
            m1_index = np.where(times_of_point == m1)[0][0]
            symbols[m1_index] = "circle"
            sizes[m1_index] = 8
            colors[m1_index] = "orange"
        except IndexError:
            print(f"M1 {m1} not in list of times")

    if np.isfinite(m2):
        try:
            m2_index = np.where(times_of_point == m2)[0][0]
            symbols[m2_index] = "circle"
            sizes[m2_index] = 8
            colors[m2_index] = "yellow"
        except IndexError:
            print(f"M2 {m2} not in list of times")

    if np.isfinite(m3):
        try:
            m3_index = np.where(times_of_point == m3)[0][0]
            symbols[m3_index] = "circle"
            sizes[m3_index] = 8
            colors[m3_index] = "green"
        except IndexError:
            print(f"M3 {m3} not in list of times")

    if np.isfinite(m4):
        try:
            m4_index = np.where(times_of_point == m4)[0][0]
            symbols[m4_index] = "circle"
            sizes[m4_index] = 8
            colors[m4_index] = "blue"
        except IndexError:
            print(f"M4 {m4} not in list of times")

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
