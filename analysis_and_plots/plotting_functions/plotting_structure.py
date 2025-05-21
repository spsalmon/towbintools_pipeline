import os
from collections import defaultdict
from time import perf_counter

import numpy as np
import polars as pl
import yaml
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.foundation.worm_features import get_features_to_compute_at_molt

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()

# THIS PART HANDLES THE PROCESSING OF THE EXPERIMENT FILEMAP AND THE CREATION OF THE PLOTTING STRUCTURE


def build_conditions(conditions_yaml):
    """
    Process a conditions YAML file structured in a factorized way. Refer to the documentation for the expected format.
    The function will return a list of dictionaries, each representing a condition with its parameters.

    Parameters:
        conditions_yaml (str or dict): Path to the YAML file or a dictionary containing the conditions.
    Returns:
        list: A list of dictionaries, each representing a condition with its parameters.
    Raises:
        ValueError: If the conditions YAML file is not structured correctly.
    """
    if isinstance(conditions_yaml, str):
        with open(conditions_yaml) as file:
            conditions_dict = yaml.safe_load(file)
            file.close()
    elif isinstance(conditions_yaml, dict):
        conditions_dict = conditions_yaml

    conditions = []
    condition_id = 0

    for condition in conditions_dict["conditions"]:
        condition = {
            key: [val] if not isinstance(val, list) else val
            for key, val in condition.items()
        }

        lengths = {len(val) for val in condition.values()}
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


def _add_conditions_to_filemap(experiment_filemap, conditions):
    """Add conditions to experiment filemap based on point ranges or pad values."""
    if "Pad" in experiment_filemap.columns:
        new_columns = experiment_filemap.select(pl.col("Point"), pl.col("Pad"))
    else:
        new_columns = experiment_filemap.select(pl.col("Point"))
    for condition in conditions:
        # Get condition rows mask
        if "point_range" in condition:
            point_range = condition["point_range"]
            # handle both single range and list of ranges
            if isinstance(point_range[0], list):
                # multiple point ranges - build expressions list
                mask_exprs = [
                    (pl.col("Point") >= pr[0]) & (pl.col("Point") <= pr[1])
                    for pr in point_range
                ]
                # Combine with or operator
                condition_mask = mask_exprs[0]
                for expr in mask_exprs[1:]:
                    condition_mask = condition_mask | expr
            else:
                # single point range
                condition_mask = (pl.col("Point") >= point_range[0]) & (
                    pl.col("Point") <= point_range[1]
                )
            filter_key = "point_range"
        elif "pad" in condition:
            pad = condition["pad"]
            condition_mask = pl.col("Pad") == pad
            filter_key = "pad"
        else:
            print(
                "Condition does not contain 'point_range' or 'pad' key, impossible to add condition to filemap, skipping."
            )
            continue

        # Extract the condition attributes to add (excluding the filter key)
        conditions_to_add = {k: v for k, v in condition.items() if k != filter_key}

        column_ops = []
        for key, val in conditions_to_add.items():
            # For new columns, add with null values first if needed
            if key not in new_columns.columns:
                new_columns = new_columns.with_columns(pl.lit(None).alias(key))

            # Build the column operation but don't apply it yet
            column_ops.append(
                pl.when(condition_mask)
                .then(pl.lit(val))
                .otherwise(pl.col(key))
                .alias(key)
            )

        # Add the new columns to the DataFrame
        if column_ops:
            new_columns = new_columns.with_columns(column_ops)

    if "Pad" in new_columns.columns:
        new_columns = new_columns.drop(["Point", "Pad"])
    else:
        new_columns = new_columns.drop(["Point"])

    # Apply the new columns to the experiment filemap
    if not new_columns.is_empty():
        experiment_filemap = experiment_filemap.with_columns(new_columns)

    return experiment_filemap


# TODO: Instead of doing it for each condition, do it for all conditions at once, then split
def _process_condition_id_plotting_structure(
    experiment_dir,
    experiment_filemap,
    filemap_path,
    organ_channels,
    conditions_keys,
    condition_id,
    recompute_values_at_molt=False,
    rescale_n_points=100,
):
    condition_df = experiment_filemap.filter(pl.col("condition_id") == condition_id)
    condition_dict = {}

    for key in conditions_keys:
        condition_dict[key] = condition_df.select(pl.col(key))[0].item()

    (
        time,
        experiment_time,
        ecdysis_index,
        ecdysis_time_step,
        ecdysis_experiment_time,
        larval_stage_durations_time_step,
        larval_stage_durations_experiment_time,
    ) = _get_time_ecdysis_and_durations(condition_df)

    n_points = condition_df.select(pl.col("Point")).n_unique()

    # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
    worm_type_column = [col for col in condition_df.columns if "worm_type" in col][0]
    worm_types = separate_column_by_point(condition_df, worm_type_column)

    condition_dict.update(
        {
            "condition_id": int(condition_dict["condition_id"]),
            "ecdysis_index": ecdysis_index,
            "ecdysis_time_step": ecdysis_time_step,
            "larval_stage_durations_time_step": larval_stage_durations_time_step,
            "ecdysis_experiment_time": ecdysis_experiment_time,
            "ecdysis_experiment_time_hours": ecdysis_experiment_time / 3600,
            "larval_stage_durations_experiment_time": larval_stage_durations_experiment_time,
            "larval_stage_durations_experiment_time_hours": larval_stage_durations_experiment_time
            / 3600,
            "experiment": np.full((n_points, 1), experiment_dir),
            "filemap_path": np.full((n_points, 1), filemap_path),
            "point": condition_df.select(pl.col("Point").unique()).to_numpy(),
            "time": time,
            "experiment_time": experiment_time,
            "experiment_time_hours": experiment_time / 3600,
            "worm_type": worm_types,
        }
    )

    for organ in organ_channels.keys():
        organ_channel = organ_channels[organ]
        organ_columns = [
            col for col in condition_df.columns if col.startswith(organ_channel)
        ]

        # remove any column with _at_ in it
        organ_columns = [col for col in organ_columns if "_at_" not in col]

        # get the columns that contain the interesting features
        organ_feature_columns = []
        for feature in FEATURES_TO_COMPUTE_AT_MOLT:
            organ_feature_columns.extend(
                [col for col in organ_columns if feature in col]
            )

        for organ_feature_column in organ_feature_columns:
            # rename the column to remove the organ channel
            renamed_feature_organ_column = organ_feature_column.replace(
                organ_channel, organ
            )

            condition_dict[renamed_feature_organ_column] = separate_column_by_point(
                condition_df, organ_feature_column
            )
            condition_dict[
                f"{renamed_feature_organ_column}_at_ecdysis"
            ] = _get_values_at_molt(condition_df, organ_feature_column)

            if "worm_type" not in organ_feature_column:
                condition_dict = _compute_values_at_molt(
                    condition_dict,
                    renamed_feature_organ_column,
                    worm_types,
                    recompute_values_at_molt=recompute_values_at_molt,
                )

    return condition_dict


def build_plotting_struct(
    experiment_dir,
    filemap_path,
    conditions_yaml_path,
    organ_channels={"body": "ch2", "pharynx": "ch1"},
    recompute_values_at_molt=False,
    rescale_n_points=100,
):
    experiment_filemap = pl.read_csv(
        filemap_path,
        infer_schema_length=10000,
        null_values=["np.nan", "[nan]", ""],
    )

    conditions = build_conditions(conditions_yaml_path)
    conditions_keys = list(conditions[0].keys())

    # remove 'point_range' and 'pad' from the conditions keys if they are present
    if "point_range" in conditions_keys:
        conditions_keys.remove("point_range")
    if "pad" in conditions_keys:
        conditions_keys.remove("pad")

    experiment_filemap = _add_conditions_to_filemap(
        experiment_filemap,
        conditions,
    )

    experiment_filemap.write_csv("test.csv")

    # if ExperimentTime is not present in the filemap, add it
    if "ExperimentTime" not in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.with_columns(
            pl.lit(np.nan).alias("ExperimentTime")
        )

    # remove rows where condition_id is null
    experiment_filemap = experiment_filemap.filter(~pl.col("condition_id").is_null())

    start = perf_counter()

    # set molts that should be ignored to NaN
    if "Ignore" in experiment_filemap.columns:
        experiment_filemap = remove_ignored_molts(experiment_filemap)

    # remove rows where Ignore is True
    if "Ignore" in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.filter(~pl.col("Ignore"))

    print(f"Processing ignore times took {perf_counter() - start:.2f} seconds")
    conditions_struct = []

    for condition_id in (
        experiment_filemap.select(pl.col("condition_id"))
        .unique(maintain_order=True)
        .to_numpy()
        .squeeze()
    ):
        condition_dict = _process_condition_id_plotting_structure(
            experiment_dir,
            experiment_filemap,
            filemap_path,
            organ_channels,
            conditions_keys,
            condition_id,
            recompute_values_at_molt=recompute_values_at_molt,
            rescale_n_points=rescale_n_points,
        )

        conditions_struct.append(condition_dict)

    conditions_info = [
        {key: condition[key] for key in conditions_keys}
        for condition in conditions_struct
    ]

    # sort the conditions and conditions_info by condition_id
    conditions_struct = sorted(conditions_struct, key=lambda x: x["condition_id"])
    conditions_info = sorted(conditions_info, key=lambda x: x["condition_id"])

    return conditions_struct, conditions_info


def _compute_larval_stage_duration(ecdysis_array):
    durations = np.full(len(ecdysis_array) - 1, np.nan)
    for i, (start, end) in enumerate(zip(ecdysis_array[:-1], ecdysis_array[1:])):
        # check if start or end is NaN
        if np.isnan(start) or np.isnan(end):
            durations[i] = np.nan
        else:
            durations[i] = end - start
    return durations


def _get_time_ecdysis_and_durations(filemap):
    all_ecdysis_time_step = []
    all_ecdysis_index = []
    all_durations_time_step = []

    all_ecdysis_experiment_time = []
    all_durations_experiment_time = []

    ecdysis_columns = ["HatchTime", "M1", "M2", "M3", "M4"]
    column_list = ["Point", "Time", "ExperimentTime"] + ecdysis_columns
    filemap = filemap.select(pl.col(column_list))

    ecdysis_values = (
        filemap.group_by("Point", maintain_order=True)
        .agg(pl.col(ecdysis_columns).first())
        .drop("Point")
        .cast(pl.Float64)
    )

    time = separate_column_by_point(filemap, "Time").astype(float)
    experiment_time = separate_column_by_point(filemap, "ExperimentTime").astype(float)

    for i in range(len(ecdysis_values)):
        ecdysis = ecdysis_values[i].to_numpy().squeeze()
        time_of_point = time[i]
        experiment_time_of_point = experiment_time[i]

        ecdysis_index = [
            float(np.where(time_of_point == ecdysis)[0][0])
            if ecdysis in time_of_point
            else np.nan
            for ecdysis in ecdysis
        ]
        ecdysis_experiment_time = [
            experiment_time_of_point[int(index)] if not np.isnan(index) else np.nan
            for index in ecdysis_index
        ]
        all_ecdysis_time_step.append(ecdysis)
        all_ecdysis_index.append(ecdysis_index)
        all_ecdysis_experiment_time.append(ecdysis_experiment_time)

        larval_stage_durations = _compute_larval_stage_duration(ecdysis)
        larval_stage_durations_experiment_time = _compute_larval_stage_duration(
            ecdysis_experiment_time
        )
        all_durations_time_step.append(larval_stage_durations)
        all_durations_experiment_time.append(larval_stage_durations_experiment_time)

    return (
        time,
        experiment_time,
        np.array(all_ecdysis_index),
        np.array(all_ecdysis_time_step),
        np.array(all_ecdysis_experiment_time),
        np.array(all_durations_time_step),
        np.array(all_durations_experiment_time),
    )


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


def _compute_values_at_molt(
    condition_dict,
    column,
    worm_types,
    recompute_values_at_molt=False,
):
    column_at_molt = f"{column}_at_ecdysis"

    values_at_molt = condition_dict[column_at_molt]
    updated_values_at_molt = values_at_molt.copy()

    nan_indexes_values_mask = np.isnan(values_at_molt)
    experiment_time = condition_dict["experiment_time_hours"]

    if (~np.isnan(experiment_time)).any():
        time = condition_dict["experiment_time_hours"]
        ecdysis = condition_dict["ecdysis_experiment_time_hours"]
    else:
        time = condition_dict["time"]
        ecdysis = condition_dict["ecdysis_index"]

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
            condition_dict[column][i],
            ecdys,
            time[i],
            worm_types[i],
        )

        updated_values_at_molt[i][idx_values_to_recompute] = recomputed_values

    condition_dict[column_at_molt] = updated_values_at_molt
    return condition_dict


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


def remove_ignored_molts(filemap):
    molt_columns = ["HatchTime", "M1", "M2", "M3", "M4"]

    # Only process if "Ignore" column exists
    if "Ignore" not in filemap.columns:
        return filemap

    # Get all rows where Ignore is True
    ignored_rows = filemap.filter(pl.col("Ignore"))

    # Build a set of (Point, Time) pairs to ignore
    ignored_points = ignored_rows.select(pl.col("Point")).to_numpy().flatten()
    ignored_times = ignored_rows.select(pl.col("Time")).to_numpy().flatten()
    ignored_pairs = set(zip(ignored_points, ignored_times))

    # For each molt column, set to None where (Point, molt_time) is in ignored_pairs
    for col in molt_columns:
        molt_times = filemap.select(pl.col(col)).to_numpy().flatten()
        points = filemap.select(pl.col("Point")).to_numpy().flatten()
        mask = np.array(
            [
                (p, mt) in ignored_pairs
                if mt is not None and not (isinstance(mt, float) and np.isnan(mt))
                else False
                for p, mt in zip(points, molt_times)
            ]
        )
        if mask.any():
            filemap = filemap.with_columns(
                pl.when(pl.Series(mask)).then(None).otherwise(pl.col(col)).alias(col)
            )

    return filemap


def remove_unwanted_info(conditions_info):
    for condition in conditions_info:
        if "description" in condition.keys():
            condition.pop("description")
        if "condition_id" in condition.keys():
            condition.pop("condition_id")
    return conditions_info


def combine_experiments(
    filemap_paths,
    config_paths,
    experiment_dirs=None,
    organ_channels=[{"body": "ch2", "pharynx": "ch1"}],
    recompute_values_at_molt=False,
    rescale_n_points=100,
):
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
    for i, (filemap_path, config_path, organ_channel) in enumerate(
        zip(filemap_paths, config_paths, organ_channels)
    ):
        experiment_dir = (
            experiment_dirs[i] if experiment_dirs else os.path.dirname(filemap_path)
        )

        conditions_struct, conditions_info = build_plotting_struct(
            experiment_dir,
            filemap_path,
            config_path,
            organ_channels=organ_channel,
            recompute_values_at_molt=recompute_values_at_molt,
            rescale_n_points=rescale_n_points,
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
