import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from towbintools.data_analysis import compute_larval_stage_duration
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


def add_conditions_to_filemap(experiment_filemap, conditions):
    """Add conditions to experiment filemap based on point ranges or pad values."""
    for condition in conditions:
        if "point_range" in condition:
            point_range = condition["point_range"]

            # handle both single range and list of ranges
            if isinstance(point_range[0], list):
                # multiple point ranges
                mask = False
                for pr in point_range:
                    mask = mask | experiment_filemap["Point"].between(pr[0], pr[1])
                condition_rows = mask
            else:
                # single point range
                condition_rows = experiment_filemap["Point"].between(
                    point_range[0], point_range[1]
                )

            filter_key = "point_range"

        elif "pad" in condition:
            pad = condition["pad"]
            condition_rows = experiment_filemap["Pad"] == pad
            filter_key = "pad"

        else:
            print(
                "Condition does not contain 'point_range' or 'pad' key, impossible to add condition to filemap, skipping."
            )
            continue

        # Extract the condition attributes to add (excluding the filter key)
        conditions_to_add = {k: v for k, v in condition.items() if k != filter_key}

        # Apply all conditions at once using loc
        for key, val in conditions_to_add.items():
            experiment_filemap.loc[condition_rows, key] = val

    return experiment_filemap


def _process_condition_id_plotting_structure(
    experiment_dir,
    experiment_filemap,
    filemap_path,
    organ_channels,
    conditions_keys,
    condition_id,
    recompute_values_at_molt=False,
):
    condition_df = experiment_filemap[
        experiment_filemap["condition_id"] == condition_id
    ]
    condition_dict = {}
    for key in conditions_keys:
        condition_dict[key] = condition_df[key].iloc[0]

    (
        ecdysis_index,
        ecdysis_time_step,
        ecdysis_experiment_time,
        larval_stage_durations_time_step,
        larval_stage_durations_experiment_time,
    ) = _get_ecdysis_and_durations(condition_df)

    condition_dict["condition_id"] = int(condition_dict["condition_id"])
    condition_dict["ecdysis_index"] = ecdysis_index
    condition_dict["ecdysis_time_step"] = ecdysis_time_step
    condition_dict[
        "larval_stage_durations_time_step"
    ] = larval_stage_durations_time_step
    condition_dict["ecdysis_experiment_time"] = ecdysis_experiment_time
    condition_dict[
        "larval_stage_durations_experiment_time"
    ] = larval_stage_durations_experiment_time
    condition_dict["larval_stage_durations_experiment_time_hours"] = (
        larval_stage_durations_experiment_time / 3600
    )
    condition_dict["experiment"] = np.array(
        [experiment_dir] * condition_df["Point"].nunique()
    )[:, np.newaxis]
    condition_dict["filemap_path"] = np.array(
        [filemap_path] * condition_df["Point"].nunique()
    )[:, np.newaxis]
    condition_dict["point"] = np.unique(condition_df["Point"].values)[:, np.newaxis]

    # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
    worm_type_column = [col for col in condition_df.columns if "worm_type" in col][0]
    worm_types = separate_column_by_point(condition_df, worm_type_column)

    condition_dict["time"] = separate_column_by_point(condition_df, "Time").astype(
        float
    )
    condition_dict["experiment_time"] = separate_column_by_point(
        condition_df, "ExperimentTime"
    ).astype(float)

    for organ in organ_channels.keys():
        organ_channel = organ_channels[organ]
        organ_channel = f"ch{organ_channel}"
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

        renamed_organ_feature_columns = [
            col.replace(organ_channel, organ) for col in organ_columns
        ]

        for organ_feature_column, renamed_feature_organ_column in zip(
            organ_feature_columns, renamed_organ_feature_columns
        ):
            condition_dict[renamed_feature_organ_column] = separate_column_by_point(
                condition_df, organ_feature_column
            )
            condition_dict[
                f"{renamed_feature_organ_column}_at_ecdysis"
            ] = _get_values_at_molt(condition_df, organ_feature_column)

        # remove any column with worm_type in it
        renamed_organ_feature_columns = [
            col for col in renamed_organ_feature_columns if "worm_type" not in col
        ]

        # compute the features of the organ at each molt
        for column in renamed_organ_feature_columns:
            column_at_molt = f"{column}_at_ecdysis"

            values_at_molt = condition_dict[column_at_molt]

            nan_indexes_values = np.where(np.isnan(values_at_molt))[0]
            experiment_time = condition_dict["experiment_time"]
            ecdysis = condition_dict["ecdysis_index"]

            if not np.any(np.isnan(experiment_time)):
                time = condition_dict["experiment_time"]
            else:
                time = condition_dict["time"]

            non_nan_indexes_ecdysis = np.where(~np.isnan(ecdysis))[0]

            if recompute_values_at_molt:
                idx_values_to_recompute = non_nan_indexes_ecdysis
            else:
                idx_values_to_recompute = [
                    idx for idx in nan_indexes_values if idx in non_nan_indexes_ecdysis
                ]

            for idx in idx_values_to_recompute:
                values_at_molt[idx] = compute_series_at_time_classified(
                    condition_dict[column][idx],
                    worm_types[idx],
                    ecdysis_index[idx],
                    series_time=time[idx],
                )
            condition_dict[column_at_molt] = values_at_molt

    return condition_dict


def build_plotting_struct(
    experiment_dir,
    filemap_path,
    conditions_yaml_path,
    organ_channels={"body": 2, "pharynx": 1},
    recompute_values_at_molt=False,
):
    experiment_filemap = pd.read_csv(filemap_path)

    conditions = build_conditions(conditions_yaml_path)
    conditions_keys = list(conditions[0].keys())

    # remove 'point_range' and 'pad' from the conditions keys if they are present
    if "point_range" in conditions_keys:
        conditions_keys.remove("point_range")
    if "pad" in conditions_keys:
        conditions_keys.remove("pad")

    experiment_filemap = add_conditions_to_filemap(
        experiment_filemap,
        conditions,
    )

    experiment_filemap.columns

    # if ExperimentTime is not present in the filemap, add it
    if "ExperimentTime" not in experiment_filemap.columns:
        experiment_filemap["ExperimentTime"] = np.nan

    # remove rows where condition_id is NaN
    experiment_filemap = experiment_filemap[
        ~experiment_filemap["condition_id"].isnull()
    ]

    # set molts that should be ignored to NaN
    if "Ignore" in experiment_filemap.columns:
        experiment_filemap = remove_ignored_molts(experiment_filemap)

    # remove rows where Ignore is True
    if "Ignore" in experiment_filemap.columns:
        experiment_filemap = experiment_filemap[~experiment_filemap["Ignore"]]

    conditions_struct = []

    for condition_id in experiment_filemap["condition_id"].unique():
        condition_dict = _process_condition_id_plotting_structure(
            experiment_dir,
            experiment_filemap,
            filemap_path,
            organ_channels,
            conditions_keys,
            condition_id,
            recompute_values_at_molt=recompute_values_at_molt,
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


def _get_ecdysis_and_durations(filemap):
    all_ecdysis_time_step = []
    all_ecdysis_index = []
    all_durations_time_step = []

    all_ecdysis_experiment_time = []
    all_durations_experiment_time = []

    for point in filemap["Point"].unique():
        point_df = filemap[filemap["Point"] == point]
        point_time = point_df["Time"].values
        point_ecdysis = point_df[["HatchTime", "M1", "M2", "M3", "M4"]].iloc[0]

        point_ecdysis_index = []
        for ecdysis in point_ecdysis:
            matches = np.where(point_time == ecdysis)[0]

            if len(matches) == 0:
                point_ecdysis_index.append(np.nan)
            else:
                point_ecdysis_index.append(float(matches[0]))

        larval_stage_durations = list(
            compute_larval_stage_duration(point_ecdysis).values()
        )

        point_ecdysis = point_ecdysis.to_numpy()
        all_ecdysis_time_step.append(point_ecdysis)
        all_ecdysis_index.append(point_ecdysis_index)
        all_durations_time_step.append(larval_stage_durations)

        ecdysis_experiment_time = []
        for ecdys in point_ecdysis:
            if np.isnan(ecdys):
                ecdysis_experiment_time.append(np.nan)
            else:
                # if ecdys is not in the time column, set it to nan
                if ecdys not in point_df["Time"].values:
                    ecdys_experiment_time = np.nan
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
        np.array(all_ecdysis_index),
        np.array(all_ecdysis_time_step),
        np.array(all_ecdysis_experiment_time),
        np.array(all_durations_time_step),
        np.array(all_durations_experiment_time),
    )


def _get_values_at_molt(filemap, column):
    all_values = []
    ecdysis = ["HatchTime", "M1", "M2", "M3", "M4"]

    columns_at_ecdysis = [f"{column}_at_{e}" for e in ecdysis]

    print(f"Columns at ecdysis: {columns_at_ecdysis}")

    for point in filemap["Point"].unique():
        point_df = filemap.loc[filemap["Point"] == point].copy()

        for col in columns_at_ecdysis:
            if col not in point_df.columns:
                print(f"Column {col} not found in point {point}, adding it.")
                point_df[col] = np.nan

        values_at_ecdysis_point = point_df[columns_at_ecdysis].to_numpy()

        print(
            f"Values at ecdysis for point shape {point}: {values_at_ecdysis_point.shape}"
        )

        all_values.append(values_at_ecdysis_point)

    return np.array(all_values)


def separate_column_by_point(filemap, column):
    max_number_of_values = np.max(
        [
            len(filemap[filemap["Point"] == point][column].values)
            for point in filemap["Point"].unique()
        ]
    )

    all_values = []
    for i, point in enumerate(filemap["Point"].unique()):
        point_df = filemap[filemap["Point"] == point]
        values_of_point = point_df[column].values

        if isinstance(values_of_point[0], str):
            dtype = str
            pad_value = "error"
        else:
            dtype = float
            pad_value = np.nan

        values_of_point = np.array(values_of_point, dtype=dtype)
        values_of_point = np.pad(
            values_of_point,
            (0, max_number_of_values - len(values_of_point)),
            mode="constant",
            constant_values=pad_value,
        )

        all_values.append(values_of_point)

    return np.array(all_values)


def remove_ignored_molts(filemap):
    df = filemap.copy()

    molt_columns = ["HatchTime", "M1", "M2", "M3", "M4"]

    for point in df["Point"].unique():
        point_mask = df["Point"] == point
        point_df = df[point_mask]

        if point_df.empty:
            continue

        # Get molt times for this point
        molt_times = point_df[molt_columns].iloc[0]

        # Check each molt time
        for col, molt_time in molt_times.items():
            if pd.isna(molt_time):
                continue

            # Find if this molt should be ignored
            try:
                if point_df[point_df["Time"] == molt_time]["Ignore"].iloc[0]:
                    # Use .loc to avoid chained indexing warning
                    df.loc[point_mask, col] = np.nan
            except IndexError:
                print(f"No row found for time {molt_time} in point {point}")
                df.loc[point_mask, col] = np.nan

    return df


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
    organ_channels=[{"body": 2, "pharynx": 1}],
    recompute_values_at_molt=False,
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
