# import os
# import shutil
# from typing import Any
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import statsmodels.api as sm
# from plotting_functions.utils_plotting import build_legend
# from plotting_functions.utils_plotting import set_scale
# from scipy.interpolate import make_interp_spline
# from scipy.signal import medfilt
# from tifffile import imwrite
# from towbintools.data_analysis import correct_series_with_classification
# from towbintools.data_analysis import rescale_and_aggregate
# from towbintools.foundation.image_handling import pad_images_to_same_dim
# from towbintools.foundation.image_handling import pad_to_dim
# from towbintools.foundation.image_handling import pad_to_dim_equally
# from towbintools.foundation.image_handling import read_tiff_file
# from towbintools.foundation.worm_features import get_features_to_compute_at_molt
# FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()
# # BUILDING THE PLOTTING STRUCTURE
# # def build_conditions(config):
# #     conditions = []
# #     condition_id = 0
# #     for condition in config["conditions"]:
# #         condition = {
# #             key: [val] if not isinstance(val, list) else val
# #             for key, val in condition.items()
# #         }
# #         lengths = {len(val) for val in condition.values()}
# #         if len(lengths) > 2 or (len(lengths) == 2 and 1 not in lengths):
# #             raise ValueError(
# #                 "All lists in the condition must have the same length or be of length 1."
# #             )
# #         max_length = max(lengths)
# #         for i in range(max_length):
# #             condition_dict = {
# #                 key: val[0] if len(val) == 1 else val[i]
# #                 for key, val in condition.items()
# #             }
# #             condition_dict["condition_id"] = condition_id
# #             conditions.append(condition_dict)
# #             condition_id += 1
# #     return conditions
# # def add_conditions_to_filemap(experiment_filemap, conditions, config):
# #     for condition in conditions:
# #         if "point_range" in condition.keys():
# #             point_range = condition["point_range"]
# #             # check if point range is a list of lists
# #             if isinstance(point_range[0], list):
# #                 for pr in point_range:
# #                     # Get all the rows that are in the point range
# #                     condition_rows = experiment_filemap[
# #                         experiment_filemap["Point"].between(pr[0], pr[1])
# #                     ]
# #                     # Remove the point range from the condition
# #                     conditions_to_add = {
# #                         key: val
# #                         for key, val in condition.items()
# #                         if key != "point_range"
# #                     }
# #                     for key, val in conditions_to_add.items():
# #                         # Directly fill the rows with the value for the new or existing column
# #                         experiment_filemap.loc[condition_rows.index, key] = val
# #             else:
# #                 # Get all the rows that are in the point range
# #                 condition_rows = experiment_filemap[
# #                     experiment_filemap["Point"].between(point_range[0], point_range[1])
# #                 ]
# #                 # Remove the point range from the condition
# #                 conditions_to_add = {
# #                     key: val for key, val in condition.items() if key != "point_range"
# #                 }
# #                 for key, val in conditions_to_add.items():
# #                     # Directly fill the rows with the value for the new or existing column
# #                     experiment_filemap.loc[condition_rows.index, key] = val
# #         elif "pad" in condition.keys():
# #             pad = condition["pad"]
# #             # Get all the rows that are in the pad
# #             condition_rows = experiment_filemap[experiment_filemap["Pad"] == pad]
# #             # Remove the pad from the condition
# #             conditions_to_add = {
# #                 key: val for key, val in condition.items() if key != "pad"
# #             }
# #             for key, val in conditions_to_add.items():
# #                 # Directly fill the rows with the value for the new or existing column
# #                 experiment_filemap.loc[condition_rows.index, key] = val
# #         else:
# #             print(
# #                 "Condition does not contain 'point_range' or 'pad' key, impossible to add condition to filemap, skipping."
# #             )
# #     return experiment_filemap
# # def get_ecdysis_and_durations(filemap):
# #     all_ecdysis_time_step = []
# #     all_ecdysis_index = []
# #     all_durations_time_step = []
# #     all_ecdysis_experiment_time = []
# #     all_durations_experiment_time = []
# #     for point in filemap["Point"].unique():
# #         point_df = filemap[filemap["Point"] == point]
# #         point_time = point_df["Time"].values
# #         point_ecdysis = point_df[["HatchTime", "M1", "M2", "M3", "M4"]].iloc[0]
# #         point_ecdysis_index = []
# #         for ecdysis in point_ecdysis:
# #             matches = np.where(point_time == ecdysis)[0]
# #             if len(matches) == 0:
# #                 point_ecdysis_index.append(np.nan)
# #             else:
# #                 point_ecdysis_index.append(float(matches[0]))
# #         larval_stage_durations = list(
# #             compute_larval_stage_duration(point_ecdysis).values()
# #         )
# #         point_ecdysis = point_ecdysis.to_numpy()
# #         all_ecdysis_time_step.append(point_ecdysis)
# #         all_ecdysis_index.append(point_ecdysis_index)
# #         all_durations_time_step.append(larval_stage_durations)
# #         ecdysis_experiment_time = []
# #         for ecdys in point_ecdysis:
# #             if np.isnan(ecdys):
# #                 ecdysis_experiment_time.append(np.nan)
# #             else:
# #                 # if ecdys is not in the time column, set it to nan
# #                 if ecdys not in point_df["Time"].values:
# #                     ecdys_experiment_time = np.nan
# #                 else:
# #                     ecdys_experiment_time = point_df[point_df["Time"] == ecdys][
# #                         "ExperimentTime"
# #                     ].iloc[0]
# #                 ecdysis_experiment_time.append(ecdys_experiment_time)
# #         all_ecdysis_experiment_time.append(ecdysis_experiment_time)
# #         durations_experiment_time = []
# #         for i in range(len(ecdysis_experiment_time) - 1):
# #             start = ecdysis_experiment_time[i]
# #             end = ecdysis_experiment_time[i + 1]
# #             duration_experiment_time = end - start
# #             durations_experiment_time.append(duration_experiment_time)
# #         all_durations_experiment_time.append(durations_experiment_time)
# #     return (
# #         np.array(all_ecdysis_index),
# #         np.array(all_ecdysis_time_step),
# #         np.array(all_durations_time_step),
# #         np.array(all_ecdysis_experiment_time),
# #         np.array(all_durations_experiment_time),
# #     )
# # def separate_column_by_point(filemap, column):
# #     max_number_of_values = np.max(
# #         [
# #             len(filemap[filemap["Point"] == point][column].values)
# #             for point in filemap["Point"].unique()
# #         ]
# #     )
# #     all_values = []
# #     for i, point in enumerate(filemap["Point"].unique()):
# #         point_df = filemap[filemap["Point"] == point]
# #         values_of_point = point_df[column].values
# #         if isinstance(values_of_point[0], str):
# #             dtype = str
# #             pad_value = "error"
# #         else:
# #             dtype = float
# #             pad_value = np.nan
# #         values_of_point = np.array(values_of_point, dtype=dtype)
# #         values_of_point = np.pad(
# #             values_of_point,
# #             (0, max_number_of_values - len(values_of_point)),
# #             mode="constant",
# #             constant_values=pad_value,
# #         )
# #         all_values.append(values_of_point)
# #     return np.array(all_values)
# # def remove_ignored_molts(filemap):
# #     df = filemap.copy()
# #     molt_columns = ["HatchTime", "M1", "M2", "M3", "M4"]
# #     for point in df["Point"].unique():
# #         point_mask = df["Point"] == point
# #         point_df = df[point_mask]
# #         if point_df.empty:
# #             continue
# #         # Get molt times for this point
# #         molt_times = point_df[molt_columns].iloc[0]
# #         # Check each molt time
# #         for col, molt_time in molt_times.items():
# #             if pd.isna(molt_time):
# #                 continue
# #             # Find if this molt should be ignored
# #             try:
# #                 if point_df[point_df["Time"] == molt_time]["Ignore"].iloc[0]:
# #                     # Use .loc to avoid chained indexing warning
# #                     df.loc[point_mask, col] = np.nan
# #             except IndexError:
# #                 print(f"No row found for time {molt_time} in point {point}")
# #                 df.loc[point_mask, col] = np.nan
# #     return df
# # def build_plotting_struct(
# #     experiment_dir,
# #     filemap_path,
# #     config_path,
# #     organ_channels={"body": 2, "pharynx": 1},
# #     recompute_values_at_molt=False,
# # ):
# #     experiment_filemap = pd.read_csv(filemap_path)
# #     with open(config_path) as file:
# #         config = yaml.safe_load(file)
# #         file.close()
# #     conditions = build_conditions(config)
# #     conditions_keys = list(conditions[0].keys())
# #     # remove 'point_range' and 'pad' from the conditions keys if they are present
# #     if "point_range" in conditions_keys:
# #         conditions_keys.remove("point_range")
# #     if "pad" in conditions_keys:
# #         conditions_keys.remove("pad")
# #     experiment_filemap = add_conditions_to_filemap(
# #         experiment_filemap, conditions, config
# #     )
# #     experiment_filemap.columns
# #     # if ExperimentTime is not present in the filemap, add it
# #     if "ExperimentTime" not in experiment_filemap.columns:
# #         experiment_filemap["ExperimentTime"] = np.nan
# #     # remove rows where condition_id is NaN
# #     experiment_filemap = experiment_filemap[
# #         ~experiment_filemap["condition_id"].isnull()
# #     ]
# #     # set molts that should be ignored to NaN
# #     if "Ignore" in experiment_filemap.columns:
# #         experiment_filemap = remove_ignored_molts(experiment_filemap)
# #     # remove rows where Ignore is True
# #     if "Ignore" in experiment_filemap.columns:
# #         experiment_filemap = experiment_filemap[~experiment_filemap["Ignore"]]
# #     conditions_struct = []
# #     for condition_id in experiment_filemap["condition_id"].unique():
# #         condition_df = experiment_filemap[
# #             experiment_filemap["condition_id"] == condition_id
# #         ]
# #         condition_dict = {}
# #         for key in conditions_keys:
# #             condition_dict[key] = condition_df[key].iloc[0]
# #         (
# #             ecdysis_index,
# #             ecdysis_time_step,
# #             larval_stage_durations_time_step,
# #             ecdysis_experiment_time,
# #             larval_stage_durations_experiment_time,
# #         ) = get_ecdysis_and_durations(condition_df)
# #         condition_dict["condition_id"] = int(condition_dict["condition_id"])
# #         condition_dict["ecdysis_index"] = ecdysis_index
# #         condition_dict["ecdysis_time_step"] = ecdysis_time_step
# #         condition_dict[
# #             "larval_stage_durations_time_step"
# #         ] = larval_stage_durations_time_step
# #         condition_dict["ecdysis_experiment_time"] = ecdysis_experiment_time
# #         condition_dict[
# #             "larval_stage_durations_experiment_time"
# #         ] = larval_stage_durations_experiment_time
# #         condition_dict["larval_stage_durations_experiment_time_hours"] = (
# #             larval_stage_durations_experiment_time / 3600
# #         )
# #         condition_dict["experiment"] = np.array(
# #             [experiment_dir] * condition_df["Point"].nunique()
# #         )[:, np.newaxis]
# #         condition_dict["filemap_path"] = np.array(
# #             [filemap_path] * condition_df["Point"].nunique()
# #         )[:, np.newaxis]
# #         condition_dict["point"] = np.unique(condition_df["Point"].values)[:, np.newaxis]
# #         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
# #         worm_type_column = [col for col in condition_df.columns if "worm_type" in col][
# #             0
# #         ]
# #         worm_types = separate_column_by_point(condition_df, worm_type_column)
# #         condition_dict["time"] = separate_column_by_point(condition_df, "Time").astype(
# #             float
# #         )
# #         condition_dict["experiment_time"] = separate_column_by_point(
# #             condition_df, "ExperimentTime"
# #         ).astype(float)
# #         for organ in organ_channels.keys():
# #             organ_channel = organ_channels[organ]
# #             organ_channel = f"ch{organ_channel}"
# #             organ_columns = [
# #                 col for col in condition_df.columns if col.startswith(organ_channel)
# #             ]
# #             # remove any column with _at_ in it
# #             organ_columns = [col for col in organ_columns if "_at_" not in col]
# #             # get the columns that contain the interesting features
# #             organ_feature_columns = []
# #             for feature in FEATURES_TO_COMPUTE_AT_MOLT:
# #                 organ_feature_columns.extend(
# #                     [col for col in organ_columns if feature in col]
# #                 )
# #             renamed_organ_feature_columns = [
# #                 col.replace(organ_channel, organ) for col in organ_columns
# #             ]
# #             for organ_column, renamed_organ_column in zip(
# #                 organ_columns, renamed_organ_feature_columns
# #             ):
# #                 condition_dict[renamed_organ_column] = separate_column_by_point(
# #                     condition_df, organ_column
# #                 )
# #             # remove any column with worm_type in it
# #             renamed_organ_feature_columns = [
# #                 col for col in renamed_organ_feature_columns if "worm_type" not in col
# #             ]
# #             # compute the features of the organ at each molt
# #             for column in renamed_organ_feature_columns:
# #                 column_at_molt = f"{column}_at_ecdysis"
# #                 if recompute_values_at_molt or (
# #                     column_at_molt not in condition_df.columns
# #                 ):
# #                     condition_dict[column_at_molt] = np.stack(
# #                         [
# #                             compute_series_at_time_classified(
# #                                 condition_dict[column][i],
# #                                 worm_types[i],
# #                                 ecdysis_time_step[i],
# #                                 series_time=condition_dict["time"][i],
# #                             )
# #                             for i in range(len(ecdysis_time_step))
# #                         ]
# #                     )
# #                 else:
# #                     condition_dict[column_at_molt] = separate_column_by_point(
# #                         condition_df, column_at_molt
# #                     )
# #         conditions_struct.append(condition_dict)
# #     conditions_info = [
# #         {key: condition[key] for key in conditions_keys}
# #         for condition in conditions_struct
# #     ]
# #     # sort the conditions and conditions_info by condition_id
# #     conditions_struct = sorted(conditions_struct, key=lambda x: x["condition_id"])
# #     conditions_info = sorted(conditions_info, key=lambda x: x["condition_id"])
# #     return conditions_struct, conditions_info
# # def remove_unwanted_info(conditions_info):
# #     for condition in conditions_info:
# #         if "description" in condition.keys():
# #             condition.pop("description")
# #         if "condition_id" in condition.keys():
# #             condition.pop("condition_id")
# #     return conditions_info
# # def combine_experiments(
# #     filemap_paths,
# #     config_paths,
# #     experiment_dirs=None,
# #     organ_channels=[{"body": 2, "pharynx": 1}],
# #     recompute_values_at_molt=False,
# # ):
# #     all_conditions_struct = []
# #     condition_info_merge_list = []
# #     conditions_info_keys = set()
# #     condition_id_counter = 0
# #     if isinstance(organ_channels, dict):
# #         organ_channels = [organ_channels]
# #     if len(organ_channels) == 1:
# #         organ_channels = organ_channels * len(filemap_paths)
# #     elif len(organ_channels) != len(filemap_paths):
# #         raise ValueError(
# #             "Number of organ channels must be equal to the number of experiments."
# #         )
# #     # Process each experiment
# #     for i, (filemap_path, config_path, organ_channel) in enumerate(
# #         zip(filemap_paths, config_paths, organ_channels)
# #     ):
# #         experiment_dir = (
# #             experiment_dirs[i] if experiment_dirs else os.path.dirname(filemap_path)
# #         )
# #         conditions_struct, conditions_info = build_plotting_struct(
# #             experiment_dir,
# #             filemap_path,
# #             config_path,
# #             organ_channels=organ_channel,
# #             recompute_values_at_molt=recompute_values_at_molt,
# #         )
# #         # Process conditions for this experiment
# #         for condition in conditions_struct:
# #             condition["condition_id"] = condition_id_counter
# #             condition_id_counter += 1
# #             all_conditions_struct.append(condition)
# #         # Process condition info
# #         experiment_conditions_info = remove_unwanted_info(conditions_info)
# #         condition_info_merge_list.extend(experiment_conditions_info)
# #         conditions_info_keys.update(
# #             *[condition.keys() for condition in experiment_conditions_info]
# #         )
# #     # Merge conditions based on their info
# #     condition_dict = defaultdict(list)
# #     for i, condition_info in enumerate(condition_info_merge_list):
# #         key = frozenset(condition_info.items())
# #         condition_dict[key].append(i)
# #     merged_conditions_struct = []
# #     for indices in condition_dict.values():
# #         base_condition = all_conditions_struct[indices[0]]
# #         for idx in indices[1:]:
# #             for key, value in all_conditions_struct[idx].items():
# #                 if key not in conditions_info_keys:
# #                     if isinstance(value, np.ndarray):
# #                         if value.shape[1] > base_condition[key].shape[1]:
# #                             base_condition[key] = np.pad(
# #                                 base_condition[key],
# #                                 (
# #                                     (0, 0),
# #                                     (0, value.shape[1] - base_condition[key].shape[1]),
# #                                 ),
# #                                 mode="constant",
# #                                 constant_values=np.nan,
# #                             )
# #                         elif value.shape[1] < base_condition[key].shape[1]:
# #                             value = np.pad(
# #                                 value,
# #                                 (
# #                                     (0, 0),
# #                                     (0, base_condition[key].shape[1] - value.shape[1]),
# #                                 ),
# #                                 mode="constant",
# #                                 constant_values=np.nan,
# #                             )
# #                     try:
# #                         base_condition[key] = np.concatenate(
# #                             (base_condition[key], value), axis=0
# #                         )
# #                     except ValueError as e:
# #                         print(f"Could not concatenate {key}: {e}")
# #         merged_conditions_struct.append(base_condition)
# #     # # Sort and reassign condition IDs
# #     # merged_conditions_struct.sort(key=lambda x: x['condition_id'])
# #     for i, condition in enumerate(merged_conditions_struct):
# #         condition["condition_id"] = i
# #     return merged_conditions_struct
# # PLOTTING FUNCTIONS
# # def save_figure(fig, name, directory, format="svg", dpi=300, transparent=False):
# #     """
# #     Save the current matplotlib figure to the specified directory with the given name.
# #     Parameters:
# #     fig (matplotlib.figure.Figure) : Figure to save
# #     name (str) : Name of the file (without extension)
# #     directory (str) : Directory to save the file in
# #     format (str) : File format to save the figure in
# #     dpi (int) : Resolution of the saved figure
# #     transparent (bool) : Whether to save the figure with a transparent background
# #     Returns:
# #     str : Full path to the saved file
# #     """
# #     # Create directory if it doesn't exist
# #     os.makedirs(directory, exist_ok=True)
# #     # Construct full file path
# #     filename = f"{name}.{format}"
# #     filepath = os.path.join(directory, filename)
# #     # Save the figure
# #     fig.savefig(
# #         filepath, format=format, dpi=dpi, bbox_inches="tight", transparent=transparent
# #     )
# # def build_legend(single_condition_dict, legend):
# #     if legend is None:
# #         return f'Condition {int(single_condition_dict["condition_id"])}'
# #     else:
# #         legend_string = ""
# #         for i, (key, value) in enumerate(legend.items()):
# #             if value:
# #                 legend_string += f"{single_condition_dict[key]} {value}"
# #             else:
# #                 legend_string += f"{single_condition_dict[key]}"
# #             if i < len(legend) - 1:
# #                 legend_string += ", "
# #         return legend_string
# # def set_scale(ax, log_scale):
# #     if isinstance(log_scale, bool):
# #         ax.set_yscale("log" if log_scale else "linear")
# #     elif isinstance(log_scale, tuple):
# #         ax.set_yscale("log" if log_scale[1] else "linear")
# #         ax.set_xscale("log" if log_scale[0] else "linear")
# #     elif isinstance(log_scale, list):
# #         ax.set_yscale("log" if log_scale[1] else "linear")
# #         ax.set_xscale("log" if log_scale[0] else "linear")
# def plot_aggregated_series(
#     conditions_struct,
#     series_column,
#     conditions_to_plot,
#     experiment_time=True,
#     aggregation="mean",
#     n_points=100,
#     time_step=10,
#     log_scale=True,
#     colors=None,
#     legend=None,
#     y_axis_label=None,
# ):
#     if colors is None:
#         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
#     else:
#         color_palette = colors
#     def plot_single_series(column: str):
#         for i, condition_id in enumerate(conditions_to_plot):
#             condition_dict = conditions_struct[condition_id]
#             if experiment_time:
#                 time = condition_dict["experiment_time"]
#                 larval_stage_durations = condition_dict[
#                     "larval_stage_durations_experiment_time"
#                 ]
#             else:
#                 time = condition_dict["time"]
#                 larval_stage_durations = condition_dict[
#                     "larval_stage_durations_time_step"
#                 ]
#             # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
#             worm_type_key = [
#                 key for key in condition_dict.keys() if "worm_type" in key
#             ][0]
#             rescaled_time, aggregated_series, _, ste_series = rescale_and_aggregate(
#                 condition_dict[column],
#                 time,
#                 condition_dict["ecdysis_index"],
#                 larval_stage_durations,
#                 condition_dict[worm_type_key],
#                 aggregation=aggregation,
#             )
#             ci_lower = aggregated_series - 1.96 * ste_series
#             ci_upper = aggregated_series + 1.96 * ste_series
#             if experiment_time:
#                 rescaled_time = rescaled_time / 3600
#             else:
#                 rescaled_time = rescaled_time * time_step / 60
#             label = build_legend(condition_dict, legend)
#             plt.plot(
#                 rescaled_time, aggregated_series, color=color_palette[i], label=label
#             )
#             plt.fill_between(
#                 rescaled_time, ci_lower, ci_upper, color=color_palette[i], alpha=0.2
#             )
#     if isinstance(series_column, list):
#         for column in series_column:
#             plot_single_series(column)
#     else:
#         plot_single_series(series_column)
#     # remove duplicate labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())
#     plt.xlabel("Time (h)")
#     plt.yscale("log" if log_scale else "linear")
#     if y_axis_label is not None:
#         plt.ylabel(y_axis_label)
#     else:
#         plt.ylabel(series_column)
#     fig = plt.gcf()
#     plt.show()
#     return fig
# def plot_correlation(
#     conditions_struct,
#     column_one,
#     column_two,
#     conditions_to_plot,
#     log_scale=True,
#     colors=None,
#     legend=None,
#     x_axis_label=None,
#     y_axis_label=None,
# ):
#     if colors is None:
#         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
#     else:
#         color_palette = colors
#     for i, condition_id in enumerate(conditions_to_plot):
#         condition_dict = conditions_struct[condition_id]
#         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
#         worm_type_key = [key for key in condition_dict.keys() if "worm_type" in key][0]
#         _, aggregated_series_one, _, _ = rescale_and_aggregate(
#             condition_dict[column_one],
#             condition_dict["time"],
#             condition_dict["ecdysis_index"],
#             condition_dict["larval_stage_durations_time_step"],
#             condition_dict[worm_type_key],
#             aggregation="mean",
#         )
#         _, aggregated_series_two, _, _ = rescale_and_aggregate(
#             condition_dict[column_two],
#             condition_dict["time"],
#             condition_dict["ecdysis_index"],
#             condition_dict["larval_stage_durations_time_step"],
#             condition_dict[worm_type_key],
#             aggregation="mean",
#         )
#         # sort the values
#         order = np.argsort(aggregated_series_one)
#         aggregated_series_one = aggregated_series_one[order]
#         aggregated_series_two = aggregated_series_two[order]
#         label = build_legend(condition_dict, legend)
#         plt.plot(
#             aggregated_series_one,
#             aggregated_series_two,
#             color=color_palette[i],
#             label=label,
#         )
#     if x_axis_label is not None:
#         plt.xlabel(x_axis_label)
#     else:
#         plt.xlabel(column_one)
#     if y_axis_label is not None:
#         plt.ylabel(y_axis_label)
#     else:
#         plt.ylabel(column_two)
#     set_scale(plt.gca(), log_scale)
#     plt.legend()
#     fig = plt.gcf()
#     plt.show()
#     return fig
# # def plot_correlation_at_ecdysis(
# #     conditions_struct,
# #     column_one,
# #     column_two,
# #     conditions_to_plot,
# #     remove_hatch=True,
# #     log_scale=True,
# #     colors=None,
# #     legend=None,
# #     x_axis_label=None,
# #     y_axis_label=None,
# # ):
# #     if colors is None:
# #         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
# #     else:
# #         color_palette = colors
# #     for i, condition_id in enumerate(conditions_to_plot):
# #         condition_dict = conditions_struct[condition_id]
# #         if remove_hatch:
# #             column_one_values = condition_dict[column_one][:, 1:]
# #             column_two_values = condition_dict[column_two][:, 1:]
# #         else:
# #             column_one_values = condition_dict[column_one]
# #             column_two_values = condition_dict[column_two]
# #         x = np.nanmean(column_one_values, axis=0)
# #         x_std = np.nanstd(column_one_values, axis=0)
# #         # x_ste = x_std / np.sqrt(np.sum(np.isfinite(column_one_values), axis=0))
# #         y = np.nanmean(column_two_values, axis=0)
# #         y_std = np.nanstd(column_two_values, axis=0)
# #         # y_ste = y_std / np.sqrt(np.sum(np.isfinite(column_two_values), axis=0))
# #         label = build_legend(condition_dict, legend)
# #         plt.errorbar(
# #             x,
# #             y,
# #             xerr=x_std,
# #             yerr=y_std,
# #             fmt="o",
# #             color=color_palette[i],
# #             label=label,
# #             capsize=3,
# #         )
# #         plt.plot(x, y, color=color_palette[i])
# #     if x_axis_label is not None:
# #         plt.xlabel(x_axis_label)
# #     else:
# #         plt.xlabel(column_one)
# #     if y_axis_label is not None:
# #         plt.ylabel(y_axis_label)
# #     else:
# #         plt.ylabel(column_two)
# #     set_scale(plt.gca(), log_scale)
# #     plt.legend()
# #     fig = plt.gcf()
# #     plt.show()
# #     return fig
# # def boxplot_at_molt(
# #     conditions_struct,
# #     column,
# #     conditions_to_plot,
# #     remove_hatch=False,
# #     log_scale: bool = True,
# #     figsize: tuple = None,
# #     colors=None,
# #     plot_significance: bool = False,
# #     significance_pairs=None,
# #     legend=None,
# #     y_axis_label=None,
# #     titles=None,
# #     share_y_axis: bool = False,
# # ):
# #     if colors is None:
# #         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
# #     else:
# #         color_palette = colors
# #     # Prepare data
# #     data_list = []
# #     for condition_id in conditions_to_plot:
# #         condition_dict = conditions_struct[condition_id]
# #         data = condition_dict[column]
# #         range_start = 1 if remove_hatch else 0
# #         for j in range(range_start, data.shape[1]):
# #             for value in data[:, j]:
# #                 data_list.append(
# #                     {
# #                         "Condition": condition_id,
# #                         "Molt": j,
# #                         column: np.log(value) if log_scale else value,
# #                     }
# #                 )
# #     df = pd.DataFrame(data_list)
# #     # Determine figure size
# #     if figsize is None:
# #         figsize = (6 * df["Molt"].nunique(), 10)
# #     if titles is not None and len(titles) != df["Molt"].nunique():
# #         print("Number of titles does not match the number of ecdysis events.")
# #         titles = None
# #     # Create figure with extra space on the right for legend
# #     fig, ax = plt.subplots(
# #         1,
# #         df["Molt"].nunique(),
# #         figsize=(figsize[0] + 3, figsize[1]),
# #         sharey=share_y_axis,
# #     )
# #     # Create a dummy plot to get proper legend handles
# #     dummy_ax = fig.add_axes([0, 0, 0, 0])
# #     for i, condition in enumerate(conditions_to_plot):
# #         dummy_ax.boxplot(
# #             [],
# #             [],
# #             patch_artist=True,
# #             label=build_legend(conditions_struct[condition], legend),
# #         )
# #         for j, patch in enumerate(dummy_ax.patches):
# #             patch.set_facecolor(color_palette[j])
# #     dummy_ax.set_visible(False)
# #     for i in range(df["Molt"].nunique()):
# #         if share_y_axis:
# #             if i > 0:
# #                 ax[i].tick_params(axis="y", which="both", left=False, labelleft=False)
# #         boxplot = sns.boxplot(
# #             data=df[df["Molt"] == i],
# #             x="Condition",
# #             y=column,
# #             order=conditions_to_plot,
# #             hue="Condition",
# #             palette=color_palette,
# #             showfliers=False,
# #             ax=ax[i],
# #             dodge=False,
# #             linewidth=2,
# #             legend=False,
# #         )
# #         sns.stripplot(
# #             data=df[df["Molt"] == i],
# #             x="Condition",
# #             order=conditions_to_plot,
# #             y=column,
# #             ax=ax[i],
# #             alpha=0.5,
# #             color="black",
# #             dodge=True,
# #         )
# #         ax[i].set_xlabel("")
# #         # Hide y-axis labels and ticks for all subplots except the first one
# #         if i > 0:
# #             ax[i].set_ylabel("")
# #         if titles is not None:
# #             ax[i].set_title(titles[i])
# #         # remove ticks
# #         ax[i].tick_params(
# #             axis="x", which="both", bottom=False, top=False, labelbottom=False
# #         )
# #         if plot_significance:
# #             if significance_pairs is None:
# #                 pairs = list(combinations(df["Condition"].unique(), 2))
# #             else:
# #                 pairs = significance_pairs
# #             annotator = Annotator(
# #                 ax=boxplot,
# #                 pairs=pairs,
# #                 data=df[df["Molt"] == i],
# #                 x="Condition",
# #                 order=conditions_to_plot,
# #                 y=column,
# #             )
# #             annotator.configure(
# #                 test="Mann-Whitney", text_format="star", loc="inside", verbose=False
# #             )
# #             annotator.apply_and_annotate()
# #         y_min, y_max = ax[i].get_ylim()
# #     # Set y label for the first plot
# #     if y_axis_label is not None:
# #         ax[0].set_ylabel(y_axis_label)
# #     else:
# #         ax[0].set_ylabel(column)
# #     # Add legend to the right of the subplots
# #     legend_labels = [
# #         build_legend(conditions_struct[condition_id], legend)
# #         for condition_id in conditions_to_plot
# #     ]
# #     legend_handles = dummy_ax.get_legend_handles_labels()[0]
# #     # Place legend to the right of the subplots
# #     fig.legend(
# #         legend_handles,
# #         legend_labels,
# #         bbox_to_anchor=(0.9, 0.5),
# #         loc="center left",
# #         title=None,
# #         frameon=True,
# #     )
# #     if share_y_axis:
# #         global_min = y_min
# #         global_max = y_max
# #         range_padding = (global_max - global_min) * 0.05  # 5% padding
# #         global_min = global_min - range_padding
# #         global_max = global_max + range_padding
# #         for i in range(df["Molt"].nunique()):
# #             ax[i].set_ylim(global_min, global_max)
# #     # Make subplots closer together while leaving space for legend
# #     plt.tight_layout(rect=[0, 0, 0.9, 1])
# #     fig = plt.gcf()
# #     plt.show()
# #     return fig
# # def boxplot_larval_stage(
# #     conditions_struct,
# #     column,
# #     conditions_to_plot,
# #     aggregation: str = "mean",
# #     n_points: int = 100,
# #     fraction: float = 0.8,
# #     log_scale: bool = True,
# #     figsize: tuple = None,
# #     colors=None,
# #     plot_significance: bool = False,
# #     significance_pairs=None,
# #     significance_position="inside",
# #     legend=None,
# #     y_axis_label=None,
# #     titles=None,
# #     share_y_axis: bool = False,
# # ):
# #     new_column = column + "_rescaled"
# #     struct = rescale_without_flattening(
# #         conditions_struct, column, new_column, aggregation, n_points
# #     )
# #     if colors is None:
# #         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
# #     else:
# #         color_palette = colors
# #     # Prepare data
# #     data_list = []
# #     for condition_id in conditions_to_plot:
# #         condition_dict = struct[condition_id]
# #         data = condition_dict[new_column]
# #         for i in range(data.shape[1]):
# #             data_of_stage = data[:, i]
# #             data_of_stage = data_of_stage[:, 0 : int(fraction * data_of_stage.shape[1])]
# #             data_of_stage = np.log(data_of_stage) if log_scale else data_of_stage
# #             if aggregation == "mean":
# #                 aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
# #             elif aggregation == "median":
# #                 aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)
# #             for j in range(aggregated_data_of_stage.shape[0]):
# #                 data_list.append(
# #                     {
# #                         "Condition": condition_id,
# #                         "LarvalStage": i,
# #                         column: aggregated_data_of_stage[j],
# #                     }
# #                 )
# #     df = pd.DataFrame(data_list)
# #     # Determine figure size
# #     if figsize is None:
# #         figsize = (6 * df["LarvalStage"].nunique(), 10)
# #     if titles is not None and len(titles) != df["LarvalStage"].nunique():
# #         print("Number of titles does not match the number of ecdysis events.")
# #         titles = None
# #     # Create figure with extra space on the right for legend
# #     fig, ax = plt.subplots(
# #         1,
# #         df["LarvalStage"].nunique(),
# #         figsize=(figsize[0] + 3, figsize[1]),
# #         sharey=share_y_axis,
# #     )
# #     # Create a dummy plot to get proper legend handles
# #     dummy_ax = fig.add_axes([0, 0, 0, 0])
# #     for i, condition in enumerate(conditions_to_plot):
# #         dummy_ax.boxplot(
# #             [], [], patch_artist=True, label=build_legend(struct[condition], legend)
# #         )
# #         for j, patch in enumerate(dummy_ax.patches):
# #             patch.set_facecolor(color_palette[j])
# #     dummy_ax.set_visible(False)
# #     for i in range(df["LarvalStage"].nunique()):
# #         if share_y_axis:
# #             if i > 0:
# #                 ax[i].tick_params(axis="y", which="both", left=False, labelleft=False)
# #         boxplot = sns.boxplot(
# #             data=df[df["LarvalStage"] == i],
# #             x="Condition",
# #             y=column,
# #             hue="Condition",
# #             order=conditions_to_plot,
# #             palette=color_palette,
# #             showfliers=False,
# #             ax=ax[i],
# #             dodge=False,
# #             linewidth=2,
# #             legend=False,
# #         )
# #         sns.stripplot(
# #             data=df[df["LarvalStage"] == i],
# #             x="Condition",
# #             order=conditions_to_plot,
# #             y=column,
# #             ax=ax[i],
# #             alpha=0.5,
# #             color="black",
# #             dodge=True,
# #         )
# #         ax[i].set_xlabel("")
# #         # Hide y-axis labels and ticks for all subplots except the first one
# #         if i > 0:
# #             ax[i].set_ylabel("")
# #         if titles is not None:
# #             ax[i].set_title(titles[i])
# #         # remove ticks
# #         ax[i].tick_params(
# #             axis="x", which="both", bottom=False, top=False, labelbottom=False
# #         )
# #         if plot_significance:
# #             if significance_pairs is None:
# #                 pairs = list(combinations(df["Condition"].unique(), 2))
# #             else:
# #                 pairs = significance_pairs
# #             annotator = Annotator(
# #                 ax=boxplot,
# #                 pairs=pairs,
# #                 data=df[df["LarvalStage"] == i],
# #                 x="Condition",
# #                 order=conditions_to_plot,
# #                 y=column,
# #             )
# #             annotator.configure(
# #                 test="Mann-Whitney",
# #                 text_format="star",
# #                 loc=significance_position,
# #                 verbose=False,
# #             )
# #             annotator.apply_and_annotate()
# #         y_min, y_max = ax[i].get_ylim()
# #     # Set y label for the first plot
# #     if y_axis_label is not None:
# #         ax[0].set_ylabel(y_axis_label)
# #     else:
# #         ax[0].set_ylabel(column)
# #     # Add legend to the right of the subplots
# #     legend_labels = [
# #         build_legend(struct[condition_id], legend)
# #         for condition_id in conditions_to_plot
# #     ]
# #     legend_handles = dummy_ax.get_legend_handles_labels()[0]
# #     # Place legend to the right of the subplots
# #     fig.legend(
# #         legend_handles,
# #         legend_labels,
# #         bbox_to_anchor=(0.9, 0.5),
# #         loc="center left",
# #         title=None,
# #         frameon=True,
# #     )
# #     if share_y_axis:
# #         global_min = y_min
# #         global_max = y_max
# #         range_padding = (global_max - global_min) * 0.05  # 5% padding
# #         global_min = global_min - range_padding
# #         global_max = global_max + range_padding
# #         for i in range(df["LarvalStage"].nunique()):
# #             ax[i].set_ylim(global_min, global_max)
# #     # Make subplots closer together while leaving space for legend
# #     plt.tight_layout(rect=[0, 0, 0.9, 1])
# #     fig = plt.gcf()
# #     plt.show()
# #     return fig
# def plot_developmental_success(
#     conditions_struct,
#     conditions_to_plot,
#     colors=None,
#     figsize=None,
#     legend=None,
# ):
#     if colors is None:
#         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
#     else:
#         color_palette = colors
#     if figsize is None:
#         figsize = (5 * 4, 6)
#     fig, ax = plt.subplots(1, 4, figsize=figsize)
#     for i, condition_id in enumerate(conditions_to_plot):
#         condition_dict = conditions_struct[condition_id]
#         ecdysis = condition_dict["ecdysis_time_step"]
#         m1_completion = ~np.isnan(ecdysis[:, 1])
#         m2_completion = np.logical_and(
#             ~np.isnan(ecdysis[:, 2]), ~np.isnan(ecdysis[:, 1])
#         )
#         m3_completion = np.logical_and(
#             ~np.isnan(ecdysis[:, 3]), ~np.isnan(ecdysis[:, 2])
#         )
#         m4_completion = np.logical_and(
#             ~np.isnan(ecdysis[:, 4]), ~np.isnan(ecdysis[:, 3])
#         )
#         m1_completion_rate = np.sum(m1_completion) / len(ecdysis)
#         m2_completion_rate = np.sum(m2_completion) / len(ecdysis)
#         m3_completion_rate = np.sum(m3_completion) / len(ecdysis)
#         m4_completion_rate = np.sum(m4_completion) / len(ecdysis)
#         label = build_legend(condition_dict, legend)
#         ax[0].bar(
#             i,
#             m1_completion_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#         ax[1].bar(
#             i,
#             m2_completion_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#         ax[2].bar(
#             i,
#             m3_completion_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#         ax[3].bar(
#             i,
#             m4_completion_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#     ax[0].set_title("M1")
#     ax[1].set_title("M2")
#     ax[2].set_title("M3")
#     ax[3].set_title("M4")
#     ax[0].set_ylabel("Successful molts (%)")
#     # remove ticks
#     for i in range(4):
#         ax[i].tick_params(
#             axis="x", which="both", bottom=False, top=False, labelbottom=False
#         )
#     plt.legend()
#     fig = plt.gcf()
#     plt.show()
#     return fig
# def plot_arrests(
#     conditions_struct,
#     conditions_to_plot,
#     colors=None,
#     figsize=None,
#     legend=None,
# ):
#     if colors is None:
#         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
#     else:
#         color_palette = colors
#     if figsize is None:
#         figsize = (5 * 4, 6)
#     fig, ax = plt.subplots(1, 4, figsize=figsize)
#     for i, condition_id in enumerate(conditions_to_plot):
#         condition_dict = conditions_struct[condition_id]
#         ecdysis = condition_dict["ecdysis_time_step"]
#         # m1_arrest = np.isnan(ecdysis[:, 1])
#         # m2_arrest = np.logical_and(np.isnan(ecdysis[:, 2]), ~m1_arrest)
#         # m3_arrest = np.logical_and(np.isnan(ecdysis[:, 3]), ~m2_arrest)
#         # m4_arrest = np.logical_and(np.isnan(ecdysis[:, 4]), ~m3_arrest)
#         # l1_arrest_rate = np.sum(m1_arrest) / len(ecdysis)
#         # l2_arrest_rate = np.sum(m2_arrest) / np.sum(~m1_arrest)
#         # l3_arrest_rate = np.sum(m3_arrest) / np.sum(~m2_arrest)
#         # l4_arrest_rate = np.sum(m4_arrest) / np.sum(~m3_arrest)
#         m1_arrest = np.isnan(ecdysis[:, 1])  # arrested in L1
#         m2_arrest = np.logical_and(
#             np.isnan(ecdysis[:, 2]), ~np.isnan(ecdysis[:, 1])
#         )  # passed L1 but arrested in L2
#         m3_arrest = np.logical_and(
#             np.isnan(ecdysis[:, 3]), ~np.isnan(ecdysis[:, 2])
#         )  # passed L2 but arrested in L3
#         m4_arrest = np.logical_and(
#             np.isnan(ecdysis[:, 4]), ~np.isnan(ecdysis[:, 3])
#         )  # passed L3 but arrested in L4
#         # Calculate rates - each rate is: number arrested in stage / number that entered that stage
#         l1_arrest_rate = np.sum(m1_arrest) / len(ecdysis)  # all animals enter L1
#         l2_arrest_rate = np.sum(m2_arrest) / np.sum(
#             ~np.isnan(ecdysis[:, 1])
#         )  # only L1 completers enter L2
#         l3_arrest_rate = np.sum(m3_arrest) / np.sum(
#             ~np.isnan(ecdysis[:, 2])
#         )  # only L2 completers enter L3
#         l4_arrest_rate = np.sum(m4_arrest) / np.sum(
#             ~np.isnan(ecdysis[:, 3])
#         )  # only L3 completers enter L4
#         print(
#             np.sum(m1_arrest), np.sum(m2_arrest), np.sum(m3_arrest), np.sum(m4_arrest)
#         )
#         print(len(ecdysis), np.sum(~m1_arrest), np.sum(~m2_arrest), np.sum(~m3_arrest))
#         label = build_legend(condition_dict, legend)
#         ax[0].bar(
#             i,
#             l1_arrest_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#         ax[1].bar(
#             i,
#             l2_arrest_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#         ax[2].bar(
#             i,
#             l3_arrest_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#         ax[3].bar(
#             i,
#             l4_arrest_rate,
#             color=color_palette[i],
#             label=label,
#             edgecolor="black",
#             linewidth=2,
#         )
#     ax[0].set_title("L1")
#     ax[1].set_title("L2")
#     ax[2].set_title("L3")
#     ax[3].set_title("L4")
#     ax[0].set_ylabel("Arrest rate (%)")
#     # remove ticks
#     for i in range(4):
#         ax[i].tick_params(
#             axis="x", which="both", bottom=False, top=False, labelbottom=False
#         )
#     plt.legend()
#     fig = plt.gcf()
#     plt.show()
#     return fig
# def plot_growth_curves_individuals(
#     conditions_struct,
#     column,
#     conditions_to_plot,
#     log_scale=True,
#     color_palette="colorblind",
#     figsize=None,
#     legend=None,
#     y_axis_label=None,
#     smoothing_window=21,
# ):
#     color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
#     if figsize is None:
#         figsize = (len(conditions_to_plot) * 8, 10)
#     fig, ax = plt.subplots(1, len(conditions_to_plot), figsize=figsize)
#     for i, condition_id in enumerate(conditions_to_plot):
#         condition_dict = conditions_struct[condition_id]
#         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
#         worm_type_key = [key for key in condition_dict.keys() if "worm_type" in key][0]
#         for j in range(len(condition_dict[column])):
#             time = condition_dict["experiment_time"][j] / 3600
#             data = condition_dict[column][j]
#             worm_type = condition_dict[worm_type_key][j]
#             hatch = condition_dict["ecdysis_time_step"][j][0]
#             hatch_experiment_time = (
#                 condition_dict["ecdysis_experiment_time"][j][0] / 3600
#             )
#             if not np.isnan(hatch):
#                 hatch = int(hatch)
#                 time = time[hatch:]
#                 time = time - hatch_experiment_time
#                 data = data[hatch:]
#                 worm_type = worm_type[hatch:]
#                 filtered_data = correct_series_with_classification(data, worm_type)
#                 # smooth the data
#                 filtered_data = medfilt(filtered_data, smoothing_window)
#                 label = build_legend(condition_dict, legend)
#                 try:
#                     ax[i].plot(time, filtered_data)
#                     set_scale(ax[i], log_scale)
#                 except TypeError:
#                     ax.plot(time, filtered_data)
#                     set_scale(ax, log_scale)
#         try:
#             ax[i].title.set_text(label)
#         except TypeError:
#             ax.title.set_text(label)
#     # Set labels
#     if y_axis_label is not None:
#         try:
#             ax[0].set_ylabel(y_axis_label)
#             ax[0].set_xlabel("Time (h)")
#         except TypeError:
#             ax.set_ylabel(y_axis_label)
#             ax.set_xlabel("Time (h)")
#     else:
#         try:
#             ax[0].set_ylabel(column)
#             ax[0].set_xlabel("Time (h)")
#         except TypeError:
#             ax.set_ylabel(column)
#             ax.set_xlabel("Time (h)")
#     fig = plt.gcf()
#     plt.show()
#     return fig
# # def get_proportion_model(
# #     rescaled_series_one, rescaled_series_two, x_axis_label=None, y_axis_label=None
# # ):
# #     assert len(rescaled_series_one) == len(
# #         rescaled_series_two
# #     ), "The two series must have the same length."
# #     series_one = np.array(rescaled_series_one).flatten()
# #     series_two = np.array(rescaled_series_two).flatten()
# #     # remove elements that are nan in one of the two arrays
# #     correct_indices = ~np.isnan(series_one) & ~np.isnan(series_two)
# #     series_one = series_one[correct_indices]
# #     series_two = series_two[correct_indices]
# #     # log transform the data
# #     series_one = np.log(series_one)
# #     series_two = np.log(series_two)
# #     # for duplicate values, take the mean
# #     unique_series_one = np.unique(series_one)
# #     unique_series_two = np.array(
# #         [np.mean(series_two[series_one == value]) for value in unique_series_one]
# #     )
# #     series_one = unique_series_one
# #     series_two = unique_series_two
# #     plt.scatter(series_one, series_two)
# #     if x_axis_label is not None:
# #         plt.xlabel(x_axis_label)
# #     else:
# #         plt.xlabel("column one")
# #     if y_axis_label is not None:
# #         plt.ylabel(y_axis_label)
# #     else:
# #         plt.ylabel("column two")
# #     # lowess will return our "smoothed" data with a y value for at every x-value
# #     lowess = sm.nonparametric.lowess(series_two, series_one, frac=0.1)
# #     # unpack the lowess smoothed points to their values
# #     lowess_x = list(zip(*lowess))[0]
# #     lowess_y = list(zip(*lowess))[1]
# #     # interpolate the loess curve
# #     model = make_interp_spline(lowess_x, lowess_y, k=1)
# #     x = np.linspace(min(series_one), max(series_one), 500)
# #     y = model(x)
# #     plt.plot(x, y, color="red", linewidth=2)
# #     plt.show()
# #     return model
# # def get_deviation_from_model(rescaled_series_one, rescaled_series_two, model):
# #     # log transform the data
# #     expected_series_two = np.exp(model(np.log(rescaled_series_one)))
# #     # log_residuals = rescaled_series_two - expected_series_two
# #     # residuals = np.exp(log_residuals)
# #     percentage_deviation = (
# #         (rescaled_series_two - expected_series_two) / expected_series_two * 100
# #     )
# #     mean_series_one = np.nanmean(rescaled_series_one, axis=0)
# #     # mean_residuals = np.nanmean(residuals, axis=0)
# #     # std_residuals = np.nanstd(residuals, axis=0)
# #     # ste_residuals = std_residuals / np.sqrt(np.sum(np.isfinite(residuals), axis=0))
# #     mean_residuals = np.nanmean(percentage_deviation, axis=0)
# #     std_residuals = np.nanstd(percentage_deviation, axis=0)
# #     ste_residuals = std_residuals / np.sqrt(
# #         np.sum(np.isfinite(percentage_deviation), axis=0)
# #     )
# #     return mean_series_one, mean_residuals, std_residuals, ste_residuals
# # def plot_deviation_from_model(
# #     conditions_struct,
# #     column_one,
# #     column_two,
# #     control_condition_id,
# #     conditions_to_plot,
# #     colors=None,
# #     log_scale=(True, False),
# #     legend=None,
# #     x_axis_label=None,
# #     y_axis_label=None,
# # ):
# #     if colors is None:
# #         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
# #     else:
# #         color_palette = colors
# #     xlbl = column_one
# #     ylbl = column_two
# #     x_axis_label = x_axis_label if x_axis_label is not None else xlbl
# #     y_axis_label = (
# #         y_axis_label
# #         if y_axis_label is not None
# #         else f"deviation from modeled {column_two}"
# #     )
# #     control_condition = conditions_struct[control_condition_id]
# #     control_model = get_proportion_model(
# #         control_condition[column_one],
# #         control_condition[column_two],
# #         x_axis_label=xlbl,
# #         y_axis_label=ylbl,
# #     )
# #     for i, condition_id in enumerate(conditions_to_plot):
# #         condition = conditions_struct[condition_id]
# #         body_data, pharynx_data = condition[column_one], condition[column_two]
# #         x, residuals, std_residuals, ste_residuals = get_deviation_from_model(
# #             body_data,
# #             pharynx_data,
# #             control_model,
# #         )
# #         sorted_indices = np.argsort(x)
# #         x = x[sorted_indices]
# #         residuals = residuals[sorted_indices]
# #         ste_residuals = ste_residuals[sorted_indices]
# #         label = build_legend(condition, legend)
# #         plt.plot(x, residuals, label=label, color=color_palette[i])
# #         plt.fill_between(
# #             x,
# #             residuals - 1.96 * ste_residuals,
# #             residuals + 1.96 * ste_residuals,
# #             color=color_palette[i],
# #             alpha=0.2,
# #         )
# #     plt.xlabel(x_axis_label)
# #     plt.ylabel(y_axis_label)
# #     set_scale(plt.gca(), log_scale)
# #     plt.legend()
# #     fig = plt.gcf()
# #     plt.show()
# #     return fig
# # def exclude_arrests_from_series_at_ecdysis(series_at_ecdysis):
# #     filtered_series = np.full(series_at_ecdysis.shape, np.nan)
# #     # keep only a value at one ecdys event if the next one is not nan
# #     if series_at_ecdysis.shape[0] == 1 or len(series_at_ecdysis.shape) == 1:
# #         for i in range(len(series_at_ecdysis)):
# #             if i == len(series_at_ecdysis) - 1:
# #                 filtered_series[i] = series_at_ecdysis[i]
# #             elif not np.isnan(series_at_ecdysis[i + 1]):
# #                 filtered_series[i] = series_at_ecdysis[i]
# #         return filtered_series
# #     else:
# #         for i in range(series_at_ecdysis.shape[0]):
# #             for j in range(series_at_ecdysis.shape[1]):
# #                 if j == series_at_ecdysis.shape[1] - 1:
# #                     filtered_series[i, j] = series_at_ecdysis[i, j]
# #                 elif not np.isnan(series_at_ecdysis[i, j + 1]):
# #                     filtered_series[i, j] = series_at_ecdysis[i, j]
# #         return filtered_series
# # def get_proportion_model_ecdysis(
# #     series_one_at_ecdysis,
# #     series_two_at_ecdysis,
# #     remove_hatch=True,
# #     x_axis_label=None,
# #     y_axis_label=None,
# #     exclude_arrests=False,
# # ):
# #     assert len(series_one_at_ecdysis) == len(
# #         series_two_at_ecdysis
# #     ), "The two series must have the same length."
# #     if remove_hatch:
# #         series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
# #         series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]
# #     if exclude_arrests:
# #         series_one_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
# #             series_one_at_ecdysis
# #         )
# #         series_two_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
# #             series_two_at_ecdysis
# #         )
# #     series_one_at_ecdysis = np.array(series_one_at_ecdysis).flatten()
# #     series_two_at_ecdysis = np.array(series_two_at_ecdysis).flatten()
# #     # remove elements that are nan in one of the two arrays
# #     correct_indices = ~np.isnan(series_one_at_ecdysis) & ~np.isnan(
# #         series_two_at_ecdysis
# #     )
# #     series_one_at_ecdysis = series_one_at_ecdysis[correct_indices]
# #     series_two_at_ecdysis = series_two_at_ecdysis[correct_indices]
# #     # log transform the data
# #     series_one_at_ecdysis = np.log(series_one_at_ecdysis)
# #     series_two_at_ecdysis = np.log(series_two_at_ecdysis)
# #     plt.scatter(series_one_at_ecdysis, series_two_at_ecdysis)
# #     if x_axis_label is not None:
# #         plt.xlabel(x_axis_label)
# #     else:
# #         plt.xlabel("column one")
# #     if y_axis_label is not None:
# #         plt.ylabel(y_axis_label)
# #     else:
# #         plt.ylabel("column two")
# #     fit = np.polyfit(series_one_at_ecdysis, series_two_at_ecdysis, 3)
# #     model = np.poly1d(fit)
# #     plt.plot(
# #         np.sort(series_one_at_ecdysis),
# #         model(np.sort(series_one_at_ecdysis)),
# #         color="red",
# #     )
# #     plt.show()
# #     return model
# # def get_deviation_from_model_at_ecdysis(
# #     series_one_at_ecdysis,
# #     series_two_at_ecdysis,
# #     model,
# #     remove_hatch=True,
# #     exclude_arrests=False,
# # ):
# #     if remove_hatch:
# #         series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
# #         series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]
# #     if exclude_arrests:
# #         series_one_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
# #             series_one_at_ecdysis
# #         )
# #         series_two_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
# #             series_two_at_ecdysis
# #         )
# #     # remove elements that are nan in one of the two arrays
# #     for i in range(series_one_at_ecdysis.shape[1]):
# #         nan_mask = np.isnan(series_one_at_ecdysis[:, i]) | np.isnan(
# #             series_two_at_ecdysis[:, i]
# #         )
# #         series_one_at_ecdysis[:, i][nan_mask] = np.nan
# #         series_two_at_ecdysis[:, i][nan_mask] = np.nan
# #     # log transform the data
# #     series_one_at_ecdysis = np.log(series_one_at_ecdysis)
# #     series_two_at_ecdysis = np.log(series_two_at_ecdysis)
# #     expected_series_two = model(series_one_at_ecdysis)
# #     log_residuals = series_two_at_ecdysis - expected_series_two
# #     residuals = np.exp(log_residuals)
# #     y = np.nanmean(residuals, axis=0)
# #     y_err = np.nanstd(residuals, axis=0) / np.sqrt(len(residuals))
# #     x = np.nanmean(np.exp(series_one_at_ecdysis), axis=0)
# #     return x, y, y_err
# # def get_deviation_percentage_from_model_at_ecdysis(
# #     series_one_at_ecdysis,
# #     series_two_at_ecdysis,
# #     model,
# #     remove_hatch=True,
# #     exclude_arrests=False,
# # ):
# #     if remove_hatch:
# #         series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
# #         series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]
# #     if exclude_arrests:
# #         series_one_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
# #             series_one_at_ecdysis
# #         )
# #         series_two_at_ecdysis = exclude_arrests_from_series_at_ecdysis(
# #             series_two_at_ecdysis
# #         )
# #     # remove elements that are nan in one of the two arrays
# #     for i in range(series_one_at_ecdysis.shape[1]):
# #         nan_mask = np.isnan(series_one_at_ecdysis[:, i]) | np.isnan(
# #             series_two_at_ecdysis[:, i]
# #         )
# #         series_one_at_ecdysis[:, i][nan_mask] = np.nan
# #         series_two_at_ecdysis[:, i][nan_mask] = np.nan
# #     # Apply the model to the log-transformed series_one to get expected values
# #     expected_series_two = np.exp(model(np.log(series_one_at_ecdysis)))
# #     # Calculate percentage deviation using real values
# #     percentage_deviation = (
# #         (series_two_at_ecdysis - expected_series_two) / expected_series_two * 100
# #     )
# #     y = np.nanmean(percentage_deviation, axis=0)
# #     y_err = np.nanstd(percentage_deviation, axis=0) / np.sqrt(
# #         np.sum(~np.isnan(percentage_deviation), axis=0)
# #     )
# #     x = np.nanmean(series_one_at_ecdysis, axis=0)
# #     return x, y, y_err
# # def plot_deviation_from_model_at_ecdysis(
# #     conditions_struct,
# #     column_one,
# #     column_two,
# #     control_condition_id,
# #     conditions_to_plot,
# #     remove_hatch=True,
# #     log_scale=(True, False),
# #     colors=None,
# #     legend=None,
# #     x_axis_label=None,
# #     y_axis_label=None,
# #     percentage=True,
# #     exclude_arrests=False,
# # ):
# #     if colors is None:
# #         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
# #     else:
# #         color_palette = colors
# #     xlbl = column_one
# #     ylbl = column_two
# #     x_axis_label = x_axis_label if x_axis_label is not None else xlbl
# #     y_axis_label = (
# #         y_axis_label
# #         if y_axis_label is not None
# #         else f"deviation from modeled {column_two}"
# #     )
# #     control_condition = conditions_struct[control_condition_id]
# #     control_model = get_proportion_model_ecdysis(
# #         control_condition[column_one],
# #         control_condition[column_two],
# #         remove_hatch,
# #         x_axis_label=xlbl,
# #         y_axis_label=ylbl,
# #         exclude_arrests=exclude_arrests,
# #     )
# #     for i, condition_id in enumerate(conditions_to_plot):
# #         condition = conditions_struct[condition_id]
# #         body_data, pharynx_data = condition[column_one], condition[column_two]
# #         if percentage:
# #             x, y, y_err = get_deviation_percentage_from_model_at_ecdysis(
# #                 body_data, pharynx_data, control_model, remove_hatch, exclude_arrests
# #             )
# #         else:
# #             x, y, y_err = get_deviation_from_model_at_ecdysis(
# #                 body_data, pharynx_data, control_model, remove_hatch, exclude_arrests
# #             )
# #         label = build_legend(condition, legend)
# #         plt.plot(x, y, label=label, color=color_palette[i], marker="o")
# #         plt.errorbar(x, y, yerr=y_err, color=color_palette[i], fmt="o", capsize=3)
# #     plt.xlabel(x_axis_label)
# #     plt.ylabel(y_axis_label)
# #     set_scale(plt.gca(), log_scale)
# #     plt.legend()
# #     fig = plt.gcf()
# #     plt.show()
# #     return fig
# # def plot_normalized_proportions(
# #     conditions_struct,
# #     column_one,
# #     column_two,
# #     control_condition_id,
# #     conditions_to_plot,
# #     aggregation="mean",
# #     log_scale=(True, False),
# #     legend=None,
# #     x_axis_label=None,
# #     y_axis_label=None,
# # ):
# #     color_palette = sns.color_palette("husl", len(conditions_to_plot))
# #     control_condition = conditions_struct[control_condition_id]
# #     control_column_one, control_column_two = (
# #         control_condition[column_one],
# #         control_condition[column_two],
# #     )
# #     aggregation_function = np.nanmean
# #     control_proportion = aggregation_function(
# #         control_column_two / control_column_one, axis=0
# #     )
# #     x_axis_label = x_axis_label if x_axis_label is not None else column_one
# #     y_axis_label = (
# #         y_axis_label
# #         if y_axis_label is not None
# #         else f"normalized {column_two} to {column_one} ratio"
# #     )
# #     for i, condition_id in enumerate(conditions_to_plot):
# #         condition = conditions_struct[condition_id]
# #         condition_column_one = condition[column_one]
# #         condition_column_two = condition[column_two]
# #         proportion = condition_column_two / condition_column_one
# #         normalized_proportion = proportion / control_proportion
# #         y = aggregation_function(normalized_proportion, axis=0)
# #         y_err = np.nanstd(normalized_proportion, axis=0) / np.sqrt(
# #             len(normalized_proportion)
# #         )
# #         x = aggregation_function(condition_column_one, axis=0)
# #         label = build_legend(condition, legend)
# #         plt.plot(x, y, label=label, color=color_palette[i])
# #         plt.errorbar(x, y, yerr=y_err, fmt="o", capsize=3, color=color_palette[i])
# #     plt.xlabel(x_axis_label)
# #     plt.ylabel(y_axis_label)
# #     set_scale(plt.gca(), log_scale)
# #     plt.legend()
# #     fig = plt.gcf()
# #     plt.show()
# #     return fig
# def process_series_at_ecdysis(
#     series: np.ndarray,
#     ecdysis: np.ndarray,
#     remove_hatch: bool = True,
#     exclude_arrests: bool = False,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Process and clean the input single series data.
#     """
#     if remove_hatch:
#         series = series[:, 1:]
#         ecdysis = ecdysis[:, 1:]
#     if exclude_arrests:
#         series = exclude_arrests_from_series_at_ecdysis(series)
#     # Remove nan elements
#     for i in range(series.shape[1]):
#         nan_mask = np.isnan(series[:, i])
#         series[:, i][nan_mask] = np.nan
#     return series, ecdysis
# def filter_non_worm_data(
#     data: np.ndarray, worm_type: np.ndarray, ecdysis: np.ndarray
# ) -> np.ndarray:
#     """
#     Filter out non-worm data points.
#     """
#     filtered_data = data.copy()
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             try:
#                 if (
#                     ~(np.isnan(data[i][j]))
#                     and worm_type[i][int(ecdysis[i][j])] != "worm"
#                 ):
#                     filtered_data[i][j] = np.nan
#             except ValueError:
#                 filtered_data[i][j] = np.nan
#     return filtered_data
# def get_condition_filemaps(
#     condition_dict: dict,
# ) -> dict[str, Any]:
#     """
#     Set up file mappings for image directories.
#     """
#     filemap_paths = condition_dict["filemap_path"]
#     unique_filemap_paths = np.unique(filemap_paths)
#     filemaps = {}
#     for filemap_path in unique_filemap_paths:
#         filemap = pd.read_csv(filemap_path)
#         filemaps[filemap_path] = filemap
#     return filemaps
# def keep_selected_columns(
#     filemap_dict: dict[str, Any], columns_to_keep
# ) -> dict[str, Any]:
#     columns = columns_to_keep.copy()
#     if "Point" not in columns:
#         columns.append("Point")
#     if "Time" not in columns:
#         columns.append("Time")
#     for key, filemap in filemap_dict.items():
#         filemap = filemap[columns]
#         filemap_dict[key] = filemap
#     return filemap_dict
# def get_image_paths_of_time_point(
#     point, time, filemap_path_of_point, filemaps, image_columns
# ):
#     filemap_of_point = filemaps[filemap_path_of_point]
#     filemap_of_point = filemap_of_point[filemap_of_point["Point"] == point]
#     filemap_of_point = filemap_of_point[filemap_of_point["Time"] == time]
#     image_paths = filemap_of_point[image_columns].values.flatten().tolist()
#     return image_paths
# def display_image(
#     img_path,
#     dpi: int = 300,
#     scale: float = 1.0,
#     cmap: str = "viridis",
#     backup_dir: str = None,
#     backup_file_name: str = None,
# ) -> None:
#     """
#     Display an image with the specified parameters.
#     """
#     if isinstance(img_path, str):
#         img = read_tiff_file(img_path)
#     if backup_dir is not None:
#         if backup_file_name is not None:
#             shutil.copy(img_path, os.path.join(backup_dir, backup_file_name))
#         else:
#             if isinstance(img_path, str):
#                 shutil.copy(img_path, backup_dir)
#             else:
#                 imwrite(os.path.join(backup_dir, "backup_image.tif"), img)
#     height, width = img.shape[-2:]
#     plt.figure(
#         figsize=((width / dpi) * scale, (height / dpi) * scale),
#         dpi=dpi,
#         facecolor="black",
#     )
#     plt.gca().set_facecolor("black")
#     plt.imshow(img, interpolation="none", aspect="equal", cmap=cmap)
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.axis("off")
#     plt.show()
# def get_most_average_proportions_at_ecdysis(
#     conditions_struct: dict,
#     column_one: str,
#     column_two: str,
#     img_dir_list: list[str],
#     conditions_to_plot: list[int],
#     remove_hatch: bool = True,
#     exclude_arrests: bool = False,
#     nb_per_condition: int = 1,
# ) -> None:
#     """
#     Calculate and display the most average proportions at ecdysis.
#     """
#     paths_dict = {}
#     for condition_id in conditions_to_plot:
#         condition = conditions_struct[condition_id]
#         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
#         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
#         series_one, series_two, point, experiment, ecdysis, worm_type = (
#             condition[key]
#             for key in [
#                 column_one,
#                 column_two,
#                 "point",
#                 "experiment",
#                 "ecdysis_time_step",
#                 worm_type_key,
#             ]
#         )
#         filemaps = get_condition_filemaps(condition)
#         filemaps = keep_selected_columns(filemaps, img_dir_list)
#         series_one, ecdysis = process_series_at_ecdysis(
#             series_one, ecdysis, remove_hatch, exclude_arrests
#         )
#         series_two, _ = process_series_at_ecdysis(
#             series_two, ecdysis, remove_hatch, exclude_arrests
#         )
#         series_one = filter_non_worm_data(series_one, worm_type, ecdysis)
#         series_two = filter_non_worm_data(series_two, worm_type, ecdysis)
#         ratio = series_one / series_two
#         ratio_mean = np.nanmean(ratio, axis=0)
#         image_paths = []
#         for i in range(ratio_mean.shape[0]):
#             ratio_molt = ratio[:, i]
#             ratio_mean_molt = ratio_mean[i]
#             distance_score = np.abs(ratio_molt - ratio_mean_molt)
#             sorted_idx = np.argsort(distance_score)
#             selected_idx = sorted_idx[:nb_per_condition]
#             point_of_indices = point[selected_idx].squeeze().astype(int)
#             ecdysis_of_indices = ecdysis[selected_idx, i].squeeze().astype(int)
#             # check if 0D array
#             if point_of_indices.shape == ():
#                 point_of_indices = [point_of_indices]
#                 ecdysis_of_indices = [ecdysis_of_indices]
#             filemap_paths_of_indices = condition["filemap_path"][selected_idx].squeeze()
#             if filemap_paths_of_indices.shape == ():
#                 filemap_paths_of_indices = [filemap_paths_of_indices]
#             image_paths_ecdysis = []
#             for j, (p, t, filemap_path) in enumerate(
#                 zip(point_of_indices, ecdysis_of_indices, filemap_paths_of_indices)
#             ):
#                 paths = get_image_paths_of_time_point(
#                     p, t, str(filemap_path), filemaps, img_dir_list
#                 )
#                 image_paths_ecdysis.append(paths)
#             image_paths.append(image_paths_ecdysis)
#         paths_dict[condition_id] = image_paths
#     return paths_dict
# # def get_most_average_deviations_at_ecdysis(
# #     conditions_struct: Dict,
# #     column_one: str,
# #     column_two: str,
# #     img_dir_list: List[str],
# #     control_condition_id: str,
# #     conditions_to_plot: List[int],
# #     remove_hatch: bool = True,
# #     exclude_arrests: bool = False,
# #     dpi: int = 200,
# #     nb_per_condition: int = 1,
# #     overlay: bool = True,
# #     cmap: List[str] = ['viridis'],
# #     backup_dir: str = None,
# #     backup_name = None,
# # ) -> None:
# #     """
# #     Calculate and display the most average deviations at ecdysis.
# #     """
# #     control_condition = conditions_struct[control_condition_id]
# #     control_model = get_proportion_model_ecdysis(
# #         control_condition[column_one],
# #         control_condition[column_two],
# #         remove_hatch,
# #         x_axis_label=column_one,
# #         y_axis_label=column_two,
# #         exclude_arrests=exclude_arrests,
# #     )
# #     for condition_id in conditions_to_plot:
# #         condition = conditions_struct[condition_id]
# #         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
# #         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
# #         series_one, series_two, point, experiment, ecdysis, worm_type = [
# #             condition[key] for key in [column_one, column_two, 'point', 'experiment', 'ecdysis_time_step', worm_type_key]
# #         ]
# #         filemaps = setup_image_filemaps(experiment, img_dir_list)
# #         series_one, series_two, point, ecdysis = process_series_data(
# #             series_one, series_two, point, ecdysis, remove_hatch, exclude_arrests
# #         )
# #         # Calculate expected values and deviations
# #         expected_series_two = np.exp(control_model(np.log(series_one)))
# #         percentage_deviation = ((series_two - expected_series_two) / expected_series_two * 100)
# #         percentage_deviation = filter_non_worm_data(percentage_deviation, worm_type, ecdysis)
# #         y = np.nanmean(percentage_deviation, axis=0)
# #         for i in range(percentage_deviation.shape[1]):
# #             deviation_molt = percentage_deviation[:, i]
# #             mean_deviation = y[i]
# #             sorted_idx = np.argsort(np.abs(deviation_molt - mean_deviation))
# #             valid_idx = sorted_idx[~np.isnan(deviation_molt[sorted_idx])][:nb_per_condition]
# #             for idx in valid_idx:
# #                 display_sample_images(experiment[idx][0], int(point[idx][i]),
# #                                    int(ecdysis[idx][i]), img_dir_list, filemaps, dpi, overlay=overlay, cmap=cmap, backup_dir=backup_dir)
# def get_most_average_size_at_ecdysis(
#     conditions_struct: dict,
#     column: str,
#     img_dir_list: list[str],
#     conditions_to_plot: list[int],
#     remove_hatch: bool = True,
#     exclude_arrests: bool = False,
#     dpi: int = 200,
#     nb_per_condition: int = 1,
#     overlay: bool = True,
#     cmap: list[str] = ["viridis"],
#     backup_dir: str = None,
#     backup_name=None,
# ) -> None:
#     """
#     Calculate and display the most average sizes at ecdysis.
#     """
#     paths_dict = {}
#     for condition_id in conditions_to_plot:
#         condition = conditions_struct[condition_id]
#         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
#         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
#         series, point, experiment, ecdysis, worm_type = (
#             condition[key]
#             for key in [
#                 column,
#                 "point",
#                 "experiment",
#                 "ecdysis_time_step",
#                 worm_type_key,
#             ]
#         )
#         filemaps = get_condition_filemaps(condition)
#         filemaps = keep_selected_columns(filemaps, img_dir_list)
#         series, ecdysis = process_series_at_ecdysis(
#             series, ecdysis, remove_hatch, exclude_arrests
#         )
#         series = filter_non_worm_data(series, worm_type, ecdysis)
#         size_mean = np.nanmean(series, axis=0)
#         image_paths = []
#         for i in range(size_mean.shape[0]):
#             size_molt = series[:, i]
#             size_mean_molt = size_mean[i]
#             distance_score = np.abs(size_molt - size_mean_molt)
#             sorted_idx = np.argsort(distance_score)
#             selected_idx = sorted_idx[:nb_per_condition]
#             point_of_indices = point[selected_idx].squeeze().astype(int)
#             ecdysis_of_indices = ecdysis[selected_idx, i].squeeze().astype(int)
#             # check if 0D array
#             if point_of_indices.shape == ():
#                 point_of_indices = [point_of_indices]
#                 ecdysis_of_indices = [ecdysis_of_indices]
#             filemap_paths_of_indices = condition["filemap_path"][selected_idx].squeeze()
#             if filemap_paths_of_indices.shape == ():
#                 filemap_paths_of_indices = [filemap_paths_of_indices]
#             image_paths_ecdysis = []
#             for j, (p, t, filemap_path) in enumerate(
#                 zip(point_of_indices, ecdysis_of_indices, filemap_paths_of_indices)
#             ):
#                 paths = get_image_paths_of_time_point(
#                     p, t, str(filemap_path), filemaps, img_dir_list
#                 )
#                 image_paths_ecdysis.append(paths)
#             image_paths.append(image_paths_ecdysis)
#         paths_dict[condition_id] = image_paths
#     return paths_dict
# def overlay_contours(
#     mask_one_path,
#     mask_two_path,
#     dpi: int = 300,
#     scale: float = 1.0,
#     center=False,
#     allign="left",
#     thickness=2,
# ) -> None:
#     """
#     Overlay two masks on top of each other.
#     """
#     mask_one = read_tiff_file(mask_one_path).astype(np.uint8)
#     mask_two = read_tiff_file(mask_two_path).astype(np.uint8)
#     max_height = max(mask_one.shape[0], mask_two.shape[0])
#     max_width = max(mask_one.shape[1], mask_two.shape[1])
#     m1, m2 = pad_images_to_same_dim(mask_one, mask_two)
#     diff = np.linalg.norm(m1 - m2)
#     flipped_m2 = np.flip(m2, axis=1)
#     diff_flipped = np.linalg.norm(m1 - flipped_m2)
#     if diff_flipped < diff:
#         mask_two = np.flip(mask_two, axis=1)
#     if allign == "left":
#         mask_one = pad_to_dim_equally(mask_one, max_height, mask_one.shape[1])
#         mask_two = pad_to_dim_equally(mask_two, max_height, mask_two.shape[1])
#         mask_one = pad_to_dim(mask_one, mask_one.shape[0], max_width)
#         mask_two = pad_to_dim(mask_two, mask_two.shape[0], max_width)
#     elif allign == "right":
#         mask_one = pad_to_dim_equally(mask_one, max_height, mask_one.shape[1])
#         mask_two = pad_to_dim_equally(mask_two, max_height, mask_two.shape[1])
#         mask_one = np.pad(
#             mask_one,
#             ((0, 0), (max_width - mask_one.shape[1], 0)),
#             mode="constant",
#             constant_values=0,
#         )
#         mask_two = np.pad(
#             mask_two,
#             ((0, 0), (max_width - mask_two.shape[1], 0)),
#             mode="constant",
#             constant_values=0,
#         )
#     elif allign == "center":
#         mask_one, mask_two = pad_images_to_same_dim(mask_one, mask_two)
#     # pad 5 pixels on each side
#     mask_one = cv2.copyMakeBorder(mask_one, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
#     mask_two = cv2.copyMakeBorder(mask_two, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
#     mask_one = cv2.medianBlur(mask_one, 5)
#     mask_two = cv2.medianBlur(mask_two, 5)
#     contours_one, _ = cv2.findContours(mask_one, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contour_img_one = np.zeros_like(mask_one)
#     cv2.drawContours(contour_img_one, contours_one, -1, 255, thickness)
#     contours_two, _ = cv2.findContours(mask_two, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contour_img_two = np.zeros_like(mask_two)
#     cv2.drawContours(contour_img_two, contours_two, -1, 255, thickness)
#     masked_contour_img_one = np.ma.masked_where(contour_img_one == 0, contour_img_one)
#     masked_contour_img_two = np.ma.masked_where(contour_img_two == 0, contour_img_two)
#     plt.figure(
#         figsize=((max_width / dpi) * scale, (max_height / dpi) * scale),
#         dpi=dpi,
#         facecolor="black",
#     )
#     plt.gca().set_facecolor("black")
#     plt.imshow(
#         masked_contour_img_one,
#         cmap=plt.cm.colors.LinearSegmentedColormap.from_list("", ["black", "yellow"]),
#         alpha=0.7,
#         vmin=0,
#         vmax=255,
#     )
#     plt.imshow(
#         masked_contour_img_two,
#         cmap=plt.cm.colors.LinearSegmentedColormap.from_list("", ["black", "red"]),
#         alpha=0.7,
#         vmin=0,
#         vmax=255,
#     )
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()
# def plot_heterogeneity_at_ecdysis(
#     conditions_struct: dict,
#     column: str,
#     conditions_to_plot: list[int],
#     remove_hatch=True,
#     legend=None,
#     colors=None,
#     x_axis_label=None,
#     y_axis_label=None,
#     exclude_arrests: bool = False,
# ):
#     if colors is None:
#         color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
#     else:
#         color_palette = colors
#     for i, condition in enumerate(conditions_to_plot):
#         condition_dict = conditions_struct[condition]
#         values = condition_dict[column]
#         if remove_hatch:
#             values = values[:, 1:]
#         if exclude_arrests:
#             values = exclude_arrests_from_series_at_ecdysis(values)
#         cvs = []
#         for j in range(values.shape[1]):
#             values_at_ecdysis = values[:, j]
#             cv = np.nanstd(values_at_ecdysis) / np.nanmean(values_at_ecdysis)
#             cvs.append(cv)
#         label = build_legend(condition_dict, legend)
#         plt.plot(cvs, label=label, marker="o", color=color_palette[i])
#     # replace the ticks by [L1, L2, L3, L4]
#     if remove_hatch:
#         plt.xticks(range(4), ["M1", "M2", "M3", "M4"])
#     else:
#         plt.xticks(range(5), ["Hatch", "M1", "M2", "M3", "M4"])
#     plt.xlabel(x_axis_label)
#     plt.ylabel(y_axis_label)
#     plt.legend()
#     fig = plt.gcf()
#     plt.show()
#     return fig
# def plot_heterogeneity_rescaled_data(
#     conditions_struct: dict,
#     column: str,
#     conditions_to_plot: list[int],
#     smooth: bool = False,
#     remove_hatch=True,
#     legend=None,
#     colors=None,
#     x_axis_label=None,
#     y_axis_label=None,
#     exclude_arrests: bool = False,
# ):
#     if colors is None:
#         color_palette = sns.color_palette("husl", len(conditions_to_plot))
#     else:
#         color_palette = colors
#     for i, condition in enumerate(conditions_to_plot):
#         condition_dict = conditions_struct[condition]
#         values = condition_dict[column]
#         cvs = np.nanstd(values, axis=0) / np.nanmean(values, axis=0)
#         label = build_legend(condition_dict, legend)
#         if smooth:
#             cvs = medfilt(cvs, 7)
#             # cvs = savgol_filter(cvs, 15, 3)
#         plt.plot(cvs, label=label, color=color_palette[i])
#     plt.xlabel(x_axis_label)
#     plt.ylabel(y_axis_label)
#     plt.legend()
#     fig = plt.gcf()
#     plt.show()
#     return fig
# # def combine_series(
# #     conditions_struct, series_one, series_two, operation, new_series_name
# # ):
# #     for condition in conditions_struct:
# #         series_one_values = condition[series_one]
# #         series_two_values = condition[series_two]
# #         if operation == "add":
# #             new_series_values = np.add(series_one_values, series_two_values)
# #         elif operation == "subtract":
# #             new_series_values = series_one_values - series_two_values
# #         elif operation == "multiply":
# #             new_series_values = series_one_values * series_two_values
# #         elif operation == "divide":
# #             new_series_values = np.divide(series_one_values, series_two_values)
# #         condition[new_series_name] = new_series_values
# #     return conditions_struct
# # def transform_series(conditions_struct, series, operation, new_series_name):
# #     for conditions in conditions_struct:
# #         series_values = conditions[series]
# #         if operation == "log":
# #             new_series_values = np.log(series_values)
# #         elif operation == "exp":
# #             new_series_values = np.exp(series_values)
# #         elif operation == "sqrt":
# #             new_series_values = np.sqrt(series_values)
# #         conditions[new_series_name] = new_series_values
# #     return conditions_struct
# # def compute_growth_rate(
# #     conditions_struct, series_name, gr_series_name, experiment_time=True
# # ):
# #     for condition in conditions_struct:
# #         series_values = condition[series_name]
# #         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
# #         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
# #         worm_type = condition[worm_type_key]
# #         if experiment_time:
# #             time = condition["experiment_time"]
# #         else:
# #             time = condition["time"]
# #         growth_rate = []
# #         for i in range(series_values.shape[0]):
# #             # gr = compute_instantaneous_growth_rate_classified(series_values[i], time[i], worm_type[i], smoothing_method = 'savgol', savgol_filter_window = 7)
# #             gr = compute_instantaneous_growth_rate_classified(
# #                 series_values[i],
# #                 time[i],
# #                 worm_type[i],
# #                 smoothing_method="savgol",
# #                 savgol_filter_window=5,
# #             )
# #             growth_rate.append(gr)
# #         growth_rate = np.array(growth_rate)
# #         condition[gr_series_name] = growth_rate
# #     return conditions_struct
# # def rescale(
# #     conditions_struct,
# #     series_name,
# #     rescaled_series_name,
# #     experiment_time=True,
# #     n_points=100,
# # ):
# #     for condition in conditions_struct:
# #         series_values = condition[series_name]
# #         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
# #         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
# #         worm_type = condition[worm_type_key]
# #         ecdysis = condition["ecdysis_index"]
# #         if experiment_time:
# #             time = condition["experiment_time"]
# #         else:
# #             time = condition["time"]
# #         _, rescaled_series = rescale_series(
# #             series_values, time, ecdysis, worm_type, n_points=n_points
# #         )  # shape (n_worms, 4, n_points)
# #         # reshape into (n_worms, 4*n_points)
# #         rescaled_series = rescaled_series.reshape(rescaled_series.shape[0], -1)
# #         condition[rescaled_series_name] = rescaled_series
# #     return conditions_struct
# # def rescale_without_flattening(
# #     conditions_struct,
# #     series_name,
# #     rescaled_series_name,
# #     experiment_time=True,
# #     n_points=100,
# # ):
# #     for condition in conditions_struct:
# #         series_values = condition[series_name]
# #         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
# #         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
# #         worm_type = condition[worm_type_key]
# #         ecdysis = condition["ecdysis_index"]
# #         if experiment_time:
# #             time = condition["experiment_time"]
# #         else:
# #             time = condition["time"]
# #         _, rescaled_series = rescale_series(
# #             series_values, time, ecdysis, worm_type, n_points=n_points
# #         )  # shape (n_worms, 4, n_points)
# #         condition[rescaled_series_name] = rescaled_series
# #     return conditions_struct
