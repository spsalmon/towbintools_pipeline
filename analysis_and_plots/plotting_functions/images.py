from typing import Any

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from towbintools.foundation.image_handling import read_tiff_file

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


def get_condition_filemaps(
    condition_dict: dict,
) -> dict[str, Any]:
    """
    Set up file mappings for image directories.
    """
    filemap_paths = condition_dict["filemap_path"]
    unique_filemap_paths = np.unique(filemap_paths)
    filemaps = {}
    for filemap_path in unique_filemap_paths:
        filemap = pl.read_csv(
            filemap_path,
            infer_schema_length=10000,
            null_values=["np.nan", "[nan]", ""],
        )
        filemaps[filemap_path] = filemap
    return filemaps


def keep_selected_columns(
    filemap_dict: dict[str, Any], columns_to_keep
) -> dict[str, Any]:
    columns = columns_to_keep.copy()
    if "Point" not in columns:
        columns.append("Point")
    if "Time" not in columns:
        columns.append("Time")
    for key, filemap in filemap_dict.items():
        filemap_dict[key] = filemap.select(pl.col(columns))
    return filemap_dict


def filter_non_worm_data(
    data: np.ndarray,
    worm_type: np.ndarray,
) -> np.ndarray:
    """
    Set data points to np.nan where worm_type is not 'worm'.
    Assumes data and worm_type have the same shape.
    """
    mask = (worm_type != "worm") & ~np.isnan(data)
    filtered_data = data.copy()
    filtered_data[mask] = np.nan
    return filtered_data


def get_indices_from_percentages(percentages, ecdysis):
    indices = []

    for p in percentages:
        if p == 0:
            # Handle edge case for 0%
            index = ecdysis[:, 0]
        elif p == 1:
            # Handle edge case for 100%
            index = ecdysis[:, 4]
        else:
            # Determine which stage the percentage falls into
            stage = int(p * 4)  # 0, 1, 2, or 3 for stages L1-L4
            stage = min(stage, 3)  # Ensure we don't exceed stage 3

            # Calculate relative position within the stage
            stage_start = stage * 0.25
            stage_progress = (p - stage_start) / 0.25

            # Calculate span and index for the current stage
            span = ecdysis[:, stage + 1] - ecdysis[:, stage]
            index = ecdysis[:, stage] + (span * stage_progress)

        indices.append(np.round(index))

    return np.array(indices).T


def get_image_paths_of_time_point(point, time, filemap, image_columns):
    image_paths = (
        (
            filemap.filter(pl.col("Point") == point)
            .filter(pl.col("Time") == time)
            .select(pl.col(image_columns))
        )
        .to_numpy()
        .squeeze()
        .tolist()
    )

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    return image_paths


def get_most_average_images_development_percentage(
    conditions_struct: dict,
    column: str,
    rescaled_column: str,
    img_dir_list: list[str],
    conditions_to_plot: list[int],
    percentages,
    nb_per_condition: int = 1,
    backup_dir: str = None,
    backup_name=None,
):
    paths = []

    for condition_id in conditions_to_plot:
        condition = conditions_struct[condition_id]
        condition_filemaps = get_condition_filemaps(condition)

        # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
        worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
        series, rescaled_series, point, time, filemap_path, ecdysis, worm_type = (
            condition[key]
            for key in [
                column,
                rescaled_column,
                "point",
                "time",
                "filemap_path",
                "ecdysis_index",
                worm_type_key,
            ]
        )

        condition_filemaps = keep_selected_columns(condition_filemaps, img_dir_list)

        indices = np.clip(
            (percentages * rescaled_series.shape[1]).astype(int),
            0,
            rescaled_series.shape[1] - 1,
        ).astype(int)

        series_at_percentages = rescaled_series[:, indices]

        mean = bn.nanmean(series_at_percentages, axis=0)

        non_rescaled_indices = get_indices_from_percentages(percentages, ecdysis)

        for i in range(non_rescaled_indices.shape[1]):
            print(f"Processing percentage {percentages[i]}")
            indices = non_rescaled_indices[:, i]
            time_at_indices = []
            for j, ind in enumerate(indices):
                if np.isnan(ind):
                    time_at_indices.append(np.nan)
                else:
                    time_at_indices.append(time[j, ind.astype(int)])
            time_at_indices = np.array(time_at_indices, dtype=float)[:, np.newaxis]

            non_nan = np.where(~np.isnan(indices))[0]
            indices = indices[non_nan].astype(int)

            non_nan_point = point[non_nan]
            non_nan_time = time_at_indices[non_nan]
            non_nan_series = series[non_nan, :]
            non_nan_filemap_path = filemap_path[non_nan]

            distances = mean[i] - non_nan_series[:, indices[i]]

            sorted_distances = np.argsort(np.abs(distances))
            selected = sorted_distances[:nb_per_condition].squeeze()
            selected_points = non_nan_point[selected]
            selected_times = non_nan_time[selected]
            filemap_path_of_selected = non_nan_filemap_path[selected]

            if np.isscalar(selected_points) or np.shape(selected_points) == ():
                selected_points = np.array([selected_points])
                selected_times = np.array([selected_times])
                filemap_path_of_selected = np.array([filemap_path_of_selected])

            for i in range(len(selected_points)):
                p = selected_points[i]
                t = selected_times[i]
                path = filemap_path_of_selected[i]

                filemap = condition_filemaps[str(path)]

                image_paths = get_image_paths_of_time_point(
                    p,
                    t,
                    filemap,
                    img_dir_list,
                )

                print(f"Image paths for point {p}, time {t}: {image_paths}")

                img = read_tiff_file(image_paths[0], channels_to_keep=[1])

                plt.imshow(img, cmap="gray")
                plt.title(f"Condition {condition_id}, Point {p}, Time {t}")
                plt.show()

    return paths
