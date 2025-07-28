from typing import Any
from typing import Optional

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from microfilm.microplot import microshow
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


def get_condition_filemaps_images(
    condition_dict: dict,
) -> dict[str, Any]:
    filemap_paths = condition_dict["filemap_path"]
    unique_filemap_paths = np.unique(filemap_paths)
    filemaps = {}
    for filemap_path in unique_filemap_paths:
        filemap = pl.read_csv(
            filemap_path,
            infer_schema_length=10000,
            null_values=["np.nan", "[nan]", ""],
        )
        filemap = filemap.select(
            pl.col("*").filter(
                lambda col: col.name.startswith("raw") or "analysis" in col.name
            )
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


def get_images_ecdysis(
    conditions_struct: dict,
    column: str,
    img_dir: str,
    criterion: str,
    conditions_to_plot: list[int],
    nb_per_condition: int = 1,
    molts_to_plot: list[int] = ["M1", "M2", "M3", "M4"],
    channels: Optional[list[int]] = None,
    dpi: int = 100,
    scale: float = 1.0,
    pixelsize: float = 0.65,
    cmaps: Optional[list[str]] = None,
    show_scalebar: bool = True,
    scalebar_size: float = 100,
    scalebar_thickness: float = 0.02,
    scalebar_font_size: int = 12,
    scalebar_location: str = "lower right",
):
    paths_dict = {}

    molt_to_index = {"Hatch": -5, "M1": -4, "M2": -3, "M3": -2, "M4": -1}
    if isinstance(molts_to_plot, list):
        molt_indices = [
            molt_to_index[molt] for molt in molts_to_plot if molt in molt_to_index
        ]

    for condition_id in conditions_to_plot:
        condition = conditions_struct[condition_id]
        series_at_ecdysis, points, ecdysis = (
            condition[key]
            for key in [
                column,
                "point",
                "ecdysis_time_step",
            ]
        )
        points = points.squeeze()

        series_at_ecdysis = series_at_ecdysis[:, molt_indices]
        ecdysis = ecdysis[:, molt_indices]
        filemaps = get_condition_filemaps_images(condition)
        # filemaps = keep_selected_columns(filemaps, [img_dir])

        if criterion == "mean":
            measure = bn.nanmean(series_at_ecdysis, axis=0)
        elif criterion == "median":
            measure = bn.nanmedian(series_at_ecdysis, axis=0)
        elif criterion == "min":
            measure = bn.nanmin(series_at_ecdysis, axis=0)
        elif criterion == "max":
            measure = bn.nanmax(series_at_ecdysis, axis=0)

        distance = measure - series_at_ecdysis

        sorted_indices = np.argsort(np.abs(distance), axis=0)
        selected_indices = sorted_indices[:nb_per_condition, :]

        selected_points = points[selected_indices]
        selected_ecdysis = ecdysis[
            selected_indices, np.arange(selected_indices.shape[1])
        ]

        selected_indices_filemap_path = condition["filemap_path"].squeeze()[
            selected_indices
        ]

        all_images = []
        for i in range(selected_indices.shape[0]):
            n_best_images = []
            for j in range(selected_indices.shape[1]):
                point = selected_points[i, j]
                time = selected_ecdysis[i, j]
                filemap = filemaps[selected_indices_filemap_path[i, j]]
                image_paths = get_image_paths_of_time_point(
                    point, time, filemap, img_dir
                )

                n_best_images.append(image_paths)

            all_images.append(n_best_images)

        all_images = np.array(all_images)

        paths_dict[condition_id] = all_images

        for i in range(paths_dict[condition_id].shape[0]):
            for j in range(paths_dict[condition_id].shape[1]):
                img_path = paths_dict[condition_id][i, j]
                print(f"Condition {condition_id}: {img_path}")
                img = read_tiff_file(
                    img_path,
                    channels_to_keep=channels,
                )
                plt.figure(
                    figsize=(
                        (img.shape[-1] / dpi) * scale,
                        (img.shape[-2] / dpi) * scale,
                    ),
                    dpi=dpi,
                    facecolor="black",
                )
                ax = plt.gca()

                if not show_scalebar:
                    scalebar_thickness = 1e-10
                    scalebar_kwargs = {"label": None, "scale_loc": "none"}
                else:
                    scalebar_kwargs = {}
                microshow(
                    images=img,
                    cmaps=cmaps,
                    unit="um",
                    scalebar_unit_per_pix=pixelsize,
                    scalebar_color="white",
                    scalebar_font_size=scalebar_font_size,
                    ax=ax,
                    scalebar_location=scalebar_location,
                    scalebar_size_in_units=scalebar_size,
                    scalebar_thickness=scalebar_thickness,
                    scalebar_kwargs=scalebar_kwargs,
                )
                plt.show()


# def get_images_development_percentage(
#     conditions_struct: dict,
#     column: str,
#     rescaled_column: str,
#     criterion: str,
#     img_dir_list: list[str],
#     conditions_to_plot: list[int],
#     percentages,
#     nb_per_condition: int = 1,
#     backup_dir: str = None,
#     backup_name=None,
# ):
#     paths = []

#     for condition_id in conditions_to_plot:
#         condition = conditions_struct[condition_id]
#         condition_filemaps = get_condition_filemaps_images(condition)

#         # TEMPORARY, ONLY WORKS WITH SINGLE CLASSIFICATION, FIND A WAY TO GENERALIZE
#         worm_type_key = [key for key in condition.keys() if "worm_type" in key][0]
#         series, rescaled_series, point, time, filemap_path, ecdysis, worm_type = (
#             condition[key]
#             for key in [
#                 column,
#                 rescaled_column,
#                 "point",
#                 "time",
#                 "filemap_path",
#                 "ecdysis_index",
#                 worm_type_key,
#             ]
#         )

#         condition_filemaps = keep_selected_columns(condition_filemaps, img_dir_list)

#         indices = np.clip(
#             (percentages * rescaled_series.shape[1]).astype(int),
#             0,
#             rescaled_series.shape[1] - 1,
#         ).astype(int)

#         series_at_percentages = rescaled_series[:, indices]

#         if criterion == "mean":
#             measure = bn.nanmean(series_at_percentages, axis=0)
#         elif criterion == "median":
#             measure = bn.nanmedian(series_at_percentages, axis=0)
#         elif criterion == "min":
#             measure = bn.nanmin(series_at_percentages, axis=0)
#         elif criterion == "max":
#             measure = bn.nanmax(series_at_percentages, axis=0)
#         non_rescaled_indices = get_indices_from_percentages(percentages, ecdysis)

#         for i in range(non_rescaled_indices.shape[1]):
#             print(f"Processing percentage {percentages[i]}")
#             indices = non_rescaled_indices[:, i]
#             time_at_indices = []
#             for j, ind in enumerate(indices):
#                 if np.isnan(ind):
#                     time_at_indices.append(np.nan)
#                 else:
#                     time_at_indices.append(time[j, ind.astype(int)])
#             time_at_indices = np.array(time_at_indices, dtype=float)[:, np.newaxis]

#             non_nan = np.where(~np.isnan(indices))[0]
#             indices = indices[non_nan].astype(int)

#             non_nan_point = point[non_nan]
#             non_nan_time = time_at_indices[non_nan]
#             non_nan_series = series[non_nan, :]
#             non_nan_filemap_path = filemap_path[non_nan]

#             distances = measure[i] - non_nan_series[:, indices[i]]

#             sorted_distances = np.argsort(np.abs(distances))
#             selected = sorted_distances[:nb_per_condition].squeeze()
#             selected_points = non_nan_point[selected]
#             selected_times = non_nan_time[selected]
#             filemap_path_of_selected = non_nan_filemap_path[selected]

#             if np.isscalar(selected_points) or np.shape(selected_points) == ():
#                 selected_points = np.array([selected_points])
#                 selected_times = np.array([selected_times])
#                 filemap_path_of_selected = np.array([filemap_path_of_selected])

#             for i in range(len(selected_points)):
#                 p = selected_points[i]
#                 t = selected_times[i]
#                 path = filemap_path_of_selected[i]

#                 filemap = condition_filemaps[str(path)]

#                 image_paths = get_image_paths_of_time_point(
#                     p,
#                     t,
#                     filemap,
#                     img_dir_list,
#                 )

#                 print(f"Image paths for point {p}, time {t}: {image_paths}")

#                 img = read_tiff_file(image_paths[0], channels_to_keep=[1])

#                 plt.imshow(img, cmap="gray")
#                 plt.title(f"Condition {condition_id}, Point {p}, Time {t}")
#                 plt.show()

#     return paths
