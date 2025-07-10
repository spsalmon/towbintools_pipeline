import os
import warnings

import numpy as np
import pandas as pd
import polars as pl
import torch
import utils
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from torch.utils.data import DataLoader
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.data_analysis import correct_series_with_classification
from towbintools.deep_learning.deep_learning_tools import (
    load_keypoint_detection_model_from_checkpoint,
)
from towbintools.deep_learning.utils.dataset import KeypointDetection1DPredictionDataset
from towbintools.foundation import detect_molts
from towbintools.foundation.detect_molts import find_hatch_time
from towbintools.foundation.keypoint_detection import heatmap_to_keypoints_1D
from towbintools.foundation.worm_features import get_features_to_compute_at_molt
from tqdm import tqdm

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()


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


def run_detect_molts(
    analysis_filemap,
    volume_column,
    worm_type_column,
    point,
    molt_size_range=[6.6e4, 15e4, 36e4, 102e4],
    search_width=20,
    fit_width=5,
):
    data_of_point = analysis_filemap[analysis_filemap["Point"] == point]
    volumes = data_of_point[volume_column]
    # replace '' with np.nan
    volumes = volumes.replace("", np.nan)
    volumes = volumes.values.astype(float)
    worm_types = data_of_point[worm_type_column].values
    try:
        # Detect molts
        ecdysis = detect_molts.find_molts(
            volumes, worm_types, molt_size_range, search_width, fit_width
        )
    except ValueError as e:
        print(f"Error in point {point}: {e}")
        # No molt detected
        ecdysis = {
            "hatch_time": np.nan,
            "M1": np.nan,
            "M2": np.nan,
            "M3": np.nan,
            "M4": np.nan,
        }

    print(f"Point {point} done, ecdysis: {ecdysis}")

    return {
        "Point": point,
        "HatchTime": ecdysis["hatch_time"],
        "M1": ecdysis["M1"],
        "M2": ecdysis["M2"],
        "M3": ecdysis["M3"],
        "M4": ecdysis["M4"],
    }


def run_detect_molts_deep_learning(
    analysis_filemap,
    molt_detection_columns,
    worm_type_column,
    model_path,
    batch_size=1,
):
    analysis_filemap = pl.from_pandas(analysis_filemap)

    worm_type_data = separate_column_by_point(analysis_filemap, worm_type_column)

    hatch_times = []
    for i, worm_type_series in enumerate(worm_type_data):
        hatch_time = find_hatch_time(
            worm_type_series,
        )
        hatch_times.append(hatch_time)
    hatch_times = np.array(hatch_times)

    molt_detection_data = []
    for column in molt_detection_columns:
        if column not in analysis_filemap.columns:
            raise ValueError(f"Column {column} not found in analysis filemap.")
        column_data = separate_column_by_point(analysis_filemap, column)

        corrected_column_data = []
        for i, series_i in enumerate(column_data):
            worm_type_i = worm_type_data[i]
            corrected_column_data.append(
                np.log(
                    correct_series_with_classification(
                        series_i,
                        worm_type_i,
                    )
                )
            )
        molt_detection_data.append(np.array(corrected_column_data))

    molt_detection_data = np.array(molt_detection_data).squeeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_keypoint_detection_model_from_checkpoint(model_path).to(device)
    model.eval()

    dataset = KeypointDetection1DPredictionDataset(
        inputs=molt_detection_data,
        resize_method="pad",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    molts = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, invalid_series_index, original_shapes = batch
            x = x.to(device)
            heatmaps = model(x)
            heatmaps = heatmaps.cpu().numpy()

            for j, heatmap in enumerate(heatmaps):
                if j in invalid_series_index:
                    molts.append(
                        [np.nan] * 4
                    )  # series was invalid so no molts could be detected
                else:
                    m = heatmap_to_keypoints_1D(heatmap, height_threshold=0.5)
                    print(f"Molts detected : {m}")
                    molts.append(m)

    molts = np.array(molts)

    # Create a DataFrame with the results
    points = (
        analysis_filemap.select(pl.col("Point").unique(maintain_order=True))
        .to_numpy()
        .flatten()
    )
    molt_columns = ["M1", "M2", "M3", "M4"]
    molt_data = {
        "Point": points,
        "HatchTime": hatch_times,
    }
    for i, column in enumerate(molt_columns):
        molt_data[column] = molts[:, i]
    molts_dataframe = pd.DataFrame(molt_data)
    return molts_dataframe


def compute_features_at_molt(analysis_filemap, molt_dataframe, worm_type_column, point):
    data_of_point = analysis_filemap[analysis_filemap["Point"] == point]
    molt_data_of_point = molt_dataframe[molt_dataframe["Point"] == point]
    data_of_point = data_of_point.sort_values(by=["Time"])
    worm_types = data_of_point[worm_type_column].values

    columns_to_compute = []
    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        columns_to_compute.extend(
            [
                column
                for column in data_of_point.columns
                if (feature in column) and ("at_" not in column)
            ]
        )

    # compute the features at each molt
    features_at_molt = {"Point": point}

    for column in columns_to_compute:
        column_data = data_of_point[column]
        column_data = column_data.replace("", np.nan)
        column_data = column_data.values.astype(float)

        for molt in ["HatchTime", "M1", "M2", "M3", "M4"]:
            molt_time = float(molt_data_of_point[molt].values[0])
            try:
                if not np.isnan(molt_time):
                    features_at_molt[
                        f"{column}_at_{molt}"
                    ] = compute_series_at_time_classified(
                        column_data,
                        molt_time,
                        data_of_point["Time"].values,
                        worm_types,
                    )
                else:
                    features_at_molt[f"{column}_at_{molt}"] = np.nan
            except Exception as e:
                print(f"Error in point {point}, column {column}, molt {molt}: {e}")
                features_at_molt[f"{column}_at_{molt}"] = np.nan

    return features_at_molt


def main(input_dataframe_path, output_file, config, n_jobs):
    """Main function."""

    config = utils.load_pickles(config)[0]

    method = config.get("molt_detection_method", "legacy")

    print(f"Running molt detection with method: {method}")

    if method == "deep_learning":
        model_path = config.get("molt_detection_model_path", None)
        if model_path is None:
            raise ValueError("Model path must be provided for deep learning method.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    if "molt_detection_columns" not in config and "molt_detection_volume" in config:
        warnings.warn(
            "The option 'molt_detection_volume' is deprecated and will be removed in a future version. Use 'molt_detection_columns' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if "molt_detection_columns" not in config:
        molt_detection_columns = [config["molt_detection_volume"]]
    else:
        molt_detection_columns = config["molt_detection_columns"]

    analysis_filemap = pd.read_pickle(input_dataframe_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if method == "legacy":
        with parallel_config(backend="loky", n_jobs=n_jobs):
            molts = Parallel()(
                delayed(run_detect_molts)(
                    analysis_filemap,
                    molt_detection_columns[0],
                    config["molt_detection_worm_type"],
                    point,
                )
                for point in analysis_filemap["Point"].unique()
            )
            molts_dataframe = pd.DataFrame(molts)
    elif method == "deep_learning":
        molts_dataframe = run_detect_molts_deep_learning(
            analysis_filemap,
            molt_detection_columns,
            config["molt_detection_worm_type"],
            model_path,
            batch_size=config.get("molt_detection_batch_size", 1),
        )

    # compute other features at each molt
    with parallel_config(backend="loky", n_jobs=n_jobs):
        other_features_at_molt = Parallel()(
            delayed(compute_features_at_molt)(
                analysis_filemap,
                molts_dataframe,
                config["molt_detection_worm_type"],
                point,
            )
            for point in analysis_filemap["Point"].unique()
        )

    other_features_at_molt_dataframe = pd.DataFrame(other_features_at_molt)
    molts_dataframe = molts_dataframe.merge(
        other_features_at_molt_dataframe, on="Point"
    )

    molts_dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
