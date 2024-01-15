from towbintools.foundation import image_handling, detect_molts
import argparse
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import re
import pandas as pd
import yaml
import utils


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
    data_of_point = data_of_point.sort_values(by=["Time"])
    volumes = data_of_point[volume_column].values
    worm_types = data_of_point[worm_type_column].values

    try:
        # Detect molts
        ecdysis, volume_at_ecdysis = detect_molts.find_molts(
            volumes, worm_types, molt_size_range, search_width, fit_width
        )
    except ValueError:
        # No molt detected
        ecdysis = {
            "hatch_time": np.nan,
            "M1": np.nan,
            "M2": np.nan,
            "M3": np.nan,
            "M4": np.nan,
        }
        volume_at_ecdysis = {
            "volume_at_hatch": np.nan,
            "volume_at_M1": np.nan,
            "volume_at_M2": np.nan,
            "volume_at_M3": np.nan,
            "volume_at_M4": np.nan,
        }

    volume_names = [
        f"{volume_column}_at_{molt}" for molt in ["HatchTime", "M1", "M2", "M3", "M4"]
    ]
    return {
        "Point": point,
        "HatchTime": ecdysis["hatch_time"],
        volume_names[0]: volume_at_ecdysis["volume_at_hatch"],
        "M1": ecdysis["M1"],
        volume_names[1]: volume_at_ecdysis["volume_at_M1"],
        "M2": ecdysis["M2"],
        volume_names[2]: volume_at_ecdysis["volume_at_M2"],
        "M3": ecdysis["M3"],
        volume_names[3]: volume_at_ecdysis["volume_at_M3"],
        "M4": ecdysis["M4"],
        volume_names[4]: volume_at_ecdysis["volume_at_M4"],
    }


def compute_other_features_at_molt(
    analysis_filemap, molt_dataframe, volume_column, worm_type_column, point
):
    data_of_point = analysis_filemap[analysis_filemap["Point"] == point]
    molt_data_of_point = molt_dataframe[molt_dataframe["Point"] == point]
    data_of_point = data_of_point.sort_values(by=["Time"])
    volumes = data_of_point[volume_column].values
    worm_types = data_of_point[worm_type_column].values

    # get the columns of the features to compute at each molt
    volumes_to_compute = [
        column
        for column in data_of_point.columns
        if ("volume" in column and "VolumeAt" not in column and column != volume_column)
    ]
    lengths_to_compute = [
        column for column in data_of_point.columns if ("length" in column)
    ]
    areas_to_compute = [
        column for column in data_of_point.columns if ("area" in column)
    ]

    columns_to_compute = volumes_to_compute + lengths_to_compute + areas_to_compute

    # compute the features at each molt
    features_at_molt = {"Point": point}

    for column in columns_to_compute:
        column_data = data_of_point[column].values
        for molt in ["HatchTime", "M1", "M2", "M3", "M4"]:
            molt_time = float(molt_data_of_point[molt].values[0])
            if not np.isnan(molt_time):
                features_at_molt[
                    f"{column}_at_{molt}"
                ] = detect_molts.compute_volume_at_time(
                    column_data, worm_types, molt_time
                )

    return features_at_molt


def main(input_dataframe_path, output_file, config, n_jobs):
    """Main function."""

    config = utils.load_pickles(config)[0]

    analysis_filemap = pd.read_pickle(input_dataframe_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    molts_and_volume = Parallel(n_jobs=n_jobs)(
        delayed(run_detect_molts)(
            analysis_filemap,
            config["molt_detection_volume"],
            config["molt_detection_worm_type"],
            point,
        )
        for point in analysis_filemap["Point"].unique()
    )
    molts_dataframe = pd.DataFrame(molts_and_volume)

    # compute other features at each molt
    other_features_at_molt = Parallel(n_jobs=n_jobs)(
        delayed(compute_other_features_at_molt)(
            analysis_filemap,
            molts_dataframe,
            config["molt_detection_volume"],
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
