import os

import numpy as np
import pandas as pd
import utils
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.foundation import detect_molts
from towbintools.foundation.worm_features import get_features_to_compute_at_molt

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()


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
        ecdysis, volume_at_ecdysis = detect_molts.find_molts(
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


def compute_features_at_molt(
    analysis_filemap, molt_dataframe, volume_column, worm_type_column, point
):
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

    analysis_filemap = pd.read_pickle(input_dataframe_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with parallel_config(backend="loky", n_jobs=n_jobs):
        molts_and_volume = Parallel()(
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
    with parallel_config(backend="loky", n_jobs=n_jobs):
        other_features_at_molt = Parallel()(
            delayed(compute_features_at_molt)(
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
