import logging
import os

import polars as pl
import utils
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from towbintools.foundation import image_handling
from towbintools.foundation import worm_features
from towbintools.foundation.file_handling import extract_time_point
from towbintools.foundation.file_handling import write_filemap


def compute_morphological_features_from_file_path(
    straightened_mask_path,
    pixelsize,
    features,
    time_regex=r"Time(\d+)",
    point_regex=r"Point(\d+)",
):
    """Compute the volume of a straightened mask."""
    str_mask = image_handling.read_tiff_file(straightened_mask_path)

    features = worm_features.compute_mask_morphological_features(
        str_mask, pixelsize, features
    )

    try:
        time, point = extract_time_point(
            straightened_mask_path, time_regex, point_regex
        )
        features["Time"] = time
        features["Point"] = point
        return features
    except ValueError:
        return None


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    logging.info(f"Computing volume for {len(input_files)} files.")

    time_regex = config.get("time_regex", r"Time(\d+)")
    point_regex = config.get("point_regex", r"Point(\d+)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with parallel_config(backend="loky", n_jobs=n_jobs):
        morphological_features = Parallel()(
            delayed(compute_morphological_features_from_file_path)(
                input_file,
                config["pixelsize"],
                config["morphological_features"],
                time_regex,
                point_regex,
            )
            for input_file in input_files
        )

    # filter out None values (files where time and point could not be extracted)
    morphological_features = [f for f in morphological_features if f is not None]

    morphological_features_dataframe = pl.DataFrame(morphological_features)

    # rename columns to match the rest of the pipeline
    output_file_basename = os.path.basename(output_file).split("_morphology")[0]
    for feature in config["morphological_features"]:
        morphological_features_dataframe = morphological_features_dataframe.rename(
            {
                feature: f"{output_file_basename}_{feature}",
            },
        )
    write_filemap(morphological_features_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
