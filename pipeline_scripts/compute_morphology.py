import logging
import os
import re
import pandas as pd
import utils
from joblib import Parallel, delayed, parallel_config
from towbintools.foundation import image_handling, worm_features

def compute_morphological_features_from_file_path(straightened_mask_path, pixelsize, features):
    """Compute the volume of a straightened mask."""
    str_mask = image_handling.read_tiff_file(straightened_mask_path)

    features = worm_features.compute_worm_morphological_features(str_mask, pixelsize, features)

    pattern = re.compile(r"Time(\d+)_Point(\d+)")
    match = pattern.search(straightened_mask_path)
    if match:
        time = int(match.group(1))
        point = int(match.group(2))
        features["Time"] = time
        features["Point"] = point
        return features
    else:
        raise ValueError("Could not extract time and point from file name.")


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    logging.info(f"Computing volume for {len(input_files)} files.")


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with parallel_config(backend="loky", n_jobs=n_jobs):
        morphological_features = Parallel()(
            delayed(compute_morphological_features_from_file_path)(input_file, config["pixelsize"], config["morphological_features"])
            for input_file in input_files
        )
    morphological_features_dataframe = pd.DataFrame(morphological_features)

    # rename columns to match the rest of the pipeline
    output_file_basename = os.path.basename(output_file).split("_morphology.csv")[0]
    for feature in config["morphological_features"]:
        morphological_features_dataframe.rename(
            columns={
                feature: f"{output_file_basename}_{feature}",
            },
            inplace=True,
        )

    morphological_features_dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
