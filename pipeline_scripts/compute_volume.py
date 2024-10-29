import argparse
import logging
import os
import re

import numpy as np
import pandas as pd
import utils
import yaml
from joblib import Parallel, delayed
from tifffile import imwrite
from towbintools.foundation import image_handling, worm_features
from towbintools.segmentation import segmentation_tools


def compute_volume_from_file_path(straightened_mask_path, pixelsize):
    """Compute the volume of a straightened mask."""
    logging.info(straightened_mask_path)
    str_mask = image_handling.read_tiff_file(straightened_mask_path)
    volume = worm_features.compute_worm_volume(str_mask, pixelsize)
    length = worm_features.compute_worm_length(str_mask, pixelsize)
    area = worm_features.compute_worm_area(str_mask, pixelsize)

    # replace volume and length with NaN if it is 0
    if volume == 0:
        volume = np.nan
    if length == 0:
        length = np.nan
    if area == 0:
        area = np.nan

    pattern = re.compile(r"Time(\d+)_Point(\d+)")
    match = pattern.search(straightened_mask_path)
    if match:
        time = int(match.group(1))
        point = int(match.group(2))
        return {
            "Time": time,
            "Point": point,
            "Volume": volume,
            "Length": length,
            "Area": area,
        }
    else:
        raise ValueError("Could not extract time and point from file name.")


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    logging.info(f"Computing volume for {len(input_files)} files.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    volumes = Parallel(n_jobs=n_jobs)(
        delayed(compute_volume_from_file_path)(input_file, config["pixelsize"])
        for input_file in input_files
    )
    volume_dataframe = pd.DataFrame(volumes)

    # rename columns to match the rest of the pipeline
    output_file_basename = os.path.basename(output_file).split("_volume.csv")[0]
    volume_dataframe.rename(
        columns={
            "Volume": f"{output_file_basename}_volume",
            "Length": f"{output_file_basename}_length",
            "Area": f"{output_file_basename}_area",
        },
        inplace=True,
    )
    volume_dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
