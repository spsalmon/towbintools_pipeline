import os
import re

import numpy as np
import polars as pl
import utils
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from towbintools.foundation import image_handling
from towbintools.foundation.file_handling import write_filemap
from towbintools.quantification import compute_fluorescence_in_mask


def quantify_fluorescence_from_file_path(
    source_image_path,
    source_image_channel,
    mask_path,
    aggregations=["sum"],
    background_aggregation="median",
):
    """Quantify the fluorescence of an image inside a mask."""

    if not isinstance(aggregations, list):
        aggregations = [aggregations]

    time_pattern = re.compile(r"Time(\d+)")
    point_pattern = re.compile(r"Point(\d+)")

    time_match = time_pattern.search(source_image_path)
    point_match = point_pattern.search(source_image_path)

    if time_match and point_match:
        time = int(time_match.group(1))
        point = int(point_match.group(1))
    else:
        raise ValueError("Could not extract time and point from file name.")

    source_image = image_handling.read_tiff_file(
        source_image_path, channels_to_keep=[source_image_channel]
    )
    mask = image_handling.read_tiff_file(mask_path)

    measurements = {
        "Time": time,
        "Point": point,
    }

    try:
        results = compute_fluorescence_in_mask(
            source_image,
            mask,
            aggregations=aggregations,
            background_aggregation=background_aggregation,
        )

    except Exception as e:
        print(f"Error processing {source_image_path} with mask {mask_path}: {e}")
        results = {agg: np.nan for agg in aggregations}

    measurements.update(results)

    return measurements


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    source_files = [f["source_image_path"] for f in input_files]
    mask_files = [f["mask_path"] for f in input_files]
    aggregations = config["fluorescence_quantification_aggregations"]
    background_aggregation = config["fluorescence_background_aggregation"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with parallel_config(backend="loky", n_jobs=n_jobs):
        fluo = Parallel()(
            delayed(quantify_fluorescence_from_file_path)(
                source_file,
                config["fluorescence_quantification_source"][1],
                mask_file,
                aggregations=aggregations,
                background_aggregation=background_aggregation,
            )
            for source_file, mask_file in zip(source_files, mask_files)
        )
    fluo_dataframe = pl.DataFrame(fluo)

    output_file_basename = os.path.basename(output_file).split(".")[0]
    fluo_source = output_file_basename.split("_")[0]
    mask_column = output_file_basename.split("_on_")[-1]
    for agg in aggregations:
        # rename columns to match the rest of the pipeline
        fluo_dataframe = fluo_dataframe.rename(
            {
                agg: f"{fluo_source}_fluo_{agg}_on_{mask_column}",
            },
        )

    write_filemap(fluo_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
