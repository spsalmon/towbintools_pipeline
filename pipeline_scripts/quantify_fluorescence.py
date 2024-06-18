from towbintools.foundation import image_handling
from towbintools.quantification import compute_fluorescence_in_mask, compute_background_fluorescence
import os
from joblib import Parallel, delayed
import re
import pandas as pd
import utils


def quantify_fluorescence_from_file_path(
    source_image_path, source_image_channel, mask_path, aggregation="sum", background_aggregation="median"
):
    """Quantify the fluorescence of an image inside a mask."""
    source_image = image_handling.read_tiff_file(
        source_image_path, channels_to_keep=[source_image_channel]
    )
    mask = image_handling.read_tiff_file(mask_path)

    fluo = compute_fluorescence_in_mask(
        source_image, mask, aggregation=aggregation
    )
    fluo_std = compute_fluorescence_in_mask(
        source_image, mask, aggregation="std"
    )
    background_fluo = compute_background_fluorescence(
        source_image, mask, aggregation=background_aggregation
    )
    pattern = re.compile(r"Time(\d+)_Point(\d+)")
    match = pattern.search(mask_path)
    if match:
        time = int(match.group(1))
        point = int(match.group(2))
        return {"Time": time, "Point": point, "FluoAggregation": fluo, "FluoStd": fluo_std, "BackgroundFluo": background_fluo}
    else:
        raise ValueError("Could not extract time and point from file name.")


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    source_files = [f["source_image_path"] for f in input_files]
    mask_files = [f["mask_path"] for f in input_files]
    aggregation = config["fluorescence_quantification_aggregation"]
    background_aggregation = config["fluorescence_background_aggregation"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fluo = Parallel(n_jobs=n_jobs)(
        delayed(quantify_fluorescence_from_file_path)(
            source_file,
            config["fluorescence_quantification_source"][1],
            mask_file,
            aggregation=aggregation,
            background_aggregation=background_aggregation,
        )
        for source_file, mask_file in zip(source_files, mask_files)
    )
    fluo_dataframe = pd.DataFrame(fluo)

    # rename columns to match the rest of the pipeline
    output_file_basename = os.path.basename(output_file).split(".csv")[0]
    fluo_source = output_file_basename.split("_")[0]
    mask_column = output_file_basename.split("_on_")[-1]
    fluo_dataframe.rename(
        columns={
            "FluoAggregation": f"{fluo_source}_fluo_{aggregation}_on_{mask_column}",
            "FluoStd": f"{fluo_source}_fluo_std_on_{mask_column}",
            "BackgroundFluo": f"{fluo_source}_fluo_background_{background_aggregation}_on_{mask_column}",
        },
        inplace=True,
    )

    fluo_dataframe.to_csv(output_file, index=False)




if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
