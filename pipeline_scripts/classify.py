import os
import re

import pandas as pd
import utils
import xgboost as xgb
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from towbintools.classification import classification_tools
from towbintools.foundation import image_handling


def classify_worm_type_from_file_path(
    straightened_mask_path, pixelsize, classifier, classes=["worm", "egg", "error"]
):
    print(straightened_mask_path)
    str_mask = image_handling.read_tiff_file(straightened_mask_path)
    worm_type = classification_tools.classify_worm_type(
        str_mask, pixelsize, classifier, classes
    )
    time_pattern = re.compile(r"Time(\d+)")
    point_pattern = re.compile(r"Point(\d+)")

    time_match = time_pattern.search(straightened_mask_path)
    point_match = point_pattern.search(straightened_mask_path)
    if time_match and point_match:
        time = int(time_match.group(1))
        point = int(point_match.group(1))
        return {"Time": time, "Point": point, "WormType": worm_type}
    else:
        raise ValueError("Could not extract time and point from file name.")


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""

    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]

    classifier_path = config["classifier"]
    classifier = xgb.XGBClassifier()
    classifier.load_model(classifier_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with parallel_config(backend="loky", n_jobs=n_jobs):
        worm_types = Parallel()(
            delayed(classify_worm_type_from_file_path)(
                input_file, config["pixelsize"], classifier
            )
            for input_file in input_files
        )
    worm_types_dataframe = pd.DataFrame(worm_types)

    # rename the WormType column to the output file
    worm_types_dataframe.rename(
        columns={"WormType": os.path.splitext(os.path.basename(output_file))[0]},
        inplace=True,
    )
    worm_types_dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
