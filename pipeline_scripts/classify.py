from towbintools.foundation import image_handling
from towbintools.classification import classification_tools
import os
from joblib import Parallel, delayed
import re
import pandas as pd
import utils
import xgboost as xgb


def classify_worm_type_from_file_path(
    straightened_mask_path, pixelsize, classifier, classes=["worm", "egg", "error"]
):
    print(straightened_mask_path)
    str_mask = image_handling.read_tiff_file(straightened_mask_path)
    worm_type = classification_tools.classify_worm_type(
        str_mask, pixelsize, classifier, classes
    )

    pattern = re.compile(r"Time(\d+)_Point(\d+)")
    match = pattern.search(straightened_mask_path)
    if match:
        time = int(match.group(1))
        point = int(match.group(2))
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
    worm_types = Parallel(n_jobs=n_jobs)(
        delayed(classify_worm_type_from_file_path)(
            input_file, config["pixelsize"], classifier
        )
        for input_file in input_files
    )
    worm_types_dataframe = pd.DataFrame(worm_types)

    # rename the WormType column to the output file
    worm_types_dataframe.rename(columns={"WormType": os.path.splitext(os.path.basename(output_file))[0]}, inplace=True)
    worm_types_dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
