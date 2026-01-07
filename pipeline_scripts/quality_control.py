import os
import re

import numpy as np
import pandas as pd
import utils
import xgboost as xgb
from joblib import delayed
from joblib import load
from joblib import Parallel
from towbintools.classification.qc_tools import compute_qc_features


def extract_time_point_and_worm_type(path):
    time_pattern = re.compile(r"Time(\d+)")
    point_pattern = re.compile(r"Point(\d+)")

    time_match = time_pattern.search(path)
    point_match = point_pattern.search(path)
    if time_match and point_match:
        time = int(time_match.group(1))
        point = int(point_match.group(1))
        return {"Time": time, "Point": point}
    else:
        raise ValueError("Could not extract time and point from file name.")


def main(input_pickle, output_file, config, n_jobs):
    """Main function."""

    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    input_images = [f["image_path"] for f in input_files]
    input_masks = [f["mask_path"] for f in input_files]

    # mask_only = input_images[0] is None or input_images[0][0] is None

    classifier_path = config["qc_model_path"]
    model_path_and_classes = load(classifier_path)

    egg_model_path = model_path_and_classes["egg_model_path"]
    qc_model_path = model_path_and_classes["qc_model_path"]

    egg_classifier = xgb.XGBClassifier()
    egg_classifier.load_model(egg_model_path)
    egg_classes = model_path_and_classes["egg_classes"]

    qc_classifier = xgb.XGBClassifier()
    qc_classifier.load_model(qc_model_path)
    qc_classes = model_path_and_classes["qc_classes"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    features = Parallel(n_jobs=-1)(
        delayed(compute_qc_features)(mask_path, image_path)
        for mask_path, image_path in zip(input_masks, input_images)
    )
    valid_indices = [i for i, x in enumerate(features) if x is not None]
    features = [features[i] for i in valid_indices]

    print(
        f"Ignored {(len(input_images) - len(valid_indices)) / len(input_images):.2%} of samples due to feature extraction failure / empty images."
    )

    # first, predict egg vs non-egg
    features_df = pd.concat(features, ignore_index=True)

    egg_predictions = egg_classifier.predict(features_df)
    egg_predictions = [egg_classes[pred] for pred in egg_predictions]

    # filter to only NON-egg samples for qc classification
    non_egg_indices = [i for i, pred in enumerate(egg_predictions) if pred != "egg"]
    qc_features_df = features_df.iloc[non_egg_indices].copy()

    # now, predict qc classes for non-egg samples
    qc_predictions = qc_classifier.predict(qc_features_df)
    qc_predictions = [qc_classes[pred] for pred in qc_predictions]

    # combine predictions - start with egg predictions
    predictions = egg_predictions.copy()
    # overwrite non-egg positions with qc predictions
    for i, idx in enumerate(non_egg_indices):
        predictions[idx] = qc_predictions[i]

    # map predictions back to original image indices
    full_predictions = np.full_like(
        np.array(input_masks), fill_value="error", dtype=object
    )
    for idx, valid_idx in enumerate(valid_indices):
        full_predictions[valid_idx] = predictions[idx]

    time_points = Parallel(n_jobs=-1)(
        delayed(extract_time_point_and_worm_type)(path) for path in input_masks
    )
    time_points_df = pd.DataFrame(time_points)
    worm_types_dataframe = pd.DataFrame(
        {
            "Time": time_points_df["Time"],
            "Point": time_points_df["Point"],
            f"{os.path.splitext(os.path.basename(output_file))[0]}": full_predictions,
        }
    )

    worm_types_dataframe.to_csv(output_file, index=False)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
