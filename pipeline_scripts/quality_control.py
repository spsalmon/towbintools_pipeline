import os
import re

import numpy as np
import pandas as pd
import polars as pl
import utils
import xgboost as xgb
from joblib import delayed
from joblib import load
from joblib import Parallel
from towbintools.classification.qc_tools import compute_qc_features
from towbintools.foundation.file_handling import write_filemap


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


def main(input_pickle, output_file, config, filemap, n_jobs=-1):
    """Main function."""

    config = utils.load_pickles(config)[0]

    input_files = utils.load_pickles(input_pickle)[0]
    input_images = [f["image_path"] for f in input_files]
    input_masks = [f["mask_path"] for f in input_files]

    filemap = utils.load_pickles(filemap)[0]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    classifier_path = config["qc_model_path"]
    model_path_and_classes = load(classifier_path)

    features = Parallel(n_jobs=n_jobs)(
        delayed(compute_qc_features)(mask_path, image_path)
        for mask_path, image_path in zip(input_masks, input_images)
    )

    valid_indices = [i for i, x in enumerate(features) if x is not None]
    features = [features[i] for i in valid_indices]

    features_df = pd.concat(features, ignore_index=True)

    print(
        f"Ignored {(len(input_images) - len(valid_indices)) / len(input_images):.2%} of samples due to feature extraction failure / empty images."
    )

    egg_model_path = model_path_and_classes.get("egg_model_path", None)
    qc_model_path = model_path_and_classes["qc_model_path"]
    import_eggs_from = config.get("qc_import_eggs_from", None)

    qc_classifier = xgb.XGBClassifier()
    qc_classifier.load_model(qc_model_path)
    qc_classes = model_path_and_classes["qc_classes"]

    # first, predict egg vs non-egg
    if egg_model_path is not None and import_eggs_from is None:
        egg_classifier = xgb.XGBClassifier()
        egg_classifier.load_model(egg_model_path)
        egg_classes = model_path_and_classes["egg_classes"]

        # if for some reason scikit-image decided to add features that we didn't include, we need to remove them
        egg_feature_names = egg_classifier.get_booster().feature_names
        extracted_feature_names = features_df.columns.to_list()
        extra_features = [
            fname for fname in extracted_feature_names if fname not in egg_feature_names
        ]
        if len(extra_features) > 0:
            egg_features_df = features_df.drop(columns=extra_features)

        egg_predictions = egg_classifier.predict(egg_features_df)
        egg_predictions = [egg_classes[pred] for pred in egg_predictions]

    elif import_eggs_from is not None:
        egg_predictions = filemap[import_eggs_from].to_list()
        egg_predictions = [egg_predictions[i] for i in valid_indices]

    else:
        egg_predictions = ["non-egg"] * len(features_df)

    non_egg_indices = [i for i, pred in enumerate(egg_predictions) if pred != "egg"]

    qc_features_df = features_df.iloc[non_egg_indices].copy()

    # now, predict qc classes for non-egg samples
    # if for some reason scikit-image decided to add features that we didn't include, we need to remove them
    qc_feature_names = qc_classifier.get_booster().feature_names
    extracted_feature_names = qc_features_df.columns.to_list()
    extra_features = [
        fname for fname in extracted_feature_names if fname not in qc_feature_names
    ]
    if len(extra_features) > 0:
        qc_features_df = qc_features_df.drop(columns=extra_features)

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
    time_points_df = pl.DataFrame(time_points)
    col_name = os.path.splitext(os.path.basename(output_file))[0]
    worm_types_dataframe = time_points_df.select(["Time", "Point"]).with_columns(
        pl.Series(col_name, full_predictions)
    )
    write_filemap(worm_types_dataframe, output_file)


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.filemap, args.n_jobs)
