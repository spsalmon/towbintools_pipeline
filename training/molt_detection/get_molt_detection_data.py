import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm
from towbintools.data_analysis.time_series import correct_series_with_classification
from tqdm import tqdm


def get_landmarks(annotation, landmark_names):
    landmarks = []
    for landmark_name in landmark_names:
        try:
            landmark_x = annotation[annotation["Name"] == landmark_name]["X"].values[0]
        except IndexError:
            landmark_x = np.nan
        landmarks.append(landmark_x)

    landmarks = np.array(landmarks)

    return landmarks


def landmarks_to_gaussian_keymap(
    landmarks, sigma=1.5, rescale_length=200, multi_output=False
):
    landmarks_gaussian_keymap = []

    for i in range(len(landmarks)):
        if np.isnan(landmarks[i]):
            landmarks_gaussian_keymap.append(np.zeros(rescale_length))
        else:
            X = norm(landmarks[i] * 100, sigma)
            pdf = X.pdf(np.linspace(0, 100, rescale_length))
            normalized_pdf = pdf / np.max(pdf)
            landmarks_gaussian_keymap.append(normalized_pdf)

    if multi_output:
        return np.array(landmarks_gaussian_keymap)
    gaussian_keymap = np.sum(landmarks_gaussian_keymap, axis=0)
    return gaussian_keymap


def build_gaussian_keymaps(
    filemap, landmark_names, sigma=1.5, rescale_length=200, multi_output=False
):
    keymaps = []
    for i, row in filemap.iterrows():
        annotation_path = row["annotation"]
        annotation = pd.read_csv(annotation_path)
        # get the landmarks
        landmarks = get_landmarks(annotation, landmark_names)
        # create the keymap
        keymap = landmarks_to_gaussian_keymap(
            landmarks,
            sigma=sigma,
            rescale_length=rescale_length,
            multi_output=multi_output,
        )
        keymaps.append(keymap)
    return np.array(keymaps)
    return np.array(keymaps)


def get_analysis_filemap(experiment_path):
    directories = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
    ]

    analysis_directories = [
        d for d in directories if "analysis" in d and "matlab" not in d
    ]
    report_directories = [os.path.join(d, "report") for d in analysis_directories]

    report_directories = [d for d in report_directories if os.path.isdir(d)]

    for report_dir in report_directories:
        files = [os.path.join(report_dir, f) for f in os.listdir(report_dir)]
        filemap_files = [f for f in files if "analysis_filemap_annotated" in f]

        # return the filemap that was modified most recently
        if filemap_files:
            filemap_files.sort(key=os.path.getmtime)
            return os.path.join(report_dir, filemap_files[-1])
        # check for converted experiments
    return None


storage_cluster_path = "/mnt/towbin.data/shared"
valid_experimentalists = ["kstojanovski", "plenart", "igheor", "spsalmon"]
valid_scopes = ["ti2", "orca", "lipsi", "crest"]
valid_scopes_variations = []
for scope in valid_scopes:
    valid_scopes_variations.append(scope)
    valid_scopes_variations.append(scope.upper())
    valid_scopes_variations.append(scope.capitalize())

valid_experimentalists_dir = [
    os.path.join(storage_cluster_path, exp) for exp in valid_experimentalists
]

filemaps = []
for exp_dir in valid_experimentalists_dir:
    experiment_directories = [
        os.path.join(exp_dir, d)
        for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    ]

    for exp in experiment_directories:
        experiment_name = os.path.basename(os.path.normpath(exp))

        try:
            year = int(experiment_name[:4])
            if year < 2024:
                continue
        except ValueError:
            continue

        if "10x" not in experiment_name and "10X" not in experiment_name:
            continue

        if not any(scope in experiment_name for scope in valid_scopes_variations):
            continue

        filemap = get_analysis_filemap(exp)

        if filemap:
            f = pd.read_csv(filemap, nrows=1)
            if "ch2_seg_str_volume" not in f.columns:
                continue
            filemaps.append(filemap)

print(f"Found {len(filemaps)} valid experiments")


def get_features_and_ground_truth_molts(
    filemap,
    worm_type_column="ch2_seg_str_worm_type",
    volume_column="ch2_seg_str_volume",
    slope=8,
    sigma=1.5,
    multi_output=True,
):
    X = []
    y = []
    keypoints_all = []
    for point in filemap["Point"].unique():
        point_data = filemap[filemap["Point"] == point]
        data_of_point = point_data.sort_values(by=["Time"])
        volume_data = point_data[volume_column].values

        worm_type_data = point_data[worm_type_column].values
        if np.all(worm_type_data == worm_type_data[0]):
            print(
                f"Skipping point {point} because all worm types are the same: {worm_type_data[0]}"
            )
            continue

        volume_data = np.log(
            correct_series_with_classification(volume_data, worm_type_data)
        )

        # get the molt times
        keypoints = data_of_point[["HatchTime", "M1", "M2", "M3", "M4"]].values[0]
        if np.all(np.isnan(keypoints)):
            print(f"Skipping point {point} because all keypoints are NaN")
            continue

        keypoints = keypoints[1:]

        normalized_keypoints = keypoints / volume_data.shape[0]

        rescale_length = volume_data.shape[-1]

        if rescale_length < 200 or rescale_length > 1000:
            print(
                f"Skipping point {point} because length is not in the [200, 1000] range: {rescale_length}"
            )
            continue

        gaussian_keymaps = landmarks_to_gaussian_keymap(
            normalized_keypoints,
            sigma=sigma,
            rescale_length=rescale_length,
            multi_output=multi_output,
        )
        y.append(gaussian_keymaps)

        X.append(volume_data)
        keypoints_all.append(keypoints)

    return np.array(X), np.array(y), keypoints_all


def get_all_features_and_ground_truth_molts(
    filemaps,
    keymap_type="gaussian",
    worm_type_column="ch2_seg_str_worm_type",
    volume_column="ch2_seg_str_volume",
    slope=8,
    sigma=1.5,
    multi_output=True,
):
    X = []
    y = []
    keypoints_all = []

    for filemap_path in tqdm(filemaps):
        print(f"Processing filemap: {filemap_path}")
        filemap = pd.read_csv(filemap_path, low_memory=False)
        try:
            X_i, y_i, keypoints_i = get_features_and_ground_truth_molts(
                filemap,
                keymap_type=keymap_type,
                worm_type_column=worm_type_column,
                volume_column=volume_column,
                slope=slope,
                sigma=sigma,
                multi_output=multi_output,
            )
            if X_i is not None and X_i.size > 0:
                X.extend(X_i)
                y.extend(y_i)
                keypoints_all.extend(keypoints_i)
        except Exception as e:
            print(f"Error processing filemap {filemap_path}: {e}")
            continue

    # y = np.array(y)
    keypoints_all = np.array(keypoints_all)

    return X, y, keypoints_all


keymap_type = "gaussian"
sigma = 0.5
segment_names = ["M1", "M2", "M3", "M4"]
multi_output = True
slope = 5

X, y, keypoints_all = get_all_features_and_ground_truth_molts(
    filemaps,
    keymap_type=keymap_type,
    worm_type_column="ch2_seg_str_worm_type",
    volume_column="ch2_seg_str_volume",
    slope=slope,
    sigma=sigma,
    multi_output=multi_output,
)

pickle.dump(X, open("X_molt_detection.pickle", "wb"))
pickle.dump(y, open(f"y_{keymap_type}_molt_detection.pickle", "wb"))
pickle.dump(keypoints_all, open("keypoints_molt_detection.pickle", "wb"))
