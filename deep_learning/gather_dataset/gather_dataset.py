import argparse
import os

import numpy as np
import yaml

# import re
# import shutil
# import pandas as pd
# from joblib import delayed
# from joblib import Parallel
# from tifffile import imwrite
# from towbintools.foundation.image_handling import read_tiff_file

# set random seed for reproducibility
# np.random.seed(42)
np.random.seed(387799)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


def process_strains(strains):
    # to be correct the strain number needs to be followed by a separator like -, _ or space
    correct_strains = []
    for strain in strains:
        correct_strains.append(strain + "-")
        correct_strains.append(strain + "_")
        correct_strains.append(strain + " ")
    return correct_strains


def get_analysis_filemap(experiment_path):
    directories = [
        os.path.join(experiment_path, d)
        for d in os.listdir(experiment_path)
        if os.path.isdir(os.path.join(experiment_path, d))
    ]

    analysis_directories = [d for d in directories if "analysis" in d]
    report_directories = [os.path.join(d, "report") for d in analysis_directories]

    report_directories = [d for d in report_directories if os.path.isdir(d)]

    for report_dir in report_directories:
        files = [os.path.join(report_dir, f) for f in os.listdir(report_dir)]
        filemap_files = [f for f in files if "analysis_filemap_annotated" in f]
        # mat_files = [f for f in files if f.endswith(".mat")]
        # return the filemap that was created last
        if filemap_files:
            filemap_files.sort(key=lambda x: os.path.getctime(x))
            return os.path.join(report_dir, filemap_files[-1])
        # check for converted experiments
        # elif mat_files:
        #     filemap_files = [f for f in files if "analysis_filemap" in f]
        #     if filemap_files:
        #         filemap_files.sort(key=lambda x: os.path.getctime(x))
        #         filemap = pd.read_csv(filemap_files[-1], low_memory=False)
        #         if "HatchTime" in filemap.columns and "raw" in filemap.columns:
        #             return os.path.join(report_dir, filemap_files[-1])
    return None


def find_all_relevant_filemaps(
    experiment_directories,
    experiments_to_always_include,
    keywords_to_include,
    experiments_to_exclude,
    valid_scopes_expanded,
    keywords_to_exclude,
    database_config,
):
    filemaps = []
    strains = process_strains(database_config.get("strains", []))
    magnifications = database_config.get("magnifications", [])
    processed_magnifications = []
    for mag in magnifications:
        processed_magnifications.append(mag)
        processed_magnifications.append(mag.lower())
        processed_magnifications.append(mag.upper())

    magnifications = list(set(processed_magnifications))

    for exp in experiment_directories:
        experiment_name = os.path.basename(os.path.normpath(exp))

        # check if the experiment is in the list of experiments to always include
        if experiment_name in experiments_to_always_include:
            filemap = get_analysis_filemap(exp)
            if filemap:
                filemaps.append(filemap)
                continue

        if any(keyword in experiment_name for keyword in keywords_to_include):
            filemap = get_analysis_filemap(exp)
            if filemap:
                filemaps.append(filemap)
                continue

        # check if the experiment is in the list of experiments to exclude
        if experiment_name in experiments_to_exclude:
            continue

        if not any(mag in experiment_name for mag in magnifications) and magnifications:
            continue

        if not any(scope in experiment_name for scope in valid_scopes_expanded):
            continue

        if any(keyword in experiment_name for keyword in keywords_to_exclude):
            continue

        if not any(strain in experiment_name for strain in strains) and strains:
            continue

        filemap = get_analysis_filemap(exp)

        if filemap:
            filemaps.append(filemap)

    return filemaps


config_path = get_args().config
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

storage_path = config.get("storage_path", "")
valid_subdirectories = config.get("valid_subdirectories", [])

database_path = config.get("database_path", None)
database_configs = config.get("database_configs", {})
extra_adulthood_time = config.get("extra_adulthood_time", 0)
valid_scopes = config.get("valid_scopes", [])
scopes_alt_names = config.get("scopes_alt_names", {})
keywords_to_exclude = config.get("keywords_to_exclude", [])
keywords_to_include = config.get("keywords_to_include", [])
experiments_to_consider = config.get("experiments_to_consider", [])
experiments_to_always_include = config.get("experiments_to_always_include", [])
experiments_to_exclude = config.get("experiments_to_exclude", [])

valid_scopes_expanded = []
for scope in valid_scopes:
    if scope in scopes_alt_names:
        valid_scopes_expanded.extend(scopes_alt_names[scope])
    else:
        valid_scopes_expanded.append(scope)

os.makedirs(database_path, exist_ok=True)
for sub_db in database_configs.keys():
    sub_db_dir = os.path.join(database_path, sub_db)
    os.makedirs(sub_db_dir, exist_ok=True)
    os.makedirs(os.path.join(sub_db_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(sub_db_dir, "masks"), exist_ok=True)

# get all experiment directories
experiment_directories = []
valid_subdirectories = [
    os.path.join(storage_path, sub_dir) for sub_dir in valid_subdirectories
]
if experiments_to_consider:
    experiment_directories = experiments_to_consider
else:
    for exp_dir in valid_subdirectories:
        experiment_directories.extend(
            [
                os.path.join(exp_dir, d)
                for d in os.listdir(exp_dir)
                if os.path.isdir(os.path.join(exp_dir, d))
            ]
        )

experiment_directories.extend(experiments_to_always_include)
experiment_directories = list(set(experiment_directories))

# filter experiment directories based on the criteria and get their filemaps
for database_name, database_config in database_configs.items():
    print(f"Processing database: {database_name}")
    filemaps = find_all_relevant_filemaps(
        experiment_directories,
        experiments_to_always_include,
        keywords_to_include,
        experiments_to_exclude,
        valid_scopes_expanded,
        keywords_to_exclude,
        database_config,
    )

    print(f"Found {len(filemaps)} valid experiments")
