import yaml
import os
from towbintools.foundation import file_handling as file_handling
import pandas as pd
from pipeline_scripts.utils import (
    get_and_create_folders,
    add_dir_to_experiment_filemap,
    create_temp_folders,
)
import numpy as np
import shutil
from pipeline_scripts.run_functions import (
    run_segmentation,
    run_straightening,
    run_compute_volume,
    run_classification,
    run_detect_molts,
    run_fluorescence_quantification,
    run_custom,
)
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file",
        required=True
    )
    args = parser.parse_args()
    return args


config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

create_temp_folders()

experiment_dir, raw_subdir, analysis_subdir, report_subdir = (
    get_and_create_folders(config)
)

# copy the config file to the report folder
# if it already exists, change the name of the new one by adding a number
config_dir = os.path.join(report_subdir, "config")
os.makedirs(config_dir, exist_ok=True)
if os.path.exists(os.path.join(config_dir, os.path.basename(config_file))):
    i = 1
    while os.path.exists(
        os.path.join(
            config_dir, f"{os.path.splitext(os.path.basename(config_file))[0]}_{i}.yaml"
        )
    ):
        i += 1
    shutil.copyfile(
        config_file,
        os.path.join(
            config_dir, f"{os.path.splitext(os.path.basename(config_file))[0]}_{i}.yaml"
        ),
    )
else:
    shutil.copyfile(
        config_file, os.path.join(config_dir, os.path.basename(config_file))
    )

if not os.path.exists(os.path.join(report_subdir, "analysis_filemap.csv")):
    experiment_filemap = file_handling.get_dir_filemap(raw_subdir)
    experiment_filemap.rename(columns={"ImagePath": "raw"}, inplace=True)
    experiment_filemap.to_csv(
        os.path.join(report_subdir, "analysis_filemap.csv"), index=False
    )
else:
    experiment_filemap = pd.read_csv(
        os.path.join(report_subdir, "analysis_filemap.csv")
    )
    experiment_filemap = experiment_filemap.replace(np.nan, "", regex=True)

print(experiment_filemap.head())

building_blocks = config["building_blocks"]


def count_building_blocks_types(building_blocks):
    building_block_counts = {}
    for i, building_block in enumerate(building_blocks):
        if building_block not in building_block_counts:
            building_block_counts[building_block] = []
        building_block_counts[building_block] += [i]
    return building_block_counts


def build_config_of_building_blocks(building_blocks, config):
    building_block_counts = count_building_blocks_types(building_blocks)

    options_map = {
        "segmentation": [
            "rerun_segmentation",
            "segmentation_column",
            "segmentation_method",
            "segmentation_channels",
            "augment_contrast",
            "pixelsize",
            "sigma_canny",
            "model_path",
            "tiler_config",
            "RGB",
            "activation_layer",
            "batch_size",
            "ilastik_project_path",
            "ilastik_result_channel",
        ],
        "straightening": [
            "rerun_straightening",
            "straightening_source",
            "straightening_masks",
        ],
        "volume_computation": [
            "rerun_volume_computation",
            "volume_computation_masks",
            "pixelsize",
        ],
        "classification": [
            "rerun_classification",
            "classification_source",
            "classifier",
            "pixelsize",
        ],
        "molt_detection": [
            "rerun_molt_detection",
            "molt_detection_volume",
            "molt_detection_worm_type",
        ],
        "fluorescence_quantification": [
            "rerun_fluorescence_quantification",
            "fluorescence_quantification_source",
            "fluorescence_quantification_masks",
            "fluorescence_quantification_normalization",
            "pixelsize",
        ],
        "custom": ["custom_script_path", "custom_script_parameters"],
    }

    blocks_config = {}

    for i, building_block in enumerate(building_blocks):
        config_copy = config.copy()
        if building_block in options_map:
            options = options_map[building_block]
            # assert options
            for option in options:
                assert (
                    len(config[option]) == len(building_block_counts[building_block])
                    or len(config[option]) == 1
                ), f"{config[option]} The number of {option} options ({len(config[option])}) does not match the number of {building_block} building blocks ({len(building_block_counts[building_block])})"

            # expand single options to match the number of blocks
            for option in options:
                if len(config_copy[option]) == 1:
                    config_copy[option] = config[option] * len(
                        building_block_counts[building_block]
                    )

            # find the index of the building block
            idx = np.argwhere(
                np.array(building_block_counts[building_block]) == i
            ).squeeze()

            # set the options for the building block
            block_options = {}
            for option in options:
                block_options[option] = config_copy[option][idx]

            blocks_config[i] = block_options

    return blocks_config


blocks_config = build_config_of_building_blocks(building_blocks, config)


def rename_merge_and_save_csv(
    experiment_filemap,
    report_subdir,
    csv_file,
    column_name_old,
    column_name_new,
    merge_cols=["Time", "Point"],
):
    dataframe = pd.read_csv(csv_file)
    dataframe.rename(columns={column_name_old: column_name_new}, inplace=True)
    if column_name_new in experiment_filemap.columns:
        experiment_filemap.drop(columns=[column_name_new], inplace=True)
    experiment_filemap = experiment_filemap.merge(dataframe, on=merge_cols, how="left")
    experiment_filemap.to_csv(
        os.path.join(report_subdir, "analysis_filemap.csv"), index=False
    )
    return experiment_filemap


def merge_and_save_csv(
    experiment_filemap, report_subdir, csv_file, merge_cols=["Time", "Point"]
):
    dataframe = pd.read_csv(csv_file)
    new_columns = [
        column
        for column in dataframe.columns
        if (column != "Time" and column != "Point")
    ]
    for column in new_columns:
        if column in experiment_filemap.columns:
            experiment_filemap.drop(columns=[column], inplace=True)
    experiment_filemap = experiment_filemap.merge(dataframe, on=merge_cols, how="left")
    experiment_filemap.to_csv(
        os.path.join(report_subdir, "analysis_filemap.csv"), index=False
    )
    return experiment_filemap


blocks_config = build_config_of_building_blocks(building_blocks, config)

building_block_functions = {
    "segmentation": {"func": run_segmentation, "return_subdir": True},
    "straightening": {"func": run_straightening, "return_subdir": True},
    "volume_computation": {"func": run_compute_volume, "return_subdir": False},
    "classification": {
        "func": run_classification,
        "return_subdir": False,
        "column_name_old": "WormType",
        "column_name_new_key": True,
    },
    "molt_detection": {
        "func": run_detect_molts,
        "return_subdir": False,
        "process_molt": True,
    },
    "fluorescence_quantification": {
        "func": run_fluorescence_quantification,
        "return_subdir": False,
        "column_name_old": "Fluo",
        "column_name_new_key": True,
    },
    "custom": {"func": run_custom, "return_subdir": False},
}

for i, building_block in enumerate(building_blocks):
    block_config = blocks_config[i]
    if building_block in building_block_functions:
        func_data = building_block_functions[building_block]
        result = func_data["func"](experiment_filemap, config, block_config)

        # reload the experiment filemap in case it was modified during the function call
        experiment_filemap = pd.read_csv(os.path.join(report_subdir, "analysis_filemap.csv"))

        if func_data.get("return_subdir"):
            experiment_filemap = add_dir_to_experiment_filemap(
                experiment_filemap, result, f'analysis/{result.split("analysis/")[-1]}'
            )
            experiment_filemap.to_csv(
                os.path.join(report_subdir, "analysis_filemap.csv"), index=False
            )

        elif not func_data.get("process_molt"):
            if func_data.get("column_name_old") is not None:
                column_name_new = (
                    os.path.splitext(os.path.basename(result))[0]
                    if func_data.get("column_name_new_key")
                    else func_data.get("column_name_new")
                )
                experiment_filemap = rename_merge_and_save_csv(
                    experiment_filemap,
                    report_subdir,
                    result,
                    func_data["column_name_old"],
                    column_name_new,
                )
            else:
                experiment_filemap = merge_and_save_csv(
                    experiment_filemap, report_subdir, result
                )

        elif func_data.get("process_molt"):
            ecdysis_csv = pd.read_csv(os.path.join(report_subdir, "ecdysis.csv"))
            if "M1" not in experiment_filemap.columns:
                experiment_filemap = experiment_filemap.merge(
                    ecdysis_csv, on=["Point"], how="left"
                )
                experiment_filemap.to_csv(
                    os.path.join(report_subdir, "analysis_filemap.csv"), index=False
                )

    else:
        print(f"Functionality for {building_block} not implemented yet")
