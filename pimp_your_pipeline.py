import yaml
import os
from towbintools.foundation import file_handling as file_handling
import pandas as pd
from pipeline_scripts.utils import (
    get_and_create_folders,
    add_dir_to_experiment_filemap,
    create_temp_folders,
    backup_file,
    get_experiment_time_from_filemap_parallel,
    merge_and_save_csv,
    rename_merge_and_save_csv,
)
import numpy as np
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
from pipeline_scripts.building_blocks import build_config_of_building_blocks


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

experiment_dir, raw_subdir, analysis_subdir, report_subdir, sbatch_backup_dir = (
    get_and_create_folders(config)
)

# add the directories to the config dictionary for easy access
config["raw_subdir"] = raw_subdir
config["analysis_subdir"] = analysis_subdir
config["report_subdir"] = report_subdir
config["sbatch_backup_dir"] = sbatch_backup_dir

# copy the config file to the report folder
# if it already exists, change the name of the new one by adding a number
config_dir = os.path.join(report_subdir, "config")

os.makedirs(config_dir, exist_ok=True)

backup_file(config_file, config_dir)

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

# if the ExperimentTime column is not present, create it
if "ExperimentTime" not in experiment_filemap.columns:
    print('Computing experiment time ...')
    experiment_filemap["ExperimentTime"] = get_experiment_time_from_filemap_parallel(experiment_filemap)
    experiment_filemap.to_csv(
        os.path.join(report_subdir, "analysis_filemap.csv"), index=False
    )
    

print('Building the config of the building blocks ...')

building_blocks = config["building_blocks"]

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
        "func": run_fluorescence_quantification, "return_subdir": False},
    "custom": {"func": run_custom},
}

for i, building_block in enumerate(building_blocks):
    block_config = blocks_config[i]
    print(f'Running {building_block} ...')
    if building_block in building_block_functions:
        func_data = building_block_functions[building_block]
        result = func_data["func"](experiment_filemap, config, block_config)

        # reload the experiment filemap in case it was modified during the function call
        experiment_filemap = pd.read_csv(os.path.join(report_subdir, "analysis_filemap.csv"))

        if building_block == "custom":
            # check if result is a file or a directory
            if result is None:
                continue
            elif os.path.isdir(result):
                experiment_filemap = add_dir_to_experiment_filemap(
                    experiment_filemap, result, f'{config["analysis_dir_name"]}/{os.path.basename(os.path.normpath(result))}'
                )
                experiment_filemap.to_csv(
                    os.path.join(report_subdir, "analysis_filemap.csv"), index=False
                )
            elif os.path.isfile(result) and result.endswith(".csv"):
                experiment_filemap = merge_and_save_csv(experiment_filemap, report_subdir, result)

        elif func_data.get("return_subdir"):
            experiment_filemap = add_dir_to_experiment_filemap(
                experiment_filemap, result, f'{config["analysis_dir_name"]}/{os.path.basename(os.path.normpath(result))}'
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
