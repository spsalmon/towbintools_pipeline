import argparse
import os

import numpy as np
import pandas as pd
import yaml
from towbintools.foundation import file_handling as file_handling

from pipeline_scripts.building_blocks import parse_and_create_building_blocks
from pipeline_scripts.run_functions import (
    run_classification,
    run_compute_volume,
    run_custom,
    run_detect_molts,
    run_fluorescence_quantification,
    run_segmentation,
    run_straightening,
)
from pipeline_scripts.utils import (
    add_dir_to_experiment_filemap,
    backup_file,
    create_temp_folders,
    get_and_create_folders,
    get_experiment_pads,
    get_experiment_time_from_filemap_parallel,
    merge_and_save_csv,
    rename_merge_and_save_csv,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

create_temp_folders()


def main(config, pad=None):
    experiment_dir, raw_subdir, analysis_subdir, report_subdir, sbatch_backup_dir = (
        get_and_create_folders(config)
    )

    if pad:
        raw_subdir = os.path.join(raw_subdir, pad)
        report_subdir = os.path.join(report_subdir, pad)
        sbatch_backup_dir = os.path.join(sbatch_backup_dir, pad)

    # add the directories to the config dictionary for easy access
    config["raw_subdir"] = raw_subdir
    config["analysis_subdir"] = analysis_subdir
    config["report_subdir"] = report_subdir
    config["sbatch_backup_dir"] = sbatch_backup_dir

    extract_experiment_time = config.get("extract_experiment_time", True)

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
    if ("ExperimentTime" not in experiment_filemap.columns):
        if extract_experiment_time:
            print("Computing experiment time ...")
            experiment_filemap["ExperimentTime"] = (
                get_experiment_time_from_filemap_parallel(experiment_filemap)
            )
            experiment_filemap.to_csv(
                os.path.join(report_subdir, "analysis_filemap.csv"), index=False
            )
        else:
            experiment_filemap["ExperimentTime"] = np.nan
            experiment_filemap.to_csv(
                os.path.join(report_subdir, "analysis_filemap.csv"), index=False
            )

    print("Building the config of the building blocks ...")

    building_blocks = parse_and_create_building_blocks(config)


    for building_block in building_blocks:
        print(f"Running {building_block} ...")
        result = building_block.run(experiment_filemap, config, pad=pad)

        # reload the experiment filemap in case it was modified during the function call
        experiment_filemap = pd.read_csv(
            os.path.join(report_subdir, "analysis_filemap.csv")
        )

        if building_block.return_type == "subdir":
            if pad:
                column_name = f'{config["analysis_dir_name"]}/{os.path.basename(os.path.dirname(os.path.normpath(result)))}'
            else:
                column_name = f'{config["analysis_dir_name"]}/{os.path.basename(os.path.normpath(result))}'

            experiment_filemap = add_dir_to_experiment_filemap(
                experiment_filemap,
                result,
                column_name,
            )
            experiment_filemap.to_csv(
                os.path.join(report_subdir, "analysis_filemap.csv"), index=False
            )

        elif building_block.return_type == "csv":
            if building_block.name == "molt_detection":
                experiment_filemap = merge_and_save_csv(
                    experiment_filemap, report_subdir, result, merge_cols=["Point"]
                )
            else:
                experiment_filemap = merge_and_save_csv(
                    experiment_filemap, report_subdir, result
                )

pads = get_experiment_pads(config)

print(f"Running the pipeline for pads: {pads}")
if not pads:
    main(config)
else:
    for pad in pads:
        main(config, pad)
