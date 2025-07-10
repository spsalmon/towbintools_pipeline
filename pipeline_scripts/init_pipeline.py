import argparse
import os

import numpy as np
import pandas as pd
import yaml
from towbintools.foundation import file_handling as file_handling

from pipeline_scripts.building_blocks import parse_and_create_building_blocks
from pipeline_scripts.utils import create_temp_folders
from pipeline_scripts.utils import get_and_create_folders
from pipeline_scripts.utils import get_experiment_pads
from pipeline_scripts.utils import get_experiment_time_from_filemap_parallel
from pipeline_scripts.utils import pickle_objects
from pipeline_scripts.utils import sync_backup_folder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    parser.add_argument(
        "-t",
        "--temp_dir",
        help="Path to the directory storing temporary files",
        required=False,
    )
    args = parser.parse_args()
    return args


config_file = get_args().config
temp_dir = get_args().temp_dir

with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if temp_dir:
    temp_dir = os.path.abspath(temp_dir)
else:
    temp_dir = os.path.abspath(os.path.join(os.getcwd(), "temp_files"))
    # if the temp_dir does not exist, create it
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

temp_dir_basename = os.path.basename(temp_dir)
create_temp_folders(temp_dir)


def main(config, temp_dir_basename, temp_dir, pad=None):
    print(f"### Initializing the pipeline for pad {pad} ###")
    (
        experiment_dir,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    ) = get_and_create_folders(config)

    pipeline_backup_dir = os.path.join(pipeline_backup_dir, temp_dir_basename)
    os.makedirs(pipeline_backup_dir, exist_ok=True)

    if pad:
        raw_subdir = os.path.join(raw_subdir, pad)
        report_subdir = os.path.join(report_subdir, pad)
        pipeline_backup_dir = os.path.join(pipeline_backup_dir, pad)
        temp_dir = os.path.join(temp_dir, pad)

    # add the directories to the config dictionary for easy access
    config["raw_subdir"] = raw_subdir
    config["analysis_subdir"] = analysis_subdir
    config["report_subdir"] = report_subdir
    config["pipeline_backup_dir"] = pipeline_backup_dir
    config["temp_dir"] = temp_dir

    sync_backup_folder(temp_dir, pipeline_backup_dir)

    extract_experiment_time = config.get("get_experiment_time", True)

    if not os.path.exists(os.path.join(report_subdir, "analysis_filemap.csv")):
        experiment_filemap = file_handling.get_dir_filemap(raw_subdir)
        experiment_filemap.rename(columns={"ImagePath": "raw"}, inplace=True)
        experiment_filemap.to_csv(
            os.path.join(report_subdir, "analysis_filemap.csv"), index=False
        )
    else:
        experiment_filemap = pd.read_csv(
            os.path.join(report_subdir, "analysis_filemap.csv"),
            low_memory=False,
        )
        experiment_filemap = experiment_filemap.replace(np.nan, "", regex=True)

    # if the ExperimentTime column is not present, create it
    if "ExperimentTime" not in experiment_filemap.columns:
        if extract_experiment_time:
            print("### Calculating ExperimentTime ###")
            experiment_filemap[
                "ExperimentTime"
            ] = get_experiment_time_from_filemap_parallel(experiment_filemap)
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

    building_blocks = [{"block": block, "pad": pad} for block in building_blocks]

    return building_blocks


pads = get_experiment_pads(config)
building_blocks = []
print(f"Running the pipeline for pads: {pads}")
if not pads:
    building_blocks.extend(main(config, temp_dir_basename, temp_dir))
else:
    for pad in pads:
        building_blocks.extend(main(config, temp_dir_basename, temp_dir, pad))

# initialize on 1 as we're gonna run the first block immediately
progress_tracker = {"current_block_index": 1, "building_blocks": building_blocks}
progress_tracker_pickle = {"path": "progress_tracker", "obj": progress_tracker}
config_pickle = {"path": "config", "obj": config}

_ = pickle_objects(temp_dir, progress_tracker_pickle, config_pickle)

current = building_blocks[0]
current_building_block, current_pad = current["block"], current["pad"]

experiment_filemap = pd.read_csv(
    os.path.join(config["report_subdir"], "analysis_filemap.csv"),
    low_memory=False,
)
current_building_block.run(experiment_filemap, config, pad=current_pad)
