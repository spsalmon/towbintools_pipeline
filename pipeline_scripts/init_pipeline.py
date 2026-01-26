import argparse
import os

import numpy as np
import polars as pl
import yaml
from towbintools.foundation.file_handling import get_dir_filemap
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.file_handling import write_filemap

from pipeline_scripts.building_blocks import parse_and_create_building_blocks
from pipeline_scripts.utils import create_temp_folders
from pipeline_scripts.utils import get_and_create_folders
from pipeline_scripts.utils import get_experiment_subdirs
from pipeline_scripts.utils import get_experiment_time_from_filemap
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
    global_config = yaml.load(f, Loader=yaml.FullLoader)

if temp_dir:
    temp_dir = os.path.abspath(temp_dir)
else:
    temp_dir = os.path.abspath(os.path.join(os.getcwd(), "temp_files"))
    # if the temp_dir does not exist, create it
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

temp_dir_basename = os.path.basename(temp_dir)
create_temp_folders(temp_dir)


def main(global_config, temp_dir_basename, temp_dir, subdir=None):
    print(f"### Initializing the pipeline for subdir {subdir} ###")

    (
        experiment_dir,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    ) = get_and_create_folders(global_config, subdir)

    pipeline_backup_dir = os.path.join(pipeline_backup_dir, temp_dir_basename)
    os.makedirs(pipeline_backup_dir, exist_ok=True)

    config = global_config.copy()

    config["raw_subdir"] = raw_subdir
    if subdir is None:
        raw_dir_name = os.path.basename(os.path.normpath(raw_subdir))
    else:
        # instead, get the name from the parent directory
        raw_dir_name = os.path.basename(os.path.normpath(os.path.dirname(raw_subdir)))

    config["raw_dir_name"] = raw_dir_name
    config["analysis_subdir"] = analysis_subdir
    config["report_subdir"] = report_subdir
    config["pipeline_backup_dir"] = pipeline_backup_dir
    config["temp_dir"] = temp_dir

    report_format = config.get("report_format", "csv")
    config["report_format"] = report_format

    sync_backup_folder(temp_dir, pipeline_backup_dir)

    extract_experiment_time = config.get("get_experiment_time", True)
    overwrite_annotated = config.get("overwrite_annotated_filemap", False)

    if not os.path.exists(
        os.path.join(report_subdir, f"analysis_filemap.{report_format}")
    ):
        experiment_filemap = get_dir_filemap(raw_subdir)

        # if the filemap is empty, it's probably because they do not follow the Time, Point structure
        if experiment_filemap.is_empty():
            experiment_filemap = pl.DataFrame()
            experiment_filemap = experiment_filemap.with_columns(
                pl.lit(
                    sorted(
                        [os.path.join(raw_subdir, f) for f in os.listdir(raw_subdir)]
                    )
                ).alias("ImagePath")
            )
            config["no_timepoints"] = True

        experiment_filemap = experiment_filemap.rename({"ImagePath": raw_dir_name})
        filemap_path = os.path.join(report_subdir, f"analysis_filemap.{report_format}")
        experiment_filemap = experiment_filemap.fill_nan("").fill_null("")
        write_filemap(experiment_filemap, filemap_path)

    else:
        if overwrite_annotated and os.path.exists(
            os.path.join(report_subdir, f"analysis_filemap_annotated.{report_format}")
        ):
            filemap_path = os.path.join(
                report_subdir, f"analysis_filemap_annotated.{report_format}"
            )
        else:
            filemap_path = os.path.join(
                report_subdir, f"analysis_filemap.{report_format}"
            )

        experiment_filemap = read_filemap(filemap_path)
        experiment_filemap = experiment_filemap.fill_nan("").fill_null("")

    config["filemap_path"] = filemap_path

    # if the ExperimentTime column is not present, create it
    if "ExperimentTime" not in experiment_filemap.columns:
        if extract_experiment_time:
            print("### Calculating ExperimentTime ###")
            experiment_filemap = experiment_filemap.with_columns(
                pl.lit(
                    get_experiment_time_from_filemap(experiment_filemap, config)
                ).alias("ExperimentTime")
            )
            write_filemap(experiment_filemap, filemap_path)
        else:
            experiment_filemap = experiment_filemap.with_columns(
                pl.lit(np.nan).alias("ExperimentTime")
            )
            write_filemap(experiment_filemap, filemap_path)

    print("Building the config of the building blocks ...")

    building_blocks = parse_and_create_building_blocks(config)

    building_blocks = [
        {"block": block, "subdir": subdir, "config": config}
        for block in building_blocks
    ]
    print(f"Building blocks created: {building_blocks}")

    return building_blocks


subdirs = get_experiment_subdirs(global_config)
building_blocks = []
print(f"Running the pipeline for subdirs: {subdirs}")
if not subdirs:
    building_blocks.extend(main(global_config, temp_dir_basename, temp_dir))
else:
    for subdir in subdirs:
        building_blocks.extend(main(global_config, temp_dir_basename, temp_dir, subdir))

# initialize on 1 as we're gonna run the first block immediately
progress_tracker = {"current_block_index": 1, "building_blocks": building_blocks}
progress_tracker_pickle = {"path": "progress_tracker", "obj": progress_tracker}

_ = pickle_objects(temp_dir, progress_tracker_pickle)

# Run the first building block
current = building_blocks[0]
current_building_block, current_subdir, current_config = (
    current["block"],
    current["subdir"],
    current["config"],
)

experiment_filemap = read_filemap(current_config["filemap_path"])
current_building_block.run(experiment_filemap, current_config, subdir=current_subdir)
