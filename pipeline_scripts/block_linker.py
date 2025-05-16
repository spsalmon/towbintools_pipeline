import argparse
import os

import pandas as pd

from pipeline_scripts.utils import add_dir_to_experiment_filemap
from pipeline_scripts.utils import cleanup_files
from pipeline_scripts.utils import load_pickles
from pipeline_scripts.utils import merge_and_save_csv
from pipeline_scripts.utils import pickle_objects
from pipeline_scripts.utils import sync_backup_folder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--temp_dir",
        help="Path to the directory storing temporary files",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--result",
        help="Path to the directory / csv file storing the previous block's results",
        required=False,
    )
    args = parser.parse_args()
    return args


temp_dir = get_args().temp_dir
result = get_args().result

pickle_dir = os.path.join(temp_dir, "pickles")
config_pickle_path = os.path.join(pickle_dir, "config.pkl")
progress_pickle_path = os.path.join(pickle_dir, "progress_tracker.pkl")

all_pickles = [
    os.path.join(pickle_dir, f) for f in os.listdir(pickle_dir) if f.endswith(".pkl")
]
pickles_to_delete = [
    f for f in all_pickles if f not in [config_pickle_path, progress_pickle_path]
]
cleanup_files(*pickles_to_delete)

config, progress_tracker = load_pickles(config_pickle_path, progress_pickle_path)

temp_dir = config["temp_dir"]
report_subdir = config["report_subdir"]
pipeline_backup_dir = config["pipeline_backup_dir"]

sync_backup_folder(config["temp_dir"], pipeline_backup_dir)

experiment_filemap = pd.read_csv(os.path.join(report_subdir, "analysis_filemap.csv"))

current_block_index = progress_tracker["current_block_index"]
building_blocks = progress_tracker["building_blocks"]

# increment progress tracker
progress_tracker["current_block_index"] += 1
progress_tracker_pickle = {"path": "progress_tracker", "obj": progress_tracker}
progress_tracker_pickle_path = pickle_objects(temp_dir, progress_tracker_pickle)

if current_block_index > 0:
    previous = building_blocks[current_block_index - 1]
    previous_block, previous_pad = previous["block"], previous["pad"]

    if previous_block.return_type == "subdir":
        if previous_pad:
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

    elif previous_block.return_type == "csv":
        if previous_block.name == "molt_detection":
            experiment_filemap = merge_and_save_csv(
                experiment_filemap, report_subdir, result, merge_cols=["Point"]
            )
        else:
            experiment_filemap = merge_and_save_csv(
                experiment_filemap, report_subdir, result
            )

if current_block_index < len(building_blocks):
    current = building_blocks[current_block_index]
    current_building_block, current_pad = current["block"], current["pad"]

    print(f"Running {current_building_block} ...")
    result = current_building_block.run(experiment_filemap, config, pad=current_pad)

else:
    print("End of the pipeline!")
