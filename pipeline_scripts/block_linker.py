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
    return parser.parse_args()


def cleanup_temp_pickles(pickle_dir, keep_paths):
    """Remove all pickle files in pickle_dir except those in keep_paths."""
    all_pickles = [
        os.path.join(pickle_dir, f)
        for f in os.listdir(pickle_dir)
        if f.endswith(".pkl")
    ]
    pickles_to_delete = [f for f in all_pickles if f not in keep_paths]
    cleanup_files(*pickles_to_delete)


def update_experiment_filemap(
    experiment_filemap, config, result, previous_block, previous_pad, report_subdir
):
    """Update experiment_filemap based on previous block's return type."""
    if previous_block.return_type == "subdir":
        # The column name is always the same regardless of previous_pad
        column_name = f'{config["analysis_dir_name"]}/{os.path.basename(os.path.normpath(result))}'
        experiment_filemap = add_dir_to_experiment_filemap(
            experiment_filemap, result, column_name
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
    return experiment_filemap


def main():
    args = get_args()
    temp_dir = args.temp_dir
    result = args.result

    # Prepare pickle paths
    pickle_dir = os.path.join(temp_dir, "pickles")
    config_pickle_path = os.path.join(pickle_dir, "config.pkl")
    progress_pickle_path = os.path.join(pickle_dir, "progress_tracker.pkl")

    # Cleanup unnecessary pickles
    cleanup_temp_pickles(pickle_dir, [config_pickle_path, progress_pickle_path])

    # Load config and progress tracker
    config, progress_tracker = load_pickles(config_pickle_path, progress_pickle_path)

    report_subdir = config["report_subdir"]
    pipeline_backup_dir = config["pipeline_backup_dir"]

    # Sync backup folder
    sync_backup_folder(config["temp_dir"], pipeline_backup_dir)

    # Load experiment filemap
    experiment_filemap = pd.read_csv(
        os.path.join(report_subdir, "analysis_filemap.csv"), low_memory=False
    )

    current_block_index = progress_tracker["current_block_index"]
    building_blocks = progress_tracker["building_blocks"]

    # Increment progress tracker and save
    progress_tracker["current_block_index"] += 1
    pickle_objects(temp_dir, {"path": "progress_tracker", "obj": progress_tracker})

    # Update experiment_filemap with previous block's result
    if current_block_index > 0:
        previous = building_blocks[current_block_index - 1]
        previous_block, previous_pad = previous["block"], previous["pad"]
        experiment_filemap = update_experiment_filemap(
            experiment_filemap,
            config,
            result,
            previous_block,
            previous_pad,
            report_subdir,
        )

    # Run current block if available
    if current_block_index < len(building_blocks):
        current = building_blocks[current_block_index]
        current_building_block, current_pad = current["block"], current["pad"]
        print(f"Running {current_building_block} ...")
        result = current_building_block.run(experiment_filemap, config, pad=current_pad)
    else:
        print("End of the pipeline!")


if __name__ == "__main__":
    main()
