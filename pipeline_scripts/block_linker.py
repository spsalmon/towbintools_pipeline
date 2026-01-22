import argparse
import os

import polars as pl
from towbintools.foundation.file_handling import add_dir_to_experiment_filemap
from towbintools.foundation.file_handling import read_filemap
from towbintools.foundation.file_handling import write_filemap

from pipeline_scripts.utils import cleanup_files
from pipeline_scripts.utils import load_pickles
from pipeline_scripts.utils import merge_and_save_records
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
    experiment_filemap: pl.DataFrame,
    config,
    result,
    previous_block,
    previous_subdir,
    report_subdir,
):
    """Update experiment_filemap based on previous block's return type."""
    filemap_path = config["filemap_path"]
    if previous_block.return_type == "subdir":
        # The column name is always the same regardless of previous_subdir
        column_name = f'{config["analysis_dir_name"]}/{os.path.basename(os.path.normpath(result))}'
        no_timepoint = config.get("no_timepoint", False)
        print(config)
        if no_timepoint or ("Time" not in experiment_filemap.columns):
            experiment_filemap = experiment_filemap.with_columns(
                pl.lit(
                    sorted([os.path.join(result, f) for f in os.listdir(result)])
                ).alias(column_name)
            )
        else:
            experiment_filemap = add_dir_to_experiment_filemap(
                experiment_filemap, result, column_name
            )
        write_filemap(experiment_filemap, filemap_path)
    elif previous_block.return_type == "csv":
        if previous_block.name == "molt_detection":
            experiment_filemap = merge_and_save_records(
                experiment_filemap, filemap_path, result, merge_cols=["Point"]
            )
        else:
            experiment_filemap = merge_and_save_records(
                experiment_filemap, filemap_path, result
            )
    return experiment_filemap


def main():
    args = get_args()
    temp_dir = args.temp_dir
    result = args.result

    # Prepare pickle paths
    pickle_dir = os.path.join(temp_dir, "pickles")
    # config_pickle_path = os.path.join(pickle_dir, "config.pkl")
    progress_pickle_path = os.path.join(pickle_dir, "progress_tracker.pkl")

    # Cleanup unnecessary pickles
    cleanup_temp_pickles(pickle_dir, [progress_pickle_path])

    # Load progress tracker
    progress_tracker = load_pickles(progress_pickle_path)[0]

    current_block_index = progress_tracker["current_block_index"]
    building_blocks = progress_tracker["building_blocks"]

    # Increment progress tracker and save
    progress_tracker["current_block_index"] += 1
    pickle_objects(temp_dir, {"path": "progress_tracker", "obj": progress_tracker})

    # Update experiment_filemap with previous block's result
    if current_block_index > 0:
        previous = building_blocks[current_block_index - 1]
        previous_block, previous_subdir, previous_config = (
            previous["block"],
            previous["subdir"],
            previous["config"],
        )

        report_subdir = previous_config["report_subdir"]
        pipeline_backup_dir = previous_config["pipeline_backup_dir"]

        # Sync backup folder
        sync_backup_folder(previous_config["temp_dir"], pipeline_backup_dir)

        # Load experiment filemap
        experiment_filemap = read_filemap(previous_config["filemap_path"])

        experiment_filemap = update_experiment_filemap(
            experiment_filemap,
            previous_config,
            result,
            previous_block,
            previous_subdir,
            report_subdir,
        )

    # Run current block if available
    if current_block_index < len(building_blocks):
        current = building_blocks[current_block_index]
        current_building_block, current_subdir, current_config = (
            current["block"],
            current["subdir"],
            current["config"],
        )
        experiment_filemap = read_filemap(current_config["filemap_path"])
        print(f"Running {current_building_block} ...")
        result = current_building_block.run(
            experiment_filemap, current_config, subdir=current_subdir
        )
    else:
        print("End of the pipeline!")


if __name__ == "__main__":
    main()
