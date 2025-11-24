import argparse
import os
import pickle
import shutil
import subprocess

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from towbintools.foundation import file_handling as file_handling
from towbintools.foundation.image_handling import get_acquisition_date

# ----BOILERPLATE CODE FOR FILE HANDLING----


def backup_file(file_path, destination_dir):
    # Ensure the source file exists
    if not os.path.exists(file_path):
        print(f"Source file does not exist: {file_path}")
        return False

    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        print(
            f"Destination directory does not exist, attempting to create: {destination_dir}"
        )
        try:
            os.makedirs(destination_dir)
        except OSError as e:
            print(f"Failed to create destination directory: {e}")
            return False

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1]
    destination_file_path = os.path.join(
        destination_dir, f"{base_name}{file_extension}"
    )

    # If a file with the same name exists, append an incrementing number
    i = 1
    while os.path.exists(destination_file_path):
        destination_file_path = os.path.join(
            destination_dir, f"{base_name}_{i}{file_extension}"
        )
        i += 1

    # Attempt to copy the file
    try:
        shutil.copyfile(file_path, destination_file_path)
        print(f"File backed up as: {destination_file_path}")
        return True
    except OSError as e:
        print(f"Failed to backup file: {e}")
        return False


def sync_backup_folder(dir, backup_dir):
    """
    Simple synchronization of a directory to a backup directory.
    Only copies files that don't exist or are older in the temp directory.
    """

    # Walk through backup directory
    for root, dirs, files in os.walk(dir):
        # Get the relative path
        rel_path = os.path.relpath(root, dir)
        backup_path = os.path.join(backup_dir, rel_path)

        # Create directory in temp if it doesn't exist
        os.makedirs(backup_path, exist_ok=True)

        # Copy each file if needed
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(backup_path, file)

            # Copy if destination doesn't exist or source is newer
            if not os.path.exists(dst_file) or os.path.getmtime(
                src_file
            ) > os.path.getmtime(dst_file):
                shutil.copy2(src_file, dst_file)


def get_experiment_subdirs(config):
    experiment_dir = config["experiment_dir"]
    raw_dir = config.get("raw_dir_name", "raw")
    raw_subdir = os.path.join(experiment_dir, raw_dir)
    subdirs = [
        f for f in os.listdir(raw_subdir) if os.path.isdir(os.path.join(raw_subdir, f))
    ]
    return sorted(subdirs)


def get_and_create_folders_subdir(config, subdir):
    (
        experiment_dir,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    ) = get_and_create_folders(config)

    raw_subdir = os.path.join(raw_subdir, subdir)
    report_subdir = os.path.join(report_subdir, subdir)
    os.makedirs(report_subdir, exist_ok=True)
    pipeline_backup_dir = os.path.join(pipeline_backup_dir, subdir)
    os.makedirs(pipeline_backup_dir, exist_ok=True)

    return (
        experiment_dir,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    )


def get_and_create_folders(config):
    experiment_dir = config["experiment_dir"]
    analysis_dir_name = config.get("analysis_dir_name", "analysis")
    raw_dir_name = config.get("raw_dir_name", "raw")

    raw_subdir = os.path.join(experiment_dir, raw_dir_name)
    analysis_subdir = os.path.join(experiment_dir, analysis_dir_name)
    os.makedirs(analysis_subdir, exist_ok=True)
    report_subdir = os.path.join(analysis_subdir, "report")
    os.makedirs(report_subdir, exist_ok=True)
    pipeline_backup_dir = os.path.join(report_subdir, "pipeline_backup")
    os.makedirs(pipeline_backup_dir, exist_ok=True)

    return (
        experiment_dir,
        raw_subdir,
        analysis_subdir,
        report_subdir,
        pipeline_backup_dir,
    )


def get_groups(config):
    try:
        return config["groups"]
    except KeyError:
        return None


def get_filter_rule(groups, run_on_option):
    if (groups is not None) or (run_on_option is not None):
        return groups[run_on_option]
    else:
        return None


def filter_files_with_filter_rule(file_groups, filter_rule):
    if filter_rule is not None:
        if isinstance(file_groups[0], str):
            return [
                file_group
                for file_group in file_groups
                if filter_rule.lower() in file_group.lower()
            ]
        else:
            return [
                file_group
                for file_group in file_groups
                if all(filter_rule.lower() in file.lower() for file in file_group)
            ]
    else:
        return file_groups


def filter_files_of_group(files, config, run_on_option):
    groups = get_groups(config)
    filter_rule = get_filter_rule(groups, run_on_option)
    return filter_files_with_filter_rule(files, filter_rule)


def get_output_name(
    config,
    input_name,
    task_name,
    subdir=None,
    channels=None,
    return_subdir=True,
    add_raw=False,
    suffix=None,
    custom_suffix=None,
):
    analysis_subdir = config["analysis_subdir"]
    report_subdir = config["report_subdir"]
    raw_subdir = config["raw_subdir"]

    output_name = ""
    if channels is not None:
        if isinstance(channels, list):
            for channel in channels:
                output_name += f"ch{channel+1}_"
        else:
            output_name += f"ch{channels+1}_"
    if input_name != raw_subdir or add_raw:
        output_name += os.path.basename(os.path.normpath(input_name)) + "_"
    output_name += task_name
    if suffix is not None:
        output_name += f"_{suffix}"
    if custom_suffix is not None:
        output_name += f"_{custom_suffix}"

    if return_subdir:
        output_name = os.path.join(analysis_subdir, output_name)
        if subdir is not None:
            output_name = os.path.join(output_name, subdir)
        os.makedirs(output_name, exist_ok=True)
    else:
        output_name = os.path.join(report_subdir, f"{output_name}.csv")
    return output_name


def create_temp_folders(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "pickles"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "batch"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "sbatch_output"), exist_ok=True)


def process_row_input_output_files(row, columns, output_dir, rerun):
    input_file = [row[column] for column in columns]

    if np.any(pd.isnull(input_file)):
        return None, None

    try:
        output_file = os.path.join(output_dir, os.path.basename(row[columns[0]]))
    except Exception as e:
        print(f"Raised exception {e} for row {row}")
        return None, None

    if (rerun or not os.path.exists(output_file)) and all(
        [(inp is not None) and (inp != "") for inp in input_file]
    ):
        return input_file, output_file
    else:
        return None, None


def get_input_and_output_files_parallel(
    experiment_filemap, columns: list, output_dir: str, rerun=True, n_jobs=-1
):
    with parallel_config(backend="loky", n_jobs=n_jobs):
        results = Parallel()(
            delayed(process_row_input_output_files)(row, columns, output_dir, rerun)
            for _, row in experiment_filemap.iterrows()
        )

    # Filter out None results
    results = [result for result in results if result[0] is not None]

    # Separate input and output files
    input_files, output_files = zip(*results) if results else ([], [])

    return input_files, output_files


def get_input_and_output_files(experiment_filemap, columns, output_dir, rerun=True):
    input_files = []
    output_files = []

    for _, row in experiment_filemap.iterrows():
        input_file = [row[column] for column in columns]
        output_file = os.path.join(output_dir, os.path.basename(row[columns[0]]))
        if (rerun or not os.path.exists(output_file)) and all(
            [((inp is not None) & (inp != "")) for inp in input_file]
        ):
            input_files.append(input_file)
            output_files.append(output_file)
    return input_files, output_files


def add_dir_to_experiment_filemap(experiment_filemap, dir_path, subdir_name):
    subdir_filemap = file_handling.get_dir_filemap(dir_path)
    subdir_filemap.rename(columns={"ImagePath": subdir_name}, inplace=True)
    # check if column already exists
    if subdir_name in experiment_filemap.columns:
        experiment_filemap.drop(columns=[subdir_name], inplace=True)
    experiment_filemap = experiment_filemap.merge(
        subdir_filemap, on=["Time", "Point"], how="left"
    )
    experiment_filemap = experiment_filemap.replace(np.nan, "", regex=True)
    return experiment_filemap


def get_experiment_time_from_filemap(experiment_filemap, config):
    experiment_filemap = experiment_filemap.copy()
    raw_subdir = config["raw_subdir"]

    with parallel_config(backend="multiprocessing", n_jobs=-1):
        date_result = Parallel()(
            delayed(get_acquisition_date)(raw) for raw in experiment_filemap[raw_subdir]
        )
    experiment_filemap["date"] = pd.Series(date_result, index=experiment_filemap.index)

    experiment_filemap["date"] = pd.to_datetime(experiment_filemap["date"], utc=True)

    # ensure no timezone info is present
    if (
        hasattr(experiment_filemap["date"].dtype, "tz")
        and experiment_filemap["date"].dt.tz is not None
    ):
        experiment_filemap["date"] = experiment_filemap["date"].dt.tz_localize(None)

    if experiment_filemap["date"].isnull().all():
        return pd.Series([np.nan] * len(experiment_filemap))

    try:
        first_time = (
            experiment_filemap[experiment_filemap["Time"] == 0]
            .groupby("Point")["date"]
            .first()
        )
    except KeyError:
        print(
            "### Error: Time 0 not found for all points, experiment time will be computed from lowest Time value for each point.###"
        )
        first_time = experiment_filemap.loc[
            experiment_filemap.groupby("Point")["Time"].idxmin()
        ].set_index("Point")["date"]

    if hasattr(first_time.dtype, "tz") and first_time.dt.tz is not None:
        first_time = first_time.dt.tz_localize(None)

    with parallel_config(backend="loky", n_jobs=-1):
        experiment_time = Parallel()(
            delayed(calculate_experiment_time)(point, experiment_filemap, first_time)
            for point in experiment_filemap["Point"].unique()
        )

    experiment_time = pd.concat(experiment_time)
    experiment_filemap["ExperimentTime"] = experiment_time

    return experiment_filemap["ExperimentTime"]


def calculate_experiment_time(point, experiment_filemap, first_time):
    point_indices = experiment_filemap["Point"] == point
    point_data = experiment_filemap.loc[point_indices]
    try:
        dates = pd.to_datetime(
            pd.Series(point_data["date"].values, index=point_data.index)
        )
        first_time_point = pd.to_datetime(first_time[point])
        return round((dates - first_time_point).dt.total_seconds())
    except KeyError:
        print(f"### Error calculating experiment time for point {point} ###")
        return pd.Series([np.nan] * len(point_data))
    except Exception as e:
        print(
            f"### Unexpected error calculating experiment time for point {point}: {e} ###"
        )
        print(f"dates: {dates}, first_time: {first_time[point]}")
        return pd.Series([np.nan] * len(point_data))


# ----BOILERPLATE CODE FOR PICKLING----


def load_pickles(*pickle_paths):
    loaded_pickles = []
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            files = pickle.load(f)
        loaded_pickles.append(files)
    return loaded_pickles


def pickle_objects(temp_dir, *objects):
    pickled_paths = []
    for obj in objects:
        path = obj["path"]
        pickled_path = f"{os.path.join(temp_dir, 'pickles', path)}.pkl"

        if hasattr(obj["obj"], "to_pickle"):
            obj["obj"].to_pickle(pickled_path)
            pickled_paths.append(pickled_path)
        else:
            with open(pickled_path, "wb") as f:
                pickle.dump(obj["obj"], f)
            pickled_paths.append(pickled_path)
    return pickled_paths


def cleanup_files(*filepaths):
    for filepath in filepaths:
        try:
            os.remove(filepath)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except PermissionError:
            print(f"Permission denied: {filepath}")
        except Exception as e:
            print(f"Error deleting file {filepath}: {e}")


# ----BOILERPLATE CODE FOR SLURM----


def create_linker_command(
    micromamba_path,
    temp_dir,
    result,
):
    linker_command = f"{micromamba_path} run -n towbintools python3 -m pipeline_scripts.block_linker --temp_dir {temp_dir} --result {result}"
    return linker_command


def run_command(
    command,
    script_name,
    config,
    run_linker=True,
    linker_command=None,
    requires_gpu=False,
):
    gpus = config.get("sbatch_gpus", None)
    if requires_gpu and gpus is not None:
        script_path = create_sbatch_file(
            script_name,
            config["temp_dir"],
            config["sbatch_cpus"],
            config["sbatch_time"],
            config["sbatch_memory"],
            command,
            gpus=gpus,
            run_linker=run_linker,
            linker_command=linker_command,
        )
    else:
        script_path = create_sbatch_file(
            script_name,
            config["temp_dir"],
            config["sbatch_cpus"],
            config["sbatch_time"],
            config["sbatch_memory"],
            command,
            run_linker=run_linker,
            linker_command=linker_command,
        )
    subprocess.run(["sbatch", script_path])


def create_sbatch_file(
    job_name,
    temp_dir,
    cores,
    time_limit,
    memory,
    command,
    gpus=None,
    run_linker=True,
    linker_command=None,
):
    # Ensure batch directory exists
    batch_dir = os.path.join(temp_dir, "batch")
    os.makedirs(batch_dir, exist_ok=True)

    # Build SLURM header
    content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o {os.path.join(temp_dir, 'sbatch_output', job_name)}-%j.out
#SBATCH -e {os.path.join(temp_dir, 'sbatch_output', job_name)}-%j.err
#SBATCH -c {cores}
#SBATCH -t {time_limit}
#SBATCH --mem={memory}
"""
    if gpus is not None:
        content += f"#SBATCH --gres=gpu:{gpus}\n"

    # set environment variables for single threaded execution (doesn't solve our problem, so I commented it out)
    #     content += """
    # export OMP_NUM_THREADS=1
    # export MKL_NUM_THREADS=1
    # export OPENBLAS_NUM_THREADS=1
    # """
    content += "\n" + command + "\n"

    if run_linker and linker_command is not None:
        content += linker_command + "\n"

    script_path = os.path.join(batch_dir, f"{job_name}.sh")
    with open(script_path, "w") as file:
        file.write(content)

    return script_path


# ----BOILERPLATE CODE FOR COMMAND LINE INTERFACE----


def basic_get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Read args for a piece of the pipeline."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file paths (saved in a pickle file) or single filepath (CSV file for example).",
    )
    parser.add_argument("-o", "--output", help="Output file path or pickle.")
    parser.add_argument("-c", "--config", help="Pickled config dictionary.")
    parser.add_argument(
        "-j", "--n_jobs", type=int, help="Number of jobs for parallel execution."
    )

    return parser.parse_args()


# ----BOILERPLATE CODE FOR SAVING ----


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
