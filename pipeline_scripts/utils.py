import pickle
import argparse
from towbintools.foundation import file_handling as file_handling
import os
import subprocess
import numpy as np
from joblib import Parallel, delayed
import shutil
import pandas as pd
from towbintools.foundation.image_handling import get_acquisition_date

# ----BOILERPLATE CODE FOR FILE HANDLING----

def backup_file(file_path, destination_dir):
    # Ensure the source file exists
    if not os.path.exists(file_path):
        print(f"Source file does not exist: {file_path}")
        return False

    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        print(f"Destination directory does not exist, attempting to create: {destination_dir}")
        try:
            os.makedirs(destination_dir)
        except OSError as e:
            print(f"Failed to create destination directory: {e}")
            return False

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1]
    destination_file_path = os.path.join(destination_dir, f"{base_name}{file_extension}")

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
    except IOError as e:
        print(f"Failed to backup file: {e}")
        return False

def get_experiment_pads(config):
    experiment_dir = config["experiment_dir"]
    pads = [f for f in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, f)) and "pad" in f]
    return pads

def get_and_create_folders_pad(config, pad):
    experiment_dir = config["experiment_dir"]
    try:
        analysis_dir_name = config["analysis_dir_name"]
    except KeyError:
        analysis_dir_name = "analysis"

    raw_subdir = os.path.join(experiment_dir, "raw", pad)
    analysis_subdir = os.path.join(experiment_dir, analysis_dir_name)
    os.makedirs(analysis_subdir, exist_ok=True)
    report_subdir = os.path.join(analysis_subdir, "report", pad)
    os.makedirs(report_subdir, exist_ok=True)
    sbatch_backup_dir = os.path.join(report_subdir, "sbatch_backup")
    os.makedirs(sbatch_backup_dir, exist_ok=True)

    return experiment_dir, raw_subdir, analysis_subdir, report_subdir, sbatch_backup_dir    

def get_and_create_folders(config):
    experiment_dir = config["experiment_dir"]
    try:
        analysis_dir_name = config["analysis_dir_name"]
    except KeyError:
        analysis_dir_name = "analysis"

    raw_subdir = os.path.join(experiment_dir, "raw")
    analysis_subdir = os.path.join(experiment_dir, analysis_dir_name)
    os.makedirs(analysis_subdir, exist_ok=True)
    report_subdir = os.path.join(analysis_subdir, "report")
    os.makedirs(report_subdir, exist_ok=True)
    sbatch_backup_dir = os.path.join(report_subdir, "sbatch_backup")
    os.makedirs(sbatch_backup_dir, exist_ok=True)

    return experiment_dir, raw_subdir, analysis_subdir, report_subdir, sbatch_backup_dir


def get_output_name(
    config,
    input_name,
    task_name,
    pad=None,
    channels=None,
    return_subdir=True,
    add_raw=False,
    suffix=None,
    custom_suffix=None,
):
    analysis_subdir = config["analysis_subdir"]
    report_subdir = config["report_subdir"]

    output_name = ""
    if channels is not None:
        if type(channels) == list:
            for channel in channels:
                output_name += f"ch{channel+1}_"
        else:
            output_name += f"ch{channels+1}_"
    if input_name != "raw" or add_raw:
        output_name += os.path.basename(os.path.normpath(input_name)) + "_"
    output_name += task_name
    if suffix is not None:
        output_name += f"_{suffix}"
    if custom_suffix is not None:
        output_name += f"_{custom_suffix}"

    if return_subdir:
        output_name = os.path.join(analysis_subdir, output_name)
        if pad is not None:
            output_name = os.path.join(output_name, pad)
        os.makedirs(output_name, exist_ok=True)
    else:
        output_name = os.path.join(report_subdir, f"{output_name}.csv")
    return output_name


def create_temp_folders():
    temp_dir = "./temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "pickles"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "batch"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "sbatch_output"), exist_ok=True)


def process_row_input_output_files(row, columns, output_dir, rerun):
    input_file = [row[column] for column in columns]

    try:
        output_file = os.path.join(output_dir, os.path.basename(row[columns[0]]))
    except Exception as e:
        print(f'Raised exception {e} for row {row}')
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
    results = Parallel(n_jobs=n_jobs)(
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

def get_experiment_time_from_filemap(experiment_filemap):
    print('### Calculating ExperimentTime ###')
    experiment_filemap['date'] = experiment_filemap['raw'].apply(get_acquisition_date)

    # in case all acquisition dates are None, return a nan filled ExperimentTime column
    if experiment_filemap['date'].isnull().all():
        return pd.Series([np.nan] * len(experiment_filemap))
    # grouped by Point value, calculate the time difference between the first time and all other times
    grouped = experiment_filemap.groupby('Point')
    # get the date of the raw where Time is 0
    try:
        first_time = grouped.apply(lambda x: x[x['Time'] == 0].iloc[0]['date'])
    except IndexError:
        print('### Error: Time 0 not found for all points, experiment time will be computed from lowest Time value for each point.###')
        first_time = grouped.apply(lambda x: x[x['Time'] == x['Time'].min()].iloc[0]['date'])

    # iterate over each point and calculate the time difference
    for point in experiment_filemap['Point'].unique():
        print(f'### Processing point {point} ###')
        # Use .loc to ensure you're modifying the original DataFrame
        point_indices = experiment_filemap['Point'] == point
        point_data = experiment_filemap.loc[point_indices]
        experiment_filemap.loc[point_indices, 'ExperimentTime'] = (point_data['date'] - first_time[point]).dt.total_seconds()
    
    # keep only the ExperimentTime column
    return experiment_filemap['ExperimentTime']

def get_experiment_time_from_filemap_parallel(experiment_filemap):
    print('### Calculating ExperimentTime ###')
    # copy the filemap to avoid modifying the original
    experiment_filemap = experiment_filemap.copy()
    date_result = Parallel(n_jobs=-1)(delayed(get_acquisition_date)(raw) for raw in experiment_filemap['raw'])
    experiment_filemap['date'] = date_result
    # in case all acquisition dates are None, return a None filled ExperimentTime column
    if experiment_filemap['date'].isnull().all():
        return pd.Series([np.nan] * len(experiment_filemap))
    # grouped by Point value, calculate the time difference between the first time and all other times
    grouped = experiment_filemap.groupby('Point')
    # get the date of the raw where Time is 0
    try:
        first_time = grouped.apply(lambda x: x[x['Time'] == 0].iloc[0]['date'])
    except IndexError:
        print('### Error: Time 0 not found for all points, experiment time will be computed from lowest Time value for each point.###')
        first_time = grouped.apply(lambda x: x[x['Time'] == x['Time'].min()].iloc[0]['date'])
    # iterate over each point and calculate the time difference
    experiment_time = Parallel(n_jobs=-1)(
        delayed(calculate_experiment_time)(point, experiment_filemap, first_time)
        for point in experiment_filemap['Point'].unique()
    )
    experiment_time = pd.concat(experiment_time)
    experiment_filemap['ExperimentTime'] = experiment_time
    # keep only the ExperimentTime column
    return experiment_filemap['ExperimentTime']

def calculate_experiment_time(point, experiment_filemap, first_time):
    point_indices = experiment_filemap['Point'] == point
    point_data = experiment_filemap.loc[point_indices]
    try:
        return (point_data['date'] - first_time[point]).dt.total_seconds()
    except KeyError:
        print(f'### Error calculating experiment time for point {point} ###')
        return pd.Series([np.nan] * len(point_data))


# ----BOILERPLATE CODE FOR PICKLING----


def load_pickles(*pickle_paths):
    loaded_pickles = []
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            files = pickle.load(f)
        loaded_pickles.append(files)
    return loaded_pickles


def pickle_objects(*objects):
    pickled_paths = []
    for obj in objects:
        path = obj["path"]
        pickled_path = f"./temp_files/pickles/{path}.pkl"
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


def run_command(command, script_name, config, requires_gpu=False):
    if config["sbatch_gpus"] == 0 or config["sbatch_gpus"] is None:
        sbatch_output_file, sbatch_error_file = create_sbatch_file(
            script_name,
            config["sbatch_cpus"],
            config["sbatch_time"],
            config["sbatch_memory"],
            command,
        )
    elif requires_gpu:
         sbatch_output_file,  sbatch_error_file = create_sbatch_file(
            script_name,
            config["sbatch_cpus"],
            config["sbatch_time"],
            config["sbatch_memory"],
            command,
            gpus=config["sbatch_gpus"],
        )
    else:
         sbatch_output_file,  sbatch_error_file = create_sbatch_file(
            script_name,
            config["sbatch_cpus"],
            config["sbatch_time"],
            config["sbatch_memory"],
            command,
        )
    subprocess.run(["sbatch", f"./temp_files/batch/{script_name}.sh"])
    return  sbatch_output_file,  sbatch_error_file


def create_sbatch_file(job_name, cores, time_limit, memory, command, gpus=0):
    content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./temp_files/sbatch_output/{job_name}.out
#SBATCH -e ./temp_files/sbatch_output/{job_name}.err
#SBATCH -c {cores}
#SBATCH --gres=gpu:{gpus}
#SBATCH -t {time_limit}
#SBATCH --mem={memory}
#SBATCH --wait

{command}
"""

    with open(f"./temp_files/batch/{job_name}.sh", "w") as file:
        file.write(content)
    
    sbatch_output_file = f"./temp_files/sbatch_output/{job_name}.out"
    sbatch_error_file = f"./temp_files/sbatch_output/{job_name}.err"
    return sbatch_output_file, sbatch_error_file


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