import pickle
import argparse
from towbintools.foundation import file_handling as file_handling
import os
import subprocess
import numpy as np
from joblib import Parallel, delayed
import shutil

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

def get_and_create_folders(config):
    experiment_dir = config["experiment_dir"]
    raw_subdir = os.path.join(experiment_dir, "raw")
    analysis_subdir = os.path.join(experiment_dir, "analysis")
    os.makedirs(analysis_subdir, exist_ok=True)
    report_subdir = os.path.join(analysis_subdir, "report")
    os.makedirs(report_subdir, exist_ok=True)

    return experiment_dir, raw_subdir, analysis_subdir, report_subdir


def get_output_name(
    experiment_dir,
    input_name,
    task_name,
    channels=None,
    return_subdir=True,
    add_raw=False,
    suffix=None,
    custom_suffix=None,
):
    analysis_subdir = os.path.join(experiment_dir, "analysis")
    report_subdir = os.path.join(analysis_subdir, "report")

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
        os.remove(filepath)


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
