import pickle
import argparse
from towbintools.foundation import file_handling as file_handling
import os

# ----BOILERPLATE CODE FOR FILE HANDLING----

def get_and_create_folders(config):
    experiment_dir = config['experiment_dir']
    raw_subdir = os.path.join(experiment_dir, "raw")
    analysis_subdir = os.path.join(experiment_dir, "analysis")
    os.makedirs(analysis_subdir, exist_ok=True)
    report_subdir = os.path.join(analysis_subdir, "report")
    os.makedirs(report_subdir, exist_ok=True)

    segmentation_subdir_name = "ch"
    for channel in config['segmentation_channels']:
        segmentation_subdir_name += str(channel+1) + "_"
    segmentation_subdir_name += "seg"
    segmentation_subdir = os.path.join(
        analysis_subdir, segmentation_subdir_name)
    os.makedirs(segmentation_subdir, exist_ok=True)

    straightening_subdir = os.path.join(
        analysis_subdir, f'{segmentation_subdir_name}_str')
    os.makedirs(straightening_subdir, exist_ok=True)

    return experiment_dir, raw_subdir, analysis_subdir, report_subdir, segmentation_subdir, straightening_subdir

def create_temp_folders():
    temp_dir = "./temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "pickles"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "batch"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "sbatch_output"), exist_ok=True)


def get_input_and_output_files(experiment_filemap, columns, output_dir, rerun=True, output_report = None):
    input_files = []
    output_files = []

    for _, row in experiment_filemap.iterrows():
        input_file = [row[column] for column in columns]
        output_file = os.path.join(
            output_dir, os.path.basename(row[columns[0]]))
        if output_report is None:
            if (rerun or not os.path.exists(output_file)) and all([inp is not None for inp in input_file]) :
                input_files.append(input_file)
                output_files.append(output_file)
        else:
            if (rerun or not os.path.exists(output_report)) and all([inp is not None for inp in input_file]):
                input_files.append(input_file)
                output_files.append(output_file)
    return input_files, output_files

def add_dir_to_experiment_filemap(experiment_filemap, dir_path, subdir_name):
    subdir_filemap = file_handling.get_dir_filemap(dir_path)
    subdir_filemap.rename(columns={'ImagePath': subdir_name}, inplace=True)
    experiment_filemap = experiment_filemap.merge(
        subdir_filemap, on=['Time', 'Point'], how='left')
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
        path = obj['path']
        pickled_path = f'./temp_files/pickles/{path}.pkl'
        with open(pickled_path, "wb") as f:
            pickle.dump(obj['obj'], f)

        pickled_paths.append(pickled_path)
    return pickled_paths

# ----BOILERPLATE CODE FOR SLURM----

def create_sbatch_file(job_name, cores, time_limit, memory, command):
    content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -o ./temp_files/sbatch_output/{job_name}.out
#SBATCH -e ./temp_files/sbatch_output/{job_name}.err
#SBATCH -c {cores}
#SBATCH -t {time_limit}
#SBATCH --mem={memory}
#SBATCH --wait

{command}
"""

    with open(f'./temp_files/batch/{job_name}.sh', "w") as file:
        file.write(content)

# ----BOILERPLATE CODE FOR COMMAND LINE INTERFACE----

def basic_get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Segment image and save.')
    parser.add_argument('-i', '--input_pickle', help='Input file paths (saved in a pickle file).')
    parser.add_argument('-o', '--output_file', help='Output file path.')
    parser.add_argument('-c', '--config_file', help='Path to JSON config file.')
    parser.add_argument('-j', '--n_jobs', type=int, help='Number of jobs for parallel execution.')
    
    return parser.parse_args()