import yaml
import os
import subprocess
import pickle
from towbintools.foundation import file_handling as file_handling
import pandas as pd
from pipeline_scripts.utils import pickle_objects, create_sbatch_file, get_and_create_folders, get_input_and_output_files, add_dir_to_experiment_filemap, create_temp_folders

def run_segmentation(experiment_filemap, segmentation_subdir, config_file, config):

    images_to_segment, segmentation_output_files = get_input_and_output_files(
        experiment_filemap, ['raw'], segmentation_subdir, rerun=config['rerun_segmentation'])
    
    if len(images_to_segment) != 0:
        input_pickle_path, output_pickle_path = pickle_objects({'path': 'input_files', 'obj': images_to_segment}, {
                                                            'path': 'segmentation_output_files', 'obj': segmentation_output_files})

        command = f"python3 ./pipeline_scripts/segment.py -i {input_pickle_path} -o {output_pickle_path} -c {config_file} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "seg", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/seg.sh"])
        os.remove(input_pickle_path)
        os.remove(output_pickle_path)


def run_straightening(experiment_filemap, column_to_straighten, segmentation_subdir, straightening_subdir, config_file, config):
    columns = [column_to_straighten,
               f'analysis/{segmentation_subdir.split("analysis/")[-1]}']
    input_files, straightening_output_files = get_input_and_output_files(
        experiment_filemap, columns, straightening_subdir, rerun=config['rerun_straightening'])

    if len(input_files) != 0:
        input_source_images = [input_file[0] for input_file in input_files]
        input_masks = [input_file[1] for input_file in input_files]
        input_source_images_pickle_path, input_masks_pickle_path, straightening_output_files_pickle_path = pickle_objects({'path': 'input_source_images', 'obj': input_source_images}, {
                                                                                                                        'path': 'input_masks', 'obj': input_masks}, {'path': 'straightening_output_files', 'obj': straightening_output_files})

        command = f"python3 ./pipeline_scripts/straighten.py -i {input_source_images_pickle_path} -m {input_masks_pickle_path} -o {straightening_output_files_pickle_path} -c {config_file} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "str", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/str.sh"])

        os.remove(input_source_images_pickle_path)
        os.remove(input_masks_pickle_path)
        os.remove(straightening_output_files_pickle_path)


def run_compute_volume(experiment_filemap, column, report_subdir, config_file, config):
    output_file = os.path.join(report_subdir, 'volume.csv')
    input_files, _ = get_input_and_output_files(
        experiment_filemap, [column], analysis_subdir, rerun=True, output_report=output_file)

    input_files = [input_file[0] for input_file in input_files]

    if len(input_files) != 0:
        input_files_pickle_path = pickle_objects(
            {'path': 'input_files', 'obj': input_files})[0]

        command = f"python3 ./pipeline_scripts/compute_volume.py -i {input_files_pickle_path} -o {output_file} -c {config_file} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "vol", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/vol.sh"])

        os.remove(input_files_pickle_path)

def run_classify_worm_type(experiment_filemap, column, report_subdir, config_file, config):
    output_file = os.path.join(report_subdir, 'worm_types.csv')
    input_files, _ = get_input_and_output_files(
        experiment_filemap, [column], analysis_subdir, rerun=config['rerun_worm_type_classification'], output_report=output_file)

    input_files = [input_file[0] for input_file in input_files]

    if len(input_files) != 0:

        input_files_pickle_path = pickle_objects(
            {'path': 'input_files', 'obj': input_files})[0]

        command = f"python3 ./pipeline_scripts/classify_worm_type.py -i {input_files_pickle_path} -o {output_file} -c {config_file} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "class", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/class.sh"])

        os.remove(input_files_pickle_path)

def run_detect_molts(experiment_filemap, report_subdir, config_file, config):
    experiment_filemap_pickle_path = pickle_objects({'path': 'experiment_filemap', 'obj': experiment_filemap})[0]
    output_file = os.path.join(report_subdir, 'ecdysis.csv')
    command = f"python3 ./pipeline_scripts/detect_molts.py -i {experiment_filemap_pickle_path} -o {output_file} -c {config_file} -j {config['sbatch_cpus']}"
    create_sbatch_file(
        "molt", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
    subprocess.run(["sbatch", f"./temp_files/batch/molt.sh"])


config_file = "./config.yaml"
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

create_temp_folders()

experiment_dir, raw_subdir, analysis_subdir, report_subdir, segmentation_subdir, straightening_subdir = get_and_create_folders(
    config)

experiment_filemap = file_handling.get_dir_filemap(raw_subdir)
experiment_filemap.rename(columns={'ImagePath': 'raw'}, inplace=True)

print(f'#### Running segmentation ####')
run_segmentation(experiment_filemap, segmentation_subdir,
                 config_file=config_file, config=config)

experiment_filemap = add_dir_to_experiment_filemap(
    experiment_filemap, segmentation_subdir, f'analysis/{segmentation_subdir.split("analysis/")[-1]}')

print(f'#### Running straightening ####')
run_straightening(experiment_filemap, f'analysis/{segmentation_subdir.split("analysis/")[-1]}',
                  segmentation_subdir, straightening_subdir, config_file=config_file, config=config)

experiment_filemap = add_dir_to_experiment_filemap(
    experiment_filemap, straightening_subdir, f'analysis/{straightening_subdir.split("analysis/")[-1]}')
experiment_filemap.to_csv("filemap.csv", index=False)

print(f'#### Running volume computation ####')
run_compute_volume(experiment_filemap,
                   f'analysis/{straightening_subdir.split("analysis/")[-1]}', report_subdir, config_file=config_file, config=config)

volume_csv = pd.read_csv(os.path.join(report_subdir, 'volume.csv'))
experiment_filemap = experiment_filemap.merge(volume_csv, on=['Time', 'Point'], how='left')

print(f'#### Running worm type classification ####')
run_classify_worm_type(experiment_filemap, f'analysis/{straightening_subdir.split("analysis/")[-1]}', report_subdir, config_file=config_file, config=config)
worm_types_csv = pd.read_csv(os.path.join(report_subdir, 'worm_types.csv'))
experiment_filemap = experiment_filemap.merge(worm_types_csv, on=['Time', 'Point'], how='left')

print(f'#### Running molt detection ####')
run_detect_molts(experiment_filemap, report_subdir, config_file=config_file, config=config)
ecdysis_csv = pd.read_csv(os.path.join(report_subdir, 'ecdysis.csv'))
experiment_filemap = experiment_filemap.merge(ecdysis_csv, on=['Time', 'Point'], how='left')

experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
