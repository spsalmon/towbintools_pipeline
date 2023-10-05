import yaml
import os
import subprocess
import pickle
from towbintools.foundation import file_handling as file_handling
import pandas as pd
from pipeline_scripts.utils import pickle_objects, create_sbatch_file, get_and_create_folders, get_input_and_output_files, add_dir_to_experiment_filemap, create_temp_folders
import numpy as np

def get_output_name(experiment_dir, input_name, task_name, channels = None, return_subdir = True, add_raw = False, suffix = None):

    analysis_subdir = os.path.join(experiment_dir, "analysis")
    report_subdir = os.path.join(analysis_subdir, "report")

    output_name = ""
    if channels is not None:
        if type(channels) == list:
            for channel in channels:
                output_name += f'ch{channel+1}_'
        else:
            output_name += f'ch{channels+1}_'
    if input_name != 'raw' or add_raw:
        output_name += os.path.basename(os.path.normpath(input_name)) + "_"
    output_name += task_name
    if suffix is not None:
        output_name += f'_{suffix}'
    
    if return_subdir:
        output_name = os.path.join(analysis_subdir, output_name)
        os.makedirs(output_name, exist_ok=True)
    else:
        output_name = os.path.join(report_subdir, f'{output_name}.csv')
    return output_name

def run_segmentation(experiment_filemap, config, block_config):

    # create segmentation subdir
    experiment_dir = config['experiment_dir']

    segmentation_subdir = get_output_name(experiment_dir, block_config['segmentation_column'], 'seg', block_config['segmentation_channels'], return_subdir=True, add_raw = False)

    images_to_segment, segmentation_output_files = get_input_and_output_files(
        experiment_filemap, [block_config['segmentation_column']], segmentation_subdir, rerun=block_config['rerun_segmentation'])
    
    
    if len(images_to_segment) != 0:
        input_pickle_path, output_pickle_path = pickle_objects({'path': 'input_files', 'obj': images_to_segment}, {
                                                            'path': 'segmentation_output_files', 'obj': segmentation_output_files})
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"python3 ./pipeline_scripts/segment.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "seg", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/seg.sh"])
        os.remove(input_pickle_path)
        os.remove(output_pickle_path)
        os.remove(pickled_block_config)
    
    return segmentation_subdir


def run_straightening(experiment_filemap, config, block_config):
        
    straightening_subdir = get_output_name(config['experiment_dir'], block_config['straightening_source'][0], 'str', channels = block_config['straightening_source'][1], return_subdir=True, add_raw = True)

    columns = [block_config['straightening_source'][0], block_config['straightening_masks']]
    input_files, straightening_output_files = get_input_and_output_files(
        experiment_filemap, columns, straightening_subdir, rerun=block_config['rerun_straightening'])

    if len(input_files) != 0:
        input_source_images = [input_file[0] for input_file in input_files]
        input_masks = [input_file[1] for input_file in input_files]
        input_source_images_pickle_path, input_masks_pickle_path, straightening_output_files_pickle_path = pickle_objects({'path': 'input_source_images', 'obj': input_source_images}, {
                                                                                                                        'path': 'input_masks', 'obj': input_masks}, {'path': 'straightening_output_files', 'obj': straightening_output_files})

        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"python3 ./pipeline_scripts/straighten.py -i {input_source_images_pickle_path} -m {input_masks_pickle_path} -o {straightening_output_files_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "str", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/str.sh"])

        os.remove(input_source_images_pickle_path)
        os.remove(input_masks_pickle_path)
        os.remove(straightening_output_files_pickle_path)
        os.remove(pickled_block_config)
    
    return straightening_subdir


def run_compute_volume(experiment_filemap, config, block_config):
    volume_computation_masks = [block_config['volume_computation_masks']]
    output_file = get_output_name(config['experiment_dir'], volume_computation_masks[0], 'volume', return_subdir=False, add_raw = False)

    rerun = ((block_config['rerun_volume_computation']) or (os.path.exists(output_file) == False))

    input_files, _ = get_input_and_output_files(
        experiment_filemap, volume_computation_masks, analysis_subdir, rerun=True)

    input_files = [input_file[0] for input_file in input_files]

    if len(input_files) != 0 and rerun:
        input_files_pickle_path = pickle_objects(
            {'path': 'input_files', 'obj': input_files})[0]
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"python3 ./pipeline_scripts/compute_volume.py -i {input_files_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "vol", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/vol.sh"])

        os.remove(input_files_pickle_path)
        os.remove(pickled_block_config)
    
    return output_file

def run_classification(experiment_filemap, config, block_config):
    model_name = os.path.basename(os.path.normpath(block_config['classifier']))
    model_name = model_name.split('_classifier')[0]
    classification_source = [block_config['classification_source']]

    output_file = get_output_name(config['experiment_dir'], classification_source[0], model_name, return_subdir=False, add_raw = False)

    rerun = ((block_config['rerun_classification']) or (os.path.exists(output_file) == False))

    input_files, _ = get_input_and_output_files(
        experiment_filemap, classification_source, analysis_subdir, rerun=True)

    input_files = [input_file[0] for input_file in input_files]

    if len(input_files) != 0 and rerun:

        input_files_pickle_path = pickle_objects(
            {'path': 'input_files', 'obj': input_files})[0]
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"python3 ./pipeline_scripts/classify.py -i {input_files_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "class", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/class.sh"])

        os.remove(input_files_pickle_path)
        os.remove(pickled_block_config)
    
    return output_file

def run_detect_molts(experiment_filemap, config, block_config):
    report_subdir = os.path.join(config['experiment_dir'], "analysis", "report")

    experiment_filemap_pickle_path = pickle_objects({'path': 'experiment_filemap', 'obj': experiment_filemap})[0]
    output_file = os.path.join(report_subdir, 'ecdysis.csv')

    rerun = ((block_config['rerun_molt_detection']) or (os.path.exists(output_file) == False))

    if rerun:
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]
        command = f"python3 ./pipeline_scripts/detect_molts.py -i {experiment_filemap_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        create_sbatch_file(
            "molt", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/molt.sh"])
        os.remove(experiment_filemap_pickle_path)
        os.remove(pickled_block_config)

    return output_file

def run_fluorescence_quantification(experiment_filemap, config, block_config):

    fluorescence_quantification_source = block_config['fluorescence_quantification_source'][0]
    fluorescence_quantification_channel = block_config['fluorescence_quantification_source'][1]
    normalization = block_config['fluorescence_quantification_normalization']

    output_file = get_output_name(config['experiment_dir'], fluorescence_quantification_source,'fluo', channels=fluorescence_quantification_channel,return_subdir=False, add_raw = False, suffix = normalization)

    columns = [fluorescence_quantification_source, block_config['fluorescence_quantification_masks']]

    input_files, _ = get_input_and_output_files(experiment_filemap, columns, analysis_subdir, rerun=True)

    rerun = ((block_config['rerun_fluorescence_quantification']) or (os.path.exists(output_file) == False))

    if len(input_files) != 0 and rerun:
        input_source_images = [input_file[0] for input_file in input_files]
        input_masks = [input_file[1] for input_file in input_files]
        input_source_images_pickle_path, input_masks_pickle_path = pickle_objects({'path': 'input_source_images', 'obj': input_source_images}, {'path': 'input_masks', 'obj': input_masks})
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"python3 ./pipeline_scripts/quantify_fluorescence.py -i {input_source_images_pickle_path} -m {input_masks_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"

        create_sbatch_file(
            "fluo", config['sbatch_cpus'], config['sbatch_time'], config['sbatch_memory'], command)
        subprocess.run(["sbatch", f"./temp_files/batch/fluo.sh"])

        os.remove(input_source_images_pickle_path)
        os.remove(input_masks_pickle_path)
        os.remove(pickled_block_config)

    return output_file

def run_custom(experiment_filemap, config, block_config):
    pass

config_file = "./config.yaml"
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

create_temp_folders()

experiment_dir, raw_subdir, analysis_subdir, report_subdir = get_and_create_folders(
    config)

if not os.path.exists(os.path.join(report_subdir, 'analysis_filemap.csv')):
    experiment_filemap = file_handling.get_dir_filemap(raw_subdir)
    experiment_filemap.rename(columns={'ImagePath': 'raw'}, inplace=True)
    experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
else:
    experiment_filemap = pd.read_csv(os.path.join(report_subdir, 'analysis_filemap.csv'))
    experiment_filemap.fillna('', inplace=True)

print(experiment_filemap.head())

building_blocks = config['building_blocks']

def count_building_blocks_types(building_blocks):
    building_block_counts = {}
    for i, building_block in enumerate(building_blocks):
        if building_block not in building_block_counts:
            building_block_counts[building_block] = []
        building_block_counts[building_block] += [i]
    return building_block_counts

def build_config_of_building_blocks(building_blocks, config):
    building_block_counts = count_building_blocks_types(building_blocks)
    
    options_map = {
        "segmentation": ['rerun_segmentation', 'segmentation_column', 'segmentation_method', 'segmentation_channels', 'augment_contrast', 'pixelsize', 'segmentation_backbone', 'sigma_canny'],
        "straightening": ['rerun_straightening', 'straightening_source', 'straightening_masks'],
        "volume_computation": ['rerun_volume_computation', 'volume_computation_masks', 'pixelsize'],
        "classification": ['rerun_classification', 'classification_source', 'classifier', 'pixelsize'],
        "molt_detection": ['rerun_molt_detection', 'molt_detection_volume', 'molt_detection_worm_type'],
        "fluorescence_quantification": ['rerun_fluorescence_quantification', 'fluorescence_quantification_source', 'fluorescence_quantification_masks', 'fluorescence_quantification_normalization', 'pixelsize'],
        "custom" : ['custom_script_path', 'custom_script_parameters']
    }

    blocks_config = {}

    for i, building_block in enumerate(building_blocks):
        if building_block in options_map:
            options = options_map[building_block]
            
            # assert options
            for option in options:
                assert len(config[option]) == len(building_block_counts[building_block]) or len(config[option]) == 1, f'The number of {option} options ({len(config[option])}) does not match the number of {building_block} building blocks ({len(building_block_counts[building_block])})'
            
            # expand single options to match the number of blocks
            for option in options:
                if len(config[option]) == 1:
                    config[option] = config[option] * len(building_block_counts[building_block])
            
            # find the index of the building block
            idx = np.argwhere(np.array(building_block_counts[building_block]) == i).squeeze()
            
            # set the options for the building block
            block_options = {}
            for option in options:
                block_options[option] = config[option][idx]
            
            blocks_config[i] = block_options

    return blocks_config

blocks_config = build_config_of_building_blocks(building_blocks, config)

def process_csv_results(experiment_filemap, report_subdir, csv_file, column_name_old, column_name_new, merge_cols=['Time', 'Point']):
    dataframe = pd.read_csv(csv_file)
    dataframe.rename(columns={column_name_old: column_name_new}, inplace=True)
    if column_name_new in experiment_filemap.columns:
        experiment_filemap.drop(columns=[column_name_new], inplace=True)
    experiment_filemap = experiment_filemap.merge(dataframe, on=merge_cols, how='left')
    experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
    return experiment_filemap

blocks_config = build_config_of_building_blocks(building_blocks, config)

building_block_functions = {
    "segmentation": {"func": run_segmentation, "return_subdir": True},
    "straightening": {"func": run_straightening, "return_subdir": True},
    "volume_computation": {"func": run_compute_volume, "return_subdir": False, "column_name_old": 'Volume', "column_name_new_key": True},
    "classification": {"func": run_classification, "return_subdir": False, "column_name_old": 'WormType', "column_name_new_key": True},
    "molt_detection": {"func": run_detect_molts, "return_subdir":False, "process_molt": True},
    "fluorescence_quantification": {"func": run_fluorescence_quantification, "return_subdir": False, "column_name_old": 'Fluo', "column_name_new_key": True},
    "custom": {"func": run_custom, "return_subdir": False}
}

for i, building_block in enumerate(building_blocks):
    block_config = blocks_config[i]
    if building_block in building_block_functions:
        func_data = building_block_functions[building_block]
        result = func_data["func"](experiment_filemap, config, block_config)
        
        if func_data.get("return_subdir"):
            experiment_filemap = add_dir_to_experiment_filemap(
                experiment_filemap, result, f'analysis/{result.split("analysis/")[-1]}'
            )
            experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
            
        elif func_data.get("column_name_old") is not None:
            column_name_new = os.path.splitext(os.path.basename(result))[0] if func_data.get("column_name_new_key") else func_data.get("column_name_new")
            experiment_filemap = process_csv_results(experiment_filemap, report_subdir, result, func_data["column_name_old"], column_name_new)
            
        elif func_data.get("process_molt"):
            ecdysis_csv = pd.read_csv(os.path.join(report_subdir, 'ecdysis.csv'))
            if 'M1' not in experiment_filemap.columns:
                experiment_filemap = experiment_filemap.merge(ecdysis_csv, on=['Time', 'Point'], how='left')
                experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
        
    else:
        print(f"Functionality for {building_block} not implemented yet")