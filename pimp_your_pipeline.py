import yaml
import os
import subprocess
import pickle
from towbintools.foundation import file_handling as file_handling
import pandas as pd
from pipeline_scripts.utils import pickle_objects, create_sbatch_file, get_and_create_folders, get_input_and_output_files, add_dir_to_experiment_filemap, create_temp_folders
import numpy as np

def run_segmentation(experiment_filemap, config, block_config):

    # create segmentation subdir
    experiment_dir = config['experiment_dir']
    analysis_subdir = os.path.join(experiment_dir, "analysis")
    segmentation_subdir_name = "ch"
    for channel in block_config['segmentation_channels']:
        segmentation_subdir_name += str(channel+1) + "_"
    segmentation_subdir_name += "seg"
    segmentation_subdir = os.path.join(
        analysis_subdir, segmentation_subdir_name)
    os.makedirs(segmentation_subdir, exist_ok=True)

    images_to_segment, segmentation_output_files = get_input_and_output_files(
        experiment_filemap, block_config['segmentation_column'], segmentation_subdir, rerun=block_config['rerun_segmentation'])
    
    
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

    # create straightening subdir
    if block_config['straightening_source'][0] == 'raw':
        straightening_subdir_name = "ch"
        channel = block_config['straightening_source'][1]
        straightening_subdir_name += str(channel+1) + "_"
        straightening_subdir_name += "raw_str"
    else:
        straightening_subdir_name = os.path.basename(os.path.normpath(block_config['straightening_source'][0]))
        straightening_subdir_name += "_str"
        
    straightening_subdir = os.path.join(config['experiment_dir'], "analysis", straightening_subdir_name)
    os.makedirs(straightening_subdir, exist_ok=True)
    

    columns = [block_config['straightening_source'][0], block_config['straightening_masks'][0]]
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
    os.remove(experiment_filemap_pickle_path)


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

    segmentation_options = ['rerun_segmentation', 'segmentation_column', 'segmentation_method', 'segmentation_channels', 'augment_contrast', 'pixelsize', 'segmentation_backbone', 'sigma_canny']
    straightening_options = ['rerun_straightening', 'straightening_source', 'straightening_masks']
    volume_options = ['rerun_volume_computation']
    worm_type_classification_options = ['rerun_worm_type_classification']
    molt_detection_options = ['rerun_molt_detection']

    blocks_config = {}

    for i, building_block in enumerate(building_blocks):
        if building_block == "segmentation":
            # assert that there are as many segmentation options as there are segmentation building blocks or that there is only one segmentation option
            for option in segmentation_options:
                assert len(config[option]) == len(building_block_counts[building_block]) or len(config[option]) == 1, f'The number of {option} options ({len(config[option])}) does not match the number of segmentation building blocks ({len(building_block_counts[building_block])})'
            for option in segmentation_options:
                if len(config[option]) == 1:
                    config[option] = config[option] * len(building_block_counts[building_block])

            # find the index of the segmentation building block
            idx = np.argwhere(np.array(building_block_counts[building_block]) == i).squeeze()
            
            # set the options for the segmentation building block
            options = {}
            for option in segmentation_options:
                options[option] = config[option][idx]
            
            blocks_config[i] = options
        
        elif building_block == "straightening":
            # assert that there are as many straightening options as there are straightening building blocks or that there is only one straightening option
            for option in straightening_options:
                assert len(config[option]) == len(building_block_counts[building_block]) or len(config[option]) == 1, f'The number of {option} options ({len(config[option])}) does not match the number of straightening building blocks ({len(building_block_counts[building_block])})'
            for option in straightening_options:
                if len(config[option]) == 1:
                    config[option] = config[option] * len(building_block_counts[building_block])
            
            # find the index of the straightening building block
            idx = np.argwhere(np.array(building_block_counts[building_block]) == i).squeeze()

            # set the options for the straightening building block
            options = {}
            for option in straightening_options:
                options[option] = config[option][idx]

            blocks_config[i] = options

        elif building_block == "volume_computation":
            # assert that there are as many volume computation options as there are volume computation building blocks or that there is only one volume computation option
            for option in volume_options:
                assert len(config[option]) == len(building_block_counts[building_block]) or len(config[option]) == 1, f'The number of {option} options ({len(config[option])}) does not match the number of volume computation building blocks ({len(building_block_counts[building_block])})'
            for option in volume_options:
                if len(config[option]) == 1:
                    config[option] = config[option] * len(building_block_counts[building_block])
            
            # find the index of the volume computation building block
            idx = np.argwhere(np.array(building_block_counts[building_block]) == i).squeeze()

            # set the options for the volume computation building block
            options = {}
            for option in volume_options:
                options[option] = config[option][idx]

            blocks_config[i] = options

        elif building_block == "worm_type_classification":
            # assert that there are as many worm type classification options as there are worm type classification building blocks or that there is only one worm type classification option
            for option in worm_type_classification_options:
                assert len(config[option]) == len(building_block_counts[building_block]) or len(config[option]) == 1, f'The number of {option} options ({len(config[option])}) does not match the number of worm type classification building blocks ({len(building_block_counts[building_block])})'
            for option in worm_type_classification_options:
                if len(config[option]) == 1:
                    config[option] = config[option] * len(building_block_counts[building_block])
            
            # find the index of the worm type classification building block
            idx = np.argwhere(np.array(building_block_counts[building_block]) == i).squeeze()

            # set the options for the worm type classification building block
            options = {}
            for option in worm_type_classification_options:
                options[option] = config[option][idx]
            
            blocks_config[i] = options

        elif building_block == "molt_detection":
            # assert that there are as many molt detection options as there are molt detection building blocks or that there is only one molt detection option
            for option in molt_detection_options:
                assert len(config[option]) == len(building_block_counts[building_block]) or len(config[option]) == 1, f'The number of {option} options ({len(config[option])}) does not match the number of molt detection building blocks ({len(building_block_counts[building_block])})'
            for option in molt_detection_options:
                if len(config[option]) == 1:
                    config[option] = config[option] * len(building_block_counts[building_block])
            
            # find the index of the molt detection building block
            idx = np.argwhere(np.array(building_block_counts[building_block]) == i).squeeze()

            # set the options for the molt detection building block
            options = {}
            for option in molt_detection_options:
                options[option] = config[option][idx]
            
            blocks_config[i] = options

    return blocks_config

blocks_config = build_config_of_building_blocks(building_blocks, config)

for i, building_block in enumerate(building_blocks):
    block_config = blocks_config[i]
    if building_block == "segmentation":
        segmentation_subdir = run_segmentation(experiment_filemap, config, block_config)
        experiment_filemap = add_dir_to_experiment_filemap(
            experiment_filemap, segmentation_subdir, f'analysis/{segmentation_subdir.split("analysis/")[-1]}')
        experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
    elif building_block == "straightening":
        straightening_subdir = run_straightening(experiment_filemap, config, block_config)
        experiment_filemap = add_dir_to_experiment_filemap(
            experiment_filemap, straightening_subdir, f'analysis/{straightening_subdir.split("analysis/")[-1]}')
        experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
    else:
        print("Not implemented yet")



# for building_block in building_blocks:
#     print(f'#### Running {building_block} ####')
#     if building_block == "segmentation":

#         config_of_job = 
#         run_segmentation(experiment_filemap, segmentation_subdir,
#                          config_file=config_file, config=config)
#         experiment_filemap = add_dir_to_experiment_filemap(
#             experiment_filemap, segmentation_subdir, f'analysis/{segmentation_subdir.split("analysis/")[-1]}')
    

# print(f'#### Running segmentation ####')
# run_segmentation(experiment_filemap, segmentation_subdir,
#                  config_file=config_file, config=config)

# experiment_filemap = add_dir_to_experiment_filemap(
#     experiment_filemap, segmentation_subdir, f'analysis/{segmentation_subdir.split("analysis/")[-1]}')

# experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)

# print(f'#### Running straightening ####')
# run_straightening(experiment_filemap, f'analysis/{segmentation_subdir.split("analysis/")[-1]}',
#                   segmentation_subdir, straightening_subdir, config_file=config_file, config=config)

# experiment_filemap = add_dir_to_experiment_filemap(
#     experiment_filemap, straightening_subdir, f'analysis/{straightening_subdir.split("analysis/")[-1]}')

# experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)

# print(f'#### Running volume computation ####')
# run_compute_volume(experiment_filemap,
#                    f'analysis/{straightening_subdir.split("analysis/")[-1]}', report_subdir, config_file=config_file, config=config)

# volume_csv = pd.read_csv(os.path.join(report_subdir, 'volume.csv'))
# experiment_filemap = experiment_filemap.merge(volume_csv, on=['Time', 'Point'], how='left')

# experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)

# print(f'#### Running worm type classification ####')
# run_classify_worm_type(experiment_filemap, f'analysis/{straightening_subdir.split("analysis/")[-1]}', report_subdir, config_file=config_file, config=config)
# worm_types_csv = pd.read_csv(os.path.join(report_subdir, 'worm_types.csv'))
# experiment_filemap = experiment_filemap.merge(worm_types_csv, on=['Time', 'Point'], how='left')

# experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)

# print(f'#### Running molt detection ####')
# run_detect_molts(experiment_filemap, report_subdir, config_file=config_file, config=config)
# ecdysis_csv = pd.read_csv(os.path.join(report_subdir, 'ecdysis.csv'))
# experiment_filemap = experiment_filemap.merge(ecdysis_csv, on=['Time', 'Point'], how='left')

# experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
