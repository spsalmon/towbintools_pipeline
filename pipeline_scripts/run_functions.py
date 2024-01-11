import os
from towbintools.foundation import file_handling as file_handling
from pipeline_scripts.utils import pickle_objects, get_input_and_output_files_parallel, add_dir_to_experiment_filemap, get_output_name, run_command, cleanup_files

def run_segmentation(experiment_filemap, config, block_config):

    # create segmentation subdir
    experiment_dir = config['experiment_dir']

    segmentation_subdir = get_output_name(experiment_dir, block_config['segmentation_column'], 'seg', block_config['segmentation_channels'], return_subdir=True, add_raw = False)

    images_to_segment, segmentation_output_files = get_input_and_output_files_parallel(
        experiment_filemap, [block_config['segmentation_column']], segmentation_subdir, rerun=block_config['rerun_segmentation'])
    
    
    if len(images_to_segment) != 0:
        input_pickle_path, output_pickle_path = pickle_objects({'path': 'input_files', 'obj': images_to_segment}, {
                                                            'path': 'segmentation_output_files', 'obj': segmentation_output_files})
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/segment.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"

        run_command(command, "seg", config)

        cleanup_files(input_pickle_path, output_pickle_path, pickled_block_config)
    
    return segmentation_subdir


def run_straightening(experiment_filemap, config, block_config):    
    straightening_subdir = get_output_name(config['experiment_dir'], block_config['straightening_source'][0], 'str', channels = block_config['straightening_source'][1], return_subdir=True, add_raw = True)

    columns = [block_config['straightening_source'][0], block_config['straightening_masks']]

    for column in columns:
        if column not in experiment_filemap.columns:
            try:
                experiment_dir = config['experiment_dir']
                report_subdir = os.path.join(experiment_dir, "analysis", "report")
                column_subdir = os.path.join(experiment_dir, column)
                experiment_filemap = add_dir_to_experiment_filemap(experiment_filemap, column_subdir, column)
                experiment_filemap.to_csv(os.path.join(report_subdir, 'analysis_filemap.csv'), index=False)
            except Exception as e:
                print(e)
                print(f'Could not find {column} in the experiment_filemap and could not infer the files that it would contain.')
                return straightening_subdir

    input_files, straightening_output_files = get_input_and_output_files_parallel(
        experiment_filemap, columns, straightening_subdir, rerun=block_config['rerun_straightening'])

    if len(input_files) != 0:
        input_files = [{'source_image_path': input_source_image, 'mask_path': input_mask} for input_source_image, input_mask in input_files]

        input_pickle_path, output_pickle_path = pickle_objects({'path': 'input_files', 'obj': input_files}, {'path': 'straightening_output_files', 'obj': straightening_output_files})

        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/straighten.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        run_command(command, "str", config)
        cleanup_files(input_pickle_path, output_pickle_path, pickled_block_config)
    
    return straightening_subdir


def run_compute_volume(experiment_filemap, config, block_config):
    analysis_subdir = os.path.join(config['experiment_dir'], "analysis")

    volume_computation_masks = [block_config['volume_computation_masks']]
    output_file = get_output_name(config['experiment_dir'], volume_computation_masks[0], 'volume', return_subdir=False, add_raw = False)

    rerun = ((block_config['rerun_volume_computation']) or (os.path.exists(output_file) == False))

    input_files, _ = get_input_and_output_files_parallel(
        experiment_filemap, volume_computation_masks, analysis_subdir, rerun=True)

    input_files = [input_file[0] for input_file in input_files]

    if len(input_files) != 0 and rerun:
        input_files_pickle_path = pickle_objects(
            {'path': 'input_files', 'obj': input_files})[0]
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/compute_volume.py -i {input_files_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        run_command(command, "vol", config)
        cleanup_files(input_files_pickle_path, pickled_block_config)
    
    return output_file

def run_classification(experiment_filemap, config, block_config):
    analysis_subdir = os.path.join(config['experiment_dir'], "analysis")

    model_name = os.path.basename(os.path.normpath(block_config['classifier']))
    model_name = model_name.split('_classifier')[0]
    classification_source = [block_config['classification_source']]

    output_file = get_output_name(config['experiment_dir'], classification_source[0], model_name, return_subdir=False, add_raw = False)

    rerun = ((block_config['rerun_classification']) or (os.path.exists(output_file) == False))

    input_files, _ = get_input_and_output_files_parallel(
        experiment_filemap, classification_source, analysis_subdir, rerun=True)

    input_files = [input_file[0] for input_file in input_files]

    if len(input_files) != 0 and rerun:

        input_files_pickle_path = pickle_objects(
            {'path': 'input_files', 'obj': input_files})[0]
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/classify.py -i {input_files_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        run_command(command, "class", config)
        cleanup_files(input_files_pickle_path, pickled_block_config)
    
    return output_file

def run_detect_molts(experiment_filemap, config, block_config):
    report_subdir = os.path.join(config['experiment_dir'], "analysis", "report")

    experiment_filemap_pickle_path = pickle_objects({'path': 'experiment_filemap', 'obj': experiment_filemap})[0]
    output_file = os.path.join(report_subdir, 'ecdysis.csv')

    rerun = ((block_config['rerun_molt_detection']) or (os.path.exists(output_file) == False))

    if rerun:
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/detect_molts.py -i {experiment_filemap_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        run_command(command, "molt", config)
        cleanup_files(experiment_filemap_pickle_path, pickled_block_config)

    return output_file

def run_fluorescence_quantification(experiment_filemap, config, block_config):
    analysis_subdir = os.path.join(config['experiment_dir'], "analysis")

    fluorescence_quantification_source = block_config['fluorescence_quantification_source'][0]
    fluorescence_quantification_channel = block_config['fluorescence_quantification_source'][1]

    fluorescence_quantification_masks_name = block_config['fluorescence_quantification_masks'].split('/')[-1]
    normalization = block_config['fluorescence_quantification_normalization']

    output_name_suffix = f'{normalization}_on_{fluorescence_quantification_masks_name}'

    output_file = get_output_name(config['experiment_dir'], fluorescence_quantification_source,'fluo', channels=fluorescence_quantification_channel,return_subdir=False, add_raw = False, suffix = output_name_suffix)

    columns = [fluorescence_quantification_source, block_config['fluorescence_quantification_masks']]

    input_files, _ = get_input_and_output_files_parallel(experiment_filemap, columns, analysis_subdir, rerun=True)

    rerun = ((block_config['rerun_fluorescence_quantification']) or (os.path.exists(output_file) == False))

    if len(input_files) != 0 and rerun:

        input_files = [{'source_image_path': input_source_image, 'mask_path': input_mask} for input_source_image, input_mask in input_files]
        input_pickle_path = pickle_objects({'path': 'input_files', 'obj': input_files})[0]
        
        pickled_block_config = pickle_objects({'path': 'block_config', 'obj': block_config})[0]

        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/quantify_fluorescence.py -i {input_pickle_path} -o {output_file} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        run_command(command, "fluo", config)
        cleanup_files(input_pickle_path, pickled_block_config)

    return output_file

def run_custom(experiment_filemap, config, block_config):
    pass