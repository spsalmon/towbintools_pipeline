import os
from abc import ABC, abstractmethod

import numpy as np
from towbintools.foundation import file_handling as file_handling

from pipeline_scripts.utils import (
    add_dir_to_experiment_filemap,
    backup_file,
    cleanup_files,
    get_input_and_output_files_parallel,
    get_output_name,
    pickle_objects,
    run_command,
    filter_files_of_group,
)

OPTIONS_MAP = {
    "segmentation": [
        "rerun_segmentation",
        "segmentation_column",
        "segmentation_name_suffix",
        "segmentation_method",
        "segmentation_channels",
        "pixelsize",
        "gaussian_filter_sigma",
        "model_path",
        "predict_on_tiles",
        "tiler_config",
        "enforce_n_channels",
        "activation_layer",
        "batch_size",
        "ilastik_project_path",
        "ilastik_result_channel",
        "run_segmentation_on",
    ],
    "straightening": [
        "rerun_straightening",
        "straightening_source",
        "straightening_masks",
        "keep_biggest_object",
    ],
    "morphology_computation": [
        "rerun_morphology_computation",
        "morphology_computation_masks",
        "pixelsize",
        "morphological_features",
    ],
    "classification": [
        "rerun_classification",
        "classification_source",
        "classifier",
        "pixelsize",
    ],
    "molt_detection": [
        "rerun_molt_detection",
        "molt_detection_volume",
        "molt_detection_worm_type",
    ],
    "fluorescence_quantification": [
        "rerun_fluorescence_quantification",
        "fluorescence_quantification_source",
        "fluorescence_quantification_masks",
        "fluorescence_quantification_aggregation",
        "fluorescence_background_aggregation",
    ],
    "custom": [
        "rerun_custom_script",
        "custom_script_path",
        "custom_script_name",
        "custom_script_return_type",
        "custom_script_parameters",
    ],
}

DEFAULT_OPTIONS = {
    "segmentation":
        {
            # default options for segmentation, allows user to make their config file shorter
            "rerun_segmentation": [False],
            "segmentation_column": ["raw"],
            "segmentation_name_suffix": [None],
            "gaussian_filter_sigma": [1.0],
            "predict_on_tiles": [False],
            "tiler_config": [None],
            "enforce_n_channels": [None],
            "activation_layer": [None],

            # default options for segmentation that are either almost never used, or allow the user to make their config file shorter
            # if some of those options are missing, some methods will not work (ie. ilastik if ilastik_project_path is missing)
            "run_segmentation_on": [None],
            "ilastik_project_path": [None],
            "ilastik_result_channel": [None],
            "model_path": [None],
            "batch_size": [1],
        },
    "straightening":
        {
            "rerun_straightening": [False],
            "keep_biggest_object": [False],
        },
    "morphology_computation":
        {
            "rerun_morphology_computation": [False],
            "morphological_features": [["volume", "length", "area"]],

        },
    "classification":
        {
            "rerun_classification": [False],
        },
    "molt_detection":
        {
            "rerun_molt_detection": [False],
        },
    "fluorescence_quantification":
        {
            "rerun_fluorescence_quantification": [False],
            "fluorescence_quantification_aggregation": ["median"],
            "fluorescence_background_aggregation": ["median"],
        },
    "custom":
        {
            "rerun_custom_script": [False],
        },
}




class BuildingBlock(ABC):
    def __init__(self, name, options, block_config, return_type):
        self.name = name
        self.options = options
        self.block_config = block_config
        # self.script_path = script_path
        self.return_type = return_type

    def __str__(self):
        return f"{self.name}: {self.block_config}"

    @abstractmethod
    def get_output_name(self, config, pad):
        pass

    @abstractmethod
    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        pass

    @abstractmethod
    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        pass

    @abstractmethod
    def run_command(self, command, name, config):
        pass   

    def run(self, experiment_filemap, config, pad=None):
        block_config = self.block_config
        sbatch_backup_dir = config["sbatch_backup_dir"]

        if self.return_type == "subdir":
            subdir = self.get_output_name(config, pad)
            input_files, output_files = self.get_input_and_output_files(
                config, experiment_filemap, subdir
            )

            if len(input_files) != 0:
                input_pickle_path, output_pickle_path = pickle_objects(
                    {"path": "input_files", "obj": input_files},
                    {"path": f"{self.name}_output_files", "obj": output_files},
                )

                pickled_block_config = pickle_objects(
                    {"path": "block_config", "obj": block_config}
                )[0]

                command = self.create_command(
                    input_pickle_path, output_pickle_path, pickled_block_config, config
                )

                sbatch_output_file, sbatch_error_file = self.run_command(
                    command, self.name, config
                )

                backup_file(sbatch_output_file, sbatch_backup_dir)
                backup_file(sbatch_error_file, sbatch_backup_dir)

                cleanup_files(
                    input_pickle_path, output_pickle_path, pickled_block_config
                )

            return subdir

        elif self.return_type == "csv":
            output_file = self.get_output_name(config, pad)
            input_files, _ = self.get_input_and_output_files(
                config, experiment_filemap, config["analysis_subdir"]
            )

            rerun = (self.block_config[f"rerun_{self.name}"]) or (
                os.path.exists(output_file) is False
            )

            if len(input_files) != 0 and rerun:
                input_pickle_path = pickle_objects(
                    {"path": "input_files", "obj": input_files}
                )[0]

                pickled_block_config = pickle_objects(
                    {"path": "block_config", "obj": block_config}
                )[0]

                command = self.create_command(
                    input_pickle_path, output_file, pickled_block_config, config
                )

                sbatch_output_file, sbatch_error_file = self.run_command(
                    command, self.name, config
                )

                backup_file(sbatch_output_file, sbatch_backup_dir)
                backup_file(sbatch_error_file, sbatch_backup_dir)

                cleanup_files(input_pickle_path, pickled_block_config)

            return output_file


class SegmentationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "segmentation", OPTIONS_MAP["segmentation"], block_config, "subdir"
        )

    def get_output_name(self, config, pad):
        return get_output_name(
            config,
            self.block_config["segmentation_column"],
            "seg",
            channels=self.block_config["segmentation_channels"],
            pad=pad,
            return_subdir=True,
            add_raw=False,
            custom_suffix=self.block_config["segmentation_name_suffix"],
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        input_files, output_files = get_input_and_output_files_parallel(
            experiment_filemap,
            [self.block_config["segmentation_column"]],
            subdir,
            rerun=self.block_config["rerun_segmentation"],
        )

        input_files = filter_files_of_group(input_files, config, self.block_config["run_segmentation_on"])
        output_files = filter_files_of_group(output_files, config, self.block_config["run_segmentation_on"])
        
        return input_files, output_files


    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        NON_LEARNING_METHODS = ["double_threshold", "edge_based"]
        LEARNING_BASED_METHODS = ["deep_learning", "ilastik"]

        if self.block_config["segmentation_method"] in NON_LEARNING_METHODS:
            command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/non_learning_segment.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        elif self.block_config["segmentation_method"] in LEARNING_BASED_METHODS:
            command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/learning_based_segment.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        else:
            raise ValueError(
                f"Segmentation method {self.block_config['segmentation_method']} not supported."
            )
        return command

    def run_command(self, command, name, config):
        if self.block_config["segmentation_method"] == "deep_learning":
            return run_command(command, name, config, requires_gpu=True)
        else:
            return run_command(command, name, config)


class StraighteningBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "straightening", OPTIONS_MAP["straightening"], block_config, "subdir"
        )

    def get_output_name(self, config, pad):
        return get_output_name(
            config,
            self.block_config["straightening_source"][0],
            "str",
            pad=pad,
            channels=self.block_config["straightening_source"][1],
            return_subdir=True,
            add_raw=True,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        block_config = self.block_config
        columns = [
            block_config["straightening_source"][0],
            block_config["straightening_masks"],
        ]

        for column in columns:
            if column not in experiment_filemap.columns:
                try:
                    report_subdir = config["report_subdir"]
                    column_subdir = os.path.join(config['experiment_dir'], column)
                    experiment_filemap = add_dir_to_experiment_filemap(
                        experiment_filemap, column_subdir, column
                    )
                    experiment_filemap.to_csv(
                        os.path.join(report_subdir, "analysis_filemap.csv"), index=False
                    )
                except Exception as e:
                    print(e)
                    print(
                        f"Could not find {column} in the experiment_filemap and could not infer the files that it would contain."
                    )
                    return subdir

        input_files, straightening_output_files = get_input_and_output_files_parallel(
            experiment_filemap,
            columns,
            subdir,
            rerun=block_config["rerun_straightening"],
        )

        if len(input_files) != 0:
            input_files = [
                {"source_image_path": input_source_image, "mask_path": input_mask}
                for input_source_image, input_mask in input_files
            ]

        return input_files, straightening_output_files

    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/straighten.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        return command

    def run_command(self, command, name, config):
        return run_command(command, name, config)


class MorphologyComputationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "morphology_computation", OPTIONS_MAP["morphology_computation"], block_config, "csv"
        )

    def get_output_name(self, config, pad):
        return get_output_name(
            config,
            self.block_config["morphology_computation_masks"],
            "morphology",
            pad=pad,
            return_subdir=False,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        morphology_computation_masks = [self.block_config["morphology_computation_masks"]]
        analysis_subdir = config["analysis_subdir"]

        input_files, _ = get_input_and_output_files_parallel(
            experiment_filemap, morphology_computation_masks, analysis_subdir, rerun=True
        )

        input_files = [input_file[0] for input_file in input_files]

        return input_files, None

    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/compute_morphology.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        return command

    def run_command(self, command, name, config):
        return run_command(command, name, config)


class ClassificationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "classification", OPTIONS_MAP["classification"], block_config, "csv"
        )

    def get_output_name(self, config, pad):
        model_name = os.path.basename(os.path.normpath(self.block_config["classifier"]))
        model_name = model_name.split("_classifier")[0]
        return get_output_name(
            config,
            self.block_config["classification_source"],
            model_name,
            pad=pad,
            return_subdir=False,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        classification_source = [self.block_config["classification_source"]]
        analysis_subdir = config["analysis_subdir"]

        input_files, _ = get_input_and_output_files_parallel(
            experiment_filemap, classification_source, analysis_subdir, rerun=True
        )

        input_files = [input_file[0] for input_file in input_files]

        return input_files, None

    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/classify.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        return command

    def run_command(self, command, name, config):
        return run_command(command, name, config)


class MoltDetectionBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "molt_detection", OPTIONS_MAP["molt_detection"], block_config, "csv"
        )

    def get_output_name(self, config, pad):
        return os.path.join(config["report_subdir"], "ecdysis.csv")

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        return experiment_filemap, None

    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/detect_molts.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        return command

    def run_command(self, command, name, config):
        return run_command(command, name, config)


class FluorescenceQuantificationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "fluorescence_quantification",
            OPTIONS_MAP["fluorescence_quantification"],
            block_config,
            "csv",
        )

    def get_output_name(self, config, pad):
        fluorescence_quantification_source = self.block_config[
            "fluorescence_quantification_source"
        ][0]

        fluorescence_quantification_channel = self.block_config[
            "fluorescence_quantification_source"
        ][1]

        fluorescence_quantification_masks_name = self.block_config[
            "fluorescence_quantification_masks"
        ].split("/")[-1]
        aggregation = self.block_config["fluorescence_quantification_aggregation"]

        output_name_suffix = (
            f"{aggregation}_on_{fluorescence_quantification_masks_name}"
        )

        return get_output_name(
            config,
            fluorescence_quantification_source,
            "fluo",
            channels=fluorescence_quantification_channel,
            return_subdir=False,
            add_raw=False,
            suffix=output_name_suffix,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        fluorescence_quantification_source = self.block_config[
            "fluorescence_quantification_source"
        ][0]

        columns = [
            fluorescence_quantification_source,
            self.block_config["fluorescence_quantification_masks"],
        ]

        input_files, _ = get_input_and_output_files_parallel(
            experiment_filemap, columns, subdir, rerun=True
        )

        if len(input_files) != 0:
            input_files = [
                {"source_image_path": input_source_image, "mask_path": input_mask}
                for input_source_image, input_mask in input_files
            ]

        return input_files, None

    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        command = f"~/.local/bin/micromamba run -n towbintools python3 ./pipeline_scripts/quantify_fluorescence.py -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        return command

    def run_command(self, command, name, config):
        return run_command(command, name, config)


class CustomBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        super().__init__(
            "custom",
            OPTIONS_MAP["custom"],
            block_config,
            block_config["custom_script_return_type"],
        )

    def get_output_name(self, config, pad):
        custom_script_name = self.block_config["custom_script_name"]
        analysis_subdir = config["analysis_subdir"]
        report_subdir = config["report_subdir"]

        if self.return_type == "subdir":
            output = os.path.join(analysis_subdir, custom_script_name)
        if pad is not None:
            output = os.path.join(output, pad)

        elif self.return_type == "csv":
            if pad is not None:
                output = os.path.join(report_subdir, f"{pad}_{custom_script_name}.csv")
            else:
                output = os.path.join(report_subdir, f"{pad}_{custom_script_name}.csv")

        return output

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        return experiment_filemap, None

    def create_command(
        self, input_pickle_path, output_pickle_path, pickled_block_config, config
    ):
        custom_script_path = self.block_config["custom_script_path"]

        # concatenate the elements of the custom_script_parameters list
        custom_script_parameters = " ".join(
            self.block_config["custom_script_parameters"]
        )

        if custom_script_path.endswith(".sh"):
            command = f"bash {custom_script_path} -f {input_pickle_path} -o {output_pickle_path} {custom_script_parameters}"
        elif custom_script_path.endswith(".py"):
            command = f"~/.local/bin/micromamba run -n towbintools python3 {custom_script_path} -f {input_pickle_path} -o {output_pickle_path} {custom_script_parameters}"
        else:
            print(
                f"Script type of {custom_script_path} is not supported. The pipeline only supports bash or python scripts."
            )
        return command

    def run_command(self, command, name, config):
        return run_command(command, name, config)


def count_building_blocks_types(building_block_names):
    building_block_counts = {}
    for i, building_block in enumerate(building_block_names):
        if building_block not in building_block_counts:
            building_block_counts[building_block] = []
        building_block_counts[building_block] += [i]
    return building_block_counts


def parse_building_blocks_config(config):
    building_block_names = config["building_blocks"]
    building_block_counts = count_building_blocks_types(building_block_names)

    blocks_config = {}

    for i, building_block_name in enumerate(building_block_names):
        config_copy = config.copy()
        if building_block_name in OPTIONS_MAP:
            options = OPTIONS_MAP[building_block_name]
            # assert options
            for option in options:
                try:
                    assert (
                        len(config[option])
                        == len(building_block_counts[building_block_name])
                        or len(config[option]) == 1
                    ), f"{config[option]} The number of {option} options ({len(config[option])}) does not match the number of {building_block_name} building blocks ({len(building_block_counts[building_block_name])})"

                except KeyError:
                    if option in DEFAULT_OPTIONS[building_block_name]:
                        config_copy[option] = DEFAULT_OPTIONS[building_block_name][option]
                        print(f'{option} not found in config file, using default value: {config_copy[option]}')
                    else:
                        raise KeyError(
                            f"{option} is not in the config file, but is required for the {building_block_name} building block."
                        )
                        
            # expand single options to match the number of blocks
            for option in options:
                if len(config_copy[option]) == 1:
                    config_copy[option] = config_copy[option] * len(
                        building_block_counts[building_block_name]
                    )

            # find the index of the building block
            idx = np.argwhere(
                np.array(building_block_counts[building_block_name]) == i
            ).squeeze()

            # set the options for the building block
            block_options = {}
            for option in options:
                block_options[option] = config_copy[option][idx]

            # add the building block name to the block options
            block_options["name"] = building_block_name

            blocks_config[i] = block_options
        else:
            raise ValueError(f"Unknown building block name : {building_block_name}")

    return blocks_config


def create_building_blocks(blocks_config):
    building_blocks = []
    for i, block_config in blocks_config.items():
        if block_config["name"] == "segmentation":
            building_block = SegmentationBuildingBlock(block_config)
        elif block_config["name"] == "straightening":
            building_block = StraighteningBuildingBlock(block_config)
        elif block_config["name"] == "morphology_computation" or block_config["name"] == "volume_computation": # volume_computation is there for backward compatibility
            building_block = MorphologyComputationBuildingBlock(block_config)
        elif block_config["name"] == "classification":
            building_block = ClassificationBuildingBlock(block_config)
        elif block_config["name"] == "molt_detection":
            building_block = MoltDetectionBuildingBlock(block_config)
        elif block_config["name"] == "fluorescence_quantification":
            building_block = FluorescenceQuantificationBuildingBlock(block_config)
        elif block_config["name"] == "custom":
            building_block = CustomBuildingBlock(block_config)
        else:
            raise ValueError(f"Building block {block_config['name']} not supported.")
        building_blocks.append(building_block)

    return building_blocks


def parse_and_create_building_blocks(config):
    blocks_config = parse_building_blocks_config(config)
    building_blocks = create_building_blocks(blocks_config)
    return building_blocks
