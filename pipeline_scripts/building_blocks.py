import os
from abc import ABC
from abc import abstractmethod

import numpy as np

from pipeline_scripts.utils import add_dir_to_experiment_filemap
from pipeline_scripts.utils import create_linker_command
from pipeline_scripts.utils import filter_files_of_group
from pipeline_scripts.utils import get_input_and_output_files_parallel
from pipeline_scripts.utils import get_output_name
from pipeline_scripts.utils import pickle_objects
from pipeline_scripts.utils import run_command

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
        "molt_detection_method",
        "molt_detection_columns",
        "molt_detection_model_path",
        "molt_detection_volume",  # deprecated, use "molt_detection_columns" instead
        "molt_detection_worm_type",
        "molt_detection_batch_size",
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
    "segmentation": {
        # default options for segmentation, allows user to make their config file shorter
        "rerun_segmentation": [False],
        "segmentation_column": ["raw"],
        "segmentation_name_suffix": [None],
        "gaussian_filter_sigma": [1.0],
        "predict_on_tiles": [False],
        "tiler_config": [None],
        "enforce_n_channels": [None],
        "activation_layer": [None],
        "run_segmentation_on": [None],
        "model_path": [None],
        "batch_size": [1],
    },
    "straightening": {
        "rerun_straightening": [False],
        "keep_biggest_object": [False],
    },
    "morphology_computation": {
        "rerun_morphology_computation": [False],
        "morphological_features": [["volume", "length", "area"]],
    },
    "classification": {
        "rerun_classification": [False],
    },
    "molt_detection": {
        "rerun_molt_detection": [False],
        "molt_detection_method": ["legacy"],
        "molt_detection_model_path": ["./models/molt_detection_model.ckpt"],
        "molt_detection_batch_size": [1],
        "molt_detection_volume": [
            None
        ],  # deprecated, use "molt_detection_columns" instead, this is for backward compatibility
    },
    "fluorescence_quantification": {
        "rerun_fluorescence_quantification": [False],
        "fluorescence_quantification_aggregation": ["median"],
        "fluorescence_background_aggregation": ["median"],
    },
    "custom": {
        "rerun_custom_script": [False],
    },
}


class BuildingBlock(ABC):
    def __init__(
        self, name, options, block_config, return_type, script_path, requires_gpu=False
    ):
        self.name = name
        self.options = options
        self.block_config = block_config
        self.return_type = return_type
        self.script_path = script_path
        self.requires_gpu = requires_gpu

    def __str__(self):
        return f"{self.name}: {self.block_config}"

    @abstractmethod
    def get_output_name(self, config, subdir):
        pass

    @abstractmethod
    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        pass

    def create_command(
        self,
        micromamba_path,
        input_pickle_path,
        output_pickle_path,
        pickled_block_config,
        config,
    ):
        script_path = self.script_path

        if script_path.endswith(".sh"):
            command = f"bash {script_path} -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config}"
        elif script_path.endswith(".py"):
            command = f"{micromamba_path} run -n towbintools python3 {script_path} -i {input_pickle_path} -o {output_pickle_path} -c {pickled_block_config} -j {config['sbatch_cpus']}"
        else:
            raise ValueError(
                f"Script type of {script_path} is not supported. The pipeline only supports bash or python scripts."
            )
        return command

    def run(self, experiment_filemap, config, subdir=None):
        block_config = self.block_config
        micromamba_path = config.get("micromamba_path", "~/.local/bin/micromamba")
        temp_dir = config["temp_dir"]

        if self.return_type == "subdir":
            subdir = self.get_output_name(config, subdir)
            input_files, output_files = self.get_input_and_output_files(
                config, experiment_filemap, subdir
            )

            if len(input_files) != 0:
                (
                    input_pickle_path,
                    output_pickle_path,
                    pickled_block_config,
                ) = pickle_objects(
                    temp_dir,
                    {"path": f"{self.name}_input_files", "obj": input_files},
                    {"path": f"{self.name}_output_files", "obj": output_files},
                    {"path": f"{self.name}_block_config", "obj": block_config},
                )

                command = self.create_command(
                    micromamba_path,
                    input_pickle_path,
                    output_pickle_path,
                    pickled_block_config,
                    config,
                )

                linker_command = create_linker_command(
                    micromamba_path, temp_dir, subdir
                )

                run_command(
                    command,
                    self.name,
                    config,
                    requires_gpu=self.requires_gpu,
                    run_linker=True,
                    linker_command=linker_command,
                )

            else:
                linker_command = create_linker_command(
                    micromamba_path, temp_dir, subdir
                )
                run_command(
                    "# No input files found, skipping this building block.",
                    self.name,
                    config,
                    requires_gpu=False,
                    run_linker=True,
                    linker_command=linker_command,
                )

            return subdir

        elif self.return_type == "csv":
            output_file = self.get_output_name(config, subdir)
            input_files, _ = self.get_input_and_output_files(
                config, experiment_filemap, config["analysis_subdir"]
            )

            rerun = (self.block_config[f"rerun_{self.name}"]) or (
                os.path.exists(output_file) is False
            )

            if len(input_files) != 0 and rerun:
                input_pickle_path, pickled_block_config = pickle_objects(
                    temp_dir,
                    {"path": "input_files", "obj": input_files},
                    {"path": "block_config", "obj": block_config},
                )

                command = self.create_command(
                    micromamba_path,
                    input_pickle_path,
                    output_file,
                    pickled_block_config,
                    config,
                )

                linker_command = create_linker_command(
                    micromamba_path, temp_dir, output_file
                )

                run_command(
                    command,
                    self.name,
                    config,
                    requires_gpu=self.requires_gpu,
                    run_linker=True,
                    linker_command=linker_command,
                )

            else:
                linker_command = create_linker_command(
                    micromamba_path, temp_dir, output_file
                )
                run_command(
                    "# No input files found, skipping this building block.",
                    self.name,
                    config,
                    requires_gpu=False,
                    run_linker=True,
                    linker_command=linker_command,
                )

            return output_file


class SegmentationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        NON_LEARNING_METHODS = ["double_threshold", "edge_based", "threshold"]
        LEARNING_BASED_METHODS = ["deep_learning", "ilastik"]

        if block_config["segmentation_method"] in NON_LEARNING_METHODS:
            requires_gpu = False
            script_path = "./pipeline_scripts/non_learning_segment.py"
        elif block_config["segmentation_method"] in LEARNING_BASED_METHODS:
            requires_gpu = True
            script_path = "./pipeline_scripts/learning_based_segment.py"
        else:
            raise ValueError(
                f"Segmentation method {block_config['segmentation_method']} not supported."
            )

        super().__init__(
            "segmentation",
            OPTIONS_MAP["segmentation"],
            block_config,
            "subdir",
            script_path,
            requires_gpu,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["segmentation_column"],
            "seg",
            channels=self.block_config["segmentation_channels"],
            subdir=subdir,
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

        input_files = filter_files_of_group(
            input_files, config, self.block_config["run_segmentation_on"]
        )
        output_files = filter_files_of_group(
            output_files, config, self.block_config["run_segmentation_on"]
        )

        return input_files, output_files


class StraighteningBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = "./pipeline_scripts/straighten.py"
        super().__init__(
            "straightening",
            OPTIONS_MAP["straightening"],
            block_config,
            "subdir",
            script_path,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["straightening_source"][0],
            "str",
            subdir=subdir,
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
                    column_subdir = os.path.join(config["experiment_dir"], column)
                    experiment_filemap = add_dir_to_experiment_filemap(
                        experiment_filemap, column_subdir, column
                    )
                    experiment_filemap.to_csv(config["filemap_path"], index=False)
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


class MorphologyComputationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = "./pipeline_scripts/compute_morphology.py"
        super().__init__(
            "morphology_computation",
            OPTIONS_MAP["morphology_computation"],
            block_config,
            "csv",
            script_path,
        )

    def get_output_name(self, config, subdir):
        return get_output_name(
            config,
            self.block_config["morphology_computation_masks"],
            "morphology",
            subdir=subdir,
            return_subdir=False,
        )

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        morphology_computation_masks = [
            self.block_config["morphology_computation_masks"]
        ]
        analysis_subdir = config["analysis_subdir"]

        input_files, _ = get_input_and_output_files_parallel(
            experiment_filemap,
            morphology_computation_masks,
            analysis_subdir,
            rerun=True,
        )

        input_files = [input_file[0] for input_file in input_files]

        return input_files, None


class ClassificationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = "./pipeline_scripts/classify.py"
        super().__init__(
            "classification",
            OPTIONS_MAP["classification"],
            block_config,
            "csv",
            script_path,
        )

    def get_output_name(self, config, subdir):
        model_name = os.path.basename(os.path.normpath(self.block_config["classifier"]))
        model_name = model_name.split("_classifier")[0]
        return get_output_name(
            config,
            self.block_config["classification_source"],
            model_name,
            subdir=subdir,
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


class MoltDetectionBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = "./pipeline_scripts/detect_molts.py"
        super().__init__(
            "molt_detection",
            OPTIONS_MAP["molt_detection"],
            block_config,
            "csv",
            script_path,
        )

    def get_output_name(self, config, subdir):
        return os.path.join(config["report_subdir"], "ecdysis.csv")

    def get_input_and_output_files(self, config, experiment_filemap, subdir):
        return experiment_filemap, None


class FluorescenceQuantificationBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = "./pipeline_scripts/quantify_fluorescence.py"
        super().__init__(
            "fluorescence_quantification",
            OPTIONS_MAP["fluorescence_quantification"],
            block_config,
            "csv",
            script_path,
        )

    def get_output_name(self, config, subdir):
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


class CustomBuildingBlock(BuildingBlock):
    def __init__(self, block_config):
        script_path = block_config["custom_script_path"]
        super().__init__(
            "custom",
            OPTIONS_MAP["custom"],
            block_config,
            block_config["custom_script_return_type"],
            script_path,
        )

    def get_output_name(self, config, subdir):
        custom_script_name = self.block_config["custom_script_name"]
        analysis_subdir = config["analysis_subdir"]
        report_subdir = config["report_subdir"]

        if self.return_type == "subdir":
            output = os.path.join(analysis_subdir, custom_script_name)
        if subdir is not None:
            output = os.path.join(output, subdir)

        elif self.return_type == "csv":
            if subdir is not None:
                output = os.path.join(
                    report_subdir, f"{subdir}_{custom_script_name}.csv"
                )
            else:
                output = os.path.join(
                    report_subdir, f"{subdir}_{custom_script_name}.csv"
                )

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
                        config_copy[option] = DEFAULT_OPTIONS[building_block_name][
                            option
                        ]
                        print(
                            f"{option} not found in config file, using default value: {config_copy[option]}"
                        )
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
        elif (
            block_config["name"] == "morphology_computation"
            or block_config["name"] == "volume_computation"
        ):  # volume_computation is there for backward compatibility
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
