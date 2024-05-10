import numpy as np


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
        "segmentation": [
            "rerun_segmentation",
            "segmentation_column",
            "segmentation_name_suffix",
            "segmentation_method",
            "segmentation_channels",
            "augment_contrast",
            "pixelsize",
            "sigma_canny",
            "model_path",
            "tiler_config",
            "RGB",
            "activation_layer",
            "batch_size",
            "ilastik_project_path",
            "ilastik_result_channel",
        ],
        "straightening": [
            "rerun_straightening",
            "straightening_source",
            "straightening_masks",
        ],
        "volume_computation": [
            "rerun_volume_computation",
            "volume_computation_masks",
            "pixelsize",
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
            "fluorescence_quantification_normalization",
            "pixelsize",
        ],
        "custom": [
            "rerun_custom_script",
            "custom_script_path",
            "custom_script_name",
            "custom_script_return_type",
            "custom_script_parameters",
        ],
    }

    blocks_config = {}

    for i, building_block in enumerate(building_blocks):
        config_copy = config.copy()
        if building_block in options_map:
            options = options_map[building_block]
            # assert options
            for option in options:
                assert (
                    len(config[option]) == len(building_block_counts[building_block])
                    or len(config[option]) == 1
                ), f"{config[option]} The number of {option} options ({len(config[option])}) does not match the number of {building_block} building blocks ({len(building_block_counts[building_block])})"

            # expand single options to match the number of blocks
            for option in options:
                if len(config_copy[option]) == 1:
                    config_copy[option] = config[option] * len(
                        building_block_counts[building_block]
                    )

            # find the index of the building block
            idx = np.argwhere(
                np.array(building_block_counts[building_block]) == i
            ).squeeze()

            # set the options for the building block
            block_options = {}
            for option in options:
                block_options[option] = config_copy[option][idx]

            blocks_config[i] = block_options

    return blocks_config
