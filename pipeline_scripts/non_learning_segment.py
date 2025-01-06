import logging
import os

import numpy as np
import utils
from joblib import Parallel, delayed
from tifffile import imwrite
from towbintools.foundation import image_handling
from towbintools.segmentation import segmentation_tools

logging.basicConfig(level=logging.INFO)


def segment_and_save(
    image_path,
    output_path,
    method,
    channels=None,
    pixelsize=None,
    is_zstack=False,
    **kwargs,
):
    """Segment image and save to output_path."""
    try:
        image = image_handling.read_tiff_file(
            image_path, channels_to_keep=channels
        ).squeeze()

        mask = segmentation_tools.segment_image(
            image,
            method,
            pixelsize=pixelsize,
            is_zstack=is_zstack,
            **kwargs,
        )

        imwrite(output_path, mask.astype(np.uint8), compression="zlib", ome=True)
    except Exception as e:
        logging.error(f"Caught exception while segmenting {image_path}: {e}")
        return False

def main(input_pickle, output_pickle, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    is_zstack = image_handling.check_if_zstack(input_files[0][0])

    if config["segmentation_method"] == "edge_based":
        Parallel(n_jobs=n_jobs)(
            delayed(segment_and_save)(
                input_file,
                output_path,
                method="edge_based",
                channels=config["segmentation_channels"],
                is_zstack=is_zstack,
                pixelsize=config["pixelsize"],
                gaussian_filter_sigma=config["gaussian_filter_sigma"],
            )
            for input_file, output_path in zip(input_files, output_files)
        )

    elif config["segmentation_method"] == "double_threshold":
        Parallel(n_jobs=n_jobs)(
            delayed(segment_and_save)(
                input_file,
                output_path,
                method="double_threshold",
                channels=config["segmentation_channels"],
                is_zstack=is_zstack,
            )
            for input_file, output_path in zip(input_files, output_files)
        )

if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
