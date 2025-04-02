import logging
import os

import numpy as np
import utils
from joblib import delayed
from joblib import Parallel
from joblib import parallel_config
from tifffile import imwrite
from towbintools.foundation.image_handling import check_if_zstack
from towbintools.foundation.image_handling import read_tiff_file
from towbintools.segmentation.segmentation_tools import segment_image

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
        image = read_tiff_file(image_path, channels_to_keep=channels).squeeze()

        mask = segment_image(
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

    is_zstack = check_if_zstack(input_files[0][0])

    with parallel_config(backend="loky", n_jobs=n_jobs):
        Parallel()(
            delayed(segment_and_save)(
                input_file,
                output_path,
                method=config["segmentation_method"],
                channels=config["segmentation_channels"],
                is_zstack=is_zstack,
                pixelsize=config["pixelsize"],
                gaussian_filter_sigma=config["gaussian_filter_sigma"],
            )
            for input_file, output_path in zip(input_files, output_files)
        )


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
