from towbintools.straightening import Warper
from towbintools.foundation import image_handling, binary_image

import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
from scipy.ndimage import binary_fill_holes
import argparse
import yaml
import utils

def straighten_and_save(source_image_path, source_image_channels, mask_path, output_path, is_zstack=False, channel_to_allign=None):
    """Straighten image and save to output_path."""
    mask = image_handling.read_tiff_file(mask_path)
    image = get_image(source_image_path, mask, is_zstack, channel_to_allign, source_image_channels)
    
    try:
        if is_zstack:
            straightened_image = straighten_zstack_image(image, mask)
        else:
            straightened_image = straighten_2D_image(image, mask)
    except:
        straightened_image = np.zeros_like(mask).astype(np.uint8)
    
    imwrite(output_path, straightened_image, compression="zlib")

def get_image(source_image_path, mask, is_zstack, channel_to_allign, source_image_channels):
    """Get the image to be straightened."""
    if source_image_path is None:
        return mask
    
    if is_zstack:
        full_image = image_handling.read_tiff_file(source_image_path)
        return {
            "allign": full_image[:, channel_to_allign, ...],
            "straighten": full_image[:, source_image_channels, ...]
        }
    else:
        return image_handling.read_tiff_file(source_image_path, channels_to_keep=source_image_channels)

def straighten_zstack_image(image, mask):
    """Straighten each plane of a zstack image."""
    warper = Warper.from_img(image["allign"], mask)
    straightened_channels = [warper.warp_3D_img(image["straighten"][:, channel, ...]) for channel in range(image["straighten"].shape[1])]
    return np.stack(straightened_channels, axis=1)

def straighten_2D_image(image, mask):
    """Straighten a 2D image."""
    warper = Warper.from_img(image, mask)
    return warper.warp_2D_img(image)


def main(input_pickle, output_pickle, config, n_jobs):
    config = utils.load_pickles(config)[0]

    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    source_files = [f['source_image_path'] for f in input_files]
    mask_files = [f['mask_path'] for f in input_files]
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    is_zstack = image_handling.check_if_zstack(source_files[0])

    Parallel(n_jobs=n_jobs)(delayed(straighten_and_save)(source_file, config['straightening_source'][1], mask_file, output_path, is_zstack=is_zstack) for source_file, mask_file, output_path in zip(source_files, mask_files, output_files))

if __name__ == '__main__':
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)