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

def straighten_and_save(source_image_path, source_image_channels, mask_path, output_path, channel_to_allign=0):
    """Straighten image and save to output_path."""
    mask = image_handling.read_tiff_file(mask_path)
    if source_image_path is None:
        image = mask
    else:
        image = image_handling.read_tiff_file(source_image_path)


        image_to_allign = image[channel_to_allign]

    try:
        mask = binary_image.get_biggest_object(mask)
        warper = Warper.from_img(image, mask)

        straightened_channels = []
        for channel in source_image_channels:
            straightened_channels.append(warper.warp_2D_img(image[channel], 0))

        straightened_image = np.stack(straightened_channels, axis=0)

        if np.max(straightened_image) == 1 and np.min(straightened_image) == 0:
            straightened_image = binary_fill_holes(straightened_image)
            straightened_image = straightened_image.astype(np.uint8)
        else:
            straightened_image = straightened_image.astype(np.uint16)
    except:
        straightened_image = np.zeros_like(mask).astype(np.uint8)
    
    imwrite(output_path, straightened_image, compression="zlib")

def main(input_pickle, output_pickle, config, n_jobs):
    config = utils.load_pickles(config)[0]

    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    source_files = [f['source_image_path'] for f in input_files]
    mask_files = [f['mask_path'] for f in input_files]
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    Parallel(n_jobs=n_jobs)(delayed(straighten_and_save)(source_file, config['straightening_source'][1], mask_file, output_path) for source_file, mask_file, output_path in zip(source_files, mask_files, output_files))

if __name__ == '__main__':
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)