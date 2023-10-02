from towbintools.foundation import image_handling
from towbintools.segmentation import segmentation_tools
import argparse
import yaml
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import utils

def segment_and_save(image_path, output_path, method, augment_contrast=False, channels=[], pixelsize=None, edge_based_backbone="skimage", sigma_canny=1,):
    """Segment image and save to output_path."""
    print(image_path)
    image = image_handling.read_tiff_file(
        image_path, channels_to_keep=channels)

    if method == "edge_based":
        if augment_contrast:
            image = image_handling.augment_contrast(image)
        else:
            image = image_handling.normalize_image(image)
        mask = segmentation_tools.edge_based_segmentation(
            image, pixelsize, edge_based_backbone, sigma_canny)
    else:
        raise ValueError(
            "Invalid segmentation method. Use 'edge_based' or 'threshold_based'.")
    imwrite(output_path, mask.astype(np.uint8), compression="zlib")

def main(input_pickle, output_pickle, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]
    print(config)
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)


    Parallel(n_jobs=n_jobs)(delayed(segment_and_save)(input_file, output_path, method=config['segmentation_method'], augment_contrast=config['augment_contrast'], channels=config[
        'segmentation_channels'], pixelsize=config['pixelsize'], edge_based_backbone=config['segmentation_backbone'], sigma_canny=config['sigma_canny']) for input_file, output_path in zip(input_files, output_files))

if __name__ == '__main__':
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
