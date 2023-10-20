from towbintools.foundation import image_handling
from towbintools.segmentation import segmentation_tools
from towbintools.foundation.image_handling import get_image_size_metadata
import argparse
import yaml
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import utils

def segment_and_save(image_path, output_path, method, augment_contrast=False, clip_limit=5, channels=[], pixelsize=None, edge_based_backbone="skimage", sigma_canny=1, is_zstack=False):
    """Segment image and save to output_path."""
    if method != "edge_based":
        raise ValueError("Invalid segmentation method. Use 'edge_based'.")

    image = image_handling.read_tiff_file(image_path, channels_to_keep=channels)
    if is_zstack:
        mask = segment_zstack(image, augment_contrast, clip_limit, pixelsize, edge_based_backbone, sigma_canny)
    else:
        mask = segment_single_image(image, augment_contrast, clip_limit, pixelsize, edge_based_backbone, sigma_canny)

    imwrite(output_path, mask.astype(np.uint8), compression="zlib")

def segment_zstack(zstack, augment_contrast, clip_limit, pixelsize, edge_based_backbone, sigma_canny):
    """Segment each plane of a zstack."""
    masks = np.zeros(zstack.shape, dtype=np.uint8)
    for i, plane in enumerate(zstack):
        masks[i] = segment_single_image(plane, augment_contrast, clip_limit, pixelsize, edge_based_backbone, sigma_canny)
    return masks

def segment_single_image(image, augment_contrast, clip_limit, pixelsize, edge_based_backbone, sigma_canny):
    """Segment a single image."""
    if augment_contrast:
        image = image_handling.augment_contrast(image, clip_limit=clip_limit)
    else:
        image = image_handling.normalize_image(image)
    return segmentation_tools.edge_based_segmentation(image, pixelsize, edge_based_backbone, sigma_canny)


def main(input_pickle, output_pickle, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]
    print(config)
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    
    size_metadata = get_image_size_metadata(input_files[0])
    if size_metadata is not None:
        is_zstack = size_metadata['z_dim']>1
    else:
        is_zstack = False

    Parallel(n_jobs=n_jobs)(delayed(segment_and_save)(input_file, output_path, method=config['segmentation_method'], augment_contrast=config['augment_contrast'], channels=config[
        'segmentation_channels'], pixelsize=config['pixelsize'], edge_based_backbone=config['segmentation_backbone'], sigma_canny=config['sigma_canny'], is_zstack=is_zstack) for input_file, output_path in zip(input_files, output_files))

if __name__ == '__main__':
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
