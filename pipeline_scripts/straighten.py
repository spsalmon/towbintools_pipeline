from worm_tools import straightening
from towbintools.foundation import image_handling, binary_image

import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
from scipy.ndimage import binary_fill_holes
import argparse
import yaml
import utils

# ----BOILERPLATE CODE FOR COMMAND LINE INTERFACE----

def get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Segment image and save.')
    parser.add_argument('-i', '--source_pickle', help='Input source images paths (saved in a pickle file).')
    parser.add_argument('-m', '--mask_pickle', help='Input mask images paths (saved in a pickle file).')
    parser.add_argument('-o', '--output_pickle', help='Output file paths (saved in a pickle file).')
    parser.add_argument('-c', '--config_file', help='Path to JSON config file.')
    parser.add_argument('-j', '--n_jobs', type=int, help='Number of jobs for parallel execution.')
    
    return parser.parse_args()

# ----END BOILERPLATE CODE FOR COMMAND LINE INTERFACE----

def straighten_and_save(source_image_path, mask_path, output_path):
    """Straighten image and save to output_path."""
    mask = image_handling.read_tiff_file(mask_path)
    if source_image_path == mask_path:
        image = mask
    else:
        image = image_handling.read_tiff_file(source_image_path)
    try:
        mask = binary_image.get_biggest_object(mask)
        transformer = straightening.Warper.from_img(image, mask)
        straightened_image = transformer.warp_2D_img(image, 0)
        straightened_image = binary_fill_holes(straightened_image)
    except:
        straightened_image = np.zeros_like(mask)
    
    imwrite(output_path, straightened_image.astype(np.uint8), compression="zlib")

def main(source_pickle, mask_pickle, output_pickle, config_file, n_jobs):
    """Main function."""
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    source_files, mask_files, output_files = utils.load_pickles(source_pickle, mask_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    Parallel(n_jobs=n_jobs)(delayed(straighten_and_save)(source_image, input_mask, output_path) for source_image, input_mask, output_path in zip(source_files, mask_files, output_files))

if __name__ == '__main__':
    args = get_args()
    main(args.source_pickle, args.mask_pickle, args.output_pickle, args.config_file, args.n_jobs)