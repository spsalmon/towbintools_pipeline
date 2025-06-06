import argparse
import os

import numpy as np
from joblib import delayed
from joblib import Parallel
from skimage.morphology import remove_small_objects
from tifffile import imwrite
from towbintools.foundation.image_handling import read_tiff_file


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess masks for deep learning pipeline."
    )
    parser.add_argument(
        "--database_path", type=str, required=True, help="Path to the database."
    )
    return parser.parse_args()


args = get_args()
database_path = args.database_path

img_dir = "images"
mask_dir = "masks"

img_output_dir = "good_images"
output_dir = "binarized_and_cleaned_masks"

img_dir = os.path.join(database_path, img_dir)
mask_dir = os.path.join(database_path, mask_dir)

img_output_dir = os.path.join(database_path, img_output_dir)
output_dir = os.path.join(database_path, output_dir)

os.makedirs(img_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

mask_files = sorted(mask_files)
img_files = sorted(img_files)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_output_dir, exist_ok=True)


def binarize_and_clean_mask(mask_file, img_file, output_dir, img_output_dir):
    mask = read_tiff_file(mask_file)
    mask = mask > 0
    mask = remove_small_objects(mask, min_size=10)
    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        print(f"Empty mask: {mask_file}")
        return

    imwrite(
        os.path.join(output_dir, os.path.basename(mask_file)), mask, compression="zlib"
    )
    imwrite(
        os.path.join(img_output_dir, os.path.basename(img_file)),
        read_tiff_file(img_file),
        compression="zlib",
    )


masks = Parallel(n_jobs=-1)(
    delayed(binarize_and_clean_mask)(mask_file, img_file, output_dir, img_output_dir)
    for mask_file, img_file in zip(mask_files, img_files)
)
