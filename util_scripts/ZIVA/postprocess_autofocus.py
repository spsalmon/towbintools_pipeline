import argparse
import os

import numpy as np
import ome_types
from bioio import BioImage
from bioio_ome_tiff.writers import OmeTiffWriter
from joblib import delayed
from joblib import Parallel
from ome_types.model import Image
from ome_types.model import Pixels
from towbintools.foundation.image_handling import read_tiff_file
from towbintools.foundation.zstack import find_best_plane
from tqdm import tqdm


def extract_best_plane_and_save(
    image_path, output_path, measure, norm_each_plane, contrast_augmentation, channel=0
):
    zstack = read_tiff_file(image_path)
    _, best_plane = find_best_plane(
        zstack,
        measure=measure,
        channel=channel,
        dest_dtype=np.uint16,
        each_plane=norm_each_plane,
        contrast_augmentation=contrast_augmentation,
    )  # change the measuring type here, either type "shannon_entropy", "mean", "normalized_variance"

    img = BioImage(image_path)
    instrument = img.metadata.instruments
    old_pixels = img.metadata.images[0].pixels

    new_pixels = Pixels(
        size_x=old_pixels.size_x,
        size_y=old_pixels.size_y,
        size_c=old_pixels.size_c,
        size_t=old_pixels.size_t,
        size_z=1,
        dimension_order=old_pixels.dimension_order,
        physical_size_x=old_pixels.physical_size_x,
        physical_size_y=old_pixels.physical_size_y,
        id=0,
        type=old_pixels.type,
        channels=old_pixels.channels,
        tiff_data_blocks=[{}],
    )
    image = Image(pixels=new_pixels)
    image.acquisition_date = img.metadata.images[0].acquisition_date
    ome = ome_types.OME(images=[image], instruments=instrument)

    OmeTiffWriter.save(
        best_plane,
        output_path,
        dim_order="ZCYX",
        ome_xml=ome,
        tifffile_kwargs={
            "compression": "zlib",
            "compressionargs": {"level": 8},
        },
    )


def get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="Extract the best plane from a z-stack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add the arguments to the parser
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to the input directory",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    measure = "normalized_variance"
    channel = [1]
    norm_each_plane = False
    contrast_augmentation = False

    os.makedirs(output_dir, exist_ok=True)

    images = sorted([os.path.join(input_dir, x) for x in os.listdir(input_dir)])
    best_plane_indexes = Parallel(n_jobs=-1, prefer="processes")(
        delayed(extract_best_plane_and_save)(
            image,
            os.path.join(output_dir, os.path.basename(image)),
            measure,
            norm_each_plane,
            contrast_augmentation,
            channel,
        )
        for image in tqdm(images)
    )
