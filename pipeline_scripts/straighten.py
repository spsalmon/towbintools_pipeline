import os

import cv2
import numpy as np
import utils
from joblib import Parallel, delayed, parallel_config
from scipy.ndimage import binary_fill_holes
from tifffile import imwrite
from towbintools.foundation import binary_image, image_handling
from towbintools.straightening import Warper


def mask_preprocessing(mask):
    if mask.ndim == 2:
        mask = binary_fill_holes(mask)
        mask = cv2.medianBlur(mask.astype(np.uint8), 5)
        mask = binary_image.get_biggest_object(mask)
        return mask

    # find mean mask size for each plane
    mask = np.array([binary_image.get_biggest_object(m) for m in mask])

    sum_mask = [np.sum(m) for m in mask]
    # remove zeros
    sum_mask = [s for s in sum_mask if s != 0]
    mean_mask_size = np.mean(sum_mask)

    # remove planes with masks that are too small to zero
    for i, m in enumerate(mask):
        if np.sum(m) < mean_mask_size * 0.9:
            mask[i] = np.zeros(m.shape, dtype=np.uint8)

    # remove small holes and median blur
    # mask = np.array([binary_fill_holes(m) for m in mask]).astype(np.uint8)
    mask = np.array([cv2.medianBlur(m, 7) for m in mask])
    return mask

def image_preprocessing(image, keep_biggest_object = False):
    if image.ndim == 2:
        if (np.unique(image).size == 2) and keep_biggest_object:
            image = binary_image.get_biggest_object(image)
            image = binary_fill_holes(image)
        return image

    if (np.unique(image).size == 2) and keep_biggest_object:
        image = np.array([binary_image.get_biggest_object(i) for i in image])
        image = np.array([binary_fill_holes(i) for i in image])
    return image


def straighten_and_save(
    source_image_path,
    source_image_channels,
    mask_path,
    output_path,
    is_zstack=False,
    channel_to_allign=[2],
    keep_biggest_object = False,
):
    """Straighten image and save to output_path."""
    mask = image_handling.read_tiff_file(mask_path)
    mask = mask_preprocessing(mask)
    try:
        image = get_image(
            source_image_path, mask, is_zstack, channel_to_allign, source_image_channels
        )

        image = image_preprocessing(image, keep_biggest_object)

        if is_zstack:
            straightened_image = straighten_zstack_image(image, mask)
        else:
            straightened_image = straighten_2D_image(image, mask)
    except Exception as e:
        print(e)
        straightened_image = np.zeros_like(mask).astype(np.uint8)
        # add empty channel dimension if is_zstack is True
        if is_zstack:
            straightened_image = straightened_image[:, np.newaxis, ...]

    if straightened_image.ndim == 2:
        imwrite(
            output_path,
            straightened_image,
            imagej=True,
            compression="zlib",
            metadata={"axes": "YX"},
        )

    elif straightened_image.ndim == 3:
        imwrite(
            output_path,
            straightened_image,
            imagej=True,
            compression="zlib",
            metadata={"axes": "ZYX"},
        )
    else:
        imwrite(
            output_path,
            straightened_image,
            imagej=True,
            compression="zlib",
            metadata={"axes": "ZCYX"},
        )


def get_image(
    source_image_path, mask, is_zstack, channel_to_allign, source_image_channels
):
    """Get the image to be straightened."""
    if source_image_path is None:
        return mask

    if is_zstack:
        full_image = image_handling.read_tiff_file(source_image_path)
        return {
            "allign": full_image[:, channel_to_allign, ...].squeeze(),
            "straighten": full_image[:, source_image_channels, ...].squeeze(),
        }
    else:
        return image_handling.read_tiff_file(
            source_image_path, channels_to_keep=source_image_channels
        )


def straighten_zstack_image(image, mask):
    """Straighten each plane of a zstack image."""
    if np.unique(image).size == 2:
        interpolation_order = 0
    else:
        interpolation_order = 1
    warper = Warper.from_img(image["allign"], mask)
    if image["straighten"].ndim > 3:
        straightened_channels = [
            warper.warp_3D_img(
                image["straighten"][:, channel, ...],
                interpolation_order=interpolation_order,
                preserve_range=True,
                preserve_dtype=True,
            )
            for channel in range(image["straighten"].shape[1])
        ]
    else:
        straightened_channels = [
            warper.warp_3D_img(
                image["straighten"],
                interpolation_order=interpolation_order,
                preserve_range=True,
                preserve_dtype=True,
            )
        ]
    return np.stack(straightened_channels, axis=1).astype(np.uint16)


def straighten_2D_image(image, mask):
    """Straighten a 2D image."""
    # set interpolation order to 0 if the image is binary
    if np.unique(image).size == 2:
        interpolation_order = 0
    else:
        interpolation_order = 1

    if image.ndim == 3:
        warper = Warper.from_img(image[0], mask)
        straightened_channels = [
            warper.warp_2D_img(
                image[channel, ...],
                0,
                interpolation_order=interpolation_order,
                preserve_range=True,
                preserve_dtype=True,
            )
            for channel in range(image.shape[0])
        ]
        return np.stack(straightened_channels, axis=0).astype(np.uint16)
    else:
        warper = Warper.from_img(image, mask)
        return warper.warp_2D_img(
            image,
            0,
            interpolation_order=interpolation_order,
            preserve_range=True,
            preserve_dtype=True,
        ).astype(np.uint16)


def main(input_pickle, output_pickle, config, n_jobs):
    config = utils.load_pickles(config)[0]

    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    source_files = [f["source_image_path"] for f in input_files]
    mask_files = [f["mask_path"] for f in input_files]
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    is_zstack = image_handling.check_if_zstack(source_files[0])

    keep_biggest_object = config.get("keep_biggest_object", False)
    channel_to_allign = config.get("channel_to_allign", [2])

    with parallel_config(backend="loky", n_jobs=n_jobs):
        Parallel()(
            delayed(straighten_and_save)(
                source_file,
                config["straightening_source"][1],
                mask_file,
                output_path,
                is_zstack=is_zstack,
                channel_to_allign=channel_to_allign,
                keep_biggest_object = keep_biggest_object,
            )
            for source_file, mask_file, output_path in zip(
                source_files, mask_files, output_files
            )
        )


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
