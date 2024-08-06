import logging
import os

import numpy as np
import torch
import utils
from ilastik.experimental.api import PixelClassificationPipeline
from joblib import Parallel, delayed
from pytorch_toolbelt.inference.tiles import ImageSlicer
from tifffile import imwrite
from towbintools.deep_learning.deep_learning_tools import (
    load_segmentation_model_from_checkpoint,
)
from towbintools.deep_learning.utils.augmentation import get_prediction_augmentation
from towbintools.foundation import image_handling
from towbintools.segmentation import segmentation_tools
from xarray import DataArray

logging.basicConfig(level=logging.INFO)


def segment_and_save(
    image_path,
    output_path,
    method,
    augment_contrast=False,
    clip_limit=5,
    channels=None,
    pixelsize=None,
    sigma_canny=1,
    preprocessing_fn=None,
    model=None,
    tiler=None,
    RGB=False,
    activation=None,
    device=None,
    batch_size=-1,
    is_zstack=False,
):
    """Segment image and save to output_path."""
    try:
        image = image_handling.read_tiff_file(
            image_path, channels_to_keep=channels
        ).squeeze()
        if augment_contrast:
            image = image_handling.augment_contrast(image, clip_limit=clip_limit)

            mask = segmentation_tools.segment_image(
                image,
                method,
                pixelsize=pixelsize,
                sigma_canny=sigma_canny,
                preprocessing_fn=preprocessing_fn,
                model=model,
                device=device,
                tiler=tiler,
                RGB=RGB,
                activation=activation,
                batch_size=batch_size,
                is_zstack=is_zstack,
            )

        imwrite(output_path, mask.astype(np.uint8), compression="zlib", ome=True)
    except Exception as e:
        logging.error(f"Caught exception while segmenting {image_path}: {e}")
        return False


def segment_image_ilastik(image, pipeline, result_channel=0):
    """Segment image using ilastik pipeline."""
    image = DataArray(image, dims=["y", "x"])
    mask = pipeline.get_probabilities(image)[..., result_channel] > 0.5
    return mask


def segment_and_save_ilastik(
    image_path,
    output_path,
    ilastik_project_path,
    augment_contrast=False,
    clip_limit=5,
    channels=None,
    is_zstack=False,
    result_channel=0,
):
    """Segment image using ilastik and save to output_path."""
    pipeline = PixelClassificationPipeline.from_ilp_file(ilastik_project_path)
    try:
        image = image_handling.read_tiff_file(
            image_path, channels_to_keep=channels
        ).squeeze()

        if (image.ndim > 2) and (is_zstack is False):
            raise ValueError(
                "The image is not a z-stack, but has more than 2 dimensions. Ilastik only works on 2D single channel images."
            )

        if augment_contrast:
            image = image_handling.augment_contrast(image, clip_limit=clip_limit)

        if is_zstack:
            mask = np.zeros(image.shape, dtype=np.uint8)
            for i, plane in enumerate(image):
                mask[i] = segment_image_ilastik(plane, pipeline)
        else:
            mask = segment_image_ilastik(image, pipeline, result_channel=result_channel)

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
                method=config["segmentation_method"],
                augment_contrast=config["augment_contrast"],
                channels=config["segmentation_channels"],
                pixelsize=config["pixelsize"],
                sigma_canny=config["sigma_canny"],
                is_zstack=is_zstack,
            )
            for input_file, output_path in zip(input_files, output_files)
        )

    elif config["segmentation_method"] == "double_threshold":
        Parallel(n_jobs=n_jobs)(
            delayed(segment_and_save)(
                input_file,
                output_path,
                method=config["segmentation_method"],
                channels=config["segmentation_channels"],
                is_zstack=is_zstack,
            )
            for input_file, output_path in zip(input_files, output_files)
        )

    elif config["segmentation_method"] == "deep_learning":
        device_count = 1
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                logging.info(f"Using {device_count} GPUs")
                devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
            else:
                devices = [torch.device("cuda:0")]
        else:
            devices = [torch.device("cpu")]

        tiler_config = config["tiler_config"]

        tile_size = tiler_config[0]
        tile_step = tiler_config[1]

        first_image_shape = image_handling.read_tiff_file(input_files[0], [1]).shape[
            -2:
        ]
        tiler = ImageSlicer(
            image_shape=first_image_shape,
            tile_size=tile_size,
            tile_step=tile_step,
            weight="pyramid",
        )

        if len(devices) == 1:
            device = devices[0]
            model = load_segmentation_model_from_checkpoint(config["model_path"]).to(
                device
            )
            model.eval()
            model.freeze()
            # except KeyError:
            #     raise KeyError('The model path does not correspond to a valid model architecture.')

            normalization_type = model.normalization["type"]
            normalization_params = model.normalization
            if normalization_type == "percentile":
                preprocessing_fn = get_prediction_augmentation(
                    normalization_type=normalization_type,
                    lo=normalization_params["lo"],
                    hi=normalization_params["hi"],
                )
            elif normalization_type == "mean_std":
                preprocessing_fn = get_prediction_augmentation(
                    normalization_type=normalization_type,
                    mean=normalization_params["mean"],
                    std=normalization_params["std"],
                )
            elif normalization_type == "data_range":
                preprocessing_fn = get_prediction_augmentation(
                    normalization_type=normalization_type
                )
            else:
                preprocessing_fn = None

            output = [
                segment_and_save(
                    input_file,
                    output_path,
                    method=config["segmentation_method"],
                    augment_contrast=config["augment_contrast"],
                    channels=config["segmentation_channels"],
                    model=model,
                    tiler=tiler,
                    RGB=config["RGB"],
                    preprocessing_fn=preprocessing_fn,
                    activation=config["activation_layer"],
                    device=device,
                    batch_size=config["batch_size"],
                    is_zstack=is_zstack,
                )
                for input_file, output_path in zip(input_files, output_files)
            ]

        else:
            modelz = [
                load_segmentation_model_from_checkpoint(config["model_path"]).to(device)
                for device in devices
            ]
            normalization_type = modelz[0].normalization["type"]
            normalization_params = modelz[0].normalization
            if normalization_type == "percentile":
                preprocessing_fn = get_prediction_augmentation(
                    normalization_type=normalization_type,
                    lo=normalization_params["lo"],
                    hi=normalization_params["hi"],
                )
            elif normalization_type == "mean_std":
                preprocessing_fn = get_prediction_augmentation(
                    normalization_type=normalization_type,
                    mean=normalization_params["mean"],
                    std=normalization_params["std"],
                )
            elif normalization_type == "data_range":
                preprocessing_fn = get_prediction_augmentation(
                    normalization_type=normalization_type
                )
            else:
                preprocessing_fn = None

            for model in modelz:
                model.eval()
                model.freeze()

            def process_on_gpu(input_data, output_path, gpu_id):
                device = devices[gpu_id]
                model = modelz[gpu_id]
                return segment_and_save(
                    input_data,
                    output_path,
                    method=config["segmentation_method"],
                    augment_contrast=config["augment_contrast"],
                    channels=config["segmentation_channels"],
                    model=model,
                    tiler=tiler,
                    RGB=config["RGB"],
                    preprocessing_fn=preprocessing_fn,
                    activation=config["activation_layer"],
                    device=device,
                    batch_size=config["batch_size"],
                    is_zstack=is_zstack,
                )

            Parallel(n_jobs=device_count, backend="threading")(
                delayed(process_on_gpu)(input_file, output_path, i % device_count)
                for i, (input_file, output_path) in enumerate(
                    zip(input_files, output_files)
                )
            )

    elif config["segmentation_method"] == "ilastik":
        Parallel(n_jobs=n_jobs)(
            delayed(segment_and_save_ilastik)(
                input_file,
                output_path,
                ilastik_project_path=config["ilastik_project_path"],
                augment_contrast=config["augment_contrast"],
                channels=config["segmentation_channels"],
                is_zstack=is_zstack,
                result_channel=config["ilastik_result_channel"],
            )
            for input_file, output_path in zip(input_files, output_files)
        )


if __name__ == "__main__":
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
