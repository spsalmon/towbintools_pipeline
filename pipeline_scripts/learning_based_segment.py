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
from towbintools.deep_learning.utils.augmentation import get_prediction_augmentation_from_model
from towbintools.foundation import image_handling
from towbintools.segmentation import segmentation_tools
from xarray import DataArray
from towbintools.deep_learning.utils.dataset import SegmentationPredictionDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)

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

def reshape_images_to_original_shape(images, original_shapes, padded_or_cropped = "pad"):
    reshaped_images = []
    for image, original_shape in zip(images, original_shapes):
        if padded_or_cropped == "pad":
            reshaped_image = image_handling.crop_to_dim_equally(image, original_shape[-2], original_shape[-1])
        elif padded_or_cropped == "crop":
            reshaped_image = image_handling.pad_to_dim_equally(image, original_shape[-2], original_shape[-1])
        reshaped_images.append(reshaped_image)
    return reshaped_images


def main(input_pickle, output_pickle, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    is_zstack = image_handling.check_if_zstack(input_files[0][0])

    if config["segmentation_method"] == "deep_learning":
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_segmentation_model_from_checkpoint(config["model_path"]).to(device)
        preprocessing_fn = get_prediction_augmentation_from_model(model)
        batch_size = config["batch_size"]
        dataset = SegmentationPredictionDataset(input_files, config['segmentation_channels'], preprocessing_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_jobs//2, pin_memory=True, collate_fn=dataset.collate_fn)

        # Make predictions
        model.eval()

        def save_prediction(prediction, output_path):
            imwrite(output_path, prediction, compression="zlib")

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                image_paths, images, image_shapes = batch
                images = torch.from_numpy(images)
                images = images.to(device)
                predictions = model(images)
                predictions = predictions.cpu().numpy()
                predictions = np.squeeze(predictions) > 0.5
                predictions = predictions.astype(np.uint8)

                # Reshape predictions to original shape
                predictions = reshape_images_to_original_shape(predictions, image_shapes, padded_or_cropped="pad")

                # Save predictions
                Parallel(n_jobs=n_jobs//2, prefer="threads")(
                    delayed(save_prediction)(prediction, output_path) for prediction, output_path in zip(predictions, output_files)
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
