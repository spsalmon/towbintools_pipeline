from towbintools.foundation import image_handling
from towbintools.segmentation import segmentation_tools
import argparse
import yaml
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import utils
import torch
from towbintools.deep_learning.architectures import models
from towbintools.deep_learning.utils.augmentation import get_prediction_augmentation
from pytorch_toolbelt.inference.tiles import ImageSlicer

def segment_and_save(image_path, output_path, method, augment_contrast=False, clip_limit=5, channels=[], pixelsize=None, sigma_canny=1, preprocessing_fn = None, model=None, tiler=None, RGB=False, activation=None, device=None, batch_size=-1,is_zstack=False):
    """Segment image and save to output_path."""
    image = image_handling.read_tiff_file(image_path, channels_to_keep=channels)
    if augment_contrast:
        image = image_handling.augment_contrast(image, clip_limit=clip_limit)

    mask = segmentation_tools.segment_image(image, method, pixelsize=pixelsize, sigma_canny=sigma_canny, preprocessing_fn=preprocessing_fn, model=model, device=device, tiler=tiler, RGB=RGB, activation=activation, batch_size=batch_size, is_zstack=is_zstack)

    imwrite(output_path, mask.astype(np.uint8), compression="zlib")

def main(input_pickle, output_pickle, config, n_jobs):
    """Main function."""
    config = utils.load_pickles(config)[0]
    print(config)
    input_files, output_files = utils.load_pickles(input_pickle, output_pickle)
    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    
    is_zstack = image_handling.check_if_zstack(input_files[0])
        

    if config['segmentation_method'] == 'edge_based':
        Parallel(n_jobs=n_jobs)(delayed(segment_and_save)(input_file, output_path, method=config['segmentation_method'], augment_contrast=config['augment_contrast'], channels=config[
            'segmentation_channels'], pixelsize=config['pixelsize'], sigma_canny=config['sigma_canny'], is_zstack=is_zstack) for input_file, output_path in zip(input_files, output_files))
        
    elif config['segmentation_method'] == 'deep_learning':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = models.LightningPretrained.load_from_checkpoint(config['model_path'], map_location=device)
            model.eval()
            model.freeze()
        except KeyError:
            raise KeyError('The model path does not correspond to a valid model architecture.')
        
        normalization_type = model.normalization['type']
        normalization_params = model.normalization
        if normalization_type == "percentile":
            preprocessing_fn = get_prediction_augmentation(normalization_type=normalization_type, lo = normalization_params['lo'], hi = normalization_params['hi'])
        elif normalization_type == "mean_std":
            preprocessing_fn = get_prediction_augmentation(normalization_type=normalization_type, mean = normalization_params['mean'], std = normalization_params['std'])
        elif normalization_type == "data_range":
            preprocessing_fn = get_prediction_augmentation(normalization_type=normalization_type)
        else:
            preprocessing_fn = None

        tiler_config = config['tiler_config']
        tile_size = tiler_config[0]
        tile_step = tiler_config[1]

        first_image_shape = image_handling.read_tiff_file(input_files[0], [1]).shape[-2:]
        tiler = ImageSlicer(image_shape=first_image_shape, tile_size=tile_size, tile_step=tile_step, weight='pyramid')


        output = [segment_and_save(input_file, output_path, method=config['segmentation_method'], augment_contrast=config['augment_contrast'], channels=config[
            'segmentation_channels'], model=model, tiler=tiler, RGB=config['RGB'], preprocessing_fn=preprocessing_fn,activation=config['activation_layer'], batch_size=config['batch_size'], is_zstack=is_zstack) for input_file, output_path in zip(input_files, output_files)]
if __name__ == '__main__':
    args = utils.basic_get_args()
    main(args.input, args.output, args.config, args.n_jobs)
