from towbintools.foundation import image_handling, worm_features
from towbintools.quantification import fluorescence_in_mask
import argparse
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import re
import pandas as pd
import utils
import yaml

import logging

def quantify_fluorescence_from_file_path(source_image_path, source_image_channel, mask_path, pixelsize, normalization = "mean"):
	"""Quantify the fluorescence of an image inside a mask."""
	source_image = image_handling.read_tiff_file(source_image_path, channels_to_keep=[source_image_channel])
	mask = image_handling.read_tiff_file(mask_path)

	fluo = fluorescence_in_mask(source_image, mask, pixelsize, normalization = normalization)
	pattern = re.compile(r'Time(\d+)_Point(\d+)')
	match = pattern.search(mask_path)
	if match:
		time = int(match.group(1))
		point = int(match.group(2))
		return {'Time': time, 'Point': point, 'Fluo': fluo}
	else:
		raise ValueError("Could not extract time and point from file name.")

def main(input_pickle, output_file, config, n_jobs):
	"""Main function."""
	config = utils.load_pickles(config)[0]

	input_files = utils.load_pickles(input_pickle)[0]
	source_files = [f['source_image_path'] for f in input_files]
	mask_files = [f['mask_path'] for f in input_files]

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	fluo = Parallel(n_jobs=n_jobs)(delayed(quantify_fluorescence_from_file_path)(source_file, config['fluorescence_quantification_source'][1], mask_file, config['pixelsize'], normalization = config['fluorescence_quantification_normalization']) for source_file, mask_file in zip(source_files, mask_files))
	fluo_dataframe = pd.DataFrame(fluo)
	fluo_dataframe.to_csv(output_file, index=False)

if __name__ == '__main__':
	args = utils.basic_get_args()
	main(args.input, args.output, args.config, args.n_jobs)