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
	parser.add_argument('-o', '--output', help='Output file paths (saved in a pickle file).')
	parser.add_argument('-c', '--config', help='Pickled config dictionary.')
	parser.add_argument('-j', '--n_jobs', type=int, help='Number of jobs for parallel execution.')
	
	return parser.parse_args()

# ----END BOILERPLATE CODE FOR COMMAND LINE INTERFACE----

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

def main(source_pickle, mask_pickle, output_file, config, n_jobs):
	"""Main function."""
	print(source_pickle)
	print(mask_pickle)
	config = utils.load_pickles(config)[0]

	source_files, mask_files = utils.load_pickles(source_pickle, mask_pickle)

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	fluo = Parallel(n_jobs=n_jobs)(delayed(quantify_fluorescence_from_file_path)(source_file, config['fluorescence_quantification_source'][1], mask_file, config['pixelsize'], normalization = config['fluorescence_quantification_normalization']) for source_file, mask_file in zip(source_files, mask_files))
	fluo_dataframe = pd.DataFrame(fluo)
	fluo_dataframe.to_csv(output_file, index=False)

if __name__ == '__main__':
	args = get_args()
	main(args.source_pickle, args.mask_pickle, args.output, args.config, args.n_jobs)