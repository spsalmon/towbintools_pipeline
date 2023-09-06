from towbintools.foundation import image_handling, worm_features
from towbintools.classification import classification_tools
import argparse
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import re
import pandas as pd
from .utils import load_pickles, basic_get_args
import xgboost as xgb
import yaml

def classify_worm_type_from_file_path(straightened_mask_path, pixelsize, classifier, classes = ["worm", "egg", "error"]):
    """Compute the volume of a straightened mask."""
    print(straightened_mask_path)
    str_mask = image_handling.read_tiff_file(straightened_mask_path)
    worm_type = classification_tools.classify_worm_type(str_mask, pixelsize, classifier, classes)

    pattern = re.compile(r'Time(\d+)_Point(\d+)')
    match = pattern.search(straightened_mask_path)
    if match:
        time = int(match.group(1))
        point = int(match.group(2))
        return {'Time': time, 'Point': point, 'WormType': worm_type}
    else:
        raise ValueError("Could not extract time and point from file name.")

def main(input_pickle, output_file, config_file, n_jobs):
    """Main function."""
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_files = load_pickles(input_pickle)[0]
    
    classifier_path = config['worm_type_classifier']
    classifier = xgb.XGBClassifier()
    classifier.load_model(classifier_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    worm_types = Parallel(n_jobs=n_jobs)(delayed(classify_worm_type_from_file_path)(input_file, config['pixelsize'], classifier) for input_file in input_files)
    worm_types_dataframe = pd.DataFrame(worm_types)
    worm_types_dataframe.to_csv(output_file, index=False)

if __name__ == '__main__':
    args = basic_get_args()
    main(args.input_pickle, args.output_file, args.config_file, args.n_jobs)