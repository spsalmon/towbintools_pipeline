from towbintools.foundation import image_handling, detect_molts
import argparse
import numpy as np
from tifffile import imwrite
import os
from joblib import Parallel, delayed
import re
import pandas as pd
import yaml
import utils

def run_detect_molts(analysis_filemap, point, molt_size_range=[6.6e4, 15e4, 36e4, 102e4], search_width=20, fit_width=5):
    data_of_point = analysis_filemap[analysis_filemap['Point'] == point]
    data_of_point = data_of_point.sort_values(by=['Time'])
    volumes = data_of_point['Volume'].values
    worm_types = data_of_point['WormType'].values

    try:
        # Detect molts
        ecdysis, volume_at_ecdysis = detect_molts.find_molts(
            volumes, worm_types, molt_size_range, search_width, fit_width)
    except ValueError:
        # No molt detected
        ecdysis = {'hatch_time': np.nan, 'M1': np.nan, 'M2': np.nan, 'M3': np.nan, 'M4': np.nan}
        volume_at_ecdysis = {'volume_at_hatch': np.nan, 'volume_at_M1': np.nan, 'volume_at_M2': np.nan, 'volume_at_M3': np.nan, 'volume_at_M4': np.nan}

    return {'Time': 0, 'Point': point, 'HatchTime': ecdysis['hatch_time'], 'VolumeAtHatch': volume_at_ecdysis['volume_at_hatch'], 'M1': ecdysis['M1'], "VolumeAtM1": volume_at_ecdysis['volume_at_M1'], 'M2': ecdysis['M2'], "VolumeAtM2": volume_at_ecdysis['volume_at_M2'],
            'M3': ecdysis['M3'], "VolumeAtM3": volume_at_ecdysis['volume_at_M3'], 'M4': ecdysis['M4'], "VolumeAtM4": volume_at_ecdysis['volume_at_M4']}

def main(input_dataframe_path, output_file, config_file, n_jobs):
    """Main function."""
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    analysis_filemap = pd.read_pickle(input_dataframe_path)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    molts_and_volume = Parallel(n_jobs=n_jobs)(delayed(run_detect_molts)(
        analysis_filemap, point) for point in analysis_filemap['Point'].unique())
    molts_dataframe = pd.DataFrame(molts_and_volume)
    molts_dataframe.to_csv(output_file, index=False)


if __name__ == '__main__':
    args = utils.basic_get_args()
    main(args.input, args.output, args.config_file, args.n_jobs)
