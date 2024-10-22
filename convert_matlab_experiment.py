import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from towbintools.foundation.file_handling import add_dir_to_experiment_filemap
from pipeline_scripts.utils import (
    get_experiment_time_from_filemap_parallel,
)

KEY_CONVERSION_MAP = {
    "vol": "volume",
    "len": "length",
    "strClass": "worm_type",
    'ecdys': 'ecdysis',
}

def convert_matlab_experiment(experiment_dir, matlab_report_dir, matlab_report_files, column_names, add_raw = False, analysis_to_add = None):
    matlab_report_files = [os.path.join(matlab_report_dir, f) for f in matlab_report_files]
    dfs = []
    for matlab_report_file, column_name in zip(matlab_report_files, column_names):
        matlab_report = sio.loadmat(matlab_report_file, chars_as_strings=False)

        # Convert keys
        new_matlab_report = {}
        for key, value in matlab_report.items():
            if key.startswith("__"):
                continue
            try:
                new_key = KEY_CONVERSION_MAP.get(key)
                new_matlab_report[new_key] = value
            except Exception as e:
                continue

        # convert worm_type from [' ', 'w', 'o', 'e'] to ['worm', 'egg', 'error']
        new_matlab_report["worm_type"] = np.where(new_matlab_report["worm_type"] == 'w', "worm", new_matlab_report["worm_type"])
        new_matlab_report["worm_type"] = np.where(new_matlab_report["worm_type"] == 'e', "egg", new_matlab_report["worm_type"])
        new_matlab_report["worm_type"] = np.where(new_matlab_report["worm_type"] == 'o', "error", new_matlab_report["worm_type"])
        new_matlab_report["worm_type"] = np.where(new_matlab_report["worm_type"] == ' ', "error", new_matlab_report["worm_type"])

        ecdysis = new_matlab_report["ecdysis"]
        hatch, M1, M2, M3, M4 = np.split(ecdysis, 5, axis=1)

        point = np.arange(0, hatch.shape[0])
        time = np.arange(0, new_matlab_report["volume"].shape[1])

        # combine each point with every time
        time_point = np.array(np.meshgrid(point, time)).T.reshape(-1, 2)

        time = time_point[:, 1]
        point = time_point[:, 0]

        # put everything into a nice dataframe
        data = {
            "Point": point,
            "Time": time,
            f"{column_name}_volume": new_matlab_report["volume"].flatten(),
            f"{column_name}_length": new_matlab_report["length"].flatten(),
            f"{column_name}_worm_type": new_matlab_report["worm_type"].flatten()
        }

        df = pd.DataFrame(data)

        for point, molts in enumerate(ecdysis):
            HatchTime, M1, M2, M3, M4 = molts
            df.loc[df["Point"] == point, "HatchTime"] = HatchTime - 1
            df.loc[df["Point"] == point, "M1"] = M1 - 1
            df.loc[df["Point"] == point, "M2"] = M2 - 1
            df.loc[df["Point"] == point, "M3"] = M3 - 1
            df.loc[df["Point"] == point, "M4"] = M4 - 1

        dfs.append(df)

    # combine the two dataframes by adding the columns of the second dataframe to the first dataframe
    df = pd.concat(dfs, axis=1)

    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # add the raw data to the experiment filemap
    if add_raw:
        df = add_dir_to_experiment_filemap(df, os.path.join(experiment_dir, "raw"), "raw")

    # add the analysis data to the experiment filemap
    if analysis_to_add is not None:
        for key, value in analysis_to_add.items():
            df = add_dir_to_experiment_filemap(df, os.path.join(experiment_dir, 'analysis', key), f'analysis/{value}')


    # save the dataframe to a csv file
    save_path = os.path.join(matlab_report_dir, "analysis_filemap.csv")
    df.to_csv(save_path, index=False)

    return df, save_path

experiment_dir = "/mnt/towbin.data/shared/igheor/20240304_Ti2_10x_rpl22AID_titration_356_369_25C_20240304_163731_651"
matlab_report_dir = os.path.join(experiment_dir, "analysis/report/")
old_matlab_report_files = ["ch1_il_strS_cor_sec2_validlength.mat", "ch2_sobel_str_molts_nw_cor_sec2_validlength.mat"]
old_matlab_report_files = [os.path.join(matlab_report_dir, f) for f in old_matlab_report_files]
better_column_names = ["ch1_seg_str", "ch2_seg_str"]

extract_experiment_time = False
add_raw = False
analysis_to_add = None

# analysis_to_add = {
#     'ch1_il' : 'ch1_seg',
#     'ch2_sobel' : 'ch2_seg',
#     # 'ch1_il_strS' : 'ch1_seg_str',
#     # 'ch2_sobel_str' : 'ch2_seg_str',
# }

experiment_filemap, experiment_filemap_path = convert_matlab_experiment(experiment_dir, matlab_report_dir, old_matlab_report_files, better_column_names, add_raw = add_raw, analysis_to_add = analysis_to_add)

if "ExperimentTime" not in experiment_filemap.columns and extract_experiment_time:
    experiment_filemap["ExperimentTime"] = get_experiment_time_from_filemap_parallel(experiment_filemap)
    experiment_filemap.to_csv(
        experiment_filemap_path, index=False
    )