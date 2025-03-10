import os

import pandas as pd
from towbintools.foundation import file_handling as file_handling

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_scripts.utils import get_experiment_time_from_filemap_parallel

if __name__ == "__main__":
    experiment_path = "/mnt/towbin.data/shared/spsalmon/20250127_ORCA_10x_chambers_for_lucien"
    filemap_name = "analysis_filemap.csv"
    experiment_filemap_path = os.path.join(
        experiment_path, "analysis_sacha", "report", filemap_name
    )

    experiment_filemap = pd.read_csv(experiment_filemap_path)

    experiment_filemap = pd.read_csv(experiment_filemap_path)
    # if the ExperimentTime column is not present, create it
    if "ExperimentTime" not in experiment_filemap.columns:
        experiment_filemap["ExperimentTime"] = get_experiment_time_from_filemap_parallel(
            experiment_filemap
        )
        experiment_filemap.to_csv(experiment_filemap_path, index=False)

