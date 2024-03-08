import os
from towbintools.foundation import file_handling as file_handling
import pandas as pd
from pipeline_scripts.utils import (
    get_experiment_time_from_filemap,
)

experiment_path = "/path/to/experiment"
filemap_name = "analysis_filemap.csv"
experiment_filemap_path = os.path.join(experiment_path, "analysis", "report", filemap_name)

experiment_filemap = pd.read_csv(experiment_filemap_path)
# if the ExperimentTime column is not present, create it
if "ExperimentTime" not in experiment_filemap.columns:
    experiment_filemap["ExperimentTime"] = get_experiment_time_from_filemap(experiment_filemap)
    experiment_filemap.to_csv(
        experiment_filemap_path, index=False
    )
    