import os
import re

import polars as pl
from towbintools.foundation import image_handling

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.graph_objs as go
# import scipy.io as sio
# from shiny import App
# from shiny import reactive
# from shiny import render
# from shiny import ui
# from shinywidgets import output_widget
# from shinywidgets import render_widget
# from towbintools.data_analysis import compute_series_at_time_classified
# from towbintools.foundation.worm_features import get_features_to_compute_at_molt


def get_backup_path(filemap_folder, filemap_name):
    # check if the filemap is already annotated
    match = re.search(r"annotated_v(/d+)", filemap_name)
    if not match:
        iteration = 1
    else:
        iteration = int(match.group(1))

    filemap_save_path = f"{filemap_name}_v{iteration}.csv"
    while os.path.exists(os.path.join(filemap_folder, filemap_save_path)):
        iteration += 1
        filemap_save_path = f"{filemap_name}_v{iteration}.csv"

    filemap_save_path = os.path.join(filemap_folder, filemap_save_path)
    return filemap_save_path


def open_filemap(filemap_path, open_annotated=True):
    filemap_folder = os.path.dirname(filemap_path)
    filemap_name = os.path.basename(filemap_path)
    filemap_name = filemap_name.split(".")[0]

    if "annotated" not in filemap_name and open_annotated:
        filemap_save_path = f"{filemap_name}_annotated.csv"
        filemap_save_path = os.path.join(filemap_folder, filemap_save_path)

        if os.path.exists(filemap_save_path):
            print(f"Annotated filemap already exists at {filemap_save_path}")
            print("Opening the existing filemap instead ...")
            filemap = pl.read_csv(filemap_save_path, infer_schema_length=10000)
            filemap_path = filemap_save_path
            filemap_name = os.path.basename(filemap_path)
            filemap_name = filemap_name.split(".")[0]

            # backup the filemap
            backup_path = get_backup_path(filemap_folder, filemap_name)
            filemap.write_csv(backup_path)
    else:
        # backup the filemap
        filemap = pl.read_csv(filemap_path, infer_schema_length=10000)
        backup_path = get_backup_path(filemap_folder, filemap_name)
        filemap.write_csv(backup_path)
        filemap_save_path = filemap_path

    return filemap, filemap_save_path


def infer_n_channels(filemap):
    first_image_path = filemap.select(pl.col("raw")).to_numpy().squeeze()[0]
    first_image = image_handling.read_tiff_file(first_image_path)

    if first_image.ndim == 3:
        n_channels = first_image.shape[0]
    elif first_image.ndim == 4:
        n_channels = first_image.shape[1]
    elif first_image.ndim == 2:
        n_channels = 1
    else:
        raise ValueError("Unknown number of channels")

    return n_channels
