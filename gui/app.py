import os
import re

import aicsimageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.foundation import image_handling
from towbintools.foundation.worm_features import get_features_to_compute_at_molt
import scipy.io as sio

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()

KEY_CONVERSION_MAP = {
    "vol": "volume",
    "len": "length",
    "strClass": "worm_type",
    "ecdys": "ecdysis",
}

filemap_path = "/mnt/towbin.data/shared/plenart/20252501_squid_10x_wBT446_NaCl/analysis_Peter/report/analysis_filemap.csv"

filemap = pd.read_csv(filemap_path)

filemap_folder = os.path.dirname(filemap_path)
filemap_name = os.path.basename(filemap_path)
filemap_name = filemap_name.split(".")[0]

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


if "annotated" not in filemap_name:
    filemap_save_path = f"{filemap_name}_annotated.csv"
    filemap_save_path = os.path.join(filemap_folder, filemap_save_path)

    if os.path.exists(filemap_save_path):
        print(f"Annotated filemap already exists at {filemap_save_path}")
        print("Opening the existing filemap instead ...")
        filemap = pd.read_csv(filemap_save_path)
        filemap_path = filemap_save_path
        filemap_name = os.path.basename(filemap_path)
        filemap_name = filemap_name.split(".")[0]

        # backup the filemap
        backup_path = get_backup_path(filemap_folder, filemap_name)
        filemap.to_csv(backup_path, index=False)

else:
    # backup the filemap
    backup_path = get_backup_path(filemap_folder, filemap_name)
    filemap.to_csv(backup_path, index=False)
    filemap_save_path = filemap_path

times = filemap["Time"].unique().tolist()
points = filemap["Point"].unique().tolist()

# channels = image_handling.read_tiff_file(filemap["raw"].iloc[0]).shape[0]
channels = int(aicsimageio.AICSImage(filemap["raw"].iloc[1]).dims["C"][0])
# channels = 3
list_channels = [f"Channel {i+1}" for i in range(channels)]
list_channels = ["None"] + list_channels

usual_columns = [
    "Time",
    "Point",
    "raw",
    "HatchTime",
    "VolumeAtHatch",
    "M1",
    "VolumeAtM1",
    "M2",
    "VolumeAtM2",
    "M3",
    "VolumeAtM3",
    "M4",
    "VolumeAtM4",
    "date",
    "ExperimentTime",
]

usual_columns.extend(
    [column for column in filemap.columns.tolist() if "worm_type" in column]
)
usual_columns.extend([column for column in filemap.columns.tolist() if "seg" in column])
usual_columns.extend([column for column in filemap.columns.tolist() if "str" in column])
usual_columns.extend(
    [column for column in filemap.columns.tolist() if "length" in column]
)
list_custom_columns = [
    column for column in filemap.columns.tolist() if column not in usual_columns
]
# add None to the list of custom columns
list_custom_columns = ["None"] + list_custom_columns

print(f"usual columns : {usual_columns}")
print(f"custom columns : {list_custom_columns}")
print(f"all columns : {filemap.columns.tolist()}")

try:
    worm_type_column = [
        column for column in filemap.columns.tolist() if "worm_type" in column
    ][0]
except IndexError:
    print("No worm_type column found in the filemap")
    worm_type_column = "placeholder_worm_type"
    filemap["placeholder_worm_type"] = "worm"

base_volume_column = [
    column
    for column in filemap.columns.tolist()
    if "volume" in column and "_at_" not in column
][0]

segmentation_columns = [
    column
    for column in filemap.columns.tolist()
    if "seg" in column
    and "str" not in column
    and "worm_type" not in column
    and "length" not in column
    and "volume" not in column
    and "area" not in column
    and "width" not in column
    and "fluo" not in column
]
overlay_segmentation_choices = ["None"] + segmentation_columns

print("Adding molts column to filemap if they do not exist...")
# check if hatch columns exist
if "HatchTime" not in filemap.columns.tolist():
    filemap["HatchTime"] = np.nan
# check if molt columns exist
if "M1" not in filemap.columns.tolist():
    filemap["M1"] = np.nan
if "M2" not in filemap.columns.tolist():
    filemap["M2"] = np.nan
if "M3" not in filemap.columns.tolist():
    filemap["M3"] = np.nan
if "M4" not in filemap.columns.tolist():
    filemap["M4"] = np.nan

# create columns for features at molts if they do not exist

def create_feature_at_molt_columns(filemap, recompute_features=False):
    columns = filemap.columns.tolist()
    feature_columns = []
    for feature in FEATURES_TO_COMPUTE_AT_MOLT:
        feature_columns.extend(
            [col for col in columns if feature in col and "_at_" not in col]
        )

    ecdys_event_list = ['HatchTime', 'M1', 'M2', 'M3', 'M4']
    for feature_column in feature_columns:
        for ecdys_event in ecdys_event_list:
            feature_at_ecdysis_column = f"{feature_column}_at_{ecdys_event}"
            if (feature_at_ecdysis_column not in columns) or recompute_features:
                # filemap[feature_at_ecdysis_column] = np.nan
                for point in points:
                    data_of_point = filemap.loc[filemap["Point"] == point]
                    filemap.loc[filemap["Point"] == point, [feature_at_ecdysis_column]] = compute_series_at_time_classified(
                        data_of_point[feature_column].values, data_of_point[worm_type_column].values, data_of_point[ecdys_event].values[0]
                    )
    return filemap, feature_columns

filemap, feature_columns = create_feature_at_molt_columns(filemap)


def save_filemap(filemap=filemap):
    print("Saving filemap ...")
    filemap.to_csv(filemap_save_path, index=False)
    print("Filemap saved !")


print("Initializing the UI ...")
molt_annotator = ui.column(
    7,
    ui.row(output_widget("volume_plot")),
    ui.row(
        ui.column(
            2,
            ui.row(ui.output_text("hatch_text")),
            ui.row(ui.input_action_button("set_hatch", "hatch")),
            ui.row(ui.input_action_button("reset_hatch", "Reset Hatch")),
            ui.row(ui.input_action_button("set_value_at_hatch", "Set value at hatch")),
        ),
        ui.column(
            2,
            ui.row(ui.output_text("m1_text")),
            ui.row(ui.input_action_button("set_m1", "M1")),
            ui.row(ui.input_action_button("reset_m1", "Reset M1")),
            ui.row(ui.input_action_button("set_value_at_m1", "Set value at M1")),
        ),
        ui.column(
            2,
            ui.row(ui.output_text("m2_text")),
            ui.row(ui.input_action_button("set_m2", "M2")),
            ui.row(ui.input_action_button("reset_m2", "Reset M2")),
            ui.row(ui.input_action_button("set_value_at_m2", "Set value at M2")),
        ),
        ui.column(
            2,
            ui.row(ui.output_text("m3_text")),
            ui.row(ui.input_action_button("set_m3", "M3")),
            ui.row(ui.input_action_button("reset_m3", "Reset M3")),
            ui.row(ui.input_action_button("set_value_at_m3", "Set value at M3")),
        ),
        ui.column(
            2,
            ui.row(ui.output_text("m4_text")),
            ui.row(ui.input_action_button("set_m4", "M4")),
            ui.row(ui.input_action_button("reset_m4", "Reset M4")),
            ui.row(ui.input_action_button("set_value_at_m4", "Set value at M4")),
        ),
        ui.row(
            ui.column(
                4,
                ui.input_selectize(
                    "custom_column", "Select custom column", choices=list_custom_columns
                ),
            ),
            ui.column(
                4, ui.input_text("new_custom_column", "Insert new custom column")
            ),
            ui.column(
                2,
                ui.input_action_button("custom_annotation", "Annotate"),
                ui.input_action_button("reset_custom_annotation", "Reset"),
            ),
        ),
        ui.row(
            ui.input_slider(
                "volume_plot_size",
                "Volume plot size",
                min=300,
                max=1000,
                step=10,
                value=700,
            ),
            ui.input_checkbox("log_scale", "Log scale", value=True),
            ui.input_file("import_file", "Import Molts", accept=[".csv", ".mat"], multiple=False),
        ),
        align="center",
    ),
    align="center",
)

timepoint_selector = ui.column(
    5,
    ui.output_plot("plot", height="60vh"),
    ui.row(
        ui.column(
            4, ui.input_action_button("previous_time", "previous time", height="15vh")
        ),
        ui.column(4, ui.input_selectize("time", "Select time", choices=times)),
        ui.column(4, ui.input_action_button("next_time", "next time", height="15vh")),
    ),
    ui.row(
        ui.column(4, ui.input_action_button("previous_point", "previous point")),
        ui.column(4, ui.input_selectize("point", "Select point", choices=points)),
        ui.column(4, ui.input_action_button("next_point", "next point")),
    ),
    ui.row(
        ui.input_selectize(
            "channel",
            "Select channel",
            choices=list_channels,
            selected=list_channels[1],
        ),
        ui.input_selectize(
            "channel_overlay", "Overlay channel", choices=list_channels, selected="None"
        ),
        ui.input_selectize(
            "segmentation_overlay",
            "Overlay segmentation",
            choices=overlay_segmentation_choices,
            selected="None",
        ),
    ),
    ui.row(
        ui.input_selectize(
            "column_to_plot",
            "Select column to plot",
            selected=base_volume_column,
            choices=feature_columns,
        )
    ),
    ui.row(ui.input_action_button("save", "Save")),
)

app_ui = ui.page_fluid(ui.row(molt_annotator, timepoint_selector))


def set_marker_shape(
    selected_time_index, worm_types, hatch_time, m1, m2, m3, m4, custom_annotations: list = []
):
    symbols = []
    for worm_type in worm_types:
        if worm_type == "egg":
            symbol = "square-open"
        elif worm_type == "worm":
            symbol = "circle-open"
        else:
            symbol = "triangle-up-open"
        symbols.append(symbol)

    # create a list full of "blue"
    sizes = [4] * len(worm_types)
    colors = ["black"] * len(worm_types)

    for custom_annotation in custom_annotations:
        if np.isfinite(custom_annotation):
            symbols[int(custom_annotation)] = "circle"
            sizes[int(custom_annotation)] = 8
            colors[int(custom_annotation)] = "pink"

    if np.isfinite(hatch_time):
        symbols[int(hatch_time)] = "square"
        sizes[int(hatch_time)] = 8
        colors[int(hatch_time)] = "red"

    if np.isfinite(m1):
        symbols[int(m1)] = "circle"
        sizes[int(m1)] = 8
        colors[int(m1)] = "orange"

    if np.isfinite(m2):
        symbols[int(m2)] = "circle"
        sizes[int(m2)] = 8
        colors[int(m2)] = "yellow"

    if np.isfinite(m3):
        symbols[int(m3)] = "circle"
        sizes[int(m3)] = 8
        colors[int(m3)] = "green"

    if np.isfinite(m4):
        symbols[int(m4)] = "circle"
        sizes[int(m4)] = 8
        colors[int(m4)] = "blue"

    widths = [1] * len(worm_types)
    widths[int(selected_time_index)] = 4
    markers = dict(symbol=symbols, size=sizes, color=colors, line=dict(width=widths))
    return markers

def get_points_for_value_at_molts(hatch, m1, m2, m3, m4, value_at_hatch, value_at_m1, value_at_m2, value_at_m3, value_at_m4):
    ecdys_list = [hatch, m1, m2, m3, m4]
    value_at_ecdys_list = [value_at_hatch, value_at_m1, value_at_m2, value_at_m3, value_at_m4]
    symbols = ["cross", "cross", "cross", "cross", "cross"]
    colors = ["red", "orange", "yellow", "green", "blue"]
    sizes = [8, 8, 8, 8, 8]
    widths = [4, 4, 4, 4, 4]

    # Use numpy to handle NaN values efficiently
    ecdys_array = np.array(ecdys_list)
    value_at_ecdys_array = np.array(value_at_ecdys_list)
    valid_mask = np.isfinite(ecdys_array) & np.isfinite(value_at_ecdys_array)

    # Filter arrays using the mask
    ecdys_filtered = ecdys_array[valid_mask]
    value_at_ecdys_filtered = value_at_ecdys_array[valid_mask]
    symbols_filtered = np.array(symbols)[valid_mask]
    colors_filtered = np.array(colors)[valid_mask]
    sizes_filtered = np.array(sizes)[valid_mask]
    widths_filtered = np.array(widths)[valid_mask]

    return (
        ecdys_filtered,
        value_at_ecdys_filtered,
        symbols_filtered,
        colors_filtered,
        sizes_filtered,
        widths_filtered,
    )

def update_molt_and_ecdysis_columns(ecdys_event, time, point, selected_column):
    data_of_point = filemap.loc[filemap["Point"] == point]
    filemap.loc[filemap["Point"] == point, [ecdys_event]] = time
    value_at_ecdys_columns = [
        column
        for column in data_of_point.columns.tolist()
        if f"_at_{ecdys_event}" in column
    ]

    if f"{selected_column}_at_{ecdys_event}" not in value_at_ecdys_columns:
        print(f"Adding column {selected_column}_at_{ecdys_event} to filemap ...")
        filemap[f"{selected_column}_at_{ecdys_event}"] = np.nan
        value_at_ecdys_columns = [
            column
            for column in data_of_point.columns.tolist()
            if f"_at_{ecdys_event}" in column
        ]

    value_columns = [
        column.replace(f"_at_{ecdys_event}", "")
        for column in value_at_ecdys_columns
    ]

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        if np.isnan(time):
            new_column_value = np.nan
        else:
            new_column_value = compute_series_at_time_classified(
                data_of_point[value_column].values,
                data_of_point[worm_type_column].values,
                time,
            )

        print(
            f'Old value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}'
        )

        filemap.loc[(filemap["Point"] == point), [value_at_ecdys_column]] = (
            new_column_value
        )

        print(
            f'New value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}'
        )

def correct_ecdysis_columns(ecdys_event, time, point, selected_column):
    data_of_point = filemap.loc[filemap["Point"] == point]
    value_at_ecdys_columns = [
        column
        for column in data_of_point.columns.tolist()
        if f"_at_{ecdys_event}" in column
    ]

    if f"{selected_column}_at_{ecdys_event}" not in value_at_ecdys_columns:
        print(f"Adding column {selected_column}_at_{ecdys_event} to filemap ...")
        filemap[f"{selected_column}_at_{ecdys_event}"] = np.nan
        value_at_ecdys_columns = [
            column
            for column in data_of_point.columns.tolist()
            if f"_at_{ecdys_event}" in column
        ]

    value_columns = [
        column.replace(f"_at_{ecdys_event}", "")
        for column in value_at_ecdys_columns
    ]

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        new_column_value = data_of_point[data_of_point["Time"] == time][value_column].values[0]

        print(
            f'Old value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}'
        )

        filemap.loc[(filemap["Point"] == point), [value_at_ecdys_column]] = (
            new_column_value
        )

        print(
            f'New value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}')

def server(input, output, session):
    print("Initializing the server ...")
    hatch = reactive.Value("")
    m1 = reactive.Value("")
    m2 = reactive.Value("")
    m3 = reactive.Value("")
    m4 = reactive.Value("")

    value_at_hatch = reactive.Value("")
    value_at_m1 = reactive.Value("")
    value_at_m2 = reactive.Value("")
    value_at_m3 = reactive.Value("")
    value_at_m4 = reactive.Value("")


    @reactive.calc
    def import_molts():
        file = input.import_file()
        if file is None:
            return None
        if file[0]["datapath"].endswith(".csv"):
            print(f"Importing molts from {file[0]['datapath']} ...")
            molts_df = pd.read_csv(file[0]["datapath"])
            
            # keep only the relevant columns
            molts_df = molts_df[["Point", "Time", "HatchTime", "M1", "M2", "M3", "M4"]]
            return molts_df

        elif file[0]["datapath"].endswith(".mat"):
            matlab_report = sio.loadmat(file[0]["datapath"], chars_as_strings=False)
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
            
            point = np.arange(0, new_matlab_report["volume"].shape[0])
            time = np.arange(0, new_matlab_report["volume"].shape[1])

            # combine each point with every time
            time_point = np.array(np.meshgrid(point, time)).T.reshape(-1, 2)

            time = time_point[:, 1]
            point = time_point[:, 0]

            # put everything into a nice dataframe
            data = {
                "Point": point,
                "Time": time,
            }
            df = pd.DataFrame(data)

            try:
                ecdysis = new_matlab_report["ecdysis"]
                hatch, M1, M2, M3, M4 = np.split(ecdysis, 5, axis=1)

                for point, molts in enumerate(ecdysis):
                    HatchTime, M1, M2, M3, M4 = molts
                    # convert to int if not nan
                    if not np.isnan(HatchTime):
                        HatchTime = int(HatchTime) - 1
                    if not np.isnan(M1):
                        M1 = int(M1) - 1
                    if not np.isnan(M2):
                        M2 = int(M2) - 1
                    if not np.isnan(M3):
                        M3 = int(M3) - 1
                    if not np.isnan(M4):
                        M4 = int(M4) - 1

                    df.loc[df["Point"] == point, "HatchTime"] = HatchTime
                    df.loc[df["Point"] == point, "M1"] = M1
                    df.loc[df["Point"] == point, "M2"] = M2
                    df.loc[df["Point"] == point, "M3"] = M3
                    df.loc[df["Point"] == point, "M4"] = M4
            except KeyError:
                print("No molts found in the matlab report")

            return df

    @reactive.Effect
    @reactive.event(input.import_file)
    def replace_molts():
        print("Replacing molts ...")
        global filemap
        molts_df = import_molts()

        if molts_df is None:
            return

        print(molts_df.head())
        if molts_df is not None:
            for molt_column in molts_df.columns:
                try:
                    new_molt = molts_df[molt_column]
                    molt = filemap[molt_column]
                    if new_molt.values.shape == molt.values.shape:
                        filemap[molt_column] = new_molt
                except KeyError:
                    print(f"Column {molt_column} not found in the imported file")

            filemap, _ = create_feature_at_molt_columns(filemap, recompute_features=True)

    @reactive.Calc
    def get_images_of_point():
        images_of_point_paths = filemap[filemap["Point"] == int(input.point())][
            "raw"
        ].values.tolist()

        images_of_point = images_of_point_paths
        # images_of_point = Parallel(n_jobs=-1)(delayed(image_handling.read_tiff_file)(path) for path in images_of_point_paths['raw'].values.tolist())
        return images_of_point

    @reactive.Calc
    def get_segmentation_of_point():
        if input.segmentation_overlay() == "None":
            return []

        segmentation_of_point_paths = filemap[filemap["Point"] == int(input.point())][
            input.segmentation_overlay()
        ].values.tolist()

        segmentation_of_point = segmentation_of_point_paths
        return segmentation_of_point

    def get_custom_annotations():
        if input.custom_column() == "" or input.custom_column() == "None":
            return []
        custom_annotations = filemap[filemap["Point"] == int(input.point())][
            input.custom_column()
        ].values.tolist()
        return custom_annotations

    @reactive.Effect
    @reactive.event(input.save)
    def save():
        save_filemap(filemap)

    @reactive.Effect
    def get_hatch_and_molts():
        hatch_and_molts = filemap[filemap["Point"] == int(input.point())][
            ["HatchTime", "M1", "M2", "M3", "M4"]
        ].values.tolist()[0]

        hatch.set(hatch_and_molts[0])
        m1.set(hatch_and_molts[1])
        m2.set(hatch_and_molts[2])
        m3.set(hatch_and_molts[3])
        m4.set(hatch_and_molts[4])

        try:
            value_at_hatch.set(
                filemap.loc[
                    (filemap["Point"] == int(input.point())),
                    f"{input.column_to_plot()}_at_HatchTime",
                ].values[0]
            )
            value_at_m1.set(
                filemap.loc[
                    (filemap["Point"] == int(input.point())),
                    f"{input.column_to_plot()}_at_M1",
                ].values[0]
            )
            value_at_m2.set(
                filemap.loc[
                    (filemap["Point"] == int(input.point())),
                    f"{input.column_to_plot()}_at_M2",
                ].values[0]
            )
            value_at_m3.set(
                filemap.loc[
                    (filemap["Point"] == int(input.point())),
                    f"{input.column_to_plot()}_at_M3",
                ].values[0]
            )
            value_at_m4.set(
                filemap.loc[
                    (filemap["Point"] == int(input.point())),
                    f"{input.column_to_plot()}_at_M4",
                ].values[0]
            )
        except KeyError:
            value_at_hatch.set(np.nan)
            value_at_m1.set(np.nan)
            value_at_m2.set(np.nan)
            value_at_m3.set(np.nan)
            value_at_m4.set(np.nan)

    @reactive.Effect
    @reactive.event(input.previous_time)
    def previous_time():
        print("previous_time")
        # new_time = max(int(input.time()) - 1, np.min(times))
        new_time_index = max(np.where(np.array(times) == int(input.time()))[0][0] - 1, 0)
        new_time = times[new_time_index]
        ui.update_selectize("time", selected=str(int(new_time)))

    @reactive.Effect
    @reactive.event(input.next_time)
    def next_time():
        print("next_time")
        # new_time = min(int(input.time()) + 1, np.max(times))
        new_time_index = min(np.where(np.array(times) == int(input.time()))[0][0] + 1, len(times) - 1)
        new_time = times[new_time_index]
        ui.update_selectize("time", selected=str(int(new_time)))

    @reactive.Effect
    @reactive.event(input.previous_point)
    def previous_point():
        print("previous_point")
        # new_point = max(int(input.point()) - 1, np.min(points))
        new_point_index = max(np.where(np.array(points) == int(input.point()))[0][0] - 1, 0)
        new_point = points[new_point_index]
        ui.update_selectize("point", selected=str(int(new_point)))

    @reactive.Effect
    @reactive.event(input.next_point)
    def next_point():
        print("next_point")
        # new_point = min(int(input.point()) + 1, np.max(points))
        new_point_index = min(np.where(np.array(points) == int(input.point()))[0][0] + 1, len(points) - 1)
        new_point = points[new_point_index]
        ui.update_selectize("point", selected=str(int(new_point)))

    @output
    @render_widget
    def volume_plot():
        data_of_point = filemap.loc[
            filemap["Point"] == int(input.point()),
            [
                "Time",
                input.column_to_plot(),
                worm_type_column,
                "HatchTime",
                "M1",
                "M2",
                "M3",
                "M4",
            ],
        ]
        
        markers = set_marker_shape(
            np.where(np.array(times) == int(input.time()))[0][0],
            data_of_point[worm_type_column],
            hatch(),
            m1(),
            m2(),
            m3(),
            m4(),
            custom_annotations=get_custom_annotations(),
        )
        fig = go.FigureWidget()
        fig.add_trace(
            go.Scatter(
                x=data_of_point["Time"],
                y=data_of_point[input.column_to_plot()],
                mode="markers",
                marker=markers,
            )
        )

        ecdys_list, value_at_ecdys_list, symbols, colors, sizes, widths = (
            get_points_for_value_at_molts(hatch(), m1(), m2(), m3(), m4(), value_at_hatch(), value_at_m1(), value_at_m2(), value_at_m3(), value_at_m4())
        )

        fig.add_trace(
            go.Scatter(
                x=ecdys_list,
                y=value_at_ecdys_list,
                mode="markers",
                marker=dict(
                    symbol=symbols,
                    size=sizes,
                    color=colors,
                    line=dict(width=widths, color=colors),
                ),
                hoverinfo="none",  # Disable hover information
                hoverlabel=None,  # Disable hover label
                hoveron=None,  # Disable hover interaction
            )
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=input.column_to_plot(),
            margin=dict(l=20, r=20, t=50, b=50),
            height=input.volume_plot_size(),
            showlegend=False,
        )

        fig.update_yaxes(type="log" if input.log_scale() else "linear")

        def update_selected_time(trace, points, selector):
            print("update_selected_time")
            for point in points.xs:
                ui.update_selectize("time", selected=str(point))

        fig.data[0].on_click(update_selected_time)

        return fig

    @output
    @render.text
    def hatch_text():
        return f"Hatch : {str(hatch())}"

    @output
    @render.text
    def m1_text():
        return f"M1 : {str(m1())}"

    @output
    @render.text
    def m2_text():
        return f"M2: {str(m2())}"

    @output
    @render.text
    def m3_text():
        return f"M3 : {str(m3())}"

    @output
    @render.text
    def m4_text():
        return f"M4 : {str(m4())}"

    @reactive.Effect
    @reactive.event(input.set_hatch)
    def set_hatch():
        print("set_hatch")
        new_hatch = float(input.time())
        update_molt_and_ecdysis_columns(
            "HatchTime", new_hatch, int(input.point()), input.column_to_plot()
        )
        hatch.set(new_hatch)
        value_at_hatch.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_HatchTime",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_m1)
    def set_m1():
        print("set_m1")
        new_m1 = float(input.time())
        update_molt_and_ecdysis_columns("M1", new_m1, int(input.point()), input.column_to_plot())
        m1.set(new_m1)
        value_at_m1.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M1",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_m2)
    def set_m2():
        print("set_m2")
        new_m2 = float(input.time())
        update_molt_and_ecdysis_columns("M2", new_m2, int(input.point()), input.column_to_plot())
        m2.set(new_m2)
        value_at_m2.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M2",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_m3)
    def set_m3():
        print("set_m3")
        new_m3 = float(input.time())
        update_molt_and_ecdysis_columns("M3", new_m3, int(input.point()), input.column_to_plot())
        m3.set(new_m3)
        value_at_m3.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M3",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_m4)
    def set_m4():
        print("set_m4")
        new_m4 = float(input.time())
        update_molt_and_ecdysis_columns("M4", new_m4, int(input.point()), input.column_to_plot())
        m4.set(new_m4)
        value_at_m4.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M4",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.reset_hatch)
    def reset_hatch():
        print("reset_hatch")
        update_molt_and_ecdysis_columns(
            "HatchTime", np.nan, int(input.point()), input.column_to_plot()
        )
        hatch.set(np.nan)
        value_at_hatch.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m1)
    def reset_m1():
        print("reset_m1")
        update_molt_and_ecdysis_columns("M1", np.nan, int(input.point()), input.column_to_plot())
        m1.set(np.nan)
        value_at_m1.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m2)
    def reset_m2():
        print("reset_m2")
        update_molt_and_ecdysis_columns("M2", np.nan, int(input.point()), input.column_to_plot())
        m2.set(np.nan)
        value_at_m2.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m3)
    def reset_m3():
        print("reset_m3")
        update_molt_and_ecdysis_columns("M3", np.nan, int(input.point()), input.column_to_plot())
        m3.set(np.nan)
        value_at_m3.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m4)
    def reset_m4():
        print("reset_m4")
        update_molt_and_ecdysis_columns("M4", np.nan, int(input.point()), input.column_to_plot())
        m4.set(np.nan)
        value_at_m4.set(np.nan)

    @reactive.Effect
    @reactive.event(input.set_value_at_hatch)
    def set_value_at_hatch():
        print("set_value_at_hatch")
        correct_ecdysis_columns("HatchTime", int(input.time()), int(input.point()), input.column_to_plot())
        value_at_hatch.set(filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_HatchTime",
            ].values[0])

    @reactive.Effect
    @reactive.event(input.set_value_at_m1)
    def set_value_at_m1():
        print("set_value_at_m1")
        correct_ecdysis_columns("M1", int(input.time()), int(input.point()), input.column_to_plot())
        value_at_m1.set(filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M1",
            ].values[0])

    @reactive.Effect
    @reactive.event(input.set_value_at_m2)
    def set_value_at_m2():
        print("set_value_at_m2")
        correct_ecdysis_columns("M2", int(input.time()), int(input.point()), input.column_to_plot())
        value_at_m2.set(filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M2",
            ].values[0])

    @reactive.Effect
    @reactive.event(input.set_value_at_m3)
    def set_value_at_m3():
        print("set_value_at_m3")
        correct_ecdysis_columns("M3", int(input.time()), int(input.point()), input.column_to_plot())
        value_at_m3.set(filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M3",
            ].values[0])

    @reactive.Effect
    @reactive.event(input.set_value_at_m4)
    def set_value_at_m4():
        print("set_value_at_m4")
        correct_ecdysis_columns("M4", int(input.time()), int(input.point()), input.column_to_plot())
        value_at_m4.set(filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M4",
            ].values[0])

    @reactive.Effect
    @reactive.event(input.custom_annotation)
    def custom_annotation():
        new_column_name = input.new_custom_column()
        custom_column = input.custom_column()

        if new_column_name != "" and (new_column_name not in list_custom_columns):
            print("new column")
            list_custom_columns.append(new_column_name)
            ui.update_selectize(
                "custom_column", choices=list_custom_columns, selected=new_column_name
            )
            ui.update_text("new_custom_column", value="")
            filemap[new_column_name] = np.nan
            filemap.loc[
                (
                    (filemap["Point"] == int(input.point()))
                    & (filemap["Time"] == int(input.time()))
                ),
                new_column_name,
            ] = float(input.time())

            print(
                filemap.loc[
                    (
                        (filemap["Point"] == int(input.point()))
                        & (filemap["Time"] == int(input.time()))
                    )
                ]
            )

        if new_column_name == "" and custom_column != "":
            print("add custom annotation")
            filemap.loc[
                (
                    (filemap["Point"] == int(input.point()))
                    & (filemap["Time"] == int(input.time()))
                ),
                custom_column,
            ] = float(input.time())
            print(
                filemap.loc[
                    (
                        (filemap["Point"] == int(input.point()))
                        & (filemap["Time"] == int(input.time()))
                    )
                ]
            )

    @reactive.Effect
    @reactive.event(input.reset_custom_annotation)
    def reset_custom_annotation():
        custom_column = input.custom_column()
        if custom_column != "":
            print("reset custom annotation")
            filemap.loc[
                (
                    (filemap["Point"] == int(input.point()))
                    & (filemap["Time"] == int(input.time()))
                ),
                custom_column,
            ] = np.nan
            print(
                filemap.loc[
                    (
                        (filemap["Point"] == int(input.point()))
                        & (filemap["Time"] == int(input.time()))
                    )
                ]
            )

    @output
    @render.plot
    def plot():
        channel = input.channel()
        channel = channel.split(" ")[-1]
        channel = int(channel) - 1

        images_of_point = get_images_of_point()
        segmentation_of_point = get_segmentation_of_point()

        img = images_of_point[np.where(np.array(times) == int(input.time()))[0][0]]
        img = image_handling.read_tiff_file(img)

        if img.ndim == 3:
            img_to_plot = img[channel]
        elif img.ndim == 4:
            img_to_plot = img[img.shape[0] // 2, channel, ...]
        elif img.ndim == 2:
            img_to_plot = img

        plot_overlay = False
        if input.channel_overlay() != "None":
            channel_overlay = input.channel_overlay()
            channel_overlay = channel_overlay.split(" ")[-1]
            channel_overlay = int(channel_overlay) - 1

            overlay = img[channel_overlay]
            overlay = image_handling.normalize_image(overlay, dest_dtype=np.float64)
            plot_overlay = True
        else:
            overlay = np.zeros_like(img[channel])

        img_to_plot = image_handling.normalize_image(img_to_plot, dest_dtype=np.float64)

        if len(segmentation_of_point) > 0:
            segmentation = segmentation_of_point[int(input.time())]
            segmentation = image_handling.read_tiff_file(segmentation)
            fig, ax = plt.subplots()
            ax.imshow(img_to_plot, cmap="viridis")
            if plot_overlay:
                ax.imshow(overlay, cmap="magma", alpha=0.7)
            ax.imshow(segmentation.squeeze(), cmap="gray", alpha=0.3)
            return fig
        else:
            fig, ax = plt.subplots()
            ax.imshow(img_to_plot, cmap="viridis")
            if plot_overlay:
                ax.imshow(overlay, cmap="magma", alpha=0.7)
            return fig

    session.on_ended(save_filemap)

    @reactive.Effect
    @reactive.event(input.close)
    async def _():
        await session.close()


print("Creating the app ...")
app = App(app_ui, server)
