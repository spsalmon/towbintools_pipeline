from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import polars as pl
import scipy.io as sio
from app_components import build_single_values_df
from app_components import get_points_for_value_at_molts
from app_components import get_time_and_ecdysis
from app_components import infer_n_channels
from app_components import open_filemap
from app_components import populate_column_choices
from app_components import process_feature_at_molt_columns
from app_components import save_filemap
from app_components import set_marker_shape
from shiny import App
from shiny import module
from shiny import reactive
from shiny import render
from shiny import ui
from shinywidgets import output_widget
from shinywidgets import render_widget
from towbintools.data_analysis import compute_series_at_time_classified
from towbintools.foundation import image_handling
from towbintools.foundation.worm_features import get_features_to_compute_at_molt

FEATURES_TO_COMPUTE_AT_MOLT = get_features_to_compute_at_molt()

KEY_CONVERSION_MAP = {
    "vol": "volume",
    "len": "length",
    "strClass": "worm_type",
    "ecdys": "ecdysis",
}

ECDYSIS_COLUMNS = ["HatchTime", "M1", "M2", "M3", "M4"]
recompute_features_at_molt = True

filemap_path = "/mnt/towbin.data/shared/spsalmon/20250314_squid_10x_yap_aid_160_438_492_493/analysis/report/analysis_filemap.csv"

start = perf_counter()
filemap, filemap_save_path = open_filemap(filemap_path)
print(f"Filemap opened in {perf_counter() - start:.2f} seconds")

times = (
    filemap.select(pl.col("Time"))
    .unique(maintain_order=True)
    .to_numpy()
    .squeeze()
    .tolist()
)
points = (
    filemap.select(pl.col("Point"))
    .unique(maintain_order=True)
    .to_numpy()
    .squeeze()
    .tolist()
)

n_channels = infer_n_channels(filemap)
list_channels = ["None"] + [f"Channel {i+1}" for i in range(n_channels)]

start = perf_counter()
(
    feature_columns,
    custom_columns_choices,
    worm_type_column,
    default_plotted_column,
    overlay_segmentation_choices,
) = populate_column_choices(filemap)
print(f"Column choices populated in {perf_counter() - start:.2f} seconds")

start = perf_counter()
filemap = process_feature_at_molt_columns(
    filemap,
    feature_columns,
    worm_type_column,
    recompute_features_at_molt=recompute_features_at_molt,
)
ecdysis_list_id = ["hatch", "m1", "m2", "m3", "m4"]


@module.ui
def molt_annotation_buttons(molt, width=2):
    return ui.column(
        width,
        ui.row(ui.input_action_button("set_molt", f"{molt.capitalize()}")),
        ui.row(ui.input_action_button("reset_molt", f"Reset {molt.capitalize()}")),
        ui.row(
            ui.input_action_button(
                "set_value_at_molt", f"Set value at {molt.capitalize()}"
            )
        ),
    )


print("Initializing the UI ...")
molt_annotator = ui.column(
    7,
    ui.row(output_widget("volume_plot")),
    ui.row(
        [molt_annotation_buttons(molt, molt=molt) for molt in ecdysis_list_id],
        ui.column(
            2,
            ui.row(ui.input_action_button("set_arrest", "Arrest")),
            ui.row(ui.input_action_button("set_death", "Dead")),
            ui.row(ui.input_action_button("set_ignore_after", "Ignore After")),
            ui.row(ui.input_action_button("set_ignore_point", "Ignore Point")),
        ),
        ui.row(
            ui.column(
                4,
                ui.input_selectize(
                    "custom_column",
                    "Select custom column",
                    choices=custom_columns_choices,
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
            ui.input_file(
                "import_file", "Import Molts", accept=[".csv", ".mat"], multiple=False
            ),
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
            selected=default_plotted_column,
            choices=feature_columns,
        )
    ),
    ui.row(ui.input_action_button("save", "Save")),
)

app_ui = ui.page_fluid(ui.row(molt_annotator, timepoint_selector))


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
        column.replace(f"_at_{ecdys_event}", "") for column in value_at_ecdys_columns
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
                series_time=data_of_point["Time"].values,
            )

        print(
            f'Old value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}'
        )

        filemap.loc[
            (filemap["Point"] == point), [value_at_ecdys_column]
        ] = new_column_value

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
        column.replace(f"_at_{ecdys_event}", "") for column in value_at_ecdys_columns
    ]

    for value_column, value_at_ecdys_column in zip(
        value_columns, value_at_ecdys_columns
    ):
        new_column_value = data_of_point[data_of_point["Time"] == time][
            value_column
        ].values[0]

        print(
            f'Old value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}'
        )

        filemap.loc[
            (filemap["Point"] == point), [value_at_ecdys_column]
        ] = new_column_value

        print(
            f'New value {value_at_ecdys_column}: {filemap.loc[(filemap["Point"] == point), value_at_ecdys_column].values[0]}'
        )


def set_ignore_start(point):
    try:
        data_of_point = filemap.loc[filemap["Point"] == point]
        # set ignore_start to the first time point that is ignored
        ignore_start = data_of_point[data_of_point["Ignore"]]["Time"].min()

        print(f"Ignore start: {ignore_start}")
        return ignore_start
    except KeyError:
        return ""


def server(input, output, session):
    global filemap
    unique_points = (
        filemap.select(pl.col("Point")).unique(maintain_order=True).to_numpy().squeeze()
    )
    point_filemaps = filemap.partition_by("Point", maintain_order=True)
    (
        time,
        experiment_time,
        ecdysis_time,
        ecdysis_index,
        ecdysis_experiment_time,
    ) = get_time_and_ecdysis(filemap)
    single_values_df = build_single_values_df(filemap)

    single_values_of_points = single_values_df.partition_by(
        "Point", maintain_order=True
    )

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

    ignore_start = reactive.Value("")
    death = reactive.Value("")
    arrest = reactive.Value("")

    current_point = reactive.Value("")
    current_point_filemap = reactive.Value("")
    single_values_of_point = reactive.Value("")

    @reactive.Effect
    @reactive.event(input.point)
    def update_current_point():
        current_point.set(int(input.point()))
        current_point_index = np.where(unique_points == int(input.point()))[0][0]
        current_point_filemap.set(point_filemaps[current_point_index])
        single_values_of_point.set(single_values_of_points[current_point_index])

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
                    print(f"Error converting key {key}: {e}")
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

    # @reactive.Effect
    # @reactive.event(input.import_file)
    # def replace_molts():
    #     print("Replacing molts ...")
    #     global filemap
    #     molts_df = import_molts()

    #     if molts_df is None:
    #         return

    #     print(molts_df.head())
    #     if molts_df is not None:
    #         for molt_column in molts_df.columns:
    #             try:
    #                 new_molt = molts_df[molt_column]
    #                 molt = filemap[molt_column]
    #                 if new_molt.values.shape == molt.values.shape:
    #                     filemap[molt_column] = new_molt
    #             except KeyError:
    #                 print(f"Column {molt_column} not found in the imported file")

    #         filemap, _ = create_feature_at_molt_columns(
    #             filemap, recompute_features=True
    #         )

    @reactive.calc
    def get_images_of_point():
        return current_point_filemap().select("raw").to_numpy().squeeze().tolist()

    @reactive.calc
    def get_segmentation_of_point():
        if input.segmentation_overlay() == "None":
            return []

        return (
            current_point_filemap()
            .select(input.segmentation_overlay())
            .to_numpy()
            .squeeze()
            .tolist()
        )

    def get_custom_annotations():
        if input.custom_column() == "" or input.custom_column() == "None":
            return []

        return (
            current_point_filemap()
            .select(input.custom_column())
            .to_numpy()
            .squeeze()
            .tolist()
        )

    @reactive.Effect
    @reactive.event(input.save)
    def save():
        save_filemap(filemap)

    @reactive.Effect
    def get_hatch_and_molts():
        hatch_and_molts = (
            single_values_of_point()
            .select(pl.col(ECDYSIS_COLUMNS))
            .to_numpy()
            .squeeze()
        )

        hatch.set(hatch_and_molts[0])
        m1.set(hatch_and_molts[1])
        m2.set(hatch_and_molts[2])
        m3.set(hatch_and_molts[3])
        m4.set(hatch_and_molts[4])

        try:
            value_at_hatch.set(
                single_values_of_point()
                .select(pl.col(f"{input.column_to_plot()}_at_HatchTime"))
                .to_numpy()
                .squeeze()
            )
            value_at_m1.set(
                single_values_of_point()
                .select(pl.col(f"{input.column_to_plot()}_at_M1"))
                .to_numpy()
                .squeeze()
            )
            value_at_m2.set(
                single_values_of_point()
                .select(pl.col(f"{input.column_to_plot()}_at_M2"))
                .to_numpy()
                .squeeze()
            )
            value_at_m3.set(
                single_values_of_point()
                .select(pl.col(f"{input.column_to_plot()}_at_M3"))
                .to_numpy()
                .squeeze()
            )
            value_at_m4.set(
                single_values_of_point()
                .select(pl.col(f"{input.column_to_plot()}_at_M4"))
                .to_numpy()
                .squeeze()
            )
        except Exception as e:
            print(f"Exception caught while getting value at hatch and molts: {e}")
            value_at_hatch.set(np.nan)
            value_at_m1.set(np.nan)
            value_at_m2.set(np.nan)
            value_at_m3.set(np.nan)
            value_at_m4.set(np.nan)

    # @reactive.Effect
    # def set_ignore_start():
    #     ignore_start.set(set_ignore_start(int(input.point())))

    # @reactive.Effect
    # def get_death():
    #     try:
    #         death.set(
    #             filemap.loc[filemap["Point"] == int(input.point()), "Death"].values[0]
    #         )
    #     except KeyError:
    #         death.set("")

    #     print(f"Death: {death()}")

    # @reactive.Effect
    # def get_arrest():
    #     try:
    #         arrest.set(
    #             filemap.loc[filemap["Point"] == int(input.point()), "Arrest"].values[0]
    #         )
    #     except KeyError:
    #         arrest.set(False)

    @reactive.Effect
    @reactive.event(input.previous_time)
    def previous_time():
        print("previous_time")
        # new_time = max(int(input.time()) - 1, np.min(times))
        new_time_index = max(
            np.where(np.array(times) == int(input.time()))[0][0] - 1, 0
        )
        new_time = times[new_time_index]
        ui.update_selectize("time", selected=str(int(new_time)))

    @reactive.Effect
    @reactive.event(input.next_time)
    def next_time():
        print("next_time")
        # new_time = min(int(input.time()) + 1, np.max(times))
        new_time_index = min(
            np.where(np.array(times) == int(input.time()))[0][0] + 1, len(times) - 1
        )
        new_time = times[new_time_index]
        ui.update_selectize("time", selected=str(int(new_time)))

    @reactive.Effect
    @reactive.event(input.previous_point)
    def previous_point():
        print("previous_point")
        # new_point = max(int(input.point()) - 1, np.min(points))
        new_point_index = max(
            np.where(np.array(points) == int(input.point()))[0][0] - 1, 0
        )
        new_point = points[new_point_index]
        ui.update_selectize("point", selected=str(int(new_point)))

    @reactive.Effect
    @reactive.event(input.next_point)
    def next_point():
        print("next_point")
        # new_point = min(int(input.point()) + 1, np.max(points))
        new_point_index = min(
            np.where(np.array(points) == int(input.point()))[0][0] + 1, len(points) - 1
        )
        new_point = points[new_point_index]
        ui.update_selectize("point", selected=str(int(new_point)))

    # @reactive.Effect
    # @reactive.event(input.set_ignore_point)
    # def set_ignore_point():
    #     global filemap
    #     print("set ignore point")
    #     current_point = int(input.point())

    #     mask = filemap["Point"] == current_point

    #     try:
    #         if np.all(filemap.loc[mask, "Ignore"]):
    #             filemap.loc[mask, "Ignore"] = False
    #         else:
    #             filemap.loc[mask, "Ignore"] = True
    #     except KeyError:
    #         filemap["Ignore"] = False
    #         filemap.loc[mask, "Ignore"] = True

    #     ignore_start.set(set_ignore_start(current_point))

    # @reactive.Effect
    # @reactive.event(input.set_ignore_after)
    # def set_ignore_after():
    #     global filemap
    #     print("set ignore after")
    #     current_point = int(input.point())
    #     current_time = int(input.time())

    #     mask_point = filemap["Point"] == current_point
    #     mask_after = (filemap["Point"] == current_point) & (
    #         filemap["Time"] >= current_time
    #     )

    #     try:
    #         if np.all(filemap.loc[mask_point, "Ignore"]):
    #             filemap.loc[mask_point, "Ignore"] = False
    #             filemap.loc[mask_after, "Ignore"] = True
    #         elif np.all(filemap.loc[mask_after, "Ignore"]):
    #             filemap.loc[mask_point, "Ignore"] = False
    #         else:
    #             filemap.loc[mask_point, "Ignore"] = False
    #             filemap.loc[mask_after, "Ignore"] = True
    #     except KeyError:
    #         filemap["Ignore"] = False
    #         filemap.loc[mask_after, "Ignore"] = True

    #     ignore_start.set(set_ignore_start(current_point))

    # @reactive.Effect
    # @reactive.event(input.set_death)
    # def set_death():
    #     global filemap
    #     print("set death")
    #     current_point = int(input.point())
    #     current_time = int(input.time())

    #     mask = filemap["Point"] == current_point

    #     try:
    #         if filemap.loc[mask, "Death"].values[0] == current_time:
    #             filemap.loc[mask, "Death"] = np.nan
    #         else:
    #             filemap.loc[mask, "Death"] = current_time
    #     except KeyError:
    #         filemap["Death"] = np.nan
    #         filemap.loc[mask, "Death"] = current_time

    #     death.set(filemap.loc[mask, "Death"].values[0])

    # @reactive.Effect
    # @reactive.event(input.set_arrest)
    # def set_arrest():
    #     global filemap
    #     print("set arrest")
    #     current_point = int(input.point())

    #     mask = filemap["Point"] == current_point

    #     try:
    #         if np.all(filemap.loc[mask, "Arrest"]):
    #             filemap.loc[mask, "Arrest"] = False
    #         else:
    #             filemap.loc[mask, "Arrest"] = True
    #     except KeyError:
    #         filemap["Arrest"] = False
    #         filemap.loc[mask, "Arrest"] = True

    #     arrest.set(filemap.loc[mask, "Arrest"].values[0])

    @output
    @render_widget
    def volume_plot():
        data_of_point = current_point_filemap().select(
            pl.col(
                [
                    "Time",
                    input.column_to_plot(),
                    worm_type_column,
                    "HatchTime",
                    "M1",
                    "M2",
                    "M3",
                    "M4",
                ]
            )
        )

        times_of_point = data_of_point.select(pl.col("Time")).to_numpy().squeeze()
        values_of_point = (
            data_of_point.select(pl.col(input.column_to_plot())).to_numpy().squeeze()
        )

        markers = set_marker_shape(
            times_of_point,
            int(input.time()),
            values_of_point,
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
                x=times_of_point,
                y=values_of_point,
                mode="markers",
                marker=markers,
            )
        )

        (
            ecdys_list,
            value_at_ecdys_list,
            symbols,
            colors,
            sizes,
            widths,
        ) = get_points_for_value_at_molts(
            hatch(),
            m1(),
            m2(),
            m3(),
            m4(),
            value_at_hatch(),
            value_at_m1(),
            value_at_m2(),
            value_at_m3(),
            value_at_m4(),
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

        if ignore_start() != "":
            if not np.isnan(float(ignore_start())):
                fig.update_layout(
                    shapes=[
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=float(ignore_start()),
                            x1=max(data_of_point["Time"]) + 1,
                            y0=0,
                            y1=1,
                            fillcolor="gray",
                            opacity=0.5,
                            layer="above",
                            line_width=0,
                        )
                    ]
                )

        if death() != "":
            if not np.isnan(float(death())):
                fig.add_shape(
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=float(death()),
                        y0=0,
                        x1=float(death()),
                        y1=1,
                        line=dict(color="black", width=2),
                    )
                )

        if arrest() != "" and arrest():
            fig.add_shape(
                dict(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=0,
                    x1=1,
                    y0=0,
                    y1=1,
                    fillcolor="red",
                    opacity=0.5,
                    layer="above",
                    line_width=0,
                )
            )

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
        update_molt_and_ecdysis_columns(
            "M1", new_m1, int(input.point()), input.column_to_plot()
        )
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
        update_molt_and_ecdysis_columns(
            "M2", new_m2, int(input.point()), input.column_to_plot()
        )
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
        update_molt_and_ecdysis_columns(
            "M3", new_m3, int(input.point()), input.column_to_plot()
        )
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
        update_molt_and_ecdysis_columns(
            "M4", new_m4, int(input.point()), input.column_to_plot()
        )
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
        update_molt_and_ecdysis_columns(
            "M1", np.nan, int(input.point()), input.column_to_plot()
        )
        m1.set(np.nan)
        value_at_m1.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m2)
    def reset_m2():
        print("reset_m2")
        update_molt_and_ecdysis_columns(
            "M2", np.nan, int(input.point()), input.column_to_plot()
        )
        m2.set(np.nan)
        value_at_m2.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m3)
    def reset_m3():
        print("reset_m3")
        update_molt_and_ecdysis_columns(
            "M3", np.nan, int(input.point()), input.column_to_plot()
        )
        m3.set(np.nan)
        value_at_m3.set(np.nan)

    @reactive.Effect
    @reactive.event(input.reset_m4)
    def reset_m4():
        print("reset_m4")
        update_molt_and_ecdysis_columns(
            "M4", np.nan, int(input.point()), input.column_to_plot()
        )
        m4.set(np.nan)
        value_at_m4.set(np.nan)

    @reactive.Effect
    @reactive.event(input.set_value_at_hatch)
    def set_value_at_hatch():
        print("set_value_at_hatch")
        correct_ecdysis_columns(
            "HatchTime", int(input.time()), int(input.point()), input.column_to_plot()
        )
        value_at_hatch.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_HatchTime",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_value_at_m1)
    def set_value_at_m1():
        print("set_value_at_m1")
        correct_ecdysis_columns(
            "M1", int(input.time()), int(input.point()), input.column_to_plot()
        )
        value_at_m1.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M1",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_value_at_m2)
    def set_value_at_m2():
        print("set_value_at_m2")
        correct_ecdysis_columns(
            "M2", int(input.time()), int(input.point()), input.column_to_plot()
        )
        value_at_m2.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M2",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_value_at_m3)
    def set_value_at_m3():
        print("set_value_at_m3")
        correct_ecdysis_columns(
            "M3", int(input.time()), int(input.point()), input.column_to_plot()
        )
        value_at_m3.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M3",
            ].values[0]
        )

    @reactive.Effect
    @reactive.event(input.set_value_at_m4)
    def set_value_at_m4():
        print("set_value_at_m4")
        correct_ecdysis_columns(
            "M4", int(input.time()), int(input.point()), input.column_to_plot()
        )
        value_at_m4.set(
            filemap.loc[
                (filemap["Point"] == int(input.point())),
                f"{input.column_to_plot()}_at_M4",
            ].values[0]
        )

    # @reactive.Effect
    # @reactive.event(input.custom_annotation)
    # def custom_annotation():
    #     new_column_name = input.new_custom_column()
    #     custom_column = input.custom_column()

    #     if new_column_name != "" and (new_column_name not in list_custom_columns):
    #         print("new column")
    #         list_custom_columns.append(new_column_name)
    #         ui.update_selectize(
    #             "custom_column", choices=list_custom_columns, selected=new_column_name
    #         )
    #         ui.update_text("new_custom_column", value="")
    #         filemap[new_column_name] = np.nan
    #         filemap.loc[
    #             (
    #                 (filemap["Point"] == int(input.point()))
    #                 & (filemap["Time"] == int(input.time()))
    #             ),
    #             new_column_name,
    #         ] = float(input.time())

    #         print(
    #             filemap.loc[
    #                 (
    #                     (filemap["Point"] == int(input.point()))
    #                     & (filemap["Time"] == int(input.time()))
    #                 )
    #             ]
    #         )

    #     if new_column_name == "" and custom_column != "":
    #         print("add custom annotation")
    #         filemap.loc[
    #             (
    #                 (filemap["Point"] == int(input.point()))
    #                 & (filemap["Time"] == int(input.time()))
    #             ),
    #             custom_column,
    #         ] = float(input.time())
    #         print(
    #             filemap.loc[
    #                 (
    #                     (filemap["Point"] == int(input.point()))
    #                     & (filemap["Time"] == int(input.time()))
    #                 )
    #             ]
    #         )

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
            segmentation = segmentation_of_point[
                np.where(np.array(times) == int(input.time()))[0][0]
            ]
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
