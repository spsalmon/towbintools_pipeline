from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import polars as pl
import scipy.io as sio
from app_components.app_components import build_single_values_df
from app_components.app_components import correct_ecdysis_columns
from app_components.app_components import get_points_for_value_at_molts
from app_components.app_components import infer_n_channels
from app_components.app_components import open_filemap
from app_components.app_components import populate_column_choices
from app_components.app_components import process_feature_at_molt_columns
from app_components.app_components import save_filemap
from app_components.app_components import set_marker_shape
from app_components.app_components import update_molt_and_ecdysis_columns
from app_components.ui_components import create_molt_annotator
from app_components.ui_components import create_timepoint_selector
from shiny import App
from shiny import module
from shiny import reactive
from shiny import render
from shiny import ui
from shinywidgets import render_widget
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


@module.server
def molt_annotation_buttons_server(
    input,
    output,
    session,
    point_filemap,
    single_values_df_of_point,
    column_to_plot,
    current_time,
    current_time_index,
    molt_time,
    value_at_molt,
    molt_name,
):
    @output
    @render.text
    def text():
        return f"{molt_name} : {str(molt_time())}"

    @reactive.Effect
    @reactive.event(input.set_molt)
    def set_molt():
        new_molt = float(current_time())
        new_molt_index = int(current_time_index())

        single_values_df_of_point.set(
            update_molt_and_ecdysis_columns(
                point_filemap(),
                single_values_df_of_point(),
                molt_name,
                new_molt,
                new_molt_index,
                worm_type_column,
            )
        )
        molt_time.set(new_molt)
        value_at_molt.set(
            single_values_df_of_point()
            .select(pl.col(f"{column_to_plot()}_at_{molt_name}"))
            .to_numpy()
            .squeeze()
        )

    @reactive.Effect
    @reactive.event(input.reset_molt)
    def reset_molt():
        new_molt = np.nan
        new_molt_index = int(current_time_index())

        single_values_df_of_point.set(
            update_molt_and_ecdysis_columns(
                point_filemap(),
                single_values_df_of_point(),
                molt_name,
                new_molt,
                new_molt_index,
                worm_type_column,
            )
        )
        molt_time.set(new_molt)
        value_at_molt.set(
            single_values_df_of_point()
            .select(pl.col(f"{column_to_plot()}_at_{molt_name}"))
            .to_numpy()
            .squeeze()
        )

    @reactive.Effect
    @reactive.event(input.set_value_at_molt)
    def set_value_at_molt():
        index = int(current_time_index())
        single_values_df_of_point.set(
            correct_ecdysis_columns(
                point_filemap(), single_values_df_of_point(), molt_name, index
            )
        )
        value_at_molt.set(
            single_values_df_of_point()
            .select(pl.col(f"{column_to_plot()}_at_{molt_name}"))
            .to_numpy()
            .squeeze()
        )


@module.server
def time_point_navigator_server(
    input,
    output,
    session,
    clicked_value=None,
    choices=[],
    all_df=None,
    current_df=None,
    save_on_switch=False,
):
    index = reactive.Value("")
    value = reactive.Value("")

    def save():
        if all_df is None or current_df is None:
            raise ValueError("You need to specify which dataframe to save")
        if index() != "":
            print(f"previous values at index {index()}: {all_df[index()]}")
            all_df[index()] = current_df()
            print(f"updated values at index {index()}: {all_df[index()]}")

    @reactive.Effect
    def update_on_click():
        if clicked_value is not None:
            if clicked_value() == "":
                return
            print(f"Clicked value: {clicked_value()}")
            ui.update_selectize("current", selected=clicked_value())

    @reactive.Effect
    @reactive.event(input.previous)
    def previous():
        current = int(input.current())
        previous_index = max(np.where(np.array(choices) == current)[0][0] - 1, 0)
        previous_time = choices[previous_index]
        ui.update_selectize("current", selected=str(int(previous_time)))

    @reactive.Effect
    @reactive.event(input.next)
    def next():
        current = int(input.current())
        next_index = min(
            np.where(np.array(choices) == current)[0][0] + 1, len(choices) - 1
        )
        next_time = choices[next_index]
        ui.update_selectize("current", selected=str(int(next_time)))

    @reactive.Effect
    @reactive.event(input.current)
    def update_current():
        if save_on_switch:
            save()

        value.set(input.current())
        index.set(np.where(np.array(choices) == int(input.current()))[0][0])

    return (index, value)


print("Initializing the UI ...")

molt_annotator = create_molt_annotator(ECDYSIS_COLUMNS, custom_columns_choices)
timepoint_selector = create_timepoint_selector(
    list_channels,
    times,
    points,
    feature_columns,
    overlay_segmentation_choices,
    default_plotted_column,
)
app_ui = ui.page_fluid(ui.row(molt_annotator, timepoint_selector))


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
    global custom_columns_choices
    point_filemaps = filemap.partition_by("Point", maintain_order=True)
    work_df = filemap.select((pl.col("Point"), pl.col("Time"))).cast(pl.Int32)
    work_df = reactive.Value(work_df)
    single_values_df = build_single_values_df(filemap)

    single_values_of_points = single_values_df.partition_by(
        "Point", maintain_order=True
    )

    custom_columns_list = reactive.Value(custom_columns_choices)
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

    clicked_time = reactive.Value("")

    current_point_filemap = reactive.Value("")
    single_values_of_point = reactive.Value("")

    custom_column_values = reactive.Value([])
    column_to_plot = reactive.Value(default_plotted_column)

    (current_point_index, current_point) = time_point_navigator_server(
        "point",
        choices=points,
        all_df=single_values_of_points,
        current_df=single_values_of_point,
        save_on_switch=True,
    )
    (current_time_index, current_time) = time_point_navigator_server(
        "time", choices=times, clicked_value=clicked_time
    )

    [
        molt_annotation_buttons_server(
            molt_name,
            current_point_filemap,
            single_values_of_point,
            column_to_plot,
            current_time,
            current_time_index,
            molt_time,
            value_at_molt,
            molt_name=molt_name,
        )
        for molt_name, molt_time, value_at_molt in zip(
            ECDYSIS_COLUMNS,
            [hatch, m1, m2, m3, m4],
            [value_at_hatch, value_at_m1, value_at_m2, value_at_m3, value_at_m4],
        )
    ]

    @reactive.Effect
    def update_point_filemap():
        current_point_filemap.set(point_filemaps[current_point_index()])
        single_values_of_point.set(single_values_of_points[current_point_index()])

    @reactive.Effect
    def update_column_to_plot():
        column_to_plot.set(input.column_to_plot())

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

    @reactive.calc
    def get_custom_annotations():
        if input.custom_column() == "" or input.custom_column() == "None":
            return []

        return (
            work_df()
            .select(input.custom_column())
            .to_numpy()
            .squeeze()
            .astype(float)
            .tolist()
        )

    @reactive.Effect
    @reactive.event(input.save)
    def save():
        save_filemap(filemap)

    @reactive.Effect
    def get_hatch_and_molts():
        start = perf_counter()
        # Get all molt times
        hatch_and_molts = (
            single_values_of_point()
            .select(pl.col(ECDYSIS_COLUMNS))
            .to_numpy()
            .squeeze()
        )

        # Set molt times using unpacking
        molt_values = [hatch, m1, m2, m3, m4]
        for i, molt_value in enumerate(molt_values):
            molt_value.set(hatch_and_molts[i])

        # Get values at each molt
        try:
            column = input.column_to_plot()
            molt_feature_values = [
                value_at_hatch,
                value_at_m1,
                value_at_m2,
                value_at_m3,
                value_at_m4,
            ]

            for i, (molt_value, molt_name) in enumerate(
                zip(molt_feature_values, ECDYSIS_COLUMNS)
            ):
                molt_value.set(
                    single_values_of_point()
                    .select(pl.col(f"{column}_at_{molt_name}"))
                    .to_numpy()
                    .squeeze()
                )
        except Exception as e:
            print(f"Exception caught while getting value at hatch and molts: {e}")
            for molt_value in molt_feature_values:
                molt_value.set(np.nan)

        print(f"Getting hatch and molts took {perf_counter() - start:.2f} seconds")

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
        overall_start = perf_counter()
        start = perf_counter()
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
        print(f"Getting data of point took {perf_counter() - start:.2f} seconds")

        times_of_point = data_of_point.select(pl.col("Time")).to_numpy().squeeze()
        values_of_point = (
            data_of_point.select(pl.col(input.column_to_plot())).to_numpy().squeeze()
        )
        worm_types_of_point = (
            data_of_point.select(pl.col(worm_type_column)).to_numpy().squeeze()
        )

        markers = set_marker_shape(
            times_of_point,
            current_time_index(),
            worm_types_of_point,
            hatch(),
            m1(),
            m2(),
            m3(),
            m4(),
            custom_annotations=custom_column_values(),
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
                clicked_time.set(str(point))

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

        print(f"Creating volume plot took {perf_counter() - overall_start:.2f} seconds")
        return fig

    @reactive.Effect
    def get_custom_column_values():
        print("get_custom_column_values")
        print(f"custom_column: {input.custom_column()}")
        custom_column = input.custom_column()

        local_work_df = work_df()
        work_df_columns = local_work_df.columns

        if custom_column == "None" or custom_column == "":
            custom_column_values.set([])
            return
        if custom_column not in work_df_columns:
            local_work_df = local_work_df.with_columns(
                pl.lit(np.nan).alias(custom_column)
            )
            work_df.set(local_work_df)
            custom_column_values.set([])
        else:
            values = (
                local_work_df.filter(pl.col("Point") == int(input.point()))
                .select(pl.col(custom_column))
                .to_numpy()
                .squeeze()
                .tolist()
            )
            print(f"new values: {values}")
            custom_column_values.set(values)

        print(f"Custom column values: {custom_column_values()}")

    @reactive.Effect
    @reactive.event(input.custom_annotation)
    def set_custom_annotation():
        new_column_name = input.new_custom_column()
        custom_column = input.custom_column()

        local_work_df = work_df()

        filemap_columns = filemap.columns

        custom_columns = custom_columns_list()

        if new_column_name != "" and (new_column_name not in custom_columns):
            custom_columns.append(new_column_name)
            custom_columns_list.set(custom_columns)

            ui.update_selectize(
                "custom_column",
                choices=custom_columns_list(),
                selected=new_column_name,
            )
            ui.update_text("new_custom_column", value="")

        if custom_column == "" or custom_column == "None":
            return

        if new_column_name not in filemap_columns:
            local_work_df.with_columns(pl.lit(np.nan).alias(new_column_name))
        else:
            # set the column in work df how it is in filemap
            new_column_values = (
                filemap.select(pl.col(new_column_name)).to_numpy().squeeze()
            )
            local_work_df.with_columns(pl.lit(new_column_values).alias(new_column_name))

        local_work_df = local_work_df.with_columns(
            pl.when(pl.col("Point") == int(input.point()))
            .when(pl.col("Time") == int(input.time()))
            .then(float(input.time()))
            .otherwise(pl.col(custom_column))
            .alias(new_column_name),
        )
        work_df.set(local_work_df)

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
        start = perf_counter()
        idx = current_time_index()
        channel_str = input.channel()
        channel = int(channel_str.split(" ")[-1]) - 1

        images_of_point = get_images_of_point()
        segmentation_of_point = get_segmentation_of_point()

        img = images_of_point[idx]
        img = image_handling.read_tiff_file(img)

        # Select the correct channel/slice
        if img.ndim == 3:
            img_to_plot = img[channel]
        elif img.ndim == 4:
            img_to_plot = img[img.shape[0] // 2, channel, ...]
        elif img.ndim == 2:
            img_to_plot = img
        else:
            raise ValueError("Unexpected image dimensions")

        channel_overlay_str = input.channel_overlay()
        plot_overlay = channel_overlay_str != "None"
        if plot_overlay:
            channel_overlay = int(channel_overlay_str.split(" ")[-1]) - 1
            overlay = img[channel_overlay]
            overlay = image_handling.normalize_image(overlay, dest_dtype=np.float64)
        else:
            overlay = None  # Only create if needed

        img_to_plot = image_handling.normalize_image(img_to_plot, dest_dtype=np.float64)

        fig, ax = plt.subplots()
        ax.imshow(img_to_plot, cmap="viridis")
        if plot_overlay:
            ax.imshow(overlay, cmap="magma", alpha=0.7)

        if len(segmentation_of_point) > 0:
            segmentation_path = segmentation_of_point[idx]
            segmentation = image_handling.read_tiff_file(segmentation_path)
            ax.imshow(segmentation.squeeze(), cmap="gray", alpha=0.3)

        # disable axis
        ax.axis("off")

        print(f"Plotting took {perf_counter() - start:.3f} seconds")
        return fig

    session.on_ended(save_filemap)

    @reactive.Effect
    @reactive.event(input.close)
    async def _():
        await session.close()


print("Creating the app ...")
app = App(app_ui, server)
