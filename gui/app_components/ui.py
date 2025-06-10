import traceback

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import polars as pl
import scipy.io as sio
from app_components.backend import build_single_values_df
from app_components.backend import check_use_experiment_time
from app_components.backend import correct_ecdysis_columns
from app_components.backend import get_points_for_value_at_molts
from app_components.backend import infer_n_channels
from app_components.backend import populate_column_choices
from app_components.backend import process_feature_at_molt_columns
from app_components.backend import set_marker_shape
from app_components.backend import update_molt_and_ecdysis_columns
from polars.exceptions import ColumnNotFoundError
from shiny import module
from shiny import reactive
from shiny import render
from shiny import ui
from shinywidgets import output_widget
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


@module.ui
def molt_annotation_buttons(molt, width=2):
    return ui.column(
        width,
        ui.row(ui.output_text("text")),
        ui.row(ui.input_action_button("set_molt", f"{molt}")),
        ui.row(ui.input_action_button("reset_molt", f"Reset {molt}")),
        ui.row(ui.input_action_button("set_value_at_molt", f"Set value at {molt}")),
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
    worm_type_column,
    use_experiment_time=False,
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
                experiment_time=use_experiment_time,
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
                experiment_time=use_experiment_time,
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


def create_molt_annotator(ecdysis_list_id, custom_columns_choices):
    return ui.column(
        7,
        ui.row(output_widget("plot_curve")),
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
                    "import_file",
                    "Import Molts",
                    accept=[".csv", ".mat"],
                    multiple=False,
                ),
            ),
            align="center",
        ),
        align="center",
    )


@module.ui
def time_point_navigator(width=4, height="15vh", name="", choices=[]):
    return ui.row(
        ui.column(
            4, ui.input_action_button("previous", f"previous {name}", height="15vh")
        ),
        ui.column(4, ui.input_selectize("current", f"Select {name}", choices=choices)),
        ui.column(4, ui.input_action_button("next", f"next {name}", height="15vh")),
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
        if all_df() is None or current_df is None:
            raise ValueError("You need to specify which dataframe to save")
        if index() != "":
            df_list = all_df()
            df_list[index()] = current_df()
            all_df.set(df_list)

    @reactive.Effect
    def update_on_click():
        if clicked_value is not None:
            if clicked_value() == "":
                return
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


def create_timepoint_selector(
    list_channels,
    times,
    points,
    feature_columns,
    overlay_segmentation_choices,
    default_plotted_column,
):
    return ui.column(
        5,
        ui.output_plot("plot_image", height="60vh"),
        time_point_navigator(
            "time", width=4, height="15vh", name="time", choices=times
        ),
        time_point_navigator(
            "point", width=4, height="15vh", name="point", choices=points
        ),
        ui.row(
            ui.input_selectize(
                "channel",
                "Select channel",
                choices=list_channels,
                selected=list_channels[1],
            ),
            ui.input_selectize(
                "channel_overlay",
                "Overlay channel",
                choices=list_channels,
                selected="None",
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


def initialize_ui(filemap, recompute_features_at_molt=False):
    print("Initializing the UI ...")

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

    (
        feature_columns,
        custom_columns_choices,
        worm_type_column,
        default_plotted_column,
        overlay_segmentation_choices,
    ) = populate_column_choices(filemap)

    filemap = process_feature_at_molt_columns(
        filemap,
        feature_columns,
        worm_type_column,
        recompute_features_at_molt=recompute_features_at_molt,
    )

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

    return (
        app_ui,
        filemap,
        feature_columns,
        custom_columns_choices,
        points,
        times,
        worm_type_column,
        default_plotted_column,
    )


def main_server(
    input,
    output,
    session,
    filemap=None,
    filemap_save_path=None,
    points=None,
    times=None,
    feature_columns=None,
    custom_columns_choices=None,
    worm_type_column=None,
    default_plotted_column=None,
):
    use_experiment_time = check_use_experiment_time(filemap)
    point_filemaps = filemap.partition_by("Point", maintain_order=True)

    work_df_columns = [
        "Time",
        "Point",
        "Death",
        "Ignore",
        "Arrest",
    ] + custom_columns_choices

    columns_to_get = [c for c in work_df_columns if c in filemap.columns]
    # Remove duplicates while preserving order
    seen = set()
    columns_to_get_unique = []
    for c in columns_to_get:
        if c not in seen:
            columns_to_get_unique.append(c)
            seen.add(c)

    print(f"Columns to get: {columns_to_get_unique}")
    work_df = filemap.select(pl.col(columns_to_get_unique))
    work_df = reactive.Value(work_df)

    single_values_df = build_single_values_df(filemap)

    single_values_of_points = single_values_df.partition_by(
        "Point", maintain_order=True
    )

    single_values_of_points = reactive.Value(single_values_of_points)

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
            worm_type_column=worm_type_column,
            use_experiment_time=use_experiment_time,
        )
        for molt_name, molt_time, value_at_molt in zip(
            ECDYSIS_COLUMNS,
            [hatch, m1, m2, m3, m4],
            [value_at_hatch, value_at_m1, value_at_m2, value_at_m3, value_at_m4],
        )
    ]

    # Create non-reactive holders for the latest values
    latest_point = {"value": None}
    latest_single_values = {"value": None}
    latest_single_values_df = {"value": None}
    latest_work_df = {"value": None}

    @reactive.Effect
    def cache_latest_for_save():
        latest_point["value"] = current_point_index()
        latest_single_values["value"] = single_values_of_point()
        latest_single_values_df["value"] = single_values_of_points()
        latest_work_df["value"] = work_df()

    @reactive.Effect
    def update_point_filemap():
        current_point_filemap.set(point_filemaps[current_point_index()])
        single_values_of_point.set(single_values_of_points()[current_point_index()])

    @reactive.Effect
    def update_column_to_plot():
        column_to_plot.set(input.column_to_plot())

    @reactive.Effect
    @reactive.event(input.import_file)
    def import_molts():
        file = input.import_file()
        point = int(current_point_index())
        if file is None:
            return

        (
            _,
            custom_columns_choices,
            worm_type_column,
            _,
            _,
        ) = populate_column_choices(filemap)

        datapath = file[0]["datapath"]
        if datapath.endswith(".csv"):
            print(f"Importing molts from {datapath} ...")
            imported_df = pl.read_csv(
                datapath,
                infer_schema_length=10000,
                null_values=["np.nan", "NaN", "[nan]", ""],
            )

            work_df_columns = [
                "Time",
                "Point",
                "Death",
                "Ignore",
                "Arrest",
            ] + custom_columns_choices

            missing_work_df_columns = [
                col for col in work_df_columns if col not in imported_df.columns
            ]

            for col in missing_work_df_columns:
                if col in work_df().columns:
                    default_value = work_df().select(pl.col(col)).to_numpy().squeeze()
                else:
                    if col == "Death":
                        default_value = np.nan
                    elif col == "Ignore":
                        default_value = False
                    elif col == "Arrest":
                        default_value = False
                    else:
                        default_value = np.nan

                imported_df = imported_df.with_columns(pl.lit(default_value).alias(col))

            # Find missing feature-at-molt columns
            current_single_values_df = single_values_of_point()
            missing_columns = [
                col
                for col in current_single_values_df.columns
                if col not in imported_df.columns
            ]

            imported_df = process_feature_at_molt_columns(
                imported_df,
                missing_columns,
                worm_type_column,
                recompute_features_at_molt=False,
            )

        elif datapath.endswith(".mat"):
            matlab_report = sio.loadmat(datapath, chars_as_strings=False)
            new_matlab_report = {}
            for key, value in matlab_report.items():
                if key.startswith("__"):
                    continue
                try:
                    new_key = KEY_CONVERSION_MAP.get(key)
                    new_matlab_report[new_key] = value
                except Exception as e:
                    print(f"Error converting key {key}: {e}")

            try:
                ecdysis = new_matlab_report["ecdysis"]
                ecdysis_df = pl.DataFrame(
                    {
                        "Point": range(len(ecdysis)),
                        **{
                            column: [
                                (
                                    float(molts[i]) - 1
                                    if not np.isnan(molts[i])
                                    else np.nan
                                )
                                for molts in ecdysis
                            ]
                            for i, column in enumerate(ECDYSIS_COLUMNS)
                        },
                    }
                )

                updated_filemap = filemap.drop(ECDYSIS_COLUMNS).join(
                    ecdysis_df, on="Point", how="left"
                )
            except KeyError:
                print("No molts found in the matlab report")
                return

            imported_df = process_feature_at_molt_columns(
                updated_filemap,
                feature_columns,
                worm_type_column,
                recompute_features_at_molt=True,
            )

        import_single_values_df = build_single_values_df(imported_df)
        import_single_values_df = import_single_values_df.partition_by(
            "Point", maintain_order=True
        )
        single_values_of_points.set(import_single_values_df)
        single_values_of_point.set(import_single_values_df[point])

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

    def save_filemap(
        filemap,
        filemap_save_path,
        current_index,
        current_single_values_df,
        single_values_df,
        work_df,
    ):
        print(f"Saving filemap to {filemap_save_path} ...")

        # because the list of single values is only updated when the point is changed, change it here to make sure to include the latest changes
        single_values_df[current_index] = current_single_values_df

        single_values_columns = single_values_df[0].columns
        single_values_columns.remove("Point")
        columns_to_drop = [c for c in single_values_columns if c in filemap.columns]
        if len(columns_to_drop) > 0:
            filemap_to_save = filemap.drop(columns_to_drop)
        else:
            filemap_to_save = filemap

        work_df_columns = work_df.columns
        work_df_columns.remove("Point")
        work_df_columns.remove("Time")

        columns_to_drop = [c for c in work_df_columns if c in filemap.columns]
        if len(columns_to_drop) > 0:
            filemap_to_save = filemap_to_save.drop(columns_to_drop)

        # combine the single_values_df into a big dataframe
        combined_df = pl.concat(single_values_df, how="vertical")

        filemap_to_save = filemap_to_save.join(combined_df, on=["Point"], how="left")

        filemap_to_save = filemap_to_save.join(
            work_df, on=["Point", "Time"], how="left"
        )

        filemap_to_save.write_csv(filemap_save_path, null_value="")
        print("Filemap saved!")

    @reactive.Effect
    @reactive.event(input.save)
    def save():
        save_filemap(
            filemap=filemap,
            current_index=current_point_index(),
            current_single_values_df=single_values_of_point(),
            filemap_save_path=filemap_save_path,
            single_values_df=single_values_of_points(),
            work_df=work_df(),
        )

    @reactive.Effect
    def get_hatch_and_molts():
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

    @reactive.Effect
    def set_ignore_start():
        try:
            df = work_df().filter(pl.col("Point") == int(current_point()))
            print(df)
            ignore_values = df.select(pl.col("Ignore")).to_numpy().squeeze()
            time_values = df.select(pl.col("Time")).to_numpy().squeeze()
            # find the first True value
            true_indices = np.where(ignore_values)[0]
            if true_indices.size > 0:
                ignore_start.set(time_values[true_indices[0]])
            else:
                ignore_start.set("")

        except ColumnNotFoundError:
            ignore_start.set("")

        print(f"Ignore start: {ignore_start()}")

    @reactive.Effect
    @reactive.event(input.set_ignore_point)
    def set_ignore_point():
        point = int(current_point())
        df = work_df()

        print(f"Setting ignore for point {point}")
        if "Ignore" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("Ignore"))

        if (
            df.filter(pl.col("Point") == point)
            .select(pl.col("Ignore"))
            .to_numpy()
            .all()
        ):
            work_df.set(
                df.with_columns(
                    pl.when(pl.col("Point") == point)
                    .then(False)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )
        else:
            work_df.set(
                df.with_columns(
                    pl.when(pl.col("Point") == point)
                    .then(True)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )

        print(work_df().filter(pl.col("Point") == point).select(pl.col("Ignore")))

    @reactive.Effect
    @reactive.event(input.set_ignore_after)
    def set_ignore_after():
        point = int(current_point())
        time = int(current_time())
        df = work_df()
        print(f"Setting ignore after for point {point}")
        if "Ignore" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("Ignore"))

        if (
            df.filter((pl.col("Point") == point) & (pl.col("Time") >= time))
            .select(pl.col("Ignore"))
            .to_numpy()
            .all()
        ):
            work_df.set(
                df.with_columns(
                    pl.when((pl.col("Point") == point) & (pl.col("Time") >= time))
                    .then(False)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )
        else:
            work_df.set(
                df.with_columns(
                    pl.when((pl.col("Point") == point) & (pl.col("Time") >= time))
                    .then(True)
                    .otherwise(pl.col("Ignore"))
                    .alias("Ignore")
                )
            )

        print(work_df().filter(pl.col("Point") == point).select(pl.col("Ignore")))

    @reactive.Effect
    def get_death():
        try:
            death.set(
                work_df()
                .filter(pl.col("Point") == int(current_point()))
                .select(pl.col("Death"))
                .to_numpy()
                .squeeze()[0]
            )
        except ColumnNotFoundError:
            death.set("")

    @reactive.Effect
    @reactive.event(input.set_death)
    def set_death():
        point = int(current_point())
        time = int(current_time())

        df = work_df()

        if "Death" not in df.columns:
            df = df.with_columns(pl.lit(np.nan).alias("Death"))

        point_df = df.filter(pl.col("Point") == point)

        if point_df.select(pl.col("Death")).to_numpy().squeeze()[0] == time:
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(np.nan)
                .otherwise(pl.col("Death"))
                .alias("Death")
            )

        else:
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(time)
                .otherwise(pl.col("Death"))
                .alias("Death")
            )

        work_df.set(df)

    @reactive.Effect
    @reactive.event(input.set_arrest)
    def set_arrest():
        point = int(current_point())

        df = work_df()

        if "Arrest" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("Arrest"))

        point_df = df.filter(pl.col("Point") == point)

        if np.all(point_df.select(pl.col("Arrest")).to_numpy().squeeze()):
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(False)
                .otherwise(pl.col("Arrest"))
                .alias("Arrest")
            )

        else:
            df = df.with_columns(
                pl.when(pl.col("Point") == point)
                .then(True)
                .otherwise(pl.col("Arrest"))
                .alias("Arrest")
            )

        work_df.set(df)

    @reactive.Effect
    def get_arrest():
        try:
            arrest.set(
                work_df()
                .filter(pl.col("Point") == int(current_point()))
                .select(pl.col("Arrest"))
                .to_numpy()
                .squeeze()[0]
            )
        except ColumnNotFoundError:
            arrest.set(False)

    @reactive.Effect
    def reactive_get_custom_column_values():
        print("reactive_get_custom_column_values")
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
            custom_column_values.set(
                get_custom_column_values(
                    local_work_df, custom_column, int(current_point())
                )
            )
        print(f"Custom column values: {custom_column_values()}")

    def get_custom_column_values(df, column, point):
        if column == "None" or column == "":
            return []
        if column not in df.columns:
            df = df.with_columns(pl.lit(np.nan).alias(column))
        intermediate_df = df.filter(pl.col("Point") == point)

        # convert the column to index in the time values
        time = intermediate_df.select(pl.col("Time")).to_numpy().squeeze()
        column_values = intermediate_df.select(pl.col(column)).to_numpy().squeeze()

        non_nan_column_values = column_values[~np.isnan(column_values)]
        column_values_as_indexes = np.full_like(non_nan_column_values, np.nan)
        for i, value in enumerate(non_nan_column_values):
            index = np.where(time == value)[0]
            if index.size > 0:
                column_values_as_indexes[i] = index[0]
        return column_values_as_indexes

    @reactive.Effect
    @reactive.event(input.custom_annotation)
    def set_custom_annotation():
        try:
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
                custom_column = new_column_name

            if custom_column == "" or custom_column == "None":
                return

            if custom_column not in filemap_columns:
                local_work_df = local_work_df.with_columns(
                    pl.lit(np.nan).alias(new_column_name)
                )
            elif custom_column not in local_work_df.columns:
                new_column_values = (
                    filemap.select(pl.col(custom_column)).to_numpy().squeeze()
                )
                local_work_df = local_work_df.with_columns(
                    pl.lit(new_column_values).alias(custom_column)
                )

            time = int(current_time())
            point = int(current_point())

            local_work_df = local_work_df.with_columns(
                pl.when((pl.col("Point") == point) & (pl.col("Time") == time))
                .then(float(time))
                .otherwise(pl.col(custom_column))
                .alias(custom_column),
            )
            work_df.set(local_work_df)

        except Exception as e:
            print(f"Exception in set_custom_annotation: {e}")
            traceback.print_exc()

    @reactive.Effect
    @reactive.event(input.reset_custom_annotation)
    def reset_custom_annotation():
        custom_column = input.custom_column()
        if custom_column != "":
            work_df.set(
                work_df().with_columns(
                    pl.when(pl.col("Point") == int(current_point()))
                    .then(np.nan)
                    .otherwise(pl.col(custom_column))
                    .alias(custom_column),
                )
            )

    @output
    @render.plot
    def plot_image():
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

        return fig

    @output
    @render_widget
    def plot_curve():
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
        worm_types_of_point = (
            data_of_point.select(pl.col(worm_type_column)).to_numpy().squeeze()
        )

        print(f"Custom annotations: {custom_column_values()}")
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
        return fig

    def save_on_session_end():
        save_filemap(
            filemap=filemap,
            current_index=latest_point["value"],
            current_single_values_df=latest_single_values["value"],
            filemap_save_path=filemap_save_path,
            single_values_df=latest_single_values_df["value"],
            work_df=latest_work_df["value"],
        )

    session.on_ended(save_on_session_end)
