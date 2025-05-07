import numpy as np
import polars as pl
from app_components.app_components import correct_ecdysis_columns
from app_components.app_components import update_molt_and_ecdysis_columns
from shiny import module
from shiny import reactive
from shiny import render
from shiny import ui
from shinywidgets import output_widget


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


def create_molt_annotator(ecdysis_list_id, custom_columns_choices):
    return ui.column(
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
        ui.output_plot("plot", height="60vh"),
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
