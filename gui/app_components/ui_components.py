from shiny import module
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
