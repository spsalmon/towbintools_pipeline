import numpy as np
import polars as pl
from app_components.backend import correct_ecdysis_columns
from app_components.backend import update_molt_and_ecdysis_columns
from shiny import module
from shiny import reactive
from shiny import render
from shiny import ui


# annotation buttons
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


# time and point navigator
@module.ui
def time_point_navigator(name="", choices=[]):
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
            df_list[int(index())] = current_df()
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
