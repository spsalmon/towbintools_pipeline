from app_components.backend import open_filemap
from app_components.server import main_server
from app_components.ui import initialize_ui
from shiny import App

recompute_features_at_molt = False

filemap_path = r"/mnt/towbin.data/shared/plenart/20261403_squid_10x_wBT318_heath_stress_27_degree/analysis/report/analysis_filemap.csv"

print("Opening the filemap ...")

filemap, filemap_save_path = open_filemap(
    filemap_path, open_annotated=True, lazy_loading=False
)

print("Creating the app ...")

(
    app_ui,
    filemap,
    feature_columns,
    custom_columns_choices,
    points,
    times,
    default_plotted_column,
) = initialize_ui(filemap, recompute_features_at_molt=recompute_features_at_molt)


def s(input, output, session):
    return main_server(
        input,
        output,
        session,
        filemap=filemap,
        filemap_save_path=filemap_save_path,
        feature_columns=feature_columns,
        custom_columns_choices=custom_columns_choices,
        points=points,
        times=times,
        default_plotted_column=default_plotted_column,
    )


app = App(app_ui, s)
