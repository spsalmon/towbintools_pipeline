import os

from app_components.backend import open_filemap
from app_components.server import main_server
from app_components.ui import initialize_ui
from shiny import App

recompute_features_at_molt = os.environ.get("RECOMPUTE_FEATURES", "0") == "1"
open_annotated = os.environ.get("OPEN_ANNOTATED", "1") == "1"
filemap_path = os.environ.get("FILEMAP_PATH", "")

print("Opening the filemap ...")
filemap, filemap_save_path = open_filemap(
    filemap_path, open_annotated=open_annotated, lazy_loading=False
)
print("Creating the app ...")
(
    app_ui,
    filemap,
    raw_column,
    feature_columns,
    custom_columns_choices,
    points,
    times,
    default_plotted_column,
) = initialize_ui(filemap, recompute_features_at_molt=recompute_features_at_molt)


def server(input, output, session):
    return main_server(
        input,
        output,
        session,
        filemap=filemap,
        filemap_save_path=filemap_save_path,
        raw_column=raw_column,
        feature_columns=feature_columns,
        custom_columns_choices=custom_columns_choices,
        points=points,
        times=times,
        default_plotted_column=default_plotted_column,
    )


app = App(app_ui, server)
