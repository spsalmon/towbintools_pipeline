from app_components.backend import open_filemap
from app_components.ui import initialize_ui
from app_components.ui import main_server
from shiny import App

recompute_features_at_molt = False

filemap_path = "/mnt/towbin.data/shared/fdell/20250704_Kinetix_wBT525_wBT553_ribosomaldecay/analysis/report/analysis_filemap.csv"

filemap, filemap_save_path = open_filemap(filemap_path, open_annotated=True)

print("Creating the app ...")
(
    app_ui,
    filemap,
    feature_columns,
    custom_columns_choices,
    points,
    times,
    worm_type_column,
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
        worm_type_column=worm_type_column,
        default_plotted_column=default_plotted_column,
    )


app = App(app_ui, s)
