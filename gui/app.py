from app_components.app_components import open_filemap
from app_components.ui_components import initialize_ui
from app_components.ui_components import main_server
from shiny import App

recompute_features_at_molt = True

filemap_path = "/mnt/towbin.data/shared/spsalmon/20250314_squid_10x_yap_aid_160_438_492_493/analysis/report/analysis_filemap.csv"

filemap, filemap_save_path = open_filemap(filemap_path, open_annotated=True)

print("Creating the app ...")
(
    app_ui,
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
