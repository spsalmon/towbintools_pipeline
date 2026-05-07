import numpy as np
import polars as pl
from app_components.backend import build_single_values_df
from app_components.backend import populate_column_choices


def make_minimal_filemap():
    """Filemap with only Time, Point, raw — as produced by a barely-processed experiment."""
    return pl.DataFrame(
        {
            "Time": [0, 1, 2, 0, 1, 2],
            "Point": [0, 0, 0, 1, 1, 1],
            "raw": [
                "/fake/0_0.tif",
                "/fake/0_1.tif",
                "/fake/0_2.tif",
                "/fake/1_0.tif",
                "/fake/1_1.tif",
                "/fake/1_2.tif",
            ],
        }
    )


def test_populate_column_choices_creates_qc_placeholder():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    assert "placeholder_qc" in result_filemap.columns
    qc_values = result_filemap.select("placeholder_qc").to_numpy().squeeze()
    assert all(v == "worm" for v in qc_values)


def test_populate_column_choices_placeholder_feature_is_one():
    filemap = make_minimal_filemap()
    result_filemap, _, feature_columns, _, default_plotted_column, _ = (
        populate_column_choices(filemap)
    )
    assert "placeholder_feature" in result_filemap.columns
    values = (
        result_filemap.select("placeholder_feature").to_numpy().squeeze().astype(float)
    )
    assert np.all(values == 1.0), f"Expected all 1.0, got {values}"


def test_populate_column_choices_feature_columns_not_empty():
    filemap = make_minimal_filemap()
    _, _, feature_columns, _, default_plotted_column, _ = populate_column_choices(
        filemap
    )
    assert len(feature_columns) > 0, "feature_columns must not be empty"
    assert "placeholder_feature" in feature_columns
    assert default_plotted_column == "placeholder_feature"


def test_build_single_values_df_no_ecdysis_columns():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    # Should not raise even though HatchTime, M1, M2, M3, M4 may be absent
    try:
        df = build_single_values_df(result_filemap)
    except Exception as e:
        raise AssertionError(f"build_single_values_df raised with minimal filemap: {e}")
    assert "Point" in df.columns
