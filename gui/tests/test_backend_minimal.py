import numpy as np
import polars as pl
from app_components.backend import build_single_values_df
from app_components.backend import MOLT_ENTRY_COLUMNS
from app_components.backend import populate_column_choices
from app_components.backend import process_feature_at_molt_columns


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


# --- populate_column_choices ---


def test_populate_column_choices_creates_qc_placeholder():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    assert "placeholder_qc" in result_filemap.columns
    qc_values = result_filemap.select("placeholder_qc").to_numpy().squeeze()
    assert all(v == "worm" for v in qc_values)


def test_populate_column_choices_placeholder_feature_is_one():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
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


def test_populate_column_choices_segmentation_overlay_is_none_only():
    filemap = make_minimal_filemap()
    _, _, _, _, _, overlay_seg = populate_column_choices(filemap)
    assert overlay_seg == ["None"]


# --- build_single_values_df ---


def test_build_single_values_df_without_ecdysis_columns():
    """Must not raise even though HatchTime, M1, M2, M3, M4 are absent."""
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    df = build_single_values_df(result_filemap)
    assert "Point" in df.columns


def test_build_single_values_df_adds_ecdysis_columns_as_nan():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    df = build_single_values_df(result_filemap)
    for col in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert col in df.columns
        values = df.select(col).to_numpy().squeeze().astype(float)
        assert np.all(np.isnan(values)), f"{col} should be all NaN, got {values}"


def test_build_single_values_df_one_row_per_point():
    filemap = make_minimal_filemap()
    result_filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result_filemap = process_feature_at_molt_columns(result_filemap, feature_columns)
    df = build_single_values_df(result_filemap)
    n_points = filemap.select("Point").unique().height
    assert df.height == n_points


# --- process_feature_at_molt_columns ---


def test_process_feature_at_molt_columns_adds_ecdysis_columns():
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for col in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert col in result.columns


def test_process_feature_at_molt_columns_adds_feature_at_molt_columns():
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for ecdys in ["HatchTime", "M1", "M2", "M3", "M4"]:
        col = f"placeholder_feature_at_{ecdys}"
        assert col in result.columns


def test_full_startup_pipeline_does_not_crash():
    """Simulate the full initialize_ui backend flow (minus infer_n_channels)."""
    filemap = make_minimal_filemap()
    (
        filemap,
        raw_column,
        feature_columns,
        custom_columns_choices,
        default_plotted_column,
        overlay_seg,
    ) = populate_column_choices(filemap)
    filemap = process_feature_at_molt_columns(filemap, feature_columns)
    single_values_df = build_single_values_df(filemap)

    assert raw_column == "raw"
    assert feature_columns == ["placeholder_feature"]
    assert default_plotted_column == "placeholder_feature"
    assert overlay_seg == ["None"]
    assert "Point" in single_values_df.columns


def test_build_single_values_df_adds_entry_columns_as_nan():
    filemap = make_minimal_filemap()
    result_filemap, *_ = populate_column_choices(filemap)
    df = build_single_values_df(result_filemap)
    for col in MOLT_ENTRY_COLUMNS:
        assert col in df.columns
        values = df.select(col).to_numpy().squeeze().astype(float)
        assert np.all(np.isnan(values)), f"{col} should be all NaN, got {values}"


def test_process_feature_at_molt_columns_adds_feature_at_entry_columns():
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for entry in MOLT_ENTRY_COLUMNS:
        col = f"placeholder_feature_at_{entry}"
        assert col in result.columns


def test_existing_ecdysis_columns_still_present():
    """Canonical developmental-event columns must remain untouched."""
    filemap = make_minimal_filemap()
    filemap, _, feature_columns, *_ = populate_column_choices(filemap)
    result = process_feature_at_molt_columns(filemap, feature_columns)
    for col in ["HatchTime", "M1", "M2", "M3", "M4"]:
        assert col in result.columns
        assert f"placeholder_feature_at_{col}" in result.columns
