from app_components.backend import ECDYSIS_COLUMNS
from app_components.backend import MOLT_ENTRY_COLUMNS
from app_components.ui import create_molt_annotator
from app_components.ui import create_timepoint_selector
from app_components.ui_components import time_point_navigator


def test_navigator_no_forced_height():
    """Buttons must not have a hardcoded height attribute."""
    tag = time_point_navigator("test_nav", name="time", choices=[0, 1, 2])
    html = str(tag)
    assert 'height="15vh"' not in html


def test_navigator_uses_flexbox_class():
    """Navigator root element must carry the navigator-row CSS class."""
    tag = time_point_navigator("test_nav", name="time", choices=[0, 1, 2])
    html = str(tag)
    assert "navigator-row" in html


def test_annotation_buttons_have_padding_container():
    """Annotation buttons must be wrapped in annotation-buttons-container."""
    tag = create_molt_annotator(
        ECDYSIS_COLUMNS, MOLT_ENTRY_COLUMNS, custom_columns_choices=[]
    )
    html = str(tag)
    assert "annotation-buttons-container" in html


def _make_selector():
    return create_timepoint_selector(
        list_channels=["None", "Channel 1", "Channel 2"],
        times=[0, 1, 2],
        points=[0, 1],
        feature_columns=["volume"],
        overlay_segmentation_choices=["None"],
        default_plotted_column="volume",
    )


def test_image_has_padding_container():
    """Image output must be wrapped in image-container div."""
    html = str(_make_selector())
    assert "image-container" in html


def test_channel_selectors_centered():
    """Channel selectors must be wrapped in channel-selectors div."""
    html = str(_make_selector())
    assert "channel-selectors" in html


def test_dark_mode_toggle_present():
    """Dark mode toggle must be present in the timepoint selector."""
    html = str(_make_selector())
    assert "dark-mode" in html.lower() or "data-bs-theme" in html


def test_save_row_present():
    """The save button and dark mode toggle must share a save-row container."""
    html = str(_make_selector())
    assert "save-row" in html
