from app_components.backend import ECDYSIS_COLUMNS
from app_components.ui import create_molt_annotator
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
    tag = create_molt_annotator(ECDYSIS_COLUMNS, custom_columns_choices=[])
    html = str(tag)
    assert "annotation-buttons-container" in html
