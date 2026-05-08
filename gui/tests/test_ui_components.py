import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
