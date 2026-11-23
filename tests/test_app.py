"""Smoke test: dashboard renders for every city without exceptions."""

import pytest

from airwatch import config as C

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.parametrize("city", list(C.CITIES))
def test_app_renders(city):
    at = AppTest.from_file(str(C.ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(city).run()
    assert not at.exception
    assert C.CITIES[city].name_vi in at.title[0].value
    assert len(at.tabs) == 3
