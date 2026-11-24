"""The SVG default route stays figureless until a figure-backed view is selected."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("startup", ["cold", "preloaded-pyplot"])
def test_default_and_legacy_od_routes_remain_figureless(startup):
    # Other tests import pyplot during collection. Measure the normal app entry
    # in a fresh interpreter and stub only the browser's custom-component mount.
    script = """
import sys
from unittest.mock import patch

if sys.argv[1] == "preloaded-pyplot":
    # Use a different noninteractive backend to exercise import order without
    # creating a native window on a Streamlit or pytest thread.
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    assert plt.get_backend().lower() == "svg"

import streamlit as st
from streamlit.testing.v1 import AppTest

with patch.object(st.components.v2, "component", return_value=lambda *a, **kw: {}):
    default = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not default.exception and not default.error
    assert any(button.label == "Result bundle (JSON)" for button in default.get("download_button"))
    if sys.argv[1] == "cold":
        assert "matplotlib" not in sys.modules

    detail = AppTest.from_file("streamlit_app.py", default_timeout=60)
    detail.query_params["compare"] = "0"
    detail.run()
    assert not detail.exception and not detail.error
    assert not any(widget.label == "Measurement CSV" for widget in detail.get("file_uploader"))
    assert not any(button.label == "Spectrum view" for button in detail.get("button_group"))
    assert detail.button(key="sas_open_csv").label == "Open CSV..."
    if sys.argv[1] == "cold":
        assert "matplotlib" not in sys.modules
    else:
        assert sys.modules["matplotlib"].get_backend().lower() == "agg"
"""
    subprocess.run([sys.executable, "-c", script, startup], cwd=Path(__file__).resolve().parents[1],
                   check=True, timeout=150)
