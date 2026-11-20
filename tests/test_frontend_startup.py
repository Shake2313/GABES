"""The SVG default route stays figureless until a figure-backed view is selected."""
from pathlib import Path
import subprocess
import sys


def test_default_od_workspace_defers_matplotlib_until_fit_view():
    # Other tests import pyplot during collection. Measure the normal app entry
    # in a fresh interpreter and stub only the browser's custom-component mount.
    script = """
import sys
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

with patch.object(st.components.v2, "component", return_value=lambda *a, **kw: {}):
    default = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not default.exception and not default.error
    assert any(button.label == "Result bundle (JSON)" for button in default.get("download_button"))
    assert "matplotlib" not in sys.modules

    detail = AppTest.from_file("streamlit_app.py", default_timeout=60)
    detail.query_params["compare"] = "0"
    detail.run()
    assert not detail.exception and not detail.error
    assert any(widget.label == "Measurement CSV" for widget in detail.get("file_uploader"))
    assert "matplotlib.pyplot" in sys.modules
    assert sys.modules["matplotlib"].get_backend().lower() == "agg"
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                   check=True, timeout=150)
