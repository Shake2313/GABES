"""Drive the scope channel picker through reruns, route changes and ambiguous records."""
import csv
import io
import json

import numpy as np
import pytest


def scope_csv(columns, signals):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(("Time", *columns))
    writer.writerows(zip(*signals, strict=True))
    return out.getvalue().encode()


@pytest.fixture
def picker(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from gabes import _scope_channels
    from gabes_ui.od_measurement import _project

    _project.clear()
    observed = []
    identify = _scope_channels.identify_scope_channels

    def counted(signals):
        observed.append(signals.shape)
        return identify(signals)

    monkeypatch.setattr(_scope_channels, "identify_scope_channels", counted)
    app = AppTest.from_string("""
from types import SimpleNamespace
import streamlit as st
from gabes_ui.od_measurement import select_upload
from gabes_ui.plotcard import _forget_retained_csv

if st.session_state.get("in_workspace", True):
    data = st.session_state.get("source")
    uploaded = SimpleNamespace(name="scope.csv", getvalue=lambda: data) if data is not None else None
    select_upload(st, "sas", uploaded, _forget_retained_csv)
""")
    return app, observed


def test_ambiguous_scope_shows_waveforms_without_guessing_and_only_detects_once(picker):
    app, observed = picker
    rows = np.arange(250)
    missing = np.full(len(rows), np.nan)
    # The scope wrote the same label twice. Select the correct physical column by position.
    app.session_state["source"] = scope_csv(("CH1", "CH1", "CH3"), (rows // 2, rows, 2 * rows, missing))
    app.run()
    assert not app.exception and not app.error
    assert app.selectbox[0].value is None
    assert "sas___csv_retained_file" not in app.session_state
    assert "trigger" in app.info[0].value
    preview = app.get("arrow_vega_lite_chart")[0].proto
    spec = json.loads(preview.spec)
    assert spec["encoding"]["detail"]["field"] == "segment"
    assert spec["encoding"]["row"]["sort"] == ["Column 1 · CH1", "Column 2 · CH1", "Column 3 · CH3"]
    app.session_state["source"] = None
    app.run()
    app.selectbox[0].select("Column 3 · CH3").run()
    assert "fewer than two finite samples" in app.warning[0].value
    assert "sas___csv_retained_file" not in app.session_state
    app.selectbox[0].select("Column 2 · CH1").run()
    chosen = app.session_state["sas___csv_retained_file"][1]
    assert chosen.scope.od == 1 and chosen.scope.manual
    measured = np.loadtxt(io.BytesIO(chosen.payload), delimiter=",", skiprows=1)
    np.testing.assert_array_equal(measured[:, 0], rows)
    np.testing.assert_array_equal(measured[:, 1], 2 * rows)
    assert observed == [(250, 3)]


def test_retained_scope_can_switch_between_gated_and_full_recording(picker):
    from test_od_measurement_io import _scope_waveforms

    app, observed = picker
    times, waves = _scope_waveforms()
    app.session_state["source"] = scope_csv(("1", "2", "3"), (times, waves["trigger"], waves["od"], waves["sas"]))
    app.run()
    assert not app.exception and not app.error
    assert app.selectbox[0].value == "Automatic · CH2"
    assert any("SAS CH3 · Trigger CH1" in c.value for c in app.caption)
    original = app.session_state["sas___csv_retained_file"][1]
    assert (original.scope.od, original.scope.start, original.scope.stop) == (1, 422, 1178)
    app.selectbox[0].select("CH2").run()
    full = app.session_state["sas___csv_retained_file"][1]
    assert full.scope.od == 1 and full.scope.manual
    assert (full.scope.start, full.scope.stop) == (0, 1600)
    np.testing.assert_array_equal(np.loadtxt(io.BytesIO(full.payload), delimiter=",", skiprows=1)[:, 1],
                                  waves["od"])
    # The app remounts the native uploader empty after visiting another workspace.
    app.session_state["in_workspace"] = False
    app.session_state["source"] = None
    app.run()
    app.session_state["in_workspace"] = True
    app.run()
    assert app.selectbox[0].value == "CH2"
    app.selectbox[0].select("CH3").run()
    assert app.session_state["sas___csv_retained_file"][1].scope.od == 2
    app.selectbox[0].select("Automatic · CH2").run()
    assert app.session_state["sas___csv_retained_file"][1].payload == original.payload
    assert observed == [(1600, 3)]
    app.button[0].click().run()
    assert not app.selectbox
    assert "sas___csv_retained_file" not in app.session_state
    assert "sas___csv_retained_scope" not in app.session_state
