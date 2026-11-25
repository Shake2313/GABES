"""The precision editor keeps a file's native calibration separate from alignment drafts."""

import pytest
from streamlit.testing.v1 import AppTest


def _editor():
    return AppTest.from_string("""
import streamlit as st
from gabes.experimental_alignment import AxisAlignment
from gabes_ui.od_fit_alignment import (clear_draft, draft_alignment,
                                       render_alignment_controls, set_draft_alignment)
initial = AxisAlignment(scale=.003, shift=-1202.123456789, signal_scale=2., signal_offset=.1)
state = st.session_state
if 'committed' not in state:
    state.committed = initial
absolute = st.checkbox("Calibrated transmission")
draft = render_alignment_controls(st, draft_alignment(state.committed),
                                  absolute=absolute, span_x=8., span_y=.6)
state.preview = draft
st.button("Discard", on_click=clear_draft)
st.button("Fast", on_click=set_draft_alignment, args=(AxisAlignment(scale=.07, shift=-9.),))
""").run()


def test_sliders_and_numeric_values_share_the_full_trace_coordinates():
    app = _editor()
    assert not app.exception
    initial = app.session_state["committed"]
    app.slider(key="sas_precise_slider_scale").set_value(.5).run()
    assert app.number_input(key="sas___csv_x_scale").value == pytest.approx(.006)
    app.slider(key="sas_precise_slider_shift").set_value(.5).run()
    assert app.number_input(key="sas___csv_x_shift").value == pytest.approx(initial.shift + 4.)
    app.slider(key="sas_precise_slider_signal_scale").set_value(-1.).run()
    assert app.number_input(key="sas___csv_y_scale").value == pytest.approx(.5)
    app.slider(key="sas_precise_slider_signal_offset").set_value(-.25).run()
    assert app.number_input(key="sas___csv_y_offset").value == pytest.approx(-.05)
    app.checkbox(key="sas_precise_reverse").check().run()
    assert app.session_state["preview"].reverse
    assert app.session_state["committed"] == initial

    # Typed values can set both a precise in-range position and a new center.
    app.number_input(key="sas___csv_x_shift").set_value(initial.shift+.31415926535).run()
    assert app.slider(key="sas_precise_slider_shift").value == pytest.approx(.31415926535 / 8.)
    app.number_input(key="sas___csv_x_scale").set_value(.123456789012345).run()
    assert app.slider(key="sas_precise_slider_scale").value == 0.
    assert app.session_state["preview"].scale == .123456789012345
    app.slider(key="sas_precise_slider_scale").set_value(-.5).run()
    assert app.session_state["preview"].scale == pytest.approx(.123456789012345 / 2.)
    app.number_input(key="sas___csv_x_scale").set_value(0.).run()
    assert app.warning and app.session_state["preview"].scale > 0.
    app.button[0].click().run()
    assert app.session_state["preview"] == initial and not app.warning
    assert app.number_input(key="sas___csv_x_shift").value == initial.shift
    assert not app.exception


def test_fast_fit_rebases_sliders_and_calibrated_y_is_fixed():
    app = _editor()
    app.button[1].click().run()
    assert app.session_state["preview"].scale == .07 and app.slider(key="sas_precise_slider_scale").value == 0.
    app.slider(key="sas_precise_slider_scale").set_value(1.).run()
    assert app.session_state["preview"].scale == pytest.approx(.28)
    app.number_input(key="sas___csv_y_scale").set_value(1.5).run()
    app.checkbox[0].check().run()
    assert app.session_state["preview"].signal_scale == 1.
    assert app.session_state["preview"].signal_offset == 0.
    assert app.slider(key="sas_precise_slider_signal_scale").disabled
    assert app.number_input(key="sas___csv_y_offset").disabled
    assert not app.exception


def test_signed_recorded_detector_law_and_invalid_draft_calibration():
    app = AppTest.from_string("""
import streamlit as st
from dataclasses import replace
from gabes.od_measurement import prepare_measurement_upload, load_measurement
from gabes_ui.od_fit_alignment import render_detector_calibration, clear_draft

upload = prepare_measurement_upload(b'sample,voltage\\n0,1.\\n1,.5\\n2,.8\\n3,.7\\n')
upload = replace(upload, mode="relative", metadata={
    "experiment_calibration_mode": "absolute_gain_offset",
    "experiment_calibration_gain": -2.,
    "experiment_calibration_offset": 1.2,
})
trace = load_measurement(upload, x_unit="arbitrary", denoise=False)
draft, override = render_detector_calibration(st, upload, trace, mode="relative",
                                               unit="arbitrary", denoise=False)
st.session_state.result = (draft.correction_diagnostics.calibration_gain if draft else None, override)
st.button("Discard", on_click=clear_draft)
""").run()
    assert not app.exception
    assert app.session_state["result"] == (-2., {})
    app.number_input(key="sas_precise_detector_gain").set_value(0.).run()
    assert app.error and app.session_state["result"] == (None, None)
    app.number_input(key="sas_precise_detector_gain").set_value(-1.).run()
    assert not app.error and app.session_state["result"] == (
        -1., {"calibration_mode": "absolute_gain_offset", "gain": -1., "offset": 1.2})
    app.button[0].click().run()
    assert not app.exception and app.session_state["result"] == (-2., {})
