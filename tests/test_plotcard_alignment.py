"""Explicit CSV alignment actions fence pending graph gestures."""


def test_explicit_alignment_actions_advance_revision_but_rerenders_do_not():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
from types import SimpleNamespace
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gabes_ui.plotcard import _render_experimental_comparison

x = np.linspace(-5., 5., 101)
y = 1. - .3 * np.exp(-x * x)
csv = "\\n".join(f"{a},{b}" for a, b in zip(x, y)).encode()
uploaded = SimpleNamespace(name='trace.csv', getvalue=lambda: csv)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlim(-10, 10)
view = dict(figure=fig, comparison=dict(axis_index=0, x_unit='MHz'))
with patch('streamlit.delta_generator.DeltaGenerator.file_uploader', return_value=uploaded):
    st.session_state['comparison'] = _render_experimental_comparison(view, 'sas', st.container())
plt.close(fig)
''').run(timeout=20)
    assert not app.exception
    revision = app.session_state["comparison"]["alignment_revision"]
    app.run(timeout=20)
    assert app.session_state["comparison"]["alignment_revision"] == revision
    for key in ("sas___csv_apply", "sas___csv_reset", "sas___csv_frame"):
        next(button for button in app.button if button.key == key).click().run(timeout=20)
        assert not app.exception
        revision += 1
        assert app.session_state["comparison"]["alignment_revision"] == revision
        app.run(timeout=20)
        assert app.session_state["comparison"]["alignment_revision"] == revision

def test_physical_fit_runs_on_explicit_action_and_discards_stale_result():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(r"""
from types import SimpleNamespace
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gabes import species
from gabes.od_spectrum_fit import od_transmission
from gabes.schemes.sas import SASScheme
from gabes_ui.plotcard import _render_experimental_comparison

x = np.linspace(-3.0, 3.0, 201)
p = dict(SASScheme().defaults(), species=species.SPECIES_ORDER[1],
         temp_c=60., pump_power_mw=0., probe_power_uw=0., cell_mm=23.)
y = od_transmission(x * 1000, dict(p, temp_c=63.), offset_mhz=8.)
y += np.random.default_rng(12).normal(0., .0006, x.size)
csv = "\n".join(f"{a},{b}" for a, b in zip(x, y)).encode()
uploaded = SimpleNamespace(name='trace.csv', getvalue=lambda: csv)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlim(-3, 3)
view = dict(figure=fig, comparison=dict(axis_index=0, x_unit='GHz'))
with patch('streamlit.delta_generator.DeltaGenerator.file_uploader', return_value=uploaded):
    st.session_state['comparison'] = _render_experimental_comparison(view, 'sas', st.container(), p)
plt.close(fig)
""").run(timeout=25)
    assert not app.exception
    assert "physical_od_fit" not in app.session_state["comparison"]
    next(box for box in app.selectbox if box.key == "sas___csv_calibration_mode").select_index(1).run(timeout=25)
    next(button for button in app.button if button.key == "sas___csv_reset").click().run(timeout=25)
    fit_button = next(button for button in app.button if button.key == "sas_run_od_fit")
    assert fit_button.disabled
    next(box for box in app.checkbox if box.key == "sas_od_fit_frequency_scale").check().run(timeout=25)
    assert "physical_od_fit" not in app.session_state["comparison"]
    next(button for button in app.button if button.key == "sas_run_od_fit").click().run(timeout=25)
    if "sas_od_fit_job" in app.session_state:
        app.session_state["sas_od_fit_job"].future.result(timeout=25)
        app.run(timeout=25)
    assert not app.exception
    fit = app.session_state["comparison"]["physical_od_fit"]
    assert fit["scope"] == "calibration_only" and fit["fit_residuals"]["rms"] < .001
    next(box for box in app.checkbox if box.key == "sas_od_fit_frequency_scale").uncheck().run(timeout=25)
    assert "physical_od_fit" not in app.session_state["comparison"]
