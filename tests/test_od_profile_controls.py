"""Starting values and fit bounds in the precise OD dialog use physical units."""

import pytest
from streamlit.testing.v1 import AppTest

from test_od_fit_workspace import workspace


def _condition_table():
    import streamlit as st
    from gabes import schemes
    from gabes_ui.od_profile_fit import _condition_rows

    params = dict(schemes.get("sas").defaults(), pump_power_mw=0., line="D2",
                  species=st.session_state.get("species", "Rb (natural)"),
                  rb_composition="Custom", probe_power_uw=20., probe_waist_mm=1.5,
                  rb85_fraction=.7217)
    preview, bounds = _condition_rows(st, params, st.session_state.setdefault("settings", {}))
    st.session_state["preview"] = preview
    st.session_state["bounds"] = bounds


def test_diameter_and_percent_convert_once_without_releasing_beam_size():
    app = AppTest.from_function(_condition_table).run()
    assert not app.exception
    assert app.number_input(key="sas_precise_base_probe_waist_mm").value == 3.
    assert app.checkbox(key="sas_profile_vary_probe_waist_mm").disabled
    assert app.checkbox(key="sas_profile_vary_probe_power_uw").disabled is False
    assert app.number_input(key="sas_precise_base_rb85_fraction").value == pytest.approx(72.17)
    assert app.session_state["bounds"] == {}
    app.number_input(key="sas_precise_base_probe_waist_mm").set_value(4.)
    app.number_input(key="sas_precise_base_rb85_fraction").set_value(95.)
    app.checkbox(key="sas_profile_vary_rb85_fraction").check().run()
    app.number_input(key="sas_profile_delta_rb85_fraction").set_value(8.).run()
    assert not app.exception
    assert app.session_state["preview"]["probe_waist_mm"] == 2.
    assert app.session_state["preview"]["rb85_fraction"] == .95
    assert app.session_state["bounds"]["rb85_fraction"] == pytest.approx((.87, 1.))
    assert "probe_waist_mm" not in app.session_state["bounds"]


def test_pure_abundance_display_does_not_replace_the_stored_custom_fraction():
    from gabes import species
    app = AppTest.from_function(_condition_table)
    app.session_state["species"] = species.RB87.label
    app.run()
    assert not app.exception
    value = app.number_input(key="sas_precise_base_rb85_fraction")
    assert value.value == 0. and value.disabled
    assert app.checkbox(key="sas_profile_vary_rb85_fraction").disabled
    assert app.session_state["preview"]["rb85_fraction"] == .7217


def test_equal_length_chart_updates_remount_on_alignment_but_keep_the_brush_for_roi_changes():
    from unittest.mock import Mock
    import numpy as np
    from gabes.experimental_alignment import AxisAlignment
    from gabes.od_measurement import load_measurement, prepare_measurement_upload
    from gabes_ui.od_profile_fit import _chart

    source = prepare_measurement_upload(b"detuning_GHz,transmission\n-1,.9\n0,.6\n1,.9\n")
    trace = load_measurement(source, mode="fraction", x_unit="GHz", denoise=False)
    theory_x = np.array([-3., 0., 3.])
    panel = Mock()

    def draw(shift, indices, depth=.5):
        _chart(panel, trace, AxisAlignment(shift=shift), False, (-3., 3.), indices,
               theory_x, np.array([1., depth, 1.]), None)
        return panel.vega_lite_chart.call_args.kwargs

    first = draw(.03, (0, 1))
    shifted = draw(.04, (0, 1))
    assert len(first["data"]) == len(shifted["data"])
    assert shifted["data"][0]["frequency"] - first["data"][0]["frequency"] == pytest.approx(.01)
    assert first["key"] != shifted["key"]
    assert draw(.04, (1, 2))["key"] == shifted["key"]
    assert draw(.04, (1, 2), depth=.6)["key"] != shifted["key"]


@pytest.mark.parametrize("axis_change", ["line", "mixture"])
def test_changed_spectrum_uses_the_grid_that_apply_will_commit(workspace, monkeypatch, axis_change):
    import numpy as np
    from gabes_ui import od_profile_fit as profile
    from gabes_ui import od_fit_panel

    app, exports, _, _ = workspace
    app.session_state["sas__species"] = "Rb (natural)"
    app.session_state["sas__rb_composition"] = "Custom"
    app.session_state["sas__rb85_fraction"] = 1.
    samples = {}
    chart, alignment_editor = profile._chart, od_fit_panel._trace_alignment_editor

    def capture_chart(box, trace, alignment, invert, window, indices, x, y, result):
        samples["chart"] = np.asarray(x).copy(), np.asarray(y).copy(), window
        chart(box, trace, alignment, invert, window, indices, x, y, result)

    def capture_matcher(panel, trace, window, indices, tx, ty, initial):
        samples["matcher"] = np.asarray(tx).copy(), np.asarray(ty).copy(), window
        return alignment_editor(panel, trace, window, indices, tx, ty, initial)

    monkeypatch.setattr(profile, "_chart", capture_chart)
    monkeypatch.setattr(od_fit_panel, "_trace_alignment_editor", capture_matcher)
    app.run()
    app.button(key="sas_open_precise").click().run()
    old_x = samples["chart"][0]
    if axis_change == "line":
        app.selectbox(key="sas_precise_base_line").select("D1").run()
    else:
        app.number_input(key="sas_precise_base_rb85_fraction").set_value(50.).run()
    assert not app.exception and not app.error
    new_x, new_y, window = samples["chart"]
    assert window == (new_x[0], new_x[-1])
    assert (old_x[0], old_x[-1]) != window
    for values, expected in zip(samples["matcher"], samples["chart"]):
        np.testing.assert_array_equal(values, expected)
    app.button(key="sas___csv_apply").click().run()
    assert not app.exception and not app.error
    main = exports[-1]["comparison_payload"]["theory_trace"]
    np.testing.assert_array_equal(new_x, main["detuning"])
    np.testing.assert_array_equal(new_y, main["transmission"])

def test_changed_finite_field_conditions_wait_for_explicit_background_fit(workspace, monkeypatch):
    from gabes.schemes.sas import FIELD_HISTORY_TRANSIT
    from gabes_ui import od_profile_fit as profile

    app, _, _, _ = workspace
    app.session_state["sas__od_transit_model"] = FIELD_HISTORY_TRANSIT
    app.session_state["sas__od_field_z_ut"] = 50.
    app.run()
    assert not app.exception and not app.error
    assert any("Magnetic field" in w.label for w in app.text_input)  # CSV mode from a pumped preset
    app.button(key="sas_open_precise").click().run()
    assert not app.button(key="sas_profile_frame").disabled

    chart = profile._chart
    sampled = []
    def draw(box, trace, alignment, invert, window, indices, x, y, result):
        sampled.append(y)
        return chart(box, trace, alignment, invert, window, indices, x, y, result)
    monkeypatch.setattr(profile, "_chart", draw)
    def synchronous_solve(*args):
        pytest.fail("A finite field draft must be solved after an explicit request")
    monkeypatch.setattr(profile, "_preview_spectrum", synchronous_solve)
    app.number_input(key="sas_precise_base_probe_power_uw").set_value(8.).run()
    assert not app.exception and not app.error
    assert sampled[-1] is None
    assert app.button(key="sas_profile_frame").disabled
    assert not app.button(key="sas_profile_run").disabled
    assert any("in the background" in item.value for item in app.caption)
