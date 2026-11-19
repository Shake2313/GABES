"""Single-beam profiler UI: explicit runs, honest provenance, and cache tiers.

The numerical backend is replaced by a small passive-field fixture. These tests
exercise real Streamlit widgets, source adapters, plots and exports; independent
atomic/propagation accuracy belongs to the numerical integration tests.
"""
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gabes import single_beam_profile as model
from gabes.nonlinear_response import ComplexSusceptibilityTable
from gabes_ui.sabes import beam_profiler as ui
from sabes.beamline import build_source_chain
from sabes.presets import LAB_PROFILE, profile_calibration, profile_settings
from sabes.single_beam import frequency_references


APP = """
import streamlit as st
show = st.checkbox("Show profiler", key="show_profiler", value=True)
from gabes_ui.sabes import beam_profiler
from sabes.beamline import build_source_chain
from sabes.presets import LAB_PROFILE, profile_calibration, profile_settings
settings, _ = profile_settings(LAB_PROFILE)
chain = build_source_chain(settings, profile_calibration(LAB_PROFILE))
if show:
    beam_profiler.render_profiler(chain, temperature_c=settings.cell_temp_c, cell_length_m=.0125)
"""


@pytest.fixture
def harness(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import matplotlib.pyplot as plt

    ui._cached_camera.clear()
    calls = SimpleNamespace(cell=[], camera=[], plots=[], fail=None, chi=1e-9+1e-9j)
    real_camera = model.profile_at_distance

    def cell(settings, bound, length, max_nodes, power, radius, curvature,
             window, samples, steps, refine, version, _progress=None):
        calls.cell.append(dict(settings=settings, bound=bound, length=length,
            power=power, radius=radius, curvature=curvature, refine=refine))
        if calls.fail:
            raise ValueError(calls.fail)
        table = ComplexSusceptibilityTable([0., bound], [calls.chi]*2)
        response = dict(settings=settings, table=table, refined_table=table,
            cell_length_m=length, checks=dict(assessed=True, passed=True),
            cache_key=model.response_cache_key(settings, bound, cell_length_m=length,
                                               max_nodes=max_nodes))
        value = model.propagate_cell(response, power_w=power, radius_m=radius,
            curvature_radius_m=curvature, cell_length_m=length,
            half_width_m=window, samples=64, steps=4, refine=False)
        # Deliberately controlled validation status for the UI badge contract;
        # this fixture does not claim independent atomic or grid convergence.
        value["checks"]["propagation"] = dict(assessed=refine, passed=refine)
        value["checks"]["passed"] = bool(refine and value["checks"]["physical_budgets"]["passed"])
        return value

    def camera(value, distance, **kwargs):
        calls.camera.append(distance)
        return real_camera(value, distance, **kwargs)

    def plot(figure):
        calls.plots.append([(axis.get_xlabel(), axis.get_ylabel()) for axis in figure.axes])
        plt.close(figure)

    monkeypatch.setattr(ui, "_cached_cell", cell)
    monkeypatch.setattr(ui, "cache_version", lambda: "ui-fixture-v1")
    monkeypatch.setattr(model, "profile_at_distance", camera)
    monkeypatch.setattr(ui, "render_fig", plot)
    app = AppTest.from_string(APP, default_timeout=20).run()
    assert not app.exception
    yield app, calls
    ui._cached_camera.clear()


def _run(app):
    app.button(key=ui.key("run")).click().run()
    assert not app.exception
    return app


def _text(elements):
    return "\n".join(str(item.value) for item in elements)


def test_reference_is_available_without_canvas_and_waits_for_run(harness):
    app, calls = harness
    assert not calls.cell and not calls.camera
    assert app.selectbox(key=ui.key("source")).value == "Custom single beam"
    assert app.number_input(key=ui.key("detuning_mhz")).value == 1500.
    assert app.number_input(key=ui.key("rb85_percent")).value == 0.
    assert app.checkbox(key=ui.key("refine")).value is True
    assert not app.get("download_button")
    assert "Other optical drives" in _text(app.caption)
    _run(app)
    assert len(calls.cell) == len(calls.camera) == 1
    assert "Cell refinement passed" in _text(app.success)
    assert len(app.get("download_button")) == 2
    assert "Second-moment radius" in _text(app.markdown)
    app.run()
    assert not app.exception
    assert len(calls.cell) == len(calls.camera) == 1


def test_camera_and_view_reuse_cell_but_physical_edit_hides_stale_result(harness):
    app, calls = harness
    _run(app)
    app.number_input(key=ui.key("camera_mm")).set_value(250.).run()
    assert not app.exception
    assert len(calls.cell) == 1 and calls.camera == [.5, .25]
    app.radio(key=ui.key("view")).set_value("Beam radius").run()
    assert not app.exception
    assert len(calls.cell) == 1 and len(calls.camera) == 2
    assert calls.plots[-1][0] == ("Distance from cell entrance (mm)", "Second-moment radius (mm)")
    app.radio(key=ui.key("view")).set_value("2D intensity").run()
    assert not app.exception
    assert len(calls.cell) == 1 and len(calls.camera) == 2
    assert calls.plots[-1][0] == ("Camera x (mm)", "Camera y (mm)")
    app.number_input(key=ui.key("power_mw")).set_value(7.).run()
    assert not app.exception
    assert "Inputs changed" in _text(app.info)
    assert not app.get("download_button") and not app.success
    assert len(calls.cell) == 1
    _run(app)
    assert len(calls.cell) == 2 and calls.cell[-1]["power"] == pytest.approx(.007)


def test_frequency_reference_is_coordinate_only_and_detuning_changes_frequency(harness):
    app, calls = harness
    _run(app)
    initial = calls.cell[0]["settings"].frequency
    target = next(ref for ref in frequency_references("D1") if ref.key == "rb85_d1_f2_fp3")
    app.selectbox(key=ui.key("reference")).set_value(target.key).run()
    assert not app.exception
    assert app.number_input(key=ui.key("detuning_mhz")).value == pytest.approx(target.detuning_mhz(initial))
    assert app.session_state[ui.key("frequency")] == initial
    assert len(calls.cell) == len(calls.camera) == 1
    assert len(app.get("download_button")) == 2 and app.success
    new_detuning = target.detuning_mhz(initial)+100.
    app.number_input(key=ui.key("detuning_mhz")).set_value(new_detuning).run()
    assert not app.exception
    assert not app.get("download_button") and "Inputs changed" in _text(app.info)
    assert app.session_state[ui.key("frequency")].offset_hz == pytest.approx(initial.offset_hz+100e6)
    _run(app)
    assert len(calls.cell) == 2


def test_red_preset_and_d2_custom_conditions_reach_backend(harness):
    app, calls = harness
    app.button(key=ui.key("red")).click().run()
    assert not app.exception and not calls.cell
    assert app.number_input(key=ui.key("detuning_mhz")).value == -1500.
    app.selectbox(key=ui.key("line")).set_value("D2").run()
    assert not app.exception
    assert app.selectbox(key=ui.key("reference")).value == "rb87_d2_upper_ground_centroid"
    assert len(app.selectbox(key=ui.key("reference")).options) == len(frequency_references("D2"))
    app.number_input(key=ui.key("rb85_percent")).set_value(37.).run()
    app.checkbox(key=ui.key("planar")).uncheck().run()
    app.number_input(key=ui.key("curvature_m")).set_value(-2.).run()
    _run(app)
    value = calls.cell[-1]
    assert value["settings"].frequency.line == "D2"
    assert value["settings"].rb85_fraction == .37
    assert value["settings"].frequency.detuning_mhz("Rb87", reference="upper_ground_centroid") == -1500.
    assert value["curvature"] == -2.


def test_seed_component_import_locks_source_conditions_and_exports_omitted_drives(harness):
    app, calls = harness
    app.selectbox(key=ui.key("source")).set_value("Cell seed line").run()
    assert not app.exception and not calls.cell
    for control in ("detuning_mhz", "temp_c", "length_mm"):
        assert app.number_input(key=ui.key(control)).disabled
    assert app.selectbox(key=ui.key("line")).disabled
    assert app.number_input(key=ui.key("length_mm")).value == 12.5
    assert "Nominal cell-centre" in _text(app.caption)
    settings, _ = profile_settings(LAB_PROFILE)
    chain = build_source_chain(settings, profile_calibration(LAB_PROFILE))
    assert app.selectbox(key=ui.key("seed_offset")).value == chain.seed_offset_hz
    _run(app)
    assert calls.cell[-1]["power"] == chain.wanted_seed_sideband_power_w
    assert calls.cell[-1]["settings"].temp_c == settings.cell_temp_c
    assert calls.cell[-1]["settings"].frequency == chain.carrier_frequency.shifted_hz(chain.seed_offset_hz)
    source = app.session_state[ui.key("result")]["source"]
    assert source["excluded_power_w"] > 0.
    assert source["excluded_other_arm_power_w"] == chain.pump.total_power_w
    assert "other optical drives absent" in source["scope"]
    assert "unmeasured" in source["source_reference_plane"]
    app.selectbox(key=ui.key("seed_offset")).set_value(0.).run()
    assert not app.exception and not app.get("download_button")
    _run(app)
    assert calls.cell[-1]["power"] == chain.seed.power_at(0.)
    assert calls.cell[-1]["settings"].frequency == chain.carrier_frequency


def test_unassessed_refinement_and_failure_do_not_keep_a_successful_profile(harness):
    app, calls = harness
    app.checkbox(key=ui.key("refine")).uncheck().run()
    _run(app)
    assert "Exploratory profile" in _text(app.warning)
    assert not app.success
    calls.fail = "intensity exceeds the susceptibility table"
    _run(app)
    assert "Calculation stopped" in _text(app.error)
    assert "intensity ceiling" in _text(app.error)
    assert ui.key("result") not in app.session_state
    assert not app.get("download_button")
    calls.fail = None
    _run(app)
    assert len(app.get("download_button")) == 2


def test_invalid_curvature_is_actionable_and_camera_failure_keeps_cell_for_recovery(harness):
    app, calls = harness
    app.checkbox(key=ui.key("planar")).uncheck().run()
    app.number_input(key=ui.key("curvature_m")).set_value(0.).run()
    _run(app)
    assert "curvature_radius_m" in _text(app.error)
    assert not app.get("download_button")
    app.checkbox(key=ui.key("planar")).check().run()
    _run(app)
    count = len(calls.cell)
    app.number_input(key=ui.key("camera_mm")).set_value(5000.).run()
    assert not app.exception
    assert "Camera plane unavailable" in _text(app.error)
    assert not app.get("download_button")
    assert ui.key("result") in app.session_state
    app.number_input(key=ui.key("camera_mm")).set_value(100.).run()
    assert not app.exception and len(app.get("download_button")) == 2
    assert len(calls.cell) == count


def test_download_contains_strict_conditions_and_native_complex_fields(harness):
    app, _ = harness
    _run(app)
    stored = app.session_state[ui.key("result")]
    cell = stored["cell"]
    camera = model.profile_at_distance(cell, .5)
    text, raw = ui.export_profile(cell, camera, stored["source"])
    record = json.loads(text, parse_constant=lambda value: pytest.fail("nonfinite JSON: "+value))
    assert record["conditions"]["input_plane"] == "cell entrance"
    assert record["source"]["curvature_radius_m"] is None
    assert record["source"]["frequency"] == cell["settings"].frequency.as_dict()
    assert record["conditions"]["atomic"]["rb85_fraction"] == 0.
    assert record["camera_distance_m"] == .5
    assert "no downstream optics" in record["camera_plane"]
    assert "transverse atomic transport" in record["model_scope"]
    with np.load(BytesIO(raw), allow_pickle=False) as arrays:
        assert json.loads(str(arrays["metadata_json"])) == record
        np.testing.assert_array_equal(arrays["x_m"], cell["grid"].x_m)
        for name in ("full", "absorption_only", "vacuum"):
            assert np.iscomplexobj(arrays[name+"_exit_field"])
            np.testing.assert_array_equal(arrays[name+"_exit_field"], cell["paths"][name].final_field)
            np.testing.assert_array_equal(arrays[name+"_camera_field"], camera["paths"][name].final_field)
            for axis in ("z_m", "radius_m", "power_w", "centerline_intensity_w_m2"):
                np.testing.assert_array_equal(arrays[name+"_"+axis], camera["combined"][name][axis])



def test_custom_geometry_survives_switching_to_source_and_back(harness):
    app, calls = harness
    app.selectbox(key=ui.key("line")).set_value("D2").run()
    app.number_input(key=ui.key("detuning_mhz")).set_value(-723.).run()
    app.number_input(key=ui.key("rb85_percent")).set_value(42.).run()
    app.number_input(key=ui.key("temp_c")).set_value(86.).run()
    frequency = app.session_state[ui.key("frequency")]
    app.number_input(key=ui.key("power_mw")).set_value(9.).run()
    app.number_input(key=ui.key("radius_mm")).set_value(.63).run()
    app.checkbox(key=ui.key("planar")).uncheck().run()
    app.number_input(key=ui.key("curvature_m")).set_value(-3.2).run()
    app.selectbox(key=ui.key("source")).set_value("Cell pump").run()
    assert not app.exception
    app.selectbox(key=ui.key("source")).set_value("Custom single beam").run()
    assert not app.exception and not calls.cell
    assert app.number_input(key=ui.key("power_mw")).value == 9.
    assert app.number_input(key=ui.key("radius_mm")).value == .63
    assert app.checkbox(key=ui.key("planar")).value is False
    assert app.number_input(key=ui.key("curvature_m")).value == -3.2
    assert app.selectbox(key=ui.key("line")).value == "D2"
    assert app.number_input(key=ui.key("detuning_mhz")).value == -723.
    assert app.number_input(key=ui.key("rb85_percent")).value == 42.
    assert app.number_input(key=ui.key("temp_c")).value == 86.
    assert app.session_state[ui.key("frequency")] == frequency


def test_nonplanar_curvature_survives_planar_control_hiding(harness):
    app, calls = harness
    app.checkbox(key=ui.key("planar")).uncheck().run()
    app.number_input(key=ui.key("curvature_m")).set_value(-3.2).run()
    app.checkbox(key=ui.key("planar")).check().run()
    assert not app.exception
    app.checkbox(key=ui.key("planar")).uncheck().run()
    assert not app.exception and not calls.cell
    assert app.number_input(key=ui.key("curvature_m")).value == -3.2


def test_profiler_hide_show_preserves_inputs_and_frequency_coordinate(harness):
    app, calls = harness
    app.selectbox(key=ui.key("line")).set_value("D2").run()
    app.selectbox(key=ui.key("reference")).set_value("rb85_d2_f3_fp4").run()
    app.number_input(key=ui.key("detuning_mhz")).set_value(333.125).run()
    app.number_input(key=ui.key("power_mw")).set_value(9.).run()
    app.number_input(key=ui.key("radius_mm")).set_value(.63).run()
    app.number_input(key=ui.key("rb85_percent")).set_value(42.).run()
    initial = app.session_state[ui.key("frequency")]
    app.checkbox(key="show_profiler").uncheck().run()
    assert not app.exception
    app.checkbox(key="show_profiler").check().run()
    assert not app.exception and not calls.cell
    assert app.selectbox(key=ui.key("line")).value == "D2"
    assert app.selectbox(key=ui.key("reference")).value == "rb85_d2_f3_fp4"
    assert app.number_input(key=ui.key("detuning_mhz")).value == 333.125
    assert app.number_input(key=ui.key("power_mw")).value == 9.
    assert app.number_input(key=ui.key("radius_mm")).value == .63
    assert app.number_input(key=ui.key("rb85_percent")).value == 42.
    assert app.session_state[ui.key("frequency")] == initial



def test_dark_output_does_not_report_a_radius_or_equivalent_lens(harness):
    app, calls = harness
    calls.chi = 1e-3j
    _run(app)
    assert not app.error
    assert "unavailable" in _text(app.markdown)
    assert "Equivalent lens unavailable: dark output" in _text(app.caption)
    assert "1/f =" not in _text(app.caption)
    assert len(app.get("download_button")) == 2
