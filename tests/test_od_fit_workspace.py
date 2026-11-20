"""Exercise the real pump-free desktop/web boundary using Streamlit's app runner."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def workspace(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1.element_tree import ButtonGroup
    from streamlit.delta_generator import DeltaGenerator
    from gabes import schemes, species
    from gabes.od_model import ODConditions, evaluate_od
    from gabes_ui import export

    scheme = schemes.get("sas")
    params = dict(scheme.defaults(), species=species.RB85.label, line="D2", temp_c=50.,
                  probe_power_uw=0., cell_mm=75., cell_propagation="Uniform", scan_points=401)
    x = np.linspace(-2, 2, 501)
    y = evaluate_od(ODConditions.from_params(params), x * 1.e9).transmission
    content = ("detuning_GHz,transmission\n" + "\n".join(f"{a:.17g},{b:.17g}" for a,b in zip(x,y))).encode()
    uploads = [SimpleNamespace(name="measured.csv", size=len(content), getvalue=lambda: content)]
    # AppTest has no browser registry for v2 components. Its ButtonGroup assumes a
    # list for the native single-select control; encode the same frontend indices.
    monkeypatch.setattr(st, "_bidi_component", lambda *a, **k: {})
    def indices(widget):
        values = widget.value if isinstance(widget.value, list) else [widget.value]
        return [widget.options.index(widget.format_func(v)) for v in values if v is not None]
    monkeypatch.setattr(ButtonGroup, "indices", property(indices))
    real_uploader = DeltaGenerator.file_uploader
    def uploader(panel, label, *args, **kwargs):
        if label == "Measurement CSV":
            return uploads.pop(0) if uploads else None
        return real_uploader(panel, label, *args, **kwargs)
    monkeypatch.setattr(DeltaGenerator, "file_uploader", uploader)
    exports, solves = [], []
    monkeypatch.setattr(export, "render_export", lambda *a, **kw: exports.append(kw))
    compute = scheme.compute
    def observed(p):
        solves.append(p.copy())
        return compute(p)
    monkeypatch.setattr(scheme, "compute", observed)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=60)
    app.query_params.update(scheme="sas", compare="0")
    app.session_state["sas___defaults_version"] = scheme.defaults_version
    for k, v in dict(params, temp_c=49., pump_power_mw=1.5, waist_mm=1.4, axial_pump_depletion=True).items():
        app.session_state[f"sas__{k}"] = v
    return app, exports, solves, uploads


def test_pump_free_fit_apply_and_preset_round_trip(workspace):
    app, exports, solves, _ = workspace
    app.run()
    assert not app.exception
    payload = exports[-1]["comparison_payload"]
    assert payload["model"]["pump_power_mw"] == 0
    assert exports[-1]["params"]["waist_mm"] != 1.4
    assert payload["alignment"]["scale"] == 1 and payload["alignment"]["shift"] == 0
    assert all("pump" not in (widget.key or "") for widget in app.text_input)
    assert not any(widget.label == "Preset" for widget in app.selectbox)
    previous = len(solves)
    app.session_state["sas__waist_mm"] = 1.6
    app.run()
    assert len(solves) == previous
    app.session_state["sas__waist_mm"] = 1.4
    app.button(key="sas___save_browser").click().run()
    preset = app.session_state["sas___defaults_command"]["record"]["params"]
    assert (preset["pump_power_mw"], preset["waist_mm"], preset["axial_pump_depletion"]) == (1.5, 1.4, True)
    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    app.button(key="sas_run_od_fit").click().run()
    fit = exports[-1]["comparison_payload"]["physical_od_fit"]
    assert fit["points_used"] == 501 and fit["fit_residuals"]["rms"] < 1.e-12
    app.button(key="sas_od_apply_temperature").click().run()
    assert app.session_state["sas__temp_c"] == pytest.approx(50.)
    # Native toggle and URL navigation both use the same route and persisted projection.
    toggle = next(x for x in app.get("button_group") if x.key == "sas___spectra_view")
    toggle.set_value(["Simulate spectra"]).run()
    assert not app.exception
    assert app.query_params.get("compare") is None
    assert (app.session_state["sas__pump_power_mw"], app.session_state["sas__waist_mm"],
            app.session_state["sas__axial_pump_depletion"]) == (1.5, 1.4, True)
    app.query_params["compare"] = ["0"]
    app.run()
    assert not app.exception
    assert exports[-1]["comparison_payload"]["source_fingerprint_sha256"] == payload["source_fingerprint_sha256"]
    assert app.session_state["sas__temp_c"] == pytest.approx(50.)


def test_reject_stale_result_and_keep_finite_power_overlay(workspace):
    app, exports, _, _ = workspace
    app.run()
    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    app.button(key="sas_run_od_fit").click().run()
    # A result button already in the browser must also re-check live canonical settings.
    app.session_state["sas__line"] = "D1"
    app.button(key="sas_od_apply_temperature").click().run()
    assert not app.exception
    assert app.session_state["sas__temp_c"] == 49.
    assert any("changed" in note.value for note in app.info)
    app.session_state["sas__probe_power_uw"] = 100.
    app.run()
    assert not app.exception
    payload = exports[-1]["comparison_payload"]
    assert payload["enabled"] and payload["model"]["probe_power_uw"] == 100
    assert payload["model"]["pump_power_mw"] == 0
    assert "physical_od_fit" not in payload
    assert any("magnetic-sublevel" in note.value for note in app.caption)
    assert app.session_state["sas__pump_power_mw"] == 1.5


@pytest.mark.parametrize("line,expected", [("D1",-0.07769), ("D2",-0.078095)])
def test_native_frequency_origin_is_independent_of_file_theory_metadata(line, expected):
    from gabes import species
    from gabes.schemes.sas import SASScheme
    from gabes_ui.od_measurement import native_reference_shift

    params = dict(SASScheme().defaults(), species=species.RB87.label, line=line, cell_propagation="Uniform")
    upload = SimpleNamespace(metadata={"frequency_reference": "Rb85", "line": line})
    assert native_reference_shift(upload, params) == (0., None)
    upload.metadata["experiment_frequency_reference"] = {"isotope": "Rb85", "line": line}
    shift, warning = native_reference_shift(upload, params)
    assert warning is None and shift == pytest.approx(expected, abs=1.e-12)
    upload.metadata["experiment_frequency_reference"]["line"] = "D2" if line == "D1" else "D1"
    assert native_reference_shift(upload, params)[1]


@pytest.mark.parametrize("change", ["branch", "source", "scale"])
def test_apply_compares_new_measurement_widgets_in_the_same_rerun(workspace, change):
    app, exports, _, uploads = workspace
    original = uploads[0].getvalue()
    header, *rows = original.splitlines()
    content = b"\n".join([header, *rows, *reversed(rows[:-1])])
    def send(data):
        return SimpleNamespace(name="same-name.csv", size=len(data), getvalue=lambda: data)
    uploads[:] = [send(content)]
    app.run()
    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    app.button(key="sas_run_od_fit").click().run()
    assert exports[-1]["comparison_payload"]["physical_od_fit"]["identifiable"]
    if change == "branch":
        app.selectbox(key="sas_od_fit_branch").select(1)
    elif change == "source":
        uploads[:] = [send(original)]
    else:
        app.number_input(key="sas___csv_x_shift").set_value(.03)
    app.button(key="sas_od_apply_temperature").click().run()
    assert not app.exception
    assert app.session_state["sas__temp_c"] == 49.
    assert "physical_od_fit" not in exports[-1]["comparison_payload"]


@pytest.mark.parametrize("initial_shift", [0.0, -0.245])
def test_frequency_alignment_survives_line_change_and_view_switch(workspace, initial_shift):
    app, exports, _, _ = workspace
    app.run()
    app.number_input(key="sas___csv_x_shift").set_value(initial_shift)
    app.button(key="sas___csv_apply").click().run()
    app.selectbox(key="conditions-sas-line").select("D1").run()
    assert exports[-1]["comparison_payload"]["alignment"]["shift"] == initial_shift
    app.selectbox(key="conditions-sas-line").select("D2").run()
    assert exports[-1]["comparison_payload"]["alignment"]["shift"] == initial_shift
    app.run()
    assert exports[-1]["comparison_payload"]["alignment"]["shift"] == initial_shift
    app.query_params.pop("compare",None)
    app.run()
    app.query_params["compare"] = ["0"]
    app.run()
    assert not app.exception
    assert exports[-1]["comparison_payload"]["alignment"]["shift"] == initial_shift


@pytest.mark.parametrize("mode", ["absolute_dark_reference", "absolute_gain_offset"])
def test_comparison_reopens_with_declared_raw_detector_law(workspace, mode):
    from gabes.od_measurement import prepare_measurement_upload, load_measurement, export_comparison_csv
    from gabes.experimental_csv import CALIBRATION_ABSOLUTE_DARK_REFERENCE, CALIBRATION_ABSOLUTE_GAIN_OFFSET
    app, exports, _, uploads = workspace
    dark = mode == "absolute_dark_reference"
    calibration = CALIBRATION_ABSOLUTE_DARK_REFERENCE if dark else CALIBRATION_ABSOLUTE_GAIN_OFFSET
    native = prepare_measurement_upload(b"detuning_GHz,voltage\n-1,1.5\n0,0.7\n1,1.5\n")
    detector = dict(dark_signal=.5, reference_signal=1.5) if dark else dict(gain=1., offset=.5)
    trace = load_measurement(native, denoise=False, calibration_mode=calibration, **detector)
    content = export_comparison_csv(native, trace, [-1,0,1], [1.,.2,1.],
        metadata={"model_id":"weak_probe_legacy","frequency_reference":{"isotope":"Rb85","line":"D2"}})
    upload = SimpleNamespace(name="calibrated.csv", size=len(content), getvalue=lambda:content)
    uploads[:] = [upload]
    app.run()
    assert not app.exception and not app.error
    payload = exports[-1]["comparison_payload"]
    assert payload["calibration_mode"] == calibration
    inputs = payload["calibration_inputs"]
    if dark:
        assert (inputs["dark_signal"], inputs["reference_signal"]) == (.5,1.5)
    else:
        assert (inputs["detector_offset"], inputs["detector_gain"]) == (.5,1.)


def test_manual_channel_override_updates_the_projected_scope_samples(workspace):
    import csv
    import io
    from test_od_measurement_io import _scope_waveforms

    app, exports, _, uploads = workspace
    times, waves = _scope_waveforms()
    data = io.StringIO()
    writer = csv.writer(data)
    writer.writerow(["Time", "CH1", "CH2", "CH3"])
    writer.writerows(zip(times, waves["trigger"], waves["od"], waves["sas"]))
    content = data.getvalue().encode()
    scope = SimpleNamespace(name="scope.csv", size=len(content), getvalue=lambda: content)
    uploads[:] = [scope, scope]
    app.run()
    assert not app.exception and not app.error
    before = app.session_state["sas___csv_retained_file"][1]
    assert before.scope.od == 1 and not before.scope.manual
    picker = next(widget for widget in app.selectbox if widget.label == "OD signal channel")
    picker.select("CH3").run()
    assert not app.exception and not app.error
    after = app.session_state["sas___csv_retained_file"][1]
    assert after.sha256 == before.sha256 and after.payload != before.payload
    assert after.scope.od == 2 and after.scope.manual


def test_quantitative_fit_converts_declared_native_centroid_once(workspace):
    from gabes import species
    from gabes.od_measurement import prepare_measurement_upload, load_measurement, export_comparison_csv
    from gabes.od_model import ODConditions, evaluate_od
    from gabes.schemes.sas import SASScheme

    app, exports, _, uploads = workspace
    params = dict(SASScheme().defaults(), species=species.RB87.label, line="D2", temp_c=50.,
                  probe_power_uw=0., cell_mm=75., cell_propagation="Uniform")
    canonical = np.linspace(-2, 2, 501)
    signal = evaluate_od(ODConditions.from_params(params), canonical * 1.e9).transmission
    body = ("detuning_GHz,transmission\n" + "\n".join(
        f"{a:.17g},{b:.17g}" for a,b in zip(canonical, signal))).encode()
    native = prepare_measurement_upload(body)
    trace = load_measurement(native, denoise=False)
    reference = {"isotope":"Rb85", "line":"D2", "point":"fine_centroid"}
    content = export_comparison_csv(native, trace, canonical, signal,
        metadata={"model_id": "weak_probe_legacy", "frequency_reference":reference,
                  "experiment_frequency_reference":reference, "line":"D2"})
    uploads[:] = [SimpleNamespace(name="rb85-origin.csv", size=len(content), getvalue=lambda:content)]
    app.session_state["sas__species"] = species.RB87.label
    app.run()
    assert not app.exception and not app.error
    displayed = exports[-1]["comparison_payload"]
    assert displayed["alignment"]["shift"] == pytest.approx(-.078095)
    assert displayed["model"]["frequency_reference"]["isotope"] == "Rb87"
    assert displayed["model"]["experiment_frequency_reference"]["isotope"] == "Rb85"
    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    app.button(key="sas_run_od_fit").click().run()
    result = exports[-1]["comparison_payload"]["physical_od_fit"]
    np.testing.assert_allclose(result["frequency_mhz"], (canonical - .078095)*1.e3, atol=1.e-10, rtol=0)
    assert result["fit_residuals"]["rms"] < 1.e-12


def test_history_bridge_refreshes_each_traversal_once_and_detaches(tmp_path):
    import shutil
    import subprocess
    from gabes_ui.paths import asset_text

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to exercise browser history events")
    script = tmp_path / "history.mjs"
    script.write_text(asset_text("ui/sas_history.js") + """
import assert from 'node:assert/strict';
globalThis.window = new EventTarget();
window.location = {search: '?compare=0'};
const seen = [];
const component = {setTriggerValue: (key, search) => seen.push([key, search])};
const first = bindSasHistory(component);
assert.deepEqual(seen, []);  // Mounting must not create a frontend rerun.
window.dispatchEvent(new Event('popstate'));
first();
window.location.search = '';
window.dispatchEvent(new Event('popstate')); // No handler on the unmounted component.
const second = bindSasHistory(component);
window.dispatchEvent(new Event('popstate'));
second();
assert.deepEqual(seen, [['route', '?compare=0'], ['route', '']]);
""", encoding="utf-8")
    subprocess.run([node, str(script)], check=True, capture_output=True)
