"""Exercise the real pump-free desktop/web boundary using Streamlit's app runner."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _upload(content, name="comparison.csv"):
    return SimpleNamespace(name=name, size=len(content), getvalue=lambda: content)


def _legacy_comparison(model_id, x, signal, *, origin=False):
    """Construct the historical wire format independently of the current writer."""
    import csv
    import io
    import json

    display = {"isotope": "Rb87", "line": "D2", "point": "fine_centroid"}
    old_metadata = {
        "format_version": 1, "generated_by": "GABES", "model_id": model_id,
        "fit_model_id": "od_weak_exact_voigt_transit_v1",
        "line": "D2", "frequency_reference": display,
        "display_reference_offset_hz": 78_095_000,
        "experiment_frequency_reference": dict(display, isotope="Rb85"),
        "experiment_input_mode": "fraction", "experiment_x_unit": "GHz",
    }
    columns = ["detuning_GHz", "theory_transmission", "experiment_display_transmission_on_theory_grid",
               "experiment_input_x", "experiment_input_signal", "experiment_calibrated_transmission",
               "experiment_aligned_detuning_GHz", "experiment_display_transmission", "gabes_comparison_v1"]
    stream = io.StringIO(newline="")
    rows = csv.writer(stream)
    if origin:
        rows.writerow([c.replace("_", " ") for c in columns[:-1]] + columns[-1:])
        rows.writerow(["GHz", "1", "1", "GHz", "1", "1", "GHz", "1", ""])
        rows.writerow(["Theory detuning", "Theory transmission", "Measured transmission",
                       "Native detuning", "Detector signal", "Calibrated transmission",
                       "Aligned detuning", "Display transmission", "Metadata"])
        rows.writerow([])
    else:
        rows.writerow(columns)
    for index, (native_x, t) in enumerate(zip(x, signal, strict=True)):
        display_x = native_x - .078095
        rows.writerow([display_x, t, t, native_x, t, t, display_x, t,
                       json.dumps(old_metadata) if index == 0 else ""])
    return stream.getvalue().encode("utf-8-sig")


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
    # ⟦AI⟧ M2b: under the d20 default a finite Rb probe is fitted by the rate model;
    # the pre-D20 set keeps the two-level overlay this test covers.
    app.session_state["sas__od_model"] = "pre_d20"
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
    reference = {"isotope": "Rb85", "line": "D2", "point": "fine_centroid"}
    content = export_comparison_csv(native, trace, [-1,0,1], [1.,.2,1.], metadata={
        "model_id": "weak_probe_legacy", "axis_reference": reference,
        "contract_reference": reference, "display_reference_offset_hz": 0.,
    })
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


@pytest.mark.parametrize("origin", [False, True], ids=["Standard-v1", "Origin-v1"])
def test_legacy_replay_fits_recorded_model_and_converts_native_centroid_once(workspace, origin):
    from gabes import species
    from gabes.od_model import ODConditions, evaluate_od
    from gabes.schemes.sas import SASScheme

    app, exports, _, uploads = workspace
    ordinary_csv = uploads[0].getvalue()
    model_id = "weak_probe_exact_maxwell_voigt"
    params = dict(SASScheme().defaults(), species=species.RB87.label, line="D2", temp_c=50.,
                  probe_power_uw=0., cell_mm=75., cell_propagation="Uniform", od_model=model_id)
    canonical = np.linspace(-2, 2, 501)
    signal = evaluate_od(ODConditions.from_params(params), canonical * 1.e9).transmission
    uploads[:] = [_upload(_legacy_comparison(model_id, canonical, signal, origin=origin))]
    app.session_state["sas__species"] = species.RB87.label
    app.run()
    assert not app.exception and not app.error
    displayed = exports[-1]["comparison_payload"]
    assert displayed["alignment"]["shift"] == pytest.approx(-.078095)
    assert displayed["model"]["axis_reference"]["isotope"] == "Rb87"
    assert displayed["model"]["contract_reference"]["isotope"] == "Rb85"
    assert displayed["model"]["display_reference_offset_hz"] == pytest.approx(78_095_000)
    assert displayed["model"]["experiment_frequency_reference"]["isotope"] == "Rb85"
    assert displayed["model"]["model_id"] == exports[-1]["raw"]["probe_model"] == model_id
    assert app.session_state["sas__od_model"] == "d20"
    assert "physical_od_fit" not in displayed

    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    app.button(key="sas_run_od_fit").click().run()
    result = exports[-1]["comparison_payload"]["physical_od_fit"]
    assert result["probe_model"] == model_id and result["identifiable"]
    np.testing.assert_allclose(result["frequency_mhz"], (canonical - .078095)*1.e3, atol=1.e-10, rtol=0)
    assert result["fit_residuals"]["rms"] < 1.e-12
    # A click from the previous waveform may arrive while a replacement file
    # is being chosen. Its fitted temperature belongs to the prior model/file.
    if origin:
        uploads[:] = [_upload(ordinary_csv)]
    app.button(key="sas_od_apply_temperature").click().run()
    assert not app.exception and not app.error
    if origin:
        assert app.session_state["sas__temp_c"] == 49.
        assert "physical_od_fit" not in exports[-1]["comparison_payload"]
    else:
        assert app.session_state["sas__temp_c"] == pytest.approx(50.)


def test_recorded_two_level_model_remains_an_overlay(workspace):
    app, exports, _, uploads = workspace
    model_id = "finite_power_2level_gaussian_propagation"
    app.session_state["sas__probe_power_uw"] = 25.
    uploads[:] = [_upload(_legacy_comparison(model_id, [-1., 0., 1.], [.9, .4, .9]))]
    app.run()
    assert not app.exception and not app.error
    comparison = exports[-1]["comparison_payload"]
    assert comparison["model"]["model_id"] == exports[-1]["raw"]["probe_model"] == model_id
    assert "physical_od_fit" not in comparison
    assert any("magnetic-sublevel" in note.value for note in app.caption)
    assert app.session_state["sas__od_model"] == "d20"


def test_autood_source_metadata_does_not_override_the_active_gabes_model(workspace):
    app, exports, _, uploads = workspace
    # AutoOD permits extra columns. Text labels must not turn an AutoOD file
    # into a versioned GABES comparison or supply a native measurement origin.
    body = (
        "detuning_GHz,theory_transmission,experiment_input_x,experiment_input_signal,"
        "experiment_input_mode,experiment_x_unit,autood_version,gabes_commit,model,"
        "frequency_reference,generated_by,format_version,model_id\n"
        "-1,.9,-1,.9,fraction,GHz,0.2.2,8846193,weak_probe_legacy,Rb85,GABES,2,weak_probe_legacy\n"
        "0,.5,0,.5,,,,,,,,,\n1,.9,1,.9,,,,,,,,,\n"
    ).encode()
    uploads[:] = [_upload(body)]
    app.run()
    assert not app.error and not app.exception
    payload = exports[-1]["comparison_payload"]
    assert payload["input_metadata"]["model"] == "weak_probe_legacy"
    assert payload["model"]["model_id"] == "od_weak_exact_voigt_transit_v1"
    assert "experiment_frequency_reference" not in payload["model"]


@pytest.mark.parametrize(("model_id", "power"), [
    ("unknown_model_from_future_version", 0.),
    ("finite_power_2level_gaussian_propagation", 0.),
    ("d20", 0.),
    ("od_weak_exact_voigt_transit_v1", 5.),
])
def test_unusable_recorded_model_can_be_replaced_on_the_same_route(workspace, model_id, power):
    app, exports, _, uploads = workspace
    plain = uploads[0].getvalue()
    app.session_state["sas__probe_power_uw"] = power
    uploads[:] = [_upload(_legacy_comparison(model_id, [-1., 0., 1.], [.9, .4, .9]))]
    app.run()
    assert not app.exception and app.error
    assert not any(output.get("comparison_payload") for output in exports)
    assert app.session_state["sas__od_model"] == "d20"

    uploads[:] = [_upload(plain)]
    app.run()
    assert not app.error and not app.exception
    model = exports[-1]["comparison_payload"]["model"]
    expected = "od_finite_rb_mf_rate_v1" if power else "od_weak_exact_voigt_transit_v1"
    assert model["model_id"] == expected


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


@pytest.mark.parametrize(("model_id", "power"), [
    ("od_finite_rb_mf_rate_v1", 100.), ("od_weak_exact_voigt_transit_v1", 0.)])
def test_explicit_model_drives_curves_export_and_measured_point_fit(workspace, monkeypatch, model_id, power):
    import csv
    import io
    from streamlit.delta_generator import DeltaGenerator
    from gabes import species
    from gabes.od_measurement import prepare_measurement_upload
    from gabes.od_model import ODConditions, evaluate_od_display
    from gabes.schemes.sas import SASScheme
    from gabes_ui.export import build_export_payload

    app, exports, _, uploads = workspace
    p = dict(SASScheme().defaults(), species=species.RB87.label, line="D2", temp_c=49.,
             cell_mm=75., cell_propagation="Uniform", probe_power_uw=power,
             od_model=model_id, line_strength=0.5, scan_points=401)
    conditions = ODConditions.from_params(p)
    x = np.linspace(-2, 2, 201)
    signal = evaluate_od_display(conditions, x * (2*np.pi*1.e9)).transmission
    body = ("detuning_GHz,transmission\n" + "\n".join(
        f"{a:.17g},{b:.17g}" for a,b in zip(x,signal))).encode()
    uploads[:] = [SimpleNamespace(name="model.csv", size=len(body), getvalue=lambda: body)]
    for key in ("species", "line", "line_strength", "probe_power_uw", "od_model"):
        app.session_state[f"sas__{key}"] = p[key]

    for size in (401,1401):
        app.session_state["sas__scan_points"] = size
        app.run()
        assert not app.exception and not app.error
        output = exports[-1]
        comparison = output["comparison_payload"]
        raw = output["raw"]
        expected = evaluate_od_display(conditions, raw["scan"])
        assert len(expected.transmission) == size
        assert comparison["model"]["model_id"] == raw["probe_model"] == model_id
        assert comparison["model"]["axis_reference"]["isotope"] == "Rb87"
        assert comparison["model"]["contract_reference"]["isotope"] == "Rb85"
        assert comparison["model"]["conditions"]["line_strength"] == .5
        np.testing.assert_array_equal(comparison["theory_trace"]["transmission"], expected.transmission)
        np.testing.assert_array_equal(output["view"]["optical_readout"]["optical_depth"], expected.optical_depth)
        np.testing.assert_array_equal(output["plot_payloads"][0]["traces"][0]["y_data"], expected.transmission)
        snapshot = build_export_payload(**output, include_quantitative=True)
        assert snapshot["raw"]["probe_model"] == snapshot["view"]["optical_readout"]["model_id"] == model_id
        np.testing.assert_array_equal(snapshot["raw"]["offsets_hz"], expected.offsets_hz)
    app.multiselect(key="sas_od_fit_free").set_value(["offset_mhz"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    app.button(key="sas_run_od_fit").click().run()
    assert not app.exception and not app.error
    comparison = exports[-1]["comparison_payload"]
    fit = comparison["physical_od_fit"]
    assert fit["probe_model"] == comparison["model"]["model_id"] == comparison["model"]["fit_model_id"] == model_id
    assert fit["points_used"] == len(x) and fit["probe_power_uw"] == power
    assert abs(fit["parameters"][0]["value"]) < 1.e-8 and fit["fit_residuals"]["rms"] < 1.e-12
    assert app.session_state["sas__pump_power_mw"] == 1.5

    downloads = []
    native_button = DeltaGenerator.download_button
    def capture(panel, label, *args, **kwargs):
        if label == "Download comparison CSV":
            downloads.append(kwargs["data"])
        return native_button(panel, label, *args, **kwargs)
    monkeypatch.setattr(DeltaGenerator, "download_button", capture)
    if power == 0.:
        app.selectbox(key="sas_measurement_export_format").select("Origin").run()
    app.button(key="sas_prepare_measurement_csv").click().run()
    assert not app.error and not app.exception and len(downloads) == 1
    metadata = prepare_measurement_upload(downloads[0]).metadata
    from gabes.od_model import MODELS
    if MODELS[model_id].accuracy:
        assert metadata["model_accuracy"] == MODELS[model_id].accuracy
        assert metadata["model_accuracy"] == snapshot["view"]["optical_readout"]["model_accuracy"]
        assert any(item.value == MODELS[model_id].accuracy for item in app.caption)
    else:
        assert "model_accuracy" not in metadata
    assert metadata["fit_model_id"] == metadata["model_id"] == model_id
    assert metadata["axis_reference"]["isotope"] == "Rb87"
    assert metadata["contract_reference"]["isotope"] == "Rb85"
    assert metadata["format_version"] == 2
    assert "frequency_reference" not in metadata and "experiment_frequency_reference" not in metadata
    rows = list(csv.reader(io.StringIO(downloads[0].decode("utf-8-sig"))))
    assert rows[0][-1] == "gabes_comparison_v2"
    records = rows[4:] if power == 0. else rows[1:]
    np.testing.assert_array_equal([float(row[1]) for row in records], expected.transmission)

    # Reopen the downloaded file with a different active preset. The pump-free
    # view uses its recorded ID immediately, without changing the theory preset.
    app.session_state["sas__od_model"] = "pre_d20"
    uploads[:] = [_upload(downloads[0])]
    app.run()
    assert not app.error and not app.exception
    replay = exports[-1]["comparison_payload"]
    assert exports[-1]["raw"]["probe_model"] == replay["model"]["model_id"] == model_id
    np.testing.assert_array_equal(replay["theory_trace"]["transmission"], expected.transmission)
    assert "physical_od_fit" not in replay and "fit_model_id" not in replay["model"]
    app.button(key="sas___save_browser").click().run()
    saved = app.session_state["sas___defaults_command"]["record"]["params"]
    assert saved["od_model"] == "pre_d20" and saved["pump_power_mw"] == 1.5
    app.query_params.pop("compare", None)
    app.run()
    app.query_params["compare"] = ["0"]
    app.run()
    assert not app.exception and not app.error
    assert exports[-1]["comparison_payload"]["model"]["model_id"] == model_id

    restored = "finite_power_2level_gaussian_propagation" if power else "weak_probe_exact_maxwell_voigt"
    if power == 0.:
        app.button(key="sas___csv_remove_retained").click().run()
        assert not app.exception and not app.error
        assert exports[-1]["raw"]["probe_model"] == restored
    uploads[:] = [_upload(body)]
    app.run()
    assert not app.error and not app.exception
    assert exports[-1]["comparison_payload"]["model"]["model_id"] == restored


@pytest.mark.parametrize("invalid", ["unknown_saved_model", "", "   ", 0, [], {}, ["od_finite_rb_mf_rate_v1"]])
def test_invalid_explicit_model_cannot_fall_back_silently(workspace, invalid):
    app, exports, _, _ = workspace
    app.session_state["sas__od_model"] = invalid
    app.run()
    assert not app.exception and not exports
    assert any("OD model ID" in error.value for error in app.error)


def test_null_saved_model_uses_the_same_weak_spectrum_as_an_absent_setting(workspace):
    app, exports, _, _ = workspace
    app.run()
    original = exports[-1]["comparison_payload"]
    app.session_state["sas__od_model"] = None
    app.run()
    assert not app.exception and not app.error
    replay = exports[-1]["comparison_payload"]
    assert replay["model"]["model_id"] == original["model"]["model_id"]
    assert replay["theory_trace"] == original["theory_trace"]
