"""Exercise the real pump-free desktop/web boundary using Streamlit's app runner."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _run_fit_to_completion(app):
    app.button(key="sas_run_od_fit").click().run()
    # AppTest has no browser timer. Advance the final rerun after the real worker.
    if "sas_od_fit_job" in app.session_state:
        app.session_state["sas_od_fit_job"].future.result(timeout=60)
        app.run()
    assert not app.exception and not app.error


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
    _run_fit_to_completion(app)
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


def test_fit_picker_filters_unavailable_estimates_as_conditions_change(workspace):
    from gabes.od_spectrum_fit import PARAMETERS, CANDIDATE_PARAMETERS

    app, _, _, _ = workspace
    app.session_state["sas__species"] = "Rb (natural)"
    app.session_state["sas__rb_composition"] = "Custom"
    app.session_state["sas__probe_power_uw"] = 15.
    app.run()

    def picker():
        assert not app.exception and not app.error
        return app.multiselect(key="sas_od_fit_free")

    base = [spec[0] for spec in PARAMETERS.values()]
    fraction = CANDIDATE_PARAMETERS["rb85_fraction"][0]
    power = CANDIDATE_PARAMETERS["probe_power_uw"][0]
    assert picker().options == [*base, fraction, power]
    picker().set_value(["offset_mhz", "rb85_fraction", "probe_power_uw"]).run()
    app.selectbox(key="conditions-sas-line").select("D1").run()
    assert picker().options == [*base, fraction]
    assert picker().value == ["offset_mhz", "rb85_fraction"]
    app.selectbox(key="conditions-sas-isotope-value").select("Natural").run()
    assert picker().options == base and picker().value == ["offset_mhz"]


@pytest.mark.parametrize("at_bound", [False, True], ids=["joint-fit", "power-upper-bound"])
def test_probe_intensity_and_custom_mixture_apply_only_after_identifiable_fit(workspace, monkeypatch, at_bound):
    from gabes import od_spectrum_fit as od
    # The default set (d20_d2sb) fits with its rate model.
    from gabes.schemes.sas import SASScheme, RATE_MODEL_V2 as RATE_MODEL
    from gabes_ui.export import build_export_payload

    # Keep both cases on the actual worker and bounded least-squares path.
    monkeypatch.setattr(od, "GLOBAL_LOG2_SAMPLES", 2)
    monkeypatch.setattr(od, "LOCAL_STARTS", 1)
    app, exports, _, uploads = workspace
    truth_power = 125. if at_bound else 25.
    truth_fraction = .62
    initial_fraction = truth_fraction if at_bound else .55
    truth = dict(SASScheme().defaults(), species="Rb (natural)", line="D2", rb_composition="Custom",
                 rb85_fraction=truth_fraction, probe_power_uw=truth_power, probe_waist_mm=.8,
                 temp_c=49., pump_power_mw=0., cell_mm=75.)
    x = np.linspace(-3.5, 5., 160)
    transmission = od.od_transmission(1000*x, truth, model_id=RATE_MODEL)
    content = ("detuning_GHz,transmission\n" + "\n".join(
        f"{v:.17g},{y:.17g}" for v, y in zip(x, transmission))).encode()
    uploads[:] = [_upload(content)]
    for key in ("species", "rb_composition", "probe_waist_mm"):
        app.session_state["sas__"+key] = truth[key]
    app.session_state["sas__probe_power_uw"] = 15.
    app.session_state["sas__rb85_fraction"] = initial_fraction
    app.run()
    keys = ["probe_power_uw"] if at_bound else ["probe_power_uw", "rb85_fraction"]
    app.multiselect(key="sas_od_fit_free").set_value(keys)
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    _run_fit_to_completion(app)

    output = exports[-1]
    fit = output["comparison_payload"]["physical_od_fit"]
    assert fit["probe_model"] == RATE_MODEL
    assert fit["candidate_parameters"]["scope"] == {k: od.VALIDATED_SCOPE[k] for k in keys}
    captions = " ".join(item.value for item in app.caption)
    for key in keys:
        assert od.VALIDATED_SCOPE[key] in captions
    fitted = {row["key"]: row for row in fit["parameters"]}
    intensity = fit["probe_peak_intensity"]
    assert intensity["value_mw_cm2"] == pytest.approx(
        od.probe_peak_intensity_mw_cm2(fitted["probe_power_uw"]["value"], .8))
    metric = next(item for item in app.metric if item.label == "Probe peak intensity")
    assert float(metric.value.split()[0]) == pytest.approx(intensity["value_mw_cm2"], rel=.0005)
    assert "mW/cm²" in metric.value and "beam radius 0.8 mm" in captions
    snapshot = build_export_payload(**output)
    exported = snapshot["view"]["comparison"]["physical_od_fit"]
    assert exported["candidate_parameters"] == fit["candidate_parameters"]
    assert exported["probe_peak_intensity"] == intensity
    assert exported["probe_power_role"] == "fit_starting_value" and exported["probe_power_uw"] == 15.
    assert snapshot["params"]["probe_power_uw"] == 15.
    assert snapshot["params"]["rb85_fraction"] == initial_fraction
    assert app.session_state["sas__probe_power_uw"] == 15.
    assert app.session_state["sas__rb85_fraction"] == initial_fraction
    apply = app.button(key="sas_od_apply_conditions")
    if at_bound:
        assert fitted["probe_power_uw"]["value"] == pytest.approx(100.)
        assert fitted["probe_power_uw"]["on_bound"] and not fit["identifiable"]
        assert intensity["standard_error_mw_cm2"] is None and apply.disabled
        assert "1σ unavailable" in captions
        assert any("100 µW" in item.value for item in app.warning)
        return

    assert fit["identifiable"] and not apply.disabled
    assert fitted["probe_power_uw"]["value"] == pytest.approx(truth_power, rel=1e-5)
    assert fitted["rb85_fraction"]["value"] == pytest.approx(truth_fraction, abs=1e-5)
    sigma = intensity["standard_error_mw_cm2"]
    assert sigma > 0. and f"±{sigma:.3g} mW/cm²" in captions
    # A click from an older render must not overwrite a new draft or partially
    # apply the isotope fraction if the displayed power changed in that rerun.
    app.text_input(key="power-input-sas-probe_power_uw").set_value("17")
    apply.click().run()
    assert not app.exception and app.session_state["sas__probe_power_uw"] == 17.
    assert app.session_state["sas__rb85_fraction"] == initial_fraction
    assert any("changed" in item.value for item in app.info)
    app.button(key="sas_od_apply_conditions").click().run()
    assert not app.exception and not app.error
    for key, widget_key in (("probe_power_uw", "power-input-sas-probe_power_uw"),
                            ("rb85_fraction", "conditions-sas-rb85_fraction")):
        value = fitted[key]["value"]
        assert app.session_state["sas__"+key] == pytest.approx(value)
        widget = next(w for w in app.text_input if w.key.startswith(widget_key))
        assert float(widget.value) == pytest.approx(value)
    app.button(key="sas___save_browser").click().run()
    record = app.session_state["sas___defaults_command"]["record"]
    saved = record["params"]
    assert (saved["probe_power_uw"], saved["rb85_fraction"]) == pytest.approx((truth_power, truth_fraction))
    assert saved["pump_power_mw"] == 1.5 and saved["probe_waist_mm"] == .8

    # AppTest cannot attach a file. Replay the browser record after portable
    # JSON validation and load it with the same native Defaults selector.
    from gabes_ui.defaults_store import dump_default, parse_default
    restored = parse_default(SASScheme(), dump_default(record).encode("utf-8"))
    app.session_state["sas___saved_defaults"] = {restored["name"]: restored}
    app.session_state["sas__probe_power_uw"] = 7.
    app.session_state["sas__rb85_fraction"] = .51
    app.run()
    selector = next(w for w in app.selectbox if w.label == "Defaults")
    selector.select("saved:" + restored["name"]).run()
    assert not app.exception and not app.error
    assert app.session_state["sas__probe_power_uw"] == pytest.approx(truth_power)
    assert app.session_state["sas__rb85_fraction"] == pytest.approx(truth_fraction)
    assert app.session_state["sas__pump_power_mw"] == 1.5


@pytest.mark.parametrize("action", ["cancel", "change_scale", "change_line", "leave_workspace"])
def test_running_fit_is_cancelled_without_publishing_a_partial_result(workspace, monkeypatch, action):
    from threading import Event
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from gabes import od_spectrum_fit as od

    started, resume = Event(), Event()

    def controlled_fit(data, params, *, progress, **options):
        assert get_script_run_ctx(suppress_warning=True) is None
        assert progress(od.ODFitProgress("global", 1, 32.))
        started.set()
        assert resume.wait(15)
        update = od.ODFitProgress("local", 2, 28.)
        if not progress(update):
            raise od.ODFitCancelled(update)
        pytest.fail("The detached fit still has a live session")

    monkeypatch.setattr(od, "fit_od_spectrum", controlled_fit)
    app, exports, _, _ = workspace
    app.session_state["sas__probe_power_uw"] = 15.
    app.run()
    app.multiselect(key="sas_od_fit_free").set_value(["probe_power_uw"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    try:
        app.button(key="sas_run_od_fit").click().run()
        assert started.wait(5)
        job = app.session_state["sas_od_fit_job"]
        app.run()
        assert any("Global search · 1 evaluation · Best weighted cost 32" in c.value for c in app.caption)
        assert not any(b.key == "sas_run_od_fit" for b in app.button)
        if action == "cancel":
            app.button(key="sas_cancel_od_fit").click().run()
            assert any(i.value == "Fit cancelled." for i in app.info)
        elif action == "change_scale":
            app.checkbox(key="sas_od_fit_frequency_scale").uncheck().run()
            assert any("fit inputs changed" in i.value for i in app.info)
        elif action == "change_line":
            app.selectbox(key="conditions-sas-line").select("D1").run()
            assert "probe_power_uw" not in app.multiselect(key="sas_od_fit_free").value
        else:
            route = next(w for w in app.get("button_group") if w.key == "sas___spectra_view")
            route.set_value(["Simulate spectra"]).run()
        assert not app.exception
        assert "sas_od_fit_job" not in app.session_state
        assert "sas_od_physical_fit" not in app.session_state
        assert not any(b.key in ("sas_od_apply_temperature", "sas_od_apply_conditions") for b in app.button)
        assert app.session_state["sas__probe_power_uw"] == 15.
        assert not any("physical_od_fit" in (e.get("comparison_payload") or {}) for e in exports)
    finally:
        resume.set()
    with pytest.raises(od.ODFitCancelled):
        job.future.result(timeout=5)


def test_fit_worker_queues_sessions_and_keeps_its_model_snapshot(monkeypatch):
    from threading import Event
    from gabes import od_spectrum_fit
    from gabes_ui.od_fit_job import ODFitJob

    entered, resume = Event(), Event()
    def controlled_fit(data, params, *, progress, **options):
        entered.set()
        assert resume.wait(10)
        return params["startup"]

    monkeypatch.setattr(od_spectrum_fit, "fit_od_spectrum", controlled_fit)
    params = {"startup": [49.]}
    first = ODFitJob(("first",), None, params)
    try:
        assert entered.wait(5)
        params["startup"][0] = 55.
        second = ODFitJob(("second",), None, params)
        assert not second.future.running()
        second.cancel()
        assert second.future.cancelled()
    finally:
        resume.set()
    assert first.future.result(timeout=5) == [49.]


def test_reject_stale_result_and_keep_finite_power_overlay(workspace):
    app, exports, _, _ = workspace
    app.run()
    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    _run_fit_to_completion(app)
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
    _run_fit_to_completion(app)
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
    assert not app.checkbox(key="sas___csv_auto_correct").value
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
    assert app.session_state["sas__od_model"] == "d20_d2sb"
    assert "physical_od_fit" not in displayed

    app.multiselect(key="sas_od_fit_free").set_value(["temp_c"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    _run_fit_to_completion(app)
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
    assert app.session_state["sas__od_model"] == "d20_d2sb"


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
    assert payload["model"]["model_id"] == "od_weak_exact_voigt_transit_v2"
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
    assert app.session_state["sas__od_model"] == "d20_d2sb"

    uploads[:] = [_upload(plain)]
    app.run()
    assert not app.error and not app.exception
    model = exports[-1]["comparison_payload"]["model"]
    expected = "od_finite_rb_mf_rate_v2" if power else "od_weak_exact_voigt_transit_v2"
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
        np.testing.assert_array_equal(comparison["theory_trace"]["od0"], expected.od0)
        assert comparison["theory_trace"]["n_eff"] == output["view"]["optical_readout"]["n_eff"] > 0
        np.testing.assert_array_equal(output["view"]["optical_readout"]["optical_depth"], expected.optical_depth)
        np.testing.assert_array_equal(output["plot_payloads"][0]["traces"][0]["y_data"], expected.transmission)
        snapshot = build_export_payload(**output, include_quantitative=True)
        assert snapshot["raw"]["probe_model"] == snapshot["view"]["optical_readout"]["model_id"] == model_id
        np.testing.assert_array_equal(snapshot["raw"]["offsets_hz"], expected.offsets_hz)
    app.multiselect(key="sas_od_fit_free").set_value(["offset_mhz"])
    app.checkbox(key="sas_od_fit_frequency_scale").check().run()
    _run_fit_to_completion(app)
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
    offered = app.multiselect(key="sas_od_fit_free").options
    assert ("Probe power" in offered) == (power > 0.)
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


def _run_profile_to_completion(app):
    app.button(key="sas_profile_run").click().run()
    if "sas_od_profile_job" in app.session_state:
        app.session_state["sas_od_profile_job"].future.result(timeout=60)
        app.run()
    assert not app.exception and not app.error


def test_profile_workflow_refines_relative_detector_and_preserves_native_data(workspace):
    app, exports, _, uploads = workspace
    absolute_csv = uploads[0].getvalue().decode().splitlines()[1:]
    rows = [tuple(map(float, row.split(","))) for row in absolute_csv]
    uploads[0] = _upload(("detuning_GHz,signal\n" + "\n".join(
        f"{x:.17g},{.4 + 1.6 * t:.17g}" for x, t in rows)).encode())
    app.run()
    assert not app.exception and not app.error
    app.checkbox(key="sas_profile_open").check().run()
    app.multiselect(key="sas_profile_vary").set_value(["temp_c"]).run()
    app.number_input(key="sas_profile_delta_temp_c").set_value(3.).run()
    app.slider(key="sas_profile_roi").set_range(15, 480).run()
    app.checkbox(key="sas___csv_auto_correct").uncheck().run()
    assert app.slider(key="sas_profile_roi").value == (15, 480)
    assert not any(button.key == "sas_run_od_fit" for button in app.button)
    before_inputs = exports[-1]["comparison_payload"]["alignment"]
    _run_profile_to_completion(app)
    result = app.session_state["sas_profile_result"][1]
    assert result.relative and result.identifiable and result.n_points == 466
    assert result.condition_values["temp_c"] == pytest.approx(50., abs=.08)
    assert app.session_state["sas__temp_c"] == 49.
    assert exports[-1]["comparison_payload"]["alignment"] == before_inputs
    app.button(key="sas_profile_discard").click().run()
    assert "sas_profile_result" not in app.session_state and app.session_state["sas__temp_c"] == 49.
    app.checkbox(key="sas_profile_open").uncheck().run()
    app.checkbox(key="sas_profile_open").check().run()
    assert app.slider(key="sas_profile_roi").value == (15, 480)
    assert app.number_input(key="sas_profile_delta_temp_c").value == 3.
    _run_profile_to_completion(app)
    app.button(key="sas_profile_apply").click().run()
    assert not app.exception and not app.error
    assert app.session_state["sas__temp_c"] == pytest.approx(50., abs=.08)
    alignment = exports[-1]["comparison_payload"]["alignment"]
    assert alignment["signal_offset"] == pytest.approx(result.alignment.signal_offset)
    assert alignment["signal_scale"] == pytest.approx(result.alignment.signal_scale)


def test_profile_group_uses_manual_seed_and_resets_for_new_measurement(workspace):
    app, _, _, uploads = workspace
    app.run()
    app.checkbox(key="sas_profile_open").check().run()
    app.selectbox(key="sas_profile_group").set_value("85-F3").run()
    group_indices = tuple(app.slider(key="sas_profile_roi").value)
    assert group_indices != (0, 500)
    app.button(key="sas_profile_frame").click().run()
    assert "frame" in app.session_state["sas_profile_options"]
    assert app.number_input(key="sas___csv_x_shift").value == 0.
    app.button(key="sas_profile_unframe").click().run()
    assert "frame" not in app.session_state["sas_profile_options"]
    app.number_input(key="sas___csv_x_shift").set_value(.012)
    app.button(key="sas___csv_apply").click().run()
    assert tuple(app.slider(key="sas_profile_roi").value) == group_indices
    _run_profile_to_completion(app)
    identity, result = app.session_state["sas_profile_result"]
    assert identity[3].shift == pytest.approx(.012) and result.relative is False
    assert result.alignment.signal_scale == 1. and result.alignment.signal_offset == 0.
    app.checkbox(key="sas___csv_auto_correct").uncheck().run()
    assert "sas_profile_result" not in app.session_state
    assert tuple(app.slider(key="sas_profile_roi").value) == group_indices
    uploads.append(_upload(b"detuning_GHz,transmission\n0,.7\n.1,.8\n.2,.9\n.3,1\n.4,.9\n"))
    app.run()
    assert not app.exception and not app.error
    assert not app.checkbox(key="sas_profile_open").value
    app.checkbox(key="sas_profile_open").check().run()
    assert app.selectbox(key="sas_profile_group").value == "full"
    assert tuple(app.slider(key="sas_profile_roi").value) == (0, 4)


@pytest.mark.parametrize("action", ["cancel", "change_roi", "replace_upload"])
def test_profile_job_cancels_without_applying(workspace, monkeypatch, action):
    from threading import Event
    from gabes import od_profile_fit
    from gabes.od_spectrum_fit import ODFitCancelled, ODFitProgress

    started, resume = Event(), Event()

    def controlled_fit(trace, params, *, progress, **kwargs):
        assert progress(ODFitProgress("global", 1, 2.))
        started.set()
        assert resume.wait(15)
        update = ODFitProgress("local", 2, 1.)
        assert not progress(update)
        raise ODFitCancelled(update)

    monkeypatch.setattr(od_profile_fit, "fit_od_profile", controlled_fit)
    app, exports, _, uploads = workspace
    app.run()
    app.checkbox(key="sas_profile_open").check().run()
    try:
        app.button(key="sas_profile_run").click().run()
        assert started.wait(5)
        job = app.session_state["sas_od_profile_job"]
        if action == "cancel":
            app.button(key="sas_cancel_profile_fit").click().run()
        elif action == "change_roi":
            app.slider(key="sas_profile_roi").set_range(10, 457).run()
        else:
            uploads.append(_upload(b"detuning_GHz,transmission\n0,.7\n.1,.8\n.2,.9\n.3,1\n.4,.9\n"))
            app.run()
        assert not app.exception and not app.error
        assert "sas_od_profile_job" not in app.session_state and "sas_profile_result" not in app.session_state
        assert app.session_state["sas__temp_c"] == 49.
        assert exports[-1]["comparison_payload"]["alignment"]["shift"] == 0.
    finally:
        resume.set()
    with pytest.raises(ODFitCancelled):
        job.future.result(timeout=5)


def test_profile_refuses_incompatible_recorded_frequency_line(workspace):
    from gabes.schemes.sas import EXACT_MODEL

    app, _, _, uploads = workspace
    rows = np.loadtxt(uploads[0].getvalue().decode().splitlines()[1:], delimiter=",")
    uploads[:] = [_upload(_legacy_comparison(EXACT_MODEL, rows[:, 0], rows[:, 1]))]
    app.run()
    app.checkbox(key="sas_profile_open").check().run()
    app.selectbox(key="conditions-sas-line").select("D1").run()
    assert not app.exception
    assert any("recorded measurement origin differs" in message.value for message in app.warning)
    assert not any(widget.key == "sas_profile_run" for widget in app.button)
    assert not any(widget.key == "sas_profile_apply" for widget in app.button)
