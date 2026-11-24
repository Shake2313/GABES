"""Source selection reaches all FWM gain readouts without changing the solve.

The distinct boundary implementation exposes routing/label errors. Actual
runtime tests also exercise the packaged numerical model and its validation
metadata through the focused/full-scan, cursor, and export paths.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gabes import fwm_experiment_gain as lab
from gabes import fwm_gain_calibration as calibration
from gabes.schemes import fwm
from gabes_ui.export import build_export_payload


@pytest.fixture(scope="module")
def case():
    scheme = fwm.FWMScheme()
    params = dict(scheme.defaults(), temp_c=95., pump_mw=300., probe_uw=8.,
                  resolution=fwm.FIDELITY_FAST, tpd=-7.314159,
                  experimental_gain_calibration=True)
    return scheme, params, scheme.compute(params)


@pytest.fixture
def literature_boundary(monkeypatch):
    def profile():
        return dict(model_id="test_literature", method="Shared gain correction",
                    validation_status="GROUPED_VALIDATION_NOT_PASSED",
                    noise_covariance_calibrated=False)

    def evaluate(raw, delta_mhz, *, enabled=True):
        delta = np.asarray(delta_mhz, dtype=float)
        axis = lab._axis(raw)
        gs = np.interp(delta, axis, raw["G_s"])
        gc = np.interp(delta, axis, raw["G_c"])
        if enabled:
            gs, gc = .7 * gs + .3, .7 * gc
        return dict(applied=enabled, model_id="test_literature",
                    display_name="Test shared literature correction",
                    G_s=gs, G_c=gc, internal_delta_mhz=delta.copy(),
                    within_training_domain=False,
                    validation_status="GROUPED_VALIDATION_NOT_PASSED",
                    source_delta_mhz=dict(probe=delta.copy(), conjugate=delta.copy()),
                    frequency_support_mhz=None, profile=profile(),
                    gain_quantity="atomic-medium power gain with a literature correction")

    def spectrum(raw, *, enabled=False, selected_delta_mhz=None):
        result = evaluate(raw, lab._axis(raw), enabled=enabled)
        if not enabled:
            result.update(G_s=raw["G_s"].copy(), G_c=raw["G_c"].copy())
        if selected_delta_mhz is not None:
            point = evaluate(raw, selected_delta_mhz, enabled=enabled)
            result["selected_point"] = dict(
                G_s=float(point["G_s"]), G_c=float(point["G_c"]),
                available=True, applied=enabled,
                status="LITERATURE_CALIBRATION; CONDITION_EXTRAPOLATION")
        return result

    original = calibration._module
    module = SimpleNamespace(profile=profile, evaluate=evaluate, calibrate_spectrum=spectrum)
    monkeypatch.setattr(calibration, "_module", lambda model: module if model == calibration.MODEL_LITERATURE else original(model))
    return module


def test_model_control_is_readout_only_and_lab_preset_is_stable():
    scheme = fwm.FWMScheme()
    specs = {s.name: s for s in scheme.param_schema()}
    control = specs["gain_calibration_model"]
    assert control.default == calibration.MODEL_LAB_CURRENT
    assert control.choices == calibration.MODELS
    assert not control.recompute
    assert "gain_calibration_model" not in scheme.recompute_keys()
    presets = {p.name: p.values for p in scheme.presets()}
    assert presets["2026-10-13 experiment"]["gain_calibration_model"] == calibration.MODEL_LAB_CURRENT
    assert presets["2026-10-13 experiment"]["experimental_gain_calibration"]
    assert not presets["Sim Gold Reference"]["experimental_gain_calibration"]


@pytest.mark.parametrize("selector", [None, "2026-10-13 lab", "2026-10-13 lab (exact-source refit)",
                                      calibration.MODEL_LAB_CURRENT])
def test_live_lab_selectors_use_the_corrected_model(case, selector):
    _, _, raw = case
    options = {} if selector is None else {"model": selector}
    expected = lab.evaluate(raw, [-100., -3.2, 200.], model_id=lab.CURRENT_MODEL_ID)
    actual = calibration.evaluate(raw, [-100., -3.2, 200.], **options)
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(actual[key], expected[key])
    profile = calibration.profile(**options)
    assert profile["model_id"] == lab.CURRENT_MODEL_ID
    assert profile["calibration_model"] == calibration.MODEL_LAB_CURRENT
    for result in (actual, calibration.calibrate_spectrum(raw, **options)):
        assert result["profile"]["model_id"] == lab.CURRENT_MODEL_ID
        assert result["calibration_model"] == calibration.MODEL_LAB_CURRENT


def test_sabes_imports_saved_fwm_selection_using_the_current_lab_profile():
    from sabes import bridge
    from tests.test_fwm_measured_gain_integration import reference_raw

    result = bridge.run(solve=False, gain_calibration_model="2026-10-13 lab (exact-source refit)",
                        experimental_gain_calibration=True)
    assert result.params["gain_calibration_model"] == calibration.MODEL_LAB_CURRENT
    selected = bridge.noise_observables(reference_raw(), result.params)["experimental_gain_calibration"]
    assert selected["profile"]["model_id"] == lab.CURRENT_MODEL_ID


def test_unknown_calibration_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        calibration.profile(model="not a calibration model")


def test_lab_preset_weak_gains_and_positive_noise_use_the_unrounded_operating_point():
    from gabes.core import blas_single_thread
    from gabes_ui.readout import present_metrics

    scheme = fwm.FWMScheme()
    lab_preset = next(p for p in scheme.presets() if p.name == "2026-10-13 experiment")
    params = {**scheme.defaults(), **lab_preset.values, "tpd": 50.}
    with blas_single_thread():
        raw = scheme.compute({k: params[k] for k in scheme.recompute_keys()})
        view = scheme.headless_observables(raw, params)
    original_metrics = deepcopy(view["metrics"])
    noise, probe, conjugate = present_metrics(view["metrics"], scheme_name="fwm", raw=raw)[:3]
    nf_db = view["noise_operating_point"]["NF_dB"]
    assert nf_db > .005
    assert noise["numeric_value"] == nf_db
    assert noise["value"] == f"{nf_db:+.2f} dB"
    assert "above shot noise" in noise["delta"] and "Jasperse" in noise["delta"]
    for key, arm in (("G_s", probe), ("G_c", conjugate)):
        gain = view["gain_operating_point"][key]
        assert 0.0 < gain < .02
        assert arm["numeric_value"] == gain
        assert "RF extrapolation" in arm["delta"]
    assert 1e-4 <= probe["numeric_value"] < 1e-3
    assert probe["value"] == f"{probe['numeric_value'] * 1e4:.2f} × 10⁻⁴"
    assert conjugate["value"] == f"{conjugate['numeric_value']:.3g}"
    assert view["metrics"] == original_metrics


@pytest.mark.parametrize("model", calibration.MODELS)
def test_disabled_selection_keeps_atomic_gains_exact(case, literature_boundary, model):
    _, _, raw = case
    result = calibration.calibrate_spectrum(raw, model=model, enabled=False)
    assert not result["applied"]
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(result[key], raw[key])
        assert not np.shares_memory(result[key], raw[key])


def test_literature_focus_cursor_export_and_source_labels_agree(case, literature_boundary):
    import matplotlib.pyplot as plt
    from gabes_ui.liveplot import preview_data

    scheme, params, raw = case
    before = deepcopy(raw)
    params = dict(params, gain_calibration_model=calibration.MODEL_LITERATURE)
    view = scheme.observables(raw, params)
    headless = scheme.headless_observables(raw, params)
    try:
        assert view["metrics"] == headless["metrics"]
        gain = view["experimental_gain_calibration"]
        point = gain["selected_point"]
        assert gain["calibration_model"] == calibration.MODEL_LITERATURE
        assert "not established" in view["metrics"][1]["delta"]
        assert "condition extrapolation" in view["metrics"][1]["delta"]
        assert view["metrics"][1]["label"] == "Literature-corrected probe gain"
        assert "gain validation does not validate noise" in view["metrics"][0]["delta"]
        figure = view["figure"]
        assert figure.axes[0].get_ylabel() == "Literature-corrected power gain"
        assert not figure.axes[0].patches  # The 10/13 RF support is not this model's data.
        assert len(figure.axes[0].lines[0].get_xdata()) == len(raw["G_s"]) + 1
        data, _ = preview_data(figure, "fwm", canonical_noise=True)
        assert np.interp(params["tpd"], data["curves"][0]["x"], data["curves"][0]["y"]) == pytest.approx(point["G_s"])
        assert float(figure.axes[0].collections[0].get_offsets()[0, 1]) == pytest.approx(point["G_s"])
        assert "Multiple 85Rb D1 experimental papers" in view["tables"][0]["markdown"]
        assert "2026-10-13 measurements, 23 points" not in view["tables"][0]["markdown"]
        payload = build_export_payload(
            scheme=scheme, raw=raw, params=params, view=view,
            plot_payloads=[], comparison_payload=None,
            cache_version=scheme.cache_version, readout_cache_version="literature-routing-test")
        decoded = json.loads(json.dumps(payload, allow_nan=False))
        exported = decoded["view"]["experimental_gain_calibration"]
        assert exported["calibration_model"] == calibration.MODEL_LITERATURE
        assert exported["validation_status"] == "GROUPED_VALIDATION_NOT_PASSED"
        for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
            np.testing.assert_array_equal(raw[key], before[key])
        assert raw["claim_gate"] == before["claim_gate"]
    finally:
        plt.close(view["figure"])


def test_literature_full_scan_routes_both_branches(case, literature_boundary):
    import matplotlib.pyplot as plt

    scheme, params, raw = case
    params = dict(params, gain_calibration_model=calibration.MODEL_LITERATURE)
    plus = dict(raw, branch=1, probe_axis_GHz=raw["probe_axis_GHz"]
                + raw["raman_center_plus_GHz"] - raw["raman_center_minus_GHz"])
    full = dict(D_GHz=params["opd"], minus=raw, plus=plus)
    figure = scheme.extra_views()[0].render(full, params)
    try:
        assert figure._gabes_gain_calibration_model == calibration.MODEL_LITERATURE
        assert set(figure._gabes_experimental_gain_calibration) == {"minus", "plus"}
        assert figure.axes[0].get_ylabel() == "Literature-corrected power gain"
        for index, branch in enumerate(("minus", "plus")):
            expected = literature_boundary.calibrate_spectrum(full[branch], enabled=True)
            np.testing.assert_allclose(figure.axes[0].lines[index].get_ydata(), expected["G_s"])
    finally:
        plt.close(figure)


def test_actual_literature_runtime_focus_domain_cursor_and_json(case):
    import matplotlib.pyplot as plt
    from gabes import fwm_literature_gain as literature
    from gabes_ui.liveplot import preview_data

    scheme, params, raw = case
    before = deepcopy(raw)
    params = dict(params, gain_calibration_model=calibration.MODEL_LITERATURE)
    direct = literature.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=params["tpd"])
    dispatched = calibration.calibrate_spectrum(
        raw, enabled=True, selected_delta_mhz=params["tpd"], model=calibration.MODEL_LITERATURE)
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(dispatched[key], direct[key])
    assert dispatched["selected_point"] == direct["selected_point"]
    # The chosen point is covered even though the wide plotted scan is not.
    assert not direct["within_training_domain"]
    assert direct["selected_point"]["within_training_domain"]
    assert direct["selected_point"]["status"] == "LITERATURE_CALIBRATION"
    record = calibration.profile(model=calibration.MODEL_LITERATURE)
    assert record["validation_status"] == "GROUPED_PREDICTION_NOT_ESTABLISHED"
    assert not record["predictive_gate_passed"]
    assert not record["noise_covariance_calibrated"]
    assert not record["susceptibility_identified"]
    assert record["calibration_model"] == calibration.MODEL_LITERATURE
    # Metadata must remain ordinary strict JSON for source inspection/export.
    assert json.loads(json.dumps(record, allow_nan=False)) == record

    view = scheme.observables(raw, params)
    headless = scheme.headless_observables(raw, params)
    try:
        assert view["metrics"] == headless["metrics"]
        point = view["experimental_gain_calibration"]["selected_point"]
        assert point == direct["selected_point"]
        assert "not established" in view["metrics"][1]["delta"]
        assert "condition extrapolation" not in view["metrics"][1]["delta"]
        assert "gain validation does not validate noise" in view["metrics"][0]["delta"]
        data, _ = preview_data(view["figure"], "fwm", canonical_noise=True)
        probe_curve = data["curves"][0]
        assert np.interp(params["tpd"], probe_curve["x"], probe_curve["y"]) == pytest.approx(point["G_s"])
        conjugate_curve = view["figure"].axes[0].lines[3]
        assert np.interp(params["tpd"], conjugate_curve.get_xdata(), conjugate_curve.get_ydata()) == pytest.approx(point["G_c"])
        marker = view["figure"].axes[0].collections[0].get_offsets()[0, 1]
        assert float(marker) == pytest.approx(point["G_s"])
        assert not view["figure"].axes[0].patches
        payload = build_export_payload(
            scheme=scheme, raw=raw, params=params, view=view, plot_payloads=[],
            comparison_payload=None, cache_version=scheme.cache_version,
            readout_cache_version="actual-literature-test")
        exported = json.loads(json.dumps(payload, allow_nan=False))["view"]["experimental_gain_calibration"]
        assert exported["profile"]["validation_status"] == record["validation_status"]
        assert exported["profile"]["coefficients"] == record["coefficients"]
        assert exported["dipole_normalization"] == {
            "training": "summed-hyperfine-over-12", "applied": "uniform-manifold-rms"}
        assert "summed-hyperfine-over-12 / uniform-manifold-rms" in view["tables"][0]["markdown"]
        assert not exported["profile"]["noise_covariance_calibrated"]
        assert exported["selected_point"]["within_training_domain"]
        assert not exported["within_training_domain"]
        assert exported["frequency_support_mhz"] is None
        assert exported["source_delta_mhz"]["probe"] == exported["source_delta_mhz"]["conjugate"]
        for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
            np.testing.assert_array_equal(raw[key], before[key])
        assert raw["claim_gate"] == before["claim_gate"]
    finally:
        plt.close(view["figure"])

    extrapolated = scheme.headless_observables(raw, dict(params, tpd=100.))
    outside = extrapolated["experimental_gain_calibration"]["selected_point"]
    assert outside["applied"] and not outside["within_training_domain"]
    assert "CONDITION_EXTRAPOLATION" in outside["status"]
    assert "condition extrapolation" in extrapolated["metrics"][1]["delta"]


def test_actual_literature_runtime_full_scan_metadata_and_noise(case):
    import matplotlib.pyplot as plt
    from gabes import fwm_literature_gain as literature

    scheme, params, raw = case
    params = dict(params, gain_calibration_model=calibration.MODEL_LITERATURE)
    plus = dict(raw, branch=1, probe_axis_GHz=raw["probe_axis_GHz"]
                + raw["raman_center_plus_GHz"] - raw["raman_center_minus_GHz"])
    full = dict(D_GHz=params["opd"], minus=raw, plus=plus)
    figure = scheme.extra_views()[0].render(full, params)
    try:
        assert figure._gabes_gain_calibration_model == calibration.MODEL_LITERATURE
        assert "not established" in figure.axes[0].get_title()
        assert "Gain validation does not validate noise" in figure.axes[1].get_title()
        stored = figure._gabes_experimental_gain_calibration
        for index, branch in enumerate(("minus", "plus")):
            direct = literature.calibrate_spectrum(full[branch], enabled=True)
            for key in ("G_s", "G_c"):
                np.testing.assert_array_equal(stored[branch][key], direct[key])
            np.testing.assert_array_equal(figure.axes[0].lines[index].get_ydata(), direct["G_s"])
            assert stored[branch]["validation_status"] == "NOT_REVALIDATED_FOR_CHANGED_DIPOLE_NORMALIZATION"
            assert stored[branch]["archived_validation_status"] == "GROUPED_PREDICTION_NOT_ESTABLISHED"
            assert not stored[branch]["profile"]["noise_covariance_calibrated"]
            assert not stored[branch]["within_training_domain"]
            np.testing.assert_array_equal(stored[branch]["point_status"], direct["point_status"])
            np.testing.assert_array_equal(direct["source_delta_mhz"]["probe"], direct["source_delta_mhz"]["conjugate"])
            noise = fwm.experimental_jasperse_readout(full[branch], params, gains=direct)
            np.testing.assert_array_equal(figure.axes[1].lines[index].get_ydata(), noise["NF_dB"])
        assert np.all(stored["plus"]["point_status"] == "LITERATURE_CALIBRATION; CONDITION_EXTRAPOLATION")
        assert np.any(stored["minus"]["point_status"] == "LITERATURE_CALIBRATION")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("model", (calibration.MODEL_LITERATURE, calibration.MODEL_CHI, calibration.MODEL_STARK))
def test_grouped_gain_diagnostics_follow_the_actual_atomic_response(case, model, monkeypatch):
    scheme, params, raw = case
    module = calibration._module(model)
    archived = module.profile()["validation_status"]
    legacy_id = "summed-hyperfine-over-12"
    rms_id = "uniform-manifold-rms"
    for source_id, expected_status in ((rms_id, "NOT_REVALIDATED_FOR_CHANGED_DIPOLE_NORMALIZATION"),
                                       (legacy_id, archived), (None, "SOURCE_DIPOLE_NORMALIZATION_UNKNOWN")):
        # Hold numerical gains fixed to isolate the provenance logic.
        source = dict(raw, gain_source_model={"dipole_normalization": source_id},
                      coupling_normalization_ledger={})
        native = source
        if model == calibration.MODEL_STARK:
            # The applied normalization must come from the recomputed response.
            native = dict(raw, gain_source_model={"dipole_normalization": "incoming-spectrum"})
            monkeypatch.setattr(module, "corrected_response", lambda *args: source)
        result = module.calibrate_spectrum(native, enabled=True, selected_delta_mhz=params["tpd"])
        assert result["applied"] and np.isfinite(result["selected_point"]["G_s"])
        assert result["dipole_normalization"] == {"training": legacy_id, "applied": source_id}
        assert result["archived_validation_status"] == archived
        assert result["validation_status"] == result["selected_point"]["validation_status"] == expected_status


def test_generic_rail_model_selection_and_lab_preset_reset(literature_boundary):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        "import streamlit as st\n"
        "from gabes.schemes.fwm import FWMScheme\n"
        "from gabes_ui.controls import render_rail,skey\n"
        "from gabes_ui.defaults import render_defaults\n"
        "from tests.test_fwm_measured_gain_integration import reference_raw\n"
        "scheme=FWMScheme()\n"
        "for key,value in scheme.defaults().items():\n"
        "    sk=skey(scheme.name,key)\n"
        "    st.session_state[sk]=st.session_state.get(sk,value)\n"
        "render_defaults(st,scheme)\n"
        "specs=scheme.param_schema()\n"
        # AppTest cannot serialize the formatted segmented choice on rerun.
        # Keep it in state, as in the measured-reference preset integration.
        "params=render_rail(st,scheme,specs,skip=[s for s in specs if s.name in ('mode','squeezing_view')])\n"
        "st.session_state['view']=scheme.headless_observables(reference_raw(params),params)\n",
        default_timeout=30,
    )
    app.session_state["fwm__gain_calibration_model"] = "2026-10-13 lab (exact-source refit)"
    app.run()
    assert not app.exception
    assert "fwm__gain_calibration_model" not in [s.key for s in app.selectbox]
    app.checkbox(key="fwm__experimental_gain_calibration").check().run()
    assert not app.exception
    assert app.selectbox(key="fwm__gain_calibration_model").value == calibration.MODEL_LAB_CURRENT
    assert app.session_state["view"]["experimental_gain_calibration"]["profile"]["model_id"] == lab.CURRENT_MODEL_ID
    app.selectbox(key="fwm___named_default").select("builtin:0").run()
    assert not app.exception
    assert app.selectbox(key="fwm__gain_calibration_model").value == calibration.MODEL_LAB_CURRENT
    app.selectbox(key="fwm__gain_calibration_model").select(calibration.MODEL_LITERATURE).run()
    assert not app.exception
    assert app.session_state["view"]["experimental_gain_calibration"]["model_id"] == "test_literature"
    app.selectbox(key="fwm___named_default").select("builtin:1").run()
    app.selectbox(key="fwm___named_default").select("builtin:0").run()
    assert not app.exception
    assert app.selectbox(key="fwm__gain_calibration_model").value == calibration.MODEL_LAB_CURRENT


@pytest.mark.parametrize("selector", [None, "2026-10-13 lab", "2026-10-13 lab (exact-source refit)"])
@pytest.mark.parametrize("enabled", [False, True])
def test_saved_lab_settings_migrate_the_selector_and_preserve_other_controls(selector, enabled):
    from gabes_ui.defaults_store import create_default, parse_default

    scheme = fwm.FWMScheme()
    params = dict(scheme.defaults(), experimental_gain_calibration=enabled, tpd=-16.0)
    record = create_default(scheme, "Old lab settings", params)
    if selector is None:
        del record["params"]["gain_calibration_model"]
    else:
        record["params"]["gain_calibration_model"] = selector
    restored = parse_default(scheme, record)["params"]
    assert restored == params
    assert restored["gain_calibration_model"] == calibration.MODEL_LAB_CURRENT
    assert calibration.profile(model=restored["gain_calibration_model"])["model_id"] == lab.CURRENT_MODEL_ID
    assert calibration.MODEL_LAB_REFIT == calibration.MODEL_LAB_CURRENT
