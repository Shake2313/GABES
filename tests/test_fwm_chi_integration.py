"""The actual susceptibility helper reaches every practical gain readout."""
from copy import deepcopy
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gabes import fwm_chi_gain as chi
from gabes import fwm_gain_calibration as calibration
from gabes.schemes import fwm
from gabes_ui.export import build_export_payload


@pytest.fixture(scope="module")
def case():
    scheme = fwm.FWMScheme()
    params = dict(scheme.defaults(), opd=-1.04858, tpd=-16., temp_c=89.,
                  pump_mw=1000., probe_uw=8., cell_mm=12.,
                  pump_waist_um=950., probe_waist_um=300., seeded_angle_deg=.45,
                  resolution=fwm.FIDELITY_FAST, experimental_gain_calibration=True,
                  gain_calibration_model=calibration.MODEL_CHI)
    return scheme, params, scheme.compute(params)


def test_chi_selection_is_readout_only_and_lab_defaults_survive():
    scheme = fwm.FWMScheme()
    assert calibration.MODELS == (calibration.MODEL_LAB, calibration.MODEL_LITERATURE,
                                  calibration.MODEL_CHI, calibration.MODEL_STARK,
                                  calibration.MODEL_ZEEMAN)
    control = next(spec for spec in scheme.param_schema() if spec.name == "gain_calibration_model")
    assert control.choices == calibration.MODELS
    assert not control.recompute
    assert control.default == calibration.MODEL_LAB
    assert scheme.defaults()["gain_calibration_model"] == calibration.MODEL_LAB
    assert "gain_calibration_model" not in scheme.recompute_keys()
    preset = next(preset for preset in scheme.presets() if preset.name == "2026-10-13 experiment")
    assert preset.values["gain_calibration_model"] == calibration.MODEL_LAB
    assert calibration.profile(model=calibration.MODEL_CHI)["model_id"] == chi.MODEL_ID


def test_chi_off_is_exact_identity_and_does_not_require_a_snapshot(case):
    _, params, raw = case
    legacy = {key: value for key, value in raw.items() if key != "propagation_snapshot"}
    result = calibration.calibrate_spectrum(
        legacy, enabled=False, selected_delta_mhz=params["tpd"], model=calibration.MODEL_CHI)
    assert not result["applied"]
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(result[key], raw[key])
        assert not np.shares_memory(result[key], raw[key])
    with pytest.raises(ValueError, match="fresh propagation snapshot"):
        calibration.calibrate_spectrum(legacy, enabled=True, model=calibration.MODEL_CHI)


def test_actual_chi_focus_cursor_formula_and_strict_json(case):
    import matplotlib.pyplot as plt
    from gabes_ui.liveplot import preview_data

    scheme, params, raw = case
    before = deepcopy(raw)
    direct = chi.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=params["tpd"])
    view = scheme.observables(raw, params)
    headless = scheme.headless_observables(raw, params)
    try:
        result = view["experimental_gain_calibration"]
        assert result["model_id"] == chi.MODEL_ID
        assert result["calibration_model"] == calibration.MODEL_CHI
        assert result["corrected_susceptibility"].shape == raw["propagation_snapshot"]["chi"].shape
        assert result["corrected_field_transfer_small_signal"].shape == raw["T_field_small_signal"].shape
        assert "not a new canonical/noise certification" in result["corrected_transfer_scope"]
        assert result["selected_point"] == direct["selected_point"]
        for key in ("G_s", "G_c"):
            np.testing.assert_array_equal(result[key], direct[key])
            np.testing.assert_array_equal(raw[key], before[key])
        assert view["metrics"] == headless["metrics"]
        assert view["metrics"][1]["label"] == "χ-propagated probe gain"
        assert "before segmented propagation" in view["metrics"][1]["help"]
        assert "gain validation does not validate noise" in view["metrics"][0]["delta"]
        if not result["profile"]["predictive_gate_passed"]:
            assert "Experimental" in view["metrics"][1]["delta"]
            assert "not established" in view["metrics"][1]["delta"]
        figure = view["figure"]
        assert figure.axes[0].get_ylabel() == "Chi-corrected propagation gain"
        assert not figure.axes[0].patches
        expected_axis = np.unique(np.r_[chi._axis(raw), params["tpd"]])
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), expected_axis)
        cursor, _ = preview_data(figure, "fwm", canonical_noise=True)
        assert np.interp(params["tpd"], cursor["curves"][0]["x"], cursor["curves"][0]["y"]) == pytest.approx(direct["selected_point"]["G_s"])
        assert float(figure.axes[0].collections[0].get_offsets()[0, 1]) == pytest.approx(direct["selected_point"]["G_s"])
        table = view["tables"][0]["markdown"]
        assert result["display_name"] in table
        assert result["profile"]["formula"] in table
        assert "χ off-diagonal scale κ" in table
        assert "χ diagonal scale d (absorption and dispersion)" in table
        assert "segmented overlap and pump depletion" in table
        assert "noise covariance or squeezing prediction" in table
        assert "23 points, RF 3022" not in table
        payload = build_export_payload(
            scheme=scheme, raw=raw, params=params, view=view, plot_payloads=[],
            comparison_payload=None, cache_version=scheme.cache_version,
            readout_cache_version="chi-integration-test")
        exported = json.loads(json.dumps(payload, allow_nan=False))["view"]["experimental_gain_calibration"]
        assert exported["model_id"] == chi.MODEL_ID
        assert exported["profile"]["formula"] == result["profile"]["formula"]
        assert exported["validation_status"] == result["validation_status"]
        assert not exported["profile"]["noise_covariance_calibrated"]
        assert not exported["profile"]["susceptibility_identified"]
        assert exported["source_delta_mhz"]["probe"] == exported["source_delta_mhz"]["conjugate"]
        assert exported["selected_point"]["offdiagonal_scale"] == direct["selected_point"]["offdiagonal_scale"]
        for key in ("T_canonical_small_signal", "gain_referred_noise_dB"):
            np.testing.assert_array_equal(raw[key], before[key])
        for key in ("chi", "drift_per_m", "segment_profile"):
            np.testing.assert_array_equal(raw["propagation_snapshot"][key], before["propagation_snapshot"][key])
        assert raw["claim_gate"] == before["claim_gate"]
    finally:
        plt.close(view["figure"])


def test_chi_extrapolation_stays_applied_and_disclosed(case):
    scheme, params, raw = case
    view = scheme.headless_observables(raw, dict(params, tpd=100.))
    result = view["experimental_gain_calibration"]
    point = result["selected_point"]
    assert result["applied"] and point["applied"]
    assert not point["within_training_domain"]
    assert "CONDITION_EXTRAPOLATION" in point["status"]
    assert "condition extrapolation" in view["metrics"][1]["delta"]
    assert np.isfinite([point["G_s"], point["G_c"]]).all()
    assert point["G_s"] >= 0. and point["G_c"] >= 0.


def test_actual_chi_full_scan_both_branches_and_effective_noise(case):
    import matplotlib.pyplot as plt

    scheme, params, raw = case
    plus = dict(raw, branch=1, probe_axis_GHz=raw["probe_axis_GHz"]
                + raw["raman_center_plus_GHz"] - raw["raman_center_minus_GHz"])
    full = dict(D_GHz=params["opd"], minus=raw, plus=plus)
    figure = scheme.extra_views()[0].render(full, params)
    try:
        assert figure._gabes_gain_calibration_model == calibration.MODEL_CHI
        assert figure.axes[0].get_ylabel() == "Chi-corrected propagation gain"
        assert "experimental chi propagation" in figure.axes[0].get_title()
        assert "Gain validation does not validate noise" in figure.axes[1].get_title()
        for index, branch in enumerate(("minus", "plus")):
            expected = chi.calibrate_spectrum(full[branch], enabled=True)
            stored = figure._gabes_experimental_gain_calibration[branch]
            assert stored["model_id"] == chi.MODEL_ID
            assert stored["calibration_model"] == calibration.MODEL_CHI
            assert stored["validation_status"] == expected["validation_status"]
            np.testing.assert_array_equal(figure.axes[0].lines[index].get_ydata(), expected["G_s"])
            np.testing.assert_array_equal(stored["point_status"], expected["point_status"])
            noise = fwm.experimental_jasperse_readout(full[branch], params, gains=expected)
            np.testing.assert_array_equal(figure.axes[1].lines[index].get_ydata(), noise["NF_dB"])
        assert not figure._gabes_experimental_gain_calibration["plus"]["within_training_domain"]
    finally:
        plt.close(figure)


def test_generic_rail_uses_actual_chi_model(case):
    from streamlit.testing.v1 import AppTest

    _, params, raw = case
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from gabes.schemes.fwm import FWMScheme\n"
        "from gabes_ui.controls import render_rail,skey\n"
        "scheme=FWMScheme()\n"
        "for key,value in st.session_state['initial_params'].items():\n"
        "    sk=skey(scheme.name,key)\n"
        "    st.session_state[sk]=st.session_state.get(sk,value)\n"
        "specs=scheme.param_schema()\n"
        # Keep the formatted segmented control in state for AppTest reruns.
        "params=render_rail(st,scheme,specs,skip=[s for s in specs if s.name in ('mode','squeezing_view')])\n"
        "st.session_state['view']=scheme.headless_observables(st.session_state['raw'],params)\n",
        default_timeout=30)
    app.session_state["raw"] = raw
    app.session_state["initial_params"] = dict(params, gain_calibration_model=calibration.MODEL_LAB)
    app.run()
    assert not app.exception
    selector = app.selectbox(key="fwm__gain_calibration_model")
    assert selector.options == list(calibration.MODELS)
    assert selector.value == calibration.MODEL_LAB
    selector.select(calibration.MODEL_CHI).run()
    assert not app.exception
    assert app.session_state["view"]["experimental_gain_calibration"]["model_id"] == chi.MODEL_ID
    assert app.session_state["view"]["metrics"][1]["label"] == "χ-propagated probe gain"
