"""Actual Stark dispatcher/helper through focused/full plots, cursor and export.

The expensive atomic recompute boundary is deterministic here; the actual
Stark helper, complex correction and optical propagation remain in the route.
"""
from copy import deepcopy
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gabes import fwm_chi_gain as chi
from gabes import fwm_gain_calibration as calibration
from gabes import fwm_stark_gain as stark
from gabes.schemes import fwm
from gabes_ui.export import build_export_payload


@pytest.fixture(scope="module")
def case():
    scheme = fwm.FWMScheme()
    params = dict(scheme.defaults(), opd=-1.04858, tpd=-16., temp_c=89.,
                  pump_mw=1000., probe_uw=8., cell_mm=12.,
                  pump_waist_um=950., probe_waist_um=300., seeded_angle_deg=.45,
                  resolution=fwm.FIDELITY_FAST, experimental_gain_calibration=True,
                  noise_model=fwm.NOISE_EXPERIMENTAL_JASPERSE,
                  gain_calibration_model=calibration.MODEL_STARK)
    return scheme, params, scheme.compute(params)


@pytest.fixture
def response_boundary(case, monkeypatch):
    _, _, native = case
    record = chi.profile()
    record.update(pump_stark_fraction=.25, coefficients=[np.log(.9), 0., 0., np.log(1.02)],
                  predictive_gate_passed=False, validation_status="NOT_EVALUATED_TEST_FIXTURE")
    monkeypatch.setattr(stark, "_record", lambda: deepcopy(record))
    calls = []

    def recompute(encoded_inputs, fraction):
        inputs = json.loads(encoded_inputs)
        calls.append((inputs, fraction))
        response = deepcopy(native)
        if inputs["branch"] == 1:
            response["branch"] = 1
            response["probe_axis_GHz"] += (native["raman_center_plus_GHz"]
                                             - native["raman_center_minus_GHz"])
        snap = response["propagation_snapshot"]
        snap["drift_per_m"][:, 0, 1] *= 1.7
        snap["drift_per_m"][:, 1, 0] *= 1.7
        snap["chi"][:, 0, 1] *= 1.7
        snap["chi"][:, 1, 0] *= 1.7
        response["pump_stark_response"] = dict(fraction=fraction, applied=True,
            native_shift_mhz=-160., counterterm_mhz=-120., corrected_gap_shift_mhz=-40.,
            response_scope="same corrected generator for populations and all susceptibilities")
        return response

    monkeypatch.setattr(stark, "_solve_response", recompute)
    return calls


def test_stark_choice_is_opt_in_lazy_and_lab_default_is_preserved(case, response_boundary):
    scheme, _, _ = case
    control = next(s for s in scheme.param_schema() if s.name == "gain_calibration_model")
    assert calibration.MODELS[3] == calibration.MODEL_STARK
    assert calibration._module(calibration.MODEL_STARK) is stark
    assert calibration.profile(model=calibration.MODEL_STARK)["model_id"] == stark.MODEL_ID
    assert control.choices == calibration.MODELS and not control.recompute
    assert control.default == scheme.defaults()["gain_calibration_model"] == calibration.MODEL_LAB
    assert "gain_calibration_model" not in scheme.recompute_keys()
    assert next(p for p in scheme.presets() if p.name == "2026-10-13 experiment").values["gain_calibration_model"] == calibration.MODEL_LAB


def test_stark_off_is_exact_without_recompute_context(case, response_boundary):
    _, params, raw = case
    legacy = {k: v for k, v in raw.items() if k not in ("propagation_snapshot", "stark_recompute_context")}
    off = calibration.calibrate_spectrum(legacy, enabled=False,
        selected_delta_mhz=params["tpd"], model=calibration.MODEL_STARK)
    assert not off["applied"] and off["pump_stark_fraction"] == 1.
    assert not response_boundary
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(off[key], raw[key])
        assert not np.shares_memory(off[key], raw[key])
    with pytest.raises(ValueError, match="fresh atomic spectrum"):
        calibration.calibrate_spectrum(legacy, enabled=True, model=calibration.MODEL_STARK)


def test_actual_stark_helper_focus_cursor_export_and_raw_nonmutation(case, response_boundary):
    import matplotlib.pyplot as plt
    from gabes_ui.liveplot import preview_data

    scheme, params, raw = case
    before = deepcopy(raw)
    direct = stark.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=params["tpd"])
    view = scheme.observables(raw, params)
    headless = scheme.headless_observables(raw, params)
    try:
        result = view["experimental_gain_calibration"]
        assert result["model_id"] == stark.MODEL_ID
        assert result["calibration_model"] == calibration.MODEL_STARK
        assert result["selected_point"] == direct["selected_point"]
        assert result["pump_stark_fraction"] == .25
        assert result["pump_stark_response"]["counterterm_mhz"] == -120.
        assert result["corrected_atomic_floquet_status"] == "CONVERGED"
        assert all(fraction == .25 for _, fraction in response_boundary)
        assert view["metrics"] == headless["metrics"]
        assert view["metrics"][1]["label"] == "Stark + χ probe gain"
        assert "pump Hamiltonian self-energy" in view["metrics"][1]["help"]
        assert "noise" in view["metrics"][0]["delta"]
        ax = view["figure"].axes[0]
        assert ax.get_ylabel() == "Stark + chi propagation gain"
        assert not ax.patches
        cursor, _ = preview_data(view["figure"], "fwm", canonical_noise=True)
        point_gain = direct["selected_point"]["G_s"]
        assert np.interp(params["tpd"], cursor["curves"][0]["x"], cursor["curves"][0]["y"]) == pytest.approx(point_gain)
        assert float(ax.collections[0].get_offsets()[0, 1]) == pytest.approx(point_gain)
        table = view["tables"][0]["markdown"]
        for label in ("Pump/Stark self-energy fraction", "Pump self-energy counterterm [MHz]",
                      "Corrected pump-dressed gap shift [MHz]", "Corrected atomic Floquet status"):
            assert label in table
        assert "23 points, RF 3022" not in table
        payload = build_export_payload(scheme=scheme, raw=raw, params=params, view=view,
            plot_payloads=[], comparison_payload=None, cache_version=scheme.cache_version,
            readout_cache_version="stark-integration-test")
        exported = json.loads(json.dumps(payload, allow_nan=False))["view"]["experimental_gain_calibration"]
        assert exported["model_id"] == stark.MODEL_ID
        assert exported["pump_stark_fraction"] == .25
        assert exported["pump_stark_response"]["counterterm_mhz"] == -120.
        assert not exported["profile"]["noise_covariance_calibrated"]
        assert exported["source_delta_mhz"]["probe"] == exported["source_delta_mhz"]["conjugate"]
        for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
            np.testing.assert_array_equal(raw[key], before[key])
        for key in ("chi", "drift_per_m", "segment_profile"):
            np.testing.assert_array_equal(raw["propagation_snapshot"][key], before["propagation_snapshot"][key])
        assert raw["claim_gate"] == before["claim_gate"]
    finally:
        plt.close(view["figure"])


def test_actual_stark_full_scan_routes_both_branches_and_preserves_noise_scope(case, response_boundary):
    import matplotlib.pyplot as plt

    scheme, params, raw = case
    plus = deepcopy(raw)
    plus["branch"] = 1
    plus["probe_axis_GHz"] += raw["raman_center_plus_GHz"]-raw["raman_center_minus_GHz"]
    plus["stark_recompute_context"]["branch"] = 1
    full = dict(D_GHz=params["opd"], minus=raw, plus=plus)
    figure = scheme.extra_views()[0].render(full, params)
    try:
        assert figure._gabes_gain_calibration_model == calibration.MODEL_STARK
        assert figure.axes[0].get_ylabel() == "Stark + chi propagation gain"
        assert "experimental Stark + chi propagation" in figure.axes[0].get_title()
        assert "Gain validation does not validate noise" in figure.axes[1].get_title()
        for i, branch in enumerate(("minus", "plus")):
            expected = stark.calibrate_spectrum(full[branch], enabled=True)
            stored = figure._gabes_experimental_gain_calibration[branch]
            assert stored["model_id"] == stark.MODEL_ID
            assert stored["pump_stark_fraction"] == .25
            np.testing.assert_array_equal(figure.axes[0].lines[i].get_ydata(), expected["G_s"])
            noise = fwm.experimental_jasperse_readout(full[branch], params, gains=expected)
            np.testing.assert_array_equal(figure.axes[1].lines[i].get_ydata(), noise["NF_dB"])
        assert not figure._gabes_experimental_gain_calibration["plus"]["within_training_domain"]
        assert {inputs["branch"] for inputs, _ in response_boundary} == {-1, 1}
    finally:
        plt.close(figure)


def test_stark_extrapolation_remains_applied(case, response_boundary):
    scheme, params, raw = case
    view = scheme.headless_observables(raw, dict(params, tpd=100.))
    result = view["experimental_gain_calibration"]
    assert result["applied"] and result["selected_point"]["applied"]
    assert "condition extrapolation" in view["metrics"][1]["delta"]
    assert np.isfinite([result["selected_point"]["G_s"], result["selected_point"]["G_c"]]).all()


def test_generated_production_profile_reaches_real_atomic_recomputation_and_export(case):
    scheme, params, raw = case
    before = deepcopy(raw)
    stark._record.cache_clear()
    stark._solve_response.cache_clear()
    record = stark.profile()
    fraction = record["pump_stark_fraction"]
    result = calibration.calibrate_spectrum(raw, enabled=True,
        selected_delta_mhz=params["tpd"], model=calibration.MODEL_STARK)
    assert result["model_id"] == stark.MODEL_ID
    assert result["pump_stark_fraction"] == fraction
    assert result["corrected_atomic_floquet_status"] == "CONVERGED"
    assert result["pump_stark_response"]["fraction"] == fraction
    if fraction != 1.:
        assert stark._solve_response.cache_info().misses == 1
        assert result["pump_stark_response"]["applied"]
        assert result["pump_stark_response"]["counterterm_mhz"] != 0.
    view = scheme.headless_observables(raw, params)
    assert view["experimental_gain_calibration"]["selected_point"] == result["selected_point"]
    assert view["metrics"][1]["label"] == "Stark + χ probe gain"
    payload = build_export_payload(scheme=scheme, raw=raw, params=params, view=view,
        plot_payloads=[], comparison_payload=None, cache_version=scheme.cache_version,
        readout_cache_version="production-stark-profile")
    exported = json.loads(json.dumps(payload, allow_nan=False))["view"]["experimental_gain_calibration"]
    assert exported["profile"]["source_sha256"] == record["source_sha256"]
    assert exported["pump_stark_response"] == result["pump_stark_response"]
    assert not exported["profile"]["noise_covariance_calibrated"]
    for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
        np.testing.assert_array_equal(raw[key], before[key])
    assert raw["claim_gate"] == before["claim_gate"]


def test_generic_rail_selects_stark_and_returns_to_the_lab_model(case):
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
    app.selectbox(key="fwm__gain_calibration_model").select(calibration.MODEL_STARK).run()
    assert not app.exception
    result = app.session_state["view"]["experimental_gain_calibration"]
    assert result["model_id"] == stark.MODEL_ID
    assert result["pump_stark_response"]["fraction"] == stark.profile()["pump_stark_fraction"]
    assert app.session_state["view"]["metrics"][1]["label"] == "Stark + χ probe gain"
    app.selectbox(key="fwm__gain_calibration_model").select(calibration.MODEL_LAB).run()
    assert not app.exception
    assert app.session_state["view"]["experimental_gain_calibration"]["calibration_model"] == calibration.MODEL_LAB
