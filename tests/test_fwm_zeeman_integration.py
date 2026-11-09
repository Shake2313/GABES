"""Actual Zeeman dispatcher/helper at the UI, cursor and export boundaries.

Only the expensive atomic response boundary is mocked in the focused routing
checks; the real residual-chi correction and optical propagation are exercised.
"""
from copy import deepcopy
import json

import matplotlib
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")

from gabes import fwm_chi_gain as chi
from gabes import fwm_gain_calibration as calibration
from gabes import fwm_zeeman_gain as zeeman
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
                  gain_calibration_model=calibration.MODEL_ZEEMAN)
    return scheme, params, scheme.compute(params)


@pytest.fixture
def atomic_boundary(monkeypatch):
    record = chi.profile()
    record.update(coefficients=[float(np.log(.9)), 0.], velocity_step_m_s=2.,
                  velocity_cutoff_sigma=6., predictive_gate_passed=False,
                  validation_status="NOT_EVALUATED_TEST_FIXTURE")
    monkeypatch.setattr(zeeman, "_record", lambda: deepcopy(record))
    calls = []

    def build_response(raw, delta_mhz, **settings):
        delta = np.atleast_1d(np.asarray(delta_mhz, float))
        calls.append((int(raw["branch"]), delta.copy(), settings))
        matrix, mismatch, _ = chi.sample_snapshot(raw, delta)
        matrix[:, 0, 1] *= 1.8
        matrix[:, 1, 0] *= 1.8
        source_chi = raw["propagation_snapshot"]["chi"]
        source_axis = chi._axis(raw)
        sampled = (PchipInterpolator(source_axis, source_chi.real, axis=0)(delta)
                   +1j*PchipInterpolator(source_axis, source_chi.imag, axis=0)(delta))
        sampled[:, 0, 1] *= 1.8
        sampled[:, 1, 0] *= 1.8
        snapshot = dict(raw["propagation_snapshot"], drift_per_m=matrix, chi=sampled)
        center = raw["raman_center_minus_GHz"] if raw["branch"] == -1 else raw["raman_center_plus_GHz"]
        return dict(probe_axis_GHz=center+delta*.001, delta_k_z=mismatch,
            propagation_snapshot=snapshot,
            zeeman_response_provenance=dict(magnetic_field_G=[0., 0., 0.],
                polarization="orthogonal linear; coherent sigma components",
                external_sublevel_population_factor=1., practical_residual=.74,
                pump_phase=dict(pump_index_added_mismatch_per_m=25., fixed_phase_coefficient=1.)),
            zeeman_pump_populations=dict(ground_F2=.6, ground_F3=.35, excited_total=.05,
                trace=1., sublevels=[], ground_alignment={}),
            corrected_atomic_response_diagnostics=dict(finite=True,
                max_pump_trace_error=1e-14, minimum_pump_eigenvalue=0.))

    monkeypatch.setattr(zeeman, "build_response", build_response)
    return calls


def test_zeeman_selector_is_opt_in_with_stable_native_and_lab_defaults(case, atomic_boundary):
    scheme, _, _ = case
    assert calibration.MODELS == (calibration.MODEL_LAB, calibration.MODEL_LITERATURE,
        calibration.MODEL_CHI, calibration.MODEL_STARK, calibration.MODEL_ZEEMAN)
    assert calibration._module(calibration.MODEL_ZEEMAN) is zeeman
    assert calibration.profile(model=calibration.MODEL_ZEEMAN)["model_id"] == zeeman.MODEL_ID
    control = next(s for s in scheme.param_schema() if s.name == "gain_calibration_model")
    assert control.choices == calibration.MODELS and not control.recompute
    assert "cached 24-state response" in control.help
    assert control.default == scheme.defaults()["gain_calibration_model"] == calibration.MODEL_LAB
    assert "gain_calibration_model" not in scheme.recompute_keys()
    assert not scheme.defaults()["experimental_gain_calibration"]
    assert next(p for p in scheme.presets() if p.name == "2026-10-13 experiment").values["gain_calibration_model"] == calibration.MODEL_LAB


def test_zeeman_off_is_exact_and_avoids_atomic_response(case, atomic_boundary):
    _, params, raw = case
    legacy = {k: v for k, v in raw.items() if k not in ("propagation_snapshot", "stark_recompute_context")}
    off = calibration.calibrate_spectrum(legacy, enabled=False,
        selected_delta_mhz=params["tpd"], model=calibration.MODEL_ZEEMAN)
    assert not off["applied"] and not atomic_boundary
    assert off["zeeman_response_provenance"] == {}
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(off[key], raw[key])
        assert not np.shares_memory(off[key], raw[key])


def test_actual_zeeman_helper_focus_cursor_export_and_raw_provenance(case, atomic_boundary):
    import matplotlib.pyplot as plt
    from gabes_ui.liveplot import preview_data

    scheme, params, raw = case
    before = deepcopy(raw)
    direct = zeeman.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=params["tpd"])
    view = scheme.observables(raw, params)
    headless = scheme.headless_observables(raw, params)
    try:
        result = view["experimental_gain_calibration"]
        assert result["model_id"] == zeeman.MODEL_ID
        assert result["calibration_model"] == calibration.MODEL_ZEEMAN
        assert result["selected_point"] == direct["selected_point"]
        assert result["diagonal_susceptibility_scale"] == 1.
        assert result["zeeman_pump_populations"]["ground_F2"] == .6
        assert result["corrected_atomic_response_diagnostics"]["finite"]
        assert atomic_boundary and all(x[2]["method"] == "poles" for x in atomic_boundary)
        assert view["metrics"] == headless["metrics"]
        assert view["metrics"][1]["label"] == "Zeeman pump + χ probe gain"
        assert "pump optical pumping" in view["metrics"][1]["help"]
        assert "zero-field" in view["metrics"][1]["help"]
        ax = view["figure"].axes[0]
        assert ax.get_ylabel() == "Zeeman pump + chi propagation gain"
        assert not ax.patches
        cursor, _ = preview_data(view["figure"], "fwm", canonical_noise=True)
        gain = direct["selected_point"]["G_s"]
        assert np.interp(params["tpd"], cursor["curves"][0]["x"], cursor["curves"][0]["y"]) == pytest.approx(gain)
        assert float(ax.collections[0].get_offsets()[0, 1]) == pytest.approx(gain)
        table = view["tables"][0]["markdown"]
        for term in ("Zeeman atomic response", "Conditional zero field", "Pump populations Fg2 / Fg3",
                     "Pump excited-state population", "Zeeman response finite",
                     "Effective phase mismatch", "Pump-index mismatch contribution", "Fixed physical coefficient 1"):
            assert term in table
        assert "Pump/Stark self-energy fraction" not in table
        assert "Pump self-energy counterterm" not in table
        payload = build_export_payload(scheme=scheme, raw=raw, params=params, view=view,
            plot_payloads=[], comparison_payload=None, cache_version=scheme.cache_version,
            readout_cache_version="zeeman-integration-test")
        exported = json.loads(json.dumps(payload, allow_nan=False))["view"]["experimental_gain_calibration"]
        assert exported["model_id"] == zeeman.MODEL_ID
        assert exported["zeeman_pump_populations"]["trace"] == 1.
        assert exported["zeeman_response_provenance"]["external_sublevel_population_factor"] == 1.
        assert not exported["profile"]["noise_covariance_calibrated"]
        assert "B=0" in exported["profile"]["physical_assumptions"]
        assert exported["source_delta_mhz"]["probe"] == exported["source_delta_mhz"]["conjugate"]
        for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
            np.testing.assert_array_equal(raw[key], before[key])
        for key in ("chi", "drift_per_m", "segment_profile"):
            np.testing.assert_array_equal(raw["propagation_snapshot"][key], before["propagation_snapshot"][key])
        assert raw["claim_gate"] == before["claim_gate"]
    finally:
        plt.close(view["figure"])


def test_actual_zeeman_full_scan_routes_both_physical_branches(case, atomic_boundary):
    import matplotlib.pyplot as plt

    scheme, params, raw = case
    plus = deepcopy(raw)
    plus["branch"] = 1
    plus["probe_axis_GHz"] += raw["raman_center_plus_GHz"]-raw["raman_center_minus_GHz"]
    full = dict(D_GHz=params["opd"], minus=raw, plus=plus)
    figure = scheme.extra_views()[0].render(full, params)
    try:
        assert figure._gabes_gain_calibration_model == calibration.MODEL_ZEEMAN
        assert figure.axes[0].get_ylabel() == "Zeeman pump + chi propagation gain"
        assert "experimental Zeeman pump + chi propagation" in figure.axes[0].get_title()
        assert "Gain validation does not validate noise" in figure.axes[1].get_title()
        for i, branch in enumerate(("minus", "plus")):
            expected = zeeman.calibrate_spectrum(full[branch], enabled=True)
            stored = figure._gabes_experimental_gain_calibration[branch]
            assert stored["model_id"] == zeeman.MODEL_ID
            assert stored["diagonal_susceptibility_scale"] == 1.
            np.testing.assert_array_equal(figure.axes[0].lines[i].get_ydata(), expected["G_s"])
            noise = fwm.experimental_jasperse_readout(full[branch], params, gains=expected)
            np.testing.assert_array_equal(figure.axes[1].lines[i].get_ydata(), noise["NF_dB"])
        assert {branch for branch, _, _ in atomic_boundary} == {-1, 1}
        assert not figure._gabes_experimental_gain_calibration["plus"]["within_training_domain"]
    finally:
        plt.close(figure)


def test_zeeman_changed_conditions_retain_requested_correction(case, atomic_boundary):
    scheme, params, raw = case
    view = scheme.headless_observables(raw, dict(params, tpd=100.))
    result = view["experimental_gain_calibration"]
    assert result["applied"] and result["selected_point"]["applied"]
    assert "condition extrapolation" in view["metrics"][1]["delta"]
    assert result["zeeman_response_provenance"]
    assert np.isfinite([result["selected_point"]["G_s"], result["selected_point"]["G_c"]]).all()


def test_generic_ui_rail_selects_actual_zeeman_helper(case, atomic_boundary):
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
    app.selectbox(key="fwm__gain_calibration_model").select(calibration.MODEL_ZEEMAN).run()
    assert not app.exception
    result = app.session_state["view"]["experimental_gain_calibration"]
    assert result["model_id"] == zeeman.MODEL_ID
    assert result["zeeman_pump_populations"]["ground_F2"] == .6
    assert app.session_state["view"]["metrics"][1]["label"] == "Zeeman pump + χ probe gain"


def test_generated_profile_real_24state_response_through_rail_and_strict_export():
    from streamlit.testing.v1 import AppTest

    scheme = fwm.FWMScheme()
    params = dict(scheme.defaults(), opd=-1.04858, tpd=-16., temp_c=89.,
        pump_mw=1000., probe_uw=8., cell_mm=12., pump_waist_um=950.,
        probe_waist_um=300., seeded_angle_deg=.45, resolution=fwm.FIDELITY_FAST,
        experimental_gain_calibration=True, gain_calibration_model=calibration.MODEL_LAB)
    center = fwm.branch_center_GHz(params["opd"], -1)
    raw = fwm.compute_spectrum(params["opd"], T=362.15, P_pump=1., P_probe=8e-6,
        w_pump=950e-6, w_probe=300e-6, L=.012, line_strength=params["line_strength"],
        coarse_points=41, fine_points=0, scan_min=center-.04, scan_max=center+.04,
        phase_detail=fwm.PHASE_ULTRA, model_fidelity=fwm.FIDELITY_FAST,
        response_method=fwm.RESPONSE_POLE, velocity_step=1., velocity_cutoff=4.,
        pump_probe_angle_deg=.45, branch=-1)
    assert len(raw["probe_axis_GHz"]) == 41
    before = deepcopy(raw)
    zeeman._record.cache_clear()
    zeeman._atomic_response.cache_clear()
    record = zeeman.profile()
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
        "st.session_state['params']=params\n"
        "st.session_state['view']=scheme.headless_observables(st.session_state['raw'],params)\n",
        default_timeout=90)
    app.session_state["raw"] = raw
    app.session_state["initial_params"] = params
    app.run()
    assert not app.exception
    app.selectbox(key="fwm__gain_calibration_model").select(calibration.MODEL_ZEEMAN).run()
    assert not app.exception
    assert zeeman._atomic_response.cache_info().misses > 0
    view = app.session_state["view"]
    result = view["experimental_gain_calibration"]
    assert result["model_id"] == zeeman.MODEL_ID
    assert result["profile"]["coefficients"] == record["coefficients"]
    assert result["zeeman_response_provenance"]["external_sublevel_population_factor"] == 1.
    populations = result["zeeman_pump_populations"]
    assert len(populations["sublevels"]) == 24
    assert populations["trace"] == pytest.approx(1., abs=1e-8)
    assert result["corrected_atomic_response_diagnostics"]["finite"]
    assert view["metrics"][1]["label"] == "Zeeman pump + χ probe gain"
    payload = build_export_payload(scheme=scheme, raw=raw, params=app.session_state["params"],
        view=view, plot_payloads=[], comparison_payload=None, cache_version=scheme.cache_version,
        readout_cache_version="production-zeeman-41point-test")
    exported = json.loads(json.dumps(payload, allow_nan=False))["view"]["experimental_gain_calibration"]
    assert exported["zeeman_pump_populations"]["sublevels"] == populations["sublevels"]
    assert exported["profile"]["source_sha256"] == record["source_sha256"]
    assert not exported["profile"]["noise_covariance_calibrated"]
    for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
        np.testing.assert_array_equal(raw[key], before[key])
    assert raw["claim_gate"] == before["claim_gate"]
