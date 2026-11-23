"""SABES uses current seed-noise NF and preserves unavailable readouts."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gabes.fwm_experiment_noise import experimental_jasperse_readout
from gabes.schemes import fwm
from sabes import bridge
from sabes.apparatus import default_apparatus
from sabes.calibration import default_calibration
from sabes.detection import DetectionSettings
from sabes.presets import PAPER_PROFILE
from sabes.session_export import session_payload
from gabes_ui.sabes import page as sabes_page


def _result():
    result = bridge.run(solve=False)
    center = fwm.branch_center_GHz(result.params["opd"], -1)
    amplifier = np.array([[2.0, np.sqrt(3.0)], [np.sqrt(3.0), 2.0]], complex)
    raw = {
        "D_GHz": result.params["opd"],
        "raman_center_minus_GHz": center,
        "probe_axis_GHz": center + np.array([-8.0, 0.0, 8.0])*1e-3,
        "T_canonical_small_signal": np.tile(amplifier, (3, 1, 1)),
        "G_s": np.full(3, 4.0), "G_c": np.full(3, 3.0),
        "input_seed_power_W": 8e-6,
        "eta": 1.0,
        "gain_referred_noise_dB": np.full(3, -20.0),
        "N_atoms": 1e18, "sigma_v": 180.0, "n_velocity": 3,
        "Op_A_2pi_GHz": 0.1, "Os_2pi_MHz": 1.0,
    }
    return replace(result, raw=raw)


def test_geometry_and_legacy_overrides_map_to_the_current_external_efficiency():
    result = bridge.run(solve=False)
    assert result.params["after_cell_eff_pct"] == pytest.approx(100-result.geometry.loss_pct)
    assert result.eta == pytest.approx(
        result.params["after_cell_eff_pct"]*result.params["qe_pct"]/1e4)
    old = bridge.run(solve=False, loss_pct=25.0, qe_pct=80.0)
    assert old.params["after_cell_eff_pct"] == 75.0
    assert old.eta == pytest.approx(0.6)
    explicit = bridge.run(solve=False, loss_pct=25.0, after_cell_eff_pct=90.0)
    assert explicit.params["after_cell_eff_pct"] == 90.0
    assert explicit.params["loss_pct"] == 10.0
    assert bridge.solve_key(old.params) == bridge.solve_key(result.params)
    stale = bridge.run(solve=False, noise_model="Passive L1", velocity_measure="truncated")
    assert stale.params == result.params
    assert stale.noise_model == fwm.NOISE_EXPERIMENTAL_JASPERSE


def test_page_noise_readout_is_cheap_current_and_matches_the_jasperse_api(monkeypatch):
    result = _result()
    params = dict(result.params, tpd=0.0, seed_classical_rin_scaled=2.0,
                  after_cell_eff_pct=75.0, qe_pct=80.0)
    # This page path must not invoke figures, headless tables, or the optional
    # pump-dressed absorption/QRT extraction.
    monkeypatch.setattr(fwm.FWMScheme, "headless_observables", lambda *a, **kw:
                        pytest.fail("full headless observables are not needed"))
    view = bridge.noise_observables(result.raw, params)
    current = replace(result, params=params, observables=view)
    expected = experimental_jasperse_readout(result.raw, params)
    assert current.noise_factor == pytest.approx(expected["NF"][1])
    assert current.gain_referred_noise_db == pytest.approx(expected["NF_dB"][1])
    assert current.squeezing_db == current.gain_referred_noise_db
    assert current.gain_referred_noise_db != -20.0
    assert current.physical_squeezing_db is None


def test_invalid_current_nf_never_falls_back_to_finite_legacy_raw_noise():
    result = _result()
    unavailable = {"jasperse_readout": {}, "passive_l1_operating_point": {
        "NF": float("nan"), "NF_dB": float("nan")}}
    current = replace(result, observables=unavailable, readout=SimpleNamespace())
    assert current.gain_referred_noise_db is None
    assert current.noise_factor is None
    assert current.eom_shot_noise_only_db is None
    assert current.eom_rin_loaded_noise_db is None
    assert current.eom_rin_penalty_db is None
    assert sabes_page._noise_db_text(current.gain_referred_noise_db) == "unavailable"
    assert sabes_page._noise_db_text(float("nan")) == "unavailable"
    assert sabes_page._noise_db_text(-3.0) == "-3.00 dB"
    # Old fixtures with no current-view descriptor retain their explicit alias.
    assert result.gain_referred_noise_db == -20.0


def test_outside_point_keeps_unavailable_status():
    result = _result()
    outside_params = dict(result.params, tpd=30.0)
    outside = replace(result, params=outside_params,
        observables=bridge.noise_observables(result.raw, outside_params))
    assert outside.gain_referred_noise_db is None
    assert outside.readout_evidence["noise"]["status"] == "UNAVAILABLE_AT_SELECTED_DETUNING"


@pytest.mark.parametrize("calibrated", [False, True])
def test_selected_gain_noise_and_evidence_match_fwm_headless(calibrated, monkeypatch):
    result = _result()
    params = dict(result.params, tpd=-1.0,
                  experimental_gain_calibration=calibrated)
    expected = fwm.FWMScheme().headless_observables(result.raw, params)
    monkeypatch.setattr(fwm.FWMScheme, "compute", lambda *_:
                        pytest.fail("changing readout must not solve again"))
    monkeypatch.setattr(fwm, "_seeded_absorption_readout", lambda *_:
                        pytest.fail("the bridge does not need the absorption audit"))
    view = bridge.noise_observables(result.raw, params)
    current = replace(result, params=params, observables=view)
    assert current.gains == tuple(expected["gain_operating_point"][k]
                                  for k in ("G_s", "G_c"))
    assert current.noise_model == fwm.NOISE_EXPERIMENTAL_JASPERSE
    assert current.gain_referred_noise_db == expected["noise_operating_point"]["NF_dB"]
    assert current.noise_factor == expected["noise_operating_point"]["NF"]
    assert current.raw_gains == (4.0, 3.0)
    assert (current.gains != current.raw_gains) is calibrated
    assert bridge.solve_key(params) == bridge.solve_key(result.params)
    payload = session_payload(current, DetectionSettings(), default_calibration(),
                              default_apparatus(), PAPER_PROFILE)
    selected = payload["selected_readout"]
    assert selected["noise_model"] == fwm.NOISE_EXPERIMENTAL_JASPERSE
    assert selected["raw_atomic_gains"] == [4.0, 3.0]
    assert selected["physical_squeezing_dB"] is None
    assert selected["evidence"]["provenance"] == "computed"
    assert selected["evidence"]["gain"]["calibration_applied"] is calibrated
    assert selected["evidence"]["noise"]["operating_point"]["NF"] == current.noise_factor


def test_jasperse_no_detection_is_unavailable_without_legacy_fallback():
    result = _result()
    params = dict(result.params, noise_model=fwm.NOISE_EXPERIMENTAL_JASPERSE,
                  tpd=0.0, after_cell_eff_pct=0.0)
    current = replace(result, params=params,
                      observables=bridge.noise_observables(result.raw, params))
    assert current.gain_referred_noise_db is None
    assert current.noise_factor is None
    assert current.readout_evidence["noise"]["status"] == "UNAVAILABLE_AT_SELECTED_DETUNING"
