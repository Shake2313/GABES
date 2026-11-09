"""Invariants of the gain-only literature readout; no atomic solves required."""
from copy import deepcopy

import numpy as np
import pytest

from gabes import fwm_literature_gain as literature


@pytest.fixture
def fixed_profile(monkeypatch):
    record = {
        "coefficients": [float(np.log(1.3)), 0.0, 0.0],
        "gain_quantity": "effective optical output/input gain",
        "predictive_gate_passed": False,
        "validation_status": "GROUPED_PREDICTION_NOT_ESTABLISHED",
        "raw_training_bounds": {"D_GHz": [.8, 1.0], "temperature_K": [380., 390.]},
        "training_delta_bounds_mhz": [-10., 10.],
    }
    monkeypatch.setattr(literature, "profile", lambda: deepcopy(record))
    return record


@pytest.fixture
def raw():
    return {
        "probe_axis_GHz": np.array([-.010, -.003, .002, .010]),
        "raman_center_minus_GHz": 0.0,
        "G_s": np.array([.5, 2., 7., 1.2]),
        "G_c": np.array([.1, 1., 3., .05]),
        "D_GHz": .9, "temperature_K": 386.15,
        "input_pump_power_W": .3, "input_seed_power_W": 10e-6,
        "w_pump_m": 530e-6, "branch": -1,
    }


def test_same_frequency_map_retains_arm_ratio(raw, fixed_profile):
    delta = np.array([-10., -3., 2., 10.])
    result = literature.evaluate(raw, delta)
    np.testing.assert_array_equal(result["source_delta_mhz"]["probe"], delta)
    np.testing.assert_array_equal(result["source_delta_mhz"]["conjugate"], delta)
    np.testing.assert_allclose(result["G_s"] / (1 + result["G_c"]),
                               raw["G_s"] / (1 + raw["G_c"]), rtol=3e-15)
    assert np.all(result["G_c"] > raw["G_c"])
    assert result["effective_generation_scale"] == pytest.approx(1.3)
    assert not np.any(result["pump_budget_capped"])


def test_off_is_exact_and_both_modes_leave_input_untouched(raw, fixed_profile):
    raw["G_c"][0] = 0.0
    before = deepcopy(raw)
    off = literature.calibrate_spectrum(raw, enabled=False, selected_delta_mhz=2.)
    assert not off["applied"]
    assert off["selected_point"]["status"] == "DISABLED"
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(off[key], before[key])
        off[key][:] = -1.0
    on = literature.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=2.)
    assert on["selected_point"]["applied"]
    for key in raw:
        np.testing.assert_array_equal(raw[key], before[key])


def test_identity_returns_exact_original_even_if_optional_budget_is_exceeded():
    gs = np.array([0., .5, 1., 2., 1e250])
    gc = np.array([0., .1, 0., 1., 1e250])
    result_s, result_c, capped = literature.coupled_gains(
        gs, gc, 1., pump_power_W=1., seed_power_W=1.)
    np.testing.assert_array_equal(result_s, gs)
    np.testing.assert_array_equal(result_c, gc)
    assert not np.any(capped)


def test_zero_conjugate_remains_zero_for_any_scale():
    gs = np.array([0., .4, 1., 2.])
    out_s, out_c, _ = literature.coupled_gains(gs, np.zeros(4), [1e-4, .5, 2., 1e4])
    np.testing.assert_array_equal(out_s, gs)
    np.testing.assert_array_equal(out_c, np.zeros(4))


def test_zero_pump_preserves_passive_input(raw, fixed_profile):
    passive = dict(raw, input_pump_power_W=0., G_s=np.array([.4, .7, 1., .9]), G_c=np.zeros(4))
    result = literature.calibrate_spectrum(passive, enabled=True)
    np.testing.assert_array_equal(result["G_s"], passive["G_s"])
    np.testing.assert_array_equal(result["G_c"], passive["G_c"])
    assert result["effective_generation_scale"] == 1.
    assert np.all(result["pump_budget_feasible"])


def test_generated_power_budget_holds_above_and_below_probe_unity():
    ratio = np.array([.1, .5, 1., 1.5, 2.])
    original_c = np.full(5, .1)
    original_s = ratio * (1 + original_c)
    budget = 3.
    assert np.all(np.maximum(original_s - 1, 0) + original_c <= budget)
    out_s, out_c, capped = literature.coupled_gains(
        original_s, original_c, 100., pump_power_W=budget, seed_power_W=1.)
    assert np.all(capped)
    generated = np.maximum(out_s - 1, 0) + out_c
    np.testing.assert_allclose(generated, budget, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(out_s / (1 + out_c), ratio, rtol=2e-15)


def test_infeasible_raw_ratio_is_reported(raw, fixed_profile):
    impossible = dict(raw, G_s=np.full(4, 10.), G_c=np.zeros(4),
                      input_pump_power_W=1e-6, input_seed_power_W=1e-6)
    result = literature.calibrate_spectrum(impossible, enabled=True, selected_delta_mhz=2.)
    assert not np.any(result["pump_budget_feasible"])
    assert result["selected_point"]["pump_budget_feasible"] is False


@pytest.mark.parametrize("scale", [1e-300, .05, 20., 1e300])
def test_finite_extreme_gains_remain_finite_nonnegative(scale):
    gs = np.array([0., 1e-300, 1e300, 1e308, 1.])
    gc = np.array([1e300, 1e-300, 1e300, 0., 1e300])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        out_s, out_c, _ = literature.coupled_gains(gs, gc, scale)
    assert np.all(np.isfinite(out_s)) and np.all(np.isfinite(out_c))
    assert np.all(out_s >= 0) and np.all(out_c >= 0)


@pytest.mark.parametrize("changed", [{"D_GHz": -2.}, {"D_GHz": None}, {"branch": 1}])
def test_unknown_or_changed_domain_retains_requested_correction(raw, fixed_profile, changed):
    result = literature.evaluate(dict(raw, **changed), np.array([2., 30.]))
    assert result["applied"] and result["context_changed"]
    assert not result["within_training_domain"]
    assert not np.any(result["within_measured_support"])
    assert np.all(["CONDITION_EXTRAPOLATION" in str(s) for s in result["point_status"]])
    assert result["effective_generation_scale"] == pytest.approx(1.3)
    assert np.all(np.isfinite(result["G_s"])) and np.all(result["G_c"] > 0)


def test_profile_metadata_is_independent_and_does_not_claim_covariance():
    first = literature.profile()
    first["coefficients"][0] = 999.
    assert literature.profile()["coefficients"][0] != 999.
    assert first["noise_covariance_calibrated"] is False
    assert first["susceptibility_identified"] is False


@pytest.mark.parametrize("args", [(-1., 1., 1.), (1., np.inf, 1.), (1., 1., 0.), (1., 1., np.nan)])
def test_invalid_gain_or_scale_is_rejected(args):
    with pytest.raises(ValueError):
        literature.coupled_gains(*args)


def test_invalid_detuning_or_context_is_rejected(raw, fixed_profile):
    with pytest.raises(ValueError, match="finite"):
        literature.evaluate(raw, [np.nan])
    with pytest.raises(ValueError, match="metadata"):
        literature.evaluate({k: v for k, v in raw.items() if k != "w_pump_m"}, [2.])
