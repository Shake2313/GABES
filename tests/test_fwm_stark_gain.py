"""Atomic recomputation, raw provenance and practical-model continuation."""
from copy import deepcopy

import numpy as np
import pytest

from gabes import fwm_chi_gain as chi
from gabes import fwm_stark_gain as model


@pytest.fixture
def record(monkeypatch):
    value = dict(pump_stark_fraction=.1, coefficients=[np.log(1.4), 0., 0., np.log(.8)],
        gain_quantity="effective optical output/input gain", predictive_gate_passed=False,
        validation_status="GROUPED_PREDICTION_NOT_ESTABLISHED",
        raw_training_bounds={"D_GHz": [.8, 1.], "temperature_K": [380., 390.]},
        training_delta_bounds_mhz=[-10., 10.])
    monkeypatch.setattr(model, "profile", lambda: deepcopy(value))
    clear = model._solve_response.cache_clear
    clear()
    yield value
    clear()


def spectrum(cross):
    matrix = np.array([[[-4.+35j, cross*1j], [-cross*1j, -6.-35j]]]*3)
    profile = np.array([.7, 1., .7])
    gs, gc, _ = chi.propagate(matrix, .012, profile, .3, 8e-6)
    return dict(probe_axis_GHz=np.array([-.03, 0., .03]), raman_center_minus_GHz=0.,
        D_GHz=.9, branch=-1, temperature_K=386.15, input_pump_power_W=.3,
        input_seed_power_W=8e-6, w_pump_m=530e-6, cell_length_m=.012,
        delta_k_z=np.zeros(3), G_s=gs, G_c=gc,
        propagation_snapshot=dict(drift_per_m=matrix, segment_profile=profile,
            conjugate_power_ratio=1., dynamic_depletion=True),
        T_canonical_small_signal=np.broadcast_to(np.eye(2), (3, 2, 2)).copy(),
        floquet_convergence=dict(status="CONVERGED"),
        stark_recompute_context=dict(D_GHz=.9, P_pump=.3, P_probe=8e-6,
            w_pump=530e-6, w_probe=330e-6, scan_min=-.03, scan_max=.03,
            phase_detail="Ultra", branch=-1))


def test_recomputation_uses_all_original_inputs_and_is_cached_without_mutating_raw(record, monkeypatch):
    from gabes.schemes import fwm
    raw, changed = spectrum(60.), spectrum(120.)
    before = deepcopy(raw)
    calls = []

    def solve(**inputs):
        calls.append(inputs)
        return changed

    monkeypatch.setattr(fwm, "compute_spectrum", solve)
    result = model.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=-7.)
    again = model.evaluate(raw, [-7., 0., 50.])
    assert calls == [dict(raw["stark_recompute_context"], pump_stark_fraction=.1)]
    corrected = chi.correct_drift(changed["propagation_snapshot"]["drift_per_m"], 0., 1.4, .8)
    expected = chi.propagate(corrected, .012, [.7, 1., .7], .3, 8e-6)
    np.testing.assert_allclose(result["G_s"], expected[0])
    np.testing.assert_allclose(result["G_c"], expected[1])
    assert result["selected_point"]["G_s"] == pytest.approx(float(again["G_s"][0]))
    for key in ("G_s", "G_c", "T_canonical_small_signal"):
        np.testing.assert_array_equal(raw[key], before[key])
    assert raw["stark_recompute_context"] == before["stark_recompute_context"]
    assert not result["profile"]["predictive_gate_passed"]
    assert not again["within_training_domain"]
    assert again["applied"] and np.isfinite(again["G_s"]).all()
    assert "EXTRAPOLATION" in again["point_status"][-1]


def test_disabled_stale_spectrum_needs_no_recompute_context_or_snapshot(record, monkeypatch):
    raw = spectrum(60.)
    raw.pop("stark_recompute_context")
    raw.pop("propagation_snapshot")
    monkeypatch.setattr(model, "_solve_response", lambda *args: pytest.fail("off must not solve"))
    result = model.calibrate_spectrum(raw, enabled=False, selected_delta_mhz=0.)
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(result[key], raw[key])
        assert not np.shares_memory(result[key], raw[key])
    with pytest.raises(ValueError, match="fresh atomic spectrum"):
        model.evaluate(raw, 0., enabled=True)


def test_pump_zero_retains_passive_response_and_fraction_one_needs_no_context(record, monkeypatch):
    raw = spectrum(0.)
    raw["input_pump_power_W"] = 0.
    raw.pop("stark_recompute_context")
    monkeypatch.setattr(model, "_solve_response", lambda *args: pytest.fail("pump zero must not solve"))
    result = model.evaluate(raw, [0.], enabled=True)
    assert result["offdiagonal_scale"] == result["diagonal_susceptibility_scale"] == 1.
    np.testing.assert_allclose(result["G_s"], raw["G_s"][:1])
    np.testing.assert_array_equal(result["G_c"], [0.])
    assert model.corrected_response(raw, 1.) is raw
    for invalid in (-.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            model.corrected_response(raw, invalid)


def test_identity_fraction_retains_native_atomic_drift(record):
    record["pump_stark_fraction"] = 1.
    record["coefficients"] = [0., 0., 0., 0.]
    raw = spectrum(100.)
    raw.pop("stark_recompute_context")
    actual = model.calibrate_spectrum(raw, enabled=True)
    np.testing.assert_array_equal(actual["corrected_drift_per_m"], raw["propagation_snapshot"]["drift_per_m"])
    np.testing.assert_allclose(actual["G_s"], raw["G_s"], rtol=1e-14)
    np.testing.assert_allclose(actual["G_c"], raw["G_c"], rtol=1e-14)


def test_native_request_reverses_an_existing_opt_in_response_and_serializes_numpy_inputs(record, monkeypatch):
    from gabes.schemes import fwm
    raw, native = spectrum(120.), spectrum(60.)
    raw["pump_stark_response"] = dict(applied=True, fraction=.1)
    raw["stark_recompute_context"]["branches"] = np.array([-1], dtype=np.int64)
    raw["stark_recompute_context"]["P_pump"] = np.float64(.3)
    calls = []

    def solve(**inputs):
        calls.append(inputs)
        return native

    monkeypatch.setattr(fwm, "compute_spectrum", solve)
    assert model.corrected_response(raw, .1) is raw
    assert model.corrected_response(raw, 1.) is native
    assert calls[0]["pump_stark_fraction"] == 1.
    assert calls[0]["branches"] == [-1]
    assert calls[0]["P_pump"] == .3
