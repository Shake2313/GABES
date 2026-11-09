"""χ correction and independent checks of the optical repropagation readout."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm

from gabes import constants, observables
from gabes import fwm_chi_gain as correction
from gabes.schemes import fwm


def _atomic_drifts():
    return np.array([
        [[-4.+30j, 160j], [-160j, -6.-20j]],
        [[-7.-15j, 90.+20j], [90.-20j, -2.+25j]],
        [[-18.+80j, 45.-10j], [45.+10j, -12.-50j]],
    ])


def _add_vacuum(atomic, mismatch):
    result = np.array(atomic, complex, copy=True)
    result[:, 0, 0] -= .5j*np.asarray(mismatch)
    result[:, 1, 1] += .5j*np.asarray(mismatch)
    return result


def _chi_from_atomic_drift(atomic, density, k_probe, k_conjugate, line_strength):
    factor = observables.chi_phys(1., density, dipole=constants.DIPOLE_D1,
                                  line_strength=line_strength)
    ss = atomic[:, 0, 0] / (.5j*k_probe*factor)
    sc = atomic[:, 0, 1] / (.5j*k_probe*factor)
    cs = np.conj(atomic[:, 1, 0] / (-.5j*k_conjugate*factor))
    cc = np.conj(atomic[:, 1, 1] / (-.5j*k_conjugate*factor))
    return ss, sc, cs, cc


@pytest.fixture
def fixed_profile(monkeypatch):
    record = {
        "coefficients": [float(np.log(1.3)), 0., 0., float(np.log(.8))],
        "gain_quantity": "effective optical output/input gain",
        "predictive_gate_passed": False,
        "validation_status": "NOT_EVALUATED",
        "raw_training_bounds": {"D_GHz": [.8, 1.], "temperature_K": [380., 390.]},
        "training_delta_bounds_mhz": [-10., 10.],
    }
    monkeypatch.setattr(correction, "profile", lambda: deepcopy(record))
    return record


@pytest.fixture
def raw():
    mismatch = np.array([70., 70., 70.])
    matrix = _add_vacuum(_atomic_drifts(), mismatch)
    profile = np.array([.6, .9, 1., .9, .6])
    gs, gc, _ = correction.propagate(matrix, .0125, profile, .3, 10e-6, 1.4)
    return {
        "probe_axis_GHz": np.array([-.010, .0, .010]),
        "raman_center_minus_GHz": 0., "delta_k_z": mismatch,
        "G_s": gs, "G_c": gc, "D_GHz": .9, "temperature_K": 386.15,
        "cell_length_m": .0125, "input_pump_power_W": .3,
        "input_seed_power_W": 10e-6, "w_pump_m": 530e-6, "branch": -1,
        "propagation_snapshot": {"drift_per_m": matrix, "segment_profile": profile,
            "conjugate_power_ratio": 1.4, "dynamic_depletion": True},
    }


def test_full_complex_diagonals_and_paired_couplings_preserve_vacuum_terms():
    atomic, dk = _atomic_drifts(), np.array([50., 100., -70.])
    matrix = _add_vacuum(atomic, dk)
    before = matrix.copy()
    actual = correction.correct_drift(matrix, dk, 1.7, .4)
    changed = atomic.copy()
    changed[:, 0, 0] *= .4
    changed[:, 1, 1] *= .4
    changed[:, 0, 1] *= 1.7
    changed[:, 1, 0] *= 1.7
    np.testing.assert_allclose(actual, _add_vacuum(changed, dk), rtol=2e-15, atol=2e-14)
    np.testing.assert_array_equal(matrix, before)
    vacuum_only = _add_vacuum(np.zeros_like(atomic), dk)
    np.testing.assert_array_equal(correction.correct_drift(vacuum_only, dk, 20., .05), vacuum_only)


@pytest.mark.parametrize("area", [.7, 1., 1.6])
def test_serial_repropagation_matches_existing_dynamic_depletion_kernel(area):
    atomic = _atomic_drifts()
    dk = np.array([70., 90., 110.])
    density, kp, kc, strength = 1e19, 7.9e6, 7.91e6, 1./12.
    ss, sc, cs, cc = _chi_from_atomic_drift(atomic, density, kp, kc, strength)
    segment = np.array([.7, .9, 1., 1., .9, .7])
    spatial = np.array([.6, .8, 1., 1., .8, .6])
    pump, seed, length = .002, 10e-6, .0125
    old_s, old_c, _, _ = fwm._ultra_segmented_gain(
        ss, sc, cs, cc, kp, kc, length, density, strength, dk,
        segment, spatial, pump, seed, conjugate_power_ratio=area)
    expected = observables.pump_depletion_saturation(old_s, old_c*area, pump, seed)
    actual_s, actual_c, clipped = correction.propagate(
        _add_vacuum(atomic, dk), length, segment*spatial, pump, seed, area)
    np.testing.assert_allclose(actual_s, expected[0], rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(actual_c, expected[1], rtol=3e-13, atol=1e-14)
    assert not np.any(clipped)


def test_linear_repropagation_matches_independent_exponential_and_existing_kernel():
    matrix = _add_vacuum(_atomic_drifts(), [70., 90., 110.])
    length, pump, seed = .0125, .3, 10e-6
    transfer = np.stack([expm(m*length) for m in matrix])
    native = observables.matrix_exp_2x2(matrix, length)
    np.testing.assert_allclose(native, transfer, rtol=3e-14, atol=1e-14)
    expected = observables.pump_depletion_saturation(
        abs(transfer[:, 0, 0])**2, abs(transfer[:, 1, 0])**2, pump, seed)
    actual_s, actual_c, _ = correction.propagate(matrix, length, np.ones(7), pump, seed,
                                               dynamic_depletion=False)
    np.testing.assert_allclose(actual_s, expected[0], rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(actual_c, expected[1], rtol=3e-13, atol=1e-14)


def test_fresh_fast_atomic_snapshot_reproduces_all_raw_gain_nodes():
    scheme = fwm.FWMScheme()
    raw = scheme.compute(dict(scheme.defaults(), resolution=fwm.FIDELITY_FAST,
                              experimental_gain_calibration=False))
    snap = raw["propagation_snapshot"]
    gs, gc, clipped = correction.propagate(snap["drift_per_m"], raw["cell_length_m"],
        snap["segment_profile"], raw["input_pump_power_W"], raw["input_seed_power_W"],
        snap["conjugate_power_ratio"], snap["dynamic_depletion"])
    assert raw["floquet_convergence"]["status"] == "CONVERGED"
    np.testing.assert_allclose(gs, raw["G_s"], rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(gc, raw["G_c"], rtol=1e-11, atol=1e-14)
    assert not np.any(clipped)
    transfer, _ = correction.undepleted_transfer(snap["drift_per_m"], raw["cell_length_m"], snap["segment_profile"])
    np.testing.assert_allclose(transfer, raw["T_field_small_signal"], rtol=3e-12, atol=2e-13)


def test_identity_profile_reproduces_native_node_propagation(raw, fixed_profile):
    fixed_profile["coefficients"] = [0., 0., 0., 0.]
    actual = correction.calibrate_spectrum(raw, enabled=True, selected_delta_mhz=0.)
    np.testing.assert_allclose(actual["G_s"], raw["G_s"], rtol=3e-14)
    np.testing.assert_allclose(actual["G_c"], raw["G_c"], rtol=3e-14)
    assert actual["selected_point"]["G_s"] == pytest.approx(raw["G_s"][1])
    assert actual["offdiagonal_scale"] == actual["diagonal_susceptibility_scale"] == 1.


def test_off_bit_exact_and_enabled_does_not_mutate_raw_or_snapshot(raw, fixed_profile):
    before = deepcopy(raw)
    off = correction.calibrate_spectrum(raw, enabled=False)
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(off[key], before[key])
        off[key][:] = -1.
    on = correction.calibrate_spectrum(raw, enabled=True)
    assert on["applied"]
    for key in ("G_s", "G_c", "delta_k_z", "probe_axis_GHz"):
        np.testing.assert_array_equal(raw[key], before[key])
    for key in ("drift_per_m", "segment_profile"):
        np.testing.assert_array_equal(raw["propagation_snapshot"][key], before["propagation_snapshot"][key])
    on["corrected_drift_per_m"][:] = 0.
    np.testing.assert_array_equal(raw["propagation_snapshot"]["drift_per_m"], before["propagation_snapshot"]["drift_per_m"])


def test_zero_length_and_zero_cross_limits():
    matrix = _add_vacuum(_atomic_drifts(), [70., 90., 110.])
    gs, gc, _ = correction.propagate(matrix, 0., np.ones(7), .3, 10e-6)
    np.testing.assert_array_equal(gs, np.ones(3))
    np.testing.assert_array_equal(gc, np.zeros(3))


def test_separately_saved_field_transfer_matches_independent_exponential():
    from scipy.linalg import expm
    matrix = _atomic_drifts()
    length = .0125
    transfer, clipped = correction.undepleted_transfer(matrix, length, np.ones(8))
    expected = np.asarray([expm(m*length) for m in matrix])
    np.testing.assert_allclose(transfer, expected, rtol=2e-13, atol=2e-14)
    assert not np.any(clipped)
    matrix[:, 0, 1] = matrix[:, 1, 0] = 0.
    gs, gc, _ = correction.propagate(matrix, .0125, np.ones(7), .3, 10e-6)
    np.testing.assert_allclose(gs, np.exp(2*matrix[:, 0, 0].real*.0125), rtol=2e-14)
    np.testing.assert_array_equal(gc, np.zeros(3))


def test_zero_pump_forces_identity_scales_and_passive_gains(raw, fixed_profile):
    matrix = raw["propagation_snapshot"]["drift_per_m"].copy()
    matrix[:, 0, 1] = matrix[:, 1, 0] = 0.
    raw["propagation_snapshot"]["drift_per_m"] = matrix
    raw["input_pump_power_W"] = 0.
    result = correction.calibrate_spectrum(raw, enabled=True)
    assert result["offdiagonal_scale"] == result["diagonal_susceptibility_scale"] == 1.
    np.testing.assert_allclose(result["G_s"], np.exp(2*matrix[:, 0, 0].real*.0125), rtol=2e-14)
    np.testing.assert_array_equal(result["G_c"], np.zeros(3))


def test_extreme_scales_are_clipped_and_repropagation_obeys_generated_energy_budget(raw, fixed_profile):
    fixed_profile["coefficients"] = [1e8, 0., 0., 1e8]
    result = correction.calibrate_spectrum(raw, enabled=True)
    assert result["offdiagonal_scale"] == pytest.approx(20.)
    assert result["diagonal_susceptibility_scale"] == pytest.approx(20.)
    budget = raw["input_pump_power_W"] / raw["input_seed_power_W"]
    assert np.all(np.maximum(result["G_s"]-1., 0.)+result["G_c"] <= budget*(1+1e-14))
    high = np.array([[[1e6, 5e4], [5e4, 1e6]]], complex)
    gs, gc, clipped = correction.propagate(high, 1., np.ones(8), 1., 1e-6)
    assert np.all(np.isfinite(gs)) and np.all(np.isfinite(gc))
    assert np.all(clipped)
    assert np.all(np.maximum(gs-1., 0.)+gc <= 1e6*(1+1e-14))


def test_same_delta_outside_training_domain_stays_applied_and_geometry_is_retained(raw, fixed_profile):
    delta = np.array([0., 30., 1e6])
    result = correction.evaluate(dict(raw, D_GHz=-1.), delta)
    assert result["applied"] and not result["within_training_domain"]
    np.testing.assert_array_equal(result["source_delta_mhz"]["probe"], delta)
    np.testing.assert_array_equal(result["source_delta_mhz"]["conjugate"], delta)
    assert result["offdiagonal_scale"] == pytest.approx(1.3)
    vacuum = _add_vacuum(np.zeros((1, 2, 2), complex), [70.])
    np.testing.assert_allclose(result["corrected_drift_per_m"][-1:], vacuum, atol=1e-14)
    assert result["G_s"][-1] == pytest.approx(1., abs=1e-12)
    assert result["G_c"][-1] == 0.


def test_legacy_fixture_and_noise_metadata_do_not_select_or_fit_chi_correction(raw, fixed_profile):
    actual = correction.evaluate(raw, [0.])
    modified = dict(raw, legacy_gain_calibration_fixture=True,
                    gain_closure_enabled=True, legacy_mean_field_fixture=True,
                    seed_classical_noise_a=1e9, detector_quantum_efficiency=.01,
                    gain_closure={"applied": True, "coupling_multiplier": 999.})
    other = correction.evaluate(modified, [0.])
    np.testing.assert_array_equal(actual["G_s"], other["G_s"])
    np.testing.assert_array_equal(actual["G_c"], other["G_c"])
    assert actual["offdiagonal_scale"] == other["offdiagonal_scale"]
    assert not actual["profile"]["predictive_gate_passed"]


def test_invalid_context_or_absent_snapshot_is_reported(raw, fixed_profile):
    with pytest.raises(ValueError, match="snapshot"):
        correction.evaluate({k: v for k, v in raw.items() if k != "propagation_snapshot"}, [0.])
    with pytest.raises(ValueError, match="finite"):
        correction.evaluate(raw, [np.nan])
    with pytest.raises(ValueError, match="positive"):
        correction.correct_drift(raw["propagation_snapshot"]["drift_per_m"], raw["delta_k_z"], 0., 1.)
    with pytest.raises(ValueError, match="representable"):
        correction.propagate(np.full((1, 2, 2), 1e200+0j), 1., np.ones(8), .3, 8e-6)
