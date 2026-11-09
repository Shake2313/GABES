"""The resolved atomic map, population ledger and gain-only residual route."""
from copy import deepcopy

import numpy as np
import pytest

from gabes import constants, fwm_chi_gain as optical, fwm_zeeman_gain as model
from gabes import hyperfine, observables
from gabes.schemes import fwm


@pytest.fixture
def case():
    context = model.context_from_params(dict(opd=-1.04858, temp_c=89., pump_mw=1000.,
        probe_uw=8., cell_mm=12., pump_waist_um=950., probe_waist_um=300., seeded_angle_deg=.45))
    center = context["raman_center_minus_GHz"]
    context.update(probe_axis_GHz=center+np.array([-.02, -.016, -.01]),
        G_s=np.array([10., 20., 30.]), G_c=np.array([9., 19., 29.]),
        T_canonical_small_signal=np.broadcast_to(np.eye(2), (3, 2, 2)).copy())
    return context


@pytest.fixture
def record(monkeypatch):
    value = dict(coefficients=[np.log(1.5), 0.], velocity_step_m_s=2., velocity_cutoff_sigma=6.,
        validation_status="NOT_EVALUATED", gain_quantity="effective optical output/input gain",
        predictive_gate_passed=False, raw_training_bounds={"D_GHz": [-1.2, 1.2]},
        training_delta_bounds_mhz=[-16., 26.])
    monkeypatch.setattr(model, "profile", lambda: deepcopy(value))
    return value


@pytest.fixture
def atomic_boundary(monkeypatch):
    chi = np.array([[1e-12-3e-11j, 2e-10+1e-11j], [2e-10-1e-11j, -1e-12-3e-11j]])
    calls = []

    def evaluate(encoded, delta, branch, method):
        import json
        calls.append((json.loads(encoded), delta, branch, method))
        state = np.diag([.5/5]*5+[.5/7]*7+[0.]*12).astype(complex)
        return np.broadcast_to(chi, (len(delta), 2, 2)).copy(), state, dict(finite=True), dict(scope="unit fixture")

    monkeypatch.setattr(model, "_atomic_response", evaluate)
    return chi, calls


def test_offline_context_never_runs_the_native_atom(monkeypatch):
    monkeypatch.setattr(fwm, "compute_spectrum", lambda *args, **kwargs: pytest.fail("native solve unnecessary"))
    context = model.context_from_params(dict(opd=-1., temp_c=89.))
    assert context["combined_line_strength_residual"] == fwm.SEEDED_REFERENCE_RESIDUAL
    assert context["transit_reset_rate_rad_s"] == pytest.approx(2*np.pi*100e3)
    assert "G_s" not in context


def test_actual_dipole_prefactor_and_vacuum_geometry_are_counted_once(case, atomic_boundary):
    chi, calls = atomic_boundary
    before = deepcopy(case)
    response = model.build_response(case, [-16.])
    assert calls[0][2:] == (-1, "poles")
    assert calls[0][0]["pump_power_W"] == 1.
    _, kp, kc = fwm.seeded_option_a_wavenumbers(case["D_GHz"], response["probe_axis_GHz"])
    factor = observables.chi_phys(1., hyperfine.number_density(case["temperature_K"]),
        dipole=constants.DIPOLE_D1, line_strength=.74/3.)
    expected = np.array([[[.5j*kp[0]*factor*chi[0, 0]-.5j*response["delta_k_z"][0],
                           .5j*kp[0]*factor*chi[0, 1]],
                          [-.5j*kc[0]*np.conj(factor*chi[1, 0]),
                           -.5j*kc[0]*np.conj(factor*chi[1, 1])+.5j*response["delta_k_z"][0]]]])
    np.testing.assert_allclose(response["propagation_snapshot"]["drift_per_m"], expected)
    assert response["zeeman_response_provenance"]["external_sublevel_population_factor"] == 1.
    assert response["zeeman_pump_populations"]["ground_F2"] == pytest.approx(.5)
    assert response["zeeman_pump_populations"]["ground_F3"] == pytest.approx(.5)
    assert len(response["zeeman_pump_populations"]["sublevels"]) == 24
    np.testing.assert_array_equal(case["G_s"], before["G_s"])
    assert "T_canonical_small_signal" not in response


def test_calibration_repropagates_new_atomic_map_without_rescaling_native_outputs(case, record, atomic_boundary):
    result = model.calibrate_spectrum(case, enabled=True, selected_delta_mhz=-16.)
    response = model.build_response(case, [-20., -16., -10.])
    snapshot = response["propagation_snapshot"]
    changed = optical.correct_drift(snapshot["drift_per_m"], response["delta_k_z"], 1.5, 1.)
    expected = optical.propagate(changed, case["cell_length_m"], snapshot["segment_profile"],
        case["input_pump_power_W"], case["input_seed_power_W"])
    np.testing.assert_allclose(result["G_s"], expected[0], rtol=1e-12)
    np.testing.assert_allclose(result["G_c"], expected[1], rtol=1e-12)
    assert result["selected_point"]["G_s"] == pytest.approx(result["G_s"][1])
    assert result["corrected_field_transfer_small_signal"].shape == (3, 2, 2)
    assert result["offdiagonal_scale"] == pytest.approx(1.5)
    assert result["diagonal_susceptibility_scale"] == 1.
    assert not np.allclose(result["G_s"], case["G_s"]*1.5)
    np.testing.assert_array_equal(case["T_canonical_small_signal"], np.broadcast_to(np.eye(2), (3, 2, 2)))


def test_disabled_old_raw_needs_no_atomic_context_and_pump_zero_has_no_fitted_cross_scale(case, record, atomic_boundary, monkeypatch):
    off = model.calibrate_spectrum(case, enabled=False, selected_delta_mhz=-16.)
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(off[key], case[key])
        assert not np.shares_memory(off[key], case[key])
    case["input_pump_power_W"] = 0.
    zero = model.evaluate(case, -16., enabled=True)
    assert zero["offdiagonal_scale"] == 1.
    assert zero["diagonal_susceptibility_scale"] == 1.
    extrapolation = model.evaluate(case, 100., enabled=True)
    assert extrapolation["applied"] and not extrapolation["within_training_domain"]
    assert "EXTRAPOLATION" in str(extrapolation["point_status"])


def test_plus_branch_uses_its_own_external_carrier(case, atomic_boundary):
    case["branch"] = 1
    value = model.build_response(case, [-16.])
    assert atomic_boundary[1][0][2] == 1
    np.testing.assert_allclose(value["probe_axis_GHz"], [case["raman_center_plus_GHz"]-.016])


def test_same_state_pump_index_matches_explicit_rotating_cross_phase(case, atomic_boundary, monkeypatch):
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    chi, calls = atomic_boundary
    state = np.diag([.95/12]*12+[.05]+[0.]*11).astype(complex)
    state[0, 12] = state[12, 0] = .025
    assert np.linalg.eigvalsh(state).min() >= -1e-15
    monkeypatch.setattr(model, "_atomic_response", lambda encoded, delta, branch, method:
        (np.broadcast_to(chi, (len(delta), 2, 2)).copy(), state, dict(finite=True), {}))
    case["propagation_snapshot"] = dict(segment_profile=np.ones(1), dynamic_depletion=False, conjugate_power_ratio=1.)
    raw = model.build_response(case, [-16.])
    added = raw["zeeman_response_provenance"]["pump_phase"]["pump_index_added_mismatch_per_m"]
    assert abs(added) > 0.
    assert raw["delta_k_z"][0] == pytest.approx(raw["vacuum_delta_k_z"][0]+added)
    effective = raw["propagation_snapshot"]["drift_per_m"][0]
    theta = float(raw["delta_k_z"][0])
    bare = effective.copy()
    bare[0, 0] += .5j*theta
    bare[1, 1] -= .5j*theta

    def rhs(z, field):
        matrix = bare.copy()
        matrix[0, 1] *= np.exp(1j*theta*z)
        matrix[1, 0] *= np.exp(-1j*theta*z)
        return matrix@field

    length = case["cell_length_m"]
    initial = np.array([1.+0j, 0j])
    solution = solve_ivp(rhs, (0., length), initial, method="DOP853", rtol=1e-11, atol=1e-12)
    actual = np.diag([np.exp(.5j*theta*length), np.exp(-.5j*theta*length)])@expm(effective*length)@initial
    np.testing.assert_allclose(actual, solution.y[:, -1], rtol=1e-9, atol=1e-10)
