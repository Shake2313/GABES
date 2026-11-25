"""Independent atomic, passive and axial propagation checks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gabes import constants, hyperfine, linear_hyperfine
from gabes.schemes import fwm
from gabes.schemes.fwm.longitudinal import LongitudinalSeededInputs, LongitudinalSeededMedium


def _medium(**overrides):
    inputs = LongitudinalSeededInputs(
        D_GHz=0.9, delta_mhz=-8.0, T=362.15, L=0.0004, P_pump=0.12,
        P_probe=8e-6, w_pump=950e-6, w_probe=300e-6, w_conj=400e-6,
        angle_deg=0.32, transit_rate=constants.GAMMA_GG, coupling_residual=1.0,
    )
    return LongitudinalSeededMedium(
        replace(inputs, **overrides),
        velocity_step=100., velocity_cutoff=0.5, transverse_order=2, floquet_order=3)


@pytest.mark.parametrize("branch", [-1, +1])
def test_local_three_carrier_state_matches_dense_floquet_and_pump_reference(branch):
    medium = _medium(branch=branch)
    pump = 2*np.pi*40e6
    finite_rabi = np.array([pump, 2e6+1e6j, 3e6-4e6j])
    actual = medium.atomic_harmonics(finite_rabi)
    np.testing.assert_allclose(actual, medium.atomic_harmonics(finite_rabi, dense=True),
                               rtol=1e-9, atol=2e-13)

    reference = fwm.pump_only_weak_response_reference(
        pump, pump, [2*np.pi*1e6*medium.inputs.delta_mhz], medium.deff,
        atom=medium.atom, branch=branch)
    n = medium.floquet_order
    pump_only = medium.atomic_harmonics([pump, 0., 0.])
    np.testing.assert_allclose(pump_only[:, n], reference.pump_state,
                               rtol=1e-10, atol=1e-14)
    weak_rabi = np.array([100., 60.-15j])
    weak = medium.atomic_harmonics([pump, *weak_rabi])
    p = np.einsum("pij,vpji->vp", medium.lowering[1:], weak[:, [n+1, n-1]])
    expected = reference.chi_matrix[0, 0] @ np.array([weak_rabi[0], weak_rabi[1].conjugate()])
    np.testing.assert_allclose(np.column_stack([p[:, 0], p[:, 1].conj()]),
                               expected, rtol=1e-6, atol=5e-15)


@pytest.mark.parametrize("branch", [-1, +1])
def test_passive_propagation_recovers_beer_lambert_for_each_manifold(branch):
    # The v=0 class probes the strong F'=3 passive line at a tiny Rabi rate.
    D = -constants.NU_HF/1e9 if branch > 0 else 0.0
    medium = _medium(branch=branch, D_GHz=D, delta_mhz=3., P_pump=0., P_probe=1e-15,
                     L=4e-7)
    inputs = medium.inputs
    probe_F = 3 if branch == -1 else 2
    origin = 2*np.pi*hyperfine.LINE_SHIFT_HZ[2, 3]
    detuning = medium.deff + 2*np.pi*1e6*inputs.delta_mhz + branch*constants.OMEGA_HF
    alpha, _, _ = linear_hyperfine.rb85_d1_alpha_faddeeva(
        origin+detuning, inputs.T, doppler_on=False, buffer_gamma=2*inputs.transit_rate,
        lines=((probe_F, 2), (probe_F, 3)))
    alpha = (medium.velocity_weights @ alpha)*medium.k[1]/constants.K_VEC
    expected = np.exp(-alpha*inputs.L)
    solution = medium.propagate(8)
    assert solution["probe_gain"] == pytest.approx(expected, abs=2e-10, rel=0.)
    assert solution["conjugate_gain"] == 0.0
    assert np.max(np.abs(solution["photon_balance_relative"])) < 1e-10


def test_segmented_axial_fields_match_adaptive_ode_and_close_the_photon_ledger():
    medium = _medium()
    fixed = medium.converged(8, noise_ledger=True)
    assert fixed["longitudinal_convergence"]["passed"]
    initial = fixed["envelope_sqrt_W"][0]
    reference = solve_ivp(medium.carrier_rate, (0., medium.inputs.L), initial,
                          rtol=1e-11, atol=1e-14, method="DOP853")
    assert reference.success
    np.testing.assert_allclose(fixed["envelope_sqrt_W"][-1], reference.y[:, -1],
                               rtol=1e-5, atol=1e-11)
    assert fixed["power_W"][-1, 0] < fixed["power_W"][0, 0]
    assert fixed["conjugate_gain"] > 0.
    assert np.max(np.abs(fixed["photon_balance_relative"])) < 1e-7
    assert np.max(np.abs(fixed["power_balance_residual_W"])) < 1e-7*medium.inputs.P_pump
    assert np.any(fixed["spectral_power_change_W"] != 0.)
    sinks = fixed["cumulative_photon_sinks_s_inverse"]
    assert (np.diff(sinks, axis=0) > 0.).all()
    noise = fixed["noise_ledger"]
    matrices = noise["ordered_atomic_diffusion_s_inverse"]
    assert matrices.shape == (16, len(noise["reservoirs"]), 6, 6)
    np.testing.assert_allclose(matrices, matrices.conj().swapaxes(-1, -2), atol=1e-8)
    assert np.min(np.linalg.eigvalsh(matrices)) > -1e-8


def test_zero_coupling_leaves_all_three_input_carriers_unchanged():
    solution = _medium(coupling_residual=0.).propagate(2, noise_ledger=True)
    np.testing.assert_array_equal(solution["envelope_sqrt_W"],
                                  np.tile(solution["envelope_sqrt_W"][0], (3, 1)))
    assert not np.any(solution["cumulative_photon_sinks_s_inverse"])
    assert not np.any(solution["photon_balance_relative"])
