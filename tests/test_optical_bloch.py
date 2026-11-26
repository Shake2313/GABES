"""Generic optical-OBE API checks beyond the independent physics suite."""

import numpy as np
import pytest

from gabes import constants, species
from gabes.optical_bloch import solve_optical_pump


@pytest.mark.parametrize(("isotope", "line"), [("Rb87", "D1"), ("Rb85", "D2")])
def test_compact_mean_omits_only_atomic_state_payload(isotope, line):
    inputs = dict(intensity_mw_cm2=35., ground_relaxation_hz=5e5,
                  jones_xy=(3 / 5, 4j / 5), density_m3=1e15)
    state = solve_optical_pump(isotope, line, 400., **inputs)
    mean = solve_optical_pump(isotope, line, 400., **inputs, include_atomic_state=False)
    omitted = {"rho", "hamiltonian_rad_s", "generator_s_inverse", "reservoirs",
               "optical_coherence_eg"}
    assert set(state) - set(mean) == omitted
    for key in ("chi_probe", "alpha_per_m", "excited_fraction"):
        assert mean[key] == state[key]
    for key in ("polarization_xy_c_m2", "ground_populations", "excited_populations"):
        np.testing.assert_array_equal(mean[key], state[key])
    assert mean["diagnostics"] == state["diagnostics"]


def test_explicit_laser_frequency_sets_doppler_shift_and_propagation_wave_number():
    atom = species.level_structure("Rb85", "D2")
    frequency = 1.001 * atom.frequency_hz
    velocity = 37.
    detuning = 400.
    common = dict(intensity_mw_cm2=12.5, ground_relaxation_hz=5e5, density_m3=1e15)
    moving = solve_optical_pump("Rb85", "D2", detuning, velocity_m_s=velocity,
                                laser_frequency_hz=frequency, **common)
    shifted = detuning - frequency * velocity / constants.C_LIGHT / 1e6
    stationary = solve_optical_pump("Rb85", "D2", shifted,
                                    laser_frequency_hz=frequency, **common)
    np.testing.assert_array_equal(moving["rho"], stationary["rho"])
    assert moving["conditions"]["effective_detuning_mhz"] == shifted
    assert moving["conditions"]["optical_frequency_hz"] == frequency
    assert moving["alpha_per_m"] == pytest.approx(
        2 * np.pi * frequency / constants.C_LIGHT * moving["chi_probe"].imag, rel=1e-12)

    # At rest frequency changes propagation k only; local centroid detuning
    # and atomic density operator retain their specified values.
    original = solve_optical_pump("Rb85", "D2", detuning, **common)
    physical = solve_optical_pump("Rb85", "D2", detuning,
                                  laser_frequency_hz=frequency, **common)
    np.testing.assert_array_equal(original["rho"], physical["rho"])
    assert physical["alpha_per_m"] == pytest.approx(1.001 * original["alpha_per_m"], rel=1e-12)


@pytest.mark.parametrize("frequency", [0., -1., np.nan, np.inf])
def test_explicit_laser_frequency_must_be_finite_positive(frequency):
    with pytest.raises(ValueError, match="laser_frequency_hz"):
        solve_optical_pump("Rb85", "D2", 0., laser_frequency_hz=frequency)


@pytest.mark.parametrize(("isotope", "line"), [
    ("Rb87", "D1"), ("Rb85", "D1"), ("Rb87", "D2"), ("Rb85", "D2"),
])
@pytest.mark.parametrize("jones_xy", [
    (1., 0.),
    (0., 1.),
    (np.cos(0.37), np.sin(0.37)),
    tuple(np.exp(0.6j) * np.array([np.cos(-0.83), np.sin(-0.83)])),
])
def test_linear_pi_mirror_matches_dense_reference_and_compact_contract(isotope, line, jones_xy):
    inputs = dict(intensity_mw_cm2=85., velocity_m_s=73., temp_c=85.,
                  ground_relaxation_hz=3e5, density_m3=1e15, jones_xy=jones_xy)
    auto_state = solve_optical_pump(isotope, line, 450., **inputs, solver="auto", include_atomic_state=True)
    auto_compact = solve_optical_pump(isotope, line, 450., **inputs, solver="auto", include_atomic_state=False)
    ref_state = solve_optical_pump(isotope, line, 450., **inputs, solver="full", include_atomic_state=True)

    # Full z-basis state and populations must match dense reference to <= 1e-12
    np.testing.assert_allclose(auto_state["rho"], ref_state["rho"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(auto_state["ground_populations"], ref_state["ground_populations"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(auto_state["excited_populations"], ref_state["excited_populations"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(auto_state["chi_probe"], ref_state["chi_probe"], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(auto_state["polarization_xy_c_m2"], ref_state["polarization_xy_c_m2"], rtol=1e-12, atol=1e-26)
    assert auto_state["excited_fraction"] == pytest.approx(ref_state["excited_fraction"], rel=1e-12, abs=1e-14)

    # Compact and full payloads must match identically
    for key in ("chi_probe", "alpha_per_m", "excited_fraction"):
        assert auto_compact[key] == auto_state[key]
    for key in ("polarization_xy_c_m2", "ground_populations", "excited_populations"):
        np.testing.assert_array_equal(auto_compact[key], auto_state[key])
    assert auto_compact["diagnostics"] == auto_state["diagnostics"]
