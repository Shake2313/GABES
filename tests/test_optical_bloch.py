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
