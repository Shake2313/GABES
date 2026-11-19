"""Independent thermal-line reference for frequency-aware Rb mixtures.

The generic response uses one physical laser wavevector for both Doppler shifts
and propagating absorption; the legacy pure87 D1 adapter retains centroid k.
"""
import numpy as np
import pytest

from gabes import constants, linear_hyperfine, species
from gabes.nonlinear_response import solve_rb_mean, solve_rb87_d1_mean
from gabes.optical_frequency import RbOpticalFrequency

DENSITY = 1e16
TEMPERATURE_C = 50.
RESET = 1e5


def _voigt_chi(frequency, isotope, density, *, physical_doppler=True):
    manifold = species.build_manifold(species.ISOTOPES[isotope], frequency.line)
    axis = np.array([frequency.detuning_hz(isotope)])*2*np.pi
    laser_k = 2*np.pi/frequency.wavelength_m if physical_doppler else None
    alpha, chi_real = linear_hyperfine.species_complex_voigt(
        axis, manifold, density, TEMPERATURE_C+273.15, manifold.gamma+2*RESET,
        doppler_wavevector_per_m=laser_k)
    return complex(chi_real[0], alpha[0]/manifold.k_vec)


@pytest.fixture(scope="module")
def thermal_cases():
    cache = {}
    cases = {}
    for line in ("D1", "D2"):
        frequency = RbOpticalFrequency(line, -600e6)
        pure = {}
        for isotope, fraction in (("Rb85", 1.), ("Rb87", 0.)):
            pure[isotope] = solve_rb_mean(
                frequency, 0., rb85_fraction=fraction, density_m3=DENSITY,
                temp_c=TEMPERATURE_C, ground_relaxation_hz=RESET,
                velocity_order=6, cutoff_sigma=6., component_cache=cache)
        cases[line] = dict(frequency=frequency, pure=pure)
    return cases, cache


@pytest.mark.parametrize("line", ["D1", "D2"])
@pytest.mark.parametrize("isotope", ["Rb85", "Rb87"])
def test_all_four_thermal_response_endpoints_match_independent_voigt(thermal_cases, line, isotope):
    cases, _ = thermal_cases
    case = cases[line]
    result = case["pure"][isotope]
    expected = _voigt_chi(case["frequency"], isotope, DENSITY)
    # Separate quadratures: a large dispersive wing must not mask absorption.
    assert result["chi"].real == pytest.approx(expected.real, rel=2e-6, abs=1e-20)
    assert result["chi"].imag == pytest.approx(expected.imag, rel=2e-6, abs=1e-20)
    assert result["chi"].imag > 0
    laser_k = 2*np.pi/case["frequency"].wavelength_m
    assert result["alpha_per_m"] == pytest.approx(laser_k*expected.imag, rel=2e-6, abs=1e-14)
    assert tuple(result["components"]) == (isotope,)
    assert result["conditions"]["isotope_density_m3"][isotope] == DENSITY
    assert result["class_count"] > 6
    assert result["diagnostics"]["steady_state_relative_residual"] < 1e-9
    assert not result["convergence"]["assessed"]


@pytest.mark.parametrize("line", ["D1", "D2"])
@pytest.mark.parametrize("fraction", [.5, species.RB85.abundance])
def test_actual_thermal_mixtures_use_density_once_and_one_frequency(thermal_cases, line, fraction):
    cases, cache = thermal_cases
    frequency = cases[line]["frequency"]
    count = len(cache)
    result = solve_rb_mean(
        frequency, 0., rb85_fraction=fraction, density_m3=DENSITY,
        temp_c=TEMPERATURE_C, ground_relaxation_hz=RESET,
        velocity_order=6, cutoff_sigma=6., component_cache=cache)
    assert len(cache) == count  # Reuse the actual pure atomic solutions, not mocks.
    expected = (fraction*_voigt_chi(frequency, "Rb85", DENSITY)
                +(1-fraction)*_voigt_chi(frequency, "Rb87", DENSITY))
    assert result["chi"].real == pytest.approx(expected.real, rel=2e-6, abs=1e-20)
    assert result["chi"].imag == pytest.approx(expected.imag, rel=2e-6, abs=1e-20)
    pure = cases[line]["pure"]
    weighted = fraction*pure["Rb85"]["chi"]+(1-fraction)*pure["Rb87"]["chi"]
    assert result["chi"] == weighted
    laser_k = 2*np.pi/frequency.wavelength_m
    assert result["alpha_per_m"] == pytest.approx(laser_k*result["chi"].imag, rel=1e-14)
    for isotope, share in (("Rb85", fraction), ("Rb87", 1-fraction)):
        assert result["components"][isotope]["conditions"]["density_m3"] == share*DENSITY
        assert result["components"][isotope]["chi"] == share*pure[isotope]["chi"]
        assert result["components"][isotope]["alpha_per_m"] == pytest.approx(
            laser_k*result["components"][isotope]["chi"].imag, rel=1e-14)
    assert result["conditions"]["frequency"] == frequency.as_dict()


def test_generic_and_legacy_conventions_match_their_independent_voigt_limits(thermal_cases):
    cases, _ = thermal_cases
    frequency = cases["D1"]["frequency"]
    result = cases["D1"]["pure"]["Rb87"]
    old = solve_rb87_d1_mean(
        frequency.detuning_mhz("Rb87", reference="upper_ground_centroid"), 0.,
        density_m3=DENSITY, temp_c=TEMPERATURE_C, ground_relaxation_hz=RESET,
        velocity_order=6, cutoff_sigma=6.)
    old_reference = _voigt_chi(frequency, "Rb87", DENSITY, physical_doppler=False)
    new_reference = _voigt_chi(frequency, "Rb87", DENSITY)
    assert old["chi"].real == pytest.approx(old_reference.real, rel=2e-6, abs=1e-20)
    assert old["chi"].imag == pytest.approx(old_reference.imag, rel=2e-6, abs=1e-20)
    actual_shift = result["chi"]-old["chi"]
    expected_shift = new_reference-old_reference
    # Resolve the small deliberate convention change instead of allowing the
    # ordinary Voigt tolerance to conceal a centroid-k Doppler implementation.
    assert actual_shift.real == pytest.approx(expected_shift.real, rel=1e-4, abs=2e-20)
    assert actual_shift.imag == pytest.approx(expected_shift.imag, rel=1e-4, abs=2e-21)
    old_k = 2*np.pi*species.RB87.nu_D1/constants.C_LIGHT
    assert old["alpha_per_m"] == pytest.approx(old_k*old["chi"].imag, rel=1e-12)
    assert old["model_id"] == "rb87_d1_maxwell_full_obe_mean_v1"
