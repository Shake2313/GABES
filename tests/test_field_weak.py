"""Weak-field Gaussian-flight response, including the zero-drive finite-grid subtraction."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import voigt_profile

from gabes import constants, field_history, field_weak, species
from gabes.ballistic_pumping import BallisticProbe85D1, TrajectoryGrid, _geometry

TEMPERATURE_K = 372.15
WAIST_M = 0.00053


def _cross_section(iso, line, field, polarization, detuning, **kwargs):
    width = species.level_structure(iso, line).gamma / 2 + 2e6
    return field_weak.weak_cross_sections(
        iso, line, width, field, polarization, detuning,
        temperature_k=TEMPERATURE_K, waist_m=kwargs.pop("waist_m", WAIST_M), **kwargs)


@pytest.mark.parametrize("iso,line,field,jones,rest_mhz", [
    ("Rb85", "D1", (0., 0., 0.), (1., 0.), -1114.22976625),
    ("Rb87", "D1", (120., 90., 45.), (1., .3 + .7j), -2868.210059),
    ("Rb87", "D2", (0., 0., -270.), (1., 1j), -2516.),
])
def test_discrete_weak_matches_full_obe_photon_flux(iso, line, field, jones, rest_mhz):
    grid = TrajectoryGrid(nx=40, nb=1, nv=1, bcut=.8, vcut=2.)
    impacts, wb, q, wq, *_ = _geometry(grid, ())
    atom = species.level_structure(iso, line)
    speed = q[0] * constants.thermal_speed(TEMPERATURE_K, atom.mass)
    ops = field_history.operators(iso, line, rest_mhz, field_ut=field, jones=jones,
                                  optical_hwhm_increment_rad_s=2e6)
    peak_intensity = 1e-8
    photons = field_history.crossing(ops, peak_intensity, WAIST_M, speed,
                                     steps=grid.nx, impact_over_waist=impacts[0])["photons"]
    flux = 2 * WAIST_M * wb[0] * wq[0] * q[0] * math.exp(-q[0]**2/2) * speed
    power = math.pi * WAIST_M**2 * peak_intensity / 2
    expected = flux * photons * constants.PLANCK * atom.frequency_hz / power
    actual = _cross_section(iso, line, field, jones, rest_mhz * species.MHZ, grid=grid)
    assert actual[0] == pytest.approx(expected, rel=3e-7, abs=0.)


@pytest.mark.parametrize("thermal", [False, True])
@pytest.mark.parametrize("waist", [WAIST_M, 1.5e-6])
def test_continuous_transverse_and_longitudinal_average(thermal, waist):
    iso = species.RB85
    field, pol = (20., -50., 350.), (1+0j, 0j)
    atom = species.level_structure(iso, "D1")
    width = atom.gamma/2 + 2e6
    centers, strengths = field_weak._transitions(iso, "D1", field, pol)
    center = centers[len(centers)//2]
    speed = constants.thermal_speed(TEMPERATURE_K, iso.mass)
    sigma_z = 2 * np.pi * atom.frequency_hz * speed / constants.C_LIGHT if thermal else 0.

    def integrand(q):
        sigma = math.hypot(sigma_z, q*speed/waist)
        profiles = np.pi * voigt_profile(center - centers, sigma, width)
        return q * math.exp(-q*q/2) * (profiles @ strengths) * 1e15

    expected = 1e-15 * quad(integrand, 0., 14., epsrel=1e-11, epsabs=1e-13)[0]
    actual = _cross_section(iso, "D1", field, pol, center, thermal=thermal, waist_m=waist)
    assert actual[0] == pytest.approx(expected, rel=1e-9, abs=0.)


def test_signed_helicity_basis_rotations_and_optically_driven_centers():
    iso, line = "Rb87", "D2"
    field = np.array([200., -170., 300.])
    jones = np.array([.7+.05j, .3-.8j])
    angle = .79
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    rotated = np.r_[rotation @ field[:2], field[2]]
    axis = np.linspace(-3500., 4500., 43) * species.MHZ
    response = _cross_section(iso, line, field, jones, axis)
    for other_field, other_pol in [(rotated, rotation @ jones),
                                    (-field, jones.conjugate()), (field, -2j*jones)]:
        np.testing.assert_allclose(_cross_section(iso, line, other_field, other_pol, axis),
                                   response, rtol=1e-12, atol=0.)
    forward = field_weak.transition_centers(iso, line, (0., 0., 300.), (1., 1j))
    reverse = field_weak.transition_centers(iso, line, (0., 0., 300.), (1., -1j))
    both = field_weak.transition_centers(iso, line, (0., 0., 300.), (1., 0.))
    assert not forward.flags.writeable
    np.testing.assert_array_equal(both, np.union1d(forward, reverse))


def test_zero_field_long_flight_recovers_stationary_weak_limit():
    probe = BallisticProbe85D1(TEMPERATURE_K, species.level_structure("Rb85", "D1").gamma/2 + 2e6)
    axis = np.array([-1114.22976625, -960., 1559.92167275]) * species.MHZ
    result = _cross_section("Rb85", "D1", (0., 0., 0.), (0.2j, .7), axis, waist_m=100.)
    np.testing.assert_allclose(result, probe.weak_cross_sections(axis), rtol=1e-10, atol=0.)
