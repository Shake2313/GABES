"""Isotope-resolved, zero-field beam crossings for both rubidium D lines."""
import numpy as np
import pytest
from scipy.linalg import expm

from gabes import constants, species
from gabes.ballistic_atom import BallisticProbe
from gabes.ballistic_pumping import BallisticProbe85D1, TrajectoryGrid


@pytest.fixture(params=[(species.RB85, "D1"), (species.RB87, "D1"), (species.RB85, "D2"), (species.RB87, "D2")],
                ids=["85-D1", "87-D1", "85-D2", "87-D2"])
def atom(request):
    isotope, line = request.param
    return BallisticProbe(isotope, line, 372.15)


def test_thermal_atom_recovers_hyperfine_lorentzian_absorption(atom):
    m = atom.manifold
    areas = species.line_integrated_alpha(m.iso, m.line, 1.0)
    detuning = np.unique(np.r_[m.omega, 0.0, m.omega + 5 * atom.width])
    expected = np.zeros_like(detuning)
    for (ground, excited), area in areas.items():
        center = m.e_excited[m.Fe.index(excited)] - m.e_ground[m.Fg.index(ground)]
        expected += (area / np.pi) * atom.width / ((detuning - center)**2 + atom.width**2)
    np.testing.assert_allclose(atom.weak_cross_sections(detuning), expected, rtol=3e-13, atol=0)
    assert atom.speed_sigma == constants.thermal_speed(372.15, m.iso.mass)
    density = species.number_density(m.iso, 372.15)
    assert atom.width == 0.5 * (m.gamma + species.self_broadening(m.iso, density, m.line, per_line=True))


def test_native_history_matches_dense_and_constant_light_propagator(atom):
    intensity = 60.0
    duration = 0.32e-6
    steps = 200
    drive = np.full(steps, intensity)
    m = atom.manifold
    detuning = m.omega[-1] + 0.5 * atom.width
    native = atom.path(detuning, drive, drive, duration / steps)
    dense = atom.path(detuning, drive, drive, duration / steps, accelerated=False)
    np.testing.assert_allclose(native.populations, dense.populations, atol=1e-13, rtol=0)
    np.testing.assert_allclose(native.absorbed_photons_per_step, dense.absorbed_photons_per_step, rtol=2e-12, atol=0)

    rate, _ = atom._drive(detuning)
    exact = expm(duration * (atom._free + intensity * rate)) @ atom.thermal
    np.testing.assert_allclose(native.populations[-1], exact, atol=2e-6, rtol=0)
    np.testing.assert_allclose(native.populations.sum(axis=-1), 1, atol=2e-13, rtol=0)
    np.testing.assert_array_equal(atom._free @ atom.thermal, 0)


def test_rb85_d1_keeps_v2_trajectory_bytes():
    original = BallisticProbe85D1(370.0, 2e7)
    generic = BallisticProbe(species.RB85, "D1", 370.0, 2e7)
    delta = original.manifold.omega[:2] + 1e7
    grid = TrajectoryGrid(nx=24, nb=4, nv=4)
    args = (delta, [0., 3.], 0.53e-3)
    np.testing.assert_array_equal(generic.cross_sections(*args, grid=grid),
                                  original.cross_sections(*args, grid=grid))


def test_domain_and_width_contract():
    for isotope, line in ((species.CS133, "D1"), (species.RB87, "D3")):
        with pytest.raises(ValueError, match="Rb85 or Rb87"):
            BallisticProbe(isotope, line, 370.0)
    with pytest.raises(ValueError, match="below the natural HWHM"):
        BallisticProbe(species.RB87, "D2", 370.0, 10.0)
