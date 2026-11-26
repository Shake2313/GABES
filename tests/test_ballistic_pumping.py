"""One-velocity ballistic 85Rb D1 populations, quadrature and absorbed power."""
import numpy as np
import pytest
from scipy.linalg import expm

from gabes import species
from gabes.ballistic_pumping import BallisticProbe85D1, SDIRK_DIAGONAL, TrajectoryGrid
from gabes.optical_pumping import MAGNETIC_PI, ProjectedProbe


WAIST = 530e-6


@pytest.fixture(scope="module")
def atom():
    m = species.build_manifold(species.RB85, "D1")
    return BallisticProbe85D1(372.15, 0.5 * m.gamma + 0.5e6)


@pytest.fixture(scope="module")
def resonance(atom):
    m = atom.manifold
    return m.e_excited[list(m.Fe).index(3.0)] - m.e_ground[list(m.Fg).index(3.0)]


def test_unilluminated_thermal_entry_and_column_balance(atom, resonance):
    for method in (False, True):
        result = atom.path(resonance, np.zeros(4), np.zeros(4), 2e-8, accelerated=method)
        np.testing.assert_allclose(result.populations, np.broadcast_to(atom.thermal, (5, 14)), atol=1e-14)
        np.testing.assert_array_equal(result.absorbed_photons_per_step, 0)
    assert np.max(abs(atom._free.sum(axis=0))) < 3e-8
    np.testing.assert_array_equal(atom._free @ atom.thermal, np.zeros(14))


def test_cw_turn_on_direct_matrix_exponential_and_photon_conservation(atom, resonance):
    intensity = 114.0
    duration = 0.25e-6
    steps = 200
    x = np.full(steps, intensity)
    rate, _ = atom._drive(resonance)
    exact = expm(duration * (atom._free + intensity * rate)) @ atom.thermal
    dense = atom.path(resonance, x, x, duration/steps, accelerated=False)
    native = atom.path(resonance, x, x, duration/steps, accelerated=True)
    np.testing.assert_allclose(native.populations, dense.populations, atol=4e-14)
    np.testing.assert_allclose(native.absorbed_photons_per_step, dense.absorbed_photons_per_step,
                               rtol=8e-13, atol=1e-16)
    np.testing.assert_allclose(dense.populations[-1], exact, atol=3e-6)
    assert np.max(abs(dense.populations.sum(axis=1)-1.0)) < 1e-12
    excited = dense.populations[:, atom._ng:].sum(axis=-1)
    total = excited[-1] + atom.manifold.gamma * np.trapezoid(excited, dx=duration/steps)
    assert dense.absorbed_photons == pytest.approx(total, rel=1e-4)


def test_exponential_age_recovers_constant_intensity_thermal_reset(atom, resonance):
    # An exponential age distribution on exact CW trajectories obeys
    # (r I - L) mean(p) = r p_thermal. Use independent stationary rate code.
    reset_rate = 0.5e6
    probe = ProjectedProbe(atom.manifold, atom.width, reset_rate, resolution=MAGNETIC_PI)
    intensity = 10.0
    target = probe.populations(intensity, np.array([resonance]))[0]
    times = np.linspace(0.0, 22/reset_rate, 4401)
    inputs = np.full(len(times)-1, intensity)
    result = atom.path(resonance, inputs, inputs, times[1])
    avg = np.trapezoid(result.populations * (reset_rate*np.exp(-reset_rate*times))[:,None], x=times, axis=0)
    np.testing.assert_allclose(avg, target, atol=6e-6, rtol=0)


def test_gaussian_cross_section_parity_and_weak_limit(atom, resonance):
    grid = TrajectoryGrid(nx=96, nb=8, nv=10)
    delta = [resonance, resonance+2*np.pi*1e8]
    intensity = [0, 1e-8, 20]
    dense = atom.cross_sections(delta, intensity, WAIST, grid=grid, accelerated=False)
    compiled = atom.cross_sections(delta, intensity, WAIST, grid=grid)
    np.testing.assert_allclose(compiled, dense, rtol=2e-12, atol=0)
    np.testing.assert_allclose(dense[0], atom.weak_cross_sections(delta)*grid.weak_factor(), rtol=1e-14)
    np.testing.assert_allclose(dense[0], dense[1], rtol=3e-8)
    assert np.all(dense[-1] < dense[0])


def test_radial_rates_use_path_history_and_same_weak_annulus_weights(atom, resonance):
    nodes = np.linspace(0., 4., 161)
    intensity = np.exp(-2*nodes**2)
    edges = [0., np.sqrt(np.log(2)/2), 4.]
    grid = TrajectoryGrid(nx=96, nb=6, nv=10)
    weights = atom.radial_weak_weights(nodes, intensity, edges, WAIST, grid=grid)
    assert weights[0]/weights.sum() == pytest.approx(0.5, abs=0.002)
    weak = atom.radial_rates([resonance], nodes, 1e-8*intensity, edges, WAIST, grid=grid)
    np.testing.assert_allclose(weak, atom.weak_cross_sections(resonance)[:,None]*weights*1e-8, rtol=3e-8)
    high = atom.radial_rates(resonance, nodes, 12*intensity, edges, WAIST, grid=grid)
    direct = atom.radial_rates(resonance, nodes, 12*intensity, edges, WAIST, grid=grid, accelerated=False)
    np.testing.assert_allclose(high, direct, rtol=2e-12)
    assert np.all(high > 0)


def test_domain_and_negative_illumination(atom):
    with pytest.raises(ValueError):
        BallisticProbe85D1(300., 1.)
    with pytest.raises(ValueError):
        atom.cross_sections(0.0, -1.0, WAIST)
    with pytest.raises(ValueError):
        atom.radial_rates(0., [0., 2., 1.], [1., 2., 0.], [0., 2.], WAIST)
    with pytest.raises(ValueError):
        atom.path(0., [1., 2.], [1.], 0.01)


def test_default_optical_width_and_gaussian_weak_measure():
    temp = 372.15
    atom = BallisticProbe85D1(temp)
    density = species.number_density(species.RB85, temp)
    expected = 0.5 * (atom.manifold.gamma +
                      species.self_broadening(species.RB85, density, "D1", per_line=True))
    assert atom.width == expected
    assert abs(TrajectoryGrid().weak_factor() - 1) < 1e-6
    with pytest.raises(ValueError, match="positive integer"):
        TrajectoryGrid(nx=1.5)


def test_one_sided_hard_aperture_weak_power(atom):
    # The physical profile cuts off inside an independently defined output ring.
    grid = TrajectoryGrid(nx=24, nb=24, nv=12)
    radius = 0.6 * WAIST
    received = atom.radial_weak_weights([0., .6], [1., 1.], [0., 1.], WAIST, grid=grid)[0]
    assert received == pytest.approx(np.pi * radius**2, rel=2e-5)
