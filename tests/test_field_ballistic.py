"""Crossing-flux normalization, coherent symmetry reductions and radial photon accounting."""
from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from gabes import beam, field_history, species
from gabes.ballistic_atom import BallisticProbe
from gabes.ballistic_pumping import TrajectoryGrid, _geometry
from gabes.field_ballistic import FieldBallisticProbe


WAIST = 0.53e-3
TEMPERATURE = 372.15
GRID = TrajectoryGrid(nx=48, nb=2, nv=2, xcut=4, bcut=4, vcut=7)


def test_gaussian_flux_matches_independent_crossings_and_finite_grid_weak_limit():
    atom = FieldBallisticProbe(species.RB85, "D1", TEMPERATURE)
    frequency = -1114.22976625 * species.MHZ
    peak = beam.peak_intensity(3e-6, WAIST)
    offsets, *_ = _geometry(GRID, ())
    *_, scales, flux = atom._geometry._measure(GRID, WAIST)
    ops = atom._operators(frequency)
    photons = 0.
    for b, impact in enumerate(offsets):
        for v, scale in enumerate(scales):
            counts = field_history.crossing(ops, peak, WAIST, WAIST / scale,
                                           impact_over_waist=impact, steps=GRID.nx,
                                           half_span_waist=GRID.xcut)
            photons += flux[b, v] * counts["photons"]

    measured = atom.cross_sections([frequency], [0., 1e-10, peak], WAIST, grid=GRID)[:, 0]
    np.testing.assert_allclose(measured[-1], photons * atom.photon_energy_j / 3e-6,
                               rtol=2e-9, atol=0)
    np.testing.assert_allclose(measured[0], measured[1], rtol=1e-9, atol=0)
    np.testing.assert_allclose(measured[0], atom.weak_cross_sections(frequency, WAIST, grid=GRID),
                               rtol=2e-9, atol=0)
    direct = atom.cross_sections(frequency, [0., peak], WAIST, grid=GRID, accelerated=False)
    np.testing.assert_allclose(measured[[0, -1]], direct[:, 0], rtol=2e-9, atol=0)


@pytest.mark.parametrize("field,jones", [
    ((21., 28., 0.), (0.6j, 0.8j)),
    ((28., -21., 0.), (0.6j, 0.8j)),
    ((0., 0., -35.), (1., 0.)),
    ((2., 0., 4.), (1., 1j)),
])
def test_quantization_agrees_with_original_lab_jones_basis(field, jones):
    frequency = -1114.22976625
    atom = FieldBallisticProbe(species.RB85, "D1", TEMPERATURE, field_ut=field, jones=jones)
    direct = field_history.operators("Rb85", "D1", frequency, field_ut=field, jones=jones,
                                     optical_hwhm_increment_rad_s=atom.width - atom.manifold.gamma / 2)
    oriented = atom._operators(frequency * species.MHZ)
    args = (15., WAIST, 275.)
    reference = field_history.crossing(direct, *args, steps=48)["photons"]
    actual = field_history.crossing(oriented, *args, steps=48)["photons"]
    assert actual == pytest.approx(reference, rel=2e-9)


def test_annular_loss_is_spontaneous_emissions_plus_excited_population_change_in_each_bin():
    atom = FieldBallisticProbe(species.RB87, "D1", TEMPERATURE, field_ut=(22., 0., 0.))
    frequency = atom.manifold.omega[-1]
    ops = atom._operators(frequency)
    nodes = [0., 0.55, 1.3]
    intensities = [14., 9., 2.]
    edges = [0., 0.6, 1.2]
    grid = TrajectoryGrid(nx=192, nb=1, nv=1, xcut=2.5, bcut=1.5, vcut=2.5)
    rates = atom.radial_rates([frequency], nodes, intensities, edges, WAIST, grid=grid)
    first, second, steps, bins, scales, flux, n = atom._geometry._radial_geometry(
        nodes, intensities, edges, WAIST, grid)
    expected = np.zeros(n)
    eye = sparse.eye(len(ops.initial), format="csc")
    d = field_history.SDIRK_DIAGONAL
    drive = field_history.rabi_per_sqrt_intensity(ops) * ops.driver
    for b in range(len(first)):
        for v, scale in enumerate(scales):
            state = ops.initial.copy()
            for j in np.flatnonzero(steps[b]):
                dt = steps[b, j] * scale
                prev = ops.excited_population @ state
                a = spsolve(eye - d * dt * (ops.base + np.sqrt(first[b, j]) * drive), state)
                state = spsolve(eye - d * dt * (ops.base + np.sqrt(second[b, j]) * drive),
                                state + (1 - d) / d * (a - state))
                excited = ops.excited_population @ state
                emitted = ops.gamma * dt * ((1 - d) * (ops.excited_population @ a) + d * excited)
                if bins[b, j] >= 0:
                    expected[bins[b, j]] += flux[b, v] * (excited - prev + emitted).real
    np.testing.assert_allclose(rates[0], expected * atom.photon_energy_j, rtol=2e-9, atol=0)
    np.testing.assert_array_equal(atom.radial_rates([frequency], nodes, [0.] * 3, edges, WAIST, grid=grid),
                                  [[0., 0.]])


def test_a7_atom_frame_inner_edges_match_rate_projection_at_three_probe_powers():
    atom = FieldBallisticProbe(species.RB85, "D1", TEMPERATURE)
    rate = BallisticProbe(species.RB85, "D1", TEMPERATURE)
    detuning = species.MHZ * np.array([-558.7708085, 1125.9761453])
    intensity = [beam.peak_intensity(p * 1e-6, WAIST) for p in (1., 10., 100.)]
    grid = TrajectoryGrid(nx=96, nb=10, nv=12, xcut=4, bcut=4, vcut=7)
    expected = rate.cross_sections(detuning, intensity, WAIST, grid=grid)
    actual = atom.cross_sections(detuning, intensity, WAIST, grid=grid)
    np.testing.assert_allclose(actual, expected, rtol=0.0015, atol=0)


def test_transverse_ensemble_is_field_sensitive_and_handles_ill_conditioned_eigenbasis(monkeypatch):
    parallel = FieldBallisticProbe(species.RB87, "D1", TEMPERATURE, field_ut=(35., 0., 0.))
    perpendicular = FieldBallisticProbe(species.RB87, "D1", TEMPERATURE, field_ut=(0., 35., 0.))
    grid = TrajectoryGrid(nx=48, nb=2, nv=2, xcut=3, bcut=2.5, vcut=5)
    args = (parallel.manifold.omega[-1], [2., 14.], WAIST)
    along = parallel.cross_sections(*args, grid=grid)
    across = perpendicular.cross_sections(*args, grid=grid)
    assert across[-1, 0] > 1.3 * along[-1, 0]

    # A nearly singular basis must go back to sparse linear-system solves,
    # without losing the transverse Zeeman contribution.
    monkeypatch.setattr("gabes.field_ballistic.linalg.get_lapack_funcs",
                        lambda *args: lambda *factors: (0.0, 0))
    fallback = perpendicular.cross_sections(*args, grid=grid)
    np.testing.assert_allclose(fallback, across, rtol=2e-9, atol=0)


@pytest.mark.parametrize("accelerated", [True, False])
def test_inflight_calculation_can_be_cancelled(accelerated):
    atom = FieldBallisticProbe(species.RB85, "D1", TEMPERATURE)

    def cancel():
        raise CancelledError()

    with pytest.raises(CancelledError):
        atom.cross_sections(0., 2., WAIST, grid=GRID, accelerated=accelerated, checkpoint=cancel)


def test_arbitrary_jones_normalization_is_shared_with_the_weak_limit():
    tiny = np.nextafter(0., 1.)
    atom = FieldBallisticProbe(species.RB85, "D1", TEMPERATURE, jones=(tiny, tiny*1j))
    ordinary = FieldBallisticProbe(species.RB85, "D1", TEMPERATURE, jones=(1., 1j))
    np.testing.assert_array_equal(atom.jones, ordinary.jones)



@pytest.mark.parametrize("isotope,line", [(species.RB85, "D1"), (species.RB87, "D1"),
                                           (species.RB85, "D2"), (species.RB87, "D2")])
def test_oblique_reduction_matches_original_at_the_same_stages(isotope, line, monkeypatch):
    from gabes import field_ballistic

    atom = FieldBallisticProbe(isotope, line, TEMPERATURE, field_ut=(25., -18., 35.), jones=(.8, .6j))
    delta = -1114.23 * species.MHZ
    intensity = [0., 50., 200., 500.]
    # The production path doubles nx if the original mesh has negative populations.
    grid = replace(GRID, nx=192) if isotope == species.RB85 and line == "D1" else GRID
    # Evaluate the full connected-system reference on identical quadrature.
    with monkeypatch.context() as patch:
        patch.setattr(FieldBallisticProbe, "_bipartite_model", lambda self: None)
        expected = atom.cross_sections(delta, intensity, WAIST, grid=grid)

    def unexpected(*args):
        pytest.fail("reduced solver fell back on a validated same-grid input")

    monkeypatch.setattr(field_ballistic, "_fixed_steps", unexpected)
    actual = atom.cross_sections(delta, intensity, WAIST, grid=grid)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)
    weak = atom.weak_cross_sections(delta, WAIST, grid=grid)
    np.testing.assert_allclose(actual[0], weak, rtol=2e-9, atol=0.)


def test_reduced_stage_fallback_and_cancellation(monkeypatch):
    from gabes import field_ballistic, _field_sdirk

    atom = FieldBallisticProbe(species.RB87, "D1", TEMPERATURE, field_ut=(23., -42., 57.),
                              jones=(.8, .6j))
    frequency, peak = -1114.23 * species.MHZ, [0., 70.]
    def cancel():
        raise CancelledError()

    with pytest.raises(CancelledError):
        atom.cross_sections(frequency, peak, WAIST, grid=GRID, checkpoint=cancel)

    monkeypatch.setattr(_field_sdirk, "_iterative_stage", lambda *args: None)
    original = field_ballistic._fixed_steps
    seen = []
    def fallback(*args):
        seen.append(True)
        return original(*args)
    fast = atom.cross_sections(frequency, peak, WAIST, grid=GRID)
    monkeypatch.setattr(field_ballistic, "_fixed_steps", fallback)
    with monkeypatch.context() as patch:
        patch.setattr(_field_sdirk, "_eigen_blocks", lambda matrix: None)
        recovered = atom.cross_sections(frequency, peak, WAIST, grid=GRID)
        np.testing.assert_allclose(recovered, fast, rtol=1e-12, atol=0.)
    with monkeypatch.context() as patch:
        patch.setattr(_field_sdirk, "_within_residual", lambda *args: False)
        recovered = atom.cross_sections(frequency, peak, WAIST, grid=GRID)
        np.testing.assert_allclose(recovered, fast, rtol=1e-12, atol=0.)
    assert len(seen) == 2


def test_neumann_residual_cap():
    from gabes import _field_sdirk

    # The zero RHS passes immediately. The second RHS needs nine updates.
    rhs = np.array([[0., 2.]])
    amplitude_squared = np.ones(2)
    operator = np.array([[0.06]])
    assert _field_sdirk._neumann(rhs, amplitude_squared, operator, max_updates=8) is None
    result = _field_sdirk._neumann(rhs, amplitude_squared, operator, max_updates=9)
    assert result is not None
    assert result[0, 0] == 0.
    assert abs((rhs - result + operator @ result)[0, 1]) <= 1e-12 * rhs[0, 1]


def test_neumann_crossing_and_capped_a1_replay(monkeypatch):
    from functools import partial

    from gabes import _field_sdirk

    atom = FieldBallisticProbe(species.RB87, "D1", TEMPERATURE, field_ut=(23., -42., 57.),
                              jones=(.8, .6j))
    frequency = -500. * species.MHZ
    peak = [0., .1, .27]
    fixed = _field_sdirk.fixed_steps
    with monkeypatch.context() as patch:
        patch.setattr(_field_sdirk, "fixed_steps", partial(fixed, neumann=False))
        expected = atom.cross_sections(frequency, peak, WAIST, grid=GRID)
    with monkeypatch.context() as patch:
        def unexpected(matrix):
            pytest.fail("weak off-resonant crossing fell back to A1")

        patch.setattr(_field_sdirk, "_eigen_blocks", unexpected)
        actual = atom.cross_sections(frequency, peak, WAIST, grid=GRID)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.)

    solve = _field_sdirk._neumann
    monkeypatch.setattr(_field_sdirk, "_neumann", partial(solve, max_updates=0))
    actual = atom.cross_sections(frequency, peak, WAIST, grid=GRID)
    np.testing.assert_array_equal(actual, expected)


def test_d2_refined_a1_crossing_matches_direct_schur(monkeypatch):
    from functools import partial

    from scipy import linalg

    from gabes import _field_sdirk

    atom = FieldBallisticProbe(species.RB87, "D2", 348.15, field_ut=(23., -42., 57.),
                              jones=(.8, .6j))
    args = (500. * species.MHZ, [3 * .272 / 7], .00053)
    grid = TrajectoryGrid(nx=48, nb=4, nv=4)
    fixed = _field_sdirk.fixed_steps

    def direct(rhs, squared, operator):
        identity = np.eye(len(operator))
        return np.column_stack([linalg.solve(identity - power * operator, column, check_finite=False)
                                for power, column in zip(squared, rhs.T)])

    with monkeypatch.context() as patch:
        patch.setattr(_field_sdirk, "_neumann", direct)
        patch.setattr(_field_sdirk, "fixed_steps", partial(fixed, neumann=True))
        expected = atom.cross_sections(*args, grid=grid)
    with monkeypatch.context() as patch:
        patch.setattr(_field_sdirk, "fixed_steps", partial(fixed, neumann=False))
        corrected = atom.cross_sections(*args, grid=grid)
    np.testing.assert_allclose(corrected, expected, rtol=1e-12, atol=0.)

    def unexpected(*args):
        pytest.fail("Eligible D2 crossing should use Neumann without the eigenbasis")

    with monkeypatch.context() as patch:
        patch.setattr(_field_sdirk, "_eigen_blocks", unexpected)
        actual = atom.cross_sections(*args, grid=grid)
    np.testing.assert_allclose(actual, corrected, rtol=1e-12, atol=0.)

    solve = _field_sdirk._neumann
    monkeypatch.setattr(_field_sdirk, "_neumann", partial(solve, max_updates=0))
    replay = atom.cross_sections(*args, grid=grid)
    np.testing.assert_array_equal(replay, corrected)


def test_refined_eigen_stage_checks_residual_after_correction(monkeypatch):
    from gabes import _field_sdirk

    operator = np.diag([.1, .2])
    incoming = np.array([[1.], [-2.]])
    roots = np.array([1.])
    stage = _field_sdirk._refined_eigen_stage(
        roots, operator, operator, lambda *args: (incoming, incoming), lambda *args: incoming,
        _field_sdirk._eigen_blocks(operator),
    )
    apply = _field_sdirk._apply_blocks
    seen = []

    def perturbed(*args):
        seen.append(True)
        solution = apply(*args)
        return solution + 1e-8 if len(seen) == 1 else solution

    monkeypatch.setattr(_field_sdirk, "_apply_blocks", perturbed)
    dark, optical = stage(incoming, incoming, roots, False)
    expected = np.linalg.solve(np.eye(2) - operator, incoming)
    np.testing.assert_allclose(optical, expected, atol=0., rtol=1e-12)
    np.testing.assert_allclose(dark, incoming + operator @ expected, atol=0., rtol=1e-12)
    assert len(seen) == 2

    monkeypatch.setattr(_field_sdirk, "_apply_blocks", lambda *args: apply(*args) + 1e-8)
    with pytest.raises(ArithmeticError, match="Unresolved refined"):
        stage(incoming, incoming, roots, False)
