"""ProjectedProbe(resolution="magnetic_pi") against the unfolded sublevel rate equations.

⟦AI⟧ Ported from AutoOD 0.2.2 tests/test_probe_populations.py (tag baseline-0.2.2-src,
commit 1bd6615), where they checked AutoOD's ZeemanProbe. Only the construction changed.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from gabes import beam, constants, core, linear_hyperfine as linear, species
from gabes.optical_pumping import MAGNETIC_PI, ProjectedProbe


def zeeman_probe(manifold, width, transit):
    return ProjectedProbe(manifold, width, transit, resolution=MAGNETIC_PI)


def physical_fixture(isotope, line):
    transit = 2 * np.pi * 100e3
    temperature = 363.15
    density = species.number_density(isotope, temperature)
    manifold = species.build_manifold(isotope, line, transit_rate=transit)
    width = (manifold.gamma + species.self_broadening(isotope, density)) / 2 + transit
    sigma = manifold.k_vec * np.sqrt(constants.KB * temperature / isotope.mass)
    return manifold, width, transit, sigma


def unfolded_populations(man, width, transit, intensity, detuning):
    """Independent full +/-m population system, with every stimulated pair."""
    atom = species.level_structure(man.iso.name, man.line)
    ng, ne = len(atom.ground), len(atom.excited)
    g_energy = np.array([man.e_ground[man.Fg.index(f)] for f, m in atom.ground])
    e_energy = np.array([man.e_excited[man.Fe.index(f)] for f, m in atom.excited])
    line_center = e_energy[None, :] - g_energy[:, None]
    rabi_sq = atom.dipole_sq * beam.field_amplitude(intensity)**2 / constants.HBAR**2
    rates = (rabi_sq * atom.dipoles[0]**2 * width / 2
             / ((np.asarray(detuning)[:, None, None] - line_center)**2 + width**2))
    n = ng + ne
    # Inflows into row, sources by column; thermal replacement goes to ground sublevels.
    matrix = np.zeros((len(detuning), n, n))
    matrix[:, :ng, ng:] = rates + man.gamma * sum(c**2 for c in atom.dipoles.values())
    matrix[:, ng:, :ng] = rates.transpose(0, 2, 1)
    diag = np.arange(n)
    matrix[:, diag, diag] = -matrix.sum(axis=1) - transit
    source = np.zeros((len(detuning), n, 1))
    source[:, :ng, 0] = -transit / ng
    return np.linalg.solve(matrix, source)[..., 0]


@pytest.mark.parametrize("isotope", [species.RB85, species.RB87], ids=lambda iso: iso.name)
@pytest.mark.parametrize("line", ["D1", "D2"])
def test_weak_areas_match_elemental_faddeeva(isotope, line):
    with core.blas_single_thread():
        man, width, transit, sigma = physical_fixture(isotope, line)
        probe = zeeman_probe(man, width, transit)
        axis = np.linspace(-6e9, 7e9, 71) * 2 * np.pi
        temperature = (sigma / man.k_vec)**2 * isotope.mass / constants.KB
        expected, _ = linear.species_complex_voigt(axis, man, 1., temperature, 2 * width)
        result = probe.absorption_cross_section(np.array([0., 1e-10]), axis, sigma)
        np.testing.assert_allclose(result[0], expected, rtol=2e-13, atol=0)
        np.testing.assert_allclose(result[1], expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize(("isotope", "line"), [(species.RB85, "D2"), (species.RB87, "D1")],
                         ids=["Rb85-D2", "Rb87-D1"])
def test_separate_m_and_hyperfine_populations_against_unfolded_system(isotope, line):
    with core.blas_single_thread():
        man, width, transit, sigma = physical_fixture(isotope, line)
        atom = species.level_structure(isotope.name, line)
        probe = zeeman_probe(man, width, transit)
        frequency = man.omega[0:1]
        direct = unfolded_populations(man, width, transit, 200., frequency)[0]
        ground = direct[:len(atom.ground)]
        # Population moves between hyperfine levels AND m sublevels of the same F.
        gmask = np.array([f == atom.ground[0][0] for f, m in atom.ground])
        assert abs(ground[gmask].sum() - gmask.mean()) > .01
        assert np.ptp(ground[gmask]) > .0001
        grouped = []
        for labels, basis, values in ((probe.ground_groups, atom.ground, ground),
                                      (probe.excited_groups, atom.excited, direct[len(atom.ground):])):
            for label in labels:
                grouped.append(sum(v for (f, m), v in zip(basis, values) if (f, abs(m)) == label))
        result = probe.populations(200., frequency)[0]
        np.testing.assert_allclose(result, grouped, rtol=1e-10, atol=2e-15)
        assert sum(result) == pytest.approx(1., abs=1e-13)
        assert min(result) >= -1e-15


def test_guarded_fallback_covers_full_maxwell_and_matches_poles():
    with core.blas_single_thread():
        man, width, transit, sigma = physical_fixture(species.RB87, "D2")
        probe = zeeman_probe(man, width, transit)
        x = np.linspace(-5e9, 6e9, 42) * 2 * np.pi
        intensities = np.array([1e-7, 60., 1e5])
        expected = probe.absorption_cross_section(intensities, x, sigma)
        # Force failure inside the residue factory so the public cross-section
        # call exercises the full numerical recovery route.
        with patch.object(probe._poles, "residues", side_effect=np.linalg.LinAlgError):
            actual = probe.absorption_cross_section(intensities, x, sigma)
        np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=0)


def test_finite_maxwell_mean_matches_unfolded_direct_quadrature():
    # Gauss-Legendre panels around every Doppler-shifted resonance; +/-10 sigma
    # leaves <2e-23 Gaussian probability outside the integral.
    with core.blas_single_thread():
        man, width, transit, sigma = physical_fixture(species.RB85, "D2")
        probe = zeeman_probe(man, width, transit)
        scan = np.array([man.omega[2], man.omega[-1] + 2.3 * sigma])
        nodes, weights = np.polynomial.legendre.leggauss(32)
        ng = len(species.level_structure(man.iso.name, man.line).ground)
        for intensity in (2., 2500.):
            result = probe.absorption_cross_section([intensity], scan, sigma)[0]
            for frequency, actual in zip(scan, result):
                pt = np.array([-40., -8., -1., 0., 1., 8., 40.]) * width
                centers = (frequency - man.omega[:, None] + pt) / sigma
                edges = np.unique(np.clip(np.r_[centers.ravel(), -10., 10.], -10., 10.))
                half = np.diff(edges)[:, None] / 2
                x = (edges[:-1, None] + half * (nodes + 1)).ravel()
                weights_v = (half * weights).ravel() * np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
                p = unfolded_populations(man, width, transit, intensity, frequency - sigma * x)
                mean = weights_v @ p[:, ng:].sum(axis=1)
                expected = constants.PLANCK * man.nu0 * (man.gamma + transit) * mean / intensity
                assert actual / expected == pytest.approx(1., abs=2e-8)


def test_resolution_is_validated_and_hyperfine_stays_default():
    man, width, transit, _ = physical_fixture(species.RB87, "D2")
    assert ProjectedProbe(man, width, transit).resolution == "hyperfine"
    with pytest.raises(ValueError, match="resolution"):
        ProjectedProbe(man, width, transit, resolution="zeeman")
