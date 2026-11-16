"""Independent checks for the opt-in projected hyperfine probe model."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import wofz

from gabes import constants, species
from gabes.atoms import AtomModel
from gabes.optical_pumping import ProjectedProbe
from gabes.schemes.sas import SASScheme


def _line_areas(manifold):
    ng = len(manifold.Fg)
    areas = species.line_integrated_alpha(manifold.iso, manifold.line, N=1.0)
    return np.array([
        areas[(manifold.Fg[g], manifold.Fe[e - ng])]
        for g, e in zip(manifold.g_idx, manifold.e_idx)
    ])


def _weak_cross_section(manifold, detunings, optical_hwhm, doppler_sigma):
    delta = np.asarray(detunings)[:, None] - manifold.omega
    z = (delta + 1j * optical_hwhm) / (np.sqrt(2.0) * doppler_sigma)
    profile = wofz(z).real / (np.sqrt(2.0 * np.pi) * doppler_sigma)
    return profile @ _line_areas(manifold)


@pytest.mark.parametrize(("isotope", "line"), [
    (species.RB85, "D1"), (species.RB87, "D2"),
])
def test_weak_cross_section_matches_thermal_link_areas_and_complex_voigt(isotope, line):
    transit = 2.0 * np.pi * 0.21e6
    elastic_dephasing = 2.0 * np.pi * 0.42e6
    manifold = species.build_manifold(isotope, line)
    optical_hwhm = manifold.gamma / 2.0 + transit + elastic_dephasing
    probe = ProjectedProbe(manifold, optical_hwhm, transit)
    detunings = np.array([
        manifold.omega.min() - 2.0 * np.pi * 950e6,
        manifold.omega[0],
        manifold.omega[-1] + 2.0 * np.pi * 130e6,
    ])
    for doppler_sigma in (optical_hwhm * 1e-5, 2.0 * np.pi * 220e6):
        expected = _weak_cross_section(manifold, detunings, optical_hwhm, doppler_sigma)
        linear = probe.linear_cross_section(detunings, doppler_sigma)
        np.testing.assert_allclose(linear, expected, rtol=3e-10, atol=0.0)
        # A strictly positive intensity exercises the driven stationary path.
        finite = probe.absorption_cross_section(np.array([1e-9]), detunings, doppler_sigma)
        np.testing.assert_allclose(finite, expected[None, :], rtol=2e-7, atol=0.0)


def _stationary_rate_equations(manifold, detuning, intensity, optical_hwhm, transit):
    """Build the classical generator from oscillator areas and Einstein balance."""
    n = manifold.n_levels
    ng = len(manifold.Fg)
    degeneracy = 2.0 * np.array(manifold.Fg + manifold.Fe) + 1.0
    reservoir = np.r_[manifold.p_ground, np.zeros(n - ng)]
    generator = transit * (reservoir[:, None] - np.eye(n))
    for source, dest, rate in manifold.atom.decay:
        generator[dest, source] += rate
        generator[source, source] -= rate
    detuning_from_link = detuning - manifold.omega
    weak_lines = (_line_areas(manifold) * (optical_hwhm / np.pi)
                  / (detuning_from_link**2 + optical_hwhm**2))
    photon_energy = constants.PLANCK * manifold.nu0
    upward = intensity * weak_lines / (photon_energy * manifold.p_ground[manifold.g_idx])
    downward = upward * degeneracy[manifold.g_idx] / degeneracy[manifold.e_idx]
    for g, e, up, down in zip(manifold.g_idx, manifold.e_idx, upward, downward):
        generator[e, g] += up
        generator[g, g] -= up
        generator[g, e] += down
        generator[e, e] -= down

    # Replace one row by normalization; scale all other rows without changing
    # the stationary state to keep the solve well conditioned.
    system = generator / manifold.gamma
    system[-1, :] = 1.0
    target = np.zeros(n)
    target[-1] = 1.0
    population = np.linalg.solve(system, target)
    net_per_ground = population[manifold.g_idx] / manifold.p_ground[manifold.g_idx]
    net_per_ground -= (
        population[manifold.e_idx] / manifold.p_ground[manifold.g_idx]
        * degeneracy[manifold.g_idx] / degeneracy[manifold.e_idx]
    )
    return population, net_per_ground @ weak_lines


@pytest.mark.parametrize(("isotope", "line"), [
    (species.RB85, "D1"), (species.RB87, "D2"),
])
def test_stationary_velocity_class_matches_independent_einstein_balance(isotope, line):
    transit = 2.0 * np.pi * 0.13e6
    manifold = species.build_manifold(isotope, line)
    optical_hwhm = manifold.gamma / 2.0 + transit
    probe = ProjectedProbe(manifold, optical_hwhm, transit)
    intensities = np.array([0.015, 23.0, 4.2e3])
    detunings = np.array([manifold.omega.min(), manifold.omega.max() + 0.7 * optical_hwhm])
    reference = [[_stationary_rate_equations(manifold, d, intensity, optical_hwhm, transit)
                  for d in detunings] for intensity in intensities]
    expected_pop = np.array([[state for state, _ in row] for row in reference])
    expected_cross_section = np.array([[sigma for _, sigma in row] for row in reference])

    populations = np.stack([probe.populations(intensity, detunings) for intensity in intensities])
    assert populations.shape == expected_pop.shape
    assert np.isfinite(populations).all()
    assert np.min(populations) >= -5e-14
    np.testing.assert_allclose(populations.sum(axis=-1), 1.0, atol=5e-14, rtol=0.0)
    np.testing.assert_allclose(populations, expected_pop, atol=3e-12, rtol=2e-10)

    # A very narrow Maxwell distribution converges to the directly solved
    # stationary class without requiring a zero-temperature API mode.
    cross_section = probe.absorption_cross_section(intensities, detunings, optical_hwhm * 1e-5)
    assert np.all(cross_section >= 0.0)
    np.testing.assert_allclose(cross_section, expected_cross_section, atol=1e-28, rtol=3e-9)



def test_driven_multilevel_doppler_mean_matches_independent_rate_integration():
    manifold = species.build_manifold(species.RB87, "D2")
    transit = 2.0 * np.pi * 0.08e6
    optical_hwhm = manifold.gamma / 2.0 + transit
    sigma = 2.0 * np.pi * 220e6
    intensity = 110.0
    model = ProjectedProbe(manifold, optical_hwhm, transit)
    detunings = np.array([manifold.omega.min(), manifold.omega.max() + 0.4 * sigma])
    # Normalize before integration so SciPy's tolerance applies to the line
    # shape instead of to an SI cross section of order 1e-15 m².
    scale = float(_line_areas(manifold).max() / optical_hwhm)
    expected = []
    for detuning in detunings:
        def integrand(v_over_sigma):
            _, cross_section = _stationary_rate_equations(
                manifold, detuning - sigma * v_over_sigma, intensity, optical_hwhm, transit,
            )
            return (cross_section / scale * np.exp(-0.5 * v_over_sigma**2)
                    / np.sqrt(2.0 * np.pi))

        centers = (detuning - manifold.omega) / sigma
        points = centers[(centers > -9.0) & (centers < 9.0)]
        value, error = quad(integrand, -9.0, 9.0, points=points, epsabs=1e-12, limit=300)
        assert error < 1e-6 * value
        expected.append(scale * value)

    actual = model.absorption_cross_section(np.array([intensity]), detunings, sigma)
    np.testing.assert_allclose(actual[0], expected, rtol=2e-7, atol=0.0)

def test_isolated_equal_degeneracy_closed_pair_matches_saturated_voigt(monkeypatch):
    # A single allowed F=2 -> F'=2 dipole line, with an artificial closed
    # decay. Equal degeneracies reduce the projected equations to the
    # ordinary closed two-level population equations.
    original = species.build_manifold(species.RB87, "D1")
    transit = 2.0 * np.pi * 0.11e6
    gamma = original.gamma
    optical_hwhm = 0.5 * gamma + transit
    atom = AtomModel(
        name="closed pair fixture", n_levels=2,
        labels=("g", "e"), ground=(0,), excited=(1,),
        decay=((1, 0, gamma),), dephasing=(), doppler_levels=(1,),
    )
    index = next(i for i, (g, e) in enumerate(zip(original.g_idx, original.e_idx))
                 if original.Fg[g] == original.Fe[e - len(original.Fg)] == 2)
    manifold = replace(
        original, atom=atom, Fg=[2], Fe=[2],
        e_ground=np.zeros(1), e_excited=np.zeros(1), p_ground=np.ones(1),
        g_idx=np.array([0]), e_idx=np.array([1]), omega=np.zeros(1),
        S_abs=original.S_abs[[index]], rabi_rel=original.rabi_rel[[index]],
        H_couplings=[(0, 1, original.rabi_rel[index])],
    )
    # Provide the same isolated pair as the shared natural-branching input.
    monkeypatch.setattr(species, "build_manifold", lambda isotope, line: manifold)
    model = ProjectedProbe(manifold, optical_hwhm, transit)
    intensity = np.array([0.1, 24.0, 1400.0])
    detuning = 2.0 * np.pi * np.array([-290e6, -12e6, 0.0, 6e6, 340e6])
    area = _line_areas(manifold)[0]
    rate_center = intensity * area / (np.pi * optical_hwhm * constants.PLANCK * manifold.nu0)
    broadening = np.sqrt(1.0 + 2 * rate_center / (gamma + transit))
    saturated_hwhm = optical_hwhm * broadening

    lorentz = optical_hwhm / (
        np.pi * (detuning[None, :] ** 2 + saturated_hwhm[:, None] ** 2)
    )
    actual = model.absorption_cross_section(intensity, detuning, optical_hwhm * 1e-5)
    np.testing.assert_allclose(actual, area * lorentz, rtol=2e-9, atol=0.0)

    doppler_sigma = 2.0 * np.pi * 210e6
    z = ((detuning[None, :] + 1j * saturated_hwhm[:, None])
         / (np.sqrt(2.0) * doppler_sigma))
    voigt = wofz(z).real / (np.sqrt(2.0 * np.pi) * doppler_sigma * broadening[:, None])
    actual = model.absorption_cross_section(intensity, detuning, doppler_sigma)
    np.testing.assert_allclose(actual, area * voigt, rtol=3e-8, atol=0.0)


def _od_conditions(scheme, **overrides):
    params = dict(
        scheme.defaults(), species=species.SPECIES_ORDER[2], line="D1",
        pump_power_mw=0.0, probe_power_uw=0.0, probe_waist_mm=1.0, temp_c=43.0,
        cell_mm=20.0, transit_khz=150.0, scan_points=101,
    )
    return dict(params, **overrides)


def test_advanced_zero_power_matches_its_own_transit_broadened_linear_od():
    scheme = SASScheme()
    params = _od_conditions(scheme)
    assert scheme.extra_view_unavailable("Finite-probe pumping reference", params) is None

    result = scheme.projected_probe_reference(params)
    assert result["status"] == "converged"
    assert result["refinement"]["levels"] == 0

    isotope = species.RB87
    manifold = species.build_manifold(isotope, params["line"])
    temperature = params["temp_c"] + 273.15
    density = species.number_density(isotope, temperature)
    transit = 2.0 * np.pi * params["transit_khz"] * 1e3
    optical_hwhm = 0.5 * (manifold.gamma + species.self_broadening(isotope, density)) + transit
    sigma = manifold.k_vec * np.sqrt(constants.KB * temperature / isotope.mass)
    x_ghz = np.asarray(result["x_ghz"])
    assert np.all(np.diff(x_ghz) > 0.0)
    cross_section = _weak_cross_section(manifold, x_ghz * 2.0 * np.pi * 1e9, optical_hwhm, sigma)
    tau = density * cross_section * params["cell_mm"] * 1e-3 * params["line_strength"]
    np.testing.assert_allclose(result["tau"], tau, rtol=3e-9, atol=0.0)
    np.testing.assert_allclose(result["transmission"], np.exp(-tau), rtol=3e-9, atol=0.0)
    np.testing.assert_array_equal(result["transmission"], result["linear_transmission"])


def test_advanced_finite_probe_propagates_passively_with_refinement():
    scheme = SASScheme()
    params = _od_conditions(scheme, probe_power_uw=5.0)
    result = scheme.projected_probe_reference(params)
    assert result["status"] == "converged", result.get("notes")
    x = np.asarray(result["x_ghz"])
    transmission = np.asarray(result["transmission"])
    tau = np.asarray(result["tau"])
    linear = np.asarray(result["linear_transmission"])
    assert transmission.shape == tau.shape == x.shape == linear.shape
    assert np.all(np.diff(x) > 0.0)
    assert np.all(np.isfinite(tau) & (tau >= 0.0))
    assert np.all((transmission >= 0.0) & (transmission <= 1.0))
    np.testing.assert_allclose(np.exp(-tau), transmission, rtol=3e-12, atol=1e-14)
    assert np.max(np.abs(transmission - linear)) > 1e-4
    for key in ("max_abs_transmission_delta", "max_scaled_tau_delta",
                "tail_max_relative_alpha_delta"):
        assert np.isfinite(result["refinement"][key])
        assert result["refinement"][key] >= 0.0
    assert result["refinement"]["levels"] >= 2
    assert result["provenance"]["isotope_fractions"] == {species.RB87.name: 1.0}
    assert "numerical readiness" in result["provenance"]["validation"].lower()

    legacy = scheme.compute(params)
    assert legacy["probe_model"] == "finite_power_2level_gaussian_propagation"


@pytest.mark.parametrize("unsupported", [
    {"pump_power_mw": 0.01},
    {"paraffin_coated": True},
    {"ne_pressure_torr": 1.0},
    {"species": species.SPECIES_ORDER[-1]},
])
def test_advanced_reference_rejects_out_of_scope_conditions(unsupported):
    scheme = SASScheme()
    params = _od_conditions(scheme, **unsupported)
    assert scheme.extra_view_unavailable("Finite-probe pumping reference", params)
    with pytest.raises(ValueError):
        scheme.projected_probe_reference(params)


def test_advanced_reference_rejects_beam_path_colder_than_cold_spot():
    scheme = SASScheme()
    params = _od_conditions(scheme, temp_c=150., doppler_temp_c=20., constrain_doppler_temp=False)
    assert "beam-path temperature" in scheme.extra_view_unavailable(
        "Finite-probe pumping reference", params)
    with pytest.raises(ValueError, match="at or above the cold-spot temperature"):
        scheme.projected_probe_reference(params)
    assert scheme.compute(params)["probe_model"] == "weak_probe_legacy"


def test_advanced_line_strength_is_reported_as_opacity_calibration():
    scheme = SASScheme()
    params = _od_conditions(scheme, species="Rb (natural)", temp_c=40.,
                            probe_power_uw=10., transit_khz=20., scan_points=401)
    scaled = scheme.projected_probe_reference(dict(params, line_strength=.5, cell_mm=10.))
    short = scheme.projected_probe_reference(dict(params, line_strength=1., cell_mm=5.))
    assert scaled["status"] == short["status"] == "converged"
    assert scaled["provenance"]["line_strength"] == {
        "factor": .5, "applies_to": "propagated absorption coefficient"}
    assert any("Optical drive and level populations keep their calculated values" in note
               for note in scaled["notes"])
    np.testing.assert_allclose(scaled["transmission"], short["transmission"], rtol=0., atol=2e-15)


def test_advanced_reference_withholds_finite_curve_on_deadline(monkeypatch):
    from gabes.schemes import sas_probe_reference

    scheme = SASScheme()
    monkeypatch.setattr(sas_probe_reference, "MAX_SECONDS", 0.0)
    result = scheme.projected_probe_reference(_od_conditions(scheme, probe_power_uw=5.0))
    assert result["status"] == "unresolved"
    assert "limit was reached" in result["reason"]
    assert "transmission" not in result and "tau" not in result
