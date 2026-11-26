"""Classical mean response preserves the full OBE and passive interpolation."""
import numpy as np
import pytest

from gabes import beam, constants, linear_hyperfine, magneto_multilevel, species
from gabes.nonlinear_response import ComplexSusceptibilityTable, solve_rb87_d1_mean
from gabes.psr_full_obe import solve_psr_pump


def test_mean_matches_finite_drive_obe_and_si_units_without_rf(monkeypatch):
    from gabes import psr_spectrum

    def forbidden(*args, **kwargs):
        raise AssertionError("Classical mean response must not calculate RF noise")

    monkeypatch.setattr(psr_spectrum, "psr_local_generator", forbidden)
    atom = magneto_multilevel.build_multilevel_atom("Rb87", "D1")
    velocities, weights = np.array([-31., 84.]), np.array([.3, .7])
    shifts = velocities * atom.frequency_hz / constants.C_LIGHT / 1e6
    monkeypatch.setattr(magneto_multilevel, "velocity_quadrature",
                        lambda *args, **kwargs: (shifts, weights, 200.))
    inputs = dict(temp_c=70., ground_relaxation_hz=1e5, density_m3=1e16)
    intensity = 125.
    result = solve_rb87_d1_mean(400., intensity, **inputs, velocity_order=2)
    states = [solve_psr_pump(400., intensity_mw_cm2=intensity / 10., velocity_m_s=v,
                             jones_xy=(1., 0.), **inputs) for v in velocities]
    expected_chi = sum(w * state["chi_probe"] for w, state in zip(weights, states))
    np.testing.assert_allclose(result["chi"], expected_chi, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(result["polarization_xy_c_m2"],
                               sum(w * s["polarization_xy_c_m2"] for w, s in zip(weights, states)),
                               rtol=1e-12, atol=1e-26)
    field = beam.field_amplitude(intensity)
    assert result["polarization_xy_c_m2"][0] == pytest.approx(
        constants.EPS_0 * field * result["chi"], rel=1e-12, abs=1e-26)
    assert abs(result["polarization_xy_c_m2"][1]) < 1e-24
    k = 2 * np.pi * atom.frequency_hz / constants.C_LIGHT
    assert result["alpha_per_m"] == pytest.approx(k * result["chi"].imag, rel=1e-12)
    assert result["class_count"] == 2
    assert result["conditions"]["intensity_w_m2"] == intensity
    assert result["conditions"]["intensity_mw_cm2"] == intensity / 10.
    assert result["conditions"]["reference_centroid_mhz"] == states[0]["conditions"]["reference_centroid_mhz"]
    assert "velocity_m_s" not in result["conditions"]
    assert "effective_detuning_mhz" not in result["conditions"]
    assert result["diagnostics"]["trace_error"] < 1e-9
    assert not result["convergence"]["assessed"]
    assert "rho" not in result and "generator_s_inverse" not in result


@pytest.mark.parametrize(("detuning", "intensity"), [(-700., 0.), (700., 1e-8)])
def test_thermal_weak_response_matches_independent_complex_voigt(detuning, intensity):
    density, temperature, reset = 1e16, 50., 1e5
    result = solve_rb87_d1_mean(detuning, intensity, temp_c=temperature,
                               ground_relaxation_hz=reset, density_m3=density,
                               velocity_order=6, cutoff_sigma=6.)
    man = species.build_manifold(species.RB87, "D1")
    absolute_axis = np.array([detuning + result["conditions"]["reference_centroid_mhz"]]) * species.MHZ
    alpha, real = linear_hyperfine.species_complex_voigt(
        absolute_axis, man, density, temperature + 273.15, man.gamma + 2 * reset)
    expected = real[0] + 1j * alpha[0] / man.k_vec
    np.testing.assert_allclose(result["chi"], expected, rtol=2e-6, atol=0.)
    assert result["chi"].imag > 0
    assert result["weight_sum"] == pytest.approx(1., abs=1e-15)
    assert result["omitted_maxwell_probability"] < 2e-9
    assert result["class_count"] > result["velocity_order"]


def test_zero_density_is_transparent_for_finite_drive(monkeypatch):
    monkeypatch.setattr(magneto_multilevel, "velocity_quadrature",
                        lambda *args, **kwargs: (np.array([0.]), np.array([1.]), 0.))
    result = solve_rb87_d1_mean(400., 125., density_m3=0.)
    assert result["chi"] == 0j and result["alpha_per_m"] == 0.
    np.testing.assert_array_equal(result["polarization_xy_c_m2"], np.zeros(2))


@pytest.mark.parametrize("kwargs", [
    dict(intensity_w_m2=-1.), dict(intensity_w_m2=float("nan")),
    dict(detuning_mhz=float("inf")), dict(temp_c=-273.15),
    dict(ground_relaxation_hz=0.), dict(density_m3=-1.),
    dict(velocity_order=1), dict(velocity_order=2.5), dict(velocity_order=True),
    dict(cutoff_sigma=0.),
])
def test_invalid_mean_inputs_rejected_before_any_atom_solve(monkeypatch, kwargs):
    from gabes import nonlinear_response

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid inputs must fail before atomic solves")

    monkeypatch.setattr(nonlinear_response, "solve_psr_pump", forbidden)
    inputs = dict(detuning_mhz=400., intensity_w_m2=1.)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        solve_rb87_d1_mean(**inputs)


def test_complex_table_preserves_nodes_shape_passivity_and_caller_arrays():
    intensities = np.array([0., .2, 1., 4., 16.])
    values = -2e-6 / (1 + intensities) + 1j * 1e-6 / (1 + .4 * intensities)
    table = ComplexSusceptibilityTable(intensities, values)
    np.testing.assert_allclose(table(intensities), values, rtol=1e-14, atol=0.)
    assert table(0.) == values[0]
    points = np.linspace(0., 16., 120).reshape(10, 12)
    result = table(points)
    assert result.shape == points.shape
    assert np.all(result.imag > 0)
    assert np.all(np.diff(result.imag.ravel()) <= 0)
    assert np.all(np.diff(result.real.ravel()) >= 0)
    assert table.max_intensity_w_m2 == 16.
    intensities[-1] = 100.
    values[0] = 5j
    assert table.max_intensity_w_m2 == 16.
    assert table(0.) == -2e-6 + 1e-6j
    assert not table.intensity_w_m2.flags.writeable
    assert not table.chi.flags.writeable


@pytest.mark.parametrize("points", [-1., float("nan"), [0., float("inf")], 1.00001])
def test_complex_table_never_extrapolates(points):
    table = ComplexSusceptibilityTable([0., 1.], [1e-6j, .5e-6j])
    with pytest.raises(ValueError):
        table(points)


@pytest.mark.parametrize(("intensities", "values"), [
    ([1., 2.], [1j, 2j]), ([0., -1.], [1j, 2j]),
    ([0., 0.], [1j, 2j]), ([0., 1.], [1j]),
    ([0., 1.], [1j, complex(float("nan"), 0.)]),
])
def test_complex_table_requires_explicit_finite_weak_node(intensities, values):
    with pytest.raises(ValueError):
        ComplexSusceptibilityTable(intensities, values)


@pytest.mark.parametrize(("isotope", "line"), [
    ("Rb87", "D1"), ("Rb85", "D1"), ("Rb87", "D2"), ("Rb85", "D2"),
])
def test_pole_maxwell_mean_matches_direct_velocity_quadrature(isotope, line):
    from gabes import nonlinear_response as nr
    from gabes.optical_bloch import solve_optical_pump

    detuning, intensity, temp_c, reset, density = 600., 5000., 90., 3e5, 1e16
    fast = nr._maxwell_mean(isotope, line, detuning, intensity, temp_c=temp_c,
                            ground_relaxation_hz=reset, density_m3=density,
                            velocity_order=3, cutoff_sigma=5., legacy=False)
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    centroid = magneto_multilevel.component_centroid_mhz(atom)
    shifts, weights, _ = magneto_multilevel.velocity_quadrature(
        atom, centroid + detuning, temp_c + 273.15, order=3, cutoff_sigma=5.)
    chi_ref = 0j
    exc_ref = 0.0
    for shift, weight in zip(shifts, weights, strict=True):
        vel = float(shift * 1e6 * constants.C_LIGHT / atom.frequency_hz)
        st = solve_optical_pump(isotope, line, detuning, intensity_mw_cm2=intensity / 10.,
                                velocity_m_s=vel, temp_c=temp_c, ground_relaxation_hz=reset,
                                density_m3=density, jones_xy=(1., 0.), solver="full",
                                include_atomic_state=False)
        chi_ref += weight * st["chi_probe"]
        exc_ref += weight * st["excited_fraction"]
    np.testing.assert_allclose(fast["chi"], chi_ref, rtol=1e-12, atol=1e-20)
    assert fast["excited_fraction"] == pytest.approx(exc_ref, rel=1e-12, abs=1e-14)
    assert fast["alpha_per_m"] == pytest.approx(
        2 * np.pi * atom.frequency_hz / constants.C_LIGHT * chi_ref.imag, rel=1e-12)


def test_pole_maxwell_mean_falls_back_cleanly_on_pole_failure(monkeypatch):
    from gabes import nonlinear_response as nr, pole_doppler

    ref = nr._maxwell_mean("Rb87", "D1", 400., 250., temp_c=70.,
                           ground_relaxation_hz=2e5, density_m3=1e16,
                           velocity_order=3, cutoff_sigma=5., legacy=False)

    def fail_residues(self, t):
        raise np.linalg.LinAlgError("forced pole failure")

    monkeypatch.setattr(pole_doppler.AffineShiftSystem, "residues", fail_residues)
    fallback = nr._maxwell_mean("Rb87", "D1", 400., 250., temp_c=70.,
                                ground_relaxation_hz=2e5, density_m3=1e16,
                                velocity_order=3, cutoff_sigma=5., legacy=False)
    np.testing.assert_allclose(fallback["chi"], ref["chi"], rtol=1e-12, atol=1e-20)
    assert fallback["excited_fraction"] == pytest.approx(ref["excited_fraction"], rel=1e-12, abs=1e-14)
