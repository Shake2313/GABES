"""Independent limits and convergence checks for coherent paraxial propagation."""
import numpy as np
import pytest
from scipy.optimize import brentq

from gabes.paraxial import (ParaxialGrid, RadialQDHTGrid, gaussian_field,
                            gaussian_radial_field, propagate,
                            propagate_axisymmetric, relative_lens_power, thin_lens)

WAVELENGTH = 795e-9


def test_vacuum_gaussian_matches_analytic_width_power_and_phase():
    waist, power = 180e-6, .004
    grid = ParaxialGrid.square(1.2e-3, 128)
    field = gaussian_field(grid, power, waist, WAVELENGTH)
    zr = np.pi*waist**2/WAVELENGTH
    result = propagate(field, grid, wavelength_m=WAVELENGTH,
                       length_m=zr, steps=12, snapshot_stride=5)
    expected = waist*np.sqrt(1+(result.z_m/zr)**2)
    np.testing.assert_allclose(result.radius_m, expected, rtol=2e-9)
    np.testing.assert_allclose(result.power_w, power, rtol=5e-14)
    assert result.z_m[-1] == zr
    assert len(result.z_m) == 4
    analytic = np.sqrt(2*power/(np.pi*waist**2))/(1+1j)*np.exp(
        -grid.radius_squared_m2/(waist**2*(1+1j)))
    np.testing.assert_allclose(result.final_field, analytic, atol=2e-6)
    assert abs(result.power_balance_error_w) < 1e-15
    assert not result.final_field.flags.writeable


def test_uniform_complex_slab_matches_beer_lambert_and_phase():
    grid = ParaxialGrid.square(.001, 32)
    field = np.full(grid.shape, 3+2j)
    chi = 1.7e-6+2.3e-6j
    length, k = .021, 2*np.pi/WAVELENGTH
    result = propagate(field, grid, wavelength_m=WAVELENGTH, length_m=length,
                       steps=7, susceptibility=lambda intensity: chi,
                       boundary_tolerance=None)
    np.testing.assert_allclose(result.final_field, field*np.exp(.5j*k*length*chi), rtol=2e-14)
    assert result.power_w[-1]/result.power_w[0] == pytest.approx(np.exp(-k*chi.imag*length))
    assert result.material_power_change_w < 0
    assert abs(result.power_balance_error_w) < 1e-16


def test_thin_lens_sign_and_curvature_match_independent_gaussian_formula():
    waist, focal, distance = 180e-6, .12, .08
    grid = ParaxialGrid.square(.0012, 128)
    field = gaussian_field(grid, .001, waist, WAVELENGTH)
    lens_field = thin_lens(field, grid, wavelength_m=WAVELENGTH, focal_length_m=focal)
    fit = relative_lens_power(lens_field, field, grid, wavelength_m=WAVELENGTH,
                              fit_radius_m=waist/2)
    assert fit['lens_power_m_inverse'] == pytest.approx(1/focal, rel=1e-12)
    assert fit['phase_rms_rad'] < 1e-13
    curved = gaussian_field(grid, .001, waist, WAVELENGTH, curvature_radius_m=-focal)
    np.testing.assert_allclose(curved, lens_field, atol=1e-12)
    result = propagate(lens_field, grid, wavelength_m=WAVELENGTH, length_m=distance, steps=16)
    zr = np.pi*waist**2/WAVELENGTH
    expected = waist*np.sqrt((1-distance/focal)**2+(distance/zr)**2)
    assert result.radius_m[-1] == pytest.approx(expected, rel=1e-8)
    vacuum_width = waist*np.sqrt(1+(distance/zr)**2)
    assert result.radius_m[-1] < vacuum_width


def test_saturable_absorption_midpoint_is_second_order_against_exact_implicit_law():
    grid = ParaxialGrid.square(.001, 16)
    intensity, isat, alpha0, length = 4., 1., 30., .03
    field = np.full(grid.shape, np.sqrt(intensity), complex)
    k = 2*np.pi/WAVELENGTH
    exact = brentq(lambda out: out+isat*np.log(out)-intensity-isat*np.log(intensity)
                   +alpha0*isat*length, 1e-10, intensity)
    errors = []
    for steps in (4, 8, 16):
        result = propagate(field, grid, wavelength_m=WAVELENGTH, length_m=length,
                           steps=steps, susceptibility=lambda value: 1j*alpha0/k/(1+value/isat),
                           boundary_tolerance=None)
        errors.append(abs(abs(result.final_field[0, 0])**2-exact))
        assert abs(result.power_balance_error_w) < 1e-17
    assert 3.7 < errors[0]/errors[1] < 4.3
    assert 3.7 < errors[1]/errors[2] < 4.3


def test_two_level_saturation_blue_focuses_red_defocuses_against_absorption_control():
    grid = ParaxialGrid.square(.0015, 128)
    waist, power, delta, scale = 180e-6, .001, 10., 2e-4
    field = gaussian_field(grid, power, waist, WAVELENGTH)
    isat = (2*power/(np.pi*waist**2))/(1+delta**2)
    outputs = []
    for sign in (0, 1, -1):
        # Delta = omega_laser - omega_atom. Positive Delta gives negative Re chi.
        susceptibility = lambda intensity, sign=sign: scale*(-sign*delta+1j)/(1+delta**2+intensity/isat)
        cell = propagate(field, grid, wavelength_m=WAVELENGTH, length_m=.012,
                         steps=80, susceptibility=susceptibility)
        screen = propagate(cell.final_field, grid, wavelength_m=WAVELENGTH,
                           length_m=.10, steps=10)
        outputs.append((cell, screen))
    control, blue, red = outputs
    assert blue[1].radius_m[-1] < control[1].radius_m[-1] < red[1].radius_m[-1]
    for pair, expected_sign in ((blue, 1), (red, -1)):
        lens = relative_lens_power(pair[0].final_field, control[0].final_field, grid,
                                   wavelength_m=WAVELENGTH, fit_radius_m=waist/2)
        assert expected_sign*lens['lens_power_m_inverse'] > 0
        assert pair[0].power_w[-1] < pair[0].power_w[0]
        assert abs(pair[0].power_balance_error_w)/power < 1e-12


def test_boundary_guard_rejects_truncated_gaussian_without_silent_renormalization():
    grid = ParaxialGrid.square(180e-6, 32)
    field = gaussian_field(grid, .001, 180e-6, WAVELENGTH)
    with pytest.raises(ValueError, match='enlarge the grid window'):
        propagate(field, grid, wavelength_m=WAVELENGTH, length_m=.01, steps=2)
    result = propagate(field, grid, wavelength_m=WAVELENGTH, length_m=.01,
                       steps=2, boundary_tolerance=None)
    assert result.power_w[0] < .001
    assert result.max_boundary_power_fraction > .01


def test_zero_distance_preserves_field_and_rejects_invalid_inputs():
    grid = ParaxialGrid.square(.001, 16)
    field = gaussian_field(grid, .001, 180e-6, WAVELENGTH)
    result = propagate(field, grid, wavelength_m=WAVELENGTH, length_m=0, steps=1)
    np.testing.assert_array_equal(result.final_field, field)
    np.testing.assert_array_equal(result.z_m, [0])
    with pytest.raises(ValueError, match='shape'):
        propagate(field[:2], grid, wavelength_m=WAVELENGTH, length_m=.01, steps=1)
    with pytest.raises(ValueError, match='nonfinite'):
        propagate(field, grid, wavelength_m=WAVELENGTH, length_m=.01, steps=1,
                  susceptibility=lambda intensity: np.nan)
    with pytest.raises(ValueError, match='broadcast'):
        propagate(field, grid, wavelength_m=WAVELENGTH, length_m=.01, steps=1,
                  susceptibility=lambda intensity: np.ones(3))
    with pytest.raises(ValueError, match='even'):
        ParaxialGrid.square(.001, 17)


def test_diffracting_saturable_medium_has_second_order_step_convergence():
    grid = ParaxialGrid.square(.0015, 128)
    field = gaussian_field(grid, .001, 180e-6, WAVELENGTH)
    isat = np.max(abs(field)**2)/101
    fields = [propagate(field, grid, wavelength_m=WAVELENGTH, length_m=.012,
                        steps=steps, susceptibility=lambda i: 2e-4*(-10+1j)/(101+i/isat)).final_field
              for steps in (20, 40, 80)]
    errors = [np.linalg.norm(fields[i]-fields[i+1])/np.linalg.norm(fields[i+1]) for i in (0, 1)]
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert errors[1] < 1e-6


def test_qdht_vacuum_gaussian_exact_on_axis_and_2d_reconstruction():
    waist, power = 180e-6, .004
    rgrid = RadialQDHTGrid(1.2e-3, 128)
    rfield = gaussian_radial_field(rgrid, power, waist, WAVELENGTH)
    zr = np.pi * waist**2 / WAVELENGTH
    res = propagate_axisymmetric(rfield, rgrid, wavelength_m=WAVELENGTH, length_m=zr, steps=12, snapshot_stride=4)
    expected_w = waist * np.sqrt(1 + (res.z_m / zr)**2)
    np.testing.assert_allclose(res.radius_m, expected_w, rtol=1e-10)
    np.testing.assert_allclose(res.power_w, power, rtol=1e-13)
    assert abs(res.power_balance_error_w) / power < 1e-13
    # Exact r=0 Hankel intensity vs analytic 2*P/(pi*w(z)^2)
    expected_i0 = 2 * power / (np.pi * expected_w**2)
    np.testing.assert_allclose(res.on_axis_intensity_w_m2, expected_i0, rtol=1e-10)
    # On-demand 2D reconstruction inside aperture matches analytic Gaussian
    cart = ParaxialGrid.square(1.2e-3, 64)
    field_2d = res.to_2d_field(cart)
    analytic = np.sqrt(2 * power / (np.pi * waist**2)) / (1 + 1j) * np.exp(
        -cart.radius_squared_m2 / (waist**2 * (1 + 1j)))
    inside = cart.radius_squared_m2 <= (0.9 * 1.2e-3)**2
    np.testing.assert_allclose(field_2d[inside], analytic[inside], atol=2e-6)
    # Circular Dirichlet: outside max_radius_m evaluates to 0
    assert np.all(field_2d[cart.radius_squared_m2 > 1.2e-3**2] == 0.0)


def test_qdht_saturable_cell_matches_2d_fft_power_balance_and_radius():
    waist, power, delta, scale = 180e-6, .001, 10., 2e-4
    R = 1.5e-3
    isat = (2 * power / (np.pi * waist**2)) / (1 + delta**2)
    chi = lambda intensity: scale * (-delta + 1j) / (1 + delta**2 + intensity / isat)
    grid = ParaxialGrid.square(R, 128)
    fft_res = propagate(gaussian_field(grid, power, waist, WAVELENGTH), grid,
                        wavelength_m=WAVELENGTH, length_m=.012, steps=80, susceptibility=chi)
    rgrid = RadialQDHTGrid(R, 128)
    qdht_res = propagate_axisymmetric(gaussian_radial_field(rgrid, power, waist, WAVELENGTH), rgrid,
                                      wavelength_m=WAVELENGTH, length_m=.012, steps=80, susceptibility=chi)
    assert abs(qdht_res.power_balance_error_w) / power < 1e-12
    assert qdht_res.power_w[-1] == pytest.approx(fft_res.power_w[-1], rel=1e-10)
    assert qdht_res.radius_m[-1] == pytest.approx(fft_res.radius_m[-1], rel=1e-10)
    # Boundary guard on small aperture
    small = RadialQDHTGrid(180e-6, 32)
    with pytest.raises(ValueError, match='enlarge the grid window'):
        propagate_axisymmetric(gaussian_radial_field(small, .001, 180e-6, WAVELENGTH), small,
                               wavelength_m=WAVELENGTH, length_m=.01, steps=2)
