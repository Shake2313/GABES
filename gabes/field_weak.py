"""Linear optical response of thermal Rb atoms crossing a transverse Gaussian beam.

The incident density operator is uniform in the ground manifold. In the eigenbasis
of a static field each allowed transition therefore contributes an independent
optical coherence. All widths, laser offsets and Zeeman shifts are angular rates.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_laguerre

from . import beam, constants, linear_hyperfine, species, zeeman
from ._arrays import readonly_float_array
from .ballistic_pumping import SDIRK_DIAGONAL, _axis, _geometry, _positive


def _sublevel_energies(states, zero_field, iso, j, gj, field_t):
    energy = np.array(zero_field, dtype=float) * species.MHZ
    for f in sorted({f for f, _ in states}):
        idx = [i for i, (level, _) in enumerate(states) if level == f]
        gf = species.lande_gf(gj, iso.g_I, iso.I, j, f)
        h_b = zeeman.zeeman_hamiltonian(f, gf, (0., 0., field_t))
        energy[idx] += h_b.diagonal().real
    return energy


@lru_cache(maxsize=32)
def _transitions(iso, line, field, jones):
    atom = species.level_structure(iso, line)
    b_ut = math.hypot(*field)
    axis = np.array(field) / b_ut if b_ut else np.array([0., 0., 1.])
    # In an orthonormal right-handed basis quantized along B, only one q
    # connects each magnetic-sublevel pair. Squaring each amplitude directly
    # keeps a nearly forbidden polarization component nonnegative.
    transverse = math.hypot(axis[0], axis[1])
    y_axis = np.array([-axis[1], axis[0], 0.]) / transverse if transverse else np.array([0., 1., 0.])
    x_axis = np.cross(y_axis, axis)
    electric = np.array([*jones, 0j])
    ex, ey = electric @ x_axis, electric @ y_axis
    weights = {0: abs(electric @ axis)**2, 1: abs(ex - 1j*ey)**2/2,
               -1: abs(ex + 1j*ey)**2/2}

    ground_levels = sorted({f for f, _ in atom.ground})
    ground_mhz = dict(zip(ground_levels, atom.ground_energy_mhz))
    ground = _sublevel_energies(atom.ground, [ground_mhz[f] for f, _ in atom.ground],
                                iso, iso.Jg, iso.gJ_S, b_ut*1e-6)
    je = iso.line(line)[0]
    gj = iso.gJ_P12 if line == "D1" else iso.gJ_P32
    excited = _sublevel_energies(atom.excited, atom.excited_energy_mhz, iso, je, gj, b_ut*1e-6)
    dipoles = sum(weights[q] * atom.dipoles[q]**2 for q in weights)
    g, e = np.nonzero(dipoles > np.finfo(float).eps**2 * dipoles.max())
    centers, idx = np.unique(excited[e]-ground[g], return_inverse=True)
    # h nu (Omega_reduced^2 / I) / (2 n_g) times signed-dipole magnitude squared.
    rabi_sq = atom.dipole_sq * beam.field_amplitude(1.)**2 / constants.HBAR**2
    coeff = constants.PLANCK * atom.frequency_hz * rabi_sq / (2*len(atom.ground))
    strengths = coeff * np.bincount(idx, weights=dipoles[g, e], minlength=len(centers))
    return readonly_float_array(centers, copy=False), readonly_float_array(strengths, copy=False)


@lru_cache(maxsize=8)
def _speed_rule(order):
    # u=q^2/2 turns the Rayleigh density q exp(-q²/2) dq into exp(-u) du.
    u, weights = roots_laguerre(order)
    return np.sqrt(2*u), weights


def _continuous(detunings, strengths, width, sigma_v, waist, sigma_z):
    # Power spectrum of exp[-(vt/w)²] is a normalized Gaussian of
    # angular standard deviation v/w. Its convolution with the independent
    # longitudinal Doppler shift is Gaussian at fixed transverse speed v.
    def response_at_speed(q):
        sigma = math.hypot(sigma_z, q * sigma_v / waist)
        profile = linear_hyperfine.line_absorption(detunings, np.pi, 2*width, sigma)
        return profile @ strengths

    last = None
    for order in (1, 2, 4, 8, 16, 32, 64, 128):
        speeds, weights = _speed_rule(order)
        result = np.zeros(detunings.shape[0])
        for q, weight in zip(speeds, weights):
            result += weight * response_at_speed(q)
        if last is not None and np.all(np.abs(result-last) <= 3e-11*result):
            return result
        last = result
    # Very narrow beams broaden even a natural-width line strongly; resolve
    # the low-speed boundary layer directly if fixed Laguerre orders disagree.
    scale = np.maximum(last, np.finfo(float).tiny)

    def integrand(q):
        return q * math.exp(-q*q/2) * response_at_speed(q) / scale

    result, error = quad_vec(integrand, 0., np.inf, epsabs=1e-11, epsrel=1e-11, norm="max")
    if error > 3e-10:
        raise ArithmeticError("Weak Gaussian flight speed integral did not converge.")
    return result * scale


def _discrete(detunings, strengths, width, sigma_v, waist, grid):
    # Stage-by-stage linearization of the same SDIRK optical coherence.
    # Its in-phase quadrature is drive work, hence includes residual
    # excited population at the end of the interval. At linear order the
    # impact dependence factors into exp(-2 b²).
    b, wb, q, wq, step, r1, r2, _ = _geometry(grid, ())
    dx = step[0]
    d = SDIRK_DIAGONAL
    # All impact paths use the same longitudinal abscissae on this grid.
    a1 = np.exp(b[0]**2 - r1[0]**2)
    a2 = np.exp(b[0]**2 - r2[0]**2)
    b_factor = (4./np.pi) * (wb @ np.exp(-2*b**2))
    z = width + 1j * detunings
    result = np.zeros(detunings.shape[0])
    for speed, weight in zip(q, wq*q*np.exp(-q*q/2.)):
        time_per_waist = waist / (sigma_v*speed)
        dt = 2. * grid.xcut / grid.nx * time_per_waist
        denom = 1. / (1. + d*dt*z)
        c = np.zeros_like(z)
        signal = np.zeros(z.shape)
        for length, first, second in zip(dx, a1, a2):
            stage = (c + d*dt*first) * denom
            c = (c + (1.-d)/d*(stage-c) + d*dt*second) * denom
            signal += length*((1.-d)*first*stage.real + d*second*c.real)
        result += weight * (signal @ strengths)
    return b_factor * result


def _validated_inputs(isotope, line, field_ut, jones):
    iso = species.ISOTOPES.get(isotope) if isinstance(isotope, str) else isotope
    if iso not in (species.RB85, species.RB87) or line not in ("D1", "D2"):
        raise ValueError("Field histories require Rb85 or Rb87 on D1 or D2.")
    field = np.asarray(field_ut, dtype=float)
    pol = np.asarray(jones, dtype=complex)
    if field.shape != (3,) or not np.isfinite(field).all():
        raise ValueError("Three finite Jones-frame field components (µT) are required.")
    if pol.shape != (2,) or not np.isfinite(pol).all() or not np.any(pol):
        raise ValueError("Two finite, nonzero-norm Jones components are required.")
    scale = max(np.max(np.abs(pol.real)), np.max(np.abs(pol.imag)))
    scaled = pol.real / scale + 1j * (pol.imag / scale)
    return iso, tuple(field), tuple(scaled/np.linalg.norm(scaled))


def transition_centers(isotope, line, field_ut, jones):
    """Driven magnetic transition centers in rad/s from this isotope's centroid."""
    iso, field, pol = _validated_inputs(isotope, line, field_ut, jones)
    return _transitions(iso, line, field, pol)[0]


def weak_cross_sections(isotope, line, width, field_ut, jones, detuning, *,
                        temperature_k, waist_m, thermal=False, grid=None):
    """Weak absorption per atom (m²), relative to the isotope's D-line centroid.

    With no grid, integrate entire Gaussian flights and the bulk transverse
    speed distribution. Set thermal=True to include the axial Maxwell average.
    A supplied trajectory grid gives the linear limit of the discrete, rest-frame
    SDIRK solver for subtracting that limit from finite-power calculations.
    """
    iso, field, pol = _validated_inputs(isotope, line, field_ut, jones)
    width = _positive(width, "Optical HWHM (rad/s)")
    if width < species.level_structure(iso, line).gamma / 2:
        raise ValueError("Intrinsic optical width is below the natural HWHM.")
    temperature_k = _positive(temperature_k, "Kinetic temperature (K)")
    waist_m = _positive(waist_m, "Gaussian waist (m)")
    axis = _axis(detuning, "Detuning (rad/s)")
    centers, strengths = _transitions(iso, line, field, pol)
    offsets = axis[:, None] - centers
    sigma_v = constants.thermal_speed(temperature_k, iso.mass)
    if grid is not None:
        if thermal:
            raise ValueError("Discrete weak subtraction requires a rest-frame detuning.")
        return _discrete(offsets, strengths, width, sigma_v, waist_m, grid)
    sigma_z = species.build_manifold(iso, line).k_vec * sigma_v if thermal else 0.
    return _continuous(offsets, strengths, width, sigma_v, waist_m, sigma_z)
