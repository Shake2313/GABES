"""Weak-probe line profiles with explicit line areas and frequency axes."""

import math

import numpy as np

from . import constants, doppler, hyperfine, species
from .constants import K_VEC


def line_absorption(detuning, integrated_alpha, homogeneous_fwhm, sigma=0.0):
    """Absorption [1/m] for a line of known area [rad/s/m].

    Detuning, homogeneous FWHM and Gaussian sigma use angular frequency.
    Zero sigma gives the homogeneous Lorentzian limit.
    """
    if sigma == 0.0:
        hwhm = 0.5 * homogeneous_fwhm
        detuning = np.asarray(detuning, dtype=float)
        profile = hwhm / (math.pi * (detuning**2 + hwhm**2))
    else:
        profile = doppler.voigt_profile(detuning, homogeneous_fwhm, sigma)
    return integrated_alpha * profile


def rb85_d1_alpha_faddeeva(scan, T, *, line_strength=1.0, doppler_on=True,
                           buffer_gamma=0.0, shift_rad_s=0.0):
    """Four Rb-85 D1 lines referenced to the Rb-87 F=2 to F'=2 marker.

    The historical OD/FWM line areas already include thermal population
    through the population per magnetic sublevel. Keep that absolute scale
    while evaluating the profile with the exact Maxwell average.
    """
    scan = np.asarray(scan, dtype=float)
    density = hyperfine.number_density(T)
    gamma = hyperfine.self_broadened_gamma(density) + buffer_gamma
    sigma_v = np.sqrt(constants.KB * T / constants.MASS_85RB)
    sigma = K_VEC * sigma_v if doppler_on else 0.0
    area_scale = (math.pi * K_VEC * hyperfine.DIPOLE_SQ * density
                  / (constants.HBAR * constants.EPS_0) / hyperfine.N_GROUND_SUBLEVELS)
    alpha = np.zeros_like(scan)
    components = {}
    for line in hyperfine.TRANSITIONS:
        center = 2.0 * np.pi * hyperfine.LINE_SHIFT_HZ[line] + shift_rad_s
        area = line_strength * area_scale * hyperfine.CF2[line]
        component = line_absorption(scan - center, area, gamma, sigma)
        alpha += component
        components[line] = component
    info = dict(N=density, gamma_eff=gamma, buffer_gamma=buffer_gamma,
                sigma_v=sigma_v, dopp_fwhm=np.sqrt(8 * np.log(2)) * K_VEC * sigma_v)
    return alpha, components, info


def _fwm_absorption_axis(beam_GHz):
    beam = np.atleast_1d(np.asarray(beam_GHz, dtype=float))
    return 2.0 * np.pi * (beam * 1e9 + hyperfine.LINE_SHIFT_HZ[(2, 3)])


def analytic_hyperfine_line(beam_GHz, T, Fg, Fe):
    """Rb-85 D1 component on the FWM frequency axis, in inverse metres."""
    _, lines, _ = rb85_d1_alpha_faddeeva(_fwm_absorption_axis(beam_GHz), T)
    return lines[Fg, Fe]


def hyperfine_alpha_analytic(beam_GHz, T, ground_F=None, population_factors=None):
    """Rb-85 D1 absorption on the FWM axis (zero at F=2 to F'=3).

    An optional ground-manifold filter gives one arm's lines. The historical
    `population_factors` argument is retained for explicit external callers.
    """
    total, lines, _ = rb85_d1_alpha_faddeeva(_fwm_absorption_axis(beam_GHz), T)
    if ground_F is None and population_factors is None:
        return total
    alpha = np.zeros_like(total)
    for (Fg, _), component in lines.items():
        if ground_F is None or Fg == ground_F:
            weight = 1.0 if population_factors is None else population_factors[Fg]
            alpha = alpha + weight * component
    return alpha


def species_complex_voigt(scan, manifold, density_m3, temperature_k,
                          homogeneous_fwhm, *, offset=0.0):
    """Absorption and dispersion with an untruncated Maxwell velocity distribution.

    The scan, isotope centroid offset, and homogeneous FWHM are in rad/s.
    Thermal populations are already included in the integrated line strengths.
    The returned absorption is per metre and the real susceptibility is dimensionless.
    """
    scan = np.asarray(scan, dtype=float)
    sigma = manifold.k_vec * math.sqrt(constants.KB * temperature_k / manifold.iso.mass)
    scale = sigma * math.sqrt(2.0)
    normalization = sigma * math.sqrt(2.0 * math.pi)
    strengths = species.line_integrated_alpha(manifold.iso, manifold.line, density_m3)
    ground_count = len(manifold.Fg)
    alpha = np.zeros_like(scan)
    chi_real = np.zeros_like(scan)
    for center, g, e in zip(manifold.omega, manifold.g_idx, manifold.e_idx, strict=True):
        area = strengths[manifold.Fg[g], manifold.Fe[e - ground_count]]
        z = (scan - offset - center + 0.5j * homogeneous_fwhm) / scale
        profile = doppler.faddeeva_upper(z) / normalization
        alpha += area * profile.real
        chi_real -= (area / manifold.k_vec) * profile.imag
    return alpha, chi_real
