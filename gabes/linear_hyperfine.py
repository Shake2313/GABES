"""85Rb D1 linear hyperfine absorption on the historical OD and FWM axes.

The OBE and analytic profiles preserve their established normalization and
velocity measures. The resolved OD/SAS manifold uses its own species model.
"""
import math

import numpy as np

from . import atoms, constants, doppler, hyperfine, observables
from .constants import GAMMA, K_VEC

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def hyperfine_alpha_obe(scan, params, line_strength, *, solve_chi, probe_rabi):
    """
    Realistic 85Rb D1 absorption coefficient α(scan) [1/m]: an incoherent sum of
    the four hyperfine transitions (Fg∈{2,3}→Fe∈{2,3}), each a Doppler-broadened
    Voigt placed at its line-center detuning, weighted by the validated
    Clebsch-Gordan strength C_F² and thermal sublevel population, with self-broadening.

    The Voigt *shape* is produced once by the OBE Doppler kernel (a genuine OBE
    solve — no analytic wofz) on a fine grid, normalised to unit area, then
    shifted/scaled per line. Since C_F² sums initial magnetic sublevels,
    the thermal population is included exactly once:
        ∫α_F dδ = π · k · C_F² · |d|² · N / (ε₀ℏ) / (2(2I+1)).

    Returns (alpha, components, info) where `components` maps (Fg,Fe)→α_F(scan)
    and `info` carries N, Γ_eff, σ_v, Doppler FWHM for the readout.
    """
    T = params["temp_c"] + 273.15
    doppler_on = params.get("doppler", "on") == "on"
    ls = (params.get("line_strength", 1.0)
          if line_strength is None else float(line_strength))

    N = hyperfine.number_density(T)
    buffer_gamma = constants.neon_buffer_broadening(params.get("ne_pressure_torr", 0.0))
    gamma_eff = hyperfine.self_broadened_gamma(N) + buffer_gamma
    atom = atoms.two_level(gamma=gamma_eff)

    def build_H0(_s):
        H = np.zeros((2, 2), dtype=complex)
        H[0, 1] = H[1, 0] = probe_rabi * GAMMA / 2
        return H

    sigma_v = np.sqrt(constants.KB * T / constants.MASS_85RB)
    dopp_fwhm = np.sqrt(8 * np.log(2)) * K_VEC * sigma_v
    width = dopp_fwhm if doppler_on else gamma_eff

    shifts = {k: 2 * np.pi * v for k, v in hyperfine.LINE_SHIFT_HZ.items()}
    smin, smax = min(shifts.values()), max(shifts.values())

    # One unit-area Voigt shape on a fine grid covering (scan − shift) ± wings.
    lo = scan.min() - smax - 12 * width
    hi = scan.max() - smin + 12 * width
    grid = np.linspace(lo, hi, int((hi - lo) / (width / 40.0)) + 2)
    chi = solve_chi(atom, build_H0, (1, 0), grid, params, h_dep=False)
    shape, _ = observables.absorption_coefficient(chi, K_VEC, 1.0, line_strength=1.0)
    S_unit = shape / _trapz(shape, grid)          # ∫ S_unit dδ = 1

    # Absolute per-line integrated absorption (AutoOD normalisation).
    K = (np.pi * K_VEC * hyperfine.DIPOLE_SQ * N
         / (constants.HBAR * constants.EPS_0) / hyperfine.N_GROUND_SUBLEVELS)

    alpha = np.zeros_like(scan)
    components = {}
    for (Fg, Fe) in hyperfine.TRANSITIONS:
        I_line = ls * K * hyperfine.CF2[(Fg, Fe)]
        line = I_line * np.interp(scan - shifts[(Fg, Fe)], grid, S_unit,
                                  left=0.0, right=0.0)
        alpha += line
        components[(Fg, Fe)] = line

    info = dict(N=N, gamma_eff=gamma_eff, buffer_gamma=buffer_gamma,
                sigma_v=sigma_v, dopp_fwhm=dopp_fwhm)
    return alpha, components, info


def analytic_hyperfine_line(beam_GHz, T, Fg, Fe):
    """Analytic Voigt absorption of one hyperfine line weighted by C_F² alone [1/m].

    Same line center, self-broadened Lorentzian and Doppler width as
    ``absorption._hyperfine_alpha`` (untruncated Maxwell, no table).
    """
    beam = np.atleast_1d(np.asarray(beam_GHz, dtype=float))
    density = hyperfine.number_density(T)
    gamma = hyperfine.self_broadened_gamma(density)
    sigma = K_VEC * math.sqrt(constants.KB * float(T) / constants.MASS_85RB)
    scale = (math.pi * K_VEC * hyperfine.DIPOLE_SQ * density
             / (constants.HBAR * constants.EPS_0) / hyperfine.N_GROUND_SUBLEVELS)
    detuning = 2 * np.pi * (beam * 1e9 + hyperfine.LINE_SHIFT_HZ[(2, 3)]
                            - hyperfine.LINE_SHIFT_HZ[(Fg, Fe)])
    return scale * hyperfine.CF2[(Fg, Fe)] * doppler.voigt_profile(detuning, gamma, sigma)


def hyperfine_alpha_analytic(beam_GHz, T, ground_F=None, population_factors=None):
    """Closed-form counterpart of ``absorption._hyperfine_alpha`` on the FWM axis.

    That path Doppler-averages a weak-probe two-level OBE (saturation ~2e-6)
    numerically, i.e. it tabulates a Voigt profile.  The analytic twin uses the
    same lines through the Faddeeva function.  ``population_factors`` maps each
    ground manifold to the extra weight the reference applies (see
    :func:`_reference_population_factors`); Ultra keeps the numerical table.
    """
    beam = np.atleast_1d(np.asarray(beam_GHz, dtype=float))
    alpha = np.zeros_like(beam)
    for (Fg, Fe) in hyperfine.TRANSITIONS:
        if ground_F is not None and Fg != ground_F:
            continue
        weight = 1.0 if population_factors is None else population_factors[Fg]
        alpha = alpha + weight * analytic_hyperfine_line(beam, T, Fg, Fe)
    return alpha
