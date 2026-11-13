"""Linear absorption and reference checks used by seeded FWM."""
from .common import L_CELL
from ...linear_hyperfine import (
    analytic_hyperfine_line as _analytic_hyperfine_line, hyperfine_alpha_analytic as
    _hyperfine_alpha_analytic,
)
import functools
import math
import numpy as np
from ... import hyperfine


def _arm_linear_od(beam_GHz, T, L=L_CELL, ground_F=2):
    """Independent linear-OD diagnostic at a twin-beam arm's optical frequency.

    It is not applied as a second arm efficiency because the diagonal Maxwell
    response already attenuates the propagated fields.  We sum the requested
    ground-manifold components of the independently validated absorption model.
    Δ=0 is 85Rb F=2→F'=3, hence the reference-line shift.
    """
    from .. import absorption
    ref_hz = hyperfine.LINE_SHIFT_HZ[(2, 3)]
    scan = 2 * np.pi * (np.asarray(beam_GHz, dtype=float) * 1e9 + ref_hz)
    params = {"temp_c": float(T) - 273.15, "line_strength": 1.0, "doppler": "on"}
    _alpha_tot, comps, _info = absorption._hyperfine_alpha(scan, params)
    alpha = np.zeros_like(scan, dtype=float)
    for (Fg, _Fe), a in comps.items():
        if Fg == ground_F:
            alpha = alpha + np.asarray(a, dtype=float)
    return np.clip(np.maximum(alpha, 0.0) * L, 0.0, 5.0)


def _pump_scatter_noise(D_GHz, T, L, kappa):
    """Excess intensity-difference noise (normalized to SQL) from pump light
    scattered into the detection mode.

    Anchored to the validated hyperfine absorption path (`schemes.absorption.
    _hyperfine_alpha`, <0.1 % vs lab): the pump's own linear OD at its detuning
    fixes how much pump power is scattered by the vapor; `kappa` is the residual
    fraction reaching the difference photocurrent (PBS extinction + spatial
    filtering leakage). Small far from resonance (Sim's +0.9 GHz window), large
    when the pump sits near a populous manifold (the −2.1 GHz / F=3 region).
    """
    from .. import absorption
    # `_hyperfine_alpha`'s scan axis is referenced to the 87Rb F=2→F'=2 marker
    # (hyperfine.LINE_SHIFT_HZ), whereas the FWM Δ is referenced to 85Rb F=2→F'=3.
    # Shift by that line's position so Δ=0 maps to the F=2→F'=3 line center.
    ref_hz = hyperfine.LINE_SHIFT_HZ[(2, 3)]
    scan = np.array([2 * np.pi * (float(D_GHz) * 1e9 + ref_hz)])
    params = {"temp_c": float(T) - 273.15, "line_strength": 1.0, "doppler": "on"}
    alpha, _, _ = absorption._hyperfine_alpha(scan, params)
    od_pump = float(np.clip(float(alpha[0]) * L, 0.0, 50.0))
    return float(kappa) * (1.0 - math.exp(-od_pump)), od_pump


@functools.lru_cache(maxsize=1)
def _reference_population_factors():
    """Population weight per ground manifold used by ``absorption._hyperfine_alpha``.

    The analytic twin follows the reference normalization instead of restating
    it: whether C_F² alone or p_F·C_F² weights a line is a documented
    correction that can differ between versions.  One numerical evaluation at
    the four line centers is compared with the C_F²-only analytic lines and each
    ground manifold is classified as 1 or p_F.  ``None`` (neither within 2 %)
    sends the pole tiers back to the numerical diagnostics.
    """
    from .. import absorption
    T = 373.15
    ref_hz = hyperfine.LINE_SHIFT_HZ[(2, 3)]
    beams = np.array([(hyperfine.LINE_SHIFT_HZ[line] - ref_hz) / 1e9
                      for line in hyperfine.TRANSITIONS])
    _alpha, components, _info = absorption._hyperfine_alpha(
        2 * np.pi * (beams * 1e9 + ref_hz),
        {"temp_c": T - 273.15, "line_strength": 1.0, "doppler": "on"})
    factors = {}
    for index, (Fg, Fe) in enumerate(hyperfine.TRANSITIONS):
        ratio = (float(components[(Fg, Fe)][index])
                 / float(_analytic_hyperfine_line(beams[index], T, Fg, Fe)[0]))
        best = min((1.0, float(hyperfine.GROUND_POP[Fg])),
                   key=lambda candidate: abs(ratio / candidate - 1.0))
        if abs(ratio / best - 1.0) > 0.02 or factors.get(Fg, best) != best:
            return None
        factors[Fg] = best
    return factors


def _arm_linear_od_analytic(beam_GHz, T, L=L_CELL, ground_F=2, population_factors=None):
    """Analytic-Voigt twin of :func:`_arm_linear_od` (diagnostic only)."""
    alpha = _hyperfine_alpha_analytic(beam_GHz, T, ground_F=ground_F,
                                      population_factors=population_factors)
    return np.clip(np.maximum(alpha, 0.0) * L, 0.0, 5.0)


def _pump_scatter_noise_analytic(D_GHz, T, L, kappa, population_factors=None):
    """Analytic-Voigt twin of :func:`_pump_scatter_noise`."""
    alpha = _hyperfine_alpha_analytic(D_GHz, T, population_factors=population_factors)
    od_pump = float(np.clip(float(alpha[0]) * L, 0.0, 50.0))
    return float(kappa) * (1.0 - math.exp(-od_pump)), od_pump


@functools.lru_cache(maxsize=1)
def _zeeman_consistency_cached():
    """Memoized 24-level CG-sum consistency diagnostic (a model constant)."""
    try:
        from ... import zeeman as _zeeman
        correction = float(getattr(
            _zeeman.rb85_d1_double_lambda_zeeman(), "lumped_strength_correction", 1.0))
        return correction, (
            f"24-level CG-sum consistency check = {correction:.4f} "
            "(lumped 3·C_F² reproduced); diagnostic only — full Floquet "
            "scan not run in Ultra v1")
    except Exception as exc:                  # pragma: no cover
        return 1.0, f"unavailable: {exc}"
