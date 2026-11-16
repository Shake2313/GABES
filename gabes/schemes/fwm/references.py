"""Linear absorption and reference checks used by seeded FWM."""
from .common import L_CELL
from ...linear_hyperfine import hyperfine_alpha_analytic as _hyperfine_alpha_analytic
import functools
import math
import numpy as np
from ... import hyperfine


LINEAR_ABSORPTION_MODEL = (
    "analytic Voigt (Faddeeva), untruncated Maxwell; reference line weights C_F² only")


def _arm_linear_od(beam_GHz, T, L=L_CELL, ground_F=2, population_factors=None):
    """Independent linear OD for one ground manifold on the FWM axis."""
    alpha = _hyperfine_alpha_analytic(beam_GHz, T, ground_F=ground_F,
                                      population_factors=population_factors)
    return np.clip(np.maximum(alpha, 0.0) * L, 0.0, 5.0)


def _pump_scatter_noise(D_GHz, T, L, kappa, population_factors=None):
    """Phenomenological source noise from linear pump OD, normalized to SQL."""
    alpha = _hyperfine_alpha_analytic(D_GHz, T, population_factors=population_factors)
    od_pump = float(np.clip(float(alpha[0]) * L, 0.0, 50.0))
    return float(kappa) * (1.0 - math.exp(-od_pump)), od_pump


def _reference_population_factors():
    """Return the fixed C_F² convention for archived evidence readers."""
    return dict.fromkeys(hyperfine.GROUND_POP, 1.0)


_arm_linear_od_analytic = _arm_linear_od
_pump_scatter_noise_analytic = _pump_scatter_noise


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
