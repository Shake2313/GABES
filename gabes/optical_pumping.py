"""Stationary populations of an optically driven atom."""
import numpy as np

from . import core


def steady_populations(L0, deff_axis, S_v, n, chunk=1500):
    """Diagonal populations ρ_ii(Δ_eff) on a fine axis, in memory-safe chunks."""
    pops = np.empty((deff_axis.size, n))
    for s in range(0, deff_axis.size, chunk):
        sl = slice(s, s + chunk)
        rho = core.steady_state_batched(L0, deff_axis[sl], S_v, n)
        pops[sl] = np.einsum("vii->vi", rho).real
    return pops
