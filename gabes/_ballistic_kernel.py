"""Small-matrix SDIRK stages for transverse atomic trajectories."""

import math

import numpy as np

from .kernels import njit


SDIRK_DIAGONAL = 1.0 - 1.0 / math.sqrt(2.0)


@njit(cache=True)
def _stage(intensity, h, source, out, free_ge, drive_ge, drive_eg, free_ee,
           drive_ee, drive_gg, up_indices, up_counts, down_indices, down_counts,
           schur, rhs, inv_ee, coupling):
    """Solve (1-hL)p=source by eliminating the diagonal excited block exactly."""
    ng, ne = len(drive_gg), len(free_ee)
    for g in range(ng):
        rhs[g] = source[g]
        for j in range(ng):
            schur[g, j] = 0.0
        schur[g, g] = 1.0 - h * intensity * drive_gg[g]
    for e in range(ne):
        inv_ee[e] = 1.0 / (1.0 - h * (free_ee[e] + intensity * drive_ee[e]))
        for k in range(down_counts[e]):
            g = down_indices[e, k]
            coupling[e, k] = -h * intensity * drive_eg[e, g]
        for j in range(up_counts[e]):
            g = up_indices[e, j]
            factor = -h * (free_ge[e, g] + intensity * drive_ge[e, g]) * inv_ee[e]
            rhs[g] -= factor * source[ng + e]
            for k in range(down_counts[e]):
                schur[g, down_indices[e, k]] -= factor * coupling[e, k]
    # 1-hL, and hence its Schur complement, is a nonsingular M-matrix.
    for j in range(ng):
        pivot = schur[j, j]
        if not pivot > 0.0:
            raise ArithmeticError("Nonpositive ballistic population pivot.")
        for g in range(j + 1, ng):
            if schur[g, j] != 0.0:
                factor = schur[g, j] / pivot
                rhs[g] -= factor * rhs[j]
                for k in range(j + 1, ng):
                    schur[g, k] -= factor * schur[j, k]
    for i in range(ng - 1, -1, -1):
        value = rhs[i]
        for j in range(i + 1, ng):
            value -= schur[i, j] * out[j]
        out[i] = value / schur[i, i]
    for e in range(ne):
        value = source[ng + e]
        for k in range(down_counts[e]):
            value -= coupling[e, k] * out[down_indices[e, k]]
        out[ng + e] = value * inv_ee[e]


@njit(cache=True, nogil=True)
def integrate_native(first, second, step, bin_ids, seconds_per_unit, flux, peak,
                     thermal, observable, blocks, n_annuli, keep_history):
    """Integrate paths in time order; reduce the absorbed photon flux by annulus."""
    (free_ge, drive_ge, drive_eg, free_ee, drive_ee, drive_gg,
     up_indices, up_counts, down_indices, down_counts) = blocks
    ng, ne = len(drive_gg), len(free_ee)
    ns = len(thermal)
    schur = np.empty((ng, ng))
    rhs = np.empty(ng)
    inv_ee = np.empty(ne)
    coupling = np.empty((ne, ng))
    p = np.empty(ns)
    p1 = np.empty(ns)
    source = np.empty(ns)
    nsteps = first.shape[1]
    history = np.empty((nsteps + 1, ns)) if keep_history else np.empty((0, ns))
    absorbed = np.empty(nsteps) if keep_history else np.empty(0)
    photons = np.zeros(n_annuli)
    min_pop, max_trace = 1.0, 0.0
    s = SDIRK_DIAGONAL
    for b in range(len(first)):
        for v in range(len(seconds_per_unit)):
            p[:] = thermal
            if keep_history:
                history[0] = p
            for i in range(nsteps):
                dt = seconds_per_unit[v] * step[b, i]
                if dt == 0.0:
                    continue
                i1, i2 = peak * first[b, i], peak * second[b, i]
                h = s * dt
                _stage(i1, h, p, p1, free_ge, drive_ge, drive_eg, free_ee, drive_ee, drive_gg,
                       up_indices, up_counts, down_indices, down_counts, schur, rhs, inv_ee, coupling)
                for j in range(ns):
                    source[j] = p[j] + (1.0 - s) / s * (p1[j] - p[j])
                _stage(i2, h, source, p, free_ge, drive_ge, drive_eg, free_ee, drive_ee, drive_gg,
                       up_indices, up_counts, down_indices, down_counts, schur, rhs, inv_ee, coupling)
                obs1, obs2, trace, trace1 = 0.0, 0.0, 0.0, 0.0
                for j in range(ns):
                    obs1 += p1[j] * observable[j]
                    obs2 += p[j] * observable[j]
                    trace += p[j]
                    trace1 += p1[j]
                    min_pop = min(min_pop, p[j], p1[j])
                count = dt * ((1.0 - s) * i1 * obs1 + s * i2 * obs2)
                annulus = bin_ids[b, i]
                if annulus >= 0:
                    photons[annulus] += flux[b, v] * count
                max_trace = max(max_trace, abs(trace - 1.0), abs(trace1 - 1.0))
                if keep_history:
                    history[i + 1] = p
                    absorbed[i] = count
    return photons, history, absorbed, min_pop, max_trace
