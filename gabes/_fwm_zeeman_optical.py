"""Exact optical block elimination for the secular Zeeman weak response."""
from dataclasses import dataclass

import numpy as np

from .kernels import NUMBA_AVAILABLE, njit, prange


@dataclass(frozen=True)
class OpticalBlocks:
    retained: np.ndarray
    optical: np.ndarray
    to_retained: np.ndarray
    to_optical: np.ndarray
    retained_doppler: np.ndarray
    optical_doppler: np.ndarray
    readout: np.ndarray
    schur_indices: np.ndarray
    schur_values: np.ndarray
    off_diagonal_norm_sq: float


def prepare_blocks(base, doppler, readout, optical):
    """Build an exact Schur complement recipe when the optical block is diagonal."""
    retained = ~optical
    optical_block = base[np.ix_(optical, optical)]
    diagonal = optical_block.diagonal().copy()
    if not np.array_equal(optical_block, np.diag(diagonal)) or np.any(readout[:, retained]):
        return None
    to_retained = base[np.ix_(retained, optical)]
    to_optical = base[np.ix_(optical, retained)]
    products = to_retained[:, :, None]*to_optical[None, :, :]
    entries = np.argwhere(products != 0)
    base_off_diagonal = base.copy()
    np.fill_diagonal(base_off_diagonal, 0.)
    return OpticalBlocks(
        base[np.ix_(retained, retained)], diagonal, to_retained, to_optical,
        doppler[retained], doppler[optical], np.ascontiguousarray(readout[:, optical]),
        np.ascontiguousarray(entries[:, [0, 2, 1]]), products[tuple(entries.T)],
        float(np.sum(abs(base_off_diagonal)**2)))


@njit(cache=True)
def _solve(retained, optical, to_retained, to_optical, retained_doppler,
           optical_doppler, readout, schur_indices, schur_values, off_diagonal_norm_sq,
           rhs_retained, rhs_optical, effective_detuning, relative_frequency):
    nv = len(effective_detuning)
    chi = np.empty((nv, 2, 2), np.complex128)
    residual = np.zeros(nv)
    n = retained.shape[0]
    no = len(optical)
    for v in range(nv):
        shift = 1j*relative_frequency[v]
        d = effective_detuning[v]
        optical_diag = optical - d*optical_doppler + shift
        inverse = 1./optical_diag
        retained_shift = -d*retained_doppler + shift
        schur = retained.copy()
        for i in range(n):
            schur[i, i] += retained_shift[i]
        for k in range(len(schur_values)):
            i, j, o = schur_indices[k]
            schur[i, j] -= schur_values[k]*inverse[o]

        driven_optical = inverse.reshape((no, 1))*rhs_optical[v]
        reduced_rhs = rhs_retained[v] - to_retained @ driven_optical
        scale = np.empty((n, 1))
        for i in range(n):
            scale[i, 0] = np.max(np.abs(schur[i]))
        x = np.linalg.solve(schur/scale, reduced_rhs/scale)
        y = driven_optical - inverse.reshape((no, 1))*(to_optical @ x)
        chi[v] = 2.*(readout @ y)

        # Measure backward error in the original paired block.
        r_retained = retained @ x + retained_shift.reshape((n, 1))*x + to_retained @ y - rhs_retained[v]
        r_optical = to_optical @ x + optical_diag.reshape((no, 1))*y - rhs_optical[v]
        matrix_norm_sq = off_diagonal_norm_sq + np.sum(np.abs(optical_diag)**2)
        for i in range(n):
            matrix_norm_sq += abs(retained[i, i] + retained_shift[i])**2
        solution_norm = np.sqrt(np.sum(np.abs(x)**2) + np.sum(np.abs(y)**2))
        drive_norm = np.sqrt(np.sum(np.abs(rhs_retained[v])**2) + np.sum(np.abs(rhs_optical[v])**2))
        defect = np.sqrt(np.sum(np.abs(r_retained)**2) + np.sum(np.abs(r_optical)**2))
        residual[v] = defect/max(np.sqrt(matrix_norm_sq)*solution_norm+drive_norm, 1e-300)
    return chi, residual



@njit(cache=True, parallel=True)
def _solve_sweep(retained, optical, to_retained, to_optical, retained_doppler,
                 optical_doppler, readout, schur_indices, schur_values, off_diagonal_norm_sq,
                 rhs_retained, rhs_optical, effective_detuning, frequency_scale, weights,
                 frequencies, return_velocity_chi):
    count = len(frequencies)
    avg = np.zeros((count, 2, 2), np.complex128)
    raw = np.empty((count if return_velocity_chi else 0, len(weights), 2, 2), np.complex128)
    residual = np.zeros(count)
    for i in prange(count):
        chi, defect = _solve(retained, optical, to_retained, to_optical, retained_doppler,
                            optical_doppler, readout, schur_indices, schur_values,
                            off_diagonal_norm_sq, rhs_retained, rhs_optical, effective_detuning,
                            frequencies[i]*frequency_scale)
        for v in range(len(weights)):
            avg[i] += weights[v]*chi[v]
        if return_velocity_chi:
            raw[i] = chi
        residual[i] = np.max(defect)
    return avg, raw, np.max(residual)


def evaluate(blocks, rhs_retained, rhs_optical, effective_detuning, frequency_scale,
             weights, frequencies, return_velocity_chi):
    args = (blocks.retained, blocks.optical, blocks.to_retained, blocks.to_optical,
            blocks.retained_doppler, blocks.optical_doppler, blocks.readout, blocks.schur_indices,
            blocks.schur_values, blocks.off_diagonal_norm_sq, rhs_retained, rhs_optical,
            effective_detuning)
    flat = frequencies.ravel()
    # Avoid scheduling overhead on cursor and short diagnostic queries.
    if NUMBA_AVAILABLE and flat.size >= 32:
        output, raw, residual = _solve_sweep(
            *args, frequency_scale, weights, flat, return_velocity_chi)
    else:
        output = np.empty((len(flat), 2, 2), complex)
        raw = (np.empty((len(flat), len(weights), 2, 2), complex)
               if return_velocity_chi else None)
        residual = 0.
        for i, frequency in enumerate(flat):
            values, defect = _solve(*args, frequency*frequency_scale)
            output[i] = np.einsum("vij,v->ij", values, weights)
            residual = max(residual, float(np.max(defect)))
            if raw is not None:
                raw[i] = values
    return (output.reshape(frequencies.shape+(2, 2)),
            raw.reshape(frequencies.shape+(len(weights), 2, 2)) if return_velocity_chi else None,
            float(residual))
