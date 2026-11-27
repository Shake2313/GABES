"""Fixed-step SDIRK in the ground/excited Zeeman eigenbases."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import linalg, sparse

from . import core, field_history, species
from .optical_bloch import _hamiltonian, _prepared_reservoirs


def _real_coordinates(matrix):
    return np.vstack((matrix.real, matrix.imag))


def _block_to_real(matrix):
    return np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])


def _energy_differences(energies):
    return (energies[:, None] - energies).ravel()


@dataclass(frozen=True)
class _DarkBlock:
    frequencies: np.ndarray
    basis: sparse.csc_matrix
    decay: float

    def inverse(self, shift):
        diagonal = 1 / (1 + shift * (self.decay + 1j * self.frequencies))
        return (self.basis.conj().T @ self.basis.multiply(diagonal[:, None])).real.toarray()


@dataclass(frozen=True)
class _Reduction:
    line: str
    ground: _DarkBlock
    excited: _DarkBlock
    optical_frequencies: np.ndarray
    optical_hwhm: float
    recycling: np.ndarray
    to_optical: np.ndarray
    from_optical: np.ndarray
    populations: np.ndarray
    source: np.ndarray
    loss: np.ndarray
    thermal: np.ndarray

    def dark_inverse(self, shift):
        ground = self.ground.inverse(shift)
        excited = self.excited.inverse(shift)
        feed = shift * (ground @ self.recycling @ excited)
        return np.block([[ground, feed], [np.zeros(feed.T.shape), excited]])


@lru_cache(maxsize=8)
def prepare(isotope, line, field, jones, quantization, optical_increment):
    atom, excitation, zeeman_g, zeeman_e = field_history._field_terms(isotope, line, field, jones, quantization)
    ng, ne = len(atom.ground), len(atom.excited)
    dark = _hamiltonian(atom, 0., 0., excitation)
    eg, ug = linalg.eigh(dark[:ng, :ng] + zeeman_g)
    ee, ue = linalg.eigh(dark[ng:, ng:] + zeeman_e)
    ug_h = sparse.csc_matrix(core.hermitian_basis(ng))
    ue_h = sparse.csc_matrix(core.hermitian_basis(ne))
    drive = -0.5 * field_history.rabi_per_sqrt_intensity(atom) * (ue.conj().T @ excitation @ ug)
    # Hermitian coordinates for gg and ee, and sqrt(2)*[Re(eg), Im(eg)] for optical coherences.
    root = np.sqrt(2.)
    to_g = (-1j * np.kron(drive, np.eye(ng))) @ ug_h
    to_e = (1j * np.kron(np.eye(ne), drive.T)) @ ue_h
    to_optical = root * _real_coordinates(np.hstack((to_g, to_e)))
    # The Hamiltonian commutator is antisymmetric in this real orthonormal basis.
    from_optical = -to_optical.T.copy()
    reservoirs, _ = _prepared_reservoirs(isotope, line, 0.)
    recycling = np.zeros((ng**2, ne**2), complex)
    damping = np.zeros((ne, ne), complex)
    for channel in reservoirs.channels:
        jump = channel.operator
        if np.any(jump[ng:]) or np.any(jump[:, :ng]):
            return None
        rotated = ug.conj().T @ jump[:ng, ng:] @ ue
        recycling += np.kron(rotated, rotated.conj())
        damping += rotated.conj().T @ rotated
    if not np.allclose(damping, atom.gamma * np.eye(ne), rtol=0., atol=2e-14 * atom.gamma):
        return None
    recycling = (ug_h.conj().T @ recycling @ ue_h).real
    # Recover the existing positivity diagnostic in the physical quantization basis.
    pg = (np.einsum("ij,ik->ijk", ug, ug.conj()).reshape(ng, -1) @ ug_h).real
    pe = (np.einsum("ij,ik->ijk", ue, ue.conj()).reshape(ne, -1) @ ue_h).real
    populations = linalg.block_diag(pg, pe)
    thermal = np.r_[np.full(ng, 1./ng), np.zeros(ne)]
    return _Reduction(line, _DarkBlock(_energy_differences(eg), ug_h, 0.),
                      _DarkBlock(_energy_differences(ee), ue_h, atom.gamma),
                      (ee[:, None] - eg).ravel(), atom.gamma / 2 + optical_increment,
                      recycling, to_optical, from_optical, populations,
                      to_optical[:, :ng] @ thermal[:ng],
                      from_optical[ng**2:ng**2+ne].sum(axis=0), thermal)


def _eigen_blocks(matrix):
    # Real LAPACK packs the two eigenvectors of a conjugate pair into adjacent real columns.
    geev = linalg.get_lapack_funcs("geev", (matrix,))
    real, imag, _, vectors, info = geev(matrix, compute_vl=False, compute_vr=True)
    if info or not np.isfinite(vectors).all():
        return None
    # Factor without scipy's singular-matrix warning. A singular basis is a normal fallback.
    lu, pivots, info = linalg.get_lapack_funcs("getrf", (vectors,))(vectors, overwrite_a=False)
    if info:
        return None
    rcond, info = linalg.get_lapack_funcs("gecon", (lu,))(lu, linalg.norm(vectors, 1))
    if info or not np.isfinite(rcond) or rcond < 1e-9:
        return None
    return real, imag, vectors, (lu, pivots)


def _apply_blocks(rhs, amplitude_squared, real, imag, pairs):
    diagonal = 1. - real[:, None] * amplitude_squared
    off = imag[pairs, None] * amplitude_squared
    first = pairs + 1
    d = diagonal[pairs]
    norm = d*d + off*off
    # Pair denominators use the full 2x2 blocks. Avoid dividing by their
    # individual diagonal entries, which can vanish in a regular block.
    diagonal[pairs] = diagonal[first] = 1.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = rhs / diagonal
        result[pairs] = (d*rhs[pairs] + off*rhs[first]) / norm
        result[first] = (d*rhs[first] - off*rhs[pairs]) / norm
    return result


def _within_residual(residual, *terms):
    scale = np.maximum.reduce([np.max(abs(term), axis=0) for term in terms])
    return np.all(np.max(abs(residual), axis=0) <= 3e-11 * np.maximum(scale, np.finfo(float).tiny))


def _neumann(rhs, amplitude_squared, operator, max_updates=10):
    """Check the reduced SDIRK equation per RHS, including at the update cap."""
    tolerance_squared = 1e-24 * np.einsum("ij,ij->j", rhs, rhs)
    if not np.isfinite(tolerance_squared).all():
        return None
    solution = rhs.copy()
    residual = amplitude_squared * (operator @ solution)
    for update in range(max_updates + 1):
        if np.all(np.einsum("ij,ij->j", residual, residual) <= tolerance_squared):
            return solution
        if update == max_updates:
            return None
        solution += residual
        residual = (rhs - solution) + amplitude_squared * (operator @ solution)


def _flight(model, first, second, roots, dt, stage, check):
    state_d = np.zeros((model.populations.shape[1], len(roots)))
    state_o = np.zeros((len(model.loss), len(roots)))
    removed = np.zeros(len(roots))
    d = field_history.SDIRK_DIAGONAL

    def checked(old_d, old_o, shape, audit):
        actual_d, actual_o = stage(old_d, old_o, shape, audit)
        pop = model.populations @ actual_d
        check(model.thermal[:, None] + roots * pop, 1. + roots * pop.sum(axis=0))
        return actual_d, actual_o

    try:
        for step, (a, c) in enumerate(zip(first.T, second.T)):
            audit = step in (0, first.shape[1]//2, first.shape[1]-1)
            u_d, u_o = checked(state_d, state_o, a, audit)
            rhs_d = state_d + ((1-d)/d) * (u_d - state_d)
            rhs_o = state_o + ((1-d)/d) * (u_o - state_o)
            state_d, state_o = checked(rhs_d, rhs_o, c, audit)
            removed += dt * ((1-d) * a * (model.loss @ u_o) + d * c * (model.loss @ state_o))
    except ArithmeticError:
        return None
    return removed


def _stage_rhs(dark_d, coherence, q, forcing):
    dark_ground_excited = sparse.csr_matrix(dark_d)
    no = len(coherence)
    c = np.r_[coherence.real, coherence.real][:, None]
    s = coherence.imag[:, None]

    def rhs(old_d, old_o, shape, amplitude):
        dark_rhs = dark_ground_excited @ old_d
        optical_rhs = c * old_o + np.vstack((-s * old_o[no:], s * old_o[:no]))
        optical_rhs += forcing[:, None] * shape + amplitude * (q @ dark_rhs)
        return dark_rhs, optical_rhs

    return rhs


def _iterative_stage(roots, p, operator, stage_rhs):
    # The norm bound selects candidates; the residual certifies each RHS.
    if max(roots)**2 * np.linalg.norm(operator, ord=np.inf) > 0.2:
        return None

    def stage(old_d, old_o, shape, _audit):
        amplitude = roots * shape
        dark_rhs, rhs = stage_rhs(old_d, old_o, shape, amplitude)
        actual_o = _neumann(rhs, amplitude**2, operator)
        if actual_o is None:
            raise ArithmeticError("Unresolved Neumann drive stage.")
        return dark_rhs + p @ (amplitude * actual_o), actual_o

    return stage


def _refined_eigen_stage(roots, p, operator, stage_rhs, modal_rhs, eigen):
    real, imag, basis, factors = eigen
    pairs = np.flatnonzero(imag > 0)

    def stage(old_d, old_o, shape, _audit):
        amplitude = roots * shape
        squared = amplitude**2
        dark_rhs, rhs = stage_rhs(old_d, old_o, shape, amplitude)
        modal = modal_rhs(old_d, old_o, shape, amplitude)
        actual_o = basis @ _apply_blocks(modal, squared, real, imag, pairs)
        residual = (rhs - actual_o) + squared * (operator @ actual_o)
        rhs_norm_squared = np.einsum("ij,ij->j", rhs, rhs)
        # Keep a margin under the stage acceptance tolerance. The eigenbasis
        # is only a preconditioner for the independently reconstructed residual.
        refine = np.einsum("ij,ij->j", residual, residual) > 1e-28 * rhs_norm_squared
        if np.any(refine):
            correction = linalg.lu_solve(factors, residual[:, refine], check_finite=False)
            actual_o[:, refine] += basis @ _apply_blocks(correction, squared[refine], real, imag, pairs)
            residual = (rhs - actual_o) + squared * (operator @ actual_o)
        if (not np.isfinite(rhs_norm_squared).all()
                or not np.all(np.einsum("ij,ij->j", residual, residual) <= 1e-24 * rhs_norm_squared)):
            raise ArithmeticError("Unresolved refined eigenbasis drive stage.")
        return dark_rhs + p @ (amplitude * actual_o), actual_o

    return stage


def fixed_steps(model, detuning, first, second, dx, scales, flux, roots, checkpoint, check, *,
                neumann=None):
    """Eliminate gg/ee from both stages, with the original time and flux weights.

    Return None to request a retry with the original solver.
    """
    if neumann is None:
        neumann = model.line == "D1"
    nb = len(first)
    count = len(roots)
    first = np.repeat(first, count, axis=0)
    second = np.repeat(second, count, axis=0)
    roots = np.tile(roots, nb)
    photons = np.zeros((nb, count))
    d = field_history.SDIRK_DIAGONAL
    for v, scale in enumerate(scales):
        if checkpoint is not None:
            checkpoint()
        dt = dx * scale
        shift = d * dt
        dark_d = model.dark_inverse(shift)
        coherence = 1 / (1 + shift * (model.optical_hwhm + 1j * (model.optical_frequencies - detuning)))
        dark_o = _block_to_real(np.diag(coherence))
        p = shift * (dark_d @ model.from_optical)
        q = shift * (dark_o @ model.to_optical)
        operator = q @ p
        if neumann or model.line == "D2":
            stage_rhs = _stage_rhs(dark_d, coherence, q, shift * (dark_o @ model.source))
        if neumann:
            stage = _iterative_stage(roots, p, operator, stage_rhs)
            if stage is not None:
                removed = _flight(model, first, second, roots, dt, stage, check)
                if removed is not None:
                    photons += flux[:, v, None] * removed.reshape(nb, count)
                    continue
        eigen = _eigen_blocks(operator)
        if eigen is None:
            return None
        re, im, basis, factors = eigen
        pairs = np.flatnonzero(im > 0)
        to_o = linalg.lu_solve(factors, dark_o, check_finite=False)
        to_d = linalg.lu_solve(factors, q @ dark_d, check_finite=False)
        source = shift * (to_o @ model.source)

        def modal_rhs(old_d, old_o, shape, amplitude):
            return to_o @ old_o + source[:, None] * shape + amplitude * (to_d @ old_d)

        if model.line == "D2":
            stage = _refined_eigen_stage(roots, p, operator, stage_rhs, modal_rhs, eigen)
        else:
            restore_d = p @ basis

            def stage(old_d, old_o, shape, audit):
                amplitude = roots * shape
                modal = modal_rhs(old_d, old_o, shape, amplitude)
                solution = _apply_blocks(modal, amplitude**2, re, im, pairs)
                actual_o = basis @ solution
                dark_rhs = dark_d @ old_d
                actual_d = dark_rhs + restore_d @ (amplitude * solution)
                if audit:
                    rhs_o = dark_o @ (old_o + shift * model.source[:, None] * shape)
                    coupling = amplitude * (q @ actual_d)
                    feedback = amplitude * (p @ actual_o)
                    if not (_within_residual(actual_o - coupling - rhs_o, actual_o, coupling, rhs_o)
                            and _within_residual(actual_d - feedback - dark_rhs, actual_d, feedback, dark_rhs)):
                        raise ArithmeticError("Unresolved reduced drive stage.")
                return actual_d, actual_o

        removed = _flight(model, first, second, roots, dt, stage, check)
        if removed is None:
            return None
        photons += flux[:, v, None] * removed.reshape(nb, count)
    return photons.sum(axis=0)
