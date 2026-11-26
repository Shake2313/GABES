"""Coherent beam-crossing absorption for a thermal Rb gas in a uniform magnetic field.

The incident irradiance drives a full density matrix along each
crossing. The flux measure, radial-profile interpolation, and annulus boundaries
are the same as in ballistic_pumping. Detunings are angular frequencies in
the atom frame. Longitudinal Maxwell averaging and propagation are separate.
"""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu, spsolve

from . import core, field_history, species
from .ballistic_atom import BallisticProbe
from .ballistic_pumping import TrajectoryGrid, _axis, _positive
from .field_weak import _validated_inputs, weak_cross_sections

_D = field_history.SDIRK_DIAGONAL


def _check(populations, trace):
    minimum = populations.min()
    error = np.max(np.abs(trace - 1.0))
    if not np.isfinite(minimum) or not np.isfinite(error) or minimum < -1e-10 or error > 1e-10:
        raise ArithmeticError(f"Unresolved field-history populations or trace: minimum {minimum:g}, "
                              f"trace error {error:g}. Refine the crossing time steps.")


def _direct(ops, first, second, step, bins, scales, flux, roots, annuli, checkpoint):
    """Direct sparse SDIRK, including unequal steps at radial boundaries.

    Evolve (rho - rho_thermal)/sqrt(I_peak); rho_thermal is an exact dark
    stationary state. This avoids a cancellation in weak-drive absorption.
    The excited-state derivative due to light gives *net optical absorption*
    where it occurred. Spontaneously emitted photons may leave another annulus.
    """
    identity = sparse.eye(len(ops.initial), format="csc")
    drive = field_history.rabi_per_sqrt_intensity(ops) * ops.driver
    source = drive @ ops.initial
    observable = ops.excited_population @ drive
    light = np.zeros((len(roots), annuli))
    for b in range(len(first)):
        intervals = np.flatnonzero(step[b] > 0)
        for v, scale in enumerate(scales):
            if checkpoint is not None:
                checkpoint()
            for i, root in enumerate(roots):
                state = np.zeros_like(ops.initial)
                for j in intervals:
                    if checkpoint is not None and j % 8 == 0:
                        checkpoint()
                    dt = step[b, j] * scale
                    a, c = first[b, j], second[b, j]
                    dark = identity - (_D * dt) * ops.base
                    first_stage = spsolve(dark - (_D * dt * root * a) * drive,
                                          state + (_D * dt * a) * source)
                    rhs = state + ((1 - _D) / _D) * (first_stage - state) + (_D * dt * c) * source
                    next_state = spsolve(dark - (_D * dt * root * c) * drive, rhs)
                    for stage in (first_stage, next_state):
                        populations = ops.initial[ops.population_rows] + root * stage[ops.population_rows]
                        _check(populations.real, 1.0 + root * (ops.trace @ stage))
                    ring = bins[b, j]
                    if ring >= 0:
                        first_loss = a * (observable @ first_stage).real
                        second_loss = c * (observable @ next_state).real
                        photons_per_intensity = dt * ((1 - _D) * first_loss + _D * second_loss)
                        light[i, ring] += flux[b, v] * photons_per_intensity
                    state = next_state
    return light


def _fixed_steps(ops, first, second, dx, scales, flux, roots, checkpoint):
    """Exact SDIRK stage solves in an intensity-independent eigenbasis.

    For a fixed time step, factor A=1-d*dt*L_dark and diagonalize
    A^-1 (d*dt*L_light)=V diag(lambda) V^-1. All impact parameters and
    peak irradiances now share the same basis, time step and operator.
    The solve of (A-a*d*dt*L_light) costs one matrix application and
    diagonal divisions. Revert to sparse solves if the basis is singular.
    """
    n = len(ops.initial)
    eye = sparse.eye(n, format="csc")
    drive = field_history.rabi_per_sqrt_intensity(ops) * ops.driver
    observable = ops.excited_population @ drive
    source = drive @ ops.initial
    nb = len(first)
    rays = np.repeat(np.arange(nb), len(roots))
    amplitudes = np.tile(roots, nb)
    photons = np.zeros((nb, len(roots)))
    for v, scale in enumerate(scales):
        if checkpoint is not None:
            checkpoint()
        dt = dx * scale
        shift = _D * dt
        dark = splu(eye - shift * ops.base)
        affine = dark.solve((shift * drive).toarray())
        eigenvalues, basis = linalg.eig(affine, check_finite=False)
        lu, pivots = linalg.lu_factor(basis, check_finite=False)
        condition_estimate = linalg.get_lapack_funcs("gecon", (lu,))
        rcond, info = condition_estimate(lu, linalg.norm(basis, 1))
        if info or not np.isfinite(rcond) or rcond < 1e-9:
            return None
        factors = lu, pivots
        propagate = linalg.lu_solve(factors, dark.solve(basis), check_finite=False)
        forcing = linalg.lu_solve(factors, dark.solve(shift * source), check_finite=False)
        population = basis[ops.population_rows]
        trace = ops.trace @ basis
        loss = observable @ basis
        thermal = ops.initial[ops.population_rows, None].real
        state = np.zeros((n, len(amplitudes)), dtype=complex)
        accumulated = np.zeros(len(amplitudes))
        for a, c in zip(first.T, second.T):
            a, c = a[rays], c[rays]
            first_stage = ((propagate @ state + forcing[:, None] * a)
                           / (1 - eigenvalues[:, None] * (amplitudes * a)))
            rhs = state + ((1 - _D) / _D) * (first_stage - state)
            next_state = ((propagate @ rhs + forcing[:, None] * c)
                          / (1 - eigenvalues[:, None] * (amplitudes * c)))
            first_loss = a * (loss @ first_stage).real
            second_loss = c * (loss @ next_state).real
            accumulated += dt * ((1 - _D) * first_loss + _D * second_loss)
            for stage in (first_stage, next_state):
                _check(thermal + (population @ stage).real * amplitudes,
                       1.0 + (trace @ stage) * amplitudes)
            state = next_state
        photons += flux[:, v, None] * accumulated.reshape(nb, -1)
    return photons.sum(axis=0)


class FieldBallisticProbe:
    """One-isotope atom-frame absorption for a single beam and uniform static field."""

    def __init__(self, isotope, line, temperature_k, optical_hwhm_rad_s=None, *,
                 field_ut=(0., 0., 0.), jones=(1., 0.)):
        self.isotope, self.field_ut, self.jones = _validated_inputs(isotope, line, field_ut, jones)
        self.line = line
        self.temperature_k = _positive(temperature_k, "Kinetic temperature")
        # Reuse the established path measure, state-independent geometry,
        # intrinsic width and dipole normalization for this isotope/line.
        self._geometry = BallisticProbe(self.isotope, line, temperature_k, optical_hwhm_rad_s)
        self.manifold = self._geometry.manifold
        self.width = self._geometry.width
        self.speed_sigma = self._geometry.speed_sigma
        self.photon_energy_j = self._geometry.photon_energy_j

    def _operators(self, delta):
        b, e, quant = self.field_ut, self.jones, "k"
        # The dipole Hamiltonian depends only on B and E. A real (up to
        # global phase) polarization allows exact rotations when B is parallel
        # or perpendicular to E. They preserve coherent dynamics and expose
        # smaller invariant subspaces to the sparse operator builder.
        e = np.asarray(e)
        pivot = e[np.argmax(np.abs(e))]
        linear = e * (pivot.conjugate() / abs(pivot))
        tolerance = 32 * np.finfo(float).eps
        if np.max(np.abs(linear.imag)) <= tolerance:
            direction = linear.real / np.linalg.norm(linear.real)
            parallel = np.dot(b[:2], direction)
            transverse = np.hypot(b[2], np.dot(b[:2], (-direction[1], direction[0])))
            field_norm = np.hypot(parallel, transverse)
            if transverse <= tolerance * field_norm:
                b, e, quant = (parallel, 0., 0.), (1., 0.), "E"
            elif abs(parallel) <= tolerance * field_norm:
                b, e = (0., 0., transverse), (1., 0.)
        return field_history.operators(
            self.isotope.name, self.line, delta / species.MHZ, field_ut=b, jones=e, quantization=quant,
            optical_hwhm_increment_rad_s=self.width - 0.5 * self.manifold.gamma)

    def cross_sections(self, detuning_rad_s, peak_intensity_w_m2, waist_m, *,
                       grid=TrajectoryGrid(), accelerated=True, checkpoint=None):
        """Total removed power per density, path length and incident power [m²]."""
        delta = _axis(detuning_rad_s, "Detuning (rad/s)")
        peak = _axis(peak_intensity_w_m2, "Peak intensity (W/m²)")
        if np.any(peak < 0):
            raise ValueError("Peak intensities must be nonnegative.")
        waist = _positive(waist_m, "Beam radius (m)")
        step, r1, r2, bins, scales, flux = self._geometry._measure(grid, waist)
        first, second = np.exp(-r1**2), np.exp(-r2**2)
        roots = np.sqrt(peak)
        result = np.empty((len(peak), len(delta)))
        with core.blas_single_thread():
            for j, freq in enumerate(delta):
                ops = self._operators(freq)
                counts = None
                if accelerated:
                    counts = _fixed_steps(ops, first, second, 2.0 * grid.xcut / grid.nx,
                                          scales, flux, roots, checkpoint)
                if counts is None:
                    counts = _direct(ops, first, second, step, bins, scales, flux, roots, 1, checkpoint)[:, 0]
                result[:, j] = counts * self.photon_energy_j / (0.5 * np.pi * waist**2)
        if not np.isfinite(result).all() or np.any(result < 0.):
            raise ArithmeticError("Unresolved nonfinite or negative total field-history absorption.")
        return result

    def weak_cross_sections(self, detuning_rad_s, waist_m, *, grid=None):
        """Linear atom-frame absorption including the finite crossing envelope [m²]."""
        return weak_cross_sections(self.isotope, self.line, self.width, self.field_ut, self.jones,
                                   detuning_rad_s, temperature_k=self.temperature_k, waist_m=waist_m, grid=grid)

    def radial_rates(self, detuning_rad_s, radial_nodes_waist, intensity_w_m2, annulus_edges_waist,
                     waist_m, *, grid=TrajectoryGrid(), accelerated=True, checkpoint=None):
        """Net optical power absorbed per density and axial length in each annulus [W m²]."""
        delta = _axis(detuning_rad_s, "Detuning (rad/s)")
        first, second, step, bins, scales, flux, n = self._geometry._radial_geometry(
            radial_nodes_waist, intensity_w_m2, annulus_edges_waist, waist_m, grid)
        result = np.zeros((len(delta), n))
        peak = float(np.max(intensity_w_m2))
        if peak:
            with core.blas_single_thread():
                # Crossing a radial boundary splits the step for this impact
                # parameter. Each profile therefore uses direct variable-step SDIRK.
                for j, freq in enumerate(delta):
                    photons = _direct(self._operators(freq), np.sqrt(first / peak), np.sqrt(second / peak),
                                      step, bins, scales, flux, [np.sqrt(peak)], n, checkpoint)[0]
                    result[j] = photons * self.photon_energy_j * peak
        return result
