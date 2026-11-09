"""Resolved 85Rb D1 pump optical pumping and infinitesimal FWM response.

The 24-level density matrix has trace one over the actual 12 ground sublevels.
Signed dipoles are the Steck/Wigner-Eckart matrices shared with the multilevel
magneto-optics model; no hyperfine-block fit or external 1/12 factor is used.
Pump polarization defines the pi axis. The orthogonal linear weak modes use
the same coherent (sigma-minus - sigma-plus)/sqrt(2) polarization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import constants, core, doppler, hyperfine, observables
from .atoms import AtomModel
from .magneto_multilevel import build_multilevel_atom

N_LEVELS = 24
N_GROUND = 12
EMISSION_MODES = ("hyperfine_secular", "global_polarization")


@dataclass(frozen=True)
class ZeemanModel:
    atom: AtomModel
    level_F: np.ndarray
    level_m: np.ndarray
    sectors: dict
    dipoles: dict
    pump_coupling: np.ndarray
    weak_coupling: np.ndarray
    thermal: np.ndarray
    provenance: dict


@dataclass(frozen=True)
class PreparedPump:
    model: ZeemanModel
    pump_rabi_rad_s: float
    delta_eff_axis_rad_s: np.ndarray
    velocity_weights: np.ndarray
    pump_state: np.ndarray
    response_bases: dict
    velocity_diagonals: dict
    sources: dict
    readouts: dict
    diagnostics: dict
    frequency_doppler_scale: np.ndarray
    pole_data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ZeemanResponse:
    chi_matrix: np.ndarray
    pump_state: np.ndarray
    mean_pump_state: np.ndarray
    delta_axis_rad_s: np.ndarray
    analysis_frequency_axis_rad_s: np.ndarray
    relative_frequency_rad_s: np.ndarray
    delta_eff_axis_rad_s: np.ndarray
    velocity_weights: np.ndarray
    velocity_chi_matrix: np.ndarray | None
    diagnostics: dict
    provenance: dict

    @property
    def chi_ss(self):
        return self.chi_matrix[..., 0, 0]

    @property
    def chi_sc(self):
        return self.chi_matrix[..., 0, 1]

    @property
    def chi_cs(self):
        return np.conj(self.chi_matrix[..., 1, 0])

    @property
    def chi_cc(self):
        return np.conj(self.chi_matrix[..., 1, 1])

    @property
    def positive_frequency_chi_matrix(self):
        value = self.chi_matrix.copy()
        value[..., 1, :] = value[..., 1, :].conj()
        return value

    def physical_chi(self, density_m3, *, residual=1.):
        """Positive-frequency (ss,sc;cs,cc) susceptibility, dimensionless.

        Reduced response has units s per reference angular Rabi frequency.
        The reference Rabi uses d0/sqrt(3), so the propagation dipole squared
        is d0^2/3. Actual sublevel populations are already in rho; no additional
        1/12 average is allowed. Residual is a separate practical coefficient.
        """
        density, factor = float(density_m3), float(residual)
        if not np.isfinite([density, factor]).all() or density < 0 or factor < 0:
            raise ValueError("density and residual must be finite and nonnegative")
        return observables.chi_phys(self.positive_frequency_chi_matrix, density,
                                    line_strength=factor/3.)


@lru_cache(maxsize=12)
def build_model(T=394.15, *, transit_rate=constants.GAMMA_GG,
                ground_dephasing_rate=None, optical_dephasing_rate=None,
                emission_mode="hyperfine_secular"):
    """Build full angular operators and a per-temperature dissipator.

    Rates are exponential angular rates in s^-1. Hyperfine-projector collision
    dephasing preserves rotation covariance and within-F Zeeman coherence. The
    thermal reset supplies transit population/coherence relaxation exactly once.
    """
    temperature, reset = float(T), float(transit_rate)
    if not np.isfinite([temperature, reset]).all() or temperature <= 0 or reset < 0:
        raise ValueError("positive temperature and nonnegative transit rate required")
    if emission_mode not in EMISSION_MODES:
        raise ValueError(f"emission_mode must be one of {EMISSION_MODES}")
    density = hyperfine.number_density(temperature)
    ground_rate = (constants.ground_coherence_dephasing(temperature, density, floor=0.)
                   if ground_dephasing_rate is None else float(ground_dephasing_rate))
    optical_rate = (.5*(hyperfine.self_broadened_gamma(density)-constants.GAMMA)
                    if optical_dephasing_rate is None else float(optical_dephasing_rate))
    if not np.isfinite([ground_rate, optical_rate]).all() or min(ground_rate, optical_rate) < 0:
        raise ValueError("dephasing rates must be finite and nonnegative")
    angular = build_multilevel_atom("Rb85", "D1")
    levels = angular.ground+angular.excited
    fs = np.array([level[0] for level in levels])
    ms = np.array([level[1] for level in levels])
    dipoles = {}
    for q in (-1, 0, 1):
        value = np.zeros((N_LEVELS, N_LEVELS), complex)
        value[:N_GROUND, N_GROUND:] = angular.dipoles[q]
        dipoles[q] = value
    emission = []
    if emission_mode == "hyperfine_secular":
        for fg in (2, 3):
            for fe in (2, 3):
                mask = ((fs[:, None] == fg) & (fs[None, :] == fe))
                for q in (-1, 0, 1):
                    emission.append(np.sqrt(constants.GAMMA)*dipoles[q]*mask)
    else:
        emission = [np.sqrt(constants.GAMMA)*dipoles[q] for q in (-1, 0, 1)]
    collapse = []
    if ground_rate:
        for fg in (2, 3):
            projection = np.zeros(N_LEVELS)
            projection[:N_GROUND] = fs[:N_GROUND] == fg
            collapse.append(np.diag(np.sqrt(ground_rate)*projection))
    if optical_rate:
        projection = np.r_[np.zeros(N_GROUND), np.ones(N_LEVELS-N_GROUND)]
        collapse.append(np.diag(np.sqrt(2*optical_rate)*projection))
    atom = AtomModel(
        name="fwm_rb85_d1_resolved_zeeman", n_levels=N_LEVELS,
        labels=tuple(("g" if i < N_GROUND else "e")+f"F{f}m{m:+d}"
                     for i, (f, m) in enumerate(levels)),
        ground=tuple(range(N_GROUND)), excited=tuple(range(N_GROUND, N_LEVELS)),
        decay=(), dephasing=(), doppler_levels=tuple(range(N_GROUND, N_LEVELS)),
        emission_ops=tuple(emission), collapse_ops=tuple(collapse))
    thermal = np.diag(np.r_[np.full(N_GROUND, 1./N_GROUND), np.zeros(N_LEVELS-N_GROUND)])
    trace = np.eye(N_LEVELS).reshape(-1)
    atom.lindblad += reset*(np.outer(thermal.reshape(-1), trace)-np.eye(N_LEVELS**2))
    delta_m = (ms[:, None]-ms[None, :]).ravel()
    sectors = {q: np.flatnonzero(delta_m == q) for q in range(-6, 7)}
    pump_coupling = np.sqrt(3.)*dipoles[0]
    weak_coupling = np.sqrt(1.5)*(dipoles[-1]-dipoles[1])
    provenance = dict(
        species="85Rb D1", levels=N_LEVELS, ground_sublevels=N_GROUND,
        dipole_source="magneto_multilevel signed Steck/Wigner-Eckart Cq; sum_q Cq†Cq=I_excited",
        reference_rabi="constants.rabi_freq(P,w); reference dipole d0/sqrt(3)",
        drive_and_readout="sqrt(3)*Cq; same signed coherent polarization in source and readout",
        pump_polarization="pi q=0; quantization along pump linear polarization",
        weak_polarization="orthogonal linear: (sigma-minus - sigma-plus)/sqrt(2)",
        emission_mode=emission_mode,
        emission_scope=("frequency-secular (Fg,Fe,q) jumps; coherent m-channel sums within each hyperfine transition"
                        if emission_mode == "hyperfine_secular" else
                        "diagnostic nonsecular global-q jumps; combines resolved optical hyperfine frequencies"),
        transit_reset_rate_rad_s=reset, ground_hyperfine_dephasing_rate_rad_s=ground_rate,
        optical_elastic_dephasing_rate_rad_s=optical_rate,
        ground_collision_scope="scalar P_F projectors; inter-F rate gamma; optical extra gamma/2; no individual-m phase randomization",
        propagation_line_strength=1./3., external_sublevel_population_factor=1.,
        residual_applied=1., microscopic_noise_covariance_available=False,
        scope="pump-only state, infinitesimal full two-frequency response; uniform pi pump; zero B; collinear atomic Doppler")
    for value in (fs, ms, thermal, pump_coupling, weak_coupling, atom.lindblad, atom.S_v, *dipoles.values()):
        value.flags.writeable = False
    return ZeemanModel(atom, fs, ms, sectors, dipoles, pump_coupling,
                       weak_coupling, thermal, provenance)


def pump_hamiltonian(pump_rabi_rad_s, model):
    """Physical pump frame at Delta_eff=0 on the F=2 -> F'=3 origin."""
    omega = float(pump_rabi_rad_s)
    if not np.isfinite(omega) or omega < 0:
        raise ValueError("pump Rabi frequency must be finite and nonnegative")
    diagonal = np.zeros(N_LEVELS)
    diagonal[:N_GROUND] = np.where(model.level_F[:N_GROUND] == 3, constants.OMEGA_HF, 0.)
    diagonal[N_GROUND:] = np.where(model.level_F[N_GROUND:] == 2, -constants.OMEGA_EXCITED_HF, 0.)
    return np.diag(diagonal)+.5*omega*(model.pump_coupling+model.pump_coupling.conj().T)


def _chunks(size, chunk_size):
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    return (slice(start, min(start+int(chunk_size), size)) for start in range(0, size, int(chunk_size)))


def prepare_pump(pump_rabi_rad_s, delta_eff_axis_rad_s, *, T=394.15,
                 velocity_weights=None, frequency_doppler_scale=None,
                 chunk_size=64, **model_options):
    """Solve the 88-dimensional Delta-m=0 state, retaining reusable ±1 blocks."""
    deff = np.atleast_1d(np.asarray(delta_eff_axis_rad_s, float))
    if deff.ndim != 1 or deff.size < 1 or not np.isfinite(deff).all():
        raise ValueError("finite nonempty one-dimensional Delta_eff axis required")
    weights = (np.full(len(deff), 1./len(deff)) if velocity_weights is None
               else np.asarray(velocity_weights, float))
    if weights.shape != deff.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("nonnegative finite velocity weights must match Delta_eff axis")
    weights = weights/weights.sum()
    frequency_scale = (np.ones(len(deff)) if frequency_doppler_scale is None
                       else np.asarray(frequency_doppler_scale, float))
    if frequency_scale.shape != deff.shape or not np.isfinite(frequency_scale).all():
        raise ValueError("frequency Doppler scales must be finite and match velocities")
    model = build_model(float(T), **model_options)
    h = pump_hamiltonian(pump_rabi_rad_s, model)
    generator = core.build_liouvillian(h, model.atom)
    indices = model.sectors[0]
    base = generator[np.ix_(indices, indices)]
    velocity_diag = np.diag(model.atom.S_v)[indices]
    trace = np.eye(N_LEVELS).reshape(-1)[indices]
    rho = np.zeros((len(deff), N_LEVELS, N_LEVELS), complex)
    maximum_residual = 0.
    with core.blas_single_thread():
        for chunk in _chunks(len(deff), chunk_size):
            matrices = np.broadcast_to(base, (len(deff[chunk]), len(indices), len(indices))).copy()
            diagonal = np.arange(len(indices))
            matrices[:, diagonal, diagonal] -= deff[chunk, None]*velocity_diag
            if float(pump_rabi_rad_s) == 0.:
                states = np.broadcast_to(model.thermal.reshape(-1)[indices], (len(matrices), len(indices))).copy()
            else:
                constrained = matrices.copy()
                scale = np.maximum(np.max(abs(matrices), axis=(1, 2)), 1.)
                constrained[:, 0, :] = trace[None, :]*scale[:, None]
                rhs = np.zeros((len(matrices), len(indices), 1), complex)
                rhs[:, 0, 0] = scale
                states = np.linalg.solve(constrained, rhs)[..., 0]
            rho[chunk].reshape(-1, N_LEVELS**2)[:, indices] = states
            residual = np.einsum("bij,bj->bi", matrices, states)
            norm = np.linalg.norm(matrices, axis=(1, 2))*np.linalg.norm(states, axis=1)
            maximum_residual = max(maximum_residual, float(np.max(np.linalg.norm(residual, axis=1)/np.maximum(norm, 1e-300))))
    lower = model.weak_coupling
    raise_ = lower.conj().T
    perturbations = np.stack((.5*raise_, .5*lower))
    source = -1j*(np.einsum("rij,vjk->vrik", perturbations, rho)
                  - np.einsum("vij,rjk->vrik", rho, perturbations))
    source = source.reshape(len(deff), 2, N_LEVELS**2).transpose(0, 2, 1)
    readout = np.stack((lower.T.reshape(-1), raise_.T.reshape(-1)))
    bases, diagonals, sources, readouts = {}, {}, {}, {}
    for q in (-1, 1):
        ind = model.sectors[q]
        bases[q] = generator[np.ix_(ind, ind)].copy()
        diagonals[q] = np.diag(model.atom.S_v)[ind].copy()
        sources[q] = source[:, ind, :].copy()
        readouts[q] = readout[:, ind].copy()
    diagnostics = dict(
        max_pump_normalized_residual=maximum_residual,
        max_pump_trace_error=float(np.max(abs(np.trace(rho, axis1=-2, axis2=-1)-1.))),
        max_pump_hermiticity_error=float(np.max(abs(rho-rho.conj().swapaxes(-1, -2)))),
        minimum_pump_eigenvalue=float(np.min(np.linalg.eigvalsh(rho))),
        steady_sector_dimension=len(indices), response_sector_dimensions={str(q):len(model.sectors[q]) for q in (-1, 1)},
        velocity_classes=len(deff))
    return PreparedPump(model, float(pump_rabi_rad_s), deff.copy(), weights.copy(), rho,
                        bases, diagonals, sources, readouts, diagnostics, frequency_scale.copy())


def _pole_response(prepared, frequencies, chunk_size, return_velocity_chi):
    """Reusable residues with direct-resolvent guards; fallback is explicit."""
    nv = len(prepared.delta_eff_axis_rad_s)
    output = np.zeros(frequencies.shape+(2, 2), complex)
    velocity_chi = np.zeros(frequencies.shape+(nv, 2, 2), complex) if return_velocity_chi else None
    guards = np.unique(np.linspace(0, frequencies.size-1, min(3, frequencies.size), dtype=int))
    maximum_error, fallbacks = 0., 0
    with core.blas_single_thread():
        for chunk in _chunks(nv, chunk_size):
            values = np.zeros(frequencies.shape+(len(prepared.delta_eff_axis_rad_s[chunk]), 2, 2), complex)
            for q in (-1, 1):
                base, source = prepared.response_bases[q], prepared.sources[q][chunk]
                n = len(base)
                indices = np.arange(n)
                matrix = np.broadcast_to(base, (len(source), n, n)).copy()
                matrix[:, indices, indices] -= prepared.delta_eff_axis_rad_s[chunk, None]*prepared.velocity_diagonals[q]
                cache_key = (q, chunk.start, chunk.stop)
                if cache_key not in prepared.pole_data:
                    eigenvalues, vectors = np.linalg.eig(matrix)
                    projection = np.linalg.solve(vectors, source)
                    read = np.einsum("oi,vik->vok", prepared.readouts[q], vectors)
                    residues = -np.einsum("vok,vki->voik", read, projection)
                    prepared.pole_data[cache_key] = (eigenvalues, residues)
                eigenvalues, residues = prepared.pole_data[cache_key]
                inverse = 1./(eigenvalues[None, None, :, :]+1j*frequencies[:, :, None, None]
                              *prepared.frequency_doppler_scale[None, None, chunk, None])
                estimate = np.einsum("voik,advk->advoi", residues, inverse)
                error = 0.
                for index in guards:
                    frequency = frequencies.reshape(-1)[index]
                    shifted = matrix.copy()
                    shifted[:, indices, indices] += 1j*frequency*prepared.frequency_doppler_scale[chunk, None]
                    direct = np.einsum("oi,vik->vok", prepared.readouts[q], np.linalg.solve(shifted, -source))
                    approx = estimate.reshape(-1, len(source), 2, 2)[index]
                    denominator = np.maximum(np.linalg.norm(direct, axis=(1, 2)), 1e-25)
                    error = max(error, float(np.max(np.linalg.norm(approx-direct, axis=(1, 2))/denominator)))
                maximum_error = max(maximum_error, error)
                if not np.isfinite(estimate).all() or error > 1e-6:
                    fallbacks += 1
                    for a in range(frequencies.shape[0]):
                        for d in range(frequencies.shape[1]):
                            shifted = matrix.copy()
                            shifted[:, indices, indices] += 1j*frequencies[a, d]*prepared.frequency_doppler_scale[chunk, None]
                            estimate[a, d] = np.einsum("oi,vik->vok", prepared.readouts[q], np.linalg.solve(shifted, -source))
                values += estimate
            output += np.einsum("advij,v->adij", values, prepared.velocity_weights[chunk])
            if velocity_chi is not None:
                velocity_chi[:, :, chunk] = values
    return output, velocity_chi, dict(max_pole_guard_relative_error=maximum_error,
                                    pole_direct_fallback_chunks=fallbacks,
                                    pole_guard_relative_tolerance=1e-6)


def evaluate_prepared(prepared, delta_axis_rad_s, *, branch=-1,
                      analysis_frequency_axis_rad_s=(0.,), chunk_size=64,
                      return_velocity_chi=False, method="direct"):
    """Resolve the complete 2x2 Nambu response using two 80-dimensional sectors."""
    delta = np.atleast_1d(np.asarray(delta_axis_rad_s, float))
    analysis = np.atleast_1d(np.asarray(analysis_frequency_axis_rad_s, float))
    if branch not in (-1, 1):
        raise ValueError("branch must be -1 or +1")
    if any(axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() for axis in (delta, analysis)):
        raise ValueError("finite nonempty one-dimensional frequency axes required")
    if method not in ("direct", "poles"):
        raise ValueError("method must be direct or poles")
    frequencies = branch*constants.OMEGA_HF+delta[None, :]+analysis[:, None]
    nv = len(prepared.delta_eff_axis_rad_s)
    output = np.zeros((len(analysis), len(delta), 2, 2), complex)
    velocity_chi = (np.zeros((len(analysis), len(delta), nv, 2, 2), complex)
                    if return_velocity_chi else None)
    maximum_residual = 0.
    with core.blas_single_thread():
        for chunk in _chunks(nv, chunk_size):
            if method == "poles":
                break
            class_response = np.zeros((len(analysis), len(delta), len(prepared.delta_eff_axis_rad_s[chunk]), 2, 2), complex)
            for q in (-1, 1):
                base = prepared.response_bases[q]
                n = len(base)
                diagonal = np.arange(n)
                generator = np.broadcast_to(base, (class_response.shape[2], n, n)).copy()
                generator[:, diagonal, diagonal] -= prepared.delta_eff_axis_rad_s[chunk, None]*prepared.velocity_diagonals[q]
                source = prepared.sources[q][chunk]
                for a in range(len(analysis)):
                    for d in range(len(delta)):
                        matrix = generator.copy()
                        matrix[:, diagonal, diagonal] += 1j*frequencies[a, d]*prepared.frequency_doppler_scale[chunk, None]
                        solution = np.linalg.solve(matrix, -source)
                        class_response[a, d] += np.einsum("oi,bik->bok", prepared.readouts[q], solution)
                        residual = np.einsum("bij,bjk->bik", matrix, solution)+source
                        norm = np.linalg.norm(matrix, axis=(1, 2))*np.linalg.norm(solution, axis=(1, 2))+np.linalg.norm(source, axis=(1, 2))
                        maximum_residual = max(maximum_residual, float(np.max(np.linalg.norm(residual, axis=(1, 2))/np.maximum(norm, 1e-300))))
            output += np.einsum("advij,v->adij", class_response, prepared.velocity_weights[chunk])
            if velocity_chi is not None:
                velocity_chi[:, :, chunk] = class_response
    diagnostics = dict(prepared.diagnostics,
        max_response_normalized_residual=maximum_residual,
        max_response_trace_error=0.,
        finite=bool(np.isfinite(output).all() and np.isfinite(prepared.pump_state).all()))
    if method == "poles":
        output, velocity_chi, pole_diagnostics = _pole_response(prepared, frequencies, chunk_size, return_velocity_chi)
        diagnostics.update(pole_diagnostics, max_response_normalized_residual=None,
                           finite=bool(np.isfinite(output).all() and np.isfinite(prepared.pump_state).all()))
    provenance = dict(prepared.model.provenance,
        branch=branch, method=method,
        relative_frequency="(branch*Omega_HF+delta+Omega_analysis)*frequency_doppler_scale; laboratory beat stays separate",
        nambu_order="outputs/inputs=(probe,conjugate-star); conjugate second row only for historical positive-frequency tuple",
        all_hyperfine_weak_couplings=True, block_reduction="exact axial Delta-m sectors; no state elimination")
    return ZeemanResponse(output, prepared.pump_state,
        np.einsum("v,vij->ij", prepared.velocity_weights, prepared.pump_state),
        delta.copy(), analysis.copy(), frequencies, prepared.delta_eff_axis_rad_s.copy(),
        prepared.velocity_weights.copy(), velocity_chi, diagnostics, provenance)


def pump_weak_response_reference(pump_rabi_rad_s, delta_axis_rad_s, delta_eff_axis_rad_s,
                                *, T=394.15, velocity_weights=None, branch=-1,
                                analysis_frequency_axis_rad_s=(0.,), chunk_size=64,
                                return_velocity_chi=False, frequency_doppler_scale=None,
                                method="direct", **model_options):
    prepared = prepare_pump(pump_rabi_rad_s, delta_eff_axis_rad_s, T=T,
        velocity_weights=velocity_weights, frequency_doppler_scale=frequency_doppler_scale,
        chunk_size=chunk_size, **model_options)
    return evaluate_prepared(prepared, delta_axis_rad_s, branch=branch,
        analysis_frequency_axis_rad_s=analysis_frequency_axis_rad_s, chunk_size=chunk_size,
        return_velocity_chi=return_velocity_chi, method=method)


def prepare_response(D_GHz, *, T, pump_power_W, pump_waist_m,
                     velocity_step=4., velocity_cutoff=4., velocities=None,
                     velocity_weights=None, chunk_size=64, **model_options):
    """Prepare the exact physical velocity/carrier map for reusable δ queries."""
    values = np.asarray([D_GHz, T, pump_power_W, pump_waist_m], float)
    if not np.isfinite(values).all() or T <= 0 or pump_power_W < 0 or pump_waist_m <= 0:
        raise ValueError("finite detuning, positive T/waist and nonnegative pump power required")
    if velocities is None:
        velocity, weights = doppler.velocity_grid(T, dv=velocity_step, cutoff_sigma=velocity_cutoff)
    else:
        velocity = np.asarray(velocities, float)
        weights = np.asarray(velocity_weights, float) if velocity_weights is not None else None
        if velocity.ndim != 1 or not velocity.size or not np.isfinite(velocity).all() or weights is None:
            raise ValueError("provided finite velocity nodes require explicit weights")
    origin = (7*constants.OMEGA_HF+5*constants.OMEGA_EXCITED_HF)/12.
    pump_k = (constants.OMEGA_D1+origin+2*np.pi*float(D_GHz)*1e9)/constants.C_LIGHT
    effective = 2*np.pi*float(D_GHz)*1e9-pump_k*velocity
    return prepare_pump(constants.rabi_freq(pump_power_W, pump_waist_m), effective,
        T=T, velocity_weights=weights, frequency_doppler_scale=1.-velocity/constants.C_LIGHT,
        chunk_size=chunk_size, **model_options)


def compute_response(D_GHz, delta_mhz, *, T, pump_power_W, pump_waist_m,
                     velocity_step=4., velocity_cutoff=4., branch=-1,
                     analysis_frequency_mhz=(0.,), chunk_size=64,
                     return_velocity_chi=False, velocities=None, velocity_weights=None,
                     method="direct", **model_options):
    """Convenience Maxwell response with the unchanged physical pump power map."""
    prepared = prepare_response(D_GHz, T=T, pump_power_W=pump_power_W,
        pump_waist_m=pump_waist_m, velocity_step=velocity_step, velocity_cutoff=velocity_cutoff,
        velocities=velocities, velocity_weights=velocity_weights, chunk_size=chunk_size, **model_options)
    return evaluate_prepared(prepared, np.asarray(delta_mhz, float)*2*np.pi*1e6,
        branch=branch, analysis_frequency_axis_rad_s=np.asarray(analysis_frequency_mhz)*2*np.pi*1e6,
        chunk_size=chunk_size, return_velocity_chi=return_velocity_chi, method=method)


def pump_weak_response(D_GHz, delta_mhz, *, T, pump_power_W, pump_waist_m,
                       velocity_step=4., velocity_cutoff=4., velocities=None,
                       velocity_weights=None, method="direct", **kwargs):
    """Adapter dictionary; chi is positive-frequency [[ss,sc],[cs,cc]] in s.

    The physical prefactor is observables.chi_phys(..., line_strength=residual/3).
    Uniform pump, zero B and collinear Doppler are fixed model assumptions.
    """
    response = compute_response(D_GHz, delta_mhz, T=T, pump_power_W=pump_power_W,
        pump_waist_m=pump_waist_m, velocity_step=velocity_step, velocity_cutoff=velocity_cutoff,
        velocities=velocities, velocity_weights=velocity_weights, method=method, **kwargs)
    if velocities is None:
        velocity, _ = doppler.velocity_grid(T, dv=velocity_step, cutoff_sigma=velocity_cutoff)
    else:
        velocity = np.asarray(velocities, float)
    chi = response.positive_frequency_chi_matrix
    return dict(chi=chi[0] if chi.shape[0] == 1 else chi,
        chi_by_analysis=chi, chi_units="seconds per reference angular Rabi frequency",
        physical_chi=response.physical_chi(hyperfine.number_density(T))[0] if chi.shape[0] == 1
                     else response.physical_chi(hyperfine.number_density(T)),
        pump_state=response.mean_pump_state, diagnostics=response.diagnostics,
        provenance=response.provenance, delta_mhz=np.atleast_1d(np.asarray(delta_mhz, float)),
        velocities_m_s=velocity.copy(), velocity_weights=response.velocity_weights,
        delta_eff_axis_rad_s=response.delta_eff_axis_rad_s,
        normalization=dict(propagation_line_strength_multiplier=1./3., residual_applied=1.,
                           external_sublevel_population_factor=1., reference_dipole="d0/sqrt(3)",
                           signed_drive_and_readout="sqrt(3)*C", thermal_ground_population_each=1./12.))
