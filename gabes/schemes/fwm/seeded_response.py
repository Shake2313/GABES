"""Finite-order Floquet and pump-only atomic response."""
from .. import fwm
from .common import (
    ATOM, BRANCHES, E2, E3, EXCITED_STATES, G1, G2, N_LEVELS,
    NoncollinearPumpWeakResponseReference, PumpWeakResponseReference, RESPONSE_POLE,
    RESPONSE_POLE_ADAPTIVE, SCAN_TOLERANCE_FLOOR_DB, SCAN_TOLERANCE_FLOOR_LOG10_GAIN,
    SCAN_TOLERANCE_RANGE_FRACTION, SEEDED_FLOQUET_ORDER, TRANSITION_DIPOLE_SCALE, T_CELL,
    noncollinear_doppler_reference_provenance, pump_weak_response_reference_provenance,
    seeded_sideband_beat,
)
from ...constants import OMEGA_HF
from ...core import (
    _transpose_perm, build_liouvillian, comm_super, floquet_solve_truncated, hermitian_basis,
    liouvillian_pole_residue_response, steady_state_batched, trace_zero_liouvillian_response,
)
from ... import adaptive_scan, constants, doppler, kernels, pole_doppler
import math
import numpy as np


# =========================================================
# Hamiltonians
# =========================================================
def _add_static_drive(H, ground, omega):
    for excited in EXCITED_STATES:
        omega_ge = omega * TRANSITION_DIPOLE_SCALE[ground, excited]
        H[ground, excited] += omega_ge / 2
        H[excited, ground] += omega_ge / 2


def _add_sideband_drive(H, ground, omega):
    for excited in EXCITED_STATES:
        H[excited, ground] += omega * TRANSITION_DIPOLE_SCALE[ground, excited] / 2


def _polarization_coherence(rho, ground):
    return sum(TRANSITION_DIPOLE_SCALE[ground, e] * rho[:, e, ground]
               for e in EXCITED_STATES)


def pump_hamiltonian_at_Deff_zero(Op_A, Op_B):
    """Static physical pump frame, before the longitudinal Doppler shift.

    For the standard minus Raman branch, applying
    ``U(t)=exp(+i*Omega_beat*t*|g2><g2|)`` to the seeded frame makes the second
    pump leg static and changes its diagonal from ``delta`` to
    ``delta + Omega_beat = OMEGA_HF``.  The resulting pump Hamiltonian is
    independent of probe detuning, as a pump-only state must be.
    """
    H = np.zeros((N_LEVELS, N_LEVELS), dtype=complex)
    H[G2, G2] = OMEGA_HF
    H[E2, E2] = -fwm.OMEGA_EXCITED_HF
    H[E3, E3] = 0.0
    _add_static_drive(H, G1, Op_A)
    _add_static_drive(H, G2, Op_B)
    return H


def pump_frame_to_seeded_harmonics(pump_state, n_f=1, branch=-1):
    """Reconstruct seeded-frame pump harmonics from the static pump state.

    This executable gauge map is currently certified only for the standard
    minus Raman branch.  In that frame coherences with ``g2`` in the row occupy
    harmonic ``-1`` and those with ``g2`` in the column occupy ``+1``; every
    ``|n|>=2`` block is identically zero.
    """
    if branch != -1:
        raise NotImplementedError(
            "the inherited plus-branch seeded frame is not gauge-equivalent "
            "to the physical static pump frame")
    n_f = int(n_f)
    if n_f < 1:
        raise ValueError("n_f must be at least 1")
    rho = np.asarray(pump_state, dtype=complex)
    if rho.shape[-2:] != (N_LEVELS, N_LEVELS):
        raise ValueError(
            f"pump_state must end in shape {(N_LEVELS, N_LEVELS)}")
    harmonics = np.zeros(
        rho.shape[:-2] + (2 * n_f + 1, N_LEVELS, N_LEVELS), dtype=complex)
    for row in range(N_LEVELS):
        for column in range(N_LEVELS):
            if row == G2 and column != G2:
                harmonic = -1
            elif row != G2 and column == G2:
                harmonic = +1
            else:
                harmonic = 0
            harmonics[..., n_f + harmonic, row, column] = rho[..., row, column]
    return harmonics


def _pump_nambu_operators(branch):
    """Return weak Nambu drive superoperators and polarization readouts."""
    if branch != -1:
        raise NotImplementedError(
            "pump-only weak response is certified only for the standard minus "
            "Raman branch; the inherited plus-branch frame fails gauge parity")
    probe_ground = G2
    conjugate_ground = G1

    probe_raise = np.zeros((N_LEVELS, N_LEVELS), dtype=complex)
    conjugate_star_lower = np.zeros_like(probe_raise)
    readouts = np.zeros((2, N_LEVELS * N_LEVELS), dtype=complex)
    for excited in EXCITED_STATES:
        probe_scale = TRANSITION_DIPOLE_SCALE[probe_ground, excited]
        conjugate_scale = TRANSITION_DIPOLE_SCALE[conjugate_ground, excited]
        probe_raise[excited, probe_ground] = 0.5 * probe_scale
        conjugate_star_lower[conjugate_ground, excited] = 0.5 * conjugate_scale
        readouts[0, excited * N_LEVELS + probe_ground] = probe_scale
        readouts[1, conjugate_ground * N_LEVELS + excited] = conjugate_scale
    drives = np.stack((comm_super(probe_raise),
                       comm_super(conjugate_star_lower)))
    return drives, readouts


def _pump_reference_system(Op_A, Op_B, Delta_eff_axis, atom, branch):
    """Build the shared state/source/readout pieces for the slow reference."""
    if branch != -1:
        _pump_nambu_operators(branch)  # raises the documented frame error
    deff = np.asarray(Delta_eff_axis, dtype=float)
    if deff.ndim != 1 or deff.size < 1 or not np.isfinite(deff).all():
        raise ValueError("Delta_eff_axis must be a finite non-empty 1-D array")
    H_pump = pump_hamiltonian_at_Deff_zero(Op_A, Op_B)
    L0 = build_liouvillian(H_pump, atom)
    L_batch = L0[None, :, :] - deff[:, None, None] * atom.S_v[None, :, :]
    rho = steady_state_batched(L0, deff, atom.S_v, N_LEVELS)
    rho_vec = rho.reshape(deff.size, N_LEVELS * N_LEVELS)
    drives, readouts = _pump_nambu_operators(branch)
    sources = np.einsum("rij,bj->bir", drives, rho_vec)
    return deff, L0, L_batch, rho, rho_vec, sources, readouts


def pump_only_weak_response_reference(
        Op_A, Op_B, delta_axis, Delta_eff_axis, *, branch=-1, atom=None,
        analysis_frequency_axis_rad_s=(0.0,), T=T_CELL):
    """Solve the self-consistent pump state and infinitesimal 2x2 response.

    ``delta_axis`` is the optical two-photon detuning and
    ``analysis_frequency_axis_rad_s`` is an independent RF/sideband analysis
    frequency.  They are never folded into one input.  In the static physical
    pump frame the Nambu sector frequency is

    ``omega_relative = -OMEGA_HF + delta + Omega_SA``

    for the supported minus Raman branch.  If ``atom`` is omitted, the reference
    uses the same temperature-dependent collisional and thermal-transit-reset
    model as :func:`compute_spectrum`.  The DC singularity is handled by the
    trace-zero stationary-subspace projection in
    :func:`gabes.core.trace_zero_liouvillian_response`.
    """
    if branch != -1:
        _pump_nambu_operators(branch)  # raises with the explicit limitation
    delta = np.atleast_1d(np.asarray(delta_axis, dtype=float))
    analysis = np.atleast_1d(
        np.asarray(analysis_frequency_axis_rad_s, dtype=float))
    if delta.ndim != 1 or delta.size < 1 or not np.isfinite(delta).all():
        raise ValueError("delta_axis must be a finite non-empty 1-D array")
    if analysis.ndim != 1 or analysis.size < 1 or not np.isfinite(analysis).all():
        raise ValueError(
            "analysis_frequency_axis_rad_s must be a finite non-empty 1-D array")
    if atom is None:
        atom = fwm.collisional_atom(T)

    (deff, _L0, L_batch, rho, rho_vec, sources, readouts) = (
        _pump_reference_system(
            Op_A, Op_B, Delta_eff_axis, atom, branch))
    relative = -OMEGA_HF + delta[None, :] + analysis[:, None]
    chi = np.empty(
        (analysis.size, delta.size, deff.size, 2, 2), dtype=complex)
    max_response_trace = 0.0
    max_response_residual = 0.0
    eye = np.eye(N_LEVELS * N_LEVELS, dtype=complex)
    diagonal = np.arange(N_LEVELS) * (N_LEVELS + 1)
    for analysis_index in range(analysis.size):
        for delta_index in range(delta.size):
            omega = float(relative[analysis_index, delta_index])
            response = trace_zero_liouvillian_response(
                L_batch, sources, omega, N_LEVELS)
            chi[analysis_index, delta_index] = np.einsum(
                "oi,bik->bok", readouts, response)
            traces = np.sum(response[:, diagonal, :], axis=1)
            max_response_trace = max(
                max_response_trace,
                float(np.max(np.abs(traces), initial=0.0)))
            residual = np.einsum(
                "bij,bjk->bik", L_batch + 1j * omega * eye, response)
            residual += sources
            scale = (
                np.linalg.norm(
                    np.einsum("bij,bjk->bik", L_batch, response), axis=1)
                + abs(omega) * np.linalg.norm(response, axis=1)
                + np.linalg.norm(sources, axis=1))
            normalized = np.linalg.norm(residual, axis=1) / np.maximum(
                scale, np.finfo(float).tiny)
            max_response_residual = max(
                max_response_residual,
                float(np.max(normalized, initial=0.0)))

    pump_residual = np.einsum("bij,bj->bi", L_batch, rho_vec)
    pump_scale = np.linalg.norm(L_batch, axis=(1, 2)) * np.linalg.norm(
        rho_vec, axis=1)
    pump_normalized = np.linalg.norm(pump_residual, axis=1) / np.maximum(
        pump_scale, np.finfo(float).tiny)
    traces = np.trace(rho, axis1=-2, axis2=-1)
    hermiticity = rho - rho.conj().swapaxes(-1, -2)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(rho)))
    diagnostics = {
        "finite": bool(np.isfinite(rho).all() and np.isfinite(chi).all()),
        "max_pump_normalized_residual": float(
            np.max(pump_normalized, initial=0.0)),
        "max_pump_trace_error": float(
            np.max(np.abs(traces - 1.0), initial=0.0)),
        "max_pump_hermiticity_error": float(
            np.max(np.abs(hermiticity), initial=0.0)),
        "minimum_pump_eigenvalue": minimum_eigenvalue,
        "max_response_normalized_residual": max_response_residual,
        "max_response_trace_error": max_response_trace,
        "velocity_classes": int(deff.size),
        "two_photon_points": int(delta.size),
        "analysis_frequency_points": int(analysis.size),
    }
    return PumpWeakResponseReference(
        branch=branch,
        delta_axis_rad_s=delta.copy(),
        analysis_frequency_axis_rad_s=analysis.copy(),
        relative_frequency_rad_s=relative,
        delta_eff_axis_rad_s=deff.copy(),
        pump_state=rho,
        chi_matrix=chi,
        diagnostics=diagnostics,
        provenance=pump_weak_response_reference_provenance(),
    )


def pump_only_weak_response_noncollinear_reference(
        Op_A, Op_B, lab_delta_axis_rad_s, one_photon_detuning_rad_s, *,
        T, pump_k_rad_m, probe_k_axis_rad_m, crossing_angle_rad,
        lab_optical_beat_axis_rad_s=None, branch=-1, atom=None,
        analysis_frequency_axis_rad_s=(0.0,), quadrature_order=24,
        cutoff_sigma=5.0):
    """Two-dimensional Maxwell reference with separated lab/atomic frequencies.

    The standard minus branch is evaluated in the certified static pump frame.
    The laboratory beat is fixed for every velocity class, while the atomic
    weak-field frequency acquires the Raman shift returned by
    :func:`gabes.doppler.noncollinear_raman_shift_rad_s`.  Passing
    ``delta_eff`` through :func:`chi_matrix_table` would be wrong because that
    finite-seed API also reconstructs the laboratory beat from its delta input.

    This is an opt-in slow reference.  It does not replace the production
    one-dimensional finite-seed gain path.
    """
    if branch != -1:
        _pump_nambu_operators(branch)  # raises the certified frame limitation
    T = float(T)
    one_photon = float(one_photon_detuning_rad_s)
    pump_k = float(pump_k_rad_m)
    theta = float(crossing_angle_rad)
    delta_lab = np.atleast_1d(np.asarray(lab_delta_axis_rad_s, dtype=float))
    analysis = np.atleast_1d(
        np.asarray(analysis_frequency_axis_rad_s, dtype=float))
    if delta_lab.ndim != 1 or delta_lab.size < 1 or not np.isfinite(delta_lab).all():
        raise ValueError(
            "lab_delta_axis_rad_s must be a finite non-empty one-dimensional array")
    if analysis.ndim != 1 or analysis.size < 1 or not np.isfinite(analysis).all():
        raise ValueError(
            "analysis_frequency_axis_rad_s must be a finite non-empty 1-D array")
    if not all(np.isfinite(value) for value in (T, one_photon, pump_k, theta)):
        raise ValueError("T, one-photon detuning, pump k, and angle must be finite")
    if T <= 0.0 or pump_k <= 0.0:
        raise ValueError("T and pump_k_rad_m must be positive")

    probe_k = np.asarray(probe_k_axis_rad_m, dtype=float)
    try:
        probe_k = np.broadcast_to(probe_k, delta_lab.shape).astype(float, copy=True)
    except ValueError as exc:
        raise ValueError(
            "probe_k_axis_rad_m must be scalar or match lab_delta_axis_rad_s") from exc
    if not np.isfinite(probe_k).all() or np.any(probe_k <= 0.0):
        raise ValueError("probe wave numbers must be finite and positive")

    if lab_optical_beat_axis_rad_s is None:
        lab_beat = seeded_sideband_beat(delta_lab, branch)
    else:
        try:
            lab_beat = np.broadcast_to(
                np.asarray(lab_optical_beat_axis_rad_s, dtype=float),
                delta_lab.shape,
            ).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError(
                "lab_optical_beat_axis_rad_s must be scalar or match the delta axis") from exc
    if not np.isfinite(lab_beat).all() or np.any(lab_beat <= 0.0):
        raise ValueError("laboratory optical beat values must be finite and positive")
    if atom is None:
        atom = fwm.collisional_atom(T)

    velocity, weights = doppler.maxwell_legendre_grid(
        T, order=quadrature_order, cutoff_sigma=cutoff_sigma)
    vx = velocity.copy()
    vz = velocity.copy()
    wx = weights.copy()
    wz = weights.copy()
    deff_z = one_photon - pump_k * vz
    (_deff, _L0, L_z, rho_z, rho_vec_z, sources_z, readouts) = (
        _pump_reference_system(Op_A, Op_B, deff_z, atom, branch))

    nz = vz.size
    nx = vx.size
    M = N_LEVELS * N_LEVELS
    L_pairs = np.broadcast_to(L_z[:, None, :, :], (nz, nx, M, M))
    source_pairs = np.broadcast_to(
        sources_z[:, None, :, :], (nz, nx, M, sources_z.shape[-1]))
    pair_weights = wz[:, None] * wx[None, :]
    shift = np.empty((delta_lab.size, nz, nx), dtype=float)
    chi_average = np.empty(
        (analysis.size, delta_lab.size, 2, 2), dtype=complex)
    diagonal = np.arange(N_LEVELS) * (N_LEVELS + 1)
    max_response_trace = 0.0
    max_response_residual = 0.0
    eye = np.eye(M, dtype=complex)

    for delta_index, probe_wave_number in enumerate(probe_k):
        shift[delta_index] = doppler.noncollinear_raman_shift_rad_s(
            vx[None, :], vz[:, None], pump_k, probe_wave_number, theta)
        for analysis_index, analysis_frequency in enumerate(analysis):
            relative = (
                -lab_beat[delta_index]
                + analysis_frequency
                + shift[delta_index]
            )
            response = trace_zero_liouvillian_response(
                L_pairs, source_pairs, relative, N_LEVELS)
            class_chi = np.einsum("om,zxmi->zxoi", readouts, response)
            chi_average[analysis_index, delta_index] = np.einsum(
                "zx,zxoi->oi", pair_weights, class_chi)

            traces = np.sum(response[..., diagonal, :], axis=-2)
            max_response_trace = max(
                max_response_trace,
                float(np.max(np.abs(traces), initial=0.0)))
            residual = np.einsum("zxij,zxjk->zxik", L_pairs, response)
            residual += 1j * relative[..., None, None] * response
            residual += source_pairs
            response_action = np.einsum(
                "zxij,zxjk->zxik", L_pairs, response)
            scale = (
                np.linalg.norm(response_action, axis=-2)
                + np.abs(relative)[..., None] * np.linalg.norm(response, axis=-2)
                + np.linalg.norm(source_pairs, axis=-2)
            )
            normalized = np.linalg.norm(residual, axis=-2) / np.maximum(
                scale, np.finfo(float).tiny)
            max_response_residual = max(
                max_response_residual,
                float(np.max(normalized, initial=0.0)))

    rho_vector = rho_z.reshape(nz, M)
    pump_residual = np.einsum("zij,zj->zi", L_z, rho_vector)
    pump_scale = np.linalg.norm(L_z, axis=(1, 2)) * np.linalg.norm(
        rho_vector, axis=1)
    pump_normalized = np.linalg.norm(pump_residual, axis=1) / np.maximum(
        pump_scale, np.finfo(float).tiny)
    pump_traces = np.trace(rho_z, axis1=-2, axis2=-1)
    pump_hermiticity = rho_z - rho_z.conj().swapaxes(-1, -2)

    quadrature_rms = []
    analytic_rms = []
    for delta_index, probe_wave_number in enumerate(probe_k):
        mean = float(np.sum(pair_weights * shift[delta_index]))
        variance = float(np.sum(
            pair_weights * (shift[delta_index] - mean)**2))
        quadrature_rms.append(np.sqrt(max(variance, 0.0)))
        analytic_rms.append(doppler.noncollinear_raman_rms_budget(
            T, pump_k, probe_wave_number, theta)["total_rms_rad_s"])

    diagnostics = {
        "finite": bool(np.isfinite(rho_z).all() and np.isfinite(chi_average).all()),
        "max_pump_normalized_residual": float(
            np.max(pump_normalized, initial=0.0)),
        "max_pump_trace_error": float(
            np.max(np.abs(pump_traces - 1.0), initial=0.0)),
        "max_pump_hermiticity_error": float(
            np.max(np.abs(pump_hermiticity), initial=0.0)),
        "minimum_pump_eigenvalue": float(np.min(np.linalg.eigvalsh(rho_z))),
        "max_response_normalized_residual": max_response_residual,
        "max_response_trace_error": max_response_trace,
        "quadrature_order_per_axis": int(vx.size),
        "velocity_pairs": int(nx * nz),
        "cutoff_sigma": float(cutoff_sigma),
        "quadrature_raman_rms_rad_s": np.asarray(quadrature_rms),
        "analytic_raman_rms_rad_s": np.asarray(analytic_rms),
        "lab_beat_velocity_invariant": True,
        "two_photon_points": int(delta_lab.size),
        "analysis_frequency_points": int(analysis.size),
    }
    return NoncollinearPumpWeakResponseReference(
        branch=branch,
        lab_delta_axis_rad_s=delta_lab.copy(),
        lab_optical_beat_axis_rad_s=lab_beat.copy(),
        analysis_frequency_axis_rad_s=analysis.copy(),
        one_photon_detuning_rad_s=one_photon,
        pump_k_rad_m=pump_k,
        probe_k_axis_rad_m=probe_k,
        crossing_angle_rad=theta,
        vx_axis_m_s=vx,
        vz_axis_m_s=vz,
        vx_weights=wx,
        vz_weights=wz,
        raman_shift_grid_rad_s=shift,
        one_photon_effective_axis_rad_s=deff_z,
        pump_state=rho_z,
        chi_matrix=chi_average,
        diagnostics=diagnostics,
        provenance=noncollinear_doppler_reference_provenance(),
    )


def pump_only_pole_residue_reference(
        Op_A, Op_B, delta, Delta_eff, *, branch=-1, atom=None,
        analysis_frequency_rad_s=0.0, T=T_CELL):
    """One-velocity pole/residue audit of the pump-only Nambu response.

    Omitting ``atom`` selects the production collisional/transit-reset model at
    ``T``; an explicit atom keeps archived or adversarial audits reproducible.
    """
    if atom is None:
        atom = fwm.collisional_atom(T)
    (deff, _L0, L_batch, rho, rho_vec, sources, readouts) = (
        _pump_reference_system(
            Op_A, Op_B, np.asarray([Delta_eff], dtype=float), atom, branch))
    carrier_relative = -OMEGA_HF + float(delta)
    omega = carrier_relative + float(analysis_frequency_rad_s)
    result = liouvillian_pole_residue_response(
        L_batch[0], sources[0], readouts, omega)
    result.update({
        "branch": branch,
        "delta_rad_s": float(delta),
        "delta_eff_rad_s": float(deff[0]),
        "analysis_frequency_rad_s": float(analysis_frequency_rad_s),
        "carrier_relative_frequency_rad_s": carrier_relative,
        "resolvent_frequency_rad_s": omega,
        "analysis_pole_centers_rad_s": (
            -carrier_relative - np.imag(result["eigenvalues"])),
        "pump_state": rho[0],
    })
    return result


def static_hamiltonian_at_Deff_zero(Op_A, Op_B, Os, delta, branch):
    """H₀ with Δ_eff = 0, so the only v / Δ_eff dependence is added later."""
    H0 = np.zeros((N_LEVELS, N_LEVELS), dtype=complex)
    H0[G2, G2] = delta
    H0[E2, E2] = -fwm.OMEGA_EXCITED_HF
    H0[E3, E3] = 0.0
    if branch == -1:
        _add_static_drive(H0, G1, Op_A)
        _add_static_drive(H0, G2, Os)
        return H0
    if branch == +1:
        _add_static_drive(H0, G1, Os)
        _add_static_drive(H0, G2, Op_B)
        return H0
    raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")


def sideband_hamiltonian(Op_A, Op_B, Oc, branch):
    Hp = np.zeros((N_LEVELS, N_LEVELS), dtype=complex)
    if branch == -1:
        _add_sideband_drive(Hp, G1, Oc)
        _add_sideband_drive(Hp, G2, Op_B)
        return Hp
    if branch == +1:
        _add_sideband_drive(Hp, G1, Op_A)
        _add_sideband_drive(Hp, G2, Oc)
        return Hp
    raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")


def sideband_template(Op_A, Op_B, Oc, branch):
    Hp = sideband_hamiltonian(Op_A, Op_B, Oc, branch)
    Cp = comm_super(Hp)
    Cm = comm_super(Hp.conj().T)
    return Cp, Cm


# =========================================================
# χ-matrix table for one fixed atom/dissipator on (δ, Δ_eff)
# =========================================================
def _coherence_weights(ground):
    """w such that Σ_k w[k]·vec(ρ)[k] = Σ_e scale[g,e]·ρ[e,g] (vec row-major)."""
    w = np.zeros(N_LEVELS * N_LEVELS, dtype=complex)
    for e in EXCITED_STATES:
        w[e * N_LEVELS + ground] = TRANSITION_DIPOLE_SCALE[ground, e]
    return w


def chi_matrix_table(Op_A, Op_B, Os_ref, Oc_ref, delta_axis, Delta_eff_axis, branch,
                     atom=ATOM, n_f=1):
    """
    Two finite-Floquet solves per probe-detuning point to extract
    (chi_ss, chi_cs, chi_sc, chi_cc) on a 2-D (delta, Delta_eff) grid.
    ``n_f`` selects the retained harmonics ``-n_f,...,+n_f``.

    `atom` carries the level scheme and its dissipator; pass a temperature-
    dependent model (collisional ground/optical dephasing) to fold density-
    dependent decoherence into the solve. Defaults to the natural-linewidth
    module `ATOM` so the kernel/NumPy regression callers are unchanged.

    With numba available the whole grid runs in one fused compiled continued-
    fraction kernel (`kernels.floquet_chi_grid`); the NumPy delta-loop below is
    the fallback and reference implementation.  Both paths support ``n_f>=1``.
    """
    n_f = int(n_f)
    if n_f < 1:
        raise ValueError("n_f must be at least 1")
    probe_ground = G2 if branch == -1 else G1
    conj_ground = G1 if branch == -1 else G2
    n_d = delta_axis.size
    n_de = Delta_eff_axis.size

    Cp_no_c, Cm_no_c = sideband_template(Op_A, Op_B, 0.0, branch)      # solve 1
    Cp_c, Cm_c = sideband_template(Op_A, Op_B, Oc_ref, branch)        # solve 2

    if kernels.available():
        # H₀(δ) is affine in δ (only H₀[G2,G2] = δ), so L₀(δ) = L₀(0) + δ·C_δ.
        E_g2 = np.zeros((N_LEVELS, N_LEVELS), dtype=complex)
        E_g2[G2, G2] = 1.0
        C_delta = comm_super(E_g2)
        w_probe = _coherence_weights(probe_ground)
        w_conj = _coherence_weights(conj_ground)
        delta_axis = np.ascontiguousarray(delta_axis, dtype=float)
        deff_axis = np.ascontiguousarray(Delta_eff_axis, dtype=float)

        L0_1 = build_liouvillian(
            fwm.static_hamiltonian_at_Deff_zero(Op_A, Op_B, Os_ref, 0.0, branch), atom)
        probe_a, conj_a = kernels.floquet_chi_grid(
            L0_1, C_delta, atom.S_v, Cp_no_c, Cm_no_c, delta_axis, deff_axis,
            OMEGA_HF, float(branch), w_probe, w_conj, N_LEVELS, n_f)

        L0_2 = build_liouvillian(
            fwm.static_hamiltonian_at_Deff_zero(Op_A, Op_B, 0.0, 0.0, branch), atom)
        probe_b, conj_b = kernels.floquet_chi_grid(
            L0_2, C_delta, atom.S_v, Cp_c, Cm_c, delta_axis, deff_axis,
            OMEGA_HF, float(branch), w_probe, w_conj, N_LEVELS, n_f)

        return probe_a / Os_ref, conj_a / Os_ref, probe_b / Oc_ref, conj_b / Oc_ref

    chi_ss = np.zeros((n_d, n_de), dtype=complex)
    chi_cs = np.zeros((n_d, n_de), dtype=complex)
    chi_sc = np.zeros((n_d, n_de), dtype=complex)
    chi_cc = np.zeros((n_d, n_de), dtype=complex)

    for i, delta in enumerate(delta_axis):
        Omega_beat = seeded_sideband_beat(delta, branch)

        # ---- Solve 1: probe drive only ----
        H0_1 = fwm.static_hamiltonian_at_Deff_zero(Op_A, Op_B, Os_ref, delta, branch)
        L0_1 = build_liouvillian(H0_1, atom)
        rho0_a, rhop_a = floquet_solve_truncated(
            L0_1, Cp_no_c, Cm_no_c, Omega_beat, Delta_eff_axis,
            atom.S_v, N_LEVELS, n_f)
        probe_a = _polarization_coherence(rho0_a, probe_ground)
        conj_a = _polarization_coherence(rhop_a, conj_ground)
        chi_ss[i] = probe_a / Os_ref
        chi_cs[i] = conj_a / Os_ref

        # ---- Solve 2: conjugate seed only ----
        H0_2 = fwm.static_hamiltonian_at_Deff_zero(Op_A, Op_B, 0.0, delta, branch)
        L0_2 = build_liouvillian(H0_2, atom)
        rho0_b, rhop_b = floquet_solve_truncated(
            L0_2, Cp_c, Cm_c, Omega_beat, Delta_eff_axis,
            atom.S_v, N_LEVELS, n_f)
        probe_b = _polarization_coherence(rho0_b, probe_ground)
        conj_b = _polarization_coherence(rhop_b, conj_ground)
        chi_sc[i] = probe_b / Oc_ref
        chi_cc[i] = conj_b / Oc_ref

    return chi_ss, chi_cs, chi_sc, chi_cc


def _seeded_pole_systems(Op_A, Op_B, Os_ref, Oc_ref, branch, atom, orders, *,
                         return_real_forms=False):
    """Real pole–residue systems for both finite seed configurations and orders.

    Returns ``{(seed, order): (AffineShiftSystem, reference Rabi frequency)}``
    with readouts (probe coherence at harmonic 0, conjugate coherence at
    harmonic +1), or ``None`` when the exact Hermiticity/adjoint identities of
    the real finite-Floquet form do not hold.  The identities are compared
    exactly in raw vec coordinates, as in ``kernels.floquet_chi_grid``.
    """
    n = N_LEVELS
    perm = _transpose_perm(n)
    U = hermitian_basis(n)
    Uh = U.conj().T
    E_g2 = np.zeros((n, n), dtype=complex)
    E_g2[G2, G2] = 1.0
    C_delta = comm_super(E_g2)
    probe_ground = G2 if branch == -1 else G1
    conj_ground = G1 if branch == -1 else G2
    w_probe = _coherence_weights(probe_ground) @ U
    w_conj = _coherence_weights(conj_ground) @ U

    def j_symmetric(X):
        return np.array_equal(X, X.conj()[perm][:, perm])

    seeds = {
        1: (build_liouvillian(
                fwm.static_hamiltonian_at_Deff_zero(Op_A, Op_B, Os_ref, 0.0, branch), atom),
            sideband_template(Op_A, Op_B, 0.0, branch), Os_ref),
        2: (build_liouvillian(
                fwm.static_hamiltonian_at_Deff_zero(Op_A, Op_B, 0.0, 0.0, branch), atom),
            sideband_template(Op_A, Op_B, Oc_ref, branch), Oc_ref),
    }
    if not (j_symmetric(C_delta) and j_symmetric(atom.S_v)):
        return None
    S_r = np.ascontiguousarray((Uh @ atom.S_v @ U).real)
    Cd_r = np.ascontiguousarray((Uh @ C_delta @ U).real)
    systems = {}
    for seed, (L0, (Cp, Cm), reference) in seeds.items():
        if not (j_symmetric(L0) and np.array_equal(Cm, Cp.conj()[perm][:, perm])):
            return None
        L_r = np.ascontiguousarray((Uh @ L0 @ U).real)
        Cp_r = Uh @ Cp @ U
        for order in orders:
            form = pole_doppler.floquet_real_form(
                L_r, Cd_r, Cp_r, S_r, order, OMEGA_HF, float(branch), n,
                ((0, w_probe), (1, w_conj)))
            systems[(seed, order)] = (form if return_real_forms else
                                      pole_doppler.AffineShiftSystem(*form), reference)
    return systems


def _finite_seed_angular_rows(forms, delta, one_photon, T, pump_k, probe_k, theta,
                              transverse_order, transverse_cutoff):
    """Conditional analytic longitudinal Maxwell mean for each transverse node."""
    vx, weights = doppler.maxwell_legendre_grid(T, order=transverse_order,
                                               cutoff_sigma=transverse_cutoff)
    sigma_v = np.sqrt(constants.KB * T / constants.MASS_85RB)
    rows = {key: np.empty((len(delta), 2), dtype=complex) for key in forms}
    theta = float(theta)
    for row, (lab_delta, k_probe) in enumerate(zip(delta, probe_k)):
        # D = Delta - k_p v_z. The atom-frame Raman detuning is t-rD;
        # (L0 + t A1) - D(S+rA1) includes the corresponding atom-frame beat.
        ratio = (pump_k - k_probe * np.cos(theta)) / pump_k
        t = lab_delta + ratio * one_photon - k_probe * np.sin(theta) * vx
        for key, ((A0, A1, S, rhs, readout), reference) in forms.items():
            system = pole_doppler.ShiftInvertSystem(A0, A1, S + ratio * A1, rhs, readout)
            residues = system.residues(t, one_photon)
            mean = residues.gaussian_mean(one_photon, pump_k * sigma_v)
            rows[key][row] = (weights @ mean) / reference
    return rows


def finite_seed_angular_reference(
        Op_A, Op_B, Os_ref, Oc_ref, lab_delta_axis_rad_s, one_photon_rad_s, *,
        T, pump_k_rad_m, probe_k_axis_rad_m, crossing_angle_rad, atom=None,
        floquet_order=SEEDED_FLOQUET_ORDER, transverse_order=18, cutoff_sigma=5.0,
        branch=-1):
    """Bounded finite-seed minus-branch mean field with transverse Raman shift.

    Longitudinal averaging is analytic at each transverse quadrature node.
    The static Raman diagonal and the internal harmonic frequency receive
    opposite velocity shifts. Their sum keeps the pump-only gauge invariant.
    Laboratory beat and RF analysis frequency are separate, and this steady
    mean-field calculation evaluates the carrier (zero RF offset) only.
    """
    if branch != -1:
        raise NotImplementedError("finite-seed angular Doppler supports only the minus Raman branch")
    delta = np.atleast_1d(np.asarray(lab_delta_axis_rad_s, dtype=float))
    probe_k = np.broadcast_to(np.asarray(probe_k_axis_rad_m, dtype=float), delta.shape)
    if (delta.ndim != 1 or not 1 <= delta.size <= 121 or not np.isfinite(delta).all()
            or not np.isfinite(probe_k).all() or np.any(probe_k <= 0)
            or not np.isfinite(pump_k_rad_m) or pump_k_rad_m <= 0
            or not np.isfinite(one_photon_rad_s)
            or not np.isfinite(crossing_angle_rad) or abs(crossing_angle_rad) > np.deg2rad(2.0)
            or transverse_order not in range(8, 33) or not 4.0 <= cutoff_sigma <= 6.0
            or floquet_order not in (3, 4, 5)):
        raise ValueError("angular reference requires 1–121 rows, angle at most 2 degrees, "
                         "8–32 transverse nodes, a 4–6 sigma cutoff and Floquet order 3–5")
    if not all(np.isfinite(ref) and ref > 0 for ref in (Os_ref, Oc_ref)):
        raise ValueError("finite-seed angular reference requires positive reference Rabi frequencies")
    if atom is None:
        atom = fwm.collisional_atom(T)
    orders = (floquet_order, floquet_order - 1)
    forms = fwm._seeded_pole_systems(Op_A, Op_B, Os_ref, Oc_ref, branch, atom, orders,
                                return_real_forms=True)
    if forms is None:
        raise ValueError("angular reference requires a Hermiticity-preserving seeded model")
    rows = _finite_seed_angular_rows(forms, delta, one_photon_rad_s, T, pump_k_rad_m,
                                     probe_k, crossing_angle_rad, transverse_order, cutoff_sigma)
    responses = {}
    for order in orders:
        probe, conjugate = rows[(1, order)], rows[(2, order)]
        responses[order] = (probe[:, 0], probe[:, 1], conjugate[:, 0], conjugate[:, 1])
    if not all(np.isfinite(response).all() for parts in responses.values() for response in parts):
        raise FloatingPointError("non-finite finite-seed angular response")
    budgets = [doppler.noncollinear_raman_rms_budget(T, pump_k_rad_m, k, crossing_angle_rad)
               for k in probe_k]
    return {
        "responses": responses,
        "lab_two_photon_detuning_rad_s": delta,
        "lab_optical_beat_rad_s": seeded_sideband_beat(delta, branch),
        "analysis_frequency_rad_s": 0.0,
        "one_photon_lab_rad_s": float(one_photon_rad_s),
        **{f"{component}_raman_rms_hz": np.array([budget[f"{component}_rms_hz"] for budget in budgets])
           for component in ("axial", "transverse", "total")},
        "transverse_order": int(transverse_order),
        "transverse_cutoff_sigma": float(cutoff_sigma),
        "longitudinal_measure": "analytic conditional untruncated Maxwell",
        "scope": "minus Raman branch, finite seed, carrier mean field; RMS is a velocity-distribution scale",
    }


def _pole_scan_response(systems, *, orders, delta_axis, probe_axis_GHz, Delta,
                        gaussian_sigma=None,
                        Delta_eff_axis, idx_lo, frac, weights, adaptive, score,
                        tolerance_dB, tolerance_log10_gain, start_stride,
                        guard, grid_rows):
    """Maxwell-averaged (ss, cs, sc, cc) responses from pole–residue rows.

    The untruncated measure uses the analytic Gaussian pole mean. The
    historical measure spreads the truncated Maxwell weights onto the same
    Δ_eff grid nodes by linear interpolation. With ``adaptive`` the probe axis is refined by
    bisection, the averaged complex responses are spline-interpolated between
    solved rows, and refinement is judged in the readout (``score``).

    After refinement every solved row is checked once against the compiled
    kernel at Δ_eff = Δ.  A row that misses ``POLE_GUARD_RTOL`` is replaced by
    the grid solution for that row and, in adaptive mode, refinement is re-run
    on the corrected values (solved rows are memoized).
    """
    n_points = probe_axis_GHz.size
    node_weights = np.zeros(Delta_eff_axis.size)
    np.add.at(node_weights, idx_lo, weights * (1.0 - frac))
    np.add.at(node_weights, idx_lo + 1, weights * frac)
    high_order = orders[0]
    keys = [(seed, order) for order in orders for seed in (1, 2)]
    stored = {order: np.zeros((n_points, 4), dtype=complex) for order in orders}
    poles = {}
    solved = np.zeros(n_points, dtype=bool)
    guarded = np.zeros(n_points, dtype=bool)
    exact_scores = {}
    ledger = {"guard": 0.0, "fallback": 0, "min_imag": math.inf}

    def solve(rows):
        rows = np.asarray(rows, dtype=int)
        rows = rows[~solved[rows]]
        if rows.size == 0:
            return
        batch = pole_doppler.residues_many(
            [(systems[key][0], delta_axis[rows]) for key in keys])
        means = {}
        for key, result in zip(keys, batch):
            ledger["min_imag"] = min(ledger["min_imag"], result.min_abs_imag)
            if key not in poles:
                poles[key] = tuple(
                    np.zeros((n_points,) + arr.shape[1:], dtype=complex)
                    for arr in (result.lam, result.res, result.c0))
            for store, arr in zip(poles[key], (result.lam, result.res, result.c0)):
                store[rows] = arr
            if gaussian_sigma is None:
                mean = result.discrete_mean(Delta_eff_axis, node_weights)
            else:
                mean = result.gaussian_mean(Delta, gaussian_sigma)
            means[key] = mean / systems[key][1]
        for order in orders:
            probe_seed, conj_seed = means[(1, order)], means[(2, order)]
            stored[order][rows] = np.column_stack(
                (probe_seed[:, 0], probe_seed[:, 1], conj_seed[:, 0], conj_seed[:, 1]))
        solved[rows] = True

    def check_guard(rows):
        rows = np.asarray(rows, dtype=int)
        rows = rows[~guarded[rows]]
        if rows.size == 0:
            return rows
        failed = np.zeros(rows.size, dtype=bool)
        for order in orders:
            values = []
            for seed in (1, 2):
                lam, res, c0 = (store[rows] for store in poles[(seed, order)])
                values.append(pole_doppler.PoleResidues(lam, res, c0).value_at(Delta)
                              / systems[(seed, order)][1])
            at_delta = np.column_stack(
                (values[0][:, 0], values[0][:, 1], values[1][:, 0], values[1][:, 1]))
            kernel = np.column_stack([table[:, 0] for table in guard(rows, order)])
            relative = np.abs(at_delta - kernel) / np.maximum(
                np.abs(kernel), np.finfo(float).tiny)
            relative = np.where(np.isfinite(relative), relative, np.inf)
            ledger["guard"] = max(ledger["guard"], float(np.max(relative)))
            failed |= np.any(relative > fwm.POLE_GUARD_RTOL, axis=1)
        guarded[rows] = True
        bad = rows[failed]
        if bad.size:
            for order in orders:
                stored[order][bad] = np.column_stack(grid_rows(bad, order))
            for r in bad:
                exact_scores.pop(int(r), None)
            ledger["fallback"] += int(bad.size)
        return bad

    def disagrees(nodes, candidates):
        candidates = np.asarray(candidates, dtype=int)
        predicted = adaptive_scan.cubic_spline(
            probe_axis_GHz[nodes], stored[high_order][nodes], probe_axis_GHz[candidates])
        need = np.asarray([r for r in np.flatnonzero(solved)
                           if int(r) not in exact_scores], dtype=int)
        rows = np.concatenate((candidates, need))
        chi = np.vstack((predicted, stored[high_order][need]))
        G_all, S_all = score(tuple(chi[:, q] for q in range(4)), rows)
        split = candidates.size
        for r, g_value, s_value in zip(need, G_all[split:], S_all[split:]):
            exact_scores[int(r)] = (float(g_value), float(s_value))
        exact = np.array([exact_scores[int(r)] for r in candidates], dtype=float)
        known = np.array(list(exact_scores.values()), dtype=float)
        log_known = np.log10(np.maximum(known[:, 0], 1e-300))
        tol_S = max(min(float(tolerance_dB),
                        SCAN_TOLERANCE_RANGE_FRACTION * float(np.ptp(known[:, 1]))),
                    SCAN_TOLERANCE_FLOOR_DB)
        tol_G = max(min(float(tolerance_log10_gain),
                        SCAN_TOLERANCE_RANGE_FRACTION * float(np.ptp(log_known))),
                    SCAN_TOLERANCE_FLOOR_LOG10_GAIN)
        with np.errstate(divide="ignore", invalid="ignore"):
            dG = np.abs(np.log10(np.maximum(G_all[:split], 1e-300))
                        - np.log10(np.maximum(exact[:, 0], 1e-300)))
            dS = np.abs(S_all[:split] - exact[:, 1])
        return (dS > tol_S) | (dG > tol_G) | ~np.isfinite(dS) | ~np.isfinite(dG)

    for _attempt in range(3):
        if adaptive:
            nodes, rounds = adaptive_scan.refine_scan_nodes(
                n_points, solve, disagrees, start_stride=start_stride,
                x=probe_axis_GHz)
        else:
            nodes, rounds = np.arange(n_points), 1
            solve(nodes)
        if check_guard(nodes).size == 0 or not adaptive:
            break
    interpolated = nodes.size < n_points

    def display(order):
        if not interpolated:
            return stored[order]
        full = adaptive_scan.cubic_spline(
            probe_axis_GHz[nodes], stored[order][nodes], probe_axis_GHz)
        full[nodes] = stored[order][nodes]
        return full

    def columns(array):
        return tuple(np.ascontiguousarray(array[:, q]) for q in range(4))

    high = columns(display(high_order))
    low = columns(display(orders[1])) if len(orders) > 1 else None
    estimator = {
        "method": RESPONSE_POLE_ADAPTIVE if adaptive else RESPONSE_POLE,
        "velocity_average": (
            "analytic untruncated Maxwell pole mean" if gaussian_sigma is not None
            else "exact pole-residue sum over the truncated Maxwell nodes"),
        "solved_probe_points": int(nodes.size),
        "display_probe_points": int(n_points),
        "solved_probe_indices": tuple(int(i) for i in nodes),
        "interpolation": (
            "not-a-knot cubic spline of Maxwell-averaged complex responses"
            if interpolated else "none"),
        "adaptive_rounds": int(rounds),
        "scan_tolerance_dB": float(tolerance_dB) if adaptive else None,
        "scan_tolerance_log10_gain": float(tolerance_log10_gain) if adaptive else None,
        "scan_start_stride": int(start_stride) if adaptive else None,
        "scan_start_rule": (
            "starting nodes no farther apart than the stride times the finest "
            "display spacing that persists over four steps; coarse samples of a "
            "mixed axis are all solved" if adaptive else None),
        "audit_scope": (
            "solved probe detunings; displayed points between them are interpolated"
            if interpolated else "every displayed probe detuning"),
        "pole_guard_rtol": fwm.POLE_GUARD_RTOL,
        "pole_guard_max_relative": float(ledger["guard"]),
        "pole_guard_fallback_rows": int(ledger["fallback"]),
        "pole_min_distance_from_real_axis_MHz": float(
            ledger["min_imag"] / (2 * np.pi * 1e6)),
        "absorption_diagnostics": "analytic Voigt (Faddeeva), untruncated Maxwell",
    }
    return high, low, estimator


class ChiPoleRows:
    """Maxwell sums of :func:`chi_matrix_table` rows in pole–residue form."""

    def __init__(self, probe_seed, conj_seed, references, guard_max_relative=None):
        self._rows = (probe_seed, conj_seed)
        self._references = references
        self.guard_max_relative = guard_max_relative

    def mean(self, nodes, weights):
        """``(chi_ss, chi_cs, chi_sc, chi_cc)`` equal to ``table[:, j] @ weights``
        for Δ_eff samples ``nodes`` [rad/s], without a Δ_eff table."""
        nodes = np.ascontiguousarray(nodes, dtype=float)
        weights = np.ascontiguousarray(weights, dtype=float)
        probe = self._rows[0].discrete_mean(nodes, weights) / self._references[0]
        conj = self._rows[1].discrete_mean(nodes, weights) / self._references[1]
        return probe[:, 0], probe[:, 1], conj[:, 0], conj[:, 1]

    def gaussian_mean(self, center, sigma):
        """Evaluate the untruncated Maxwell mean in the same response order."""
        probe = self._rows[0].gaussian_mean(center, sigma) / self._references[0]
        conj = self._rows[1].gaussian_mean(center, sigma) / self._references[1]
        return probe[:, 0], probe[:, 1], conj[:, 0], conj[:, 1]


def chi_pole_rows(Op_A, Op_B, Os_ref, Oc_ref, delta_axis, branch, atom=ATOM, n_f=1,
                  guard_delta_eff=None):
    """Pole–residue form of :func:`chi_matrix_table` for velocity sums.

    One eigendecomposition per probe detuning replaces one Floquet solve per
    (probe detuning, Δ_eff) sample; :meth:`ChiPoleRows.mean` then evaluates any
    set of Δ_eff nodes exactly.  With ``guard_delta_eff`` every row is compared
    with the table solve at that single Δ_eff.  Returns ``None`` when the real
    finite-Floquet identities fail or a row misses ``POLE_GUARD_RTOL``; callers
    then use the table.
    """
    n_f = int(n_f)
    delta_axis = np.ascontiguousarray(delta_axis, dtype=float)
    systems = fwm._seeded_pole_systems(Op_A, Op_B, Os_ref, Oc_ref, branch, atom, (n_f,))
    if systems is None:
        return None
    keys = ((1, n_f), (2, n_f))
    probe_seed, conj_seed = pole_doppler.residues_many(
        [(systems[key][0], delta_axis) for key in keys])
    rows = ChiPoleRows(probe_seed, conj_seed, tuple(systems[key][1] for key in keys))
    if guard_delta_eff is not None:
        node = np.array([float(guard_delta_eff)])
        worst = 0.0
        for value, table in zip(
                rows.mean(node, np.ones(1)),
                fwm.chi_matrix_table(Op_A, Op_B, Os_ref, Oc_ref, delta_axis, node, branch,
                                 atom=atom, n_f=n_f)):
            reference = table[:, 0]
            relative = np.abs(value - reference) / np.maximum(
                np.abs(reference), np.finfo(float).tiny)
            worst = max(worst, float(np.max(np.where(np.isfinite(relative), relative, np.inf))))
        if not worst <= fwm.POLE_GUARD_RTOL:
            return None
        rows.guard_max_relative = worst
    return rows
