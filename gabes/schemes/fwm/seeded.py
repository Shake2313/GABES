"""Seeded spectrum calculation, propagation and scan entry points."""
from .. import fwm
from .common import (
    BRANCHES, DEFAULT_BRANCH, G1, G2, GROUND_F, HARDENED_PUMP_SCATTER_KAPPA, LOSS_FRAC, L_CELL,
    LEGACY_TRANSITION_DIPOLE_SCALE,
    PHASE_BALANCED, PHASE_FINE, PHASE_LEGACY, PHASE_ULTRA, PUMP_OVERLAP_EXCLUSION_MHZ, P_PROBE,
    P_PUMP, QE_DETECTOR, RESONANCE_WINDOW_MHZ, RESPONSE_GRID, RESPONSE_METHODS,
    RESPONSE_POLE_ADAPTIVE, SCAN_COARSE_POINTS, SCAN_FINE_POINTS, SCAN_MAX_GHZ, SCAN_MIN_GHZ,
    SCAN_START_STRIDE, SCAN_TOLERANCE_DB, SCAN_TOLERANCE_LOG10_GAIN, SEEDED_FLOQUET_ORDER,
    SEEDED_PHASE_ANGLE_DEG, SEEDED_REFERENCE_RESIDUAL, SeededCouplingFactors, T_CELL,
    ULTRA_PROPAGATION_SEGMENTS, VELOCITY_CUTOFF_SIGMA, VELOCITY_STEP_MPS, W_PROBE, W_PUMP,
    noncollinear_doppler_reference_provenance, physical_coupling_ledger,
    pump_weak_response_reference_provenance, seeded_atomic_solver_provenance,
    seeded_parameter_provenance, seeded_validation_claim_gate,
)
from ...constants import K_VEC, rabi_freq
from ...core import _EXP_ARG_CLAMP, blas_single_thread
from .phase_matching import (
    _gaussian_overlap_profile, seeded_option_a_wavenumbers, seeded_phase_mismatch_z,
)
from .references import LINEAR_ABSORPTION_MODEL, _zeeman_consistency_cached
from ... import (
    constants, doppler, fwm_gain_closure, fwm_stark_response, gain_reference, hyperfine, kernels,
    observables,
)
import copy
import math
import numpy as np
from ...fwm_efficiency import dressed_cell_transmission
from .seeded_response import finite_seed_angular_reference, pump_hamiltonian_at_Deff_zero
from ...passive_readout import passive_l1_from_spectrum


# =========================================================
# Probe-detuning axis
# =========================================================
def branch_center_GHz(Delta_GHz, branch):
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")
    return Delta_GHz + branch * constants.NU_HF / 1e9


def probe_scan_axis_GHz(Delta_GHz, coarse_points=None, fine_points=None,
                        window_mhz=None, scan_min=None, scan_max=None,
                        branches=BRANCHES):
    """Probe-frequency scan axis [GHz], referenced to the 85Rb F=2 → F'=3 line."""
    coarse_points = SCAN_COARSE_POINTS if coarse_points is None else coarse_points
    fine_points = SCAN_FINE_POINTS if fine_points is None else fine_points
    window_mhz = RESONANCE_WINDOW_MHZ if window_mhz is None else window_mhz
    scan_min = SCAN_MIN_GHZ if scan_min is None else scan_min
    scan_max = SCAN_MAX_GHZ if scan_max is None else scan_max

    coarse = np.linspace(scan_min, scan_max, coarse_points)
    half_window = window_mhz * 1e-3
    parts = [coarse]
    for branch in branches:
        center = branch_center_GHz(Delta_GHz, branch)
        fmin = max(scan_min, center - half_window)
        fmax = min(scan_max, center + half_window)
        if fmin < fmax:
            parts.append(np.linspace(fmin, fmax, fine_points))
    axis = np.unique(np.concatenate(parts))
    exclusion_GHz = PUMP_OVERLAP_EXCLUSION_MHZ * 1e-3
    return axis[np.abs(axis - Delta_GHz) > exclusion_GHz]


def two_photon_detuning_from_probe_scan(probe_GHz, Delta_GHz, branch):
    delta_Hz = (probe_GHz - branch_center_GHz(Delta_GHz, branch)) * 1e9
    return 2 * np.pi * delta_Hz


def seeded_linear_gain(
        response, k_probe, k_conj, length, density, line_strength, *,
        collected_modes=None, delta_k_z=None, propagation_segments=1,
        segment_profile=None):
    """Propagate averaged (ss, cs, sc, cc) susceptibility and return arm gains.

    With `collected_modes=(omega_probe, omega_conj, area_probe, area_conj)`,
    gains use the collected power convention and include canonical diagnostics.
    Omit it to obtain the field-power coefficients used by the historical
    delta-T scanner. The transfer always maps electric-field amplitudes.
    """
    ss, cs, sc, cc = response
    probe, conjugate, transfer = observables.gain_from_chi(
        ss, sc, cs, cc, k_probe, k_conj, length, density,
        line_strength=line_strength, delta_k_z=delta_k_z,
        propagation_segments=propagation_segments, segment_profile=segment_profile)
    canonical = None
    if collected_modes is not None:
        canonical = observables.canonical_transfer_diagnostics(transfer, *collected_modes)
        probe = canonical["probe_power_gain"]
        conjugate = canonical["conjugate_power_gain"]
    return probe, conjugate, transfer, canonical


def seeded_balanced_gain_noise_dB(
        G_s, G_c, eta_s, eta_c, *, seed_excess_noise=0.0, detected_excess_noise=None):
    """Gain diagnostic with explicit arm efficiencies and noise reference plane.

    Both legacy excess-noise keywords add PSD relative to the detected shot
    noise. The engine uses this convention for its pump-scatter diagnostic.
    """
    noise = observables.balanced_twin_beam_noise(
        G_s, G_c, eta_s, eta_c, reference_weight="dc",
        reference_excess_noise=seed_excess_noise)
    if detected_excess_noise is not None:
        noise = noise + detected_excess_noise
    return 10.0 * np.log10(np.maximum(noise, 1e-30))


def _uniform_segment_profile_and_probe_od(
        chi_bar, N_atoms, line_strength, nseg, L=L_CELL):
    """Return a neutral coupling profile plus a probe-OD diagnostic.

    ``chi_bar`` already supplies the probe's local diagonal attenuation in the
    Maxwell matrix.  It cannot also be used as a pump-depletion law for the
    off-diagonal FWM coupling.  Until a pump-frequency propagation solve is
    available, keep that coupling uniform and report the probe OD separately.
    """
    chi = observables.chi_phys(chi_bar, N_atoms, line_strength=line_strength)
    alpha = np.maximum(K_VEC * np.imag(chi), 0.0)
    od = float(np.nanmedian(alpha) * L) if alpha.size else 0.0
    od = max(od, 0.0)
    return np.ones(int(nseg), dtype=float), od


def _ultra_segmented_gain(chi_ss_avg, chi_sc_avg, chi_cs_avg, chi_cc_avg,
                          k_probe, k_conj, L, N_atoms, line_strength,
                          delta_k_z, segment_profile, spatial_profile,
                          P_pump, P_seed, conjugate_power_ratio=1.0):
    """Segmented propagation with approximate dynamic pump depletion.

    Diagonal susceptibility already attenuates each field.  ``segment_profile``
    may scale only independently known pump/crossing overlap; a probe-absorption
    estimate must not be recycled as an off-diagonal pump profile or as a second
    post-source loss.
    """
    nseg = int(segment_profile.size)
    dz = L / max(nseg, 1)
    M = observables._gain_matrix_from_chi(
        chi_ss_avg, chi_sc_avg, chi_cs_avg, chi_cc_avg,
        k_probe, k_conj, N_atoms, constants.DIPOLE_D1, line_strength,
        delta_k_z=delta_k_z)
    n = M.shape[0]
    amp = np.zeros((n, 2), dtype=complex)
    amp[:, 0] = math.sqrt(max(float(P_seed), 1e-30))
    T_total = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
    pump_remaining = np.full(n, max(float(P_pump), 1e-30), dtype=float)

    for coupling_scale, spatial_scale in zip(segment_profile, spatial_profile):
        pump_scale = np.sqrt(np.clip(pump_remaining / max(float(P_pump), 1e-30),
                                     0.0, 1.0))
        Mz = M.copy()
        scale = float(coupling_scale) * float(spatial_scale) * pump_scale
        Mz[:, 0, 1] *= scale
        Mz[:, 1, 0] *= scale
        Tseg = observables.matrix_exp_2x2(Mz, dz)
        amp = np.einsum("nij,nj->ni", Tseg, amp)
        T_total = Tseg @ T_total
        seed_added = np.maximum(np.abs(amp[:, 0]) ** 2 - float(P_seed), 0.0)
        conj_power = np.maximum(
            np.abs(amp[:, 1]) ** 2 * float(conjugate_power_ratio), 0.0)
        pump_remaining = np.maximum(float(P_pump) - seed_added - conj_power, 0.0)

    G_s = np.abs(amp[:, 0]) ** 2 / max(float(P_seed), 1e-30)
    G_c = np.abs(amp[:, 1]) ** 2 / max(float(P_seed), 1e-30)
    return G_s, G_c, T_total, pump_remaining


def _single_branch(branch, branches):
    """Resolve one physical FWM channel; do not merge distinct Raman branches."""
    if branches is None:
        if branch not in BRANCHES:
            raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")
        return branch

    branches = tuple(branches)
    if len(branches) != 1:
        raise ValueError(
            "FWM Raman branches are separate probe/conjugate mode pairs; "
            "compute them one at a time instead of summing susceptibilities."
        )
    only = branches[0]
    if only not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {only}")
    return only


# =========================================================
# High-level spectrum  (one call → gain + gain-referred diagnostic curves)
# =========================================================
def compute_spectrum(D_GHz, *,
                     T=T_CELL, P_pump=P_PUMP, P_probe=P_PROBE,
                     w_pump=W_PUMP, w_probe=W_PROBE, w_conj=None,
                     line_strength=None,
                     mode_overlap_penalty=1.0, polarization_penalty=1.0,
                     zeeman_participation_penalty=1.0,
                     loss_frac=LOSS_FRAC, qe=QE_DETECTOR,
                     detection_efficiency=None,
                     L=L_CELL,
                     coarse_points=None, fine_points=None, window_mhz=None,
                     scan_min=None, scan_max=None, extra_probe_points_GHz=(),
                     velocity_step=None, velocity_cutoff=None,
                     velocity_measure="truncated", angular_transverse_order=0,
                     angular_transverse_cutoff=5.0,
                     branch=DEFAULT_BRANCH, branches=None,
                     phase_detail=PHASE_LEGACY,
                     pump_probe_angle_deg=SEEDED_PHASE_ANGLE_DEG,
                     model_fidelity=None,
                     gain_closure_enabled=False,
                     legacy_gain_calibration_fixture=False,
                     legacy_mean_field_fixture=False,
                     legacy_coupling_fixture=False,
                     measured_gain_calibration=False,
                     seed_input_assumption="",
                     excess_noise_model=None,
                     floquet_order=SEEDED_FLOQUET_ORDER,
                     enforce_floquet_convergence=True,
                     response_method=RESPONSE_GRID,
                     scan_tolerance_dB=SCAN_TOLERANCE_DB,
                     scan_tolerance_log10_gain=SCAN_TOLERANCE_LOG10_GAIN,
                     scan_start_stride=SCAN_START_STRIDE,
                     transit_rate=constants.GAMMA_GG,
                     pump_stark_fraction=1.0,
                     eom_residual_carrier_power=0.0,
                     eom_other_sidebands_power=0.0,
                     eom_seed_spectrum_provenance="not supplied",
                     eom_seed_spectrum_status="not supplied",
                     eom_seed_spectrum_application="unapplied",
                     _readout_probe_points_GHz=(), _sparse_pole_schur=False):
    """Mean-field seeded-FWM pipeline at Δ = 2π·D_GHz·1e9.

    ``excess_noise_model`` controls only an optional phenomenological pump-scatter
    term in the Ultra gain-referred diagnostic.  Arm attenuation is already in
    the Maxwell drift; distributed loss vacuum and atomic diffusion are not
    supplied.  EOM residual carrier/sideband powers are carried as explicit input
    provenance but remain unapplied until a measured coupling model exists.
    The former affine replay flags are accepted for compatibility. An explicit
    replay request raises ValueError with the historical archive tag. A separate,
    optional measured pump-off-reference interpolation is returned under
    ``measured_gain_reference``. It
    never changes atomic gains, propagation, adaptive sampling, or noise NF.
    ``legacy_mean_field_fixture=True`` additionally restores the historical dipoles, optical
    carrier origin and unity floor on absorbed probe gains for frozen fixtures.
    Use only legacy_coupling_fixture to replay the historical dipoles with current carriers.
    Normal calls preserve absorption and use the transition-referenced carrier.
    Set ``velocity_measure="exact"`` for the untruncated Maxwell mean with
    the pole solver. Untiered callers default to their historical grid.
    Input seed power and small-signal gains refer to the atomic medium
    boundaries; entrance/exit windows and detector optics are not in that map.
    """
    fwm_gain_closure.reject_retired_replay(
        gain_closure_enabled, legacy_gain_calibration_fixture)
    stark_recompute_context = copy.deepcopy({
        key: value for key, value in locals().items()
        if key not in {"pump_stark_fraction", "_readout_probe_points_GHz"}
        and (key != "extra_probe_points_GHz" or np.size(value))
        and (key != "_sparse_pole_schur" or value)
        and (key not in {"angular_transverse_order", "angular_transverse_cutoff"}
             or angular_transverse_order)})
    branch = _single_branch(branch, branches)
    floquet_order = int(floquet_order)
    if floquet_order < 1:
        raise ValueError("floquet_order must be at least 1")
    if enforce_floquet_convergence and floquet_order < 3:
        raise ValueError(
            "reported scans require floquet_order >= 3; set "
            "enforce_floquet_convergence=False only for an explicitly historical fixture")
    if line_strength is None:
        line_strength = SEEDED_REFERENCE_RESIDUAL
    factors = SeededCouplingFactors(
        reference_residual=line_strength,
        mode_overlap_penalty=mode_overlap_penalty,
        polarization_penalty=polarization_penalty,
        zeeman_participation_penalty=zeeman_participation_penalty,
    )
    # The reference residual and the three lab-facing relative factors sit on
    # top of the first-principles macroscopic normalization. ``coupling_ls`` is
    # what enters every χ_phys / gain call below.
    historical_coupling = legacy_coupling_fixture or legacy_mean_field_fixture
    dipole_kwargs = {"transition_scales": LEGACY_TRANSITION_DIPOLE_SCALE} if historical_coupling else {}
    coupling_ledger = physical_coupling_ledger(branch, legacy=historical_coupling)
    coupling_norm = coupling_ledger["macroscopic_coupling_norm"]
    coupling_ls = factors.combined_residual * coupling_norm
    gain_closure = fwm_gain_closure.provenance(
        w_pump=w_pump, w_probe=w_probe, w_conjugate=w_conj,
        gain_closure_enabled=gain_closure_enabled,
        legacy_gain_calibration_fixture=legacy_gain_calibration_fixture)
    eta = (qe * (1.0 - loss_frac) if detection_efficiency is None
           else float(detection_efficiency))
    if not 0.0 <= eta <= 1.0:
        raise ValueError("detection efficiency must be between 0 and 1")

    eom_residual_carrier_power = float(eom_residual_carrier_power)
    eom_other_sidebands_power = float(eom_other_sidebands_power)
    if eom_residual_carrier_power < 0.0 or eom_other_sidebands_power < 0.0:
        raise ValueError("EOM component powers must be non-negative")
    eom_seed_spectrum_status = str(eom_seed_spectrum_status).strip().lower()
    eom_seed_spectrum_application = str(
        eom_seed_spectrum_application).strip().lower()
    if eom_seed_spectrum_status not in {"not supplied", "unsupported"}:
        raise ValueError(
            "EOM spectrum status cannot be promoted without a calibrated "
            "carrier/sideband transfer model")
    if eom_seed_spectrum_application != "unapplied":
        raise ValueError(
            "EOM residual components must remain unapplied until their atomic "
            "and detector transfer is calibrated")
    wanted_power = max(float(P_probe), 0.0)
    wanted_denominator = max(wanted_power, 1e-30)
    eom_seed_spectrum = {
        "wanted_sideband_power_w": wanted_power,
        "residual_carrier_power_w": eom_residual_carrier_power,
        "other_sidebands_power_w": eom_other_sidebands_power,
        "residual_carrier_to_wanted_ratio": (
            eom_residual_carrier_power / wanted_denominator),
        "other_sidebands_to_wanted_ratio": (
            eom_other_sidebands_power / wanted_denominator),
        "provenance": str(eom_seed_spectrum_provenance),
        "status": eom_seed_spectrum_status,
        "application": eom_seed_spectrum_application,
        "physics_effect": (
            "unapplied: carrier/sideband coupling, phase, polarization, and "
            "technical-noise transfer are not calibrated"),
    }

    Op_A = rabi_freq(P_pump, w_pump)
    Op_B = Op_A
    Os = rabi_freq(P_probe, w_probe)
    Os_ref = Os
    # Equal reference powers; finite-drive chi can depend on |Omega_ref|.
    # These two driven states are not related by the +/- harmonic symmetry;
    # keep both seed configurations even when their harmonic chains are reduced.
    Oc_ref = Os

    # Pure-85Rb CRC vapor density, consistent with the AutoOD-validated
    # absorption path (`hyperfine.number_density`). The natural-abundance
    # share from `atoms.rb85_density` is for a different cell composition.
    N_atoms = hyperfine.number_density(T)
    Delta = 2 * np.pi * D_GHz * 1e9

    # Density-dependent collisions plus trace-preserving transit replenishment.
    atom_T = fwm.collisional_atom(T, N_atoms, transit_rate=transit_rate)
    transit_rate = atom_T.transit_reset_rate
    atom_T, pump_stark_response = fwm_stark_response.apply(
        atom_T, pump_hamiltonian_at_Deff_zero(Op_A, Op_B, **dipole_kwargs), Delta,
        fraction=pump_stark_fraction, pump_power_W=P_pump)

    probe_axis_GHz = fwm.probe_scan_axis_GHz(
        D_GHz, coarse_points, fine_points, window_mhz, scan_min, scan_max,
        branches=(branch,))
    extra = np.asarray(extra_probe_points_GHz, dtype=float)
    readout_points = np.asarray(_readout_probe_points_GHz, dtype=float)
    if readout_points.size and extra.size:
        raise ValueError("readout-only coordinates cannot extend a native scan")
    requested = readout_points if readout_points.size else extra
    if (requested.ndim != 1 or np.any(~np.isfinite(requested))
            or np.any(requested < probe_axis_GHz[0]) or np.any(requested > probe_axis_GHz[-1])
            or np.any(np.abs(requested - D_GHz) <= PUMP_OVERLAP_EXCLUSION_MHZ * 1e-3)):
        raise ValueError("extra probe points must be finite and within the sampled probe span")
    if readout_points.size:
        requested = np.unique(readout_points)
        if requested.size < 2:
            # The adjacent-order audit requires two distinct coordinates.
            anchor = probe_axis_GHz[0] if requested[0] != probe_axis_GHz[0] else probe_axis_GHz[-1]
            requested = np.sort(np.r_[requested, anchor])
        probe_axis_GHz = requested
    elif extra.size:
        probe_axis_GHz = np.unique(np.r_[probe_axis_GHz, extra])
    k_pump_vac, k_probe_vac, k_conj_vac = seeded_option_a_wavenumbers(
        D_GHz, probe_axis_GHz, legacy_centroid=legacy_mean_field_fixture)
    omega_probe = constants.C_LIGHT * k_probe_vac
    omega_conj = constants.C_LIGHT * k_conj_vac
    area_probe = float(observables.gaussian_mode_area(w_probe))
    if w_conj is None:
        w_conj_effective = float(w_probe)
        canonical_mode_status = (
            "conditional: conjugate collection waist not supplied; matched to probe")
    else:
        w_conj_effective = float(w_conj)
        canonical_mode_status = "explicit probe and conjugate collected-mode waists"
    area_conj = float(observables.gaussian_mode_area(w_conj_effective))
    if velocity_measure not in ("exact", "truncated"):
        raise ValueError("velocity_measure must be exact or truncated")
    exact_velocity = velocity_measure == "exact"
    if exact_velocity and response_method == RESPONSE_GRID:
        raise ValueError("The exact Maxwell measure requires a pole response method")
    gaussian_sigma = (K_VEC * constants.thermal_speed(T, constants.MASS_85RB)
                      if exact_velocity else None)
    velocity_step = VELOCITY_STEP_MPS if velocity_step is None else velocity_step
    velocity_cutoff = VELOCITY_CUTOFF_SIGMA if velocity_cutoff is None else velocity_cutoff
    # The grid is a numerical fallback when the exact pole form is rejected.
    fallback_cutoff = max(8.0, velocity_cutoff) if exact_velocity else velocity_cutoff
    v_grid, weights = doppler.velocity_grid(
        T, dv=velocity_step, cutoff_sigma=fallback_cutoff)
    Delta_eff_axis = doppler.build_Delta_eff_axis(Delta, Delta, v_grid)

    delta_axis = two_photon_detuning_from_probe_scan(probe_axis_GHz, D_GHz, branch)
    # All four susceptibility tables share the same Δ_eff interpolation geometry.
    # Build it once for this solve instead of repeating the index/fraction work.
    idx_lo, frac = doppler.interpolation_weights(
        Delta_eff_axis, Delta, v_grid)

    def averaged_response_at_order(order):
        tables = fwm.chi_matrix_table(
            Op_A, Op_B, Os_ref, Oc_ref, delta_axis, Delta_eff_axis,
            branch, atom=atom_T, n_f=order, **dipole_kwargs)
        return tuple(
            doppler.apply_doppler_average(table, idx_lo, frac, weights)
            for table in tables)

    phase_detail = (phase_detail or PHASE_LEGACY).lower()
    response_method = str(response_method)
    if response_method not in RESPONSE_METHODS:
        raise ValueError(f"response_method must be one of {RESPONSE_METHODS}")
    response_orders = ((floquet_order, floquet_order - 1)
                       if enforce_floquet_convergence else (floquet_order,))
    pole_systems = None
    if response_method != RESPONSE_GRID and not angular_transverse_order:
        pole_systems = fwm._seeded_pole_systems(
            Op_A, Op_B, Os_ref, Oc_ref, branch, atom_T, response_orders,
            sparse_schur=_sparse_pole_schur, **dipole_kwargs)
    angular_reference = None
    if angular_transverse_order:
        if not enforce_floquet_convergence or legacy_mean_field_fixture:
            raise ValueError("angular reference requires adjacent orders and the current carrier origin")
        angular_reference = finite_seed_angular_reference(
            Op_A, Op_B, Os_ref, Oc_ref, delta_axis, Delta,
            T=T, pump_k_rad_m=k_pump_vac, probe_k_axis_rad_m=k_probe_vac,
            crossing_angle_rad=np.deg2rad(pump_probe_angle_deg), atom=atom_T,
            floquet_order=floquet_order, transverse_order=angular_transverse_order,
            cutoff_sigma=angular_transverse_cutoff, branch=branch, **dipole_kwargs)
        chi_ss_avg, chi_cs_avg, chi_sc_avg, chi_cc_avg = angular_reference["responses"].pop(floquet_order)
        lower_order_response = angular_reference["responses"].pop(floquet_order - 1)
        del angular_reference["responses"]
        response_estimator = {
            "method": "angular_conditional_pole_reference",
            "velocity_average": "conditional longitudinal Maxwell pole mean, transverse Gauss–Legendre",
            "solved_probe_points": int(probe_axis_GHz.size),
            "display_probe_points": int(probe_axis_GHz.size),
            "interpolation": "none",
            "audit_scope": "every angular reference detuning",
        }
    elif pole_systems is None:
        chi_ss_avg, chi_cs_avg, chi_sc_avg, chi_cc_avg = (
            averaged_response_at_order(floquet_order))
        lower_order_response = None
        if enforce_floquet_convergence:
            lower_order_response = averaged_response_at_order(floquet_order - 1)
        response_estimator = {
            "method": RESPONSE_GRID,
            "velocity_average": (
                "one compiled finite-Floquet solve per Maxwell class with linear "
                "Δ_eff interpolation"),
            "solved_probe_points": int(probe_axis_GHz.size),
            "display_probe_points": int(probe_axis_GHz.size),
            "interpolation": "none",
            "audit_scope": "every displayed probe detuning",
        }
        if response_method != RESPONSE_GRID:
            response_estimator["fallback_reason"] = (
                "exact Hermiticity/adjoint identities of the real pole form failed")
    else:
        score_scatter = 0.0
        score_profile = np.ascontiguousarray(_gaussian_overlap_profile(
            ULTRA_PROPAGATION_SEGMENTS, L, w_pump, w_probe, pump_probe_angle_deg))
        score_noise_enabled = excess_noise_model is not False
        if phase_detail == PHASE_ULTRA and score_noise_enabled:
            score_cfg = ({} if excess_noise_model in (None, True)
                         else dict(excess_noise_model))
            score_kappa = score_cfg.get(
                "pump_scatter_kappa", HARDENED_PUMP_SCATTER_KAPPA)
            score_scatter, _ = fwm._pump_scatter_noise(D_GHz, T, L, score_kappa)

        def score_readout(chi4, rows):
            """Pointwise copy of the readout below, used only to steer refinement."""
            chi_ss, chi_cs, chi_sc, chi_cc = chi4
            rows = np.asarray(rows, dtype=int)
            if phase_detail == PHASE_LEGACY:
                k_probe_rows = np.full(rows.size, K_VEC, dtype=float)
                k_conj_rows = k_probe_rows
                delta_k_rows = None
            else:
                k_probe_rows = k_probe_vac[rows]
                k_conj_rows = k_conj_vac[rows]
                delta_k_rows = seeded_phase_mismatch_z(
                    D_GHz, probe_axis_GHz[rows], angle_deg=pump_probe_angle_deg,
                    legacy_centroid=legacy_mean_field_fixture)
            if phase_detail == PHASE_ULTRA:
                if kernels.available():
                    # Same closed-form steps and pump budget, compiled (steering only).
                    maxwell_rows = observables._gain_matrix_from_chi(
                        chi_ss, chi_sc, chi_cs, chi_cc, k_probe_rows, k_conj_rows,
                        N_atoms, constants.DIPOLE_D1, coupling_ls,
                        delta_k_z=delta_k_rows)
                    G_s_rows, G_c_rows = kernels.segmented_depletion_gain(
                        np.ascontiguousarray(maxwell_rows),
                        L / ULTRA_PROPAGATION_SEGMENTS, score_profile,
                        float(P_pump), float(P_probe), area_conj / area_probe,
                        _EXP_ARG_CLAMP)
                else:
                    G_s_rows, G_c_rows, _, _ = _ultra_segmented_gain(
                        chi_ss, chi_sc, chi_cs, chi_cc, k_probe_rows, k_conj_rows,
                        L, N_atoms, coupling_ls, delta_k_rows,
                        np.ones(ULTRA_PROPAGATION_SEGMENTS), score_profile,
                        P_pump, P_probe, conjugate_power_ratio=area_conj / area_probe)
                G_c_rows = G_c_rows * area_conj / area_probe
            else:
                nseg = 16 if phase_detail == PHASE_FINE else 1
                _, _, transfer = observables.gain_from_chi(
                    chi_ss, chi_sc, chi_cs, chi_cc, k_probe_rows, k_conj_rows, L,
                    N_atoms, line_strength=coupling_ls, delta_k_z=delta_k_rows,
                    propagation_segments=nseg,
                    segment_profile=np.ones(nseg) if nseg > 1 else None)
                canonical_rows = observables.canonical_transfer_diagnostics(
                    transfer, omega_probe[rows], omega_conj[rows],
                    area_probe, area_conj)
                G_s_rows = canonical_rows["probe_power_gain"]
                G_c_rows = canonical_rows["conjugate_power_gain"]
            G_s_rows, G_c_rows = observables.pump_depletion_saturation(
                G_s_rows, G_c_rows, P_pump, P_probe,
                legacy_probe_floor=legacy_mean_field_fixture)
            # In the new calibration context the atomic sampling must not depend
            # on external detector efficiency. Keep a fixed steering readout;
            # the actual noise readout below still uses the requested eta.
            score_eta = (eta if legacy_gain_calibration_fixture or legacy_mean_field_fixture
                         else 1.0)
            if phase_detail == PHASE_ULTRA and score_noise_enabled:
                noise = observables.balanced_twin_beam_noise(
                    G_s_rows, G_c_rows, score_eta, score_eta, reference_weight="dc",
                    reference_excess_noise=score_scatter)
                return G_s_rows, 10.0 * np.log10(np.maximum(noise, 1e-30))
            return G_s_rows, observables.gain_referred_noise_dB(
                G_s_rows, G_c_rows, score_eta)

        def guard_rows(rows, order):
            return fwm.chi_matrix_table(
                Op_A, Op_B, Os_ref, Oc_ref, delta_axis[rows], np.array([Delta]),
                branch, atom=atom_T, n_f=order, **dipole_kwargs)

        def grid_rows(rows, order):
            return tuple(
                doppler.apply_doppler_average(table, idx_lo, frac, weights)
                for table in fwm.chi_matrix_table(
                    Op_A, Op_B, Os_ref, Oc_ref, delta_axis[rows], Delta_eff_axis,
                    branch, atom=atom_T, n_f=order, **dipole_kwargs))

        high_response, lower_order_response, response_estimator = fwm._pole_scan_response(
            pole_systems, orders=response_orders, delta_axis=delta_axis,
            probe_axis_GHz=probe_axis_GHz, Delta=Delta,
            gaussian_sigma=gaussian_sigma,
            Delta_eff_axis=Delta_eff_axis, idx_lo=idx_lo, frac=frac,
            weights=weights, adaptive=response_method == RESPONSE_POLE_ADAPTIVE,
            score=score_readout, tolerance_dB=scan_tolerance_dB,
            tolerance_log10_gain=scan_tolerance_log10_gain,
            start_stride=scan_start_stride, guard=guard_rows, grid_rows=grid_rows)
        chi_ss_avg, chi_cs_avg, chi_sc_avg, chi_cc_avg = high_response
    if _sparse_pole_schur and pole_systems is not None:
        response_estimator["schur_products"] = (
            "compiled structural blocks and nonzeros" if kernels.available() else "dense NumPy")
    response_estimator["absorption_diagnostics"] = LINEAR_ABSORPTION_MODEL
    response_estimator["velocity_classes"] = int(v_grid.size)
    response_estimator["velocity_measure"] = velocity_measure
    if exact_velocity:
        response_estimator["fallback_velocity_cutoff_sigma"] = float(fallback_cutoff)
        if pole_systems is None and angular_reference is None:
            response_estimator["velocity_average"] = (
                f"Maxwell grid fallback ({fallback_cutoff:g} sigma); analytic pole form unavailable")
    atomic_high_response = (chi_ss_avg, chi_cs_avg, chi_sc_avg, chi_cc_avg)
    delta_k_z = None
    delta_k_z_vacuum = None
    k_probe_prop = np.full_like(probe_axis_GHz, K_VEC, dtype=float)
    k_conj_prop = np.full_like(probe_axis_GHz, K_VEC, dtype=float)
    propagation_segments = 1
    segment_profile = None
    segment_od = 0.0
    spatial_profile = None
    ultra_phase_iterations = 0
    ultra_phase_max_change = 0.0
    ultra_pump_remaining_min = float(P_pump)
    zeeman_status = "inactive"
    zeeman_correction = 1.0
    if phase_detail != PHASE_LEGACY:
        delta_k_z_vacuum = seeded_phase_mismatch_z(
            D_GHz, probe_axis_GHz, angle_deg=pump_probe_angle_deg,
            legacy_centroid=legacy_mean_field_fixture)
        delta_k_z = delta_k_z_vacuum
        # Option A: frequency-specific bare k in the Maxwell matrix and vacuum /
        # geometric mismatch only. Re(chi) remains in the diagonal response.
        k_probe_prop = k_probe_vac
        k_conj_prop = k_conj_vac
        if phase_detail == PHASE_FINE:
            propagation_segments = 16
            segment_profile, segment_od = _uniform_segment_profile_and_probe_od(
                chi_ss_avg, N_atoms, coupling_ls, propagation_segments, L=L)
        elif phase_detail == PHASE_ULTRA:
            propagation_segments = ULTRA_PROPAGATION_SEGMENTS
            segment_profile, segment_od = _uniform_segment_profile_and_probe_od(
                chi_ss_avg, N_atoms, coupling_ls, propagation_segments, L=L)
            spatial_profile = _gaussian_overlap_profile(
                propagation_segments, L, w_pump, w_probe, pump_probe_angle_deg)
            if pole_systems is not None:
                zeeman_correction, zeeman_status = _zeeman_consistency_cached()
            else:
                try:
                    from ... import zeeman as _zeeman
                    z_atom = _zeeman.rb85_d1_double_lambda_zeeman()
                    zeeman_correction = float(
                        getattr(z_atom, "lumped_strength_correction", 1.0))
                    zeeman_status = (
                        f"24-level CG-sum consistency check = {zeeman_correction:.4f} "
                        "(lumped 3·C_F² reproduced); diagnostic only — full Floquet "
                        "scan not run in Ultra v1")
                except Exception as exc:                  # pragma: no cover
                    zeeman_status = f"unavailable: {exc}"

    if phase_detail == PHASE_ULTRA:
        if segment_profile is None:
            segment_profile = np.ones(propagation_segments, dtype=float)
        if spatial_profile is None:
            spatial_profile = np.ones(propagation_segments, dtype=float)
        # zeeman_correction is a CG-sum consistency diagnostic (≡1.0 by
        # construction), not an active correction, so it is no longer multiplied
        # into the coupling.
        G_s_effective, G_c_effective, _T_effective, pump_remaining = _ultra_segmented_gain(
            chi_ss_avg, chi_sc_avg, chi_cs_avg, chi_cc_avg,
            k_probe_prop, k_conj_prop, L, N_atoms,
            coupling_ls, delta_k_z,
            segment_profile, spatial_profile, P_pump, P_probe,
            conjugate_power_ratio=area_conj / area_probe)
        ultra_pump_remaining_min = float(np.nanmin(pump_remaining))
        # Canonical diagnostics require a linear map. Remove the seed-dependent
        # dynamic-depletion scale while retaining the fixed spatial/pump profile.
    undepleted_profile = (
        segment_profile * spatial_profile if phase_detail == PHASE_ULTRA else segment_profile)
    collected_modes = (omega_probe, omega_conj, area_probe, area_conj)
    G_s_smallsignal, G_c_smallsignal, T_small_signal, canonical = seeded_linear_gain(
        atomic_high_response, k_probe_prop, k_conj_prop, L, N_atoms, coupling_ls,
        collected_modes=collected_modes, delta_k_z=delta_k_z,
        propagation_segments=propagation_segments, segment_profile=undepleted_profile)
    if lower_order_response is not None:
        _, _, _, canonical_lower = seeded_linear_gain(
            lower_order_response, k_probe_prop, k_conj_prop, L, N_atoms, coupling_ls,
            collected_modes=collected_modes, delta_k_z=delta_k_z,
            propagation_segments=propagation_segments, segment_profile=undepleted_profile)
        floquet_convergence = fwm.assess_floquet_scan_convergence(
            high_order=floquet_order,
            low_order=floquet_order - 1,
            high_response={
                name: value for name, value in zip(
                    ("chi_ss", "chi_cs", "chi_sc", "chi_cc"), atomic_high_response)
            },
            low_response={
                name: value for name, value in zip(
                    ("chi_ss", "chi_cs", "chi_sc", "chi_cc"), lower_order_response)
            },
            high_transfer=canonical["transfer_canonical"],
            low_transfer=canonical_lower["transfer_canonical"],
            scan_axis=probe_axis_GHz,
        )
    else:
        floquet_convergence = {
            "status": "NOT_EVALUATED",
            "passed": False,
            "high_order": floquet_order,
            "comparison_order": None,
            "full_scan_points": int(probe_axis_GHz.size),
            "reason": (
                "adjacent-order full-scan comparison explicitly disabled; "
                "result is a historical/numerical fixture only"),
        }
    if phase_detail == PHASE_ULTRA:
        # The dynamic-depletion path propagates sqrt(power) amplitudes. Convert
        # its conjugate coefficient for an explicitly unequal collected area.
        G_s = G_s_effective
        G_c = G_c_effective * area_conj / area_probe
    else:
        G_s = G_s_smallsignal
        G_c = G_c_smallsignal

    # The pre-cap gains are the output of the chosen propagation model.
    # Keep this stage distinct from the undepleted canonical transfer and
    # the post-hoc pump-budget heuristic.
    pre_cap_s, pre_cap_c = G_s, G_c
    capped_s, capped_c = observables.pump_depletion_saturation(
        pre_cap_s, pre_cap_c, P_pump, P_probe)
    legacy_floor_mask = (capped_s < 1.0) if legacy_mean_field_fixture else np.zeros_like(
        capped_s, dtype=bool)
    G_s = np.where(legacy_floor_mask, 1.0, capped_s) if legacy_mean_field_fixture else capped_s
    G_c = capped_c
    dynamic_depletion = phase_detail == PHASE_ULTRA
    gain_stages = {
        "small_signal": (G_s_smallsignal, G_c_smallsignal),
        "pre_cap": (pre_cap_s, pre_cap_c),
        "final_cap": (capped_s, capped_c),
        "displayed": (G_s, G_c),
    }
    hardened_noise = None
    pump_scatter_kappa_used = HARDENED_PUMP_SCATTER_KAPPA
    if phase_detail == PHASE_ULTRA:
        if excess_noise_model is False:
            # The diagonal Maxwell drift already contains the arm attenuation.
            # A second post-source transmission would double count it; explicit
            # distributed Langevin diffusion is a separate, unfinished model.
            gain_referred_noise_db = observables.gain_referred_noise_dB(
                G_s, G_c, eta)
        else:
            cfg = {} if excess_noise_model in (None, True) else dict(excess_noise_model)
            kappa = cfg.get("pump_scatter_kappa", HARDENED_PUMP_SCATTER_KAPPA)
            pump_scatter_kappa_used = float(kappa)
            # The F=2→F'=3 manifold is the linear absorption the lumped double-Λ
            # omits for the generated sidebands. Apply it at BOTH sidebands' own
            # optical frequencies: without the probe term the deep-blue-Δ region
            # parks the probe ON F=2→F'=3 with no penalty (a probe-side mirror of
            # the conjugate artifact).
            conj_ground = G1 if branch == -1 else G2
            manifold_F = GROUND_F[conj_ground]
            conj_GHz = 2.0 * float(D_GHz) - probe_axis_GHz
            od_conj_arr = fwm._arm_linear_od(conj_GHz, T, L, ground_F=manifold_F)
            od_probe_lin = fwm._arm_linear_od(probe_axis_GHz, T, L, ground_F=manifold_F)
            pump_scatter, od_pump = fwm._pump_scatter_noise(D_GHz, T, L, kappa)
            # These OD curves remain useful diagnostics, but applying them as
            # post-source efficiencies would count absorption already present in
            # the diagonal Maxwell drift a second time.  Atomic vacuum/excess
            # noise needs the future distributed covariance solve instead.
            eta_s_arm = eta
            eta_c_arm = eta
            gain_referred_noise_db = seeded_balanced_gain_noise_dB(
                G_s, G_c, eta_s_arm, eta_c_arm, seed_excess_noise=pump_scatter)
            hardened_noise = {
                "kappa": float(kappa),
                "pump_scatter_noise": float(pump_scatter),
                "od_pump": float(od_pump),
                "segment_od_probe": float(segment_od),
                "od_conj_max": float(np.nanmax(od_conj_arr)),
                "od_probe_lin_max": float(np.nanmax(od_probe_lin)),
                "od_conj_arr": od_conj_arr,
                "od_probe_lin_arr": od_probe_lin,
                "atomic_od_application": (
                    "diagnostic only; distributed Langevin covariance unavailable"),
                "absorption_model": response_estimator["absorption_diagnostics"],
            }
    else:
        gain_referred_noise_db = observables.gain_referred_noise_dB(G_s, G_c, eta)

    if pole_systems is not None or angular_reference is not None:
        floquet_convergence["scope"] = response_estimator["audit_scope"]
        floquet_convergence["solved_scan_points"] = (
            response_estimator["solved_probe_points"])
    atomic_solver_provenance = seeded_atomic_solver_provenance(
        floquet_order=floquet_order, convergence=floquet_convergence)
    parameter_provenance = seeded_parameter_provenance(
        line_strength=line_strength,
        transit_rate=transit_rate,
        pump_scatter_kappa=pump_scatter_kappa_used,
    )
    claim_gate = seeded_validation_claim_gate(
        canonical_mode_status=canonical_mode_status,
        commutator_defect_max=canonical["commutator_defect_max"],
        floquet_convergence=floquet_convergence,
        eom_residual_carrier_power=eom_residual_carrier_power,
        eom_other_sidebands_power=eom_other_sidebands_power,
        eom_spectrum_status=eom_seed_spectrum_status,
        eom_spectrum_application=eom_seed_spectrum_application,
        response_estimator=response_estimator,
    )
    if legacy_mean_field_fixture:
        claim_gate["badges"] += ("HISTORICAL_MEAN_FIELD_FIXTURE",)
        claim_gate["reasons"] += (
            "historical carrier origin and absorbed-probe unity floor requested",)

    cell_drift = observables._gain_matrix_from_chi(
        chi_ss_avg, chi_sc_avg, chi_cs_avg, chi_cc_avg,
        k_probe_prop, k_conj_prop, N_atoms, constants.DIPOLE_D1,
        coupling_ls, delta_k_z=delta_k_z)
    cell_transmission = dressed_cell_transmission(cell_drift, L)
    spectrum = {
        # Reusable input to the optical propagator. Empirical propagation
        # readouts operate on a copy; the raw response and audits stay intact.
        "propagation_snapshot": {
            "version": "fwm-chi-propagation-v1",
            "drift_per_m": cell_drift.copy(),
            "chi": np.stack((chi_ss_avg, chi_sc_avg, chi_cs_avg, chi_cc_avg), axis=-1).reshape(-1, 2, 2),
            "segment_profile": (
                np.asarray(segment_profile) * np.asarray(spatial_profile)
                if phase_detail == PHASE_ULTRA else
                np.ones(propagation_segments) if segment_profile is None else np.asarray(segment_profile).copy()),
            "dynamic_depletion": phase_detail == PHASE_ULTRA,
            "conjugate_power_ratio": float(area_conj / area_probe),
            "source": "Doppler-averaged response before empirical literature correction",
        },
        "dressed_cell_transmission": cell_transmission,
        "pump_stark_response": pump_stark_response,
        "stark_recompute_context": stark_recompute_context,
        "D_GHz": D_GHz,
        "probe_axis_GHz": probe_axis_GHz,
        "G_s": G_s,
        "G_c": G_c,
        "gain_referred_noise_dB": gain_referred_noise_db,
        "legacy_gain_diagnostic": {
            "noise_dB": gain_referred_noise_db,
            "model": ("effective hardened mean-field noise" if hardened_noise is not None
                      else "historical two-gain algebra"),
            "physical_squeezing_claim_allowed": False,
        },
        "gain_budget": {
            "power_plane": "atomic medium exit, before exit window",
            "gain_reference": "input probe power at the atomic medium entrance",
            "power_definition": "each gain multiplied by input_seed_power_W",
            **{stage: {"G_s": probe, "G_c": conjugate,
                       "probe_W": probe * P_probe, "conjugate_W": conjugate * P_probe}
               for stage, (probe, conjugate) in gain_stages.items()},
            "dynamic_depletion": {
                "enabled": dynamic_depletion,
                "probe_changed": (pre_cap_s != G_s_smallsignal) & dynamic_depletion,
                "conjugate_changed": (pre_cap_c != G_c_smallsignal) & dynamic_depletion,
            },
            "posthoc_pump_cap": {
                "probe_reduced": capped_s < pre_cap_s,
                "conjugate_reduced": capped_c < pre_cap_c,
            },
            "historical_probe_floor": {
                "enabled": bool(legacy_mean_field_fixture),
                "probe_raised": legacy_floor_mask,
            },
        },
        # Deliberately unavailable until the frequency-dependent microscopic
        # Langevin drift/diffusion and detector transfer are implemented.
        "physical_squeezing_dB": None,
        # Backward-compatible data key.  It is the same algebraic diagnostic,
        # not a physical squeezing prediction.
        "S_dB": gain_referred_noise_db,
        "G_s_smallsignal": G_s_smallsignal,
        "G_c_smallsignal": G_c_smallsignal,
        "G_s_smallsignal_peak": float(np.nanmax(G_s_smallsignal)),
        "G_c_smallsignal_peak": float(np.nanmax(G_c_smallsignal)),
        "T_field_small_signal": T_small_signal,
        "Q_photon_flux": canonical["Q"],
        "T_canonical_small_signal": canonical["transfer_canonical"],
        "conjugate_photon_flux_gain_smallsignal": canonical[
            "conjugate_photon_flux_gain"],
        "photon_flux_gap_smallsignal": canonical["photon_flux_gap"],
        "commutator_defect_smallsignal": canonical["commutator_defect"],
        "commutator_defect_max_smallsignal": canonical["commutator_defect_max"],
        "canonical_mode_status": canonical_mode_status,
        "canonical_mode_area_probe_m2": area_probe,
        "canonical_mode_area_conjugate_m2": area_conj,
        "omega_probe_rad_s": omega_probe,
        "omega_conjugate_rad_s": omega_conj,
        "legacy_mean_field_fixture": bool(legacy_mean_field_fixture),
        "legacy_coupling_fixture": bool(historical_coupling),
        "optical_carrier_origin": (
            "historical fine-structure centroid" if legacy_mean_field_fixture
            else "85Rb D1 F=2 -> F'=3"),
        "gain_reference": dict(
            gain_reference.pump_off_referenced_gains(
                G_s_smallsignal, G_c_smallsignal),
            source_gain_keys=("G_s_smallsignal", "G_c_smallsignal")),
        "input_seed_power_W": float(P_probe),
        "input_pump_power_W": float(P_pump),
        "temperature_K": float(T),
        "velocity_cutoff_sigma": float(velocity_cutoff),
        "velocity_measure": velocity_measure,
        "input_seed_plane": "atomic medium entrance, after entrance window",
        "output_gain_plane": "atomic medium exit, before exit window",
        "smallsignal_output_gain_plane": "atomic medium exit, before exit window",
        "displayed_gains_posthoc_saturated": True,  # compatibility: the stage was evaluated
        "pump_depletion_cap": 1.0 + 0.5 * P_pump / max(P_probe, 1e-30),
        "phase_detail": phase_detail,
        "model_fidelity": model_fidelity or phase_detail,
        "pump_probe_angle_deg": pump_probe_angle_deg,
        "propagation_convention": (
            "legacy fixed-k" if phase_detail == PHASE_LEGACY
            else "Option A: bare frequency-specific k plus vacuum mismatch"),
        "k_probe_propagation_per_m": k_probe_prop,
        "k_conjugate_propagation_per_m": k_conj_prop,
        "delta_k_z": delta_k_z,
        "delta_k_z_vacuum": delta_k_z_vacuum,
        "phase_segments": propagation_segments,
        "segment_absorption_od": segment_od,
        "ultra_phase_iterations": ultra_phase_iterations,
        "ultra_phase_max_change": ultra_phase_max_change,
        "ultra_dynamic_depletion": phase_detail == PHASE_ULTRA,
        "ultra_in_cell_loss_noise": False,
        "squeezing_status": (
            "unavailable: gain-referred diagnostic only; microscopic distributed "
            "atomic diffusion is not implemented"),
        "validation_level": claim_gate["level"],
        "claim_gate": claim_gate,
        "atomic_solver_provenance": atomic_solver_provenance,
        "pump_weak_response_reference_provenance": (
            pump_weak_response_reference_provenance()),
        "noncollinear_doppler_reference_provenance": (
            noncollinear_doppler_reference_provenance()),
        "floquet_convergence": floquet_convergence,
        "response_estimator": response_estimator,
        "gain_source_model": {
            "dipole_normalization": coupling_ledger["dipole_normalization"],
            "response_method": response_estimator["method"],
            "scan_points": int(probe_axis_GHz.size),
            "velocity_measure": velocity_measure,
            "velocity_cutoff_sigma": None if exact_velocity else float(velocity_cutoff),
            "velocity_step_mps": float(velocity_step),
            "floquet_order": int(floquet_order),
            "phase_detail": phase_detail,
            "linewidth_hz": float(constants.GAMMA_2PI),
            "rb85_mass_kg": float(constants.MASS_85RB),
            "d1_excited_splitting_hz": float(constants.NU_EXCITED_HF_D1),
        },
        "gain_closure": gain_closure,
        "floquet_order": floquet_order,
        "floquet_convergence_enforced": bool(enforce_floquet_convergence),
        "parameter_provenance": parameter_provenance,
        "eom_seed_spectrum": eom_seed_spectrum,
        "transit_reset_rate_rad_s": float(transit_rate),
        "ground_collision_dephasing_rate_rad_s": float(
            atom_T.ground_collision_dephasing_rate),
        "ultra_pump_remaining_min": ultra_pump_remaining_min,
        "ultra_spatial_overlap_min": (float(np.nanmin(spatial_profile))
                                      if spatial_profile is not None else 1.0),
        "zeeman_status": zeeman_status,
        "zeeman_correction": zeeman_correction,
        "eta": eta,
        "qe": qe,
        "w_pump_m": w_pump,
        "w_probe_m": w_probe,
        "w_conjugate_m": w_conj_effective,
        "cell_length_m": L,
        "branch": branch,
        "coupling_norm": coupling_norm,
        "coupling_normalization_ledger": coupling_ledger,
        # Keep the historical key equal to the direct API input.  Consumers can
        # opt into the factorized bookkeeping through the new explicit keys.
        "line_strength_residual": line_strength,
        "mode_overlap_penalty": factors.mode_overlap_penalty,
        "polarization_penalty": factors.polarization_penalty,
        "zeeman_participation_penalty": factors.zeeman_participation_penalty,
        "lab_coupling_factor": factors.lab_factor,
        "combined_line_strength_residual": factors.combined_residual,
        "effective_line_strength": coupling_ls,
        "N_atoms": N_atoms,
        "sigma_v": constants.thermal_speed(T, constants.MASS_85RB),
        "n_velocity": v_grid.size,
        "Op_A_2pi_GHz": Op_A / (2 * np.pi) / 1e9,
        "Os_2pi_MHz": Os / (2 * np.pi) / 1e6,
        "raman_center_minus_GHz": branch_center_GHz(D_GHz, -1),
        "raman_center_plus_GHz": branch_center_GHz(D_GHz, +1),
        "hardened_noise": hardened_noise,
    }
    # This separate readout uses the canonical photon-flux map before the
    # post-hoc power cap/output fit. Any susceptibility coupling closure is
    # already included in that map. The legacy gain diagnostic above is retained
    # verbatim for callers and historical numerical fixtures.
    if angular_reference is not None:
        spectrum["finite_seed_angular_reference"] = angular_reference
    spectrum.update(passive_l1_from_spectrum(spectrum))
    spectrum["measured_gain_reference"] = fwm._measured_gain_reference_readout(
        spectrum, enabled=measured_gain_calibration,
        seed_input_assumption=seed_input_assumption)
    return spectrum


def operating_point(spectrum, delta_mhz, branch=-1):
    """Read gains and the gain-referred diagnostic at a selected δ (MHz)."""
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")
    center = spectrum["raman_center_minus_GHz" if branch == -1 else "raman_center_plus_GHz"]
    probe_GHz = center + delta_mhz * 1e-3
    pump_GHz = spectrum.get("D_GHz", center - branch * constants.NU_HF / 1e9)
    x = np.asarray(spectrum["probe_axis_GHz"])
    exclusion = PUMP_OVERLAP_EXCLUSION_MHZ * 1e-3
    # Do not interpolate across the unsampled pump interval, even for a probe
    # just outside it whose two neighboring samples straddle the pump.
    if probe_GHz < pump_GHz:
        segment = slice(None, np.searchsorted(x, pump_GHz - exclusion, side="left"))
    else:
        segment = slice(np.searchsorted(x, pump_GHz + exclusion, side="right"), None)
    axis = x[segment]
    branch_matches = spectrum.get("branch", branch) == branch
    available = bool(branch_matches and axis.size and axis[0] <= probe_GHz <= axis[-1])

    def interpolate(values):
        if not available:
            return float("nan")
        return float(np.interp(probe_GHz, axis, np.asarray(values)[segment]))

    diagnostic = spectrum.get("gain_referred_noise_dB")
    if diagnostic is None:
        diagnostic = spectrum["S_dB"]
    diagnostic_value = interpolate(diagnostic)
    out = {
        "probe_GHz": probe_GHz,
        "G_s": interpolate(spectrum["G_s"]),
        "G_c": interpolate(spectrum["G_c"]),
        "gain_referred_noise_dB": diagnostic_value,
        "physical_squeezing_dB": None,
        "S_dB": diagnostic_value,  # compatibility alias; not physical squeezing
    }
    for source, target in (
        ("G_s_smallsignal", "G_s_smallsignal"),
        ("G_c_smallsignal", "G_c_smallsignal"),
        ("conjugate_photon_flux_gain_smallsignal", "G_c_flux_smallsignal"),
        ("photon_flux_gap_smallsignal", "photon_flux_gap_smallsignal"),
        ("commutator_defect_max_smallsignal", "commutator_defect_max_smallsignal"),
    ):
        if source in spectrum:
            out[target] = interpolate(spectrum[source])
    budget = spectrum.get("gain_budget")
    if budget is not None:
        out["gain_budget"] = {
            stage: {arm: interpolate(gains)
                    for arm, gains in budget[stage].items()}
            for stage in ("small_signal", "pre_cap", "final_cap", "displayed")
        }
    if spectrum.get("delta_k_z") is not None:
        out["delta_k_z"] = interpolate(spectrum["delta_k_z"])
    if spectrum.get("delta_k_z_vacuum") is not None:
        out["delta_k_z_vacuum"] = interpolate(spectrum["delta_k_z_vacuum"])
    measured = spectrum.get("measured_gain_reference", {})
    if measured.get("requested"):
        point = fwm._measured_gain_reference_readout(
            spectrum, enabled=True,
            seed_input_assumption=measured.get("seed_input_assumption", ""),
            selected_delta_mhz=delta_mhz, requested_branch=branch)["selected_point"]
        if not available:
            point.update(available=False,
                         status="OUTSIDE_SPECTRUM_SUPPORT" if branch_matches else "INELIGIBLE_CONTEXT",
                         measured_gain_probe=float("nan"), measured_gain_conjugate=float("nan"))
        out["measured_gain_reference"] = point
    return out


def full_spectrum(D_GHz, T_K, P_pump_mW, P_probe_uW, line_strength, loss_pct,
                  L=L_CELL, *, mode_overlap_penalty=1.0,
                  polarization_penalty=1.0,
                   zeeman_participation_penalty=1.0,
                   w_pump=W_PUMP, w_probe=W_PROBE, qe=QE_DETECTOR,
                   detection_efficiency=None,
                   floquet_order=SEEDED_FLOQUET_ORDER,
                   phase_detail=PHASE_BALANCED,
                   pump_probe_angle_deg=SEEDED_PHASE_ANGLE_DEG,
                   model_fidelity=None,
                   gain_closure_enabled=False,
                   legacy_gain_calibration_fixture=False,
                   legacy_mean_field_fixture=False,
                   legacy_coupling_fixture=False,
                   measured_gain_calibration=False,
                   seed_input_assumption="",
                   velocity_step=4.0, velocity_cutoff=3.0,
                   velocity_measure="truncated",
                   response_method=RESPONSE_GRID,
                   scan_tolerance_dB=SCAN_TOLERANCE_DB,
                   scan_tolerance_log10_gain=SCAN_TOLERANCE_LOG10_GAIN,
                   scan_start_stride=SCAN_START_STRIDE,
                   transit_rate=constants.GAMMA_GG,
                  eom_residual_carrier_power=0.0,
                  eom_other_sidebands_power=0.0,
                  eom_seed_spectrum_provenance="direct FWM input",
                  eom_seed_spectrum_status="not supplied",
                  eom_seed_spectrum_application="unapplied"):
    """Wide scan with the two Raman channels calculated independently.

    The minus/plus branches are independent. They are evaluated sequentially
    because each compiled Floquet kernel already occupies the Numba worker pool;
    wrapping both in Python threads oversubscribes that pool and is slower on the
    benchmark hardware. This scheduling choice is not a numerical approximation.
    ``response_method`` and the scan settings are forwarded unchanged, so the
    pole-residue tiers solve each branch exactly as ``compute_spectrum`` does.
    Untiered calls retain the original model. The historical affine replay
    flags have the same compatibility policy as in ``compute_spectrum``.
    """
    fwm_gain_closure.reject_retired_replay(
        gain_closure_enabled, legacy_gain_calibration_fixture)
    common = dict(
        T=T_K, P_pump=P_pump_mW * 1e-3, P_probe=P_probe_uW * 1e-6,
        line_strength=line_strength, mode_overlap_penalty=mode_overlap_penalty,
        polarization_penalty=polarization_penalty,
        zeeman_participation_penalty=zeeman_participation_penalty,
        L=L, loss_frac=loss_pct / 100.0,
        w_pump=w_pump, w_probe=w_probe, qe=qe,
        detection_efficiency=detection_efficiency,
        floquet_order=floquet_order, enforce_floquet_convergence=True,
        phase_detail=phase_detail,
        pump_probe_angle_deg=pump_probe_angle_deg,
        model_fidelity=model_fidelity,
        gain_closure_enabled=gain_closure_enabled,
        legacy_gain_calibration_fixture=legacy_gain_calibration_fixture,
        legacy_mean_field_fixture=legacy_mean_field_fixture,
        legacy_coupling_fixture=legacy_coupling_fixture,
        measured_gain_calibration=measured_gain_calibration,
        seed_input_assumption=seed_input_assumption,
        transit_rate=transit_rate,
        eom_residual_carrier_power=eom_residual_carrier_power,
        eom_other_sidebands_power=eom_other_sidebands_power,
        eom_seed_spectrum_provenance=eom_seed_spectrum_provenance,
        eom_seed_spectrum_status=eom_seed_spectrum_status,
        eom_seed_spectrum_application=eom_seed_spectrum_application,
        coarse_points=301, fine_points=401,
        velocity_step=velocity_step, velocity_cutoff=velocity_cutoff,
        velocity_measure=velocity_measure,
        response_method=response_method, scan_tolerance_dB=scan_tolerance_dB,
        scan_tolerance_log10_gain=scan_tolerance_log10_gain,
        scan_start_stride=scan_start_stride)

    def _branch(b):
        with blas_single_thread():
            return fwm.compute_spectrum(D_GHz, branch=b, **common)

    minus = _branch(-1)
    plus = _branch(+1)
    return {"D_GHz": D_GHz, "minus": minus, "plus": plus}
