"""FWM model constants, settings, atomic definitions and claim metadata."""
from ...constants import OMEGA_HF
from ... import atoms, constants, core, detection, hyperfine, species
from dataclasses import dataclass
import numpy as np


# =========================================================
# Level indices (this scheme's labelling of the atom model)
# =========================================================
ATOM = atoms.get("double_lambda_rb85")
G1, G2, E2, E3 = 0, 1, 2, 3
GROUND_STATES = (G1, G2)
EXCITED_STATES = (E2, E3)
N_LEVELS = ATOM.n_levels
GROUND_F = {G1: 2, G2: 3}
EXCITED_F = {E2: 2, E3: 3}
TRANSITION_DIPOLE_SCALE = np.zeros((N_LEVELS, N_LEVELS), dtype=float)
for _g in GROUND_STATES:
    for _e in EXCITED_STATES:
        TRANSITION_DIPOLE_SCALE[_g, _e] = np.sqrt(
            3.0 * hyperfine.CF2[(GROUND_F[_g], EXCITED_F[_e])])


def physical_coupling_ledger(branch):
    """One-use ledger for the reduced four-level susceptibility scale.

    ``rho_ss`` is trace normalized over the two representative hyperfine
    manifolds, so its ground-state diagonal already supplies the pump-modified
    manifold population. Multiplying by the equilibrium ``p_F`` again would
    duplicate that population. The remaining structural factor is the
    ``1/[2(2I+1)] = 1/12`` sublevel average used by the AutoOD reference.

    The microscopic drive and polarization readout each contain
    ``sqrt(3 C_F^2)``; their product supplies ``3 C_F^2`` exactly once. Isotope
    abundance is absent here because ``hyperfine.number_density`` already
    returns the pure-85Rb cell density. Residual experimental factors are
    applied separately by ``SeededCouplingFactors``.
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")
    probe_F = GROUND_F[G2] if branch == -1 else GROUND_F[G1]
    return {
        "probe_ground_F": probe_F,
        "atomic_density_source": "pure-85Rb total density",
        "isotope_factor_applied_here": 1.0,
        "manifold_population_source": "trace-normalized rho_ss",
        "equilibrium_manifold_population_reference": hyperfine.GROUND_POP[probe_F],
        "external_manifold_population_factor": 1.0,
        "ground_sublevel_average": 1.0 / hyperfine.N_GROUND_SUBLEVELS,
        "dipole_strength_source": "sqrt(3 C_F^2) in drive and readout",
        "macroscopic_coupling_norm": 1.0 / hyperfine.N_GROUND_SUBLEVELS,
    }


def physical_coupling_norm(branch):
    """Macroscopic factor from :func:`physical_coupling_ledger`."""
    return physical_coupling_ledger(branch)["macroscopic_coupling_norm"]


@dataclass(frozen=True)
class SeededCouplingFactors:
    """Scalar residuals multiplying the seeded-FWM macroscopic coupling.

    ``reference_residual`` preserves an inherited historical factor.  It has not
    been refitted after the normalization and propagation corrections and is not
    an experimental gain or squeezing calibration.
    The three lab-facing terms are one-sided *additional effective-coupling
    penalties*, not independently calibrated power fractions. Keeping their
    defaults at one avoids inventing an arbitrary decomposition of the reference
    residual already absorbed by the anchor.
    """

    reference_residual: float
    mode_overlap_penalty: float = 1.0
    polarization_penalty: float = 1.0
    zeeman_participation_penalty: float = 1.0

    @property
    def lab_factor(self):
        return (float(self.mode_overlap_penalty)
                * float(self.polarization_penalty)
                * float(self.zeeman_participation_penalty))

    @property
    def combined_residual(self):
        return float(self.reference_residual) * self.lab_factor


@dataclass(frozen=True)
class PumpWeakResponseReference:
    """Slow pump-only, infinitesimal Nambu-response reference.

    ``chi_matrix`` has axes
    ``(analysis_frequency, two_photon_detuning, velocity, output, input)``
    with output/input order ``(probe, conjugate-star)``.  Its four entries are
    reduced susceptibilities per unit angular Rabi frequency and therefore have
    units of seconds.  The named ``chi_cs``/``chi_cc`` properties conjugate the
    second Nambu output row to match the historical positive-frequency tuple
    returned by ``chi_matrix_table``.
    """

    branch: int
    delta_axis_rad_s: np.ndarray
    analysis_frequency_axis_rad_s: np.ndarray
    relative_frequency_rad_s: np.ndarray
    delta_eff_axis_rad_s: np.ndarray
    pump_state: np.ndarray
    chi_matrix: np.ndarray
    diagnostics: dict
    provenance: dict

    @property
    def chi_ss(self):
        return self.chi_matrix[..., 0, 0]

    @property
    def chi_cs(self):
        return np.conj(self.chi_matrix[..., 1, 0])

    @property
    def chi_sc(self):
        return self.chi_matrix[..., 0, 1]

    @property
    def chi_cc(self):
        return np.conj(self.chi_matrix[..., 1, 1])


@dataclass(frozen=True)
class NoncollinearPumpWeakResponseReference:
    """Slow ``(v_z,v_x)`` Maxwell average of the pump-only Nambu response.

    The laboratory optical beat, RF analysis frequency, velocity-shifted Raman
    detuning, and one-photon detuning are stored separately.  ``chi_matrix`` has
    axes ``(analysis_frequency, lab_two_photon_detuning, output, input)`` in the
    raw ``(probe, conjugate-star)`` Nambu convention.
    """

    branch: int
    lab_delta_axis_rad_s: np.ndarray
    lab_optical_beat_axis_rad_s: np.ndarray
    analysis_frequency_axis_rad_s: np.ndarray
    one_photon_detuning_rad_s: float
    pump_k_rad_m: float
    probe_k_axis_rad_m: np.ndarray
    crossing_angle_rad: float
    vx_axis_m_s: np.ndarray
    vz_axis_m_s: np.ndarray
    vx_weights: np.ndarray
    vz_weights: np.ndarray
    raman_shift_grid_rad_s: np.ndarray
    one_photon_effective_axis_rad_s: np.ndarray
    pump_state: np.ndarray
    chi_matrix: np.ndarray
    diagnostics: dict
    provenance: dict

    @property
    def chi_ss(self):
        return self.chi_matrix[..., 0, 0]

    @property
    def chi_cs(self):
        return np.conj(self.chi_matrix[..., 1, 0])

    @property
    def chi_sc(self):
        return self.chi_matrix[..., 0, 1]

    @property
    def chi_cc(self):
        return np.conj(self.chi_matrix[..., 1, 1])

# =========================================================
# FWM experiment configuration (cell, beams, detection, scan)
# =========================================================
L_CELL = 12.5e-3
W_PUMP = 530e-6
W_PROBE = 330e-6
P_PUMP, P_PROBE = 600e-3, 10e-6
T_CELL = 394.15

# Sim et al. reference detection inputs. GABES exposes only their product η;
# SABES retains the device-level optical-loss/QE split.
QE_DETECTOR = 0.92
SEEDED_POST_CELL_LOSS_PCT = 5.5
SEEDED_DETECTION_EFFICIENCY_PCT = (
    QE_DETECTOR * (1.0 - SEEDED_POST_CELL_LOSS_PCT / 100.0) * 100.0)
RESPONSIVITY_AW = 0.59
LOSS_FRAC = 0.0
ETA_TOTAL = QE_DETECTOR * (1.0 - LOSS_FRAC)

PROBE_WAVELENGTH_NM = 795.0


def responsivity_AW(qe=QE_DETECTOR, wavelength_nm=PROBE_WAVELENGTH_NM):
    """Photodiode responsivity [A/W] implied by a quantum efficiency."""
    return detection.photodiode_responsivity_a_per_w(qe, float(wavelength_nm) * 1e-9)

OMEGA_C_SEED = 0.0
SCAN_MIN_GHZ = -8.0
SCAN_MAX_GHZ = 12.0
SCAN_COARSE_POINTS = 401
RESONANCE_WINDOW_MHZ = 80.0
SCAN_FINE_POINTS = 801
PUMP_OVERLAP_EXCLUSION_MHZ = 1e-3
VELOCITY_STEP_MPS = 1.0
VELOCITY_CUTOFF_SIGMA = 3.0
DELTA_GHZ_LIST = [0.9]
BRANCHES = (-1, +1)
DEFAULT_BRANCH = -1
PHASE_LEGACY = "legacy"
PHASE_BALANCED = "balanced"
PHASE_FINE = "fine"
PHASE_ULTRA = "ultra"
SEEDED_PHASE_ANGLE_DEG = 0.32
ULTRA_PHASE_ITERATIONS = 0   # Option A keeps dispersion in diagonal χ, not in Δk
ULTRA_PROPAGATION_SEGMENTS = 64
# Maxwell-averaged response estimators behind the Solver detail tiers.  All
# three evaluate the same finite-Floquet steady state on the same truncated
# Maxwell measure; they differ only in how that sum is carried out.
RESPONSE_GRID = "maxwell_grid"                    # one Floquet solve per velocity class
RESPONSE_POLE = "pole_residue"                    # exact pole sum, every probe point
RESPONSE_POLE_ADAPTIVE = "pole_residue_adaptive"  # exact pole sum, adaptive probe points
RESPONSE_METHODS = (RESPONSE_GRID, RESPONSE_POLE, RESPONSE_POLE_ADAPTIVE)
POLE_GUARD_RTOL = 1e-6              # pole form vs compiled kernel at Δ_eff = Δ, every row
SCAN_TOLERANCE_DB = 0.05            # adaptive probe scan: indicator disagreement
SCAN_TOLERANCE_LOG10_GAIN = 0.01    # adaptive probe scan: log10 G_s disagreement
SCAN_START_STRIDE = 32              # initial display stride (401 points -> 14 nodes)
SCAN_TOLERANCE_RANGE_FRACTION = 0.05  # never looser than 5 % of the curve's own range
SCAN_TOLERANCE_FLOOR_DB = 2e-3
SCAN_TOLERANCE_FLOOR_LOG10_GAIN = 2e-4
SEEDED_REFERENCE_RESIDUAL = 0.74
SEEDED_FLOQUET_ORDER = 3
FLOQUET_COMPLEX_RTOL = 0.01
FLOQUET_TRANSFER_ATOL = 1e-10
FLOQUET_RESPONSE_FLOOR_FRACTION = 1e-8
FLOQUET_GAIN_RTOL = 0.01
FLOQUET_PHASE_TOL_DEG = 0.5
FLOQUET_PHASE_AMPLITUDE_FLOOR = 1e-8

# Optional Ultra diagnostic.  The Maxwell diagonal already carries mean-field arm
# absorption, so this path applies only the phenomenological pump-scatter term.
# Distributed loss vacuum and atomic Langevin diffusion remain unavailable.
# ``kappa`` is an inherited, uncalibrated coefficient; disabling the option removes
# that diagnostic term but does not recover a physical squeezing prediction.
HARDENED_PUMP_SCATTER_KAPPA = 0.1


def assess_floquet_scan_convergence(
        *, high_order, low_order, high_response, low_response,
        high_transfer, low_transfer, scan_axis,
        complex_rtol=FLOQUET_COMPLEX_RTOL,
        transfer_atol=FLOQUET_TRANSFER_ATOL,
        response_floor_fraction=FLOQUET_RESPONSE_FLOOR_FRACTION,
        gain_rtol=FLOQUET_GAIN_RTOL,
        phase_tol_deg=FLOQUET_PHASE_TOL_DEG,
        phase_amplitude_floor=FLOQUET_PHASE_AMPLITUDE_FLOOR):
    """Apply the declared full-scan Floquet truncation criteria.

    ``high_response`` and ``low_response`` are mappings of named complex arrays;
    the transfer arrays have shape ``(scan, 2, 2)`` in the canonical photon-flux
    basis.  A single operating point cannot make this gate pass.
    """
    high_order = int(high_order)
    low_order = int(low_order)
    if high_order < 2 or low_order != high_order - 1:
        raise ValueError("Floquet convergence requires adjacent orders N and N-1")
    if set(high_response) != set(low_response):
        raise ValueError("high/low response components must have identical names")
    high_transfer = np.asarray(high_transfer, dtype=complex)
    low_transfer = np.asarray(low_transfer, dtype=complex)
    scan_axis = np.asarray(scan_axis, dtype=float)
    if scan_axis.ndim != 1 or scan_axis.size < 2:
        raise ValueError("a full-scan Floquet gate requires at least two scan points")
    expected_transfer_shape = (scan_axis.size, 2, 2)
    if (high_transfer.shape != expected_transfer_shape
            or low_transfer.shape != expected_transfer_shape):
        raise ValueError(
            f"high/low transfer arrays must have shape {expected_transfer_shape}")
    nonfinite_entries = int(
        high_transfer.size - np.count_nonzero(np.isfinite(high_transfer))
        + low_transfer.size - np.count_nonzero(np.isfinite(low_transfer))
        + scan_axis.size - np.count_nonzero(np.isfinite(scan_axis)))

    complex_components = {}
    complex_pass = True
    for name in sorted(high_response):
        high = np.asarray(high_response[name], dtype=complex)
        low = np.asarray(low_response[name], dtype=complex)
        if high.shape != (scan_axis.size,) or low.shape != (scan_axis.size,):
            raise ValueError(
                f"response component {name!r} must have full-scan shape "
                f"({scan_axis.size},)")
        component_nonfinite = int(
            high.size - np.count_nonzero(np.isfinite(high))
            + low.size - np.count_nonzero(np.isfinite(low)))
        nonfinite_entries += component_nonfinite
        difference = np.abs(high - low)
        finite_magnitudes = np.concatenate((
            np.abs(high[np.isfinite(high)]),
            np.abs(low[np.isfinite(low)]),
        ))
        component_scale = max(
            float(np.max(finite_magnitudes, initial=0.0)),
            np.finfo(float).tiny)
        # Reduced susceptibilities carry seconds, whereas transfer coefficients
        # are dimensionless.  A fixed 1e-10 absolute term would therefore allow
        # order-unity changes in a 1e-12 s response.  Use a full-scan component
        # scale for the response floor and reserve the fixed SI-free atol for T.
        response_floor = float(response_floor_fraction) * component_scale
        allowed = response_floor + float(complex_rtol) * np.maximum(
            np.abs(high), np.abs(low))
        ratio = np.full(difference.shape, np.inf, dtype=float)
        finite = np.isfinite(difference) & np.isfinite(allowed)
        ratio[finite] = difference[finite] / np.maximum(
            allowed[finite], np.finfo(float).tiny)
        passed = bool(np.all(ratio <= 1.0))
        complex_pass = complex_pass and passed
        complex_components[name] = {
            "max_abs_difference": (
                float("inf") if component_nonfinite else
                float(np.max(difference, initial=0.0))),
            "component_scale": component_scale,
            "absolute_floor": response_floor,
            "max_tolerance_ratio": float(np.max(ratio, initial=0.0)),
            "failed_points": int(np.count_nonzero(ratio > 1.0)),
            "nonfinite_entries": component_nonfinite,
            "passed": passed,
        }

    transfer_difference = np.abs(high_transfer - low_transfer)
    transfer_allowed = float(transfer_atol) + float(complex_rtol) * np.maximum(
        np.abs(high_transfer), np.abs(low_transfer))
    transfer_ratio = np.full(transfer_difference.shape, np.inf, dtype=float)
    finite_transfer = np.isfinite(transfer_difference) & np.isfinite(transfer_allowed)
    transfer_ratio[finite_transfer] = transfer_difference[finite_transfer] / np.maximum(
        transfer_allowed[finite_transfer], np.finfo(float).tiny)
    transfer_pass = bool(np.all(transfer_ratio <= 1.0))

    gain_high = {
        "probe_power": np.abs(high_transfer[:, 0, 0]) ** 2,
        "conjugate_photon_flux": np.abs(high_transfer[:, 1, 0]) ** 2,
    }
    gain_low = {
        "probe_power": np.abs(low_transfer[:, 0, 0]) ** 2,
        "conjugate_photon_flux": np.abs(low_transfer[:, 1, 0]) ** 2,
    }
    gain_components = {}
    gain_pass = True
    for name in gain_high:
        relative = np.full(gain_high[name].shape, np.inf, dtype=float)
        finite_gain = np.isfinite(gain_high[name]) & np.isfinite(gain_low[name])
        relative[finite_gain] = (
            np.abs(gain_high[name][finite_gain] - gain_low[name][finite_gain])
            / np.maximum(1.0, np.abs(gain_high[name][finite_gain])))
        passed = bool(np.all(relative < float(gain_rtol)))
        gain_pass = gain_pass and passed
        gain_components[name] = {
            "max_relative_difference": float(np.max(relative, initial=0.0)),
            "failed_points": int(np.count_nonzero(relative >= float(gain_rtol))),
            "passed": passed,
        }

    transfer_scale = np.maximum(
        1.0,
        np.maximum(
            np.max(np.abs(high_transfer), axis=(-2, -1)),
            np.max(np.abs(low_transfer), axis=(-2, -1))),
    )
    amplitude_threshold = float(phase_amplitude_floor) * transfer_scale[:, None, None]
    phase_mask = ((np.abs(high_transfer) > amplitude_threshold)
                  & (np.abs(low_transfer) > amplitude_threshold))
    wrapped_phase = np.degrees(np.angle(high_transfer * np.conj(low_transfer)))
    reported_phase = np.abs(wrapped_phase[phase_mask])
    max_phase = float(np.max(reported_phase, initial=0.0))
    phase_pass = bool(
        nonfinite_entries == 0 and max_phase < float(phase_tol_deg))

    probe_high = gain_high["probe_power"]
    probe_low = gain_low["probe_power"]
    optimum_finite = (
        probe_high.size > 0
        and np.isfinite(scan_axis).all()
        and np.isfinite(probe_high).all()
        and np.isfinite(probe_low).all())
    if optimum_finite:
        high_index = int(np.nanargmax(probe_high))
        low_index = int(np.nanargmax(probe_low))
        optimum_index_shift = abs(high_index - low_index)
        optimum_axis_shift = float(abs(scan_axis[high_index] - scan_axis[low_index]))
        optimum_pass = optimum_index_shift <= 1
    else:
        high_index = low_index = None
        optimum_index_shift = None
        optimum_axis_shift = None
        optimum_pass = False

    passed = bool(
        nonfinite_entries == 0 and complex_pass and transfer_pass
        and gain_pass and phase_pass and optimum_pass)
    return {
        "status": "CONVERGED" if passed else "UNCONVERGED",
        "passed": passed,
        "high_order": high_order,
        "comparison_order": low_order,
        "full_scan_points": int(scan_axis.size),
        "finite": nonfinite_entries == 0,
        "nonfinite_entries": nonfinite_entries,
        "criteria": {
            "complex_rtol": float(complex_rtol),
            "transfer_atol": float(transfer_atol),
            "response_floor_fraction_of_component_scan_max": float(
                response_floor_fraction),
            "gain_rtol": float(gain_rtol),
            "phase_tolerance_deg": float(phase_tol_deg),
            "phase_amplitude_floor": float(phase_amplitude_floor),
            "optimum_max_grid_intervals": 1,
        },
        "response_components": complex_components,
        "transfer": {
            "max_abs_difference": (
                float("inf") if not np.isfinite(transfer_difference).all()
                else float(np.max(transfer_difference, initial=0.0))),
            "max_tolerance_ratio": float(np.max(transfer_ratio, initial=0.0)),
            "failed_coefficients": int(np.count_nonzero(transfer_ratio > 1.0)),
            "passed": transfer_pass,
        },
        "gains": gain_components,
        "wrapped_phase": {
            "max_difference_deg": max_phase,
            "reported_coefficients": int(np.count_nonzero(phase_mask)),
            "passed": phase_pass,
        },
        "probe_gain_optimum": {
            "high_index": high_index,
            "comparison_index": low_index,
            "index_shift": optimum_index_shift,
            "axis_shift": optimum_axis_shift,
            "passed": bool(optimum_pass),
        },
    }


def pump_weak_response_reference_provenance():
    """Machine-readable boundary of the independent pump-only reference."""
    return {
        "solver_id": "static_pump_trace_zero_nambu_reference",
        "state_equation": "self-consistent trace-one 4-level pump Liouvillian",
        "rotating_frame": (
            "U=exp(+i*Omega_beat*t*|g2><g2|), default minus Raman branch"),
        "frame_equivalence": (
            "unitarily equivalent to the pump-only finite-Floquet state; "
            "executable gauge-parity fixture required"),
        "weak_response": (
            "analytic complex-amplitude derivative on the trace-zero subspace"),
        "dc_solver": "trace-constrained stationary-subspace projection",
        "analysis_frequency": "independent angular-frequency input",
        "reference_fields": "none",
        "supported_branches": (-1,),
        "unsupported_branches": {
            +1: (
                "the inherited plus-branch finite-seed frame is not gauge-"
                "equivalent to the physical static pump frame")},
        "production_default": False,
        "doppler_geometry": "longitudinal one-dimensional reference only",
        "quantum_noise": "not implemented",
    }


def noncollinear_doppler_reference_provenance():
    """Machine-readable boundary of the two-dimensional Maxwell reference."""
    return {
        "solver_id": "pump_trace_zero_nambu_2d_maxwell_reference",
        "atomic_response": pump_weak_response_reference_provenance()["solver_id"],
        "velocity_geometry": "tensor (v_z,v_x) truncated-Maxwell quadrature",
        "one_photon_detuning": "Delta_eff=Delta_lab-k_pump*v_z",
        "two_photon_detuning": (
            "delta_eff=delta_lab+(k_pump-k_probe*cos(theta))*v_z"
            "-k_probe*sin(theta)*v_x"),
        "lab_optical_beat": "independent input; never velocity shifted",
        "analysis_frequency": "independent angular-frequency input",
        "floquet_certificate": (
            "minus-branch static pump frame pinned to the N_F=3 pump-only "
            "Floquet state"),
        "supported_branches": (-1,),
        "production_default": False,
        "quadrature": "Gauss-Legendre nodes with explicit Maxwell weights/cutoff",
        "angular_distribution": "single declared crossing angle; no beam divergence",
        "segmentwise_pump_recomputation": False,
        "quantum_noise": "not implemented",
    }


def seeded_atomic_solver_provenance(*, floquet_order=SEEDED_FLOQUET_ORDER,
                                    convergence=None):
    """Machine-readable provenance for the production seeded atomic response."""
    floquet_order = int(floquet_order)
    convergence_status = (
        "not evaluated" if convergence is None else convergence.get("status"))
    return {
        "solver_id": "finite_seed_finite_floquet_density_matrix",
        "state_equation": "full density-matrix Liouvillian null solve",
        "reference_fields": "finite probe/conjugate reference fields",
        "floquet_order": floquet_order,
        "floquet_modes": tuple(range(-floquet_order, floquet_order + 1)),
        "floquet_convergence_status": convergence_status,
        "floquet_comparison_order": (
            None if convergence is None else convergence.get("comparison_order")),
        "pump_only_self_consistent_nullspace": False,
        "pump_only_weak_response_reference_available": True,
        "noncollinear_doppler_reference_available": True,
        "pump_only_reference_solver_id": (
            pump_weak_response_reference_provenance()["solver_id"]),
        "rate_sylvester_approximation": False,
        "scope": "production seeded-FWM response",
        "target_model_status": (
            "pump-only weak-response and two-dimensional non-collinear Doppler "
            "references are separate from production; quantum-noise response "
            "is not implemented"),
    }


def seeded_parameter_provenance(*, line_strength, transit_rate,
                                pump_scatter_kappa=HARDENED_PUMP_SCATTER_KAPPA):
    """Provenance ledger; sweep ranges are illustrative, not uncertainties."""
    return {
        "ell_s": {
            "value": float(line_strength),
            "source": "inherited historical residual",
            "calibration_status": "not refitted after normalization/sign corrections",
            "illustrative_sweep": (0.666, 0.74, 0.814),
            "range_kind": "illustrative sensitivity sweep; not an uncertainty interval",
        },
        "kappa": {
            "value": float(pump_scatter_kappa),
            "source": "phenomenological pump-scatter diagnostic coefficient",
            "calibration_status": "unvalidated and not fitted",
            "illustrative_sweep": (0.0, 0.1, 0.2),
            "range_kind": "illustrative sensitivity sweep; not an uncertainty interval",
        },
        "gamma_transit": {
            "value_rad_s": float(transit_rate),
            "value_hz": float(transit_rate) / (2.0 * np.pi),
            "source": "inherited transit/residual floor",
            "calibration_status": "independent fit unavailable",
            "illustrative_sweep_hz": (90e3, 100e3, 110e3),
            "range_kind": "illustrative sensitivity sweep; not an uncertainty interval",
        },
    }


def seeded_validation_claim_gate(*, canonical_mode_status,
                                 commutator_defect_max,
                                 floquet_convergence=None,
                                 eom_residual_carrier_power=0.0,
                                 eom_other_sidebands_power=0.0,
                                 eom_spectrum_status="not supplied",
                                 eom_spectrum_application="unapplied",
                                 response_estimator=None):
    """Return the explicit validation boundary for seeded-FWM outputs."""
    defect = float(np.nanmax(np.asarray(commutator_defect_max, dtype=float)))
    reasons = [
        "atomic response uses finite probe/conjugate reference fields",
        "pump-only weak response exists only as a separate slow reference",
        "two-dimensional non-collinear Doppler averaging exists only as a "
        "separate slow reference; production remains one-dimensional",
        "ell_s, kappa, and gamma_transit lack independent calibration",
        "frequency-dependent atomic Langevin diffusion/covariance is unavailable",
    ]
    floquet_status = (
        "NOT_EVALUATED" if floquet_convergence is None
        else str(floquet_convergence.get("status", "NOT_EVALUATED")))
    if floquet_status != "CONVERGED":
        reasons.append(
            f"full-scan Floquet truncation status is {floquet_status}")
    if response_estimator and response_estimator.get("interpolation", "none") != "none":
        solved = response_estimator.get("solved_probe_points")
        shown = response_estimator.get("display_probe_points")
        reasons.append(
            f"{shown - solved} of {shown} displayed probe detunings are cubic-spline "
            "interpolated between exactly solved detunings (Fast solver detail)")
    if "conditional" in str(canonical_mode_status).lower():
        reasons.append("conjugate collected-mode area is assumed rather than measured")
    eom_unapplied = str(eom_spectrum_application).strip().lower() != "applied"
    eom_unsupported = str(eom_spectrum_status).strip().lower() not in {
        "supported", "calibrated"}
    if (float(eom_residual_carrier_power) > 0.0
            or float(eom_other_sidebands_power) > 0.0
            or eom_unapplied or eom_unsupported):
        reasons.append(
            "declared EOM residual components are passed through but their "
            f"spectrum model is {eom_spectrum_status}/{eom_spectrum_application}")
    badges = [
        "MEAN_FIELD_DIAGNOSTIC",
        "QUANTITATIVE_GAIN_UNSUPPORTED",
        "PHYSICAL_SQUEEZING_UNAVAILABLE",
    ]
    if floquet_status != "CONVERGED":
        badges.append("FLOQUET_UNCONVERGED")
    return {
        "level": "MEAN_FIELD_DIAGNOSTIC",
        "badges": tuple(badges),
        "mean_field_gain_diagnostic_available": True,
        "quantitative_gain_supported": False,
        "physical_squeezing_prediction": False,
        "experimental_agreement_claim_allowed": False,
        "physical_claims_allowed": False,
        "commutator_defect_max": defect,
        "floquet_convergence_status": floquet_status,
        "blocked_claims": (
            "absolute quantitative gain",
            "physical squeezing spectrum or bandwidth",
            "agreement with experimental gain/squeezing",
        ),
        "reasons": tuple(reasons),
    }


def seeded_sideband_beat(delta, branch):
    """Positive pump-sideband spacing for the seeded Raman branch.

    The scan coordinate is probe = pump + branch*nu_HF + delta.  Therefore the
    physical optical beat magnitude is nu_HF - delta on the (-) branch and
    nu_HF + delta on the (+) branch.
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch}")
    return OMEGA_HF + branch * delta


# =========================================================
# Generic SFWM / biphoton topology layer
# =========================================================
NM = 1e-9
MHZ_ANG = 2 * np.pi * 1e6

MODE_SEEDED = "Gain diagnostic"  # stored value; UI label is "Squeezing"
MODE_SEEDED_LEGACY = "Squeezing"  # accepted implicitly by compute() as seeded
MODE_BIPHOTON = "Biphoton"
MODE_LABELS = {
    MODE_SEEDED: "Squeezing",
    MODE_BIPHOTON: "Biphoton",
}

NOISE_CONSTANT = "constant"
NOISE_GAIN_PROPORTIONAL = "gain_proportional"
NOISE_MODE_LABELS = {
    NOISE_CONSTANT: "Constant Noise N",
    NOISE_GAIN_PROPORTIONAL: "Gain proportional Noise a",
}

# Historical stored model keys below are retained for compatibility. The reduced
# coherent waveform uses Chen et al. PRR4,023132(2024), Kim et al. QST9,045006(2024)
# and Du et al. JOSAB25,C98(2008); this does not derive the absolute collection
# rate. Rb/Cs rates are anchored and diamond coefficients are assumed template
# values. Numerical velocity refinement checks FWHM, not the complete waveform.
BIPHOTON_PREDICTIVE = "Predictive (first-principles)"
BIPHOTON_CALIBRATED = "Calibrated (reference)"
BIPHOTON_MODELS = (BIPHOTON_PREDICTIVE, BIPHOTON_CALIBRATED)
BIPHOTON_MODEL_LABELS = {
    BIPHOTON_PREDICTIVE: "Reduced model",
    BIPHOTON_CALIBRATED: "Reference model",
}

# Two-photon (ground) coherence dephasing as a fraction of the intermediate Γ;
# sets the EIT/Raman two-photon linewidth. Chen et al. fit γ ≈ 0.02–0.03 Γ.
GROUND_DEPHASING_FRAC = 0.02
# Regularization clip for the complex longitudinal function ρ̄ at high OD
# (a numerical regularizer for the separate biphoton longitudinal model).
PRED_RHO_CLIP = 60.0

# Predictive velocity-grid auto-refinement. The nonlinear source |amp(v)| that the
# velocity-class coherent sum integrates is only ~Γ/k — a few m/s — wide, far
# narrower than the Doppler width σ_v the navigate-only `biphoton_velocity_step`
# is sized for. A step coarser than that resonance aliases the (Fourier) sum, so
# the reported absolute BTW width tracks numerical undersampling, not physics
# (e.g. a factor-~20, non-monotonic swing over vstep=1–12 m/s in the 780/1529 nm
# telecom cascade). The predictive path therefore starts from a step that
# oversamples the *measured* resonance, then halves until the |ψ|² FWHM is stable;
# if the point cap is hit first the width is flagged unconverged.
PRED_V_OVERSAMPLE = 16.0      # starting step = probe-measured resonance FWHM / this
PRED_V_FWHM_TOL = 0.03        # relative |ψ|² FWHM change accepted as converged
PRED_V_MAX_REFINE = 10        # max successive halvings of the velocity step
PRED_V_MAX_POINTS = 40000     # velocity-grid point cap (guards runtime)

TOPOLOGY_RB87_TELECOM = "cascade_rb87_telecom"
TOPOLOGY_CS_BTW = "cascade_cs_btw"
TOPOLOGY_DIAMOND = "diamond_generic"
TOPOLOGY_LABELS = {
    TOPOLOGY_RB87_TELECOM: "⁸⁷Rb telecom cascade",
    TOPOLOGY_CS_BTW: "¹³³Cs cascade",
    TOPOLOGY_DIAMOND: "Generic diamond SFWM",
}
CS_CHANNEL_917 = "6D5/2: 852-917 nm"
CS_CHANNEL_795 = "8S1/2: 852-795 nm"
CS_CHANNEL_LABELS = {
    CS_CHANNEL_917: "852/917 nm (6D₅/₂)",
    CS_CHANNEL_795: "852/795 nm (8S₁/₂)",
}
SIDE_PLUS = "+"
SIDE_MINUS = "-"
SIDE_CHOICES = (SIDE_PLUS, SIDE_MINUS)


@dataclass(frozen=True)
class LevelSpec:
    """Lightweight FWM level metadata used by the generic topology layer."""
    name: str
    energy_hz: float
    gamma_mhz: float = 0.0


@dataclass(frozen=True)
class FieldSpec:
    """A driven or generated optical field in a four-wave-mixing topology."""
    role: str
    lower: int
    upper: int
    wavelength_nm: float
    detuning_mhz: float = 0.0
    rabi_mhz: float = 0.0
    phase_sign: float = 1.0
    direction: float = 1.0
    angle_deg: float = 0.0
    side_sign: float = 0.0

    @property
    def k(self):
        return 2 * np.pi / (self.wavelength_nm * NM)

    @property
    def frequency_hz(self):
        return constants.C_LIGHT / (self.wavelength_nm * NM)


@dataclass(frozen=True)
class TopologySpec:
    """Generic SFWM topology; presets carry the reference-calibrated constants."""
    name: str
    label: str
    family: str
    isotope_name: str
    levels: tuple
    fields: tuple
    signal_role: str
    idler_role: str
    default_temp_c: float
    default_cell_mm: float
    default_pump_uw: float
    default_coupling_mw: float
    pair_rate_cps_per_mw: float
    emission_decay_ns: float
    target_g2_peak: float | None = None
    reference_fwhm_ns: float | None = None
    reference_od: float | None = None
    reference_bandwidth_mhz: float | None = None
    reference_width_ratio: float | None = None
    reference_delta_k: float | None = None
    notes: str = ""

    @property
    def isotope(self):
        return species.ISOTOPES[self.isotope_name]

    @property
    def field_map(self):
        return {f.role: f for f in self.fields}


def thermal_transit_reset_superoperator(rate, populations=None):
    """Trace-preserving ``rate·(rho_th Tr(rho) - rho)`` superoperator.

    ``rho_th`` defaults to an unpolarized thermal mixture of the representative
    F=2/F=3 ground manifolds with degeneracy weights 5/12 and 7/12.  The reset
    models atoms leaving the optical mode and fresh thermal atoms entering it;
    unlike pure ground-coherence dephasing, it also replenishes populations.
    """
    rate = float(rate)
    if rate < 0.0:
        raise ValueError("transit reset rate must be non-negative")
    if populations is None:
        populations = (
            hyperfine.GROUND_POP[GROUND_F[G1]],
            hyperfine.GROUND_POP[GROUND_F[G2]],
            0.0,
            0.0,
        )
    populations = np.asarray(populations, dtype=float)
    if populations.shape != (N_LEVELS,):
        raise ValueError(f"populations must have shape ({N_LEVELS},)")
    if np.any(populations < 0.0) or not np.isfinite(populations).all():
        raise ValueError("transit target populations must be finite and non-negative")
    total = float(np.sum(populations))
    if total <= 0.0:
        raise ValueError("transit target populations must have positive trace")
    return core.thermal_reset_superoperator(rate, populations / total)


def collisional_atom(T, density=None, *, transit_rate=None):
    """Per-temperature double-Λ atom with collisions and thermal transit reset.

      • optical (g,e) coherence — Rb self-broadening, added collisional
        half-width (Γ_eff − Γ)/2 via the AutoOD-validated
        `hyperfine.self_broadened_gamma`;
      • ground Raman (g₁,g₂) coherence — Rb–Rb pure dephasing plus a
        trace-preserving thermal transit reset.  The reset supplies the inherited
        transit/residual floor exactly once while replenishing populations.

    Because χ̄ now depends on temperature through this dissipator, any cached-χ̄
    scanner must rebuild its table per temperature with this atom (a single
    all-temperature table is no longer valid). Shared by `compute_spectrum` and
    the analysis scanners so the optimum-finder sees the same physics.
    """
    if density is None:
        density = hyperfine.number_density(T)
    transit_rate = constants.GAMMA_GG if transit_rate is None else float(transit_rate)
    if transit_rate < 0.0:
        raise ValueError("transit_rate must be non-negative")
    gamma_collision = constants.ground_coherence_dephasing(
        T, density, floor=0.0)
    gamma_opt = 0.5 * (hyperfine.self_broadened_gamma(density) - constants.GAMMA)
    atom = atoms.double_lambda_rb85(
        gamma_gg=gamma_collision, gamma_opt=max(gamma_opt, 0.0))
    atom.lindblad = atom.lindblad + thermal_transit_reset_superoperator(transit_rate)
    atom.transit_reset_rate = transit_rate
    atom.ground_collision_dephasing_rate = gamma_collision
    atom.thermal_reset_populations = (
        hyperfine.GROUND_POP[GROUND_F[G1]],
        hyperfine.GROUND_POP[GROUND_F[G2]],
        0.0,
        0.0,
    )
    return atom


# =========================================================
# Scheme wrapper for the generic front-end
# =========================================================
WINDOW_GHZ = 0.55          # half-width of the focused probe window around (−) Raman
TPD_LIMIT_MHZ = 500.0
# Three tiers share the atomic solve. Ultra is the brute-force reference: one compiled
# finite-Floquet solve per (probe detuning, Maxwell class) on a 1 m/s, 4σ grid.
# Fast and Balanced evaluate the same steady state, the same Maxwell measure and
# the same atomic response exactly through poles and residues (one eigenproblem
# per probe detuning instead of ~1,600 solves); Balanced solves all 401 probe
# detunings, Fast solves an adaptively refined subset and splines the averaged
# responses between them.  The two-branch full probe-scan view uses the same
# model and measure but solves every displayed probe detuning for both pole
# tiers: its 0.4 MHz resonance windows are finer than the display resolution the
# adaptive refinement was validated at. Retired cross-coupling and local gain
# calibrations are available only through the explicit historical fixture;
# current tiers use neither calibration. Historical lab fixtures include up to
# ten additional 1-MHz gain nodes. See
# analysis/fwm_lite/DEVLOG.md and analysis/fwm_gain_hotfix/DEVLOG.md.
# Labels estimate the reference (deployment) environment the earlier labels used.
FIDELITY_FAST = "Fast  (~1 s)"
FIDELITY_BALANCED = "Balanced  (~2 s)"
FIDELITY_ULTRA = "Ultra  (slow)"
FIDELITY_LABELS = {
    FIDELITY_FAST: "Fast",
    FIDELITY_BALANCED: "Balanced",
    FIDELITY_ULTRA: "Ultra",
}
FWM_FIDELITY = {
    FIDELITY_FAST:     dict(coarse_points=401, velocity_step=1.0,
                            velocity_cutoff=4.0, phase_detail=PHASE_ULTRA,
                            response_method=RESPONSE_POLE_ADAPTIVE,
                            scan_tolerance_dB=SCAN_TOLERANCE_DB,
                            scan_tolerance_log10_gain=SCAN_TOLERANCE_LOG10_GAIN,
                            scan_start_stride=SCAN_START_STRIDE),
    FIDELITY_BALANCED: dict(coarse_points=401, velocity_step=1.0,
                            velocity_cutoff=4.0, phase_detail=PHASE_ULTRA,
                            response_method=RESPONSE_POLE),
    FIDELITY_ULTRA:    dict(coarse_points=401, velocity_step=1.0,
                            velocity_cutoff=4.0, phase_detail=PHASE_ULTRA,
                            response_method=RESPONSE_POLE),
}
RESOLUTION = FWM_FIDELITY


def _tier_response_method(fidelity, velocity_measure, *, full_scan=False):
    if fidelity == FIDELITY_ULTRA and velocity_measure == "truncated":
        return RESPONSE_GRID
    method = FWM_FIDELITY[fidelity]["response_method"]
    return RESPONSE_POLE if full_scan else method

# Old saved fidelity labels → current labels. Settings-preserving where a tier
# survived (old Balanced/High kept their solver settings under the new names);
# the removed coarse tier falls to the cheapest survivor.
_FIDELITY_LEGACY = {
    "Fast  (~4 s)":           FIDELITY_FAST,      # grid Fast before the pole remaster
    "Balanced  (~12 s)":      FIDELITY_BALANCED,  # grid Balanced before the pole remaster
    "Fast  (~3 s)":           FIDELITY_FAST,      # removed 121-pt tier → cheapest now
    "Balanced  (~6 s)":       FIDELITY_FAST,      # same 181-pt settings, renamed
    "High fidelity  (~20 s)": FIDELITY_BALANCED,  # same 301-pt settings, renamed
    "Fine  (~20 s)":          FIDELITY_BALANCED,  # legacy alias of the old High tier
}


def normalize_fidelity(value):
    """Map old saved labels onto current user-facing fidelity labels."""
    if value in FWM_FIDELITY:
        return value
    return _FIDELITY_LEGACY.get(value, FIDELITY_FAST)
