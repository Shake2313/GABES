"""Cluster D — 85Rb D1 double-Λ four-wave mixing.

The seeded path is a maintained descendant of ``fwm_obe.py``.  It now differs
materially in normalization, Maxwell-sign/phase conventions, collected-mode
conversion, transit reset, and validation metadata.  Its present atomic response
is a finite-reference-field density-matrix Floquet solve, reported at N_F>=3
only after an adjacent-order full-scan check; it remains a mean-field diagnostic,
not a quantum-Langevin model.  A separate slow reference now solves the exact
static pump state and its infinitesimal trace-zero Nambu response for the
standard minus Raman branch; it is not silently substituted into production.

``compute_spectrum`` / ``operating_point`` retain compatibility keys for older
callers.  ``FWMScheme`` wraps them for the generic Streamlit front-end.
"""

# Keep the historical module API, including scanner and archive imports.
import copy
import functools
import json
import math
import numpy as np
from dataclasses import dataclass
from ... import (
    adaptive_scan, atoms, beam, constants, detection, doppler, fwm_gain_closure,
    fwm_stark_response, fwm_experiment_gain, fwm_measured_gain, gain_reference, hyperfine, kernels,
    observables, pole_doppler, species,
)
from ...constants import K_VEC, OMEGA_HF, OMEGA_EXCITED_HF, rabi_freq
from ...core import (
    _EXP_ARG_CLAMP, _transpose_perm, blas_single_thread, build_liouvillian, comm_super,
    hermitian_basis, floquet_solve_truncated, liouvillian_pole_residue_response,
    steady_state_batched, trace_zero_liouvillian_response,
)
from ...linear_hyperfine import (
    analytic_hyperfine_line as _analytic_hyperfine_line, hyperfine_alpha_analytic as
    _hyperfine_alpha_analytic,
)
from ...lineshape import fwhm_interp
from ...biphoton_evidence import biphoton_output_evidence
from ...passive_readout import passive_l1_from_spectrum
from ...jasperse import seed_noise_components
from ...fwm_gain_calibration import (
    MODEL_LAB, MODEL_LITERATURE, MODEL_CHI, MODEL_STARK, MODEL_ZEEMAN, MODELS as
    GAIN_CALIBRATION_MODELS, calibrate_spectrum as calibrate_experimental_gain, evaluate as
    evaluate_experimental_gain,
)
from ...fwm_experiment_noise import experimental_jasperse_readout
from ...fwm_quadrature import experimental_quadrature_phase_scan
from ...fwm_efficiency import dressed_cell_transmission
from ...dressed_cell_absorption import selected_dressed_cell_absorption
from ..base import ExtraView, ParamSpec, Preset, Scheme
from .common import (
    ATOM, G1, G2, E2, E3, GROUND_STATES, EXCITED_STATES, N_LEVELS, GROUND_F, EXCITED_F,
    TRANSITION_DIPOLE_SCALE, _g, _e, physical_coupling_ledger, physical_coupling_norm,
    SeededCouplingFactors, PumpWeakResponseReference, NoncollinearPumpWeakResponseReference,
    L_CELL, W_PUMP, W_PROBE, P_PUMP, P_PROBE, T_CELL, QE_DETECTOR, SEEDED_POST_CELL_LOSS_PCT,
    SEEDED_DETECTION_EFFICIENCY_PCT, RESPONSIVITY_AW, LOSS_FRAC, ETA_TOTAL, PROBE_WAVELENGTH_NM,
    responsivity_AW, OMEGA_C_SEED, SCAN_MIN_GHZ, SCAN_MAX_GHZ, SCAN_COARSE_POINTS,
    RESONANCE_WINDOW_MHZ, SCAN_FINE_POINTS, PUMP_OVERLAP_EXCLUSION_MHZ, VELOCITY_STEP_MPS,
    VELOCITY_CUTOFF_SIGMA, DELTA_GHZ_LIST, BRANCHES, DEFAULT_BRANCH, PHASE_LEGACY, PHASE_BALANCED,
    PHASE_FINE, PHASE_ULTRA, SEEDED_PHASE_ANGLE_DEG, ULTRA_PHASE_ITERATIONS,
    ULTRA_PROPAGATION_SEGMENTS, RESPONSE_GRID, RESPONSE_POLE, RESPONSE_POLE_ADAPTIVE,
    RESPONSE_METHODS, POLE_GUARD_RTOL, SCAN_TOLERANCE_DB, SCAN_TOLERANCE_LOG10_GAIN,
    SCAN_START_STRIDE, SCAN_TOLERANCE_RANGE_FRACTION, SCAN_TOLERANCE_FLOOR_DB,
    SCAN_TOLERANCE_FLOOR_LOG10_GAIN, SEEDED_REFERENCE_RESIDUAL, SEEDED_FLOQUET_ORDER,
    FLOQUET_COMPLEX_RTOL, FLOQUET_TRANSFER_ATOL, FLOQUET_RESPONSE_FLOOR_FRACTION,
    FLOQUET_GAIN_RTOL, FLOQUET_PHASE_TOL_DEG, FLOQUET_PHASE_AMPLITUDE_FLOOR,
    HARDENED_PUMP_SCATTER_KAPPA, assess_floquet_scan_convergence,
    pump_weak_response_reference_provenance, noncollinear_doppler_reference_provenance,
    seeded_atomic_solver_provenance, seeded_parameter_provenance, seeded_validation_claim_gate,
    seeded_sideband_beat, NM, MHZ_ANG, MODE_SEEDED, MODE_SEEDED_LEGACY, MODE_BIPHOTON, MODE_LABELS,
    NOISE_CONSTANT, NOISE_GAIN_PROPORTIONAL, NOISE_MODE_LABELS, BIPHOTON_PREDICTIVE,
    BIPHOTON_CALIBRATED, BIPHOTON_MODELS, BIPHOTON_MODEL_LABELS, GROUND_DEPHASING_FRAC,
    PRED_RHO_CLIP, PRED_V_OVERSAMPLE, PRED_V_FWHM_TOL, PRED_V_MAX_REFINE, PRED_V_MAX_POINTS,
    TOPOLOGY_RB87_TELECOM, TOPOLOGY_CS_BTW, TOPOLOGY_DIAMOND, TOPOLOGY_LABELS, CS_CHANNEL_917,
    CS_CHANNEL_795, CS_CHANNEL_LABELS, SIDE_PLUS, SIDE_MINUS, SIDE_CHOICES, LevelSpec, FieldSpec,
    TopologySpec, thermal_transit_reset_superoperator, collisional_atom, WINDOW_GHZ, TPD_LIMIT_MHZ,
    FIDELITY_FAST, FIDELITY_BALANCED, FIDELITY_ULTRA, FIDELITY_LABELS, FWM_FIDELITY, RESOLUTION,
    _tier_response_method, _FIDELITY_LEGACY, normalize_fidelity,
)
from .phase_matching import (
    _wavevector_nm, _side_sign, _side_label, transverse_matched_angle_deg, _field_with,
    phase_mismatch, phase_mismatch_vector, phase_matching_weight, _sinc_complex,
    _bandwidth_from_waveform_mhz, phase_mismatch_grid, phase_mismatch_vector_grid,
    biphoton_phase_matching_map, energy_mismatch_hz, _optical_k_from_offset,
    seeded_option_a_wavenumbers, seeded_phase_mismatch_z, _gaussian_overlap_profile,
)
from .biphoton import (
    _raw_delta_k, _with_reference_delta_k, _rb87_telecom_spec, _cs_btw_spec, _diamond_generic_spec,
    _optional_float, _topology_cache_key, _topology_from_key, topology_from_params,
    _default_biphoton_geometry, GenericFWMSolver, _solver_from_topology_key,
    _BIPHOTON_READOUT_PARAM_KEYS, _biphoton_source_items, _cached_biphoton_source,
)
from .references import (
    _arm_linear_od, _pump_scatter_noise, _reference_population_factors, _arm_linear_od_analytic,
    _pump_scatter_noise_analytic, _zeeman_consistency_cached,
)
from .seeded_response import (
    _add_static_drive, _add_sideband_drive, _polarization_coherence, pump_hamiltonian_at_Deff_zero,
    pump_frame_to_seeded_harmonics, _pump_nambu_operators, _pump_reference_system,
    pump_only_weak_response_reference, pump_only_weak_response_noncollinear_reference,
    pump_only_pole_residue_reference, static_hamiltonian_at_Deff_zero, sideband_hamiltonian,
    sideband_template, _coherence_weights, chi_matrix_table, _seeded_pole_systems,
    _finite_seed_angular_rows, finite_seed_angular_reference, _pole_scan_response, ChiPoleRows,
    chi_pole_rows,
)
from .seeded import (
    branch_center_GHz, probe_scan_axis_GHz, two_photon_detuning_from_probe_scan,
    _uniform_segment_profile_and_probe_od, _ultra_segmented_gain, _single_branch, compute_spectrum,
    operating_point, full_spectrum,
)
from .readout import (
    _measured_gain_context, _measured_gain_rf_roundoff, _measured_gain_reference_readout,
    _detection_efficiency_from_params, _seeded_excess_noise, _seeded_noise_caption,
    _seeded_passive_l1_readout, _seeded_sidebar_readouts, _seeded_absorption_readout,
    _passive_l1_selected, NOISE_PASSIVE_L1, NOISE_EXPERIMENTAL_JASPERSE,
    SQUEEZING_INTENSITY_DIFFERENCE, SQUEEZING_QUADRATURE_PHASE, _power_gain_metric,
    _experimental_noise_selected, _focused_readout_reference, _focused_readout_source,
    _seeded_readout_rows, seeded_operating_readout, _plot_reference, _label_noise_axis,
)
from .scheme import FWMScheme
