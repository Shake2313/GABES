"""Practical resolved-Zeeman pump response with train-only cross-chi residual.

Native spectra and their canonical/noise provenance are retained separately.
This model assumes crossed linear polarization, zero B and a uniform pump;
its microscopic state/linear response does not supply a quantum diffusion map.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import constants, doppler, fwm_chi_gain as propagation, fwm_zeeman_response as atom
from . import hyperfine, observables
from .fwm_experiment_gain import _axis, _sample_model
from .fwm_literature_gain import _domain, features

MODEL_ID = "fwm_literature_zeeman_response_v1"
PROFILE_PATH = Path(__file__).with_name("data") / "fwm_zeeman_gain_v1.json"
PHYSICAL_KEYS = ("D_GHz", "temperature_K", "input_pump_power_W", "input_seed_power_W",
    "w_pump_m", "w_probe_m", "cell_length_m", "pump_probe_angle_deg", "branch",
    "raman_center_minus_GHz", "raman_center_plus_GHz")


@lru_cache(maxsize=1)
def _record():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def profile():
    record = copy.deepcopy(_record())
    record.update(model_id=MODEL_ID,
        method="Resolved 24-state optical pumping, consistent pump refractive phase and segmented propagation",
        formula="24-state steady state -> chi; delta_k=delta_k_vac+kp*Re(chi_pump); chi_offdiag'=kappa*chi_offdiag; chi_diag unchanged",
        noise_covariance_calibrated=False, susceptibility_identified=False,
        physical_assumptions="B=0; crossed linear fields; uniform pump; infinitesimal weak fields; collinear atomic Doppler")
    return record


def context_from_params(params):
    """Offline physical/geometry context without solving the native atom."""
    from .schemes import fwm
    settings = dict(fwm.FWMScheme().defaults(), **params)
    branch = int(settings.get("branch", -1))
    detail = fwm.FWM_FIDELITY[settings.get("resolution", fwm.FIDELITY_FAST)]["phase_detail"]
    residual = settings["line_strength"]
    for key in ("mode_overlap_penalty", "polarization_penalty", "zeeman_participation_penalty"):
        residual *= settings.get(key, 1.)
    return dict(D_GHz=float(settings["opd"]), temperature_K=float(settings["temp_c"])+273.15,
        input_pump_power_W=float(settings["pump_mw"])*1e-3,
        input_seed_power_W=float(settings["probe_uw"])*1e-6,
        w_pump_m=float(settings["pump_waist_um"])*1e-6, w_probe_m=float(settings["probe_waist_um"])*1e-6,
        cell_length_m=float(settings["cell_mm"])*1e-3, pump_probe_angle_deg=float(settings["seeded_angle_deg"]),
        branch=branch, raman_center_minus_GHz=fwm.branch_center_GHz(settings["opd"], -1),
        raman_center_plus_GHz=fwm.branch_center_GHz(settings["opd"], 1),
        combined_line_strength_residual=float(residual), phase_detail=detail,
        transit_reset_rate_rad_s=2*np.pi*settings.get("transit_rate_khz", 100.)*1e3)


def atomic_inputs(raw, velocity_step, velocity_cutoff):
    original = raw.get("stark_recompute_context", {})
    return dict(D_GHz=float(raw["D_GHz"]), T=float(raw["temperature_K"]),
        pump_power_W=float(raw["input_pump_power_W"]), pump_waist_m=float(raw["w_pump_m"]),
        velocity_step=float(velocity_step), velocity_cutoff=float(velocity_cutoff),
        transit_rate=float(raw.get("transit_reset_rate_rad_s", original.get("transit_rate", constants.GAMMA_GG))))


@lru_cache(maxsize=4)
def _prepared(encoded_inputs):
    return atom.prepare_response(**json.loads(encoded_inputs))


@lru_cache(maxsize=8)
def _atomic_response(encoded_inputs, delta_tuple, branch, method):
    prepared = _prepared(encoded_inputs)
    response = atom.evaluate_prepared(prepared, np.asarray(delta_tuple)*2*np.pi*1e6,
                                      branch=branch, method=method)
    return (response.positive_frequency_chi_matrix[0], response.mean_pump_state,
            response.diagnostics, response.provenance)


def _populations(state):
    populations = np.real(np.diag(state))
    rows, alignment = [], {}
    offset = 0
    for label, F in (("ground_F2", 2), ("ground_F3", 3), ("excited_F2", 2), ("excited_F3", 3)):
        values = populations[offset:offset+2*F+1]
        for m, value in zip(range(-F, F+1), values):
            rows.append(dict(manifold=label, F=F, m=m, population=float(value)))
        if label.startswith("ground"):
            weights = (3*np.arange(-F, F+1)**2-F*(F+1))/(F*(2*F-1))
            alignment[label] = float(weights@values)
        offset += 2*F+1
    return dict(ground_F2=float(np.sum(populations[:5])), ground_F3=float(np.sum(populations[5:12])),
        excited_total=float(np.sum(populations[12:])), trace=float(np.sum(populations)),
        sublevels=rows, ground_alignment=alignment)


def pump_phase_mismatch(raw, mean_pump_state):
    """Physical pump-index gauge term from the SAME steady state as weak chi.

    A pump phase phi(z) multiplies the parametric vertices by exp(+-2i*phi).
    Removing it adds kp*Re(chi_p) to vacuum mismatch, without altering weak
    diagonal susceptibility. The multiplier is fixed by this gauge identity.
    """
    from .schemes import fwm
    omega = constants.rabi_freq(raw["input_pump_power_W"], raw["w_pump_m"])
    original = raw.get("stark_recompute_context", {})
    residual = raw.get("combined_line_strength_residual")
    if residual is None:
        residual = original.get("line_strength", fwm.SEEDED_REFERENCE_RESIDUAL)
        for key in ("mode_overlap_penalty", "polarization_penalty", "zeeman_participation_penalty"):
            residual *= original.get(key, raw.get(key, 1.))
    model = atom.build_model(raw["temperature_K"],
        transit_rate=atomic_inputs(raw, 2., 6.)["transit_rate"])
    reduced = (np.trace(model.pump_coupling@np.asarray(mean_pump_state))/omega if omega > 0. else 0j)
    physical = complex(observables.chi_phys(reduced, hyperfine.number_density(raw["temperature_K"]),
        dipole=constants.DIPOLE_D1, line_strength=float(residual)/3.))
    pump_k, _, _ = fwm.seeded_option_a_wavenumbers(raw["D_GHz"], np.array([raw["D_GHz"]]))
    correction = float(pump_k*np.real(physical))
    return correction, dict(pump_chi_real=float(physical.real), pump_chi_imag=float(physical.imag),
        pump_index_added_mismatch_per_m=correction, fixed_phase_coefficient=1.,
        formula="delta_k_effective=delta_k_vacuum+kp*Re(chi_pump); chi_pump from same steady state",
        scope="Uniform pump real refractive phase; pump attenuation and spatial state evolution not re-solved")


def build_response(raw, delta_mhz, *, velocity_step=2., velocity_cutoff=6., method="poles"):
    """Return an uncalibrated full-Zeeman propagation snapshot at exact deltas.

    Accepts native raw or context_from_params(). No native atom is solved here.
    Returned chi is in seconds, positive-frequency [[ss,sc],[cs,cc]].
    """
    from .schemes import fwm
    delta = np.atleast_1d(np.asarray(delta_mhz, float))
    if delta.ndim != 1 or not delta.size or not np.isfinite(delta).all():
        raise ValueError("finite nonempty one-dimensional detunings required")
    inputs = atomic_inputs(raw, velocity_step, velocity_cutoff)
    encoded = json.dumps(inputs, sort_keys=True, allow_nan=False)
    branch = int(raw.get("branch", -1))
    chi, state, diagnostics, provenance = _atomic_response(encoded, tuple(delta), branch, method)
    original = raw.get("stark_recompute_context", {})
    residual = raw.get("combined_line_strength_residual")
    if residual is None:
        residual = original.get("line_strength", fwm.SEEDED_REFERENCE_RESIDUAL)
        for key in ("mode_overlap_penalty", "polarization_penalty", "zeeman_participation_penalty"):
            residual *= original.get(key, raw.get(key, 1.))
    residual = float(residual)
    center = raw["raman_center_minus_GHz"] if branch == -1 else raw["raman_center_plus_GHz"]
    axis = float(center)+delta*1e-3
    phase = raw.get("phase_detail", original.get("phase_detail", fwm.PHASE_ULTRA))
    if phase == fwm.PHASE_LEGACY:
        kp = kc = np.full(len(delta), constants.K_VEC)
        mismatch = np.zeros(len(delta))
    else:
        _, kp, kc = fwm.seeded_option_a_wavenumbers(raw["D_GHz"], axis)
        mismatch = fwm.seeded_phase_mismatch_z(raw["D_GHz"], axis, angle_deg=raw["pump_probe_angle_deg"])
    vacuum_mismatch = np.array(mismatch, copy=True)
    pump_index, pump_phase = pump_phase_mismatch(raw, state)
    mismatch = mismatch+pump_index
    snapshot = raw.get("propagation_snapshot")
    if snapshot is not None:
        segment_profile = np.asarray(snapshot["segment_profile"], float).copy()
        dynamic = bool(snapshot["dynamic_depletion"])
        area_ratio = float(snapshot["conjugate_power_ratio"])
    else:
        nseg = fwm.ULTRA_PROPAGATION_SEGMENTS if phase == fwm.PHASE_ULTRA else 16 if phase == fwm.PHASE_FINE else 1
        segment_profile = (fwm._gaussian_overlap_profile(nseg, raw["cell_length_m"], raw["w_pump_m"],
            raw["w_probe_m"], raw["pump_probe_angle_deg"]) if phase == fwm.PHASE_ULTRA else np.ones(nseg))
        dynamic = phase == fwm.PHASE_ULTRA
        area_ratio = float(raw.get("w_conjugate_m", raw["w_probe_m"])/raw["w_probe_m"])**2
    matrix = observables._gain_matrix_from_chi(chi[:, 0, 0], chi[:, 0, 1], chi[:, 1, 0], chi[:, 1, 1],
        kp, kc, hyperfine.number_density(raw["temperature_K"]), constants.DIPOLE_D1, residual/3., delta_k_z=mismatch)
    gs, gc, clipped = propagation.propagate(matrix, raw["cell_length_m"], segment_profile,
        raw["input_pump_power_W"], raw["input_seed_power_W"], area_ratio, dynamic)
    return dict(**{key: raw[key] for key in PHYSICAL_KEYS}, probe_axis_GHz=axis, delta_k_z=mismatch,
        vacuum_delta_k_z=vacuum_mismatch,
        G_s=gs, G_c=gc, propagation_snapshot=dict(version="fwm-zeeman-response-v1", drift_per_m=matrix,
            chi=np.array(chi, copy=True), segment_profile=segment_profile, dynamic_depletion=dynamic,
            conjugate_power_ratio=area_ratio, source="resolved24-state pump and infinitesimal polarized atomic response"),
        zeeman_response_provenance=dict(provenance, practical_residual=residual, pump_phase=pump_phase,
            effective_propagation_line_strength=residual/3., external_sublevel_population_factor=1.),
        zeeman_pump_populations=_populations(state), zeeman_mean_pump_state=np.array(state, copy=True),
        corrected_atomic_response_diagnostics=copy.deepcopy(diagnostics), stability_clipped=clipped,
        atomic_effective_inputs=inputs)


def evaluate(raw, delta_mhz, *, enabled=True):
    delta = np.asarray(delta_mhz, float)
    if not np.isfinite(delta).all():
        raise ValueError("finite detunings required")
    record = profile()
    cross, matrix, response = 1., None, {}
    clipped = np.zeros(delta.size, bool)
    if enabled:
        response = build_response(raw, delta.ravel(), velocity_step=record.get("velocity_step_m_s", 2.),
            velocity_cutoff=record.get("velocity_cutoff_sigma", 6.), method="poles")
        snapshot = response["propagation_snapshot"]
        beta = np.asarray(record["coefficients"], float)
        cross = float(np.exp(np.clip(features(raw["temperature_K"], raw["input_pump_power_W"], raw["w_pump_m"])[:2]@beta,
            -propagation.LOG_LIMIT, propagation.LOG_LIMIT)))
        if float(raw["input_pump_power_W"]) == 0.:
            cross = 1.
        matrix = propagation.correct_drift(snapshot["drift_per_m"], response["delta_k_z"], cross, 1.)
        gs, gc, clipped = propagation.propagate(matrix, raw["cell_length_m"], snapshot["segment_profile"],
            raw["input_pump_power_W"], raw["input_seed_power_W"], snapshot["conjugate_power_ratio"], snapshot["dynamic_depletion"])
        gs, gc = gs.reshape(delta.shape), gc.reshape(delta.shape)
    else:
        gs, outside = _sample_model(_axis(raw), raw["G_s"], delta, arm="probe")
        gc, _ = _sample_model(_axis(raw), raw["G_c"], delta, arm="conjugate")
    within, changed = _domain(raw, delta, record)
    status = np.where(within, "ZEEMAN_PUMP_CHI_CALIBRATION", "ZEEMAN_PUMP_CHI_CALIBRATION; CONDITION_EXTRAPOLATION")
    if not enabled:
        status = np.full(delta.shape, "DISABLED", object)
    result = dict(applied=bool(enabled), G_s=gs, G_c=gc, internal_delta_mhz=delta.copy(),
        model_id=MODEL_ID, display_name="Multi-paper Zeeman pump + chi", profile=record,
        gain_quantity=record["gain_quantity"], validation_status=record["validation_status"],
        validation_label="Repeated grouped prediction passed" if record.get("predictive_gate_passed", False)
            else "Experimental · grouped prediction not established",
        within_training_domain=bool(np.all(within)), point_within_training_domain=within,
        within_measured_support=np.zeros(delta.shape, bool), context_changed=bool(changed), changed_context_inputs=changed,
        point_status=status, frequency_support_mhz=None, outside_atomic_scan=dict(probe=np.zeros(delta.shape, bool) if enabled else outside,
            conjugate=np.zeros(delta.shape, bool) if enabled else outside),
        source_delta_mhz=dict(probe=delta.copy(), conjugate=delta.copy()), offdiagonal_scale=cross,
        diagonal_susceptibility_scale=1., corrected_drift_per_m=matrix, stability_clipped=clipped.reshape(delta.shape),
        status="ZEEMAN_PUMP_CHI_EMPIRICAL_CORRECTION" if enabled else "DISABLED")
    for key in ("zeeman_response_provenance", "zeeman_pump_populations", "corrected_atomic_response_diagnostics"):
        result[key] = copy.deepcopy(response.get(key, {}))
    if enabled:
        value = response["propagation_snapshot"]["chi"].copy()
        value[:, 0, 1] *= cross; value[:, 1, 0] *= cross
        result["corrected_susceptibility"] = value
    return result


def calibrate_spectrum(raw, *, enabled=False, selected_delta_mhz=None):
    result = evaluate(raw, _axis(raw), enabled=enabled)
    if enabled:
        response = build_response(raw, _axis(raw), velocity_step=result["profile"].get("velocity_step_m_s", 2.),
            velocity_cutoff=result["profile"].get("velocity_cutoff_sigma", 6.), method="poles")
        transfer, clipped = propagation.undepleted_transfer(result["corrected_drift_per_m"], raw["cell_length_m"],
            response["propagation_snapshot"]["segment_profile"])
        result.update(corrected_field_transfer_small_signal=transfer, transfer_stability_clipped=clipped,
            corrected_transfer_scope="Resolved atomic field response; not a new canonical/noise certification")
    else:
        result.update(G_s=np.array(raw["G_s"], copy=True), G_c=np.array(raw["G_c"], copy=True))
    if selected_delta_mhz is not None:
        point = evaluate(raw, float(selected_delta_mhz), enabled=enabled)
        result["selected_point"] = dict(G_s=float(point["G_s"]), G_c=float(point["G_c"]),
            internal_delta_mhz=float(selected_delta_mhz), available=True, applied=bool(enabled),
            status=str(point["point_status"]), within_training_domain=point["within_training_domain"],
            within_measured_support=False, context_changed=point["context_changed"],
            offdiagonal_scale=point["offdiagonal_scale"], diagonal_susceptibility_scale=1.)
    return result
