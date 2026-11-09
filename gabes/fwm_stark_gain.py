"""Opt-in pump self-energy, atomic response and susceptibility calibration.

The original atomic spectrum remains the reference. A separate response is
computed with the selected ground Hamiltonian counterterm before applying the
paired complex susceptibility scales and the existing optical propagation.
No canonical map or noise covariance is inferred from this gain calibration.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import fwm_chi_gain as chi
from .fwm_experiment_gain import _axis, _sample_model
from .fwm_literature_gain import _domain, features

MODEL_ID = "fwm_literature_stark_response_v1"
PROFILE_PATH = Path(__file__).with_name("data") / "fwm_stark_gain_v1.json"


@lru_cache(maxsize=1)
def _record():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def profile():
    record = copy.deepcopy(_record())
    record.update(model_id=MODEL_ID,
        method="Pump Hamiltonian self-energy, recomputed susceptibility and segmented propagation",
        formula="H'=H+(1-s)*S0*|g2><g2|; chi_offdiag'=kappa*chi_offdiag; chi_diag'=d*chi_diag",
        noise_covariance_calibrated=False, susceptibility_identified=False)
    return record


@lru_cache(maxsize=8)
def _solve_response(encoded_inputs, fraction):
    # Lazy import avoids a scheme/dispatcher cycle. The key contains every
    # original compute_spectrum argument, including fidelity and scan bounds.
    from .schemes import fwm
    return fwm.compute_spectrum(**json.loads(encoded_inputs), pump_stark_fraction=fraction)


def _json_inputs(value):
    if isinstance(value, np.ndarray):
        return _json_inputs(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_inputs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_inputs(item) for item in value]
    return value


def corrected_response(raw, fraction):
    """Return a separate consistently solved atomic response; never mutate raw."""
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError("pump Stark fraction must be between zero and one")
    if float(raw["input_pump_power_W"]) == 0.:
        return raw
    provenance = raw.get("pump_stark_response", {})
    if provenance.get("fraction", 1.) == fraction:
        return raw
    if fraction == 1. and not provenance.get("applied", False):
        return raw
    inputs = raw.get("stark_recompute_context")
    if inputs is None:
        raise ValueError("pump/Stark calibration requires a fresh atomic spectrum with recompute context")
    inputs = dict(inputs)
    inputs.pop("pump_stark_fraction", None)
    encoded = json.dumps(_json_inputs(inputs), sort_keys=True, allow_nan=False)
    return _solve_response(encoded, fraction)


def evaluate(raw, delta_mhz, *, enabled=True):
    delta = np.asarray(delta_mhz, float)
    if not np.isfinite(delta).all():
        raise ValueError("finite detunings required")
    record = profile()
    fraction = float(record["pump_stark_fraction"]) if enabled else 1.
    response = corrected_response(raw, fraction) if enabled else raw
    cross, diagonal, matrix = 1., 1., None
    clipped = np.zeros(delta.size, bool)
    if enabled:
        matrix, mismatch, outside = chi.sample_snapshot(response, delta.ravel())
        beta = np.asarray(record["coefficients"], float)
        x = features(raw["temperature_K"], raw["input_pump_power_W"], raw["w_pump_m"])
        cross = float(np.exp(np.clip(x@beta[:3], -chi.LOG_LIMIT, chi.LOG_LIMIT)))
        diagonal = float(np.exp(np.clip(beta[3], -chi.LOG_LIMIT, chi.LOG_LIMIT)))
        if float(raw["input_pump_power_W"]) == 0.:
            cross = diagonal = 1.
        matrix = chi.correct_drift(matrix, mismatch, cross, diagonal)
        snapshot = response["propagation_snapshot"]
        gs, gc, clipped = chi.propagate(matrix, raw["cell_length_m"], snapshot["segment_profile"],
            raw["input_pump_power_W"], raw["input_seed_power_W"],
            snapshot["conjugate_power_ratio"], snapshot["dynamic_depletion"])
        gs, gc = gs.reshape(delta.shape), gc.reshape(delta.shape)
    else:
        gs, outside = _sample_model(_axis(raw), raw["G_s"], delta, arm="probe")
        gc, _ = _sample_model(_axis(raw), raw["G_c"], delta, arm="conjugate")
    within, changed = _domain(raw, delta, record)
    status = np.where(within, "PUMP_STARK_CHI_CALIBRATION",
        "PUMP_STARK_CHI_CALIBRATION; CONDITION_EXTRAPOLATION")
    if not enabled:
        status = np.full(delta.shape, "DISABLED", object)
    passed = record.get("predictive_gate_passed", False)
    return dict(applied=bool(enabled), G_s=gs, G_c=gc, internal_delta_mhz=delta.copy(),
        model_id=MODEL_ID, display_name="Multi-paper pump/Stark + chi", profile=record,
        gain_quantity=record["gain_quantity"], validation_status=record["validation_status"],
        validation_label="Repeated grouped prediction passed" if passed else "Experimental · grouped prediction not established",
        within_training_domain=bool(np.all(within)), point_within_training_domain=within,
        within_measured_support=np.zeros(delta.shape, bool), context_changed=bool(changed),
        changed_context_inputs=changed, point_status=status, frequency_support_mhz=None,
        outside_atomic_scan=dict(probe=outside, conjugate=outside),
        source_delta_mhz=dict(probe=delta.copy(), conjugate=delta.copy()),
        offdiagonal_scale=cross, diagonal_susceptibility_scale=diagonal,
        pump_stark_fraction=fraction,
        pump_stark_response=copy.deepcopy(response.get("pump_stark_response", {})),
        corrected_atomic_floquet_status=response.get("floquet_convergence", {}).get("status", "UNKNOWN"),
        corrected_drift_per_m=matrix, stability_clipped=clipped.reshape(delta.shape),
        status="PUMP_STARK_CHI_EMPIRICAL_CORRECTION" if enabled else "DISABLED")


def calibrate_spectrum(raw, *, enabled=False, selected_delta_mhz=None):
    axis = _axis(raw)
    result = evaluate(raw, axis, enabled=enabled)
    if enabled:
        response = corrected_response(raw, result["pump_stark_fraction"])
        snapshot = response["propagation_snapshot"]
        transfer, clipped = chi.undepleted_transfer(result["corrected_drift_per_m"], raw["cell_length_m"], snapshot["segment_profile"])
        result.update(corrected_field_transfer_small_signal=transfer, transfer_stability_clipped=clipped,
            corrected_transfer_scope="Field basis from recomputed atomic response; not a new canonical/noise certification")
        if "chi" in snapshot:
            value = np.asarray(snapshot["chi"], complex)
            source_axis = _axis(response)
            if np.array_equal(axis, source_axis):
                value = value.copy()
            else:
                from scipy.interpolate import PchipInterpolator
                value = (PchipInterpolator(source_axis, value.real, axis=0)(axis)
                         + 1j*PchipInterpolator(source_axis, value.imag, axis=0)(axis))
            for j in (0, 1):
                value[:, j, j] *= result["diagonal_susceptibility_scale"]
                value[:, j, 1-j] *= result["offdiagonal_scale"]
            result["corrected_susceptibility"] = value
    else:
        result.update(G_s=np.array(raw["G_s"], copy=True), G_c=np.array(raw["G_c"], copy=True))
    if selected_delta_mhz is not None:
        point = evaluate(raw, float(selected_delta_mhz), enabled=enabled)
        result["selected_point"] = dict(G_s=float(point["G_s"]), G_c=float(point["G_c"]),
            internal_delta_mhz=float(selected_delta_mhz), available=True, applied=bool(enabled),
            status=str(point["point_status"]), within_training_domain=point["within_training_domain"],
            within_measured_support=False, context_changed=point["context_changed"],
            offdiagonal_scale=point["offdiagonal_scale"],
            diagonal_susceptibility_scale=point["diagonal_susceptibility_scale"],
            pump_stark_fraction=point["pump_stark_fraction"])
    return result
