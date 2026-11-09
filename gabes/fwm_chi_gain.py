"""Empirical complex-susceptibility correction followed by optical propagation.

Raw atomic spectra/certification are never overwritten. Both complex cross
couplings share one positive scale; both FULL complex diagonal susceptibilities
share another. Vacuum/geometric phase mismatch is retained exactly.
"""
from __future__ import annotations

import cmath
import copy
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import kernels
from .core import matrix_exp_2x2
from .fwm_experiment_gain import _axis, _sample_model
from .fwm_literature_gain import features, _domain

MODEL_ID = "fwm_literature_chi_propagation_v1"
PROFILE_PATH = Path(__file__).with_name("data") / "fwm_chi_gain_v1.json"
LOG_LIMIT = float(np.log(20.0))


@lru_cache(maxsize=1)
def _record():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def profile():
    result = copy.deepcopy(_record())
    result.update(model_id=MODEL_ID, method="Complex susceptibility correction and segmented repropagation",
                  formula="chi_offdiag'=kappa*chi_offdiag; chi_diag'=d*chi_diag; vacuum mismatch unchanged",
                  gain_quantity="effective optical output/input gain; literature reference planes remain conditional",
                  noise_covariance_calibrated=False, susceptibility_identified=False)
    return result


def correct_drift(drift, mismatch, cross_scale, diagonal_scale):
    """Scale atomic parts of Maxwell drift, keeping known vacuum terms exact."""
    matrix = np.array(drift, dtype=complex, copy=True)
    dk, cross, diagonal = np.broadcast_arrays(np.asarray(mismatch, float),
        np.asarray(cross_scale, float), np.asarray(diagonal_scale, float))
    if (not np.isfinite(matrix).all() or not np.isfinite([dk, cross, diagonal]).all()
            or np.any(cross <= 0) or np.any(diagonal <= 0)):
        raise ValueError("finite drift and positive susceptibility scales required")
    matrix[..., 0, 1] *= cross
    matrix[..., 1, 0] *= cross
    matrix[..., 0, 0] = diagonal*(matrix[..., 0, 0]+.5j*dk)-.5j*dk
    matrix[..., 1, 1] = diagonal*(matrix[..., 1, 1]-.5j*dk)+.5j*dk
    return matrix


@kernels.njit(cache=True)
def _propagate_rows(matrix, lengths, profiles, budget, area_ratio, dynamic):
    """Serial batched version of the existing 2x2 depletion recurrence.

    Unit input seed. Normalize amplitudes in log space for finite extreme
    continuation. The local exponential uses the same clamp as core.py.
    """
    n, nseg = matrix.shape[0], profiles.shape[1]
    gs, gc, clipped = np.empty(n), np.empty(n), np.zeros(n, np.bool_)
    for i in range(n):
        a0, a1, log_norm = 1.+0j, 0.+0j, 0.
        remaining = budget[i]
        dz = lengths[i]/nseg
        for seg in range(nseg):
            pump_scale = math.sqrt(min(max(remaining/max(budget[i], 1e-30), 0.), 1.)) if dynamic[i] else 1.
            scale = profiles[i, seg]*pump_scale
            m00, m11 = matrix[i, 0, 0], matrix[i, 1, 1]
            m01, m10 = matrix[i, 0, 1]*scale, matrix[i, 1, 0]*scale
            half = .5*(m00+m11)
            q00, q11 = m00-half, m11-half
            root = cmath.sqrt(-(q00*q11-m01*m10)+0j)
            cL, sL = root*dz, half*dz
            cL = min(max(cL.real, -350.), 350.)+1j*cL.imag
            sL = min(max(sL.real, -350.), 350.)+1j*sL.imag
            sinh = cmath.sinh(cL)/root if abs(root)>1e-30 else dz+0j
            cosine, exp = cmath.cosh(cL), cmath.exp(sL)
            t00, t01 = exp*(cosine+sinh*q00), exp*sinh*m01
            t10, t11 = exp*sinh*m10, exp*(cosine+sinh*q11)
            tnorm = max(abs(t00), abs(t01), abs(t10), abs(t11), 1e-300)
            b0, b1 = (t00/tnorm)*a0+(t01/tnorm)*a1, (t10/tnorm)*a0+(t11/tnorm)*a1
            norm = max(abs(b0), abs(b1), 1e-300)
            a0, a1 = b0/norm, b1/norm
            log_norm += math.log(tnorm)+math.log(norm)
            lp = 2*log_norm+2*math.log(max(abs(a0), 1e-300))
            lc = 2*log_norm+2*math.log(max(abs(a1), 1e-300))
            gp, gcf = math.exp(min(lp, 690.)), math.exp(min(lc, 690.))
            if abs(a0)==0.: gp=0.
            if abs(a1)==0.: gcf=0.
            if lp>690. or lc>690.: clipped[i]=True
            remaining = max(budget[i]-max(gp-1., 0.)-gcf*area_ratio[i], 0.)
        gs[i], gc[i] = gp, gcf*area_ratio[i]
    return gs, gc, clipped


def propagate(drift, length_m, segment_profile, pump_power_W, seed_power_W,
              conjugate_power_ratio=1., dynamic_depletion=True):
    """Re-use the segmented depletion and final smooth pump-budget model."""
    matrix = np.asarray(drift, complex).reshape(-1, 2, 2)
    n = len(matrix)
    lengths, pump, seed, area, dynamic = [np.broadcast_to(np.asarray(value), (n,)).copy()
        for value in (length_m, pump_power_W, seed_power_W, conjugate_power_ratio, dynamic_depletion)]
    profile_array = np.asarray(segment_profile, float)
    if profile_array.ndim == 1:
        profile_array = np.broadcast_to(profile_array, (n, len(profile_array))).copy()
    if (profile_array.ndim != 2 or profile_array.shape[0] != n or profile_array.shape[1] < 1
            or not np.isfinite(matrix).all() or not np.isfinite(profile_array).all()
            or np.any(profile_array < 0) or not np.isfinite([lengths, pump, seed, area]).all()
            or np.any(lengths < 0) or np.any(pump < 0) or np.any(seed < 0) or np.any(area <= 0)):
        raise ValueError("invalid propagation context")
    # The local exponential squares matrix entries before its exponent clamp.
    # Reject unrepresentable inputs rather than returning an unflagged NaN.
    if np.max(np.abs(matrix), initial=0.) > 1e150:
        raise ValueError("drift magnitude exceeds representable 2x2 propagation range")
    budget = pump/np.maximum(seed, 1e-30)
    gs, gc, clipped = _propagate_rows(np.ascontiguousarray(matrix), lengths.astype(float),
        np.ascontiguousarray(profile_array), budget.astype(float), area.astype(float), dynamic.astype(bool))
    cap = np.maximum(.5*budget, 1e-30)
    extra = np.maximum(gs-1., 0.)
    return np.where(gs <= 1., gs, 1.+extra/(1.+extra/cap)), gc/(1.+gc/cap), clipped


def sample_snapshot(raw, delta_mhz):
    """Interpolate COMPLEX atomic drift, then restore known vacuum mismatch."""
    snapshot = raw.get("propagation_snapshot")
    if not snapshot:
        raise ValueError("chi propagation calibration requires a fresh propagation snapshot")
    axis = _axis(raw)
    delta = np.atleast_1d(np.asarray(delta_mhz, float))
    outside = (delta < axis[0]) | (delta > axis[-1])
    query = np.clip(delta, axis[0], axis[-1])
    matrix = np.asarray(snapshot["drift_per_m"], complex)
    mismatch = np.broadcast_to(np.asarray(raw.get("delta_k_z") if raw.get("delta_k_z") is not None else 0., float), axis.shape)
    if len(axis) == 1:
        sampled = np.broadcast_to(matrix[0], (len(delta), 2, 2)).copy()
    else:
        sampled = PchipInterpolator(axis, matrix.real, axis=0)(query)+1j*PchipInterpolator(axis, matrix.imag, axis=0)(query)
    dk = np.interp(query, axis, mismatch)
    if np.any(outside):
        # Smooth finite continuation of the atomic part; the vacuum geometry
        # survives at infinity, where gains approach passive unity/zero.
        distance = np.abs(delta-query)
        fade = np.exp(-np.minimum((distance/50.)**2, 745.))
        sampled[:, 0, 0] += .5j*dk
        sampled[:, 1, 1] -= .5j*dk
        sampled *= fade[:, None, None]
        sampled[:, 0, 0] -= .5j*dk
        sampled[:, 1, 1] += .5j*dk
    return sampled, dk, outside


def undepleted_transfer(drift, length_m, segment_profile):
    """Field-basis transfer with fixed pump/profile, separate from noise audit."""
    matrix = np.asarray(drift, complex)
    profile_array = np.asarray(segment_profile, float)
    result = np.broadcast_to(np.eye(2, dtype=complex), matrix.shape).copy()
    log_scale = np.zeros(len(matrix))
    for overlap in profile_array:
        step = matrix.copy()
        step[:, 0, 1] *= overlap
        step[:, 1, 0] *= overlap
        result = matrix_exp_2x2(step, float(length_m)/len(profile_array)) @ result
        norm = np.maximum(np.max(np.abs(result), axis=(1, 2)), 1e-300)
        result /= norm[:, None, None]
        log_scale += np.log(norm)
    clipped = log_scale > 690.
    result *= np.exp(np.minimum(log_scale, 690.))[:, None, None]
    return result, clipped


def evaluate(raw, delta_mhz, *, enabled=True):
    delta = np.asarray(delta_mhz, float)
    if not np.isfinite(delta).all():
        raise ValueError("finite detunings required")
    axis = _axis(raw)
    record = profile()
    cross, diagonal, clipped, changed_matrix = 1., 1., np.zeros(delta.size, bool), None
    if enabled:
        matrix, dk, outside = sample_snapshot(raw, delta.ravel())
        x = features(raw["temperature_K"], raw["input_pump_power_W"], raw["w_pump_m"])
        beta = np.asarray(record["coefficients"], float)
        cross = float(np.exp(np.clip(beta[:3]@x, -LOG_LIMIT, LOG_LIMIT)))
        diagonal = float(np.exp(np.clip(beta[3], -LOG_LIMIT, LOG_LIMIT)))
        if float(raw["input_pump_power_W"]) == 0.:
            cross = diagonal = 1.
        changed_matrix = correct_drift(matrix, dk, cross, diagonal)
        snap = raw["propagation_snapshot"]
        gs, gc, clipped = propagate(changed_matrix, raw["cell_length_m"], snap["segment_profile"],
            raw["input_pump_power_W"], raw["input_seed_power_W"],
            snap["conjugate_power_ratio"], snap["dynamic_depletion"])
        gs, gc = gs.reshape(delta.shape), gc.reshape(delta.shape)
    else:
        gs, outside = _sample_model(axis, raw["G_s"], delta, arm="probe")
        gc, _ = _sample_model(axis, raw["G_c"], delta, arm="conjugate")
    within, changed = _domain(raw, delta, record)
    statuses = np.where(within, "CHI_PROPAGATION_CALIBRATION", "CHI_PROPAGATION_CALIBRATION; CONDITION_EXTRAPOLATION")
    if not enabled: statuses = np.full(delta.shape, "DISABLED", object)
    passed = record.get("predictive_gate_passed", False)
    return dict(applied=bool(enabled), G_s=gs, G_c=gc, internal_delta_mhz=delta.copy(),
        model_id=MODEL_ID, display_name="Multi-paper chi propagation", profile=record,
        gain_quantity=record["gain_quantity"], validation_status=record["validation_status"],
        validation_label="Repeated grouped prediction passed" if passed else "Experimental · grouped prediction not established",
        within_training_domain=bool(np.all(within)), point_within_training_domain=within,
        within_measured_support=np.zeros(delta.shape, bool), context_changed=bool(changed),
        changed_context_inputs=changed, point_status=statuses, frequency_support_mhz=None,
        outside_atomic_scan=dict(probe=outside, conjugate=outside),
        source_delta_mhz=dict(probe=delta.copy(), conjugate=delta.copy()),
        offdiagonal_scale=cross, diagonal_susceptibility_scale=diagonal,
        corrected_drift_per_m=changed_matrix, stability_clipped=clipped.reshape(delta.shape),
        status="CHI_PROPAGATION_EMPIRICAL_CORRECTION" if enabled else "DISABLED")


def calibrate_spectrum(raw, *, enabled=False, selected_delta_mhz=None):
    result = evaluate(raw, _axis(raw), enabled=enabled)
    if enabled:
        snapshot = raw["propagation_snapshot"]
        transfer, clipped = undepleted_transfer(result["corrected_drift_per_m"], raw["cell_length_m"], snapshot["segment_profile"])
        result.update(corrected_field_transfer_small_signal=transfer, transfer_stability_clipped=clipped,
            corrected_transfer_scope="Field basis, undepleted fixed profile; not a new canonical/noise certification")
        if "chi" in snapshot:
            chi = np.array(snapshot["chi"], complex, copy=True)
            for j in (0, 1):
                chi[:, j, j] *= result["diagonal_susceptibility_scale"]
                chi[:, j, 1-j] *= result["offdiagonal_scale"]
            result["corrected_susceptibility"] = chi
    if not enabled:
        result.update(G_s=np.array(raw["G_s"], copy=True), G_c=np.array(raw["G_c"], copy=True))
    if selected_delta_mhz is not None:
        point = evaluate(raw, float(selected_delta_mhz), enabled=enabled)
        result["selected_point"] = dict(G_s=float(point["G_s"]), G_c=float(point["G_c"]),
            internal_delta_mhz=float(selected_delta_mhz), available=True, applied=bool(enabled),
            status=str(point["point_status"]), within_training_domain=point["within_training_domain"],
            within_measured_support=False, context_changed=point["context_changed"],
            offdiagonal_scale=point["offdiagonal_scale"],
            diagonal_susceptibility_scale=point["diagonal_susceptibility_scale"])
    return result
