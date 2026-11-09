"""Coupled empirical gain correction from grouped literature experiments.

This readout maps two gains at the SAME frequency. It does not change the atomic
solver or identify a microscopic susceptibility/noise covariance.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import hyperfine
from .fwm_experiment_gain import _axis, _sample_model

MODEL_ID = "fwm_literature_coupled_generation_v1"
PROFILE_PATH = Path(__file__).with_name("data") / "fwm_literature_gain_v1.json"
FEATURE_LIMIT = 3.0
LOG_SCALE_LIMIT = float(np.log(20.0))
PARAMETER_KEYS = {
    "opd": "D_GHz", "temp_c": "temperature_K", "pump_mw": "input_pump_power_W",
    "probe_uw": "input_seed_power_W", "cell_mm": "cell_length_m",
    "pump_waist_um": "w_pump_m", "probe_waist_um": "w_probe_m",
    "seeded_angle_deg": "pump_probe_angle_deg",
}


@lru_cache(maxsize=1)
def _frozen_profile():
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def profile():
    """Return independent metadata, including the failed/passed outer gate."""
    result = copy.deepcopy(_frozen_profile())
    result.setdefault("model_id", MODEL_ID)
    result.setdefault("method", "common latent-generation scale; original arm asymmetry retained")
    result.setdefault("formula", "r=asinh(sqrt(Gc)); R=Gs/(1+Gc); Gc'=sinh(a*r)^2; Gs'=R*(1+Gc')")
    result.setdefault("gain_quantity", "effective optical output/input gain; literature reference planes remain conditional")
    result["noise_covariance_calibrated"] = False
    result["susceptibility_identified"] = False
    return result


def features(temperature_K, pump_power_W, pump_waist_m):
    """Fixed dimensionless physics coordinates, never test-set standardization."""
    t, power, waist = map(float, (temperature_K, pump_power_W, pump_waist_m))
    if not np.isfinite([t, power, waist]).all() or t <= 0 or power < 0 or waist <= 0:
        raise ValueError("literature gain context must have finite positive T/waist and nonnegative pump")
    density_ratio = hyperfine.number_density(t) / hyperfine.number_density(386.15)
    intensity_ratio = (max(power, 1e-30) / waist**2) / (0.3 / (530e-6)**2)
    return np.array([1.0, np.clip(np.log(density_ratio), -FEATURE_LIMIT, FEATURE_LIMIT),
                     np.clip(np.log(intensity_ratio), -FEATURE_LIMIT, FEATURE_LIMIT)])


def coupled_gains(probe_gain, conjugate_gain, scale, *, pump_power_W=None, seed_power_W=None):
    """Positive common generation scale with optional total pump-energy bound.

    At scale=1 return exact input gains. Zero conjugate gain stays zero. The
    optional bound limits max(Gs-1, 0)+Gc when the retained raw arm ratio admits
    a feasible pair. An already infeasible raw ratio cannot satisfy both that
    bound and ratio preservation; it is reported separately by evaluate().
    """
    gs, gc, a = np.broadcast_arrays(np.asarray(probe_gain, dtype=float),
                                   np.asarray(conjugate_gain, dtype=float), np.asarray(scale, dtype=float))
    if not np.isfinite(gs).all() or not np.isfinite(gc).all() or np.any(gs < 0) or np.any(gc < 0):
        raise ValueError("gain arrays must be finite and nonnegative")
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("generation scale must be finite and positive")
    ratio = gs / (1.0 + gc)
    with np.errstate(over="ignore"):
        radius = np.minimum(a * np.arcsinh(np.sqrt(gc)), 350.0)
    # sinh(x)^2 remains finite for x<=350. Preserve exact zeros explicitly.
    proposed_c = np.sinh(radius)**2
    capped = np.zeros(gs.shape, dtype=bool)
    if pump_power_W is not None and seed_power_W is not None:
        power, seed = np.asarray(pump_power_W, dtype=float), np.asarray(seed_power_W, dtype=float)
        if not np.isfinite(power).all() or not np.isfinite(seed).all() or np.any(power < 0) or np.any(seed <= 0):
            raise ValueError("pump/seed power must be finite, nonnegative pump and positive seed")
        with np.errstate(over="ignore", invalid="ignore"):
            budget = power / seed
            # Below probe unity the conjugate alone consumes the budget;
            # above unity both generated arms consume it.
            cap_c = np.maximum(np.minimum(budget, (budget + 1.0 - ratio) / (ratio + 1.0)), 0.0)
        capped = proposed_c > cap_c
        proposed_c = np.minimum(proposed_c, cap_c)
    # Keep finite inputs finite even without a physical power bound. A small
    # rounding margin prevents the final product exceeding float64 max.
    with np.errstate(over="ignore", divide="ignore"):
        representable_c = np.maximum(np.finfo(float).max / ratio / (1.0 + 8*np.finfo(float).eps) - 1.0, 0.0)
    proposed_c = np.minimum(proposed_c, representable_c)
    proposed_s = ratio * (1.0 + proposed_c)
    identity = a == 1.0
    out_s, out_c = np.where(identity, gs, proposed_s), np.where(identity, gc, proposed_c)
    capped = np.where(identity, False, capped)
    return out_s, out_c, capped


def _context(raw):
    needed = ("temperature_K", "input_pump_power_W", "w_pump_m")
    if any(key not in raw for key in needed):
        raise ValueError("literature gain calibration requires temperature, pump power and waist metadata")
    return features(*(raw[key] for key in needed))


def _domain(raw, delta, record):
    outside = []
    for key, limits in record.get("raw_training_bounds", {}).items():
        value = raw.get(key)
        if value is None or not np.isfinite(float(value)) or not limits[0]-1e-12 <= float(value) <= limits[1]+1e-12:
            outside.append(key)
    bounds = record.get("training_delta_bounds_mhz", [-16.0, 26.0])
    within = (delta >= bounds[0]-1e-9) & (delta <= bounds[1]+1e-9) & (not outside)
    if int(raw.get("branch", -1)) != -1:
        within = np.zeros(np.shape(delta), dtype=bool)
        outside.append("branch")
    return within, outside


def evaluate(raw, delta_mhz, *, enabled=True):
    delta = np.asarray(delta_mhz, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("calibration detunings must be finite")
    axis = _axis(raw)
    gs, out_s = _sample_model(axis, raw["G_s"], delta, arm="probe")
    gc, out_c = _sample_model(axis, raw["G_c"], delta, arm="conjugate")
    record = profile()
    scale = 1.0
    capped = np.zeros(delta.shape, dtype=bool)
    if enabled:
        f = _context(raw)
        beta = np.asarray(record["coefficients"], dtype=float)
        scale = float(np.exp(np.clip(np.dot(beta, f), -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT)))
        if float(raw["input_pump_power_W"]) == 0.0:
            scale = 1.0
        gs, gc, capped = coupled_gains(gs, gc, scale,
            pump_power_W=raw.get("input_pump_power_W"), seed_power_W=raw.get("input_seed_power_W"))
    within, changed = _domain(raw, delta, record)
    statuses = np.where(within, "LITERATURE_CALIBRATION", "LITERATURE_CALIBRATION; CONDITION_EXTRAPOLATION")
    if not enabled:
        statuses = np.full(delta.shape, "DISABLED", dtype=object)
    passed = record.get("predictive_gate_passed", False)
    return {
        "applied": bool(enabled), "G_s": gs, "G_c": gc, "internal_delta_mhz": delta.copy(),
        "model_id": MODEL_ID, "display_name": "Multi-paper literature",
        "profile": record, "gain_quantity": record["gain_quantity"],
        "validation_status": record.get("validation_status", "NOT_EVALUATED"),
        "validation_label": ("Conditional grouped prediction passed" if passed else "Experimental · grouped prediction not established"),
        "within_training_domain": bool(np.all(within)), "point_within_training_domain": within,
        "within_measured_support": np.zeros(delta.shape, dtype=bool),
        "context_changed": bool(changed), "changed_context_inputs": changed,
        "point_status": statuses, "frequency_support_mhz": None,
        "outside_atomic_scan": {"probe": out_s, "conjugate": out_c},
        "source_delta_mhz": {"probe": delta.copy(), "conjugate": delta.copy()},
        "effective_generation_scale": scale, "pump_budget_capped": capped,
        "pump_budget_feasible": _budget_feasible(raw, gs, gc),
        "status": "LITERATURE_EMPIRICAL_CALIBRATION" if enabled else "DISABLED",
    }


def _budget_feasible(raw, gs, gc):
    """Whether the retained arm ratio admits any pair within the pump budget."""
    power, seed = raw.get("input_pump_power_W"), raw.get("input_seed_power_W")
    if power is None or seed is None or float(seed) <= 0:
        return None
    with np.errstate(over="ignore"):
        return np.asarray(gs) / (1.0 + np.asarray(gc)) <= 1.0 + float(power)/float(seed)


def calibrate_spectrum(raw, *, enabled=False, selected_delta_mhz=None):
    axis = _axis(raw)
    result = evaluate(raw, axis, enabled=enabled)
    if not enabled:
        result["G_s"] = np.array(raw["G_s"], copy=True)
        result["G_c"] = np.array(raw["G_c"], copy=True)
    if selected_delta_mhz is not None:
        selected = evaluate(raw, float(selected_delta_mhz), enabled=enabled)
        result["selected_point"] = {
            "G_s": float(selected["G_s"]), "G_c": float(selected["G_c"]),
            "internal_delta_mhz": float(selected_delta_mhz), "available": True,
            "applied": bool(enabled), "status": str(selected["point_status"]),
            "within_training_domain": selected["within_training_domain"],
            "within_measured_support": False, "context_changed": selected["context_changed"],
            "generation_scale": selected["effective_generation_scale"],
            "pump_budget_capped": bool(selected["pump_budget_capped"]),
            "pump_budget_feasible": (None if selected["pump_budget_feasible"] is None
                                     else bool(selected["pump_budget_feasible"])),
        }
    return result
