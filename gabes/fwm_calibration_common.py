"""Shared coordinates, gain sampling and profile loading for FWM calibration."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import hyperfine


FEATURE_LIMIT = 3.0


def cached_profile_loader(path: Callable[[], Path]):
    """Make a model's one-entry cache. Callers copy before exposing the profile.

    Resolve the path on cache miss so clearing one model's cache reloads its
    current profile path without invalidating another model's cached record.
    """
    @lru_cache(maxsize=1)
    def load():
        return json.loads(path().read_text(encoding="utf-8"))

    return load


def calibration_axis(raw):
    branch = raw.get("branch", -1)
    center_key = "raman_center_plus_GHz" if branch == 1 else "raman_center_minus_GHz"
    center = raw.get(center_key, raw.get("raman_center_minus_GHz"))
    if center is None:
        raise ValueError("calibration requires the Raman scan center")
    axis = (np.asarray(raw["probe_axis_GHz"], dtype=float)-float(center))*1e3
    if axis.ndim != 1 or axis.size < 2 or np.any(~np.isfinite(axis)):
        raise ValueError("calibration requires at least two finite scan points")
    return axis


def sample_model_gain(axis, gain, positions, *, arm):
    """Positive interpolation with a finite explicit continuation past the scan.

    Log interpolation avoids negative cubic overshoot. Beyond the computed scan,
    a positive exponential continuation approaches the passive asymptote over
    one fifth of the computed span. That scope is reported separately from the
    much narrower experimental calibration interval.
    """
    gain = np.asarray(gain, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if gain.shape != axis.shape or np.any(~np.isfinite(gain)) or np.any(gain < 0):
        raise ValueError("calibration gain arrays must be finite, nonnegative and match scan points")
    order = np.argsort(axis)
    x, g = axis[order], gain[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("calibration scan points must be distinct")
    outside = (positions < x[0]) | (positions > x[-1])
    if not np.any(g):
        return np.zeros_like(positions), outside
    bounded = np.clip(positions, x[0], x[-1])
    sampled = np.exp(PchipInterpolator(x, np.log(np.maximum(g, 1e-30)), extrapolate=False)(bounded))
    asymptote = 1.0 if arm == "probe" else 0.0
    length = max(float((x[-1]-x[0])/5.0), 1.0)
    distance = np.abs(positions-bounded)
    sampled = np.where(outside, asymptote+(sampled-asymptote)*np.exp(-distance/length), sampled)
    return sampled, outside


def literature_features(temperature_K, pump_power_W, pump_waist_m):
    """Fixed dimensionless physics coordinates, never test-set standardization."""
    t, power, waist = map(float, (temperature_K, pump_power_W, pump_waist_m))
    if not np.isfinite([t, power, waist]).all() or t <= 0 or power < 0 or waist <= 0:
        raise ValueError("literature gain context must have finite positive T/waist and nonnegative pump")
    density_ratio = hyperfine.number_density(t) / hyperfine.number_density(386.15)
    intensity_ratio = (max(power, 1e-30) / waist**2) / (0.3 / (530e-6)**2)
    return np.array([1.0, np.clip(np.log(density_ratio), -FEATURE_LIMIT, FEATURE_LIMIT),
                     np.clip(np.log(intensity_ratio), -FEATURE_LIMIT, FEATURE_LIMIT)])


def literature_domain(raw, delta, record):
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
