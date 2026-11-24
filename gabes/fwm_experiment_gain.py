"""Practical 10/13/2026 FWM gain calibration, separate from atomic/GC solves.

The measured pump-on / pump-off detector ratios calibrate a frequency-warped
model. Frozen positive correction factors act on each CURRENT model spectrum;
they do not replace it with a measured lookup curve. Outside measured support,
only the correction coefficient holds its boundary value: the model continues.
"""
import copy
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import fwm_measured_gain as measured
from .fwm_calibration_common import cached_profile_loader, calibration_axis, sample_model_gain

# Historical evaluators import the previous private names directly.
_axis = calibration_axis
_sample_model = sample_model_gain


LEGACY_MODEL_ID = "fwm_20261013_warp_affine_residual_v1"
# Frozen offline fit, reproduced by analysis/fwm_experiment_20261022/fit_gain_profile.py.
# BEGIN FITTED COEFFICIENTS
_ARMS = {'probe': {'model_peak_delta_mhz': 24.749999999999606,
           'measured_peak_delta_mhz': 2.2324390000003405,
           'source_center_delta_mhz': 22.827113312494454,
           'frequency_scale': 1.2392520825296043,
           'gain_scale': 0.09401476007742224,
           'gain_offset': 0.015291576826316149,
           'affine_max_relative_error': 0.25546760317046724,
           'affine_rms_log_error': 0.12136298700494211,
           'log_total_coefficient_at_nodes': [-2.517957851200495,
                                              -2.503995545291295,
                                              -2.5137673070531177,
                                              -2.4485481032565097,
                                              -2.375518819648362,
                                              -2.3266045808010096,
                                              -2.33077689206703,
                                              -2.2870940788820855,
                                              -2.25240072839961,
                                              -2.2386856298749525,
                                              -2.192754219042939,
                                              -2.2193116033664104,
                                              -2.2897526825171437,
                                              -2.3567679242015243,
                                              -2.395190087235892,
                                              -2.4574839135036313,
                                              -2.4650009808774835,
                                              -2.503097335818218,
                                              -2.181096369153177,
                                              -2.323827115233929,
                                              -1.748098562172115,
                                              -1.5595865045394315,
                                              -1.907233438221969]},
 'conjugate': {'model_peak_delta_mhz': 24.749999999999606,
               'measured_peak_delta_mhz': 2.7324390000003405,
               'source_center_delta_mhz': 22.689633036415053,
               'frequency_scale': 1.294620567486792,
               'gain_scale': 0.09409948262794274,
               'gain_offset': 0.17628769405430758,
               'affine_max_relative_error': 0.2876951260476732,
               'affine_rms_log_error': 0.14779889425949747,
               'log_total_coefficient_at_nodes': [-2.566744988284615,
                                                  -2.5837915556673354,
                                                  -2.5077399275123704,
                                                  -2.4262740624082477,
                                                  -2.334979254310135,
                                                  -2.248954102201915,
                                                  -2.2745780939838216,
                                                  -2.227927558690042,
                                                  -2.198546198566624,
                                                  -2.17171466872008,
                                                  -2.198459747510887,
                                                  -2.2051736208963955,
                                                  -2.2670844254470475,
                                                  -2.3737022816810356,
                                                  -2.4278681759505734,
                                                  -2.4995453484526693,
                                                  -2.504238713367706,
                                                  -2.4762389345054436,
                                                  -1.9858426601340071,
                                                  -1.7040635113291327,
                                                  -1.2348914272121547,
                                                  -0.9499447191181953,
                                                  -1.0393488456724986]}}
# END FITTED COEFFICIENTS

_OBSERVATIONS = measured.observations()
_NODES = measured.internal_delta_mhz_from_eom(_OBSERVATIONS["eom_MHz"])
_NOMINAL = measured.nominal_context()
# Solver controls held fixed in the 10/13 source calculations. Older
# saved arrays predate these metadata fields and used the fixed defaults.
_FIT_CONTROLS = {
    "transit_reset_rate_rad_s": 2.0*np.pi*100e3,
    "line_strength_residual": 0.74,
    "mode_overlap_penalty": 1.0,
    "polarization_penalty": 1.0,
    "zeeman_participation_penalty": 1.0,
}
_GAIN_QUANTITY = "pump-on arm output / measured pump-off seed detector output"
# The archived fit script builds the Balanced spectrum from the historical
# solver. The exact measured-node identity is with the recorded input arrays.
_FROZEN_SOURCE_MODEL = {
    "response_method": "pole_residue",
    "scan_points": 401,
    "velocity_measure": "truncated",
    "velocity_cutoff_sigma": 4.0,
    "velocity_step_mps": 1.0,
    "floquet_order": 3,
    "phase_detail": "ultra",
    "linewidth_hz": 5.746e6,
    "rb85_mass_kg": 1.4100e-25,
    "d1_excited_splitting_hz": 361.58e6,
}


REFIT_MODEL_ID = "fwm_20261013_warp_affine_residual_v2"
CURRENT_MODEL_ID = "fwm_20261013_warp_affine_residual_v3"
MODEL_ID = CURRENT_MODEL_ID
_current_record = cached_profile_loader(
    lambda: Path(__file__).with_name("data") / "fwm_lab_gain_refit_v3.json")
_refit_record = cached_profile_loader(
    lambda: Path(__file__).with_name("data") / "fwm_lab_gain_refit_v2.json")


def _arm_coefficients(model_id):
    if model_id == LEGACY_MODEL_ID:
        return _ARMS
    if model_id == CURRENT_MODEL_ID:
        return _current_record()["arms"]
    if model_id == REFIT_MODEL_ID:
        return _refit_record()["arms"]
    raise ValueError(f"Unknown lab gain calibration version: {model_id!r}")


def profile(*, model_id=CURRENT_MODEL_ID):
    """Fresh calibration metadata; empirical fit, not independent validation."""
    arms = _arm_coefficients(model_id)
    refit = model_id in (REFIT_MODEL_ID, CURRENT_MODEL_ID)
    result = {
        "model_id": model_id,
        "measurement_date": "2026-10-13",
        "source_workbook_sha256": measured.SOURCE_WORKBOOK_SHA256,
        "calibration_record": "analysis/fwm_experiment_20261022/gain_profile.json",
        "calibration_record_commit": "4fbf04b7b39a4ea71612ac8d306d4ce964224aad",
        "frozen_source_model": dict(_FROZEN_SOURCE_MODEL),
        "nominal_context": copy.deepcopy(_NOMINAL),
        "eom_support_MHz": list(measured.EOM_SUPPORT_MHZ),
        "internal_delta_support_mhz": [float(_NODES[0]), float(_NODES[-1])],
        "gain_quantity": _GAIN_QUANTITY,
        "gain_reference_plane": "pump-off seed detector output, common measured voltage conversion",
        "method": "per-arm frequency shift/scale, positive affine gain fit and bounded residual correction",
        "formula": "x_j=c_j+w_j*(delta-d_j); R_j=G_j(current,x_j)*C_j(delta)",
        "coefficient_formula": "C_j(delta_i)=measured_ratio_j_i/G_j(nominal,x_j_i)",
        "residual_interpolation": "shape-preserving cubic in log coefficient, within measured support",
        "outside_measurement": "hold boundary correction coefficient; continue current warped model",
        "outside_atomic_scan": "positive continuation to unseeded vacuum/conjugate=0 and probe=1",
        "context_policy": "changed operating conditions retain calibration and are marked as extrapolation",
        "independent_validation": False,
        "node_agreement": "Exact 23-point development fit on the frozen source arrays only",
        "canonical_transfer_modified": False,
        "noise_covariance_calibrated": False,
        "arms": copy.deepcopy(arms),
    }
    if refit:
        current = model_id == CURRENT_MODEL_ID
        record = _current_record() if current else _refit_record()
        result.pop("calibration_record_commit")
        result.update(
            calibration_record=("gabes/data/fwm_lab_gain_refit_v3.json" if current else
                                "analysis/fast_tier/fwm_lab_gain_refit_2026_11_18/fit/profile.json"),
            frozen_source_model=dict(record["frozen_source_model"]),
            source_coordinates="exact arm-specific detunings in the recorded Ultra pole solution",
            node_agreement="23 development points at exact source coordinates on the pinned Ultra model",
        )
    result["retired"] = model_id != CURRENT_MODEL_ID
    if result["retired"]:
        result["frozen_source_model"]["dipole_normalization"] = "summed-hyperfine-over-12"
    result["nominal_context"].update(_FIT_CONTROLS)
    return result


def _context_changes(raw):
    keys = {
        "D_GHz": "D_GHz", "T": "temperature_K", "P_pump": "input_pump_power_W",
        "P_probe": "input_seed_power_W", "w_pump": "w_pump_m", "w_probe": "w_probe_m",
        "L": "cell_length_m", "pump_probe_angle_deg": "pump_probe_angle_deg", "branch": "branch",
    }
    changes = []
    for nominal_key, raw_key in keys.items():
        value = raw.get(raw_key)
        try:
            matches = bool(np.isfinite(float(value)) and np.isclose(
                float(value), float(_NOMINAL[nominal_key]), rtol=1e-6, atol=1e-12))
        except (TypeError, ValueError):
            matches = False
        if not matches:
            changes.append(raw_key)
    for key, nominal in _FIT_CONTROLS.items():
        if key in raw and not np.isclose(raw[key], nominal, rtol=1e-6, atol=1e-12):
            changes.append(key)
    return changes


def source_detunings(delta_mhz, *, enabled=True, model_id=CURRENT_MODEL_ID):
    """Return the atomic detunings sampled for each lab calibration arm."""
    delta = np.asarray(delta_mhz, dtype=float)
    return {
        arm: (fit["source_center_delta_mhz"]
              + fit["frequency_scale"] * (delta - fit["measured_peak_delta_mhz"]))
        if enabled else delta
        for arm, fit in _arm_coefficients(model_id).items()
    }


def evaluate(raw, delta_mhz, *, enabled=True, model_id=CURRENT_MODEL_ID):
    """Evaluate corrected gains at arbitrary internal detunings, including tails.

    Detector/noise sliders are absent from this API. Its frozen correction acts
    on a newly computed model when physical controls change, without refitting.
    """
    axis = _axis(raw)
    delta = np.asarray(delta_mhz, dtype=float)
    if np.any(~np.isfinite(delta)):
        raise ValueError("calibration detunings must be finite")
    tolerance = 64*np.finfo(float).eps*max(1.0, float(np.max(np.abs(_NODES))))
    within = (delta >= _NODES[0]-tolerance) & (delta <= _NODES[-1]+tolerance)
    changed = _context_changes(raw)
    calibration_profile = profile(model_id=model_id)
    source_model = raw.get("gain_source_model")
    source_model_differences = None if source_model is None else [
        key for key, frozen_value in calibration_profile["frozen_source_model"].items()
        if key in source_model and source_model[key] != frozen_value
        and not (model_id in (REFIT_MODEL_ID, CURRENT_MODEL_ID) and key == "scan_points"
                 and source_model[key] >= frozen_value)
    ]
    result = {
        "applied": bool(enabled),
        "internal_delta_mhz": delta.copy(),
        "eom_MHz": measured.eom_mhz_from_internal_delta(delta),
        "within_measured_support": within,
        "context_changed": bool(changed),
        "changed_context_inputs": changed,
        "source_model": None if source_model is None else dict(source_model),
        "source_model_differences": source_model_differences,
        "status": ("DISABLED" if not enabled else
                   "EMPIRICAL_CALIBRATION_CHANGED_CONDITIONS" if changed else "EMPIRICAL_CALIBRATION"),
        "point_status": np.where(within, "CALIBRATED_MEASUREMENT_REGION", "EXTRAPOLATED_FREQUENCY"),
        "source_delta_mhz": {}, "outside_atomic_scan": {},
        "profile": calibration_profile,
        "gain_quantity": _GAIN_QUANTITY,
    }
    source_positions = source_detunings(delta, enabled=enabled, model_id=model_id)
    for arm, key in (("probe", "G_s"), ("conjugate", "G_c")):
        fit = calibration_profile["arms"][arm]
        positions = source_positions[arm]
        sample, outside = _sample_model(axis, raw[key], positions, arm=arm)
        if enabled:
            bounded = np.clip(delta, _NODES[0], _NODES[-1])
            coefficient = np.exp(PchipInterpolator(
                _NODES, fit["log_total_coefficient_at_nodes"], extrapolate=False)(bounded))
            sample = sample*coefficient
        result[key] = sample
        result["source_delta_mhz"][arm] = positions
        result["outside_atomic_scan"][arm] = outside
    return result


def calibrate_spectrum(raw, *, enabled=False, selected_delta_mhz=None, model_id=CURRENT_MODEL_ID):
    """Return a nonmutating calibrated view on the cached spectrum's own axis."""
    axis = _axis(raw)
    result = evaluate(raw, axis, enabled=enabled, model_id=model_id)
    if not enabled:
        # Preserve the exact uncalibrated result; log interpolation at its knots
        # would otherwise introduce unnecessary roundoff in a disabled toggle.
        result["G_s"] = np.array(raw["G_s"], copy=True)
        result["G_c"] = np.array(raw["G_c"], copy=True)
    result["nominal_node_deviation"] = None
    if enabled and not result["context_changed"] and result["source_model"] is not None:
        fitted_nodes = evaluate(raw, _NODES, model_id=model_id)
        result["nominal_node_deviation"] = {
            "scope": "training nodes at nominal apparatus settings; no acceptance threshold",
            "point_count": int(_NODES.size),
            "max_relative_error": {
                arm: float(np.max(np.abs(fitted_nodes[key] / _OBSERVATIONS[f"{arm}_ratio"] - 1)))
                for arm, key in (("probe", "G_s"), ("conjugate", "G_c"))
            },
        }
    if selected_delta_mhz is not None:
        selected = evaluate(raw, float(selected_delta_mhz), enabled=enabled, model_id=model_id)
        result["selected_point"] = {
            "G_s": float(selected["G_s"]), "G_c": float(selected["G_c"]),
            "available": True, "applied": bool(enabled),
            "internal_delta_mhz": float(selected_delta_mhz),
            "eom_MHz": float(selected["eom_MHz"]),
            "within_measured_support": bool(selected["within_measured_support"]),
            "context_changed": bool(selected["context_changed"]),
            "status": str(selected["point_status"])+( "; CHANGED_OPERATING_CONDITIONS" if selected["context_changed"] else ""),
        }
    return result
