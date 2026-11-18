"""FWM gain calibration, noise and selected-point readout."""
from .. import fwm
from ...fwm_gain_calibration import (
    MODEL_LAB, MODEL_LAB_REFIT, calibrate_spectrum as calibrate_experimental_gain, evaluate as
    evaluate_experimental_gain,
)
from .common import (
    NOISE_CONSTANT, NOISE_GAIN_PROPORTIONAL, PUMP_OVERLAP_EXCLUSION_MHZ, QE_DETECTOR,
    RESPONSE_GRID, RESPONSE_POLE, SEEDED_DETECTION_EFFICIENCY_PCT, SEEDED_POST_CELL_LOSS_PCT,
)
from .seeded import branch_center_GHz
from ... import constants, fwm_experiment_gain, fwm_measured_gain, observables
import functools
import json
import numpy as np
from ...passive_readout import passive_l1_from_spectrum
from ...jasperse import seed_noise_components
from .phase_matching import seeded_option_a_wavenumbers
from ...dressed_cell_absorption import selected_dressed_cell_absorption


def _measured_gain_context(raw, seed_input_assumption=""):
    """Use solved physical metadata, not fitted/capped gain, to gate a profile."""
    return {
        "D_GHz": raw.get("D_GHz"), "T": raw.get("temperature_K"),
        "P_pump": raw.get("input_pump_power_W"),
        "P_probe": raw.get("input_seed_power_W"),
        "w_pump": raw.get("w_pump_m"), "w_probe": raw.get("w_probe_m"),
        "L": raw.get("cell_length_m"),
        "pump_probe_angle_deg": raw.get("pump_probe_angle_deg"),
        "branch": raw.get("branch"),
        "seed_input_assumption": seed_input_assumption,
        "legacy_mean_field_fixture": raw.get("legacy_mean_field_fixture", False),
        "legacy_gain_calibration_fixture": raw.get("gain_closure", {}).get(
            "historical_fixture_requested", False),
    }


def _measured_gain_rf_roundoff(eom_mhz):
    """Normalize only machine roundoff at the two measured RF endpoints."""
    eom = np.asarray(eom_mhz, dtype=float)
    for boundary in fwm_measured_gain.EOM_SUPPORT_MHZ:
        eom = np.where(np.abs(eom-boundary) <= 16*np.spacing(boundary),
                       boundary, eom)
    return eom


def _measured_gain_reference_readout(raw, *, enabled=False,
                                      seed_input_assumption="",
                                      selected_delta_mhz=None,
                                      requested_branch=None):
    """Separate measured detector-reference ratios from the atomic transfer.

    The scalar operating point is evaluated directly at its RF frequency;
    neither cached display-grid interpolation nor atomic gains set this reference.
    All out-of-support or unmatched-context values remain unavailable.
    """
    context = _measured_gain_context(raw, seed_input_assumption)
    if requested_branch is not None:
        context["branch"] = requested_branch
    result = fwm_measured_gain.context_eligibility(context, enabled=enabled)
    result["seed_input_assumption"] = seed_input_assumption
    result["context"] = context
    center = raw.get("raman_center_minus_GHz")
    if center is None:
        center = branch_center_GHz(raw.get("D_GHz", np.nan), -1)
    delta = (np.asarray(raw["probe_axis_GHz"], dtype=float)-center)*1e3
    eom = _measured_gain_rf_roundoff(
        fwm_measured_gain.eom_mhz_from_internal_delta(delta))
    curve = fwm_measured_gain.reference_curve(eom)
    active = bool(result["applied"])
    within = np.asarray(curve["within_support"], dtype=bool)
    result.update(eom_MHz=eom, internal_delta_mhz=delta, within_support=within)
    for key in ("measured_gain_probe", "measured_gain_conjugate"):
        result[key] = (np.asarray(curve[key], dtype=float) if active
                       else np.full(eom.shape, np.nan))
    result["context_applied"] = active
    result["applied"] = bool(active and np.any(within))
    if active and not np.any(within):
        result["status"] = "OUTSIDE_RF_SUPPORT"
    if selected_delta_mhz is not None:
        selected_eom = float(_measured_gain_rf_roundoff(
            fwm_measured_gain.eom_mhz_from_internal_delta(selected_delta_mhz)))
        selected = fwm_measured_gain.reference_curve(selected_eom)
        inside = bool(selected["within_support"])
        available = bool(active and inside)
        point = {
            "eom_MHz": selected_eom,
            "internal_delta_mhz": float(selected_delta_mhz),
            "available": available, "within_support": inside,
            "status": (result["status"] if not active else
                       str(selected["status"]) if inside else "OUTSIDE_RF_SUPPORT"),
        }
        for key in ("measured_gain_probe", "measured_gain_conjugate"):
            point[key] = float(selected[key]) if available else float("nan")
        result["selected_point"] = point
    return result


def _detection_efficiency_from_params(params):
    """Return η while preserving legacy loss/QE overrides from SABES and APIs."""
    if "after_cell_eff_pct" in params:
        after = float(params["after_cell_eff_pct"]) / 100.0
        qe = float(params.get("qe_pct", QE_DETECTOR * 100.0)) / 100.0
        if not (0.0 <= after <= 1.0 and 0.0 <= qe <= 1.0):
            raise ValueError("after-cell transmission and detector QE must be in [0,100]%")
        return after * qe
    loss_pct = float(params.get("loss_pct", SEEDED_POST_CELL_LOSS_PCT))
    qe_pct = float(params.get("qe_pct", QE_DETECTOR * 100.0))
    legacy_eta = (qe_pct / 100.0) * (1.0 - loss_pct / 100.0)
    direct_pct = params.get("detection_eff_pct")
    if direct_pct is None:
        return legacy_eta

    legacy_inputs_changed = (
        not np.isclose(loss_pct, SEEDED_POST_CELL_LOSS_PCT, rtol=0.0, atol=1e-12)
        or not np.isclose(qe_pct, QE_DETECTOR * 100.0, rtol=0.0, atol=1e-12)
    )
    return legacy_eta if legacy_inputs_changed else float(direct_pct) / 100.0


def _seeded_excess_noise(G_s, params):
    """Source-referred noise selected by cheap readout controls only."""
    mode = params.get("excess_noise_mode") or NOISE_CONSTANT
    if mode == NOISE_CONSTANT:
        return float(params.get("excess_noise", 0.0))
    if mode == NOISE_GAIN_PROPORTIONAL:
        return observables.gain_proportional_excess_noise(
            G_s, params.get("excess_noise_slope", 0.0))
    raise ValueError(f"Unknown excess noise mode: {mode}")


def _seeded_noise_caption(params):
    return f"seed RIN a = {float(params.get('seed_classical_rin_scaled', 0.0))*1e-14:.3g} /Hz"


def _seeded_passive_l1_readout(raw, params):
    """Evaluate readout-only inputs without mutating the cached atomic solve."""
    power = raw.get("input_seed_power_W", float(params.get("probe_uw", 8.0))*1e-6)
    omega = raw.get("omega_probe_rad_s")
    if omega is None:
        _, k, _ = seeded_option_a_wavenumbers(
            raw.get("D_GHz", params.get("opd", 0.9)), raw["probe_axis_GHz"])
        omega = constants.C_LIGHT * k
    seed = seed_noise_components(
        power, omega,
        classical_rin_per_Hz=float(params.get("seed_classical_rin_scaled", 0.0))*1e-14)
    eta = (_detection_efficiency_from_params(params)
           if any(key in params for key in ("after_cell_eff_pct", "qe_pct", "loss_pct", "detection_eff_pct"))
           else float(raw.get("eta", SEEDED_DETECTION_EFFICIENCY_PCT / 100.0)))
    result = passive_l1_from_spectrum(
        dict(raw, eta=eta), seed_fano=seed["seed_fano"])
    result["seed_noise"] = seed
    result["eta_after_cell_QE"] = eta
    # Compatibility alias only; the retired constant/gain-linear laws are not
    # applied. All current classical seed noise enters through Var(N0).
    result["passive_l1_with_excess_noise_dB"] = result["NF_dB"]
    return result


def _seeded_sidebar_readouts(raw, params, readout, probe_GHz, absorption=None):
    """Display calculated quantities without adding them to solver inputs."""
    axis = np.asarray(raw["probe_axis_GHz"])
    shot = float(np.interp(probe_GHz, axis, readout["seed_noise"]["shot_rin_per_Hz"]))
    entries = {"seed_shot_noise_b": {
        "value": shot * 1e14,
        "status": "Calculated one-sided shot RIN at the seed input; not an adjustable floor."}}
    eta_ext = readout["eta_after_cell_QE"]
    for arm in ("probe", "conjugate"):
        value = None
        status = "Absorption/emission separation has not been calculated."
        if absorption is not None:
            transmission = absorption.get(arm, {}).get("in_cell_absorption_transmission")
            if transmission is not None and np.isfinite(transmission):
                value = 100.0 * float(transmission) * eta_ext
                status = "Calculated absorption transmission × after-cell transmission × detector QE."
        entries[f"eta_{arm}_pct"] = {"value": value, "status": status}
    return entries


def _seeded_absorption_readout(raw, probe_GHz):
    """Selected-point ordered absorption; no efficiency is inferred from gain."""
    required = ("temperature_K", "input_pump_power_W", "input_seed_power_W",
                "w_pump_m", "w_probe_m", "cell_length_m", "effective_line_strength")
    if any(key not in raw for key in required):
        return {"status": "UNAVAILABLE_LEGACY_ATOMIC_INPUTS",
                "reason": "The saved spectrum lacks the physical atomic inputs."}
    delta_mhz = (probe_GHz - branch_center_GHz(raw["D_GHz"], -1))*1e3
    try:
        return selected_dressed_cell_absorption(
            D_GHz=raw["D_GHz"], T=raw["temperature_K"],
            P_pump=raw["input_pump_power_W"], P_probe=raw["input_seed_power_W"],
            w_pump=raw["w_pump_m"], w_probe=raw["w_probe_m"], delta_mhz=delta_mhz,
            L=raw["cell_length_m"], weak_line_strength=raw["effective_line_strength"],
            transit_rate=raw.get("transit_reset_rate_rad_s", constants.GAMMA_GG),
            velocity_cutoff=raw.get("velocity_cutoff_sigma", 4.0),
            branch=raw.get("branch", -1))
    except (ValueError, NotImplementedError, np.linalg.LinAlgError) as error:
        return {"status": "UNAVAILABLE_ABSORPTION_VALIDATION", "reason": str(error)}


def _passive_l1_selected(readout, raw, probe_GHz):
    """Read a point only within a contiguous valid part of the sampled curve."""
    x = np.asarray(raw["probe_axis_GHz"], dtype=float)
    idx = int(np.searchsorted(x, probe_GHz))
    if idx >= x.size:
        brackets = (x.size - 1,)
    elif idx == 0 or np.isclose(x[idx], probe_GHz, rtol=0.0, atol=1e-12):
        brackets = (idx,)
    else:
        brackets = (idx - 1, idx)
    valid = np.asarray(readout["passive_completion_valid"], dtype=bool)
    noise = np.asarray(readout["passive_l1_noise_dB"], dtype=float)
    completion_valid = bool(np.all(valid[list(brackets)]))
    in_scan = bool(x[0] - 1e-12 <= probe_GHz <= x[-1] + 1e-12)
    available = bool(completion_valid and in_scan
                     and np.all(np.isfinite(noise[list(brackets)])))
    statuses = np.asarray(readout["passive_l1_status"])
    status = (str(statuses[brackets[0]]) if len(brackets) == 1 or available
              else "L1_UNAVAILABLE_BETWEEN_INVALID_POINTS")
    if not in_scan:
        status = "L1_OUTSIDE_SAMPLED_SCAN"
    point = {"passive_completion_valid": completion_valid,
             "passive_l1_available": available,
             "passive_l1_status": status}
    point["NF"] = (float(np.interp(probe_GHz, x, readout["NF"]))
                   if available else float("nan"))
    point["NF_dB"] = (float(10 * np.log10(point["NF"]))
                      if available and point["NF"] > 0 else float("nan"))
    for key in ("passive_l1_noise_dB", "passive_l1_with_excess_noise_dB"):
        point[key] = point["NF_dB"]
    for key in ("passive_l1_G_s", "passive_l1_G_c", "passive_l1_D", "passive_l1_T"):
        point[key] = float(np.interp(probe_GHz, x, readout[key]))
    point["passive_completion_min_eig"] = float(np.min(
        np.asarray(readout["passive_completion_min_eig"])[list(brackets)]))
    return point


NOISE_PASSIVE_L1 = "Passive L1"
NOISE_EXPERIMENTAL_JASPERSE = "Experimental Jasperse"
SQUEEZING_INTENSITY_DIFFERENCE = "Intensity difference"
SQUEEZING_QUADRATURE_PHASE = "Quadrature phase scan"


def _power_gain_metric(label, gain, **context):
    """Keep the unrounded operating value beside the legacy export text."""
    return dict(label=label, value=f"{gain:.2f}", numeric_value=float(gain) if np.isfinite(gain) else None,
                quantity="power_gain", **context)


def _experimental_noise_selected(readout, raw, delta_mhz):
    """Interpolate the linear NF; the metric and plotted marker share this value."""
    axis = readout.get("internal_delta_mhz")
    if axis is None:
        axis = (np.asarray(raw["probe_axis_GHz"]) - raw["raman_center_minus_GHz"]) * 1e3
    axis = np.asarray(axis)
    order = np.argsort(axis)
    x, nf = axis[order], np.asarray(readout["NF"])[order]
    idx = min(int(np.searchsorted(x, delta_mhz)), len(x)-1)
    brackets = [idx] if idx == 0 or abs(x[idx]-delta_mhz) < 1e-9 else [idx-1, idx]
    available = bool(x[0]-1e-9 <= delta_mhz <= x[-1]+1e-9
                     and np.all(np.isfinite(nf[brackets])))
    value = float(np.interp(delta_mhz, x, nf)) if available else float("nan")
    point = {"NF": value, "NF_dB": float(10*np.log10(value)) if value > 0 else float("nan"),
             "available": available, "model": NOISE_EXPERIMENTAL_JASPERSE,
             "inference_approximate": bool(np.any(readout["inference_approximate"][order][brackets])),
             "fit_relative_residual": float(np.max(readout["fit_relative_residual"][order][brackets]))}
    for key in ("G_s", "G_c", "input_G_s", "input_G_c"):
        point[key] = float(np.interp(delta_mhz, x, np.asarray(readout[key])[order]))
    return point


@functools.lru_cache(maxsize=8)
def _focused_readout_reference(encoded_context, extra_probe_points):
    """Dense pole solve for explicit coordinates, cached independently of detection."""
    return fwm.compute_spectrum(**json.loads(encoded_context),
                            extra_probe_points_GHz=extra_probe_points)


def _focused_readout_source(raw, tpd, gain_axis, *, lab_calibration, lab_model=MODEL_LAB):
    if not raw.get("focused_readout_reference"):
        return raw, "CACHED_GRID_INTERPOLATION"
    expected = raw.get("gain_budget", {}).get("displayed", {})
    if not all(key in expected and np.array_equal(raw[key], expected[key]) for key in ("G_s", "G_c")):
        return raw, "CACHED_GRID_OVERRIDE"
    context = dict(raw["stark_recompute_context"])
    native = np.asarray(raw["probe_axis_GHz"])
    center = raw["raman_center_minus_GHz"]
    def in_span(delta):
        points = center + np.asarray(delta, dtype=float) * 1e-3
        return points[(points >= native[0]) & (points <= native[-1])
                      & (np.abs(points - raw["D_GHz"]) > PUMP_OVERLAP_EXCLUSION_MHZ * 1e-3)]

    extra = in_span([tpd])
    if lab_calibration:
        measured_nodes = fwm_measured_gain.internal_delta_mhz_from_eom(
            fwm_measured_gain.observations()["eom_MHz"])
        model_id = (fwm_experiment_gain.REFIT_MODEL_ID if lab_model == MODEL_LAB_REFIT
                    else fwm_experiment_gain.MODEL_ID)
        warp = fwm_experiment_gain.source_detunings(np.r_[gain_axis, measured_nodes], model_id=model_id)
        extra = np.concatenate((extra, *(in_span(delta) for delta in warp.values())))
    extra = tuple(np.setdiff1d(extra, native).tolist())
    estimator = raw["response_estimator"]
    if estimator["method"] == RESPONSE_GRID:
        return raw, "GRID_FALLBACK_INTERPOLATION"
    if not extra and estimator.get("interpolation") == "none":
        return raw, "NATIVE_SOURCE_GRID"
    context.update(response_method=RESPONSE_POLE,
                   pump_stark_fraction=raw.get("pump_stark_response", {}).get("fraction", 1.0))
    source = _focused_readout_reference(json.dumps(context, sort_keys=True, allow_nan=False), extra)
    status = ("DENSE_GRID_FALLBACK" if source["response_estimator"]["method"] == RESPONSE_GRID
              else "DENSE_POLE_READOUT_COORDINATES")
    return source, status


def _seeded_readout_rows(result, rows, source_size):
    """Select native rows while keeping nested provenance aligned to the readout."""
    projected = {}
    for key, value in result.items():
        if isinstance(value, dict):
            value = _seeded_readout_rows(value, rows, source_size)
        elif isinstance(value, np.ndarray) and value.shape[:1] == (source_size,):
            value = value[rows]
        projected[key] = value
    return projected


def seeded_operating_readout(raw, params, *, include_passive=False):
    """Resolve FWM gain and NF selections without figures or absorption solves."""
    tpd = params["tpd"]
    phase_view = params.get("squeezing_view") == SQUEEZING_QUADRATURE_PHASE
    gain_model = params.get("gain_calibration_model", MODEL_LAB)
    gain_requested = params.get("experimental_gain_calibration", False)
    lab_calibration = gain_requested and gain_model in (MODEL_LAB, MODEL_LAB_REFIT)
    gain_axis = (raw["probe_axis_GHz"] - raw["raman_center_minus_GHz"]) * 1e3
    if gain_requested:
        gain_axis = np.unique(np.r_[gain_axis, tpd])
        if lab_calibration:
            support = sorted(fwm_measured_gain.internal_delta_mhz_from_eom(
                np.array(fwm_measured_gain.EOM_SUPPORT_MHZ)))
            gain_axis = np.unique(np.r_[gain_axis, np.linspace(*support, 221)])
    source = raw
    coordinate_status = "CACHED_GRID_INTERPOLATION"
    if params.get("exact_readout_coordinates", False):
        source, coordinate_status = _focused_readout_source(raw, tpd, gain_axis,
                                                            lab_calibration=lab_calibration, lab_model=gain_model)
    op = fwm.operating_point(source, tpd, branch=-1)
    native_indices = np.searchsorted(source["probe_axis_GHz"], raw["probe_axis_GHz"])
    experimental_gain = calibrate_experimental_gain(
        raw, enabled=gain_requested, selected_delta_mhz=tpd, model=gain_model)
    if lab_calibration and source is not raw:
        exact_gain = calibrate_experimental_gain(
            source, enabled=True, selected_delta_mhz=tpd, model=gain_model)
        exact_gain = _seeded_readout_rows(exact_gain, native_indices,
                                         source["probe_axis_GHz"].size)
        exact_gain["exact_nominal_node_deviation"] = exact_gain["nominal_node_deviation"]
        exact_gain["nominal_node_deviation"] = experimental_gain["nominal_node_deviation"]
        experimental_gain = exact_gain
    gain_active = experimental_gain["applied"]
    displayed_gain = experimental_gain
    noise_raw = raw
    if gain_active:
        displayed_gain = evaluate_experimental_gain(
            source if lab_calibration else raw, gain_axis, enabled=True, model=gain_model)
        noise_raw = dict(raw, probe_axis_GHz=raw["raman_center_minus_GHz"]+gain_axis*1e-3)
        omega = np.asarray(raw.get("omega_probe_rad_s", constants.OMEGA_D1))
        if omega.ndim:
            noise_raw["omega_probe_rad_s"] = np.interp(
                noise_raw["probe_axis_GHz"], raw["probe_axis_GHz"], omega)
    displayed_gs = displayed_gain["G_s"] if gain_active else raw["G_s"]
    displayed_gc = displayed_gain["G_c"] if gain_active else raw["G_c"]
    displayed_point = experimental_gain["selected_point"] if gain_active else op

    noise_model = (NOISE_EXPERIMENTAL_JASPERSE if phase_view else
                   params.get("noise_model", NOISE_PASSIVE_L1))
    if source is not raw and not gain_active:
        displayed_gs, displayed_gc = (source[key][native_indices] for key in ("G_s", "G_c"))
        noise_raw = dict(raw, G_s=displayed_gs, G_c=displayed_gc)
    passive_readout = None
    passive_op = None
    experimental_noise = None
    if ("T_canonical_small_signal" in raw
            and (include_passive or noise_model == NOISE_PASSIVE_L1)):
        point_readout = _seeded_passive_l1_readout(source, params)
        passive_op = _passive_l1_selected(point_readout, source, op["probe_GHz"])
        passive_readout = point_readout
        if source is not raw:
            passive_readout = _seeded_readout_rows(point_readout, native_indices,
                                                  source["probe_axis_GHz"].size)
    noise_point = passive_op
    if noise_model == NOISE_EXPERIMENTAL_JASPERSE:
        experimental_noise = fwm.experimental_jasperse_readout(
            noise_raw, params, gains=displayed_gain if gain_active else None)
        experimental_noise["internal_delta_mhz"] = gain_axis
        noise_point = _experimental_noise_selected(experimental_noise, raw, tpd)
        if not gain_active and source is not raw:
            selected_idx = np.searchsorted(source["probe_axis_GHz"], op["probe_GHz"])
            if (selected_idx < source["probe_axis_GHz"].size
                    and source["probe_axis_GHz"][selected_idx] == op["probe_GHz"]):
                point_source = dict(source, omega_probe_rad_s=source["omega_probe_rad_s"][selected_idx])
                point_gain = {key: source[key][selected_idx] for key in ("G_s", "G_c")}
                point_noise = fwm.experimental_jasperse_readout(point_source, params, gains=point_gain)
                noise_point = {
                    "NF": float(point_noise["NF"]), "NF_dB": float(point_noise["NF_dB"]),
                    "available": bool(np.isfinite(point_noise["NF"])),
                    "model": NOISE_EXPERIMENTAL_JASPERSE,
                    "inference_approximate": bool(point_noise["inference_approximate"]),
                    "fit_relative_residual": float(point_noise["fit_relative_residual"]),
                    **{key: float(point_noise[key]) for key in ("G_s", "G_c", "input_G_s", "input_G_c")},
                }

    gain_point = {
        "G_s": float(displayed_point["G_s"]),
        "G_c": float(displayed_point["G_c"]),
        "two_photon_detuning_mhz": float(tpd),
        "gain_basis": (experimental_gain.get("gain_quantity", "effective source power gain")
                       if gain_active else "atomic-medium power gain"),
        "phase_independent": True,
    }
    if phase_view and not gain_axis[0]-1e-9 <= tpd <= gain_axis[-1]+1e-9:
        gain_point.update(G_s=float("nan"), G_c=float("nan"))
    return {
        "atomic_operating_point": op,
        "gain_operating_point": gain_point,
        "gain_curve": {
            "probe_axis_GHz": raw["raman_center_minus_GHz"] + gain_axis * 1e-3
            if gain_active else raw["probe_axis_GHz"],
            "internal_delta_mhz": gain_axis, "G_s": displayed_gs, "G_c": displayed_gc,
        },
        "experimental_gain_calibration": experimental_gain,
        "noise_model": noise_model,
        "readout_coordinate_status": coordinate_status,
        "readout_certification_scope": (
            "LAB_AND_UNCALIBRATED_EXACT_COORDINATES"
            if coordinate_status in {"NATIVE_SOURCE_GRID", "DENSE_POLE_READOUT_COORDINATES"}
            and not phase_view and (not gain_active or lab_calibration)
            else "UNCERTIFIED_CALIBRATION_OR_SOURCE"),
        "noise_operating_point": noise_point,
        "jasperse_readout": experimental_noise if experimental_noise is not None else passive_readout,
        "experimental_jasperse_readout": experimental_noise,
        "passive_l1_readout": passive_readout,
        "passive_l1_operating_point": passive_op,
    }


def _plot_reference(axis, value, label):
    """Label the physical reference on the plot without adding a data trace."""
    axis.axhline(value, color="#47555a", lw=.8, ls=":")
    axis.annotate(label, xy=(.99, value), xycoords=axis.get_yaxis_transform(),
                  xytext=(0, 2), textcoords="offset points", ha="right",
                  va="bottom", fontsize=8, color="#34464b", clip_on=True,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.7))


def _label_noise_axis(axis, modeled):
    reference = "shot noise" if modeled else "diagnostic reference"
    _plot_reference(axis, 0.0, f"0 dB ({reference})")
    axis.set_ylabel("Intensity-difference noise [dB]" if modeled
                    else "Gain-referred noise diagnostic [dB]")
