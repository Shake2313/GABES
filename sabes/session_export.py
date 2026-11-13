"""Portable SABES session record with measurement planes and unknowns intact."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math


def json_value(value):
    """Strict JSON: unavailable numeric values become null, not NaN/Infinity."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported SABES export value: {type(value).__name__}")


def _gold_reference():
    """Separate reported and model-derived IDS for the 85Rb operating point."""
    paper_eta = 0.865
    code_eta = 0.945 * 0.92
    slope_nf = 0.134
    # Invert NF_det = 1 - eta + eta * NF_src for the slope-derived readout.
    inferred_nf = (slope_nf - (1.0 - code_eta)) / code_eta
    return {
        "source": "https://doi.org/10.1038/s41598-025-86479-w",
        "detector_efficiency": {
            "paper": {"eta": paper_eta, "uncertainty": 0.002,
                      "convention": "one minus the reported total loss"},
            "code": {"eta": code_eta, "uncertainty": None,
                     "convention": "sequential optics transmission times detector QE"},
        },
        "rounded_output_power": {
            "probe_uW": 111.0, "conjugate_uW": 109.0,
            "paper_measurement_plane": None, "uncertainty_uW": None,
            "v7_model_plane": "cell exit before external detection loss",
        },
        "ids": {
            "direct_detector_dB": -7.8,
            "slope_detector_dB": 10.0 * math.log10(slope_nf),
            "slope_detector_analysis_frequency_Hz": 150e3,
            "intrinsic_inferred_from_slope": {
                "dB": 10.0 * math.log10(inferred_nf),
                "assumed_eta": code_eta,
                "method": "invert symmetric vacuum loss with slope NF=0.134",
                "uncertainty": None,
            },
        },
        "v7_gaussian": {
            "evidence_class": "output-constrained model estimate",
            "record": "docs/squeezing_report/v7_science/theory_results.json",
            "predicted_ids_dB_at_eta_0p865": -7.70926151275678,
            "predicted_ids_dB_at_eta_0p8694": -7.81890347354036,
        },
    }


def session_payload(result, detection, calibration, apparatus, profile):
    """Export actual active inputs, not merely the shipped apparatus JSON."""
    raw = result.raw or {}
    evidence = result.readout_evidence
    dependencies = calibration.output_evidence(
        gain_calibrated=bool(evidence and evidence["gain"]["calibration_applied"]),
        recorded_transmission=detection.post_cell_transmission_override is not None,
        recorded_qe=detection.qe_pct_override is not None,
    )
    rows = [{"path": path, "stage": stage, "power_w": power,
             "waist_radius_m": waist}
            for path, stage, power, waist in result.chain.budget_rows()]
    return json_value({
        "schema_version": "sabes-session-v1",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_profile": profile,
        "apparatus": apparatus.as_dict(),
        "settings": result.settings,
        "detection_settings": detection,
        "calibration": {c.name: c.as_dict() for c in calibration},
        "derived_parameters": result.params,
        "calibration_dependencies": dependencies,
        "literature_reference": _gold_reference(),
        "detection_geometry": result.geometry,
        "source_power_budget": rows,
        "operating_point": {
            "gains": result.gains,
            "NF_dB": result.gain_referred_noise_db,
            "claim_gate": raw.get("claim_gate"),
        },
        "selected_readout": {
            "noise_model": result.noise_model,
            "raw_atomic_gains": result.raw_gains,
            "physical_squeezing_dB": result.physical_squeezing_db,
            "evidence": result.readout_evidence,
        },
        "scope": {
            "display_coordinates": "schematic units; physical x/y/z unknown",
            "distance_plane": "cell output end to D-mirror",
            "rf_generator_display_dbm": apparatus.get_fact("rf_display_setting").value,
            "rf_eom_port_measured_dbm": apparatus.get_fact("eom_port_power").value,
            "rf_model_input_dbm": result.settings.eom_rf_dbm,
            "detector_gain_switch_v_per_a": apparatus.get_fact("detector_gain_switch").value,
            "detector_load_measured_ohm": apparatus.get_fact("detector_load").value,
            "instrument_models": "Virtual traces; 50-ohm SA conversion and "
                                 "stock noise floor are assumptions. Actual "
                                 "Monitor +/- volts are not calibrated here.",
            "noise_eater": "Transmission only; no suppression model.",
            "sql_reference": "Ideal coherent bypass power before unmeasured "
                             "SQL attenuation; not a measured matched SQL.",
        },
    })


def session_json(result, detection, calibration, apparatus, profile):
    return json.dumps(session_payload(result, detection, calibration,
                                     apparatus, profile),
                      ensure_ascii=False, indent=2, allow_nan=False)
