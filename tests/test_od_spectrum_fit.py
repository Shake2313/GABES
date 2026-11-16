"""Physical OD fit checks using independent data noise and the real OD kernel."""
import json

import numpy as np
import pytest

from gabes import species
from gabes.experimental_alignment import AxisAlignment
from gabes.experimental_csv import CALIBRATION_ABSOLUTE_GAIN_OFFSET, load_experimental_csv
from gabes.od_spectrum_fit import ODFitInput, fit_od_spectrum, od_transmission, prepare_od_fit
from gabes.schemes.sas import EXACT_WEAK_PROBE, SASScheme


@pytest.fixture
def conditions():
    return dict(SASScheme().defaults(), species=species.SPECIES_ORDER[1],
                pump_power_mw=0.0, probe_power_uw=0.0, temp_c=63.0, cell_mm=23.0)


def test_exact_spectrum_recovers_temperature_and_offset(conditions):
    x = np.linspace(-3100.0, 3100.0, 501)
    truth = dict(conditions, temp_c=67.2)
    y = od_transmission(x, truth, offset_mhz=21.7)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    fit = fit_od_spectrum(data, conditions)
    assert fit["identifiable"]
    assert fit["scope"] == "calibration_only"
    assert fit["probe_model"] == "weak_probe_legacy"
    assert "weak_probe_profile" not in fit
    assert [item["value"] for item in fit["parameters"]] == pytest.approx([67.2, 21.7], abs=1e-7)
    assert fit["fit_residuals"]["rms"] < 1e-10
    assert fit["global_evaluations"] == 65
    json.dumps(fit, allow_nan=False)


def test_noisy_absolute_trace_keeps_holdout_separate_and_flags_density_length(conditions):
    x = np.linspace(-3200.0, 3200.0, 601)
    known = (67.2, 11.2, 21.7)
    clean = od_transmission(x, dict(conditions, temp_c=known[0]),
                            extra_width_mhz=known[1], offset_mhz=known[2])
    noisy = clean + np.random.default_rng(31).normal(0.0, 0.0009, x.size)
    data = "\n".join(f"{a / 1000},{b}" for a, b in zip(x, noisy)).encode()
    trace = load_experimental_csv(data, calibration_mode=CALIBRATION_ABSOLUTE_GAIN_OFFSET,
                                  gain=1.0, offset=0.0)
    prepared = prepare_od_fit(trace, AxisAlignment())
    fit = fit_od_spectrum(prepared, conditions, free=("temp_c", "extra_width_mhz", "offset_mhz"),
                          fit_range_mhz=(-3200, 2700), heldout_range_mhz=(2701, 3200))
    assert fit["identifiable"] and fit["scope"] == "fit_with_disjoint_holdout"
    assert not np.any(np.asarray(fit["train_mask"]) & fit["heldout_mask"])
    assert fit["heldout_residuals"]["rms"] < 0.0015
    for param, value in zip(fit["parameters"], known):
        assert abs(param["value"] - value) < param["standard_error"]
    degenerate = fit_od_spectrum(prepared, conditions,
                                 free=("temp_c", "cell_mm", "extra_width_mhz", "offset_mhz"))
    assert abs(degenerate["correlation"][0][1]) > 0.98
    assert not degenerate["identifiable"]
    assert all(p["standard_error"] is None for p in degenerate["parameters"])
    with pytest.raises(ValueError, match="disjoint"):
        fit_od_spectrum(prepared, conditions, fit_range_mhz=(-1000, 500),
                        heldout_range_mhz=(0, 1000))
    with pytest.raises(ValueError, match="absolute"):
        prepare_od_fit(load_experimental_csv(data), AxisAlignment())
    with pytest.raises(ValueError, match="weak-probe"):
        fit_od_spectrum(prepared, dict(conditions, pump_power_mw=1.0))
    with pytest.raises(ValueError, match="supported list"):
        fit_od_spectrum(prepared, conditions, free=("line_strength",))


@pytest.mark.parametrize(("coated", "status", "model"), [
    (False, "active", "weak_probe_exact_maxwell_voigt"),
    (True, "inactive", "weak_probe_legacy"),
])
def test_standalone_fit_reports_actual_weak_probe_backend(conditions, coated, status, model):
    conditions = dict(conditions, weak_probe_profile=EXACT_WEAK_PROBE, paraffin_coated=coated)
    x = np.linspace(-3100.0, 3100.0, 77)
    y = od_transmission(x, conditions)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    fit = fit_od_spectrum(data, conditions, free=("temp_c",))
    raw = SASScheme()._compute_species(conditions, scan_rad_s=x * (2.0 * np.pi * 1e6))
    assert fit["probe_model"] == raw["probe_model"] == model
    assert fit["weak_probe_profile"] == raw["weak_probe_profile"]
    assert fit["weak_probe_profile"]["status"] == status
    assert fit["fit_residuals"]["rms"] < 1e-10
