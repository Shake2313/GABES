"""Physical OD fit checks using independent data noise and the real OD kernel."""
import json

import numpy as np
import pytest

from gabes import species
from gabes.experimental_alignment import AxisAlignment
from gabes.experimental_csv import CALIBRATION_ABSOLUTE_GAIN_OFFSET, load_experimental_csv
from gabes import od_spectrum_fit
from gabes.od_spectrum_fit import (
    CANDIDATE_PARAMETERS, PARAMETERS, VALIDATED_SCOPE, ODFitInput, fit_od_spectrum,
    od_transmission, offered_fit_parameters, prepare_od_fit, probe_peak_intensity_mw_cm2,
)
from gabes.schemes.sas import AUTO_WEAK_PROBE, EXACT_WEAK_PROBE, LEGACY_WEAK_PROBE, SASScheme


@pytest.fixture
def conditions():
    # ⟦AI⟧ pre-D20 model set: these fits pin the Exact and Legacy backends (M2b).
    return dict(SASScheme().defaults(), species=species.SPECIES_ORDER[1],
                pump_power_mw=0.0, probe_power_uw=0.0, temp_c=63.0, cell_mm=23.0,
                od_model="pre_d20")


def test_exact_spectrum_recovers_temperature_and_offset(conditions):
    conditions = dict(conditions, weak_probe_profile=LEGACY_WEAK_PROBE)
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


@pytest.mark.parametrize(("profile", "coated", "status", "model"), [
    (AUTO_WEAK_PROBE, False, "active", "weak_probe_exact_maxwell_voigt"),
    (EXACT_WEAK_PROBE, False, "active", "weak_probe_exact_maxwell_voigt"),
    (EXACT_WEAK_PROBE, True, "inactive", "weak_probe_legacy"),
])
def test_standalone_fit_reports_actual_weak_probe_backend(
        conditions, profile, coated, status, model):
    conditions = dict(conditions, paraffin_coated=coated)
    if profile != AUTO_WEAK_PROBE:
        conditions["weak_probe_profile"] = profile
    assert conditions["weak_probe_profile"] == profile
    x = np.linspace(-3100.0, 3100.0, 77)
    y = od_transmission(x, conditions)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    fit = fit_od_spectrum(data, conditions, free=("temp_c",))
    raw = SASScheme()._compute_species(conditions, scan_rad_s=x * (2.0 * np.pi * 1e6))
    assert fit["probe_model"] == raw["probe_model"] == model
    assert fit["weak_probe_profile"] == raw["weak_probe_profile"]
    assert fit["weak_probe_profile"]["status"] == status
    assert fit["weak_probe_profile"]["requested"] == profile
    assert fit["fit_residuals"]["rms"] < 1e-10


def test_rate_model_fit_uses_the_same_evaluator_at_finite_probe_power():
    """A finite probe needs an explicit model; offset fits reuse the cached table."""
    rate = "od_finite_rb_mf_rate_v1"
    params = dict(SASScheme().defaults(), species="Rb (natural)", line="D2", pump_power_mw=0.0,
                  probe_power_uw=100.0, temp_c=45.0, cell_mm=75.0)
    x = np.linspace(-3500.0, 5000.0, 160)
    y = od_transmission(x, params, offset_mhz=-8.4, model_id=rate)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    with pytest.raises(ValueError, match="weak-probe"):
        fit_od_spectrum(data, params, free=("offset_mhz",))
    fit = fit_od_spectrum(data, params, free=("offset_mhz",), model_id=rate)
    assert fit["probe_model"] == rate and fit["probe_power_uw"] == 100.0
    assert fit["model"] == "resolved uniform pump-off OD, finite probe"
    assert fit["parameters"][0]["value"] == pytest.approx(-8.4, abs=1e-6)
    assert fit["fit_residuals"]["rms"] < 1e-9


def test_default_d20_fit_reports_the_transit_voigt(conditions):
    conditions = dict(conditions, od_model="d20")
    x = np.linspace(-3100.0, 3100.0, 77)
    y = od_transmission(x, conditions)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    fit = fit_od_spectrum(data, conditions, free=("temp_c",))
    assert fit["probe_model"] == "od_weak_exact_voigt_transit_v1"
    assert fit["fit_residuals"]["rms"] < 1e-10


def test_progress_callback_reports_every_evaluation_and_leaves_the_fit_unchanged(conditions):
    x = np.linspace(-3100.0, 3100.0, 121)
    y = od_transmission(x, dict(conditions, temp_c=66.0), offset_mhz=12.0)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    plain = fit_od_spectrum(data, conditions)
    seen = []
    tracked = fit_od_spectrum(data, conditions, progress=seen.append)
    for key in ("parameters", "residual", "global_evaluations", "local_evaluations", "status"):
        assert tracked[key] == plain[key]
    total = plain["global_evaluations"] + plain["local_evaluations"]
    assert [state.evaluations for state in seen] == list(range(1, total + 1))
    stages = [state.stage for state in seen]
    assert stages == ["global"] * plain["global_evaluations"] + ["local"] * plain["local_evaluations"]
    costs = [state.best_cost for state in seen]
    assert all(later <= earlier for earlier, later in zip(costs, costs[1:]))
    residual = np.asarray(plain["residual"]) / np.asarray(plain["sigma"])
    assert costs[-1] <= 0.5 * float(np.dot(residual, residual)) * (1 + 1e-12)


@pytest.mark.parametrize(("stop_at", "stage"), [(10, "global"), (70, "local")])
def test_progress_callback_can_cancel_between_evaluations(conditions, stop_at, stage):
    from gabes.od_spectrum_fit import ODFitCancelled

    x = np.linspace(-3100.0, 3100.0, 121)
    y = od_transmission(x, conditions, offset_mhz=5.0)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    calls = []

    def stop(state):
        calls.append(state.evaluations)
        return state.evaluations < stop_at

    with pytest.raises(ODFitCancelled) as cancelled:
        fit_od_spectrum(data, conditions, progress=stop)
    assert calls[-1] == stop_at == cancelled.value.progress.evaluations
    assert cancelled.value.progress.stage == stage
    assert not isinstance(cancelled.value, ValueError)


RATE = "od_finite_rb_mf_rate_v1"


def _rate_params(**changes):
    base = dict(SASScheme().defaults(), species="Rb (natural)", line="D2", pump_power_mw=0.0,
                probe_power_uw=50.0, probe_waist_mm=1.0, temp_c=40.0, cell_mm=75.0)
    return {**base, **changes}


def test_candidate_variables_stay_out_of_the_app_list():
    assert set(CANDIDATE_PARAMETERS) == {"probe_power_uw", "probe_waist_mm", "rb85_fraction"}
    assert not set(CANDIDATE_PARAMETERS) & set(PARAMETERS)


def test_rate_model_sees_the_probe_only_through_peak_intensity():
    x = np.linspace(-3500.0, 5000.0, 120)
    same = od_transmission(x, _rate_params(), model_id=RATE)
    np.testing.assert_array_equal(
        same, od_transmission(x, _rate_params(probe_power_uw=12.5, probe_waist_mm=0.5), model_id=RATE))
    other = od_transmission(x, _rate_params(probe_waist_mm=1.2), model_id=RATE)
    assert np.max(np.abs(other - same)) > 1e-2
    assert probe_peak_intensity_mw_cm2(50.0, 1.0) == pytest.approx(0.2 * 50.0 / np.pi, rel=1e-15)


def test_candidate_variables_have_an_explicit_scope(conditions):
    x = np.linspace(-3100.0, 3100.0, 120)
    y = od_transmission(x, conditions)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    with pytest.raises(ValueError, match="rate model"):
        fit_od_spectrum(data, conditions, free=("temp_c", "probe_power_uw"))
    with pytest.raises(ValueError, match="custom Rb composition"):
        fit_od_spectrum(data, dict(conditions, species="Rb (natural)"), free=("rb85_fraction",))
    rate = _rate_params()
    y = od_transmission(x, rate, model_id=RATE)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    with pytest.raises(ValueError, match="peak intensity"):
        fit_od_spectrum(data, rate, free=("probe_power_uw", "probe_waist_mm"), model_id=RATE)


def test_probe_power_is_recovered_and_reported_as_peak_intensity(monkeypatch):
    # A shorter deterministic search keeps the rate-model test quick.
    monkeypatch.setattr(od_spectrum_fit, "GLOBAL_LOG2_SAMPLES", 2)
    monkeypatch.setattr(od_spectrum_fit, "LOCAL_STARTS", 1)
    x = np.linspace(-3500.0, 5000.0, 160)
    y = od_transmission(x, _rate_params(), model_id=RATE)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    start = _rate_params(probe_power_uw=30.0)
    start["probe_power_mw"] = 0.03          # the legacy key must not pin the free power
    fit = fit_od_spectrum(data, start, free=("probe_power_uw",), model_id=RATE)
    assert fit["parameters"][0]["value"] == pytest.approx(50.0, rel=1e-5)
    assert fit["candidate_parameters"]["keys"] == ["probe_power_uw"]
    assert fit["candidate_parameters"]["scope"]["probe_power_uw"] == VALIDATED_SCOPE["probe_power_uw"]
    intensity = fit["probe_peak_intensity"]
    assert intensity["value_mw_cm2"] == pytest.approx(probe_peak_intensity_mw_cm2(50.0, 1.0), rel=1e-5)
    json.dumps(fit, allow_nan=False)


def test_custom_isotope_fraction_is_recovered_with_the_weak_probe_model():
    p = dict(SASScheme().defaults(), species="Rb (natural)", rb_composition="Custom",
             rb85_fraction=0.6, line="D2", pump_power_mw=0.0, probe_power_uw=0.0,
             temp_c=45.0, cell_mm=75.0)
    x = np.linspace(-3500.0, 5000.0, 301)
    y = od_transmission(x, p, offset_mhz=6.0)
    data = ODFitInput(x, y, np.full_like(x, 0.001), 0, x.size, "known test uncertainty")
    fit = fit_od_spectrum(data, dict(p, rb85_fraction=0.72, temp_c=43.0),
                          free=("temp_c", "offset_mhz", "rb85_fraction"))
    assert fit["identifiable"]
    assert [q["value"] for q in fit["parameters"]] == pytest.approx([45.0, 6.0, 0.6], abs=1e-5)
    assert "probe_peak_intensity" not in fit


def test_app_offers_the_fraction_and_d2_power_only():
    """User decision 2026-11-23: expose the isotope fraction and D2 probe power."""
    base = dict(SASScheme().defaults(), species="Rb (natural)", pump_power_mw=0.0)
    assert offered_fit_parameters(dict(base, probe_power_uw=0.0)) == PARAMETERS
    custom = offered_fit_parameters(dict(base, rb_composition="Custom", probe_power_uw=0.0))
    assert list(custom) == [*PARAMETERS, "rb85_fraction"]
    d2 = offered_fit_parameters(dict(base, line="D2", probe_power_uw=50.0))
    assert list(d2) == [*PARAMETERS, "probe_power_uw"]
    assert d2["probe_power_uw"] == CANDIDATE_PARAMETERS["probe_power_uw"]
    assert "probe_power_uw" not in offered_fit_parameters(dict(base, line="D1", probe_power_uw=50.0))
    assert "probe_power_uw" not in offered_fit_parameters(dict(base, line="D2", probe_power_uw=0.0))
    assert offered_fit_parameters(dict(base, line="D2", probe_power_uw=50.0, od_model="pre_d20")) == PARAMETERS
    cs = dict(base, species=species.CS133.label, line="D2", probe_power_uw=50.0)
    assert offered_fit_parameters(cs) == PARAMETERS
    assert all("probe_waist_mm" not in offered for offered in (custom, d2))
