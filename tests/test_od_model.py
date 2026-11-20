"""M0 contract for the pump-free OD evaluator (gabes.od_model, decision D20)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gabes import observables, species
from gabes.od_model import (
    MODEL_KEY, TWO_PI, ODConditions, ODUnsupportedError, display_reference_offset_hz,
    evaluate_od, od_fit_conditions, od_unavailable,
)
from gabes.od_spectrum_fit import od_transmission
from gabes.optical_frequency import reference_offset_hz
from gabes.schemes.sas import (
    EXACT_WEAK_PROBE, LEGACY_WEAK_PROBE, RB_COMPOSITION_CUSTOM, RB_COMPOSITION_NATURAL, SASScheme,
)

FIXTURE = Path(__file__).parent / "fixtures" / "autood_baseline_0_2_2" / "spectra.npz"
FIXTURE_SHA256 = "25393b1006c3e71243b8b9151145c0dba7a8d00d8d49267de90d2ac05441a3d1"
OFFSETS = np.linspace(-4.5e9, 5.5e9, 401)


def sas_params(**changes):
    return {**SASScheme().defaults(), "species": "Rb (natural)", "line": "D2", "temp_c": 60.0,
            "cell_mm": 75.0, "probe_power_uw": 0.0, **changes}


def direct_transmission(conditions, offsets):
    params = conditions.to_params()
    raw = SASScheme()._compute_species(
        params, scan_rad_s=(offsets - display_reference_offset_hz(conditions)) * TWO_PI,
        extra_homogeneous_fwhm_hz=conditions.extra_homogeneous_fwhm_hz)
    return observables.transmission(raw["alpha_unit"] * conditions.line_strength,
                                    conditions.cell_mm * 1e-3)


def test_fit_conditions_reset_only_pump_controls_without_mutating_input():
    theory = sas_params(pump_power_mw=1.5, waist_mm=1.4, axial_pump_depletion=True)
    frozen = dict(theory)
    derived = od_fit_conditions(theory)
    assert theory == frozen
    assert (derived["pump_power_mw"], derived["waist_mm"], derived["axial_pump_depletion"]) == (0.0, 1.0, False)
    assert {k: v for k, v in derived.items() if k not in ("pump_power_mw", "waist_mm", "axial_pump_depletion")} \
        == {k: v for k, v in theory.items() if k not in ("pump_power_mw", "waist_mm", "axial_pump_depletion")}
    other = sas_params(pump_power_mw=0.2, waist_mm=0.3, axial_pump_depletion=False)
    assert od_fit_conditions(other) == derived
    assert ODConditions.from_params(theory) == ODConditions.from_params(other)


@pytest.mark.parametrize(("changes", "model_id"), [
    ({}, "weak_probe_exact_maxwell_voigt"),
    ({"weak_probe_profile": LEGACY_WEAK_PROBE}, "weak_probe_legacy"),
    ({"ne_pressure_torr": 10.0}, "weak_probe_legacy"),
    ({"probe_power_uw": 100.0}, "finite_power_2level_gaussian_propagation"),
    ({MODEL_KEY: "weak_probe_legacy"}, "weak_probe_legacy"),
])
def test_model_id_is_explicit_and_replays_todays_scheme_choice(changes, model_id):
    conditions = ODConditions.from_params(sas_params(pump_power_mw=1.5, **changes))
    assert conditions.model_id == model_id
    assert conditions.to_params()[MODEL_KEY] == model_id


def test_unsupported_inputs_give_reasons():
    with pytest.raises(ODUnsupportedError, match="Uniform"):
        ODConditions.from_params(sas_params(cell_propagation="Segmented reference"))
    finite = ODConditions.from_params(sas_params(probe_power_uw=100.0))
    assert finite.model_id == "finite_power_2level_gaussian_propagation"
    assert "od_finite_rb_mf_rate_v1" in od_unavailable(finite)
    with pytest.raises(ODUnsupportedError, match="uniform scan axis"):
        evaluate_od(finite, OFFSETS)
    for weak in ("weak_probe_exact_maxwell_voigt", "od_weak_exact_voigt_transit_v1"):
        with pytest.raises(ODUnsupportedError, match="probe power to zero"):
            evaluate_od(replace(finite, model_id=weak), OFFSETS)
    cesium = ODConditions.from_params(sas_params(species="¹³³Cs", probe_power_uw=10.0))
    with pytest.raises(ODUnsupportedError, match="rubidium"):
        evaluate_od(replace(cesium, model_id="od_finite_rb_mf_rate_v1"), OFFSETS)
    exact_with_buffer = replace(ODConditions.from_params(sas_params(ne_pressure_torr=10.0)),
                                model_id="weak_probe_exact_maxwell_voigt")
    assert "does not apply" in od_unavailable(exact_with_buffer)


@pytest.mark.parametrize("changes", [
    {},
    {"weak_probe_profile": LEGACY_WEAK_PROBE, "line": "D1"},
    {"line_strength": 0.5},
    {"constrain_doppler_temp": False, "doppler_temp_c": 80.0},
    {"species": "⁸⁵Rb", "temp_c": 90.0},
    {"species": "¹³³Cs", "line": "D1"},
    {"rb_composition": RB_COMPOSITION_CUSTOM, "rb85_fraction": 0.4},
])
def test_weak_evaluation_is_the_scheme_solve_bit_for_bit(changes):
    conditions = ODConditions.from_params(sas_params(**changes), extra_homogeneous_fwhm_hz=3e6)
    result = evaluate_od(conditions, OFFSETS)
    np.testing.assert_array_equal(result.transmission, direct_transmission(conditions, OFFSETS))
    np.testing.assert_array_equal(np.exp(-result.optical_depth), result.transmission)
    assert not result.transmission.flags.writeable
    fit = od_transmission(OFFSETS / 1e6, od_fit_conditions(sas_params(**changes)), extra_width_mhz=3.0)
    np.testing.assert_allclose(result.transmission, fit, rtol=1e-12, atol=1e-15)


def test_nonunit_line_strength_matches_the_plotted_scheme_spectrum():
    params = sas_params(line_strength=0.5, pump_power_mw=0.0)
    raw = SASScheme().compute(params)
    plotted = observables.transmission(raw["alpha_unit"] * 0.5, params["cell_mm"] * 1e-3)
    result = evaluate_od(ODConditions.from_params(params), raw["scan"] / TWO_PI)
    np.testing.assert_allclose(result.transmission, plotted, rtol=1e-12, atol=1e-15)
    unit = evaluate_od(ODConditions.from_params(dict(params, line_strength=1.0)), raw["scan"] / TWO_PI)
    assert np.max(np.abs(unit.transmission - result.transmission)) > 0.1


@pytest.mark.parametrize("profile", [EXACT_WEAK_PROBE, LEGACY_WEAK_PROBE])
def test_each_value_depends_only_on_its_own_frequency(profile):
    conditions = ODConditions.from_params(sas_params(weak_probe_profile=profile))
    grid = np.linspace(-4.5e9, 5.5e9, 2001)
    full = evaluate_od(conditions, grid).transmission
    rng = np.random.default_rng(7)
    picks = (np.arange(0, grid.size, 5), np.r_[np.arange(700), np.arange(1300, grid.size)],
             rng.choice(grid.size, 600, replace=True), np.arange(grid.size)[::-1])
    for pick in picks:
        np.testing.assert_array_equal(evaluate_od(conditions, grid[pick]).transmission, full[pick])


@pytest.mark.parametrize("line", ["D1", "D2"])
def test_pure_rb87_and_custom_zero_share_agree_at_the_same_frequency(line):
    pure = ODConditions.from_params(sas_params(species="⁸⁷Rb", line=line))
    custom = ODConditions.from_params(sas_params(line=line, rb_composition=RB_COMPOSITION_CUSTOM,
                                                 rb85_fraction=0.0))
    shift = display_reference_offset_hz(pure)
    assert shift == reference_offset_hz("Rb87", line)
    assert abs(shift / 1e6 - {"D1": 77.69, "D2": 78.095}[line]) < 1e-3
    assert display_reference_offset_hz(custom) == 0.0
    assert pure.to_params()["species"] == species.RB87.label
    np.testing.assert_allclose(evaluate_od(pure, OFFSETS).transmission,
                               evaluate_od(custom, OFFSETS).transmission, rtol=1e-12, atol=1e-14)


def test_autood_fixture_bytes_are_frozen():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    meta = json.loads(str(np.load(FIXTURE)["metadata"]))
    assert meta["gabes_pin"] == "8846193ffcef99f831452ac516db1c500daeafc4"
    assert meta["adapter_sha256"] == "7063e3471c71bcaeaffc33d6f5e6d960f65ef9728f93f83ecbf0db89cfd84f13"


def autood_conditions(settings, entry, model_id):
    percent = settings.get("rb85_percent", 72.17)
    return ODConditions(
        species="Rb (natural)", line=settings["line"], temp_c=settings["temperature_c"],
        cell_mm=settings["cell_mm"], model_id=model_id,
        rb_composition=RB_COMPOSITION_NATURAL if percent == 72.17 else RB_COMPOSITION_CUSTOM,
        rb85_fraction=percent / 100.0 if percent != 72.17 else species.RB85.abundance,
        probe_power_uw=entry["probe_mw"] * 1000.0, probe_waist_mm=settings["diameter_mm"] / 2.0,
        transit_khz=entry["transit_khz"],
    )


def test_ported_rate_model_reproduces_every_autood_spectrum():
    """Step 1: AutoOD's algorithm in GABES. Bit-identical with Numba off;
    GABES's Numba kernels move transmission by <1e-13 here."""
    from gabes.od_model import _rate_model_on_axis, _rate_model_scan

    data = np.load(FIXTURE)
    meta = json.loads(str(data["metadata"]))
    for name, case in meta["cases"].items():
        for entry in case["spectra"]:
            conditions = autood_conditions(case["settings"], entry, "od_finite_rb_mf_rate_v1")
            scan = _rate_model_scan(conditions)
            np.testing.assert_array_equal(scan, data[f"{name}__scan_rad_s"])
            transmission, depth, od0 = _rate_model_on_axis(conditions, scan)
            key = entry["key"]
            np.testing.assert_allclose(transmission, data[f"{key}__transmission"], rtol=0, atol=1e-12,
                                       err_msg=key)
            np.testing.assert_allclose(depth, data[f"{key}__optical_depth"], rtol=1e-12, atol=1e-14,
                                       err_msg=key)
            np.testing.assert_array_equal(od0, data[f"{key}__od0"])


def test_rate_model_rejects_conditions_outside_its_scope():
    from gabes.od_model import _rate_model_scan

    base = ODConditions.from_params(sas_params(probe_power_uw=100.0))
    for changes in ({"doppler_temp_c": 80.0}, {"ne_pressure_torr": 5.0}, {"species": "¹³³Cs"}):
        with pytest.raises(ODUnsupportedError):
            _rate_model_scan(replace(base, **changes))


def test_autood_weak_probe_equals_exact_voigt_plus_twice_the_transit_rate():
    """AutoOD adds the transit rate to the optical HWHM; GABES Exact adds none (U1)."""
    data = np.load(FIXTURE)
    meta = json.loads(str(data["metadata"]))
    for name, case in meta["cases"].items():
        settings = case["settings"]
        percent = settings.get("rb85_percent", 72.17)
        weak = next(entry for entry in case["spectra"] if entry["probe_mw"] == 0.0)
        assert weak["model"] == "weak_probe_exact_maxwell_voigt"
        conditions = ODConditions(
            species="Rb (natural)", line=settings["line"], temp_c=settings["temperature_c"],
            cell_mm=settings["cell_mm"], model_id="weak_probe_exact_maxwell_voigt",
            rb_composition=RB_COMPOSITION_NATURAL if percent == 72.17 else RB_COMPOSITION_CUSTOM,
            rb85_fraction=percent / 100.0 if percent != 72.17 else species.RB85.abundance,
            probe_waist_mm=settings["diameter_mm"] / 2.0, transit_khz=weak["transit_khz"],
            extra_homogeneous_fwhm_hz=2.0 * weak["transit_khz"] * 1e3,
        )
        result = evaluate_od(conditions, data[f"{name}__scan_rad_s"] / TWO_PI)
        expected = data[f"{weak['key']}__transmission"]
        np.testing.assert_allclose(result.transmission, expected, rtol=0, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(result.optical_depth, data[f"{weak['key']}__od0"], rtol=1e-11,
                                   err_msg=name)
        without = evaluate_od(replace(conditions, extra_homogeneous_fwhm_hz=0.0),
                              data[f"{name}__scan_rad_s"] / TWO_PI)
        assert np.max(np.abs(without.transmission - expected)) > 1e-6


RATE = "od_finite_rb_mf_rate_v1"
TRANSIT = "od_weak_exact_voigt_transit_v1"


def rate_conditions(**changes):
    params = sas_params(temp_c=25.0, probe_power_uw=100.0, **changes)
    return replace(ODConditions.from_params(params), model_id=RATE)


def test_transit_model_is_the_exact_solve_with_twice_the_transit_rate():
    conditions = replace(ODConditions.from_params(sas_params(line_strength=0.7)), model_id=TRANSIT)
    result = evaluate_od(conditions, OFFSETS)
    widened = replace(conditions, model_id="weak_probe_exact_maxwell_voigt",
                      extra_homogeneous_fwhm_hz=2 * conditions.transit_khz * 1e3)
    np.testing.assert_array_equal(result.transmission, evaluate_od(widened, OFFSETS).transmission)
    assert result.model_id == TRANSIT
    buffered = replace(ODConditions.from_params(sas_params(ne_pressure_torr=5.0)), model_id=TRANSIT)
    assert "does not apply" in od_unavailable(buffered)


def test_transit_and_rate_models_reproduce_autood_through_evaluate_od():
    data = np.load(FIXTURE)
    meta = json.loads(str(data["metadata"]))
    for name, case in meta["cases"].items():
        offsets = data[f"{name}__scan_rad_s"] / TWO_PI
        for entry in case["spectra"]:
            key = entry["key"]
            models = (TRANSIT, RATE) if entry["probe_mw"] == 0.0 else (RATE,)
            for model_id in models:
                result = evaluate_od(autood_conditions(case["settings"], entry, model_id), offsets)
                np.testing.assert_allclose(result.transmission, data[f"{key}__transmission"],
                                           rtol=0, atol=1e-12, err_msg=f"{key} {model_id}")
                np.testing.assert_allclose(result.od0, data[f"{key}__od0"], rtol=1e-11,
                                           err_msg=f"{key} {model_id}")


def test_rate_model_values_do_not_depend_on_the_requested_axis():
    from gabes.od_model import _rate_model_scan

    conditions = rate_conditions()
    offsets = _rate_model_scan(conditions) / TWO_PI
    full = evaluate_od(conditions, offsets)
    rng = np.random.default_rng(11)
    for pick in (np.arange(0, offsets.size, 5), rng.choice(offsets.size, 700, replace=True),
                 np.arange(offsets.size)[::-1], np.r_[np.arange(700), np.arange(1300, offsets.size)]):
        part = evaluate_od(conditions, offsets[pick])
        for field_name in ("transmission", "optical_depth", "od0"):
            np.testing.assert_array_equal(getattr(part, field_name), getattr(full, field_name)[pick])


def test_rate_model_reuses_its_table_and_is_unsaturated_far_outside_it():
    from gabes.od_model import _rate_model_spline

    _rate_model_spline.cache_clear()
    conditions = rate_conditions()
    evaluate_od(conditions, OFFSETS)
    evaluate_od(replace(conditions, cell_mm=10.0, line_strength=0.5), OFFSETS[::3] + 1e6)
    info = _rate_model_spline.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    far = np.array([-40e9, 40e9])
    weak = evaluate_od(replace(conditions, model_id=TRANSIT, probe_power_uw=0.0), far)
    np.testing.assert_allclose(evaluate_od(conditions, far).transmission, weak.transmission,
                               rtol=1e-12)


def test_rate_model_zero_power_limit_is_continuous_and_linear():
    """T(P) - T(0) is linear in P: transit-limited pumping saturates at nW powers."""
    from gabes.od_model import _rate_model_scan

    base = rate_conditions(cell_mm=75.0)
    offsets = _rate_model_scan(base) / TWO_PI
    zero = evaluate_od(replace(base, probe_power_uw=0.0), offsets).transmission
    weak = evaluate_od(replace(base, model_id=TRANSIT, probe_power_uw=0.0), offsets).transmission
    np.testing.assert_allclose(zero, weak, rtol=1e-13, atol=1e-15)
    slopes = []
    for power_uw in (1e-3, 1e-4, 1e-5):
        change = evaluate_od(replace(base, probe_power_uw=power_uw), offsets).transmission - zero
        slopes.append(np.max(np.abs(change)) / power_uw)
    assert slopes[0] > 0
    assert abs(slopes[1] - slopes[2]) < abs(slopes[0] - slopes[1]) + 1e-12 * slopes[0]
    assert abs(slopes[2] / slopes[1] - 1) < 1e-3


def test_display_axis_matches_the_plot_and_the_fit_for_pure_rb87():
    from gabes.od_model import evaluate_od_display

    params = sas_params(species="⁸⁷Rb", pump_power_mw=0.0, line_strength=0.8)
    raw = SASScheme().compute(params)
    plotted = observables.transmission(raw["alpha_unit"] * 0.8, params["cell_mm"] * 1e-3)
    conditions = ODConditions.from_params(params)
    shown = evaluate_od_display(conditions, raw["scan"])
    np.testing.assert_array_equal(shown.transmission, plotted)
    shift = display_reference_offset_hz(conditions)
    np.testing.assert_allclose(shown.offsets_hz, raw["scan"] / TWO_PI + shift, rtol=1e-15)
    mhz = raw["scan"][::50] / (TWO_PI * 1e6)
    fit_curve = od_transmission(mhz, params)
    np.testing.assert_allclose(fit_curve, evaluate_od(conditions, mhz * 1e6 + shift).transmission,
                               rtol=1e-12, atol=1e-15)
