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
    assert "Finite probe power" in od_unavailable(finite)
    with pytest.raises(ODUnsupportedError, match="Finite probe power"):
        evaluate_od(finite, OFFSETS)
    reserved = replace(ODConditions.from_params(sas_params()), model_id="od_weak_exact_voigt_transit_v1")
    with pytest.raises(ODUnsupportedError, match="reserved"):
        evaluate_od(reserved, OFFSETS)
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
