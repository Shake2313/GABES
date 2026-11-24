"""SAS model selection via the hidden od_model setting (decision D20, M2a)."""
from __future__ import annotations

import numpy as np
import pytest

from gabes import observables, species
from gabes.od_model import TWO_PI, ODConditions, evaluate_od_display
from gabes.schemes.sas import (
    OD_MODEL_KEY, OD_MODELS_D2SB, OD_MODELS_D20, OD_MODELS_PRE_D20, ODModelSettingError, PUMPED_MODEL,
    PUMPED_TRANSIT_MODEL, RATE_MODEL, RATE_MODEL_V2, TRANSIT_WEAK_MODEL, TRANSIT_WEAK_MODEL_V2, SASScheme,
    _collected_response,
)


def params(**changes):
    """Pre-D20 baseline unless a test sets od_model."""
    return {**SASScheme().defaults(), "species": "Rb (natural)", "line": "D2", "temp_c": 50.0,
            "cell_mm": 50.0, "pump_power_mw": 0.0, "probe_power_uw": 0.0,
            OD_MODEL_KEY: OD_MODELS_PRE_D20, **changes}


def plotted(raw, p):
    alpha, _chi, length = _collected_response(raw, p)
    return observables.transmission(alpha, length)


def test_d20_d2sb_is_the_default_and_old_settings_replay():
    scheme = SASScheme()
    # User decision 2026-11-24: d20_d2sb replaced d20 as the default; d20 replays.
    assert scheme.defaults()[OD_MODEL_KEY] == OD_MODELS_D2SB
    assert scheme.migrate_default_params({})[OD_MODEL_KEY] == OD_MODELS_PRE_D20
    assert scheme.migrate_default_params({OD_MODEL_KEY: OD_MODELS_D20})[OD_MODEL_KEY] == OD_MODELS_D20
    bare = {k: v for k, v in params().items() if k != OD_MODEL_KEY}
    assert OD_MODEL_KEY not in scheme.solve_param_keys(bare)
    assert OD_MODEL_KEY in scheme.solve_param_keys(params())
    for p in (params(), params(pump_power_mw=0.6), params(probe_power_uw=40.0)):
        absent = {k: v for k, v in p.items() if k != OD_MODEL_KEY}
        np.testing.assert_array_equal(scheme.compute(p)["alpha_unit"], scheme.compute(absent)["alpha_unit"])
        # None is an unset session value: the current default, like a fresh session.
        assert (scheme.compute(dict(p, **{OD_MODEL_KEY: None}))["probe_model"]
                == scheme.compute(dict(p, **{OD_MODEL_KEY: OD_MODELS_D2SB}))["probe_model"])


def test_d20_weak_probe_is_the_transit_voigt_on_both_views():
    p = params(**{OD_MODEL_KEY: OD_MODELS_D20}, line_strength=0.7)
    raw = SASScheme().compute(p)
    assert raw["probe_model"] == TRANSIT_WEAK_MODEL and raw["od_model_set"] == OD_MODELS_D20
    conditions = ODConditions.from_params(p)
    assert conditions.model_id == TRANSIT_WEAK_MODEL
    np.testing.assert_array_equal(evaluate_od_display(conditions, raw["scan"]).transmission, plotted(raw, p))
    pre = SASScheme().compute(params(line_strength=0.7))
    assert np.max(np.abs(plotted(pre, p) - plotted(raw, p))) > 1e-6


def test_d20_finite_probe_is_the_rate_model_on_both_views():
    p = params(**{OD_MODEL_KEY: OD_MODELS_D20}, probe_power_uw=100.0, line_strength=0.5)
    raw = SASScheme().compute(p)
    assert raw["probe_model"] == RATE_MODEL and raw["probe_radial_order"] == 96
    evaluation = evaluate_od_display(ODConditions.from_params(p), raw["scan"])
    alpha, chi, length = _collected_response(raw, p)
    np.testing.assert_array_equal(alpha * length, evaluation.optical_depth)
    np.testing.assert_array_equal(raw["alpha_unit"] * 0.5 * length, evaluation.od0)
    assert chi is None
    view = SASScheme().observables(raw, p, include_figures=False)
    np.testing.assert_array_equal(view["optical_readout"]["od0"], evaluation.od0)
    np.testing.assert_array_equal(view["optical_readout"]["optical_depth"], evaluation.optical_depth)
    assert any(m.get("label") == "Probe model" for m in view["metrics"])


def test_d20_finite_probe_outside_the_rate_scope_keeps_the_two_level_overlay():
    p = params(**{OD_MODEL_KEY: OD_MODELS_D20}, probe_power_uw=50.0, ne_pressure_torr=5.0)
    assert SASScheme().compute(p)["probe_model"] == "finite_power_2level_gaussian_propagation"


def test_d20_pumped_exact_profile_carries_the_transit_width_continuously():
    pumped = SASScheme().compute(params(**{OD_MODEL_KEY: OD_MODELS_D20}, pump_power_mw=0.8))
    assert pumped["probe_model"] == PUMPED_TRANSIT_MODEL
    assert SASScheme().compute(params(pump_power_mw=0.8))["probe_model"] == PUMPED_MODEL
    # U3: at a vanishing pump the pumped profile meets the U1 weak-probe curve.
    tiny = params(**{OD_MODEL_KEY: OD_MODELS_D20}, pump_power_mw=1e-9)
    weak = params(**{OD_MODEL_KEY: OD_MODELS_D20})
    raw_tiny, raw_weak = SASScheme().compute(tiny), SASScheme().compute(weak)
    np.testing.assert_allclose(raw_tiny["alpha_unit"], raw_weak["alpha_unit"], rtol=1e-6,
                               atol=1e-9 * np.max(raw_weak["alpha_unit"]))
    pre_tiny = SASScheme().compute(params(pump_power_mw=1e-9))
    assert np.max(np.abs(pre_tiny["alpha_unit"] - raw_tiny["alpha_unit"])) > 1e-6 * np.max(raw_weak["alpha_unit"])


def test_explicit_model_ids_fix_the_pump_off_solve_and_pick_a_set_for_the_rest():
    explicit = params(**{OD_MODEL_KEY: RATE_MODEL}, probe_power_uw=30.0)
    assert SASScheme().compute(explicit)["probe_model"] == RATE_MODEL
    assert SASScheme().compute(dict(explicit, pump_power_mw=0.8, probe_power_uw=0.0))["probe_model"] \
        == PUMPED_TRANSIT_MODEL
    legacy = params(**{OD_MODEL_KEY: "weak_probe_legacy"})
    assert SASScheme().compute(legacy)["probe_model"] == "weak_probe_legacy"
    with pytest.raises(ValueError, match="Exact Maxwell Voigt models"):
        SASScheme().compute(params(**{OD_MODEL_KEY: TRANSIT_WEAK_MODEL}, ne_pressure_torr=5.0))
    with pytest.raises(ValueError, match="zero probe power"):
        SASScheme().compute(params(**{OD_MODEL_KEY: TRANSIT_WEAK_MODEL}, probe_power_uw=5.0))


@pytest.mark.parametrize("value", ["unknown", "", "   ", 0, [], {}])
def test_invalid_settings_raise_a_visible_error(value):
    with pytest.raises(ODModelSettingError, match="Invalid OD model ID"):
        SASScheme().compute(params(**{OD_MODEL_KEY: value}))


def test_pure_rb87_display_axis_under_d20():
    p = params(**{OD_MODEL_KEY: OD_MODELS_D20}, species=species.RB87.label, probe_power_uw=60.0)
    raw = SASScheme().compute(p)
    evaluation = evaluate_od_display(ODConditions.from_params(p), raw["scan"])
    alpha, _chi, length = _collected_response(raw, p)
    np.testing.assert_array_equal(alpha * length, evaluation.optical_depth)
    np.testing.assert_allclose(evaluation.offsets_hz, raw["scan"] / TWO_PI + 78.095e6, rtol=1e-15)


@pytest.mark.parametrize("probe_uw", [0.0, 100.0])
def test_scheme_and_evaluator_resolve_the_same_model_from_one_dictionary(probe_uw):
    """Missing key: pre_d20 in both. None: the current default. An ID: that model."""
    base = {k: v for k, v in params(probe_power_uw=probe_uw).items() if k != OD_MODEL_KEY}
    explicit = TRANSIT_WEAK_MODEL if probe_uw == 0.0 else RATE_MODEL
    for setting in ("missing", None, OD_MODELS_D2SB, OD_MODELS_D20, OD_MODELS_PRE_D20, explicit):
        p = dict(base) if setting == "missing" else dict(base, **{OD_MODEL_KEY: setting})
        scheme_model = SASScheme().compute(p)["probe_model"]
        assert ODConditions.from_params(p).model_id == scheme_model, setting
    assert SASScheme().compute(base)["probe_model"] == (
        "weak_probe_exact_maxwell_voigt" if probe_uw == 0.0 else "finite_power_2level_gaussian_propagation")
    assert ODConditions.from_params(dict(base, **{OD_MODEL_KEY: None})).model_id == (
        TRANSIT_WEAK_MODEL_V2 if probe_uw == 0.0 else RATE_MODEL_V2)


def test_a_valid_model_outside_its_scope_is_a_displayed_input_error():
    with pytest.raises(ODModelSettingError, match="needs zero probe power"):
        SASScheme().compute(params(**{OD_MODEL_KEY: TRANSIT_WEAK_MODEL}, probe_power_uw=5.0))
    with pytest.raises(ODModelSettingError, match="Exact Maxwell Voigt models"):
        SASScheme().compute(params(**{OD_MODEL_KEY: TRANSIT_WEAK_MODEL}, ne_pressure_torr=5.0))
    with pytest.raises(ODModelSettingError, match="rubidium"):
        SASScheme().compute(params(**{OD_MODEL_KEY: RATE_MODEL}, species=species.CS133.label,
                                   probe_power_uw=5.0))
