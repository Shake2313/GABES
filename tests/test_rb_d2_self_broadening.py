"""Measured Rb D2 self-broadening in the d20_d2sb model set (user decision 2026-11-24).

Checklist item rb-d2-self-broadening-default-value: D2 takes the measured
coefficient (Kondo 2006) through new _v2 model IDs; every older set and ID
keeps the D1 value on D2, and D1 does not move.
"""
import math

import numpy as np
import pytest

from gabes import collisions, species
from gabes.od_model import MODELS, ODConditions, evaluate_od_display, is_rate_model, rate_model_axis
from gabes.schemes.sas import (
    LEGACY_WEAK_PROBE, OD_MODEL_KEY, OD_MODEL_SETS, OD_MODELS_D2SB, OD_MODELS_D20,
    PUMPED_TRANSIT_MODEL_V2, RATE_MODEL_V2, TRANSIT_WEAK_MODEL_V2, TWO_LEVEL_MODEL_V2, SASScheme,
    _od_model_setting, _pump_free_model_id,
)

DENSITY = 1.0e20


def test_d2_takes_the_measured_coefficient_and_d1_keeps_beta_self():
    for isotope in (species.RB85, species.RB87):
        d2 = species.self_broadening(isotope, DENSITY, "D2", per_line=True)
        reference, _ = collisions.self_broadening_reference(isotope.name, "D2", DENSITY)
        assert d2 == pytest.approx(reference, rel=1e-15)
        assert d2 == pytest.approx(2 * math.pi * 1.10e-13 * DENSITY, rel=1e-12)
        assert species.self_broadening(isotope, DENSITY, "D1", per_line=True) == species.BETA_SELF * DENSITY
        # Without the flag every Rb line keeps the D1 value (older model IDs).
        assert species.self_broadening(isotope, DENSITY) == species.BETA_SELF * DENSITY
        assert species.self_broadening(isotope, DENSITY, "D2") == species.BETA_SELF * DENSITY
    assert species.self_broadening(species.CS133, DENSITY, "D2", per_line=True) == 0.0
    with pytest.raises(ValueError):
        species.self_broadening(species.RB85, DENSITY, per_line=True)


def _sas_params(**changes):
    return {**SASScheme().defaults(), "species": "Rb (natural)", "line": "D2", "temp_c": 60.0,
            "scan_points": 201, "pump_power_mw": 0.0, OD_MODEL_KEY: OD_MODELS_D2SB, **changes}


@pytest.mark.parametrize(("changes", "model"), [
    ({}, TRANSIT_WEAK_MODEL_V2),
    ({"probe_power_uw": 50.0}, RATE_MODEL_V2),
    ({"probe_power_uw": 50.0, "ne_pressure_torr": 5.0}, TWO_LEVEL_MODEL_V2),
])
def test_the_new_set_records_v2_model_ids(changes, model):
    params = _sas_params(**changes)
    assert _pump_free_model_id(params) == model
    assert MODELS[model].per_line_self_broadening
    assert is_rate_model(model) == (model == RATE_MODEL_V2)


def test_pumped_and_legacy_requests():
    raw = SASScheme().compute(_sas_params(pump_power_mw=0.5))
    assert raw["probe_model"] == PUMPED_TRANSIT_MODEL_V2
    # The Legacy grid replays the historical spectrum: the set falls back to d20.
    legacy = _sas_params(weak_probe_profile=LEGACY_WEAK_PROBE)
    assert _od_model_setting(legacy) == (OD_MODELS_D20, None)
    old = SASScheme().compute({**legacy, OD_MODEL_KEY: OD_MODELS_D20})
    new = SASScheme().compute(legacy)
    assert new["probe_model"] == old["probe_model"] == "weak_probe_legacy"
    assert np.array_equal(new["alpha_unit"], old["alpha_unit"])


@pytest.mark.parametrize("line", ["D1", "D2"])
def test_only_d2_moves_and_an_explicit_id_replays_the_set(line):
    scheme = SASScheme()
    new = scheme.compute(_sas_params(line=line))
    old = scheme.compute(_sas_params(line=line, **{OD_MODEL_KEY: OD_MODELS_D20}))
    explicit = scheme.compute(_sas_params(line=line, **{OD_MODEL_KEY: TRANSIT_WEAK_MODEL_V2}))
    assert np.array_equal(explicit["alpha_unit"], new["alpha_unit"])
    if line == "D1":
        assert np.array_equal(new["alpha_unit"], old["alpha_unit"])
    else:
        assert not np.allclose(new["alpha_unit"], old["alpha_unit"], rtol=1e-6, atol=0)
        # More homogeneous width puts more absorption in the far wings.
        assert new["alpha_unit"][0] > old["alpha_unit"][0]


def test_pump_free_evaluator_uses_the_model_flag():
    base = {"species": "Rb (natural)", "line": "D2", "temp_c": 150.0, "cell_mm": 12.5}
    old = ODConditions.from_params({**base, OD_MODEL_KEY: OD_MODELS_D20})
    new = ODConditions.from_params({**base, OD_MODEL_KEY: OD_MODELS_D2SB})
    assert (old.model_id, new.model_id) == ("od_weak_exact_voigt_transit_v1", TRANSIT_WEAK_MODEL_V2)
    axis = rate_model_axis(old, points=401)
    assert np.array_equal(axis, rate_model_axis(new, points=401))
    t_old = evaluate_od_display(old, axis).transmission
    t_new = evaluate_od_display(new, axis).transmission
    # 150 °C, 12.5 mm: the wings carry a change of several percentage points.
    assert 0.03 < float(np.max(np.abs(t_new - t_old))) < 0.15


def test_sets_are_listed_for_the_ui():
    assert OD_MODEL_SETS == ("pre_d20", "d20", "d20_d2sb")
