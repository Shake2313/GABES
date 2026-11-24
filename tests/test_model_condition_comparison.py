"""Condition comparison shares solver samples, units and resolution diagnostics."""
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gabes import schemes, species
from gabes.core import blas_single_thread
from gabes.model_comparison import comparison_values, sample_values, same_grid
from gabes.schemes.sas import (
    AUTO_WEAK_PROBE, EXACT_WEAK_PROBE, GENERIC, LEGACY_WEAK_PROBE,
    PUMPED_MAXWELL, RB_COMPOSITION_CUSTOM,
)


def _compare(**changes):
    scheme = schemes.get("sas")
    params = {**scheme.defaults(), **changes}
    solves = []

    def resolve(physical_params):
        solves.append(physical_params["pump_power_mw"])
        return scheme.compute({k: physical_params[k] for k in scheme.recompute_keys()})

    with blas_single_thread():
        current = resolve(params)
        view = scheme.headless_observables(current, params)
        solves.clear()
        before = plt.get_fignums()
        comparison = scheme.model_comparison(
            params, current_raw=current, current_view=view, resolve_raw=resolve)
        assert before == plt.get_fignums()
    json.dumps(comparison, allow_nan=False)
    return comparison, solves, view


def test_full_resolution_natural_rb_reuses_current_and_differences_at_every_point():
    model, solves, current_view = _compare(scan_points=4001, pump_power_mw=2.)
    assert solves == [0.]
    x = np.array(model["x"])
    np.testing.assert_array_equal(x, current_view["optical_readout"]["x"] * 1000.)
    assert len(x) == model["fixed"]["scan_points"] == 4001
    assert "pump_power_mw" not in model["fixed"]
    assert model["axis"]["reference"] == "Rb85 D1 line centroid"
    assert model["axis"]["overview_unit"] == "GHz"
    off = np.asarray(model["series"][0]["transmission"])
    for s in model["series"]:
        np.testing.assert_array_equal(s["delta"], np.asarray(s["transmission"]) - off)
    current = model["series"][-1]
    np.testing.assert_array_equal(current["transmission"],
                                  current_view["optical_readout"]["transmission"])
    np.testing.assert_array_equal(current["optical_depth"],
                                  current_view["optical_readout"]["optical_depth"])
    assert not current["effective"]
    # With no probe saturation OD0 and OD_eff coincide even with the pump on.
    for row in model["series"]:
        np.testing.assert_array_equal(row["od0"], row["optical_depth"])
    n = species.number_density(species.RB85, model["fixed"]["temp_c"] + 273.15)
    area = np.pi * (model["fixed"]["probe_waist_mm"] * 1e-3) ** 2 / 2
    assert model["n_eff"] == pytest.approx(n * model["fixed"]["cell_mm"] * 1e-3 * area)
    assert current["resolution"] == current_view["optical_readout"]["resolution"]
    assert model["series"][0]["resolution"]["width_mhz"] is None
    assert [g["label"] for g in model["groups"]] == [
        "⁸⁷Rb F=2", "⁸⁵Rb F=3", "⁸⁵Rb F=2", "⁸⁷Rb F=1"]
    for g in model["groups"]:
        lo, hi = g["window_mhz"]
        actual = [m["x_mhz"] for m in g["markers"] if m["kind"] == "transition"]
        candidate = [m["x_mhz"] for m in g["markers"] if m["kind"] == "crossover_candidate"]
        assert len(actual) == 2 and candidate == [sum(actual)/2]
        assert x[0] <= lo < min(actual) < max(actual) < hi <= x[-1]


def test_mixed_rb_composition_isotope_absence_and_native_reference_do_not_conflate():
    model, solves, _ = _compare(line="D2", rb_composition=RB_COMPOSITION_CUSTOM,
                                rb85_fraction=0., scan_points=401, pump_power_mw=.7,
                                temp_c=40., constrain_doppler_temp=False, doppler_temp_c=60.,
                                cell_mm=25., probe_waist_mm=.4, waist_mm=2., line_strength=.6)
    assert solves == [0.]
    density = species.number_density(species.RB87, 313.15) * (313.15 / 333.15)
    assert model["n_eff"] == pytest.approx(density * .025 * np.pi * .0004**2 / 2, rel=1e-13)
    assert model["default_power_mw"] == .7
    assert len(model["groups"]) == 2 and all("⁸⁷Rb" in g["label"] for g in model["groups"])
    assert model["axis"]["reference"].startswith("Rb85 D2")
    for group in model["groups"]:
        lines = [m["x_mhz"] for m in group["markers"] if m["kind"] == "transition"]
        mids = [m["x_mhz"] for m in group["markers"] if m["kind"] == "crossover_candidate"]
        assert len(lines) == 3
        assert mids == sorted((a+b)/2 for a,b in combinations(lines, 2))


def test_explicit_exact_mixed_comparison_uses_same_legacy_grid_for_both_curves():
    profile = EXACT_WEAK_PROBE
    conditions = dict(line="D2", temp_c=90., cell_mm=12.5, probe_power_uw=0.,
                      scan_points=401, pump_power_mw=.01)
    expected, _, _ = _compare(**conditions, weak_probe_profile=LEGACY_WEAK_PROBE)
    actual, solves, _ = _compare(**conditions, weak_probe_profile=profile)
    assert solves == [0.]
    assert actual["axis"] == expected["axis"] and actual["x"] == expected["x"]
    assert actual["fixed"]["weak_probe_profile"] == LEGACY_WEAK_PROBE
    for actual_curve, expected_curve in zip(actual["series"], expected["series"], strict=True):
        np.testing.assert_array_equal(actual_curve["transmission"], expected_curve["transmission"])
        np.testing.assert_array_equal(actual_curve["delta"], expected_curve["delta"])
        assert actual_curve["probe_model"] == "weak_probe_legacy"
        assert actual_curve["weak_probe_profile"]["requested"] == profile
        assert actual_curve["weak_probe_profile"]["applied"] == LEGACY_WEAK_PROBE

    # An explicit comparison can request both powers while the active curve is the exact pump-off one.
    conditions["pump_power_mw"] = 0.
    off_only, solves, current = _compare(**conditions, weak_probe_profile=profile)
    assert solves == []
    # Default set d20_d2sb: an Exact request at zero pump is the transit Voigt (_v2).
    assert off_only["series"][0]["probe_model"] == "od_weak_exact_voigt_transit_v2"
    assert off_only["series"][0]["weak_probe_profile"]["requested"] == profile
    np.testing.assert_array_equal(off_only["series"][0]["transmission"],
                                  current["optical_readout"]["transmission"])
    expected_delta = {row["power_mw"]: row["delta"] for row in expected["series"]}
    for powers in ([0., .01], [.01, 0.], [.01]):
        with comparison_values("pump_power_mw", powers):
            sweep, solves, _ = _compare(**conditions, weak_probe_profile=profile)
        assert solves == [0., .01]
        assert sweep["varied"]["values"] == powers
        assert [row["power_mw"] for row in sweep["series"]] == powers
        for row in sweep["series"]:
            np.testing.assert_array_equal(row["delta"], expected_delta[row["power_mw"]])


def test_automatic_pump_comparison_shares_exact_background_and_reuses_zero():
    base = dict(line="D2", temp_c=90., cell_mm=75., probe_power_uw=0.,
                scan_points=1201, pump_power_mw=0.)
    tiny = float(np.nextafter(0., 1.))
    powers = [tiny, 0.01, 0., 1e-10, .5]
    with comparison_values("pump_power_mw", powers):
        sweep, solves, view = _compare(**base)
    assert solves == [tiny, 0.01, 1e-10, .5]
    assert sweep["fixed"]["weak_probe_profile"] == AUTO_WEAK_PROBE
    assert sweep["source"]["weak_probe_profile"]["applied"] == PUMPED_MAXWELL
    rows = {row["power_mw"]: row for row in sweep["series"]}
    assert list(rows) == powers
    off = rows[0.]
    assert off["weak_probe_profile"]["applied"] == EXACT_WEAK_PROBE
    np.testing.assert_array_equal(off["transmission"], view["optical_readout"]["transmission"])
    assert rows[tiny]["weak_probe_profile"]["applied"] == EXACT_WEAK_PROBE
    np.testing.assert_array_equal(rows[tiny]["transmission"], off["transmission"])
    for pump in (0.01, 1e-10, .5):
        assert rows[pump]["weak_probe_profile"]["applied"] == PUMPED_MAXWELL
        np.testing.assert_array_equal(
            rows[pump]["delta"], np.asarray(rows[pump]["transmission"]) - off["transmission"])
    assert np.max(np.abs(rows[1e-10]["delta"])) < 1e-6


@pytest.mark.parametrize("atom,reference,count,unit", [
    ("¹³³Cs", "Cs133 D1 line centroid", 2, "GHz"),
    (GENERIC, "Generic model zero detuning", 0, "MHz"),
])
def test_other_manifolds_use_own_grid_and_no_invented_rb_groups(atom, reference, count, unit):
    model, solves, current = _compare(species=atom, scan_points=401,
                                      temp_c=65., pump_power_mw=0.)
    assert solves == []
    assert model["default_power_mw"] == 0
    assert [s["power_mw"] for s in model["series"]] == [0.]
    assert model["axis"]["overview_unit"] == unit
    assert model["axis"]["reference"] == reference
    assert len(model["groups"]) == count and model["fixed"]["temp_c"] == 65.
    if atom == GENERIC:
        assert model["n_eff"] is None
    else:
        assert model["n_eff"] > 0
    factor = 1000 if unit == "GHz" else 1
    np.testing.assert_array_equal(model["x"], current["optical_readout"]["x"] * factor)


def test_axis_mismatch_and_missing_values_cannot_be_subtracted_as_valid_samples():
    with pytest.raises(ValueError, match="same sampled axis"):
        same_grid([0, 1, 2], [0, 1.01, 2])
    assert sample_values([0, np.nan, 1.0, np.inf]) == [0., None, 1., None]


def test_native_depth_survives_underflow_and_carries_finite_probe_scope():
    opaque, _, _ = _compare(scan_points=401, temp_c=170., cell_mm=150.)
    c = opaque["series"][-1]
    deep = int(np.argmax(c["optical_depth"]))
    assert c["transmission"][deep] == 0
    assert c["od0"][deep] == c["optical_depth"][deep]
    assert np.isfinite(c["optical_depth"][deep]) and c["optical_depth"][deep] > 745.

    finite, solves, current = _compare(scan_points=401, line="D2", probe_power_uw=.1,
                                       pump_power_mw=.83)
    assert solves == [0.]
    assert finite["varied"]["values"] == [0., .83]
    assert all(c["effective"] for c in finite["series"])
    off, on = finite["series"]
    assert np.max(np.abs(np.subtract(off["od0"], off["optical_depth"]))) > 1e-6
    assert np.max(np.abs(np.subtract(on["od0"], off["od0"]))) > 1e-6
    np.testing.assert_array_equal(on["od0"], current["optical_readout"]["od0"])
    np.testing.assert_array_equal(finite["series"][-1]["optical_depth"],
                                  current["optical_readout"]["optical_depth"])


def test_unused_scheme_does_not_resolve_an_optional_comparison():
    def unexpected(*args):
        pytest.fail("Unexpected solver call")
    other = schemes.get("fwm")
    assert other.model_comparison_parameter is None
    assert other.model_comparison({}, resolve_raw=unexpected) is None
