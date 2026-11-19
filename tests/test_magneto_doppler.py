"""
Unregistered single-pair reference: exact Maxwell light-region average.

  Exactness : the closed-form average matches a converged velocity grid solved
              by the unchanged steady-state kernels (forced guard fallback).
  Off path  : doppler="off" never builds a pole system.
  Fallback  : both circular coherences must converge or the B row is withheld.
  Provenance: raw keys and the Derived table name the Doppler method.

    pytest tests/test_magneto_doppler.py
"""
from unittest.mock import patch

import numpy as np
import pytest


from gabes import pole_doppler  # noqa: E402
from gabes.schemes import magneto  # noqa: E402
from gabes.schemes.magneto import SinglePairMagnetoReference  # noqa: E402

FIELDS = ("chi_probe", "chi_p", "chi_m")


def _params(regime, **over):
    params = SinglePairMagnetoReference().defaults()
    for regimes in SinglePairMagnetoReference._REGIMES.values():
        if regime in regimes:
            params.update(regimes[regime])
    params.update(over)
    return params


CASES = {
    "paraffin EIT dip": _params("EIT dip", scan_points=5),
    "paraffin detuned": _params("EIT dip", scan_points=5, laser_detuning_mhz=200.0),
    # No transverse field: the Doppler-free Zeeman populations have no relaxation
    # of their own, so only the shift-invert pole form applies.
    "paraffin F1-F2 no transverse field": _params(
        "EIT dip", scan_points=5, transition=magneto._transition_label(1, 2),
        residual_transverse_b_ut=0.0),
    "buffer LCA F2-F2": _params("Buffer LCA", scan_points=5),
}


def _relative(a, b):
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


@pytest.mark.parametrize("name", list(CASES))
def test_exact_average_matches_a_converged_velocity_grid(name):
    params = CASES[name]
    exact = SinglePairMagnetoReference().compute(params)
    with patch.object(magneto, "POLE_GUARD_RTOL", -1.0):
        grid = SinglePairMagnetoReference().compute(params)

    assert exact["doppler_method"] == magneto.DOPPLER_EXACT
    assert exact["doppler_fallback_rows"] == 0
    assert exact["doppler_guard_max_relative"] <= magneto.POLE_GUARD_RTOL
    assert exact["doppler_min_pole_distance_mhz"] > 0.0
    assert grid["doppler_fallback_rows"] == params["scan_points"]
    details = grid["doppler_fallback_details"]
    assert {record["row"] for record in details} == set(range(params["scan_points"]))
    assert all(record["status"] == "converged" for record in details)
    assert all(record["relative_change"] <= 1e-8 for record in details)
    assert grid["doppler_fallback_classes"] == max(record["classes"] for record in details)
    for key in FIELDS:
        assert _relative(exact[key], grid[key]) <= 1e-5, key


@pytest.mark.parametrize(("regime", "feature"), [
    ("EIT dip", "EIT-like dip"),
    ("EIA peak", "EIA/MIA-like peak"),
    ("Buffer Hanle", "EIT-like dip"),
    ("Buffer LCA", "crossover"),
    ("NMOR", "EIT-like dip"),
])
def test_preset_feature_is_recovered_by_exact_maxwell_average(regime, feature):
    scheme = SinglePairMagnetoReference()
    params = _params(regime)
    raw = scheme.compute(params)
    readout = scheme.headless_observables(raw, params)["magneto_readout"]

    assert raw["doppler_method"] == magneto.DOPPLER_EXACT
    assert raw["doppler_fallback_rows"] == 0
    assert readout["feature"] == feature
    if regime == "EIT dip":
        assert params["qwp_deg"] == 0
        assert readout["feature_amplitude_per_m"] < -0.2 * readout["background_absorption_per_m"]
    elif regime == "Buffer LCA":
        spec = next(spec for spec in scheme.param_schema() if spec.name == "regime")
        stored_key = next(key for key in spec.choices if key.endswith(regime))
        assert spec.choice_labels[stored_key].endswith("Buffer crossover")
        assert stored_key in scheme.recommended_defaults(scheme.defaults())


def test_doppler_off_never_builds_a_pole_system():
    params = _params("EIT dip", scan_points=5, doppler="off")
    with patch.object(pole_doppler, "ShiftInvertSystem",
                      side_effect=AssertionError("pole path used")), \
            patch.object(pole_doppler, "AffineShiftSystem",
                         side_effect=AssertionError("pole path used")):
        raw = SinglePairMagnetoReference().compute(params)
    assert raw["doppler_method"] == magneto.DOPPLER_OFF
    assert raw["doppler_guard_max_relative"] is None
    assert raw["doppler_fallback_rows"] == 0


@pytest.mark.parametrize("singular", [False, True])
def test_rejected_row_requires_both_circular_coherences(monkeypatch, singular):
    scheme = SinglePairMagnetoReference()
    params = _params("Buffer Hanle", scan_points=5)
    baseline = scheme.compute(params)
    original_mean = pole_doppler.PoleResidues.gaussian_mean
    original_average = scheme._exact_doppler_average

    def reject_two(self, *args):
        means = original_mean(self, *args)
        means[[0, 2], 0] = np.nan
        return means

    def controlled_average(*args, **kwargs):
        def grid(rows, velocity, weights):
            if singular and 2 in rows:
                raise np.linalg.LinAlgError("forced singular steady state")
            # A tiny second component must pass on its own scale. Row 0
            # agrees on the first two grids, then changes only in phase.
            changed = velocity.size >= 143
            other = [1j if slot == 0 and changed else -1j for slot in rows]
            for i in np.flatnonzero(rows == 2):
                other[i] = np.nan
            return np.column_stack([np.full(rows.size, 1e-10), 1e-30 * np.array(other)])
        kwargs["grid_average"] = grid
        return original_average(*args, **kwargs)

    monkeypatch.setattr(pole_doppler.PoleResidues, "gaussian_mean", reject_two)
    monkeypatch.setattr(scheme, "_exact_doppler_average", controlled_average)
    raw = scheme.compute(params)
    records = {row["row"]: row for row in raw["doppler_fallback_details"]}
    assert records[0]["status"] == "converged" and records[0]["refinements"] == 4
    assert records[2]["status"] == "unresolved"
    assert records[2]["classes"] == raw["doppler_fallback_classes"] == 351_811
    assert records[2]["relative_change"] is None
    assert raw["chi_p"][0] == 1e-10 and raw["chi_m"][0] == 1e-30j
    accepted = [1, 3, 4]
    for key in FIELDS:
        assert np.isnan(raw[key][2].real) and np.isnan(raw[key][2].imag)
        assert raw[key][accepted].tobytes() == baseline[key][accepted].tobytes()
    view = scheme.headless_observables(raw, params)
    assert view["magneto_readout"]["unresolved_rows"] == [2]
    assert "cap" in view["metrics"][0]["value"]
    assert all("SNR" not in metric["label"] for metric in view["metrics"])
    derived = next(table["markdown"] for table in view["tables"]
                   if table["title"] == "Derived quantities")
    assert "Unresolved Doppler B rows" in derived


def test_exact_average_is_reported_and_replaces_the_class_knob():
    params = _params("EIT dip", scan_points=5)
    scheme = SinglePairMagnetoReference()
    raw = scheme.compute(params)
    view = scheme.headless_observables(raw, params)
    derived = next(table["markdown"] for table in view["tables"]
                   if table["title"] == "Derived quantities")
    assert magneto.DOPPLER_EXACT in derived
    assert "Grid-fallback B rows" in derived
    assert raw["doppler_scale"] == 1.0 and raw["N_eff"] == raw["N"]
    assert "velocity_classes" not in {spec.name for spec in scheme.param_schema()}
