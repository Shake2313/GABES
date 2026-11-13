"""Acceptance checks for the opt-in segmented OD/SAS reference."""
import numpy as np
import pytest

from gabes import constants, species, vapor_cell
from gabes.schemes.absorption import ODScheme
from gabes.schemes.sas import SASScheme
from gabes.schemes.sas_axial import AxialInputError, SEGMENTED


def _params(scheme, **edits):
    return {**scheme.defaults(), "species": "\u2078\u2075Rb", "line": "D1",
            "scan_points": 401, **edits}


@pytest.mark.parametrize("count,power", [(1, 0.5), (1, 0.), (4, 0.), (16, 0.)])
def test_flat_profile_recovers_uniform_raw_and_absorption_anchor(count, power):
    scheme = SASScheme()
    uniform = _params(scheme, pump_power_mw=power, temp_c=90., scan_points=1401)
    segmented = dict(uniform, cell_propagation=SEGMENTED, axial_segments=count)
    reference = scheme.compute(uniform)
    raw = scheme.compute(segmented)
    axial = raw.pop("axial_cell")
    assert axial["status"] == "exact_uniform_limit"
    assert axial["segment_count"] == count
    assert raw.keys() == reference.keys()
    for field in ("scan", "alpha_unit", "chi_real_unit"):
        np.testing.assert_array_equal(raw[field], reference[field])
    if power == 0.:
        od = ODScheme()
        independent = od.compute({**od.defaults(), "model": "85Rb D1 hyperfine", "temp_c": 90.})
        for value, target in (
                (np.trapezoid(raw["alpha_unit"], raw["scan"]),
                 np.trapezoid(independent["alpha"], independent["scan"])),
                (raw["alpha_unit"].max(), independent["alpha"].max())):
            assert value == pytest.approx(target, rel=1e-3)


def test_graded_cell_carries_local_pressure_field_and_segment_integral():
    scheme = SASScheme()
    params = _params(scheme, cell_propagation=SEGMENTED, axial_segments=4,
                     temp_c=40., constrain_doppler_temp=False, doppler_temp_c=50.,
                     axial_temperature_span_c=12., axial_field_span_ut=80., pump_power_mw=0.)
    raw = scheme.compute(params)
    view = scheme.headless_observables(raw, params)["optical_readout"]
    axial = view["axial_cell"]
    assert axial["status"] == "converged"
    np.testing.assert_allclose(axial["field_ut"], [-30., -10., 10., 30.])
    np.testing.assert_allclose(axial["temperature_c"], [45.5, 48.5, 51.5, 54.5])
    np.testing.assert_allclose(axial["density_m3"]["Rb85"] * constants.KB
                               * (axial["temperature_c"] + 273.15),
                               species.rb_vapor_pressure_pa(313.15), rtol=1e-13)
    np.testing.assert_allclose(view["optical_depth"], axial["optical_depth_unit"], rtol=1e-14)
    assert "Axially integrated" in view["attenuation_scope"]
    assert axial["difference_from_uniform"]["transmission_absolute_max"] > 1e-6
    assert max(axial["segment_change"].values()) < axial["segment_relative_tolerance"]


def test_counterpropagating_pump_has_loss_and_converged_independent_refinements():
    scheme = SASScheme()
    params = _params(scheme, cell_propagation=SEGMENTED, axial_segments=2,
                     pump_power_mw=0.5, axial_pump_depletion=True)
    raw = scheme.compute(params)
    readout = scheme.headless_observables(raw, params)["optical_readout"]
    axial = readout["axial_cell"]
    pump = axial["pump_power_mw_z"]
    np.testing.assert_array_equal(pump[-1], np.full(raw["scan"].size, 0.5))
    assert (np.diff(pump, axis=0) >= 0.0).all()
    assert np.min(pump[0]) < 0.45
    assert max(axial["segment_change"].values()) < 0.01
    assert max(axial["pump_grid_change"].values()) < 0.01
    assert axial["status"] == "converged"
    assert axial["difference_from_undepleted"]["transmission_absolute_max"] > 1e-3


def test_conditional_compute_keys_keep_uniform_readout_navigation():
    scheme = SASScheme()
    uniform = _params(scheme)
    axial = dict(uniform, cell_propagation=SEGMENTED)

    def cache_key(params):
        return tuple((key, params[key]) for key in scheme.solve_param_keys(params))

    for key, value in (("cell_mm", 80.), ("line_strength", 0.5), ("axial_segments", 8)):
        assert cache_key(dict(uniform, **{key: value})) == cache_key(uniform)
        assert cache_key(dict(axial, **{key: value})) != cache_key(axial)
    assert scheme.compute(dict(cache_key(axial)))["axial_cell"]["length_m"] == 0.075


@pytest.mark.parametrize("changes,reason", [
    ({"probe_power_uw": 1.}, "weak probe"),
    ({"paraffin_coated": True}, "paraffin"),
    ({"axial_temperature_span_c": 10.}, "cold spot"),
    ({"axial_center_field_ut": 150., "axial_field_span_ut": 200.}, "field endpoints"),
])
def test_invalid_reference_inputs_are_explicit(changes, reason):
    scheme = SASScheme()
    params = _params(scheme, cell_propagation=SEGMENTED, **changes)
    with pytest.raises(AxialInputError, match=reason):
        scheme.compute(params)


def test_midpoint_beer_lambert_uses_length_and_preserves_trailing_axis():
    edges = [0., 0.02, 0.03]
    alpha = [[20., 4.], [10., 7.]]
    got = vapor_cell.integrate_beer_lambert(edges, alpha, sample_location="midpoints")
    np.testing.assert_allclose(got.optical_depth, [0.5, 0.15], rtol=1e-14)
    np.testing.assert_allclose(got.transmission, np.exp(-np.asarray([0.5, 0.15])), rtol=1e-14)
