"""Joint OD fit on fixed measurement samples, including relative and reversed scans."""
from dataclasses import replace
import io

import numpy as np
import pytest

from gabes.experimental_alignment import AxisAlignment
from gabes.od_measurement import load_measurement, prepare_measurement_upload
from gabes.od_model import ODConditions, evaluate_od_display, rate_model_axis
from gabes.od_profile_fit import fit_od_profile, profile_fit_parameters, profile_group_windows
from gabes.od_spectrum_fit import ODFitCancelled, RATE_MODEL

GHZ = 2 * np.pi * 1e9
WEAK_MODEL = "od_weak_exact_voigt_transit_v1"


def medium(**overrides):
    return replace(ODConditions("Rb (natural)", "D2", 50., 12.5, WEAK_MODEL), **overrides)


def measurement(x, y, relative=False):
    buffer = io.StringIO()
    np.savetxt(buffer, np.column_stack((x, y)), delimiter=",", header="native,detector", comments="")
    return load_measurement(prepare_measurement_upload(buffer.getvalue().encode("utf-8")),
                            mode="relative" if relative else "fraction", x_unit="arbitrary", denoise=False)


def weak_curve(conditions, x):
    grid = np.linspace(-6., 6., 2001)
    transmission = evaluate_od_display(conditions, grid * GHZ).transmission
    return np.interp(x, grid, transmission)


def test_relative_reversed_native_roi_full_trace_pivot_and_all_points():
    conditions = medium()
    native = np.linspace(1250., 9750., 10501)
    pivot = (native[0] + native[-1]) / 2
    expected = AxisAlignment(scale=0.0011, shift=-pivot + 0.4, reverse=True)
    transmission = weak_curve(conditions, expected.transform_x(native))
    # Uncalibrated inverted detector, with periodic pickup. Keep every sample
    # in the asymmetric ROI and carry the full acquisition's X pivot back out.
    trace = measurement(native, -2.6 * transmission + 3.1 + 0.0003 * np.sin(np.arange(len(native))),
                        relative=True)
    first, last = 330, 9742
    initial = replace(expected, scale=0.001, shift=expected.shift + .12)
    result = fit_od_profile(trace, conditions.to_params(), initial, native[[first, last]],
                            theory_range_ghz=(-6., 6.), invert=True)
    assert result.relative and result.identifiable and result.condition_values == {}
    assert result.n_points == last - first + 1
    assert result.alignment.reverse
    np.testing.assert_allclose(result.alignment.transform_x(native)[[first, last]],
                               expected.transform_x(native)[[first, last]], atol=3e-4)
    assert result.rmse < result.initial_rmse
    assert result.correlation > 0.9999
    assert result.alignment.signal_scale != 1


def test_known_group_absolute_y_and_preview_match_shared_model():
    conditions = medium(line="D1")
    params = conditions.to_params()
    group = "87-F1"
    lo, hi = profile_group_windows(params, (-6., 6.))[group]
    x = np.linspace(lo + 0.10, hi - 0.20, 251)
    trace = measurement(x, weak_curve(conditions, x))
    result = fit_od_profile(trace, params, AxisAlignment(shift=.05, signal_scale=.3, signal_offset=.5),
                            (x[0], x[-1]), {"temp_c": (40, 60)}, group=group, theory_range_ghz=(-6., 6.))
    assert result.identifiable and not result.relative
    assert result.condition_values["temp_c"] == pytest.approx(50., abs=1e-3)
    assert result.alignment.signal_scale == 1. and result.alignment.signal_offset == 0.
    assert result.rmse < 1e-5
    np.testing.assert_allclose(result.preview_frequency_ghz[[0, -1]], (-6., 6.))
    assert not result.preview_frequency_ghz.flags.writeable and not result.preview_transmission.flags.writeable
    fitted = result.alignment.transform_x(x)
    assert lo <= min(fitted) < max(fitted) <= hi
    np.testing.assert_allclose(result.preview_transmission, weak_curve(conditions, result.preview_frequency_ghz),
                               atol=2e-6)


def test_degenerate_and_bound_conditions_report_preview_only():
    # At fixed Doppler temperature the data nearly determine only density times length.
    true = medium(doppler_temp_c=50.)
    x = np.linspace(-4.7, 5.3, 401)
    trace = measurement(x, weak_curve(true, x))
    fit = fit_od_profile(trace, true.to_params(), AxisAlignment(), (x[0], x[-1]),
                         {"temp_c": (40, 60), "cell_mm": (5, 20)}, theory_range_ghz=(-6., 6.))
    assert not fit.identifiable and fit.correlation > .999
    assert fit.correlated_pairs and abs(fit.correlated_pairs[0][2]) > .98
    high = medium(temp_c=55.)
    trace = measurement(x, weak_curve(high, x))
    fit = fit_od_profile(trace, medium(temp_c=52.).to_params(), AxisAlignment(), (x[0], x[-1]),
                         {"temp_c": (45, 54.9)}, theory_range_ghz=(-6., 6.))
    assert not fit.identifiable and fit.condition_values["temp_c"] == pytest.approx(54.9, abs=.001)
    assert any("fit bound" in warning for warning in fit.warnings)


def test_zero_and_driven_d2_power_branch_obey_physical_guard():
    true = medium(model_id=RATE_MODEL)
    axis = rate_model_axis(true)
    x = axis[100:-100:5] / GHZ
    trace = measurement(x, evaluate_od_display(true, axis).transmission[100:-100:5])
    start = replace(true, probe_power_uw=10.)
    fit = fit_od_profile(trace, start.to_params(), AxisAlignment(), (x[0], x[-1]),
                         {"probe_power_uw": (0, 25)}, theory_range_ghz=(axis[0]/GHZ,axis[-1]/GHZ))
    assert fit.condition_values["probe_power_uw"] == 0
    assert not fit.identifiable
    assert any("probe_power_uw" in warning for warning in fit.warnings)
    assert fit.rmse < 1e-8


def test_scope_bounds_bad_match_cancel_and_group_reference():
    params = medium(line="D1", model_id=RATE_MODEL).to_params()
    x = np.linspace(-4.4, 4.4, 201)
    trace = measurement(x, np.random.default_rng(0).normal(0.5, .1, size=len(x)))
    args = (trace, params, AxisAlignment(), (x[0], x[-1]))
    assert "probe_power_uw" not in profile_fit_parameters(params)
    for bounds in ({"probe_power_uw": (0, 25)}, {"probe_waist_mm": (0.4, 3)}):
        with pytest.raises(ValueError, match="not offered"):
            fit_od_profile(*args, bounds, theory_range_ghz=(-6, 6))
    with pytest.raises(ValueError, match="within"):
        fit_od_profile(trace, dict(params, line="D2"), AxisAlignment(), (-4,4),
                       {"probe_power_uw": (0, 101)}, theory_range_ghz=(-6, 6))
    with pytest.raises(ValueError, match="do not match"):
        fit_od_profile(*args, theory_range_ghz=(-6, 6))
    with pytest.raises(ODFitCancelled):
        fit_od_profile(*args, theory_range_ghz=(-6, 6), cancelled=lambda: True)
    with pytest.raises(ODFitCancelled):
        fit_od_profile(*args, theory_range_ghz=(-6, 6), progress=lambda state: False)
    with pytest.raises(ValueError, match="12 distinct"):
        fit_od_profile(trace, params, AxisAlignment(), x[[1, 10]], theory_range_ghz=(-6, 6))
    custom = dict(params, rb_composition="Custom", rb85_fraction=1.)
    with pytest.raises(ValueError, match="isotope is absent"):
        fit_od_profile(trace, custom, AxisAlignment(), (-4,4), group="87-F1", theory_range_ghz=(-6, 6))
    from gabes import species
    pure = profile_group_windows(dict(params, species=species.RB87.label), (-10, 10))
    mixed = profile_group_windows(params, (-10,10))
    assert set(pure) == {"full", "87-F1", "87-F2"}
    delta = (species.RB87.nu_D1 - species.RB85.nu_D1) * 1e-9
    np.testing.assert_allclose(pure["87-F1"], np.array(mixed["87-F1"]) - delta)


def test_group_preview_frames_only_selected_native_samples():
    from gabes.od_profile_fit import profile_frame_alignment

    conditions = medium()
    params = conditions.to_params()
    window = profile_group_windows(params, (-6., 6.))["85-F3"]
    native = np.linspace(100., 200., 401)
    trace = measurement(native, np.sin(np.linspace(0, 9., len(native))), relative=True)
    first, last = 71, 244
    roi = native[[first, last]]
    theory_x = np.linspace(-6., 6., 2001)
    theory = weak_curve(conditions, theory_x)
    frame = profile_frame_alignment(trace, params, roi, group="85-F3", theory_range_ghz=(-6., 6.),
                                    reverse=True, invert=True, theory_x_ghz=theory_x,
                                    theory_transmission=theory)
    margin = .05 * (window[1] - window[0])
    np.testing.assert_allclose(frame.transform_x(native)[[first, last]], (window[1] - margin, window[0] + margin))
    reference = theory[(theory_x >= window[0]) & (theory_x <= window[1])]
    shown = frame.transform_y(trace.transmission[first:last + 1], invert=True)
    np.testing.assert_allclose((shown.min(), shown.max()), (reference.min(), reference.max()))


@pytest.mark.parametrize("line", ["D1", "D2"])
def test_custom_rb_fraction_recovered_on_both_lines(line):
    truth = medium(line=line, rb_composition="Custom", rb85_fraction=.81)
    start = replace(truth, rb85_fraction=.7)
    x = np.linspace(-4.7, 5.3, 401)
    trace = measurement(x, weak_curve(truth, x))
    result = fit_od_profile(trace, start.to_params(), AxisAlignment(shift=.03), (x[0], x[-1]),
                            {"rb85_fraction": (.3, .95)}, theory_range_ghz=(-6., 6.))
    assert result.identifiable
    assert result.condition_values["rb85_fraction"] == pytest.approx(.81, abs=1e-4)


def test_d2_joint_driven_branch_recovers_power_and_temperature_from_noisy_trace():
    truth = medium(model_id=RATE_MODEL, temp_c=48.6, probe_power_uw=18.)
    axis = rate_model_axis(truth)
    x = axis[120:-120:4] / GHZ
    transmission = evaluate_od_display(truth, axis).transmission[120:-120:4]
    y = np.random.default_rng(20261123).poisson(100000 * transmission) / 100000
    trace = measurement(x, y)
    start = replace(truth, temp_c=45., probe_power_uw=0.)
    result = fit_od_profile(trace, start.to_params(), AxisAlignment(shift=.04), (x[0], x[-1]),
                            {"temp_c": (40, 58), "probe_power_uw": (0, 45)},
                            theory_range_ghz=(axis[0] / GHZ, axis[-1] / GHZ))
    assert result.identifiable and result.n_points == len(x)
    assert result.condition_values["temp_c"] == pytest.approx(48.6, abs=.15)
    assert result.condition_values["probe_power_uw"] == pytest.approx(18., abs=1.2)
    assert result.rmse < .004


def test_straight_relative_segment_cannot_identify_an_affine_frequency_map():
    conditions = medium()
    x = np.linspace(-.8252, -.8248, 101)
    # All 101 theoretical samples lie in a single linear interpolation interval.
    trace = measurement(x, weak_curve(conditions, x), relative=True)
    result = fit_od_profile(trace, conditions.to_params(), AxisAlignment(), (x[0], x[-1]),
                            theory_range_ghz=(-6., 6.))
    assert result.correlation > .99
    assert not result.identifiable
    assert any("frequency map" in text for text in result.warnings)


def test_pinned_frequency_endpoint_blocks_physical_condition_apply():
    conditions = medium(line="D1")
    x = np.linspace(3.3, 4.9, 201)
    trace = measurement(x, weak_curve(conditions, x + .38))
    args = (trace, conditions.to_params(), AxisAlignment(), (x[0], x[-1]))
    for bounds, identifiable in (({}, True), ({"temp_c": (45, 55)}, False)):
        result = fit_od_profile(*args, bounds, group="87-F1", theory_range_ghz=(-6., 6.))
        assert result.identifiable == identifiable
        assert result.correlation > .99
        assert any("alignment reached" in text for text in result.warnings)


def test_temperature_confounded_with_frequency_endpoint_blocks_apply():
    conditions = medium()
    x = np.linspace(1.2, 1.55, 101)
    trace = measurement(x, weak_curve(conditions, x))
    result = fit_od_profile(trace, conditions.to_params(), AxisAlignment(), (x[0], x[-1]),
                            {"temp_c": (40, 55)}, theory_range_ghz=(-6., 6.))
    assert result.rmse < 1e-8 and not result.identifiable
    assert any(left == "axis_right_ghz" and right == "temp_c" and abs(rho) > .98
               for left, right, rho in result.correlated_pairs)
