"""Numerical checks of fixed OD instrument and cell calibration inputs."""
from dataclasses import replace
import json

import numpy as np
import pytest

from gabes import observables, vapor_cell
from gabes.experimental_alignment import AxisAlignment
from gabes.experimental_csv import CALIBRATION_ABSOLUTE_GAIN_OFFSET, load_experimental_csv
from gabes.od_spectrum_fit import (
    ODCellCalibration,
    ODFrequencyCalibration,
    ODOpticalCalibration,
    fit_od_spectrum,
    od_transmission,
    prepare_od_fit,
)
from gabes.schemes.sas import EXACT_WEAK_PROBE, SASScheme


@pytest.fixture
def conditions():
    return dict(SASScheme().defaults(), line="D2", pump_power_mw=0.0,
                probe_power_uw=0.0, temp_c=63.0, cell_mm=23.0)


def test_identity_optics_preserves_legacy_direct_samples(conditions):
    x = np.array([200.0, -50.0, 1200.0, -50.0, -500.0])
    offset = 3.0
    width = 7.0
    raw = SASScheme()._compute_species(
        conditions, scan_rad_s=(x - offset) * (2.0 * np.pi * 1e6),
        extra_homogeneous_fwhm_hz=width * 1e6,
    )
    expected = observables.transmission(
        raw["alpha_unit"] * conditions["line_strength"], conditions["cell_mm"] * 1e-3)
    for optics in (None, ODOpticalCalibration(), ODOpticalCalibration(laser_weights=(3.0,))):
        got = od_transmission(x, conditions, offset_mhz=offset,
                              extra_width_mhz=width, optics=optics)
        np.testing.assert_array_equal(got, expected)


def test_asymmetric_laser_averages_transmitted_power_before_bypass_and_baseline(conditions):
    x = np.array([220.0, -50.0, -700.0, 1200.0])
    optics = ODOpticalCalibration(
        laser_offsets_mhz=(-34.0, 85.0), laser_weights=(1.0, 3.0),
        bypass_fraction=0.02, baseline_frequency_mhz=(-700.0, 1200.0),
        baseline_transmission=(0.97, 1.01),
    )
    baseline = 0.97 + 0.04 * (x + 700.0) / 1900.0
    line_a = od_transmission(x - 34.0, conditions, offset_mhz=9.0)
    line_b = od_transmission(x + 85.0, conditions, offset_mhz=9.0)
    expected = baseline * (0.02 + 0.98 * (line_a + 3.0 * line_b) / 4.0)
    got = od_transmission(x, conditions, offset_mhz=9.0, optics=optics)
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=0.0)
    # Taking the logarithm before averaging would instead combine transmissions
    # geometrically. The asymmetric fixture resolves the difference.
    geometric_average = np.power(line_a, 0.25) * np.power(line_b, 0.75)
    assert np.max(np.abs(got - baseline * (0.02 + 0.98 * geometric_average))) > 1e-5
    with pytest.raises(ValueError, match="Optical baseline must bracket"):
        od_transmission(np.array([1201.0]), conditions, optics=optics)


def test_bypass_sets_deep_od_floor(conditions):
    thick = dict(conditions, temp_c=175.0, cell_mm=130.0)
    x = np.linspace(-4000.0, 4000.0, 65)
    clear = od_transmission(x, thick)
    assert np.min(clear) < 1e-25
    got = od_transmission(x, thick, optics=ODOpticalCalibration(bypass_fraction=0.001))
    assert np.min(got) == pytest.approx(0.001)
    assert np.max(-np.log(got)) == pytest.approx(-np.log(0.001))


def test_frequency_markers_fix_reversed_raw_branch_without_display_transform():
    t = np.linspace(2.0, 12.0, 33)
    noisy = 0.68 + 0.07 * np.sin(2 * t)
    noisy[[1, 4]] = [1.03, -0.05]
    raw_signal = 2.7 - 1.2 * noisy
    data = "\n".join(f"{a:.14g},{b:.14g}" for a, b in zip(t[::-1], raw_signal[::-1])).encode()
    trace = load_experimental_csv(data, calibration_mode=CALIBRATION_ABSOLUTE_GAIN_OFFSET,
                                  gain=-1.2, offset=2.7)
    raw_markers = (2.0, 6.0, 12.0)
    frequency_mhz = (800.0, -300.0, -5300.0)
    markers = ODFrequencyCalibration(raw_markers, frequency_mhz)
    prepared = prepare_od_fit(
        trace, AxisAlignment(scale=4.1, shift=2.0, reverse=True, signal_scale=8.0,
                             signal_offset=0.5), frequency_calibration=markers,
    )
    expected_x = np.where(t <= 6.0, 800.0 - 275.0 * (t - 2.0),
                          -300.0 - (5000.0 / 6.0) * (t - 6.0))
    np.testing.assert_allclose(prepared.frequency_mhz, expected_x[::-1], atol=1e-10)
    np.testing.assert_allclose(prepared.transmission, noisy[::-1], atol=1e-12)
    assert prepared.frequency_calibration is markers
    with pytest.raises(ValueError, match="Frequency markers must bracket"):
        prepare_od_fit(trace, frequency_calibration=ODFrequencyCalibration((3, 12), (-100, 200)))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ODFrequencyCalibration((0.0, 1.0, 2.0), (1.0, 0.0, 2.0)),
        lambda: ODFrequencyCalibration((0.0, 0.0), (1.0, 2.0)),
        lambda: ODOpticalCalibration(laser_offsets_mhz=(0, 1), laser_weights=(0, 0)),
        lambda: ODOpticalCalibration(laser_offsets_mhz=(0, 1), laser_weights=(2, -1)),
        lambda: ODOpticalCalibration(bypass_fraction=float("nan")),
        lambda: ODOpticalCalibration(baseline_frequency_mhz=(0.0,), baseline_transmission=(1.0,)),
        lambda: ODOpticalCalibration(baseline_frequency_mhz=(0.0, 2.0),
                                     baseline_transmission=(0.0, 1.0)),
        lambda: ODCellCalibration(cell_length_mm=0),
        lambda: ODCellCalibration(temperature=vapor_cell.resolve_cell_temperature(
            60.0, effective_vapor_temp_c=50.0, cold_spot_temp_c=60.0)),
    ],
)
def test_invalid_fixed_calibration_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_calibrated_synthetic_fit_recovers_width_and_predicts_disjoint_holdout(conditions):
    rng = np.random.default_rng(38)
    x = np.linspace(-3900.0, 4000.0, 231)
    x[1:-1] += rng.uniform(-2.0, 2.0, x.size - 2)
    marker_map = ODFrequencyCalibration((0.0, 0.5, 1.0), (-3900.0, -800.0, 4000.0))
    physical = dict(conditions, temp_c=70.0, cell_mm=31.0, doppler_temp_c=86.0,
                    constrain_doppler_temp=False)
    width = 14.0
    state = vapor_cell.resolve_cell_temperature(
        85.0, effective_vapor_temp_c=86.0, cold_spot_temp_c=70.0)
    cell = ODCellCalibration(state, 31.0, source="synthetic cell")
    optics = ODOpticalCalibration(
        laser_offsets_mhz=(0.0, 89.0), laser_weights=(0.84, 0.16), bypass_fraction=0.035,
        baseline_frequency_mhz=(x[0], x[-1]), baseline_transmission=(0.99, 0.98),
        source="synthetic optical calibration",
    )
    on_carrier = od_transmission(x, physical, extra_width_mhz=width)
    sideband = od_transmission(x + 89, physical, extra_width_mhz=width)
    baseline = 0.99 - 0.01 * (x - x[0]) / (x[-1] - x[0])
    clean = baseline * (0.035 + 0.965 * (0.84 * on_carrier + 0.16 * sideband))
    sigma = 0.0002
    noise = rng.normal(0.0, sigma, x.size)
    raw_axis = np.interp(x, marker_map.frequency_mhz, marker_map.raw_axis)
    csv = "\n".join(f"{a:.17g},{b:.17g}" for a, b in zip(raw_axis, clean + noise)).encode()
    trace = load_experimental_csv(csv, calibration_mode=CALIBRATION_ABSOLUTE_GAIN_OFFSET,
                                  gain=1.0, offset=0.0)
    data = prepare_od_fit(trace, frequency_calibration=marker_map)
    data = replace(data, sigma=np.full_like(x, sigma), uncertainty_source="known synthetic uncertainty")
    np.testing.assert_allclose(data.frequency_mhz, x, atol=1e-11)
    opts = dict(fit_range_mhz=(x[0], 1100.0), heldout_range_mhz=(1101.0, x[-1]),
                free=("extra_width_mhz",), cell=cell)
    fit = fit_od_spectrum(data, conditions, optics=optics, **opts)
    uncorrected = fit_od_spectrum(data, conditions, **opts)
    assert fit["identifiable"]
    assert fit["parameters"][0]["value"] == pytest.approx(width, abs=0.1)
    assert fit["heldout_residuals"]["rms"] < 1.5 * sigma
    np.testing.assert_array_equal(fit["sigma"], data.sigma)
    assert fit["uncertainty_source"] == data.uncertainty_source
    assert uncorrected["heldout_residuals"]["rms"] > 10 * sigma
    assert fit["frequency_calibration"]["residual_offset_fitted"] is False
    assert fit["cell_calibration"]["temperature"]["doppler_temp_c"] == 86.0
    assert fit["optical_calibration"]["source"] == "synthetic optical calibration"
    assert not np.any(np.asarray(fit["train_mask"]) & np.asarray(fit["heldout_mask"]))
    with pytest.raises(ValueError, match="fixed cell constraint"):
        fit_od_spectrum(data, conditions, free=("temp_c",), cell=cell)
    with pytest.raises(ValueError, match="default fit parameters are fixed"):
        fit_od_spectrum(data, conditions, cell=cell)
    with pytest.raises(ValueError, match="uncoated, buffer-free Rb"):
        od_transmission(x, dict(conditions, ne_pressure_torr=1), optics=optics)
    json.dumps(fit, allow_nan=False)


def test_heldout_csv_noise_cannot_change_any_training_result(conditions):
    conditions = dict(conditions, weak_probe_profile=EXACT_WEAK_PROBE)
    x = np.linspace(-3900.0, 4000.0, 501)
    markers = ODFrequencyCalibration((x[0], x[-1]), (x[0], x[-1]))
    train = x <= -600.0
    noise = np.random.default_rng(13).normal(size=x.size)
    clean = od_transmission(x, conditions)
    training_y = clean + np.where(x > -1000.0, 0.06, 0.0) + 0.0002 * noise

    prepared = []
    fits = []
    restricted_fits = []
    for heldout_noise in (0.0002, 0.01):
        signal = np.where(train, training_y, clean + heldout_noise * noise)
        csv = "\n".join(f"{a:.17g},{b:.17g}" for a, b in zip(x, signal)).encode()
        trace = load_experimental_csv(
            csv, calibration_mode=CALIBRATION_ABSOLUTE_GAIN_OFFSET,
            gain=1.0, offset=0.0, denoise=False,
        )
        data = prepare_od_fit(trace, frequency_calibration=markers)
        prepared.append(data)
        fits.append(fit_od_spectrum(
            data, conditions, fit_range_mhz=(-3900.0, -600.0),
            heldout_range_mhz=(-590.0, 4000.0),
        ))
        restricted_fits.append(fit_od_spectrum(data, conditions, fit_range_mhz=(-3900., -600.)))

    # Reproduces the source of leakage: the legacy whole-trace noise estimate
    # changes inside the train window even though all its signal stays fixed.
    assert np.max(np.abs(prepared[0].sigma[train] - prepared[1].sigma[train])) > 1e-4
    low_noise, high_noise = fits
    for result in fits:
        assert result["uncertainty_source"] == "Robust noise estimated from retained training samples only"
        np.testing.assert_array_equal(result["train_mask"], train)
    for key in ("sigma", "parameters", "covariance", "reduced_chi_squared",
                "fit_residuals", "model_transmission"):
        assert low_noise[key] == high_noise[key]
        assert restricted_fits[0][key] == restricted_fits[1][key] == low_noise[key]
    assert low_noise["heldout_residuals"] != high_noise["heldout_residuals"]

    fixed = replace(prepared[0], sigma=prepared[0].sigma.copy(), uncertainty_source="independent weights")
    fit = fit_od_spectrum(fixed, conditions, fit_range_mhz=(-3900., -600.),
                          heldout_range_mhz=(-590., 4000.))
    assert fit["uncertainty_source"] == "independent weights"
    np.testing.assert_array_equal(fit["sigma"], fixed.sigma)


def test_holdout_with_no_full_trace_noise_estimate_can_still_fit(conditions):
    conditions = dict(conditions, line="D1", temp_c=40., weak_probe_profile=EXACT_WEAK_PROBE)
    x = np.linspace(-3500., 4500., 801)
    markers = ODFrequencyCalibration((x[0], x[-1]), (x[0], x[-1]))
    train = x <= -3300.
    noise = np.random.default_rng(7).normal(scale=0.001, size=x.shape)
    training_signal = od_transmission(x, conditions) + noise
    results = []
    for heldout_signal in (np.full(x.shape, 0.5), 0.5 + noise):
        signal = np.where(train, training_signal, heldout_signal)
        csv = "\n".join(f"{a:.17g},{b:.17g}" for a, b in zip(x, signal)).encode()
        trace = load_experimental_csv(csv, calibration_mode=CALIBRATION_ABSOLUTE_GAIN_OFFSET,
                                      gain=1.0, offset=0.0, denoise=False)
        data = prepare_od_fit(trace, frequency_calibration=markers)
        if not results:
            assert data.sigma is None
            with pytest.raises(ValueError, match="Noise could not be estimated from this trace"):
                fit_od_spectrum(data, conditions)
        else:
            assert data.sigma is not None
        results.append(fit_od_spectrum(data, conditions, fit_range_mhz=(-3500., -3300.),
                                       heldout_range_mhz=(-3290., 4500.)))
    for key in ("parameters", "sigma", "covariance", "reduced_chi_squared", "fit_residuals"):
        assert results[0][key] == results[1][key]
    assert all(result["status"] == "converged" for result in results)
