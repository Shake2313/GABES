"""Practical FWM models stay active across detuning and apparatus controls."""
from copy import deepcopy
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gabes.schemes import fwm
from gabes_ui.export import build_export_payload


@pytest.fixture(scope="module")
def experimental_case():
    scheme = fwm.FWMScheme()
    params = dict(scheme.defaults(), opd=-1.4, temp_c=113.0,
                  pump_mw=300.0, probe_uw=3.0, cell_mm=12.5,
                  pump_waist_um=530.0, probe_waist_um=330.0,
                  seeded_angle_deg=0.32, transit_rate_khz=100.0,
                  resolution=fwm.FIDELITY_FAST, tpd=-124.317286437,
                  after_cell_eff_pct=94.5, qe_pct=92.0,
                  seed_classical_rin_scaled=0.0,
                  noise_model=fwm.NOISE_EXPERIMENTAL_JASPERSE,
                  experimental_gain_calibration=True, gain_calibration_model=fwm.MODEL_LAB)
    raw = scheme.compute(params)
    return scheme, params, raw


def test_practical_defaults_and_controls_only_change_readout():
    scheme = fwm.FWMScheme()
    specs = {spec.name: spec for spec in scheme.param_schema()}
    assert scheme.defaults()["noise_model"] == fwm.NOISE_EXPERIMENTAL_JASPERSE
    assert scheme.defaults()["experimental_gain_calibration"] is False
    for name in ("noise_model", "experimental_gain_calibration"):
        assert specs[name].recompute is False
        assert name not in scheme.recompute_keys()
    assert fwm.NOISE_PASSIVE_L1 in specs["noise_model"].choices


def test_experimental_nf_fills_the_screenshot_gap_and_preserves_cached_model(experimental_case):
    scheme, params, raw = experimental_case
    original = deepcopy(raw)
    view = scheme.headless_observables(raw, params)
    assert view["noise_model"] == fwm.NOISE_EXPERIMENTAL_JASPERSE
    assert view["experimental_gain_calibration"]["applied"]
    assert "updated source model" in view["metrics"][1]["delta"]
    assert "Updated atomic model with the frozen fit coefficients" in view["tables"][0]["markdown"]
    assert view["experimental_jasperse_readout"] is not None
    point = view["noise_operating_point"]
    assert np.isfinite(point["NF"]) and point["NF"] > 0.0
    assert point["NF_dB"] == pytest.approx(10*np.log10(point["NF"]), abs=1e-12)
    assert view["metrics"][0]["value"] == f"{point['NF_dB']:.2f} dB"
    assert not view["passive_l1_operating_point"]["passive_l1_available"]
    for key in ("G_s", "G_c", "T_canonical_small_signal", "gain_referred_noise_dB"):
        np.testing.assert_array_equal(raw[key], original[key])
    assert raw["claim_gate"] == original["claim_gate"]


def test_extrapolated_selected_points_keep_gain_and_noise_available(experimental_case):
    scheme, params, raw = experimental_case
    for delta in (-200.0, -40.0, 40.0, 200.0):
        view = scheme.headless_observables(raw, dict(params, tpd=delta))
        calibration = view["experimental_gain_calibration"]
        assert calibration["applied"]
        point = calibration["selected_point"]
        assert point["available"]
        assert np.isfinite(point["G_s"]) and point["G_s"] >= 0.0
        assert np.isfinite(point["G_c"]) and point["G_c"] >= 0.0
        assert np.isfinite(view["noise_operating_point"]["NF_dB"])


def test_operating_condition_changes_keep_correction_active_and_model_responsive(experimental_case):
    scheme, params, raw = experimental_case
    first = scheme.headless_observables(raw, params)
    changed_params = dict(params, pump_mw=330.0, temp_c=115.0, opd=-1.2)
    changed_raw = scheme.compute(changed_params)
    changed = scheme.headless_observables(changed_raw, changed_params)
    assert changed["experimental_gain_calibration"]["applied"]
    assert changed["experimental_gain_calibration"]["selected_point"]["available"]
    assert np.isfinite(changed["noise_operating_point"]["NF_dB"])
    assert not np.allclose(first["experimental_gain_calibration"]["G_s"],
                           changed["experimental_gain_calibration"]["G_s"])


@pytest.mark.parametrize("calibrated", [False, True])
def test_focused_plot_marker_headless_and_strict_export_agree(experimental_case, calibrated):
    import matplotlib.pyplot as plt
    scheme, params, raw = experimental_case
    params = dict(params, experimental_gain_calibration=calibrated)
    rendered = scheme.observables(raw, params)
    headless = scheme.headless_observables(raw, params)
    try:
        assert rendered["metrics"] == headless["metrics"]
        gain_axis = rendered["figure"].axes[0]
        labels = [line.get_label() for line in gain_axis.lines]
        conjugate_label = "Calibrated conjugate" if calibrated else "Conjugate"
        conjugate = gain_axis.lines[labels.index(conjugate_label)]
        assert conjugate_label in gain_axis.get_legend_handles_labels()[1]
        if not calibrated:
            np.testing.assert_array_equal(conjugate.get_ydata(), raw["G_c"])
        point = rendered["noise_operating_point"]
        axis = rendered["figure"].axes[1]
        assert float(axis.collections[0].get_offsets()[0, 1]) == pytest.approx(point["NF_dB"])
        assert np.interp(params["tpd"], *axis.lines[0].get_data()) == pytest.approx(point["NF_dB"])
        payload = build_export_payload(
            scheme=scheme, raw=raw, params=params, view=rendered,
            plot_payloads=[], comparison_payload=None,
            cache_version=scheme.cache_version, readout_cache_version="experiment-test")
        decoded = json.loads(json.dumps(payload, allow_nan=False))
        assert decoded["view"]["experimental_gain_calibration"]["applied"] is calibrated
        assert decoded["view"]["experimental_jasperse_readout"] is not None
        assert decoded["view"]["noise_operating_point"]["NF"] == pytest.approx(point["NF"])
    finally:
        plt.close(rendered["figure"])


def test_detection_and_classical_noise_change_nf_without_refitting_gain(experimental_case):
    scheme, params, raw = experimental_case
    first = scheme.headless_observables(raw, params)
    changed = scheme.headless_observables(raw, dict(
        params, seed_classical_rin_scaled=20.0, qe_pct=60.0, after_cell_eff_pct=80.0))
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(first["experimental_gain_calibration"][key],
                                      changed["experimental_gain_calibration"][key])
    assert first["noise_operating_point"]["NF"] != changed["noise_operating_point"]["NF"]
    zero = scheme.headless_observables(raw, dict(params, qe_pct=0.0))
    assert np.all(np.isnan(zero["experimental_jasperse_readout"]["NF"]))
    assert not zero["noise_operating_point"]["available"]


def test_extreme_effective_gain_mismatch_is_visible_in_selected_readout(experimental_case):
    scheme, params, raw = experimental_case
    extreme = dict(raw, G_s=np.full_like(raw["G_s"], 14.6092),
                   G_c=np.full_like(raw["G_c"], .0143289))
    view = scheme.headless_observables(extreme, dict(params, experimental_gain_calibration=False))
    point = view["noise_operating_point"]
    assert point["available"] and point["inference_approximate"]
    assert point["fit_relative_residual"] > .01
    assert "approximate gain match" in view["metrics"][0]["delta"]
    assert "Assumed Jasperse common loss" in view["tables"][0]["markdown"]
    assert "cannot reproduce this pair exactly" in view["tables"][0]["markdown"]


def test_live_cursor_uses_calibrated_gain_and_committed_nf(experimental_case):
    import matplotlib.pyplot as plt
    from gabes_ui.liveplot import preview_data
    scheme, params, raw = experimental_case
    view = scheme.observables(raw, params)
    try:
        figure = view["figure"]
        data, moving = preview_data(figure, "fwm", canonical_noise=True)
        assert data["original"]["tpd"] == pytest.approx(params["tpd"])
        gain_point = view["experimental_gain_calibration"]["selected_point"]
        gain_curve = data["curves"][0]
        assert np.interp(params["tpd"], gain_curve["x"], gain_curve["y"]) == pytest.approx(gain_point["G_s"])
        assert float(figure.axes[0].collections[0].get_offsets()[0, 1]) == pytest.approx(gain_point["G_s"])
        for axis in figure.axes:
            assert axis.lines[1] in moving
            assert axis.collections[0] in moving
        assert data["linear_noise"]
        assert data["curves"][1]["x"] == list(figure.axes[1].lines[0].get_xdata())
    finally:
        plt.close(view["figure"])


@pytest.mark.parametrize("selection", [fwm.MODEL_LAB, fwm.MODEL_LAB_REFIT])
def test_full_view_routes_both_practical_models_and_branch_coordinates(experimental_case, selection):
    import matplotlib.pyplot as plt
    from gabes.fwm_gain_calibration import calibrate_spectrum
    from gabes.fwm_experiment_noise import experimental_jasperse_readout
    scheme, params, raw = experimental_case
    params = dict(params, gain_calibration_model=selection)
    plus = dict(raw, branch=1, probe_axis_GHz=(raw["probe_axis_GHz"]
                +raw["raman_center_plus_GHz"]-raw["raman_center_minus_GHz"]))
    full = {"D_GHz": params["opd"], "minus": raw, "plus": plus}
    figure = scheme.extra_views()[0].render(full, params)
    try:
        assert set(figure._gabes_experimental_gain_calibration) == {"minus", "plus"}
        assert set(figure._gabes_experimental_jasperse_readout) == {"minus", "plus"}
        for index, branch in enumerate(("minus", "plus")):
            spec = full[branch]
            gains = calibrate_spectrum(spec, enabled=True, model=selection)
            noise = experimental_jasperse_readout(spec, params, gains=gains)
            assert figure._gabes_experimental_gain_calibration[branch]["applied"]
            np.testing.assert_array_equal(figure.axes[0].lines[index].get_xdata(), spec["probe_axis_GHz"])
            np.testing.assert_allclose(figure.axes[0].lines[index].get_ydata(), gains["G_s"])
            np.testing.assert_allclose(figure.axes[1].lines[index].get_ydata(), noise["NF_dB"], equal_nan=True)
    finally:
        plt.close(figure)


def test_opt_in_exact_readout_recovers_native_passive_eligibility_without_changing_raw():
    scheme = fwm.FWMScheme()
    gold = next(p.values for p in scheme.presets() if p.name == "Sim Gold Reference")
    params = dict(gold, tpd=393.25, noise_model=fwm.NOISE_PASSIVE_L1)
    raw = scheme.compute(params)
    approximate = fwm.seeded_operating_readout(raw, params)
    exact = fwm.seeded_operating_readout(raw, dict(params, exact_readout_coordinates=True))
    assert approximate["readout_certification_scope"] == "UNCERTIFIED_CALIBRATION_OR_SOURCE"
    assert exact["readout_coordinate_status"] == "DENSE_POLE_READOUT_COORDINATES"
    assert not approximate["noise_operating_point"]["passive_l1_available"]
    assert exact["noise_operating_point"]["passive_l1_available"]
    assert raw["response_estimator"]["solved_probe_points"] < len(raw["probe_axis_GHz"])
    balanced = scheme.compute(dict(params, resolution=fwm.FIDELITY_BALANCED))
    expected = fwm.seeded_operating_readout(balanced, params)
    for key in ("G_s", "G_c"):
        np.testing.assert_array_equal(exact["gain_curve"][key], expected["gain_curve"][key])
    np.testing.assert_array_equal(exact["passive_l1_readout"]["NF_dB"],
                                  expected["passive_l1_readout"]["NF_dB"])
    phase = fwm.seeded_operating_readout(raw, dict(
        params, exact_readout_coordinates=True, squeezing_view=fwm.SQUEEZING_QUADRATURE_PHASE))
    assert phase["readout_coordinate_status"] == "DENSE_POLE_READOUT_COORDINATES"
    assert phase["readout_certification_scope"] == "UNCERTIFIED_CALIBRATION_OR_SOURCE"


@pytest.mark.parametrize("selection", [fwm.MODEL_LAB, fwm.MODEL_LAB_REFIT])
def test_opt_in_lab_warp_uses_exact_mapped_gains_and_reuses_source_on_detector_change(selection):
    from gabes import fwm_experiment_gain as lab

    scheme = fwm.FWMScheme()
    params = next(p.values for p in scheme.presets() if p.name == "2026-10-13 experiment")
    params = dict(params, gain_calibration_model=selection)
    model_id = lab.REFIT_MODEL_ID if selection == fwm.MODEL_LAB_REFIT else lab.MODEL_ID
    raw = scheme.compute(params)
    exact_params = dict(params, exact_readout_coordinates=True)
    fwm._focused_readout_reference.cache_clear()
    exact = fwm.seeded_operating_readout(raw, exact_params)
    prior = fwm.seeded_operating_readout(raw, params)
    tpd = params["tpd"]
    coordinates = lab.source_detunings(tpd, model_id=model_id)
    probe = raw["raman_center_minus_GHz"]
    context = dict(raw["stark_recompute_context"], response_method=fwm.RESPONSE_POLE)
    dense = fwm.compute_spectrum(**context, extra_probe_points_GHz=(
        probe + float(coordinates["probe"]) * 1e-3,
        probe + float(coordinates["conjugate"]) * 1e-3,
    ))
    expected = lab.evaluate(dense, tpd, model_id=model_id)
    for key in ("G_s", "G_c"):
        assert exact["gain_operating_point"][key] == pytest.approx(expected[key], rel=2e-12)
    assert exact["gain_operating_point"]["G_c"] != pytest.approx(prior["gain_operating_point"]["G_c"])
    assert exact["experimental_gain_calibration"]["exact_nominal_node_deviation"]["point_count"] == 23
    misses = fwm._focused_readout_reference.cache_info().misses
    changed = fwm.seeded_operating_readout(raw, dict(exact_params, qe_pct=60.0))
    assert fwm._focused_readout_reference.cache_info().misses == misses
    assert changed["noise_operating_point"]["NF"] != exact["noise_operating_point"]["NF"]
    np.testing.assert_array_equal(changed["gain_curve"]["G_c"], exact["gain_curve"]["G_c"])
