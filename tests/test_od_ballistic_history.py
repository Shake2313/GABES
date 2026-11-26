"""Beam-crossing history through the scheme, OD fitter and saved worksheet."""
from dataclasses import replace

import numpy as np
import pytest

from gabes import species
from gabes.core import blas_single_thread
from gabes.od_measurement import export_comparison_csv, load_measurement, prepare_measurement_upload
from gabes.od_model import ODConditions, evaluate_od_display, od_unavailable
from gabes.od_spectrum_fit import od_fit_unavailable, od_transmission, offered_fit_parameters
from gabes.schemes.sas import (
    BALLISTIC_FAST, BALLISTIC_FAST_V1, BALLISTIC_FAST_TRANSIT, BALLISTIC_REFERENCE, BALLISTIC_TRANSIT,
    EXACT_MODEL, GAUSSIAN_TRANSIT, GAUSSIAN_TRANSIT_D20_MODEL, OD_MODELS_D20,
    OD_MODELS_D2SB, OD_MODELS_NTOTAL, OD_MODELS_PRE_D20, RB_HISTORY_FAST, RB_HISTORY_REFERENCE,
    ODModelSettingError, SASScheme,
)


def parameters(**overrides):
    params = dict(SASScheme().defaults(), species=species.RB85.label, line="D1",
                  temp_c=99., cell_mm=12.5, probe_waist_mm=.53, pump_power_mw=0.,
                  probe_power_uw=0., od_transit_model=BALLISTIC_TRANSIT)
    return dict(params, **overrides)


@pytest.mark.parametrize(("transit", "model"), [(BALLISTIC_TRANSIT, BALLISTIC_REFERENCE),
                                                (BALLISTIC_FAST_TRANSIT, BALLISTIC_FAST)])
def test_history_weak_limit_sampled_view_and_fitter(transit, model):
    scheme = SASScheme()
    p = parameters(scan_points=101, od_transit_model=transit)
    c = ODConditions.from_params(p)
    assert c.model_id == model
    assert ODConditions.from_params(c.to_params()) == c
    assert c.transit_rates_s() == {}
    keys = scheme.solve_param_keys(p)
    assert {"cell_mm", "line_strength"} <= set(keys) and "transit_khz" not in keys
    raw = scheme.compute({key: p[key] for key in keys})
    readout = scheme.headless_observables(raw, p)["optical_readout"]
    assert raw["probe_model"] == model
    weak = evaluate_od_display(replace(c, model_id=EXACT_MODEL), raw["scan"])
    np.testing.assert_allclose(readout["transmission"], weak.transmission, atol=1e-13)
    np.testing.assert_allclose(readout["od0"], weak.od0, rtol=1e-13)
    x = raw["scan"][[70, 10, 70]]
    np.testing.assert_allclose(od_transmission(x / (2*np.pi*1e6), p), weak.transmission[[70, 10, 70]])
    assert ODConditions.from_params(dict(p, od_model=EXACT_MODEL)).model_id == EXACT_MODEL


@pytest.mark.parametrize("conditions", [
    {"species": species.RB87.label}, {"species": "Rb (natural)"}, {"line": "D2"},
    {"ne_pressure_torr": 5.}, {"paraffin_coated": True},
    {"cell_propagation": "Segmented reference"}, {"probe_waist_mm": 0.},
])
def test_history_scope_rejects_before_solve(conditions):
    with pytest.raises(ODModelSettingError):
        SASScheme().compute(parameters(od_model=BALLISTIC_REFERENCE, **conditions))


@pytest.mark.parametrize(("setting", "selector", "sas_set", "off_id", "probe_uw"), [
    (OD_MODELS_D2SB, BALLISTIC_TRANSIT, OD_MODELS_D2SB, BALLISTIC_REFERENCE, 0.),
    (OD_MODELS_PRE_D20, BALLISTIC_FAST_TRANSIT, OD_MODELS_PRE_D20, BALLISTIC_FAST, 3.),
    (BALLISTIC_REFERENCE, "fixed", OD_MODELS_D2SB, BALLISTIC_REFERENCE, 0.),
    (OD_MODELS_PRE_D20, GAUSSIAN_TRANSIT, OD_MODELS_PRE_D20, GAUSSIAN_TRANSIT_D20_MODEL, 0.),
    (GAUSSIAN_TRANSIT_D20_MODEL, "fixed", OD_MODELS_D20, GAUSSIAN_TRANSIT_D20_MODEL, 0.),
])
def test_geometry_preference_survives_driven_sas(setting, selector, sas_set, off_id, probe_uw):
    scheme = SASScheme()
    p = parameters(od_model=setting, od_transit_model=selector, pump_power_mw=.02,
                   transit_khz=250., probe_power_uw=probe_uw, scan_points=101)
    original = p.copy()
    fixed = dict(p, od_model=sas_set, od_transit_model="fixed")
    assert "transit_khz" in scheme.solve_param_keys(p)
    with blas_single_thread():
        raw = scheme.compute({k: p[k] for k in scheme.solve_param_keys(p)})
        control = scheme.compute({k: fixed[k] for k in scheme.solve_param_keys(fixed)})
    assert raw["od_model_set"] == sas_set
    assert raw["probe_model"] == control["probe_model"]
    assert "ballistic_diagnostics" not in raw and "transit_rates_s" not in raw
    readout = scheme.headless_observables(raw, p)["optical_readout"]
    expected = scheme.headless_observables(control, fixed)["optical_readout"]
    np.testing.assert_array_equal(readout["transmission"], expected["transmission"])
    assert p == original
    # CSV fitting always derives pump-off conditions, even when SAS is on.
    assert ODConditions.from_params(p).model_id == off_id
    off = dict(p, pump_power_mw=0., probe_power_uw=0.)
    assert scheme.compute({k: off[k] for k in scheme.solve_param_keys(off)})["probe_model"] == off_id


@pytest.mark.parametrize("overrides", [
    {"species": species.CS133.label}, {"ne_pressure_torr": 1.}, {"paraffin_coated": True},
    {"cell_propagation": "Segmented reference"},
])
def test_isotope_resolved_histories_require_uniform_uncoated_rb(overrides):
    p = parameters(species="Rb (natural)", line="D2", od_model=RB_HISTORY_REFERENCE)
    with pytest.raises(ODModelSettingError):
        SASScheme().compute(dict(p, **overrides))


def test_pumped_mixture_uses_current_sas_and_recovers_general_history_on_return():
    scheme = SASScheme()
    p = parameters(species="Rb (natural)", rb_composition="Custom", rb85_fraction=.38,
                   line="D2", od_model=OD_MODELS_NTOTAL, pump_power_mw=.02, scan_points=101)
    fixed = dict(p, od_transit_model="fixed")
    with blas_single_thread():
        raw = scheme.compute(p)
        expected = scheme.compute(fixed)
    np.testing.assert_array_equal(raw["alpha_unit"], expected["alpha_unit"])
    assert raw["probe_model"] == expected["probe_model"]
    assert "ballistic_depth" not in raw
    p["pump_power_mw"] = 0.
    assert ODConditions.from_params(p).model_id == RB_HISTORY_REFERENCE
    assert scheme.compute(p)["probe_model"] == RB_HISTORY_REFERENCE


@pytest.mark.parametrize("model_set", [OD_MODELS_PRE_D20, OD_MODELS_D20, OD_MODELS_D2SB, OD_MODELS_NTOTAL])
@pytest.mark.parametrize("transit,general,pure", [
    (BALLISTIC_TRANSIT, RB_HISTORY_REFERENCE, BALLISTIC_REFERENCE),
    (BALLISTIC_FAST_TRANSIT, RB_HISTORY_FAST, BALLISTIC_FAST),
])
def test_automatic_history_resolves_mixtures_without_changing_recorded_pure_ids(model_set, transit, general, pure):
    scheme = SASScheme()
    for line in ("D1", "D2"):
        for composition, fraction in (("Natural", .7217), ("Custom", .38), ("Custom", 0.), ("Custom", 1.)):
            p = parameters(species="Rb (natural)", line=line, rb_composition=composition,
                           rb85_fraction=fraction, od_model=model_set, od_transit_model=transit, scan_points=51)
            old_endpoint = line == "D1" and fraction == 1. and model_set != OD_MODELS_NTOTAL
            expected = pure if old_endpoint else general
            c = ODConditions.from_params(p)
            assert c.model_id == expected and od_unavailable(c) is None
            raw = scheme.compute(p)
            assert raw["probe_model"] == expected
            explicit = scheme.compute(dict(p, od_model=expected))
            np.testing.assert_array_equal(raw["ballistic_depth"], explicit["ballistic_depth"])
    pure85 = parameters(od_model=model_set, od_transit_model=transit)
    assert ODConditions.from_params(pure85).model_id == pure


def test_invalid_transit_selector_still_errors_with_pump_on():
    p = parameters(od_transit_model="unknown", pump_power_mw=1.)
    with pytest.raises(ODModelSettingError, match="transit"):
        SASScheme().compute(p)


def test_pure_custom_and_fit_conditions():
    p = parameters(species="Rb (natural)", rb_composition="Custom", rb85_fraction=1., probe_power_uw=100.)
    assert od_fit_unavailable(p) is None
    assert od_unavailable(ODConditions.from_params(dict(p, rb85_fraction=.999))) is None
    assert od_unavailable(ODConditions.from_params(dict(p, rb85_fraction=.999, od_model=BALLISTIC_REFERENCE)))
    assert not {"rb85_fraction", "probe_power_uw", "probe_waist_mm"}.intersection(offered_fit_parameters(p))


@pytest.mark.parametrize(("model_id", "route"), [
    (BALLISTIC_REFERENCE, "reference"), (BALLISTIC_FAST, "fast"), (BALLISTIC_FAST_V1, "fast_v1"),
])
def test_nonzero_power_scheme_uses_propagated_depth_only_once(monkeypatch, model_id, route):
    from gabes import ballistic_spectrum
    scheme = SASScheme()
    p = parameters(probe_power_uw=25., line_strength=.7, od_model=model_id)
    axis = np.linspace(-1., 1., 7) * 2*np.pi*1e9
    def sampled(conditions, scan, *, fidelity):
        assert conditions.model_id == model_id and fidelity == route
        assert conditions.cell_mm == 12.5 and conditions.line_strength == .7
        depth = .35 + .1 * np.cos(scan / 1e9)
        return ballistic_spectrum.BallisticSpectrum(np.exp(-depth), depth, depth*2, {"fidelity_used": fidelity})
    monkeypatch.setattr(ballistic_spectrum, "evaluate_ballistic_spectrum", sampled)
    raw = scheme._compute_species(p, scan_rad_s=axis)
    readout = scheme.headless_observables(raw, p)["optical_readout"]
    np.testing.assert_allclose(readout["transmission"], od_transmission(axis / (2*np.pi*1e6), p))
    assert readout["ballistic_diagnostics"]["fidelity_used"] == route
    assert readout["effective"]


@pytest.mark.parametrize(("transit", "model"), [
    (BALLISTIC_TRANSIT, BALLISTIC_REFERENCE),
    (BALLISTIC_FAST_TRANSIT, BALLISTIC_FAST),
    (BALLISTIC_FAST_TRANSIT, "od_rb85_d1_ballistic_history_fast_v1"),
])
def test_gabes_csv_roundtrip_preserves_history_without_reset(transit, model):
    from gabes_ui.od_measurement import geometry_replay_params, plot_metadata, workspace_params
    p = parameters(probe_power_uw=25., od_transit_model=transit, od_model=model)
    c = ODConditions.from_params(p)
    metadata = plot_metadata(p, c.model_id)
    upload = prepare_measurement_upload(b"frequency_GHz,transmission\n-1,0.96\n0,0.81\n1,0.93\n")
    trace = load_measurement(upload, mode="fraction", x_unit="GHz")
    saved = export_comparison_csv(upload, trace, [-1.,0.,1.], [.96,.81,.93], metadata=metadata)
    restored = prepare_measurement_upload(saved)
    assert restored.metadata["model_id"] == c.model_id
    assert restored.metadata["transit"]["rate_s_inverse"] == {}
    controls = geometry_replay_params(restored)
    assert controls["od_transit_model"] == transit
    replay = workspace_params(dict(SASScheme().defaults(), **controls), {}, restored)
    assert ODConditions.from_params(replay).model_id == model
    assert "probe_waist_mm" not in offered_fit_parameters(replay)
