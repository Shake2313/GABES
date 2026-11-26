"""Beam-crossing history through the scheme, OD fitter and saved worksheet."""
from dataclasses import replace

import numpy as np
import pytest

from gabes import species
from gabes.od_measurement import export_comparison_csv, load_measurement, prepare_measurement_upload
from gabes.od_model import ODConditions, evaluate_od_display, od_unavailable
from gabes.od_spectrum_fit import od_fit_unavailable, od_transmission, offered_fit_parameters
from gabes.schemes.sas import (
    BALLISTIC_FAST, BALLISTIC_FAST_TRANSIT, BALLISTIC_REFERENCE, BALLISTIC_TRANSIT, EXACT_MODEL, ODModelSettingError, SASScheme,
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
    {"pump_power_mw": 1.}, {"ne_pressure_torr": 5.}, {"paraffin_coated": True},
    {"cell_propagation": "Segmented reference"}, {"probe_waist_mm": 0.},
])
def test_history_scope_rejects_before_solve(conditions):
    with pytest.raises(ODModelSettingError):
        SASScheme().compute(parameters(**conditions))


def test_pure_custom_and_fit_conditions():
    p = parameters(species="Rb (natural)", rb_composition="Custom", rb85_fraction=1., probe_power_uw=100.)
    assert od_fit_unavailable(p) is None
    assert od_unavailable(ODConditions.from_params(dict(p, rb85_fraction=.999))) is not None
    assert not {"rb85_fraction", "probe_power_uw", "probe_waist_mm"}.intersection(offered_fit_parameters(p))


def test_nonzero_power_scheme_uses_propagated_depth_only_once(monkeypatch):
    from gabes import ballistic_spectrum
    scheme = SASScheme()
    p = parameters(probe_power_uw=25., line_strength=.7)
    axis = np.linspace(-1., 1., 7) * 2*np.pi*1e9
    def sampled(conditions, scan, *, fidelity):
        assert conditions.model_id == BALLISTIC_REFERENCE and fidelity == "reference"
        assert conditions.cell_mm == 12.5 and conditions.line_strength == .7
        depth = .35 + .1 * np.cos(scan / 1e9)
        return ballistic_spectrum.BallisticSpectrum(np.exp(-depth), depth, depth*2, {"fidelity_used": fidelity})
    monkeypatch.setattr(ballistic_spectrum, "evaluate_ballistic_spectrum", sampled)
    raw = scheme._compute_species(p, scan_rad_s=axis)
    readout = scheme.headless_observables(raw, p)["optical_readout"]
    np.testing.assert_allclose(readout["transmission"], od_transmission(axis / (2*np.pi*1e6), p))
    assert readout["ballistic_diagnostics"]["fidelity_used"] == "reference"
    assert readout["effective"]


@pytest.mark.parametrize("transit", [BALLISTIC_TRANSIT, BALLISTIC_FAST_TRANSIT])
def test_gabes_csv_roundtrip_preserves_history_without_reset(transit):
    from gabes_ui.od_measurement import geometry_replay_params, plot_metadata
    p = parameters(probe_power_uw=25., od_transit_model=transit)
    c = ODConditions.from_params(p)
    metadata = plot_metadata(p, c.model_id)
    upload = prepare_measurement_upload(b"frequency_GHz,transmission\n-1,0.96\n0,0.81\n1,0.93\n")
    trace = load_measurement(upload, mode="fraction", x_unit="GHz")
    saved = export_comparison_csv(upload, trace, [-1.,0.,1.], [.96,.81,.93], metadata=metadata)
    restored = prepare_measurement_upload(saved)
    assert restored.metadata["model_id"] == c.model_id
    assert restored.metadata["transit"]["rate_s_inverse"] == {}
    assert geometry_replay_params(restored)["od_transit_model"] == transit
