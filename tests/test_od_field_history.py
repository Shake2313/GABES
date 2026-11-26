"""Static-field OD inputs through the scheme, fitter, readout and comparison CSV."""
import numpy as np
import pytest

from gabes import species
from gabes.ballistic_spectrum import BallisticSpectrum
from gabes.core import blas_single_thread
from gabes.od_measurement import export_comparison_csv, load_measurement, prepare_measurement_upload
from gabes.od_model import ODConditions, ODUnsupportedError, evaluate_od, od_unavailable
from gabes.od_spectrum_fit import od_transmission
from gabes.schemes.sas import (
    BALLISTIC_REFERENCE, BALLISTIC_TRANSIT, FIELD_HISTORY_KEYS, FIELD_HISTORY_REFERENCE, FIELD_HISTORY_TRANSIT,
    OD_MODELS_NTOTAL, RB_HISTORY_REFERENCE, SASScheme,
)
from gabes_ui.od_measurement import geometry_replay_params, plot_metadata, workspace_params


def field_params(**changes):
    p = dict(SASScheme().defaults(), species=species.RB85.label, line="D1", pump_power_mw=0.,
             probe_power_uw=0., cell_mm=12.5, line_strength=.7, od_transit_model=FIELD_HISTORY_TRANSIT,
             od_field_x_ut=5., od_field_y_ut=-8., od_field_z_ut=23.,
             od_probe_azimuth_deg=32., od_probe_ellipticity_deg=-19.)
    return dict(p, **changes)


@pytest.mark.parametrize("medium,line,fraction,old", [
    (species.RB85.label, "D1", 1., BALLISTIC_REFERENCE), (species.RB87.label, "D2", 0., RB_HISTORY_REFERENCE),
    ("Rb (natural)", "D1", .38, RB_HISTORY_REFERENCE),
])
def test_field_opt_in_is_distinct_and_other_histories_ignore_retained_inputs(medium, line, fraction, old):
    p = field_params(species=medium, line=line, rb_composition="Custom", rb85_fraction=fraction)
    c = ODConditions.from_params(p)
    assert c.model_id == FIELD_HISTORY_REFERENCE
    assert od_unavailable(c) is None and c.transit_rates_s() == {}
    assert c.field_ut == (5., -8., 23.)
    ex, ey = c.probe_jones
    s1, s2, s3 = abs(ex)**2 - abs(ey)**2, 2*(ex.conjugate()*ey).real, 2*(ex.conjugate()*ey).imag
    assert np.linalg.norm((ex, ey)) == pytest.approx(1.)
    assert np.rad2deg(np.arctan2(s2, s1))/2 == pytest.approx(32.)
    assert np.rad2deg(np.arcsin(s3))/2 == pytest.approx(-19.)
    assert ODConditions.from_params(c.to_params()) == c
    previous = ODConditions.from_params(dict(p, od_transit_model=BALLISTIC_TRANSIT))
    assert previous.model_id == old
    assert previous.field_ut == (0., 0., 0.) and previous.probe_jones == (1., 0.)
    assert set(FIELD_HISTORY_KEYS).isdisjoint(previous.recorded_conditions())


@pytest.mark.parametrize("key,bad", [("od_field_x_ut", np.inf), ("od_field_y_ut", "7"),
                                     ("od_probe_azimuth_deg", True), ("od_probe_ellipticity_deg", 45.01)])
def test_reject_invalid_field_settings(key, bad):
    with pytest.raises(ODUnsupportedError):
        ODConditions.from_params(field_params(**{key: bad}))


def test_supported_cell_and_ui_visibility_follow_model_and_pump(monkeypatch):
    from gabes_ui import controls

    scheme = SASScheme()
    p = field_params()
    for updates in ({"species": species.CS133.label}, {"paraffin_coated": True}, {"ne_pressure_torr": 1.}):
        assert od_unavailable(ODConditions.from_params(dict(p, **updates)))
    assert set(FIELD_HISTORY_KEYS) <= set(scheme.solve_param_keys(p))
    specs = {sp.name: sp for sp in scheme.param_schema()}
    for changed, visible in [({}, True), ({"pump_power_mw": .05}, False), ({"od_transit_model": "fixed"}, False)]:
        params = dict(p, **changed)
        state = {controls.skey(scheme.name, key): val for key, val in params.items()}
        monkeypatch.setattr(controls.st, "session_state", state)
        assert all(controls.param_visible(scheme.name, specs[key]) is visible for key in FIELD_HISTORY_KEYS)
        assert set(FIELD_HISTORY_KEYS).issubset(scheme.solve_param_keys(params)) is visible
    # CSV mode derives a pump-off view without overwriting the saved SAS pump setting.
    state = {controls.skey(scheme.name, key): val for key, val in dict(p, pump_power_mw=.05).items()}
    monkeypatch.setattr(controls.st, "session_state", state)
    assert all(controls.param_visible(scheme.name, specs[key], {"pump_power_mw": 0.}) for key in FIELD_HISTORY_KEYS)
    p.update(pump_power_mw=.05, scan_points=51)
    fixed = dict(p, od_transit_model="fixed", od_model=OD_MODELS_NTOTAL)
    assert scheme.solve_param_keys(p) == scheme.solve_param_keys(fixed)
    with blas_single_thread():
        raw = scheme.compute({key: p[key] for key in scheme.solve_param_keys(p)})
        ref = scheme.compute({key: fixed[key] for key in scheme.solve_param_keys(fixed)})
    assert raw["probe_model"] == ref["probe_model"] and "ballistic_depth" not in raw
    np.testing.assert_array_equal(raw["alpha_unit"], ref["alpha_unit"])
    # Uploaded data are fitted with the pump off, independently of the displayed SAS power.
    assert ODConditions.from_params(p).model_id == FIELD_HISTORY_REFERENCE


@pytest.mark.parametrize("power,length", [(0., 12.5), (10., 12.5), (10., 0.)])
def test_field_readout_and_fit_use_requested_axis_and_field_dependent_weak_od(monkeypatch, power, length):
    from gabes import field_spectrum

    scheme = SASScheme()
    p = field_params(species=species.RB87.label, probe_power_uw=power, cell_mm=length)
    c = ODConditions.from_params(p)
    mhz = np.array([-3100., -2200., 210., 600., 3100.])

    def evaluate(conditions, scan):
        assert conditions.field_ut == c.field_ut and conditions.probe_jones == c.probe_jones
        od0 = conditions.cell_mm * conditions.line_strength * (.08 + .03 * np.cos(scan/1e9))
        depth = od0 / (1. + .02*conditions.probe_power_uw)
        return BallisticSpectrum(np.exp(-depth), depth, od0, {"fidelity_used": "reference"})

    monkeypatch.setattr(field_spectrum, "evaluate_field_ballistic_spectrum", evaluate)
    raw = scheme._compute_species(p, scan_rad_s=mhz*2*np.pi*1e6)
    view = scheme.headless_observables(raw, p)["optical_readout"]
    actual = evaluate(c, raw["scan"])
    assert raw["chi_real_unit"] is None
    from gabes_ui import od_measurement as measurement
    def unexpected_solve(*args):
        pytest.fail("A completed field solve must be reused by the measurement workspace")
    monkeypatch.setattr(measurement, "_spectrum", unexpected_solve)
    reopened = measurement.evaluate_workspace(dict(p, transit_khz=429.), raw)
    assert reopened.conditions.transit_khz == 429.
    np.testing.assert_array_equal(reopened.transmission, actual.transmission)
    if power and length:
        with pytest.raises(ODUnsupportedError, match="current conditions"):
            measurement.evaluate_workspace(dict(p, od_field_x_ut=19.), raw)
        with pytest.raises(ODUnsupportedError, match="current conditions"):
            measurement.evaluate_workspace(p, dict(raw, scan=raw["scan"][::-1]))
    np.testing.assert_allclose(view["optical_depth"], actual.optical_depth, rtol=2e-14)
    np.testing.assert_allclose(view["od0"], actual.od0, rtol=2e-14)
    order = [2, 0, 2, 4]
    # Public OD uses the 85Rb centroid; the fitter and display use the pure-87Rb origin.
    shift = species.RB87.line(c.line)[1] - species.RB85.line(c.line)[1]
    native = evaluate_od(c, mhz[order]*1e6 + shift)
    fit = od_transmission(mhz[order], dict(p, od_transit_model="fixed"), model_id=c.model_id)
    np.testing.assert_allclose(native.transmission, actual.transmission[order], rtol=1e-13)
    np.testing.assert_allclose(fit, actual.transmission[order], rtol=1e-13)


def test_field_csv_round_trip_uses_all_five_recorded_inputs_and_rejects_incomplete_csv():
    from gabes.experimental_csv import ExperimentalCSVError

    p = field_params()
    c = ODConditions.from_params(p)
    metadata = plot_metadata(p, c.model_id)
    upload = prepare_measurement_upload(b"frequency_GHz,transmission\n-1,0.96\n0,0.81\n1,0.93\n")
    trace = load_measurement(upload, mode="fraction", x_unit="GHz")

    def reopen(record):
        saved = export_comparison_csv(upload, trace, [-1., 0., 1.], [.96, .81, .93], metadata=record)
        return prepare_measurement_upload(saved)

    restored = reopen(metadata)
    controls = geometry_replay_params(restored)
    assert restored.metadata["conditions"] == c.recorded_conditions()
    assert controls["od_transit_model"] == FIELD_HISTORY_TRANSIT
    assert [controls[k] for k in FIELD_HISTORY_KEYS] == [p[k] for k in FIELD_HISTORY_KEYS]
    replay = workspace_params(dict(SASScheme().defaults(), **controls), {}, restored)
    assert ODConditions.from_params(replay) == c
    for key in FIELD_HISTORY_KEYS:
        incomplete = dict(metadata["conditions"])
        incomplete.pop(key)
        with pytest.raises(ExperimentalCSVError, match="static field or probe polarization"):
            reopen(dict(metadata, conditions=incomplete))
