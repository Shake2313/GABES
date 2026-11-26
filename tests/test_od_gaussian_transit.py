"""Single-beam Gaussian replacement: scheme, pointwise OD, fitting and replay."""
from dataclasses import replace

import numpy as np
import pytest

from gabes import species
from gabes.od_model import MODELS, ODConditions, evaluate_od_display, od_unavailable
from gabes.od_spectrum_fit import offered_fit_parameters, od_transmission
from gabes.schemes.sas import (
    GAUSSIAN_TRANSIT, GAUSSIAN_TRANSIT_D20_MODEL, GAUSSIAN_TRANSIT_MODEL,
    OD_MODELS_D20, OD_MODELS_D2SB, ODModelSettingError, RATE_MODEL, RATE_MODEL_V2, SASScheme,
)


@pytest.mark.parametrize("power_uw", [0., 25.])
@pytest.mark.parametrize(("setting", "geometry", "fixed"), [
    (OD_MODELS_D20, GAUSSIAN_TRANSIT_D20_MODEL, RATE_MODEL),
    (OD_MODELS_D2SB, GAUSSIAN_TRANSIT_MODEL, RATE_MODEL_V2),
])
def test_d2_spectrum_matches_its_own_fixed_width_recipe(setting, geometry, fixed, power_uw):
    scheme = SASScheme()
    p = dict(scheme.defaults(), pump_power_mw=0., temp_c=74., probe_waist_mm=.53,
             species=species.RB85.label, line="D2", cell_mm=12.5, od_model=setting,
             od_transit_model=GAUSSIAN_TRANSIT, probe_power_uw=power_uw)
    conditions = ODConditions.from_params(p)
    assert conditions.model_id == geometry and conditions.transit_metadata()["mode"] == GAUSSIAN_TRANSIT
    rate = conditions.transit_rates_s()["Rb85"]
    control = replace(conditions, model_id=fixed, transit_khz=rate / (2*np.pi*1e3))
    scan = 2*np.pi*1e9*np.array([-1.45, -.73, .12, .88, 1.72])
    a = evaluate_od_display(conditions, scan)
    b = evaluate_od_display(control, scan)
    np.testing.assert_allclose(od_transmission(scan / (2*np.pi*1e6), p), a.transmission, rtol=1e-12)
    for name in ("od0", "optical_depth", "transmission"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-12, atol=0)
    raw = scheme._compute_species(p, scan_rad_s=scan)
    view = scheme.headless_observables(raw, p)
    table = next(table for table in view["tables"] if table["title"] == "Finite-probe model")
    assert MODELS[geometry].accuracy in table["markdown"]
    assert MODELS[fixed].accuracy not in table["markdown"]
    display = view["optical_readout"]
    assert raw["probe_model"] == geometry and display["transit_rates_s"] == conditions.transit_rates_s()
    np.testing.assert_allclose(display["transmission"], a.transmission, rtol=2e-13)
    assert ODConditions.from_params(dict(p, od_model=fixed)).model_id == fixed
    assert "probe_power_uw" not in offered_fit_parameters(p)
    assert "probe_power_uw" in offered_fit_parameters(dict(p, od_model=fixed))


def test_isotope_rates_use_path_temperature_and_waist():
    c = ODConditions("Rb (natural)", "D1", 85., 12.5, GAUSSIAN_TRANSIT_MODEL,
                     doppler_temp_c=99., probe_waist_mm=.53)
    metadata = c.transit_metadata()
    assert metadata["kinetic_temperature_k"] == 372.15
    assert metadata["probe_waist_m"] == pytest.approx(.53e-3)
    rates = metadata["rate_s_inverse"]
    assert rates["Rb85"] / (2*np.pi*1e3) == pytest.approx(77.69, abs=.02)
    assert rates["Rb85"] / rates["Rb87"] == pytest.approx(np.sqrt(species.RB87.mass/species.RB85.mass))
    assert replace(c, temp_c=30.).transit_rates_s() == rates
    assert replace(c, probe_waist_mm=1.06).transit_rates_s()["Rb85"] == pytest.approx(rates["Rb85"]/2)
    assert replace(c, doppler_temp_c=40.).transit_rates_s()["Rb85"] == pytest.approx(
        rates["Rb85"]*np.sqrt(313.15/372.15))
    assert replace(c, rb_composition="Custom", rb85_fraction=0.).transit_rates_s().keys() == {"Rb87"}
    assert od_unavailable(c) is None
    scan = 2*np.pi*1e9*np.array([-.65, .35, 1.02])
    weak = evaluate_od_display(c, scan)
    tiny = evaluate_od_display(replace(c, probe_power_uw=1e-10), scan)
    np.testing.assert_allclose(weak.od0, tiny.od0, rtol=1e-13)
    np.testing.assert_allclose(weak.transmission, tiny.transmission, rtol=1e-7)
    np.testing.assert_array_equal(weak.transmission, evaluate_od_display(replace(c, transit_khz=500), scan).transmission)


@pytest.mark.parametrize("unsupported", [
    {"paraffin_coated": True}, {"ne_pressure_torr": 10.},
    {"species": species.CS133.label}, {"cell_propagation": "Segmented reference"},
])
def test_geometry_rejects_unsupported_conditions(unsupported):
    scheme = SASScheme()
    p = dict(scheme.defaults(), pump_power_mw=0., od_transit_model=GAUSSIAN_TRANSIT)
    p.update(unsupported)
    with pytest.raises(ODModelSettingError):
        scheme.compute(p)


def test_gaussian_weak_fit_recovers_temperature_and_reports_fitted_rates():
    from gabes.od_spectrum_fit import ODFitInput, fit_od_spectrum, od_transmission

    p = dict(SASScheme().defaults(), pump_power_mw=0., probe_power_uw=0., line="D1",
             temp_c=68., cell_mm=12.5, probe_waist_mm=.53, od_transit_model=GAUSSIAN_TRANSIT)
    x = np.linspace(-3000., 3000., 121)
    truth = dict(p, temp_c=65.)
    y = od_transmission(x, truth)
    fit = fit_od_spectrum(ODFitInput(x, y, np.full_like(x, .001), 0, len(x), "test uncertainty"),
                          p, free=("temp_c",))
    assert fit["probe_model"] == GAUSSIAN_TRANSIT_MODEL
    assert fit["parameters"][0]["value"] == pytest.approx(65., abs=1e-7)
    assert fit["transit"]["rate_s_inverse"] == pytest.approx(ODConditions.from_params(truth).transit_rates_s())
    ref = SASScheme().projected_probe_reference(truth)
    assert ref["provenance"]["transit_rates_s"] == pytest.approx(fit["transit"]["rate_s_inverse"])
