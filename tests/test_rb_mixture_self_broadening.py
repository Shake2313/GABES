"""2026-11-26 mixture-density decision and archived OD/SAS model replay."""
from dataclasses import replace

import numpy as np
import pytest

from gabes import collisions, species
from gabes.od_model import ODConditions, evaluate_od, evaluate_od_display, is_rate_model
from gabes.schemes.sas import OD_MODEL_KEY, OD_MODELS_D2SB, RATE_MODEL_V3, SASScheme


@pytest.mark.parametrize("line", ["D1", "D2"])
@pytest.mark.parametrize("fraction", [0.5, species.RB85.abundance])
def test_uniform_mixture_has_pure_elemental_widths_and_partial_absorber_counts(line, fraction):
    scheme = SASScheme()
    p = dict(scheme.defaults(), species="Rb (natural)", line=line, temp_c=150., cell_mm=2.,
             scan_points=101, pump_power_mw=0., rb_composition="Custom", rb85_fraction=fraction)
    c = ODConditions.from_params(p)
    offsets = np.linspace(-4e9, 5e9, 201)
    # Single-isotope controls keep the original elementary response on the same frequency axis.
    controls = [ODConditions.from_params(dict(p, od_model=OD_MODELS_D2SB, species=iso.label))
                for iso in (species.RB85, species.RB87)]
    pure_depths = [evaluate_od(control, offsets).od0 for control in controls]
    expected = fraction * pure_depths[0] + (1 - fraction) * pure_depths[1]
    np.testing.assert_allclose(evaluate_od(c, offsets).od0, expected, rtol=3e-13)
    previous = ODConditions.from_params(dict(p, od_model=OD_MODELS_D2SB))
    assert np.max(abs(evaluate_od(previous, offsets).od0 - expected)) > .01
    # Rate-model weak limit and the projected reference use the same collision partners.
    np.testing.assert_allclose(evaluate_od(replace(c, model_id=RATE_MODEL_V3), offsets).od0,
                               expected, rtol=3e-13)
    widths = scheme.projected_probe_reference(p)["provenance"]["homogeneous_fwhm_mhz"]
    for iso, control in zip((species.RB85, species.RB87), controls, strict=True):
        pure = scheme.projected_probe_reference(dict(p, species=iso.label, od_model=control.model_id))
        assert widths[iso.name] == pure["provenance"]["homogeneous_fwhm_mhz"][iso.name]


def test_measured_uniform_collision_partners_and_segmented_legacy_limit():
    scheme = SASScheme()
    p = dict(scheme.defaults(), temp_c=150., line="D2", species="Rb (natural)",
             cell_mm=2., scan_points=101, pump_power_mw=0.)
    conditions = dict(p, optical_collision_model=collisions.REFERENCE_COLLISIONS)
    raw = scheme.compute(conditions)
    assert raw["probe_model"] == "weak_probe_grid_ntotal_v1"
    total = species.number_density(species.RB85, p["temp_c"] + 273.15)
    for iso in (species.RB85, species.RB87):
        report = raw["self_broadening_reference"][iso.name]
        assert report["density_m3"] == total
    old = scheme.compute(dict(conditions, od_model=OD_MODELS_D2SB))
    for iso in (species.RB85, species.RB87):
        assert old["self_broadening_reference"][iso.name]["density_m3"] == total * iso.abundance
    segmented = dict(p, cell_propagation="Segmented reference", axial_field_span_ut=10.)
    old = scheme.compute(dict(segmented, od_model=OD_MODELS_D2SB))
    new = scheme.compute(segmented)
    assert old["probe_model"] == new["probe_model"] == "weak_probe_legacy"
    for key in ("scan", "alpha_unit", "chi_real_unit"):
        np.testing.assert_array_equal(old[key], new[key])


@pytest.mark.parametrize(("changes", "model"), [
    ({"pump_power_mw": .5}, "sas_pumped_exact_voigt_transit_v3"),
    ({"probe_power_uw": 20.}, "od_finite_rb_mf_rate_v3"),
    ({"probe_power_uw": 20., "od_transit_model": "gaussian_fwhm"}, "od_finite_rb_mf_gaussian_transit_v2"),
    ({"probe_power_uw": 20., "ne_pressure_torr": .1}, "finite_power_2level_gaussian_propagation_v3"),
])
def test_pumped_and_finite_mixtures_select_new_widths_only_for_the_new_set(changes, model):
    scheme = SASScheme()
    p = dict(scheme.defaults(), temp_c=110., line="D2", cell_mm=12.5, scan_points=101, pump_power_mw=0.)
    p.update(changes)
    raw = scheme.compute(p)
    previous = scheme.compute(dict(p, od_model=OD_MODELS_D2SB))
    assert raw["probe_model"] == model
    assert raw["gamma_eff_max"] > previous["gamma_eff_max"]
    if changes.get("probe_power_uw"):
        view = scheme.headless_observables(raw, p)
        table = next(table for table in view["tables"] if table["title"] == "Finite-probe model")
        assert f"`{model}`" in table["markdown"]
    if is_rate_model(model):
        view = view["optical_readout"]
        c = ODConditions.from_params(p)
        np.testing.assert_allclose(view["transmission"], evaluate_od_display(c, raw["scan"]).transmission,
                                   rtol=3e-13, atol=1e-16)


def test_older_standalone_projected_reference_without_model_keeps_previous_width():
    scheme = SASScheme()
    params = dict(scheme.defaults(), line="D2", temp_c=110., pump_power_mw=0., scan_points=101)
    del params[OD_MODEL_KEY]
    recorded = scheme.projected_probe_reference(dict(params, od_model=OD_MODELS_D2SB))
    replay = scheme.projected_probe_reference(params)
    assert recorded["provenance"] == replay["provenance"]
    np.testing.assert_array_equal(recorded["transmission"], replay["transmission"])
