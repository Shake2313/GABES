"""Source scopes, sign, zero-pressure limits and public reference routing."""

import numpy as np
import pytest

from gabes import atoms, collisions, species
from gabes.schemes.absorption import LambdaScheme, ODScheme
from gabes.schemes.magneto_unified import BUFFER, UnifiedMagnetoScheme
from gabes.schemes.sas import LEGACY_WEAK_PROBE, SASScheme
from gabes.schemes.sas_axial import AxialInputError, SEGMENTED


def test_pressure_coefficients_and_unsupported_domain():
    effects = collisions.pressure_effects("Rb87", "D1", 10.0, 393.15)
    report = effects.readout()
    assert report["shift_mhz"] == pytest.approx(-11.0)
    assert report["fwhm_mhz"] == pytest.approx(100.0)
    assert report["optical_dephasing_mhz"] == pytest.approx(50.0)
    assert report["shift_uncertainty_mhz"] > 2.0
    assert report["fwhm_uncertainty_mhz"] > 20.0
    assert len(report["warnings"]) == 2
    extrapolated = collisions.pressure_effects("Rb87", "D1", 100.0, 300.0).readout()
    assert any("Pressure outside" in warning for warning in extrapolated["warnings"])
    assert any("Temperature transfer" in warning for warning in extrapolated["warnings"])
    assert report["shift_mhz"] != collisions.pressure_effects(
        "Rb87", "D2", 10.0, 393.15).readout()["shift_mhz"]
    unsupported = collisions.pressure_effects("K39", "D1", 10.0, 393.15)
    assert unsupported.status == "unsupported"
    assert unsupported.shift_rad_s == unsupported.optical_dephasing_rad_s == 0.0
    assert collisions.pressure_effects("Rb87", "D1", 10.0, 393.15, gas="Ar").status == "unsupported"
    zero = collisions.pressure_effects("Rb87", "D1", 0.0, 393.15)
    assert zero.shift_rad_s == zero.optical_dephasing_rad_s == 0.0
    with pytest.raises(ValueError):
        collisions.pressure_effects("Rb87", "D1", -1.0, 393.15)
    density = 1e20
    d1, _ = collisions.self_broadening_reference("Rb85", "D1", density)
    d2, _ = collisions.self_broadening_reference("Rb85", "D2", density)
    assert d1 == pytest.approx(species.self_broadening(species.RB85, density))
    assert d2 > d1
    assert collisions.self_broadening_reference("Cs133", "D2", density)[1]["status"] == "unsupported"


def test_shared_elastic_projector_has_no_radiative_or_ground_term():
    bare = atoms.lambda3()
    rate = 25e6
    assert bare.with_optical_dephasing(0.0) is bare
    changed = bare.with_optical_dephasing(rate)
    assert bare.decay == changed.decay
    assert bare.dephasing == changed.dephasing
    increment = changed.lindblad - bare.lindblad
    expected = np.zeros_like(increment)
    for g in bare.ground:
        for i in (bare.rho_index(g, 2), bare.rho_index(2, g)):
            expected[i, i] = -rate
    np.testing.assert_allclose(increment, expected)


def test_od_single_peak_moves_by_the_signed_shift_and_zero_is_exact():
    scheme = ODScheme()
    p = dict(scheme.defaults(), model="single 2-level", doppler="off")
    before = scheme.compute(p)
    p["optical_collision_model"] = collisions.REFERENCE_COLLISIONS
    zero = scheme.compute(p)
    np.testing.assert_array_equal(zero["chi_bar"], before["chi_bar"])
    np.testing.assert_array_equal(zero["scan"], before["scan"])
    p["ne_pressure_torr"] = 10.0
    raw = scheme.compute(p)
    peak = raw["scan"][np.argmin(raw["chi_bar"].imag)] / species.MHZ
    assert peak == pytest.approx(-11.0)
    assert scheme.headless_observables(raw, p)["optical_collisions"]["fwhm_mhz"] == pytest.approx(100.0)


def test_lambda_common_optical_shift_leaves_raman_scan_coordinate():
    scheme = LambdaScheme()
    p = dict(scheme.defaults(), doppler="off", optical_collision_model=collisions.REFERENCE_COLLISIONS)
    zero = scheme.compute(p)
    raw = scheme.compute(dict(p, ne_pressure_torr=10.0))
    np.testing.assert_array_equal(raw["scan"], zero["scan"])
    assert raw["Dc"] == zero["Dc"]
    assert raw["optical_collisions"]["shift_mhz"] == pytest.approx(-11)
    assert not np.array_equal(raw["chi_bar"], zero["chi_bar"])


def test_magneto_reference_reaches_both_public_axes():
    scheme = UnifiedMagnetoScheme()
    p = dict(scheme.defaults(), doppler="off", scan_points=3, detuning_points=3,
             cell_type=BUFFER, use_ne_optical_reference=True, ne_pressure_torr=10.0)
    raw = scheme.compute(p)
    optical = scheme.compute(dict(p, scan_axis="Optical detuning"))
    assert raw["conditions"]["elastic_hwhm"] == pytest.approx(50 * species.MHZ)
    assert raw["conditions"]["optical_shift_mhz"] == pytest.approx(-11)
    assert raw["optical_collisions"] == optical["optical_collisions"]
    assert scheme.headless_observables(raw, p)["magneto_readout"]["optical_collisions"] == raw["optical_collisions"]


def test_sas_pressure_applies_same_width_and_shift_to_pump_and_probe(monkeypatch):
    scheme = SASScheme()
    p = dict(scheme.defaults(), species=species.RB87.label, line="D1", scan_points=161)
    legacy = scheme.compute(dict(p, pump_power_mw=0.0, weak_probe_profile=LEGACY_WEAK_PROBE))
    zero = scheme.compute(dict(p, pump_power_mw=0.0,
                               optical_collision_model=collisions.REFERENCE_COLLISIONS))
    np.testing.assert_array_equal(zero["alpha_unit"], legacy["alpha_unit"])
    np.testing.assert_array_equal(zero["scan"], legacy["scan"])
    p.update(optical_collision_model=collisions.REFERENCE_COLLISIONS,
             ne_pressure_torr=10.0, pump_power_mw=0.1)

    original = scheme._component_response
    observed = []
    def measure(component, scan, temperature):
        observed.append(component)
        return original(component, scan, temperature)
    monkeypatch.setattr(scheme, "_component_response", measure)
    raw = scheme.compute(p)
    assert len(observed) == 1
    component = observed[0]
    atom = component["pump_atom"]
    bare = component["man"].atom
    assert atom.decay == bare.decay
    offdiag = bare.rho_index(bare.ground[0], bare.excited[0])
    damping = -(atom.lindblad - bare.lindblad)[offdiag, offdiag].real
    assert 2.0 * damping == pytest.approx(component["gamma_eff"] - component["man"].gamma)
    assert raw["markers"][0][0] - zero["markers"][0][0] == pytest.approx(-.011)
    assert raw["optical_collisions"]["Rb87"]["fwhm_mhz"] == pytest.approx(100.0)
    assert scheme.headless_observables(raw, p)["optical_readout"]["optical_collisions"] == raw["optical_collisions"]


def test_sas_segmented_ne_scope_is_explicit():
    scheme = SASScheme()
    p = dict(scheme.defaults(), cell_propagation=SEGMENTED,
             optical_collision_model=collisions.REFERENCE_COLLISIONS, ne_pressure_torr=10.0)
    with pytest.raises(AxialInputError, match="uniform-cell propagation"):
        scheme.compute(p)


@pytest.mark.parametrize(("line", "slope", "quoted", "bound"), [
    ("D1", 5.7e-13, 1.0e-13, 0.2297),
    ("D2", 6.7e-13, 1.1e-13, 0.2179),
])
def test_cs_reference_uses_angular_fwhm_and_separate_spread_bound(line, slope, quoted, bound):
    density = 1e20
    width, report = collisions.self_broadening_reference("Cs133", line, density, cs_reference=True)
    assert width == pytest.approx(slope * density)
    assert report["coefficient_rad_s_m3"] == slope
    assert report["measurement_uncertainty_rad_s_m3"] == quoted
    assert report["relative_uncertainty"] == bound
    assert report["uncertainty_rad_s"] == pytest.approx(bound * width)
    assert f"{bound:.2%}" in report["warning"]
    assert collisions.self_broadening_reference("Cs133", line, 0, cs_reference=True)[0] == 0
    assert collisions.self_broadening_reference("Cs133", line, density)[0] == 0
    assert collisions.self_broadening_reference("Rb85", line, density, cs_reference=True) == (
        collisions.self_broadening_reference("Rb85", line, density))


@pytest.mark.parametrize("mode", collisions.COLLISION_CHOICES)
def test_cs_reference_independent_of_ne_old_settings_and_public_readout(mode, monkeypatch):
    scheme = SASScheme()
    control = next(spec for spec in scheme.param_schema() if spec.name == "cs_self_broadening_reference")
    assert control.default is False and control.recompute and control.visible_if == {"species": species.CS133.label}
    p = dict(scheme.defaults(), species=species.CS133.label, line="D2", temp_c=90.,
             ne_pressure_torr=5., optical_collision_model=mode, pump_power_mw=.2,
             probe_power_uw=5., scan_points=101)
    legacy = p.copy()
    legacy.pop("cs_self_broadening_reference")
    assert "cs_self_broadening_reference" not in scheme.solve_param_keys(legacy)
    assert "cs_self_broadening_reference" in scheme.solve_param_keys(p)
    components = []
    original = scheme._component_response

    def capture(component, *args, **kwargs):
        components.append(component)
        return original(component, *args, **kwargs)

    monkeypatch.setattr(scheme, "_component_response", capture)
    old, off, on = [scheme.compute(params) for params in (legacy, p, dict(p, cs_self_broadening_reference=True))]
    np.testing.assert_array_equal(old["alpha_unit"], off["alpha_unit"])
    assert old["self_broadening"] == off["self_broadening"]
    assert old["markers"] == on["markers"]
    assert on.get("optical_collisions") == old.get("optical_collisions")
    width = on["self_broadening"]["applied_fwhm_rad_s"]
    assert width == pytest.approx(6.7e-13 * components[-1]["N"])
    assert on["gamma_eff_max"] - off["gamma_eff_max"] == pytest.approx(width)
    off_atom, on_atom = [component["pump_atom"] for component in components[-2:]]
    assert on_atom.decay == off_atom.decay
    optical = off_atom.rho_index(off_atom.ground[0], off_atom.excited[0])
    assert (off_atom.lindblad - on_atom.lindblad)[optical, optical] == pytest.approx(width / 2)
    view = scheme.headless_observables(on, dict(p, cs_self_broadening_reference=True))
    assert view["optical_readout"]["self_broadening"] == on["self_broadening"]
    metric = next(row for row in view["metrics"] if row["label"] == "Self-broadening")
    assert "high uncertainty" in metric["value"] and "21.79%" in metric["value"]
    assert "inter-study spread" in metric["help"]
    assert "self-broadening contribution is omitted" not in str(view)


def test_segmented_cs_reference_uses_local_density_on_the_gridded_profile():
    scheme = SASScheme()
    p = dict(scheme.defaults(), species=species.CS133.label, line="D1",
             cs_self_broadening_reference=True, pump_power_mw=0., temp_c=90.,
             constrain_doppler_temp=False, doppler_temp_c=110., axial_segments=2,
             axial_temperature_span_c=30., scan_points=101, cell_propagation=SEGMENTED)
    raw = scheme.compute(p)
    axial = raw["axial_cell"]
    width = axial["cs_self_fwhm_rad_s"]
    np.testing.assert_allclose(width, 5.7e-13 * axial["density_m3"]["Cs133"])
    assert width[0] > width[1]
    # Independent local uniform responses at the same two midpoints and detunings.
    local = [scheme._compute_species(dict(p, cell_propagation="Uniform", doppler_temp_c=float(temp)),
                                     scan_rad_s=raw["scan"])["alpha_unit"]
             for temp in axial["temperature_c"]]
    np.testing.assert_allclose(raw["alpha_unit"], np.mean(local, axis=0), rtol=1e-12)
