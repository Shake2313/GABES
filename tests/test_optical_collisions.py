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
