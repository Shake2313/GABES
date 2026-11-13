"""Shared photodetection: photon energy/flux, responsivity, shot noise."""

import math

import numpy as np
import pytest


from gabes import constants, detection, jasperse, magneto_noise  # noqa: E402
from gabes import rydberg_electrometry  # noqa: E402

EXACT_SI_PLANCK = 6.62607015e-34


def test_rydberg_electrometry_reexports_the_detector_model():
    for name in ("PhotodiodeChannel", "BalancedDetector", "DetectorNoiseBudget",
                 "balanced_detector_noise", "photodiode_responsivity_a_per_w", "asd_to_rms"):
        assert getattr(rydberg_electrometry, name) is getattr(detection, name), name


def test_detection_relations_against_an_independent_photon_count():
    # Independent reference: a coherent beam of power P delivers P/(h nu) photons
    # per second, each giving one electron at unit efficiency, with Poisson
    # statistics; every helper must follow from that one picture.
    nu = constants.C_LIGHT / 780.241e-9
    omega = 2.0 * math.pi * nu
    power = np.array([1e-9, 3.7e-6, 2.5e-3])
    flux = detection.photon_flux(power, nu)
    np.testing.assert_allclose(flux * EXACT_SI_PLANCK * nu, power, rtol=1e-9)
    np.testing.assert_allclose(detection.photon_flux(power, nu, 0.8), 0.8 * flux, rtol=1e-15)
    responsivity = detection.photodiode_responsivity_a_per_w(1.0, constants.C_LIGHT / nu)
    current = responsivity * power
    np.testing.assert_allclose(current, constants.ELEMENTARY_CHARGE * flux, rtol=1e-12)
    # S_i = 2 e I equals R^2 times the optical-power shot PSD 2 hbar omega P.
    np.testing.assert_allclose(detection.shot_noise_current_psd(current),
                               responsivity**2 * detection.optical_shot_power_psd(power, omega),
                               rtol=1e-12)
    np.testing.assert_allclose(detection.optical_shot_rin(power, omega),
                               detection.optical_shot_power_psd(power, omega) / power**2, rtol=1e-12)


def test_callers_use_the_shared_detection():
    nu = 377.107385690e12
    sens = magneto_noise.magnetic_sensitivity([0.9], [1e5], frequency_hz=nu, intensity_mw_cm2=1.0)
    assert sens["photon_energy_j"] == detection.photon_energy(nu)
    omega = 2.0 * math.pi * nu
    seed = jasperse.seed_noise_components(2e-4, omega)
    assert seed["shot_rin_per_Hz"] == detection.optical_shot_rin(2e-4, omega)
    assert seed["shot_power_psd_W2_per_Hz"] == detection.optical_shot_power_psd(2e-4, omega)


def test_exact_si_planck_value_is_shared_by_fwm_and_sabes():
    from gabes.schemes import fwm
    from sabes.instruments import base

    assert constants.PLANCK == EXACT_SI_PLANCK
    assert constants.HBAR == EXACT_SI_PLANCK / (2.0 * np.pi)
    assert fwm.responsivity_AW(0.58, 794.979) == pytest.approx(
        0.58 * 1.602176634e-19 * 794.979e-9 / (EXACT_SI_PLANCK * 299792458.0),
        rel=1e-15)
    assert base.quantum_efficiency(0.58, 795.0) == pytest.approx(
        0.9045388061793227, rel=1e-15)
    preset = next(p for p in fwm.FWMScheme().presets() if "2026-10-13" in p.name)
    assert preset.values["qe_pct"] == pytest.approx(90.45388061793227, rel=1e-15)
