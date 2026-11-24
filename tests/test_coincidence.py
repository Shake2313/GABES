"""
Ideal-canonical fixtures and explicitly unverified gain-only comparisons.
Production mean-field gains alone do not establish physical nonclassicality.

    python -m pytest -q tests/test_coincidence.py
"""

import numpy as np
import pytest


from gabes import observables  # noqa: E402
from gabes.schemes import fwm  # noqa: E402


def test_ideal_two_mode_squeezer():
    r = 0.8
    G_s = np.array([np.cosh(r) ** 2])
    G_c = np.array([np.sinh(r) ** 2])
    c = observables.coincidence_stats(G_s, G_c, ideal_canonical=True)
    n = np.sinh(r) ** 2                      # mean photons per mode
    assert np.isclose(c["n_s"][0], n)        # G_s − 1 = sinh²r
    assert np.isclose(c["g2_ss"][0], 2.0) and np.isclose(c["g2_cc"][0], 2.0)
    assert np.isclose(c["g2_sc"][0], 2.0 + 1.0 / n)        # textbook TMSV
    assert np.isclose(c["cauchy_schwarz"][0], (2.0 + 1.0 / n) ** 2 / 4.0)
    assert c["cauchy_schwarz"][0] > 1.0                    # nonclassical
    assert c["evidence"]["ideal_fixture_opt_in"]
    assert c["ideal_nonclassicality"][0]
    assert not c["evidence"]["physical_nonclassicality_available"]
    assert c["physical_nonclassicality"] is None


def test_fwm_spectrum_coincidence_is_an_unverified_gain_only_comparison():
    center = fwm.branch_center_GHz(0.9, -1)
    spec = fwm.compute_spectrum(
        0.9, T=394.15, P_pump=0.6, P_probe=8e-6, line_strength=1.0, loss_frac=0.055,
        coarse_points=121, fine_points=0, scan_min=center - 0.55, scan_max=center + 0.55,
        velocity_step=5.0, velocity_cutoff=3.0, branch=-1)
    c = observables.coincidence_stats(spec["G_s"], spec["G_c"])
    gain = spec["G_s"] > 1.0
    assert gain.any()                                       # there is a gain region
    # Preserve compatibility algebra without treating its built-in violation
    # as evidence about the noncanonical production transfer.
    np.testing.assert_allclose(c["g2_sc"][gain], 2.0 + 1.0/(spec["G_s"][gain]-1.0))
    assert np.all(np.isnan(c["g2_sc"][~gain]))              # undefined without gain
    imax = int(np.argmax(spec["G_s"]))                      # peak gain
    assert np.isfinite(c["g2_sc"][imax])
    assert c["evidence"]["origin"] == "legacy_gain_only_comparison"
    assert not c["evidence"]["ideal_nonclassicality_available"]
    assert not c["evidence"]["physical_nonclassicality_available"]
    assert c["ideal_nonclassicality"] is None
    assert c["physical_nonclassicality"] is None


def test_ideal_opt_in_rejects_inconsistent_or_nonfinite_gains():
    for probe, conjugate in ((3.0, 1.0), (0.5, 0.0), (np.inf, 1.0), (2.0, -1.0)):
        with pytest.raises(ValueError, match="Ideal canonical fixture"):
            observables.coincidence_stats(probe, conjugate, ideal_canonical=True)


def test_canonical_gain_relation_does_not_silently_opt_in():
    result = observables.coincidence_stats(np.array([2.0]), np.array([1.0]))
    assert result["ideal_nonclassicality"] is None
    assert not result["evidence"]["ideal_fixture_opt_in"]


def test_detected_moments_drive_counts_and_arm_correlations():
    vacuum = observables.ideal_twin_beam_moments(4., .4, .8)
    stats = observables.coincidence_stats(moments=vacuum)
    assert stats["n_s"] == pytest.approx(1.2)
    assert stats["n_c"] == pytest.approx(2.4)
    assert stats["g2_ss"] == pytest.approx(2.)
    assert stats["g2_sc"] == pytest.approx(2 + 1/3)
    seeded = observables.ideal_twin_beam_moments(
        4., .4, .8, seed_photons=3., env_photons_c=.2)
    result = observables.coincidence_stats(moments=seeded)
    assert result["g2_ss"] < 2 and result["g2_cc"] < 2
    assert result["evidence"]["environment"] == "thermal_attenuation"
    assert result["physical_nonclassicality"] is None
    dark = observables.coincidence_stats(moments=observables.ideal_twin_beam_moments(4., 0., .8))
    assert not dark["populated_arms"] and np.isnan(dark["g2_sc"])
    with pytest.raises(ValueError, match="absolute photon-count"):
        observables.coincidence_stats(moments=observables.ideal_twin_beam_moments(
            4., bright_seed_limit=True))
    with pytest.raises(ValueError, match="moments or gain"):
        observables.coincidence_stats(4., 3., moments=vacuum)
