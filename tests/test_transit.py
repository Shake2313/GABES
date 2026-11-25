"""Transit input conventions used by saved schemes."""
import math

import numpy as np
import pytest

from gabes import constants, species, transit


def test_manual_rate_and_raw_rate_have_distinct_time_constants():
    assert transit.rate_from_khz_over_2pi(0.) == 0.
    assert 1e6 / transit.rate_from_khz_over_2pi(100.) == pytest.approx(1.59154943092)
    atom = species.build_manifold(species.RB85, "D1").atom
    coherence = np.zeros((atom.n_levels, atom.n_levels))
    coherence[0, 1] = 1.
    for rate, lifetime_us in [(100_000., 10.), (transit.rate_from_khz_over_2pi(100.), 1.59154943)]:
        replaced = species.build_manifold(species.RB85, "D1", transit_rate=rate).atom
        decay = (replaced.lindblad - atom.lindblad) @ coherence.ravel()
        assert -1e6 / decay.reshape(coherence.shape)[0, 1].real == pytest.approx(lifetime_us)


def test_rydberg_rate_preserves_the_most_probable_speed_convention():
    temp, diameter, mass = 310., 0.5e-3, constants.MASS_85RB
    expected = 0.6 * math.sqrt(2. * constants.KB * temp / mass) / diameter
    assert transit.most_probable_transit_rate_s(temp, diameter, mass, factor=.6) == expected


def test_gaussian_rate_matches_beam_chord_and_thermal_scaling():
    waist, temp, mass = 530e-6, 372.15, species.RB85.mass
    rate = transit.gaussian_transit_rate_s(waist, temp, mass)
    # Rayleigh transverse speed and mean chord through the half-intensity disk.
    v_perp = math.sqrt(math.pi * constants.KB * temp / (2. * mass))
    chord = math.pi / 4. * waist * math.sqrt(2. * math.log(2.))
    assert rate == pytest.approx(v_perp / chord, rel=1e-14)
    assert 1e6 / rate == pytest.approx(2.04853, rel=1e-5)
    assert transit.gaussian_transit_rate_s(waist * 2, temp * 4, mass) == pytest.approx(rate)
    assert transit.gaussian_transit_rate_s(waist, temp, species.RB87.mass) == pytest.approx(
        rate * math.sqrt(mass / species.RB87.mass))


@pytest.mark.parametrize(("waist", "temp", "mass"), [(0., 372., 1.), (1., -1., 1.), (1., 1., math.nan)])
def test_invalid_geometry_is_reported(waist, temp, mass):
    with pytest.raises(ValueError, match="finite and positive"):
        transit.gaussian_transit_rate_s(waist, temp, mass)
