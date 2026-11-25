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
