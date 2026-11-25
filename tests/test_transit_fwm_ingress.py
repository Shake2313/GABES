"""FWM transit ingress points keep their historical arithmetic (transit-time stage A).

The seeded main path computed 2π·(x·1e3); the full spectrum and the Zeeman gain
context computed (2π·x)·1e3. The two orders differ in the last bit for some
inputs (100 kHz among them), so each site keeps its own order.
"""
import math

import numpy as np
import pytest

from gabes import fwm_zeeman_gain, transit
from gabes.schemes import fwm as fwm_pkg

VALUES = (100.0, 37.3, 123.456789, 0.1, 1999.97, 1e-3, 7.0, 0.333, 2.0 / 3.0, 1234.5678)


def _seeded(x):
    return 2.0 * np.pi * (x * 1e3)


def _khz_first(x):
    return 2.0 * np.pi * x * 1e3


def test_the_two_historical_orders_really_differ():
    assert any(_seeded(x) != _khz_first(x) for x in VALUES)


@pytest.mark.parametrize("x", VALUES)
def test_helpers_repeat_each_order_bit_for_bit(x):
    assert transit.rate_from_khz_over_2pi(x) == _khz_first(x)
    if hasattr(transit, "rate_from_hz_over_2pi"):
        assert transit.rate_from_hz_over_2pi(x * 1e3) == _seeded(x)


@pytest.mark.parametrize("x", VALUES)
def test_each_fwm_site_keeps_its_order(x, monkeypatch):
    scheme = fwm_pkg.FWMScheme()
    params = {**scheme.defaults(), "transit_rate_khz": x}
    assert scheme._seeded_compute_inputs(params)["transit_rate"] == _seeded(x)

    recorded = {}
    monkeypatch.setattr(fwm_pkg, "full_spectrum", lambda *a, **k: recorded.update(k) or {})
    view = next(v for v in scheme.extra_views() if "Full probe scan" in v.key)
    view.compute(params)
    assert recorded["transit_rate"] == _khz_first(x)

    context = fwm_zeeman_gain.context_from_params(params)
    assert context["transit_reset_rate_rad_s"] == _khz_first(x)
    assert math.isfinite(context["transit_reset_rate_rad_s"])
