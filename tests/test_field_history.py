"""Static-field beam crossings (gabes.field_history).

Recorded values come from runs/od_history_2026_11_26/controls/obe.py (85Rb D1,
99 C, w = 0.53 mm, 100 uW, b = 0, v_perp = sigma_v, nt = 1536), an
independent script that built the same signed 24-level density matrix by hand.
"""
import math

import numpy as np
import pytest

from gabes import field_history as fh, species
from gabes.ballistic_pumping import BallisticProbe85D1
from gabes.schemes.sas import _species_path_density

TEMP = 372.15
WAIST = 0.00053
INTENSITY = 2e-4 / (math.pi * WAIST * WAIST)
RESONANT = -1114.22976625   # rest-frame MHz, resonant class of the left inner edge

# (rest MHz, B along k in uT) -> recorded absorbed photons at nt = 1536
RECORDED = {
    (-1114.22976625, 0.0): 0.8139761183660889,
    (-1114.22976625, 50.0): 1.0507654366032402,
    (1559.92167275, 0.0): 0.4047489568653562,
    (1559.92167275, 50.0): 0.5363321324081975,
}


@pytest.fixture(scope="module")
def setting():
    iso = species.RB85
    density = _species_path_density(iso, TEMP, TEMP, 1.0)
    hcoll = species.self_broadening(iso, density, "D1", per_line=True) / 2
    return hcoll, BallisticProbe85D1(TEMP).speed_sigma


def _photons(setting, rest, field=(0.0, 0.0, 0.0), jones=(1.0, 0.0), steps=384, quantization="k",
             intensity=INTENSITY):
    hcoll, speed = setting
    ops = fh.operators("Rb85", "D1", rest, field_ut=field, jones=jones,
                       optical_hwhm_increment_rad_s=hcoll, quantization=quantization)
    return fh.crossing(ops, intensity, WAIST, speed, steps=steps)


@pytest.mark.parametrize("key", sorted(RECORDED))
def test_reproduces_the_independent_obe_control(setting, key):
    rest, field = key
    # 768 steps: the 768 -> 1536 change is ~5e-10 relative (runs/field_history_2026_11_26).
    result = _photons(setting, rest, (0.0, 0.0, field), steps=768)
    assert result["photons"] == pytest.approx(RECORDED[key], rel=1e-6)
    assert result["trace_error"] < 1e-12
    assert result["min_population"] > -1e-12


def test_field_along_k_raises_resonant_absorption_by_the_recorded_amount(setting):
    zero = _photons(setting, RESONANT)["photons"]
    perp = _photons(setting, RESONANT, (0.0, 0.0, 50.0))["photons"]
    assert perp / zero - 1 == pytest.approx(0.2909, abs=2e-3)


@pytest.mark.parametrize("jones, parallel, perpendicular", [
    ((1.0, 0.0), (50.0, 0.0, 0.0), (0.0, 50.0, 0.0)),
    ((0.0, 1.0), (0.0, 50.0, 0.0), (50.0, 0.0, 0.0)),
    ((1.0, 1.0), (35.0, 35.0, 0.0), (35.0, -35.0, 0.0)),
])
def test_field_parallel_to_linear_light_does_not_mix(setting, jones, parallel, perpendicular):
    zero = _photons(setting, RESONANT, jones=jones)["photons"]
    assert _photons(setting, RESONANT, parallel, jones)["photons"] == pytest.approx(zero, rel=1e-6)
    assert _photons(setting, RESONANT, perpendicular, jones)["photons"] / zero - 1 > 0.2


@pytest.mark.parametrize("field", [(50.0, 0.0, 0.0), (0.0, 50.0, 0.0), (30.0, 20.0, -40.0)])
def test_beam_axis_and_polarization_axis_quantization_agree(setting, field):
    k = _photons(setting, RESONANT, field)["photons"]
    e = _photons(setting, RESONANT, field, quantization="E")["photons"]
    assert e == pytest.approx(k, rel=1e-9)


def test_weak_drive_is_linear(setting):
    low = _photons(setting, RESONANT, intensity=INTENSITY * 1e-8)["photons"]
    double = _photons(setting, RESONANT, intensity=INTENSITY * 2e-8)["photons"]
    assert double / low == pytest.approx(2.0, rel=1e-5)


def test_step_doubling_converges(setting):
    coarse = _photons(setting, RESONANT, (0.0, 50.0, 0.0), steps=384)["photons"]
    fine = _photons(setting, RESONANT, (0.0, 50.0, 0.0), steps=768)["photons"]
    assert fine == pytest.approx(coarse, rel=1e-5)


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        fh.operators("Cs133", "D1", 0.0)
    with pytest.raises(ValueError):
        fh.operators("Rb85", "D1", 0.0, jones=(0.0, 0.0))
    with pytest.raises(ValueError):
        fh.operators("Rb85", "D1", 0.0, jones=(0.0, 1.0), quantization="E")
