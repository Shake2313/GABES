"""Mixture density, common frequency, cache and transparent-limit contracts."""
import numpy as np
import pytest

from gabes import nonlinear_response as response
from gabes.optical_frequency import RbOpticalFrequency


def fake_components(monkeypatch):
    calls = []
    def solve(isotope, line, detuning, intensity, **kw):
        calls.append((isotope, line, detuning, intensity, kw))
        chi = (2+3j if isotope == "Rb85" else 5+7j)*kw["density_m3"]
        return dict(chi=chi, alpha_per_m=chi.imag,
            polarization_xy_c_m2=np.array([chi, 0j]), excited_fraction=.1,
            conditions=dict(density_m3=kw["density_m3"]),
            diagnostics=dict(trace_error=0., hermiticity_error=0.,
                min_eigenvalue=0., steady_state_relative_residual=0.), class_count=1)
    monkeypatch.setattr(response, "_maxwell_mean", solve)
    return calls


@pytest.mark.parametrize("fraction", [0., .01, .25, .7217, .999, 1.])
@pytest.mark.parametrize("line", ["D1", "D2"])
def test_mixture_uses_one_frequency_and_intensity_and_density_once(monkeypatch, fraction, line):
    calls = fake_components(monkeypatch)
    frequency = RbOpticalFrequency(line, 1234.567e6)
    result = response.solve_rb_mean(frequency, 123., rb85_fraction=fraction, density_m3=10.)
    expected = 10*(fraction*(2+3j)+(1-fraction)*(5+7j))
    assert result["chi"] == pytest.approx(expected)
    assert result["alpha_per_m"] == pytest.approx(2*np.pi/frequency.wavelength_m*expected.imag)
    assert len(calls) == (1 if fraction in (0., 1.) else 2)
    for isotope, selected_line, delta, intensity, kw in calls:
        assert selected_line == line and intensity == 123. and kw["density_m3"] == 10.
        assert delta == frequency.detuning_mhz(isotope, reference="upper_ground_centroid")
        weight = fraction if isotope == "Rb85" else 1-fraction
        assert result["components"][isotope]["conditions"]["density_m3"] == 10*weight
    assert sum(result["conditions"]["isotope_density_m3"].values()) == pytest.approx(10.)


def test_cache_reuses_pure_response_without_mutating_it_or_shifting_frequency(monkeypatch):
    calls = fake_components(monkeypatch)
    cache = {}
    frequency = RbOpticalFrequency("D1", -2500e6)
    kwargs = dict(density_m3=10., component_cache=cache)
    first = response.solve_rb_mean(frequency, 12., rb85_fraction=.25, **kwargs)
    second = response.solve_rb_mean(frequency, 12., rb85_fraction=.9, **kwargs)
    assert len(calls) == 2
    assert first["chi"] != second["chi"]
    first["components"]["Rb85"]["polarization_xy_c_m2"][:] = 999
    again = response.solve_rb_mean(frequency, 12., rb85_fraction=.25, **kwargs)
    assert again["polarization_xy_c_m2"][0] == pytest.approx(first["chi"])
    response.solve_rb_mean(frequency.shifted_hz(1.), 12., rb85_fraction=.25, **kwargs)
    assert len(calls) == 4
    response.solve_rb_mean(frequency, 12., rb85_fraction=.25, temp_c=71., **kwargs)
    assert len(calls) == 6


def test_zero_total_density_is_transparent_without_atomic_solves(monkeypatch):
    calls = fake_components(monkeypatch)
    result = response.solve_rb_mean(RbOpticalFrequency("D2", 0.), 1e6, density_m3=0.)
    assert not calls and not result["components"]
    assert result["chi"] == 0j and result["alpha_per_m"] == 0.
    assert result["class_count"] == 0
    assert not result["convergence"]["assessed"]


@pytest.mark.parametrize("kwargs", [dict(rb85_fraction=-.01), dict(rb85_fraction=1.01),
    dict(rb85_fraction=np.nan), dict(intensity_w_m2=-1.), dict(density_m3=-1.),
    dict(density_m3=np.inf), dict(temp_c=-273.15), dict(ground_relaxation_hz=0.),
    dict(velocity_order=True), dict(velocity_order=2.1), dict(cutoff_sigma=0.)])
def test_invalid_mixture_inputs_fail_before_solves(monkeypatch, kwargs):
    calls = fake_components(monkeypatch)
    params = dict(intensity_w_m2=1.)
    params.update(kwargs)
    with pytest.raises(ValueError):
        response.solve_rb_mean(RbOpticalFrequency("D1", 0.), **params)
    assert not calls


def test_bare_detuning_is_not_an_absolute_frequency():
    with pytest.raises(TypeError):
        response.solve_rb_mean(1500., 1.)
