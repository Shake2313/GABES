"""Runtime atomic/propagation contracts for the interactive beam profiler."""
from dataclasses import replace

import numpy as np
import pytest

from gabes import single_beam_profile as profile
from gabes.nonlinear_response import ComplexSusceptibilityTable
from gabes.optical_frequency import RbOpticalFrequency


def settings(**changes):
    return profile.AtomicSettings(RbOpticalFrequency("D1", 1.5e9), **changes)


def response(chi=0j, *, length=.02, bound=1e5, atomic=None):
    atom = settings() if atomic is None else atomic
    table = ComplexSusceptibilityTable([0., bound], [chi, chi])
    return dict(settings=atom, table=table, refined_table=table,
        cache_key=profile.response_cache_key(atom, bound, cell_length_m=length),
        checks=dict(assessed=True, passed=True), cell_length_m=length,
        max_intensity_w_m2=bound)


def cell_inputs(**changes):
    return dict(dict(power_w=.001, radius_m=.0004, curvature_radius_m=2.,
        cell_length_m=.02, half_width_m=.002, samples=64, steps=8), **changes)


def fake_mean(monkeypatch, *, quadrature_error=0.):
    calls = []
    def solve(frequency, intensity, **kwargs):
        calls.append((float(intensity), kwargs.copy()))
        chi = (1e-5+2e-5j)/(1+intensity/10.)
        chi += quadrature_error*(kwargs["velocity_order"]-6)
        return dict(chi=chi, class_count=3,
            diagnostics=dict(trace_error=0., hermiticity_error=0., min_eigenvalue=0.,
                             steady_state_relative_residual=0.))
    monkeypatch.setattr(profile, "solve_rb_mean", solve)
    return calls


def test_atomic_settings_and_acceptance_cache_key_cover_conditions():
    original = settings()
    key = profile.response_cache_key(original, 1e5, cell_length_m=.05)
    for changed in (
        replace(original, frequency=original.frequency.shifted_hz(1.)),
        replace(original, rb85_fraction=.5), replace(original, temp_c=71.),
        replace(original, ground_relaxation_hz=1e4), replace(original, density_m3=1e15),
        replace(original, velocity_order=8), replace(original, cutoff_sigma=7.)):
        assert profile.response_cache_key(changed, 1e5, cell_length_m=.05) != key
    for kwargs in (dict(cell_length_m=.06), dict(initial_nodes=19),
                   dict(max_nodes=257), dict(tolerance=2e-3)):
        assert profile.response_cache_key(original, 1e5,
            **dict(dict(cell_length_m=.05), **kwargs)) != key
    assert profile.response_cache_key(original, 2e5, cell_length_m=.05) != key
    assert original.as_dict()["frequency"]["absolute_hz"] == original.frequency.absolute_hz


@pytest.mark.parametrize("changes", [dict(rb85_fraction=-.1), dict(temp_c=-273.15),
    dict(ground_relaxation_hz=0.), dict(density_m3=-1.), dict(velocity_order=True),
    dict(velocity_order=2.5), dict(cutoff_sigma=0.)])
def test_atomic_settings_reject_invalid_inputs(changes):
    with pytest.raises(ValueError):
        settings(**changes)


def test_adaptive_atomic_table_uses_held_out_points_and_independent_quadratures(monkeypatch):
    calls = fake_mean(monkeypatch)
    progress = []
    bundle = profile.build_response_table(settings(), 400., cell_length_m=.02,
        initial_nodes=3, max_nodes=129, progress=lambda fraction, message: progress.append((fraction, message)))
    assert bundle["checks"]["passed"]
    assert bundle["table"].intensity_w_m2.size > 3
    assert bundle["refined_table"].intensity_w_m2.size > bundle["table"].intensity_w_m2.size
    assert len(bundle["checks"]["interpolation"]["history"]) > 1
    assert any(abs(intensity-200.00005) < 1e-9 for intensity, _ in calls)
    assert {(kw["velocity_order"], kw["cutoff_sigma"]) for _, kw in calls} == {(6, 6.), (8, 6.), (8, 7.)}
    identities = [(intensity, kw["velocity_order"], kw["cutoff_sigma"]) for intensity, kw in calls]
    assert len(identities) == len(set(identities))
    assert all(0 <= value <= 1 for value, _ in progress) and progress[-1][0] == 1.


def test_failed_interpolation_or_quadrature_never_claims_convergence(monkeypatch):
    fake_mean(monkeypatch)
    limited = profile.build_response_table(settings(), 400., cell_length_m=.02,
                                            initial_nodes=3, max_nodes=3)
    assert not limited["checks"]["passed"]
    assert not limited["checks"]["interpolation"]["passed"]
    fake_mean(monkeypatch, quadrature_error=1e-5)
    failed = profile.build_response_table(settings(), 400., cell_length_m=.02)
    assert not failed["checks"]["quadrature"]["passed"]
    assert not failed["checks"]["passed"]


def test_curved_input_and_camera_follow_physical_wavelength_without_atomic_resolve(monkeypatch):
    # Production transparent medium exercises the real mixture interface while
    # its exact zero-density shortcut avoids unnecessary atomic states.
    atom = settings(density_m3=0.)
    bundle = profile.build_response_table(atom, 1e5, cell_length_m=.02, initial_nodes=3)
    assert bundle["checks"]["passed"] and bundle["checks"]["sampled_velocity_states"] == 0
    cell = profile.propagate_cell(bundle, **cell_inputs())
    assert cell["checks"]["passed"]
    assert cell["checks"]["propagation"]["assessed"]
    def forbidden(*args, **kwargs):
        raise AssertionError("camera navigation must never rebuild atomic response")
    monkeypatch.setattr(profile, "solve_rb_mean", forbidden)
    camera = profile.profile_at_distance(cell, .3)
    assert camera["checks"]["passed"]
    assert camera["paths"]["full"].final_field.shape == (64, 64)
    combined = camera["combined"]["full"]
    z, radius, curvature = combined["z_m"], .0004, 2.
    expected = np.sqrt(radius**2*(1+z/curvature)**2
                       +(atom.frequency.wavelength_m*z/(np.pi*radius))**2)
    np.testing.assert_allclose(combined["radius_m"], expected, rtol=2e-6)
    np.testing.assert_allclose(combined["power_w"], .001, rtol=1e-12)
    assert combined["z_m"][-1] == pytest.approx(.32)
    assert combined["centerline_intensity_w_m2"].shape == (len(z), 64)


def test_unrefined_result_is_explicitly_unassessed_even_with_good_budgets():
    cell = profile.propagate_cell(response(), **cell_inputs(refine=False))
    assert cell["checks"]["physical_budgets"]["passed"]
    assert not cell["checks"]["propagation"]["assessed"]
    assert not cell["checks"]["passed"]
    camera = profile.profile_at_distance(cell, 0.)
    assert camera["checks"]["free_space_budgets"]["passed"]
    assert not camera["checks"]["camera_convergence"]["assessed"]
    assert not camera["checks"]["passed"]


@pytest.mark.parametrize("failure", [None, "atomic", "vacuum_budget"])
def test_lens_sign_requires_atomic_and_baseline_budget_acceptance(monkeypatch, failure):
    bundle = response()
    bundle["checks"]["passed"] = failure != "atomic"
    # Hold a clearly resolved, refinement-stable curvature fixed. This isolates
    # acceptance gating from the existing numerical phase-fit tests.
    monkeypatch.setattr(profile, "_lens", lambda *args: dict(
        available=True, lens_power_m_inverse=2., phase_fit_passed=True))
    propagate = profile.propagate

    def with_budget_fault(*args, **kwargs):
        result = propagate(*args, **kwargs)
        if failure == "vacuum_budget" and kwargs.get("susceptibility") is None:
            # Only the baseline vacuum control is affected; material-path
            # refinements still pass, and every propagated field is unchanged.
            return replace(result, power_balance_error_w=1e-5)
        return result

    monkeypatch.setattr(profile, "propagate", with_budget_fault)
    cell = profile.propagate_cell(bundle, **cell_inputs())
    assert cell["checks"]["propagation"]["passed"]
    assert cell["checks"]["physical_budgets"]["passed"] == (failure != "vacuum_budget")
    assert cell["checks"]["passed"] == (failure is None)
    assert cell["nonlinear_lens"]["sign_resolved"] == (failure is None)
    assert cell["nonlinear_lens"]["sign_status"] == (
        "focusing" if failure is None else "numerical sign unresolved")


def test_stale_length_conditions_and_insufficient_intensity_bound_are_rejected():
    with pytest.raises(ValueError, match="cell length"):
        profile.propagate_cell(response(), **cell_inputs(cell_length_m=.03))
    stale = response()
    stale["settings"] = replace(stale["settings"], temp_c=80.)
    with pytest.raises(ValueError, match="cached table"):
        profile.propagate_cell(stale, **cell_inputs())
    with pytest.raises(ValueError, match="input intensity"):
        profile.propagate_cell(response(bound=1.), **cell_inputs())


def test_passive_slab_and_dark_output_keep_energy_budgets_and_suppress_radius():
    bundle = response(chi=1e-6+2e-6j)
    cell = profile.propagate_cell(bundle, **cell_inputs(refine=False))
    k = 2*np.pi/cell["wavelength_m"]
    expected = np.exp(-k*2e-6*.02)
    budget = cell["checks"]["physical_budgets"]["paths"]["full"]
    assert budget["transmission"] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(cell["paths"]["full"].final_field,
        cell["paths"]["vacuum"].final_field*np.exp(.5j*k*(1e-6+2e-6j)*.02), rtol=1e-10, atol=1e-13)
    dark = profile.propagate_cell(response(chi=.1j), **cell_inputs(refine=False))
    assert dark["checks"]["physical_budgets"]["passed"]
    assert dark["checks"]["physical_budgets"]["paths"]["full"]["radius_m"] is None
    assert not dark["nonlinear_lens"]["available"]
    camera = profile.profile_at_distance(dark, .1)
    assert camera["checks"]["free_space_budgets"]["passed"]
    assert not camera["checks"]["camera_convergence"]["assessed"]
    assert camera["checks"]["free_space_budgets"]["paths"]["full"]["radius_m"] is None


def test_complex_field_comparison_removes_only_global_phase():
    cell = profile.propagate_cell(response(), **cell_inputs(refine=False))
    field, grid = cell["paths"]["full"].final_field, cell["grid"]
    assert profile.field_shape_difference(field, grid, field*np.exp(.4j), grid) < 1e-14
    distorted = field*np.exp(1j*grid.radius_squared_m2/(.0004**2))
    assert profile.field_shape_difference(field, grid, distorted, grid) > .1


def test_single_beam_profile_opt_in_qdht_matches_fft_and_documents_boundary():
    atom = settings(density_m3=0.)
    bundle = profile.build_response_table(atom, 1e5, cell_length_m=.02, initial_nodes=3)
    cell_fft = profile.propagate_cell(bundle, **cell_inputs(samples=64, steps=16, refine=True, propagator="fft"))
    cell_qdht = profile.propagate_cell(bundle, **cell_inputs(samples=64, steps=16, refine=True, propagator="qdht"))
    assert cell_qdht["propagator"] == "qdht"
    assert "circular Dirichlet" in cell_qdht["conditions"]["boundary_condition"]
    assert cell_qdht["checks"]["passed"]
    assert cell_qdht["input_key"] != cell_fft["input_key"]
    assert cell_qdht["paths"]["full"].power_w[-1] == pytest.approx(cell_fft["paths"]["full"].power_w[-1], rel=1e-9)
    assert cell_qdht["paths"]["full"].radius_m[-1] == pytest.approx(cell_fft["paths"]["full"].radius_m[-1], rel=1e-9)
    assert cell_qdht["checks"]["physical_budgets"]["paths"]["full"]["relative_power_balance_error"] <= 1e-9

    cam_fft = profile.profile_at_distance(cell_fft, .25, steps=16)
    cam_qdht = profile.profile_at_distance(cell_qdht, .25, steps=16)
    assert cam_qdht["propagator"] == "qdht"
    assert cam_qdht["checks"]["passed"]
    assert cam_qdht["paths"]["full"].radius_m[-1] == pytest.approx(cam_fft["paths"]["full"].radius_m[-1], rel=1e-6)
    assert cam_qdht["paths"]["full"].power_w[-1] == pytest.approx(cam_fft["paths"]["full"].power_w[-1], rel=1e-9)

    with pytest.raises(ValueError, match="propagator"):
        profile.propagate_cell(bundle, **cell_inputs(refine=False, propagator="unknown"))
