"""The figure validation gate must reject unresolved curvature or broken budgets."""
from types import SimpleNamespace

import numpy as np
import pytest

from gabes.paraxial import ParaxialGrid, gaussian_field, thin_lens
from gabes_lab.pipelines import beam_self_action as pipeline

WAVELENGTH = 795e-9


def _path(args, table, *, mode="full", steps=None, reverse_refined=False,
          budget_error=0., nonquadratic=False):
    # Synthetic paths isolate validation logic; they are not physics evidence.
    grid = ParaxialGrid.square(.002, 64)
    waist = args.waist_mm*1e-3
    field = gaussian_field(grid, .001, waist, WAVELENGTH)
    if mode == "full":
        power = -1. if table.startswith("red") else 1.
        if reverse_refined and steps is not None:
            power *= -1
        field = thin_lens(field, grid, wavelength_m=WAVELENGTH, focal_length_m=1/power)
        if nonquadratic:
            field *= np.exp(.6j*np.cos(grid.x_m[None, :]*40000.))
    length, distance = args.cell_length_mm*1e-3, args.camera_distance_m
    z = np.array([0., length, length+distance])
    radius = waist*np.sqrt(1+(z/(np.pi*waist**2/WAVELENGTH))**2)
    transmission = 1. if mode == "vacuum" else .8
    powers = .001*np.array([1., transmission, transmission])
    cuts = np.broadcast_to(abs(field[len(grid.y_m)//2])**2, (2, len(grid.x_m)))
    cell = SimpleNamespace(power_w=powers[:2], radius_m=radius[:2], final_field=field,
                           max_boundary_power_fraction=0., power_balance_error_w=budget_error)
    camera = SimpleNamespace(power_w=powers[1:], radius_m=radius[1:], final_field=field,
                             max_boundary_power_fraction=0., power_balance_error_w=0.,
                             centerline_intensity_w_m2=cuts)
    return dict(grid=grid, cell=cell, camera=camera, z=z, widths=radius,
                powers=powers, cuts=np.vstack((cuts, cuts[-1:])))


def _calculate(monkeypatch, **fault):
    args = pipeline.parse_args([])
    tables = dict(red="red", blue="blue", red_refined="red_refined", blue_refined="blue_refined")
    def fake(args, table, wavelength, *, mode="full", steps=None, **kwargs):
        return _path(args, table, mode=mode, steps=steps, **fault)
    monkeypatch.setattr(pipeline, "cell_and_camera", fake)
    return pipeline.calculate(args, tables, WAVELENGTH)


def test_matched_curvature_and_conserved_passive_paths_pass_the_gate(monkeypatch):
    _, _, checks = _calculate(monkeypatch)
    assert checks["passed"]
    assert checks["physical_budgets"]["passed"]
    assert checks["red"]["sign_margin"]["passed"]


def test_refinement_sign_reversal_cannot_pass_unchanged_intensity_checks(monkeypatch):
    _, _, checks = _calculate(monkeypatch, reverse_refined=True)
    assert checks["sign_switch"]["passed"]  # Baseline alone looks correct.
    assert checks["red"]["longitudinal_step"]["max_width_relative_difference"] == 0
    assert not checks["red"]["longitudinal_step"]["lens_sign_stable"]
    assert not checks["red"]["sign_margin"]["passed"]
    assert not checks["passed"]


def test_power_budget_failure_cannot_pass_a_converged_shape(monkeypatch):
    _, _, checks = _calculate(monkeypatch, budget_error=1e-8)
    assert not checks["physical_budgets"]["passed"]
    assert not checks["passed"]


def test_nonquadratic_phase_does_not_certify_a_thin_lens_sign(monkeypatch):
    _, _, checks = _calculate(monkeypatch, nonquadratic=True)
    assert not checks["phase_fit"]["passed"]
    assert not checks["passed"]


@pytest.mark.parametrize("fault", ["gain", "drift", "edge"])
def test_metrics_rejects_gain_vacuum_power_drift_and_edge_power(fault):
    args = pipeline.parse_args([])
    path = _path(args, "red")
    if fault == "gain":
        path["cell"].power_w = np.array([.001, .0011])
    elif fault == "drift":
        path["camera"].power_w = np.array([.0008, .00080001])
    else:
        path["camera"].max_boundary_power_fraction = .001
    assert not pipeline.metrics(path)["passed"]
