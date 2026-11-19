"""Dark outputs and cached atomic tables must not generate false validation."""
import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gabes.optical_frequency import RbOpticalFrequency
from gabes.paraxial import ParaxialGrid, gaussian_field
from gabes_lab.pipelines import rb_beam_validation as pipeline


def _path(transmission=0., *, balance_error=0., camera_drift=0., edge=0.):
    grid=ParaxialGrid.square(.002,32)
    field=gaussian_field(grid,.001,.0004,795e-9)*np.sqrt(transmission)
    radius=np.array([.0004,.0005]) if transmission else np.array([.0004,np.nan])
    cell=SimpleNamespace(power_w=np.array([.001,.001*transmission]),radius_m=radius,
        final_field=field,max_boundary_power_fraction=edge,power_balance_error_w=balance_error)
    camera=SimpleNamespace(power_w=np.array([.001*transmission,.001*transmission+camera_drift]),
        radius_m=np.array([radius[-1],radius[-1]]),final_field=field,
        max_boundary_power_fraction=edge,power_balance_error_w=0.,
        centerline_intensity_w_m2=np.stack([abs(field[16])**2]*2))
    return dict(grid=grid,cell=cell,camera=camera,z=np.array([0.,.02,.52]),
        widths=np.array([radius[0],radius[1],radius[1]]),
        powers=np.array([.001,.001*transmission,.001*transmission+camera_drift]))


def test_zero_output_has_valid_budgets_but_no_radius_or_fictitious_lens():
    path=_path(0.)
    metrics=pipeline.path_metrics(path)
    assert metrics["passed"]
    assert metrics["camera_radius_mm"] is None and metrics["exit_radius_mm"] is None
    assert metrics["post_cell_power_drift"] == 0.
    lens=pipeline.optional_lens(path,path,795e-9,.0002)
    assert not lens["available"]
    assert "lens_power_m_inverse" not in lens


@pytest.mark.parametrize("fault", [dict(balance_error=1e-8),dict(camera_drift=1e-8),dict(edge=1e-3)])
def test_dark_output_still_requires_physical_budgets(fault):
    metrics=pipeline.path_metrics(_path(1e-15,**fault))
    assert metrics["camera_radius_mm"] is None
    assert not metrics["passed"]


def test_dark_to_resolved_change_under_refinement_cannot_pass(monkeypatch):
    args=pipeline.parse_args([])
    def fake(args,table,wavelength,**options):
        return _path(1.1e-12 if "samples" in options else .9e-12)
    monkeypatch.setattr(pipeline,"cell_and_camera",fake)
    _,_,_,checks=pipeline.validate_path(args,None,None,795e-9)
    assert checks["transverse_grid"]["transmission_absolute_difference"] < 1e-12
    assert not checks["transverse_grid"]["passed"]
    assert not checks["passed"]


def test_unilluminated_fit_and_nonquadratic_fit_remain_unavailable(monkeypatch):
    path=_path(.5)
    def no_fit(*args,**kwargs):
        raise ValueError("at least five illuminated samples are required")
    monkeypatch.setattr(pipeline,"fit_relative_lens",no_fit)
    assert not pipeline.optional_lens(path,path,795e-9,.0002)["available"]
    monkeypatch.setattr(pipeline,"fit_relative_lens",lambda *args,**kwargs:
        dict(lens_power_m_inverse=2.,phase_fit_passed=False))
    lens=pipeline.optional_lens(path,path,795e-9,.0002)
    assert not lens["available"] and "nonquadratic" in lens["status"]


class _AnalyticResponse:
    def __init__(self):
        self.calls=[]
        self.saves=0
    def solve(self,frequency,intensity,fraction,**changes):
        self.calls.append((float(intensity),fraction))
        return dict(chi=(1e-5+2e-5j)*(1+.2*fraction)/(1+intensity/10.))
    def save(self):
        self.saves+=1


def test_adaptive_table_checks_held_out_midpoints_and_refines_curvature():
    args=pipeline.parse_args(["--fractions","0.5","--table-nodes","3",
                              "--peak-intensity-w-m2","100"])
    response=_AnalyticResponse()
    tables,refined,checks=pipeline.adaptive_tables(args,RbOpticalFrequency("D1",0.),response)
    assert len(tables[.5].intensity_w_m2)>3
    assert len(refined[.5].intensity_w_m2)>len(tables[.5].intensity_w_m2)
    assert len(checks["interpolation"])>1 and checks["passed"]
    assert any(abs(intensity-200.00005)<1e-9 for intensity,_ in response.calls)
    assert response.saves>1


def test_adaptive_table_cannot_claim_pass_when_node_budget_is_exhausted():
    args=pipeline.parse_args(["--fractions","0.5","--table-nodes","3",
                              "--max-table-nodes","3","--peak-intensity-w-m2","100"])
    with pytest.raises(ValueError,match="interpolation failed"):
        pipeline.adaptive_tables(args,RbOpticalFrequency("D1",0.),_AnalyticResponse())


class _Run:
    def __init__(self,root):
        self.root=root
    def path(self,*parts):
        target=self.root.joinpath(*parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        return target


def test_response_cache_roundtrips_complex_arrays_and_rejects_changed_sources(tmp_path,monkeypatch):
    source_root=tmp_path/"source"
    for name in ("gabes/constants.py","gabes_lab/pipelines/rb_beam_validation.py"):
        target=source_root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text("# source v1\n",encoding="utf-8")
    monkeypatch.setattr(pipeline,"REPO_ROOT",source_root)
    monkeypatch.setattr(pipeline,"SOURCES",("gabes/constants.py",))
    args=argparse.Namespace(reuse_cache=None)
    run=_Run(tmp_path/"run")
    cache=pipeline.ResponseCache(args,run)
    key=("model","Rb85","D1",123.,1e16)
    cache.cache[key]=dict(chi=1+2j,polarization=np.array([3+4j,0j]),count=5)
    cache.save()
    cached_file=run.path("atomic_cache.json")
    args.reuse_cache=cached_file
    restored=pipeline.ResponseCache(args,_Run(tmp_path/"restored"))
    assert restored.cache[key]["chi"]==1+2j
    np.testing.assert_array_equal(restored.cache[key]["polarization"],[3+4j,0j])
    (source_root/"gabes/constants.py").write_text("# changed atomic constant\n",encoding="utf-8")
    with pytest.raises(ValueError,match="fingerprint differs"):
        pipeline.ResponseCache(args,_Run(tmp_path/"rejected"))
