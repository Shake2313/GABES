"""Rb D1/D2 arbitrary-mixture single-beam propagation and convergence campaign.

The physical frequency is a fixed Rb85 fine-centroid offset, independent of
composition. Examples: --lines D1 D2 --offsets-mhz -4500 6500 --fractions 0 .5 .7217 1
No frequency interpolation/extrapolation and no SABES UI are used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import scipy
from gabes import single_beam_profile
from gabes.nonlinear_response import ComplexSusceptibilityTable, solve_rb_mean
from gabes.optical_frequency import RbOpticalFrequency, RbComposition
from gabes_lab.blocks import figures, provenance
from gabes_lab.blocks.io import write_json
from gabes_lab.blocks.paths import REPO_ROOT
from gabes_lab.blocks.runner import start_run
from .beam_self_action import SOURCES, cell_and_camera, metrics, compare_paths, fit_relative_lens


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out")
    p.add_argument("--reuse-cache", type=Path)
    p.add_argument("--lines", nargs="+", choices=("D1", "D2"), default=["D1", "D2"])
    p.add_argument("--offsets-mhz", type=float, nargs="+", default=[-4500., 6500.])
    p.add_argument("--fractions", type=float, nargs="+", default=[0., .5, .7217, 1.])
    p.add_argument("--temperature-c", type=float, default=70.)
    p.add_argument("--reset-rate", type=float, default=5e5)
    p.add_argument("--peak-intensity-w-m2", type=float, default=2e4)
    p.add_argument("--waist-mm", type=float, default=.4)
    p.add_argument("--cell-length-mm", type=float, default=20.)
    p.add_argument("--camera-distance-m", type=float, default=.5)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--half-width-mm", type=float, default=2.)
    p.add_argument("--cell-steps", type=int, default=80)
    p.add_argument("--table-nodes", type=int, default=17)
    p.add_argument("--max-table-nodes", type=int, default=257)
    p.add_argument("--table-ceiling-factor", type=float, default=4.)
    p.add_argument("--velocity-order", type=int, default=6)
    p.add_argument("--cutoff-sigma", type=float, default=6.)
    a=p.parse_args(argv)
    for f in a.fractions:
        RbComposition(f)
    if not np.isfinite(a.table_ceiling_factor) or a.table_ceiling_factor <= 1.:
        p.error("table-ceiling-factor must be finite and greater than one")
    if a.table_nodes < 3 or a.max_table_nodes < a.table_nodes:
        p.error("table node limits must be >=3 and ordered")
    return a


def _encode(value):
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray):
        return {"__array__": _encode(value.tolist())}
    if isinstance(value, dict):
        return {k:_encode(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [_encode(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value):
    if isinstance(value,dict):
        if "__complex__" in value:
            return complex(*value["__complex__"])
        if "__array__" in value:
            return np.asarray(_decode(value["__array__"]))
        return {k:_decode(v) for k,v in value.items()}
    if isinstance(value,list):
        return [_decode(v) for v in value]
    return value


class ResponseCache:
    def __init__(self, args, run):
        self.args,self.run,self.cache=args,run,{}
        paths=tuple(dict.fromkeys(SOURCES+("gabes_lab/pipelines/rb_beam_validation.py",)))
        self.fingerprint=dict(numpy=np.__version__,scipy=scipy.__version__,
            sources={p:provenance.sha256_file(REPO_ROOT/p) for p in paths if p.startswith("gabes/")})
        if args.reuse_cache:
            data=json.loads(args.reuse_cache.read_text(encoding="utf-8"))
            if data["fingerprint"] != self.fingerprint:
                raise ValueError("response cache source/runtime fingerprint differs")
            self.cache={tuple(key):_decode(value) for key,value in data["entries"]}
        for p in paths:
            run.path("sources",p).write_bytes((REPO_ROOT/p).read_bytes())
        write_json(run.path("source_manifest.json"),dict(atomic_fingerprint=self.fingerprint,
            sources={p:provenance.sha256_file(REPO_ROOT/p) for p in paths}))

    def solve(self, frequency, intensity, fraction, **changes):
        return solve_rb_mean(frequency,float(intensity),rb85_fraction=fraction,
            component_cache=self.cache, **dict(dict(temp_c=self.args.temperature_c,
            ground_relaxation_hz=self.args.reset_rate,velocity_order=self.args.velocity_order,
            cutoff_sigma=self.args.cutoff_sigma),**changes))

    def save(self):
        write_json(self.run.path("atomic_cache.json"),dict(fingerprint=self.fingerprint,
            entries=[(key,_encode(value)) for key,value in self.cache.items()]))


def adaptive_tables(args, frequency, response):
    """Direct held-out midpoints refine intensity interpolation, never frequency."""
    axis=np.r_[0.,np.geomspace(args.peak_intensity_w_m2*1e-6,
                getattr(args,"table_ceiling_factor",4.)*args.peak_intensity_w_m2,args.table_nodes-1)]
    kL=2*np.pi/frequency.wavelength_m*args.cell_length_mm*1e-3
    history=[]
    while True:
        middle=.5*(axis[1:]+axis[:-1])
        tables,refined,errors={},{},{}
        bad=np.zeros(len(middle),bool)
        for fraction in args.fractions:
            values=np.array([response.solve(frequency,i,fraction)["chi"] for i in axis])
            direct=np.array([response.solve(frequency,i,fraction)["chi"] for i in middle])
            table=ComplexSusceptibilityTable(axis,values)
            delta=direct-table(middle)
            phase=.5*kL*abs(delta.real)
            od=kL*abs(delta.imag)
            bad |= (phase>=1e-3)|(od>=1e-3)
            errors[str(fraction)]=dict(max_phase_error_rad=float(max(phase)),
                max_optical_depth_error=float(max(od)))
            fine_axis=np.sort(np.r_[axis,middle])
            fine_values=np.empty(len(fine_axis),complex)
            fine_values[np.searchsorted(fine_axis,axis)]=values
            fine_values[np.searchsorted(fine_axis,middle)]=direct
            tables[fraction]=table
            refined[fraction]=ComplexSusceptibilityTable(fine_axis,fine_values)
        history.append(dict(nodes=len(axis),errors=errors))
        print(f"{frequency.line} {frequency.offset_hz/1e6:g} MHz: {len(axis)} nodes, "
              f"{sum(bad)} intervals need refinement",flush=True)
        response.save()
        if not np.any(bad):
            break
        if len(axis)+sum(bad)>args.max_table_nodes:
            raise ValueError("intensity interpolation failed; increase --max-table-nodes")
        axis=np.sort(np.r_[axis,middle[bad]])
    checks=dict(interpolation=history,quadrature={},passed=True)
    for fraction in args.fractions:
        errors=[]
        for intensity in (0.,.01*args.peak_intensity_w_m2,args.peak_intensity_w_m2,
                          getattr(args,"table_ceiling_factor",4.)*args.peak_intensity_w_m2):
            base=response.solve(frequency,intensity,fraction)["chi"]
            fine=response.solve(frequency,intensity,fraction,
                velocity_order=args.velocity_order+2)["chi"]
            tail=response.solve(frequency,intensity,fraction,
                velocity_order=args.velocity_order+2,cutoff_sigma=args.cutoff_sigma+1)["chi"]
            errors.append(dict(intensity_w_m2=intensity,
                order_phase_error_rad=float(.5*kL*abs((fine-base).real)),
                order_od_error=float(kL*abs((fine-base).imag)),
                cutoff_phase_error_rad=float(.5*kL*abs((tail-fine).real)),
                cutoff_od_error=float(kL*abs((tail-fine).imag))))
        passed=all(v<1e-3 for e in errors for k,v in e.items() if k!="intensity_w_m2")
        checks["quadrature"][str(fraction)]=dict(probes=errors,passed=passed)
        checks["passed"] &= passed
    response.save()
    return tables,refined,checks


def path_metrics(path):
    result=metrics(path,dark_threshold=1e-12)
    return dict(result,status=("resolved output" if result["camera_radius_mm"] is not None
        else "dark: radius and phase unavailable below 1e-12 transmission"))


def optional_lens(full,control,wavelength,fit_radius):
    if any(path_metrics(p)["camera_radius_mm"] is None for p in (full,control)):
        return dict(status="unavailable: dark output",available=False)
    try:
        fit=fit_relative_lens(full,control,wavelength,fit_radius)
    except ValueError as error:
        return dict(status="unavailable: "+str(error),available=False)
    fit.update(available=bool(fit["phase_fit_passed"]),status=("quadratic lens resolved"
        if fit["phase_fit_passed"] else "nonquadratic phase; equivalent focal length unavailable"))
    return fit


def exit_field_difference(reference,candidate):
    """Compare exit fields with the shared phase-invariant shape diagnostic."""
    return single_beam_profile.field_shape_difference(
        reference["cell"].final_field, reference["grid"],
        candidate["cell"].final_field, candidate["grid"])


def validate_path(args,table,fine,wavelength):
    baseline=cell_and_camera(args,table,wavelength)
    control=cell_and_camera(args,table,wavelength,mode="absorption_only")
    summary=path_metrics(baseline)
    checks={}
    lens=optional_lens(baseline,control,wavelength,.5*args.waist_mm*1e-3)
    for name,kwargs,candidate_table in (
        ("transverse_grid",dict(samples=args.samples*2),table),
        ("longitudinal_step",dict(steps=args.cell_steps*2),table),
        ("window",dict(samples=args.samples*3//2,half_width_mm=args.half_width_mm*1.5),table),
        ("intensity_table",{},fine)):
        candidate=cell_and_camera(args,candidate_table,wavelength,**kwargs)
        candidate_summary=path_metrics(candidate)
        absorber=cell_and_camera(args,candidate_table,wavelength,mode="absorption_only",**kwargs)
        budgets=candidate_summary["passed"] and path_metrics(absorber)["passed"]
        if summary["camera_radius_mm"] is None or candidate_summary["camera_radius_mm"] is None:
            difference=abs(candidate_summary["transmission"]-summary["transmission"])
            checks[name]=dict(transmission_absolute_difference=difference,
                phase_status="unavailable for dark output",passed=bool(budgets and difference<1e-12
                    and summary["camera_radius_mm"] is None and candidate_summary["camera_radius_mm"] is None))
        else:
            checks[name]=compare_paths(baseline,candidate)
            field_error=exit_field_difference(baseline,candidate)
            refined_lens=optional_lens(candidate,absorber,wavelength,.5*args.waist_mm*1e-3)
            checks[name].update(exit_field_shape_relative_error=field_error,
                                refined_lens=refined_lens)
            checks[name]["passed"] &= bool(budgets and field_error<.01)
            if lens["available"] and refined_lens["available"]:
                delta=abs(lens["lens_power_m_inverse"]-refined_lens["lens_power_m_inverse"])
                tolerance=max(.01*abs(lens["lens_power_m_inverse"]),1e-3)
                checks[name].update(lens_power_absolute_difference_m_inverse=delta,
                    lens_power_tolerance_m_inverse=tolerance,lens_converged=bool(delta<tolerance))
                checks[name]["passed"] &= bool(delta<tolerance)
            elif lens["available"] != refined_lens["available"]:
                checks[name].update(lens_status="fit validity changed under refinement",passed=False)
    weak_args=argparse.Namespace(**dict(vars(args),peak_intensity_w_m2=args.peak_intensity_w_m2*1e-6))
    weak=cell_and_camera(weak_args,table,wavelength)
    linear=cell_and_camera(weak_args,table,wavelength,mode="linear")
    if path_metrics(weak)["camera_radius_mm"] is not None and path_metrics(linear)["camera_radius_mm"] is not None:
        difference=compare_paths(linear,weak)
        difference["passed"] &= all(difference[k]<1e-4 for k in (
            "max_width_relative_difference","transmission_relative_difference","camera_cut_total_variation"))
        checks["weak_intensity"]=difference
    else:
        delta=abs(path_metrics(weak)["transmission"]-path_metrics(linear)["transmission"])
        checks["weak_intensity"]=dict(transmission_absolute_difference=delta,passed=bool(delta<1e-12
            and path_metrics(weak)["passed"] and path_metrics(linear)["passed"]
            and path_metrics(weak)["camera_radius_mm"] is None and path_metrics(linear)["camera_radius_mm"] is None))
    if lens["available"]:
        largest_change=max((v.get("lens_power_absolute_difference_m_inverse",0.)
                            for v in checks.values()),default=0.)
        resolved=abs(lens["lens_power_m_inverse"])>max(5*largest_change,1e-3)
        lens.update(sign_resolved=bool(resolved),max_refinement_difference_m_inverse=largest_change,
                    sign_status=("focusing" if lens["lens_power_m_inverse"]>0 else "defocusing")
                    if resolved else "numerical sign unresolved")
    summary.update(absorption_only=path_metrics(control),nonlinear_lens=lens)
    checks["passed"]=bool(summary["passed"] and path_metrics(control)["passed"]
                          and all(v["passed"] for v in checks.values()))
    return baseline,control,summary,checks


def main(argv=None):
    args=parse_args(argv)
    with start_run("rb_beam_validation",out=args.out,config=vars(args)) as run:
        response=ResponseCache(args,run)
        evidence,arrays,plot_data={},{},[]
        for line in args.lines:
            for offset in args.offsets_mhz:
                frequency=RbOpticalFrequency(line,offset*1e6)
                tables,refined,atomic=adaptive_tables(args,frequency,response)
                key=f"{line}_{offset:g}"
                evidence[key]=dict(frequency=frequency.as_dict(),atomic=atomic,cases={})
                for fraction in args.fractions:
                    print(f"propagate {line}, offset {offset:g} MHz, Rb85={fraction:g}",flush=True)
                    full,control,summary,checks=validate_path(args,tables[fraction],
                        refined[fraction],frequency.wavelength_m)
                    evidence[key]["cases"][str(fraction)]=dict(summary=summary,checks=checks)
                    prefix=key+f"_f{fraction:g}"
                    arrays.update({prefix+"_"+name:value for name,value in dict(
                        z=full["z"],radius_m=full["widths"],power_w=full["powers"],
                        x_m=full["grid"].x_m,intensity_cut=full["cuts"],
                        exit_field=full["cell"].final_field,camera_field=full["camera"].final_field,
                        absorption_camera_field=control["camera"].final_field,
                        intensity_nodes=tables[fraction].intensity_w_m2,
                        chi_nodes=tables[fraction].chi).items()})
                    plot_data.append((line,offset,fraction,full,control,summary))
                    write_json(run.path("convergence.json"),evidence)
        passed=all(v["atomic"]["passed"] and all(c["checks"]["passed"]
            for c in v["cases"].values()) for v in evidence.values())
        source_record=json.loads(run.path("source_manifest.json").read_text(encoding="utf-8"))
        unchanged=all(provenance.sha256_file(REPO_ROOT/p)==digest
                      for p,digest in source_record["sources"].items())
        evidence["sources_unchanged"]=unchanged
        passed=bool(passed and unchanged)
        evidence["passed"]=passed
        evidence["scope"]="local single CW x-linear beam; no universal red/blue sign gate; numerical not experimental validation"
        write_json(run.path("convergence.json"),evidence)
        np.savez_compressed(run.path("propagation.npz"),**arrays)
        plt=figures.pyplot()
        groups=[(line,offset) for line in args.lines for offset in args.offsets_mhz]
        fig,axes=plt.subplots(len(groups),2,figsize=(12,3*len(groups)),squeeze=False,constrained_layout=True)
        for row,(line,offset) in enumerate(groups):
            for l,o,f,full,control,summary in plot_data:
                if (l,o)!=(line,offset):
                    continue
                label=f"Rb85 fraction {f:g}"
                visible_radius=np.where(full["powers"]/full["powers"][0]>=1e-12,full["widths"],np.nan)
                axes[row,0].plot(full["z"]*1e3,visible_radius*1e3,label=label)
                x=full["grid"].x_m*1e3
                axes[row,1].plot(x,full["camera"].centerline_intensity_w_m2[-1],label=label)
            axes[row,0].set(title=f"{line}, fixed Rb85-centroid offset {offset:g} MHz",
                            xlabel="z (mm)",ylabel="Second-moment radius (mm)")
            axes[row,1].set(xlabel="Camera x (mm)",ylabel="Intensity (W/m^2)",xlim=(-1.2,1.2))
            for ax in axes[row]:
                ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle("Rb D-line mixture propagation | "+("CONVERGED" if passed else "VALIDATION FAILED"))
        figures.save_figure(fig,run.path("rb_mixture_beams"),formats=("png","pdf"),dpi=170)
        if not passed:
            raise RuntimeError("beam campaign convergence failed; inspect convergence.json")
        print(f"PASS: {run.dir}",flush=True)
    return evidence


if __name__=="__main__":
    main()
