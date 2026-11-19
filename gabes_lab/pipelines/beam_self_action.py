"""Reproduce the Rb87 D1 single-beam nonlinear-refraction validation.

Run: python -m gabes_lab.pipelines.beam_self_action --out runs/beam_self_action/reference
Inputs: CLI geometry/atomic settings; existing full 16-state OBE.
Outputs: figures, sampled fields/tables, convergence.json and run.json under runs/.
No SABES UI or apparatus defaults are changed.
"""
from __future__ import annotations

import argparse
import json
import inspect

import scipy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gabes import constants, magneto_multilevel, species, single_beam_profile
from gabes.beam import peak_intensity
from gabes.nonlinear_response import ComplexSusceptibilityTable, solve_rb87_d1_mean
from gabes.paraxial import ParaxialGrid, gaussian_field, propagate
from gabes_lab.blocks import figures, provenance
from gabes_lab.blocks.io import write_json, write_text
from gabes_lab.blocks.paths import REPO_ROOT
from gabes_lab.blocks.runner import start_run

SOURCES = (
    "gabes/nonlinear_response.py", "gabes/paraxial.py", "gabes/psr_full_obe.py",
    "gabes/psr_conventions.py", "gabes/magneto_multilevel.py", "gabes/core.py",
    "gabes/species.py", "gabes/constants.py", "gabes/beam.py",
    "gabes/quantum/reservoirs.py", "gabes/doppler.py",
    "gabes/angular.py", "gabes/_arrays.py", "gabes/quantum/contracts.py",
    "gabes/zeeman.py", "gabes/magneto_coordinates.py", "gabes/magneto_optical.py",
    "gabes/atoms.py", "gabes/optical_bloch.py", "gabes/optical_frequency.py",
    "gabes/single_beam_profile.py",
    "gabes_lab/pipelines/beam_self_action.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--reuse-tables", type=Path)
    parser.add_argument("--temperature-c", type=float, default=100.)
    parser.add_argument("--reset-rate", type=float, default=5e5)
    parser.add_argument("--detuning-mhz", type=float, default=1500.)
    parser.add_argument("--waist-mm", type=float, default=.4)
    parser.add_argument("--peak-intensity-w-m2", type=float, default=2e4)
    parser.add_argument("--cell-length-mm", type=float, default=50.)
    parser.add_argument("--camera-distance-m", type=float, default=.5)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--half-width-mm", type=float, default=2.)
    parser.add_argument("--cell-steps", type=int, default=160)
    parser.add_argument("--table-nodes", type=int, default=49)
    parser.add_argument("--velocity-order", type=int, default=6)
    parser.add_argument("--cutoff-sigma", type=float, default=6.)
    parser.add_argument("--skip-refinement", action="store_true",
                        help="Exploratory output only; never marks numerical validation passed.")
    return parser.parse_args(argv)


def atomic_config(args):
    return dict(temp_c=args.temperature_c, ground_relaxation_hz=args.reset_rate,
                velocity_order=args.velocity_order, cutoff_sigma=args.cutoff_sigma)


def compact(result):
    return dict(chi_real=result["chi"].real, chi_imag=result["chi"].imag,
                class_count=result["class_count"], diagnostics=result["diagnostics"],
                omitted_maxwell_probability=result["omitted_maxwell_probability"])


def make_tables(args, run, wavelength):
    atomic = atomic_config(args)
    axis = np.r_[0., np.geomspace(args.peak_intensity_w_m2*1e-5,
                                 4*args.peak_intensity_w_m2, args.table_nodes-1)]
    cache_meta = dict(numpy_version=np.__version__, scipy_version=scipy.__version__,
                      atomic=atomic, detuning_mhz=args.detuning_mhz,
                      peak_intensity_w_m2=args.peak_intensity_w_m2,
                      table_nodes=args.table_nodes,
                      sources={p: provenance.sha256_file(REPO_ROOT/p) for p in SOURCES[:-1]})
    cached = None
    inherited_health = None
    inherited_audit = None
    if args.reuse_tables:
        with np.load(args.reuse_tables, allow_pickle=False) as data:
            if json.loads(str(data["metadata"])) != cache_meta:
                raise ValueError("cached table settings/source hashes differ from this run")
            cached = {key: data[key].copy() for key in ("red_chi", "blue_chi")}
            if not np.array_equal(axis, data["intensity_w_m2"]):
                raise ValueError("cached table intensity axis differs")
        health_path = args.reuse_tables.with_name("atomic_nodes.json")
        if health_path.exists():
            inherited_health = json.loads(health_path.read_text(encoding="utf-8"))
        audit_path = args.reuse_tables.with_name("atomic_refinement.npz")
        audit_metadata = dict(table=cache_meta, cell_length_mm=args.cell_length_mm,
                              wavelength_m=wavelength, code_sha256=provenance.sha256_bytes(
            inspect.getsource(make_tables).encode("utf-8")))
        if audit_path.exists() and not args.skip_refinement:
            with np.load(audit_path, allow_pickle=False) as data:
                if json.loads(str(data["metadata"])) == audit_metadata:
                    inherited_audit = {key: data[key].copy() for key in data.files}
    tables, audit = {}, {}
    for label, detuning in (("red", -args.detuning_mhz), ("blue", args.detuning_mhz)):
        values, health = [], []
        if cached is None:
            for index, intensity in enumerate(axis):
                result = solve_rb87_d1_mean(detuning, intensity, **atomic)
                values.append(result["chi"])
                health.append(compact(result))
                if index % 12 == 0 or index == len(axis)-1:
                    print(f"{label} atomic table {index+1}/{len(axis)}", flush=True)
        else:
            values = cached[label+"_chi"]
        tables[label] = ComplexSusceptibilityTable(axis, values)
        if cached is not None and inherited_health:
            health = inherited_health.get(label, {}).get("nodes", [])
        audit[label] = dict(nodes=health, detuning_mhz=detuning,
                            reused=bool(cached is not None),
                            node_health_available=bool(health))
        if cached is not None:
            audit[label]["source_table"] = provenance.file_record(args.reuse_tables)
            if inherited_health is not None:
                audit[label]["source_node_health"] = provenance.file_record(health_path)
    np.savez_compressed(run.path("susceptibility_tables.npz"),
                        intensity_w_m2=axis, metadata=json.dumps(cache_meta),
                        **{label+"_chi": table.chi for label, table in tables.items()})
    write_json(run.path("atomic_nodes.json"), audit)
    if args.skip_refinement:
        return tables, dict(passed=False, status="refinement skipped")
    audit_metadata = dict(table=cache_meta, cell_length_mm=args.cell_length_mm,
                              wavelength_m=wavelength, code_sha256=provenance.sha256_bytes(
        inspect.getsource(make_tables).encode("utf-8")))
    if inherited_audit is not None:
        for label in ("red", "blue"):
            tables[label+"_refined"] = ComplexSusceptibilityTable(
                inherited_audit[label+"_axis"], inherited_audit[label+"_chi"])
        checks = json.loads(str(inherited_audit["checks"]))
        # Save exact audit evidence with its dependency/configuration fingerprint.
        np.savez_compressed(run.path("atomic_refinement.npz"), **inherited_audit)
        return tables, checks

    kL = 2*np.pi/wavelength*args.cell_length_mm*1e-3
    checks = {}
    for label, detuning in (("red", -args.detuning_mhz), ("blue", args.detuning_mhz)):
        table = tables[label]
        # Direct OBE evaluations between every tabulated pair; none are fit nodes.
        middle = .5*(axis[1:]+axis[:-1])
        midpoint_chi = []
        for index, intensity in enumerate(middle):
            result = solve_rb87_d1_mean(detuning, intensity, **atomic)
            midpoint_chi.append(result["chi"])
            if index % 16 == 0:
                print(f"{label} held-out interpolation {index+1}/{len(middle)}", flush=True)
        midpoint_chi = np.asarray(midpoint_chi)
        error = midpoint_chi-table(middle)
        refined_axis = np.sort(np.r_[axis, middle])
        refined_values = np.empty(refined_axis.size, complex)
        refined_values[np.searchsorted(refined_axis, axis)] = table.chi
        refined_values[np.searchsorted(refined_axis, middle)] = midpoint_chi
        tables[label+"_refined"] = ComplexSusceptibilityTable(refined_axis, refined_values)
        # Independent order and cutoff checks spanning the entire used table.
        probes = args.peak_intensity_w_m2*np.array([0., .001, .01, .1, .4, 1., 2., 4.])
        order_error, tail_error, details = [], [], []
        for intensity in probes:
            base = solve_rb87_d1_mean(detuning, intensity, **atomic)
            fine = solve_rb87_d1_mean(
                detuning, intensity, **dict(atomic, velocity_order=args.velocity_order+2))
            tail = solve_rb87_d1_mean(
                detuning, intensity, **dict(atomic, velocity_order=args.velocity_order+2,
                                           cutoff_sigma=args.cutoff_sigma+1))
            order_error.append(fine["chi"]-base["chi"])
            tail_error.append(tail["chi"]-fine["chi"])
            details.append(dict(intensity_w_m2=float(intensity), base=compact(base),
                                order=compact(fine), cutoff=compact(tail)))
        def errors(values):
            values = np.asarray(values)
            return dict(max_phase_error_rad=float(.5*kL*np.max(abs(values.real))),
                        max_optical_depth_error=float(kL*np.max(abs(values.imag))))
        checked = dict(interpolation=errors(error), velocity_order=errors(order_error),
                       velocity_cutoff=errors(tail_error), quadrature_probes=details)
        checked["passed"] = all(
            checked[name]["max_phase_error_rad"] < 1e-3
            and checked[name]["max_optical_depth_error"] < 1e-3
            for name in ("interpolation", "velocity_order", "velocity_cutoff"))
        checks[label] = checked
    checks["passed"] = all(checks[label]["passed"] for label in ("red", "blue"))
    checks["criterion"] = "direct-response phase and optical-depth errors below 0.001 over cell"
    np.savez_compressed(
        run.path("atomic_refinement.npz"), metadata=json.dumps(audit_metadata),
        checks=json.dumps(checks),
        **{label+"_axis": tables[label+"_refined"].intensity_w_m2 for label in ("red", "blue")},
        **{label+"_chi": tables[label+"_refined"].chi for label in ("red", "blue")})
    return tables, checks


def cell_and_camera(args, table, wavelength, *, samples=None, half_width_mm=None,
                    steps=None, mode="full"):
    grid = ParaxialGrid.square(
        (half_width_mm or args.half_width_mm)*1e-3, samples or args.samples)
    waist = args.waist_mm*1e-3
    power = args.peak_intensity_w_m2*np.pi*waist**2/2
    field = gaussian_field(grid, power, waist, wavelength)
    response = table
    if mode == "absorption_only":
        response = lambda intensity: table(0.).real + 1j*table(intensity).imag
    elif mode == "linear":
        response = lambda intensity: table(0.)
    elif mode == "vacuum":
        response = None
    cell = propagate(field, grid, wavelength_m=wavelength,
                     length_m=args.cell_length_mm*1e-3,
                     steps=steps or args.cell_steps, susceptibility=response,
                     snapshot_stride=max(1, (steps or args.cell_steps)//80))
    camera = propagate(cell.final_field, grid, wavelength_m=wavelength,
                       length_m=args.camera_distance_m, steps=100)
    return dict(grid=grid, cell=cell, camera=camera,
                z=np.r_[cell.z_m, args.cell_length_mm*1e-3+camera.z_m[1:]],
                widths=np.r_[cell.radius_m, camera.radius_m[1:]],
                powers=np.r_[cell.power_w, camera.power_w[1:]],
                cuts=np.concatenate((cell.centerline_intensity_w_m2,
                                     camera.centerline_intensity_w_m2[1:])))


def metrics(result, *, dark_threshold=0.):
    cell, camera = result["cell"], result["camera"]
    transmission = float(cell.power_w[-1]/cell.power_w[0])
    if dark_threshold > 0 and 0 <= transmission < dark_threshold:
        balance = float(max(abs(cell.power_balance_error_w), abs(camera.power_balance_error_w))/cell.power_w[0])
        drift = float(abs(camera.power_w[-1]-camera.power_w[0])/cell.power_w[0])
        edge = max(cell.max_boundary_power_fraction,camera.max_boundary_power_fraction)
        valid = bool(np.isfinite([transmission,balance,drift,edge]).all()
                     and balance <= 1e-9 and drift <= 1e-9 and edge <= 1e-5)
        return dict(transmission=transmission,exit_radius_mm=None,camera_radius_mm=None,
            max_boundary_power_fraction=edge,relative_power_balance_error=balance,
            post_cell_power_drift=drift,post_cell_drift_normalization="input power for dark output",
            passed=valid)
    values = dict(transmission=float(cell.power_w[-1]/cell.power_w[0]),
                  exit_radius_mm=float(cell.radius_m[-1]*1e3),
                  camera_radius_mm=float(camera.radius_m[-1]*1e3),
                  max_boundary_power_fraction=max(cell.max_boundary_power_fraction,
                                                  camera.max_boundary_power_fraction),
                  relative_power_balance_error=float(
                      max(abs(cell.power_balance_error_w), abs(camera.power_balance_error_w))
                      /cell.power_w[0]),
                  post_cell_power_drift=float(abs(camera.power_w[-1]/camera.power_w[0]-1)))
    values["passed"] = bool(
        all(np.isfinite(value) for value in values.values())
        and 0 <= values["transmission"] <= 1+1e-10
        and values["relative_power_balance_error"] <= 1e-9
        and values["post_cell_power_drift"] <= 1e-9
        and values["max_boundary_power_fraction"] <= 1e-5)
    return values


def compare_paths(reference, candidate, *, reference_lens=None, candidate_lens=None):
    width = np.interp(reference["z"], candidate["z"], candidate["widths"])
    result = dict(max_width_relative_difference=float(
        np.max(abs(width-reference["widths"])/reference["widths"])),
        transmission_relative_difference=float(abs(
            metrics(candidate)["transmission"]/metrics(reference)["transmission"]-1)))
    # Shape comparison at camera, independent of total transmitted power.
    x = reference["grid"].x_m
    before = reference["camera"].centerline_intensity_w_m2[-1]
    after = np.interp(x, candidate["grid"].x_m,
                      candidate["camera"].centerline_intensity_w_m2[-1])
    before = before/np.trapezoid(before, x)
    after = after/np.trapezoid(after, x)
    result["camera_cut_total_variation"] = float(.5*np.trapezoid(abs(before-after), x))
    passed = all(np.isfinite(value) and value < .01 for value in result.values())
    result["physical_budgets_passed"] = metrics(reference)["passed"] and metrics(candidate)["passed"]
    if (reference_lens is None) != (candidate_lens is None):
        raise ValueError("both lens diagnostics are required for curvature comparison")
    if reference_lens is not None:
        baseline = reference_lens["lens_power_m_inverse"]
        refined = candidate_lens["lens_power_m_inverse"]
        difference = abs(refined-baseline)
        result.update(
            lens_power_absolute_difference_m_inverse=float(difference),
            lens_power_relative_difference=float(difference/abs(baseline)) if baseline else float("inf"),
            lens_sign_stable=bool(baseline*refined > 0),
            phase_fit_passed=bool(reference_lens["phase_fit_passed"] and candidate_lens["phase_fit_passed"]),
            refined_lens=candidate_lens)
        passed = (passed and result["lens_power_relative_difference"] <= .01
                  and result["lens_sign_stable"] and result["phase_fit_passed"])
    result["passed"] = bool(passed and result["physical_budgets_passed"])
    return result


def fit_relative_lens(result, control, wavelength, fit_radius):
    """Use the production fixed-aperture phase fit without a second kernel."""
    return single_beam_profile.fit_relative_lens(result, control, wavelength, fit_radius)


def calculate(args, tables, wavelength):
    results = {}
    for label in ("red", "blue"):
        for mode in ("full", "absorption_only", "linear"):
            print(f"propagate {label}: {mode}", flush=True)
            results[label+"_"+mode] = cell_and_camera(args, tables[label], wavelength, mode=mode)
    results["vacuum"] = cell_and_camera(args, None, wavelength, mode="vacuum")
    checks, summaries = {}, {}
    fit_radius = .5*args.waist_mm*1e-3
    def lens_fit(result, control):
        return fit_relative_lens(result, control, wavelength, fit_radius)

    for label in ("red", "blue"):
        baseline = results[label+"_full"]
        absorber = results[label+"_absorption_only"]
        lens = lens_fit(baseline, absorber)
        summaries[label] = dict(
            full=metrics(baseline), absorption_only=metrics(absorber),
            linear=metrics(results[label+"_linear"]), nonlinear_lens=lens)
        checks[label] = {}
        if not args.skip_refinement:
            variants = (
                ("transverse_grid", dict(samples=2*args.samples), tables[label]),
                ("longitudinal_step", dict(steps=2*args.cell_steps), tables[label]),
                ("window", dict(half_width_mm=1.5*args.half_width_mm,
                                samples=3*args.samples//2), tables[label]),
                ("intensity_table", {}, tables[label+"_refined"]),
            )
            for name, options, table in variants:
                print(f"refine {label}: {name}", flush=True)
                candidate = cell_and_camera(args, table, wavelength, **options)
                candidate_absorber = cell_and_camera(
                    args, table, wavelength, mode="absorption_only", **options)
                candidate_lens = lens_fit(candidate, candidate_absorber)
                checks[label][name] = compare_paths(
                    baseline, candidate, reference_lens=lens, candidate_lens=candidate_lens)
                checks[label][name]["absorption_control_budgets_passed"] = metrics(candidate_absorber)["passed"]
                checks[label][name]["passed"] &= checks[label][name]["absorption_control_budgets_passed"]
            max_difference = max(check["lens_power_absolute_difference_m_inverse"]
                                 for check in checks[label].values())
            checks[label]["sign_margin"] = dict(
                lens_power_m_inverse=lens["lens_power_m_inverse"],
                max_refinement_difference_m_inverse=float(max_difference),
                required_multiple=5.,
                passed=bool(abs(lens["lens_power_m_inverse"]) > 5*max_difference))
        else:
            checks[label]["refinement"] = dict(passed=False)
    signs = (summaries["red"]["nonlinear_lens"]["lens_power_m_inverse"] < 0
             and summaries["blue"]["nonlinear_lens"]["lens_power_m_inverse"] > 0)
    checks["sign_switch"] = dict(passed=bool(signs),
        definition="cell-exit phase curvature relative to intensity-dependent absorption-only control")
    checks["physical_budgets"] = dict(
        paths={name: metrics(result) for name, result in results.items()},
        criterion="0 <= T <= 1+1e-10; relative power balance and camera drift <=1e-9; edge fraction <=1e-5",
        passed=all(metrics(result)["passed"] for result in results.values()))
    checks["phase_fit"] = dict(
        criterion="RMS phase residual <10% of fitted quadratic phase at fixed physical aperture edge",
        passed=all(summaries[label]["nonlinear_lens"]["phase_fit_passed"] for label in ("red", "blue")))
    checks["passed"] = bool(signs and checks["physical_budgets"]["passed"] and checks["phase_fit"]["passed"]
                            and all(check["passed"] for label in ("red", "blue") for check in checks[label].values()))
    # Exact Gaussian vacuum reference along all observed planes.
    vacuum = results["vacuum"]
    waist = args.waist_mm*1e-3
    rayleigh = np.pi*waist**2/wavelength
    expected = waist*np.sqrt(1+(vacuum["z"]/rayleigh)**2)
    vacuum_error = float(np.max(abs(vacuum["widths"]/expected-1)))
    vacuum_power_error = float(np.max(abs(vacuum["powers"]/vacuum["powers"][0]-1)))
    checks["vacuum"] = dict(max_width_relative_error=vacuum_error,
                            passed=bool(vacuum_error < 1e-6 and vacuum_power_error <= 1e-9),
                            max_power_relative_drift=vacuum_power_error)
    checks["curvature_criterion"] = "same sign and <=1% relative difference in each refinement; baseline magnitude >5x maximum absolute difference"
    checks["passed"] = checks["passed"] and checks["vacuum"]["passed"]
    checks["weak_intensity"] = {}
    for label in ("red", "blue"):
        weak_args = SimpleNamespace(**dict(vars(args),
            peak_intensity_w_m2=args.peak_intensity_w_m2*1e-6))
        weak = cell_and_camera(weak_args, tables[label], wavelength)
        linear = cell_and_camera(weak_args, tables[label], wavelength, mode="linear")
        difference = compare_paths(linear, weak)
        difference["input_intensity_fraction"] = 1e-6
        difference["passed"] = bool(difference["physical_budgets_passed"]
                                    and difference["max_width_relative_difference"] < 1e-4
                                    and difference["transmission_relative_difference"] < 1e-4
                                    and difference["camera_cut_total_variation"] < 1e-4)
        checks["weak_intensity"][label] = difference
    checks["passed"] = checks["passed"] and all(
        check["passed"] for check in checks["weak_intensity"].values())
    return results, summaries, checks


def save_arrays(run, results):
    payload = {}
    for name, result in results.items():
        payload.update({name+"_"+key: value for key, value in (
            ("z_m", result["z"]), ("x_m", result["grid"].x_m),
            ("radius_m", result["widths"]), ("power_w", result["powers"]),
            ("centerline_intensity_w_m2", result["cuts"]),
            ("exit_field_sqrt_w_m2", result["cell"].final_field),
            ("camera_field_sqrt_w_m2", result["camera"].final_field))})
    np.savez_compressed(run.path("propagation.npz"), **payload)


def plot_results(run, args, results, summaries, tables, passed):
    plt = figures.pyplot()
    colors = dict(red="#bc3f43", blue="#2269b0", vacuum="#35414c", absorption_only="#b5a581")
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    layout = fig.add_gridspec(3, 3, height_ratios=(1.05, 1., .95))
    views = (("red_full", f"RED  {-args.detuning_mhz/1000:+g} GHz"), ("vacuum", "VACUUM REFERENCE"),
             ("blue_full", f"BLUE  {args.detuning_mhz/1000:+g} GHz"))
    for column, (name, title) in enumerate(views):
        ax = fig.add_subplot(layout[0, column])
        result = results[name]
        image = ax.pcolormesh(result["z"]*100, result["grid"].x_m*1e3,
                             result["cuts"].T/args.peak_intensity_w_m2,
                             shading="auto", cmap="magma", vmin=0., vmax=2.)
        ax.axvline(args.cell_length_mm/10, color="white", lw=1, ls="--")
        ax.set(xlabel="Distance from cell entrance (cm)", ylabel="x (mm)",
               title=title, ylim=(-1.2, 1.2))
    fig.colorbar(image, ax=fig.axes[:3], label="Intensity / input peak", shrink=.85)
    ax = fig.add_subplot(layout[1, :2])
    for label in ("red", "blue"):
        ax.plot(results[label+"_full"]["z"]*100, results[label+"_full"]["widths"]*1e3,
                color=colors[label], lw=2.3, label=label.title()+": full complex response")
        ax.plot(results[label+"_absorption_only"]["z"]*100,
                results[label+"_absorption_only"]["widths"]*1e3,
                color=colors[label], lw=1.4, ls="--", label=label.title()+": absorption only")
    ax.plot(results["vacuum"]["z"]*100, results["vacuum"]["widths"]*1e3,
            color=colors["vacuum"], lw=1.5, ls=":", label="Vacuum")
    ax.axvspan(0, args.cell_length_mm/10, color="#d9dfe4", alpha=.5)
    ax.set(xlabel="Distance from cell entrance (cm)", ylabel="Second-moment radius (mm)",
           title="Beam size through and after the cell")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    ax = fig.add_subplot(layout[1, 2])
    for label in ("red", "blue"):
        result = results[label+"_full"]
        row = int(np.argmin(abs(result["grid"].y_m)))
        full = result["cell"].final_field[row]
        ref = results[label+"_absorption_only"]["cell"].final_field[row]
        x = result["grid"].x_m
        mask = abs(x) <= args.waist_mm*1e-3
        phase = np.unwrap(np.angle(full[mask]*ref[mask].conj()))
        phase -= np.interp(0., x[mask], phase)
        ax.plot(x[mask]*1e3, phase, color=colors[label], lw=2, label=label.title())
    ax.axhline(0, color="#9aa4aa", lw=.7)
    ax.set(xlabel="x at cell exit (mm)", ylabel="Relative phase (rad)",
           title="Nonlinear wavefront; centre = 0")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = fig.add_subplot(layout[2, 0])
    for label in ("red", "blue"):
        result = results[label+"_full"]
        x = result["grid"].x_m*1e3
        intensity = result["camera"].centerline_intensity_w_m2[-1]
        ax.plot(x, intensity/args.peak_intensity_w_m2, color=colors[label], lw=2, label=label.title())
    ax.set(xlabel="x at camera (mm)", ylabel="Intensity / input peak",
           title=f"Camera: {args.camera_distance_m*100:g} cm after cell", xlim=(-1.2, 1.2))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = fig.add_subplot(layout[2, 1])
    intensities = np.linspace(0., args.peak_intensity_w_m2*1.3, 150)
    for label in ("red", "blue"):
        delta_n = .5*(tables[label](intensities).real-tables[label](0.).real)
        ax.plot(intensities/1e4, delta_n*1e6, color=colors[label], lw=2, label=label.title())
    ax.axhline(0, color="#9aa4aa", lw=.7)
    ax.set(xlabel="Local intensity (W/cm^2)", ylabel="n(I) - n(0) (ppm)",
           title="Existing full 16-state OBE")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = fig.add_subplot(layout[2, 2])
    ax.axis("off")
    lines = ["NUMERICAL CHECKS PASSED" if passed else "EXPLORATORY / CHECKS INCOMPLETE", ""]
    for label in ("red", "blue"):
        data = summaries[label]
        lines += [label.upper(),
                  f"T = {data['full']['transmission']:.4f}",
                  f"1/f_NL = {data['nonlinear_lens']['lens_power_m_inverse']:+.3f} m^-1",
                  f"phase-fit RMS = {data['nonlinear_lens']['phase_rms_rad']:.2g} rad",
                  f"camera radius = {data['full']['camera_radius_mm']:.3f} mm", ""]
    lines += ["Local steady-state model; fixed transit reset.",
              "No experimental validation or thermal lens."]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="monospace")
    power_mw = args.peak_intensity_w_m2*np.pi*(args.waist_mm*1e-3)**2/2*1e3
    headline = ("Rb87 D1: red self-defocusing / blue self-focusing" if passed
                else "Rb87 D1: detuning-dependent beam self-action (unverified)")
    fig.suptitle(headline+"\n"
                 f"{args.temperature_c:g} C | L={args.cell_length_mm:g} mm | "
                 f"w_in={args.waist_mm:g} mm | P={power_mw:.3f} mW | "
                 f"reset={args.reset_rate:.2g} s^-1",
                 fontsize=16, weight="bold")
    figures.save_figure(fig, run.path("self_action_validation"), ("png", "pdf"), dpi=180)


def main(argv=None):
    args = parse_args(argv)
    atom = magneto_multilevel.build_multilevel_atom("Rb87", "D1")
    wavelength = constants.C_LIGHT/atom.frequency_hz
    config = vars(args).copy()
    config["reuse_tables"] = str(args.reuse_tables) if args.reuse_tables else None
    config.update(scipy_version=scipy.__version__,
                  isotope_density_m3=species.number_density(species.ISOTOPES["Rb87"],
                                                           args.temperature_c+273.15),
                  wavelength_m=wavelength, model="pure Rb87 D1 full 16-state OBE",
                  polarization="x-linear", magnetic_field_t=0.,
                  input_wavefront="waist at cell entrance; M2=1",
                  detuning_reference="laser minus upper F=2 line-strength centroid",
                  claim="local steady-state numerical model, not experimental validation",
                  reset_provenance="AI-chosen effective replacement rate, not a measured transit rate",
                  exclusions=["transverse atomic transport", "thermal lensing", "buffer gas",
                              "collisional broadening", "spin exchange", "window optics",
                              "coupled FWM fields", "SABES UI"])
    with start_run("beam_self_action", out=args.out, config=config, argv=argv) as run:
        write_json(run.path("source_manifest.json"), [provenance.file_record(REPO_ROOT/p) for p in SOURCES])
        # Preserve the actual uncommitted source bytes behind each result hash.
        for source in SOURCES:
            run.path("sources", source).write_bytes((REPO_ROOT/source).read_bytes())
        tables, atomic_checks = make_tables(args, run, wavelength)
        results, summaries, propagation_checks = calculate(args, tables, wavelength)
        passed = bool(atomic_checks["passed"] and propagation_checks["passed"])
        evidence = dict(passed=passed, claim=config["claim"], config=config,
                        atomic=atomic_checks, propagation=propagation_checks, readouts=summaries)
        write_json(run.path("convergence.json"), evidence)
        save_arrays(run, results)
        plot_results(run, args, results, summaries, tables, passed)
        report = ("# \u27e6AI\u27e7 Rb87 D1 \uB2E8\uC77C \uBE54 \uAC80\uC99D\n\n"
                  + ("\uC218\uCE58 \uAC80\uC99D \uD1B5\uACFC." if passed else "\uC218\uCE58 \uAC80\uC99D \uBBF8\uC644\uB8CC.")
                  + " \uC2E4\uD5D8 \uAC80\uC99D \uC544\uB2D8. SABES UI \uBBF8\uC5F0\uACB0.\n\n"
                  + "![validation](self_action_validation.png)\n\n"
                  + "\uC870\uAC74\u00B7\uC218\uB834\u00B7\uD55C\uACC4: convergence.json. \uC6D0\uC790 \uC751\uB2F5: susceptibility_tables.npz.\n"
                  + "\uC804\uD30C \uBC30\uC5F4: propagation.npz. \uCF54\uB4DC \uD574\uC2DC: source_manifest.json.\n")
        write_text(run.path("report.md"), report)
        print(json.dumps(dict(output=str(run.dir), passed=passed, readouts=summaries), indent=2), flush=True)
        if not passed:
            raise RuntimeError("numerical validation did not pass; inspect convergence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
