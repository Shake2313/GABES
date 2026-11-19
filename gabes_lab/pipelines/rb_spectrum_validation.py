"""Validate local full-OBE Rb D1/D2 spectra over all hyperfine components.

Run: python -m gabes_lab.pipelines.rb_spectrum_validation
Inputs: CLI total density/temperature, intensity and Maxwell quadrature settings.
Outputs: raw NPZ, PNG/PDF spectra, checks and provenance under runs/rb_spectrum/.
This is a local-response numerical campaign; it does not propagate a beam.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from gabes import constants, linear_hyperfine, species
from gabes.nonlinear_response import solve_rb_mean
from gabes.optical_frequency import RbOpticalFrequency, reference_offset_hz
from gabes.plot_style import apply_gabes_plot_style
from gabes_lab.blocks import figures, provenance
from gabes_lab.blocks.io import read_json, write_json
from gabes_lab.blocks.paths import REPO_ROOT
from gabes_lab.blocks.runner import start_run


FRACTIONS = (0., .5, species.RB85.abundance, 1.)
LABELS = ("Pure Rb87", "50/50", "Natural Rb", "Pure Rb85")
COLORS = ("#255f91", "#ad5c29", "#298067", "#7b4a8b")
SOURCES = (
    "gabes/nonlinear_response.py", "gabes/optical_bloch.py",
    "gabes/optical_frequency.py", "gabes/psr_full_obe.py",
    "gabes/psr_conventions.py", "gabes/magneto_multilevel.py",
    "gabes/species.py", "gabes/angular.py", "gabes/constants.py",
    "gabes/core.py", "gabes/atoms.py", "gabes/linear_hyperfine.py",
    "gabes/doppler.py", "gabes/quantum/reservoirs.py",
    "gabes_lab/pipelines/rb_spectrum_validation.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--reuse-responses", type=Path,
                        help="Reuse source-checked sampled OBE data; rebuild references/checks/figures.")
    parser.add_argument("--temperature-c", type=float, default=100.)
    parser.add_argument("--density-m3", type=float)
    parser.add_argument("--reset-rate", type=float, default=5e5)
    parser.add_argument("--intensity-w-m2", type=float, default=2e4)
    parser.add_argument("--velocity-order", type=int, default=6)
    parser.add_argument("--refined-order", type=int, default=8)
    parser.add_argument("--cutoff-sigma", type=float, default=6.)
    parser.add_argument("--tail-sigma", type=float, default=7.)
    parser.add_argument("--wing-mhz", type=float, default=1500.)
    parser.add_argument("--dense-points", type=int, default=1601)
    parser.add_argument("--reference-rtol", type=float, default=2e-5)
    parser.add_argument("--convergence-rtol", type=float, default=.005)
    parser.add_argument("--cell-length-mm", type=float, default=50.,
                        help="Length used only to express response errors as phase/optical depth.")
    args = parser.parse_args(argv)
    scalars = [args.temperature_c, args.reset_rate, args.intensity_w_m2,
               args.cutoff_sigma, args.tail_sigma, args.wing_mhz,
               args.reference_rtol, args.convergence_rtol, args.cell_length_mm]
    if (not np.isfinite(scalars).all() or args.temperature_c <= -273.15
            or min(scalars[1:]) <= 0 or args.velocity_order < 2
            or args.refined_order <= args.velocity_order or args.tail_sigma <= args.cutoff_sigma
            or args.dense_points < 101):
        parser.error("finite positive physical/numerical settings and strict order/tail refinements required")
    if args.density_m3 is not None and (not np.isfinite(args.density_m3) or args.density_m3 <= 0):
        parser.error("density-m3 must be finite and positive")
    return args


def scan_definition(line, wing_mhz):
    """Coordinates for allowed transitions; all atomic data remain shared."""
    transitions = []
    for isotope in ("Rb85", "Rb87"):
        iso = species.ISOTOPES[isotope]
        hwhm_hz = iso.line(line)[2] * 1e6 / 2
        for ground_f, excited_f in species.line_integrated_alpha(iso, line, 1.):
            center = reference_offset_hz(isotope, line, reference="hyperfine_transition",
                                         ground_f=ground_f, excited_f=excited_f)
            transitions.append(dict(isotope=isotope, ground_f=ground_f, excited_f=excited_f,
                                    offset_hz=center, natural_hwhm_hz=hwhm_hz))
    centers = np.array([row["offset_hz"] for row in transitions])
    wings = (float(centers.min() - wing_mhz * 1e6),
             float(centers.max() + wing_mhz * 1e6))
    weak = np.unique([*wings, *(row["offset_hz"] + sign * row["natural_hwhm_hz"]
                               for row in transitions for sign in (-1, 0, 1))])
    finite = np.unique([*wings, *centers])
    refine = list(wings)
    for isotope in ("Rb85", "Rb87"):
        row = max((r for r in transitions if r["isotope"] == isotope),
                  key=lambda r: (r["ground_f"], r["excited_f"]))
        refine.append(row["offset_hz"])
    return dict(transitions=transitions, weak_hz=weak, finite_hz=finite,
                refine_hz=np.unique(refine), span_hz=wings)


@lru_cache(maxsize=4)
def _manifold(isotope, line):
    return species.build_manifold(species.ISOTOPES[isotope], line)


def analytic_reference(line, offsets_hz, fractions, density, temperature_c, reset_rate):
    """Compose the shared independent scalar-strength, exact-Maxwell Voigt reference."""
    offsets = np.asarray(offsets_hz, dtype=float)
    pure = {}
    physical_k = 2 * np.pi * (species.RB85.line(line)[1] + offsets) / constants.C_LIGHT
    for isotope in ("Rb85", "Rb87"):
        manifold = _manifold(isotope, line)
        isotope_offset = reference_offset_hz(isotope, line)
        alpha, real = linear_hyperfine.species_complex_voigt(
            offsets * 2 * np.pi, manifold, density, temperature_c + 273.15,
            manifold.gamma + 2 * reset_rate, offset=isotope_offset * 2 * np.pi,
            doppler_wavevector_per_m=physical_k)
        pure[isotope] = real + 1j * alpha / manifold.k_vec
    return np.asarray([f * pure["Rb85"] + (1 - f) * pure["Rb87"] for f in fractions])


def _comparison(actual, expected, rtol, absolute_floor=1e-12):
    """Separate real/imaginary scales, with an absolute floor at zero crossings."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    checks = {}
    for name, a, b in (("real", actual.real, expected.real),
                       ("imaginary", actual.imag, expected.imag)):
        error = np.abs(a - b)
        scaled = error / (absolute_floor + rtol * np.abs(b))
        checks[name] = dict(max_absolute=float(error.max()), max_scaled=float(scaled.max()))
    return dict(passed=all(row["max_scaled"] <= 1 for row in checks.values()), checks=checks)


def _response_comparison(old, new, line, offset_hz, args):
    # Express real and imaginary error independently as phase and optical depth.
    frequency = RbOpticalFrequency(line, offset_hz)
    k_length = 2 * np.pi / frequency.wavelength_m * args.cell_length_mm * 1e-3
    phase_old = k_length * np.asarray(old).real / 2
    phase_new = k_length * np.asarray(new).real / 2
    od_old = k_length * np.asarray(old).imag
    od_new = k_length * np.asarray(new).imag
    results = {}
    for name, a, b in (("phase_rad", phase_old, phase_new),
                       ("optical_depth", od_old, od_new)):
        error = np.abs(a - b)
        scaled = error / (1e-4 + args.convergence_rtol * np.maximum(np.abs(a), np.abs(b)))
        results[name] = dict(max_absolute=float(error.max()), max_scaled=float(scaled.max()))
    return dict(passed=all(row["max_scaled"] <= 1 for row in results.values()), checks=results)


def compute_line(line, args, density, component_cache):
    scan = scan_definition(line, args.wing_mhz)
    atomic = dict(temp_c=args.temperature_c, ground_relaxation_hz=args.reset_rate,
                  density_m3=density)
    call_count = 0
    max_health = dict(trace_error=0., hermiticity_error=0., steady_state_relative_residual=0.)
    min_eigenvalue = 1.

    def evaluate(offsets, intensity, order, cutoff):
        nonlocal call_count, min_eigenvalue
        values = np.empty((len(FRACTIONS), len(offsets)), complex)
        for j, offset in enumerate(offsets):
            frequency = RbOpticalFrequency(line, float(offset))
            for i, fraction in enumerate(FRACTIONS):
                result = solve_rb_mean(frequency, intensity, rb85_fraction=fraction,
                                      velocity_order=order, cutoff_sigma=cutoff,
                                      component_cache=component_cache, **atomic)
                values[i, j] = result["chi"]
                for key in max_health:
                    max_health[key] = max(max_health[key], result["diagnostics"][key])
                min_eigenvalue = min(min_eigenvalue, result["diagnostics"]["min_eigenvalue"])
            call_count += 1
            if j % 10 == 0 or j == len(offsets) - 1:
                print(f"{line}: I={intensity:g} W/m2, order={order}, cutoff={cutoff:g}, "
                      f"frequency {j + 1}/{len(offsets)}", flush=True)
        return values

    raw = dict(
        weak_hz=scan["weak_hz"],
        weak_chi=evaluate(scan["weak_hz"], 0., args.velocity_order, args.cutoff_sigma),
        finite_hz=scan["finite_hz"],
        finite_chi=evaluate(scan["finite_hz"], args.intensity_w_m2,
                            args.velocity_order, args.cutoff_sigma),
        refinement_hz=scan["refine_hz"],
        refined_chi=evaluate(scan["refine_hz"], args.intensity_w_m2,
                             args.refined_order, args.cutoff_sigma),
        tail_chi=evaluate(scan["refine_hz"], args.intensity_w_m2,
                          args.refined_order, args.tail_sigma))
    return finalize_line(line, raw, scan, args, density,
                         dict(**max_health, min_eigenvalue=min_eigenvalue), call_count)


def finalize_line(line, raw, scan, args, density, health, call_count):
    """Rebuild reference comparisons without repeating expensive atomic samples."""
    weak, finite = raw["weak_chi"], raw["finite_chi"]
    refined, tail = raw["refined_chi"], raw["tail_chi"]
    reference = analytic_reference(line, scan["weak_hz"], FRACTIONS, density,
                                   args.temperature_c, args.reset_rate)
    linear_at_finite = analytic_reference(line, scan["finite_hz"], FRACTIONS, density,
                                          args.temperature_c, args.reset_rate)
    refine_records = []
    for j, offset in enumerate(scan["refine_hz"]):
        base_index = int(np.flatnonzero(scan["finite_hz"] == offset)[0])
        linear = linear_at_finite[:, base_index]
        base = finite[:, base_index]
        refine_records.append(dict(offset_hz=float(offset),
            order_total=_response_comparison(base, refined[:, j], line, offset, args),
            order_nonlinear=_response_comparison(base - linear, refined[:, j] - linear,
                                                 line, offset, args),
            tail_total=_response_comparison(refined[:, j], tail[:, j], line, offset, args),
            tail_nonlinear=_response_comparison(refined[:, j] - linear, tail[:, j] - linear,
                                                line, offset, args)))
    weak_check = _comparison(weak, reference, args.reference_rtol)
    mixing = {}
    for name, array in (("weak", weak), ("finite", finite)):
        expected = np.asarray([f * array[-1] + (1 - f) * array[0] for f in FRACTIONS])
        mixing[name] = _comparison(array, expected, 1e-12, absolute_floor=1e-18)
    passivity = bool(np.all(weak.imag >= 0.) and np.all(finite.imag >= 0.)
                     and np.all(refined.imag >= 0.) and np.all(tail.imag >= 0.))
    refined_pass = all(row[key]["passed"] for row in refine_records
                       for key in ("order_total", "order_nonlinear", "tail_total", "tail_nonlinear"))
    dense_hz = np.linspace(*scan["span_hz"], args.dense_points)
    dense_reference = analytic_reference(line, dense_hz, FRACTIONS, density,
                                         args.temperature_c, args.reset_rate)
    raw.update(weak_reference=reference, linear_at_finite=linear_at_finite,
               dense_hz=dense_hz, dense_reference=dense_reference)
    checks = dict(passed=weak_check["passed"] and refined_pass and passivity
                  and all(row["passed"] for row in mixing.values()),
                  weak_complex_voigt=weak_check, finite_quadrature=refine_records,
                  passivity=passivity, mixture_identity=mixing,
                  sampled_transition_count=len(scan["transitions"]),
                  weak_frequency_count=len(scan["weak_hz"]),
                  finite_frequency_count=len(scan["finite_hz"]),
                  atomic_health=health, transitions=scan["transitions"],
                  evaluated_frequency_groups=call_count)
    return raw, checks


def reuse_context(args, density):
    folder = args.reuse_responses.resolve()
    record = read_json(folder / "run.json")
    if record["status"] != "complete":
        raise ValueError("only completed raw-response runs can be reused")
    expected = dict(vars(args), total_elemental_density_m3=density)
    keys = ("temperature_c", "reset_rate", "intensity_w_m2", "velocity_order",
            "refined_order", "cutoff_sigma", "tail_sigma", "wing_mhz",
            "total_elemental_density_m3")
    if (any(record["config"].get(key) != expected[key] for key in keys)
            or record["config"].get("rb85_fractions") != list(FRACTIONS)):
        raise ValueError("raw-response physical or quadrature settings differ")
    # References and report code can change; the physics producing each saved
    # OBE sample must match exactly. Stored raw-file hashes prevent stale edits.
    old_sources = {row["path"]: row["sha256"]
                   for row in read_json(folder / "source_manifest.json")["sources"]}
    atomic_sources = [path for path in SOURCES
                      if path not in ("gabes/linear_hyperfine.py", SOURCES[-1])]
    for path in atomic_sources:
        if old_sources.get(path) != provenance.sha256_file(REPO_ROOT / path):
            raise ValueError(f"raw-response atomic source changed: {path}")
    outputs = {row["path"]: row["sha256"] for row in record["outputs"]}
    evidence = []
    for line in ("d1", "d2"):
        for suffix in ("response.npz", "checks.json"):
            file = folder / f"{line}_{suffix}"
            if outputs.get(file.name) != provenance.sha256_file(file):
                raise ValueError(f"raw-response output changed: {file.name}")
            evidence.append(provenance.file_record(file))
    return dict(folder=folder, evidence=dict(source_run=provenance.portable_path(folder),
                input_files=evidence, unchanged_atomic_sources=atomic_sources))


def reuse_line(line, args, density, context):
    folder = context["folder"]
    with np.load(folder / f"{line.lower()}_response.npz", allow_pickle=False) as saved:
        raw = {key: saved[key].copy() for key in saved.files}
    scan = scan_definition(line, args.wing_mhz)
    for stored, expected in (("weak_hz", "weak_hz"), ("finite_hz", "finite_hz"),
                             ("refinement_hz", "refine_hz")):
        if not np.array_equal(raw[stored], scan[expected]):
            raise ValueError(f"raw-response sample axis changed: {line} {stored}")
    for name, axis in (("weak_chi", "weak_hz"), ("finite_chi", "finite_hz"),
                       ("refined_chi", "refinement_hz"), ("tail_chi", "refinement_hz")):
        if raw[name].shape != (len(FRACTIONS), len(raw[axis])) or not np.isfinite(raw[name]).all():
            raise ValueError(f"invalid stored response array: {line} {name}")
    old_checks = read_json(folder / f"{line.lower()}_checks.json")
    return finalize_line(line, raw, scan, args, density, old_checks["atomic_health"],
                         old_checks["evaluated_frequency_groups"])


def plot_results(results, args, run):
    plt = figures.pyplot()
    weak_fig, weak_axes = plt.subplots(2, 2, figsize=(13.4, 8.), constrained_layout=True)
    nonlinear_fig, nonlinear_axes = plt.subplots(2, 2, figsize=(13.4, 8.), constrained_layout=True)
    for row, line in enumerate(("D1", "D2")):
        data = results[line]
        dense_k = 2 * np.pi * (species.RB85.line(line)[1] + data["dense_hz"]) / constants.C_LIGHT
        weak_k = 2 * np.pi * (species.RB85.line(line)[1] + data["weak_hz"]) / constants.C_LIGHT
        finite_k = 2 * np.pi * (species.RB85.line(line)[1] + data["finite_hz"]) / constants.C_LIGHT
        for i, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
            weak_axes[row, 0].plot(data["dense_hz"] / 1e9,
                                   data["dense_reference"][i].real, color=color, label=label)
            weak_axes[row, 0].plot(data["weak_hz"] / 1e9, data["weak_chi"][i].real,
                                   "o", color=color, ms=3, mfc="none")
            weak_axes[row, 1].plot(data["dense_hz"] / 1e9,
                                   dense_k * data["dense_reference"][i].imag, color=color)
            weak_axes[row, 1].plot(data["weak_hz"] / 1e9,
                                   weak_k * data["weak_chi"][i].imag,
                                   "o", color=color, ms=3, mfc="none")
            nonlinear_axes[row, 0].scatter(data["finite_hz"] / 1e9,
                (data["finite_chi"][i] - data["linear_at_finite"][i]).real / 2,
                color=color, label=label, s=20, alpha=.85)
            nonlinear_axes[row, 1].scatter(data["finite_hz"] / 1e9,
                finite_k * data["finite_chi"][i].imag, color=color, s=20, alpha=.85)
        weak_axes[row, 0].set(title=f"{line}: weak dispersion", ylabel="Re(chi)")
        weak_axes[row, 1].set(title=f"{line}: weak absorption", ylabel="alpha (1/m)")
        weak_axes[row, 1].set_yscale("log")
        nonlinear_axes[row, 0].set(title=f"{line}: intensity-induced index", ylabel="delta n")
        nonlinear_axes[row, 0].axhline(0, color="0.5", lw=.7)
        nonlinear_axes[row, 1].set(title=f"{line}: finite-intensity absorption", ylabel="alpha (1/m)")
        nonlinear_axes[row, 1].set_yscale("log")
        for axes in (weak_axes, nonlinear_axes):
            for ax in axes[row]:
                ax.set_xlabel(f"Laser offset from Rb85 {line} fine centroid (GHz)")
                ax.grid(alpha=.18)
                ax.ticklabel_format(axis="x", style="plain")
    weak_axes[0, 0].legend(ncol=2, fontsize=9)
    nonlinear_axes[0, 0].legend(ncol=2, fontsize=9)
    weak_fig.suptitle("Rb D-line weak spectra: exact-Maxwell Voigt curves and full-OBE samples\n"
                      f"T={args.temperature_c:g} C; reset={args.reset_rate:g} /s; total density shared across mixtures")
    nonlinear_fig.suptitle("Rb D-line local nonlinear response: sampled frequencies, no beam propagation\n"
                           f"I={args.intensity_w_m2:g} W/m2; T={args.temperature_c:g} C; no assumed red/blue sign law")
    for fig, stem in ((weak_fig, "rb_weak_spectra"), (nonlinear_fig, "rb_nonlinear_samples")):
        apply_gabes_plot_style(fig)
        figures.save_figure(fig, run.path(stem), ("png", "pdf"), dpi=180)


def main(argv=None):
    args = parse_args(argv)
    if (args.out is not None and args.reuse_responses is not None
            and Path(args.out).resolve() == args.reuse_responses.resolve()):
        raise ValueError("replayed output must differ from its raw-response source folder")
    density = (species.number_density(species.RB85, args.temperature_c + 273.15)
               if args.density_m3 is None else args.density_m3)
    config = dict(vars(args), total_elemental_density_m3=density,
                  rb85_fractions=FRACTIONS, fraction_labels=LABELS,
                  frequency_origin="fixed Rb85 fine centroid of each selected D line")
    config["reuse_responses"] = (None if args.reuse_responses is None
                                else provenance.portable_path(args.reuse_responses))
    sources = [provenance.file_record(REPO_ROOT / path) for path in SOURCES]
    reused = reuse_context(args, density) if args.reuse_responses else None
    with start_run("rb_spectrum", out=args.out, config=config, argv=argv) as run:
        cache, results, checks = {}, {}, {}
        for line in ("D1", "D2"):
            results[line], checks[line] = (reuse_line(line, args, density, reused) if reused
                                           else compute_line(line, args, density, cache))
            write_json(run.path(f"{line.lower()}_checks.json"), checks[line])
            np.savez_compressed(run.path(f"{line.lower()}_response.npz"), **results[line])
        plot_results(results, args, run)
        sources_unchanged = all(row["sha256"] == provenance.sha256_file(REPO_ROOT / row["path"])
                                for row in sources)
        passed = sources_unchanged and all(check["passed"] for check in checks.values())
        summary = dict(passed=passed, conditions=config, line_checks=checks,
            sources_unchanged=sources_unchanged, cached_component_points=len(cache),
            response_reuse=None if reused is None else reused["evidence"],
            interpretation="\u27e6AI\u27e7 Numerical validation of this local model at recorded samples; not experimental validation.",
            scope="Single selected D line, CW x-linear beam, B=0, full hyperfine/Zeeman OBE, independent Maxwell classes, fixed effective reset. No transverse atomic transport, buffer gas, spin exchange or thermal lens.",
            finite_spectra="Discrete samples, not a frequency-interpolated nonlinear model. Refinement covers declared line/wing samples, not every operating condition.",
            sign_scope="Multilevel and mixture spectra need not be odd in a detuning about an arbitrarily chosen transition.")
        write_json(run.path("checks.json"), summary)
        write_json(run.path("source_manifest.json"), dict(sources=sources, unchanged=sources_unchanged))
        print(f"OUTPUT {run.dir}", flush=True)
        print(f"NUMERICAL_CHECKS {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
