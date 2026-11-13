"""Generate three bounded current-model examples without experimental data."""
from __future__ import annotations

import argparse
import inspect
import os
from time import perf_counter

import numpy as np

from gabes import core, kernels
from gabes.schemes.absorption import LambdaScheme
from gabes.schemes.rydberg import RydbergEITScheme
from gabes.schemes.sas import SASScheme
from gabes_lab.blocks import figures, io, provenance
from gabes_lab.blocks.runner import start_run


def _conditions():
    sas, cpt, rydberg = SASScheme(), LambdaScheme("cpt"), RydbergEITScheme()
    return {
        "od": (sas, {**sas.defaults(), "species": "\u2078\u2075Rb", "line": "D1",
                     "scan_points": 601, "pump_power_mw": 0.0}, {
                         "species/line": "REFERENCE: Rb85 D1 species data",
                         "cell_mm/temp_c": "ASSUMED: selected uniform cell conditions",
                         "normalization": "REFERENCE: internal pump-off OD software closure",
                     }),
        "cpt": (cpt, cpt.defaults(), {
            "species/line": "REFERENCE: scalar D-line species data",
            "coupling/buffer": "ASSUMED: CPT preset drive and relaxation",
            "geometry": "ASSUMED: collinear and Doppler off in the CPT preset",
        }),
        "rydberg": (rydberg, rydberg.recommended_defaults({})["AT electrometry"], {
            "atomic_ladder": "REFERENCE: scalar Rb85 ladder and 37 GHz design",
            "rf_dipole": "REFERENCE: ARC 3.10.2 stretched sigma+ calculation",
            "detector_and_opacity": "ASSUMED: detector budget and effective opacity participation",
            "temperatures": "ASSUMED: linked heater, effective, and cold-spot preset",
        }),
    }


def _evaluate(name, scheme, params):
    if name == "rydberg":
        scheme = type(scheme)()
    raw = scheme.compute(params)
    view = scheme.headless_observables(raw, params)
    if name == "od":
        ro = view["optical_readout"]
        return dict(x=ro["x"], transmission=ro["transmission"], od=ro["optical_depth"]), dict(
            unit="GHz", status="MODELED", scope=ro["attenuation_scope"])
    if name == "cpt":
        ro = view["lambda_readout"]
        return dict(x=(ro["x_mhz"] - ro["center_mhz"]) * 1000.0,
                    transmission=ro["transmission"], dispersion=ro["dispersion"]), dict(
            unit="kHz", status=ro["window"]["status"], width_mhz=ro["window"]["width_mhz"],
            group_index=ro["window"]["group_index"], group_delay_s=ro["window"]["group_delay_s"],
            experimental_visibility=ro["experimental_visibility"])
    spectrum = view["rydberg_spectrum"]
    ro = view["rydberg_readout"]
    finite_if = spectrum["finite_if"]
    return dict(x=spectrum["x"], transmission=spectrum["transmission"],
                finite_if_x=finite_if["response_detuning_mhz"],
                field_asd=finite_if["total_field_asd_v_m_per_sqrt_hz"] * 1e7), dict(
        unit="MHz", status=ro["prediction_status"], opacity=ro["effective_opacity"],
        rf_dipole=ro["rf_dipole_provenance"], experimental_visibility=ro["experimental_visibility"],
        finite_if=ro["finite_if"], numerical_convergence=ro["numerical_convergence"])


def _plot(name, arrays, details, run):
    plt = figures.pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), layout="constrained")
    axes[0].plot(arrays["x"], arrays["transmission"])
    axes[0].set_ylabel("Transmission")
    if name == "od":
        axes[1].plot(arrays["x"], arrays["od"])
        axes[1].set_ylabel("Optical depth")
    elif name == "cpt":
        axes[1].plot(arrays["x"], arrays["dispersion"])
        axes[1].set_ylabel("Re chi")
        width = details["width_mhz"]
        if width is not None:
            for ax in axes:
                ax.set_xlim(-2000.0*width, 2000.0*width)
    else:
        positive = np.isfinite(arrays["field_asd"]) & (arrays["field_asd"] > 0.0)
        axes[1].semilogy(arrays["finite_if_x"][positive], arrays["field_asd"][positive])
        axes[1].set_ylabel("Total field ASD [nV/cm/sqrt(Hz)]")
    for ax in axes:
        ax.set_xlabel(f"Detuning [{details['unit']}]")
    commit = run.record["git"]["commit"] or "unknown"
    fig.suptitle(f"{name.upper()} | {details['status']} | {commit[:7]}")
    figures.save_figure(fig, run.path(name), dpi=135)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Output directory (default: runs/current_scheme_examples/...)")
    args = parser.parse_args(argv)
    conditions = _conditions()
    config = {name: {"inputs": params, "input_sources": sources, "cache_version": sc.cache_version}
              for name, (sc, params, sources) in conditions.items()}
    with start_run("current_scheme_examples", out=args.out, config=config, argv=argv) as run:
        paths = {__file__, inspect.getfile(core), inspect.getfile(kernels),
                 *(inspect.getfile(type(sc)) for sc, _, _ in conditions.values())}
        run.record["sources"] = [provenance.file_record(path) for path in sorted(paths)]
        run.record["config_sha256"] = provenance.sha256_bytes(io.dumps_json(config).encode("utf-8"))
        run.record["measured_data_status"] = "ABSENT"
        run.record["timing_conditions"] = {
            "blas_threads": 1, "numba_available": kernels.available(),
            "numba_thread_setting": os.environ.get("NUMBA_NUM_THREADS", "runtime default"),
            "method": "first and immediate repeated process-local compute; JIT disk cache may already exist",
        }
        with core.blas_single_thread():
            for name, (sc, params, _) in conditions.items():
                start = perf_counter()
                arrays, details = _evaluate(name, sc, params)
                first = perf_counter() - start
                np.savez(run.path(name + ".npz"), **arrays)
                start = perf_counter()
                fresh, _ = _evaluate(name, sc, params)
                second = perf_counter() - start
                with np.load(run.path(name + ".npz"), allow_pickle=False) as stored:
                    for key, values in fresh.items():
                        np.testing.assert_array_equal(stored[key], values)
                io.write_json(run.path(name + ".json"), {
                    **details, "first_s": first, "repeated_s": second,
                    "fresh_array_check": "exact equality", "figure": name + ".png",
                })
                _plot(name, arrays, details, run)
    print(run.dir)


if __name__ == "__main__":
    main()
