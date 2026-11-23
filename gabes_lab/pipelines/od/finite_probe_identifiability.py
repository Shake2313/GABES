"""Synthetic recovery study for the finite-probe OD fit variables.

Run: python -m gabes_lab.pipelines.od.finite_probe_identifiability [--workers 4]
Inputs: noiseless spectra from the shared OD evaluator plus seeded Gaussian
noise; natural or custom Rb, D1/D2, 75 mm cell, 1 mm 1/e² probe radius.
Outputs: fits.jsonl (one line per fit, written as fits finish), summary.json
and summary.md under runs/od/finite_probe_identifiability/.

Checklist item od-sas-autood-finite-probe-fit-extension: which of probe power,
beam radius and ⁸⁵Rb fraction can be estimated, with what bias, calibrated
uncertainty (pull) and correlation. Power and radius enter the rate model only
through 2P/(πw²), so power is fitted at a fixed radius.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from gabes_lab.blocks.io import write_json, write_text
from gabes_lab.blocks.runner import start_run

RATE = "od_finite_rb_mf_rate_v1"
TRUTH_OFFSET_MHZ = 7.0
SPAN_MHZ = {"D1": (-4500.0, 6500.0), "D2": (-3500.0, 5000.0)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--sigma", type=float, default=2e-3)
    parser.add_argument("--lines", nargs="+", default=["D1", "D2"])
    parser.add_argument("--temps", nargs="+", type=float, default=[30.0, 50.0])
    parser.add_argument("--powers", nargs="+", type=float, default=[5.0, 25.0, 100.0])
    parser.add_argument("--quick", action="store_true", help="one seed, one line, one temperature")
    return parser.parse_args(argv)


def cases(args):
    """Each case: truth, starting point, free variables and model."""
    out = []
    for line in args.lines:
        for temp in args.temps:
            for seed in range(args.seeds):
                low = seed % 2 == 0          # alternate the starting side
                for power in args.powers:
                    truth = dict(line=line, temp_c=temp, probe_power_uw=power)
                    start = dict(temp_c=temp - 3.0 if low else temp + 3.0,
                                 probe_power_uw=power * (0.5 if low else 2.0))
                    out.append(dict(study="S0_temp_offset", truth=truth, start=dict(temp_c=start["temp_c"]),
                                    free=["temp_c", "offset_mhz"], model=RATE, seed=seed))
                    out.append(dict(study="S1_plus_power", truth=truth, start=start,
                                    free=["temp_c", "offset_mhz", "probe_power_uw"], model=RATE, seed=seed))
                for fraction in (0.5, 0.7217):
                    start_fraction = 0.85 if low else 0.35
                    weak = dict(line=line, temp_c=temp, probe_power_uw=0.0, rb85_fraction=fraction)
                    out.append(dict(study="S2_fraction_weak", truth=weak,
                                    start=dict(temp_c=temp - 3.0, rb85_fraction=start_fraction),
                                    free=["temp_c", "offset_mhz", "rb85_fraction"], model=None, seed=seed))
                    finite = dict(weak, probe_power_uw=25.0)
                    out.append(dict(study="S2_fraction_rate", truth=finite,
                                    start=dict(temp_c=temp - 3.0, rb85_fraction=start_fraction),
                                    free=["temp_c", "offset_mhz", "rb85_fraction"], model=RATE, seed=seed))
                    out.append(dict(study="S3_power_and_fraction", truth=finite,
                                    start=dict(temp_c=temp - 3.0, rb85_fraction=start_fraction,
                                               probe_power_uw=50.0 if low else 12.5),
                                    free=["temp_c", "offset_mhz", "probe_power_uw", "rb85_fraction"],
                                    model=RATE, seed=seed))
    return out


def _params(values):
    from gabes.schemes.sas import RB_COMPOSITION_CUSTOM, SASScheme
    p = dict(SASScheme().defaults(), species="Rb (natural)", pump_power_mw=0.0,
             probe_waist_mm=1.0, cell_mm=75.0)
    p.update({k: v for k, v in values.items()})
    if "rb85_fraction" in values:
        p["rb_composition"] = RB_COMPOSITION_CUSTOM
    return p


def run_case(case, points, sigma):
    from gabes.od_spectrum_fit import ODFitInput, fit_od_spectrum, od_transmission
    started = time.perf_counter()
    truth = _params(case["truth"])
    x = np.linspace(*SPAN_MHZ[truth["line"]], points)
    clean = od_transmission(x, truth, offset_mhz=TRUTH_OFFSET_MHZ, model_id=case["model"])
    seed = zlib.crc32(json.dumps([case["study"], case["truth"], case["seed"]], sort_keys=True).encode())
    rng = np.random.default_rng([case["seed"], seed])
    y = clean + rng.normal(0.0, sigma, x.size)
    data = ODFitInput(x, y, np.full_like(x, sigma), 0, x.size, "synthetic Gaussian noise")
    start = _params(dict(case["truth"], **case["start"]))
    record = dict(case, points=points, sigma=sigma)
    try:
        fit = fit_od_spectrum(data, start, free=tuple(case["free"]), model_id=case["model"])
    except Exception as error:          # noqa: BLE001 - recorded, not hidden
        record.update(status="error", error=f"{type(error).__name__}: {error}",
                      seconds=time.perf_counter() - started)
        return record
    expected = dict(case["truth"], offset_mhz=TRUTH_OFFSET_MHZ)
    estimates = []
    for q in fit["parameters"]:
        value, se = q["value"], q["standard_error"]
        true = expected[q["key"]]
        estimates.append(dict(key=q["key"], truth=true, value=value, standard_error=se,
                              pull=None if se in (None, 0) else (value - true) / se,
                              on_bound=q["on_bound"]))
    corr = fit["correlation"]
    off = None if corr is None else float(np.max(np.abs(np.array(corr) - np.eye(len(corr))))) if len(corr) > 1 else 0.0
    record.update(status=fit["status"], identifiable=fit["identifiable"],
                  condition_number=fit["condition_number"], max_abs_correlation=off,
                  correlation=corr, reduced_chi_squared=fit["reduced_chi_squared"],
                  evaluations=fit["global_evaluations"] + fit["local_evaluations"],
                  estimates=estimates, probe_peak_intensity=fit.get("probe_peak_intensity"),
                  seconds=time.perf_counter() - started)
    return record


def summarize(records):
    groups = {}
    for r in records:
        groups.setdefault(r["study"], []).append(r)
    summary = {}
    for study, rows in sorted(groups.items()):
        ok = [r for r in rows if r["status"] != "error"]
        entry = dict(fits=len(rows), errors=len(rows) - len(ok),
                     identifiable=sum(bool(r.get("identifiable")) for r in ok),
                     max_abs_correlation=[min((r["max_abs_correlation"] for r in ok
                                               if r.get("max_abs_correlation") is not None), default=None),
                                          max((r["max_abs_correlation"] for r in ok
                                               if r.get("max_abs_correlation") is not None), default=None)],
                     median_seconds=float(np.median([r["seconds"] for r in rows])) if rows else None,
                     parameters={})
        keys = {e["key"] for r in ok for e in r["estimates"]}
        for key in sorted(keys):
            est = [e for r in ok for e in r["estimates"] if e["key"] == key]
            pulls = np.array([e["pull"] for e in est if e["pull"] is not None])
            rel = np.array([(e["value"] - e["truth"]) / e["truth"] for e in est if e["truth"]])
            entry["parameters"][key] = dict(
                n=len(est), with_uncertainty=int(pulls.size), on_bound=sum(e["on_bound"] for e in est),
                pull_rms=float(np.sqrt(np.mean(pulls**2))) if pulls.size else None,
                coverage_2sigma=float(np.mean(np.abs(pulls) < 2)) if pulls.size else None,
                median_relative_error=float(np.median(np.abs(rel))) if rel.size else None,
                max_relative_error=float(np.max(np.abs(rel))) if rel.size else None)
        summary[study] = entry
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.seeds, args.lines, args.temps = 1, args.lines[:1], args.temps[:1]
    todo = cases(args)
    with start_run("od/finite_probe_identifiability", out=args.out, config=vars(args)) as run:
        records = []
        with open(run.path("fits.jsonl"), "w", encoding="utf-8") as sink, \
                ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, case, args.points, args.sigma) for case in todo]
            for done, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"[{done}/{len(todo)}] {record['study']} {record['truth']} "
                      f"{record['status']} {record['seconds']:.0f}s", flush=True)
        summary = summarize(records)
        write_json(run.path("summary.json"), summary)
        lines = ["| study | fits | identifiable | max abs rho | parameter | pull RMS | 2σ coverage | median rel err | max rel err | on bound |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for study, entry in summary.items():
            for key, p in entry["parameters"].items():
                rho = entry["max_abs_correlation"]
                lines.append(f"| {study} | {entry['fits']} | {entry['identifiable']} | "
                             f"{rho[0]}–{rho[1]} | {key} | {p['pull_rms']} | {p['coverage_2sigma']} | "
                             f"{p['median_relative_error']} | {p['max_relative_error']} | {p['on_bound']} |")
        write_text(run.path("summary.md"), "\n".join(lines) + "\n")
        print(run.path("summary.md"))


if __name__ == "__main__":
    main()
