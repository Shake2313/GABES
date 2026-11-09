"""
Find the EFFICIENT FRONTIER of FWM twin-beam intensity-difference squeezing over
the (Δ, T) plane, at the HIGHEST fidelity the engine offers.

Δ = one-photon detuning (OPD) [GHz]; T = cell temperature [°C].
For each (Δ, T) the two-photon detuning δ is scanned (inside compute_spectrum) on
a SINGLE Raman branch and the deepest squeezing is taken:
    ξ(Δ, T) = min over δ of S_dB on the (−) branch.
Everything except Δ and T is held at the Sim et al. reference operating point.

FIDELITY (default: ultra -- the full model)
  --fidelity {legacy,balanced,fine,ultra}. `ultra` runs the complete propagation
  physics that compute_spectrum can model: self-consistent Δk_z phase mismatch
  (3 iterations), 64-segment propagation with dynamic per-segment pump depletion,
  in-cell propagation loss + technical-noise channels, Gaussian beam-overlap, and
  the 24-level Zeeman CG correction. `legacy` is the ideal lossless model (fast,
  good for a quick map); `balanced`/`fine` are intermediate.

WHAT 'THE FLOOR' MEANS AT EACH FIDELITY (why the target is generalized)
  - legacy: ξ saturates at the detection floor 10log10(1−η) and the whole
    high-gain region collapses onto it as a degenerate plateau -- "the maximum"
    is a meaningless plateau-edge coordinate.
  - ultra (any fidelity with in-cell loss): ξ no longer reaches the detection
    floor -- in-cell loss raises the achievable floor to ~1−η·e^(−OD), which grows
    with T (OD). ξ(T) therefore has a TRUE minimum (a real optimum operating
    point): low T is gain-limited, high T is loss-limited.
  So the target is generalized to the BEST ACHIEVABLE squeezing on the grid,
  ξ_opt = min over (Δ, T). The EFFICIENT FRONTIER is the lowest cell temperature
  (lowest gain / OD) that comes within `eps` dB of ξ_opt. Under legacy ξ_opt ==
  the detection floor, so this reduces to the original "reach the floor"
  definition. That point is experimentally cheapest and most robust to the
  excess-noise channels that limit experiments (anti-squeezed sum-noise ∝ G,
  spontaneous Raman ∝ pump, collisional dephasing ∝ density). The detection floor
  10log10(1−η) is KEPT as a hard guardrail: no point may fall below it (ultra's
  loss/noise only ever push ξ UP, never below 1−η).

DRIFT ROBUSTNESS: reference values are read from the bit-locked
  tests/test_regression.py::CONFIGS['sim_optimum'] (compute_spectrum's native
  kwargs), not hardcoded. `line_strength` there is the dimensionless residual
  (1.0); the physical coupling normalization p_F/[2(2I+1)] and the pure-85Rb CRC
  density live INSIDE compute_spectrum. If the anchor moves, the script fails loud.

SOLVE-ONCE / POST-PROCESS-η: the deepest-δ argmin is η-independent (in-cell loss
  tau_cell = e^(−OD) is one scalar per (Δ, T) spectrum, so S is monotone in
  S_ideal over the δ axis). We capture (G_s, G_c, segment-OD) at that δ and
  reconstruct ξ for any η from the segmented-loss formula -- the grid is solved
  ONCE and any η/eps follows.

TWIN-BEAM VALIDITY GATE (GAP_MIN/GAP_MAX): the ideal noise S_ideal=(G_s-G_c)^2
  /(G_s+G_c) is the exact bound only where G_s-G_c=1 (a genuine two-mode
  parametric amplifier). Away from the real FWM/Raman resonance -- background
  absorption/dispersion, or a gain pole under Ultra's segmented dynamics -- G_s
  and G_c decouple, and (G_s-G_c)^2 can vanish with NO real parametric gain,
  reporting spuriously perfect "squeezing". The deepest-delta argmin is
  restricted to delta points with G_s-G_c in [GAP_MIN, GAP_MAX]; a (Delta, T)
  point with no valid-gap delta in the window is excluded from the optimum and
  frontier search (like a delta-edge point) and counted separately.

Usage:
    python scripts/scan_squeezing_frontier.py                    # ultra frontier
    python scripts/scan_squeezing_frontier.py --fidelity legacy  # fast ideal map
    python scripts/scan_squeezing_frontier.py --loss 0           # lossless detect
    python scripts/scan_squeezing_frontier.py --metric gain
    python scripts/scan_squeezing_frontier.py --t-step 2.5       # finer min-T
NOTE: an ultra scan over the full default grid is heavy (64 segments x 3 phase
  iterations per point); use a coarser grid / fewer points while iterating.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))

# Windows consoles are often cp949 (Korean) and crash on chars like U+2212.
# Console output below is ASCII; this reconfigure is a belt-and-suspenders net.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Walk up to the repo root (the dir that contains the `gabes` package).
for cand in [ROOT, *ROOT.parents]:
    if (cand / "gabes").is_dir() and (cand / "tests").is_dir():
        REPO = cand
        break
else:  # pragma: no cover
    raise SystemExit("Could not locate the GABES repo root (needs gabes/ + tests/).")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from gabes.schemes import fwm                         # noqa: E402
from gabes.core import blas_single_thread             # noqa: E402
from gabes import observables                         # noqa: E402

# Fidelity label -> compute_spectrum kwargs. `legacy` passes nothing (defaults to
# phase_detail=legacy). The others switch on the matching propagation refinements;
# `ultra` is the full model (in-cell loss, segments, phase, Zeeman, overlap).
FIDELITY_KW = {
    "legacy":   dict(phase_detail=fwm.PHASE_LEGACY),
    "balanced": dict(phase_detail=fwm.PHASE_BALANCED, model_fidelity=fwm.FIDELITY_BALANCED),
    "fine":     dict(phase_detail=fwm.PHASE_FINE,     model_fidelity=fwm.FIDELITY_HIGH),
    "ultra":    dict(phase_detail=fwm.PHASE_ULTRA,    model_fidelity=fwm.FIDELITY_ULTRA),
}


def load_reference():
    """Sim et al. reference kwargs from the bit-locked regression anchor."""
    try:
        import test_regression as tr
        ref = dict(tr.CONFIGS["sim_optimum"])
        src = "tests/test_regression.py::CONFIGS['sim_optimum']"
        # sanity: the kwargs compute_spectrum needs must be present
        need = {"P_pump", "P_probe", "line_strength", "loss_frac"}
        if not need.issubset(ref):
            raise KeyError(f"anchor missing {need - set(ref)}")
        return ref, src
    except Exception as exc:  # pragma: no cover - drift signal
        raise SystemExit(
            "Could not read the Sim reference from "
            "tests/test_regression.py::CONFIGS['sim_optimum'].\n"
            f"  reason: {exc}\n"
            "The code has drifted past what this skill assumes. Open "
            "tests/test_regression.py, find the current Sim/optimum config (the "
            "one feeding compute_spectrum), and pass its P_pump/P_probe/"
            "line_strength via --p-pump/--p-probe/--line-strength.")


# Twin-beam validity gate on the deepest-δ argmin. `ideal_twin_beam_noise`
# S_ideal=(G_s−G_c)²/(G_s+G_c) is the EXACT bound only where the propagation is
# a genuine two-mode (Bogoliubov) parametric amplifier, G_s−G_c=1. Away from the
# actual FWM/Raman resonance feature — background absorption/dispersion, or a
# gain pole under Ultra's segmented dynamics — G_s and G_c decouple (gap far
# from 1), and (G_s−G_c)² can vanish with NO real parametric gain, making
# S_ideal spuriously ≈0 (perfect-looking "squeezing" that is not physical).
# Scanning the validated Sim reference point's full δ window (Δ=0.9 GHz,
# T=121 °C) empirically keeps gap in [0.545, 0.98] end to end (never below
# 0.5); a wide (Δ,T) exploratory grid does hit points with gap ≈0 or ≫1 (e.g.
# -0.04, 0.01, 40 observed off the reference regime). [GAP_MIN, GAP_MAX] keeps
# essentially all validated-regime points while excluding those artifacts.
GAP_MIN = 0.5
GAP_MAX = 1.5


def best_squeezing(D_GHz, T_K, ref, window, coarse, vstep, vcut, branch, fid_kw,
                   excess_noise=None):
    """Deepest squeezing over the δ scan on one branch, at the chosen fidelity.

    Returns the gains, the in-cell propagation OD and the δ at the deepest point.
    The argmin is restricted to δ points passing the [GAP_MIN, GAP_MAX] twin-beam
    validity gate (`gap_bad=True` if none in the window do, in which case the
    unrestricted argmin is reported anyway for diagnostics but the point should
    be excluded from frontier/optimum candidacy by the caller). The argmin is
    otherwise η-independent (one tau_cell per spectrum), so the captured
    (G_s, G_c, seg_od) reconstruct ξ for any η downstream.

    `excess_noise` (None → legacy gap-only S_dB; else the hardened model, Ultra
    only) makes `spec["S_dB"]` the hardened metric, so the argmin already lands on
    the hardened-deepest δ; the conjugate arm OD at that δ and the (scalar) pump
    scatter are captured so `squeezing_map` can reconstruct ξ(η) downstream.
    """
    center = fwm.branch_center_GHz(D_GHz, branch)
    with blas_single_thread():
        spec = fwm.compute_spectrum(
            D_GHz, T=T_K, P_pump=ref["P_pump"], P_probe=ref["P_probe"],
            line_strength=ref["line_strength"], loss_frac=ref["loss_frac"],
            qe=fwm.QE_DETECTOR, coarse_points=coarse, fine_points=0,
            scan_min=center - window, scan_max=center + window,
            velocity_step=vstep, velocity_cutoff=vcut, branch=branch,
            excess_noise_model=excess_noise, **fid_kw)
    Gs, Gc, S = spec["G_s"], spec["G_c"], spec["S_dB"]
    gap = Gs - Gc
    valid = np.isfinite(S) & (gap >= GAP_MIN) & (gap <= GAP_MAX)
    gap_bad = not bool(valid.any())
    S_search = S if gap_bad else np.where(valid, S, np.inf)
    i = int(np.nanargmin(S_search))                # deepest TRUSTWORTHY squeezing
    n = Gs.size
    hn = spec.get("hardened_noise") or {}
    od_conj_arr = hn.get("od_conj_arr")
    od_conj_i = float(od_conj_arr[i]) if od_conj_arr is not None else 0.0
    od_probe_arr = hn.get("od_probe_lin_arr")
    od_probe_i = float(od_probe_arr[i]) if od_probe_arr is not None else 0.0
    return dict(G_s=float(Gs[i]), G_c=float(Gc[i]), gap=float(gap[i]),
                seg_od=float(spec.get("segment_absorption_od", 0.0) or 0.0),
                od_conj=od_conj_i, od_probe=od_probe_i,
                pump_scatter=float(hn.get("pump_scatter_noise", 0.0)),
                delta_mhz=float((spec["probe_axis_GHz"][i] - center) * 1e3),
                at_edge=bool(i <= 1 or i >= n - 2), gap_bad=gap_bad)


def squeezing_map(Gs, Gc, OD, eta, od_conj=None, pump_scatter=None, od_probe=None):
    """ξ(Δ, T; η) including in-cell propagation loss.

    Delegates the ideal twin-beam noise to `observables.ideal_twin_beam_noise`
    (the conserved-(N_s−N_c) result S_ideal=(G_s−G_c)²/(G_s+G_c), which reduces
    to 1/(2G−1) in the lossless limit) so this script tracks the engine's
    squeezing formula rather than a private copy. In-cell transmission
    tau=e^(−OD) degrades S_ideal inside the cell, then detection η degrades it
    again. OD=0 (legacy) reduces this exactly to `intensity_difference_
    squeezing_dB`.

    When `od_conj` is supplied (hardened model) the probe/conjugate arms carry
    unequal in-cell transmission (probe e^(−OD), conjugate e^(−od_conj)) and are
    routed through the validated `balanced_twin_beam_noise`; `pump_scatter` is an
    η-independent additive excess added to the normalized noise. This reproduces
    `compute_spectrum`'s hardened S_dB at the same η, so ξ(η) stays remappable.
    """
    Gs = np.asarray(Gs, float); Gc = np.asarray(Gc, float); OD = np.asarray(OD, float)
    if od_conj is not None:
        od_conj = np.asarray(od_conj, float)
        probe_extra = (0.0 if od_probe is None
                       else np.maximum(np.asarray(od_probe, float), 0.0))
        eta_s = eta * np.exp(-(np.maximum(OD, 0.0) + probe_extra))
        eta_c = eta * np.exp(-np.maximum(od_conj, 0.0))
        # balanced_twin_beam_noise's scalar `tech` can't take a grid array, so add
        # the (per-point) pump-scatter excess to the returned noise ratio instead.
        S = observables.balanced_twin_beam_noise(
            Gs, Gc, eta_s, eta_c, reference_weight="dc")
        if pump_scatter is not None:
            S = S + np.maximum(np.asarray(pump_scatter, float), 0.0)
        return 10.0 * np.log10(np.maximum(S, 1e-30))
    S_ideal = observables.ideal_twin_beam_noise(Gs, Gc)
    tau = np.exp(-np.maximum(OD, 0.0))
    S_cell = tau * S_ideal + (1.0 - tau)
    S = eta * S_cell + (1.0 - eta)
    return 10.0 * np.log10(np.maximum(S, 1e-30))


def gain_knee(eta, eps):
    """Legacy analytic reference: G_s at which the lossless gap to the detection
    floor drops to `eps` dB. With the conserved-(N_s−N_c) ideal noise
    S_ideal=(G_s−G_c)²/(G_s+G_c) and the twin-beam relation G_c=G_s−1, this is
    exactly S_ideal = 1/(2 G_s − 1), so G_knee = (1/S_thr + 1)/2. Exact only at
    OD=0; under in-cell loss the frontier is determined empirically vs ξ_opt.
    At eta=1 (ideal/source-limit scan) there is no detection floor to reach, so
    no finite gain "clears the knee" -- returns +inf rather than an astronomical
    but meaningless number."""
    s_thr = (1.0 - eta) / max(eta, 1e-30) * (10 ** (eps / 10.0) - 1.0)
    if s_thr <= 0.0:
        return float("inf")
    return float(0.5 * (1.0 / s_thr + 1.0))


def frontier(Xi, Gs, Untrusted, T, target, eps, metric):
    """Lowest-T (or lowest-gain) point per Δ that reaches within eps of `target`
    (the best achievable ξ on the grid). `Untrusted` (δ-window-edge-limited OR
    twin-beam-gap-invalid, see GAP_MIN/GAP_MAX) points are excluded from
    candidacy (their gain/ξ is untrustworthy) but counted; T ascending so the
    first True per Δ is the min-T."""
    reached_all = Xi <= target + eps
    reached_ok = reached_all & (~Untrusted)
    nD = Xi.shape[0]
    Tmin_iT = np.full(nD, -1, dtype=int)
    for iD in range(nD):
        idx = np.where(reached_ok[iD, :])[0]
        if idx.size:
            Tmin_iT[iD] = int(idx[0])           # T ascending -> lowest T
    cands = [(T[jT], Gs[iD, jT], iD, jT)
             for iD, jT in enumerate(Tmin_iT) if jT >= 0]
    eff_iD = eff_jT = -1
    if cands:
        key = (lambda c: (c[0], c[1])) if metric == "temp" else (lambda c: (c[1], c[0]))
        _, _, eff_iD, eff_jT = min(cands, key=key)
    n_excluded = int(np.sum(reached_all & Untrusted))
    return Tmin_iT, eff_iD, eff_jT, n_excluded, bool(reached_all.any())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fidelity", choices=tuple(FIDELITY_KW), default="ultra",
                    help="engine fidelity (default: ultra = full propagation model)")
    ap.add_argument("--opd-min", type=float, default=-3.5)
    ap.add_argument("--opd-max", type=float, default=3.0)
    ap.add_argument("--opd-step", type=float, default=0.1)
    ap.add_argument("--t-min", type=float, default=60.0)
    ap.add_argument("--t-max", type=float, default=150.0)
    ap.add_argument("--t-step", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=0.1,
                    help="gap [dB] to the best achievable ξ that counts as "
                         "'on the frontier'. Headline 0.1 dB.")
    ap.add_argument("--metric", choices=("temp", "gain"), default="temp",
                    help="cost minimised on the frontier: lowest temperature "
                         "(default) or lowest gain")
    ap.add_argument("--pump-scatter-kappa", type=float, default=None,
                    help="override the Ultra hardened-noise pump-scatter coefficient "
                         f"kappa (default {fwm.HARDENED_PUMP_SCATTER_KAPPA}). The "
                         "hardened model (conjugate-arm linear OD + pump scatter) is "
                         "intrinsic to Ultra; this only tunes the pump-scatter term "
                         "and is ignored at other fidelities.")
    ap.add_argument("--loss", type=float, default=None,
                    help="override loss fraction (e.g. 0 for lossless/QE "
                         "detection); default = the Sim reference loss")
    ap.add_argument("--qe", type=float, default=None,
                    help="override detector quantum efficiency (e.g. 1.0 for an "
                         "ideal/source-limit scan with --loss 0); default = "
                         "fwm.QE_DETECTOR. G_s/G_c/OD never depend on qe/loss "
                         "(only the final eta-readout does), so this never "
                         "requires a re-solve of the physics.")
    ap.add_argument("--p-pump", type=float, default=None,
                    help="override pump power [W] (drift recovery)")
    ap.add_argument("--p-probe", type=float, default=None,
                    help="override seed/probe power [W] (drift recovery)")
    ap.add_argument("--line-strength", type=float, default=None,
                    help="override residual line_strength (drift recovery)")
    ap.add_argument("--coarse", type=int, default=81, help="δ scan points")
    ap.add_argument("--vstep", type=float, default=5.0, help="velocity step m/s")
    ap.add_argument("--vcut", type=float, default=3.0, help="velocity cutoff sigma")
    ap.add_argument("--window", type=float, default=0.7,
                    help="half-width of the δ scan window [GHz] (wide enough to "
                         "capture the low-gain dip on the frontier)")
    ap.add_argument("--branch", type=int, default=fwm.DEFAULT_BRANCH, choices=(-1, 1))
    ap.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--out", type=str, default=str(REPO / "analysis"))
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the matplotlib figure (still saves the .npz). Useful "
                         "for chunked/sub-grid runs whose figure will be discarded "
                         "and re-plotted after merging with other chunks.")
    args = ap.parse_args()

    fid_kw = FIDELITY_KW[args.fidelity]
    # The hardened excess-noise model is intrinsic to Ultra (see fwm.compute_spectrum);
    # nothing to switch on. A --pump-scatter-kappa override is passed through as a
    # tuning dict, otherwise the engine default is used.
    hardened = args.fidelity == "ultra"
    excess_noise = ({"pump_scatter_kappa": args.pump_scatter_kappa}
                    if args.pump_scatter_kappa is not None else None)
    ref, src = load_reference()
    overrides = {"loss_frac": args.loss, "P_pump": args.p_pump,
                 "P_probe": args.p_probe, "line_strength": args.line_strength}
    ref = dict(ref, **{k: v for k, v in overrides.items() if v is not None})
    qe_val = fwm.QE_DETECTOR if args.qe is None else float(args.qe)
    eta = qe_val * (1.0 - ref["loss_frac"])
    eta_qe = qe_val
    det_floor = 10.0 * np.log10(max(1.0 - eta, 1e-300))
    floor_qe = 10.0 * np.log10(max(1.0 - eta_qe, 1e-300))
    knee = gain_knee(eta, args.eps)

    D = np.round(np.arange(args.opd_min, args.opd_max + 1e-9, args.opd_step), 4)
    T = np.round(np.arange(args.t_min, args.t_max + 1e-9, args.t_step), 4)

    print(f"reference source : {src}")
    print(f"fixed (Sim) kwargs: P_pump={ref['P_pump']} W, P_probe={ref['P_probe']} W, "
          f"line_strength={ref['line_strength']}, loss_frac={ref['loss_frac']}")
    print(f"FIDELITY = {args.fidelity}  (kwargs {fid_kw})")
    if hardened:
        kap = (args.pump_scatter_kappa if args.pump_scatter_kappa is not None
               else fwm.HARDENED_PUMP_SCATTER_KAPPA)
        print(f"  Ultra hardened noise ACTIVE (conjugate-arm linear OD + pump "
              f"scatter, kappa={kap})")
    print(f"eta = QE*(1-loss) = {qe_val}*(1-{ref['loss_frac']}) = {eta:.5f}")
    print(f"detection floor 10log10(1-eta) = {det_floor:.4f} dB  (hard guardrail)")
    print(f"frontier: 'on frontier' = within eps={args.eps} dB of the BEST "
          f"achievable xi on the grid; cost metric = {args.metric}")
    if args.fidelity != "legacy":
        print("  NOTE: at this fidelity in-cell loss raises the achievable floor "
              "above the detection floor -> xi has a true minimum (optimum T).")
    print(f"grid: {D.size} OPD [{D[0]}..{D[-1]} GHz] x {T.size} T "
          f"[{T[0]}..{T[-1]} C] = {D.size*T.size} pts, "
          f"branch={args.branch}, coarse={args.coarse}, vstep={args.vstep}, "
          f"vcut={args.vcut}, window=+-{args.window} GHz\n")

    nD, nT = D.size, T.size
    Gs = np.full((nD, nT), np.nan)
    Gc = np.full((nD, nT), np.nan)
    Gap = np.full((nD, nT), np.nan)
    OD = np.zeros((nD, nT))
    Dlt = np.full((nD, nT), np.nan)
    Edge = np.zeros((nD, nT), dtype=bool)
    GapBad = np.zeros((nD, nT), dtype=bool)
    OdConj = np.zeros((nD, nT))
    OdProbe = np.zeros((nD, nT))
    PumpScatter = np.zeros((nD, nT))
    jobs = [(iD, iT, float(d), float(t) + 273.15)
            for iD, d in enumerate(D) for iT, t in enumerate(T)]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(best_squeezing, d, t, ref, args.window, args.coarse,
                          args.vstep, args.vcut, args.branch, fid_kw,
                          excess_noise): (iD, iT)
                for (iD, iT, d, t) in jobs}
        done = 0
        for fut in as_completed(futs):
            iD, iT = futs[fut]
            r = fut.result()
            Gs[iD, iT], Gc[iD, iT] = r["G_s"], r["G_c"]
            Gap[iD, iT] = r["gap"]
            OD[iD, iT] = r["seg_od"]
            OdConj[iD, iT] = r.get("od_conj", 0.0)
            OdProbe[iD, iT] = r.get("od_probe", 0.0)
            PumpScatter[iD, iT] = r.get("pump_scatter", 0.0)
            Dlt[iD, iT], Edge[iD, iT] = r["delta_mhz"], r["at_edge"]
            GapBad[iD, iT] = r["gap_bad"]
            done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                eta_s = el / done * (len(jobs) - done)
                print(f"  {done}/{len(jobs)}  ({el:.0f}s, ~{eta_s:.0f}s left)",
                      flush=True)

    od_conj_arg = OdConj if hardened else None
    od_probe_arg = OdProbe if hardened else None
    pscat_arg = PumpScatter if hardened else None
    Xi = squeezing_map(Gs, Gc, OD, eta, od_conj=od_conj_arg,
                       pump_scatter=pscat_arg, od_probe=od_probe_arg)
    Xi_qe = squeezing_map(Gs, Gc, OD, eta_qe, od_conj=od_conj_arg,
                          pump_scatter=pscat_arg, od_probe=od_probe_arg)
    n_below = int(np.sum(Xi[np.isfinite(Xi)] < det_floor - 1e-6))
    Untrusted = Edge | GapBad

    # Best achievable squeezing on the grid = the generalized "floor" target.
    # Excludes GapBad points: with no valid-gap delta in the window, best_squeezing
    # falls back to the raw (untrustworthy) argmin, which must not anchor xi_opt.
    Xi_trusted = np.where(Untrusted, np.nan, Xi)
    if not np.any(np.isfinite(Xi_trusted)):
        raise SystemExit(
            "No grid point passed the twin-beam validity gate "
            f"(GAP_MIN={GAP_MIN}, GAP_MAX={GAP_MAX}) or the delta-edge check -- "
            "widen --window / the (Delta, T) range and rerun.")
    oFlat = int(np.nanargmin(Xi_trusted))
    oiD, oiT = np.unravel_index(oFlat, Xi.shape)
    xi_opt = float(Xi[oiD, oiT])
    target = xi_opt
    penalty = xi_opt - det_floor                 # propagation/in-cell-loss penalty
    opt_at_edge = bool(oiD in (0, nD - 1) or oiT in (0, nT - 1))
    n_at_opt = int(np.sum(Xi_trusted < target + 0.01))
    n_gap_bad = int(np.sum(GapBad))

    Tmin_iT, eiD, ejT, n_untrusted_excl, reached_any = frontier(
        Xi, Gs, Untrusted, T, target, args.eps, args.metric)

    print("\n================  BEST ACHIEVABLE (optimum)  ================")
    print(f"deepest xi on the grid = {xi_opt:.4f} dB @ Delta={D[oiD]:+.2f} GHz, "
          f"T={T[oiT]:.0f} C  (G_s={Gs[oiD,oiT]:.3g}, in-cell OD={OD[oiD,oiT]:.3f}, "
          f"in-cell loss={1-np.exp(-OD[oiD,oiT]):.1%})")
    print(f"detection floor = {det_floor:.4f} dB; propagation penalty "
          f"(xi_opt - floor) = +{penalty:.4f} dB"
          + ("  [legacy: ~0 expected]" if args.fidelity == "legacy" else
             "  [in-cell loss cost at the optimum]"))
    if opt_at_edge:
        print("  WARNING: optimum sits on a grid boundary -- widen --t-max/"
              "--t-min/--opd-* to confirm it is the true (interior) optimum.")

    print("\n================  RESULT (efficient frontier)  ================")
    if eiD >= 0:
        gs_e, gc_e = Gs[eiD, ejT], Gc[eiD, ejT]
        gap = Xi[eiD, ejT] - target
        print(f"most efficient operating point (lowest {args.metric} within "
              f"{args.eps} dB of the optimum):")
        print(f"  Delta = {D[eiD]:+.2f} GHz, T = {T[ejT]:.0f} C, "
              f"delta = {Dlt[eiD,ejT]:.1f} MHz")
        print(f"  G_s = {gs_e:.3g}, G_c = {gc_e:.3g}, in-cell OD = {OD[eiD,ejT]:.3f} "
              f"(loss {1-np.exp(-OD[eiD,ejT]):.1%}); legacy knee ref ~= {knee:.1f}")
        print(f"  xi @ eta={eta:.4f} = {Xi[eiD,ejT]:.4f} dB  "
              f"(optimum {target:.4f}, gap +{gap:.4f}; detection floor {det_floor:.4f})")
        print(f"  xi @ eta=QE={eta_qe:.4f} (same point) = {Xi_qe[eiD,ejT]:.4f} dB  "
              f"(QE detection floor {floor_qe:.4f})")
    else:
        print("  no trustworthy operating point within eps of the optimum "
              "(see VERDICT).")

    # ---- efficient frontier table: lowest-T-to-target vs Delta ----------------
    print("\n----------------  EFFICIENT FRONTIER  ----------------")
    print("  Delta[GHz]   Tmin[C]      G_s     in-cellOD")
    valid_iD = [iD for iD in range(nD) if Tmin_iT[iD] >= 0]
    if valid_iD:
        stride = max(1, len(valid_iD) // 14)
        show = set(valid_iD[::stride]) | {eiD}
        for iD in valid_iD:
            if iD not in show:
                continue
            jT = Tmin_iT[iD]
            mark = " *" if iD == eiD else "  "
            print(f"  {D[iD]:+7.2f}    {T[jT]:7.0f}   {Gs[iD,jT]:9.1f}   "
                  f"{OD[iD,jT]:7.3f}{mark}")
        print("  (* = most efficient; rows subsampled)")
    else:
        print("  (frontier empty -- no trustworthy point within eps of optimum)")

    # ---- eps sensitivity ------------------------------------------------------
    print("\n----------------  eps SENSITIVITY  ----------------")
    print("  eps[dB]   Delta*[GHz]   T*[C]   G_s*    in-cellOD")
    for e in (0.05, 0.1, 0.2):
        _, jiD, jjT, _, _ = frontier(Xi, Gs, Untrusted, T, target, e, args.metric)
        if jiD >= 0:
            print(f"  {e:5.2f}     {D[jiD]:+8.2f}    {T[jjT]:6.0f}   "
                  f"{Gs[jiD,jjT]:7.1f}   {OD[jiD,jjT]:7.3f}")
        else:
            print(f"  {e:5.2f}     (none within eps)")

    # ---- theory-check guardrail ----------------------------------------------
    print("\n----------------  THEORY CHECK (guardrail)  ----------------")
    print(f"points below the detection floor (must be 0): {n_below}")
    print(f"points within 0.01 dB of the optimum: {n_at_opt}/{Xi.size}")
    print(f"points failing the twin-beam gap gate (G_s-G_c outside "
          f"[{GAP_MIN},{GAP_MAX}] everywhere in the delta window): {n_gap_bad} "
          f"(excluded from optimum + frontier)")
    print(f"target-reaching points excluded as delta-edge/gap-untrustworthy: "
          f"{n_untrusted_excl}")

    # ---- verdict --------------------------------------------------------------
    if n_below > 0:
        verdict = ("FAIL -- points fall BELOW the detection floor, which is "
                   "physically impossible (S = eta*S_cell+(1-eta) >= 1-eta). This "
                   "indicates a coding/numerical error; inspect those points.")
    elif eiD < 0:
        verdict = ("INCOMPLETE -- the optimum is reached only at delta-window-edge-"
                   "limited or twin-beam-gap-invalid points (untrustworthy gains). "
                   "Widen --window and rerun.")
    elif opt_at_edge:
        verdict = (f"CHECK -- frontier found (Delta={D[eiD]:+.2f} GHz, T={T[ejT]:.0f} "
                   f"C) but the GRID OPTIMUM is on a boundary, so it may not be the "
                   f"true optimum. Widen --t-max/--t-min/--opd-* and rerun to "
                   f"confirm, then trust the frontier.")
    elif args.fidelity == "legacy" and penalty > 0.5:
        verdict = (f"INCOMPLETE (legacy) -- best xi {xi_opt:.2f} dB is {penalty:.2f} "
                   f"dB short of the detection floor {det_floor:.2f}: the gain never "
                   f"cleared the knee (G_s~={knee:.0f}) on this grid. Raise --t-max "
                   f"or widen the OPD range.")
    else:
        verdict = (f"EFFICIENT -- best achievable xi = {xi_opt:.2f} dB "
                   f"(+{penalty:.2f} dB above the detection floor from in-cell loss) "
                   f"is reached at minimum {args.metric} at Delta={D[eiD]:+.2f} GHz, "
                   f"T={T[ejT]:.0f} C (G_s~={Gs[eiD,ejT]:.0f}, OD~={OD[eiD,ejT]:.2f}); "
                   f"no point beats the detection floor. This is the cheapest near-"
                   f"optimal operating point -- low T/gain/OD also minimises the "
                   f"excess-noise channels that limit real experiments.")
    print("\nVERDICT: " + verdict)
    sys.stdout.flush()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "squeezing_frontier.npz", delta_ghz=D, temp_c=T, xi_dB=Xi,
             xi_qe_dB=Xi_qe, G_s=Gs, G_c=Gc, gap=Gap, in_cell_od=OD, delta_mhz=Dlt,
             edge=Edge, gap_bad=GapBad, tmin_idx=Tmin_iT, eta=eta, eta_qe=eta_qe,
             det_floor_dB=det_floor, floor_qe_dB=floor_qe, xi_opt_dB=xi_opt,
             opt_iD=oiD, opt_iT=oiT, eps=args.eps, eff_iD=eiD, eff_jT=ejT,
             fidelity=args.fidelity, knee=knee,
             noise_model=("hardened" if hardened else "off"),
             od_conj=OdConj, od_probe=OdProbe, pump_scatter=PumpScatter)
    if args.no_plot:
        print(f"\nsaved: {out/'squeezing_frontier.npz'}  (plot skipped, --no-plot; "
              f"total {time.time()-t0:.0f}s)", flush=True)
    else:
        _plot(D, T, Xi, Gs, OD, Tmin_iT, eiD, ejT, oiD, oiT, det_floor, xi_opt,
              args.metric, args.fidelity, out)
        print(f"\nsaved: {out/'squeezing_frontier.npz'} and squeezing_frontier.png  "
              f"(total {time.time()-t0:.0f}s)", flush=True)


def _plot(D, T, Xi, Gs, OD, Tmin_iT, eiD, ejT, oiD, oiT, det_floor, xi_opt,
          metric, fidelity, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Dfront = np.array([D[iD] for iD in range(D.size) if Tmin_iT[iD] >= 0])
    Tfront = np.array([T[Tmin_iT[iD]] for iD in range(D.size) if Tmin_iT[iD] >= 0])
    Gfront = np.array([Gs[iD, Tmin_iT[iD]] for iD in range(D.size) if Tmin_iT[iD] >= 0])

    fig, (axM, axF, axT) = plt.subplots(1, 3, figsize=(17, 4.8))
    extent = [T[0], T[-1], D[0], D[-1]]

    # (1) squeezing heatmap with the frontier curve + optimum overlaid
    im = axM.imshow(Xi, origin="lower", aspect="auto", extent=extent, cmap="viridis_r")
    fig.colorbar(im, ax=axM, label="xi [dB]")
    if Tfront.size:
        axM.plot(Tfront, Dfront, "-", color="white", lw=1.4, alpha=0.9,
                 label="efficient frontier T_min(Delta)")
    axM.scatter([T[oiT]], [D[oiD]], color="gold", marker="o", s=90, edgecolor="k",
                zorder=6, label=f"optimum ({xi_opt:.2f} dB)")
    if eiD >= 0:
        axM.scatter([T[ejT]], [D[eiD]], color="crimson", marker="*", s=180,
                    edgecolor="k", zorder=7,
                    label=f"most efficient ({D[eiD]:+.1f} GHz, {T[ejT]:.0f}C)")
    axM.set_xlabel("T [C]"); axM.set_ylabel("Delta [GHz]")
    axM.set_title(f"xi(Delta, T) -- fidelity={fidelity}")
    axM.legend(loc="upper right", fontsize=7)

    # (2) cost-vs-Delta: lowest target-reaching T and the gain there (twin axes)
    if Dfront.size:
        axF.plot(Dfront, Tfront, "o-", color="#1f77b4", ms=3, lw=1.4)
        axF.set_xlabel("Delta [GHz]")
        axF.set_ylabel("frontier T_min [C]", color="#1f77b4")
        axF.tick_params(axis="y", labelcolor="#1f77b4")
        axG = axF.twinx()
        axG.plot(Dfront, Gfront, "s--", color="#d62728", ms=3, lw=1.0, alpha=0.8)
        axG.set_ylabel("G_s at frontier", color="#d62728")
        axG.tick_params(axis="y", labelcolor="#d62728")
        if eiD >= 0:
            axF.axvline(D[eiD], color="crimson", ls=":", lw=1.2)
    axF.set_title(f"cost vs Delta (min-{metric} frontier)")
    axF.grid(alpha=0.3)

    # (3) xi-vs-T slice at the optimum Delta (shows the true minimum vs floor)
    sl = oiD if eiD < 0 else eiD
    axT.plot(T, Xi[sl, :], "o-", color="#2ca02c", lw=1.6, ms=3)
    axT.scatter([T[oiT]], [xi_opt], color="gold", marker="o", s=80, edgecolor="k",
                zorder=6, label=f"optimum {xi_opt:.2f} dB")
    axT.axhline(det_floor, color="crimson", ls="--",
                label=f"detection floor {det_floor:.2f} dB")
    axT.set_title(f"xi vs T at Delta = {D[sl]:+.2f} GHz")
    axT.set_xlabel("T [C]"); axT.set_ylabel("xi [dB]")
    axT.grid(alpha=0.3); axT.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "squeezing_frontier.png", dpi=130)


if __name__ == "__main__":
    main()
