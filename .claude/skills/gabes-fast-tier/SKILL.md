---
name: gabes-fast-tier
description: Build or repair a Fast fidelity tier for any scheme in THIS GABES repository, so that it reproduces the scheme's non-Fast (reference) computation closely enough to be useful. Use it when adding a Fast or Balanced tier to a scheme, speeding up a scheme without changing its model, or when a Fast result disagrees with the reference (Fast 오차, Fast 충실도 수리, 경계 사례 dB 차이, 보간 오차, Fast vs Ultra/dense 불일치). FWM seeded Fast is the reference implementation and the first repair case. Not for choosing physics, constants or calibration values; those are user decisions.
---

# GABES Fast tier

A Fast tier is a cheaper computation of **the same model** that the scheme's
reference tier computes. It is never a different model. This skill gives the
procedure; you set the acceptance criteria yourself, before you start, and
freeze them before you collect evidence (user decision 2026-11-16).

Use the same procedure in two modes:

- **Build**: a scheme has no Fast tier, or its Fast tier is a coarser grid.
- **Repair**: someone found a case where Fast and the reference disagree.

## Read first

1. `references/techniques.md`: the acceleration techniques used in FWM Fast,
   with measured gains, applicability tests, traps and rejected attempts.
2. `references/gate_template.md`: the gate skeleton you fill in and freeze.
3. The scheme's own code and its tier table (for FWM:
   `gabes/schemes/fwm/common.py` `FWM_FIDELITY`, `gabes/pole_doppler.py`,
   `gabes/adaptive_scan.py`; contracts in `tests/test_fwm_fast_tiers.py`,
   `tests/test_pole_doppler.py`, `tests/test_fwm_fast_source_independence.py`).
4. The worked record in `GABES-archive`: `analysis/fwm_lite/GATE.md`,
   `DEVLOG.md` and `gate.py` (FWM remaster, 2026-09-11/14), and
   `analysis/pole_reuse/` (scanners and magneto, 2026-09-14).
5. The checklist item that asked for the work (`docs/checklist.json`).

## Procedure

### 1. Pin the model the Fast tier must reproduce

Write down the reference tier's complete model definition: velocity measure
and cutoff, Floquet or basis order, grid and display axis, constants,
calibration and its coordinates, readout formulas, and validity/availability
rules. A Fast tier that sums a different measure or skips a calibration step
computes a different model, even when its answer is "more accurate".

FWM lesson (DEVLOG 1.5): an untruncated Maxwell mean disagreed with Ultra's
4σ-cut grid by 0.112 dB at 150 °C. The fix was to sum Ultra's discrete
measure, not to relax the gate. If you think the reference model itself is
wrong, record that as a separate checklist item; do not fix it inside the Fast
work.

### 2. Freeze the gate before implementation

Copy `references/gate_template.md` into the run folder as `GATE.md` and fill
it in:

- **Case matrix.** Defaults; every shipped preset; both ends of each
  `recompute=True` knob; hot/cold, strong/weak drive; the narrowest feature the
  display can show; near-resonance and scan-edge cases; model boundaries
  (feasible/unavailable, caps, floors, withheld readouts); calibration on and
  off, inside and outside its measured domain. In repair mode, add the
  reported failing case, which then stays in the matrix for good.
- **Error norms per displayed quantity.** Curves (tube error with a small
  horizontal allowance on steep flanks, p95), hero readouts and their
  location, validity/availability states (exact agreement outside a declared
  tolerance band), and anything a downstream consumer reads (SABES, exports).
  Compare at every displayed point, not only at solved nodes.
- **Thresholds.** Choose them from what the user can see and act on: the
  display resolution, the readout's printed precision, and the scheme's known
  physical uncertainties. Write one line of rationale per threshold.
  Starting points, not mandates: the FWM gate in the archive and the FWM
  `validation_contract` in `fwm-fast-displayed-readout-error-certificate`.
- **Speed budget** for warm default, worst case, and first call in a new
  process with a warm Numba disk cache.
- **Integrity.** Same display axis; all outputs finite; guard of every solved
  row against the existing kernel; Numba on/off parity
  (`GABES_DISABLE_NUMBA=1`); the reference tier's outputs bit-identical
  before and after; the FWM compatibility baseline unchanged unless a value
  decision says otherwise.

After freezing, never relax a criterion to reach PASS. To change one, append
the date and reason to `GATE.md` and keep the verdict against the old
criterion too.

### 3. Capture "before"

- Reference outputs for every case through the same entry point the app uses
  (`scheme.compute(params)`), saved with
  `gabes_lab.blocks.snapshot.capture` / `save`.
- Current Fast outputs and timings: warm, median of 5, fixed thread counts,
  nothing else running. First JIT compile can take minutes; warm the cache
  before timing.
- In repair mode, reproduce the reported failure first with an independent
  dense evaluation at the same coordinates. A report is not evidence until it
  reproduces.

### 4. Diagnose before optimizing

Profile (`cProfile`) and attribute time before changing code. Structural cuts
(fewer solves, fewer grid points, symmetry) beat kernel tuning: in FWM a
kernel rewrite gave 1.36× while replacing ~1,600 solves per detuning with one
eigendecomposition gave the speed-up.

In repair mode, classify the error before fixing it:

| Class | Symptom | Usual fix |
|---|---|---|
| Model mismatch | Error at solved nodes too | Match the reference measure, order, constants (step 1) |
| Interpolation | Solved nodes exact, points between them wrong | Tighter refinement, or solve every point in that window |
| Off-grid/source coordinates | Error only when inputs fall between cached coordinates | Evaluate at exact source coordinates, or refine the cache |
| Eligibility/state | Available vs unavailable, converged vs not, differs | Judge validity from solved values, never from interpolated ones |
| Calibration coupling | Calibrated readout wrong, raw readout right | Refit the calibration against the reference model, then re-gate |
| Guard miss | Row passes guard, fails dense check | Widen the guard set or tighten its tolerance |

### 5. Implement from the technique catalogue

Pick techniques from `references/techniques.md` whose applicability test the
scheme passes. Apply one at a time, re-run the gate after each, and log the
result, including what you rejected and why.

Engine rules still hold (AGENTS "Unified engine"): put shared numerics in
`gabes/` modules that every scheme can call (`pole_doppler`,
`adaptive_scan`, `affine_doppler`, `kernels`), extend them with a parameter
instead of copying them, and keep `gabes` and `gabes_ui` free of
`gabes_lab` imports.

### 6. Handle what cannot pass cheaply

Where a region cannot meet the frozen criteria at Fast cost, choose one and
state it in the result's provenance:

1. solve every point in that region (FWM full-scan view: all points solved,
   because adaptive refinement was verified only down to the 2.75 MHz display
   spacing);
2. fall back to the reference evaluation for that region;
3. mark the region "uncertified" visibly in the readout.

Never drop a failing region silently, and never disable a requested empirical
correction because its domain is uncertified (AGENTS practical FWM rule).

### 7. Gate, iterate, land

- Write `gate.py` that runs the frozen matrix and writes a JSON report; it
  exits 0 only when every criterion passes. Run it through
  `gabes_lab.blocks.runner.start_run("fast_tier/<scheme>", ...)` so outputs go
  to `runs/`.
- Iterate steps 4–6 until the gate passes. Re-run the whole matrix after
  each change, not only the failing case.
- Land product code in `gabes/` with contract tests in `tests/`: method and
  model passed through, guard equals grid on representative rows, node rules,
  Numba on/off parity. Record the method, solved-node count, interpolation,
  guard and audit range in the result's provenance, and the interpolation
  reason in the claim gate.
- A Fast tier that matches its reference does not change values and lands as
  a refactor. If it changes any displayed value beyond the frozen tolerance,
  that is a value change: it needs a user decision in `docs/checklist.json`
  and its own commit.
- Publish `GATE.md`, `DEVLOG.md`, `gate.py` and the gate report to the archive
  with `gabes_lab.blocks.publish_to_archive` (destination
  `analysis/fast_tier/<scheme>/`). Update the checklist item with commits,
  gate result and what stays uncertified.
- Validation scope follows AGENTS "Change-scoped validation": the gate, the
  scheme's focused tests and its callers (SABES for FWM). A full suite is for
  shared-engine changes.

## Calibration rule

Fit an empirical calibration against the reference model, never against Fast
output. Fast then has to reproduce the calibrated reference like any other
readout. FWM lesson (2026-11): the lab gain fit was trained on cached Fast
source coordinates; at exact coordinates its training-point error rose from
4.9 %/3.9 % to 26.2 %/17.7 %, so the calibration had absorbed Fast artifacts.

## Report

Report to the user in Korean (AGENTS conventions): frozen thresholds with
their rationale, before/after speed, worst error per quantity and case, what
fell back or stays uncertified, commits, and validation scope.
