# <Scheme> Fast tier: acceptance gate (vN)

Frozen: <date>, before implementation. To change a criterion, append the date
and reason below and keep the verdict against the previous criterion. Never
relax a criterion to reach PASS.

## Common conditions

- **Reference**: <tier name and its full model: measure, cutoff, order, axis,
  constants, calibration and coordinates>.
- **Entry point**: the app's own `<Scheme>.compute(params)`.
- **Case matrix (N)**: <name: overrides on top of app defaults>, ... In repair
  mode include the reported failing case.
- **Timing**: thread counts <...>, warm, median of 5, nothing else running;
  first call measured separately in a new process with a warm Numba cache.
- **Display window**: <axis range and point count, identical to the reference>.
- **Tube error**: minimum vertical error within ±<h> of horizontal allowance.

## G1 Speed

| ID | Criterion | Rationale |
|---|---|---|
| G1.1 | default ≤ <t> s; every case ≤ <t> s | |
| G1.2 | speed-up vs frozen "before" ≥ <x>× | |
| G1.3 | first call in a new process ≤ <t> s | |

## G2 Agreement with the reference (every case, every displayed point)

| Quantity | Norm | Threshold | Rationale |
|---|---|---|---|
| <curve 1> | tube max / p95 | | |
| <hero readout> | absolute or relative | | |
| <readout location> | absolute | | |
| <validity / availability state> | exact match outside band <...> | | |
| <downstream consumer, e.g. SABES> | | | |

## G3 Integrity

| ID | Criterion |
|---|---|
| G3.1 | display axis `array_equal` to the reference |
| G3.2 | all outputs finite |
| G3.3 | guard: every solved row vs the existing kernel, relative ≤ <tol> |
| G3.4 | reference tier outputs bit-identical before/after |
| G3.5 | `GABES_DISABLE_NUMBA=1` parity within <tol> |
| G3.6 | provenance records method, solved nodes, interpolation, guard, audit range |
| G3.7 | compatibility baselines unchanged (or a recorded value decision) |

## G4 Tests

| ID | Criterion |
|---|---|
| G4.1 | focused tests of the scheme and its callers pass |
| G4.2 | new contract tests pass with Numba on and off |

## G5 Records

| ID | Criterion |
|---|---|
| G5.1 | `DEVLOG.md`: attempts, rejections, numbers, reuse notes |
| G5.2 | `gate.py` writes a JSON report and exits 0 only when every item passes |

## Amendments

<date>: <criterion>, <old → new>, <reason>, <verdict under the old criterion>
