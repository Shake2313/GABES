---
name: fwm-squeezing-frontier
description: >
  Find the EFFICIENT FRONTIER of FWM twin-beam intensity-difference squeezing in
  THIS GABES repo: the lowest-cost operating point (lowest cell temperature /
  lowest gain) over the one-photon-detuning (Δ/OPD) and temperature (T) plane
  that reaches the best achievable squeezing (the η-limited floor 10·log10(1−η) in
  the lossless limit; raised by in-cell loss at the default full/Ultra fidelity). Use this
  whenever the user wants to scan/optimize FWM squeezing vs detuning and
  temperature, find the cheapest or most robust squeezing operating point, locate
  the efficient frontier / floor-reaching minimum temperature, reproduce the
  ξ(Δ,T) squeezing study, or check whether a computed squeezing value agrees with
  theory or is a coding/numerical error (the η-floor guardrail is still applied).
  Triggers on "최효율점", "efficient frontier", "floor 도달 최저온도/최저이득",
  "squeezing 동작점 최적화", "squeezing 최대치", "ξ(Δ,T)", "OPD/온도 스캔",
  "is this squeezing physical", and similar — even if the user does not name this
  skill. Narrow on purpose: 85Rb D1 double-Λ FWM squeezing only.
---

# FWM squeezing efficient frontier ξ(Δ, T)

## What this does
Scans one-photon detuning Δ (OPD) and temperature T, and for each (Δ, T) takes
the deepest intensity-difference squeezing on a **single Raman branch** (the δ /
two-photon-detuning scan already inside `compute_spectrum`). It then finds the
**efficient frontier**: for each Δ, the *lowest temperature* (equivalently lowest
gain) at which ξ comes within `eps` dB of the **best achievable** squeezing on the
grid `ξ_opt = min ξ`, and reports the single cheapest such operating point. The
detection floor `10·log10(1−η)` is kept as a hard **guardrail** (no point may fall
below `1−η`, at any fidelity).

It runs at **full Ultra fidelity by default** (`--fidelity ultra`): self-consistent
Δk_z phase mismatch, 64-segment propagation with dynamic per-segment pump
depletion, in-cell propagation loss + technical-noise channels, Gaussian
beam-overlap, and the 24-level Zeeman CG correction. The target is the *achieved*
optimum rather than the bare detection floor because in-cell loss makes the floor
unreachable and gives ξ(T) a **true minimum** (see "Fidelity"). `--fidelity legacy`
runs the fast ideal model, for which `ξ_opt` collapses back onto `10·log10(1−η)`.

Everything except Δ and T is held at the Sim et al. reference operating point
(G. Sim, H. Kim, H. S. Moon, *Sci. Rep.* 15, 7727 (2025)).

## Why the frontier, not the maximum
- **Legacy (ideal):** the global maximum of ξ is **meaningless** — it is always the
  detection floor `10·log10(1−η)`, and the entire high-gain region maps onto it as a
  broad degenerate **plateau**, so "the maximum" is an arbitrary plateau-edge
  coordinate (see `analysis/squeezing_report.tex`).
- **Ultra (full model):** in-cell propagation loss raises the achievable floor to
  `~1−η·e^(−OD)`, which grows with T (OD). So there is no plateau — ξ(T) has a
  **genuine minimum** (low T is gain-limited, high T is loss-limited), and `ξ_opt`
  is a real optimum operating point.

Either way the physically meaningful singular point is the **efficient frontier** —
the cheapest (Δ, T) that comes within `eps` of `ξ_opt`. That point is (a)
experimentally cheapest (lowest cell temperature, lowest gain, lowest OD) and (b)
**most robust to the excess-noise channels that actually limit experiments**:
anti-squeezed sum noise ∝ G, spontaneous Raman ∝ pump, collisional dephasing ∝
density — all minimised at low T / low gain. So the frontier is not just "cheap",
it is the recommended operating regime.

## Run it
The work is done by the bundled script. From the repo root:

```bash
python .claude/skills/fwm-squeezing-frontier/scripts/scan_squeezing_frontier.py
```

Default grid (Δ −3.5…3.0 GHz, T 60…150 °C, coarse=81, vstep=5, window ±0.7 GHz)
at **Ultra** fidelity finishes in a few minutes on 8 cores (numba-accelerated; the
per-point 64-segment / 3-iteration cost is modest). The wide OPD range is
**required** — the frontier minimum sits at negative Δ, outside the old [−0.5, 3.0]
window. Useful flags:

- `--fidelity legacy|balanced|fine|ultra` — engine fidelity (default **ultra** =
  full propagation model). `legacy` is the fast ideal model for a quick map.
- `--eps E` — gap [dB] to `ξ_opt` that counts as "on the frontier". Headline `0.1`;
  the script also reports a 0.05/0.1/0.2 sensitivity sweep.
- `--metric temp|gain` — cost minimised on the frontier (default `temp`).
- `--loss 0` — lossless detection (η = QE); moves the detection floor to −10.2 dB
  (in-cell loss still applies under Ultra).
- `--opd-min/--opd-max/--opd-step`, `--t-min/--t-max/--t-step` — widen/refine
  (e.g. `--t-step 2.5` to pin the min-T more finely). Widen if the VERDICT is
  `CHECK` (grid optimum on a boundary).
- `--coarse N --vstep V --vcut S` — higher δ / velocity-class precision.
- `--window G` — half-width of the δ scan window in GHz (default 0.7; raise it if
  the verdict says the optimum was reached only at δ-edge-limited points).
- `--out DIR` sets the output folder. The default is a new folder at
  `runs/squeezing/frontier/<UTC>-<commit7>/`. It contains
  `squeezing_frontier.npz`, `squeezing_frontier.png` (unless `--no-plot`),
  and `run.json` with the configuration, status, and output hashes.

Relay the script's `BEST ACHIEVABLE` + `RESULT` + `EFFICIENT FRONTIER` + `VERDICT`
block to the user, and surface the saved `squeezing_frontier.png` (heatmap +
frontier overlay + optimum, the cost-vs-Δ curve, and the ξ-vs-T slice showing the
true minimum vs the detection floor).

## Fidelity — defaults to Ultra (the full model)
`--fidelity` maps onto `compute_spectrum`'s `phase_detail` / `model_fidelity`. The
default is **ultra** — the complete propagation physics this engine can model:

| Ultra element | Effect on the frontier |
|---|---|
| **In-cell propagation loss** `1−e^(−OD)` (segmented absorption) | **Dominant.** Raises the achievable floor from `1−η` to `~1−η·e^(−OD)` (T/Δ-dependent) → the detection floor is no longer reachable and ξ(T) gains a true minimum. |
| Self-consistent Δk_z phase mismatch (3 iters) | Reshapes gain vs Δ → shifts Δ*. |
| 64-segment propagation + dynamic per-segment pump depletion | More physical gain than the bulk Manley-Rowe cap. |
| Gaussian beam-overlap profile | Reduces effective gain (≈1.0 at the reference geometry). |
| Zeeman `lumped_strength_correction` (24-level CG diagnostic) | Rescales coupling (≈1.0 at the reference point). |
| Technical-noise channels (seed-excess / pump-scatter / EOM) | Plumbed but `0.0` in the current Ultra call. |
| **Density/T-dependent collisional decoherence** (`fwm.collisional_atom`, applied inside `compute_spectrum`'s χ̄ solve at **every** fidelity, not just Ultra) | Rb self-broadening of the optical coherence + ground-Raman collisions grow with T; best achievable squeezing now peaks near 121-131 °C and **worsens** at higher T instead of improving monotonically with density — an additional, physical high-T penalty on top of the in-cell-loss one. |

**Why it changes the answer** (legacy vs ultra at one point, Δ −1.1 GHz, T 95 °C):
legacy ξ = **−7.16 dB** but ultra ξ = **−4.28 dB** — a ≈2.9 dB gap, dominated by
in-cell loss (OD ≈ 0.30 → ≈26 % loss). So the ultra frontier is the *real*
operating point, while a legacy scan understates the loss penalty. The
detection-floor guardrail still holds (ultra `S` only ever sits **above** `1−η`).

**Solve-once is preserved.** The deepest-δ argmin is η-independent (in-cell loss
`tau = e^(−OD)` is one scalar per (Δ, T) spectrum, so `S` is monotone in `S_ideal`
along δ). The script captures `(G_s, G_c, OD)` at that δ and reconstructs ξ for any
η from the segmented-loss formula — so `--loss`/`--eps` variations need no re-solve.

Use `--fidelity legacy` for a fast ideal map (e.g. to locate the gain knee or sanity
-check the guardrail); the deepest-fidelity research answer is the default ultra run.

## The physics you are verifying (so you can explain the result)
Lossless twin-beam squeezing is `S(η) = η·S_ideal + (1−η)`, `ξ = 10·log10(S)`,
with the conserved-(N_s−N_c) ideal noise `S_ideal = (G_s−G_c)²/(G_s+G_c)`
(`observables.ideal_twin_beam_noise`) — `S_ideal ∈ [0,1]` → **ξ ≤ 0 dB always**,
deepest at the detection floor `10·log10(1−η)`. With the twin-beam relation
`G_c = G_s−1`, this is *exactly* `S_ideal = 1/(2 G_s − 1)` (Sim et al.'s
reference law), so the legacy gain knee is
`G_knee = (1/S_ideal_thr + 1)/2`, `S_ideal_thr = (1−η)/η·(10^(eps/10)−1)`
(printed as a reference).

Under **Ultra**, in-cell transmission `tau = e^(−OD)` degrades the twin beams
*inside* the cell before detection: `S = η·[tau·S_ideal + (1−tau)] + (1−η)`. The
achievable floor rises to `~1−η·e^(−OD)`; OD grows with T, so high-T points become
loss-limited while low-T points stay gain-limited. The engine's escape is
**detuning**: at large |Δ| the seed sits far off-resonance, OD→0, and the full
detection floor is recovered — so the Ultra optimum migrates to larger |Δ| (modest
T) rather than the legacy small-|Δ| knee. The deepest-δ argmin is η-independent (one
`tau` per spectrum), so the grid is solved **once** and any η/eps follows from the
captured `(G_s, G_c, OD)`.

Read the verdict this way:
- **EFFICIENT**: a trustworthy interior point reaches within `eps` of `ξ_opt` at
  minimum cost, and no point beats the detection floor → the cheapest near-optimal,
  most robust operating point.
- **CHECK**: a frontier was found but `ξ_opt` sits on a grid boundary, so it may not
  be the true optimum → widen `--t-*` / `--opd-*` and rerun to confirm.
- **INCOMPLETE (edge)**: the optimum is reached only at δ-window-edge-limited points
  → widen `--window` and rerun.
- **INCOMPLETE (legacy)**: legacy only — best ξ is >0.5 dB short of the detection
  floor (gain never cleared the knee) → raise `--t-max` / widen the OPD range.
- **FAIL** (points below the detection floor): physically impossible (`S ≥ 1−η`) →
  a coding/numerical error to investigate.

## Lessons baked in (do not relearn the hard way)
- **Code drifts — never hardcode reference values.** UI parameter *names* change
  (e.g. `ls` → `line_strength`). The script reads the Sim reference from the
  bit-locked `tests/test_regression.py::CONFIGS['sim_optimum']` instead, using
  `compute_spectrum`'s native kwargs. If that anchor has moved, the script exits
  with instructions; then open `tests/test_regression.py`, find the current
  optimum config, and pass `--p-pump/--p-probe/--line-strength` (or update the
  script's `load_reference`). Note: `line_strength` is now a dimensionless
  **residual** (anchor value `1.0`); the physical macroscopic scale
  (`fwm.physical_coupling_norm` = `p_F/[2(2I+1)]` × pure-85Rb CRC density) lives
  inside `compute_spectrum`, so the anchor's `1.0` is correct — not a missing
  calibration.
- **One Raman branch only.** Do not sum susceptibilities across branches — the
  engine rejects it. The δ scan on the `−1` branch already gives the unique dip.
- **η is a post-process**, not a re-solve. The grid is solved once; ξ for any
  η/loss/eps follows from the captured `(G_s, G_c, in-cell OD)` via the
  segmented-loss formula — even under Ultra, because the deepest-δ argmin is
  η-independent (one `tau = e^(−OD)` per spectrum).
- **δ-window edge points are excluded from the frontier.** At low T / low gain
  (exactly where the frontier lives) the deepest-δ can fall outside the window;
  such points have untrustworthy gains and are dropped from candidacy (and
  counted). That is why the default window is wide (±0.7 GHz).
- **Float64 cancellation** breaks the squeezing formula for `G_s ≳ 1e7`
  (spurious very-deep dB) — but the frontier lives at `G_s ≈ 30`, far below it,
  so the frontier result is numerically pristine (unlike the old plateau-edge
  maximum at the cap, `G_s ≈ 3.75e4`).
- **Analysis only.** The FWM baseline is bit-locked (`tests/baseline_focused.npz`
  at rtol 1e-9). Do **not** edit FWM physics for this; if you must, the baseline
  has to be recaptured (`python tests/capture_baseline.py`).
- **Twin-beam gap gate (`GAP_MIN`/`GAP_MAX` = `[0.5, 1.5]`).** The ideal noise
  `S_ideal=(G_s−G_c)²/(G_s+G_c)` (`observables.ideal_twin_beam_noise`) is the exact
  bound only where `G_s−G_c=1` (a genuine two-mode parametric amplifier). Off the
  real FWM/Raman resonance — background absorption/dispersion, or a gain pole
  under Ultra's segmented dynamics — `G_s` and `G_c` decouple and `(G_s−G_c)²` can
  vanish with **no real parametric gain**, reporting spuriously perfect
  "squeezing" (observed: `G_s≈G_c≈2.3` at an off-resonance δ reporting ξ within
  0.1 dB of the floor). The deepest-δ argmin is restricted to δ where
  `G_s−G_c ∈ [GAP_MIN, GAP_MAX]` (the validated Sim-reference δ scan keeps the gap
  in `[0.55, 0.98]` end to end); a `(Δ,T)` with no valid-gap δ in the window is
  excluded from the optimum/frontier like a δ-edge point (`gap_bad`, counted in the
  THEORY CHECK block). **This gate matters more after any change to the ideal-noise
  formula** — the old (pre-fix) formula required `G_s·G_c≥1` and degraded smoothly
  near `G_s≈G_c`, so it happened not to produce this artifact; the corrected
  formula does, so don't remove the gate when re-deriving the physics again.

## Sanity expectations (current code, default = Ultra)
> Full-fidelity numbers (`fwm.physical_coupling_norm` = p_F/[2(2I+1)] + pure-85Rb
> CRC density; density/T-dependent collisional decoherence via
> `fwm.collisional_atom`; corrected ideal noise `S_ideal=(G_s−G_c)²/(G_s+G_c)`;
> Ultra propagation incl. in-cell loss; twin-beam gap gate applied). Headline:
> in-cell loss pushes the optimum to LARGER |Δ| (where OD→0), not a small-|Δ| knee.
- **Default Ultra run** (η = QE·(1−loss) = 0.92·0.945 = **0.8694**, `eps=0.1`, full
  grid ~110 s on 16 cores): best achievable ξ_opt ≈ **−8.834 dB** (0.006 dB above
  the detection floor, at Δ ≈ **−2.2 GHz**, T ≈ **135 °C**, `G_s ≈ 636`, in-cell
  OD ≈ 0). Efficient frontier ≈ **Δ −2.1 GHz, T 125 °C, G_s ≈ 54**, in-cell OD ≈ 0,
  ξ ≈ **−8.77 dB** (gap +0.07), VERDICT `EFFICIENT`, 0 below floor, 0 δ-edge/gap
  exclusions. eps sweep: (Δ −2.1 GHz, T 135 °C, `G_s≈159`) at `eps=0.05`;
  (Δ −2.0 GHz, T 120 °C, `G_s≈27`) at `eps=0.2`.
- **Why Ultra ≠ legacy** (illustrative point Δ −1.1 GHz, T 95 °C): legacy
  ξ ≈ **−7.16 dB**, `G_s≈11.2`, but ultra ξ ≈ **−4.28 dB**, `G_s≈7.9` — in-cell
  OD ≈ 0.30 (≈26 % loss) degrades the twin beams inside the cell. Detuning to
  ≈ −2.1 GHz trades a little T/gain to drop OD toward 0 and recover near the
  floor. Surfacing that trade-off is the whole point of scanning at full fidelity.
- Detection floor `10·log10(1−η)` ≈ **−8.841 dB**; lossless detection (`--loss 0`,
  η=QE=0.92) ≈ **−10.97 dB** (in-cell loss still applies under Ultra).
- A `--fidelity legacy` run on the default grid finds `ξ_opt` on a **grid boundary**
  (VERDICT `CHECK` — 43/1254 points sit within 0.01 dB of it, a broad plateau at
  the floor as expected for the lossless model); its `eps=0.1` frontier lands near
  Δ −2.1 GHz, T 120 °C, `G_s≈64` — widen `--t-max` to pin the true interior optimum
  if you need one.
If the numbers are wildly off (VERDICT `FAIL`, or `CHECK` with the optimum pinned at
a grid edge, or a nonzero twin-beam-gap-gate exclusion count that swallows most of
the grid), suspect code drift / too-narrow a grid / a broken ideal-noise formula
and re-check `load_reference`, the `--t-*`/`--opd-*` ranges, and `GAP_MIN`/`GAP_MAX`
before trusting the output.
