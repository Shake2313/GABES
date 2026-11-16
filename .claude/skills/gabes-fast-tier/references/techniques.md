# Fast-tier techniques used in GABES

Measured in FWM unless noted. Sources: GABES-archive `analysis/fwm_lite/DEVLOG.md`
(2026-09-11/14), `analysis/pole_reuse/DEVLOG.md` (2026-09-14), the FWM Ultra
performance diagnosis (2026-09-09/11) and the basis and Numba work
(2026-06/07). Numbers are for the stated machine and model at the time; use
them for ordering, then re-measure.

## Order of payoff

1. Remove work structurally: fewer solves, fewer grid points, symmetry.
2. Then batch, pool and defer verification.
3. Kernel tuning last. At matrix size ~16, LU (M³/3 multiply-adds) is cheaper
   than a matrix product (M³), so iterative and low-rank tricks lose. What wins
   is cutting the number of grid points and blocks.

## 1. Pole–residue Doppler average

**What.** When a scan row has the form `(A00 + t·A01 − D·S) x = rhs`, with `t`
the scan variable and `D` the Doppler shift entering through a fixed matrix
`S`, eigendecompose once per row and write the velocity dependence as a sum of
poles. The Maxwell average then becomes a closed form (Faddeeva) or an exact
sum over the reference's discrete nodes. FWM: about 1,600 solves per probe
detuning became one eigendecomposition. Fast 0.10 s, Balanced 0.24 s; the
full-scan view went from 4.1/8.1 s to 0.72 s.

**Code.** `gabes/pole_doppler.py`: `AffineShiftSystem`, `ShiftInvertSystem`,
`PoleResidues.discrete_mean(nodes, weights)` (reproduces a reference grid
measure) and `PoleResidues.gaussian_mean` (untruncated Maxwell),
`residues_many`, `floquet_real_form`. Non-parallel beams:
`affine_doppler.joint_maxwell_mean`.

**Applicability test.**
- Λ EIT/AT/CPT: `affine_doppler.AffineScan` (formerly
  `absorption._affine_scan_coeffs`) already builds `base + s·A + kv·B`, so
  `A00 = base` (plus the trace row), `A01 = A`, `S = −B`.
- Magneto buffer (B, v) grid: Zeeman is affine in B, so `t = B`.
- Finite Floquet: `floquet_real_form` when L and S are real and C₋ = conj(C₊).
- Outside the assumption: Doppler that also enters the two-photon detuning in
  non-collinear geometry (use the joint Maxwell path), and matrices whose
  structure depends on velocity (velocity-dependent collision rates).

**Traps.**
- Use the reference's measure. An untruncated mean against a 4σ-cut reference
  gave 0.112 dB at 150 °C.
- `AffineShiftSystem` (Schur, eliminating A_NN) fails when the Doppler-free
  block has no relaxation of its own: Numba ZeroDivisionError for magneto
  paraffin F=1→F'=2 at B⊥=0. `ShiftInvertSystem` expands about a regular point
  D0 (center − σ, guard at center) and needs only A(D0) to be regular.
- Rounding gives μ≈0 modes, nearly real far poles, that trip the non-decay
  check. Remove them with `ShiftInvertSystem.residues(..., pole_limit=...)`
  (1e4σ) and judge non-decay only inside
  the Maxwell bulk.

## 2. Symmetry and real arithmetic

- **Hermitian (coherence-vector) basis.** `core.hermitian_basis(n)`: L₀ and
  S_v become real, so real LU replaces complex LU. Λ-EIT 1.61×, magneto
  two-region 1.35×.
- **± sideband conjugate symmetry.** A₊ = P·A₋*·P and Cp = P·Cm*·P, so solve
  one side and conjugate. FWM 1.69×. It was lost once in a generalization
  (7244f25); a regression test should pin it.
- **𝕁 real form.** Conjugation with harmonic inversion maps solutions to
  solutions, so the Floquet system becomes a real system of the same size:
  dgeev 0.44 ms against zgeev 1.22 ms for 56×56.
- **Floquet order.** Measure convergence. In FWM, n_f 2→3 differs by
  1.6e-12…1.7e-8 because Ω_hf ≫ Ω_pump. Lowering the order changes values
  beyond the regression tolerance, so it is a value decision.

## 3. Velocity quadrature, when you need one

- Resonances narrower than the node spacing defeat Gauss–Legendre and
  Gauss–Hermite: 25–740 % error with 16–128 nodes. Composite GL panels: 20–77 %.
  A sinh-mapped trapezoid was worse than uniform for the same node count.
- The uniform trapezoid converges as e^{−2πγ/h}. With γ ≥ Γ/2 this gives a
  node spacing of h ≲ 2–4 m/s. Do this Fermi estimate before running scans.
- Judge the resonance cluster by its distance in units of kσ, not by
  Δ = 0. Sideband resonances also exist (FWM: Δ_eff = −3.118 GHz, 1/42 of the
  main resonance, FWHM 15.3 MHz).
- Readouts that cancel two large numbers are far more grid-sensitive than the
  gains themselves (FWM S_dB against G_s).

## 4. Adaptive display axis

**What.** Solve a subset of the display points, spline the linear-response
quantities between them, and refine where the observable disagrees.
`adaptive_scan.initial_scan_nodes`, `refine_scan_nodes`, `cubic_spline`
(not-a-knot, Thomas algorithm, 4e-14 against SciPy).

**Rules.**
- Interpolate quantities that are linear in the state (averaged
  susceptibilities). Judge refinement on the observables the user sees, with
  tolerances proportional to each curve's range.
- Initial node spacing: stride × the smallest display spacing that holds for
  four consecutive points. The plain minimum collapsed when one coarse point
  sat 1 kHz from a dense window.
- Trust refinement only down to the display spacing where it was verified.
  FWM was verified at 2.75 MHz. In a 0.4 MHz window it missed a 2 MHz gain
  structure at 150 °C (log10 G_s tube 0.156) because S_dB was flat there. Such
  windows solve every point.
- Judge validity and availability states from solved values only. The
  2026-11 Fast failures included Passive L1 eligibility flipping at
  interpolated points.

**Rejected.** Uniform thinning of the δ axis (stride 2/4/8: 3.8–8.4 dB, because
it misses ~5 MHz Raman features). PCHIP and Akima (several to tens of times
worse than cubic; they flatten extrema).

## 5. Guards

- Compare every solved row of the new form against the existing compiled
  kernel (FWM: relative 1e-6, measured maximum 5e-8). Replace a failing row
  with the grid solution and re-judge refinement with the corrected value.
- Run the guard once at the end, not every round: same result, 8 % less time.
- A guard proves the solved nodes. It proves nothing about interpolated points
  or source-coordinate mapping; the gate's dense comparison covers those.

## 6. Numba and LAPACK

- Fuse batched tiny solves into one `prange` kernel with hand-written LU.
  Matching LAPACK's pivot rule (|Re| + |Im|, cabs1) agrees with
  `np.linalg.solve` to ~1e-13, so frozen baselines need no recapture. FWM
  χ̄ table 8.6–10×; magneto 4.9–10.9×; Λ and Rydberg affine scan 13–69×.
- For 2M ≥ ~128, call `np.linalg.solve` inside the kernel (LAPACK, about 3.3×
  faster than hand LU at 200²) and wrap it in `core.blas_single_thread()` so
  BLAS threads do not stack on `prange` threads.
- LAPACK through ctypes inside Numba works but cannot use `cache=True`
  ("dynamic globals"), so it recompiles in every process. Split the work:
  structural arithmetic in cached Numba, eigendecomposition with NumPy LAPACK
  in a thread pool (LAPACK releases the GIL).
- Never write a per-batch basis change as
  `np.einsum('ab,nbc,cd->nad', ...)`. NumPy picks a non-BLAS path (78× slower).
  Use batched `@`.
- The first JIT compile per signature can take ~230 s; a warm `cache=True`
  disk cache restores normal speed. Warm up before timing.
- Keep a NumPy fallback (`GABES_DISABLE_NUMBA`) and pin its parity in
  `tests/test_kernels.py`.

## 7. Batching, pooling and caching

- Score predicted and exact points together, pool eigendecompositions by
  matrix size (minimum chunk 4), and use deterministic chunk boundaries so the
  same input gives the same bits.
- Put each knob in the right compute tier: `recompute=False` knobs only
  post-process.
- Reuse tables across sweeps when the expensive object depends on fewer
  variables than the sweep. The FWM χ̄ table depends only on (δ, Δ_eff); a
  Δ ladder aligned to 16·k·dv gave 10.7× exact reuse in scans.
- Cache keys must hold every input that enters the solve. A shared cache
  keyed only on the order collided with a guard call in the tpd-angle scanner.

## 8. Cheap exact pieces for diagnostics

- Replace numerical weak-probe Doppler averages with the analytic Voigt
  (0.1 ms against 150 ms). SciPy is not a core requirement, so use the in-repo
  Weideman 32-term Faddeeva (3e-13 against SciPy).
- Do not compute expensively what is not displayed. Record the substitution in
  provenance when it changes a non-displayed array (FWM arm OD tails differed
  by up to 24 % against the 4σ-cut table).

## 9. Readout extraction

- Use cubic rather than linear interpolation for operating-point readouts. FWM
  went from 1.7e-4 to 6.6e-6, independent of the grid. An earlier "unconverged
  δ axis" was this readout error.
- Check where a built-in graded mesh is centred. FWM's mesh centred at δ = 0
  while the gain peak sits near δ ≈ −278 MHz, which made the peak error 170×
  worse.

## Rejected elsewhere (do not re-explore without new evidence)

| Attempt | Result | Why |
|---|---|---|
| Low-rank RHS Floquet kernel | 1.36× | M=16 per-point cost floor |
| AAA rational fit on the velocity axis | fit cost ≥ grid solve | SVD per row |
| Contour shift (complex Δ_eff) | impossible | 𝕁 symmetry puts poles on both sides |
| Complex eig (zgeev) | 2.8× slower than the real form | ignores 𝕁 symmetry |
| Coarse shared Δ_eff axis + interpolation | 6.5–11 % error | resonances too narrow |
| Woodbury updates | no gain | rank(C±) 6–8 of 16; base block changes per point |
| Velocity-axis Hessenberg–QZ | break-even ~237 nodes | too few nodes |
| Neumann series | 16,384 multiplications > LU 5,461 | small M |
| Adiabatic elimination | invalid | Ω/Δ ≈ 0.8 |
| χ̄ interpolation in temperature | 2.4e-3 at 20 °C anchors | too coarse |
| Merging the two linear-response solves | physics change | χ̄ depends on Ω_ref (106 % at half) |
| One-segment depletion scorer | not adopted | under-refines against the depletion the user sees |
