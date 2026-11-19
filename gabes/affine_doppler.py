"""
Doppler averages of a scan whose Liouvillian is affine in scan and velocity:

    L(s, kv) = base + s·A + kv·S_v

The Λ (two-photon detuning) and Rydberg ladder scans have this form whenever
`build_H0(s)` is affine in the scan variable s. `AffineScan` builds the three
constant operators once. Every scan point is a rational function of kv
(`pole_doppler`), so `maxwell_mean` returns the probe coherence ρ_eg / Ω_p averaged
over the Maxwell distribution in closed form: one small eigendecomposition per
point, no velocity grid and no cutoff. A point that misses the guard is summed on
successively finer grids until its complex response converges (or is withheld
at the class cap). `stationary` is the single v = 0 class.
"""
import math

import numpy as np

from . import constants, core, doppler, kernels, pole_doppler

POLE_GUARD_RTOL = 1e-6      # pole form vs a direct solve of the v = 0 class, every row
POLE_LIMIT_SIGMAS = 1e8     # farther poles are rounding modes; 1e4 already cuts a real one
BULK_SIGMAS = 6.0           # velocity span where a pole must be damped and a grid reaches
STRIP_FLOOR_SIGMAS = 1e-4   # narrowest pole a grid resolves; real ones stay above 9e-4 σ
QUADRATURE_ERROR = 1e-8     # trapezoid target and rejected-row complex refinement tolerance
_NUMPY_CLASS_BATCH = 2048   # bound the fallback stack of dense Liouvillians


def _shift_sigma(T, mass, k_vec):
    """Maxwell standard deviation of kv [rad/s]."""
    return k_vec * math.sqrt(constants.KB * T / mass)


def _strip_classes(damping, T, mass, k_vec):
    """Maxwell (v, weights) resolving the narrowest of the row `damping`s."""
    sigma = _shift_sigma(T, mass, k_vec)
    strip = float(np.clip(np.min(damping), STRIP_FLOOR_SIGMAS * sigma, sigma))
    return doppler.maxwell_strip_grid(T, mass, k_vec, strip, QUADRATURE_ERROR, BULK_SIGMAS)


def _quadrature_values(rows, velocity, weights, evaluate):
    """A singular class solve in a batch invalidates only its own scan row."""
    try:
        return np.asarray(evaluate(rows, velocity, weights), dtype=complex)
    except np.linalg.LinAlgError:
        values = np.full(rows.size, complex(np.nan, np.nan))
        if rows.size > 1:
            for i in range(rows.size):
                try:
                    values[i] = evaluate(rows[i:i + 1], velocity, weights)[0]
                except np.linalg.LinAlgError:
                    pass  # this row must refine again or reach the cap
        return values


def _refine_rejected(out, failed, evaluate, temperature, mass, k_vec, fallback_rows):
    """Refine the complex observable on rejected rows, dropping each as it converges.

    evaluate(rows, velocity, weights) returns one weighted mean per local row.
    Two consecutive relative changes must pass. The floor grid has 351,811
    classes (6 sigma, 1e-8 step criterion). Values still unresolved there
    stay unavailable, including persistent nonfinite responses.
    """
    rows = np.flatnonzero(failed)
    if not rows.size:
        return
    previous = np.full(rows.size, complex(np.nan, np.nan))
    agreements = np.zeros(rows.size, dtype=int)
    sigma = _shift_sigma(temperature, mass, k_vec)
    strip_fraction = 1.0
    total_classes = 0
    refinement = 0
    while rows.size:
        velocity, weights = _strip_classes([strip_fraction * sigma], temperature, mass, k_vec)
        classes = velocity.size
        total_classes += classes
        values = _quadrature_values(rows, velocity, weights, evaluate)
        scale = np.maximum(np.abs(previous), np.abs(values))
        # Exact zero followed by zero agrees. Near-zero values keep the same
        # relative test as every other row; there is no arbitrary absolute floor.
        finite = np.isfinite(previous) & np.isfinite(values) & np.isfinite(scale)
        change = np.full(rows.size, np.nan)
        with np.errstate(invalid="ignore", over="ignore"):
            np.divide(np.abs(values - previous), scale, out=change, where=finite & (scale > 0.))
        change[finite & (scale == 0.)] = 0.
        passed = change <= QUADRATURE_ERROR
        agreements = np.where(passed, agreements + 1, 0)
        converged = agreements >= 2
        at_cap = strip_fraction == STRIP_FLOOR_SIGMAS
        finished = converged | at_cap
        out[rows[converged]] = values[converged]
        if at_cap:
            out[rows[~converged]] = complex(np.nan, np.nan)
        if fallback_rows is not None:
            for i in np.flatnonzero(finished):
                fallback_rows.append(dict(
                    row=int(rows[i]), status="converged" if converged[i] else "unresolved",
                    classes=int(classes), total_classes=int(total_classes),
                    relative_change=float(change[i]) if np.isfinite(change[i]) else None,
                    refinements=refinement))
        rows = rows[~finished]
        previous = values[~finished]
        agreements = agreements[~finished]
        strip_fraction = max(0.5 * strip_fraction, STRIP_FLOOR_SIGMAS)
        refinement += 1


class AffineScan:
    """Constant pieces of L(s, kv) = base + s·A + kv·S_v for one probe readout.

    `scan_shift` is the optical scan operator, entering as −s·scan_shift: the Λ
    scan moves the optical detuning through it while a residual Raman Doppler
    term belongs only in S_v and must not change a stationary atom's scan
    Hamiltonian. Pass None when the scan already lives in `build_H0(s)` (the
    Rydberg probe detuning). The s-independent drive entries are bit-identical
    between build_H0(0) and build_H0(1), so their difference isolates dL/ds
    exactly with no cancellation.
    """

    def __init__(self, atom, build_H0, coherence, probe_omega, scan_shift=None):
        self.atom = atom
        self.build_H0 = build_H0
        self.coherence = coherence
        self.probe_omega = probe_omega
        self.scan_shift = scan_shift
        self.base = core.build_liouvillian(build_H0(0.0), atom)
        dL_ds = core.build_liouvillian(build_H0(1.0), atom) - self.base
        self.A = dL_ds if scan_shift is None else dL_ds - scan_shift

    def _kernel(self, scan, kv, w):
        e, g = self.coherence
        n = self.atom.n_levels
        with core.blas_single_thread():
            out = kernels.affine_scan_chi(
                self.base, self.A, self.atom.S_v,
                np.ascontiguousarray(scan, dtype=float),
                np.ascontiguousarray(kv, dtype=float),
                np.ascontiguousarray(w, dtype=float), e * n + g, n)
        return out / self.probe_omega

    def _guarded_poles(self, scan, sigma):
        try:
            return self._pole_rows(scan, sigma)
        except np.linalg.LinAlgError:
            return None, np.zeros(scan.size), np.ones(scan.size, dtype=bool)

    def maxwell_mean(self, scan, T, mass, k_vec, *, fallback_rows=None):
        """Exact Gaussian means, with independent refinement on rejected rows.

        Optional fallback_rows receives one diagnostic per rejected row.
        An unresolved value is complex NaN.
        """
        sigma = _shift_sigma(T, mass, k_vec)
        poles, _, failed = self._guarded_poles(scan, sigma)
        if poles is None:
            out = np.empty(scan.size, dtype=complex)
        else:
            out = poles.gaussian_mean(0.0, sigma)[:, 0] / self.probe_omega
            failed = failed | ~np.isfinite(out)

        def integrate(rows, velocity, weights):
            kv = k_vec * velocity
            if kernels.available() or kv.size <= _NUMPY_CLASS_BATCH:
                return self._class_sum(scan[rows], kv, weights)
            # The NumPy solver builds a dense generator stack per class.
            # Bound that temporary on rejection grids; keep global weights.
            values = np.zeros(rows.size, dtype=complex)
            for first in range(0, kv.size, _NUMPY_CLASS_BATCH):
                sl = slice(first, first + _NUMPY_CLASS_BATCH)
                values += self._class_sum(scan[rows], kv[sl], weights[sl])
            return values

        _refine_rejected(out, failed, integrate, T, mass, k_vec, fallback_rows)
        return out

    def quadrature_classes(self, scan, T, mass, k_vec):
        """Size a shared grid from accepted static poles, or use the floor if none pass.

        A grid alone cannot certify another observable. Use quadrature_mean
        to refine the actual complex readout on rejected rows.
        """
        _, damping, failed = self._guarded_poles(scan, _shift_sigma(T, mass, k_vec))
        accepted = damping[~failed]
        return _strip_classes(accepted if accepted.size else [0.0], T, mass, k_vec)

    def quadrature_mean(self, scan, T, mass, k_vec, evaluate, *, fallback_rows=None):
        """Average an independent observable on guarded and refined grids.

        evaluate(slot, velocities, weights) returns one complex weighted mean.
        Accepted slots retain their original common-grid result even in mixed
        batches. Rejected slots never use the rejected pole's reported width.
        """
        _, damping, failed = self._guarded_poles(scan, _shift_sigma(T, mass, k_vec))
        out = np.empty(scan.size, dtype=complex)
        if not np.all(failed):
            # The original finite-IF result for accepted companions used this
            # shared grid. Preserve its rounding; a rejected row never uses it.
            usable = damping[~np.isnan(damping)]
            v, w = _strip_classes(usable, T, mass, k_vec)
            accepted = np.flatnonzero(~failed)
            for slot in accepted:
                try:
                    out[slot] = evaluate(slot, v, w)
                except np.linalg.LinAlgError:
                    out[slot] = complex(np.nan, np.nan)
            # The static guard cannot catch a nonfinite external observable.
            failed = failed.copy()
            failed[accepted] = ~np.isfinite(out[accepted])
        _refine_rejected(out, failed,
                         lambda rows, v, w: [evaluate(row, v, w) for row in rows],
                         T, mass, k_vec, fallback_rows)
        return out

    def _pole_rows(self, scan, sigma):
        """Pole rows of the coherence, their damping and the rows the guard rejects.

        With the trace row inserted a row is (A00 + s·A01 − D·S) x = e₀ in the real
        Hermitian-generator basis, with D = −kv. A row is kept if its pole form
        reproduces a direct solve of the v = 0 class to POLE_GUARD_RTOL and has no
        undamped pole inside the Maxwell bulk.
        """
        e, g = self.coherence
        n = self.atom.n_levels
        A00, A01, S = (np.ascontiguousarray(core.to_real_liouvillian(X, n))
                       for X in (self.base, self.A, self.atom.S_v))
        for X in (A00, A01, S):
            X[0] = 0.0
        A00[0, :n] = 1.0                 # trace row: populations are the first n coordinates
        rhs = np.zeros(n * n)
        rhs[0] = 1.0
        # Expand about kv = +σ, not by Schur elimination: the ground populations
        # relax only through the optical coherences, so the Doppler-free block alone
        # is singular.
        system = pole_doppler.ShiftInvertSystem(A00, A01, S, rhs, core.hermitian_basis(n)[e * n + g])
        with core.blas_single_thread():
            poles = system.residues(scan, -sigma, pole_limit=POLE_LIMIT_SIGMAS * sigma)
            at_rest = system.direct(scan, 0.0)[:, 0]
        rest = poles.value_at(0.0)[:, 0]
        in_bulk = np.abs(poles.lam.real) <= BULK_SIGMAS * sigma
        damping = np.min(np.where(in_bulk, np.abs(poles.lam.imag), np.inf), axis=1)
        failed = ~((np.abs(rest - at_rest) <= POLE_GUARD_RTOL * np.abs(at_rest))
                   & (damping > 1e-9 * sigma))
        return poles, damping, failed

    def _class_sum(self, scan, kv, w):
        """χ̄(scan) = Σ_j w_j ρ_eg(s, kv_j) / Ω_p solved class by class."""
        if kernels.available():
            return self._kernel(scan, kv, w)
        e, g = self.coherence
        n = self.atom.n_levels
        out = np.zeros(scan.size, dtype=complex)
        for i, s in enumerate(scan):
            L0 = core.build_liouvillian(self.build_H0(s), self.atom)
            if self.scan_shift is not None:
                L0 = L0 - s * self.scan_shift
            # Δ = −kv reproduces the kernel's +kv·S_v velocity shift.
            rho = core.steady_state_batched(L0, -kv, self.atom.S_v, n)
            out[i] = ((rho[:, e, g] / self.probe_omega) * w).sum()
        return out

    def stationary(self, scan):
        """χ̄(scan) of the v = 0 class."""
        if kernels.available() or self.scan_shift is None:
            return self._class_sum(scan, np.zeros(1), np.ones(1))
        e, g = self.coherence
        n = self.atom.n_levels
        out = np.zeros(scan.size, dtype=complex)
        for i, s in enumerate(scan):
            L0 = core.build_liouvillian(self.build_H0(s), self.atom)
            rho = core.steady_state_batched(L0, np.array([s]), self.scan_shift, n)
            out[i] = rho[0, e, g] / self.probe_omega
        return out


def joint_maxwell_mean(scan, temperature, mass, longitudinal_k, conditional_scan, *,
                       transverse_orders=(32, 64, 128, 256), cutoff_sigma=6.0,
                       rtol=1e-4):
    """Integrate exact longitudinal means over an independent transverse velocity.

    ``conditional_scan(vx)`` provides an AffineScan with the transverse shift
    held fixed. Its single longitudinal coordinate carries all correlated
    shifts. Check refinement at fixed cutoff and compare a wider cutoff at a
    proportional order. Use both complex coherence and its scan derivative so
    a converged envelope cannot mask an unconverged dispersion readout.
    """
    longitudinal_fallback = []
    unresolved = np.zeros(scan.size, dtype=bool)

    def transverse_mean(order, cutoff):
        velocity, weights = doppler.maxwell_legendre_grid(
            temperature, mass, order, cutoff)
        mean = np.zeros_like(scan, dtype=complex)
        for vx, weight in zip(velocity, weights):
            rows = []
            mean += weight * conditional_scan(vx).maxwell_mean(
                scan, temperature, mass, longitudinal_k, fallback_rows=rows)
            for record in rows:
                longitudinal_fallback.append(dict(record, transverse_order=order,
                                                  cutoff_sigma=cutoff, velocity_m_s=float(vx)))
                if record["status"] == "unresolved":
                    unresolved[record["row"]] = True
        mean[unresolved] = complex(np.nan, np.nan)
        return mean

    def difference(a, b):
        def normalized_error(x, y):
            if unresolved.any():
                # Withhold unresolved rows and derivatives that use them.
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.any():
                    return None
                x, y = x[valid], y[valid]
            scale = max(float(np.max(np.abs(x))), np.finfo(float).tiny)
            return float(np.max(np.abs(x - y))) / scale

        return dict(
            coherence=normalized_error(a, b),
            derivative=normalized_error(np.gradient(a, scan), np.gradient(b, scan)),
        )

    def within_tolerance(errors):
        return (errors is not None
                and all(value is not None and value <= rtol for value in errors.values()))

    if len(transverse_orders) < 2 or any(
            lower >= upper for lower, upper in zip(transverse_orders, transverse_orders[1:])):
        raise ValueError("At least two increasing transverse orders are required.")
    history = []
    previous = None
    for order in transverse_orders:
        mean = transverse_mean(order, cutoff_sigma)
        refinement = None if previous is None else difference(mean, previous)
        history.append(dict(order=order, cutoff_sigma=cutoff_sigma, difference=refinement))
        if within_tolerance(refinement) or unresolved.all():
            break
        previous = mean
    cutoff_check = None
    if within_tolerance(refinement):
        wider_cutoff = cutoff_sigma + 1.0
        wider_order = math.ceil(order * wider_cutoff / cutoff_sigma)
        wider = transverse_mean(wider_order, wider_cutoff)
        cutoff_check = dict(order=wider_order, cutoff_sigma=wider_cutoff,
                            difference=difference(wider, mean))
        mean = wider
    converged = (not unresolved.any() and cutoff_check is not None
                 and within_tolerance(cutoff_check["difference"]))
    evidence = dict(status="converged" if converged else "not_converged",
                    relative_tolerance=rtol, transverse_refinement=history,
                    transverse_cutoff_check=cutoff_check,
                    longitudinal_method="conditional Maxwell pole mean with guarded direct fallback",
                    transverse_method="Gauss-Legendre weighted by the Maxwell density")
    if longitudinal_fallback:
        evidence["longitudinal_fallback"] = longitudinal_fallback
    return mean, evidence
