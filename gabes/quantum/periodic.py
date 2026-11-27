"""Microscopic cyclostationary atomic noise in a complete Hermitian basis.

H(t)/hbar=H0+V exp(-i nu t)+V† exp(i nu t). Constant explicit jumps;
periodic means and harmonic-correlated noise, never a period-averaged repair.
This is an atomic second-moment model, not a finite-seed field squeezing solver.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import schur, solve_triangular

from .. import core
from .contracts import GeneratorFrequencyAxis, readonly_array
from .diffusion import traceless_hermitian_basis
from .reservoirs import ExplicitReservoirs


def _order(value, name, minimum=0):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _series(coefficients, phases):
    values = np.asarray(coefficients)
    q = np.arange(-(len(values)//2), len(values)//2+1)
    return np.einsum("tq,q...->t...", np.exp(-1j*np.outer(phases, q)), values)


def _toeplitz(coefficients, harmonics):
    """Block (h,k) is D_(h-k); individual D_q need not be positive."""
    coeff = np.asarray(coefficients)
    order, m = len(coeff)//2, coeff.shape[-1]
    out = np.zeros((len(harmonics)*m,)*2, complex)
    for i, h in enumerate(harmonics):
        for j, k in enumerate(harmonics):
            if abs(h-k) <= order:
                out[i*m:(i+1)*m, j*m:(j+1)*m] = coeff[h-k+order]
    return out


def _psd(matrices, rtol=1e-9):
    a = np.asarray(matrices)
    adj = a.conj().swapaxes(-1, -2)
    scale = np.maximum(np.linalg.norm(a, axis=(-2, -1)), np.finfo(float).tiny)
    herm = np.linalg.norm(a-adj, axis=(-2, -1))/scale
    minimum = np.linalg.eigvalsh((a+adj)/2)[..., 0]
    return {"passed": bool(np.all(herm <= rtol) and np.all(minimum >= -rtol*scale)),
            "minimum_eigenvalue": float(np.min(minimum)),
            "minimum_relative_eigenvalue": float(np.min(minimum/scale)),
            "maximum_hermiticity_relative_error": float(np.max(herm))}


class _FloquetResolvent:
    """One complex Schur basis for the full or left-projected resolvent."""

    def __init__(self, drift, use_schur):
        self.drift = drift
        self.diagonal = np.diag_indices_from(drift)
        self.identity = np.eye(len(drift))
        self.factor = None
        if use_schur:
            try:
                self.factor = schur(drift, output="complex", check_finite=False)
            except np.linalg.LinAlgError:
                pass
        if self.factor is not None:
            self.triangle, self.basis = self.factor
            self.basis_h = self.basis.conj().T
            self.poles = self.triangle.diagonal().copy()
            diagonal = self.drift.diagonal()
            row_norm = abs(self.drift).sum(axis=1)
            self.scale = np.max(row_norm)
            self.off_diagonal_row_norm = row_norm-abs(diagonal)

    def eigenvalues(self):
        if self.factor is not None:
            # Use the reference stability test when the sign is unresolved at
            # the scale of the Schur factorization error.
            if abs(np.max(self.poles.real)) > np.sqrt(np.finfo(float).eps)*self.scale:
                return self.poles
        return np.linalg.eigvals(self.drift)

    def _solve(self, omega, triangle, projection, projected_basis):
        if self.factor is not None:
            # Small spectral gaps can amplify backward error in the Schur
            # factors. Keep the original LU at extremely narrow resonances.
            gap = np.min(abs(-1j*omega-self.poles))
            scale = max(self.scale, abs(omega))
            if gap > np.sqrt(np.finfo(float).eps)*scale:
                triangle[self.diagonal] = -self.poles-1j*omega
                projected = projection is not None

                def triangular_solve(rhs):
                    if projected:
                        return solve_triangular(triangle.T, rhs.T, lower=True,
                                                check_finite=False).T @ self.basis_h
                    return self.basis @ solve_triangular(triangle, rhs, check_finite=False)

                rhs = projected_basis if projected else self.basis_h
                value = triangular_solve(rhs)
                # Refine in physical coordinates where cancellation against
                # the lab frequency occurs, using the same triangular factors.
                eps = np.finfo(float).eps
                rhs_scale = np.max(abs(projection)) if projected else 1.
                parity_budget = (1e-12/(4*eps))*rhs_scale
                near_resonance = gap < (32*eps/1e-12)*scale
                # A nonnormal lift can amplify the solve even far from all
                # eigenvalues. Use a cheap upper bound for |value||matrix|
                # (or the reverse product in the full solve) at every shift.
                if not near_resonance:
                    matrix_norm = np.max(
                        self.off_diagonal_row_norm+abs(-1j*omega-self.drift.diagonal()))
                    product_bound = np.max(abs(value).sum(axis=1))*matrix_norm
                    if product_bound <= parity_budget and np.isfinite(value).all():
                        return value

                matrix = -1j*omega*self.identity-self.drift
                product_scale = np.max(abs(value)@abs(matrix) if projected
                                       else abs(matrix)@abs(value))
                # The residual cannot reliably resolve a more strongly
                # cancelling product. Replay it with the original LU.
                if product_scale <= parity_budget:
                    if projected:
                        defect = projection-value@matrix
                        value += triangular_solve(defect@self.basis)
                    else:
                        defect = self.identity-matrix@value
                        value += triangular_solve(self.basis_h@defect)
                    if np.isfinite(value).all():
                        return value
        matrix = -1j*omega*self.identity-self.drift
        if projection is None:
            return np.linalg.solve(matrix, self.identity)
        return np.linalg.solve(matrix.T, projection.T).T

    def sweep(self, omega, projection=None):
        projected_basis = (projection@self.basis
                           if projection is not None and self.factor is not None else None)
        triangle = -self.triangle if self.factor is not None else None
        for w in omega:
            yield self._solve(w, triangle, projection, projected_basis)


@dataclass(frozen=True)
class FloquetAtomicSpectrum:
    frequency_axis: GeneratorFrequencyAxis
    harmonics: np.ndarray
    greater: np.ndarray
    lesser: np.ndarray
    operator_count: int

    def __post_init__(self):
        for key in ("harmonics", "greater", "lesser"):
            object.__setattr__(self, key, readonly_array(getattr(self, key), real=key == "harmonics"))

    def block(self, left_harmonic, right_harmonic, *, ordering="greater"):
        if ordering not in ("greater", "lesser"):
            raise ValueError("greater or lesser ordering required")
        indices = []
        for h in (left_harmonic, right_harmonic):
            found = np.flatnonzero(self.harmonics == h)
            if len(found) != 1:
                raise ValueError("requested harmonic absent from response ladder")
            indices.append(int(found[0]))
        i, j = indices
        m = self.operator_count
        return getattr(self, ordering)[:, i*m:(i+1)*m, j*m:(j+1)*m]

    def audit(self):
        result = {name: _psd(getattr(self, name)) for name in ("greater", "lesser")}
        result["passed"] = all(v["passed"] for v in result.values())
        return result


@dataclass(frozen=True)
class FloquetAtomicLift:
    beat_rad_s: float
    harmonics: np.ndarray
    drift: np.ndarray
    greater_by_reservoir: np.ndarray
    lesser_by_reservoir: np.ndarray
    reservoir_names: tuple[str, ...]
    operator_count: int

    def __post_init__(self):
        for key in ("harmonics", "drift", "greater_by_reservoir", "lesser_by_reservoir"):
            object.__setattr__(self, key, readonly_array(getattr(self, key), real=key == "harmonics"))

    @cached_property
    def _schur_resolvent(self):
        return _FloquetResolvent(self.drift, use_schur=True)

    @cached_property
    def _drift_audit(self):
        resolvent = self.__dict__.get("_schur_resolvent")
        eigenvalues = (resolvent.eigenvalues() if resolvent is not None
                       else np.linalg.eigvals(self.drift))
        result = {"greater": _psd(self.greater_by_reservoir), "lesser": _psd(self.lesser_by_reservoir),
                  "maximum_drift_real_eigenvalue": float(np.max(eigenvalues.real))}
        result["passed"] = bool(result["greater"]["passed"] and result["lesser"]["passed"]
                                 and result["maximum_drift_real_eigenvalue"] < 0)
        return result

    def audit(self):
        """Check ordered diffusion and drift stability of the immutable lift.

        The spectral abscissa can differ by roundoff if Schur factors were
        already computed. Near zero the audit uses the direct eigensolve.
        """
        result = self._drift_audit
        return dict(result, greater=dict(result["greater"]), lesser=dict(result["lesser"]))

    def spectrum(self, frequency_axis):
        """R D R†, with R=(-i omega - A_lift)^-1; units seconds.

        x_h(omega)=delta F(omega+h*nu); omega is a Floquet quasifrequency.
        Restrict to one half-open zone to avoid redundant physical coordinates.
        """
        if not isinstance(frequency_axis, GeneratorFrequencyAxis):
            raise TypeError("explicit Floquet generator-frequency axis required")
        omega = frequency_axis.omega_rad_s
        if np.any(omega < -abs(self.beat_rad_s)/2) or np.any(omega >= abs(self.beat_rad_s)/2):
            raise ValueError("base frequencies must lie in [-abs(nu)/2, abs(nu)/2)")
        with core.blas_single_thread():
            resolvent = (self._schur_resolvent if len(omega) >= 4
                         else _FloquetResolvent(self.drift, use_schur=False))
            if not self.audit()["passed"]:
                raise ValueError("unstable or nonpositive finite Floquet lift")
            d = [self.greater_by_reservoir.sum(axis=0), self.lesser_by_reservoir.sum(axis=0)]
            result = [[], []]
            for r in resolvent.sweep(omega):
                for out, diffusion in zip(result, d):
                    out.append(r@diffusion@r.conj().T)
            spectrum = FloquetAtomicSpectrum(frequency_axis, self.harmonics, *result, self.operator_count)
            if not spectrum.audit()["passed"]:
                raise ValueError("Floquet atomic spectrum failed positivity; no clipping")
            return spectrum


@dataclass(frozen=True)
class PeriodicAtomicNoise:
    hamiltonian_zero_rad_s: np.ndarray
    hamiltonian_plus_rad_s: np.ndarray
    reservoirs: ExplicitReservoirs
    beat_rad_s: float
    operators: np.ndarray
    state_harmonics: np.ndarray
    drift_harmonics: np.ndarray
    diffusion_harmonics_by_reservoir: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        for key in ("hamiltonian_zero_rad_s", "hamiltonian_plus_rad_s", "operators", "state_harmonics",
                    "drift_harmonics", "diffusion_harmonics_by_reservoir"):
            object.__setattr__(self, key, readonly_array(getattr(self, key)))

    def at_phase(self, phases_rad):
        phases = readonly_array(phases_rad, real=True)
        if phases.ndim != 1 or not len(phases):
            raise ValueError("nonempty phase vector required")
        order = len(self.state_harmonics)//2
        derivative = (-1j*self.beat_rad_s*np.arange(-order, order+1))[:, None, None]*self.state_harmonics
        return {"state": _series(self.state_harmonics, phases), "state_derivative": _series(derivative, phases),
                "drift": _series(self.drift_harmonics, phases),
                "diffusion_by_reservoir": np.stack([_series(d, phases) for d in self.diffusion_harmonics_by_reservoir])}

    def lift(self, response_order, *, _use_schur=False):
        order = _order(response_order, "response_order")
        harmonics = np.arange(-order, order+1)
        m = len(self.operators)
        drift = _toeplitz(self.drift_harmonics, harmonics)+np.kron(np.diag(1j*harmonics*self.beat_rad_s), np.eye(m))
        greater = np.array([_toeplitz(d, harmonics) for d in self.diffusion_harmonics_by_reservoir])
        # Transpose atomic operator indices only, NOT the entire lifted matrix.
        lesser = np.array([_toeplitz(d.swapaxes(-1, -2), harmonics) for d in self.diffusion_harmonics_by_reservoir])
        result = FloquetAtomicLift(self.beat_rad_s, harmonics, drift, greater, lesser,
                                  tuple(j.name for j in self.reservoirs.channels), m)
        if _use_schur:
            result._schur_resolvent
        if not result.audit()["passed"]:
            raise ValueError("finite Floquet lift failed stability or diffusion positivity")
        return result


def periodic_atomic_noise(h0, v, reservoirs, beat_rad_s, *, mean_order, phase_samples=96):
    """Derive A_q from L_q and D_q independently from explicit jump products."""
    if not isinstance(reservoirs, ExplicitReservoirs) or not reservoirs.channels:
        raise TypeError("nonempty explicit reservoirs required")
    h0, v = readonly_array(h0), readonly_array(v)
    n = reservoirs.n_levels
    beat = float(beat_rad_s)
    mean_order = _order(mean_order, "mean_order", 1)
    phase_samples = _order(phase_samples, "phase_samples", 4*mean_order+3)
    if h0.shape != (n, n) or v.shape != (n, n) or not np.isfinite(beat) or beat == 0:
        raise ValueError("matched finite Hamiltonians and nonzero beat required")
    generators = np.array([core.comm_super(v.conj().T), reservoirs.generator(h0), core.comm_super(v)])
    operators = traceless_hermitian_basis(n)
    basis = operators.reshape(n*n-1, n*n).T
    drift = np.array([basis.conj().T@g@basis for g in generators])
    # Resolve physical mixing over one period before solving a continuous spectrum.
    m = len(operators)
    def rhs(theta, raw):
        a = drift[1]+drift[2]*np.exp(-1j*np.sign(beat)*theta)+drift[0]*np.exp(1j*np.sign(beat)*theta)
        return (a@raw.reshape(m, m)/abs(beat)).reshape(-1)
    flow = solve_ivp(rhs, (0., 2*np.pi), np.eye(m, dtype=complex).reshape(-1),
                     method="DOP853", rtol=2e-11, atol=2e-13)
    if not flow.success:
        raise ValueError("periodic stability integration failed")
    radius = float(np.max(abs(np.linalg.eigvals(flow.y[:, -1].reshape(m, m)))))
    if radius >= 1-2e-8:
        raise ValueError("nondecaying or unresolved periodic atomic modes")
    states = core.floquet_solve_truncated(generators[1], generators[2], generators[0], beat,
        [0.], np.zeros((n*n, n*n)), n, n_f=mean_order, return_harmonics=True)[0]
    # Linear functional of rho_q. rho_q (q!=0) itself is not a physical state.
    diffusion = []
    for channel in reservoirs.channels:
        comm = operators@channel.operator-channel.operator@operators
        gram_operators = comm.conj().swapaxes(-1, -2)[:, None]@comm[None, :]
        diffusion.append(np.einsum("ijab,qba->qij", gram_operators, states))
    diffusion = np.array(diffusion)
    provisional = PeriodicAtomicNoise(h0, v, reservoirs, beat, operators, states, drift, diffusion, {})
    phases = 2*np.pi*np.arange(phase_samples)/phase_samples
    sampled = provisional.at_phase(phases)
    rho, rhodot, a, d = sampled["state"], sampled["state_derivative"], sampled["drift"], sampled["diffusion_by_reservoir"]
    means = np.einsum("iab,tba->ti", operators, rho)
    mean_dot = np.einsum("iab,tba->ti", operators, rhodot)
    products = operators[:, None]@operators[None, :]
    covariance = np.einsum("ijab,tba->tij", products, rho)-means[:, :, None]*means[:, None, :]
    cdot = np.einsum("ijab,tba->tij", products, rhodot)-mean_dot[:, :, None]*means[:, None, :]-means[:, :, None]*mean_dot[:, None, :]
    total_d = d.sum(axis=0)
    terms = [cdot, -a@covariance, -covariance@a.swapaxes(-1, -2), -total_d]
    defect = sum(terms)
    scale = np.maximum(sum(np.linalg.norm(t, axis=(-2, -1)) for t in terms), np.finfo(float).tiny)
    k = covariance-covariance.swapaxes(-1, -2)
    kdot = cdot-cdot.swapaxes(-1, -2)
    kdefect = kdot-a@k-k@a.swapaxes(-1, -2)-total_d+total_d.swapaxes(-1, -2)
    derivative = _series(generators, phases)@rho.reshape(phase_samples, n*n, 1)
    generator_scale = sum(np.linalg.norm(g) for g in generators)
    state_residual = float(np.max(np.linalg.norm(derivative[..., 0]-rhodot.reshape(phase_samples, n*n), axis=-1))/generator_scale)
    diagnostics = {"phase_samples": phase_samples, "period_map_spectral_radius": radius,
        "decay_gap_s_inverse": float(-np.log(radius)*abs(beat)/(2*np.pi)),
        "maximum_state_equation_relative_residual": state_residual,
        "maximum_state_trace_error": float(np.max(abs(np.trace(rho, axis1=-2, axis2=-1)-1))),
        "maximum_state_hermiticity_error": float(np.max(np.linalg.norm(rho-rho.conj().swapaxes(-1, -2), axis=(-2, -1)))),
        "minimum_phase_sampled_state_eigenvalue": float(np.linalg.eigvalsh((rho+rho.conj().swapaxes(-1, -2))/2).min()),
        "maximum_dynamic_einstein_relative_residual": float(np.max(np.linalg.norm(defect, axis=(-2, -1))/scale)),
        "maximum_dynamic_commutator_relative_residual": float(np.max(np.linalg.norm(kdefect, axis=(-2, -1))/scale)),
        "phase_diffusion": _psd(d), "phase_ordered_covariance": _psd(covariance),
        "scope": "sampled periodic state and exact jump functional; harmonic truncation requires convergence"}
    diagnostics["passed"] = bool(state_residual < 1e-8 and diagnostics["maximum_state_trace_error"] < 1e-9
        and diagnostics["maximum_state_hermiticity_error"] < 1e-9
        and diagnostics["minimum_phase_sampled_state_eigenvalue"] >= -1e-10
        and diagnostics["maximum_dynamic_einstein_relative_residual"] < 1e-7
        and diagnostics["maximum_dynamic_commutator_relative_residual"] < 1e-7
        and diagnostics["phase_diffusion"]["passed"] and diagnostics["phase_ordered_covariance"]["passed"])
    if not diagnostics["passed"]:
        raise ValueError(f"periodic atomic consistency failed: {diagnostics}")
    return PeriodicAtomicNoise(h0, v, reservoirs, beat, operators, states, drift, diffusion, diagnostics)
