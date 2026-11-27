"""Finite Lindblad characteristics, inflow noise and correlated path readout.

Exact complete-basis second moments for a prescribed ballistic path. These
atomic wavepackets are not canonical optical ports or a self-consistent Maxwell
solution. A stationary Poisson stream additionally assumes identical age-based
protocols, independent entry states, no interatomic interactions and no feedback.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..core import comm_super
from .contracts import GeneratorFrequencyAxis, readonly_array
from .diffusion import traceless_hermitian_basis, _state
from .periodic import _psd
from .reservoirs import ExplicitReservoirs

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(10)
_GL_NODES.setflags(write=False)
_GL_WEIGHTS.setflags(write=False)


def _panel_quadrature(lo, hi, knots, max_omega):
    """Positive-weight Gauss-Legendre nodes/weights on [lo, hi] aligned to ODE knots."""
    edges = np.sort(np.asarray(knots, dtype=float))
    if edges[0] > lo:
        edges = np.r_[lo, edges]
    if edges[-1] < hi:
        edges = np.r_[edges, hi]
    nodes_list, weights_list = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        n_sub = max(1, int(np.ceil(max_omega * (b - a)))) if max_omega > 0.0 else 1
        sub = np.linspace(a, b, n_sub + 1)
        half = 0.5 * (sub[1:] - sub[:-1])
        mid = 0.5 * (sub[1:] + sub[:-1])
        nodes_list.append((mid[:, None] + half[:, None] * _GL_NODES[None, :]).ravel())
        weights_list.append((half[:, None] * _GL_WEIGHTS[None, :]).ravel())
    return np.concatenate(nodes_list), np.concatenate(weights_list)


@dataclass(frozen=True)
class BallisticPath:
    entry_position_m: np.ndarray
    velocity_m_s: np.ndarray
    residence_time_s: float
    source: str

    def __post_init__(self):
        for key in ('entry_position_m', 'velocity_m_s'):
            value = readonly_array(getattr(self, key), real=True)
            if value.shape != (3,):
                raise ValueError('Cartesian entry position and velocity required')
            object.__setattr__(self, key, value)
        duration = float(self.residence_time_s)
        if not np.isfinite(duration) or duration <= 0 or not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('positive finite residence time and path provenance required')
        object.__setattr__(self, 'residence_time_s', duration)

    def position(self, age_s):
        age = readonly_array(age_s, real=True)
        if np.any(age < 0) or np.any(age > self.residence_time_s):
            raise ValueError('age outside declared characteristic')
        return self.entry_position_m+age[..., None]*self.velocity_m_s


def _moments(rho, operators):
    means = np.einsum('iab,ba->i', operators, rho)
    raw = np.einsum('iab,jbc,ca->ij', operators, operators, rho, optimize=True)
    return means, raw-np.outer(means, means)


class AtomicCharacteristic:
    """Build with integrate_characteristic; intervals never reset the atom."""

    def __init__(self, path, hamiltonian, reservoirs, boundary_state, times, rtol, atol):
        if not isinstance(path, BallisticPath) or not isinstance(reservoirs, ExplicitReservoirs):
            raise TypeError('explicit path and reservoirs required')
        if not callable(hamiltonian):
            raise TypeError('Hamiltonian callback(age_s, position_m) required')
        self.path, self.hamiltonian, self.reservoirs = path, hamiltonian, reservoirs
        self.n = reservoirs.n_levels
        self.operators = traceless_hermitian_basis(self.n)
        self.basis = self.operators.reshape(-1, self.n*self.n).T
        self.boundary_state = _state(boundary_state, self.n)
        self.times_s = readonly_array(times, real=True)
        if (self.times_s.ndim != 1 or len(self.times_s) < 2 or self.times_s[0] != 0
                or self.times_s[-1] != path.residence_time_s or np.any(np.diff(self.times_s) <= 0)):
            raise ValueError('strictly increasing times must include entry zero and exit residence time')
        if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError('positive finite integration tolerances required')
        self.rtol, self.atol = rtol, atol
        # D_r,ij = Tr rho [Lr†,Fi][Fj,Lr]. Products are constant for fixed
        # jumps: cache them exactly, without deriving noise from A or K.
        products = []
        for channel in reservoirs.channels:
            comm = self.operators@channel.operator-channel.operator@self.operators
            products.append(np.einsum('iab,jbc->ijac', comm.conj().swapaxes(-1, -2), comm))
        m = len(self.operators)
        self.jump_products = np.array(products).reshape(len(products), m, m, self.n, self.n)
        self.source_names = ('atomic_inflow',)+tuple('jump:'+c.name for c in reservoirs.channels)
        self._dissipator = reservoirs.dissipator()
        self._dissipator_drift = (self.basis.conj().T@self._dissipator@self.basis).real
        self._solutions = []

    def generator(self, age):
        h = np.asarray(self.hamiltonian(age, self.path.position(age)), dtype=complex)
        if h.shape != (self.n, self.n) or not np.isfinite(h).all():
            raise ValueError('Hamiltonian dimension mismatch')
        scale = max(float(np.linalg.norm(h)), np.finfo(float).tiny)
        if np.linalg.norm(h - h.conj().T) > 1e-12 * scale:
            raise ValueError('Hamiltonian must be Hermitian')
        return comm_super(h) + self._dissipator

    def _drift(self, age):
        generator = self.generator(age)
        drift = self.basis.conj().T@generator@self.basis
        if np.linalg.norm(drift.imag) > 1e-11*max(np.linalg.norm(generator), 1e-300):
            raise ValueError('Hermitian atomic coordinates must give real drift')
        return generator, drift.real

    def coefficients(self, age, rho):
        generator, drift = self._drift(age)
        diffusion = np.einsum('rijab,ba->rij', self.jump_products, rho)
        return generator, drift, diffusion

    def state_at(self, age):
        age = float(age)
        self.path.position(age)
        index = min(int(np.searchsorted(self.times_s, age, side='right')-1), len(self._solutions)-1)
        return self._solutions[index].sol(age)[:self.n*self.n].reshape(self.n, self.n)

    def transition(self, later, earlier):
        if not 0 <= earlier <= later < len(self.times_s):
            raise ValueError('ordered sample indices required')
        result = np.eye(len(self.operators))
        # Exact semigroup composition R(t,s)=R(t,t_k)...R(t_1,s).
        # Never invert a long-time decaying fundamental matrix to obtain it.
        for index in range(earlier, later):
            result = self.interval_responses[index]@result
        return result

    def two_time_covariance(self, readout_operators, *, source=None, ordering='greater'):
        """Sampled cross-position atomic covariance; no delta(z-z') assumption."""
        if ordering not in ('greater', 'lesser'):
            raise ValueError('greater or lesser ordering required')
        local = self.covariance_by_source.sum(axis=0) if source is None else self.covariance_by_source[source]
        if ordering == 'lesser':
            local = local.swapaxes(-1, -2)
        c = [self.readout(readout_operators, t)[1] for t in self.times_s]
        count, ports = len(c), len(c[0])
        out = np.empty((count, count, ports, ports), complex)
        for j in range(count):
            for i in range(j+1):
                value = c[j]@self.transition(j, i)@local[i]@c[i].conj().T
                out[j, i], out[i, j] = value, value.conj().T
        return readonly_array(out)

    def readout(self, callback, age):
        ops = readonly_array(callback(age, self.path.position(age)))
        if ops.ndim != 3 or ops.shape[1:] != (self.n, self.n) or not len(ops):
            raise ValueError('nonempty readout operator rows required')
        return ops, np.einsum('jab,iba->ji', ops, self.operators)

    def retarded_kernel(self, readout_operators, drive_operators):
        """C(t) R(t,s) B(s) for sampled t>=s, zero for t<s.

        This is a causal atomic response kernel in age/position coordinates;
        no local Maxwell M or optical feedback closure is implicit. The diagonal
        is its right limit; continuum integration gives it zero measure.
        """
        c, b = [], []
        for age, rho in zip(self.times_s, self.states):
            c.append(self.readout(readout_operators, age)[1])
            v = readonly_array(drive_operators(age, self.path.position(age)))
            if v.ndim != 3 or v.shape[1:] != (self.n, self.n) or not len(v):
                raise ValueError('nonempty drive operator columns required')
            b.append(np.einsum('iab,jba->ij', self.operators, -1j*(v@rho-rho@v)))
        if any(x.shape != c[0].shape for x in c) or any(x.shape != b[0].shape for x in b):
            raise ValueError('fixed readout and drive counts required')
        out = np.zeros((len(c), len(c), c[0].shape[0], b[0].shape[1]), complex)
        for j in range(len(c)):
            for i in range(j+1):
                out[j, i] = c[j]@self.transition(j, i)@b[i]
        return readonly_array(out)

    def wavepacket(self, axis, readout_operators, *, drive_operators=None, rtol=None, atol=None):
        """Y(Omega)=integral exp(+i Omega a) O(a) da along one atomic life.

        Return connected ordered covariance, reservoir/inflow decomposition and
        retarded response to V_j(a)*epsilon_j*exp(-i Omega a). V can be complex;
        the conjugate physical drive, when needed, is a separate input column.
        """
        if not isinstance(axis, GeneratorFrequencyAxis):
            raise TypeError('explicit physical Fourier axis required')
        rtol, atol = self.rtol if rtol is None else rtol, self.atol if atol is None else atol
        if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
            raise ValueError('positive finite integration tolerances required')
        ports = len(self.readout(readout_operators, 0.)[0])
        def drives(age):
            if drive_operators is None:
                return np.zeros((0, self.n, self.n), complex)
            value = readonly_array(drive_operators(age, self.path.position(age)))
            if value.ndim != 3 or value.shape[1:] != (self.n, self.n):
                raise ValueError('drive operator columns must match atomic dimensions')
            return value
        inputs = len(drives(0.))
        nf, ns, m = len(axis.omega_rad_s), len(self.source_names), len(self.operators)
        omega = axis.omega_rad_s
        max_omega = float(np.max(np.abs(omega))) if nf else 0.0
        duration = self.path.residence_time_s
        # Solve backward adjoint propagator:
        #   -dPsi/dt = Psi A(t) + exp(+i*omega*t) * C(t) / duration,  Psi(T) = 0.
        # Duration scaling preserves conditioning on microsecond SI paths.
        # Every reservoir integral is a positive-weight Gram sum Psi D_r Psi^H >= 0.
        def rhs(age, y):
            psi = y.reshape(nf, ports, m)
            _, a = self._drift(age)
            ops, c = self.readout(readout_operators, age)
            if len(ops) != ports:
                raise ValueError('readout port count changed along characteristic')
            w = np.exp(1j * omega * age)[:, None, None] * c / duration
            return -(psi @ a + w).ravel()

        y = np.zeros(nf * ports * m, complex)
        evaluations = 0
        adjoint_odes = [None] * (len(self.times_s) - 1)
        for idx in range(len(self.times_s) - 2, -1, -1):
            lo, hi = float(self.times_s[idx]), float(self.times_s[idx + 1])
            ode = solve_ivp(
                rhs, (hi, lo), y, method='DOP853', dense_output=True,
                rtol=rtol, atol=atol, max_step=(hi - lo) / 4,
            )
            if not ode.success:
                raise ValueError('characteristic wavepacket propagation failed')
            adjoint_odes[idx] = ode
            y = ode.y[:, -1]
            evaluations += ode.nfev

        psi0 = y.reshape(nf, ports, m)
        _, initial = _moments(self.boundary_state, self.operators)
        greater = np.zeros((ns, nf, ports, ports), complex)
        lesser = np.zeros((ns, nf, ports, ports), complex)
        greater[0] = np.einsum('fpi,ij,fqj->fpq', psi0, initial, psi0.conj(), optimize=True) * duration**2
        lesser[0] = np.einsum('fpi,ij,fqj->fpq', psi0, initial.T, psi0.conj(), optimize=True) * duration**2
        mean = np.zeros((nf, ports), complex)
        response = np.zeros((nf, ports, inputs), complex)

        n2 = self.n * self.n
        for idx in range(len(self.times_s) - 1):
            lo, hi = float(self.times_s[idx]), float(self.times_s[idx + 1])
            adj_ode = adjoint_odes[idx]
            fwd_ode = self._solutions[idx]
            knots = np.union1d(adj_ode.t, fwd_ode.t)
            s_q, w_q = _panel_quadrature(lo, hi, knots, max_omega)
            nq = len(s_q)
            psi_q = adj_ode.sol(s_q).T.reshape(nq, nf, ports, m)
            rho_q = fwd_ode.sol(s_q)[:n2].T.reshape(nq, self.n, self.n)

            if ns > 1:
                diff_q = np.einsum('rijab,tba->trij', self.jump_products, rho_q, optimize=True)
                wpsi_conj = w_q[:, None, None, None] * psi_q.conj()
                tmp_g = np.einsum('tfpi,trij->trfpj', psi_q, diff_q, optimize=True)
                greater[1:] += np.einsum('trfpj,tfqj->rfpq', tmp_g, wpsi_conj, optimize=True) * duration**2
                tmp_l = np.einsum('tfpi,trji->trfpj', psi_q, diff_q, optimize=True)
                lesser[1:] += np.einsum('trfpj,tfqj->rfpq', tmp_l, wpsi_conj, optimize=True) * duration**2

            exp_p = np.exp(1j * np.outer(s_q, omega))
            ops_q = np.empty((nq, ports, self.n, self.n), complex)
            for k, age in enumerate(s_q):
                ops_k = readonly_array(readout_operators(float(age), self.path.position(float(age))))
                if ops_k.ndim != 3 or ops_k.shape[1:] != (self.n, self.n) or len(ops_k) != ports:
                    raise ValueError('readout port count changed along characteristic')
                ops_q[k] = ops_k
            tr_ops = np.einsum('qpab,qba->qp', ops_q, rho_q)
            mean += np.einsum('q,qf,qp->fp', w_q, exp_p, tr_ops, optimize=True)

            if inputs > 0:
                v_q = np.empty((nq, inputs, self.n, self.n), complex)
                for k, age in enumerate(s_q):
                    v_k = drives(float(age))
                    if len(v_k) != inputs:
                        raise ValueError('drive column count changed along characteristic')
                    v_q[k] = v_k
                comm_q = -1j * (np.einsum('qkac,qcb->qkab', v_q, rho_q)
                                - np.einsum('qac,qkcb->qkab', rho_q, v_q))
                b_q = np.einsum('iab,qkba->qik', self.operators, comm_q)
                response += np.einsum('q,qf,qfpi,qik->fpk', w_q, exp_p.conj(), psi_q, b_q, optimize=True) * duration

        # Adjoint Gram quadrature computes the full double-time kernel at the
        # ODE tolerance with manifest PSD: no sampled double integral or diagonal shortcut.
        audit = {name: _psd(value, rtol=1e-8) for name, value in (('greater', greater), ('lesser', lesser))}
        audit['passed'] = all(row['passed'] for row in audit.values())
        if not audit['passed']:
            raise ValueError('transport wavepacket ordered covariance failed PSD')
        return {'frequency_axis': axis, 'source_names': self.source_names,
            'greater_by_source': readonly_array(greater), 'lesser_by_source': readonly_array(lesser),
            'greater': readonly_array(greater.sum(axis=0)), 'lesser': readonly_array(lesser.sum(axis=0)),
            'mean_pulse': readonly_array(mean), 'retarded_response': readonly_array(response),
            'residence_time_s': self.path.residence_time_s, 'audit': audit, 'ode_evaluations': evaluations,
            'scope': 'finite atomic wavepacket moments/response, not canonical field M/D or a stationary PSD'}


def integrate_characteristic(path, hamiltonian, reservoirs, boundary_state, *, times_s=None,
                             rtol=2e-10, atol=2e-12):
    if not isinstance(path, BallisticPath):
        raise TypeError('explicit ballistic path required')
    times = np.linspace(0., path.residence_time_s, 17) if times_s is None else times_s
    result = AtomicCharacteristic(path, hamiltonian, reservoirs, boundary_state, times, rtol, atol)
    n, m, ns = result.n, len(result.operators), len(result.source_names)
    _, initial = _moments(result.boundary_state, result.operators)
    covariance = np.zeros((ns, m, m), complex)
    covariance[0] = initial
    state = result.boundary_state
    states, contributions, intervals = [state], [covariance], []
    def rhs(age, y):
        rho = y[:n*n].reshape(n, n)
        cov = y[n*n:n*n+ns*m*m].reshape(ns, m, m)
        transition = y[n*n+ns*m*m:].reshape(m, m)
        generator, a, d = result.coefficients(age, rho)
        dc = a@cov+cov@a.T
        dc[1:] += d
        return np.r_[generator@rho.ravel(), dc.ravel(), (a@transition).ravel()]
    for lo, hi in zip(result.times_s[:-1], result.times_s[1:]):
        y = np.r_[state.ravel(), covariance.ravel(), np.eye(m).ravel()]
        ode = solve_ivp(rhs, (lo, hi), y, method='DOP853', dense_output=True,
            rtol=rtol, atol=atol, max_step=(hi-lo)/4)
        if not ode.success:
            raise ValueError('characteristic state/covariance propagation failed')
        result._solutions.append(ode)
        state = ode.y[:n*n, -1].reshape(n, n)
        covariance = ode.y[n*n:n*n+ns*m*m, -1].reshape(ns, m, m)
        states.append(state); contributions.append(covariance)
        intervals.append(ode.y[n*n+ns*m*m:, -1].reshape(m, m))
    result.states = readonly_array(states)
    result.covariance_by_source = readonly_array(np.swapaxes(contributions, 0, 1))
    result.interval_responses = readonly_array(intervals)
    exact = np.array([_moments(_state(rho, n), result.operators)[1] for rho in states])
    evolved = result.covariance_by_source.sum(axis=0)
    scale = max(float(np.linalg.norm(exact)), 1e-300)
    audit = {'covariance_relative_residual': float(np.linalg.norm(evolved-exact)/scale),
        'atomic_commutator_relative_residual': float(np.linalg.norm(
            evolved-evolved.swapaxes(-1, -2)-exact+exact.swapaxes(-1, -2))/scale),
        'source_covariance': _psd(result.covariance_by_source, rtol=1e-8),
        'minimum_state_eigenvalue': float(np.linalg.eigvalsh(result.states).min())}
    audit['passed'] = bool(audit['source_covariance']['passed'] and audit['covariance_relative_residual'] < 1e-8
        and audit['atomic_commutator_relative_residual'] < 1e-8 and audit['minimum_state_eigenvalue'] >= -1e-10)
    result.audit = audit
    if not audit['passed']:
        raise ValueError('inflow plus jump covariance failed characteristic moment identity')
    return result


def poisson_beam_spectrum(wavepacket, arrival_rate_s_inverse, *, source):
    """Campbell spectrum of independent identical atom pulses at Poisson times.

    S=rate*(connected single-atom spectrum + mean_pulse mean_pulse†).
    The second term is fixed by Poisson inflow statistics, not an adjustable
    excess-noise factor. Lab-time periodic drives need entry-phase averaging
    before this reduction; identical age protocols are an explicit assumption.
    """
    rate = float(arrival_rate_s_inverse)
    if not np.isfinite(rate) or rate < 0 or not isinstance(source, str) or not source.strip():
        raise ValueError('finite nonnegative Poisson arrival rate and provenance required')
    mean = wavepacket['mean_pulse']
    number = rate*mean[:, :, None]*mean[:, None, :].conj()
    return {'greater': readonly_array(rate*wavepacket['greater']+number),
        'lesser': readonly_array(rate*wavepacket['lesser']+number),
        'internal_greater': readonly_array(rate*wavepacket['greater']),
        'internal_lesser': readonly_array(rate*wavepacket['lesser']), 'poisson_number': readonly_array(number),
        'arrival_rate_s_inverse': rate, 'mean_occupancy': rate*wavepacket['residence_time_s'], 'source': source,
        'scope': 'stationary independent Poisson atom pulses with identical age protocol; no optical SQL normalization'}
