"""Reference parity for the periodic Floquet Schur resolvent."""
import numpy as np
import pytest

from gabes import core
from gabes.quantum import periodic
from gabes.quantum.contracts import GeneratorFrequencyAxis
from gabes.quantum.periodic import FloquetAtomicLift, _FloquetResolvent, periodic_atomic_noise
from gabes.quantum.periodic_field import PeriodicFieldPorts, eliminate_periodic_atom, periodic_field_couplings
from gabes.quantum.reservoirs import CollapseChannel, ExplicitReservoirs


def periodic_fixture():
    h = np.array([[0., 0., .8, 0.],
                  [0., .2, 0., .6],
                  [.8, 0., -.5, 0.],
                  [0., .6, 0., -.3]], complex)
    v = np.zeros((4, 4), complex)
    v[0, 3], v[1, 2] = .15, .12
    jumps = []
    for i, j, rate in ((0, 2, .5), (1, 2, .5), (0, 3, .5), (1, 3, .5), (0, 1, .05), (1, 0, .05)):
        op = np.zeros_like(h)
        op[i, j] = np.sqrt(rate)
        jumps.append(CollapseChannel(f"j{i}{j}", op, "decay", "Floquet test"))
    atom = periodic_atomic_noise(h, v, ExplicitReservoirs(4, tuple(jumps)), 3., mean_order=4)
    ports = PeriodicFieldPorts(("probe", "conjugate"), [1, -1],
                               [jumps[0].operator, jumps[3].operator.conj().T], [.7, 1.])
    return atom, ports


def assert_norm_close(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.max(abs(expected), axis=(-2, -1))
    error = np.max(abs(actual-expected), axis=(-2, -1))
    assert np.all(error <= 1e-12*scale)


@pytest.mark.parametrize("order,count", [(3, 1), (3, 25), (6, 41)])
def test_full_and_projected_sweeps_match_direct(order, count):
    with core.blas_single_thread():
        atom, ports = periodic_fixture()
        axis = GeneratorFrequencyAxis(np.linspace(-1.2, 1.2, count), "Floquet test")
        lift, b, c, _, _ = periodic_field_couplings(atom, ports, response_order=order)
        spectrum = lift.spectrum(axis)
        local = eliminate_periodic_atom(atom, ports, axis, response_order=order,
                                        linear_density_m_inverse=1.2)
        eye = np.eye(len(lift.drift))
        r = np.array([np.linalg.solve(-1j*w*eye-lift.drift, eye) for w in axis.omega_rad_s])
        cr = c @ r
        assert_norm_close(local.drift, 1.2*cr @ b)
        for field, diffusion, projected in (
                (spectrum.greater, lift.greater_by_reservoir, local.noise_greater_by_reservoir),
                (spectrum.lesser, lift.lesser_by_reservoir, local.noise_lesser_by_reservoir)):
            assert_norm_close(field, r @ diffusion.sum(axis=0) @ r.conj().swapaxes(-1, -2))
            assert_norm_close(projected, 1.2*(cr[None] @ diffusion[:, None]
                                                @ cr[None].conj().swapaxes(-1, -2)))
        assert spectrum.audit()["passed"]
        assert local.audit()["passed"]


def test_narrow_resonance_uses_direct_and_audit_remains_fail_closed():
    drift = np.array([[-1e-11+.27j, .4], [0., -1.2+.1j]])
    diffusion = np.eye(2)[None]
    lift = FloquetAtomicLift(2., [0], drift, diffusion, diffusion, ("bath",), 2)
    frequencies = GeneratorFrequencyAxis([-.28, -.27, -.26, .1], "narrow resonance")
    actual = lift.spectrum(frequencies)
    reference = np.linalg.solve(-1j*frequencies.omega_rad_s[1]*np.eye(2)-drift, np.eye(2))
    np.testing.assert_array_equal(actual.greater[1], reference @ reference.conj().T)
    c = np.array([[2.+.4j, .5]])
    (result,) = _FloquetResolvent(drift, use_schur=True).sweep([-.27], projection=c)
    np.testing.assert_array_equal(result, np.linalg.solve((-.27*(-1j)*np.eye(2)-drift).T, c.T).T)
    # With the resonance mixed into both physical coordinates, the residual
    # itself loses precision well before the spectral-gap cutoff.
    rotation = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    mixed = rotation @ (drift-1e-5*np.eye(2)) @ rotation.conj().T
    schur = _FloquetResolvent(mixed, use_schur=True)
    direct = _FloquetResolvent(mixed, use_schur=False)
    for projection in (None, c):
        (approx,) = schur.sweep([-.27], projection=projection)
        (exact,) = direct.sweep([-.27], projection=projection)
        np.testing.assert_array_equal(approx, exact)
    # Returned audit dictionaries are independent of the cached audit.
    lift.audit()["greater"]["passed"] = False
    assert lift.audit()["passed"]
    for unstable, noise in ((drift+1e-4*np.eye(2), diffusion), (drift, -diffusion)):
        invalid = FloquetAtomicLift(2., [0], unstable, noise, noise, ("bath",), 2)
        with pytest.raises(ValueError, match="unstable or nonpositive"):
            invalid.spectrum(frequencies)

@pytest.mark.parametrize("projected", [False, True])
def test_moderately_narrow_resonance_exercises_schur_refinement(monkeypatch, projected):
    rotation = np.array([[1., 1j], [1j, 1.]])/np.sqrt(2.)
    drift = rotation @ np.array([[-.003+.27j, .4], [0., -1.2+.1j]]) @ rotation.conj().T
    c = np.array([[2.+.4j, .5]]) if projected else None
    old_solve = periodic.solve_triangular
    calls = []

    def count_solve(*args, **kwargs):
        calls.append(1)
        return old_solve(*args, **kwargs)

    monkeypatch.setattr(periodic, "solve_triangular", count_solve)
    actual, = _FloquetResolvent(drift, use_schur=True).sweep([-.27], projection=c)
    expected, = _FloquetResolvent(drift, use_schur=False).sweep([-.27], projection=c)
    assert len(calls) == 2
    assert_norm_close(actual, expected)


def test_nonnormal_lift_replays_even_far_from_eigenvalues():
    n = 12
    rng = np.random.default_rng(3)
    rotation, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))
    triangle = np.triu(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)), 1)
    triangle[np.diag_indices(n)] = -np.linspace(.35, 1., n) + 1j*np.linspace(-.12, .18, n)
    drift = rotation @ triangle @ rotation.conj().T
    assert np.linalg.cond(drift) > 1e4
    for projection in (None, np.eye(n, dtype=complex)[:2]):
        schur = _FloquetResolvent(drift, use_schur=True)
        assert np.min(abs(schur.poles))/schur.scale > .01
        actual, = schur.sweep([0.], projection=projection)
        expected, = _FloquetResolvent(drift, use_schur=False).sweep([0.], projection=projection)
        np.testing.assert_array_equal(actual, expected)
