"""Exact finite-cutoff symmetry: independent dense oracle and guarded fallback."""

import numpy as np
import pytest

from gabes import core, kernels


@pytest.fixture(autouse=True)
def single_blas_thread():
    with core.blas_single_thread():
        yield


def physical_grid(n=3, n_f=3, branch=-1):
    rng = np.random.default_rng(42+n)
    z = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    h0 = (z+z.conj().T)/2
    drive = .4*(rng.normal(size=(n, n))+1j*rng.normal(size=(n, n)))
    trace = np.eye(n).reshape(-1)
    populations = np.arange(1., n+1)
    target = np.diag(populations/populations.sum()).reshape(-1)
    reset = .7*(np.outer(target, trace)-np.eye(n*n))
    return [core.comm_super(h0)+reset, core.comm_super(np.diag(np.arange(n))),
        core.comm_super(np.diag(np.linspace(0., .9, n))),
        core.comm_super(drive), core.comm_super(drive.conj().T),
        np.array([-.8, .2, 1.7]), np.array([-3.1, .7]), 9.3, branch,
        rng.normal(size=n*n)+1j*rng.normal(size=n*n),
        rng.normal(size=n*n)+1j*rng.normal(size=n*n), n, n_f]


@pytest.mark.parametrize('n,n_f,branch', [(2, 1, -1), (2, 4, 1), (3, 2, -1),
    (3, 3, 1), (4, 1, 1), (4, 3, -1), (4, 5, 1)])
def test_symmetry_kernel_matches_full_density_block_at_finite_cutoff(n, n_f, branch):
    args = physical_grid(n, n_f, branch)
    actual = kernels.floquet_chi_grid(*args)
    l0, cd, sv, cp, cm, deltas, deff, whf, br, wp, wc, _, _ = args
    # A conjugate without the raw vec transpose permutation is NOT the identity.
    assert np.max(abs(cm-cp.conj())) > .1
    for i, delta in enumerate(deltas):
        modes = core.floquet_solve_direct(l0+delta*cd, cp, cm, whf+br*delta,
            deff, sv, n, n_f, return_harmonics=True)
        expected = (modes[:, n_f].reshape(len(deff), -1)@wp,
                    modes[:, n_f+1].reshape(len(deff), -1)@wc)
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value[i], reference, rtol=1e-11, atol=1e-12)
        diagnostics = core.floquet_solution_diagnostics(l0+delta*cd, cp, cm,
            whf+br*delta, deff, sv, modes)
        assert diagnostics['max_normalized_residual'] < 1e-11


@pytest.mark.parametrize('changed', [0, 1, 2, 4, 7])
def test_broken_symmetry_or_complex_frequency_uses_general_solver(monkeypatch, changed):
    args = physical_grid()
    if changed == 7:
        args[changed] += .03j
    else:
        # Even a one-ULP defect must not be called mathematically zero.
        old = args[changed][0, 1]
        args[changed][0, 1] = np.nextafter(old.real, np.inf)+1j*old.imag
    expected = kernels._floquet_chi_grid_general(*args)[:2]
    def forbidden(*args):
        raise AssertionError('non-symmetric input reached the exact reduction')
    monkeypatch.setattr(kernels, '_floquet_chi_grid_hermitian', forbidden)
    actual = kernels.floquet_chi_grid(*args)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not kernels.available(), reason='Numba call-count instrumentation')
def test_exact_reduction_requires_only_nf_complex_and_one_real_factorization(monkeypatch):
    args = physical_grid(n=2, n_f=4)
    expected = kernels._floquet_chi_grid_general(*args)[:2]
    counts = {'complex': 0, 'real': 0}
    complex_factor, real_factor = kernels._lu_factor, kernels._lu_factor_real
    def count_complex(a, p):
        counts['complex'] += 1
        return complex_factor(a, p)
    def count_real(a, p):
        counts['real'] += 1
        return real_factor(a, p)
    monkeypatch.setattr(kernels, '_lu_factor', count_complex)
    monkeypatch.setattr(kernels, '_lu_factor_real', count_real)
    monkeypatch.setattr(kernels, '_floquet_chi_grid_hermitian', kernels._floquet_chi_grid_hermitian.py_func)
    actual = kernels.floquet_chi_grid(*args)
    points = len(args[5])*len(args[6])
    assert counts == {'complex': points*args[-1], 'real': points}
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def test_wrapper_preserves_readonly_inputs_and_invalid_order_error():
    args = physical_grid()
    saved = []
    for value in args:
        if isinstance(value, np.ndarray):
            saved.append(value.copy())
            value.setflags(write=False)
    kernels.floquet_chi_grid(*args)
    for before, after in zip(saved, (a for a in args if isinstance(a, np.ndarray))):
        np.testing.assert_array_equal(before, after)
    args[-1] = 0
    with pytest.raises(ValueError, match='n_f'):
        kernels.floquet_chi_grid(*args)


@pytest.mark.parametrize("invalid", [0.0, np.nan, np.inf])
def test_lu_failure_reports_floquet_point_for_compiled_and_reference(invalid):
    # At deff[1] the zero harmonic loses its independent constraints. The
    # nonzero harmonics are invertible when the entire input is finite.
    matrix = -np.eye(4, dtype=complex)
    matrix[1, 1] = matrix[2, 2] = invalid
    shift = -np.eye(4, dtype=complex)
    shift[1, 1] = shift[2, 2] = -1.0
    zero = np.zeros_like(matrix)
    deff = np.array([0.5, 0.])
    readout = np.ones(4)
    reason = "singular" if invalid == 0.0 else "non-finite"
    failing_index = 1 if invalid == 0.0 else 0
    point = rf"deff\[{failing_index}\]"

    with pytest.raises(np.linalg.LinAlgError, match=rf"floquet_chi_grid.*{reason}.*delta\[0\].*{point}"):
        kernels.floquet_chi_grid(matrix, zero, shift, zero, zero, np.array([0.]),
                                 deff, 3., 1., readout, readout, 2)
    with pytest.raises(np.linalg.LinAlgError, match=rf"floquet_solve.*{reason}.*{point}"):
        core.floquet_solve_truncated(matrix, zero, zero, 3., deff, shift, 2)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("scale", [1e-20, 1., 1e20])
@pytest.mark.parametrize("damping", [1.0, 1e-9])
def test_checked_lu_preserves_solutions_across_scales(dtype, scale, damping):
    # A narrow but valid response has a small pivot relative to the other
    # scale in the matrix. It must retain its linewidth and amplitude.
    matrix = scale * np.array([[1e-7 * damping, 1.0],
                               [2.0 * damping, 3.0]], dtype=dtype)
    rhs = scale * np.array([[1.0], [4.0]], dtype=dtype)
    expected = core._solve_checked(matrix, rhs, "test")
    factored, actual = matrix.copy(), rhs.copy()
    pivots = np.empty(2, dtype=np.int64)
    factor, solve = ((kernels._lu_factor_real, kernels._lu_solve_real)
                     if dtype is np.float64 else (kernels._lu_factor, kernels._lu_solve))

    assert factor(factored, pivots) == kernels._LU_OK
    assert solve(factored, pivots, actual) == kernels._LU_OK
    assert np.linalg.cond(matrix) < 10.0 / damping
    assert np.linalg.norm(matrix @ actual - rhs) / np.linalg.norm(rhs) < 1e-13
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("n,n_f", [(2, 1), (3, 2), (4, 2), (4, 4)])
def test_floquet_solve_truncated_conjugate_symmetry_matches_direct_and_halves_solves(monkeypatch, n, n_f):
    l0, _, sv, cp, cm, _, deff, whf, _, _, _, _, _ = physical_grid(n=n, n_f=n_f)
    assert core._is_conjugate_symmetric_floquet(l0, cp, cm, whf, deff, sv, n)

    solve_calls = []
    real_solve = core._solve_checked
    def counting_solve(matrix, rhs, kernel, **axes):
        solve_calls.append(kernel)
        return real_solve(matrix, rhs, kernel, **axes)
    monkeypatch.setattr(core, "_solve_checked", counting_solve)

    harmonics = core.floquet_solve_truncated(
        l0, cp, cm, whf, deff, sv, n, n_f, return_harmonics=True)
    # Only positive chain (n_f solves) + zero harmonic (1 solve)
    assert len(solve_calls) == n_f + 1

    direct = core.floquet_solve_direct(
        l0, cp, cm, whf, deff, sv, n, n_f, return_harmonics=True)
    np.testing.assert_allclose(harmonics, direct, rtol=1e-12, atol=1e-13)

    # Signed harmonic ordering -n_f..+n_f obeys rho[-h] == rho[+h]^H
    for h in range(1, n_f + 1):
        np.testing.assert_allclose(
            harmonics[:, n_f - h], harmonics[:, n_f + h].conj().swapaxes(-1, -2),
            rtol=0.0, atol=1e-15)

    diag = core.floquet_solution_diagnostics(l0, cp, cm, whf, deff, sv, harmonics)
    assert diag["max_normalized_residual"] < 1e-12
    assert diag["max_trace_error"] < 1e-13
    assert diag["max_nonzero_harmonic_trace"] < 1e-13
    assert diag["max_rho0_hermiticity_error"] < 1e-13

    rho0, rho1 = core.floquet_solve_truncated(
        l0, cp, cm, whf, deff, sv, n, n_f, return_harmonics=False)
    np.testing.assert_array_equal(rho0, harmonics[:, n_f])
    np.testing.assert_array_equal(rho1, harmonics[:, n_f + 1])


@pytest.mark.parametrize("defect_target", ["L0", "S_v", "Cm"])
def test_floquet_solve_truncated_one_ulp_defect_uses_dual_chain(monkeypatch, defect_target):
    l0, _, sv, cp, cm, _, deff, whf, _, _, _, _, _ = physical_grid(n=3, n_f=2)
    l0, sv, cm = l0.copy(), sv.copy(), cm.copy()
    target = {"L0": l0, "S_v": sv, "Cm": cm}[defect_target]
    target[0, 1] = np.nextafter(target[0, 1].real, np.inf) + 1j * target[0, 1].imag
    assert not core._is_conjugate_symmetric_floquet(l0, cp, cm, whf, deff, sv, 3)

    solve_calls = []
    real_solve = core._solve_checked
    def counting_solve(matrix, rhs, kernel, **axes):
        solve_calls.append(kernel)
        return real_solve(matrix, rhs, kernel, **axes)
    monkeypatch.setattr(core, "_solve_checked", counting_solve)

    harmonics = core.floquet_solve_truncated(
        l0, cp, cm, whf, deff, sv, 3, 2, return_harmonics=True)
    # Full dual chain: 2*n_f + 1 = 5 solves
    assert len(solve_calls) == 5
    direct = core.floquet_solve_direct(
        l0, cp, cm, whf, deff, sv, 3, 2, return_harmonics=True)
    np.testing.assert_allclose(harmonics, direct, rtol=1e-12, atol=1e-13)
