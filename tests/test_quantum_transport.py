"""Tests for gabes.quantum.transport adjoint wavepacket factorization (Trick C)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gabes.quantum.contracts import GeneratorFrequencyAxis
from gabes.quantum.reservoirs import CollapseChannel, ExplicitReservoirs, thermal_reset_channels
from gabes.quantum.transport import (
    BallisticPath,
    _moments,
    integrate_characteristic,
    poisson_beam_spectrum,
)


def _reference_augmented_wavepacket(char, axis, readout_operators, *, drive_operators=None, rtol=1e-12, atol=1e-14):
    """Independent reference implementation via the forward augmented Lyapunov ODE."""
    ports = len(char.readout(readout_operators, 0.0)[0])
    def drives(age):
        if drive_operators is None:
            return np.zeros((0, char.n, char.n), complex)
        return np.asarray(drive_operators(age, char.path.position(age)), complex)
    inputs = len(drives(0.0))
    nf, ns, m = len(axis.omega_rad_s), len(char.source_names), len(char.operators)
    duration = char.path.residence_time_s
    dim = m + ports
    shape = (2, ns, nf, dim, dim)
    cov = np.zeros(shape, complex)
    _, initial = _moments(char.boundary_state, char.operators)
    cov[0, 0, :, :m, :m], cov[1, 0, :, :m, :m] = initial, initial.T
    lengths = [cov.size, nf * ports, nf * m * inputs, nf * ports * inputs]
    offsets = np.r_[0, np.cumsum(lengths)]
    y0 = np.r_[cov.ravel(), np.zeros(sum(lengths[1:]), complex)]

    def unpack(y):
        return (
            y[offsets[0]:offsets[1]].reshape(shape),
            y[offsets[1]:offsets[2]].reshape(nf, ports),
            y[offsets[2]:offsets[3]].reshape(nf, m, inputs),
            y[offsets[3]:offsets[4]].reshape(nf, ports, inputs),
        )

    def rhs(age, y):
        p, _, response, _ = unpack(y)
        rho = char.state_at(age)
        _, a, d = char.coefficients(age, rho)
        ops, c = char.readout(readout_operators, age)
        w = np.exp(1j * axis.omega_rad_s * age)[:, None, None] * c / duration
        aug = np.zeros((nf, dim, dim), complex)
        aug[:, :m, :m], aug[:, m:, :m] = a, w
        pdot = aug @ p + p @ aug.conj().swapaxes(-1, -2)
        pdot[0, 1:, :, :m, :m] += d[:, None]
        pdot[1, 1:, :, :m, :m] += d.swapaxes(-1, -2)[:, None]
        mean = np.exp(1j * axis.omega_rad_s * age)[:, None] * np.einsum("jab,ba->j", ops, rho) / duration
        v = drives(age)
        b = np.einsum("iab,jba->ij", char.operators, -1j * (v @ rho - rho @ v))
        resp_dot = (a + 1j * axis.omega_rad_s[:, None, None] * np.eye(m)) @ response + b
        out_dot = c @ response / duration
        return np.r_[pdot.ravel(), mean.ravel(), resp_dot.ravel(), out_dot.ravel()]

    y = y0
    for lo, hi in zip(char.times_s[:-1], char.times_s[1:]):
        ode = solve_ivp(rhs, (lo, hi), y, method="DOP853", rtol=rtol, atol=atol, max_step=(hi - lo) / 4)
        assert ode.success
        y = ode.y[:, -1]
    cov_out, mean, _, resp = unpack(y)
    greater, lesser = cov_out[:, :, :, m:, m:] * duration**2
    return {
        "greater_by_source": greater,
        "lesser_by_source": lesser,
        "mean_pulse": mean * duration,
        "retarded_response": resp * duration,
    }


def _two_level_fixture():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], complex)
    sy = np.array([[0.0, -1j], [1j, 0.0]], complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], complex)
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], complex)
    sp = sm.conj().T
    res = ExplicitReservoirs(2, (
        CollapseChannel("decay", np.sqrt(1.2) * sm, "radiative", "test"),
        CollapseChannel("deph", np.sqrt(0.4) * sz, "collision", "test"),
    ))
    path = BallisticPath([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.5, "two_level")
    ham = lambda a, r: 0.3 * sz + (0.8 + 0.5 * np.sin(2.5 * a)) * sx + 0.2 * np.cos(1.7 * a) * sy
    rho0 = np.array([[0.7, 0.1 - 0.05j], [0.1 + 0.05j, 0.3]], complex)
    read = lambda a, r: np.array([sm * np.exp(-0.2j * a), sp + 0.3 * sz])
    drv = lambda a, r: np.array([sm * (1.0 + 0.2 * a), sp * np.exp(0.3j * a)])
    return path, ham, res, rho0, read, drv


def test_wavepacket_adjoint_matches_augmented_reference_within_1e12():
    path, ham, res, rho0, read, drv = _two_level_fixture()
    char = integrate_characteristic(path, ham, res, rho0, times_s=np.linspace(0.0, 1.5, 9), rtol=1e-12, atol=1e-14)
    axis = GeneratorFrequencyAxis(np.array([-2.3, -0.7, 0.0, 0.7, 2.3]), "signed")
    wp = char.wavepacket(axis, read, drive_operators=drv, rtol=1e-12, atol=1e-14)
    ref = _reference_augmented_wavepacket(char, axis, read, drive_operators=drv, rtol=1e-12, atol=1e-14)

    for key in ("greater_by_source", "lesser_by_source", "mean_pulse", "retarded_response"):
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        rel = float(np.max(np.abs(wp[key] - ref[key]))) / scale
        assert rel <= 1e-12, f"{key} rel={rel:.3e} > 1e-12"

    for key in ("greater_by_source", "lesser_by_source"):
        mat = wp[key]
        herm = 0.5 * (mat + mat.conj().swapaxes(-1, -2))
        assert float(np.linalg.eigvalsh(herm).min()) >= -1e-14 * float(np.max(np.abs(mat)))
    assert wp["audit"]["passed"]


def test_wavepacket_microsecond_si_duration_conditioning_and_nodrive_poisson():
    g_rad = 2 * np.pi * 5.75e6
    duration = 1.2e-6
    j02 = np.zeros((3, 3), complex); j02[0, 2] = np.sqrt(0.6 * g_rad)
    j12 = np.zeros((3, 3), complex); j12[1, 2] = np.sqrt(0.4 * g_rad)
    jd = np.diag([1.0, -1.0, 0.0]) * np.sqrt(2 * np.pi * 0.2e6)
    res = ExplicitReservoirs(3, (
        CollapseChannel("rad0", j02, "radiative", "test"),
        CollapseChannel("rad1", j12, "radiative", "test"),
        CollapseChannel("deph", jd, "collision", "test"),
    ) + tuple(thermal_reset_channels(1.5e5, [0.45, 0.55, 0.0], source="test")))
    path = BallisticPath([0.0, 0.0, 0.0], [200.0, 0.0, 0.0], duration, "us_3lvl")

    def ham(a, r):
        env = np.exp(-((a - 0.5 * duration) / (0.35 * duration)) ** 2)
        h = np.zeros((3, 3), complex)
        h[1, 1] = 2 * np.pi * 1.2e6
        h[2, 2] = 2 * np.pi * 4.0e6
        h[0, 2] = h[2, 0] = 2 * np.pi * 3.5e6 * env
        h[1, 2] = 2 * np.pi * 1.8e6 * env * np.exp(0.4j)
        h[2, 1] = h[1, 2].conj()
        return h

    rho0 = np.diag([0.45, 0.55, 0.0]).astype(complex)
    char = integrate_characteristic(path, ham, res, rho0, times_s=np.linspace(0.0, duration, 9), rtol=1e-12, atol=1e-14)
    axis = GeneratorFrequencyAxis(2 * np.pi * np.array([-2e6, 0.0, 2e6]), "us_signed")
    p02 = np.zeros((3, 3), complex); p02[0, 2] = 1.0
    p12 = np.zeros((3, 3), complex); p12[1, 2] = 1.0
    read = lambda a, r: np.array([p02, p12])
    drv = lambda a, r: np.array([p02.conj().T * 1e6, p12.conj().T * 1e6])

    wp = char.wavepacket(axis, read, drive_operators=drv, rtol=1e-12, atol=1e-14)
    ref = _reference_augmented_wavepacket(char, axis, read, drive_operators=drv, rtol=1e-12, atol=1e-14)
    for key in ("greater_by_source", "lesser_by_source", "mean_pulse", "retarded_response"):
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        rel = float(np.max(np.abs(wp[key] - ref[key]))) / scale
        assert rel <= 1e-12, f"{key} rel={rel:.3e} > 1e-12"

    wp_nd = char.wavepacket(axis, read, drive_operators=None, rtol=1e-12, atol=1e-14)
    assert wp_nd["retarded_response"].shape == (3, 2, 0)
    np.testing.assert_allclose(wp_nd["greater"], wp["greater"], rtol=1e-13, atol=1e-30)
    pb = poisson_beam_spectrum(wp_nd, 1e4, source="test_poisson")
    assert pb["greater"].shape == (3, 2, 2)
    assert float(np.linalg.eigvalsh(0.5 * (pb["greater"] + pb["greater"].conj().swapaxes(-1, -2))).min()) >= -1e-25


def test_wavepacket_high_frequency_oscillation_matches_reference():
    path, ham, res, rho0, read, drv = _two_level_fixture()
    char = integrate_characteristic(path, ham, res, rho0, times_s=np.linspace(0.0, 1.5, 9), rtol=1e-12, atol=1e-14)
    axis = GeneratorFrequencyAxis(np.array([-30.0, 0.0, 30.0]), "high_omega")
    wp = char.wavepacket(axis, read, drive_operators=drv, rtol=1e-12, atol=1e-14)
    ref = _reference_augmented_wavepacket(char, axis, read, drive_operators=drv, rtol=1e-12, atol=1e-14)
    for key in ("greater_by_source", "lesser_by_source", "mean_pulse", "retarded_response"):
        scale = max(float(np.max(np.abs(ref[key]))), 1e-300)
        rel = float(np.max(np.abs(wp[key] - ref[key]))) / scale
        assert rel <= 1e-12, f"{key} rel={rel:.3e} > 1e-12"


def test_wavepacket_error_guards_and_hermiticity():
    path, ham, res, rho0, read, drv = _two_level_fixture()
    char = integrate_characteristic(path, ham, res, rho0, times_s=np.linspace(0.0, 1.5, 5), rtol=1e-10, atol=1e-12)
    axis = GeneratorFrequencyAxis(np.array([-1.0, 0.0, 1.0]), "guard")
    with pytest.raises(TypeError):
        char.wavepacket([0.0], read)
    with pytest.raises(ValueError):
        char.wavepacket(axis, read, rtol=-1e-8)

    bad_read = lambda a, r: read(a, r) if a == 0.0 else read(a, r)[:1]
    with pytest.raises(ValueError):
        char.wavepacket(axis, bad_read)

    bad_drv = lambda a, r: drv(a, r) if a == 0.0 else drv(a, r)[:1]
    with pytest.raises(ValueError):
        char.wavepacket(axis, read, drive_operators=bad_drv)

    wp = char.wavepacket(axis, read, drive_operators=drv)
    np.testing.assert_allclose(wp["greater"], wp["greater_by_source"].sum(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(wp["lesser"], wp["lesser_by_source"].sum(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(wp["greater"], wp["greater"].conj().swapaxes(-1, -2), atol=1e-14)
    np.testing.assert_allclose(wp["lesser"], wp["lesser"].conj().swapaxes(-1, -2), atol=1e-14)
