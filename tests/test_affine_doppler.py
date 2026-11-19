"""Affine-scan Doppler averages: exact Maxwell mean, resolving grids and per-class solves agree."""
import numpy as np
import pytest

from gabes import affine_doppler, atoms, constants, core, doppler, kernels, pole_doppler, schemes
from gabes.affine_doppler import AffineScan

T, MASS, K = 330.0, constants.MASS_85RB, constants.K_VEC
SIGMA = K * np.sqrt(constants.KB * T / MASS)


def _lambda_scan():
    atom = atoms.lambda3(two_photon_doppler_ratio=0.02)
    probe, coupling = 1e-3 * constants.GAMMA, 0.4 * constants.GAMMA

    def build_H0(s):
        H = np.zeros((3, 3), dtype=complex)
        H[1, 1] = -s
        H[0, 2] = H[2, 0] = probe / 2
        H[1, 2] = H[2, 1] = coupling / 2
        return H

    return atom, build_H0, probe


def _maxwell_scan():
    atom, build_H0, probe = _lambda_scan()
    affine = AffineScan(atom, build_H0, (2, 0), probe, atom.S_v.copy())
    return affine, np.linspace(-3, 3, 41) * constants.GAMMA


def test_maxwell_strip_grid_covers_the_bulk_at_the_quadrature_step():
    strip = 0.2 * constants.GAMMA
    v, w = doppler.maxwell_strip_grid(T, MASS, K, strip, error=1e-6, cutoff_sigma=6.0)
    step = v[1] - v[0]
    np.testing.assert_allclose(K * step, 2 * np.pi * strip / np.log(1e6), rtol=1e-12)
    assert np.isclose(w.sum(), 1.0)
    assert v[-1] >= 6.0 * SIGMA / K
    np.testing.assert_allclose(np.sum(w * (K * v) ** 2), SIGMA ** 2, rtol=1e-6)


def test_class_sum_is_the_weighted_per_class_steady_state():
    atom, build_H0, probe = _lambda_scan()
    scan = np.linspace(-2, 2, 9) * constants.GAMMA
    kv = np.array([-3.0, 0.0, 1.5]) * constants.GAMMA
    w = np.array([0.2, 0.5, 0.3])
    shift = atom.S_v.copy()
    got = AffineScan(atom, build_H0, (2, 0), probe, shift)._class_sum(scan, kv, w)
    expected = np.array([
        sum(wj * core.steady_state_batched(
            core.build_liouvillian(build_H0(s), atom) - s * shift + kj * atom.S_v,
            np.zeros(1), atom.S_v, 3)[0, 2, 0] for kj, wj in zip(kv, w)) / probe
        for s in scan])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_stationary_matches_the_single_zero_class():
    atom, build_H0, probe = _lambda_scan()
    scan = np.linspace(-1, 1, 7) * constants.GAMMA
    affine = AffineScan(atom, build_H0, (2, 0), probe, atom.S_v)
    np.testing.assert_allclose(affine.stationary(scan),
                               affine._class_sum(scan, np.zeros(1), np.ones(1)),
                               rtol=1e-10, atol=1e-13)
    if kernels.available():   # the kernel path is the same call for both
        np.testing.assert_array_equal(affine.stationary(scan),
                                      affine._class_sum(scan, np.zeros(1), np.ones(1)))


def test_stationary_with_the_scan_in_the_hamiltonian_is_the_single_zero_class():
    atom, build_H0, probe = _lambda_scan()
    scan = np.linspace(-1, 1, 7) * constants.GAMMA
    affine = AffineScan(atom, build_H0, (2, 0), probe)
    np.testing.assert_array_equal(affine.stationary(scan),
                                  affine._class_sum(scan, np.zeros(1), np.ones(1)))


def test_maxwell_mean_is_the_limit_of_fine_velocity_grids():
    affine, scan = _maxwell_scan()
    v, w = doppler.maxwell_strip_grid(T, MASS, K, 0.1 * constants.GAMMA,
                                      error=1e-10, cutoff_sigma=8.0)
    np.testing.assert_allclose(affine.maxwell_mean(scan, T, MASS, K),
                               affine._class_sum(scan, K * v, w), rtol=1e-8, atol=0.0)


def test_a_two_metre_per_second_grid_aliases_the_optical_resonance():
    affine, scan = _maxwell_scan()
    exact = affine.maxwell_mean(scan, T, MASS, K)
    v, w = doppler.velocity_grid(T, mass=MASS, dv=2.0, cutoff_sigma=8.0)
    assert np.max(np.abs(affine._class_sum(scan, K * v, w) / exact - 1.0)) > 1e-4


def test_quadrature_classes_resolve_the_narrowest_pole():
    affine, scan = _maxwell_scan()
    v, w = affine.quadrature_classes(scan, T, MASS, K)
    np.testing.assert_allclose(affine._class_sum(scan, K * v, w),
                               affine.maxwell_mean(scan, T, MASS, K), rtol=1e-6, atol=0.0)


def test_strip_classes_stay_inside_their_floor_and_ceiling():
    floored, _ = affine_doppler._strip_classes(np.zeros(1), T, MASS, K)
    capped, _ = affine_doppler._strip_classes(np.full(1, np.inf), T, MASS, K)
    step = 2 * np.pi * affine_doppler.STRIP_FLOOR_SIGMAS * SIGMA
    np.testing.assert_allclose(K * (floored[1] - floored[0]),
                               step / np.log(1.0 / affine_doppler.QUADRATURE_ERROR), rtol=1e-9)
    assert 20 < capped.size < 100


def test_rows_missing_the_guard_are_summed_on_a_resolving_grid(monkeypatch):
    affine, scan = _maxwell_scan()
    exact = affine.maxwell_mean(scan, T, MASS, K)
    pole_rows = AffineScan._pole_rows

    def one_bad_row(self, *args):
        poles, damping, failed = pole_rows(self, *args)
        poles.res[7] = 1e30
        failed[7] = True
        return poles, damping, failed

    monkeypatch.setattr(AffineScan, "_pole_rows", one_bad_row)
    got = affine.maxwell_mean(scan, T, MASS, K)
    assert got[7] != exact[7]
    np.testing.assert_allclose(got, exact, rtol=1e-6, atol=0.0)


def test_guard_rejects_a_row_off_the_direct_solve(monkeypatch):
    affine, scan = _maxwell_scan()
    assert not affine._pole_rows(scan, SIGMA)[2].any()
    monkeypatch.setattr(affine_doppler, "POLE_GUARD_RTOL", -1.0)
    assert affine._pole_rows(scan, SIGMA)[2].all()


def test_guard_rejects_an_undamped_pole_inside_the_bulk(monkeypatch):
    affine, scan = _maxwell_scan()
    residues = pole_doppler.ShiftInvertSystem.residues

    def real_pole(self, *args, **kwargs):
        poles = residues(self, *args, **kwargs)
        poles.lam[3, 0] = 0.5 * SIGMA
        return poles

    monkeypatch.setattr(pole_doppler.ShiftInvertSystem, "residues", real_pole)
    assert affine._pole_rows(scan, SIGMA)[2][3]


def test_singular_expansion_falls_back_to_a_resolving_grid(monkeypatch):
    affine, scan = _maxwell_scan()
    scan = scan[:3]
    exact = affine.maxwell_mean(scan, T, MASS, K)

    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(pole_doppler.ShiftInvertSystem, "residues", singular)
    np.testing.assert_allclose(affine.maxwell_mean(scan, T, MASS, K), exact, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("name, params", [
    ("lambda", {"view": "EIT"}),
    ("lambda", {"view": "CPT", "species": "⁸⁷Rb", "line": "D2"}),
    ("rydberg_eit", {"view": "EIT"}),
    ("rydberg_eit", {"view": "AT electrometry"}),
])
def test_doppler_on_scheme_matches_a_resolving_velocity_grid(monkeypatch, name, params):
    scheme = schemes.get(name)
    p = scheme.defaults()
    p.update(params, doppler="on")
    class_sum, maxwell_mean = AffineScan._class_sum, AffineScan.maxwell_mean
    fallback, checked = [], []

    def checked_mean(self, scan, T, mass, k_vec, **kwargs):
        out = maxwell_mean(self, scan, T, mass, k_vec, **kwargs)
        rows = np.arange(0, scan.size, 40)
        v, w = self.quadrature_classes(scan[rows], T, mass, k_vec)
        np.testing.assert_allclose(out[rows], class_sum(self, scan[rows], k_vec * v, w),
                                   rtol=1e-6, atol=0.0)
        checked.append(rows.size)
        return out

    def counted_sum(self, scan, kv, w):
        fallback.append(scan.size)
        return class_sum(self, scan, kv, w)

    monkeypatch.setattr(AffineScan, "maxwell_mean", checked_mean)
    monkeypatch.setattr(AffineScan, "_class_sum", counted_sum)
    scheme.compute(p)
    assert checked and not fallback


@pytest.mark.parametrize("name", ["lambda", "rydberg_eit"])
def test_doppler_off_never_reaches_the_maxwell_mean(monkeypatch, name):
    scheme = schemes.get(name)
    p = scheme.defaults()
    p.update(doppler="off")

    def refuse(*args):
        raise AssertionError("Doppler off is the single v = 0 class")

    monkeypatch.setattr(AffineScan, "maxwell_mean", refuse)
    assert np.all(np.isfinite(scheme.compute(p)["chi_bar"]))


def test_quadrature_classes_fall_back_when_the_pole_form_fails(monkeypatch):
    """A bare grid with no accepted rows retains the conservative floor."""
    affine, scan = _maxwell_scan()

    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(AffineScan, "_pole_rows", fail)
    v, w = affine.quadrature_classes(scan[:2], T, MASS, K)
    expected = doppler.maxwell_strip_grid(
        T, MASS, K, affine_doppler.STRIP_FLOOR_SIGMAS * SIGMA,
        affine_doppler.QUADRATURE_ERROR, affine_doppler.BULK_SIGMAS)
    np.testing.assert_array_equal(v, expected[0])
    assert w.sum() == pytest.approx(1.0)


def test_rejected_infinite_strip_converges_to_independent_two_level_voigt(monkeypatch):
    from scipy.special import wofz

    gamma = 0.002 * SIGMA
    omega = 0.05 * gamma
    atom = atoms.two_level(gamma)
    scan = SIGMA * np.array([0.21, 0.143])
    affine = AffineScan(atom, lambda s: np.array([[0., omega / 2], [omega / 2, -s]]),
                        (1, 0), omega)
    original = affine.maxwell_mean(scan, T, MASS, K)
    real_poles = affine._pole_rows

    def one_rejected(scan, sigma):
        poles, damping, failed = real_poles(scan, sigma)
        assert not failed.any()
        failed[1], damping[1] = True, np.inf
        return poles, damping, failed

    monkeypatch.setattr(affine, "_pole_rows", one_rejected)
    records = []
    value = affine.maxwell_mean(scan, T, MASS, K, fallback_rows=records)
    width = np.sqrt((gamma / 2)**2 + omega**2 / 2)
    pole = -1j * np.sqrt(np.pi / 2) / SIGMA * wofz((scan + 1j * width) / (np.sqrt(2) * SIGMA))
    expected = 0.5 * (pole.real + 1j * gamma / (2 * width) * pole.imag)
    assert value[:1].tobytes() == original[:1].tobytes()
    np.testing.assert_allclose(value[1], expected[1], rtol=1e-8, atol=0.)
    v, w = affine_doppler._strip_classes([np.inf], T, MASS, K)
    aliased = affine._class_sum(scan[1:], K * v, w)[0]
    assert abs(aliased - expected[1]) > 0.5 * abs(expected[1])
    assert [(r["row"], r["status"]) for r in records] == [(1, "converged")]
    assert records[0]["refinements"] >= 2 and records[0]["relative_change"] <= 1e-8


def test_masked_grid_and_accepted_phasor_compatibility_in_mixed_batch(monkeypatch):
    affine, _ = _maxwell_scan()
    scan = np.arange(2.)
    monkeypatch.setattr(affine, "_pole_rows", lambda *args: (
        None, SIGMA * np.array([1., 1e-2]), np.array([False, True])))
    masked, _ = affine.quadrature_classes(scan, T, MASS, K)
    assert masked.size == 37
    sampled = {0: [], 1: []}
    legacy_v, legacy_w = affine_doppler._strip_classes(SIGMA * np.array([1., 1e-2]), T, MASS, K)

    def phasor(slot, v, w):
        sampled[slot].append(v.size)
        return np.dot(w, v + 1j) if slot == 0 else 0j

    records = []
    result = affine.quadrature_mean(scan, T, MASS, K, phasor, fallback_rows=records)
    assert result[:1].tobytes() == np.array([np.dot(legacy_w, legacy_v + 1j)]).tobytes()
    assert sampled == {0: [3521], 1: [37, 73, 143]}
    assert result[1] == 0j and records[0]["relative_change"] == 0.


def test_two_complex_changes_required_and_unresolved_rows_withheld_at_cap(monkeypatch):
    affine, _ = _maxwell_scan()
    scan = np.arange(4.)
    monkeypatch.setattr(affine, "_pole_rows", lambda *args: (
        None, np.full(4, np.inf), np.ones(4, dtype=bool)))
    samples = [[] for _ in scan]

    def phasor(slot, v, w):
        samples[slot].append(v.size)
        if slot == 0:  # First two grids agree, third differs in phase only.
            return 1e-30 * (1j if len(samples[slot]) <= 2 else -1j)
        if slot == 1:
            return 0j
        if slot == 2:
            return np.nan + 1j
        return complex(0., len(samples[slot]) % 2)

    records = []
    values = affine.quadrature_mean(scan, T, MASS, K, phasor, fallback_rows=records)
    assert [len(s) for s in samples[:2]] == [5, 3]
    assert samples[2][-1] == samples[3][-1] == 351_811
    assert values[0] == -1e-30j and values[1] == 0.
    assert np.isnan(values[2:]).all()
    by_row = {r["row"]: r for r in records}
    assert [by_row[i]["status"] for i in range(4)] == ["converged"] * 2 + ["unresolved"] * 2
    assert by_row[2]["relative_change"] is None
    assert by_row[3]["relative_change"] == 1. and by_row[3]["total_classes"] == sum(samples[3])


def test_joint_mean_propagates_capped_longitudinal_rows():
    scan = np.arange(5.)

    class Conditional:
        def maxwell_mean(self, scan, *args, fallback_rows):
            fallback_rows.append(dict(row=0, status="unresolved", classes=351811,
                                      total_classes=928209, relative_change=0.3, refinements=14))
            return np.array([np.nan + 1j * np.nan, 1, 2, 3, 4])

    mean, info = affine_doppler.joint_maxwell_mean(
        scan, T, MASS, K, lambda vx: Conditional(), transverse_orders=(2, 4))
    assert np.isnan(mean[0])
    np.testing.assert_allclose(mean[1:], scan[1:])
    assert info["status"] == "not_converged"
    assert all(row["row"] == 0 and row["classes"] == 351811 and "velocity_m_s" in row
               for row in info["longitudinal_fallback"])


def test_numpy_rejection_limits_class_batch_and_preserves_global_weights(monkeypatch):
    affine, _ = _maxwell_scan()
    scan = np.arange(1.)
    monkeypatch.setattr(affine, "_pole_rows", lambda *args: (
        None, np.array([np.inf]), np.array([True])))
    monkeypatch.setattr(kernels, "available", lambda: False)
    sizes = []

    def narrow_sum(rows, kv, weights):
        sizes.append(kv.size)
        assert kv.size <= 2048
        return np.full(rows.size, weights @ (1. / (0.143 - kv / SIGMA + 0.001j)))

    monkeypatch.setattr(affine, "_class_sum", narrow_sum)
    records = []
    result = affine.maxwell_mean(scan, T, MASS, K, fallback_rows=records)
    assert max(sizes) == 2048 and records[0]["status"] == "converged"
    from scipy.special import wofz
    expected = -1j * np.sqrt(np.pi / 2) * wofz((0.143 + 0.001j) / np.sqrt(2))
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=0.)


def test_singular_class_solve_withholds_only_its_row(monkeypatch):
    affine, _ = _maxwell_scan()
    scan = np.arange(3.)
    monkeypatch.setattr(affine, "_pole_rows", lambda *args: (
        None, np.full(3, np.inf), np.ones(3, dtype=bool)))

    def class_sum(detunings, kv, weights):
        if 1. in detunings:
            raise np.linalg.LinAlgError("forced singular class")
        return (detunings + 1j) * weights.sum()

    monkeypatch.setattr(affine, "_class_sum", class_sum)
    records = []
    out = affine.maxwell_mean(scan, T, MASS, K, fallback_rows=records)
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [1j, 2 + 1j])
    cap = next(r for r in records if r["status"] == "unresolved")
    assert (cap["row"], cap["classes"], cap["relative_change"]) == (1, 351811, None)

    # A static pole can pass even though the separate phasor solve is singular.
    monkeypatch.setattr(affine, "_pole_rows", lambda *args: (
        None, np.full(3, np.inf), np.array([False, False, True])))
    records = []

    def phasor(slot, velocity, weights):
        if slot == 1:
            raise np.linalg.LinAlgError("forced singular dynamical phasor")
        return complex(slot, 1)

    out = affine.quadrature_mean(scan, T, MASS, K, phasor, fallback_rows=records)
    assert np.isnan(out[1]) and out[0] == 1j and out[2] == 2 + 1j
    assert {(r["row"], r["status"]) for r in records} == {(1, "unresolved"), (2, "converged")}
