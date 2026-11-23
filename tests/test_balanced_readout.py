
import numpy as np
import pytest


from gabes import observables  # noqa: E402


def test_asymmetric_balanced_noise_recovers_symmetric_eta_formula():
    G_s = np.array([1.2, 3.0, 15.0, 80.0])
    G_c = G_s - 1.0
    eta = 0.87

    old = observables.intensity_difference_squeezing_dB(G_s, G_c, eta)
    new = observables.balanced_twin_beam_squeezing_dB(
        G_s, G_c, eta_s=eta, eta_c=eta, reference_weight="raw")

    assert np.allclose(new, old, rtol=1e-12, atol=1e-12)


def test_shot_weight_is_optimal_only_inside_the_effective_closure():
    G_s = 12.0
    G_c = 10.8
    eta_s = 0.42
    eta_c = 0.91

    shot = observables.balanced_twin_beam_noise(
        G_s, G_c, eta_s, eta_c, reference_weight="shot")
    raw = observables.balanced_twin_beam_noise(
        G_s, G_c, eta_s, eta_c, reference_weight="raw")
    dc = observables.balanced_twin_beam_noise(
        G_s, G_c, eta_s, eta_c, reference_weight="dc")
    optimal = observables.balanced_twin_beam_noise(
        G_s, G_c, eta_s, eta_c, reference_weight="optimal")

    # Poissonian arms make V_c m_s = V_s m_c, so the covariance optimum is sqrt.
    assert np.isclose(optimal, shot, rtol=1e-12)
    assert shot <= raw
    assert shot <= dc


def test_measured_source_noise_anchor_propagates_symmetric_loss():
    source_noise = 10 ** (-8.3 / 10.0)
    eta = 0.72
    got = observables.balanced_twin_beam_noise(
        5.6, 4.6, eta_s=eta, eta_c=eta, reference_weight="raw",
        source_noise=source_noise)
    expected = eta * source_noise + (1.0 - eta)

    assert np.isclose(got, expected)


def _exact_unequal(A, eta_s, eta_c, w):
    B = A - 1.0
    var = (eta_s**2 * A * (A + B) + eta_s * (1 - eta_s) * A
           + w**2 * (eta_c**2 * B * (A + B) + eta_c * (1 - eta_c) * B)
           - 4.0 * w * eta_s * eta_c * A * B)
    return var / (eta_s * A + w**2 * eta_c * B)


@pytest.mark.parametrize("A,eta_s,eta_c", [(1.3, 0.9, 0.9), (12.0, 0.42, 0.91),
                                            (15.0, 0.8694, 0.7), (80.0, 0.99, 0.6)])
def test_ideal_canonical_matches_the_unequal_efficiency_equation(A, eta_s, eta_c):
    B = A - 1.0
    for weight, w in (("raw", 1.0), ("dc", eta_s * A / (eta_c * B)), (0.83, 0.83)):
        got = observables.balanced_twin_beam_noise(
            A, B, eta_s, eta_c, reference_weight=weight, covariance="ideal_canonical")
        assert got == pytest.approx(_exact_unequal(A, eta_s, eta_c, w), rel=1e-12)


def test_ideal_canonical_reduces_to_the_symmetric_law():
    G = np.array([1.2, 3.0, 15.0, 80.0])
    for eta in (1.0, 0.87, 0.3):
        got = observables.balanced_twin_beam_noise(
            G, G - 1.0, eta, eta, reference_weight="raw", covariance="ideal_canonical")
        np.testing.assert_allclose(got, 1 - eta + eta / (2 * G - 1), rtol=1e-12)
        # Production passes equal efficiencies: both closures agree at canonical gains.
        for weight in ("raw", "dc"):
            effective = observables.balanced_twin_beam_noise(G, G - 1.0, eta, eta,
                                                             reference_weight=weight)
            exact = observables.balanced_twin_beam_noise(
                G, G - 1.0, eta, eta, reference_weight=weight, covariance="ideal_canonical")
            # Different cancellation order: agree to rounding near S = 1/(2G-1).
            np.testing.assert_allclose(effective, exact, rtol=1e-12, atol=1e-13)


def test_bright_seed_fixture_weights():
    G, eta = 15.0, 0.8694
    moments = observables.ideal_twin_beam_moments(G, eta, eta, bright_seed_limit=True)
    raw = 10 * np.log10(observables.difference_noise(moments, "raw"))
    assert round(float(raw), 6) == -7.943104
    w_dc = moments.mean_s / moments.mean_c
    assert w_dc == pytest.approx(15.0 / 14.0, rel=1e-15)
    assert moments.mean_s - w_dc * moments.mean_c == pytest.approx(0.0, abs=1e-12)
    w_opt = observables.optimal_difference_weight(moments)
    assert w_opt == pytest.approx(np.sqrt(15.0 / 14.0), rel=1e-12)
    S_opt = observables.difference_noise(moments, "optimal")
    manual = ((moments.var_s + w_opt**2 * moments.var_c - 2 * w_opt * moments.cov)
              / (moments.mean_s + w_opt**2 * moments.mean_c))
    assert S_opt == pytest.approx(manual, rel=1e-14)
    assert round(float(10 * np.log10(S_opt)), 6) == -8.368562


def test_unequal_efficiencies_expose_the_effective_closure_and_the_true_optimum():
    A, B, eta_s, eta_c = 12.0, 11.0, 0.42, 0.91
    exact = observables.balanced_twin_beam_noise(A, B, eta_s, eta_c, reference_weight="raw",
                                                 covariance="ideal_canonical")
    effective = observables.balanced_twin_beam_noise(A, B, eta_s, eta_c, reference_weight="raw")
    assert exact / effective > 5.0          # order-unity error of the effective closure
    moments = observables.ideal_twin_beam_moments(A, eta_s, eta_c, bright_seed_limit=True)
    w = np.linspace(0.05, 3.0, 20001)
    scan = observables.difference_noise(moments, w)
    w_opt = observables.optimal_difference_weight(moments)
    assert abs(w_opt - w[np.argmin(scan)]) < 2 * (w[1] - w[0])
    S_opt = observables.difference_noise(moments, "optimal")
    assert S_opt <= scan.min() * (1 + 1e-12)
    # The square-root power weight is not the optimum here.
    assert observables.difference_noise(moments, "shot") > 1.05 * S_opt
    assert not np.isclose(w_opt, np.sqrt(moments.mean_s / moments.mean_c), rtol=1e-3)


def _thinned_tmsv(nbar, eta_s, eta_c, n_max=4000):
    """Explicit thermal pair distribution followed by binomial thinning."""
    n = np.arange(n_max)
    p = (nbar / (1 + nbar)) ** n / (1 + nbar)
    mean_n, var_n = np.sum(p * n), np.sum(p * n * n) - np.sum(p * n) ** 2
    out = {}
    for arm, eta in (("s", eta_s), ("c", eta_c)):
        out["mean_" + arm] = eta * mean_n
        out["var_" + arm] = eta**2 * var_n + eta * (1 - eta) * mean_n
    out["cov"] = eta_s * eta_c * var_n
    return out


def test_vacuum_tmsv_pure_loss_is_conditional_binomial_thinning():
    G, eta_s, eta_c = 4.5, 0.63, 0.88
    nbar = G - 1.0
    m = observables.ideal_twin_beam_moments(G, eta_s, eta_c)
    assert (m.source, m.environment) == ("vacuum_input_tmsv", "pure_loss_vacuum")
    reference = _thinned_tmsv(nbar, eta_s, eta_c)
    for key, value in reference.items():
        assert getattr(m, key) == pytest.approx(value, rel=1e-10)
    for eta, mean, var in ((eta_s, m.mean_s, m.var_s), (eta_c, m.mean_c, m.var_c)):
        assert mean == pytest.approx(eta * nbar, rel=1e-14)
        assert var == pytest.approx(eta * nbar * (1 + eta * nbar), rel=1e-14)
    g = observables.twin_beam_correlations(m)
    assert g["g2_ss"] == pytest.approx(2.0, rel=1e-13)
    assert g["g2_cc"] == pytest.approx(2.0, rel=1e-13)


def test_symmetric_and_asymmetric_loss_laws():
    G, nbar = 6.0, 5.0
    eta = 0.7
    m = observables.ideal_twin_beam_moments(G, eta, eta)
    var = m.var_s + m.var_c - 2 * m.cov
    assert var == pytest.approx(2 * eta * (1 - eta) * nbar, rel=1e-12)
    assert observables.difference_noise(m, "raw") == pytest.approx(1 - eta, rel=1e-12)

    eta_p, eta_c = 0.9, 0.55
    m = observables.ideal_twin_beam_moments(G, eta_p, eta_c)
    vacuum = eta_p * (1 - eta_p) * nbar + eta_c * (1 - eta_c) * nbar
    raw = m.var_s + m.var_c - 2 * m.cov
    assert raw - vacuum == pytest.approx((eta_p - eta_c) ** 2 * nbar * (nbar + 1), rel=1e-12)
    w = eta_p / eta_c
    weighted = m.var_s + w**2 * m.var_c - 2 * w * m.cov
    assert weighted == pytest.approx(eta_p * (1 - eta_p) * nbar
                                     + w**2 * eta_c * (1 - eta_c) * nbar, rel=1e-12)


def test_typed_sources_and_environments_are_not_one_g2_branch():
    G, eta, n_env = 3.0, 0.6, 0.8
    seeded = observables.ideal_twin_beam_moments(G, eta, eta, seed_photons=50.0)
    assert seeded.source == "coherent_seeded_displaced_thermal"
    g = observables.twin_beam_correlations(seeded)
    assert 1.0 < g["g2_ss"] < 2.0 and 1.0 < g["g2_cc"] < 2.0
    hot = observables.ideal_twin_beam_moments(G, eta, eta, env_photons_s=n_env,
                                              env_photons_c=n_env)
    assert hot.environment == "thermal_attenuation"
    n_out = eta * (G - 1.0) + (1 - eta) * n_env
    assert hot.mean_s == pytest.approx(n_out, rel=1e-14)
    assert hot.var_s == pytest.approx(n_out * (n_out + 1), rel=1e-14)   # still thermal
    bright = observables.ideal_twin_beam_moments(G, eta, 0.8, bright_seed_limit=True)
    finite = observables.ideal_twin_beam_moments(G, eta, 0.8, seed_photons=1e7)
    for key in ("mean_s", "mean_c", "var_s", "var_c", "cov"):
        assert getattr(finite, key) / 1e7 == pytest.approx(getattr(bright, key), rel=1e-6)


def test_nonclassicality_needs_a_measured_or_propagated_covariance():
    ideal = observables.ideal_twin_beam_moments(2.0, 1.0, 1.0)
    g = observables.twin_beam_correlations(ideal)
    assert g["ideal_nonclassicality"] and g["physical_nonclassicality"] is None
    measured = observables.TwinBeamMoments(
        ideal.mean_s, ideal.mean_c, ideal.var_s, ideal.var_c, ideal.cov,
        source="recorded_counts", environment="detected",
        covariance_origin=observables.COVARIANCE_MEASURED)
    g = observables.twin_beam_correlations(measured)
    assert g["physical_nonclassicality"] and g["ideal_nonclassicality"] is None
    with pytest.raises(ValueError, match="Unknown covariance origin"):
        observables.TwinBeamMoments(1.0, 1.0, 1.0, 1.0, 0.0, "x", "y", "inferred")


def test_closure_requests_are_explicit():
    with pytest.raises(observables.CovarianceUnavailable):
        observables.balanced_twin_beam_noise(5.0, 4.0, 0.9, 0.9, covariance="microscopic")
    with pytest.raises(ValueError, match="Ideal canonical fixture"):
        observables.balanced_twin_beam_noise(5.0, 3.6, 0.9, 0.9, covariance="ideal_canonical")
    with pytest.raises(ValueError, match="covariance must be"):
        observables.balanced_twin_beam_noise(5.0, 4.0, covariance="guess")
    source = 10 ** (-8.0 / 10.0)
    with pytest.raises(ValueError, match="symmetric raw difference"):
        observables.balanced_twin_beam_noise(5.6, 4.6, 0.8, 0.6, source_noise=source)
    with pytest.raises(ValueError, match="symmetric raw difference"):
        observables.balanced_twin_beam_noise(5.6, 4.6, 0.8, 0.8, reference_weight="dc",
                                             source_noise=source)
    declared = observables.balanced_twin_beam_noise(5.6, 4.6, 0.8, 0.6, reference_weight="dc",
                                                    source_noise=source, covariance="effective")
    mean_s, mean_c = 0.8 * 5.6, 0.6 * 4.6
    cov = 0.8 * 0.6 * min(0.5 * 10.2 * (1 - source), np.sqrt(5.6 * 4.6))
    w = mean_s / mean_c
    expected = (mean_s + w * w * mean_c - 2 * w * cov) / (mean_s + w * w * mean_c)
    assert declared == pytest.approx(expected, rel=1e-15)
