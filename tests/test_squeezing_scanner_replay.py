"""Scanner conventions and same-engine historical replay contracts."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gabes import observables
from gabes.schemes import fwm
from gabes_lab.pipelines.squeezing import (
    scan_squeezing_delta_T as delta,
    scan_squeezing_theory_max as theory,
    scan_squeezing_tpd_angle_detail as angle,
    scan_squeezing_v6_tolerance as v6,
)


@pytest.fixture
def frontier(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / ".claude/skills/fwm-squeezing-frontier/scripts/scan_squeezing_frontier.py"
    spec = importlib.util.spec_from_file_location("frontier_replay", path)
    scanner = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(root))
    spec.loader.exec_module(scanner)
    return scanner


def test_delta_t_replays_temperature_table_and_defaults_to_engine(monkeypatch):
    monkeypatch.setattr(delta, "DELTA_GHZ", np.array([0.9]))
    monkeypatch.setattr(delta, "TEMP_C", np.array([121.0]))
    monkeypatch.setattr(delta, "COARSE_POINTS", 41)
    monkeypatch.setattr(delta, "RESPONSE_METHOD", fwm.RESPONSE_POLE)
    historical_axis = delta._delta_axis(41)
    tables, velocities, weights, master = delta._temperature_tables(
        394.15, historical_axis, 5.0, 3.0)
    original = delta._best_from_tables(
        tables, master, velocities, weights, historical_axis, 0.9, 394.15)
    expected_replay = [original[k] for k in ("S_dB", "G_s", "G_c", "delta_mhz")]
    replay = delta.run_grid(historical_replay=True)
    assert np.asarray(replay).reshape(-1).tobytes() == np.asarray(expected_replay).tobytes()

    center = fwm.branch_center_GHz(0.9, delta.BRANCH)
    spec = fwm.compute_spectrum(
        0.9, T=394.15, P_pump=delta.P_PUMP, P_probe=delta.P_SEED,
        line_strength=delta.LINE_STRENGTH, loss_frac=delta.LOSS_FRAC, qe=delta.QE,
        coarse_points=41, fine_points=0, scan_min=center - delta.WINDOW_GHZ,
        scan_max=center + delta.WINDOW_GHZ, response_method=fwm.RESPONSE_POLE,
        velocity_measure="exact", velocity_step=1.0, velocity_cutoff=4.0, floquet_order=3)
    i = np.nanargmin(spec["gain_referred_noise_dB"])
    expected_engine = np.array([spec["gain_referred_noise_dB"][i], spec["G_s"][i],
                                spec["G_c"][i], (spec["probe_axis_GHz"][i] - center) * 1e3])
    actual_engine = np.asarray(delta.run_grid()).reshape(-1)
    np.testing.assert_allclose(actual_engine, expected_engine, rtol=1e-13, atol=1e-13)
    assert abs(actual_engine[0] - expected_replay[0]) > 1e-6


@pytest.mark.parametrize("method", [fwm.RESPONSE_POLE, fwm.RESPONSE_GRID])
def test_delta_t_historical_replay_preserves_optical_normalization(monkeypatch, method):
    # Pre-manifold-RMS scan from 3956706, on a small sampled velocity grid.
    monkeypatch.setattr(delta, "DELTA_GHZ", np.array([-0.7]))
    monkeypatch.setattr(delta, "RESPONSE_METHOD", method)
    delta_axis = 2.0 * np.pi * np.array([-8.0, -2.0, 3.0]) * 1e6
    tables, velocities, weights, master = delta._temperature_tables(
        337.0, delta_axis, 120.0, 2.0)
    assert isinstance(tables, fwm.ChiPoleRows) == (method == fwm.RESPONSE_POLE)
    result = delta._best_from_tables(
        tables, master, velocities, weights, delta_axis, -0.7, 337.0)
    np.testing.assert_allclose(
        [result["G_s"], result["G_c"], result["S_dB"], result["delta_mhz"]],
        [0.12410238212874558, 0.008022560040066098, -6.5903227392967505, -8.0],
        rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("historical_replay", [True, False])
def test_frontier_uses_engine_noise_and_replays_original_grid(frontier, historical_replay):
    d, temperature, window, qe = 0.9, 394.15, 0.7, 0.86
    ref, _ = frontier.load_reference()
    center = fwm.branch_center_GHz(d, -1)
    grid = (5.0, 3.0) if historical_replay else (1.0, 4.0)
    engine_options = (dict(qe=fwm.QE_DETECTOR) if historical_replay else
                      dict(qe=qe, response_method=fwm.RESPONSE_POLE, velocity_measure="exact"))
    spec = fwm.compute_spectrum(
        d, T=temperature, P_pump=ref["P_pump"], P_probe=ref["P_probe"],
        line_strength=ref["line_strength"], loss_frac=ref["loss_frac"],
        coarse_points=41, fine_points=0, scan_min=center-window, scan_max=center+window,
        velocity_step=grid[0], velocity_cutoff=grid[1], branch=-1,
        phase_detail=fwm.PHASE_ULTRA, model_fidelity=fwm.FIDELITY_ULTRA, **engine_options)
    gs, gc, noise = spec["G_s"], spec["G_c"], spec["gain_referred_noise_dB"]
    gap = gs - gc
    eligible = (gap >= frontier.GAP_MIN) & (gap <= frontier.GAP_MAX)
    assert eligible.any()
    idx = np.nanargmin(np.where(eligible, noise, np.inf))

    actual = frontier.best_squeezing(
        d, temperature, ref, window, 41, *grid, -1, frontier.FIDELITY_KW["ultra"],
        qe=qe, historical_replay=historical_replay)
    np.testing.assert_array_equal([actual["G_s"], actual["G_c"]], [gs[idx], gc[idx]])
    if historical_replay:
        hn = spec["hardened_noise"]
        od = spec["segment_absorption_od"]
        od_s, od_c = hn["od_probe_lin_arr"][idx], hn["od_conj_arr"][idx]
        eta = qe * (1.0 - ref["loss_frac"])
        eta_s, eta_c = eta * np.exp(-(od + od_s)), eta * np.exp(-od_c)
        var = observables.balanced_twin_beam_noise(
            gs[idx], gc[idx], eta_s, eta_c, reference_weight="dc")
        expected = 10.0 * np.log10(max(var + hn["pump_scatter_noise"], 1e-30))
        value = frontier.squeezing_map(
            actual["G_s"], actual["G_c"], actual["seg_od"], eta,
            od_conj=actual["od_conj"], od_probe=actual["od_probe"],
            pump_scatter=actual["pump_scatter"])
        assert value.tobytes() == np.asarray(expected).tobytes()
    else:
        assert actual["gain_noise_db"] == noise[idx]
        no_loss = fwm.seeded_balanced_gain_noise_dB(
            gs[idx], gc[idx], qe, qe,
            seed_excess_noise=spec["hardened_noise"]["pump_scatter_noise"])
        assert actual["gain_noise_qe_db"] == no_loss


def test_v6_loss_sweep_matches_engine_at_base_and_replays_detected_scatter():
    loss_axis = np.array([0.0, v6.LOSS0, 0.775])
    sol = v6._solve(0.9, 121.0, v6.P_PROBE0, coarse=41)
    _, best = v6._readouts(sol)
    idx = best["idx"]
    default = v6._xi_of_loss(sol, idx, loss_axis)
    assert default[1] == pytest.approx(sol["xi"][idx], rel=1e-13, abs=1e-13)

    eta = v6.QE * (1.0 - loss_axis)
    eta_s = eta * np.exp(-(sol["seg_od"] + sol["od_probe"][idx]))
    eta_c = eta * np.exp(-sol["od_conj"][idx])
    original = np.array([
        10.0 * np.log10(max(observables.balanced_twin_beam_noise(
            sol["Gs"][idx], sol["Gc"][idx], es, ec, reference_weight="dc")
            + sol["pump_scatter"], 1e-30))
        for es, ec in zip(eta_s, eta_c, strict=True)
    ])
    replay = v6._xi_of_loss(sol, idx, loss_axis, historical_replay=True)
    assert replay.tobytes() == original.tobytes()
    assert replay[1] != default[1]


def test_angle_and_theory_replay_solver_options(monkeypatch):
    calls = []

    def compute(_d, **kwargs):
        calls.append(kwargs)
        detuning = np.linspace(kwargs["scan_min"], kwargs["scan_max"], kwargs["coarse_points"])
        return dict(probe_axis_GHz=detuning, gain_referred_noise_dB=np.full(detuning.size, -1.0),
                    G_s=np.full(detuning.size, 3.0), G_c=np.full(detuning.size, 2.0),
                    delta_k_z=np.zeros(detuning.size), hardened_noise=None,
                    G_s_smallsignal_peak=3.0, pump_depletion_cap=10.0)

    monkeypatch.setattr(fwm, "compute_spectrum", compute)
    monkeypatch.setattr(theory, "RESPONSE_METHOD", fwm.RESPONSE_POLE)
    config = angle.ScanConfig("v6", np.array([0.9]), np.array([121.0]), np.array([0.3]),
                              np.array([-550.0, 0.0, 550.0]), 0.9, 121.0)
    angle._compute_one(config, 0, 0, 0)
    angle._compute_one(replace(config, historical_replay=True, velocity_step=4.0,
                               velocity_cutoff=3.0), 0, 0, 0)
    new = theory.deepest_point(0.9, 394.15, coarse=3)
    replay = theory.deepest_point(
        0.9, 394.15, coarse=3, vstep=3.0, vcut=3.0, historical_replay=True)
    assert [(c["response_method"], c["velocity_measure"]) for c in calls] == [
        (fwm.RESPONSE_POLE, "exact"), (fwm.RESPONSE_GRID, "truncated"),
        (fwm.RESPONSE_POLE, "exact"), (fwm.RESPONSE_POLE, "truncated"),
    ]
    assert calls[2]["qe"] == 1.0
    assert calls[3]["qe"] == theory.QE
    assert new["S_ideal_dB"] == -1.0
    assert replay["S_ideal_dB"] == observables.gain_referred_noise_dB(3.0, 2.0, 1.0)
