"""Opt-in self-energy changes one per-call generator and keeps the default exact."""
import inspect
import json

import numpy as np
import pytest

from gabes import constants, fwm_stark_response as stark
from gabes.core import build_liouvillian, comm_super
from gabes.schemes import fwm


@pytest.fixture(scope="module")
def context():
    detuning = -1.04858
    center = fwm.branch_center_GHz(detuning, -1)
    return dict(D_GHz=detuning, T=362.15, P_pump=1., P_probe=8e-6,
                w_pump=950e-6, w_probe=300e-6, L=.012,
                coarse_points=41, fine_points=0, scan_min=center-.22, scan_max=center+.22,
                phase_detail=fwm.PHASE_ULTRA, pump_probe_angle_deg=.45,
                response_method=fwm.RESPONSE_POLE, model_fidelity=fwm.FIDELITY_FAST,
                floquet_order=3, enforce_floquet_convergence=True,
                velocity_step=1., velocity_cutoff=4., excess_noise_model=False)


def pump_hamiltonian(power=1., waist=950e-6):
    omega = constants.rabi_freq(power, waist)
    return fwm.pump_hamiltonian_at_Deff_zero(omega, omega)


def test_ground_gap_assignment_and_counterterm_sign():
    result = stark.ground_gap_shift(pump_hamiltonian(), -1.04858*2*np.pi*1e9)
    assert result["shift_mhz"] == pytest.approx(-176.5515489041, abs=1e-7)
    assert len(set(result["ground_eigenstate_indices"])) == 2
    assert not result["assignment_ambiguous"]
    atom = fwm.collisional_atom(362.15)
    _, metadata = stark.apply(atom, pump_hamiltonian(), -1.04858*2*np.pi*1e9,
                              fraction=.1, pump_power_W=1.)
    assert metadata["counterterm_mhz"] == pytest.approx(.9*result["shift_mhz"])
    assert metadata["counterterm_mhz"] < 0.
    assert metadata["corrected_gap_shift_mhz"] == pytest.approx(-23.1374298106, abs=1e-7)
    assert metadata["corrected_gap_shift_mhz"] != pytest.approx(.1*result["shift_mhz"])
    assert "not an exact" in metadata["correction_scope"]
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata


def test_ambiguous_ground_assignment_is_distinct_and_reported():
    h = np.diag([0., 0., 8e9, 10e9]).astype(complex)
    h[0, 1] = h[1, 0] = 1e9
    result = stark.ground_gap_shift(h, 0.)
    assert len(set(result["ground_eigenstate_indices"])) == 2
    assert result["assignment_ambiguous"]
    assert result["joint_assignment_margin"] == pytest.approx(0.)


@pytest.mark.parametrize("fraction,power", [(1., 1.), (0., 0.), (.1, 0.)])
def test_identity_does_not_touch_generator(fraction, power):
    atom = fwm.collisional_atom(362.15)
    original = atom.lindblad
    before = original.copy()
    result, metadata = stark.apply(atom, pump_hamiltonian(power), -1.04858*2*np.pi*1e9,
                                   fraction=fraction, pump_power_W=power)
    assert result is atom and result.lindblad is original
    np.testing.assert_array_equal(result.lindblad, before)
    assert not metadata["applied"]
    assert metadata["counterterm_rad_s"] == 0.
    if power == 0.:
        assert metadata["native_shift_rad_s"] == 0.


def test_counterterm_is_one_coherent_generator_in_both_gauges():
    atom = fwm.collisional_atom(362.15)
    original = fwm.collisional_atom(362.15)
    registry = fwm.ATOM.lindblad.copy()
    _, metadata = stark.apply(atom, pump_hamiltonian(), -1.04858*2*np.pi*1e9,
                              fraction=.1, pump_power_W=1.)
    term = np.diag([0., metadata["counterterm_rad_s"], 0., 0.])
    np.testing.assert_array_equal(atom.lindblad, original.lindblad+comm_super(term))
    for h in (pump_hamiltonian(), fwm.static_hamiltonian_at_Deff_zero(
            constants.rabi_freq(1., 950e-6), constants.rabi_freq(1., 950e-6),
            constants.rabi_freq(8e-6, 300e-6), -16.*2*np.pi*1e6, -1)):
        np.testing.assert_allclose(build_liouvillian(h, atom), build_liouvillian(h+term, original),
                                   rtol=2e-15, atol=2e-6)
    trace = np.eye(4).reshape(-1)
    np.testing.assert_array_equal(trace@comm_super(term), np.zeros(16))
    np.testing.assert_array_equal(fwm.ATOM.lindblad, registry)


@pytest.mark.parametrize("fraction", [-.1, 1.1, np.nan, np.inf])
def test_invalid_fraction_rejected_before_mutation(fraction):
    atom = fwm.collisional_atom(362.15)
    before = atom.lindblad.copy()
    with pytest.raises(ValueError, match="pump_stark_fraction"):
        stark.apply(atom, pump_hamiltonian(), -1.04858*2*np.pi*1e9,
                    fraction=fraction, pump_power_W=1.)
    np.testing.assert_array_equal(atom.lindblad, before)


def test_default_spectrum_bitwise_matches_no_counterterm_path(context, monkeypatch):
    default = fwm.compute_spectrum(**context)
    explicit = fwm.compute_spectrum(**context, pump_stark_fraction=1.)
    with monkeypatch.context() as patch:
        patch.setattr(stark, "apply", lambda atom, *args, **kwargs: (atom, {"applied": False}))
        previous = fwm.compute_spectrum(**context)
    for key, values in previous.items():
        if isinstance(values, np.ndarray):
            np.testing.assert_array_equal(default[key], values)
            np.testing.assert_array_equal(explicit[key], values)
    for key in ("chi", "drift_per_m", "segment_profile"):
        np.testing.assert_array_equal(default["propagation_snapshot"][key], previous["propagation_snapshot"][key])
    assert not default["pump_stark_response"]["applied"]


def test_opt_in_matches_hamiltonian_counterfactual_and_backend(context, monkeypatch):
    actual = fwm.compute_spectrum(**context, pump_stark_fraction=.1)
    metadata = actual["pump_stark_response"]
    assert metadata["applied"]
    assert actual["floquet_convergence"]["status"] == "CONVERGED"
    term = np.diag([0., metadata["counterterm_rad_s"], 0., 0.])
    original_static = fwm.static_hamiltonian_at_Deff_zero
    with monkeypatch.context() as patch:
        patch.setattr(fwm, "static_hamiltonian_at_Deff_zero", lambda *args, **kwargs:
                      original_static(*args, **kwargs)+term)
        expected = fwm.compute_spectrum(**context)
    for key in ("G_s", "G_c", "T_field_small_signal", "T_canonical_small_signal"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-11, atol=1e-12)
    center = fwm.branch_center_GHz(context["D_GHz"], -1)-.016
    tiny = dict(context, coarse_points=3, scan_min=center-1e-9, scan_max=center+1e-9)
    pole = fwm.compute_spectrum(**tiny, pump_stark_fraction=.1)
    grid = fwm.compute_spectrum(**dict(tiny, response_method=fwm.RESPONSE_GRID), pump_stark_fraction=.1)
    for key in ("G_s", "G_c"):
        np.testing.assert_allclose(pole[key], grid[key], rtol=1e-9, atol=1e-12)


def test_recompute_context_keeps_original_args_and_mutable_copies(context):
    branches = [-1]
    noise = {"pump_scatter_kappa": .05}
    raw = fwm.compute_spectrum(**dict(context, branches=branches, excess_noise_model=noise),
                               pump_stark_fraction=.25)
    saved = raw["stark_recompute_context"]
    assert set(saved) == set(inspect.signature(fwm.compute_spectrum).parameters)-{"pump_stark_fraction"}
    assert saved["branches"] == [-1]
    assert saved["branches"] is not branches
    assert saved["excess_noise_model"] is not noise
    assert saved["D_GHz"] == context["D_GHz"]
    assert saved["scan_min"] == context["scan_min"]
    branches.append(1)
    noise["pump_scatter_kappa"] = .9
    assert saved["branches"] == [-1]
    assert saved["excess_noise_model"] == {"pump_scatter_kappa": .05}
    replayed = fwm.compute_spectrum(**saved, pump_stark_fraction=.25)
    for key in ("G_s", "G_c", "T_field_small_signal"):
        np.testing.assert_array_equal(raw[key], replayed[key])


def test_zero_pump_spectrum_is_exact_identity(context):
    zero = dict(context, P_pump=0.)
    original = fwm.compute_spectrum(**zero)
    changed = fwm.compute_spectrum(**zero, pump_stark_fraction=.1)
    assert not changed["pump_stark_response"]["applied"]
    assert changed["pump_stark_response"]["counterterm_rad_s"] == 0.
    for key in ("G_s", "G_c", "T_field_small_signal", "T_canonical_small_signal"):
        np.testing.assert_array_equal(original[key], changed[key])
