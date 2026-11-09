"""Physical angular normalization and exact axial-block response checks."""
import numpy as np
import pytest

from gabes import constants, core, hyperfine
from gabes import fwm_zeeman_response as zeeman


def test_signed_dipoles_emission_and_actual_sublevel_normalization():
    model = zeeman.build_model(362.15)
    excited = np.diag(np.r_[np.zeros(12), np.ones(12)])
    np.testing.assert_allclose(sum(d.conj().T@d for d in model.dipoles.values()), excited, atol=6e-16)
    np.testing.assert_allclose(sum(d.conj().T@d for d in model.atom.emission_ops), constants.GAMMA*excited, rtol=3e-15, atol=2e-8)
    assert len(model.atom.emission_ops) == 12
    assert [len(model.sectors[q]) for q in (0, -1, 1)] == [88, 80, 80]
    np.testing.assert_array_equal(model.thermal.diagonal()[:12], np.full(12, 1./12))
    assert model.provenance["propagation_line_strength"] == 1./3.
    assert model.provenance["external_sublevel_population_factor"] == 1.
    for fg in (2, 3):
        for fe in (2, 3):
            mask = ((model.level_F[:, None] == fg)&(model.level_F[None, :] == fe))
            # Coherent pi polarization is isotropically equivalent to any
            # fixed unit polarization for an unpolarized thermal ensemble.
            assert np.sum(abs(model.dipoles[0]*mask)**2) == pytest.approx(3*hyperfine.CF2[fg, fe])


def test_secular_emission_does_not_feed_hyperfine_ground_coherence():
    secular = zeeman.build_model(362.15, transit_rate=0., ground_dephasing_rate=0., optical_dephasing_rate=0.)
    global_q = zeeman.build_model(362.15, transit_rate=0., ground_dephasing_rate=0., optical_dephasing_rate=0., emission_mode="global_polarization")
    ground_cross = (secular.level_F[:12, None] != secular.level_F[None, :12])
    refill_secular = np.zeros((12, 12), complex)
    refill_global = np.zeros((12, 12), complex)
    # Excited F'=2,m=0 has distinguishable decays to F=2 and F=3.
    index = next(i for i in range(12,24) if secular.level_F[i] == 2 and secular.level_m[i] == 0)
    rho = np.zeros((24,24)); rho[index,index] = 1.
    for jump in secular.atom.emission_ops:
        refill_secular += (jump@rho@jump.conj().T)[:12,:12]
    for jump in global_q.atom.emission_ops:
        refill_global += (jump@rho@jump.conj().T)[:12,:12]
    np.testing.assert_array_equal(refill_secular[ground_cross], 0.)
    assert np.linalg.norm(refill_global[ground_cross]) > .1*constants.GAMMA


def test_pump_zero_matches_independent_hyperfine_lorentz_susceptibility():
    reset, ground, optical = 6e5, 73., 2e4
    deff = 2*np.pi*np.array([-.9e9, .8e9])
    delta = 2*np.pi*np.array([-16e6, 26e6])
    reference = zeeman.pump_weak_response_reference(0., delta, deff, T=362.15,
        transit_rate=reset, ground_dephasing_rate=ground, optical_dephasing_rate=optical,
        velocity_weights=[.4,.6], return_velocity_chi=True)
    np.testing.assert_array_equal(reference.pump_state, np.broadcast_to(zeeman.build_model(
        362.15, transit_rate=reset, ground_dephasing_rate=ground, optical_dephasing_rate=optical).thermal,
        reference.pump_state.shape))
    np.testing.assert_allclose(reference.chi_sc, 0., atol=1e-30)
    np.testing.assert_allclose(reference.chi_cs, 0., atol=1e-30)
    density = 1.234e17
    physical = reference.physical_chi(density)
    width = constants.GAMMA/2+reset+optical+ground/2
    prefactor = -density*3*constants.DIPOLE_D1**2/(12*constants.EPS_0*constants.HBAR)
    for di, detuning in enumerate(delta):
        relative = -constants.OMEGA_HF+detuning
        for arm, sign in ((0,1.), (1,-1.)):
            line = np.zeros(len(deff), complex)
            for fg in (2,3):
                for fe in (2,3):
                    optical_detuning = deff+sign*relative+(fg == 3)*constants.OMEGA_HF+(fe == 2)*constants.OMEGA_EXCITED_HF
                    line += hyperfine.CF2[fg,fe]/(optical_detuning+1j*width)
            expected = prefactor*np.array([.4,.6])@line
            assert physical[0,di,arm,arm] == pytest.approx(expected, rel=3e-14, abs=1e-20)
    assert np.all(physical[...,0,0].imag > 0)
    assert np.all(physical[...,1,1].imag > 0)


def test_hyperfine_collision_projectors_preserve_within_F_coherence():
    gamma = 125.
    model = zeeman.build_model(362.15, transit_rate=0., ground_dephasing_rate=gamma, optical_dephasing_rate=0.)
    diagonal = np.diag(model.atom.lindblad).reshape(24,24)
    same_f = np.where((model.level_F[:12,None] == model.level_F[None,:12]))
    different_f = np.where((model.level_F[:12,None] != model.level_F[None,:12]))
    np.testing.assert_allclose(diagonal[same_f], 0., atol=1e-13)
    np.testing.assert_allclose(diagonal[different_f], -gamma, atol=1e-13)


def test_block_state_and_full_resolvent_are_identical():
    pump, deff, delta = 2*np.pi*350e6, -2*np.pi*1.05e9, -2*np.pi*16e6
    prepared = zeeman.prepare_pump(pump, [deff], T=362.15)
    actual = zeeman.evaluate_prepared(prepared, [delta])
    model = prepared.model
    generator = core.build_liouvillian(zeeman.pump_hamiltonian(pump, model), model.atom)-deff*model.atom.S_v
    with core.blas_single_thread():
        state = core.steady_state_from_liouvillian(generator, 24)
        lower = model.weak_coupling
        raise_ = lower.conj().T
        sources = np.column_stack([core.comm_super(h)@state.reshape(-1) for h in (.5*raise_,.5*lower)])
        response = core.trace_zero_liouvillian_response(generator, sources, -constants.OMEGA_HF+delta, 24)
    readout = np.stack((lower.T.reshape(-1), raise_.T.reshape(-1)))
    np.testing.assert_allclose(prepared.pump_state[0], state, rtol=2e-11, atol=1e-13)
    np.testing.assert_allclose(actual.chi_matrix[0,0], readout@response, rtol=2e-10, atol=1e-22)
    assert actual.diagnostics["max_pump_normalized_residual"] < 1e-13
    assert actual.diagnostics["max_response_normalized_residual"] < 1e-13
    assert actual.diagnostics["minimum_pump_eigenvalue"] >= -1e-12


def test_reusable_poles_match_direct_and_velocity_chunking():
    prepared = zeeman.prepare_pump(2*np.pi*400e6, 2*np.pi*np.array([-1.1e9,-1.e9,-.8e9]),
                                   T=362.15, velocity_weights=[.2,.5,.3])
    delta = 2*np.pi*np.array([-80e6,-16e6,0.,26e6,80e6])
    direct = zeeman.evaluate_prepared(prepared, delta, chunk_size=1)
    poles = zeeman.evaluate_prepared(prepared, delta, method="poles", chunk_size=2)
    repeated = zeeman.evaluate_prepared(prepared, delta, method="poles", chunk_size=2)
    np.testing.assert_allclose(poles.chi_matrix, direct.chi_matrix, rtol=2e-8, atol=1e-22)
    np.testing.assert_array_equal(poles.chi_matrix, repeated.chi_matrix)
    assert poles.diagnostics["pole_direct_fallback_chunks"] == 0
    assert poles.diagnostics["max_pole_guard_relative_error"] < 1e-6


def test_low_pump_cross_response_scales_quadratically():
    values = []
    for pump in (1e6, 2e6):
        value = zeeman.pump_weak_response_reference(pump, [-2*np.pi*16e6], [-2*np.pi*1.05e9], T=362.15)
        values.append(value.chi_sc[0,0])
    assert abs(values[1]/values[0]) == pytest.approx(4., rel=.001)


def test_public_dictionary_and_positive_frequency_row_convention():
    result = zeeman.pump_weak_response(-1.05, [-16.,26.], T=362.15,
        pump_power_W=1., pump_waist_m=950e-6, velocities=[-100.,100.], velocity_weights=[.4,.6])
    reference = zeeman.compute_response(-1.05, [-16.,26.], T=362.15,
        pump_power_W=1., pump_waist_m=950e-6, velocities=[-100.,100.], velocity_weights=[.4,.6])
    assert result["chi"].shape == (2,2,2)
    np.testing.assert_array_equal(result["chi"], reference.positive_frequency_chi_matrix[0])
    np.testing.assert_array_equal(result["chi"][:,1,0], reference.chi_cs[0])
    np.testing.assert_array_equal(result["chi"][:,1,1], reference.chi_cc[0])
    assert result["normalization"]["propagation_line_strength_multiplier"] == 1./3.
    assert result["provenance"]["all_hyperfine_weak_couplings"]


def test_physical_factory_reuse_matches_compute_response():
    kwargs = dict(T=362.15, pump_power_W=1., pump_waist_m=950e-6,
                  velocities=[-100.,100.], velocity_weights=[.4,.6])
    prepared = zeeman.prepare_response(-1.05, **kwargs)
    reused = zeeman.evaluate_prepared(prepared, np.array([-16.,26.])*2*np.pi*1e6)
    direct = zeeman.compute_response(-1.05, [-16.,26.], **kwargs)
    np.testing.assert_array_equal(reused.chi_matrix, direct.chi_matrix)
    np.testing.assert_array_equal(reused.pump_state, direct.pump_state)
    np.testing.assert_array_equal(prepared.frequency_doppler_scale,
                                  1.-np.array([-100.,100.])/constants.C_LIGHT)
