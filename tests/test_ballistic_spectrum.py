"""Contract tests for the pure-Rb85 D1 distributed-trajectory spectrum."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gabes import ballistic_spectrum as history, collisions, species
from gabes.od_model import ODConditions


def _conditions(**changes):
    base = ODConditions(species=species.RB85.label, line='D1', temp_c=99., cell_mm=12.5,
                        probe_waist_mm=.53, model_id='od_finite_rb_mf_rate_v2')
    return replace(base, **changes)


def test_weak_solution_unsorted_offsets_and_empty_cell():
    conditions = _conditions(cs_self_broadening_reference=True)
    scan = 2*np.pi*np.array([1.1e9, -1.3e9, 1.1e9, 0.])
    weak = history.evaluate_ballistic_spectrum(conditions, scan)
    assert weak.optical_depth[0] == weak.optical_depth[2]
    np.testing.assert_array_equal(weak.optical_depth, weak.od0)
    np.testing.assert_array_equal(weak.transmission, np.exp(-weak.od0))
    assert np.all(np.isfinite(weak.od0)) and np.max(weak.od0) > 5.
    assert not weak.transmission.flags.writeable and not weak.od0.flags.writeable
    empty = history.evaluate_ballistic_spectrum(replace(conditions, cell_mm=0., probe_power_uw=200.), scan)
    np.testing.assert_array_equal(empty.transmission, np.ones_like(scan))
    np.testing.assert_array_equal(empty.optical_depth, np.zeros_like(scan))


@pytest.mark.parametrize('changes', [
    {'species':species.RB87.label}, {'line':'D2'}, {'paraffin_coated':True},
    {'optical_collision_model':collisions.REFERENCE_COLLISIONS}, {'ne_pressure_torr':1.},
])
def test_scope_enforced_on_direct_calls(changes):
    with pytest.raises(ValueError):
        history.evaluate_ballistic_spectrum(_conditions(**changes), [0.])


def test_isotopic_representation_uses_same_collision_and_optical_density():
    scalar = history._optics(_conditions())
    mixed = history._optics(_conditions(species='Rb (natural)', rb_composition='Custom', rb85_fraction=1.))
    assert scalar == mixed


@pytest.mark.parametrize('peak', [0.01, .5, 1., 2.7, 9.9, 10.])
def test_axial_depletion_agrees_with_independent_differential_equation(peak):
    # Unlike exp[-alpha(input)*L], the differential law weakens saturation
    # as the power depletes. Include flat rows and a finite deep-OD sample.
    nodes = np.array([.5, 1., 3., 10.])
    weak = np.array([3., 1., 1.8])
    table = np.array([[2.8, 1., 1.83], [2.4, 1., 1.9],
                      [1.1, 1., 1.75], [.45, 1., 1.6]])
    column = 4.
    def derivative(_, depth):
        current = peak*np.exp(-depth)
        return [np.interp(np.log(max(v,nodes[0])), np.log(nodes), table[:,i])
                if v>=nodes[0] else weak[i] + (table[0,i]-weak[i])*v/nodes[0]
                for i,v in enumerate(current)]
    expected = solve_ivp(derivative, (0., column), np.zeros_like(weak),
                         rtol=2e-11, atol=1e-12, max_step=.01).y[:,-1]
    actual = history._deplete(weak, nodes, table, peak, column)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)


def test_rest_frame_quadrature_at_every_component_and_at_tails():
    optics = history._optics(_conditions())
    nodes, weights = history._panels(optics.width, order=6, wing_widths=1024)
    atom = history._atom(optics.temperature, optics.width)
    scan = np.linspace(-4, 4, 501) * 2*np.pi*1e9
    density = np.exp(-.5*((scan[:,None]-nodes)/optics.sigma)**2) / (np.sqrt(2*np.pi)*optics.sigma)
    approximate = density @ (weights*atom.weak_cross_sections(nodes))
    exact = history._weak(optics, scan)
    assert np.max(abs(approximate-exact))/max(exact) < 5e-5


def test_fast_preserves_pointwise_frequencies_and_reports_reference_fallback(monkeypatch):
    # Distinct finite values reveal any accidental full-scan downgrade or reordering.
    def fake_evaluation(conditions, scan, resolution, requested):
        used = 'fast' if resolution == history._FAST else 'reference'
        depth = np.abs(scan)*1e-12 + (0.2 if used == 'fast' else 0.3)
        return history.BallisticSpectrum(np.exp(-depth), depth, depth*2,
            dict(fidelity_used=used, fidelity_requested=requested, max_weak_absorption_per_waist=0.5))
    monkeypatch.setattr(history, '_evaluate', fake_evaluation)
    c = _conditions(probe_power_uw=25.)
    scan = np.array([8., -1., -8., -1., 1.])*2*np.pi*1e9
    output = history.evaluate_ballistic_spectrum(c, scan, fidelity='fast')
    assert output.diagnostics['fidelity_used'] == 'mixed'
    assert output.diagnostics['reference_samples'] == 2
    assert [(p['samples'], p['fidelity_used']) for p in output.diagnostics['partitions']] == [(3, 'fast'), (2, 'reference')]
    separate = [history.evaluate_ballistic_spectrum(c, [x], fidelity='fast').transmission[0] for x in scan]
    np.testing.assert_array_equal(output.transmission, separate)
    assert not output.transmission.flags.writeable
    fallback = history.evaluate_ballistic_spectrum(replace(c, probe_waist_mm=1.), scan, fidelity='fast')
    assert fallback.diagnostics['fidelity_used'] == 'reference'
    assert 'validation range' in fallback.diagnostics['fallback_reason']


def test_fast_numerical_failure_retries_reference_and_retains_model_scope(monkeypatch):
    call = history._evaluate
    def reject_fast(conditions, scan, resolution, requested):
        if resolution == history._FAST:
            raise ArithmeticError('unresolved populations')
        return call(conditions, scan, resolution, requested)
    monkeypatch.setattr(history, '_evaluate', reject_fast)
    result = history.evaluate_ballistic_spectrum(_conditions(), [0.], fidelity='fast')
    assert result.diagnostics['fidelity_requested'] == 'fast'
    assert result.diagnostics['fidelity_used'] == 'reference'
    assert 'unresolved populations' in result.diagnostics['fallback_reason']
