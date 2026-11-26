"""The public coherence model keeps the isotope frequency frame, column and weak limit."""
from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
import pytest

from gabes import species
from gabes.ballistic_spectrum import _rb_components
from gabes.field_spectrum import evaluate_field_ballistic_spectrum, field_progress
from gabes.field_weak import weak_cross_sections
from gabes.od_model import ODConditions, FIELD_HISTORY_REFERENCE, evaluate_od_display
from gabes.od_spectrum_fit import od_transmission


def test_mixture_weak_od_and_fitter_share_prescribed_field_and_scan():
    c = ODConditions("Rb (natural)", "D2", 79., 12.5, FIELD_HISTORY_REFERENCE,
                     rb_composition="Custom", rb85_fraction=.38, od_field_x_ut=23.,
                     od_field_y_ut=-42., od_field_z_ut=57., od_probe_azimuth_deg=30.,
                     od_probe_ellipticity_deg=19., line_strength=.6)
    axis = np.linspace(-6e9, 6e9, 2001) * 2 * np.pi
    result = evaluate_od_display(c, axis)
    column_sum = np.zeros_like(axis)
    for part, _, offset in _rb_components(c):
        cross = weak_cross_sections(part.isotope, part.line, part.width, c.field_ut,
                                    c.probe_jones, axis - offset, temperature_k=part.temperature,
                                    waist_m=part.waist, thermal=True)
        column_sum += c.cell_mm * 1e-3 * c.line_strength * part.density * cross
    np.testing.assert_allclose(result.optical_depth, column_sum, rtol=7e-15)
    np.testing.assert_array_equal(result.optical_depth, result.od0)
    zero_field = replace(c, od_field_x_ut=0., od_field_y_ut=0., od_field_z_ut=0.)
    assert max(abs(result.transmission - evaluate_od_display(zero_field, axis).transmission)) > 1e-4
    indices = [1550, 73, 1550, 674]
    fit = od_transmission(axis[indices] / (2*np.pi*1e6), c.to_params(), model_id=c.model_id)
    np.testing.assert_allclose(fit, result.transmission[indices], rtol=5e-14)
    empty = evaluate_od_display(replace(c, cell_mm=0., probe_power_uw=1.), axis[indices])
    np.testing.assert_array_equal(empty.transmission, np.ones(len(indices)))


def test_cancellation_interrupts_real_finite_path_before_trajectories(monkeypatch):
    from gabes import field_spectrum
    c = ODConditions(species.RB87.label, "D1", 45., 2., FIELD_HISTORY_REFERENCE,
                     od_field_x_ut=50., probe_power_uw=.1)
    def unexpected(*args):
        pytest.fail("cancelled field calculation entered a trajectory")
    monkeypatch.setattr(field_spectrum, "_rest_row", unexpected)
    reports = []
    def cancel(status):
        reports.append(status)
        return False
    with field_progress(cancel), pytest.raises(CancelledError):
        evaluate_field_ballistic_spectrum(c, [0.])
    assert reports[0][0:2] == ("Rb87", 0) and reports[0][2] > 0


def test_progress_recorder_can_observe_without_requesting_cancellation():
    from gabes.field_spectrum import _report
    events = []
    with field_progress(events.append):
        _report(species.RB85, 3, 10)
    assert events == [("Rb85", 3, 10)]
