"""Absolute passive-limit check against the independent Rb-85 D1 OD profile."""

import numpy as np
import pytest

from gabes import beam, constants, hyperfine, linear_hyperfine, observables
from gabes.schemes import fwm


@pytest.mark.parametrize("branch", (-1, +1))
@pytest.mark.parametrize("legacy", (False, True))
def test_pump_off_finite_seed_response_reproduces_passive_line_strengths(branch, legacy):
    temperature = 362.15
    density = hyperfine.number_density(temperature)
    atom = fwm.collisional_atom(temperature, density)
    # Sample both D1 excited lines on both manifolds, at two weak seed offsets.
    one_photon = np.array([0., -constants.OMEGA_EXCITED_HF, -constants.OMEGA_HF,
                           -constants.OMEGA_HF - constants.OMEGA_EXCITED_HF])
    delta = 2*np.pi*1e6*np.array([-11., 9.])
    seed = 100.0  # rad/s, negligible optical pumping relative to thermal reset
    scales = fwm.LEGACY_TRANSITION_DIPOLE_SCALE if legacy else fwm.TRANSITION_DIPOLE_SCALE
    ss, cs, sc, cc = fwm.chi_matrix_table(
        0., 0., seed, seed, delta, one_photon, branch, atom=atom, transition_scales=scales)
    np.testing.assert_allclose([cs, sc], 0., atol=1e-18)
    probe_offset = branch*constants.OMEGA_HF + delta[:, None]
    probe_F = 3 if branch == -1 else 2
    od_origin = 2*np.pi*hyperfine.LINE_SHIFT_HZ[2, 3]
    for chi, ground_F, frequencies in ((ss, probe_F, one_photon + probe_offset),
                                       (cc, 5-probe_F, one_photon - probe_offset)):
        alpha, _ = observables.absorption_coefficient(
            chi, constants.K_VEC, density, line_strength=fwm.physical_coupling_norm(branch, legacy=legacy))
        reference, _, _ = linear_hyperfine.rb85_d1_alpha_faddeeva(
            od_origin + frequencies, temperature, doppler_on=False,
            buffer_gamma=2*atom.transit_reset_rate, lines=((ground_F, 2), (ground_F, 3)))
        # The old summed strength counted the equilibrium manifold fraction twice.
        expected = reference * (hyperfine.GROUND_POP[ground_F] if legacy else 1.)
        np.testing.assert_allclose(alpha, expected, rtol=1e-9, atol=0.)


def test_pump_and_weak_fields_use_the_same_absolute_uniform_manifold_dipoles():
    power, waist = 1.0, 950e-6
    electric_field = beam.field_amplitude(beam.peak_intensity(power, waist))
    rabi = fwm.rabi_freq(power, waist)
    norm = fwm.physical_coupling_norm(-1)
    for g, fg in fwm.GROUND_F.items():
        for e, fe in fwm.EXCITED_F.items():
            drive_dipole = constants.HBAR*rabi*fwm.TRANSITION_DIPOLE_SCALE[g, e]/electric_field
            weak_dipole = constants.DIPOLE_D1*np.sqrt(norm)*fwm.TRANSITION_DIPOLE_SCALE[g, e]
            expected = constants.DIPOLE_D1*np.sqrt(3*hyperfine.CF2[fg, fe]/(2*fg+1))
            np.testing.assert_allclose((drive_dipole, weak_dipole), expected, rtol=1e-14, atol=0.)
