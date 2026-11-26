"""Jones-frame fields map onto the angular-momentum frame (zeeman.field_from_jones_frame).

The circular drive weights omit the Condon-Shortley sign of q=+1, so Jones-x light
commutes with F_y of zeeman.angular_momentum_matrices. A field given along the
light's Jones x axis must therefore act as F_y. Found 2026-11-26 while testing
gabes.field_history; fixed at user request the same day.
"""
import numpy as np
import pytest

from gabes import magneto_multilevel as mm, zeeman
from gabes.psr_conventions import jones_to_circular
from gabes.schemes import magneto as mg


def _block(states, direction):
    n = len(states)
    out = np.zeros((n, n), complex)
    i = 0
    while i < n:
        f = states[i][0]
        k = int(round(2 * f + 1))
        fx, fy, fz = zeeman.angular_momentum_matrices(f)
        out[i:i + k, i:i + k] = direction[0] * fx + direction[1] * fy + direction[2] * fz
        i += k
    return out


def _commutator_norm(weights, jones_axis):
    atom = mm.build_multilevel_atom("Rb87", "D1")
    coupling = sum(weights[q] * atom.dipoles[q].T for q in (-1, 1))
    direction = np.asarray(zeeman.field_from_jones_frame(*jones_axis), dtype=float)
    return np.linalg.norm(coupling @ _block(atom.ground, direction)
                          - _block(atom.excited, direction) @ coupling)


@pytest.mark.parametrize("weights, axis", [
    (jones_to_circular(np.array([1.0, 0.0])), (1.0, 0.0, 0.0)),
    (jones_to_circular(np.array([0.0, 1.0])), (0.0, 1.0, 0.0)),
    (jones_to_circular(np.array([1.0, 1.0]) / np.sqrt(2)), (1.0, 1.0, 0.0)),
    (mm._polarization(0.0), (1.0, 0.0, 0.0)),
    ({q: w / np.sqrt(2) for q, w in mg._qwp_drive_weights(0.0).items()}, (1.0, 0.0, 0.0)),
])
def test_field_along_linear_light_commutes_with_its_coupling(weights, axis):
    assert _commutator_norm(weights, axis) < 1e-12
    perpendicular = (-axis[1], axis[0], 0.0)
    assert _commutator_norm(weights, perpendicular) > 0.1


def test_zero_components_stay_positive_zero():
    bx, by, bz = zeeman.field_from_jones_frame(0.0, 0.0, 0.0)
    assert (bx, by, bz) == (0.0, 0.0, 0.0)
    assert np.copysign(1.0, bx) == 1.0


def test_multilevel_field_parallel_to_qwp0_light_leaves_transmission_unchanged():
    kw = dict(isotope="Rb87", line="D1", ground_f=2, intensity_mw_cm2=0.1, temp_c=20.0,
              b_ut=[0.0], doppler=False, qwp_deg=0.0, magnetic_derivative=False,
              ground_relaxation_hz=100.0, wall_relaxation_hz=100.0, gamma_out_hz=1e3)

    def transmission(bx, by):
        r = mm.multilevel_response([0.0], b_x_ut=bx, b_y_ut=by, **kw)
        return float(np.asarray(r["transmission"]).ravel()[0])

    zero = transmission(0.0, 0.0)
    assert transmission(1.0, 0.0) == pytest.approx(zero, rel=1e-12)
    assert abs(transmission(0.0, 1.0) / zero - 1) > 1e-8
