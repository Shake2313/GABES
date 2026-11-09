"""Opt-in empirical pump self-energy in the reduced FWM atomic response.

The real ground-level counterterm is shared by both Floquet gauges through the
per-call atom generator. Optical Rabi couplings and the external frequencies
remain unchanged. A fitted fraction does not identify a microscopic mechanism.
"""
from __future__ import annotations

import numpy as np

from . import constants
from .core import comm_super


def ground_gap_shift(pump_hamiltonian_at_zero, one_photon_detuning_rad_s):
    """Track two distinct ground-like dressed states at zero velocity.

    The maximum joint overlap assignment avoids assigning both bare ground
    states to the same eigenvector near strong mixing. Ambiguous assignments
    remain explicit metadata rather than an unreported branch choice.
    """
    h = np.array(pump_hamiltonian_at_zero, dtype=complex, copy=True)
    detuning = float(one_photon_detuning_rad_s)
    if (h.shape != (4, 4) or not np.isfinite(h).all() or not np.isfinite(detuning)
            or not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12)):
        raise ValueError("a finite Hermitian four-level pump Hamiltonian is required")
    h[2, 2] -= detuning
    h[3, 3] -= detuning
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    overlap = abs(eigenvectors[:2])**2
    assignments = [(float(overlap[0, first]+overlap[1, second]), first, second)
                   for first in range(4) for second in range(4) if first != second]
    assignments.sort(key=lambda item: (-item[0], item[1], item[2]))
    score, first, second = assignments[0]
    weights = [float(overlap[0, first]), float(overlap[1, second])]
    margin = float(score-assignments[1][0])
    shift = float(eigenvalues[first]-eigenvalues[second]+constants.OMEGA_HF)
    return dict(shift_rad_s=shift, shift_mhz=shift/(2*np.pi*1e6),
                ground_eigenstate_indices=[first, second], ground_overlaps=weights,
                joint_assignment_margin=margin,
                assignment_ambiguous=bool(margin < 1e-8 or min(weights) < .5),
                assignment_scope="maximum joint bare-ground overlap; zero-velocity closed pump Hamiltonian")


def apply(atom, pump_hamiltonian_at_zero, one_photon_detuning_rad_s, *,
          fraction=1.0, pump_power_W=None):
    """Add ``comm_super(C|g2><g2|)`` to the supplied per-call atom.

    ``C=(1-fraction)*native_dressed_gap_shift`` is an empirical effective
    self-energy. It is a linear counterterm, not an exact scaling of the final
    dressed gap. Fraction one and a zero pump do not modify the generator.
    """
    retained = float(fraction)
    if not np.isfinite(retained) or not 0. <= retained <= 1.:
        raise ValueError("pump_stark_fraction must be finite and between 0 and 1")
    if pump_power_W is not None:
        power = float(pump_power_W)
        if not np.isfinite(power) or power < 0.:
            raise ValueError("pump power must be finite and nonnegative")
    native = ground_gap_shift(pump_hamiltonian_at_zero, one_photon_detuning_rad_s)
    pump_off = pump_power_W is not None and float(pump_power_W) == 0.
    counterterm = 0. if retained == 1. or pump_off else (1.-retained)*native["shift_rad_s"]
    active = counterterm != 0.
    corrected = native
    if active:
        term = np.zeros((4, 4), dtype=complex)
        term[1, 1] = counterterm
        atom.lindblad = atom.lindblad + comm_super(term)
        corrected = ground_gap_shift(np.asarray(pump_hamiltonian_at_zero)+term,
                                     one_photon_detuning_rad_s)
    metadata = dict(
        fraction=retained, applied=active, pump_zero=pump_off,
        native_shift_rad_s=native["shift_rad_s"], native_shift_mhz=native["shift_mhz"],
        counterterm_rad_s=float(counterterm), counterterm_mhz=float(counterterm/(2*np.pi*1e6)),
        corrected_gap_shift_rad_s=corrected["shift_rad_s"], corrected_gap_shift_mhz=corrected["shift_mhz"],
        native_assignment=native, corrected_assignment=corrected,
        assignment_ambiguous=bool(native["assignment_ambiguous"] or corrected["assignment_ambiguous"]),
        formula="C=(1-s)*delta_Stark0; H += C|g2><g2|; optical Rabi couplings unchanged",
        correction_scope="empirical effective self-energy; zero-velocity dressed-gap feature; not an exact final-gap scaling",
        response_scope="same per-call generator for pump populations and all susceptibility components",
        noise_covariance_calibrated=False, susceptibility_identified=False)
    return atom, metadata
