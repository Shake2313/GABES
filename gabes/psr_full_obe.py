"""Compatibility entry point for the original saturated Rb87 D1 PSR atom.

The shared optical Bloch solver owns the physics. This wrapper retains the
original model identifier, dense solve and complete output for existing PSR
mean-field, RF and propagation callers.
"""
from .optical_bloch import solve_optical_pump


MODEL_ID = "psr_rb87_d1_single_velocity_full_obe_v1"


def solve_psr_pump(detuning_mhz, *, intensity_mw_cm2=0.1, jones_xy=(1., 0.),
                   velocity_m_s=0., temp_c=70., ground_relaxation_hz=1000., density_m3=None):
    """Legacy pure-Rb87 D1 zero-field local state; positive detuning is blue.

    Detuning is MHz from the upper-ground-F line-strength centroid. Intensity
    is mW/cm^2 and reset is s^-1. The full state, Hamiltonian, named reservoirs
    and mean polarization retain the established PSR conventions.
    """
    result = solve_optical_pump(
        "Rb87", "D1", detuning_mhz, intensity_mw_cm2=intensity_mw_cm2,
        jones_xy=jones_xy, velocity_m_s=velocity_m_s, temp_c=temp_c,
        ground_relaxation_hz=ground_relaxation_hz, density_m3=density_m3,
        solver="full")
    result["model_id"] = MODEL_ID
    return result
