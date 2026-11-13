"""Velocity measure shared by the squeezing scanners."""

from gabes.schemes import fwm

CONVERGED_VELOCITY_GRID = (1.0, 4.0)


def velocity_measure(response_method, step, cutoff):
    """Select the exact pole mean or the explicitly sampled Maxwell grid.

    The scanners use the converged grid settings for new runs and their
    original grid settings for historical runs. The pole solver can integrate
    the converged measure analytically; the grid solver uses the sampled grid.
    """
    if response_method != fwm.RESPONSE_GRID and (step, cutoff) == CONVERGED_VELOCITY_GRID:
        return "exact"
    return "truncated"
