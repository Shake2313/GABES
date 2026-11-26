"""Per-isotope rest-frame trajectories for a single linear probe in a zero-field Rb cell.

Detunings are angular frequencies relative to this isotope's fine-structure centroid.
The magnetic-sublevel rate projection omits inter-transition coherences. A D2
trajectory from this solver therefore does not establish agreement with a full OBE.
"""
from functools import lru_cache

import numpy as np

from . import constants, species
from .ballistic_pumping import BallisticProbe85D1, _positive, _support
from .optical_pumping import MAGNETIC_PI, ProjectedProbe


@lru_cache(maxsize=4)
def _flight_structure(isotope, line):
    manifold = species.build_manifold(isotope, line)
    # ProjectedProbe stores the stationary generator with a positive reset.
    # Remove exactly that term: the flight has a thermal boundary condition.
    reset = 1.0
    probe = ProjectedProbe(manifold, 0.5 * manifold.gamma + reset, reset, resolution=MAGNETIC_PI)
    thermal = probe._thermal.copy()
    thermal.flags.writeable = False
    free = probe._unforced - reset * (thermal[:, None] - np.eye(thermal.size))

    # The sparse SDIRK stage includes spontaneous branches even when a pi
    # dipole does not connect the excited and ground groups directly.
    ng = probe._nground
    pi_links = np.zeros((thermal.size - ng, ng), dtype=bool)
    pi_links[probe._excited - ng, probe._ground] = True
    up = _support((free[:ng, ng:].T != 0) | pi_links)
    down = _support(pi_links)
    return manifold, probe, thermal, free, up, down


class BallisticProbe(BallisticProbe85D1):
    """One Rb isotope and D line. Paths and phase-space quadratures use the v2 solver.

    The optional intrinsic optical HWHM includes natural and homogeneous
    broadening only. Its default assumes a pure saturated vapor at the kinetic
    temperature. Supply the cell's width for cold-spot or mixture calculations.
    """

    def __init__(self, isotope, line, temperature_k, optical_hwhm_rad_s=None):
        if isotope not in (species.RB85, species.RB87) or line not in ("D1", "D2"):
            raise ValueError("Rb beam-crossing histories require Rb85 or Rb87 and line D1 or D2.")
        if isotope == species.RB85 and line == "D1":
            # The recorded Fast v2 coefficients authenticate this exact solver.
            super().__init__(temperature_k, optical_hwhm_rad_s)
            return

        temp = _positive(temperature_k, "Kinetic temperature (K)")
        self.manifold, self._probe, self.thermal, self._free, self._up, self._down = _flight_structure(isotope, line)
        natural = 0.5 * self.manifold.gamma
        if optical_hwhm_rad_s is None:
            density = species.number_density(isotope, temp)
            optical_hwhm_rad_s = natural + 0.5 * species.self_broadening(isotope, density, line, per_line=True)
        self.width = _positive(optical_hwhm_rad_s, "Intrinsic optical HWHM (rad/s)")
        if self.width < natural:
            raise ValueError("Intrinsic optical width is below the natural HWHM.")
        self.speed_sigma = constants.thermal_speed(temp, isotope.mass)
        self.photon_energy_j = constants.PLANCK * self.manifold.nu0
        self._ng = self._probe._nground
