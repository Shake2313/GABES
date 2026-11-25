"""Beam transit rates in s⁻¹. The calling model selects the relaxation channel."""
import math

from . import constants


def rate_from_hz_over_2pi(value):
    """Convert a displayed γ/(2π) in Hz to the exponential rate γ in s⁻¹."""
    return 2.0 * math.pi * value


def rate_from_khz_over_2pi(value):
    """Convert a displayed γ/(2π) in kHz to the exponential rate γ in s⁻¹."""
    return 2.0 * math.pi * value * 1e3


def most_probable_transit_rate_s(temperature_k, diameter_m, mass_kg, *, factor=1.0):
    """Rydberg convention: factor times the most-probable speed over the diameter."""
    speed = math.sqrt(2.0 * constants.KB * float(temperature_k) / float(mass_kg))
    return float(factor) * speed / diameter_m


def gaussian_transit_rate_s(waist_m, temperature_k, mass_kg):
    """Mean replacement rate in s⁻¹ for ballistic atoms in a circular Gaussian beam.

    Häupl et al., New J. Phys. 27 (2025), Eq. 8, doi:10.1088/1367-2630/adb77c.
    The beam diameter is its *intensity* FWHM, w sqrt(2 ln 2), with w the
    1/e² intensity radius in metres. T is the kinetic temperature in kelvin.
    """
    if not all(math.isfinite(v) and v > 0. for v in (waist_m, temperature_k, mass_kg)):
        raise ValueError("Beam waist, kinetic temperature and isotope mass must be finite and positive.")
    diameter_fwhm_m = waist_m * math.sqrt(2. * math.log(2.))
    speed_sigma = constants.thermal_speed(temperature_k, mass_kg, sqrt=math.sqrt)
    return math.sqrt(8. / math.pi) * speed_sigma / diameter_fwhm_m
