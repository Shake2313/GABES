"""Beam transit rates in s⁻¹. The calling model selects the relaxation channel."""
import math

from . import constants


def rate_from_khz_over_2pi(value):
    """Convert a displayed γ/(2π) in kHz to the exponential rate γ in s⁻¹."""
    return 2.0 * math.pi * value * 1e3


def most_probable_transit_rate_s(temperature_k, diameter_m, mass_kg, *, factor=1.0):
    """Rydberg convention: factor times the most-probable speed over the diameter."""
    speed = math.sqrt(2.0 * constants.KB * float(temperature_k) / float(mass_kg))
    return float(factor) * speed / diameter_m
