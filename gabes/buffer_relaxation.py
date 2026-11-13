"""Reference ground-polarization budget for a fully quenching cylinder."""

import math

from . import constants, species
from .collisions import Coefficient, MAX_DEFAULT_RELATIVE_UNCERTAINTY, Measurement

QUENCHING_CYLINDER = "Quenching finite cylinder"
MANUAL_GROUND_RATE = "Manual rate"
_TORR_TO_PA = 101325 / 760
_NE_MASS = 20.1797 * constants.ATOMIC_MASS_UNIT
_REFERENCE_T = 297.15
_DIFFUSION_EXPONENT = 1.756
_J0_FIRST_ZERO = 2.404825557695773
_MAX_KNUDSEN = 0.1


def _at_reference_temperature(value, temperature):
    return value * (_REFERENCE_T / temperature) ** _DIFFUSION_EXPONENT


_NE_DIFFUSION = Coefficient(
    Measurement(0.214, 0.014, "https://doi.org/10.1103/PhysRevA.111.033108",
                "Table II, natural Rb-Ne, 297.15 K and 760 Torr"),
    (
        Measurement(_at_reference_temperature(0.18, 273.15),
                    _at_reference_temperature(0.03, 273.15),
                    "https://doi.org/10.1007/s00340-013-5712-y",
                    "0.18(3) cm^2/s at 273.15 K and 760 Torr, rescaled"),
        Measurement(_at_reference_temperature(0.48, 340.15),
                    _at_reference_temperature(0.10, 340.15),
                    "https://doi.org/10.1103/PhysRev.139.A603",
                    "Table II, 0.48(10) cm^2/s at 340.15 K and 760 Torr, rescaled"),
    ),
)

# Cross section in m^2. The primary measured enriched 21Ne. Other gas
# compositions and spin moments can change the measured disorientation rate.
_SPIN = Measurement(1.9e-27, 0.2e-27, "https://doi.org/10.1103/PhysRevA.81.043415",
                    "Ghosh and Romalis; also Ghosh dissertation Table 5.4")
_SPIN_SPREAD = (
    (3.3e-28, "https://doi.org/10.1103/PhysRev.139.A603"),
    (5.2e-27, "https://doi.org/10.1103/PhysRev.115.850"),
)


def quenching_cylinder_mode(diffusivity_m2_s, radius_m, length_m):
    """Radial and axial wall-loss eigenvalues [s^-1] for Dirichlet walls."""
    return (diffusivity_m2_s * (_J0_FIRST_ZERO / radius_m) ** 2,
            diffusivity_m2_s * (math.pi / length_m) ** 2)


def cylinder_budget(isotope, gas, pressure_torr, temperature_k, beam_radius_m,
                    radius_m, length_m, *, wall="quenching"):
    """Conditional single-mode budget with an explicit calibration status.

    The beam radius sets a transport applicability check. In this lowest
    late-time mode it does not move the absorbing cell wall or add another
    loss rate. Spin destruction is an effective isotropic-reset proxy.
    """
    conditions = dict(isotope=isotope, gas=gas, pressure_torr=pressure_torr,
                      temperature_k=temperature_k, beam_radius_m=beam_radius_m,
                      radius_m=radius_m, length_m=length_m, wall=wall)
    if gas != "Ne" or isotope not in ("Rb85", "Rb87") or wall != "quenching":
        return dict(status="insufficient calibration", conditions=conditions,
                    reason="Only Rb in Ne with a fully quenching cylinder wall is supported.")
    inputs = (pressure_torr, temperature_k, beam_radius_m, radius_m, length_m)
    if any(value is None or not math.isfinite(value) or value <= 0 for value in inputs):
        return dict(status="insufficient calibration", conditions=conditions,
                    reason="Positive pressure, temperature, beam radius and cylinder dimensions are required.")
    if beam_radius_m > radius_m:
        return dict(status="insufficient calibration", conditions=conditions,
                    reason="The centered beam radius exceeds the cylinder radius.")

    scaling = 1e-4 * 760 / pressure_torr * (temperature_k / _REFERENCE_T) ** _DIFFUSION_EXPONENT
    diffusion = scaling * _NE_DIFFUSION.primary.value
    diffusion_u = scaling * _NE_DIFFUSION.uncertainty
    m_rb = species.ISOTOPES[isotope].mass
    mean_rb_speed = math.sqrt(8 * constants.KB * temperature_k / (math.pi * m_rb))
    transport_length = 3 * diffusion / mean_rb_speed
    knudsen = transport_length / min(beam_radius_m, radius_m, length_m)
    if knudsen > _MAX_KNUDSEN:
        return dict(status="insufficient calibration", conditions=conditions,
                    knudsen_proxy=knudsen,
                    reason="The transport length is too large for diffusion on the beam/cell scale.")

    radial, axial = quenching_cylinder_mode(diffusion, radius_m, length_m)
    wall_rate = radial + axial
    wall_u = wall_rate * diffusion_u / diffusion
    number_density = pressure_torr * _TORR_TO_PA / (constants.KB * temperature_k)
    reduced_mass = m_rb * _NE_MASS / (m_rb + _NE_MASS)
    relative_speed = math.sqrt(8 * constants.KB * temperature_k / (math.pi * reduced_mass))
    spin_scale = number_density * relative_speed
    spin_rate = spin_scale * _SPIN.value
    spin_uncertainty = max(_SPIN.uncertainty,
                           *(abs(value - _SPIN.value) for value, _ in _SPIN_SPREAD))
    spin_u = spin_scale * spin_uncertainty
    total = wall_rate + spin_rate
    # No published covariance across source regimes. Sum bound contributions.
    total_u = wall_u + spin_u
    relative = total_u / total
    warnings = [
        "The spin term transfers the enriched 21Ne measurement to Ne as an effective reset.",
        "Geometry and pressure are treated as exact inputs. The coefficient bound omits apparatus error.",
    ]
    if relative > MAX_DEFAULT_RELATIVE_UNCERTAINTY:
        warnings.append(f"Propagated coefficient uncertainty {relative:.0%} exceeds 20%.")
    return dict(
        status="opt-in reference", conditions=conditions, total_s_inverse=total,
        uncertainty_s_inverse=total_u, relative_uncertainty=relative, warnings=warnings,
        knudsen_proxy=knudsen,
        diffusion=dict(
            diffusivity_m2_s=diffusion, uncertainty_m2_s=diffusion_u,
            temperature_exponent=_DIFFUSION_EXPONENT,
            beam_crossing_time_s=beam_radius_m ** 2 / (4 * diffusion),
            sources=[dict(source=m.source, locator=m.locator, value_cm2_s=m.value,
                          uncertainty_cm2_s=m.uncertainty)
                     for m in (_NE_DIFFUSION.primary, *_NE_DIFFUSION.comparisons)]),
        wall_relaxation=dict(rate_s_inverse=wall_rate, uncertainty_s_inverse=wall_u,
                             radial_s_inverse=radial, endcap_s_inverse=axial,
                             model="Dirichlet cylinder fundamental eigenmode"),
        spin_destruction=dict(
            rate_s_inverse=spin_rate, uncertainty_s_inverse=spin_u,
            cross_section_m2=_SPIN.value, cross_section_uncertainty_m2=spin_uncertainty,
            source=_SPIN.source, locator=_SPIN.locator,
            comparison_sources=[dict(source=url, cross_section_m2=value,
                                     uncertainty="not reported for this cross section")
                                for value, url in _SPIN_SPREAD]),
    )
