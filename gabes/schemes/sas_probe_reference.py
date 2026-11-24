"""Single-pass transmission with a hyperfine-projected, finite-power Rb probe."""
from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline

from .. import constants, observables, probe_saturation, species
from ..core import blas_single_thread
from ..optical_pumping import ProjectedProbe


TRANSMISSION_TOL = 2e-5
SCALED_DEPTH_TOL = 2e-4
WEAK_TAIL_TOL = 3e-6
MAX_SECONDS = 30.0
FREQUENCY_CHUNK = 256
# Native detuning spacing (MHz), log-intensity solve and propagation spacing,
# and radial order. Two resolutions are compared before returning any curve.
GRIDS = ((10.0, 0.10, 0.0125, 64), (5.0, 0.05, 0.00625, 128),
         (2.5, 0.025, 0.003125, 256))


def _closed_grid(lo, hi, max_step):
    return np.linspace(lo, hi, int(np.ceil((hi - lo) / max_step)) + 1)


def _check_time(deadline):
    if perf_counter() >= deadline:
        raise TimeoutError("The 30 s reference calculation limit was reached.")


def _propagate(components, scan, alpha_linear, intensity, tail, length, grid,
               deadline):
    freq_mhz, dx_solve, dx_propagate, rays = grid
    detuning = _closed_grid(scan[0], scan[-1], 2 * np.pi * 1e6 * freq_mhz)
    x = _closed_grid(0.0, tail, dx_solve)
    x_propagate = _closed_grid(0.0, tail, dx_propagate)
    intensities = intensity * np.exp(-x)
    alpha = np.zeros((x.size, detuning.size))
    for probe, offset, sigma, density in components:
        _check_time(deadline)
        alpha += density * probe.absorption_cross_section(
            intensities, detuning - offset, sigma, deadline=deadline)
    if not np.all(np.isfinite(alpha)) or not np.all(alpha > 0.0):
        raise FloatingPointError("The projected absorption table failed the passivity check.")

    # Interpolate log absorption to preserve positive cross sections. Bound the
    # dense propagation table by processing independent frequency columns.
    by_detuning = CubicSpline(detuning, np.log(alpha), axis=1)
    transmission = np.empty_like(scan)
    depth = np.empty_like(scan)
    tail_error = 0.0
    for start in range(0, scan.size, FREQUENCY_CHUNK):
        _check_time(deadline)
        block = slice(start, start + FREQUENCY_CHUNK)
        log_alpha = by_detuning(scan[block])
        endpoint = np.exp(log_alpha[-1])
        tail_error = max(tail_error, float(np.max(abs(endpoint / alpha_linear[block] - 1.0))))
        dense_alpha = np.exp(CubicSpline(x, log_alpha, axis=0)(x_propagate))
        transmission[block], effective_alpha = probe_saturation.propagate_gaussian_spectrum(
            x_propagate, dense_alpha, alpha_linear[block], length, radial_order=rays)
        depth[block] = observables.optical_depth(effective_alpha, length)
    _check_time(deadline)
    resolution = dict(detuning_nodes=int(detuning.size), intensity_nodes=int(x.size),
                      propagation_nodes=int(x_propagate.size), radial_order=rays)
    return transmission, depth, tail_error, resolution


def _bounded_parameter(params, key, lower, upper):
    value = float(params[key])
    if not lower <= value <= upper:
        raise ValueError(f"{key} must be between {lower:g} and {upper:g} for the probe reference.")
    return value


def compute(params, isotope_densities, reference, temperature_k, probe_mw):
    """Compute the explicit advanced view after SAS has checked its scope."""
    deadline = perf_counter() + MAX_SECONDS
    with blas_single_thread():
        return _compute(params, isotope_densities, reference, temperature_k, probe_mw, deadline)


def _compute(params, isotope_densities, reference, temperature_k, probe_mw, deadline):
    line = params["line"]
    from .sas import per_line_self_broadening
    per_line = per_line_self_broadening(params)
    probe_waist = _bounded_parameter(params, "probe_waist_mm", 0.1, 5.0)
    transit_khz = _bounded_parameter(params, "transit_khz", 5.0, 2000.0)
    strength = _bounded_parameter(params, "line_strength", 0.01, 2.0)
    length_m = _bounded_parameter(params, "cell_mm", 0.5, 200.0) * 1e-3
    count = _bounded_parameter(params, "scan_points", 2, 4001)
    if int(count) != count:
        raise ValueError("Scan points must be an integer.")
    intensity = probe_saturation.gaussian_peak_intensity(probe_mw, probe_waist)
    transit = 2 * np.pi * transit_khz * 1e3
    nu_ref = reference.line(line)[1]

    components = []
    centers = []
    widths = {}
    fractions = {}
    margin = 0.0
    pumping_scale = 0.0
    for isotope, fraction, density in isotope_densities:
        man = species.build_manifold(isotope, line, transit_rate=transit)
        offset = 2 * np.pi * (man.nu0 - nu_ref)
        sigma = man.k_vec * constants.thermal_speed(temperature_k, isotope.mass)
        hwhm = (man.gamma + species.self_broadening(
            isotope, density, line, per_line=per_line)) / 2 + transit
        probe = ProjectedProbe(man, hwhm, transit)
        saturation = probe.peak_saturation(intensity)
        pumping_scale = max(pumping_scale, saturation * (man.gamma + transit) / transit)
        margin = max(margin, 3.5 * np.sqrt(8 * np.log(2)) * sigma,
                     12 * hwhm * np.sqrt(1.0 + saturation))
        components.append((probe, offset, sigma, density * strength))
        centers.extend(man.omega + offset)
        widths[isotope.name] = float(hwhm / (np.pi * 1e6))
        fractions[isotope.name] = float(fraction)

    scan = np.linspace(min(centers) - margin, max(centers) + margin, int(count))
    alpha_linear = sum(density * probe.linear_cross_section(scan - offset, sigma)
                       for probe, offset, sigma, density in components)
    linear_depth = observables.optical_depth(alpha_linear, length_m)
    linear_transmission = observables.transmission(alpha_linear, length_m)
    length_ratio = probe_saturation.cell_to_rayleigh_ratio(
        length_m, probe_waist, 2 * np.pi * nu_ref / constants.C_LIGHT)
    notes = ["Numerical reference for linearly polarized Rb D1/D2. Magnetic sublevels "
             "within each hyperfine level share equal populations."]
    if strength != 1.0:
        notes.append(f"Absorption is scaled by {strength:g}. At a given local intensity, "
                     "the atomic drive and populations are unchanged.")
    if length_ratio >= 0.5:
        notes.append(f"Cell length is {length_ratio:.2g} Rayleigh ranges. "
                     "The calculation assumes a constant beam waist and full collection.")
    result = dict(
        x_ghz=scan / (2 * np.pi * 1e9), linear_transmission=linear_transmission,
        provenance=dict(model="hyperfine_projected_probe_obe", validation="numerical readiness",
                        polarization="linear", frequency_reference=reference.name,
                        isotope_fractions=fractions, homogeneous_fwhm_mhz=widths,
                        transit_khz=transit_khz, peak_probe_intensity_w_m2=float(intensity),
                        line_strength=dict(factor=strength, applies_to="propagated absorption coefficient"),
                        linear_limit="complex Voigt with natural, self and transit broadening",
                        beam="constant Gaussian waist, full collection"),
        notes=notes,
    )
    if intensity == 0.0:
        return dict(result, status="converged", transmission=linear_transmission,
                    tau=linear_depth, refinement=dict(levels=0, method="analytic linear limit"))

    # The weak tail is set by optical pumping versus transit, rather than by
    # the closed-transition saturation parameter alone.
    tail = max(12.0, float(np.log1p(pumping_scale) + 13.0))
    refinement = dict(levels=0, transmission_tolerance=TRANSMISSION_TOL,
                      scaled_tau_tolerance=SCALED_DEPTH_TOL, tail_tolerance=WEAK_TAIL_TOL)
    previous = None
    try:
        for index, grid in enumerate(GRIDS, start=1):
            transmission, depth, tail_error, resolution = _propagate(
                components, scan, alpha_linear, intensity, tail, length_m, grid, deadline)
            refinement.update(levels=index, tail_max_relative_alpha_delta=tail_error,
                              grid=resolution)
            if previous is not None:
                transmission_delta = float(np.max(abs(transmission - previous[0])))
                depth_delta = float(np.max(abs(depth - previous[1]) / np.maximum(1.0, depth)))
                refinement.update(max_abs_transmission_delta=transmission_delta,
                                  max_scaled_tau_delta=depth_delta, previous_grid=previous[2])
                if (transmission_delta <= TRANSMISSION_TOL and depth_delta <= SCALED_DEPTH_TOL
                        and tail_error <= WEAK_TAIL_TOL):
                    notes.append(f"Grid check: max transmission change {transmission_delta:.2g}.")
                    return dict(result, status="converged", transmission=transmission,
                                tau=depth, refinement=refinement)
            previous = transmission, depth, resolution
    except (ArithmeticError, np.linalg.LinAlgError, TimeoutError) as exc:
        return dict(result, status="unresolved", reason=str(exc), refinement=refinement)
    else:
        return dict(result, status="unresolved", refinement=refinement,
                    reason="The reference did not meet the grid convergence tolerances for these inputs.")


def render(data):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.35))
    ax.plot(data["x_ghz"], data["transmission"], label="Hyperfine pumping reference")
    if data["provenance"]["peak_probe_intensity_w_m2"] > 0.0:
        ax.plot(data["x_ghz"], data["linear_transmission"], "--", label="Weak-probe limit")
        ax.legend()
    ax.set(xlabel="Relative frequency [GHz] (ref: line centroid)", ylabel="Transmission",
           ylim=(-0.02, 1.02))
    fig.tight_layout()
    return fig
