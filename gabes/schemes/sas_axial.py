"""Single-pass axial OD/SAS reference using the local species line response."""

import functools

import numpy as np

from .. import constants, observables, species, vapor_cell
from ..lineshape import subdoppler_feature

class AxialInputError(ValueError):
    """Unsupported or inconsistent parameters for the axial reference."""


SEGMENTED = "Segmented reference"
DEFAULT_SEGMENTS = 4
MAX_FIELD_UT = 200.0
SEGMENT_RTOL = 0.01
POWER_RTOL = 0.01
_POWER_COORDINATE = np.array([0., 0.125, 0.25, 0.5, 0.75, 1.])


def _finite(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise AxialInputError(f"{name} must be finite.")
    return result


def _profile_settings(params):
    count = params.get("axial_segments", DEFAULT_SEGMENTS)
    if isinstance(count, bool) or not np.isfinite(count) or int(count) != count or not 1 <= count <= 16:
        raise AxialInputError("Axial segment count must be an integer from 1 to 16.")
    center = (params["temp_c"] if params.get("constrain_doppler_temp", True)
              else params.get("doppler_temp_c", params["temp_c"]))
    temp = _finite(center, "Axial center temperature")
    gradient = _finite(params.get("axial_temperature_span_c", 0.), "Axial temperature span")
    field = _finite(params.get("axial_center_field_ut", 0.), "Axial center field")
    field_span = _finite(params.get("axial_field_span_ut", 0.), "Axial field span")
    if temp - 0.5 * abs(gradient) < float(params["temp_c"]):
        raise AxialInputError("Axial temperature endpoints must be at or above the cold spot. "
                              "Release the Doppler temperature tie and raise its value.")
    if abs(field) + 0.5 * abs(field_span) > MAX_FIELD_UT:
        raise AxialInputError(f"Axial field endpoints must be within ±{MAX_FIELD_UT:g} µT.")
    return int(count), temp, gradient, field, field_span


@functools.lru_cache(maxsize=12)
def _zeeman_pattern(isotope, line):
    structure = species.level_structure(isotope.name, line)
    excited_j = isotope.line(line)[0]
    excited_gj = isotope.gJ_P12 if line == "D1" else isotope.gJ_P32
    ground = {(f, m): species.lande_gf(isotope.gJ_S, isotope.g_I, isotope.I, isotope.Jg, f) * m
              for f, m in structure.ground}
    excited = {(f, m): species.lande_gf(excited_gj, isotope.g_I, isotope.I, excited_j, f) * m
               for f, m in structure.excited}
    result = {}
    for polarization in (-1, 1):
        dipoles = structure.dipoles[polarization]
        for g, e in zip(*np.nonzero(dipoles)):
            fg, mg = structure.ground[g]
            fe, me = structure.excited[e]
            result.setdefault((fg, fe), []).append((
                excited[fe, me] - ground[fg, mg], float(dipoles[g, e] ** 2)))
    return result


def zeeman_components(manifold, field_ut):
    """Expand parent lines into linear-polarization sigma components.

    Populations stay hyperfine-lumped and uniform within each magnetic
    sublevel. Only the optical line frequencies and strengths are resolved.
    """
    indices = np.arange(manifold.omega.size)
    if field_ut == 0.0:
        return manifold.omega, indices, np.ones(indices.size)
    patterns = _zeeman_pattern(manifold.iso, manifold.line)
    multiplier = constants.MU_B / constants.HBAR * field_ut * 1e-6
    centers, parents, weights = [], [], []
    ng = len(manifold.Fg)
    for index, (g, e) in enumerate(zip(manifold.g_idx, manifold.e_idx)):
        pattern = patterns[manifold.Fg[g], manifold.Fe[e - ng]]
        shifts, strengths = np.asarray(pattern).T
        centers.extend(manifold.omega[index] + multiplier * shifts)
        parents.extend([index] * strengths.size)
        weights.extend(strengths / strengths.sum())
    return np.asarray(centers), np.asarray(parents), np.asarray(weights)


def _interpolate_columns(axis, values, target):
    indices = np.clip(np.searchsorted(axis, target, side="right") - 1, 0, axis.size - 2)
    fraction = (target - axis[indices]) / (axis[indices + 1] - axis[indices])
    column = np.arange(target.size)
    lower = values[:, indices, column]
    upper = values[:, indices + 1, column]
    return lower + fraction * (upper - lower)


def _propagate(count, length, temp, gradient, field, field_span, params,
               components, reference, response, scan, power, *, deplete, power_axis):
    edges = np.linspace(0., length, count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    normalized = centers / length - 0.5
    temperature = temp + gradient * normalized
    field_profile = field + field_span * normalized
    # Include the endpoints in the pressure check so a coarse grid cannot
    # silently hide a beam-path endpoint below the declared cold reservoir.
    profile_z = np.concatenate(([0.], centers, [length]))
    profile_t = np.concatenate(([temp - gradient / 2], temperature, [temp + gradient / 2]))
    densities = {}
    for isotope, weight in components:
        profile = vapor_cell.axial_cell_profile(
            profile_z, profile_t, isotope=isotope,
            pressure_cold_spot_temp_c=float(params["temp_c"]))
        densities[isotope.name] = profile.density_m3[1:-1] * weight

    probe_alpha = np.empty((count, scan.size))
    probe_chi = np.empty_like(probe_alpha)
    pump = np.full(scan.size, power)
    pump_rows = [pump.copy()]
    most_negative_pump_alpha = 0.0
    line = params["line"]
    waist = float(params["waist_mm"])
    transit = 2.0 * np.pi * float(params["transit_khz"]) * 1e3
    neon = constants.neon_buffer_broadening(params.get("ne_pressure_torr", 0.))
    nu_reference = reference.line(line)[1]
    prepared = [(isotope, species.build_manifold(isotope, line, transit_rate=transit))
                for isotope, _ in components]
    line_strength = float(params["line_strength"])
    for index in range(count - 1, -1, -1):
        tk = temperature[index] + 273.15
        values = []
        for node_power in power_axis if deplete else (power,):
            local = np.zeros((3, scan.size))
            for isotope, manifold in prepared:
                density = densities[isotope.name][index]
                local += response(dict(
                    man=manifold, iso=isotope, N=density,
                    offset=2.0*np.pi*(manifold.nu0 - nu_reference),
                    Op=species.pump_rabi_from_power(node_power, waist, manifold.gamma,
                                                   iso=isotope, line=line),
                    # A weak-probe Legacy-grid path: the historical width in every set.
                    gamma_eff=manifold.gamma + species.self_broadening(isotope, density) + neon,
                    field_ut=field_profile[index]), scan, tk,
                    axial_optics=True, pump_attenuation=deplete)
            values.append(local)
        if deplete:
            lookup = np.stack(values, axis=1)
            dz = edges[index + 1] - edges[index]
            entrance = _interpolate_columns(power_axis, lookup, pump)[2] * line_strength
            most_negative_pump_alpha = min(most_negative_pump_alpha, float(entrance.min()))
            # Explicit midpoint step: sample local pump after a half-cell
            # Beer-Lambert step, then propagate the full cell from its entrance.
            middle_power = pump * observables.transmission(np.maximum(entrance, 0.), 0.5 * dz)
            alpha, chi, pump_alpha = _interpolate_columns(power_axis, lookup, middle_power)
            pump_alpha *= line_strength
            most_negative_pump_alpha = min(most_negative_pump_alpha, float(pump_alpha.min()))
            pump *= observables.transmission(np.maximum(pump_alpha, 0.), dz)
        else:
            alpha, chi, _ = values[0]
        probe_alpha[index], probe_chi[index] = alpha, chi
        pump_rows.append(pump.copy())

    integral = vapor_cell.integrate_beer_lambert(edges, probe_alpha, sample_location="midpoints")
    dz = np.diff(edges)[:, None]
    return dict(alpha_unit=integral.optical_depth / length,
                chi_real_unit=np.sum(probe_chi * dz, axis=0) / length,
                optical_depth_unit=integral.optical_depth,
                most_negative_pump_alpha_per_m=most_negative_pump_alpha,
                pump_power_mw_z=np.asarray(pump_rows[::-1]),
                z_edges_m=edges, z_centers_m=centers, temperature_c=temperature,
                field_ut=field_profile, density_m3=densities)


def _relative_change(actual, previous):
    scale = max(float(np.max(np.abs(actual))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - previous))) / scale


def _convergence(actual, previous, input_power, line_strength):
    transmission = observables.transmission(actual["optical_depth_unit"], line_strength)
    previous_transmission = observables.transmission(previous["optical_depth_unit"], line_strength)
    return dict(alpha=_relative_change(actual["alpha_unit"], previous["alpha_unit"]),
                dispersion=_relative_change(actual["chi_real_unit"], previous["chi_real_unit"]),
                transmission_absolute=float(np.max(np.abs(transmission - previous_transmission))),
                pump_output=float(np.max(np.abs(actual["pump_power_mw_z"][0]
                                               - previous["pump_power_mw_z"][0]))) / input_power
                if input_power else 0.)


def _lamb_dip_depths(x, transmission, markers):
    """Isolate positive residual peaks at resolved, labelled transition centers."""
    depths = {}
    for index, (center, label) in enumerate(markers):
        nearest = min((abs(center - other) for other, _ in markers if other != center), default=0.1)
        half = min(0.025, 0.18 * nearest)
        feature = subdoppler_feature(x, transmission, (center - half, center + half))
        if (feature.resolved and feature.polarity == 1
                and abs(feature.center - center) < half / 2.0):
            depths[index] = (label, float(feature.amplitude))
    return depths


def compute(params, *, uniform, components, reference, response):
    """Compose local SAS coefficients along z and report refinement evidence."""
    power = _finite(params["pump_power_mw"], "Pump power")
    length = 1e-3 * _finite(params.get("cell_mm", 75.), "Cell length")
    if power < 0. or length <= 0.:
        raise AxialInputError("Axial reference requires nonnegative pump power and positive cell length.")
    line_strength = _finite(params["line_strength"], "Line strength")
    if line_strength <= 0.0:
        raise AxialInputError("Axial reference requires positive line strength.")
    count, temperature, gradient, field, field_span = _profile_settings(params)
    deplete = bool(params.get("axial_pump_depletion", False)) and power > 0.
    probe_power_mw = float(params.get("probe_power_mw", float(params.get("probe_power_uw", 0.)) * 1e-3))
    if probe_power_mw != 0.:
        raise AxialInputError("Segmented reference requires 0 µW probe power (weak probe).")
    if power > 0. and params.get("paraffin_coated", False):
        raise AxialInputError("Segmented reference does not combine a pumped paraffin reservoir with axial gradients.")

    raw = uniform(params)
    scan = raw["scan"]
    state = dict(segment_count=count, default_segment_count=DEFAULT_SEGMENTS,
                 pump_depletion=deplete, length_m=length,
                 reference_line_strength=line_strength,
                 temperature_source="user-declared linear beam-path profile; cold reservoir fixes vapor pressure",
                 field_source="user-declared longitudinal linear magnetic-field profile",
                 pump_direction="single pass from probe exit to probe entrance",
                 pump_absorption_scope="local copropagating absorption; negative coefficients clipped to zero",
                 field_model="linear Zeeman optical frequencies and sigma strengths; hyperfine-lumped pump populations",
                 segment_relative_tolerance=SEGMENT_RTOL, pump_grid_relative_tolerance=POWER_RTOL)
    flat = gradient == 0. and field == 0. and field_span == 0. and not deplete
    if flat:
        edges = np.linspace(0., length, count + 1)
        state.update(status="exact_uniform_limit", z_edges_m=edges,
                     z_centers_m=0.5*(edges[:-1] + edges[1:]),
                     temperature_c=np.full(count, temperature), field_ut=np.zeros(count),
                     pump_power_mw_z=np.full((count + 1, scan.size), power),
                     density_m3={iso.name: np.full(count, weight * vapor_cell.cold_spot_limited_density(
                         species.number_density(iso, float(params["temp_c"]) + 273.15),
                         float(params["temp_c"]) + 273.15, temperature + 273.15))
                         for iso, weight in components})
        raw["axial_cell"] = state
        return raw

    def evaluate(n, nodes, depletion):
        return _propagate(n, length, temperature, gradient, field, field_span, params,
                          components, reference, response, scan, power,
                          deplete=depletion, power_axis=nodes * power)

    base_axis = _POWER_COORDINATE
    fine_axis = np.sort(np.concatenate((base_axis, 0.5*(base_axis[1:] + base_axis[:-1]))))
    actual = evaluate(count, fine_axis, deplete)
    doubled = evaluate(2 * count, fine_axis, deplete)
    segment_error = _convergence(doubled, actual, power, line_strength)
    power_error = (_convergence(actual, evaluate(count, base_axis, deplete), power, line_strength)
                   if deplete else None)
    converged = (max(segment_error.values()) <= SEGMENT_RTOL
                 and (power_error is None or max(power_error.values()) <= POWER_RTOL))
    state.update(status="converged" if converged else "refinement_limited",
                 segment_refinement_count=2 * count, segment_change=segment_error,
                 profiles_constrained_by_fit=False,
                 pump_power_nodes=fine_axis.size if deplete else 1,
                 coarse_pump_power_nodes=base_axis.size if deplete else 1,
                 pump_grid_change=power_error,
                 **{key: value for key, value in actual.items()
                    if key not in ("alpha_unit", "chi_real_unit")})
    raw["axial_cell"] = state
    raw["axial_uniform_alpha_unit"] = raw["alpha_unit"]
    if deplete:
        raw["axial_undepleted_alpha_unit"] = (
            evaluate(count, fine_axis, False)["alpha_unit"]
            if gradient or field or field_span else raw["alpha_unit"])
    raw.update(alpha_unit=actual["alpha_unit"], chi_real_unit=actual["chi_real_unit"])
    return raw


def readout(raw, params, x, transmission):
    """Carry profile provenance and quantify changes with matched baselines."""
    evidence = raw["axial_cell"].copy()
    length = evidence["length_m"]
    scale = float(params["line_strength"])
    if evidence["pump_depletion"]:
        evidence["minimum_pump_output_fraction"] = float(
            np.min(evidence["pump_power_mw_z"][0]) / params["pump_power_mw"])
    current = subdoppler_feature(x, transmission) if params["pump_power_mw"] > 0 else None
    for name, raw_key in (("uniform", "axial_uniform_alpha_unit"),
                          ("undepleted", "axial_undepleted_alpha_unit")):
        if raw_key not in raw:
            continue
        baseline = observables.transmission(raw[raw_key] * scale, length)
        base_feature = subdoppler_feature(x, baseline) if current is not None else None
        delta = transmission - baseline
        contrast_delta = None
        if (current is not None and current.detected and base_feature.detected
                and current.polarity == base_feature.polarity
                and abs(current.center - base_feature.center) <= max(current.fwhm, base_feature.fwhm)):
            contrast_delta = float(current.amplitude - base_feature.amplitude)
        comparison = dict(transmission_absolute_max=float(np.max(np.abs(delta))),
                          matched_subdoppler_contrast_change=contrast_delta,
                          matched_feature_polarity=current.polarity if contrast_delta is not None else None)
        if name == "undepleted":
            depths = _lamb_dip_depths(x, transmission, raw["markers"])
            base_depths = _lamb_dip_depths(x, baseline, raw["markers"])
            comparison["matched_lamb_dips"] = [
                dict(transition=label, depth=depth, baseline_depth=base_depths[i][1],
                     depth_change=depth - base_depths[i][1])
                for i, (label, depth) in depths.items() if i in base_depths]
        evidence[f"difference_from_{name}"] = comparison
    return evidence
