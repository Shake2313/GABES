"""Reusable single-CW-beam atomic tables, cell fields and camera planes.

All inputs use SI except temperature in Celsius and the inherited reset-rate
name ``ground_relaxation_hz`` (s^-1). The input is a coherent Gaussian at the
CELL ENTRANCE, with a specified radius and wavefront curvature. This adapter
never infers a coherent field from M^2, combines spectral lines, or applies a
second atomic interaction to an already propagated apparatus beam.

The selected D-line full OBE, Maxwell average, local complex interpolation and
FFT propagation are owned by their shared gabes modules. Runtime checks are
reported separately from the registered model-validation campaigns.
"""
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import species
from .beam import peak_intensity
from .nonlinear_response import ComplexSusceptibilityTable, solve_rb_mean
from .optical_frequency import RbComposition, RbOpticalFrequency
from .paraxial import ParaxialGrid, gaussian_field, propagate, relative_lens_power


MODEL_ID = "rb_single_beam_profile_v1"
DARK_TRANSMISSION = 1e-12


def _positive(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
            or int(value) != value or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class AtomicSettings:
    """Hashable atomic conditions; density is TOTAL elemental density in m^-3."""
    frequency: RbOpticalFrequency
    rb85_fraction: float = species.RB85.abundance
    temp_c: float = 70.
    ground_relaxation_hz: float = 5e5
    density_m3: float | None = None
    velocity_order: int = 6
    cutoff_sigma: float = 6.

    def __post_init__(self):
        if not isinstance(self.frequency, RbOpticalFrequency):
            raise TypeError("frequency must have an explicit RbOpticalFrequency reference")
        composition = RbComposition(self.rb85_fraction)
        object.__setattr__(self, "rb85_fraction", composition.rb85_fraction)
        temperature = float(self.temp_c)
        if not np.isfinite(temperature) or temperature <= -273.15:
            raise ValueError("temp_c must be finite and above absolute zero")
        object.__setattr__(self, "temp_c", temperature)
        for name in ("ground_relaxation_hz", "cutoff_sigma"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.density_m3 is not None:
            object.__setattr__(self, "density_m3", _positive(self.density_m3, "density_m3", zero=True))
        object.__setattr__(self, "velocity_order", _integer(self.velocity_order, "velocity_order", 2))

    @property
    def cache_key(self):
        return (MODEL_ID, self.frequency.line, self.frequency.offset_hz,
                self.frequency.reference_hz, self.rb85_fraction, self.temp_c,
                self.ground_relaxation_hz, self.density_m3,
                self.velocity_order, self.cutoff_sigma)

    def as_dict(self):
        return dict(frequency=self.frequency.as_dict(), rb85_fraction=self.rb85_fraction,
                    temp_c=self.temp_c, ground_relaxation_hz=self.ground_relaxation_hz,
                    density_m3=self.density_m3, velocity_order=self.velocity_order,
                    cutoff_sigma=self.cutoff_sigma,
                    density_basis="total elemental density, isotope fractions applied once")


def response_cache_key(settings, max_intensity_w_m2, *, cell_length_m,
                       initial_nodes=17, max_nodes=129, tolerance=1e-3):
    """Exact in-process table/acceptance key; persistent caches also need source hashes.

    Length belongs to the acceptance key because interpolation and quadrature
    errors are measured as optical depth and phase across that length.
    """
    if not isinstance(settings, AtomicSettings):
        raise TypeError("settings must be AtomicSettings")
    bound = _positive(max_intensity_w_m2, "max_intensity_w_m2")
    length = _positive(cell_length_m, "cell_length_m", zero=True)
    initial = _integer(initial_nodes, "initial_nodes", 3)
    maximum = _integer(max_nodes, "max_nodes", initial)
    threshold = _positive(tolerance, "tolerance")
    return settings.cache_key + (bound, length, initial, maximum, threshold)


def build_response_table(settings, max_intensity_w_m2, *, cell_length_m,
                         initial_nodes=17, max_nodes=129, tolerance=1e-3,
                         component_cache=None, progress=None):
    """Adapt chi(I), then check velocity order/cutoff independently.

    ``progress(fraction, message)`` is optional. The separate zero-drive
    susceptibility comes from the shared analytic weak-field limit. Held-out
    midpoint evaluations refine intensity interpolation only. Four quadrature
    probes cover zero, .0025, .25 and 1 times the supplied intensity ceiling.
    Failed checks are returned explicitly; a table is never silently certified.
    A caller may cache this bundle by :func:`response_cache_key`. The mutable
    component cache is strictly in-process unless its source/runtime provenance
    is separately checked, as documented by ``solve_rb_mean``.
    """
    key = response_cache_key(settings, max_intensity_w_m2, cell_length_m=cell_length_m,
                             initial_nodes=initial_nodes, max_nodes=max_nodes, tolerance=tolerance)
    bound, length, initial, maximum, threshold = key[-5:]
    cache = {} if component_cache is None else component_cache
    evaluated = {}
    health = dict(trace_error=0., hermiticity_error=0., min_eigenvalue=1.,
                  steady_state_relative_residual=0.)
    class_count = 0

    def tell(fraction, message):
        if progress is not None:
            progress(float(fraction), message)

    def solve(intensity, *, order=settings.velocity_order, cutoff=settings.cutoff_sigma):
        nonlocal class_count
        local_key = (float(intensity), int(order), float(cutoff))
        if local_key not in evaluated:
            result = solve_rb_mean(settings.frequency, float(intensity),
                rb85_fraction=settings.rb85_fraction, temp_c=settings.temp_c,
                ground_relaxation_hz=settings.ground_relaxation_hz,
                density_m3=settings.density_m3, velocity_order=order,
                cutoff_sigma=cutoff, component_cache=cache)
            evaluated[local_key] = complex(result["chi"])
            class_count += result["class_count"]
            for name in health:
                reducer = min if name == "min_eigenvalue" else max
                health[name] = reducer(health[name], result["diagnostics"][name])
        return evaluated[local_key]

    axis = np.r_[0., np.geomspace(bound*2.5e-7, bound, initial-1)]
    k_length = 2*np.pi/settings.frequency.wavelength_m*length
    history = []
    while True:
        middle = .5*(axis[1:]+axis[:-1])
        values = []
        direct = []
        for target, locations in ((values, axis), (direct, middle)):
            for intensity in locations:
                tell(min(.69, .7*len(evaluated)/(2*maximum-1)),
                     f"Atomic response: {len(evaluated)} evaluated intensity/quadrature points")
                target.append(solve(intensity))
        values, direct = np.asarray(values), np.asarray(direct)
        table = ComplexSusceptibilityTable(axis, values)
        delta = direct-table(middle)
        phase, od = .5*k_length*abs(delta.real), k_length*abs(delta.imag)
        bad = (phase >= threshold) | (od >= threshold)
        history.append(dict(nodes=len(axis), held_out_points=len(middle),
            max_phase_error_rad=float(max(phase)), max_optical_depth_error=float(max(od))))
        fine_axis = np.sort(np.r_[axis, middle])
        fine_values = np.empty(len(fine_axis), complex)
        fine_values[np.searchsorted(fine_axis, axis)] = values
        fine_values[np.searchsorted(fine_axis, middle)] = direct
        refined = ComplexSusceptibilityTable(fine_axis, fine_values)
        if not np.any(bad) or len(axis)+int(sum(bad)) > maximum:
            break
        axis = np.sort(np.r_[axis, middle[bad]])
    interpolation_passed = not bool(np.any(bad))
    probes = []
    for index, intensity in enumerate(bound*np.array([0., .0025, .25, 1.])):
        tell(.7+.075*index, f"Velocity convergence: probe {index+1}/4")
        base = solve(intensity)
        fine = solve(intensity, order=settings.velocity_order+2)
        tail = solve(intensity, order=settings.velocity_order+2, cutoff=settings.cutoff_sigma+1)
        probes.append(dict(intensity_w_m2=float(intensity),
            order_phase_error_rad=float(.5*k_length*abs((fine-base).real)),
            order_optical_depth_error=float(k_length*abs((fine-base).imag)),
            cutoff_phase_error_rad=float(.5*k_length*abs((tail-fine).real)),
            cutoff_optical_depth_error=float(k_length*abs((tail-fine).imag))))
    quadrature_passed = all(np.isfinite(value) and value < threshold
        for probe in probes for name, value in probe.items() if name != "intensity_w_m2")
    checks = dict(assessed=True, passed=bool(interpolation_passed and quadrature_passed),
        interpolation=dict(assessed=True, passed=interpolation_passed, history=history,
            status="midpoint tolerance met" if interpolation_passed else "intensity node budget exhausted"),
        quadrature=dict(assessed=True, passed=bool(quadrature_passed), probes=probes),
        criterion=dict(max_phase_error_rad=threshold, max_optical_depth_error=threshold),
        state_health=health, evaluated_response_points=len(evaluated), sampled_velocity_states=class_count)
    tell(1., "Atomic table ready" if checks["passed"] else "Atomic table convergence failed")
    return dict(model_id=MODEL_ID, settings=settings, cache_key=key, table=table,
                refined_table=refined, checks=checks, cell_length_m=length,
                max_intensity_w_m2=bound, conditions=settings.as_dict())


def field_shape_difference(reference, reference_grid, candidate, candidate_grid):
    """Relative complex-field norm after removing one unobservable global phase."""
    yy, xx = np.meshgrid(reference_grid.y_m, reference_grid.x_m, indexing="ij")
    sampled = RegularGridInterpolator((candidate_grid.y_m, candidate_grid.x_m), candidate)(
        np.stack((yy, xx), axis=-1))
    overlap = np.vdot(reference, sampled)
    if overlap:
        sampled = sampled*np.exp(-1j*np.angle(overlap))
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(sampled-reference)/norm) if norm else float("nan")


def fit_relative_lens(result, control, wavelength, fit_radius):
    """Common fixed-aperture phase fit, shared with validation pipelines."""
    fit_grid = ParaxialGrid(np.linspace(-fit_radius, fit_radius, 65),
                            np.linspace(-fit_radius, fit_radius, 9))

    def central_cut(value):
        source_grid = value["grid"]
        row = int(np.argmin(abs(source_grid.y_m)))
        field = value["cell"].final_field[row]
        sampled = np.interp(fit_grid.x_m, source_grid.x_m, field)
        return np.broadcast_to(sampled, fit_grid.shape)
    lens = relative_lens_power(central_cut(result), central_cut(control), fit_grid,
                               wavelength_m=wavelength, fit_radius_m=fit_radius)
    quadratic_edge_phase = abs(np.pi/wavelength*fit_radius**2*lens["lens_power_m_inverse"])
    relative_rms = lens["phase_rms_rad"]/quadratic_edge_phase if quadratic_edge_phase else float("inf")
    lens.update(quadratic_edge_phase_rad=float(quadratic_edge_phase),
                phase_fit_relative_rms=float(relative_rms),
                phase_fit_passed=bool(np.isfinite(relative_rms) and relative_rms < .1),
                sampling="65 fixed physical points across +/-0.5 input radius; complex-field interpolation")
    return lens


def _budget(result, incident_power_w):
    transmission = float(result.power_w[-1]/incident_power_w)
    balance = float(abs(result.power_balance_error_w)/incident_power_w)
    dark = transmission < DARK_TRANSMISSION
    values = dict(transmission=transmission, power_w=float(result.power_w[-1]),
        radius_m=None if dark else float(result.radius_m[-1]), dark=bool(dark),
        relative_power_balance_error=balance,
        max_boundary_power_fraction=float(result.max_boundary_power_fraction))
    values["passed"] = bool(np.isfinite([transmission, balance, result.max_boundary_power_fraction]).all()
        and 0 <= transmission <= float(result.power_w[0]/incident_power_w)+1e-10
        and balance <= 1e-9 and result.max_boundary_power_fraction <= 1e-5
        and (dark or np.isfinite(values["radius_m"])))
    return values


def _lens(paths, grid, wavelength, radius, incident):
    if any(_budget(paths[name], incident)["dark"] for name in ("full", "absorption_only")):
        return dict(available=False, status="dark output; wavefront unavailable", sign_resolved=False)
    try:
        lens = fit_relative_lens(dict(grid=grid, cell=paths["full"]),
                                 dict(grid=grid, cell=paths["absorption_only"]), wavelength, .5*radius)
    except ValueError as error:
        return dict(available=False, status=str(error), sign_resolved=False)
    lens.update(available=bool(lens["phase_fit_passed"]), sign_resolved=False,
        status="quadratic phase fit" if lens["phase_fit_passed"] else "nonquadratic phase; focal length unavailable")
    return lens


def _compare_cell(reference, ref_grid, candidate, candidate_grid, incident):
    before, after = _budget(reference, incident), _budget(candidate, incident)
    if before["dark"] or after["dark"]:
        difference = abs(before["transmission"]-after["transmission"])
        return dict(transmission_absolute_difference=difference,
            passed=bool(before["passed"] and after["passed"] and before["dark"] and after["dark"]
                        and difference < DARK_TRANSMISSION), phase_status="dark output")
    widths = np.interp(reference.z_m, candidate.z_m, candidate.radius_m)
    width_error = float(np.max(abs(widths/reference.radius_m-1)))
    power_error = abs(after["transmission"]/before["transmission"]-1)
    field_error = field_shape_difference(reference.final_field, ref_grid,
                                         candidate.final_field, candidate_grid)
    return dict(max_width_relative_difference=width_error,
        transmission_relative_difference=power_error, exit_field_shape_relative_error=field_error,
        passed=bool(before["passed"] and after["passed"]
                    and all(np.isfinite(v) and v < .01 for v in (width_error, power_error, field_error))))


def propagate_cell(response, *, power_w, radius_m, curvature_radius_m=np.inf,
                   cell_length_m, half_width_m, samples=128, steps=80,
                   refine=True, progress=None):
    """Propagate an entrance-plane Gaussian through one selected Rb D line.

    Returned paths are full response, intensity-dependent absorption only, and
    vacuum. Grid, longitudinal step, window and intensity-table refinements are
    optional and explicitly unassessed when omitted. Camera navigation is a
    separate call. A changed cell length requires a matching acceptance bundle.
    """
    settings = response["settings"]
    length = _positive(cell_length_m, "cell_length_m", zero=True)
    if length != response["cell_length_m"]:
        raise ValueError("cell length differs from atomic table acceptance conditions; rebuild the response table")
    if tuple(response["cache_key"][:len(settings.cache_key)]) != settings.cache_key:
        raise ValueError("atomic response settings do not match the cached table")
    power = _positive(power_w, "power_w")
    radius = _positive(radius_m, "radius_m")
    half_width = _positive(half_width_m, "half_width_m")
    count = _integer(samples, "samples", 8)
    step_count = _integer(steps, "steps", 1)
    curvature = float(curvature_radius_m)
    if np.isnan(curvature) or curvature == 0:
        raise ValueError("curvature_radius_m must be nonzero or infinite")
    wavelength = settings.frequency.wavelength_m
    if peak_intensity(power, radius) > response["table"].max_intensity_w_m2:
        raise ValueError("input intensity exceeds atomic table ceiling; rebuild with a larger bound")

    def tell(fraction, message):
        if progress is not None:
            progress(float(fraction), message)

    def run(table, *, n=count, width=half_width, nz=step_count, vacuum=False):
        grid = ParaxialGrid.square(width, n)
        field = gaussian_field(grid, power, radius, wavelength, curvature)
        paths = {}
        modes = ("full", "absorption_only", "vacuum") if vacuum else ("full", "absorption_only")
        for mode in modes:
            susceptibility = (None if mode == "vacuum" else table if mode == "full" else
                              lambda intensity: table(0.).real+1j*table(intensity).imag)
            paths[mode] = propagate(field, grid, wavelength_m=wavelength, length_m=length,
                steps=nz, susceptibility=susceptibility, snapshot_stride=max(1, nz//80))
        return grid, paths

    tell(0., "Propagating cell and controls")
    grid, paths = run(response["table"], vacuum=True)
    incident = float(paths["full"].power_w[0])
    budgets = {name: _budget(path, incident) for name, path in paths.items()}
    sampling_error = float(abs(incident/power-1))
    budgets_passed = all(value["passed"] for value in budgets.values()) and sampling_error < 1e-6
    lens = _lens(paths, grid, wavelength, radius, incident)
    spatial = dict(assessed=bool(refine), passed=False, criterion="width, transmission, and complex field differences <1%")
    if refine:
        variants = (("transverse_grid", dict(n=2*count), response["table"]),
            ("longitudinal_step", dict(nz=2*step_count), response["table"]),
            ("window", dict(n=2*int(ceil(.75*count)), width=1.5*half_width), response["table"]),
            ("intensity_table", {}, response["refined_table"]))
        refinements = {}
        for index, (name, options, table) in enumerate(variants):
            tell(.2+.2*index, "Checking "+name.replace("_", " "))
            candidate_grid, candidate = run(table, **options)
            check = _compare_cell(paths["full"], grid, candidate["full"], candidate_grid, incident)
            control_check = _compare_cell(paths["absorption_only"], grid,
                                          candidate["absorption_only"], candidate_grid, incident)
            check["absorption_only"] = control_check
            check["passed"] &= control_check["passed"]
            refined_lens = _lens(candidate, candidate_grid, wavelength, radius, incident)
            check["refined_lens"] = refined_lens
            if lens["available"] and refined_lens["available"]:
                difference = abs(lens["lens_power_m_inverse"]-refined_lens["lens_power_m_inverse"])
                tolerance = max(.01*abs(lens["lens_power_m_inverse"]), 1e-3)
                check.update(lens_power_absolute_difference_m_inverse=difference,
                             lens_power_tolerance_m_inverse=tolerance)
                check["passed"] &= difference < tolerance
            elif lens["available"] != refined_lens["available"]:
                check.update(passed=False, lens_status="phase-fit validity changed under refinement")
            refinements[name] = check
        spatial.update(passed=all(check["passed"] for check in refinements.values()), refinements=refinements)
        if lens["available"]:
            delta = max(check.get("lens_power_absolute_difference_m_inverse", 0.)
                        for check in refinements.values())
            sign_resolved = bool(response["checks"]["passed"] and budgets_passed
                and spatial["passed"] and abs(lens["lens_power_m_inverse"]) > max(5*delta, 1e-3))
            lens.update(sign_resolved=sign_resolved, max_refinement_difference_m_inverse=delta,
                sign_status=("focusing" if lens["lens_power_m_inverse"] > 0 else "defocusing")
                if sign_resolved else "numerical sign unresolved")
    else:
        spatial["status"] = "grid, step, window and intensity-table refinement unassessed"
    checks = dict(atomic=response["checks"], physical_budgets=dict(passed=bool(budgets_passed),
        paths=budgets, input_sampling_relative_error=sampling_error), propagation=spatial,
        passed=bool(response["checks"]["passed"] and budgets_passed and spatial["passed"]))
    tell(1., "Cell profile ready")
    return dict(model_id=MODEL_ID, settings=settings, response_key=response["cache_key"],
        input_key=(response["cache_key"], power, radius, curvature, length, half_width, count, step_count, bool(refine)),
        grid=grid, wavelength_m=wavelength, cell_length_m=length, paths=paths,
        checks=checks, nonlinear_lens=lens, incident_power_w=incident,
        conditions=dict(atomic=settings.as_dict(), power_w=power, radius_m=radius,
            curvature_radius_m=curvature, input_plane="cell entrance", coherent_m2=1.,
            cell_length_m=length, half_width_m=half_width, samples=count, steps=step_count,
            scope="one linearly polarized CW beam, B=0, local steady state; other drives absent"))


def profile_at_distance(cell_result, distance_m, *, steps=32):
    """Free-space camera navigation from saved cell fields; no atomic calls.

    Cell convergence is retained, while the free-space plane has its own power
    and boundary budgets. No material, lenses or downstream layout optics are
    silently inserted. Dark output has no reported radius or focal length.
    """
    distance = _positive(distance_m, "distance_m", zero=True)
    count = _integer(steps, "steps", 1)
    grid, wavelength = cell_result["grid"], cell_result["wavelength_m"]
    incident = cell_result["incident_power_w"]
    paths, combined, budgets = {}, {}, {}
    for name, cell in cell_result["paths"].items():
        camera = propagate(cell.final_field, grid, wavelength_m=wavelength,
                           length_m=distance, steps=count)
        paths[name] = camera
        budget = _budget(camera, incident)
        drift = float(abs(camera.power_w[-1]-camera.power_w[0])/incident)
        budget.update(relative_free_space_power_drift=drift)
        budget["passed"] &= drift <= 1e-9
        budgets[name] = budget
        combined[name] = dict(z_m=np.r_[cell.z_m, cell_result["cell_length_m"]+camera.z_m[1:]],
            radius_m=np.r_[cell.radius_m, camera.radius_m[1:]],
            power_w=np.r_[cell.power_w, camera.power_w[1:]],
            centerline_intensity_w_m2=np.concatenate((cell.centerline_intensity_w_m2,
                                                     camera.centerline_intensity_w_m2[1:])))
    budget_passed = all(item["passed"] for item in budgets.values())
    return dict(model_id=MODEL_ID, input_key=(cell_result["input_key"], distance, count),
        grid=grid, wavelength_m=wavelength, distance_m=distance, paths=paths, combined=combined,
        checks=dict(cell=cell_result["checks"], free_space_budgets=dict(passed=budget_passed, paths=budgets),
                    camera_convergence=dict(assessed=False, status="camera radius/profile not independently refined; cell exit convergence and camera budgets only"),
                    passed=bool(cell_result["checks"]["passed"] and budget_passed)),
        nonlinear_lens=cell_result["nonlinear_lens"], conditions=cell_result["conditions"])
