"""Thermal absorption and single-pass propagation for a distributed Gaussian flight history.

The local power-loss coefficient is a plane-crossing average over complete transverse
flights. Its argument is the whole incoming Gaussian profile at one axial position.
The scalar propagation closure holds that shape fixed while its total power decays.
Its limits are reported with the result. In particular, solving the nonlinear
absorption at each radius using an independent stationary atom would change the model.
"""
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import beam, collisions, constants, linear_hyperfine, probe_saturation, species, vapor_cell


@dataclass(frozen=True)
class BallisticSpectrum:
    transmission: np.ndarray
    optical_depth: np.ndarray
    od0: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class _Resolution:
    path_nodes: tuple = (96, 10, 12)
    panel_order: int = 4
    wing_widths: int = 512
    log_intensity_step: float = np.log(10.0)/3.0
    minimum_intensity: float = 0.003


_REFERENCE = _Resolution()
_FAST = _Resolution(path_nodes=(64, 10, 10))  # Direct mesh for recorded Fast v1 spectra.


@dataclass(frozen=True)
class _Optics:
    temperature: float
    density: float
    width: float
    waist: float

    @property
    def sigma(self):
        manifold = species.build_manifold(species.RB85, 'D1')
        return manifold.k_vec * constants.thermal_speed(self.temperature, species.RB85.mass)


def _optics(conditions):
    pure = (conditions.species == species.RB85.label or
            (conditions.species == 'Rb (natural)' and conditions.rb_composition == 'Custom'
             and conditions.rb85_fraction == 1.0))
    if not pure or conditions.line != 'D1':
        raise ValueError('Distributed transit requires pure Rb85 on D1.')
    if conditions.ne_pressure_torr or conditions.paraffin_coated:
        raise ValueError('Distributed transit requires an uncoated, buffer-gas-free cell.')
    if conditions.optical_collision_model not in (None, collisions.LEGACY_COLLISIONS):
        raise ValueError('Distributed transit requires the buffer-gas-free optical collision model.')
    cold = conditions.temp_c + 273.15
    warm = (cold if conditions.doppler_temp_c is None
            else conditions.doppler_temp_c + 273.15)
    waist = conditions.probe_waist_mm*1e-3
    if (not np.all(np.isfinite([cold, warm, waist, conditions.cell_mm, conditions.probe_power_uw]))
            or min(cold, warm, waist) <= 0 or min(conditions.cell_mm, conditions.probe_power_uw) < 0):
        raise ValueError('Distributed transit requires physical temperatures, beam waist, power and length.')
    density = vapor_cell.cold_spot_limited_density(species.number_density(species.RB85, cold), cold, warm)
    manifold = species.build_manifold(species.RB85, 'D1')
    width = .5*(manifold.gamma + species.self_broadening(species.RB85, density, 'D1', per_line=True))
    width += np.pi*conditions.extra_homogeneous_fwhm_hz
    return _Optics(warm, density, width, waist)


@lru_cache(maxsize=32)
def _atom(temperature, width):
    from .ballistic_pumping import BallisticProbe85D1

    return BallisticProbe85D1(temperature, width)


def _path_grid(resolution):
    from .ballistic_pumping import TrajectoryGrid

    nx, nb, nv = resolution.path_nodes
    return TrajectoryGrid(nx=nx, nb=nb, nv=nv, xcut=4., bcut=4., vcut=7.)


@lru_cache(maxsize=48)
def _panels(width, order, wing_widths):
    """Gauss-Legendre cells clustered at every allowed zero-velocity component.

    Every panel integrates in angular frequency. Adjacent component domains meet
    at their midpoint, so overlapping tails belong to exactly one domain.
    """
    centers = np.unique(species.build_manifold(species.RB85, 'D1').omega)
    outer = width * wing_widths
    bounds = np.r_[centers[0]-outer, (centers[:-1]+centers[1:])/2, centers[-1]+outer]
    edges = list(bounds)
    for i, center in enumerate(centers):
        edges.append(center)
        radius = width
        while radius < max(center-bounds[i], bounds[i+1]-center):
            if center-radius > bounds[i]:
                edges.append(center-radius)
            if center+radius < bounds[i+1]:
                edges.append(center+radius)
            radius *= 2.
    edges = np.unique(edges)
    nodes, weights = leggauss(order)
    halfwidth = np.diff(edges) / 2
    samples = (edges[:-1,None]+halfwidth[:,None]*(1.+nodes)).ravel()
    quadrature = (halfwidth[:,None]*weights).ravel()
    return samples, quadrature


def _weak(optics, scan):
    manifold = species.build_manifold(species.RB85, 'D1')
    absorption, _ = linear_hyperfine.species_complex_voigt(
        scan, manifold, 1., optics.temperature, 2*optics.width)
    return absorption


@lru_cache(maxsize=512)
def _rest_excess(temperature, width, waist, intensity, resolution):
    """Absolute local cross section minus its discrete linear-response limit [m²]."""
    atom = _atom(temperature, width)
    nodes, _ = _panels(width, resolution.panel_order, resolution.wing_widths)
    grid = _path_grid(resolution)
    local = np.empty(len(nodes))
    weak = atom.weak_cross_sections(nodes)
    peak = np.array([intensity])
    refined = 0
    max_steps = grid.nx

    def solve(first, last, mesh):
        nonlocal refined, max_steps
        try:
            values = atom.cross_sections(nodes[first:last], peak, waist, grid=mesh, accelerated=True)[0]
        except ArithmeticError as error:
            if not str(error).startswith("Unresolved ballistic populations"):
                raise
            if last-first > 1:
                middle = (first+last)//2
                solve(first, middle, mesh)
                solve(middle, last, mesh)
                return
            if mesh.nx >= 16*grid.nx:
                raise
            solve(first, last, replace(mesh, nx=mesh.nx*2))
            return
        local[first:last] = values - mesh.weak_factor()*weak[first:last]
        if mesh.nx > grid.nx:
            refined += last-first
            max_steps = max(max_steps, mesh.nx)

    for first in range(0, len(nodes), 16):
        solve(first, min(first+16, len(nodes)), grid)
    return local, refined, max_steps


def _thermal_table(optics, scan, intensity, resolution, *, diagnostics=None, use_table=False):
    """Convolve a local absorption *excess* with the Maxwell velocity distribution.

    The Gaussian in rest frequency has unit integral. It is never renormalized
    to the finite panel span; the remainder carries the analytical weak result.
    Integrate before axial propagation. Frequency inputs may be nonuniform.
    """
    nodes, weights = _panels(optics.width, resolution.panel_order, resolution.wing_widths)
    weak = _weak(optics, scan)
    rest = None
    if use_table:
        from ._ballistic_reference_table import lookup
        rest, provenance = lookup(optics, intensity, nodes, weights)
        if diagnostics is not None:
            diagnostics.update(provenance)
    if rest is None:
        sampled = [_rest_excess(optics.temperature, optics.width, optics.waist, float(power), resolution)
                   for power in intensity]
        rest = np.stack([entry[0] for entry in sampled])
        if diagnostics is not None:
            diagnostics['refined_path_rows'] = sum(entry[1] for entry in sampled)
            diagnostics['max_transverse_steps'] = max(entry[2] for entry in sampled)
            if use_table:
                diagnostics['rest_method'] = 'direct reference trajectories'
    table = np.empty((len(intensity), len(scan)))
    sigma = optics.sigma
    for start in range(0, len(scan), 512):
        sl = slice(start, start+512)
        density = np.exp(-.5*((scan[sl,None]-nodes)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)
        table[:,sl] = weak[sl] + rest @ (density*weights).T
    if not np.all(np.isfinite(table)) or np.any(table <= 0):
        raise ArithmeticError('Distributed-transit thermal quadrature produced a nonpositive absorption.')
    return weak, table


def _intensity_nodes(peak, resolution):
    """Shared physical intensities (W/m²) across fits and powers; bracket the input."""
    start = resolution.minimum_intensity
    count = max(1, int(np.ceil(np.log(peak/start)/resolution.log_intensity_step)))
    return start*np.exp(np.arange(count+2)*resolution.log_intensity_step)


def _refine_intensity(nodes, table, subdivisions=8):
    """Monotone cubic interpolation in log sigma/log I; retain an exact piecewise axial integral.

    A real upper guard knot beyond the input bracket supplies its endpoint
    slope, so lowering the requested launch power cannot change the model
    at lower intensities already sampled in a different scan.
    """
    from scipy.interpolate import PchipInterpolator

    log_i = np.log(nodes)
    fine = np.linspace(log_i[0], log_i[-1], (len(nodes)-1)*subdivisions+1)
    return np.exp(fine), np.exp(PchipInterpolator(log_i, np.log(table), axis=0)(fine))


def _at_peak(intensity, nodes, table):
    """Interpolate a positive local coefficient in log intensity on one bracket."""
    i = max(0, min(len(nodes)-2, np.searchsorted(nodes, intensity)-1))
    weight = np.log(intensity/nodes[i])/np.log(nodes[i+1]/nodes[i])
    return table[i] + weight*(table[i+1]-table[i])


def _deplete(weak, nodes, table, peak, column):
    """Integrate dOD/d(column) = sigma(I_in*exp(-OD)) across interpolation knots.

    Above the first knot sigma is affine in log I. Below it the coefficient
    tends linearly in I to the exact zero-power cross section. Both
    interpolants admit an exact integral for arbitrary optical depth.
    """
    remaining = np.full(len(weak), column)
    od = np.zeros_like(weak)
    current = peak
    for i in reversed(np.flatnonzero(nodes < peak)):
        if not np.any(remaining):
            break
        bottom = nodes[i]
        logspan = np.log(current/bottom)
        high = _at_peak(current, nodes, table) if current == peak else table[i+1]
        low = table[i]
        relative = (low-high)/high
        # required column to cross this entire interval; log1p handles flat responses.
        log_ratio = np.ones_like(relative)
        np.divide(np.log1p(relative), relative, out=log_ratio, where=relative!=0)
        needed = (logspan/high)*log_ratio
        used = np.minimum(needed, remaining)
        x = relative*high/logspan*used
        exp_ratio = np.ones_like(x)
        np.divide(np.expm1(x), x, out=exp_ratio, where=x!=0)
        od += high*used*exp_ratio
        remaining = np.maximum(0., remaining-used)
        current = bottom
    if np.any(remaining):
        # y=I_min/I solves a linear ODE. Use its logarithm to remain finite
        # even where the actual transmitted intensity underflows.
        difference = (table[0]/weak-1.)*min(1.,peak/nodes[0])
        if np.any(difference <= -1):
            raise ArithmeticError('Distributed-transit low-intensity extrapolation failed.')
        depth = remaining*weak
        od += depth + np.log1p(difference*(-np.expm1(-depth)))
    return od


def evaluate_ballistic_spectrum(conditions, scan_rad_s, *, fidelity='reference'):
    """Evaluate a pure 85Rb D1 scan in its fine-centroid frame."""
    if fidelity not in ('reference', 'fast', 'fast_v1'):
        raise ValueError('Choose reference, fast or recorded fast_v1 distributed-transit fidelity.')
    scan = np.asarray(scan_rad_s, dtype=float)
    if fidelity == 'reference':
        return _evaluate(conditions, scan, _REFERENCE, fidelity)
    if scan.ndim != 1 or not len(scan) or not np.all(np.isfinite(scan)):
        raise ValueError('A nonempty finite one-dimensional detuning axis is required.')
    # Mesh errors were measured in this single-beam geometry. The weak and
    # transparent-cell limits are analytical for either mesh.
    zero = conditions.probe_power_uw == 0 or conditions.cell_mm == 0
    # Unit conversions in imported data can shift the same radius by a few ulps.
    # The v1 replay keeps its original exact-equality eligibility boundary.
    radius_matches = (conditions.probe_waist_mm == 0.53 if fidelity == 'fast_v1' else
                      abs(conditions.probe_waist_mm - 0.53) <= 8*np.spacing(0.53))
    measured_geometry = (
        radius_matches and 0 < conditions.cell_mm <= 12.5
        and conditions.line_strength == 1 and conditions.extra_homogeneous_fwhm_hz == 0
        and (conditions.doppler_temp_c is None or conditions.doppler_temp_c == conditions.temp_c)
        and 95 <= conditions.temp_c <= 107 and 0 <= conditions.probe_power_uw <= 160
    )
    if not (zero or measured_geometry):
        result = _evaluate(conditions, scan, _REFERENCE, fidelity)
        result.diagnostics['fallback_reason'] = 'Conditions outside the Fast validation range. Evaluated the reference.'
        return result
    inside = np.abs(scan) <= 2*np.pi*7e9
    if zero or np.all(inside):
        return _evaluate_fast(conditions, scan, fidelity)
    if not np.any(inside):
        result = _evaluate(conditions, scan, _REFERENCE, fidelity)
        result.diagnostics['fallback_reason'] = 'Frequencies outside the Fast validation range. Evaluated the reference.'
        return result
    # Preserve the pointwise OD contract for unordered, duplicated and mixed scans.
    fast = _evaluate_fast(conditions, scan[inside], fidelity)
    ref = _evaluate(conditions, scan[~inside], _REFERENCE, fidelity)
    arrays = []
    for name in ('transmission', 'optical_depth', 'od0'):
        merged = np.empty_like(scan)
        merged[inside] = getattr(fast, name)
        merged[~inside] = getattr(ref, name)
        merged.setflags(write=False)
        arrays.append(merged)
    all_reference = fast.diagnostics['fidelity_used'] == 'reference'
    diagnostics = dict(fast.diagnostics,
                       fidelity_used='reference' if all_reference else 'mixed',
                       reference_samples=len(scan) if all_reference else int(np.sum(~inside)),
                       max_weak_absorption_per_waist=max(fast.diagnostics['max_weak_absorption_per_waist'],
                                                        ref.diagnostics['max_weak_absorption_per_waist']))
    # Each partition can have different rest and trajectory grids and recovery steps.
    grid_fields = ('transverse_nodes', 'rest_nodes', 'intensity_rows', 'refined_path_rows', 'max_transverse_steps',
                   'rest_method', 'rest_table_sha256', 'rest_temperature_nodes_c', 'rest_temperature_interpolation',
                   'rest_table_refined_rows_at_knots', 'rest_table_max_steps_at_knots', 'rest_table_fallback_reason')
    diagnostics['partitions'] = [
        dict(samples=count, fidelity_used=part['fidelity_used'],
             **{name: part[name] for name in grid_fields if name in part})
        for count, part in ((int(np.sum(inside)), fast.diagnostics), (int(np.sum(~inside)), ref.diagnostics))
    ]
    for name in grid_fields:
        diagnostics.pop(name, None)
    if not all_reference:
        diagnostics['fallback_reason'] = 'Frequencies beyond ±7 GHz were evaluated with the reference.'
    return BallisticSpectrum(*arrays, diagnostics)


def _evaluate_fast(conditions, scan, fidelity):
    try:
        if fidelity == 'fast_v1':
            return _evaluate(conditions, scan, _FAST, fidelity)
        return _evaluate(conditions, scan, _REFERENCE, fidelity, use_table=True)
    except ArithmeticError as error:
        result = _evaluate(conditions, scan, _REFERENCE, fidelity)
        result.diagnostics['fallback_reason'] = f'Fast quadrature did not converge ({error}). Evaluated the reference.'
        return result


def _evaluate(conditions, scan_rad_s, resolution, requested='reference', *, use_table=False):
    scan = np.asarray(scan_rad_s, dtype=float)
    if scan.ndim != 1 or not len(scan) or not np.all(np.isfinite(scan)):
        raise ValueError('A nonempty finite one-dimensional detuning axis is required.')
    fast_route = use_table or resolution is _FAST
    optics = _optics(conditions)
    length = conditions.cell_mm*1e-3
    column = length*optics.density*conditions.line_strength
    weak = _weak(optics, scan)
    od0 = weak*column
    peak = beam.peak_intensity(conditions.probe_power_uw*1e-6, optics.waist)
    row_meta = {}
    if peak > 0 and column > 0:
        intensity = _intensity_nodes(peak, resolution)
        weak, table = _thermal_table(optics, scan, intensity, resolution, diagnostics=row_meta, use_table=use_table)
        fine_intensity, fine_table = _refine_intensity(intensity, table)
        od = _deplete(weak, fine_intensity, fine_table, peak, column)
    else:
        intensity = np.array([])
        od = od0.copy()
    maximum_gradient = float(np.max(weak)*optics.density*conditions.line_strength)
    manifold = species.build_manifold(species.RB85, 'D1')
    rayleigh = probe_saturation.cell_to_rayleigh_ratio(length, optics.waist*1e3, manifold.k_vec)
    if 'rest_table_fallback_reason' in row_meta:
        row_meta['fallback_reason'] = row_meta['rest_table_fallback_reason'] + ' Evaluated the reference directly.'
        fast_route = False
    diagnostics = dict(fidelity_requested='fast' if requested == 'fast_v1' else requested,
                       fidelity_used='fast' if fast_route else 'reference',
                       propagation='local Gaussian-shaped transverse flux with axial power depletion',
                       physical_status=('analytical zero-power/length limit' if not peak or not column
                                        else 'conditional Gaussian profile and local-axial propagation'),
                       radial_profile_assumption='Gaussian at each plane; evolving annular deposition can deform the beam',
                       axial_history_assumption='local plane during each transverse crossing; axial flight through the absorption gradient omitted',
                       radial_and_axial_trajectories_validated=False,
                       peak_intensity_w_m2=peak, intensity_rows=len(intensity),
                       max_weak_absorption_per_waist=maximum_gradient*optics.waist,
                       cell_length_to_rayleigh_range=rayleigh,
                       thermal_quadrature='analytical Voigt plus Gaussian-weighted rest-frame excess',
                       rest_nodes=len(_panels(optics.width,resolution.panel_order,resolution.wing_widths)[0]),
                       transverse_nodes=resolution.path_nodes, **row_meta)
    transmission = np.exp(-od)
    for data in (transmission, od, od0):
        data.setflags(write=False)
    return BallisticSpectrum(transmission, od, od0, diagnostics)
