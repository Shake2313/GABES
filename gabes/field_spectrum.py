"""Single-beam rubidium absorption with full magnetic-sublevel flight histories.

The Jones vector and transverse Gaussian shape are prescribed along the cell.
This is scalar power propagation, including field-dependent absorption.
"""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import beam
from .ballistic_spectrum import (
    BallisticSpectrum, _REFERENCE, _deplete, _intensity_nodes, _panels,
    _path_grid, _rb_components, _refine_intensity,
)
from .field_weak import transition_centers, weak_cross_sections


_progress = ContextVar("field_spectrum_progress", default=None)
_current_row = ContextVar("field_spectrum_row", default=None)


@contextmanager
def field_progress(callback):
    """Observe rest-frequency integration; returning false stops the solve."""
    token = _progress.set(callback)
    try:
        yield
    finally:
        _progress.reset(token)


def _report(isotope, done, total):
    callback = _progress.get()
    if callback is not None and callback((isotope.name, done, total)) is False:
        raise CancelledError()


def _checkpoint():
    status = _current_row.get()
    if status is not None:
        _report(*status)


def _field_panels(part, field, jones, resolution):
    if not any(field):
        return _panels(part.width, resolution.panel_order, resolution.wing_widths,
                       part.isotope, part.line)
    centers = transition_centers(part.isotope, part.line, field, jones)
    # Bound the distance of every Zeeman component to a panel anchor at
    # one eighth of an optical HWHM, even when an entire multiplet overlaps.
    anchors = []
    first = 0
    while first < len(centers):
        end = np.searchsorted(centers, centers[first] + 0.25 * part.width, side="right")
        anchors.append(0.5 * (centers[first] + centers[end - 1]))
        first = end
    centers = np.asarray(anchors)
    radius = resolution.wing_widths * part.width
    bounds = np.r_[centers[0] - radius, 0.5 * (centers[1:] + centers[:-1]), centers[-1] + radius]
    edges = [*bounds, *centers]
    for center, left, right in zip(centers, bounds[:-1], bounds[1:]):
        step = part.width
        while step < max(center - left, right - center):
            if center - step > left:
                edges.append(center - step)
            if center + step < right:
                edges.append(center + step)
            step *= 2
    edges = np.unique(edges)
    nodes, weights = leggauss(resolution.panel_order)
    half = np.diff(edges)[:, None] / 2
    return (edges[:-1, None] + half * (1 + nodes)).ravel(), (half * weights).ravel()


@lru_cache(maxsize=16)
def _atom(isotope, line, temperature, width, field, jones):
    from .field_ballistic import FieldBallisticProbe
    return FieldBallisticProbe(isotope, line, temperature, width, field_ut=field, jones=jones)


@lru_cache(maxsize=2048)
def _rest_row(isotope, line, temperature, width, waist, field, jones, frequency, intensities, grid):
    atom = _atom(isotope, line, temperature, width, field, jones)
    mesh = grid
    while True:
        try:
            values = atom.cross_sections([frequency], intensities, waist, grid=mesh,
                                         checkpoint=_checkpoint)[:, 0]
            return values, mesh
        except ArithmeticError as error:
            if not str(error).startswith("Unresolved field-history populations") or mesh.nx >= 16 * grid.nx:
                raise
            mesh = replace(mesh, nx=2 * mesh.nx)


def _isotope_table(part, scan, intensities, field, jones, resolution):
    optical = (part.isotope, part.line, part.width, field, jones)
    geometry = dict(temperature_k=part.temperature, waist_m=part.waist)
    weak = weak_cross_sections(*optical, scan, thermal=True, **geometry)
    if not len(intensities):
        return weak, None, 0, 0

    nodes, weights = _field_panels(part, field, jones, resolution)
    grid = _path_grid(resolution)
    _report(part.isotope, 0, len(nodes))
    # Evaluate the derivative of the *same* discrete flight integrator.
    # This leaves no power-independent offset when the continuous Maxwell
    # response replaces the weak part of the finite integration domain.
    discrete = weak_cross_sections(*optical, nodes, grid=grid, **geometry)
    rest = np.empty((len(intensities), len(nodes)))
    powers = tuple(intensities)
    refined = 0
    for index, frequency in enumerate(nodes):
        _report(part.isotope, index, len(nodes))
        token = _current_row.set((part.isotope, index, len(nodes)))
        try:
            values, mesh = _rest_row(part.isotope, part.line, part.temperature, part.width,
                                     part.waist, field, jones, frequency, powers, grid)
        finally:
            _current_row.reset(token)
        linear = discrete[index]
        if mesh != grid:
            linear = weak_cross_sections(*optical, [frequency], grid=mesh, **geometry)[0]
            refined += 1
        rest[:, index] = values - linear
    _report(part.isotope, len(nodes), len(nodes))
    table = np.empty((len(intensities), len(scan)))
    for first in range(0, len(scan), 512):
        block = slice(first, first + 512)
        gaussian = np.exp(-0.5 * ((scan[block, None] - nodes) / part.sigma)**2)
        gaussian *= weights / (np.sqrt(2 * np.pi) * part.sigma)
        table[:, block] = weak[block] + rest @ gaussian.T
    if not np.isfinite(table).all() or np.any(table <= 0):
        raise ArithmeticError("Field-history quadrature returned a nonpositive absorption coefficient.")
    return weak, table, len(nodes), refined


def evaluate_field_ballistic_spectrum(conditions, scan_rad_s, *, _resolution=_REFERENCE):
    """Evaluate static-field OD on an arbitrary lab-frame frequency axis (rad/s)."""
    scan = np.asarray(scan_rad_s, dtype=float)
    if scan.ndim != 1 or not scan.size or not np.isfinite(scan).all():
        raise ValueError("A nonempty finite one-dimensional detuning axis is required.")
    parts = _rb_components(conditions)
    total_density = sum(part.density for part, _, _ in parts)
    column = conditions.cell_mm * 1e-3 * conditions.line_strength * total_density
    if not np.isfinite(column) or column < 0:
        raise ValueError("A finite nonnegative absorber column is required.")
    field, jones = tuple(conditions.field_ut), tuple(conditions.probe_jones)
    peak = beam.peak_intensity(conditions.probe_power_uw * 1e-6, parts[0][0].waist)
    intensities = _intensity_nodes(peak, _resolution) if column and peak else ()
    weak = np.zeros_like(scan)
    table = np.zeros((len(intensities), len(scan)))
    meta = []
    for part, fraction, offset in parts:
        linear, finite, n, refined = _isotope_table(part, scan - offset, intensities, field, jones, _resolution)
        weak += fraction * linear
        if finite is not None:
            table += fraction * finite
        meta.append(dict(isotope=part.isotope.name, line=part.line, fraction=fraction,
                         centroid_offset_rad_s=offset, optical_hwhm_rad_s=part.width, rest_nodes=n, refined_rest_nodes=refined))
    od0 = column * weak
    if len(intensities):
        fine, cross = _refine_intensity(intensities, table)
        od = _deplete(weak, fine, cross, peak, column)
    else:
        od = od0.copy()
    transmission = np.exp(-od)
    for array in (transmission, od, od0):
        array.setflags(write=False)
    diagnostics = dict(fidelity_requested="reference", fidelity_used="reference",
                       coherence_model="full magnetic-sublevel density matrix",
                       propagation="axial power depletion with fixed Gaussian profile and prescribed Jones polarization",
                       physical_status="single beam in a uniform static magnetic field",
                       perturber_density="total elemental Rb", field_ut=field,
                       optical_polarization_evolves=False, isotopes=meta,
                       intensity_rows=len(intensities), rest_nodes=sum(m["rest_nodes"] for m in meta),
                       transverse_nodes=_resolution.path_nodes if len(intensities) else None,
                       thermal_quadrature="finite-flight weak response plus Maxwell-weighted nonlinear excess")
    return BallisticSpectrum(transmission, od, od0, diagnostics)
