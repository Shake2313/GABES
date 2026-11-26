"""Time-ordered magnetic-sublevel pumping during transverse 85Rb D1 beam crossings.

One uncoated, buffer-free zero-field cell and a single linearly polarized beam.
The detuning is the atom-frame angular offset from the 85Rb fine centroid. The
default intrinsic optical HWHM includes natural width and D1 self-broadening;
callers can supply another intrinsic optical HWHM.
Atoms enter thermally, so there is no reset or transit width during a crossing.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import operator

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import constants, kernels, species
from ._ballistic_kernel import SDIRK_DIAGONAL, integrate_native
from .optical_pumping import MAGNETIC_PI, ProjectedProbe


def _axis(values, name):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite, nonempty vector.")
    return values


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _radial_axis(values, name):
    axis = _axis(values, name)
    if len(axis) < 2 or axis[0] != 0.0 or np.any(np.diff(axis) <= 0):
        raise ValueError(f"{name} must increase strictly from zero.")
    return axis


@lru_cache(maxsize=64)
def _quadrature(n, cut):
    x, weights = leggauss(n)
    return (1 + x) * (cut / 2), weights * (cut / 2)


@dataclass(frozen=True)
class TrajectoryGrid:
    """SDIRK steps across [-xcut,xcut] w, GL nodes in b/w and v_perp/sigma."""
    nx: int = 192
    nb: int = 12
    nv: int = 16
    xcut: float = 4.0
    bcut: float = 4.0
    vcut: float = 8.0

    def __post_init__(self):
        for name in ("nx", "nb", "nv"):
            value = getattr(self, name)
            try:
                count = operator.index(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a positive integer.") from exc
            if isinstance(value, (bool, np.bool_)) or count < 1:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, count)
        for name in ("xcut", "bcut", "vcut"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    @lru_cache(maxsize=64)
    def weak_factor(self):
        """Discrete Gaussian weak absorption divided by its infinite-space limit."""
        b, wb, q, wq, steps, r1, r2, _ = _geometry(self, ())
        signal = (1.0 - SDIRK_DIAGONAL) * np.exp(-2 * r1**2) + SDIRK_DIAGONAL * np.exp(-2 * r2**2)
        return float((4.0 / np.pi) * (wq @ (q * np.exp(-q**2 / 2))) * (wb @ np.sum(steps * signal, axis=1)))


@dataclass(frozen=True)
class BallisticPath:
    populations: np.ndarray
    absorbed_photons: float
    absorbed_photons_per_step: np.ndarray


@lru_cache(maxsize=24)
def _geometry(grid, annuli, support_cut=None):
    # Include annulus boundaries as panel boundaries in b and x so individual
    # rings converge without sampling a discontinuous binning indicator.
    b, wb = _quadrature(grid.nb, grid.bcut)
    q, wq = _quadrature(grid.nv, grid.vcut)
    boundaries = annuli if support_cut is None else (*annuli, support_cut)
    if boundaries:
        limits = np.unique([0.0, *(r for r in boundaries if 0.0 < r < grid.bcut), grid.bcut])
        b = np.concatenate([lo + _quadrature(grid.nb, hi - lo)[0] for lo, hi in zip(limits[:-1], limits[1:])])
        wb = np.concatenate([_quadrature(grid.nb, hi - lo)[1] for lo, hi in zip(limits[:-1], limits[1:])])
    dx = 2.0 * grid.xcut / grid.nx
    x = -grid.xcut + dx * np.arange(grid.nx + 1)
    x[-1] = grid.xcut
    nodes = []
    for impact in b:
        if not boundaries:
            nodes.append(x)
            continue
        crossings = [math.sqrt(r*r - impact*impact) for r in boundaries if impact < r < math.hypot(grid.xcut, impact)]
        nodes.append(np.unique(np.r_[x, crossings, -np.array(crossings)]))
    n = max(len(row) - 1 for row in nodes)
    steps = np.zeros((len(b), n))
    r1 = np.zeros_like(steps)
    r2 = np.zeros_like(steps)
    bins = np.full((len(b), n), -1, dtype=np.int64)
    for j, (impact, row) in enumerate(zip(b, nodes)):
        lengths = np.diff(row)
        keep = slice(0, len(lengths))
        steps[j, keep] = lengths
        r1[j, keep] = np.hypot(impact, row[:-1] + SDIRK_DIAGONAL * lengths)
        r2[j, keep] = np.hypot(impact, row[1:])
        if annuli:
            mid_radius = np.hypot(impact, row[:-1] + 0.5 * lengths)
            assignment = np.searchsorted(annuli, mid_radius, side="right") - 1
            bins[j, keep] = np.where(assignment < len(annuli)-1, assignment, -1)
        else:
            bins[j, keep] = 0
    return b, wb, q, wq, steps, r1, r2, bins


def _support(mask):
    ids = np.zeros_like(mask, dtype=np.int64)
    counts = np.count_nonzero(mask, axis=1)
    for j, count in enumerate(counts):
        ids[j, :count] = np.flatnonzero(mask[j])
    return ids, counts


def _integrate_dense(first, second, step, bins, time_scale, flux, peak, thermal, observable,
                     free, driven, n_annuli, keep_history):
    """Independent batched dense solve of the full 14-by-14 implicit system."""
    ns = len(thermal)
    nb, nt = first.shape
    shape = (nb, len(time_scale))
    p = np.broadcast_to(thermal, (*shape, ns)).copy()
    counts = np.zeros(n_annuli)
    eye = np.eye(ns)
    history = [thermal.copy()] if keep_history else []
    increments = []
    min_pop, max_trace = 1.0, 0.0
    s = SDIRK_DIAGONAL
    for i in range(nt):
        dt = step[:, i, None] * time_scale
        i1, i2 = peak * first[:, i], peak * second[:, i]
        stage1 = eye - (s * dt)[:, :, None, None] * (free + i1[:, None, None] * driven)[:, None]
        p1 = np.linalg.solve(stage1, p[..., None])[..., 0]
        source = p + ((1 - s) / s) * (p1 - p)
        stage2 = eye - (s * dt)[:, :, None, None] * (free + i2[:, None, None] * driven)[:, None]
        p = np.linalg.solve(stage2, source[..., None])[..., 0]
        photons = dt * ((1 - s) * i1[:, None] * (p1 @ observable) + s * i2[:, None] * (p @ observable))
        kept = bins[:, i] >= 0
        np.add.at(counts, bins[kept, i], np.sum(flux * photons, axis=1)[kept])
        min_pop = min(min_pop, float(p.min()), float(p1.min()))
        max_trace = max(max_trace, float(np.max(np.abs(p.sum(axis=-1) - 1.0))),
                        float(np.max(np.abs(p1.sum(axis=-1) - 1.0))))
        if keep_history:
            history.append(p[0, 0].copy())
            increments.append(photons[0, 0])
    return counts, np.array(history), np.array(increments), min_pop, max_trace


@lru_cache(maxsize=1)
def _dipoles():
    # The stationary builder requires a positive reset, which does not enter
    # its signed dipoles, frequencies, or thermal populations. Strip this
    # unit reset from the free generator before propagating any history.
    manifold = species.build_manifold(species.RB85, "D1")
    probe = ProjectedProbe(manifold, 0.5 * manifold.gamma + 1.0, 1.0, resolution=MAGNETIC_PI)
    thermal = probe._thermal.copy()
    thermal.flags.writeable = False
    n = probe._nground
    free = probe._unforced - (thermal[:, None] - np.eye(len(thermal)))
    linked = np.zeros(free.shape, dtype=bool)
    g, e = probe._ground, probe._excited
    linked[g, e] = linked[e, g] = True
    up = _support(((free != 0) | linked)[:n, n:].T)
    down = _support(linked[n:, :n])
    return manifold, probe, thermal, free, up, down


class BallisticProbe85D1:
    """Local projected populations for pure 85Rb D1 at zero B.

    With no width override the saturated Rb density at the kinetic temperature
    sets the self-broadening. An independent cold-spot temperature requires an
    explicit optical HWHM, as supplied by the cell-spectrum adapter.
    """

    def __init__(self, temperature_k, optical_hwhm_rad_s=None):
        temp = _positive(temperature_k, "Kinetic temperature (K)")
        self.manifold, self._probe, self.thermal, self._free, self._up, self._down = _dipoles()
        if optical_hwhm_rad_s is None:
            density = species.number_density(species.RB85, temp)
            self_width = species.self_broadening(species.RB85, density, "D1", per_line=True)
            optical_hwhm_rad_s = 0.5 * (self.manifold.gamma + self_width)
        self.width = _positive(optical_hwhm_rad_s, "Intrinsic optical HWHM (rad/s)")
        if self.width < 0.5 * self.manifold.gamma:
            raise ValueError("Intrinsic optical width is below the natural HWHM.")
        self.speed_sigma = constants.thermal_speed(temp, species.RB85.mass)
        self.photon_energy_j = constants.PLANCK * self.manifold.nu0
        self._ng = self._probe._nground

    def _link_rates(self, detuning):
        p = self._probe
        return (0.5 * p._rabi_scale**2 * p._strengths * self.width
                / ((detuning[..., None] - p._frequencies)**2 + self.width**2))

    def weak_cross_sections(self, detuning_rad_s):
        """Infinite-domain, zero-power atom-frame absorption per atom in m²."""
        detuning = _axis(detuning_rad_s, "Detuning (rad/s)")
        return self.photon_energy_j * (self._link_rates(detuning) @ self.thermal[self._probe._ground])

    def _drive(self, delta):
        g, e = self._probe._ground, self._probe._excited
        rates = self._link_rates(np.asarray(delta))
        drive = np.zeros_like(self._free)
        np.add.at(drive, (g, g), -rates)
        np.add.at(drive, (e, e), -rates)
        drive[g, e] = drive[e, g] = rates
        obs = np.zeros(len(self.thermal))
        np.add.at(obs, g, rates)
        np.add.at(obs, e, -rates)
        return drive, obs

    def _integrate(self, delta, scale, first, second, step, bins, time_scale, flux,
                   n_annuli, accelerated, history=False):
        drive, obs = self._drive(delta)
        if accelerated and kernels.NUMBA_AVAILABLE:
            n = self._ng
            free = self._free
            blocks = (np.ascontiguousarray(free[:n, n:].T), np.ascontiguousarray(drive[:n, n:].T),
                      np.ascontiguousarray(drive[n:, :n]), np.diag(free[n:, n:]).copy(),
                      np.diag(drive[n:, n:]).copy(), np.diag(drive[:n, :n]).copy(), *self._up, *self._down)
            result = integrate_native(first, second, step, bins, time_scale, flux, scale,
                                      self.thermal, obs, blocks, n_annuli, history)
        else:
            result = _integrate_dense(first, second, step, bins, time_scale, flux, scale,
                                      self.thermal, obs, self._free, drive, n_annuli, history)
        counts, _, _, minimum, trace_error = result
        if (not np.isfinite(minimum) or not np.isfinite(trace_error) or
                minimum < -1e-8 or trace_error > 1e-8 or not np.all(np.isfinite(counts))):
            raise ArithmeticError(f"Unresolved ballistic populations: min {minimum:g}, trace error {trace_error:g}.")
        return result

    def _measure(self, grid, waist, annuli=(), support_cut=None):
        waist = _positive(waist, "1/e² Gaussian waist (m)")
        b, wb, q, wq, steps, r1, r2, bins = _geometry(grid, annuli, support_cut)
        speeds = q * self.speed_sigma
        # Incoming crossing flux per density and axial length; dt along each
        # path supplies the reciprocal speed and recovers the bulk Rayleigh law.
        flux = 2 * waist * wb[:, None] * (wq * q * np.exp(-q*q/2) * speeds)
        return steps, r1, r2, bins, waist / speeds, flux

    def path(self, detuning_rad_s, intensity_stage1_w_m2, intensity_stage2_w_m2, dt_s, *, accelerated=True):
        """One thermal-entry path, stage intensities and step durations in seconds.

        Returns populations including the initial state and net probe photons
        absorbed at each step (stimulated absorption minus stimulated emission).
        """
        delta = float(detuning_rad_s)
        first = _axis(intensity_stage1_w_m2, "First stage intensity")
        second = _axis(intensity_stage2_w_m2, "Second stage intensity")
        dt = np.broadcast_to(dt_s, first.shape).astype(float)
        if (not np.isfinite(delta) or first.shape != second.shape or np.any(first < 0)
                or np.any(second < 0) or np.any(dt <= 0) or not np.all(np.isfinite(dt))):
            raise ValueError("Path requires matching nonnegative intensities and finite positive time steps.")
        bins = np.zeros((1, len(dt)), dtype=np.int64)
        totals, populations, absorbed, _, _ = self._integrate(
            delta, 1.0, first[None], second[None], dt[None], bins, np.ones(1),
            np.ones((1, 1)), 1, accelerated, history=True)
        return BallisticPath(populations, float(totals[0]), absorbed)

    def cross_sections(self, detuning_rad_s, peak_intensity_w_m2, waist_m, *,
                       grid=TrajectoryGrid(), accelerated=True):
        """Gaussian absorption cross sections in m², shape (intensity, detuning).

        Integrated absorbed power divided by incident Gaussian power, per
        density and axial length. Zero input uses the same finite-grid measure.
        """
        delta = _axis(detuning_rad_s, "Detuning (rad/s)")
        intensity = _axis(peak_intensity_w_m2, "Peak intensity (W/m²)")
        if np.any(intensity < 0):
            raise ValueError("Incident peak intensity must be nonnegative.")
        waist_m = _positive(waist_m, "Waist (m)")
        steps, r1, r2, bins, scale, flux = self._measure(grid, waist_m)
        first, second = np.exp(-2*r1**2), np.exp(-2*r2**2)
        result = np.empty((len(intensity), len(delta)))
        area = np.pi * waist_m**2 / 2
        for i, peak in enumerate(intensity):
            if peak == 0.0:
                result[i] = self.weak_cross_sections(delta) * grid.weak_factor()
                continue
            for j, d in enumerate(delta):
                photons = self._integrate(d, peak, first, second, steps, bins, scale, flux, 1, accelerated)[0][0]
                result[i, j] = self.photon_energy_j * photons / (area * peak)
        return result

    def _radial_geometry(self, radial_nodes_waist, intensity_w_m2, annulus_edges_waist, waist_m, grid):
        nodes = _radial_axis(radial_nodes_waist, "Radial profile knots (r/w)")
        annuli = _radial_axis(annulus_edges_waist, "Annulus edges (r/w)")
        intensity = _axis(intensity_w_m2, "Radial profile intensity (W/m²)")
        if intensity.shape != nodes.shape or np.any(intensity < 0):
            raise ValueError("Set one nonnegative radial intensity per profile knot.")
        # A nonzero last knot followed by vacuum is a jump. Split there and
        # use the one-sided intensity of the traversed interval at its endpoint.
        cut = float(nodes[-1]) if intensity[-1] > 0.0 else None
        step, r1, r2, bins, scale, flux = self._measure(grid, waist_m, tuple(annuli), cut)
        first = np.interp(r1, nodes, intensity, right=0.0)
        second = np.interp(r2, nodes, intensity, right=0.0)
        if cut is not None:
            second = np.where(r1 < cut, np.interp(np.minimum(r2, cut), nodes, intensity), 0.0)
        return first, second, step, bins, scale, flux, len(annuli)-1

    def radial_rates(self, detuning_rad_s, radial_nodes_waist, intensity_w_m2, annulus_edges_waist,
                     waist_m, *, grid=TrajectoryGrid(), accelerated=True):
        """Integrated absorbed power per density and axial length [W m²], (freq, annulus).

        I(r) is piecewise linear from r=0 and zero beyond the last knot. The
        atom retains its history across annuli. Annuli outside the requested
        aperture do not contribute to the reported power.
        """
        delta = _axis(detuning_rad_s, "Detuning (rad/s)")
        args = self._radial_geometry(radial_nodes_waist, intensity_w_m2, annulus_edges_waist, waist_m, grid)
        return np.array([self.photon_energy_j * self._integrate(d, 1.0, *args, accelerated)[0] for d in delta])

    def radial_weak_weights(self, radial_nodes_waist, intensity_w_m2, annulus_edges_waist,
                            waist_m, *, grid=TrajectoryGrid()):
        """Incident watts in each annulus under the identical discrete crossing measure."""
        first, second, step, bins, scale, flux, n = self._radial_geometry(
            radial_nodes_waist, intensity_w_m2, annulus_edges_waist, waist_m, grid)
        weighted = ((1.0 - SDIRK_DIAGONAL) * first + SDIRK_DIAGONAL * second) * step
        values = weighted * (flux @ scale)[:, None]
        use = bins >= 0
        return np.bincount(bins[use], weights=values[use], minlength=n)
