"""Scalar paraxial fields with local complex susceptibility and FFT diffraction.

The carrier is exp(i*k*z-i*omega*t); |field|**2 is intensity in W/m^2.
Propagation solves dA/dz = i*laplacian(A)/(2*k) + i*k*chi(|A|^2)*A/2.
No aperture absorber or normalization conceals power escaping the sampled window.
FFT boundaries are periodic; the boundary-power guard rejects occupied edges.
This coherent field model does not infer a wavefront from an empirical M^2.
"""
from dataclasses import dataclass
import functools

import numpy as np
from scipy import special

from ._arrays import readonly_float_array
from .beam import peak_intensity


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ParaxialGrid:
    """Uniform transverse coordinates; field shape is (len(y_m), len(x_m))."""
    x_m: np.ndarray
    y_m: np.ndarray

    def __post_init__(self):
        for name in ("x_m", "y_m"):
            axis = readonly_float_array(getattr(self, name))
            if (axis.ndim != 1 or len(axis) < 8 or not np.isfinite(axis).all()
                    or np.any(np.diff(axis) <= 0)
                    or not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-10, atol=0)):
                raise ValueError("transverse axes must be finite, uniform, increasing and at least 8 samples")
            object.__setattr__(self, name, axis)

    @classmethod
    def square(cls, half_width_m, samples):
        half_width_m = _positive(half_width_m, "half_width_m")
        samples = _integer(samples, "samples", 8)
        if samples % 2:
            raise ValueError("square-grid samples must be even so the centre is sampled")
        axis = np.linspace(-half_width_m, half_width_m, samples, endpoint=False)
        return cls(axis, axis)

    @property
    def shape(self):
        return len(self.y_m), len(self.x_m)

    @property
    def pixel_area_m2(self):
        return float((self.x_m[1]-self.x_m[0])*(self.y_m[1]-self.y_m[0]))

    @property
    def radius_squared_m2(self):
        return self.y_m[:, None]**2 + self.x_m[None, :]**2


def _field(value, grid):
    value = np.array(value, dtype=complex, copy=True)
    if value.shape != grid.shape or not np.isfinite(value).all():
        raise ValueError("field must be finite and match grid shape (y, x)")
    return value


def gaussian_field(grid, power_w, waist_m, wavelength_m, curvature_radius_m=np.inf):
    """Gaussian at this plane, with 1/e^2 intensity radius ``waist_m``.

    The argument denotes the radius HERE, not necessarily a waist when curved.
    Positive curvature gives an outward/diverging wavefront. The power is the
    infinite-plane Gaussian power; finite-window truncation is not renormalized.
    """
    power_w = float(power_w)
    if not np.isfinite(power_w) or power_w < 0:
        raise ValueError("power_w must be finite and nonnegative")
    waist_m = _positive(waist_m, "waist_m")
    k = 2*np.pi/_positive(wavelength_m, "wavelength_m")
    curvature_radius_m = float(curvature_radius_m)
    if np.isnan(curvature_radius_m) or curvature_radius_m == 0:
        raise ValueError("curvature radius must be nonzero or infinite")
    r2 = grid.radius_squared_m2
    return np.sqrt(peak_intensity(power_w, waist_m))*np.exp(
        -r2/waist_m**2 + 0.5j*k*r2/curvature_radius_m)


def thin_lens(field, grid, *, wavelength_m, focal_length_m):
    """Apply exp(-i*k*r^2/(2*f)); positive f is a converging lens."""
    wavelength_m = _positive(wavelength_m, "wavelength_m")
    focal_length_m = float(focal_length_m)
    if not np.isfinite(focal_length_m) or focal_length_m == 0:
        raise ValueError("focal_length_m must be finite and nonzero")
    return _field(field, grid)*np.exp(
        -1j*np.pi*grid.radius_squared_m2/(wavelength_m*focal_length_m))


def relative_lens_power(field, reference_field, grid, *, wavelength_m,
                        fit_radius_m, intensity_floor=0.01):
    """Fit relative central-cut phase; positive inverse focal length focuses.

    This is a quadratic wavefront diagnostic, not a Gaussian fit to intensity.
    The RMS residual exposes aberrations within the declared fitting aperture.
    """
    k = 2*np.pi/_positive(wavelength_m, "wavelength_m")
    fit_radius_m = _positive(fit_radius_m, "fit_radius_m")
    if not 0 <= intensity_floor < 1:
        raise ValueError("intensity_floor must lie in [0, 1)")
    row = int(np.argmin(abs(grid.y_m)))
    a, b = _field(field, grid)[row], _field(reference_field, grid)[row]
    ia, ib = abs(a)**2, abs(b)**2
    mask = ((abs(grid.x_m) <= fit_radius_m)
            & (ia > intensity_floor*ia.max()) & (ib > intensity_floor*ib.max()))
    if np.count_nonzero(mask) < 5:
        raise ValueError("at least five illuminated samples are required for a phase fit")
    x_scaled = grid.x_m[mask]/fit_radius_m
    phase = np.unwrap(np.angle(a[mask]*b[mask].conj()))
    design = np.column_stack((np.ones_like(x_scaled), x_scaled, x_scaled**2))
    coefficients = np.linalg.lstsq(design, phase, rcond=None)[0]
    residual = phase-design@coefficients
    return {
        "lens_power_m_inverse": float(-2*coefficients[2]/(k*fit_radius_m**2)),
        "phase_rms_rad": float(np.sqrt(np.mean(residual**2))),
        "sample_count": int(len(phase)), "fit_radius_m": fit_radius_m,
        "centerline_y_m": float(grid.y_m[row]),
    }


@dataclass(frozen=True)
class ParaxialResult:
    z_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    centerline_intensity_w_m2: np.ndarray
    radius_m: np.ndarray
    power_w: np.ndarray
    final_field: np.ndarray
    boundary_power_fraction: np.ndarray
    max_boundary_power_fraction: float
    material_power_change_w: float
    power_balance_error_w: float
    centerline_y_m: float

    def __post_init__(self):
        for name in ("z_m", "x_m", "y_m", "centerline_intensity_w_m2",
                     "radius_m", "power_w", "boundary_power_fraction"):
            object.__setattr__(self, name, readonly_float_array(getattr(self, name)))
        field = np.array(self.final_field, dtype=complex, copy=True)
        field.setflags(write=False)
        object.__setattr__(self, "final_field", field)


def propagate(field, grid, *, wavelength_m, length_m, steps,
              susceptibility=None, snapshot_stride=1, boundary_tolerance=1e-5):
    """Strang FFT diffraction plus exponential-midpoint local material steps.

    ``susceptibility(intensity_w_m2)`` returns complex chi, broadcastable to the
    grid. None means vacuum. The midpoint intensity prediction accounts for
    saturation changing through absorption within each step (second-order).
    Snapshot storage contains only one central intensity cut and scalar moments;
    the final complex field remains available for downstream propagation.

    ``radius_m`` = sqrt(2*<x^2+y^2>) about the declared optical axis, equal to
    Gaussian 1/e^2 radius for centred beams. Edge fraction is power in the outer
    5 percent strip of either array dimension, monitored after every full step.
    Set boundary_tolerance=None only to inspect an under-resolved trial domain.
    ``material_power_change_w`` is signed: negative for absorption.
    """
    wavelength_m = _positive(wavelength_m, "wavelength_m")
    length_m = float(length_m)
    if not np.isfinite(length_m) or length_m < 0:
        raise ValueError("length_m must be finite and nonnegative")
    steps = _integer(steps, "steps")
    snapshot_stride = _integer(snapshot_stride, "snapshot_stride")
    if boundary_tolerance is not None:
        if not np.isfinite(boundary_tolerance) or not 0 < boundary_tolerance < 1:
            raise ValueError("boundary_tolerance must lie in (0, 1), or be None")
    if susceptibility is not None and not callable(susceptibility):
        raise TypeError("susceptibility must be callable or None")
    a = _field(field, grid)
    k, dz = 2*np.pi/wavelength_m, length_m/steps
    qx = 2*np.pi*np.fft.fftfreq(len(grid.x_m), d=grid.x_m[1]-grid.x_m[0])
    qy = 2*np.pi*np.fft.fftfreq(len(grid.y_m), d=grid.y_m[1]-grid.y_m[0])
    half_diffraction = np.exp(-0.25j*dz*(qy[:, None]**2+qx[None, :]**2)/k)
    edge = np.zeros(grid.shape, dtype=bool)
    by, bx = max(1, int(np.ceil(.05*grid.shape[0]))), max(1, int(np.ceil(.05*grid.shape[1])))
    edge[:by, :] = edge[-by:, :] = True
    edge[:, :bx] = edge[:, -bx:] = True
    row = int(np.argmin(abs(grid.y_m)))
    r2, area = grid.radius_squared_m2, grid.pixel_area_m2
    z_values, cuts, radii, powers, edges = [], [], [], [], []
    material_change = 0.0
    largest_edge = 0.0

    def inspect(index, save):
        nonlocal largest_edge
        intensity = abs(a)**2
        total = float(np.sum(intensity))
        edge_fraction = float(np.sum(intensity[edge])/total) if total else 0.0
        largest_edge = max(largest_edge, edge_fraction)
        if boundary_tolerance is not None and edge_fraction > boundary_tolerance:
            raise ValueError(
                f"transverse boundary power fraction {edge_fraction:.3g} exceeds "
                f"{boundary_tolerance:.3g} at z={index*dz:.6g} m; enlarge the grid window")
        if save:
            z_values.append(index*dz)
            cuts.append(intensity[row].copy())
            radii.append(np.sqrt(2*np.sum(intensity*r2)/total) if total else np.nan)
            powers.append(total*area)
            edges.append(edge_fraction)

    def chi(intensity):
        result = np.asarray(susceptibility(intensity), dtype=complex)
        try:
            result = np.broadcast_to(result, grid.shape)
        except ValueError as error:
            raise ValueError("susceptibility must broadcast to grid shape") from error
        if not np.isfinite(result).all():
            raise ValueError("susceptibility returned nonfinite values")
        return result

    inspect(0, True)
    if length_m > 0:
        for index in range(1, steps+1):
            a = np.fft.ifft2(np.fft.fft2(a)*half_diffraction)
            if susceptibility is not None:
                intensity = abs(a)**2
                before = float(np.sum(intensity))*area
                with np.errstate(over="raise", invalid="raise"):
                    mid_intensity = intensity*np.exp(-.5*k*dz*chi(intensity).imag)
                    a *= np.exp(.5j*k*dz*chi(mid_intensity))
                material_change += float(np.sum(abs(a)**2))*area-before
            a = np.fft.ifft2(np.fft.fft2(a)*half_diffraction)
            if not np.isfinite(a).all():
                raise ValueError("field became nonfinite during propagation")
            inspect(index, index % snapshot_stride == 0 or index == steps)
    return ParaxialResult(
        np.asarray(z_values), grid.x_m, grid.y_m, np.asarray(cuts),
        np.asarray(radii), np.asarray(powers), a, np.asarray(edges), largest_edge,
        material_change, powers[-1]-powers[0]-material_change,
        float(grid.y_m[row]))


@functools.lru_cache(maxsize=32)
def _qdht_unit_tables(samples):
    """Normalized 0th-order QDHT tables (Yu 1998 / Guizar-Sicairos 2004).

    Independent of physical aperture ``R``; cached by radial node count ``N``.
    """
    zeros = special.jn_zeros(0, samples + 1)
    alpha = zeros[:samples]
    alpha_max = float(zeros[samples])
    j1 = special.j1(alpha)
    scale = np.abs(j1)
    arg = np.outer(alpha, alpha) / alpha_max
    transform = 2.0 * special.j0(arg) / (alpha_max * np.outer(scale, scale))
    unit_weights = 4.0 * np.pi / ((alpha_max ** 2) * (j1 ** 2))
    for arr in (alpha, scale, transform, unit_weights):
        arr.setflags(write=False)
    return alpha, alpha_max, scale, transform, unit_weights


@functools.lru_cache(maxsize=16)
def _qdht_2d_hankel_table(samples, max_radius_m, x0, x1, nx, y0, y1, ny):
    """Cached exact 0th-order Bessel evaluation matrix for 2D ParaxialGrid reconstruction."""
    alpha, _, _, _, _ = _qdht_unit_tables(samples)
    kr = alpha / max_radius_m
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    r_flat = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2).ravel()
    inside = r_flat <= max_radius_m
    if not np.any(inside):
        return inside, np.empty(0, dtype=np.intp), np.empty((0, samples), dtype=float)
    unique_r, inv = np.unique(r_flat[inside], return_inverse=True)
    j0_mat = special.j0(np.outer(unique_r, kr))
    for arr in (inside, inv, j0_mat):
        arr.setflags(write=False)
    return inside, inv, j0_mat


@dataclass(frozen=True)
class RadialQDHTGrid:
    """Axisymmetric 0th-order QDHT radial grid on ``[0, max_radius_m]`` with Dirichlet boundary.

    Unlike :class:`ParaxialGrid` (square periodic FFT domain ``[-R, R]^2``),
    QDHT imposes a circular Dirichlet boundary ``A(R) = 0`` at ``r = max_radius_m``.
    """
    max_radius_m: float
    samples: int

    def __post_init__(self):
        object.__setattr__(self, "max_radius_m", _positive(self.max_radius_m, "max_radius_m"))
        object.__setattr__(self, "samples", _integer(self.samples, "samples", 8))

    @property
    def _tables(self):
        return _qdht_unit_tables(self.samples)

    @property
    def r_m(self):
        alpha, alpha_max, _, _, _ = self._tables
        return readonly_float_array(self.max_radius_m * alpha / alpha_max)

    @property
    def kr_rad_m(self):
        alpha, _, _, _, _ = self._tables
        return readonly_float_array(alpha / self.max_radius_m)

    @property
    def weights_m2(self):
        _, _, _, _, unit_weights = self._tables
        return readonly_float_array((self.max_radius_m ** 2) * unit_weights)

    @property
    def scale(self):
        return self._tables[2]

    @property
    def transform(self):
        return self._tables[3]

    def evaluate(self, radial_field, r_eval_m):
        """Exact 0th-order Hankel evaluation at arbitrary radii ``r_eval_m`` (including ``r=0``).

        Radii outside ``max_radius_m`` evaluate to ``0`` (circular Dirichlet boundary).
        """
        a = np.asarray(radial_field, dtype=complex)
        if a.shape != (self.samples,) or not np.isfinite(a).all():
            raise ValueError("radial_field must be finite and match radial grid samples")
        r_eval = np.asarray(r_eval_m, dtype=float)
        if not np.isfinite(r_eval).all() or np.any(r_eval < 0):
            raise ValueError("r_eval_m must be finite and nonnegative")
        alpha, alpha_max, scale, transform, _ = self._tables
        spec = transform @ (a / scale)
        coeff = (2.0 / alpha_max) * (spec / scale)
        kr = alpha / self.max_radius_m
        flat = r_eval.ravel()
        out = np.zeros(flat.shape, dtype=complex)
        inside = flat <= self.max_radius_m
        if np.any(inside):
            unique_r, inv = np.unique(flat[inside], return_inverse=True)
            vals = special.j0(np.outer(unique_r, kr)) @ coeff
            out[inside] = vals[inv]
        return out.reshape(r_eval.shape)

    def reconstruct_2d(self, radial_field, cart_grid):
        """On-demand 2D complex field reconstruction on a :class:`ParaxialGrid`."""
        if not isinstance(cart_grid, ParaxialGrid):
            raise TypeError("cart_grid must be a ParaxialGrid")
        a = np.asarray(radial_field, dtype=complex)
        if a.shape != (self.samples,) or not np.isfinite(a).all():
            raise ValueError("radial_field must be finite and match radial grid samples")
        _, alpha_max, scale, transform, _ = self._tables
        spec = transform @ (a / scale)
        coeff = (2.0 / alpha_max) * (spec / scale)
        inside, inv, j0_mat = _qdht_2d_hankel_table(
            self.samples, self.max_radius_m,
            float(cart_grid.x_m[0]), float(cart_grid.x_m[-1]), len(cart_grid.x_m),
            float(cart_grid.y_m[0]), float(cart_grid.y_m[-1]), len(cart_grid.y_m),
        )
        out = np.zeros(cart_grid.shape[0] * cart_grid.shape[1], dtype=complex)
        if inv.size:
            out[inside] = (j0_mat @ coeff)[inv]
        return out.reshape(cart_grid.shape)


def gaussian_radial_field(radial_grid, power_w, waist_m, wavelength_m, curvature_radius_m=np.inf):
    """Axisymmetric Gaussian on a :class:`RadialQDHTGrid`."""
    if not isinstance(radial_grid, RadialQDHTGrid):
        raise TypeError("radial_grid must be a RadialQDHTGrid")
    power_w = float(power_w)
    if not np.isfinite(power_w) or power_w < 0:
        raise ValueError("power_w must be finite and nonnegative")
    waist_m = _positive(waist_m, "waist_m")
    k = 2 * np.pi / _positive(wavelength_m, "wavelength_m")
    curvature_radius_m = float(curvature_radius_m)
    if np.isnan(curvature_radius_m) or curvature_radius_m == 0:
        raise ValueError("curvature radius must be nonzero or infinite")
    r2 = radial_grid.r_m ** 2
    return np.sqrt(peak_intensity(power_w, waist_m)) * np.exp(
        -r2 / waist_m**2 + 0.5j * k * r2 / curvature_radius_m)


@dataclass(frozen=True)
class RadialParaxialResult:
    """Axisymmetric QDHT propagation output with on-demand 2D reconstruction."""
    z_m: np.ndarray
    r_m: np.ndarray
    radial_intensity_w_m2: np.ndarray
    on_axis_intensity_w_m2: np.ndarray
    radius_m: np.ndarray
    power_w: np.ndarray
    final_radial_field: np.ndarray
    boundary_power_fraction: np.ndarray
    max_boundary_power_fraction: float
    material_power_change_w: float
    power_balance_error_w: float
    radial_grid: RadialQDHTGrid

    def __post_init__(self):
        for name in ("z_m", "r_m", "radial_intensity_w_m2", "on_axis_intensity_w_m2",
                     "radius_m", "power_w", "boundary_power_fraction"):
            object.__setattr__(self, name, readonly_float_array(getattr(self, name)))
        field = np.array(self.final_radial_field, dtype=complex, copy=True)
        field.setflags(write=False)
        object.__setattr__(self, "final_radial_field", field)

    def to_2d_field(self, cart_grid):
        """Reconstruct ``final_radial_field`` onto a 2D :class:`ParaxialGrid`."""
        return self.radial_grid.reconstruct_2d(self.final_radial_field, cart_grid)


def propagate_axisymmetric(radial_field, radial_grid, *, wavelength_m, length_m, steps,
                           susceptibility=None, snapshot_stride=1, boundary_tolerance=1e-5):
    """Axisymmetric 0th-order QDHT Strang diffraction + exponential-midpoint material steps.

    Opt-in radial backend for axisymmetric single-beam propagation. Boundary
    condition is circular Dirichlet at ``r = radial_grid.max_radius_m`` (not
    periodic square as in :func:`propagate`); outer 5 percent radial annulus is
    monitored against ``boundary_tolerance``.
    """
    if not isinstance(radial_grid, RadialQDHTGrid):
        raise TypeError("radial_grid must be a RadialQDHTGrid")
    wavelength_m = _positive(wavelength_m, "wavelength_m")
    length_m = float(length_m)
    if not np.isfinite(length_m) or length_m < 0:
        raise ValueError("length_m must be finite and nonnegative")
    steps = _integer(steps, "steps")
    snapshot_stride = _integer(snapshot_stride, "snapshot_stride")
    if boundary_tolerance is not None:
        if not np.isfinite(boundary_tolerance) or not 0 < boundary_tolerance < 1:
            raise ValueError("boundary_tolerance must lie in (0, 1), or be None")
    if susceptibility is not None and not callable(susceptibility):
        raise TypeError("susceptibility must be callable or None")
    a = np.array(radial_field, dtype=complex, copy=True)
    if a.shape != (radial_grid.samples,) or not np.isfinite(a).all():
        raise ValueError("radial_field must be finite and match radial grid samples")

    k = 2.0 * np.pi / wavelength_m
    dz = length_m / steps
    r = radial_grid.r_m
    r2 = r ** 2
    kr = radial_grid.kr_rad_m
    weights = radial_grid.weights_m2
    scale = radial_grid.scale
    T = radial_grid.transform
    half_diff = np.exp(-0.25j * dz * (kr ** 2) / k)
    edge = r >= 0.95 * radial_grid.max_radius_m
    if not np.any(edge):
        edge[-1] = True
    alpha, alpha_max, _, _, _ = radial_grid._tables
    on_axis_coeff = (2.0 / alpha_max) / scale

    z_values, rad_ints, on_axis_ints, radii, powers, edges = [], [], [], [], [], []
    material_change = 0.0
    largest_edge = 0.0

    def inspect(index, save):
        nonlocal largest_edge
        intensity = np.abs(a) ** 2
        total_power = float(np.sum(intensity * weights))
        edge_fraction = float(np.sum(intensity[edge] * weights[edge]) / total_power) if total_power else 0.0
        largest_edge = max(largest_edge, edge_fraction)
        if boundary_tolerance is not None and edge_fraction > boundary_tolerance:
            raise ValueError(
                f"transverse boundary power fraction {edge_fraction:.3g} exceeds "
                f"{boundary_tolerance:.3g} at z={index*dz:.6g} m; enlarge the grid window")
        if save:
            z_values.append(index * dz)
            rad_ints.append(intensity.copy())
            spec = T @ (a / scale)
            a0 = complex(np.dot(on_axis_coeff, spec))
            on_axis_ints.append(abs(a0) ** 2)
            radii.append(np.sqrt(2.0 * float(np.sum(intensity * r2 * weights)) / total_power)
                         if total_power else np.nan)
            powers.append(total_power)
            edges.append(edge_fraction)

    def chi(intensity):
        result = np.asarray(susceptibility(intensity), dtype=complex)
        try:
            result = np.broadcast_to(result, r.shape)
        except ValueError as error:
            raise ValueError("susceptibility must broadcast to radial grid shape") from error
        if not np.isfinite(result).all():
            raise ValueError("susceptibility returned nonfinite values")
        return result

    inspect(0, True)
    if length_m > 0:
        if susceptibility is None:
            # Vacuum diffraction is exact in QDHT spectral space (zero splitting error).
            spec0 = T @ (a / scale)
            for index in range(1, steps + 1):
                save = (index % snapshot_stride == 0) or (index == steps)
                a = scale * (T @ (np.exp(-0.5j * (index * dz) * (kr ** 2) / k) * spec0))
                if not np.isfinite(a).all():
                    raise ValueError("field became nonfinite during propagation")
                inspect(index, save)
        else:
            for index in range(1, steps + 1):
                a = scale * (T @ (half_diff * (T @ (a / scale))))
                intensity = np.abs(a) ** 2
                before = float(np.sum(intensity * weights))
                with np.errstate(over="raise", invalid="raise"):
                    mid_intensity = intensity * np.exp(-0.5 * k * dz * chi(intensity).imag)
                    a *= np.exp(0.5j * k * dz * chi(mid_intensity))
                material_change += float(np.sum((np.abs(a) ** 2) * weights)) - before
                a = scale * (T @ (half_diff * (T @ (a / scale))))
                if not np.isfinite(a).all():
                    raise ValueError("field became nonfinite during propagation")
                inspect(index, index % snapshot_stride == 0 or index == steps)

    return RadialParaxialResult(
        np.asarray(z_values), r, np.asarray(rad_ints), np.asarray(on_axis_ints),
        np.asarray(radii), np.asarray(powers), a, np.asarray(edges), largest_edge,
        material_change, powers[-1] - powers[0] - material_change, radial_grid)
