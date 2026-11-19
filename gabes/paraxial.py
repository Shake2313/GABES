"""Scalar paraxial fields with local complex susceptibility and FFT diffraction.

The carrier is exp(i*k*z-i*omega*t); |field|**2 is intensity in W/m^2.
Propagation solves dA/dz = i*laplacian(A)/(2*k) + i*k*chi(|A|^2)*A/2.
No aperture absorber or normalization conceals power escaping the sampled window.
FFT boundaries are periodic; the boundary-power guard rejects occupied edges.
This coherent field model does not infer a wavefront from an empirical M^2.
"""
from dataclasses import dataclass

import numpy as np

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
