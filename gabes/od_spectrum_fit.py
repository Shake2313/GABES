"""Bounded calibration of absolutely calibrated weak-probe OD spectra."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from . import observables, species
from .experimental_alignment import AxisAlignment
from .experimental_csv import ExperimentalCSV, transform_x
from .schemes.sas import SASScheme, _probe_power_mw
from .vapor_cell import CellTemperatureState


MAX_FIT_POINTS = 1200
GLOBAL_LOG2_SAMPLES = 6
LOCAL_STARTS = 3
LOCAL_MAX_NFEV = 40
PARAMETERS = {
    "temp_c": ("Cold-spot temperature", "°C", 20.0, 200.0),
    "cell_mm": ("Cell length", "mm", 0.5, 200.0),
    "extra_width_mhz": ("Extra homogeneous FWHM", "MHz", 0.0, 100.0),
    "offset_mhz": ("Frequency offset", "MHz", -150.0, 150.0),
}


def _calibration_vector(values, name, *, minimum_size=1):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < minimum_size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 1-D array with at least {minimum_size} values.")
    return result


def _interpolate_calibration(x, nodes, values, name):
    if not np.isfinite(x).all():
        raise ValueError(f"{name} requires finite evaluation coordinates.")
    if np.any(x < nodes[0]) or np.any(x > nodes[-1]):
        raise ValueError(f"{name} must bracket all evaluated samples.")
    return np.interp(x, nodes, values)


@dataclass(frozen=True)
class ODFrequencyCalibration:
    """Fixed piecewise-linear marker map from the raw CSV x coordinate to MHz.

    Supply markers in increasing raw-axis order. Detuning can increase or
    decrease, and is relative to the selected isotope or mixture's line origin.
    The same map must cover the full selected acquisition branch.
    """

    raw_axis: tuple[float, ...]
    frequency_mhz: tuple[float, ...]
    source: str = "supplied frequency markers"

    def __post_init__(self):
        raw_axis = _calibration_vector(self.raw_axis, "Raw marker coordinates", minimum_size=2)
        frequency = _calibration_vector(self.frequency_mhz, "Marker detunings", minimum_size=2)
        if (raw_axis.shape != frequency.shape or np.any(np.diff(raw_axis) <= 0)
                or not (np.all(np.diff(frequency) > 0) or np.all(np.diff(frequency) < 0))):
            raise ValueError("Frequency markers need increasing raw coordinates and strictly monotonic detunings.")
        object.__setattr__(self, "raw_axis", tuple(raw_axis.tolist()))
        object.__setattr__(self, "frequency_mhz", tuple(frequency.tolist()))

    def detuning_mhz(self, raw_axis):
        return _interpolate_calibration(
            np.asarray(raw_axis, dtype=float), self.raw_axis, self.frequency_mhz, "Frequency markers"
        )

    def readout(self):
        return dict(source=self.source, raw_axis=list(self.raw_axis),
                    frequency_mhz=list(self.frequency_mhz))


@dataclass(frozen=True)
class ODOpticalCalibration:
    """Fixed optical response on the absolute detector-transmission scale.

    Laser offsets are added to nominal detuning, with weights proportional to
    incident power. Bypass is the optical power fraction unaffected by atoms.
    The optional baseline is the no-atom throughput left after dark/reference
    calibration, tabulated versus nominal detuning. It applies to both terms.
    """

    laser_offsets_mhz: tuple[float, ...] = (0.0,)
    laser_weights: tuple[float, ...] = (1.0,)
    bypass_fraction: float = 0.0
    baseline_frequency_mhz: tuple[float, ...] = ()
    baseline_transmission: tuple[float, ...] = ()
    source: str = "supplied optical calibration"

    def __post_init__(self):
        offsets = _calibration_vector(self.laser_offsets_mhz, "Laser offsets")
        weights = _calibration_vector(self.laser_weights, "Laser weights")
        total = float(weights.sum())
        if offsets.shape != weights.shape or np.any(weights < 0) or not 0 < total < np.inf:
            raise ValueError("Laser offsets and nonnegative weights must match and have finite positive total weight.")
        bypass = float(self.bypass_fraction)
        if not 0.0 <= bypass <= 1.0:
            raise ValueError("Optical bypass fraction must be between zero and one.")
        axis = _calibration_vector(self.baseline_frequency_mhz, "Baseline frequencies", minimum_size=0)
        baseline = _calibration_vector(self.baseline_transmission, "Baseline transmission", minimum_size=0)
        if (axis.shape != baseline.shape or axis.size == 1 or np.any(np.diff(axis) <= 0)
                or np.any(baseline <= 0)):
            raise ValueError("Supply at least two matching baseline samples with increasing frequencies and positive transmission.")
        object.__setattr__(self, "laser_offsets_mhz", tuple(offsets.tolist()))
        object.__setattr__(self, "laser_weights", tuple((weights / total).tolist()))
        object.__setattr__(self, "bypass_fraction", bypass)
        object.__setattr__(self, "baseline_frequency_mhz", tuple(axis.tolist()))
        object.__setattr__(self, "baseline_transmission", tuple(baseline.tolist()))

    @property
    def is_identity(self):
        return (self.laser_offsets_mhz == (0.0,) and self.bypass_fraction == 0.0
                and not self.baseline_frequency_mhz)

    def transmission(self, frequency_mhz, atomic_transmission):
        """Evaluate the detected power average on the supplied frequency samples."""
        frequency_mhz = _calibration_vector(frequency_mhz, "Evaluation frequencies")
        if self.is_identity:
            return atomic_transmission(frequency_mhz)
        if self.baseline_frequency_mhz:
            baseline = _interpolate_calibration(
                frequency_mhz, self.baseline_frequency_mhz, self.baseline_transmission, "Optical baseline"
            )
        else:
            baseline = 1.0
        average = np.zeros_like(frequency_mhz)
        for offset, weight in zip(self.laser_offsets_mhz, self.laser_weights):
            if weight > 0.0:
                average += weight * atomic_transmission(frequency_mhz + offset)
        return baseline * (self.bypass_fraction + (1.0 - self.bypass_fraction) * average)

    def readout(self):
        return dict(source=self.source, laser_offsets_mhz=list(self.laser_offsets_mhz),
                    laser_weights=list(self.laser_weights), bypass_fraction=self.bypass_fraction,
                    baseline_frequency_mhz=list(self.baseline_frequency_mhz),
                    baseline_transmission=list(self.baseline_transmission))


@dataclass(frozen=True)
class ODCellCalibration:
    """Fixed temperature inputs from the shared cell model and/or a known length.

    Fits using these conditions report uncertainty conditional on their values.
    """

    temperature: CellTemperatureState | None = None
    cell_length_mm: float | None = None
    source: str = "supplied cell constraints"

    def __post_init__(self):
        if self.temperature is None and self.cell_length_mm is None:
            raise ValueError("Supply a fixed cell temperature state, a cell length, or both.")
        if self.temperature is not None:
            cold = self.temperature.cold_spot_temp_c
            path = self.temperature.effective_vapor_temp_c
            if not 20.0 <= cold <= 200.0 or not cold <= path <= 300.0:
                raise ValueError("Fixed cell temperatures must be in the supported ranges and the path must be at least as warm as the cold spot.")
        if self.cell_length_mm is not None:
            length = float(self.cell_length_mm)
            if not 0.5 <= length <= 200.0:
                raise ValueError("Fixed cell length must be between 0.5 and 200 mm.")
            object.__setattr__(self, "cell_length_mm", length)

    def apply(self, params):
        result = dict(params)
        if self.temperature is not None:
            cold = self.temperature.cold_spot_temp_c
            path = self.temperature.effective_vapor_temp_c
            result.update(temp_c=cold, doppler_temp_c=path, constrain_doppler_temp=False)
        if self.cell_length_mm is not None:
            result["cell_mm"] = self.cell_length_mm
        return result

    def readout(self):
        result = dict(source=self.source, cell_length_mm=self.cell_length_mm)
        if self.temperature is not None:
            state = self.temperature
            result["temperature"] = dict(
                cold_spot_temp_c=state.cold_spot_temp_c, cold_spot_source=state.cold_spot_source,
                doppler_temp_c=state.effective_vapor_temp_c, doppler_temp_source=state.effective_source,
            )
        return result


@dataclass(frozen=True)
class ODFitInput:
    frequency_mhz: np.ndarray
    transmission: np.ndarray
    sigma: np.ndarray
    branch: int
    source_points: int
    uncertainty_source: str
    frequency_calibration: ODFrequencyCalibration | None = None


def od_fit_unavailable(params):
    if params.get("species") not in species.SPECIES_ORDER:
        return "Choose a resolved Rb or Cs medium."
    if float(params.get("pump_power_mw", 0.0)) != 0.0 or _probe_power_mw(params) != 0.0:
        return "Set pump and probe power to zero for weak-probe OD fitting."
    if params.get("cell_propagation", "Uniform") != "Uniform":
        return "Choose the Uniform cell for OD fitting."
    return None


def prepare_od_fit(trace: ExperimentalCSV, alignment: AxisAlignment | None = None, *,
                   branch=0, frequency_calibration: ODFrequencyCalibration | None = None):
    """Use one raw sweep and the declared detector law.

    A supplied marker map fixes the detuning directly from raw x coordinates.
    Otherwise the legacy display alignment determines the fit axis.
    """
    if not trace.is_absolute_transmission:
        raise ValueError("OD fitting requires an absolute transmission calibration.")
    if not 0 <= branch < len(trace.branches):
        raise ValueError("Select an acquisition branch for OD fitting.")
    sweep = trace.branches[branch]
    if frequency_calibration is not None:
        x = frequency_calibration.detuning_mhz(sweep.detuning)
    else:
        if alignment is None:
            raise ValueError("Supply a frequency calibration or the display alignment.")
        pivot = 0.5 * float(trace.detuning.min()) + 0.5 * float(trace.detuning.max())
        x = transform_x(sweep.detuning, scale=alignment.scale, shift=alignment.shift,
                        reverse=alignment.reverse, pivot=pivot) * 1e3  # SAS axis is GHz.
    y = trace.calibrate_signal(sweep.raw_signal)
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if x.size < 16 or np.unique(x).size < 16:
        raise ValueError("OD fitting requires at least 16 distinct frequency samples.")

    # For an irregular frequency grid, subtract linear interpolation of each
    # point's two neighbors. Scale the resulting high-pass residual back to
    # the noise of one raw sample. Robust bin estimates limit line-edge bias.
    span = x[2:] - x[:-2]
    valid = span > 0
    fraction = (x[1:-1][valid] - x[:-2][valid]) / span[valid]
    residual = (y[1:-1][valid] - (1 - fraction) * y[:-2][valid] - fraction * y[2:][valid])
    residual /= np.sqrt(1 + (1 - fraction)**2 + fraction**2)
    centers, spreads = [], []
    for group in np.array_split(np.arange(residual.size), min(12, max(1, residual.size // 18))):
        if group.size < 4:
            continue
        values = residual[group]
        spread = 1.4826 * np.median(np.abs(values - np.median(values)))
        if spread > 0 and np.isfinite(spread):
            centers.append(float(np.mean(x[1:-1][valid][group])))
            spreads.append(float(spread))
    diagnostic = trace.correction_diagnostics
    detector_sigma = float(diagnostic.noise_sigma / abs(diagnostic.calibration_gain))
    if spreads:
        floor = max(0.2 * detector_sigma, 0.2 * float(np.median(spreads)), np.finfo(float).eps)
        sigma = np.maximum(np.interp(x, centers, spreads), floor)
        source = "Interpolated robust noise estimates from raw branch frequency bins"
    elif detector_sigma > 0 and np.isfinite(detector_sigma):
        sigma = np.full_like(x, detector_sigma)
        source = "Raw detector noise estimate converted to transmission"
    else:
        raise ValueError("Noise could not be estimated from this trace. Supply raw measurements with noise.")
    keep = np.linspace(0, x.size - 1, min(x.size, MAX_FIT_POINTS), dtype=int)
    return ODFitInput(x[keep], y[keep], sigma[keep], branch, int(x.size), source,
                      frequency_calibration=frequency_calibration)


def _require_calibrated_od_scope(params):
    components = species.SPECIES[params["species"]]
    if (any(iso not in (species.RB85, species.RB87) for iso, _ in components)
            or params.get("line") not in ("D1", "D2")
            or params.get("paraffin_coated", False)
            or float(params.get("ne_pressure_torr", 0.0)) != 0.0):
        raise ValueError("The new calibration inputs cover uncoated, buffer-free Rb D1 and D2 cells.")


def od_transmission(frequency_mhz, params, *, extra_width_mhz=0.0, offset_mhz=0.0,
                    optics: ODOpticalCalibration | None = None,
                    cell: ODCellCalibration | None = None):
    """Evaluate the resolved SAS pump-off model at the supplied detunings in MHz."""
    reason = od_fit_unavailable(params)
    if reason:
        raise ValueError(reason)
    if optics is not None or cell is not None:
        _require_calibrated_od_scope(params)
    if cell is not None:
        params = cell.apply(params)

    def atomic_transmission(frequency):
        scan = (frequency - offset_mhz) * (2 * np.pi * 1e6)
        raw = SASScheme()._compute_species(
            params, scan_rad_s=scan, extra_homogeneous_fwhm_hz=extra_width_mhz * 1e6)
        alpha = raw["alpha_unit"] * float(params.get("line_strength", 1.0))
        return observables.transmission(alpha, float(params["cell_mm"]) * 1e-3)

    frequency = np.asarray(frequency_mhz, dtype=float)
    if optics is None:
        return atomic_transmission(frequency)
    return optics.transmission(frequency, atomic_transmission)


def fit_od_spectrum(data: ODFitInput, params, *, free=None,
                    fit_range_mhz=None, heldout_range_mhz=None, bounds=None,
                    optics: ODOpticalCalibration | None = None,
                    cell: ODCellCalibration | None = None):
    """Search bounded parameter space and refine weighted residuals on fit points.

    Ranges apply to the calibrated input axis before fitting the offset. The
    covariance is local, conditional on fixed model and calibration inputs.
    By default the fit estimates cold-spot temperature and frequency offset,
    omitting either when fixed by the corresponding calibration. An explicit
    offset on a marker-anchored axis is interpreted as a residual offset.
    """
    reason = od_fit_unavailable(params)
    if reason:
        raise ValueError(reason)
    markers = data.frequency_calibration
    if optics is not None or cell is not None or markers is not None:
        _require_calibrated_od_scope(params)
    fixed = set()
    if cell is not None:
        params = cell.apply(params)
        if cell.temperature is not None:
            fixed.add("temp_c")
        if cell.cell_length_mm is not None:
            fixed.add("cell_mm")
    if free is None:
        free = tuple(key for key in ("temp_c", "offset_mhz")
                     if key not in fixed and (key != "offset_mhz" or markers is None))
        if not free:
            raise ValueError("The default fit parameters are fixed. Choose an explicitly free parameter.")
    free = tuple(free)
    if fixed.intersection(free):
        raise ValueError("A fixed cell constraint cannot also be a free fit parameter.")
    if not free or len(free) != len(set(free)) or any(key not in PARAMETERS for key in free):
        raise ValueError("Choose distinct physical fit parameters from the supported list.")
    x, y, sigma = (np.asarray(values, dtype=float) for values in
                   (data.frequency_mhz, data.transmission, data.sigma))
    if (x.ndim != 1 or x.size > MAX_FIT_POINTS or y.shape != x.shape or sigma.shape != x.shape or
            not all(np.isfinite(v).all() for v in (x, y, sigma)) or np.any(sigma <= 0)):
        raise ValueError(f"Fit at most {MAX_FIT_POINTS} finite samples with positive uncertainty.")

    def points(interval, *, default):
        if interval is None:
            return np.full(x.shape, default, dtype=bool)
        lo, hi = map(float, interval)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("Detuning ranges must be finite and increasing.")
        return (x >= lo) & (x <= hi)

    if heldout_range_mhz is not None and fit_range_mhz is None:
        raise ValueError("Declare a fit range when requesting held-out residuals.")
    train = points(fit_range_mhz, default=True)
    heldout = points(heldout_range_mhz, default=False)
    if np.any(train & heldout) or np.unique(x[train]).size < max(16, 3 * len(free)):
        raise ValueError("Fit and held-out points must be disjoint. The fit needs at least 16 distinct points.")
    if heldout_range_mhz is not None and np.unique(x[heldout]).size < 4:
        raise ValueError("The held-out interval must contain at least four distinct samples.")
    domains = dict(bounds or {})
    if domains.keys() - set(free):
        raise ValueError("Fit bounds apply only to selected physical parameters.")
    base = {"temp_c": float(params["temp_c"]), "cell_mm": float(params["cell_mm"]),
            "extra_width_mhz": 0.0, "offset_mhz": 0.0}
    limits = np.array([domains.get(k, PARAMETERS[k][2:]) for k in free], dtype=float)
    registered = np.array([PARAMETERS[k][2:] for k in free])
    if (limits.shape != (len(free), 2) or not np.isfinite(limits).all() or
            np.any(limits[:, 0] >= limits[:, 1]) or
            np.any(limits[:, 0] < registered[:, 0]) or np.any(limits[:, 1] > registered[:, 1])):
        raise ValueError("Parameter bounds must be finite, increasing and within the supported physical ranges.")
    lower, upper = limits.T
    width = upper - lower
    def model(frequency, normalized):
        values = dict(base, **dict(zip(free, lower + normalized * width)))
        return od_transmission(frequency, dict(params, temp_c=values["temp_c"], cell_mm=values["cell_mm"]),
                               extra_width_mhz=values["extra_width_mhz"],
                               offset_mhz=values["offset_mhz"], optics=optics)

    x_train, y_train, sigma_train = x[train], y[train], sigma[train]
    calls = 0

    def residual(normalized):
        nonlocal calls
        calls += 1
        return (model(x_train, normalized) - y_train) / sigma_train

    initial = np.clip((np.array([base[k] for k in free]) - lower) / width, 0.0, 1.0)
    # Deterministic broad search followed by independently bounded refinements.
    candidates = np.vstack([initial, qmc.Sobol(len(free), scramble=False).random_base2(
        GLOBAL_LOG2_SAMPLES)])
    costs = [np.linalg.norm(residual(p)) for p in candidates]
    global_calls = calls
    local = [least_squares(residual, candidates[i], bounds=(0, 1), max_nfev=LOCAL_MAX_NFEV)
             for i in np.argsort(costs, kind="stable")[:LOCAL_STARTS]]
    best = min(local, key=lambda item: item.cost)
    values = lower + best.x * width
    predicted = model(x, best.x)
    residuals = predicted - y
    dof = int(np.count_nonzero(train)) - len(free)
    reduced_chi_squared = float(np.dot(best.fun, best.fun) / dof)
    _, singular, vectors = np.linalg.svd(best.jac, full_matrices=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    active = (best.x < 1e-5) | (best.x > 1 - 1e-5)
    identifiable = bool(best.success and condition <= 1e5 and not np.any(active))
    covariance = correlation = errors = None
    if np.isfinite(condition) and condition <= 1e5:
        weighted_vectors = (vectors.T * width[:, None]) / singular
        covariance = weighted_vectors @ weighted_vectors.T * max(reduced_chi_squared, 1.0)
        errors = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(errors, errors)
        if np.any(np.abs(correlation - np.eye(len(free))) > 0.98):
            identifiable = False
    if not identifiable:
        errors = None

    def distribution(mask):
        if not np.any(mask):
            return None
        r = residuals[mask]
        return dict(points=int(r.size), rms=float(np.sqrt(np.mean(r**2))),
                    mean=float(np.mean(r)), quantiles=np.quantile(r, [0.05, 0.5, 0.95]).tolist())

    result = dict(
        status="converged" if best.success else "optimizer_limit",
        uncertainty_scope="Conditional on the detector calibration and fixed model inputs",
        scope="fit_with_disjoint_holdout" if np.any(heldout) else "calibration_only",
        model="resolved uniform weak-probe pump-off OD", source_points=data.source_points,
        medium=params["species"], line=params["line"],
        acquisition_sampling="evenly spaced record indices, no averaging" if data.source_points > x.size else "all branch points",
        points_used=int(x.size), acquisition_branch=data.branch + 1,
        axis_method="direct evaluation on retained experimental points, no residual interpolation",
        uncertainty_source=data.uncertainty_source, frequency_mhz=x.tolist(),
        measured_transmission=y.tolist(), model_transmission=predicted.tolist(),
        residual=residuals.tolist(), sigma=sigma.tolist(), train_mask=train.tolist(), heldout_mask=heldout.tolist(),
        fit_range_mhz=fit_range_mhz, heldout_range_mhz=heldout_range_mhz,
        parameters=[dict(key=key, name=PARAMETERS[key][0], unit=PARAMETERS[key][1],
                         value=float(values[i]), bounds=limits[i].tolist(),
                         standard_error=None if errors is None else float(errors[i]),
                         on_bound=bool(active[i])) for i, key in enumerate(free)],
        fixed_line_strength=float(params.get("line_strength", 1.0)),
        fixed_parameters={key: value for key, value in base.items() if key not in free},
        covariance=None if covariance is None else covariance.tolist(),
        correlation=None if correlation is None else correlation.tolist(),
        condition_number=condition if np.isfinite(condition) else None,
        identifiable=identifiable, reduced_chi_squared=reduced_chi_squared,
        identifiability_scope="Local weighted Jacobian; cell length and density can be correlated. Line strength is fixed.",
        fit_residuals=distribution(train), heldout_residuals=distribution(heldout),
        global_evaluations=global_calls, local_evaluations=calls - global_calls,
        local_max_evaluations_per_start=LOCAL_MAX_NFEV, local_starts=LOCAL_STARTS,
    )

    if markers is not None:
        result["frequency_calibration"] = dict(markers.readout(), residual_offset_fitted="offset_mhz" in free)
    if optics is not None:
        result["optical_calibration"] = optics.readout()
    if cell is not None:
        result["cell_calibration"] = cell.readout()
    return result
