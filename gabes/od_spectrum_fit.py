"""Bounded calibration of absolutely calibrated weak-probe OD spectra."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from . import species
from ._arrays import readonly_float_array
from .experimental_alignment import AxisAlignment
from .experimental_csv import ExperimentalCSV, transform_x
from .od_model import (MODELS, ODConditions, ODUnsupportedError, evaluate_od_display, is_rate_model,
                       od_unavailable)
from .schemes.sas import (
    BALLISTIC_MODELS, GAUSSIAN_TRANSIT_MODELS, GEOMETRY_MODELS, RB_COMPOSITION_CUSTOM, _od_model_setting, _probe_power_mw, _weak_probe_profile,
)
from .vapor_cell import CellTemperatureState

RATE_MODEL = "od_finite_rb_mf_rate_v1"


MAX_FIT_POINTS = 1200
GLOBAL_LOG2_SAMPLES = 6
LOCAL_STARTS = 3
LOCAL_MAX_NFEV = 40
# Identifiability gate shared by the OD fits. User decision 2026-11-23:
# keep the GABES gate (not AutoOD's 0.95/0.85 warnings) for GABES fits.
MAX_CONDITION_NUMBER = 1e5
MAX_ABS_CORRELATION = 0.98
# A normalized estimate this close to 0 or 1 counts as on its bound.
BOUND_TOLERANCE = 1e-5

PARAMETERS = {
    "temp_c": ("Cold-spot temperature", "°C", 20.0, 200.0),
    "cell_mm": ("Cell length", "mm", 0.5, 200.0),
    "extra_width_mhz": ("Extra homogeneous FWHM", "MHz", 0.0, 100.0),
    "offset_mhz": ("Frequency offset", "MHz", -150.0, 150.0),
}
# Finite-probe estimates (checklist item od-sas-autood-finite-probe-fit-extension).
# fit_od_spectrum accepts all of them; the app offers only what
# offered_fit_parameters returns for the current conditions.
CANDIDATE_PARAMETERS = {
    "probe_power_uw": ("Probe power", "µW", 0.0, 100.0),
    "probe_waist_mm": ("Probe beam radius (1/e²)", "mm", 0.1, 5.0),
    "rb85_fraction": ("⁸⁵Rb fraction", "", 0.0, 1.0),
}
_FIT_PARAMETERS = {**PARAMETERS, **CANDIDATE_PARAMETERS}
_PROBE_PARAMETERS = ("probe_power_uw", "probe_waist_mm")
# User decision 2026-11-23 ("분율+D2 파워") after the 192-fit synthetic study
# (runs/od/finite_probe_identifiability/20261123-4d92bdb): the app offers the
# ⁸⁵Rb fraction for a custom Rb composition and, on D2 under the rate model,
# the probe power at a fixed beam radius. The radius and D1 power stay script-only.
VALIDATED_SCOPE = {
    "rb85_fraction": (
        "Recovered in 64 of 64 synthetic fits (D1 and D2, 30–50 °C, 0 and 25 µW). "
        "With the finite-probe rate model the reported uncertainty was about 20 % too small."),
    "probe_power_uw": (
        "D2 only. Recovered at 5–25 µW with a fixed beam radius (30–50 °C). The fit "
        "constrains the peak intensity 2P/(πw²), so an error in the radius biases the "
        "power. The temperature uncertainty grows 1.7–2.4×."),
}


def _candidate_scope(key, params, model_id):
    if model_id in GAUSSIAN_TRANSIT_MODELS:
        return "Experimental Gaussian transit. Parameter identifiability has not been validated."
    if key == "probe_power_uw" and params.get("line") != "D2":
        return ("Not validated: on D1 the probe power and temperature were "
                "indistinguishable (|ρ| 0.987–0.9996). Scripts only.")
    if key == "probe_waist_mm":
        return ("Not offered: equivalent to the power through 2P/(πw²). Scripts only.")
    return VALIDATED_SCOPE[key]


def offered_fit_parameters(params, *, model_id=None):
    """Fit variables the app offers for these conditions.

    Returns entries shaped like ``PARAMETERS`` (name, unit, lower, upper):
    ``PARAMETERS`` itself, plus ``rb85_fraction`` for a custom Rb composition
    and ``probe_power_uw`` on D2 when the finite-probe rate model applies.
    ``VALIDATED_SCOPE`` holds the evidence for each added entry.
    """
    offered = dict(PARAMETERS)
    try:
        model = _fit_conditions(params, model_id).model_id
    except (ODUnsupportedError, ValueError, KeyError):
        model = None
    # The identifiability study used a fixed transit rate and fixed beam radius.
    if model in GEOMETRY_MODELS:
        return offered
    if (params.get("species") == "Rb (natural)"
            and params.get("rb_composition") == RB_COMPOSITION_CUSTOM):
        offered["rb85_fraction"] = CANDIDATE_PARAMETERS["rb85_fraction"]
    if params.get("line") == "D2" and is_rate_model(model):
        offered["probe_power_uw"] = CANDIDATE_PARAMETERS["probe_power_uw"]
    return offered


def probe_peak_intensity_mw_cm2(power_uw, waist_mm):
    """Gaussian peak intensity I₀ = 2P/(πw²) in mW/cm² (w is the 1/e² radius).

    With the transit rate fixed, the rate model's beam-averaged transmission
    depends on the probe only through I₀: equal I₀ gives identical spectra.
    """
    return 2.0 * (power_uw * 1e-3) / (np.pi * (waist_mm * 0.1) ** 2)


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


class ODFitCancelled(Exception):
    """Raised by ``fit_od_spectrum`` when its progress callback asks to stop."""

    def __init__(self, progress):
        super().__init__(f"OD fit cancelled after {progress.evaluations} evaluations "
                         f"in the {progress.stage} stage.")
        self.progress = progress


@dataclass(frozen=True)
class ODFitProgress:
    """State after one objective evaluation of ``fit_od_spectrum``.

    ``stage`` is "global" for the deterministic broad search and "local" for
    the bounded least-squares refinements, whose finite-difference Jacobian
    calls are counted too. ``evaluations`` is the cumulative number of
    objective evaluations. ``best_cost`` is the smallest 0.5 * sum(r**2) over
    the weighted training residuals so far, the cost least_squares minimizes.
    """

    stage: str
    evaluations: int
    best_cost: float


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
    sigma: np.ndarray | None
    branch: int
    source_points: int
    uncertainty_source: str
    frequency_calibration: ODFrequencyCalibration | None = None
    _trace_estimated_sigma: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False)


def od_fit_unavailable(params, *, model_id=None):
    """Reason the pump-off OD fit cannot run, or None.

    Without ``model_id`` the fit uses today's SAS weak-probe model and needs
    zero probe power. ``model_id`` selects any implemented ``od_model`` model;
    the magnetic-sublevel rate model accepts a finite probe.
    """
    if params.get("species") not in species.SPECIES_ORDER:
        return "Choose a resolved Rb or Cs medium."
    if model_id is None:
        try:
            requested = _od_model_setting(params)[1]
        except ValueError as error:
            return str(error)
        if requested in GEOMETRY_MODELS:
            model_id = requested
    finite = model_id is not None and MODELS.get(model_id) is not None and MODELS[model_id].finite_power
    if float(params.get("pump_power_mw", 0.0)) != 0.0 or (_probe_power_mw(params) != 0.0 and not finite):
        return "Set pump and probe power to zero for weak-probe OD fitting."
    if params.get("cell_propagation", "Uniform") != "Uniform":
        return "Choose the Uniform cell for OD fitting."
    if model_id is not None:
        try:
            return od_unavailable(_fit_conditions(params, model_id))
        except ODUnsupportedError as error:
            return str(error)
    return None


def _fit_conditions(params, model_id=None, extra_homogeneous_fwhm_hz=0.0):
    conditions = ODConditions.from_params(params, extra_homogeneous_fwhm_hz=extra_homogeneous_fwhm_hz)
    if model_id is not None:
        conditions = replace(conditions, model_id=model_id)
    return conditions


def prepare_od_fit(trace: ExperimentalCSV, alignment: AxisAlignment | None = None, *,
                   branch=0, frequency_calibration: ODFrequencyCalibration | None = None):
    """Use one raw sweep and the declared detector law.

    A supplied marker map fixes the detuning directly from raw x coordinates.
    Otherwise the legacy display alignment determines the fit axis. If the
    whole branch gives no noise estimate, the fit can still estimate noise on
    a restricted fit range.
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

    diagnostic = trace.correction_diagnostics
    detector_sigma = float(diagnostic.noise_sigma / abs(diagnostic.calibration_gain))
    sigma = _branch_noise_weights(x, y, x, detector_sigma=detector_sigma)
    if sigma is not None:
        source = "Interpolated robust noise estimates from raw branch frequency bins"
    elif detector_sigma > 0 and np.isfinite(detector_sigma):
        sigma = np.full_like(x, detector_sigma)
        source = "Raw detector noise estimate converted to transmission"
    else:
        source = "No positive noise estimate from the full raw branch"
    keep = np.linspace(0, x.size - 1, min(x.size, MAX_FIT_POINTS), dtype=int)
    retained_sigma = sigma[keep] if sigma is not None else None
    estimated = None if retained_sigma is None else (retained_sigma, readonly_float_array(retained_sigma))
    return ODFitInput(x[keep], y[keep], retained_sigma, branch, int(x.size), source,
                      frequency_calibration=frequency_calibration,
                      _trace_estimated_sigma=estimated)


def _branch_noise_weights(x, y, evaluation_x, *, detector_sigma=0.0):
    """Interpolate robust noise levels estimated from the supplied samples."""
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
    if spreads:
        floor = max(0.2 * detector_sigma, 0.2 * float(np.median(spreads)), np.finfo(float).eps)
        return np.maximum(np.interp(evaluation_x, centers, spreads), floor)
    return None


def _require_calibrated_od_scope(params):
    components = species.SPECIES[params["species"]]
    if (any(iso not in (species.RB85, species.RB87) for iso, _ in components)
            or params.get("line") not in ("D1", "D2")
            or params.get("paraffin_coated", False)
            or float(params.get("ne_pressure_torr", 0.0)) != 0.0):
        raise ValueError("The new calibration inputs cover uncoated, buffer-free Rb D1 and D2 cells.")


def od_transmission(frequency_mhz, params, *, extra_width_mhz=0.0, offset_mhz=0.0,
                    optics: ODOpticalCalibration | None = None,
                    cell: ODCellCalibration | None = None, model_id=None):
    """Evaluate the pump-off OD model at the supplied detunings in MHz.

    The same ``od_model.evaluate_od`` call draws the theory curve, so a fit
    residual and the plotted spectrum sample one function.
    """
    reason = od_fit_unavailable(params, model_id=model_id)
    if reason:
        raise ValueError(reason)
    if optics is not None or cell is not None:
        _require_calibrated_od_scope(params)
    if cell is not None:
        params = cell.apply(params)
    conditions = _fit_conditions(params, model_id, extra_width_mhz * 1e6)

    def atomic_transmission(frequency):
        # MHz on the SAS display axis (pure Rb87 from its own centroid).
        scan = (frequency - offset_mhz) * (2 * np.pi * 1e6)
        return np.array(evaluate_od_display(conditions, scan).transmission)

    frequency = np.asarray(frequency_mhz, dtype=float)
    if optics is None:
        return atomic_transmission(frequency)
    return optics.transmission(frequency, atomic_transmission)


def fit_od_spectrum(data: ODFitInput, params, *, free=None,
                    fit_range_mhz=None, heldout_range_mhz=None, bounds=None,
                    optics: ODOpticalCalibration | None = None,
                    cell: ODCellCalibration | None = None, model_id=None, progress=None):
    """Search bounded parameter space and refine weighted residuals on fit points.

    ``progress``, if given, is called with an ``ODFitProgress`` after every
    objective evaluation. Returning ``False`` cancels the fit between
    evaluations and raises ``ODFitCancelled`` (which carries the last
    progress); any other return value continues. Omitting it leaves the
    arithmetic and the result unchanged.

    Ranges apply to the calibrated input axis before fitting the offset. The
    covariance is local, conditional on fixed model and calibration inputs.
    By default the fit estimates cold-spot temperature and frequency offset,
    omitting either when fixed by the corresponding calibration. An explicit
    offset on a marker-anchored axis is interpreted as a residual offset.

    The top-level ``probe_power_uw`` is always the input power.
    ``probe_power_role`` says whether that input was a fixed model condition
    (``"fixed_input"``) or only the starting value of a fitted power
    (``"fit_starting_value"``); a fitted power is the ``parameters`` row with
    key ``probe_power_uw``.
    """
    reason = od_fit_unavailable(params, model_id=model_id)
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
    if not free or len(free) != len(set(free)) or any(key not in _FIT_PARAMETERS for key in free):
        raise ValueError("Choose distinct physical fit parameters from the supported list.")
    candidate_keys = tuple(key for key in free if key in CANDIDATE_PARAMETERS)
    if candidate_keys and _fit_conditions(params, model_id).model_id in BALLISTIC_MODELS.values():
        raise ValueError("For a beam-crossing fit, fix the isotope composition, measured beam radius and power.")
    if (set(candidate_keys) & set(_PROBE_PARAMETERS)
            and not is_rate_model(_fit_conditions(params, model_id).model_id)):
        raise ValueError("Probe power and beam radius can be fitted only with the "
                         "finite-probe magnetic-sublevel rate model.")
    if set(_PROBE_PARAMETERS) <= set(candidate_keys):
        if _fit_conditions(params, model_id).model_id in GAUSSIAN_TRANSIT_MODELS:
            raise ValueError("Hold the measured beam radius or power fixed for a Gaussian transit fit.")
        raise ValueError("The probe enters this model only through its peak intensity "
                         "2P/(πw²). Fit the power or the beam radius, with the other fixed.")
    if "rb85_fraction" in candidate_keys and (
            params.get("species") != "Rb (natural)"
            or params.get("rb_composition") != RB_COMPOSITION_CUSTOM):
        raise ValueError("Fit the ⁸⁵Rb fraction with the custom Rb composition.")
    x, y = (np.asarray(values, dtype=float) for values in (data.frequency_mhz, data.transmission))
    sigma = None if data.sigma is None else np.asarray(data.sigma, dtype=float)
    invalid_sigma = sigma is not None and (sigma.shape != x.shape or not np.isfinite(sigma).all()
                                           or np.any(sigma <= 0))
    if (x.ndim != 1 or x.size > MAX_FIT_POINTS or y.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all() or invalid_sigma):
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
    uncertainty_source = data.uncertainty_source
    # CSV-prepared uncertainties can depend on excluded samples. Re-estimate
    # from training samples alone. Check both identity and contents so even
    # a replacement with equal values counts as explicitly fixed weights.
    estimated = data._trace_estimated_sigma
    automatic = estimated is not None and data.sigma is estimated[0] and np.array_equal(sigma, estimated[1])
    if not np.all(train) and (sigma is None or automatic):
        sigma = _branch_noise_weights(x[train], y[train], x)
        if sigma is None:
            raise ValueError("Noise could not be estimated from the training range. Supply fixed uncertainties.")
        uncertainty_source = "Robust noise estimated from retained training samples only"
    if sigma is None:
        raise ValueError("Noise could not be estimated from this trace. Supply raw measurements with noise.")
    domains = dict(bounds or {})
    if domains.keys() - set(free):
        raise ValueError("Fit bounds apply only to selected physical parameters.")
    base = {"temp_c": float(params["temp_c"]), "cell_mm": float(params["cell_mm"]),
            "extra_width_mhz": 0.0, "offset_mhz": 0.0}
    current = {"probe_power_uw": _probe_power_mw(params) * 1e3,
               "probe_waist_mm": float(params.get("probe_waist_mm", 1.0)),
               "rb85_fraction": float(params.get("rb85_fraction", species.RB85.abundance))}
    base.update({key: current[key] for key in candidate_keys})
    limits = np.array([domains.get(k, _FIT_PARAMETERS[k][2:]) for k in free], dtype=float)
    registered = np.array([_FIT_PARAMETERS[k][2:] for k in free])
    if (limits.shape != (len(free), 2) or not np.isfinite(limits).all() or
            np.any(limits[:, 0] >= limits[:, 1]) or
            np.any(limits[:, 0] < registered[:, 0]) or np.any(limits[:, 1] > registered[:, 1])):
        raise ValueError("Parameter bounds must be finite, increasing and within the supported physical ranges.")
    lower, upper = limits.T
    width = upper - lower
    def model(frequency, normalized):
        values = dict(base, **dict(zip(free, lower + normalized * width)))
        point = dict(params, temp_c=values["temp_c"], cell_mm=values["cell_mm"])
        if candidate_keys:
            point.update({key: values[key] for key in candidate_keys})
            if "probe_power_uw" in candidate_keys:
                point.pop("probe_power_mw", None)   # the legacy mW key would win
        return od_transmission(frequency, point,
                               extra_width_mhz=values["extra_width_mhz"],
                               offset_mhz=values["offset_mhz"], optics=optics, model_id=model_id)

    x_train, y_train, sigma_train = x[train], y[train], sigma[train]
    calls = 0
    stage = "global"
    best_cost = np.inf

    def residual(normalized):
        nonlocal calls, best_cost
        calls += 1
        weighted = (model(x_train, normalized) - y_train) / sigma_train
        if progress is not None:
            best_cost = min(best_cost, 0.5 * float(np.dot(weighted, weighted)))
            state = ODFitProgress(stage, calls, best_cost)
            if progress(state) is False:
                raise ODFitCancelled(state)
        return weighted

    initial = np.clip((np.array([base[k] for k in free]) - lower) / width, 0.0, 1.0)
    # Deterministic broad search followed by independently bounded refinements.
    candidates = np.vstack([initial, qmc.Sobol(len(free), scramble=False).random_base2(
        GLOBAL_LOG2_SAMPLES)])
    costs = [np.linalg.norm(residual(p)) for p in candidates]
    global_calls = calls
    stage = "local"
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
    active = (best.x < BOUND_TOLERANCE) | (best.x > 1 - BOUND_TOLERANCE)
    identifiable = bool(best.success and condition <= MAX_CONDITION_NUMBER and not np.any(active))
    covariance = correlation = errors = None
    if np.isfinite(condition) and condition <= MAX_CONDITION_NUMBER:
        weighted_vectors = (vectors.T * width[:, None]) / singular
        covariance = weighted_vectors @ weighted_vectors.T * max(reduced_chi_squared, 1.0)
        errors = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(errors, errors)
        if np.any(np.abs(correlation - np.eye(len(free))) > MAX_ABS_CORRELATION):
            identifiable = False
    if not identifiable:
        errors = None

    def distribution(mask):
        if not np.any(mask):
            return None
        r = residuals[mask]
        return dict(points=int(r.size), rms=float(np.sqrt(np.mean(r**2))),
                    mean=float(np.mean(r)), quantiles=np.quantile(r, [0.05, 0.5, 0.95]).tolist())

    applied_model = _fit_conditions(params, model_id).model_id
    profile = None if applied_model in BALLISTIC_MODELS.values() else _weak_probe_profile(params)
    finite = MODELS[applied_model].finite_power
    result = dict(
        status="converged" if best.success else "optimizer_limit",
        uncertainty_scope="Conditional on the detector calibration and fixed model inputs",
        scope="fit_with_disjoint_holdout" if np.any(heldout) else "calibration_only",
        model=("resolved uniform pump-off OD, finite probe" if finite
               else "resolved uniform weak-probe pump-off OD"),
        probe_model=applied_model, probe_power_uw=_probe_power_mw(params) * 1e3,
        probe_power_role="fit_starting_value" if "probe_power_uw" in free else "fixed_input",
        source_points=data.source_points,
        medium=params["species"], line=params["line"],
        acquisition_sampling="evenly spaced record indices, no averaging" if data.source_points > x.size else "all branch points",
        points_used=int(x.size), acquisition_branch=data.branch + 1,
        axis_method="direct evaluation on retained experimental points, no residual interpolation",
        uncertainty_source=uncertainty_source, frequency_mhz=x.tolist(),
        measured_transmission=y.tolist(), model_transmission=predicted.tolist(),
        residual=residuals.tolist(), sigma=sigma.tolist(), train_mask=train.tolist(), heldout_mask=heldout.tolist(),
        fit_range_mhz=fit_range_mhz, heldout_range_mhz=heldout_range_mhz,
        parameters=[dict(key=key, name=_FIT_PARAMETERS[key][0], unit=_FIT_PARAMETERS[key][1],
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

    if candidate_keys:
        result["candidate_parameters"] = dict(
            keys=list(candidate_keys),
            scope={key: _candidate_scope(key, params, applied_model) for key in candidate_keys})
    if applied_model in GEOMETRY_MODELS:
        fitted_params = dict(params, **dict(zip(free, values)))
        result["transit"] = _fit_conditions(fitted_params, applied_model).transit_metadata()
    probe_free = [i for i, key in enumerate(free) if key in _PROBE_PARAMETERS]
    if probe_free:
        i = probe_free[0]
        fitted = dict(current, **dict(zip(free, values)))
        intensity = probe_peak_intensity_mw_cm2(fitted["probe_power_uw"], fitted["probe_waist_mm"])
        relative = None
        if errors is not None:
            # I ∝ P and I ∝ w⁻²; the other quantity is fixed.
            scale = 1.0 if free[i] == "probe_power_uw" else 2.0
            relative = scale * float(errors[i]) / float(values[i])
        result["probe_peak_intensity"] = dict(
            value_mw_cm2=float(intensity),
            standard_error_mw_cm2=None if relative is None else float(intensity * relative),
            definition="I0 = 2P/(pi w^2), w the 1/e^2 radius; the only probe quantity "
                       "this model constrains, conditional on the fixed "
                       + ("beam radius" if free[i] == "probe_power_uw" else "probe power"))
        if applied_model in GAUSSIAN_TRANSIT_MODELS:
            result["probe_peak_intensity"]["definition"] = (
                "I0 = 2P/(pi w^2), w the 1/e^2 radius; w also sets the thermal replacement rate.")
    if profile is not None:
        result["weak_probe_profile"] = profile
    if markers is not None:
        result["frequency_calibration"] = dict(markers.readout(), residual_offset_fitted="offset_mhz" in free)
    if optics is not None:
        result["optical_calibration"] = optics.readout()
    if cell is not None:
        result["cell_calibration"] = cell.readout()
    return result
