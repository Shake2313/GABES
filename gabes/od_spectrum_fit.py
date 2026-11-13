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


@dataclass(frozen=True)
class ODFitInput:
    frequency_mhz: np.ndarray
    transmission: np.ndarray
    sigma: np.ndarray
    branch: int
    source_points: int
    uncertainty_source: str


def od_fit_unavailable(params):
    if params.get("species") not in species.SPECIES_ORDER:
        return "Choose a resolved Rb or Cs medium."
    if float(params.get("pump_power_mw", 0.0)) != 0.0 or _probe_power_mw(params) != 0.0:
        return "Set pump and probe power to zero for weak-probe OD fitting."
    if params.get("cell_propagation", "Uniform") != "Uniform":
        return "Choose the Uniform cell for OD fitting."
    return None


def prepare_od_fit(trace: ExperimentalCSV, alignment: AxisAlignment, *, branch=0):
    """Use one raw sweep and the measured detector law. Alignment fixes the frequency scale."""
    if not trace.is_absolute_transmission:
        raise ValueError("OD fitting requires an absolute transmission calibration.")
    if not 0 <= branch < len(trace.branches):
        raise ValueError("Select an acquisition branch for OD fitting.")
    sweep = trace.branches[branch]
    pivot = 0.5 * float(trace.detuning.min()) + 0.5 * float(trace.detuning.max())
    x = transform_x(sweep.detuning, scale=alignment.scale, shift=alignment.shift,
                    reverse=alignment.reverse, pivot=pivot) * 1e3  # SAS plot axis is GHz.
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
    return ODFitInput(x[keep], y[keep], sigma[keep], branch, int(x.size), source)


def od_transmission(frequency_mhz, params, *, extra_width_mhz=0.0, offset_mhz=0.0):
    """Evaluate the resolved SAS pump-off model directly at each input frequency."""
    reason = od_fit_unavailable(params)
    if reason:
        raise ValueError(reason)
    scan = (np.asarray(frequency_mhz, dtype=float) - offset_mhz) * (2 * np.pi * 1e6)
    raw = SASScheme()._compute_species(
        params, scan_rad_s=scan, extra_homogeneous_fwhm_hz=extra_width_mhz * 1e6)
    alpha = raw["alpha_unit"] * float(params.get("line_strength", 1.0))
    return observables.transmission(alpha, float(params["cell_mm"]) * 1e-3)


def fit_od_spectrum(data: ODFitInput, params, *, free=("temp_c", "offset_mhz"),
                    fit_range_mhz=None, heldout_range_mhz=None, bounds=None):
    """Search bounded parameter space and refine weighted residuals on fit points.

    Ranges apply to the calibrated input axis before fitting the offset. The
    covariance is local, conditional on fixed model and calibration inputs.
    """
    reason = od_fit_unavailable(params)
    if reason:
        raise ValueError(reason)
    free = tuple(free)
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
                        extra_width_mhz=values["extra_width_mhz"], offset_mhz=values["offset_mhz"])

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

    return dict(
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
