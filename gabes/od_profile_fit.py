"""AutoOD's bounded joint alignment and OD profile fit on a native measurement ROI.

The ROI fixes the data before optimizing the frequency map. Detector-affine
parameters are nuisance parameters for relative signals only. The extracted
conditions are conditional on the chosen line, model and detector assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from . import core, species
from ._arrays import readonly_float_array
from .experimental_alignment import AxisAlignment, fit_signal_affine, frame_alignment, frame_signal_range
from .experimental_csv import ExperimentalCSV
from .od_model import ODConditions, evaluate_od_display, od_unavailable, rate_model_axis
from .od_spectrum_fit import (
    BOUND_TOLERANCE, MAX_ABS_CORRELATION, MAX_CONDITION_NUMBER,
    ODFitCancelled, ODFitProgress, RATE_MODEL, offered_fit_parameters,
)
from .schemes.sas import RB_COMPOSITION_CUSTOM, _isotope_composition

_PROFILE_FIELDS = ("temp_c", "cell_mm", "probe_power_uw", "rb85_fraction")
_RAD_PER_GHZ = 2 * np.pi * 1e9


@dataclass(frozen=True)
class ODProfileFitResult:
    alignment: AxisAlignment
    condition_values: dict[str, float]
    identifiable: bool
    relative: bool
    rmse: float
    initial_rmse: float
    correlation: float
    n_points: int
    condition_number: float | None
    correlated_pairs: tuple[tuple[str, str, float], ...]
    warnings: tuple[str, ...]
    preview_frequency_ghz: np.ndarray
    preview_transmission: np.ndarray


def profile_fit_parameters(params, *, model_id=None):
    """User-approved condition ranges for a fit with a free frequency scale."""
    offered = offered_fit_parameters(params, model_id=model_id)
    return {key: offered[key] for key in _PROFILE_FIELDS if key in offered}


@lru_cache(maxsize=2)
def _rb_group_edges(line):
    """AutoOD's midpoint boundaries, in GHz from the 85Rb centroid."""
    reference = species.RB85.line(line)[1]
    groups = []
    for isotope, mass in ((species.RB85, "85"), (species.RB87, "87")):
        manifold = species.build_manifold(isotope, line)
        centers = (manifold.omega / (2 * np.pi) + (manifold.nu0 - reference)) * 1e-9
        for index, ground in enumerate(manifold.Fg):
            lines = centers[manifold.g_idx == index]
            groups.append((float(lines.min()), float(lines.max()), f"{mass}-F{int(ground)}"))
    groups.sort()
    edges = {}
    for i, (lo, hi, group) in enumerate(groups):
        left = max(lo - 0.8, (groups[i - 1][1] + lo) / 2) if i else lo - 0.8
        right = min(hi + 0.8, (hi + groups[i + 1][0]) / 2) if i + 1 < len(groups) else hi + 0.8
        edges[group] = left, right
    return edges


def profile_group_windows(params, theory_range_ghz):
    """Available transition-group windows on the active GABES display axis (GHz)."""
    lo, hi = map(float, theory_range_ghz)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("Choose an increasing finite theory frequency range in GHz.")
    windows = {"full": (lo, hi)}
    components, reference, composition, *_ = _isotope_composition(params)
    if reference not in (species.RB85, species.RB87):
        return windows
    present = {isotope for isotope, fraction in components if fraction > 0}
    shift = (reference.line(params["line"])[1] - species.RB85.line(params["line"])[1]) * 1e-9
    for group, (left, right) in _rb_group_edges(params["line"]).items():
        isotope = species.RB85 if group.startswith("85-") else species.RB87
        if isotope not in present and composition != RB_COMPOSITION_CUSTOM:
            continue
        clipped = max(left - shift, lo), min(right - shift, hi)
        if clipped[0] < clipped[1]:
            windows[group] = clipped
    return windows


def _roi_indices(trace, roi):
    lo, hi = map(float, roi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("Select a finite native interval of positive width.")
    first, last = np.searchsorted(trace.detuning, lo), np.searchsorted(trace.detuning, hi, side="right")
    if last - first < 12:
        raise ValueError("Select at least 12 distinct experimental samples.")
    return first, last


def profile_frame_alignment(trace, params, roi, *, group="full", theory_range_ghz,
                            reverse=False, current=None, invert=False, theory_x_ghz=None,
                            theory_transmission=None):
    """Frame only the selected native ROI in a group, retaining the full-trace pivot.

    Uses the existing theory arrays when supplied for a range-only relative Y
    preview. Neither the frequency span nor the signal range is a calibration.
    """
    windows = profile_group_windows(params, theory_range_ghz)
    if group not in windows:
        raise ValueError("Select a transition group present on the theory axis.")
    window = windows[group]
    first, last = _roi_indices(trace, roi)
    native = trace.detuning[first:last]
    preview = frame_alignment(native, window, reverse=reverse)
    full_pivot = 0.5 * trace.detuning[0] + 0.5 * trace.detuning[-1]
    roi_pivot = 0.5 * native[0] + 0.5 * native[-1]
    signed = -preview.scale if reverse else preview.scale
    shift = 0.5 * (window[0] + window[1]) - full_pivot - signed * (roi_pivot - full_pivot)
    preview = replace(preview, shift=shift)
    if trace.is_absolute_transmission:
        return preview
    if current is not None:
        preview = replace(preview, signal_scale=current.signal_scale, signal_offset=current.signal_offset)
    if theory_x_ghz is None and theory_transmission is None:
        return preview
    x, y = np.asarray(theory_x_ghz), np.asarray(theory_transmission)
    if x.ndim != 1 or x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Group framing needs paired finite theory frequencies and transmissions.")
    theory = y[(x >= window[0]) & (x <= window[1])]
    measured = trace.transmission[first:last]
    if invert:
        measured = 1.0 - measured
    return frame_signal_range(preview, measured, theory) if len(theory) >= 2 else preview


def _physical_limits(conditions, bounds, group):
    offered = profile_fit_parameters(conditions.to_params(), model_id=conditions.model_id)
    for key, (lo, hi) in bounds.items():
        if key not in offered:
            raise ValueError(f"{key} is not offered for this spectrum and model.")
        lower, upper = offered[key][2:]
        if not (lower <= lo <= getattr(conditions, key) <= hi <= upper and lo < hi):
            raise ValueError(f"{key} bounds must contain the starting value, within [{lower:g}, {upper:g}].")
    if conditions.rb_composition == RB_COMPOSITION_CUSTOM and conditions.species == "Rb (natural)":
        lo, hi = bounds.get("rb85_fraction", (conditions.rb85_fraction, conditions.rb85_fraction))
        if (group.startswith("85-") and hi == 0) or (group.startswith("87-") and lo == 1):
            raise ValueError("The selected isotope is absent throughout its fraction interval.")


def _solve(residual, start, sample_count):
    """Stratified coarse endpoint search; soft-L1 refinement on all samples."""
    sample = slice(None)
    if sample_count > 1400:
        edges = np.linspace(0, sample_count, 1401, dtype=int)
        sample = np.random.default_rng(0).integers(edges[:-1], edges[1:])

    def coarse_score(point):
        return np.mean(np.hypot(residual(point, sample), 0.1) - 0.1)

    best = coarse_score(start)
    physical_start = start[2:]
    for left in np.linspace(0, 1, 7):
        for right in np.linspace(0, 1, 7):
            trial = np.r_[left, right, physical_start]
            score = coarse_score(trial)
            if score < best:
                best, start = score, trial

    def jacobian(point):
        base = residual(point)
        jac = np.empty((base.size, point.size))
        for j in range(point.size):
            trial = point.copy()
            step = 2e-4 if point[j] < 0.5 else -2e-4
            trial[j] += step
            jac[:, j] = (residual(trial) - base) / step
        return jac

    with core.blas_single_thread():
        solution = least_squares(residual, start, jac=jacobian, bounds=(0, 1), loss="soft_l1",
                                 tr_solver="lsmr" if sample_count > 8192 else "exact",
                                 x_scale="jac" if sample_count > 8192 else 1.0,
                                 f_scale=0.1, max_nfev=60, xtol=2e-6, ftol=2e-6, gtol=2e-6)
    if not solution.success:
        raise ValueError("The fit did not converge. Refine the initial alignment or physical bounds.")
    return solution


def _diagnostics(fields, solution, zero_power_column, relative_linear):
    """Physical sensitivities marginalized over X endpoints, in units of the fit ranges."""
    warnings = []
    axis_limited = np.any((solution.x[:2] < 1e-3) | (solution.x[:2] > 1 - 1e-3))
    if axis_limited:
        warnings.append("The fitted frequency alignment reached its search limit.")
    # Preserve the right singular vectors without building another full-size
    # Jacobian or storing the tall left singular vectors for a scope acquisition.
    blocks = []
    for start in range(0, len(solution.jac), 8192):
        part = slice(start, start + 8192)
        block = solution.jac[part]
        if zero_power_column is not None:
            block = np.column_stack((block, zero_power_column[part]))
        blocks.append(np.linalg.qr(block, mode="r"))
    _, singular, vt = np.linalg.svd(np.vstack(blocks), full_matrices=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    scatter = max(1e-9, 1.4826 * np.median(np.abs(solution.fun - np.median(solution.fun))))
    factors = vt / np.maximum(singular, 1e-15)[:, None]
    coordinates = ("axis_left_ghz", "axis_right_ghz", *fields)
    sizes = np.linalg.norm(factors, axis=0)
    unknown = [name for name, error in zip(coordinates, scatter * sizes) if error > 1]
    columns = factors / np.maximum(sizes, 1e-30)
    correlations = np.clip(columns.T @ columns, -1, 1)
    # Exclude the descriptive correlation between the two display endpoints.
    # A condition confounded with either endpoint must remain preview-only.
    pairs = sorted(((coordinates[j], coordinates[i], float(correlations[i, j]))
                    for i in range(2, len(coordinates)) for j in range(i) if abs(correlations[i, j]) >= 0.95),
                   key=lambda pair: -abs(pair[2]))
    bound = list(solution.x[2:])
    if zero_power_column is not None:
        bound.append(0.)
    limited = [name for name, x in zip(fields, bound) if x < BOUND_TOLERANCE or x > 1 - BOUND_TOLERANCE]
    if limited:
        warnings.append("Physical condition at a fit bound: " + ", ".join(limited) + ".")
    if unknown:
        warnings.append("Unconstrained over its selected interval: " + ", ".join(unknown) + ".")
    if condition > MAX_CONDITION_NUMBER:
        warnings.append(f"The normalized fit Jacobian exceeds the condition-number limit ({MAX_CONDITION_NUMBER:,.0f}).")
    strong = any(abs(rho) > MAX_ABS_CORRELATION for _, _, rho in pairs)
    if strong:
        warnings.append(f"A physical estimate exceeds the absolute correlation limit ({MAX_ABS_CORRELATION:g}) "
                        "with another fit coordinate.")
    if relative_linear:
        warnings.append("This relative interval is linear. Detector gain and offset absorb the frequency map.")
    identifiable = not (relative_linear or (fields and axis_limited) or limited or unknown
                        or strong or condition > MAX_CONDITION_NUMBER)
    return identifiable, condition, tuple(pairs), warnings


def fit_od_profile(trace: ExperimentalCSV, params, alignment: AxisAlignment, roi, bounds=None, *,
                   group="full", model_id=None, theory_range_ghz, invert=False,
                   cancelled=None, progress=None):
    """Fit the displayed experimental trace on a fixed inclusive native X interval.

    The returned X map uses the full trace's pivot and outputs SAS display GHz.
    Free conditions use GABES units: degrees C, mm, microwatts, isotope fraction.
    Empty bounds leave every condition fixed. The absolute detector map is fixed.
    A low-quality match raises ValueError; a good shape can still have
    unidentifiable conditions, in which case the estimates are preview-only.
    """
    bounds = dict(bounds or {})
    conditions = ODConditions.from_params(params)
    if model_id is not None:
        conditions = replace(conditions, model_id=model_id)
    if reason := od_unavailable(conditions):
        raise ValueError(reason)
    if float(params.get("pump_power_mw", 0.)) != 0:
        raise ValueError("Choose the pump-off OD view for a profile fit.")
    _physical_limits(conditions, bounds, group)
    windows = profile_group_windows(conditions.to_params(), theory_range_ghz)
    if group not in windows:
        raise ValueError("Select a transition group present on the theory axis.")
    window = np.array(windows[group])
    first, last = _roi_indices(trace, roi)
    measured = trace.transmission[first:last]
    if np.ptp(measured) <= 1e-12:
        raise ValueError("The selected native interval has no measurable contrast.")
    relative = not trace.is_absolute_transmission
    if not relative and invert:
        raise ValueError("Signal inversion is available only for relative measurements.")
    if invert:
        measured = 1.0 - measured
    if not relative:
        alignment = replace(alignment, signal_scale=1.0, signal_offset=0.0)

    state = ODFitProgress("global", 0, np.inf)

    def check_cancel():
        if cancelled is not None and cancelled():
            raise ODFitCancelled(state)

    @lru_cache(maxsize=12)
    def model(trial_conditions):
        check_cancel()
        if trial_conditions.model_id == RATE_MODEL:
            axis = rate_model_axis(trial_conditions)
        else:
            axis = np.linspace(*theory_range_ghz, 2001) * _RAD_PER_GHZ
        transmission = evaluate_od_display(trial_conditions, axis).transmission
        return axis / _RAD_PER_GHZ, transmission

    theory_x, theory_y = model(conditions)
    window[0], window[1] = max(window[0], theory_x[0]), min(window[1], theory_x[-1])
    native = trace.detuning[first:last]
    mapped = alignment.transform_x(trace.detuning)[first:last]
    endpoints = np.array([min(mapped[0], mapped[-1]), max(mapped[0], mapped[-1])])
    margin = 0.23 * np.ptp(endpoints)
    lower = np.maximum(endpoints - margin, window[0])
    upper = np.minimum(endpoints + margin, window[1])
    if np.any(upper <= lower):
        raise ValueError("Align the selected native interval with the chosen transition group first.")
    axis_start = np.clip((endpoints - lower) / (upper - lower), 0, 1)
    position = (native - native[0]) / (native[-1] - native[0])
    if alignment.reverse:
        position = 1.0 - position
    initial_error = alignment.transform_y(measured) - np.interp(mapped, theory_x, theory_y)
    initial_rmse = float(np.sqrt(np.mean(initial_error**2)))
    measured_scale = max(float(np.std(measured)), 1e-12)

    def profile_residual(theory, observed):
        if relative:
            affine = fit_signal_affine(observed, theory)
            if affine is None:
                return np.full(len(observed), 30.0)
            gain, offset = affine
            return (gain * observed + offset - theory) / max(float(np.std(theory)), 1e-12)
        return (observed - theory) / measured_scale

    def fit_branch(start_conditions, fields, fixed_zero):
        lows = np.array([bounds[key][0] for key in fields])
        highs = np.array([bounds[key][1] for key in fields])
        widths = highs - lows
        physical_start = (np.array([getattr(start_conditions, key) for key in fields]) - lows) / widths

        def unpack(point):
            values = np.clip(lows + widths * point[2:], lows, highs)
            return replace(start_conditions, **dict(zip(fields, map(float, values))))

        def coordinates(point, fractions):
            left, right = lower + (upper - lower) * point[:2]
            return left + (right - left) * fractions

        def differences(trial_conditions, x, observed):
            grid, transmission = model(trial_conditions)
            difference = profile_residual(np.interp(x, grid, transmission), observed)
            outside = np.maximum(np.maximum(grid[0] - x, x - grid[-1]), 0)
            return np.copysign(np.hypot(difference, 15 * outside / np.ptp(window)), difference)

        def residual(point, sample=None):
            nonlocal state
            check_cancel()
            stage = "local" if sample is None else "global"
            sample = slice(None) if sample is None else sample
            values = differences(unpack(point), coordinates(point, position[sample]), measured[sample])
            if progress is not None or cancelled is not None:
                # Stage minima have different sample counts for long acquisitions.
                cost = np.sum(0.1 * (np.hypot(values, 0.1) - 0.1))
                best = min(state.best_cost, cost) if state.stage == stage else cost
                state = ODFitProgress(stage, state.evaluations + 1, float(best))
                if progress is not None and progress(state) is False:
                    raise ODFitCancelled(state)
            return values

        solution = _solve(residual, np.r_[axis_start, physical_start], len(measured))
        fitted = unpack(solution.x)
        grid, transmission = model(fitted)
        fitted_x = coordinates(solution.x, position)
        if fitted_x.min() < grid[0] - 1e-8 or fitted_x.max() > grid[-1] + 1e-8:
            raise ValueError("The final model does not cover the selected native interval.")
        theory = np.interp(fitted_x, grid, transmission)
        affine = fit_signal_affine(measured, theory) if relative else (1., 0.)
        if affine is None:
            raise ValueError("The selected data do not match the model reliably.")
        gain, offset = affine
        rmse = float(np.sqrt(np.mean((gain * measured + offset - theory)**2)))
        spread = float(np.std(theory))
        correlation = float(np.mean((measured - measured.mean()) * (theory - theory.mean()))
                            / max(measured_scale * spread, 1e-30))
        if spread <= 1e-10 or correlation < 0.85 or rmse > 0.40 * spread:
            raise ValueError("The selected data do not match the model reliably. Check the group and alignment.")
        power_column = None
        if fixed_zero:
            # GABES' released D2 rate model has a continuous weak limit. Test
            # the one-sided power sensitivity even if exact zero wins.
            step = 2e-4
            driven = replace(fitted, probe_power_uw=step * bounds["probe_power_uw"][1])
            shifted = differences(driven, fitted_x, measured)
            weight = np.hypot(1., solution.fun / 0.1) ** -1.5
            power_column = weight * (shifted - solution.fun) / step
            fields = (*fields, "probe_power_uw")
        # On a linear stretch, detector gain/offset absorb changes to both
        # X endpoints. Roundoff in finite differences can mask this exact
        # degeneracy even for a noiseless synthetic trace.
        chord = theory[0] + (theory[-1] - theory[0]) * ((native - native[0]) / (native[-1] - native[0]))
        relative_linear = relative and np.max(np.abs(theory - chord)) <= 64 * np.finfo(float).eps
        identifiable, condition, pairs, warnings = _diagnostics(fields, solution, power_column, relative_linear)
        if trace.diagnostics.sweep_reversal_count:
            warnings.append("The displayed trace median-merges the acquisition branches at repeated X values.")
        signed_scale = (fitted_x[-1] - fitted_x[0]) / (native[-1] - native[0])
        pivot = 0.5 * trace.detuning[0] + 0.5 * trace.detuning[-1]
        shift = (fitted_x[0] - pivot) - signed_scale * (native[0] - pivot)
        result_alignment = AxisAlignment(float(abs(signed_scale)), float(shift), alignment.reverse,
                                         float(gain), float(offset))
        result = ODProfileFitResult(
            alignment=result_alignment, condition_values={key: getattr(fitted, key) for key in bounds},
            identifiable=identifiable, relative=relative, rmse=rmse, initial_rmse=initial_rmse,
            correlation=float(np.clip(correlation, -1, 1)), n_points=len(measured),
            condition_number=condition, correlated_pairs=pairs, warnings=tuple(warnings),
            preview_frequency_ghz=readonly_float_array(grid, copy=False), preview_transmission=transmission)
        return solution.cost, result

    fields = tuple(bounds)
    if "probe_power_uw" in bounds and bounds["probe_power_uw"][0] == 0:
        positive = conditions if conditions.probe_power_uw > 0 else replace(
            conditions, probe_power_uw=0.5 * bounds["probe_power_uw"][1])
        zero_fields = tuple(key for key in fields if key != "probe_power_uw")
        candidates, errors = [], []
        for start, released, fixed_zero in ((replace(conditions, probe_power_uw=0.), zero_fields, True),
                                            (positive, fields, False)):
            try:
                candidates.append(fit_branch(start, released, fixed_zero))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                errors.append(error)
        if not candidates:
            raise errors[-1]
        _, result = min(candidates, key=lambda candidate: candidate[0])
    else:
        _, result = fit_branch(conditions, fields, False)
    check_cancel()
    return result
