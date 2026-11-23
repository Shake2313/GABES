"""Fast display alignment from edges in an already calculated spectrum."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from .experimental_alignment import AxisAlignment, frame_alignment, frame_signal_range


@dataclass(frozen=True)
class QuickFrameResult:
    alignment: AxisAlignment
    matched: bool
    candidates: int


@dataclass(frozen=True)
class _Edge:
    center: float
    width: float
    direction: int
    amplitude: float


def _edges(x, signal):
    """Find 10/50/90% crossings on substantial rising and falling sections."""
    low, high = np.percentile(signal, (1, 99))
    contrast = high - low
    noise = np.median(np.abs(np.diff(signal) - np.median(np.diff(signal))))
    if contrast <= max(1e-12, 4 * noise):
        return []
    smooth = gaussian_filter1d(median_filter(signal, size=3, mode="nearest"), 1.25, mode="nearest")
    prominence = max(0.065 * contrast, 3 * noise)
    peaks = find_peaks(smooth, prominence=prominence)[0]
    valleys = find_peaks(-smooth, prominence=prominence)[0]
    knots = np.sort(np.r_[0, peaks, valleys, len(x) - 1])
    edges = []
    for first, last in zip(knots[:-1], knots[1:]):
        change = smooth[last] - smooth[first]
        if last - first < 3 or abs(change) < prominence:
            continue
        if len(knots) == 2:
            # A featureless ramp supplies no position or width landmark.
            straight = np.linspace(smooth[first], smooth[last], last - first + 1)
            if np.max(np.abs(smooth[first:last + 1] - straight)) < 0.07 * abs(change):
                continue
        fraction = (smooth[first:last + 1] - smooth[first]) / change
        left, center, right = np.interp((0.1, 0.5, 0.9), np.maximum.accumulate(fraction),
                                        x[first:last + 1])
        if right > left:
            edges.append(_Edge(center, right - left, 1 if change > 0 else -1, abs(change)))
    # Bound the pair search even for a noisy recording with many small ripples.
    strongest = sorted(edges, key=lambda edge: edge.amplitude, reverse=True)[:8]
    return sorted(strongest, key=lambda edge: edge.center)


def _edge_seeds(measured, theory):
    pairs = [(m, t) for m in measured for t in theory if m.direction == t.direction]
    for m, t in pairs:
        width = t.width / m.width
        yield t.center - width * m.center, width
    # Spacings between two landmarks can survive a change in line broadening.
    for (m1, t1), (m2, t2) in combinations(pairs, 2):
        if m2.center > m1.center and t2.center > t1.center:
            width = (t2.center - t1.center) / (m2.center - m1.center)
            yield t1.center - width * m1.center, width


def _evaluate(centers, widths, u, signal, tx, theory, relative):
    """Score each candidate on the measured trace and penalize incomplete coverage."""
    x = centers[:, None] + widths[:, None] * u
    valid = (x >= 0) & (x <= 1)
    target = np.interp(x, tx, theory)
    count = np.maximum(1, valid.sum(axis=1))
    coverage = valid.mean(axis=1)
    mean_y = (signal * valid).sum(axis=1) / count
    mean_t = (target * valid).sum(axis=1) / count
    dy = (signal - mean_y[:, None]) * valid
    dt = (target - mean_t[:, None]) * valid
    var_y = (dy * dy).sum(axis=1)
    var_t = (dt * dt).sum(axis=1)
    covariance = (dy * dt).sum(axis=1)
    correlation = covariance / np.sqrt(np.maximum(var_y * var_t, 1e-30))
    if relative:
        gain = covariance / np.maximum(var_y, 1e-30)
        offset = mean_t - gain * mean_y
        # Normalization by local contrast prevents a tiny segment of a flat
        # theoretical wing from winning simply because its amplitude is small.
        error = np.sqrt(np.maximum(0, 1 - np.minimum(correlation, 1) ** 2))
    else:
        gain = np.ones(len(x))
        offset = np.zeros(len(x))
        residual = (signal - target) * valid
        error = np.sqrt((residual * residual).sum(axis=1) / np.maximum(var_t, 1e-30))
    invalid = (coverage < 0.88) | (gain <= 0) | (correlation < 0.65)
    score = error + (1 - coverage) * 1.2
    return np.where(invalid, np.inf, score), gain, offset


def _different_regions(centers, widths, scores, landmark):
    """One best placement per distinct location of the strongest measured edge."""
    selected = []
    locations = centers + landmark * widths
    for i in np.argsort(scores, kind="stable"):
        if not np.isfinite(scores[i]):
            break
        if all(abs(locations[i] - locations[j]) > max(0.006, 0.08 * min(widths[i], widths[j]))
               for j in selected):
            selected.append(i)
            if len(selected) == 8:
                break
    return selected


def quick_measurement_alignment(trace, theory_x_ghz, theory_transmission, *, current=None,
                                reverse=False, invert=False, fallback=None, match_signal=True):
    """Return a provisional affine display transform; require evidence in both traces."""
    mx, my = np.asarray(trace.x), np.asarray(trace.transmission)
    tx, ty = np.asarray(theory_x_ghz), np.asarray(theory_transmission)
    relative = not trace.is_absolute_transmission
    if invert and relative:
        my = 1.0 - my
    if fallback is None:
        fallback = frame_alignment(mx, tx, reverse=reverse)
        if match_signal and relative:
            fallback = frame_signal_range(fallback, my, ty, center_flat=True)
    miss = QuickFrameResult(fallback, False, 0)
    if min(len(mx), len(tx)) < 12:
        return miss
    raw_span = float(mx[-1] - mx[0])
    theory_span = float(tx[-1] - tx[0])
    raw_mid = float(0.5 * mx[0] + 0.5 * mx[-1])
    if raw_span <= 0 or theory_span <= 0:
        return miss

    # Both axes are dimensionless for the search; positive widths cover the
    # selected direction. Only the returned transform uses the original units.
    u = np.linspace(-0.5, 0.5, min(len(mx), 385))
    grid = mx[0] + (u + 0.5) * raw_span
    uniform_signal = np.interp(grid, mx, my)
    # Use real observations for scoring. Uniform interpolation across a gap
    # is allowed only while locating potential edges, and is checked below.
    after = np.clip(np.searchsorted(mx, grid), 1, len(mx) - 1)
    nearest = np.where(grid - mx[after - 1] < mx[after] - grid, after - 1, after)
    take = np.unique(nearest)
    observed = (mx[take] - mx[0]) / raw_span - 0.5
    measured = my[take]
    if fallback.reverse:
        uniform_signal = uniform_signal[::-1]
        observed, measured = -observed[::-1], measured[::-1]
    theory_x = np.linspace(0, 1, min(len(tx), 1025))
    theory = np.interp(tx[0] + theory_x * theory_span, tx, ty)
    m_edges = []
    for edge in _edges(u, uniform_signal):
        # The 50% crossing must be bracketed by actual samples inside the edge.
        left, center, right = observed.searchsorted(
            [edge.center - 0.5 * edge.width, edge.center, edge.center + 0.5 * edge.width])
        if left < center < right:
            m_edges.append(edge)
    t_edges = _edges(theory_x, theory)
    if not m_edges or not t_edges:
        return miss
    seeds = list(_edge_seeds(m_edges, t_edges))
    if not seeds:
        return miss
    current = current or fallback
    preferred_center = ((raw_mid - tx[0]) + current.shift) / theory_span
    preferred_width = current.scale * raw_span / theory_span
    if not (np.isfinite(preferred_center) and np.isfinite(preferred_width) and preferred_width > 0):
        preferred_center, preferred_width = 0.5, 0.9
    seeds.append((preferred_center, preferred_width))
    centers, widths = np.asarray(seeds).T
    keep = (widths >= 6 / (len(theory) - 1)) & (widths <= 1.16) & (centers >= 0) & (centers <= 1)
    centers, widths = centers[keep], widths[keep]
    if not len(centers):
        return miss

    measured = median_filter(measured, size=3, mode="nearest")
    fit_y = relative and match_signal
    # A calibrated transmission can distinguish a weak dip from a deep dip.
    # Use that evidence even when the display button also requests a Y frame.

    def evaluate(c, w):
        return _evaluate(c, w, observed, measured, theory_x, theory, relative)

    score, _, _ = evaluate(centers, widths)
    landmark = max(m_edges, key=lambda edge: edge.amplitude).center
    regions = _different_regions(centers, widths, score, landmark)
    if not regions:
        return miss
    centers, widths = centers[regions], widths[regions]
    # Small bounded stencils around each region improve the edge-based scale
    # and center estimates without evaluating another atomic spectrum.
    steps = np.array([-1., 0., 1.])
    for step in (0.12, 0.06, 0.03, 0.015, 0.0075, 0.00375):
        trial_width = widths[:, None, None] * (1 + steps[None, :, None] * step)
        trial_center = (centers[:, None, None] + (widths[:, None, None] - trial_width) * landmark
                        + widths[:, None, None] * steps[None, None, :] * step * 0.35)
        trial_width = np.broadcast_to(trial_width, trial_center.shape)
        score, _, _ = evaluate(trial_center.ravel(), trial_width.ravel())
        score = score.reshape(len(regions), -1)
        best = np.argmin(score, axis=1)
        centers = trial_center.reshape(score.shape)[np.arange(len(regions)), best]
        widths = trial_width.reshape(score.shape)[np.arange(len(regions)), best]

    # The regular search grid can sample a periodic acquisition at a beat
    # frequency that resembles an absorption line. Validate the finalists on
    # raw readings. For long traces, draw one independent point per stratum
    # so even a dense periodic signal contributes observations at mixed phases.
    if len(mx) > 8192:
        boundaries = np.linspace(0, len(mx), 8193, dtype=np.intp)
        verification = np.random.default_rng(742).integers(boundaries[:-1], boundaries[1:])
    else:
        verification = slice(None)
    positions = (mx[verification] - mx[0]) / raw_span - .5
    if fallback.reverse:
        positions = -positions
    score, gain, offset = _evaluate(centers, widths, positions, my[verification],
                                    (tx - tx[0]) / theory_span, ty, relative)
    regions = _different_regions(centers, widths, score, landmark)
    if not regions or score[regions[0]] > 0.48:
        return miss
    threshold = min(0.48, score[regions[0]] + max(0.018, 0.12 * score[regions[0]]))
    tied = [i for i in regions if score[i] <= threshold]
    best = min(tied, key=lambda i: (abs(centers[i] - preferred_center)
                                    + 0.12 * abs(math.log(widths[i]) - math.log(preferred_width))))
    alignment = replace(fallback, scale=float(widths[best] * theory_span / raw_span),
                        shift=float(tx[0] + centers[best] * theory_span - raw_mid),
                        signal_scale=float(gain[best]) if fit_y else fallback.signal_scale,
                        signal_offset=float(offset[best]) if fit_y else fallback.signal_offset)
    return QuickFrameResult(alignment, True, len(tied))
