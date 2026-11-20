"""Assign three oscilloscope channels from their acquisition-order waveforms."""
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


def _retained_slice(start, stop):
    # Exclude the scan reversal and settling at each end of a complete gate.
    trim = ceil(0.15 * (stop - start))
    return slice(start + trim, stop - trim)


@dataclass(frozen=True)
class ScopeChannels:
    trigger: int
    sas: int
    od: int
    start: int
    stop: int

    @property
    def retained_slice(self):
        return _retained_slice(self.start, self.stop)


def _complete_gates(signal):
    """Find complete plateaus only on a waveform concentrated around two rails."""
    valid = np.isfinite(signal)
    if np.count_nonzero(valid) < 0.8 * signal.size:
        return []
    values = signal[valid]
    low, high = np.percentile(values, [1, 99])
    if not high > low:
        return []
    middle = low + (high - low) / 2
    lower = values[values < middle]
    upper = values[values >= middle]
    if min(len(lower), len(upper)) < max(3, 0.008 * len(values)):
        return []
    low, high = np.median(lower), np.median(upper)
    distance = np.minimum(abs(values - low), abs(values - high))
    if not high > low or np.mean(distance < 0.10 * (high - low)) < 0.94:
        return []
    filled = signal if valid.all() else np.interp(np.arange(signal.size), np.flatnonzero(valid), values)
    state = median_filter((filled > low + (high - low) / 2).astype(np.uint8), size=3, mode="nearest")
    edges = np.flatnonzero(state[1:] != state[:-1]) + 1
    starts, stops = edges[:-1], edges[1:]
    # A missing sample right at a transition cannot establish the edge's location.
    good = (stops - starts >= 100) & valid[starts] & valid[starts - 1] & valid[stops] & valid[stops - 1]
    gates = []
    for level in (0, 1):
        choices = np.flatnonzero(good & (state[starts] == level))
        # Inspect either polarity, with bounded work even on long pulse trains.
        longest = np.argsort(-(stops - starts)[choices], kind="stable")[:3]
        gates.extend((int(starts[i]), int(stops[i])) for i in choices[longest])
    return gates


def _optical_features(raw):
    """Measure broad optical contrast and repeated narrow peaks of either sign."""
    valid = np.isfinite(raw)
    if np.count_nonzero(valid) < 0.8 * raw.size:
        return False, 0.0
    if not valid.all():
        raw = np.interp(np.arange(raw.size), np.flatnonzero(valid), raw[valid])
    # Binning is only for identifying channels. The imported OD channel uses native samples.
    step = ceil(len(raw) / 4096)
    if step > 1:
        raw = np.median(raw[:len(raw) // step * step].reshape(-1, step), axis=1)
    n = len(raw)
    sigma = max(0.7, n / 1500)
    fine = gaussian_filter1d(raw, sigma, mode="nearest")
    broad = gaussian_filter1d(fine, max(3., n / 35), mode="nearest")
    span = np.ptp(np.percentile(fine, [0.5, 99.5]))
    if not np.isfinite(span) or span <= 0:
        return False, 0.0
    delta = np.diff(raw)
    # MAD of first differences estimates point noise without assuming an optical polarity.
    noise = np.median(abs(delta - np.median(delta))) / 0.954
    has_broad_signal = np.ptp(np.percentile(broad, [2, 98])) > max(5 * noise, 0.18 * span)
    detail = (fine - broad) / span
    threshold = max(0.025, 4 * noise / (span * np.sqrt(sigma)))
    score = 0.
    for polarity in (1, -1):
        peaks, properties = find_peaks(
            polarity * detail, prominence=threshold, width=(2, max(2, n * 0.028)),
            distance=max(2, round(n * 0.004)))
        interior = (peaks > 0.03 * n) & (peaks < 0.97 * n)
        strengths = properties["prominences"][interior]
        if len(strengths) >= 2:
            # One switching spike is insufficient evidence for a SAS reference.
            score = max(score, float(np.partition(strengths, -2)[-2]))
    return has_broad_signal, score


def identify_scope_channels(signals):
    """Return zero-based column roles and a bounded sweep for three voltage channels.

    Preserve acquisition row positions (including missing values). Raises ValueError
    when the waveforms do not distinguish a trigger, a SAS reference and an OD signal.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2 or signals.shape[1] != 3 or len(signals) < 100:
        raise ValueError("Expected three voltage channels and a complete trigger sweep.")
    gates = [_complete_gates(channel) for channel in signals.T]
    candidates = [i for i, intervals in enumerate(gates) if intervals]
    if len(candidates) != 1:
        raise ValueError("Could not identify a unique two-level trigger with a complete sweep.")
    trigger, = candidates
    first, second = [i for i in range(3) if i != trigger]
    assignments = []
    for start, stop in gates[trigger]:
        retained = _retained_slice(start, stop)
        first_broad, first_peaks = _optical_features(signals[retained, first])
        second_broad, second_peaks = _optical_features(signals[retained, second])
        if second_broad and first_peaks > 2 * second_peaks:
            assignments.append(ScopeChannels(trigger, first, second, start, stop))
        elif first_broad and second_peaks > 2 * first_peaks:
            assignments.append(ScopeChannels(trigger, second, first, start, stop))
    if len({roles.od for roles in assignments}) != 1:
        raise ValueError("The optical waveforms do not distinguish the SAS and OD channels reliably.")
    return max(assignments, key=lambda roles: (roles.stop - roles.start, -roles.start))
