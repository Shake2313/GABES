"""Helpers shared by the CSV loaders: text decoding and duplicate merging."""

import numpy as np


def sort_and_median_duplicates(x, y):
    """Sort by x and replace duplicate-y groups with their median."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be equal one-dimensional arrays")
    if x.size == 0:
        return x.copy(), y.copy()

    order = np.lexsort((y, x))
    sorted_x = x[order]
    sorted_y = y[order]
    unique_x, starts, counts = np.unique(
        sorted_x, return_index=True, return_counts=True
    )
    middle = starts + counts // 2
    medians = sorted_y[middle].copy()
    even = counts % 2 == 0
    medians[even] = 0.5 * (
        sorted_y[middle[even] - 1] + sorted_y[middle[even]]
    )
    return unique_x, medians


def decode_csv_text(data, error=ValueError):
    """Decode CSV bytes; return ``(text, encoding)``.

    BOM-marked UTF-16 first (malformed text raises ``error``, the loader's
    exception type), then UTF-8 with or without a BOM, then CP949. Numeric
    fields and delimiters are ASCII, so the final Latin-1 step, which maps every
    byte, lets an unsupported byte in metadata or ignored columns through
    instead of rejecting otherwise valid data. UTF-8 text without a BOM is
    reported as ``utf-8-sig``, the first codec that accepts it.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError as exc:
            raise error("Malformed UTF-16 CSV text.") from exc
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


def sweep_structure(
    x: np.ndarray,
) -> tuple[tuple[slice, ...], np.ndarray, int]:
    """Classify acquisition-order samples before any sorting or merging.

    Each valid source sample belongs to exactly one monotonic branch. A turning
    point reached by a nonzero step stays with the incoming branch; when the
    instrument repeats the turning-point x value, the extra equal-x sample is
    assigned to the outgoing branch. Equal steps elsewhere inherit the nearest
    movement direction and never create a reversal by themselves.
    """

    values = np.asarray(x, dtype=float)
    n_samples = int(values.size)
    if n_samples == 0:
        return (), np.empty(0, dtype=np.int8), 0
    if n_samples == 1:
        return (slice(0, 1),), np.zeros(1, dtype=np.int8), 0

    edge_direction = np.sign(np.diff(values)).astype(np.int8)
    moving_edges = edge_direction[edge_direction != 0]
    if moving_edges.size == 0:
        labels = np.zeros(n_samples, dtype=np.int8)
        return (slice(0, n_samples),), labels, 0

    reversal_count = int(np.count_nonzero(moving_edges[1:] != moving_edges[:-1]))
    next_movement = np.zeros(n_samples, dtype=np.int8)
    next_seen = 0
    for edge_index in range(edge_direction.size - 1, -1, -1):
        if edge_direction[edge_index] != 0:
            next_seen = int(edge_direction[edge_index])
        next_movement[edge_index] = next_seen
    labels = np.empty(n_samples, dtype=np.int8)
    labels[0] = moving_edges[0]
    previous = int(labels[0])
    for sample_index in range(1, n_samples):
        incoming = int(edge_direction[sample_index - 1])
        if incoming:
            labels[sample_index] = incoming
            previous = incoming
            continue

        # A duplicate at a turn belongs to the outgoing branch; an ordinary
        # duplicate on a monotonic sweep stays with that same direction.
        outgoing = int(next_movement[sample_index]) or previous
        labels[sample_index] = outgoing if outgoing != previous else previous
        previous = int(labels[sample_index])

    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [n_samples]))
    branches = tuple(
        slice(int(start), int(stop)) for start, stop in zip(starts, stops)
    )
    return branches, labels, reversal_count
