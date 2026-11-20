"""Select a detector from a three-channel, time-domain oscilloscope export."""

import csv
from dataclasses import dataclass, field
from hashlib import sha256
import io
from itertools import chain
import operator
import re

import numpy as np

from .experimental_csv import MAX_CSV_ROWS, ExperimentalCSVError


@dataclass(frozen=True)
class ScopeImport:
    channels: tuple[str, ...]
    od: int
    sas: int | None
    trigger: int | None
    start: int
    stop: int
    total: int
    manual: bool
    source_id: str
    previews: tuple[tuple[float, ...], ...] = field(repr=False, compare=False)


class ScopeChannelChoice(ValueError):
    def __init__(self, channels, source_id, reason, previews):
        super().__init__(reason)
        self.channels = channels
        self.source_id = source_id
        self.previews = previews


_TIME = re.compile(r"(?:xaxis|time(?:stamp)?|t)(?:[munµμ]?s(?:ec)?|seconds?)?|seconds?", re.I)
_CHANNEL = re.compile(r"(?:(?:ch(?:annel)?|c|v)[ _-]*)?(?:\d+|[abcd])|sas|od|trig(?:ger)?", re.I)
_VOLTS = re.compile(r"(?:[munµμ]?v|(?:milli|micro|nano)?volts?)", re.I)


def _number(row, column):
    try:
        return float(row[column])
    except (IndexError, ValueError):
        return np.nan


def _channel_previews(signals):
    """Small display-only traces on a common acquisition-order axis."""
    sample = np.linspace(0, len(signals) - 1, min(384, len(signals)), dtype=int)
    previews = []
    for values in signals[sample].T:
        valid = np.isfinite(values)
        if not valid.any():
            previews.append(())
            continue
        low, high = np.percentile(values[valid], [1, 99])
        if low == high:
            low, high = values[valid].min(), values[valid].max()
        y = np.full(len(values), np.nan)
        y[valid] = np.clip((values[valid] - low) / (high - low), 0, 1) if high > low else 0.5
        previews.append(tuple(y))
    return tuple(previews)


def select_scope_input(header, text, payload, channel=None, source_id=None):
    """Return selected samples and provenance, or None for other CSV layouts."""
    header = list(header)
    while len(header) > 4 and not header[-1]:
        header.pop()
    if len(header) != 4:
        return None
    time_columns = [i for i, label in enumerate(header)
                    if _TIME.fullmatch(re.sub(r"[\s_()/\[\]-]", "", label))]
    if len(time_columns) != 1:
        return None
    time_col, = time_columns
    columns = [i for i in range(4) if i != time_col]
    reader = csv.reader(io.StringIO(text, newline=""))
    next(reader, None)
    first = next(reader, [])
    for i in columns:
        # Some scopes put voltage units in a second row, others in the label.
        name = re.split(r"[\[(]", header[i], maxsplit=1)[0].strip()
        unit = re.search(r"[\[(]([^\[\]()]+)[\])]$", header[i])
        if (_CHANNEL.fullmatch(name) or (unit and _VOLTS.fullmatch(unit[1].strip()))
                or (i < len(first) and _VOLTS.fullmatch(first[i].strip()))):
            continue
        return None
    names = tuple(header[i] or str(n) for n, i in enumerate(columns, 1))

    def measurements():
        for row_number, row in enumerate(chain((first,), reader), start=2):
            if row_number > MAX_CSV_ROWS:
                raise ExperimentalCSVError(f"CSV exceeds the {MAX_CSV_ROWS:,}-row import limit.")
            if np.isfinite(_number(row, time_col)):
                yield tuple(_number(row, i) for i in columns)

    signals = np.fromiter(measurements(), dtype=np.dtype((np.float64, 3)))
    if len(signals) < 2:
        raise ValueError("The scope file contains fewer than two timestamped samples.")
    digest = sha256(payload).hexdigest()
    if source_id is not None and digest != source_id:
        channel = None  # The acquisition software replaced the file while it was open.
    previews = _channel_previews(signals)
    manual = channel is not None
    if manual:
        try:
            od = names.index(channel) if isinstance(channel, str) else operator.index(channel)
        except (TypeError, ValueError):
            raise ValueError("Select one of the three scope channels.") from None
        if isinstance(channel, (bool, np.bool_)) or not 0 <= od < len(names):
            raise ValueError("Select one of the three scope channels.")
        sas = trigger = None
        selected = slice(0, len(signals))
    else:
        from ._scope_channels import identify_scope_channels

        try:
            roles = identify_scope_channels(signals)
        except ValueError as exc:
            raise ScopeChannelChoice(names, digest, str(exc), previews) from exc
        od, sas, trigger = roles.od, roles.sas, roles.trigger
        selected = roles.retained_slice
    volts = signals[selected, od]
    finite = np.isfinite(volts)
    if np.count_nonzero(finite) < 2:
        raise ValueError("The selected scope channel contains fewer than two finite samples.")
    indices = selected.start + np.flatnonzero(finite)
    converted = io.StringIO(newline="")
    writer = csv.writer(converted)
    writer.writerow(("sample", "voltage"))
    writer.writerows(zip(indices, volts[finite]))
    info = ScopeImport(names, od, sas, trigger, selected.start, selected.stop, len(signals),
                       manual, digest, previews)
    return converted.getvalue().encode("ascii"), info
