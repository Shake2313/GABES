"""OD measurement uploads, native acquisition rows and comparison downloads.

The web and desktop can retain a projected upload without keeping a second copy
of the theory grid. Only recognized comparison files get the larger byte limit.
"""
from __future__ import annotations

import codecs
import csv
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import io
from itertools import zip_longest
import json
import math
import re
import sys

import numpy as np

from . import __version__
from ._arrays import readonly_float_array
from ._csv_numeric import decode_csv_text
from ._measurement_alignment import QuickFrameResult, quick_measurement_alignment, quick_profile_alignment
from ._scope_channels import ScopeChannels, identify_scope_channels
from ._scope_measurement import ScopeChannelChoice, ScopeImport, select_scope_input
from .experimental_alignment import AxisAlignment
from .experimental_csv import (
    CALIBRATION_ABSOLUTE_DARK_REFERENCE, CALIBRATION_ABSOLUTE_GAIN_OFFSET,
    CALIBRATION_RELATIVE_EXTREMA, MAX_CSV_ROWS, MAX_FILE_BYTES, ExperimentalCSV,
    ExperimentalCSVError, _load_measured_pairs, _parse_first_two_columns,
)

MAX_COMPARISON_BYTES = 16 * 1024 * 1024
_MAX_LINE = 8192
_UNITS = {"GHz": 1., "MHz": 1.e-3, "Hz": 1.e-9, "arbitrary": 1.}
_MODES = ("relative", "fraction", "percent")
_DECLARED_TRANSMISSION = re.compile(
    r"(?:transmission|trans)(?:[\s_(\[]*(?:%|pct|percent|fraction|1)[\s)\]]*)?"
)
_INPUT_COLUMNS = ("experiment_input_x", "experiment_input_signal")
_GABES_META = "gabes_comparison_v2"
_LEGACY_GABES_META = "gabes_comparison_v1"


@dataclass(frozen=True)
class MeasurementUpload:
    """Native CSV and import choices; sha256 always identifies the entire uploaded file."""

    payload: bytes
    mode: str | None
    x_unit: str | None
    sha256: str
    metadata: dict
    scope: ScopeImport | None
    columns: tuple[str, str]


def _labels(row):
    return [value.strip().casefold().replace(" ", "_") for value in row]


def _is_comparison(names):
    if len(names) != len(set(names)) or not set(_INPUT_COLUMNS).issubset(names):
        return False
    if len(names) < 2 or names[1] != "theory_transmission":
        return False
    if names[0] not in ("detuning_ghz", "detuning_from_rb-85_d1_centroid",
                        "detuning_from_rb-85_d2_centroid"):
        return False
    return (_GABES_META in names or _LEGACY_GABES_META in names or
            {"autood_version", "gabes_commit", "experiment_input_mode", "experiment_x_unit"} <= set(names))


def _decode_row(line):
    if len(line) > _MAX_LINE:
        raise ExperimentalCSVError("Comparison CSV row is too long (8192-byte limit).")
    if not line:
        return None
    text, _ = decode_csv_text(line, ExperimentalCSVError)
    try:
        return next(csv.reader([text], strict=True))
    except csv.Error as exc:
        raise ExperimentalCSVError(f"Malformed comparison CSV: {exc}") from exc


def _compact_float(text):
    """Shortest of a plain decimal and integer-coefficient exponent, both lossless."""
    value = float(text)
    if not math.isfinite(value):
        raise ExperimentalCSVError("Invalid native comparison sample.")
    sign = "-" if math.copysign(1, value) < 0 else ""
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    if not digits:
        return sign + "0"
    coefficient = digits.rstrip("0")
    power = int(exponent or 0) - len(fraction) + len(digits) - len(coefficient)
    if power >= 0:
        decimal = coefficient + "0" * power
    else:
        point = len(coefficient) + power
        decimal = (coefficient[:point] + "." + coefficient[point:] if point > 0
                   else "." + "0" * -point + coefficient)
    return sign + min((decimal, coefficient + "e" + str(power)), key=len)


def _split_legacy_reference(metadata):
    """In a v1 GABES worksheet frequency_reference describes the plotted theory axis."""
    axis = metadata.pop("frequency_reference")
    metadata["axis_reference"] = axis
    isotope = axis.get("isotope") if isinstance(axis, dict) else axis
    if isotope not in ("Rb85", "Rb87", "Cs133"):
        return  # Preserve an older caller's axis description without guessing its origin.
    contract_isotope = "Cs133" if isotope == "Cs133" else "Rb85"
    if isinstance(axis, dict):
        line = axis.get("line", metadata.get("line"))
        metadata["contract_reference"] = {"isotope": contract_isotope, "point": "fine_centroid"}
        if line is not None:
            metadata["contract_reference"]["line"] = line
        point = axis.get("point", "fine_centroid")
    else:
        metadata["contract_reference"] = contract_isotope
        line, point = metadata.get("line"), "fine_centroid"
    if "display_reference_offset_hz" not in metadata and point == "fine_centroid":
        if isotope == contract_isotope:
            metadata["display_reference_offset_hz"] = 0.0
        elif line in ("D1", "D2"):
            from .optical_frequency import reference_offset_hz
            metadata["display_reference_offset_hz"] = reference_offset_hz(isotope, line)


def _valid_theory_references(metadata):
    line = metadata.get("line")
    isotopes = []
    for key in ("axis_reference", "contract_reference"):
        reference = metadata.get(key)
        isotope = reference
        if isinstance(reference, dict):
            isotope = reference.get("isotope")
            ref_line = reference.get("line", line)
            if (reference.get("point", "fine_centroid") != "fine_centroid"
                    or ref_line not in ("D1", "D2") or (line is not None and ref_line != line)):
                return False
            line = ref_line
        if isotope not in ("Rb85", "Rb87", "Cs133"):
            return False
        isotopes.append(isotope)
    axis, contract = isotopes
    if contract != ("Cs133" if axis == "Cs133" else "Rb85") or line not in (None, "D1", "D2"):
        return False
    offset = metadata.get("display_reference_offset_hz")
    return (type(offset) in (int, float) and abs(offset) <= sys.float_info.max
            and (axis != contract or offset == 0))


def native_frequency_reference(metadata):
    """Resolve an explicitly recorded native isotope shorthand in its source worksheet."""
    origin = metadata.get("experiment_frequency_reference")
    if isinstance(origin, str) and origin:
        # The shorthand uses the original worksheet's line, even when re-exported
        # against a theory curve on the other D line. A missing line stays unknown.
        return {"isotope": origin, "line": metadata.get("line")}
    return origin


def _validate_scope_record(scope):
    if not isinstance(scope, dict):
        raise ExperimentalCSVError("Invalid comparison scope record.")
    channels = scope.get("channels")
    if not isinstance(channels, list) or not channels or not all(isinstance(c, str) for c in channels):
        raise ExperimentalCSVError("Invalid comparison scope channels.")
    for key in ("od", "sas", "trigger"):
        if key != "od" and scope.get(key) is None:
            scope[key] = None
            continue
        index = scope.get(key)
        if type(index) is not int or not 0 <= index < len(channels):
            raise ExperimentalCSVError(f"Invalid comparison scope {key} channel.")
    samples, total = scope.get("selected_samples"), scope.get("record_samples")
    if (not isinstance(samples, list) or len(samples) != 2 or type(total) is not int
            or not all(type(i) is int for i in samples) or not 0 <= samples[0] < samples[1] <= total
            or type(scope.get("manual")) is not bool):
        raise ExperimentalCSVError("Invalid comparison scope selection.")


def _autood_acquisition_metadata(first):
    """Restore typed acquisition fields; desktop model labels stay source metadata."""
    text = first.get("autood_measurement_v1")
    if not text:
        return {}
    try:
        fields = json.loads(text)
    except ValueError as exc:
        raise ExperimentalCSVError("Invalid AutoOD acquisition metadata.") from exc
    if not isinstance(fields, dict):
        raise ExperimentalCSVError("Invalid AutoOD acquisition metadata.")
    result = {}
    for key in ("scope", "experiment_frequency_reference"):
        if key in fields:
            if not isinstance(fields[key], dict):
                raise ExperimentalCSVError(f"Invalid AutoOD {key}.")
            result[key] = fields[key]
    if "original_file_sha256" in fields:
        value = fields["original_file_sha256"]
        if not isinstance(value, str) or not re.fullmatch("[0-9a-fA-F]{64}", value):
            raise ExperimentalCSVError("Invalid AutoOD original file hash.")
        result["original_file_sha256"] = value
    if "experiment_columns" in fields:
        names = fields["experiment_columns"]
        if not isinstance(names, list) or len(names) != 2 or not all(isinstance(s, str) for s in names):
            raise ExperimentalCSVError("Invalid AutoOD acquisition columns.")
        result["experiment_columns"] = names
    if "experiment_sparse_manual_overlay" in fields:
        value = fields["experiment_sparse_manual_overlay"]
        if not isinstance(value, bool):
            raise ExperimentalCSVError("Invalid AutoOD sparse overlay setting.")
        result["experiment_sparse_manual_overlay"] = value
    return result


def _validate_transit_record(record, conditions):
    expected = conditions.transit_metadata()
    rates = record["rate_s_inverse"]
    if (record["mode"] != expected["mode"] or not isinstance(rates, dict)
            or rates.keys() != expected["rate_s_inverse"].keys()
            or not all(math.isclose(rates[k], v, rel_tol=1e-12) for k, v in expected["rate_s_inverse"].items())
            or not all(math.isclose(record[k], expected[k], rel_tol=1e-12)
                       for k in ("kinetic_temperature_k", "probe_waist_m"))):
        raise ValueError("rates or recorded geometry disagree with the conditions")
    return expected


def _validate_gabes_transit(metadata):
    from .od_model import ODConditions, od_unavailable
    from .schemes.sas import GEOMETRY_MODELS

    model = metadata.get("model_id")
    if model not in GEOMETRY_MODELS and "transit" not in metadata:
        return
    try:
        conditions = ODConditions(**metadata["conditions"])
        if conditions.model_id != model or model not in GEOMETRY_MODELS or od_unavailable(conditions):
            raise ValueError("model ID mismatch")
        _validate_transit_record(metadata["transit"], conditions)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ExperimentalCSVError(f"Invalid GABES transit metadata: {exc}") from exc


def _autood_transit_metadata(row):
    """Validate the versioned desktop geometry record against its displayed inputs."""
    from .od_model import GAUSSIAN_TRANSIT_MODELS, ODConditions
    from . import species

    text = row.get("autood_transit_v1")
    if not text:
        if row.get("model") in GAUSSIAN_TRANSIT_MODELS:
            raise ExperimentalCSVError("Missing AutoOD transit metadata for the recorded Gaussian model.")
        return {}

    try:
        record = json.loads(text)
        model = row["model"]
        if model not in GAUSSIAN_TRANSIT_MODELS or record["model_id"] != model:
            raise ValueError("model ID mismatch")
        fraction = float(row["rb85_percent"]) / 100.
        if not 0. <= fraction <= 1. or float(row["cell_mm"]) <= 0.:
            raise ValueError("invalid isotope fraction or cell length")
        conditions = ODConditions(
            species="Rb (natural)", line=row["line"], temp_c=float(row["temperature_c"]),
            cell_mm=float(row["cell_mm"]), model_id=model, rb85_fraction=fraction,
            rb_composition="Natural" if math.isclose(fraction, species.RB85.abundance, abs_tol=1e-15, rel_tol=0.) else "Custom",
            probe_waist_mm=float(row["diameter_mm"]) / 2., probe_power_uw=float(row["probe_uw"]))
        expected = _validate_transit_record(record, conditions)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ExperimentalCSVError(f"Invalid AutoOD transit metadata: {exc}") from exc
    return {"transit": {**expected, "model_id": model}, "conditions": asdict(conditions)}


def _read_comparison(read_row, header, digest):
    names = _labels(header)
    x, y = (names.index(name) for name in _INPUT_COLUMNS)
    if _GABES_META in names and _LEGACY_GABES_META in names:
        raise ExperimentalCSVError("Conflicting GABES comparison versions.")
    meta_column = next((key for key in (_GABES_META, _LEGACY_GABES_META) if key in names), None)
    if "experiment input x" in [v.strip().casefold() for v in header]:
        for _ in range(2):
            if read_row() is None:
                raise ExperimentalCSVError("Missing Origin worksheet headers.")
        separator = read_row()
        if separator is None or any(separator):
            raise ExperimentalCSVError("Missing Origin worksheet separator.")
    native = io.BytesIO()
    metadata = None
    for index in range(MAX_CSV_ROWS + 1):
        row = read_row()
        if row is None:
            break
        if index == MAX_CSV_ROWS:
            raise ExperimentalCSVError(f"CSV exceeds the {MAX_CSV_ROWS:,}-row import limit.")
        if len(row) != len(header):
            raise ExperimentalCSVError("Comparison CSV column count does not match its header.")
        if metadata is None:
            first = dict(zip(names, row))
            if meta_column is not None:
                try:
                    metadata = json.loads(first[meta_column])
                except (ValueError, TypeError) as exc:
                    raise ExperimentalCSVError("Invalid GABES comparison metadata.") from exc
                version = 1 if meta_column == _LEGACY_GABES_META else 2
                if (not isinstance(metadata, dict) or metadata.get("format_version") != version
                        or metadata.get("generated_by") != "GABES"):
                    raise ExperimentalCSVError("Unsupported GABES comparison version.")
                if not isinstance(metadata.get("model_id"), str) or not metadata["model_id"].strip():
                    raise ExperimentalCSVError("GABES comparison requires a non-blank recorded model_id.")
                if version == 1:
                    if not metadata.get("frequency_reference"):
                        raise ExperimentalCSVError("Missing v1 comparison frequency reference.")
                    _split_legacy_reference(metadata)
                elif "frequency_reference" in metadata or not _valid_theory_references(metadata):
                    raise ExperimentalCSVError("Invalid v2 comparison theory references or offset.")
                _validate_gabes_transit(metadata)
            else:
                metadata = {k: v for k, v in first.items() if k not in (*_INPUT_COLUMNS, names[0])
                            and not k.startswith(("experiment_display_", "experiment_original_",
                                                  "experiment_aligned_", "theory_"))}
                metadata.pop("autood_measurement_v1", None)
                metadata.update(_autood_acquisition_metadata(first))
                metadata.pop("autood_transit_v1", None)
                metadata.update(_autood_transit_metadata(first))
                if not metadata.get("autood_version") or not metadata.get("gabes_commit"):
                    raise ExperimentalCSVError("Missing AutoOD comparison provenance.")
            if "scope" in metadata:
                _validate_scope_record(metadata["scope"])
        if not row[x] and not row[y]:
            continue
        if not row[x] or not row[y] or max(len(row[x]), len(row[y])) > 32:
            raise ExperimentalCSVError("Invalid native sample in comparison CSV.")
        try:
            if not all(math.isfinite(float(value)) for value in (row[x], row[y])):
                raise ValueError
            native.write((row[x] + "," + row[y] + "\n").encode("ascii"))
        except (UnicodeEncodeError, ValueError) as exc:
            raise ExperimentalCSVError("Invalid native sample in comparison CSV.") from exc
    if metadata is None or not native.tell():
        raise ExperimentalCSVError("Comparison CSV has no original experimental samples.")
    if origin := native_frequency_reference(metadata):
        metadata["experiment_frequency_reference"] = origin
    mode, unit = metadata.get("experiment_input_mode"), metadata.get("experiment_x_unit")
    if mode not in _MODES or unit not in _UNITS:
        raise ExperimentalCSVError("Unknown comparison signal mode or X unit.")
    payload = native.getvalue().rstrip(b"\n")
    if len(payload) > MAX_FILE_BYTES:
        # Writing binary64 values as decimal may have expanded the original file.
        # Try the compact, round-trippable spelling before rejecting its native pair payload.
        compact = io.BytesIO()
        for pair in io.BytesIO(payload):
            x_text, y_text = pair.decode("ascii").strip().split(",")
            compact.write(f"{_compact_float(x_text)},{_compact_float(y_text)}\n".encode("ascii"))
        payload = compact.getvalue().rstrip(b"\n")
        if len(payload) > MAX_FILE_BYTES:
            raise ExperimentalCSVError("Native measurements exceed the 10 MiB import limit.")
    return MeasurementUpload(payload, mode, unit, digest.hexdigest(), metadata, None, _INPUT_COLUMNS)


def _column_indices(header):
    cells = [label.casefold() for label in header]
    x = next((i for i, label in enumerate(cells)
              if any(key in label for key in ("detuning", "frequency", "freq", "ghz"))), None)
    candidates = [(i, label) for i, label in enumerate(cells) if i != x]
    y = next((i for i, label in candidates if _DECLARED_TRANSMISSION.fullmatch(label)), None)
    if y is None:
        y = next((i for key in ("transmission", "trans", "intensity")
                  for i, label in candidates if key in label), None)
    if x is None:
        x = 1 if y == 0 else 0
    if y is None:
        y = 0 if x == 1 else 1
    if (x, y) == (0, 1) and cells[1:3] == ["transmission", "transmission"]:
        y = 2
    return x, y


def prepare_measurement_upload(data, *, scope_channel=None, scope_source_id=None,
                               max_comparison_bytes=MAX_COMPARISON_BYTES, max_scope_rows=MAX_CSV_ROWS):
    """Project native acquisition pairs, streaming recognized comparison worksheets.

    The web uses the default byte and raw scope row limits. A desktop file reader
    can allow larger comparison worksheets and limit scope rows after channel
    selection. Ordinary inputs and the projected pairs retain the CSV limits.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        stream = io.BytesIO(data)
    elif hasattr(data, "readline"):
        stream = data
    elif hasattr(data, "read"):
        stream = io.BytesIO(data.read(max_comparison_bytes + 1))
    else:
        raise TypeError("CSV input must be bytes or a binary stream.")
    first = stream.readline(_MAX_LINE + 1)
    if len(first) <= _MAX_LINE and not first.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            header = _decode_row(first) or []
        except ExperimentalCSVError:
            header = []
        if _is_comparison(_labels(header)):
            digest = sha256(first)
            consumed = len(first)

            def read_row():
                nonlocal consumed
                row = stream.readline(_MAX_LINE + 1)
                consumed += len(row)
                if consumed > max_comparison_bytes:
                    raise ExperimentalCSVError(f"Comparison CSV exceeds the {max_comparison_bytes:,}-byte limit.")
                digest.update(row)
                return _decode_row(row)

            return _read_comparison(read_row, header, digest)
    data = first + stream.read(max(0, MAX_FILE_BYTES + 1 - len(first)))
    if len(data) > MAX_FILE_BYTES:
        raise ExperimentalCSVError("Only recognized comparison exports may exceed 10 MiB.")
    digest = sha256(data).hexdigest()
    text, encoding = decode_csv_text(data, ExperimentalCSVError)
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = [v.strip() for v in next(reader, [])]
        names = _labels(header)
        if _is_comparison(names):
            # UTF-16 needs its BOM once, not at the start of every row.
            utf8 = io.BytesIO(text.encode("utf-8"))

            def read_utf8_row():
                return _decode_row(utf8.readline(_MAX_LINE + 1))

            return _read_comparison(read_utf8_row, read_utf8_row(), sha256(data))
        if (_GABES_META in names or _LEGACY_GABES_META in names or "autood_version" in names
                or names[:2] == ["detuning_ghz", "theory_transmission"]
                or (names and names[0].startswith("detuning_from_85rb_")
                    and names[1:] == ["transmission", "parameters"])):
            raise ExperimentalCSVError("Comparison CSV has no valid original measurement columns.")
        scoped = select_scope_input(header, text, data, scope_channel, scope_source_id,
                                    max_scope_rows=max_scope_rows)
        if scoped is not None:
            payload, scope = scoped
            return MeasurementUpload(payload, "relative", "arbitrary", digest, {"encoding": encoding},
                                     scope, ("sample", scope.channels[scope.od]))
        x, y = _column_indices(header)
        labels = tuple(header[i] if i < len(header) else "" for i in (x, y))
        unit = next((u for u in _UNITS if u != "arbitrary" and
                     re.search(r"(?<![a-z])" + u + r"(?![a-z])", labels[0], re.I)), None)
        signal = labels[1].casefold()
        mode = None
        if _DECLARED_TRANSMISSION.fullmatch(signal):
            mode = "percent" if any(s in signal for s in ("percent", "pct", "%")) else "fraction"
        payload = data
        if (x, y) != (0, 1) or len(header) > 2:
            out = io.StringIO(newline="")
            writer = csv.writer(out, lineterminator="\n")
            # Numeric first rows are measurements too (including headerless 3-column inputs).
            writer.writerow(labels)
            for index, row in enumerate(reader, start=2):
                if index > MAX_CSV_ROWS:
                    raise ExperimentalCSVError(f"CSV exceeds the {MAX_CSV_ROWS:,}-row import limit.")
                writer.writerow([row[i] if i < len(row) else "" for i in (x, y)])
            payload = out.getvalue().encode("utf-8")
    except csv.Error as exc:
        raise ExperimentalCSVError(f"Malformed measurement CSV: {exc}") from exc
    return MeasurementUpload(payload, mode, unit, digest, {"encoding": encoding}, None, labels)


def load_measurement(upload, *, mode=None, x_unit=None, denoise=True, calibration_mode=None,
                     dark_signal=None, reference_signal=None, gain=None, offset=None, sparse_manual_overlay=None):
    """Use explicit UI choices or declared input units. Unknown signals are relative."""
    mode = mode or upload.mode or "relative"
    unit = x_unit or upload.x_unit or "arbitrary"
    if mode not in _MODES or unit not in _UNITS:
        raise ExperimentalCSVError("Unknown measurement signal mode or X unit.")
    if calibration_mode is None:
        if "experiment_calibration_mode" in upload.metadata and mode == upload.mode:
            calibration_mode = upload.metadata["experiment_calibration_mode"]
            saved_gain = upload.metadata.get("experiment_calibration_gain")
            saved_offset = upload.metadata.get("experiment_calibration_offset")
            if calibration_mode in (CALIBRATION_ABSOLUTE_DARK_REFERENCE, CALIBRATION_ABSOLUTE_GAIN_OFFSET):
                try:
                    saved_gain, saved_offset = float(saved_gain), float(saved_offset)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ExperimentalCSVError("Invalid saved detector calibration.") from exc
            if calibration_mode == CALIBRATION_ABSOLUTE_GAIN_OFFSET:
                gain, offset = saved_gain, saved_offset
            elif calibration_mode == CALIBRATION_ABSOLUTE_DARK_REFERENCE:
                dark_signal, reference_signal = saved_offset, saved_offset + saved_gain
        elif mode == "relative":
            calibration_mode = CALIBRATION_RELATIVE_EXTREMA
        else:
            calibration_mode = CALIBRATION_ABSOLUTE_GAIN_OFFSET
            gain, offset = (100. if mode == "percent" else 1.), 0.
    if sparse_manual_overlay is None:
        sparse_manual_overlay = upload.metadata.get("experiment_sparse_manual_overlay") is True
    trace = _load_measured_pairs(
        upload.payload, denoise=denoise, calibration_mode=calibration_mode, min_unique_points=2,
        dark_signal=dark_signal, reference_signal=reference_signal, gain=gain, offset=offset,
        sparse_manual_overlay=sparse_manual_overlay,
    )
    factor = _UNITS[unit]
    if factor != 1.:
        axis = readonly_float_array(trace.detuning * factor, copy=False)
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ExperimentalCSVError("Frequency samples become indistinguishable in GHz.")
        d = trace.diagnostics
        trace = replace(trace, detuning=axis, sweep_branches=tuple(
            replace(b, detuning=readonly_float_array(b.detuning * factor, copy=False)) for b in trace.branches
        ), import_diagnostics=replace(d, typical_x_spacing=d.typical_x_spacing * factor,
                                     largest_x_gap=d.largest_x_gap * factor, gap_threshold=d.gap_threshold * factor))
    return trace


def gap_slices(trace):
    """Contiguous display/measurement slices; never interpolate across the open gaps."""
    if trace.diagnostics.segment_count <= 1:
        return (slice(0, len(trace.detuning)),)
    boundaries = np.flatnonzero(np.diff(trace.detuning) > trace.diagnostics.gap_threshold) + 1
    return tuple(slice(int(a), int(b)) for a, b in
                 zip(np.r_[0, boundaries], np.r_[boundaries, len(trace.detuning)]))


def export_comparison_csv(upload, trace, theory_x_ghz, theory_transmission, *,
                          alignment=None, invert=False, metadata, origin=False,
                          mode=None, x_unit=None):
    """Export theory/display and original pairs independently, preserving acquisition order.

    axis_reference identifies detuning_GHz, contract_reference the evaluator's origin.
    Add display_reference_offset_hz to the displayed detuning in Hz to get contract Hz.
    experiment_frequency_reference independently identifies an established native origin.
    """
    meta = dict(metadata)
    if "axis_reference" not in meta and meta.get("frequency_reference"):
        _split_legacy_reference(meta)
    meta.pop("frequency_reference", None)
    if not isinstance(meta.get("model_id"), str) or not meta["model_id"].strip():
        raise ValueError("Comparison export requires a non-blank recorded model_id.")
    _validate_gabes_transit(meta)
    if not _valid_theory_references(meta):
        raise ValueError("Comparison export requires both theory references and a finite offset.")
    tx, ty = np.asarray(theory_x_ghz, dtype=float), np.asarray(theory_transmission, dtype=float)
    if tx.ndim != 1 or tx.size == 0 or tx.shape != ty.shape or not np.all(np.isfinite(tx)) or not np.all(np.isfinite(ty)):
        raise ValueError("Theory must contain paired finite detunings and transmissions.")
    mode, unit = mode or upload.mode or "relative", x_unit or upload.x_unit or "arbitrary"
    if mode not in _MODES or unit not in _UNITS:
        raise ValueError("Unknown measurement signal mode or X unit.")
    alignment = alignment or AxisAlignment()
    aligned_x = alignment.transform_x(trace.detuning)
    aligned_y = alignment.transform_y(trace.transmission, invert=invert)
    signal = "transmission" if trace.is_absolute_transmission else "relative_signal"
    grid = np.full(tx.shape, np.nan)
    for part in gap_slices(trace):
        x, y = aligned_x[part], aligned_y[part]
        if alignment.reverse:
            x, y = x[::-1], y[::-1]
        if np.any(np.diff(x) <= 0):
            raise ValueError("Frequency samples become indistinguishable after alignment.")
        inside = (x[0] <= tx) & (tx <= x[-1])
        grid[inside] = np.interp(tx[inside], x, y)

    correction = trace.correction
    meta.update(
        format_version=2, generated_by="GABES", gabes_version=__version__,
        experiment_x_unit=unit, experiment_input_mode=mode,
        input_file_sha256=upload.sha256,
        original_file_sha256=upload.metadata.get("original_file_sha256", upload.sha256),
        experiment_columns=upload.metadata.get("experiment_columns", upload.columns),
        experiment_denoised=correction.denoise_enabled, experiment_inverted=bool(invert),
        experiment_calibration_mode=correction.calibration_mode,
        experiment_calibration_gain=correction.calibration_gain,
        experiment_calibration_offset=correction.calibration_offset,
        experiment_display_scaling="display_only", experiment_alignment=asdict(alignment),
    )
    if "experiment_sparse_manual_overlay" in upload.metadata:
        meta["experiment_sparse_manual_overlay"] = upload.metadata["experiment_sparse_manual_overlay"]
    if "scope" in upload.metadata:
        meta["scope"] = upload.metadata["scope"]
    if "autood_version" in upload.metadata:
        meta["measurement_source"] = {key: value for key, value in upload.metadata.items()
                                      if key in ("autood_version", "gabes_commit", "model", "frequency_reference")
                                      or (key in ("transit", "conditions") and isinstance(upload.metadata.get("transit"), dict))
                                      or key.startswith("scope_")}
    elif "measurement_source" in upload.metadata:
        meta["measurement_source"] = upload.metadata["measurement_source"]
    if upload.scope is not None:
        scope = upload.scope
        meta["scope"] = dict(channels=scope.channels, od=scope.od, sas=scope.sas,
                             trigger=scope.trigger, selected_samples=[scope.start, scope.stop],
                             record_samples=scope.total, manual=scope.manual, source_id=scope.source_id)
    native_key = "experiment_frequency_reference"
    # A retained pre-upgrade upload can still hold the old shorthand. Resolve
    # inherited metadata against that upload before changing the theory line.
    source = (upload.metadata if native_key not in meta or meta[native_key] == upload.metadata.get(native_key)
              else meta)
    if origin_reference := native_frequency_reference(source):
        meta[native_key] = origin_reference
    encoded_meta = json.dumps(meta, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    columns = ["detuning_GHz", "theory_transmission", f"experiment_display_{signal}_on_theory_grid",
               *_INPUT_COLUMNS, f"experiment_calibrated_{signal}", "experiment_aligned_detuning_GHz",
               f"experiment_display_{signal}", _GABES_META]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    if origin:
        writer.writerow([c.replace("_", " ") if c != _GABES_META else c for c in columns])
        writer.writerow(["GHz", "1", "1", unit, "%" if mode == "percent" else "1" if mode == "fraction" else "",
                         "1", "GHz", "1", ""])
        writer.writerow(["Theory detuning", "Theory transmission", "Only within measured coverage",
                         "Native X in acquisition order", "Native detector signal", "Before display scaling",
                         "After display alignment", "After display scaling", "Source and model metadata"])
        writer.writerow([])
    else:
        writer.writerow(columns)

    # The GHz conversion can round. Take the original binary64 coordinates from
    # the uploaded pair payload directly, not by dividing the display axis.
    native_text, _ = decode_csv_text(upload.payload, ExperimentalCSVError)
    native_x, _, _ = _parse_first_two_columns(native_text)

    def measurements():
        for branch in trace.branches:
            indices = np.searchsorted(trace.detuning, branch.detuning)
            yield from zip(native_x[branch.start_index:branch.stop_index], branch.raw_signal,
                           branch.transmission, aligned_x[indices], aligned_y[indices], strict=True)

    theory = ((x, t, "" if np.isnan(y) else y) for x, t, y in zip(tx, ty, grid, strict=True))
    # Limit the metadata row as well so every emitted comparison can be reopened.
    metadata_size = len(encoded_meta.encode("utf-8")) + encoded_meta.count('"')
    if metadata_size > _MAX_LINE - 512:
        raise ExperimentalCSVError("Comparison metadata exceeds the row limit.")
    for index, (curve, sample) in enumerate(zip_longest(theory, measurements())):
        writer.writerow([*(curve or ("",) * 3), *(sample or ("",) * 5), encoded_meta if index == 0 else ""])
    data = output.getvalue().encode("utf-8-sig")
    if len(data) > MAX_COMPARISON_BYTES:
        raise ExperimentalCSVError("Comparison export exceeds the supported upload size.")
    return data
