"""Draft-only controls for the OD precision editor's measured curve."""

from dataclasses import replace
import math

import streamlit as st

from gabes.experimental_csv import (
    CALIBRATION_ABSOLUTE_DARK_REFERENCE, CALIBRATION_ABSOLUTE_GAIN_OFFSET,
    CALIBRATION_RELATIVE_EXTREMA,
)

# Preserve the numeric input IDs used by keyboard entry in the previous editor.
_FIELDS = (
    ("scale", "X scale", "sas___csv_x_scale"),
    ("shift", "X shift (GHz)", "sas___csv_x_shift"),
    ("signal_scale", "Y scale", "sas___csv_y_scale"),
    ("signal_offset", "Y shift", "sas___csv_y_offset"),
)
_DRAFT = "sas_precise_align_draft"
_ANCHORS = "sas_precise_align_anchors"
_ERROR = "sas_precise_align_error"
_REVERSE = "sas_precise_reverse"
_CALIBRATIONS = {
    CALIBRATION_RELATIVE_EXTREMA: "Relative signal",
    CALIBRATION_ABSOLUTE_DARK_REFERENCE: "Dark / reference",
    CALIBRATION_ABSOLUTE_GAIN_OFFSET: "Detector gain / offset",
}


def clear_draft():
    """Discard just this editor's controls, leaving file and committed fit state intact."""
    state = st.session_state
    numeric_keys = {key for _, _, key in _FIELDS}
    for key in list(state):
        if (key.startswith(("sas_precise_align_", "sas_precise_slider_", "sas_precise_detector_"))
                or key == _REVERSE or key in numeric_keys):
            del state[key]


def draft_alignment(initial):
    return st.session_state.get(_DRAFT, initial)


def set_draft_alignment(alignment):
    """A new fast-fit result centers each slider on the exact fitted coordinate."""
    state = st.session_state
    state[_DRAFT] = alignment
    state.pop(_ANCHORS, None)
    state.pop(_ERROR, None)


def _slider_position(name, value, anchor, span):
    if name in ("scale", "signal_scale"):
        return (math.log(value) - math.log(anchor)) / math.log(4.)
    return (value - anchor) / span


def _change_coordinate(name, key, anchor, span, from_slider):
    state = st.session_state
    value = state[key]
    try:
        if from_slider:
            if name in ("scale", "signal_scale"):
                value = anchor * 4. ** value
            else:
                value = anchor + span * value
        state[_DRAFT] = replace(state[_DRAFT], **{name: value})
    except (ValueError, ArithmeticError, TypeError):
        state[_ERROR] = "Enter a finite alignment value. Both scale factors must be positive."
    else:
        state.pop(_ERROR, None)


def _reverse_changed():
    state = st.session_state
    state[_DRAFT] = replace(state[_DRAFT], reverse=state[_REVERSE])


def render_alignment_controls(panel, alignment, *, span_x, span_y, absolute):
    """Pair normalized sliders with physical values, matching the AutoOD editor.

    A slider covers a factor of four either way for scale, or one displayed span
    either way for shift. A typed value outside the range establishes a new center.
    """
    state = st.session_state
    draft = draft_alignment(alignment)
    if absolute and (draft.signal_scale != 1. or draft.signal_offset != 0.):
        draft = replace(draft, signal_scale=1., signal_offset=0.)
    state[_DRAFT] = draft
    anchors = state.setdefault(_ANCHORS, {})
    rows = panel.container(key="sas_precise_alignment_rows")
    rows.markdown("**Experimental trace alignment**")
    for name, label, numeric_key in _FIELDS:
        value = float(getattr(draft, name))
        span = max(float(span_x), 1.e-9) if name == "shift" else max(float(span_y), .05)
        anchor = anchors.setdefault(name, value)
        position = _slider_position(name, value, anchor, span)
        if not -1. <= position <= 1.:
            anchor = anchors[name] = value
            position = 0.
        slider_key = "sas_precise_slider_" + name
        # Mirror the canonical draft before mounting either widget. Reading the
        # slider later would round a precisely typed value to the nearest slider step.
        if state.get(numeric_key) != value:
            state[numeric_key] = value
        if state.get(slider_key) != position:
            state[slider_key] = position
        label_col, slide_col, entry_col = rows.columns([.95, 1.5, 1.45], vertical_alignment="center")
        label_col.write(label)
        disabled = absolute and name.startswith("signal_")
        slide_col.slider(label, -1., 1., step=.001, key=slider_key, label_visibility="collapsed",
                         disabled=disabled, on_change=_change_coordinate,
                         args=(name, slider_key, anchor, span, True))
        step = max(anchor / 100., math.ulp(anchor)) if name in ("scale", "signal_scale") else span / 100.
        entry_col.number_input(label, step=step, format="%.15g", key=numeric_key,
                               label_visibility="collapsed", disabled=disabled,
                               on_change=_change_coordinate, args=(name, numeric_key, anchor, span, False))
    if state.get(_REVERSE) != draft.reverse:
        state[_REVERSE] = draft.reverse
    rows.checkbox("Invert X-axis", key=_REVERSE, on_change=_reverse_changed)
    if error := state.get(_ERROR):
        rows.warning(error)
    return draft


def render_detector_calibration(panel, upload, trace, *, mode, unit, denoise):
    """Preview a detector calibration while retaining the declared native law on reopen."""
    from gabes_ui.od_measurement import load_trace

    committed = st.session_state.get("sas_csv_detector", {})
    if mode != "relative":
        return trace, committed

    diag = trace.correction_diagnostics
    current = diag.calibration_mode
    panel.markdown("**Detector calibration**")
    choice = panel.radio("Transmission calibration", tuple(_CALIBRATIONS),
                             index=tuple(_CALIBRATIONS).index(current), format_func=_CALIBRATIONS.get, horizontal=True,
                             key="sas_precise_detector_law", label_visibility="collapsed")
    calibrated = trace.is_absolute_transmission
    gain = diag.calibration_gain if calibrated else 1.
    offset = diag.calibration_offset if calibrated else 0.
    original = {"calibration_mode": current}
    candidate = {"calibration_mode": choice}
    if choice == CALIBRATION_ABSOLUTE_DARK_REFERENCE:
        defaults = {"dark_signal": offset, "reference_signal": offset + gain}
        labels = ("Dark signal", "Reference signal (T = 1)")
    elif choice == CALIBRATION_ABSOLUTE_GAIN_OFFSET:
        defaults = {"gain": gain, "offset": offset}
        labels = ("Detector gain", "Detector offset")
    else:
        defaults = {}
        labels = ()
    if defaults:
        for col, (name, default), label in zip(panel.columns(2), defaults.items(), labels):
            candidate[name] = col.number_input(label, value=float(default), format="%.12g",
                                                key="sas_precise_detector_" + name)
            original[name] = default
    if candidate == original:
        # Empty explicit overrides are significant: load_measurement restores the
        # signed detector gain and the frequency reference from the original worksheet.
        return trace, committed
    try:
        adjusted = load_trace(upload, mode=mode, x_unit=unit, denoise=denoise, **candidate)
    except ValueError as exc:
        panel.error(str(exc))
        return None, None
    return adjusted, candidate
