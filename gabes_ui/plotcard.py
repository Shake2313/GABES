"""Plot card — the figure and everything that hangs off it.

Views as underline tabs (only the selected figure is drawn), the popped figure
title as a caption, experimental CSV overlay in a side panel, and the tables,
diagnostics and heavy extra views behind the More pills
(docs/ui_redesign/decisions.md D8).

PLOT_LOCK protects process-wide Matplotlib state for GABES and SABES. The
callers hold it for observables that build figures. This module holds it for
artist access, copy, drawing and cleanup; the SABES page also uses it for
instrument traces. Cached solvers run separately.
"""
import copy
import hashlib
import importlib
import inspect
from html import escape
from threading import RLock

import numpy as np
import streamlit as st

import gabes as _gabes
import gabes.experimental_csv as _experimental_csv
from gabes import schemes
from gabes.core import blas_single_thread
from gabes.experimental_alignment import AxisAlignment, frame_alignment, fit_trace_alignment
from gabes.plot_style import PALETTE, apply_gabes_plot_style
from gabes_ui import export as ui_export
from gabes_ui.controls import render_param, skey
from gabes_ui.plot_focus import focused_axes, focus_signature, resonance_window
from gabes_ui.readout import caption_html

PLOT_LOCK = RLock()

# Temporary user preference; restore the existing tabs by setting this True.
SHOW_SAS_EXTRA_VIEWS = False

_EXPERIMENTAL_CSV_IMPORT_LOCK = _gabes.__dict__.setdefault(
    "_streamlit_experimental_csv_import_lock", RLock()
)
_EXPERIMENTAL_CSV_API = (
    "CALIBRATION_ABSOLUTE_DARK_REFERENCE",
    "CALIBRATION_ABSOLUTE_GAIN_OFFSET",
    "CALIBRATION_RELATIVE_EXTREMA",
    "MAX_FILE_BYTES",
    "ExperimentalCSVError",
    "load_experimental_csv",
)
with _EXPERIMENTAL_CSV_IMPORT_LOCK:
    if any(not hasattr(_experimental_csv, name) for name in _EXPERIMENTAL_CSV_API):
        importlib.invalidate_caches()
        _experimental_csv = importlib.reload(_experimental_csv)

    CALIBRATION_ABSOLUTE_DARK_REFERENCE = (
        _experimental_csv.CALIBRATION_ABSOLUTE_DARK_REFERENCE
    )
    CALIBRATION_ABSOLUTE_GAIN_OFFSET = (
        _experimental_csv.CALIBRATION_ABSOLUTE_GAIN_OFFSET
    )
    CALIBRATION_RELATIVE_EXTREMA = _experimental_csv.CALIBRATION_RELATIVE_EXTREMA
    MAX_FILE_BYTES = _experimental_csv.MAX_FILE_BYTES
    ExperimentalCSVError = _experimental_csv.ExperimentalCSVError
    load_experimental_csv = _experimental_csv.load_experimental_csv


class _UnresolvedExtraView(Exception):
    """Carry an unresolved result through the cache to the current view."""

    def __init__(self, data):
        super().__init__("Extra view did not resolve.")
        self.data = data


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_extra(scheme_name, view_key, param_items, cache_version, *, retry_unresolved=False):
    scheme = schemes.get(scheme_name)
    view = next(v for v in scheme.extra_views() if v.key == view_key)
    with blas_single_thread():
        data = view.compute(dict(param_items))
    if retry_unresolved and data.get("status") == "unresolved":
        # Streamlit only caches returns, so another Run can retry this calculation.
        raise _UnresolvedExtraView(data)
    return data


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_experimental_csv(
    csv_bytes,
    denoise,
    calibration_mode=CALIBRATION_RELATIVE_EXTREMA,
    dark_signal=None,
    reference_signal=None,
    gain=None,
    offset=None,
):
    """Parse/correct uploaded scope data outside the physics solve cache."""
    # Streamlit can retain a cached wrapper across a source reload while
    # replacing the module. Resolve the current class factory before pickling.
    return importlib.import_module("gabes.experimental_csv").load_experimental_csv(
        csv_bytes,
        denoise=denoise,
        calibration_mode=calibration_mode,
        dark_signal=dark_signal,
        reference_signal=reference_signal,
        gain=gain,
        offset=offset,
    )


def _close_fig(fig):
    from matplotlib import pyplot as plt
    with PLOT_LOCK:
        plt.close(fig)


def render_fig(fig):
    """Style, draw and release a figure under the app's Matplotlib lock."""
    with PLOT_LOCK:
        try:
            apply_gabes_plot_style(fig)
            st.pyplot(fig)
        finally:
            _close_fig(fig)


def _diagnostic_value(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


_OVERLAY_PREFERENCES = (
    "_overlay", "_csv_auto_correct", "_csv_show_raw", "_csv_calibration_mode",
    "_csv_dark_signal", "_csv_reference_signal", "_csv_detector_gain", "_csv_detector_offset",
    "_csv_x_scale", "_csv_x_multiplier", "_csv_x_shift", "_csv_reverse", "_csv_invert",
    "_csv_y_scale", "_csv_y_offset", "_csv_mode", "_csv_unit",
)


def _overlay_upload_key(scheme_name):
    epoch = st.session_state.get(skey(scheme_name, "_csv_upload_epoch"), 0)
    return skey(scheme_name, f"_csv_upload_{epoch}" if epoch else "_csv_upload")


def _save_overlay_preferences(scheme_name):
    state = st.session_state
    keys = [skey(scheme_name, name) for name in _OVERLAY_PREFERENCES]
    keys.extend(("sas_od_fit_branch", "sas_od_fit_free", "sas_od_fit_frequency_scale",
                 "sas_od_fit_holdout", "sas_od_fit_start", "sas_od_fit_end",
                 "sas_od_holdout_start", "sas_od_holdout_end"))
    state.setdefault(skey(scheme_name, "_model_return_overlay"), {}).update(
        {key: state[key] for key in keys if key in state})


def remember_overlay_for_comparison(scheme_name, *, entering):
    """Streamlit drops hidden widgets; keep a session-only copy across model comparison."""
    state = st.session_state
    saved_key = skey(scheme_name, "_model_return_overlay")
    if entering:
        state.pop("sas_od_pending_temperature", None)
        _save_overlay_preferences(scheme_name)
    else:
        for key, value in state.pop(saved_key, {}).items():
            state[key] = value
        if skey(scheme_name, "_csv_retained_file") in state:
            # A remounted native uploader can have an empty browser file list
            # while Python still sees its previous UploadedFile. Give the new
            # picker a fresh identity and let the visible retained-file control
            # own the previous file and its removal.
            epoch_key = skey(scheme_name, "_csv_upload_epoch")
            state[epoch_key] = state.get(epoch_key, 0) + 1


def _forget_retained_csv(scheme_name):
    if scheme_name == "sas":
        st.session_state.pop("sas_csv_source", None)
    st.session_state.pop(skey(scheme_name, "_csv_retained_file"), None)
    st.session_state.pop(skey(scheme_name, "_csv_fingerprint"), None)
    st.session_state.pop("sas_od_physical_fit", None)
    st.session_state.pop("sas_od_pending_temperature", None)
    st.session_state.pop(skey(scheme_name, "_model_return_overlay"), None)


def _csv_uploader(panel, scheme_name, *, workspace):
    uploader_options = {}
    if "max_upload_size" in inspect.signature(panel.file_uploader).parameters:
        # Streamlit added this per-widget guard after the project's oldest
        # supported release; retain the backend byte check as the fallback.
        limit = MAX_FILE_BYTES
        if workspace:
            from gabes.od_measurement import MAX_COMPARISON_BYTES
            limit = MAX_COMPARISON_BYTES
        uploader_options["max_upload_size"] = limit // (1024 * 1024)
    return panel.file_uploader(
        "Measurement CSV" if workspace else "Oscilloscope CSV",
        type=("csv",),
        key=_overlay_upload_key(scheme_name),
        on_change=_forget_retained_csv,
        args=(scheme_name,),
        help=("Open a measured trace, oscilloscope file, or a standard or Origin comparison export."
              if workspace else "A and B are read as arbitrary units; no header is required."),
        **uploader_options,
    )


def select_workspace_measurement(panel, scheme_name):
    """Read the file before computing the curve so its recorded model can select the solve."""
    from gabes_ui.od_measurement import select_upload

    uploaded = _csv_uploader(panel, scheme_name, workspace=True)
    try:
        selected = select_upload(panel, scheme_name, uploaded, _forget_retained_csv)
    except ValueError:
        _forget_retained_csv(scheme_name)
        raise
    if selected is None:
        st.session_state.pop(skey(scheme_name, "_csv_retained_file"), None)
        return None
    return selected[1]


def _render_physical_od_fit(panel, trace, alignment, params, fit_context, *, upload=None,
                            pending_temperature=None):
    from gabes.od_model import ODConditions, ODUnsupportedError, od_unavailable
    from gabes_ui.od_measurement import request_temperature, fit_conditions_key, workspace_params
    from gabes_ui.controls import current_params
    from gabes.od_spectrum_fit import (
        PARAMETERS, fit_od_spectrum, od_fit_unavailable, prepare_od_fit,
    )

    workspace = upload is not None
    box = panel.expander("Fit physical OD parameters", expanded=False)
    if workspace:
        notice = st.session_state.pop("sas_od_apply_notice", None)
        if notice:
            box.info(notice)
        try:
            conditions = ODConditions.from_params(params)
            reason = od_unavailable(conditions)
        except ODUnsupportedError as exc:
            reason = str(exc)
    else:
        reason = od_fit_unavailable(params)
    if reason:
        if pending_temperature is not None:
            box.info("The model changed. Run a new fit before applying a result.")
        box.caption(reason)
        return None
    if not trace.is_absolute_transmission:
        box.caption("Absolute transmission calibration is required for a physical fit.")
        return None
    if not trace.branches:
        box.caption("The trace has no usable acquisition branch.")
        return None
    box.caption("Uses the raw signal from one sweep and the fixed frequency scale above. "
                "Line strength stays fixed. Fit results do not apply settings automatically.")
    branch = box.selectbox(
        "Acquisition branch", range(len(trace.branches)), key="sas_od_fit_branch",
        format_func=lambda i: f"{i + 1} · {trace.branches[i].direction} · {trace.branches[i].sample_count} points",
    )
    free = box.multiselect("Free parameters", tuple(PARAMETERS), default=("temp_c", "offset_mhz"),
                           key="sas_od_fit_free", format_func=lambda key: PARAMETERS[key][0])
    calibrated_axis = box.checkbox("The frequency scale above is calibrated", key="sas_od_fit_frequency_scale")
    use_holdout = box.checkbox("Declare separate fit and held-out ranges", key="sas_od_fit_holdout")
    fit_range = holdout_range = None
    if use_holdout:
        edges = np.asarray(alignment.transform_x(trace.detuning)) * 1e3
        start, end = float(np.min(edges)), float(np.max(edges))
        split = start + 0.75 * (end - start)
        left, right = box.columns(2)
        fit_range = (
            left.number_input("Fit start (MHz)", value=start, key="sas_od_fit_start"),
            right.number_input("Fit end (MHz)", value=split, key="sas_od_fit_end"),
        )
        holdout_range = (
            left.number_input("Held-out start (MHz)", value=split + 1e-6, key="sas_od_holdout_start"),
            right.number_input("Held-out end (MHz)", value=end, key="sas_od_holdout_end"),
        )
    key = "sas_od_physical_fit"
    model_key = fit_conditions_key(params, free) if workspace else tuple(sorted(params.items()))
    identity = (fit_context, model_key, alignment.scale, alignment.shift,
                alignment.reverse, branch, tuple(free), calibrated_axis, fit_range, holdout_range)
    saved = st.session_state.get(key)
    if workspace and pending_temperature is not None:
        try:
            live = workspace_params(current_params("sas", schemes.get("sas")), st.session_state, upload)
            current_key = fit_conditions_key(live, free)
        except (ValueError, TypeError):
            current_key = None
        if (saved and saved[0] == pending_temperature == identity and current_key == model_key
                and saved[1]["identifiable"] and saved[1]["status"] == "converged"):
            value = next(row["value"] for row in saved[1]["parameters"] if row["key"] == "temp_c")
            st.session_state[skey("sas", "temp_c")] = value
            st.session_state["sas_od_apply_notice"] = f"Applied fitted cell temperature {value:.4g} °C."
            st.rerun()
        box.info("The file, calibration or model changed. Run the fit again before applying it.")
    if saved is not None and saved[0] != identity:
        if workspace:
            box.info("The file, calibration or model changed. Run the fit again before applying it.")
        st.session_state.pop(key)
        saved = None
    if box.button("Run physical OD fit", key="sas_run_od_fit", disabled=not free or not calibrated_axis):
        try:
            data = prepare_od_fit(trace, alignment, branch=branch)
            with box, st.spinner("Fitting the OD spectrum…"), blas_single_thread():
                result = fit_od_spectrum(data, params, free=free, fit_range_mhz=fit_range,
                                         heldout_range_mhz=holdout_range,
                                         model_id=conditions.model_id if workspace else None)
            saved = identity, result
            st.session_state[key] = saved
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.session_state.pop(key, None)
            saved = None
            box.error(str(exc))
    if saved is None:
        return None
    result = saved[1]
    if not result["identifiable"]:
        box.warning("The selected parameters are poorly constrained or at a bound. "
                    "Fix a parameter such as cell length and fit again.")
    box.dataframe(result["parameters"], hide_index=True)
    box.caption(f"Fit RMS {result['fit_residuals']['rms']:.4g} transmission · "
                f"{result['status'].replace('_', ' ')} · {result['scope'].replace('_', ' ')}")
    if result["heldout_residuals"] is not None:
        box.caption(f"Held-out RMS {result['heldout_residuals']['rms']:.4g} transmission")
    if workspace and "temp_c" in free:
        box.button("Apply fitted temperature", key="sas_od_apply_temperature",
                   disabled=not result["identifiable"] or result["status"] != "converged",
                   on_click=request_temperature, args=(identity,))
    return result


def _render_experimental_comparison(view, scheme_name, panel, params=None):
    """Render a scheme-declared CSV panel into `panel` and overlay its trace.

    Uploaded bytes and alignment controls intentionally stay outside `params`:
    changing them must reuse both the heavy solve and the cached base figure.
    """
    descriptor = view.get("comparison")
    fig = view.get("figure")
    if not descriptor or fig is None:
        return

    with PLOT_LOCK:
        axis_index = int(descriptor.get("axis_index", 0))
        if not 0 <= axis_index < len(fig.axes):
            return
        axis = fig.axes[axis_index]
        xlim = tuple(float(v) for v in axis.get_xlim())
        ylim = tuple(float(v) for v in axis.get_ylim())
    # Relative extrema data span 0–1 even when the theory's absorption is weak.
    # Include that range so the automatic correction cannot hide the CSV floor.
    ylim = (min(ylim[0], -0.05), max(ylim[1], 1.05))
    x_unit = descriptor.get("x_unit", "plot unit")
    raw_x_unit = descriptor.get("raw_x_unit", "Arb. unit")
    raw_y_unit = descriptor.get("raw_y_unit", "Arb. unit")

    workspace = bool(view.get("experimental_workspace")) and x_unit == "GHz"
    pending_temperature = st.session_state.pop("sas_od_pending_temperature", None) if workspace else None
    if workspace:
        from gabes import od_measurement as measurement
        from gabes_ui import od_measurement as workspace_ui

    with panel:
        if view.get("experimental_workspace") and not workspace:
            panel.caption("This generic model uses a relative MHz axis. Choose a resolved Rb or Cs line for calibrated OD fitting.")
        projection = uploaded = None
        mode = unit = None
        if view.get("experimental_workspace"):
            selected = st.session_state.get(skey(scheme_name, "_csv_retained_file"))
            if selected is None:
                return
            filename, measurement_file = selected
            if workspace:
                projection = measurement_file
            else:
                csv_bytes = measurement_file.payload
        else:
            panel.caption("CSV A = X (time/frequency), B = detector signal. Metadata and later columns are ignored.")
            uploaded = _csv_uploader(panel, scheme_name, workspace=False)
        if workspace:
            source_key = (projection.sha256, projection.columns, projection.scope.od if projection.scope else None)
            new_source = st.session_state.get("sas_csv_source") != source_key
            if new_source:
                st.session_state["sas_csv_source"] = source_key
                for control in ("_csv_mode", "_csv_unit", "_csv_calibration_mode", "_csv_dark_signal",
                                "_csv_reference_signal", "_csv_detector_gain", "_csv_detector_offset"):
                    st.session_state.pop(skey(scheme_name, control), None)
                st.session_state.pop("sas_od_fit_frequency_scale", None)
            unit_labels = {"arbitrary": "Arbitrary units", "GHz": "GHz", "MHz": "MHz", "Hz": "Hz"}
            mode_labels = {"relative": "Relative detector signal", "fraction": "Transmission (0–1)",
                           "percent": "Transmission (%)"}
            cols = panel.columns(2)
            unit = cols[0].selectbox("Measured X unit", tuple(unit_labels),
                index=tuple(unit_labels).index(projection.x_unit or "arbitrary"),
                format_func=unit_labels.get, key=skey(scheme_name, "_csv_unit"))
            mode = cols[1].selectbox("Measured signal", tuple(mode_labels),
                index=tuple(mode_labels).index(projection.mode or "relative"),
                format_func=mode_labels.get, key=skey(scheme_name, "_csv_mode"))
            raw_x_unit = ("GHz" if unit != "arbitrary" else "Arb. unit")
        auto_correct = panel.checkbox(
            "Automatic noise correction",
            value=True,
            key=skey(scheme_name, "_csv_auto_correct"),
            help="Merge repeated A values, reject isolated spikes, and apply "
                 "noise-adaptive local smoothing before the selected calibration.",
        )
        show_raw = panel.checkbox(
            "Show unfiltered trace",
            value=False,
            disabled=not auto_correct,
            key=skey(scheme_name, "_csv_show_raw"),
        )
        calibration_options = {
            "Relative normalized transmission (extrema-scaled)":
                CALIBRATION_RELATIVE_EXTREMA,
            "Absolute transmission (dark/reference)":
                CALIBRATION_ABSOLUTE_DARK_REFERENCE,
            "Absolute transmission (gain/offset)":
                CALIBRATION_ABSOLUTE_GAIN_OFFSET,
        }
        if workspace and new_source and mode == "relative":
            try:
                workspace_ui.restore_detector_law(scheme_name, projection, calibration_options)
            except ValueError as exc:
                panel.error(str(exc))
                return
        if workspace and mode != "relative":
            calibration_options = {mode_labels[mode]: None}
        calibration_choice = panel.selectbox(
            "Transmission calibration",
            tuple(calibration_options),
            key=skey(scheme_name, "_csv_calibration_mode"),
            help="Extrema scaling is relative only. Absolute transmission "
                 "requires measured dark/reference levels or an explicit "
                 "detector law: signal = offset + gain × transmission.",
        )
        calibration_mode = calibration_options[calibration_choice]
        dark_signal = reference_signal = gain = offset = None
        if calibration_mode == CALIBRATION_ABSOLUTE_DARK_REFERENCE:
            calibration_cols = panel.columns(2)
            dark_signal = calibration_cols[0].number_input(
                f"Dark signal  [{raw_y_unit}]",
                value=0.0,
                key=skey(scheme_name, "_csv_dark_signal"),
                help="Detector signal measured at zero transmitted light.",
            )
            reference_signal = calibration_cols[1].number_input(
                f"Reference signal (T = 1)  [{raw_y_unit}]",
                value=1.0,
                key=skey(scheme_name, "_csv_reference_signal"),
                help="Detector signal measured for unit-transmission reference light.",
            )
        elif calibration_mode == CALIBRATION_ABSOLUTE_GAIN_OFFSET:
            calibration_cols = panel.columns(2)
            gain = calibration_cols[0].number_input(
                f"Detector gain  [{raw_y_unit}]",
                value=1.0,
                key=skey(scheme_name, "_csv_detector_gain"),
                help="Gain in signal = offset + gain × transmission; a negative "
                     "gain represents inverted detector polarity.",
            )
            offset = calibration_cols[1].number_input(
                f"Detector offset  [{raw_y_unit}]",
                value=0.0,
                key=skey(scheme_name, "_csv_detector_offset"),
                help="Signal value corresponding to zero transmission.",
            )
        if projection is not None:
            fingerprint = projection.sha256
        elif uploaded is not None:
            filename, csv_bytes = uploaded.name, uploaded.getvalue()
        elif not view.get("experimental_workspace"):
            retained = st.session_state.get(skey(scheme_name, "_csv_retained_file"))
            if retained is None:
                return
            filename, csv_bytes = retained
            if not isinstance(csv_bytes, bytes):
                csv_bytes = csv_bytes.payload
            panel.button("Remove retained CSV", key=skey(scheme_name, "_csv_remove_retained"),
                         on_click=_forget_retained_csv, args=(scheme_name,),
                         help="Remove the file kept in this session during model comparison.")
        if projection is None:
            fingerprint = hashlib.sha256(csv_bytes).hexdigest()
        fingerprint_key = skey(scheme_name, "_csv_fingerprint")
        scale_key = skey(scheme_name, "_csv_x_scale")
        multiplier_key = skey(scheme_name, "_csv_x_multiplier")
        shift_key = skey(scheme_name, "_csv_x_shift")
        reverse_key = skey(scheme_name, "_csv_reverse")
        invert_key = skey(scheme_name, "_csv_invert")
        signal_scale_key = skey(scheme_name, "_csv_y_scale")
        signal_offset_key = skey(scheme_name, "_csv_y_offset")
        fit_key = skey(scheme_name, "_csv_alignment_fit")
        native_shift = 0.0
        if workspace and unit != "arbitrary":
            native_shift, origin_warning = workspace_ui.native_reference_shift(projection, params)
            if origin_warning:
                panel.warning(origin_warning)
        alignment_identity = (
            fingerprint, native_shift,
            axis_index,
            str(x_unit),
            str(raw_x_unit), unit, None if projection is None else projection.columns,
        )
        try:
            if workspace:
                trace = workspace_ui.load_trace(projection, mode=mode, x_unit=unit, denoise=auto_correct,
                    calibration_mode=calibration_mode, dark_signal=dark_signal,
                    reference_signal=reference_signal, gain=gain, offset=offset)
                calibration_mode = trace.correction_diagnostics.calibration_mode
            else:
                trace = _cached_experimental_csv(csv_bytes, auto_correct, calibration_mode,
                    dark_signal, reference_signal, gain, offset)
        except ValueError as exc:
            panel.error(str(exc))
            return

        detuning = trace.detuning
        plot_span = float(xlim[1] - xlim[0])
        try:
            framed = frame_alignment(detuning, xlim)
        except ValueError as exc:
            panel.error(str(exc))
            return
        framed_scale, framed_shift = framed.scale, framed.shift

        def set_alignment(alignment, *, invert=False):
            revision_key = skey(scheme_name, "_csv_alignment_revision")
            st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
            st.session_state[scale_key] = float(alignment.scale)
            st.session_state[multiplier_key] = 1.0
            st.session_state[shift_key] = float(alignment.shift)
            st.session_state[reverse_key] = bool(alignment.reverse)
            st.session_state[invert_key] = bool(invert)
            st.session_state[signal_scale_key] = float(alignment.signal_scale)
            st.session_state[signal_offset_key] = float(alignment.signal_offset)
            st.session_state.pop(fit_key, None)

        if st.session_state.get(fingerprint_key) != alignment_identity or scale_key not in st.session_state:
            st.session_state[fingerprint_key] = alignment_identity
            set_alignment(AxisAlignment(shift=native_shift) if workspace and unit != "arbitrary" else framed)
        if trace.is_absolute_transmission:
            st.session_state[invert_key] = False
            st.session_state[signal_scale_key] = 1.0
            st.session_state[signal_offset_key] = 0.0

        action_cols = panel.columns(2)
        if action_cols[0].button(
            "Bring into view",
            key=skey(scheme_name, "_csv_frame"),
            use_container_width=True,
            help="Map the imported span into the current theoretical plot; "
                 "this does not fit spectral features.",
        ):
            try:
                reframed = AxisAlignment(
                    scale=framed.scale, shift=framed.shift,
                    reverse=st.session_state[reverse_key],
                    signal_scale=st.session_state[signal_scale_key],
                    signal_offset=st.session_state[signal_offset_key],
                )
                set_alignment(reframed, invert=st.session_state[invert_key])
            except ValueError as exc:
                panel.error(str(exc))
        if action_cols[1].button(
            "Use raw axis",
            key=skey(scheme_name, "_csv_reset"),
            use_container_width=True,
        ):
            try:
                raw_axis = AxisAlignment(
                    shift=native_shift, signal_scale=st.session_state[signal_scale_key],
                    signal_offset=st.session_state[signal_offset_key],
                )
                set_alignment(raw_axis, invert=st.session_state[invert_key])
            except ValueError as exc:
                panel.error(str(exc))
        # The workspace compares against the scheme's typed attenuation readout.
        readout = view.get("optical_readout", {})
        theory = axis.lines[0] if axis.lines else None
        if workspace and "transmission" in readout and "x" in readout:
            theory_x = np.asarray(readout["x"])
            theory_y = np.asarray(readout["transmission"])
        elif theory is not None:
            theory_x, theory_y = np.asarray(theory.get_xdata()), np.asarray(theory.get_ydata())
        else:
            theory_x = theory_y = None
        fit_context_key = skey(scheme_name, "_csv_fit_context")
        fit_context = (
            fingerprint, auto_correct, calibration_mode,
            dark_signal, reference_signal, gain, offset, unit, mode,
            None if projection is None else projection.columns,
            None if theory_x is None else hashlib.sha256(np.asarray([theory_x, theory_y]).tobytes()).hexdigest(),
        )
        if st.session_state.get(fit_context_key) != fit_context:
            st.session_state.pop(fit_key, None)
            st.session_state[fit_context_key] = fit_context
        if panel.button(
            "Fit alignment", key=skey(scheme_name, "_csv_fit"),
            use_container_width=True, disabled=theory is None,
            help="Align X scale/shift to the current theoretical transmission. "
                 "For relative data, also fit detector gain/offset. Physical "
                 "parameters are held fixed; choose sweep direction first.",
        ):
            try:
                initial = AxisAlignment(
                    scale=st.session_state[scale_key] * st.session_state[multiplier_key],
                    shift=st.session_state[shift_key],
                    reverse=st.session_state[reverse_key],
                    signal_scale=st.session_state[signal_scale_key],
                    signal_offset=st.session_state[signal_offset_key],
                )
                if workspace:
                    result = measurement.quick_measurement_alignment(trace, theory_x, theory_y,
                        current=initial, reverse=initial.reverse, invert=st.session_state[invert_key])
                else:
                    result = fit_trace_alignment(trace, theory_x, theory_y,
                        initial=initial, invert=st.session_state[invert_key])
                if workspace or result.success:
                    set_alignment(result.alignment, invert=st.session_state[invert_key])
                st.session_state[fit_key] = result
            except ValueError as exc:
                panel.error(str(exc))

        panel.caption(
            "The native frequency scale is preserved. Confirm its calibration before a quantitative fit."
            if workspace and unit != "arbitrary" else
            "X span is framed automatically. Confirm the frequency calibration before a quantitative fit."
        )
        # Forms keep slider drags in the browser and send one update on Apply.
        precise = panel.expander("Precise alignment · base scale, direction and signal", expanded=False)
        with precise.form(skey(scheme_name, "_csv_alignment_form")):
            base_scale = st.number_input(
                f"Base X scale  [{x_unit}/{raw_x_unit}]", min_value=0.0,
                key=scale_key, format="%.6g",
                step=float(st.session_state[scale_key] * 0.01),
                help="Automatic coarse scale; scientific notation is accepted.",
            )
            multiplier = st.slider(
                "X fine scale ×", 0.1, 3.0, step=0.005, key=multiplier_key,
                help="Stable relative range around the base scale.",
            )
            shift_now = float(st.session_state[shift_key])
            shift_margin = 2.0 * abs(plot_span)
            if workspace:
                # Changing the plotted line must never change the measurement's calibration.
                x_shift = st.number_input(f"X shift  [{x_unit}]", key=shift_key, step=0.001, format="%.8g")
            else:
                x_shift = st.slider(
                    f"X shift  [{x_unit}]", min(xlim[0] - shift_margin, shift_now),
                    max(xlim[1] + shift_margin, shift_now),
                    step=abs(plot_span) / 1000.0, key=shift_key)
            reverse = st.checkbox("Reverse sweep", key=reverse_key)
            invert = st.checkbox(
                "Invert transmission", key=invert_key,
                disabled=trace.is_absolute_transmission,
                help="Absolute detector polarity is defined by the calibration gain.",
            )
            signal_scale = st.number_input(
                "Y scale ×", min_value=0.0, step=0.01, key=signal_scale_key,
                disabled=trace.is_absolute_transmission,
            )
            signal_offset = st.number_input(
                "Y offset", step=0.01, key=signal_offset_key,
                disabled=trace.is_absolute_transmission,
            )
            applied = st.form_submit_button(
                "Apply alignment", key=skey(scheme_name, "_csv_apply"),
                use_container_width=True,
            )
        if applied:
            revision_key = skey(scheme_name, "_csv_alignment_revision")
            st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
            st.session_state.pop(fit_key, None)
        x_scale = float(base_scale) * float(multiplier)
        try:
            alignment = AxisAlignment(
                scale=x_scale, shift=float(x_shift), reverse=bool(reverse),
                signal_scale=float(signal_scale), signal_offset=float(signal_offset),
            )
            aligned_x = alignment.transform_x(trace.detuning)
            aligned_y = alignment.transform_y(trace.transmission, invert=bool(invert))
        except ValueError as exc:
            panel.error(str(exc))
            return
        fit_result = st.session_state.get(fit_key)
        if fit_result is not None:
            if workspace:
                if fit_result.matched:
                    panel.caption(f"Feature alignment · {fit_result.candidates} candidate(s)")
                else:
                    panel.info("No matching feature. Showing the measured span within the plot.")
            elif fit_result.success:
                panel.caption(
                    f"Alignment RMSE {fit_result.rmse:.4g} · "
                    f"overlap {fit_result.overlap_fraction:.0%}. "
                    "Temperature and isotope composition were held fixed."
                )
            else:
                panel.warning(fit_result.message)

        physical_fit = None
        if scheme_name == "sas" and params is not None and x_unit == "GHz":
            # Exclude the displayed curve from the physical fit's file/calibration identity.
            physical_fit = _render_physical_od_fit(panel, trace, alignment, params,
                fit_context[:-1] if workspace else fit_context, upload=projection,
                pending_temperature=pending_temperature)

        import_diag = trace.import_diagnostics
        correction_diag = trace.correction_diagnostics
        valid_rows = _diagnostic_value(
            import_diag, "valid_rows", "accepted_rows", default="?"
        )
        ignored_rows = _diagnostic_value(
            import_diag, "ignored_rows", "skipped_rows", default="?"
        )
        merged_rows = _diagnostic_value(
            import_diag, "duplicate_rows_merged", "merged_rows",
            "duplicate_rows", default="?"
        )
        panel.caption(
            f"{filename} · " + (f"{projection.columns[0]} / {projection.columns[1]} · " if workspace else "") +
            f"{valid_rows} numeric rows · "
            f"{len(detuning)} unique detuning points · {merged_rows} duplicates "
            f"merged · {ignored_rows} rows ignored"
        )
        panel.caption(
            f"Acquisition order: {import_diag.forward_sample_count} forward · "
            f"{import_diag.reverse_sample_count} reverse · "
            f"{import_diag.sweep_reversal_count} reversal(s) · "
            f"{import_diag.sweep_branch_count} preserved branch(es)"
        )
        if import_diag.directional_merge_warning:
            panel.warning(import_diag.directional_merge_warning)
        floor = _diagnostic_value(correction_diag, "floor", "floor_level")
        ceiling = _diagnostic_value(correction_diag, "ceiling", "ceiling_level")
        window = _diagnostic_value(
            correction_diag, "smoothing_window", "window_size", default=1
        )
        if correction_diag.is_absolute:
            panel.caption(
                f"{trace.transmission_label} [{correction_diag.calibration_source}]: "
                f"offset {correction_diag.calibration_offset:.6g} · "
                f"gain {correction_diag.calibration_gain:.6g} [{raw_y_unit}] · "
                f"smoothing window {window}"
            )
        elif floor is not None and ceiling is not None:
            panel.caption(
                f"{trace.transmission_label} [{raw_y_unit}]: relative floor "
                f"{float(floor):.6g} · relative ceiling {float(ceiling):.6g} · "
                f"smoothing window {window}"
            )
        warnings = _diagnostic_value(
            correction_diag, "warnings", "warning", default=()
        )
        if isinstance(warnings, str):
            warnings = (warnings,)
        for warning in warnings or ():
            panel.warning(str(warning))

        in_view = ((aligned_x >= xlim[0]) & (aligned_x <= xlim[1])).any()
        if not in_view:
            panel.warning(
                "The imported trace is outside the theoretical x range. "
                "Use Bring into view, then refine X scale and X shift manually."
            )

        raw_overlay = None
        if auto_correct and show_raw:
            raw_x = aligned_x
            raw_y = trace.calibrate_signal(trace.raw_signal)
            try:
                raw_y = alignment.transform_y(raw_y, invert=bool(invert))
            except ValueError as exc:
                panel.error(str(exc))
                return
            raw_overlay = (raw_x, raw_y)

        overlay_label = (
            f"{descriptor.get('label', 'Experimental CSV')} — "
            f"{trace.transmission_label.lower()}"
        )

    workspace_payload = {}
    plot_x, plot_y = aligned_x, aligned_y
    if workspace:
        parts = measurement.gap_slices(trace)
        coverage = sorted(tuple(sorted((float(aligned_x[s.start]), float(aligned_x[s.stop - 1]))))
                          for s in parts)
        if len(parts) > 1:
            def break_gaps(values):
                return np.concatenate([np.r_[values[s], np.nan] for s in parts])[:-1]
            plot_x, plot_y = break_gaps(aligned_x), break_gaps(aligned_y)
            if raw_overlay is not None:
                raw_overlay = (plot_x, break_gaps(raw_overlay[1]))
        metadata = workspace_ui.plot_metadata(params, view.get("od_model_id"))
        if native_reference := measurement.native_frequency_reference(projection.metadata):
            metadata["experiment_frequency_reference"] = native_reference
        if physical_fit:
            metadata["fit_model_id"] = physical_fit.get("probe_model")
        theory_trace = {"detuning": theory_x.tolist(), "transmission": theory_y.tolist()}
        if readout.get("optical_depth") is not None:
            theory_trace["optical_depth"] = np.asarray(readout["optical_depth"]).tolist()
        workspace_payload = dict(measurement_workspace=True, coverage_ghz=coverage,
                                 theory_trace=theory_trace, model=metadata, input_metadata=projection.metadata)
        if metadata.get("axis_reference") and x_unit == "GHz":
            fmt = panel.selectbox("Comparison CSV format", ["Standard", "Origin"], key="sas_measurement_export_format")
            if panel.button("Prepare comparison CSV", key="sas_prepare_measurement_csv"):
                try:
                    download = measurement.export_comparison_csv(
                        projection, trace, theory_x, theory_y, alignment=alignment, invert=bool(invert),
                        metadata=metadata, origin=fmt == "Origin", mode=mode, x_unit=unit)
                except ValueError as exc:
                    panel.error(str(exc))
                else:
                    panel.download_button("Download comparison CSV", data=download, key="sas_download_measurement_csv",
                        file_name="od-comparison.csv", mime="text/csv", on_click="ignore")

    with PLOT_LOCK:
        if raw_overlay is not None:
            axis.plot(
                raw_overlay[0], raw_overlay[1], color=PALETTE["muted"],
                ls=":", lw=1.0, alpha=0.45, label="CSV · unfiltered", zorder=2,
            )
        axis.plot(
            plot_x, plot_y, color=PALETTE["rose"], lw=1.4, alpha=0.88,
            label=overlay_label, zorder=3,
        )
        # Arbitrary input units must never autoscale the theoretical plot away.
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.legend(loc="best")
        comparison_warnings = ()
        if isinstance(warnings, str):
            comparison_warnings = (warnings,)
        else:
            comparison_warnings = tuple(warnings or ())
        return {
            "enabled": True,
            "kind": "experimental_csv_overlay",
            **workspace_payload,
            "overlay_filename": filename,
            "source_fingerprint_sha256": fingerprint,
            "alignment_revision": st.session_state.get(
                skey(scheme_name, "_csv_alignment_revision"), 0),
            "descriptor": {
                "label": descriptor.get("label"),
                "axis_index": int(axis_index),
                "axis_x_unit": str(x_unit),
                "raw_x_unit": str(raw_x_unit),
                "raw_y_unit": str(raw_y_unit),
            },
            "calibration_mode": str(calibration_mode),
            "calibration_inputs": {
                "auto_correct": bool(auto_correct),
                "dark_signal": dark_signal,
                "reference_signal": reference_signal,
                "detector_gain": gain,
                "detector_offset": offset,
            },
            "alignment": {
                "scale": float(x_scale),
                "shift": float(x_shift),
                "reverse": bool(reverse),
                "invert": bool(invert),
                "base_scale": float(base_scale),
                "manual_multiplier": float(multiplier),
                "signal_scale": float(signal_scale),
                "signal_offset": float(signal_offset),
            },
            **({"physical_od_fit": physical_fit} if physical_fit is not None else {}),
            "alignment_fit": None if fit_result is None else ({
                "success": fit_result.matched, "candidates": fit_result.candidates,
                "physical_parameters_fitted": False,
            } if workspace else {
                "success": bool(fit_result.success),
                "rmse": float(fit_result.rmse),
                "initial_rmse": float(fit_result.initial_rmse),
                "overlap_fraction": float(fit_result.overlap_fraction),
                "points_used": int(fit_result.points_used),
                "message": str(fit_result.message),
                "physical_parameters_fitted": False,
            }),
            "show_raw": bool(show_raw),
            "in_view": bool(in_view),
            "framed_scale": float(framed_scale),
            "framed_shift": float(framed_shift),
            "comparison_trace": {
                "detuning": ui_export.to_json_primitive(aligned_x),
                "transmission": ui_export.to_json_primitive(aligned_y),
                "detuning_unit": str(x_unit),
                "transmission_unit": "1",
                "transmission_label": str(trace.transmission_label),
            },
            "overlay_statistics": {
                "detuning_points": int(len(aligned_x)),
                "valid_numeric_rows": valid_rows,
                "ignored_rows": ignored_rows,
                "merged_duplicates": merged_rows,
            },
            "warnings": ui_export.to_json_primitive(
                tuple(str(item).strip() for item in comparison_warnings if str(item).strip())
            ),
        }


def _pop_figure_title(fig, *, clear=True):
    """Take the title text off the image; the plot card shows it as a caption."""
    parts = []
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text().strip():
        parts.append(suptitle.get_text())
        if clear:
            suptitle.set_text("")
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            text = ax.get_title(loc=loc)
            if text.strip():
                parts.append(text)
                if clear:
                    ax.set_title("", loc=loc)
    return "  ·  ".join(parts)


def _plot_description(fig, scheme, selected, caption, params, metrics):
    """Describe the selected figure using its plotted axes and existing readout."""
    parts = [getattr(scheme, "title", scheme.name)]
    if selected != "Plot":
        parts.append(selected)
    if caption:
        parts.append(" ".join(caption.split()))
    for getter, label in (("get_xlabel", "Horizontal"), ("get_ylabel", "Vertical")):
        axes = dict.fromkeys(getattr(ax, getter)() for ax in fig.axes)
        labels = ", ".join(text for text in axes if text)
        if labels:
            parts.append(f"{label}: {labels}")
    if scheme.name == "fwm" and params:
        if params.get("mode") == "Biphoton":
            parts.append(str(params.get("topology", "Biphoton")))
        else:
            if "tpd" in params:
                parts.append(f"Selected two-photon detuning {params['tpd']:g} MHz")
            if "pump_mw" in params:
                parts.append(f"Pump {params['pump_mw']:g} mW")
    if scheme.name == "sas" and params:
        parts.extend(str(params[k]) for k in ("species", "line") if k in params)
        if "pump_power_mw" in params:
            pump = params["pump_power_mw"]
            parts.append(f"Pump {pump:g} mW")
            if pump == 0:
                parts.append("Pump off: Doppler absorption background")
            else:
                status = next((m["value"] for m in metrics if m.get("label") == "SAS resolution"), None)
                if status:
                    parts.append(f"Sub-Doppler feature: {status}")
    return ". ".join(parts)


def render_plot_card(view, scheme, spec_by_name, raw=None, params=None):
    """Plot views as tabs, overlay/export tools, figure. Returns (comparison
    payload, plot payloads, export popover) for the export step."""
    fig = view.get("figure")
    pages = [
        (item.get("label") or f"View {index + 1}", item.get("figure"))
        for index, item in enumerate(view.get("figure_views", []) or [])
        if isinstance(item, dict) and item.get("figure") is not None
    ]
    if not pages and fig is not None:
        pages = [("Plot", fig)]
    if not pages:
        return None, [], None

    # Temporary display policy: keep all physical readouts/export data intact.
    hidden_pages = []
    if scheme.name == "sas" and not SHOW_SAS_EXTRA_VIEWS:
        hidden_pages = [f for label, f in pages if label != "Transmission"]
        pages = [(label, f) for label, f in pages if label == "Transmission"] or [("Transmission", fig)]

    labels = [label for label, _ in pages]
    selected = labels[0]
    interaction_context = None
    if scheme.name == "fwm" and params is not None:
        interaction_context = dict(
            params=ui_export.to_json_primitive({k: v for k, v in params.items() if k != "tpd"}),
            cache_version=getattr(scheme, "cache_version", "1"))
    card = st.container(border=True, key="gabes_plotcard")
    with card:
        head = st.container(horizontal=True, vertical_alignment="center",
                            gap="small", key="gabes_plothead")
    with head:
        if len(pages) > 1:
            view_key = skey(scheme.name, "_plot_view")
            last_key = skey(scheme.name, "_plot_view_last")
            if st.session_state.get(view_key) not in labels:
                st.session_state[view_key] = (
                    st.session_state.get(last_key)
                    if st.session_state.get(last_key) in labels else labels[0])

            def _keep_view(k=view_key, lk=last_key):
                if st.session_state.get(k) not in labels:
                    st.session_state[k] = st.session_state.get(lk, labels[0])

            selected = st.segmented_control(
                "Plot view", labels, key=view_key, on_change=_keep_view,
                label_visibility="collapsed") or labels[0]
            st.session_state[last_key] = selected
        page_fig = dict(pages)[selected]
        with PLOT_LOCK:
            caption = _pop_figure_title(page_fig, clear=False)
            description = _plot_description(page_fig, scheme, selected, caption, params,
                                            view.get("metrics", ()))
        # Mount the caption before the image; never leave an empty placeholder
        # while a new figure is rendered and sent to the browser.
        if caption:
            st.markdown(
                f"<div class='g-plot-caption' title='{escape(caption, quote=True)}'>"
                f"{caption_html(caption)}</div>", unsafe_allow_html=True)
        tools = st.container(horizontal=True, horizontal_alignment="right",
                             vertical_alignment="center", gap="small",
                             key="gabes_plottools")
    with tools:
        focus_key = skey(scheme.name, f"_resonance_zoom_{selected}")
        window_key = f"{focus_key}_window"
        with PLOT_LOCK:
            candidate = resonance_window(page_fig, scheme.name, raw, view)
        focus_unavailable = candidate is None and st.session_state.get(focus_key, False)
        if candidate is None:
            st.session_state[focus_key] = False
        focus_on = st.toggle(
            "Resonance zoom", key=focus_key, disabled=candidate is None,
            help="Focus the existing samples and scale the visible Y range. "
                 "No physics recalculation or extra sampling. Turn off for the full range.")
        window = None
        if focus_on:
            # Calibrated FWM curves insert a knot at the selected point. That
            # display-only change must not move the focused viewing window.
            with PLOT_LOCK:
                signature = (interaction_context if interaction_context is not None
                             else focus_signature(page_fig))
            saved = st.session_state.get(window_key)
            if saved is None or saved[0] != signature:
                saved = (signature, candidate)
                st.session_state[window_key] = saved
            window = saved[1]
        else:
            st.session_state.pop(window_key, None)
        overlay_on = bool(view.get("comparison")) and (
            view.get("experimental_workspace") or st.toggle("Overlay data", key=skey(scheme.name, "_overlay")))
        export_box = st.popover("Export", icon=":material/download:", type="tertiary")

    with card:
        plot_area = st.container()
        measurement_panel = view.get("experimental_workspace")
        if measurement_panel and skey(scheme.name, "_csv_retained_file") not in st.session_state:
            overlay_on = False
        if overlay_on:
            label = ("Experimental CSV · calibrate and fit" if measurement_panel
                     else "Overlay settings · upload, calibrate and fit")
            panel = st.expander(label, expanded=True)
            comparison_payload = _render_experimental_comparison(view, scheme.name, panel, params)
            if view.get("experimental_workspace"):
                _save_overlay_preferences(scheme.name)
        else:
            comparison_payload = None

    with PLOT_LOCK:
        plot_payloads = ui_export.collect_view_plot_payloads(view, fig)
        _pop_figure_title(page_fig)
    with plot_area:
        # Keep this row mounted in both modes so toggling zoom does not move the
        # plot, associated controls or side-by-side diagram.
        focus_note = "\u00a0"
        if focus_unavailable:
            focus_note = "No focusable sampled resonance in this view; showing the full range."
        elif window is not None:
            focus_note = f"{window.description} · Original samples; zoom does not improve resolution."
        with st.container(key="gabes_plotfocus_note"):
            # Native pyplot images in Streamlit 1.54 use the array index as alt.
            # The browser binds this description without replacing the zoomable image.
            st.caption(f'<span data-spectrum-alt="{escape(description, quote=True)}">'
                       f'{escape(focus_note)}</span>', unsafe_allow_html=True)
        for control_name in view.get("figure_controls", []):
            control_spec = spec_by_name.get(control_name)
            if control_spec is not None:
                render_param(st, scheme.name, control_spec, scheme)
        from gabes_ui.liveplot import render_live_plot
        with st.container(key="gabes_plotstage"), PLOT_LOCK, focused_axes(page_fig, window):
            live = False
            if not (scheme.name == "fwm" and raw and raw.get("kind") == "biphoton"):
                live = render_live_plot(page_fig, scheme.name, comparison_payload,
                                        canonical_noise=bool(raw and "T_canonical_small_signal" in raw),
                                        gain_digits=6 if (raw or {}).get("claim_gate", {}).get("quantitative_gain_supported") else 3,
                                        interaction_context=interaction_context, description=description)
            if live:
                _close_fig(page_fig)
            else:
                render_fig(page_fig)
    for _label, other in pages:
        if other is not page_fig:
            _close_fig(other)
    for other in hidden_pages:
        if other is not page_fig:
            _close_fig(other)
    return comparison_payload, plot_payloads, export_box


def _render_plot_and_export(view, scheme, spec_by_name, raw, params,
                            cache_version, readout_cache_version):
    comparison, plots, export_box = render_plot_card(view, scheme, spec_by_name, raw, params)
    if export_box is not None:
        ui_export.render_export(
            export_box, scheme=scheme, raw=raw, params=params, view=view,
            plot_payloads=plots, comparison_payload=comparison,
            cache_version=cache_version, readout_cache_version=readout_cache_version,
        )


@st.fragment
def _render_comparison_fragment(view, scheme, spec_by_name, raw, params,
                                 cache_version, readout_cache_version):
    # Fragment reruns reuse their input objects. Clone only the plot figures so
    # every rerun starts with clean theory, without accumulated CSV lines/titles.
    with PLOT_LOCK:
        local_view = dict(view)
        local_view.update(copy.deepcopy({
            "figure": view.get("figure"),
            "figure_views": view.get("figure_views", []),
        }))
        try:
            _render_plot_and_export(
                local_view, scheme, spec_by_name, raw, params,
                cache_version, readout_cache_version,
            )
        finally:
            # Closed figures retain their data and can still seed the next clone.
            for item in (view, local_view):
                for figure in [item.get("figure"), *(
                        page.get("figure") for page in item.get("figure_views", []))]:
                    if figure is not None:
                        _close_fig(figure)


def render_plot_section(view, scheme, spec_by_name, raw, params,
                        cache_version, readout_cache_version):
    """Keep CSV and export reruns local; physics controls retain full reruns."""
    renderer = (_render_comparison_fragment
                if view.get("comparison") and not view.get("figure_controls")
                else _render_plot_and_export)
    renderer(view, scheme, spec_by_name, raw, params,
             cache_version, readout_cache_version)


def _extra_export_metadata(key, data, compute_params, figure):
    """Export the completed run's own branch and renderer provenance."""
    metadata = {"name": key, "compute_params": compute_params, "status": "complete"}
    for attr, label in (
        ("_gabes_measured_gain_reference", "measured_gain_reference"),
        ("_gabes_experimental_gain_calibration", "experimental_gain_calibration"),
        ("_gabes_gain_calibration_model", "gain_calibration_model"),
        ("_gabes_experimental_jasperse_readout", "experimental_jasperse_readout"),
    ):
        if hasattr(figure, attr):
            metadata[label] = getattr(figure, attr)
    branches = {name: data[name] for name in ("minus", "plus") if name in data}
    if branches:
        metadata["branches"] = {name: {k: raw[k] for k in ("branch", "validation_level", "claim_gate")
                                       if k in raw} for name, raw in branches.items()}
    return metadata


@st.fragment
def render_on_demand(scheme, params, cache_version, readout_cache_version):
    """Render retained manual runs without rerunning the spectrum workspace."""
    render_more({}, scheme, params, cache_version, readout_cache_version, remember_extra=True)


def _render_extra_view(obj, scheme, params, cache_version, readout_cache_version, *, remember):
    st.caption(obj.description)
    extra_keys = set(scheme.recompute_keys()) | set(obj.param_keys)
    extra_items = tuple(sorted((key, params[key]) for key in extra_keys if key in params))
    identity = (cache_version, extra_items)
    button_key = f"run__{scheme.name}__{obj.key}"
    result_key = f"{button_key}__result"
    saved = st.session_state.get(result_key) if remember else None
    unavailable = getattr(scheme, "extra_view_unavailable", None)
    reason = unavailable(obj.key, params) if unavailable is not None else None
    if reason:
        st.caption(reason)
    if saved is not None and (saved[0] != identity or reason):
        st.caption("Conditions changed. Run again to see the reference for the current settings.")
        saved = None
    if st.button("Run", key=button_key, icon=":material/play_arrow:", disabled=bool(reason)):
        with st.spinner("Running…"):
            try:
                data = _cached_extra(scheme.name, obj.key, extra_items, cache_version,
                                     retry_unresolved=remember)
            except _UnresolvedExtraView as exc:
                data = exc.data
        saved = (identity, data)
        if remember:
            st.session_state[result_key] = saved
    if saved is None:
        return

    data = saved[1]
    unresolved = remember and data.get("status") == "unresolved"
    if remember:
        if unresolved:
            st.warning(data.get("reason") or "Numerical refinement did not converge.")
        elif data.get("status") == "converged":
            validation = data.get("provenance", {}).get("validation")
            st.caption(f"Converged · {validation}" if validation else "Converged")
        for note in data.get("notes", ()):
            st.caption(note)

    figure = None
    plots = []
    with PLOT_LOCK:
        if not unresolved:
            figure = obj.render(data, params) if obj.render_with_params else obj.render(data)
            plots = ui_export.collect_view_plot_payloads({}, figure)
            for plot in plots:
                plot.update(label=obj.key, source="extra_view")
            render_fig(figure)
        metadata = _extra_export_metadata(obj.key, data, dict(extra_items), figure)
    if remember:
        metadata.update({key: data[key] for key in ("status", "provenance", "refinement")
                         if key in data})
    ui_export.render_export(
        st.popover("Export extra view", icon=":material/download:", type="tertiary"),
        scheme=scheme, raw=data, params=params, view={}, plot_payloads=plots,
        comparison_payload=None, cache_version=cache_version,
        readout_cache_version=readout_cache_version, extra_view=metadata,
        key_suffix="extra-view")


def render_more(view, scheme, params, cache_version, readout_cache_version=None, *, remember_extra=False):
    """Tables, diagnostic figures and heavy on-demand views behind pills."""
    items = [(table["title"], "table", table) for table in view.get("tables", [])]
    items += [(f"Diagnostic · {title}", "figure", extra_fig)
              for title, extra_fig in view.get("figures", [])]
    items += [(view_def.key, "extra", view_def) for view_def in scheme.extra_views()]
    if not items:
        return
    labels = [label for label, _, _ in items]
    more_key = skey(scheme.name, "_more")
    if st.session_state.get(more_key) not in labels:
        st.session_state[more_key] = None
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key="gabes_more"):
        st.markdown("<span class='g-more-label'>More</span>",
                    unsafe_allow_html=True, width="content")
        picked = st.pills("More", labels, key=more_key, label_visibility="collapsed")
    for label, kind, obj in items:
        if kind == "figure" and label != picked:
            _close_fig(obj)
    if picked is None:
        return
    _label, kind, obj = items[labels.index(picked)]
    with st.container(border=True, key="gabes_morepanel"):
        if kind == "table":
            if scheme.name == "fwm" and obj["title"] == "Model diagnostics":
                from gabes_ui.diagnostics import render_diagnostics
                render_diagnostics(obj["markdown"])
            else:
                st.markdown(obj["markdown"])
        elif kind == "figure":
            render_fig(obj)
        else:
            _render_extra_view(obj, scheme, params, cache_version, readout_cache_version,
                               remember=remember_extra)
