"""CSV controls in the OD/SAS spectrum and the AutoOD-style precise-fit editor."""
from dataclasses import asdict, replace
import hashlib
import inspect

import numpy as np
import streamlit as st

from gabes.experimental_alignment import AxisAlignment, frame_alignment
from gabes.experimental_csv import (CALIBRATION_ABSOLUTE_DARK_REFERENCE,
                                    CALIBRATION_ABSOLUTE_GAIN_OFFSET)
from gabes_ui.controls import skey

_FILE = skey("sas", "_csv_retained_file")
_SCOPE = skey("sas", "_csv_retained_scope")
_NOISE = skey("sas", "_csv_auto_correct")
_REVERSE = skey("sas", "_csv_reverse")


def has_csv():
    return _FILE in st.session_state or _SCOPE in st.session_state


def current_upload():
    selected = st.session_state.get(_FILE)
    if selected and isinstance(selected[1], bytes):
        # Sessions opened before the unified workspace kept native CSV bytes.
        install_csv(*selected)
        selected = st.session_state.get(_FILE)
    return selected[1] if selected else None


def _dismiss_precise():
    from gabes_ui.plotcard import cancel_od_fit

    cancel_od_fit()
    if st.session_state.pop("sas_precise_open", None):
        _clear_precise_draft()
    st.session_state.pop("sas_od_pending_fit_conditions", None)
    st.session_state.pop("sas_profile_result", None)
    st.session_state.get("sas_profile_options", {}).pop("frame", None)


def close_csv():
    _dismiss_precise()
    for key in list(st.session_state):
        if key.startswith(("sas_profile_", "sas_csv_", "sas_od_fit_", "sas___csv_")) or key in (
                "sas_od_physical_fit", "sas_od_apply_notice", "sas_od_export_context", "sas_channel_open"):
            st.session_state.pop(key)


def install_csv(filename, content):
    """Replace a file only after its format and any channel ambiguity have been identified."""
    from gabes_ui import od_measurement as inputs

    parsed = inputs._project(content)
    close_csv()
    unresolved = isinstance(parsed, inputs._UnresolvedScope)
    if unresolved or parsed.scope is not None:
        st.session_state[_SCOPE] = filename, content
    if not unresolved:
        st.session_state[_FILE] = filename, parsed


@st.dialog("Open CSV")
def _open_dialog():
    from gabes.od_measurement import MAX_COMPARISON_BYTES

    options = {}
    if "max_upload_size" in inspect.signature(st.file_uploader).parameters:
        options["max_upload_size"] = MAX_COMPARISON_BYTES // (1024 * 1024)
    uploaded = st.file_uploader("Measurement CSV", type=("csv",), key="sas_fit_file_picker", **options)
    if uploaded is not None:
        try:
            install_csv(uploaded.name, uploaded.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.rerun()


def _dismiss_channel():
    st.session_state.pop("sas_channel_open", None)


@st.dialog("OD channel", on_dismiss=_dismiss_channel)
def _channel_dialog():
    from gabes_ui import od_measurement as inputs

    if not st.session_state.get("sas_channel_open"):
        st.rerun()
    filename, content = st.session_state[_SCOPE]
    detected = inputs._project(content)
    projected = inputs._select_scope_channel(st, "sas", detected, content, current_upload())
    def confirm():
        scope, labels, automatic = inputs._scope_options(detected)
        choice = st.session_state.get(skey("sas", "_csv_scope_" + scope.source_id))
        try:
            selected = inputs._scope_projection(content, detected, scope, labels, choice, automatic)
        except ValueError:
            return  # The picker reports the failed choice on this rerun.
        close_csv()
        st.session_state[_FILE] = filename, selected
        st.session_state[_SCOPE] = filename, content

    st.button("Use channel", key="sas_confirm_od_channel", disabled=projected is None, on_click=confirm)


def _interpretation(upload):
    from gabes_ui.od_measurement import recorded_model_id

    # Raw recordings always start as relative detector signals on an arbitrary axis.
    # A comparison worksheet declares the native pair used to reproduce a saved measurement.
    if recorded_model_id(upload) or upload.metadata.get("autood_version"):
        return upload.mode, upload.x_unit
    return "relative", "arbitrary"


def _set_alignment(alignment, *, invert=False):
    st.session_state.sas_csv_alignment = alignment
    st.session_state.pop("sas_csv_fast_result", None)
    st.session_state.pop("sas_od_physical_fit", None)


def _clear_precise_draft():
    from gabes_ui.od_fit_alignment import clear_draft
    from gabes_ui.od_profile_fit import clear_spectrum_draft

    clear_draft()
    clear_spectrum_draft()
    st.session_state.pop("sas_precise_base", None)
    st.session_state.pop("sas_precise_display_origin_ghz", None)


def _trace_alignment_editor(panel, trace, window, indices, tx, ty, initial):
    from gabes.od_measurement import quick_profile_alignment
    from gabes_ui.od_fit_alignment import draft_alignment, render_alignment_controls, set_draft_alignment

    sliders = panel.container()
    actions = panel.container(horizontal=True, gap="small")
    draft = draft_alignment(initial)
    if trace.is_absolute_transmission:
        draft = replace(draft, signal_scale=1., signal_offset=0.)
    if actions.button("Fast fitting", key="sas_profile_frame", disabled=ty is None or indices[1] - indices[0] < 11,
                      help="Match the selected measurement to the model in this spectral region."):
        try:
            result = quick_profile_alignment(trace, tx, ty, indices=indices, window=window, current=draft)
        except ValueError as exc:
            panel.warning(str(exc))
        else:
            set_draft_alignment(result.alignment)
            if not result.matched:
                panel.info("No clear shape match. The preview uses the selected frequency span.")
    actions.button("Use current alignment", key="sas_profile_unframe", disabled=draft_alignment(initial) == initial,
                   help="Restore the alignment applied to the main spectrum.",
                   on_click=set_draft_alignment, args=(initial,))
    visible = np.asarray(ty)[(tx >= window[0]) & (tx <= window[1])] if ty is not None else np.empty(0)
    return render_alignment_controls(sliders, initial, span_x=window[1] - window[0],
                                     span_y=float(np.ptp(visible)) if visible.size else 1.,
                                     absolute=trace.is_absolute_transmission)


@st.dialog("Precise fitting", width="large", on_dismiss=_dismiss_precise)
def _precise_dialog(upload, params, trace, alignment, source, context, tx, ty, origin_issue):
    from gabes_ui.od_fit_alignment import draft_alignment, render_detector_calibration, set_draft_alignment
    from gabes_ui.od_measurement import native_reference_shift
    from gabes_ui.od_profile_fit import render_profile_fit
    from gabes_ui.plotcard import _dialog_fit_progress, _render_physical_od_fit

    state = st.session_state
    if not state.get("sas_precise_open"):
        st.rerun()
    base = (source, params, alignment, context[-1])
    if state.get("sas_precise_base") != base:
        _clear_precise_draft()
        state.sas_precise_base = base

    with st.container(key="sas_precise_layout"):
        preview, sidebar = st.columns([1.4, 1], gap="medium")
        left = preview.container(key="sas_precise_preview")
        right = sidebar.container(key="sas_precise_controls")
        controls = right.container()
        detector_box = right.container()
    mode, unit = _interpretation(upload)
    working_trace, detector = render_detector_calibration(detector_box, upload, trace, mode=mode,
                                                         unit=unit, denoise=state[_NOISE])
    detector_issue = None
    if working_trace is None:
        detector_issue = "The preview uses the applied detector calibration until the new values are valid."
        working_trace = trace
        detector = state.get("sas_csv_detector", {})
    working_context = (*context[:-1], tuple(sorted(detector.items())))

    def commit(value, *, params, close=False):
        shift, _ = native_reference_shift(upload, params) if unit != "arbitrary" else (0., None)
        # The new alignment already uses the selected line's display origin.
        state.sas_csv_source = source, unit, shift
        for name in ("line", "temp_c", "cell_mm", "probe_power_uw", "probe_waist_mm", "rb85_fraction"):
            if params[name] != base[1][name]:
                state[skey("sas", name)] = params[name]
        state.sas_csv_detector = detector
        state.sas_csv_pending_reverse = value.reverse
        _set_alignment(value)
        if close:
            _dismiss_precise()
        st.rerun(scope="app")

    def apply_profile(value, *, params, invert=False):
        commit(value, params=params, close=True)

    def render_alignment(panel, measured, window, indices, preview_x, preview_y):
        preview_line = state.get("sas_precise_base_line", params["line"])
        preview_shift, _ = (native_reference_shift(upload, {**params, "line": preview_line})
                            if unit != "arbitrary" else (0., None))
        previous_shift = state.get("sas_precise_display_origin_ghz", context[4])
        if preview_shift != previous_shift:
            draft = draft_alignment(alignment)
            set_draft_alignment(replace(draft, shift=draft.shift + preview_shift - previous_shift))
        state.sas_precise_display_origin_ghz = preview_shift
        initial = replace(alignment, shift=alignment.shift + preview_shift - context[4])
        return _trace_alignment_editor(panel, measured, window, indices, preview_x, preview_y, initial)

    try:
        draft, preview_params, valid = render_profile_fit(
            left, upload, params, working_trace, draft_alignment(alignment), False, source, working_context, tx, ty,
            apply_profile, _dialog_fit_progress, controls=controls, render_alignment=render_alignment,
            origin_issue=origin_issue, detector_issue=detector_issue)
        updated = preview_params != params
        label = "Apply alignment and conditions" if updated else "Apply alignment"
        if right.button(label, key=skey("sas", "_csv_apply"), disabled=not valid):
            commit(draft, params=preview_params)
        pending = state.pop("sas_od_pending_fit_conditions", None)
        applied = valid and not updated and draft == alignment and working_context == context
        if working_trace.is_absolute_transmission and applied:
            _render_physical_od_fit(left, trace, alignment, params, context, upload=upload,
                                    pending_conditions=pending, render_progress=_dialog_fit_progress)
        else:
            from gabes_ui.od_fit_job import SESSION_KEY as OD_FIT_JOB
            if job := state.pop(OD_FIT_JOB, None):
                job.cancel()
            if working_trace.is_absolute_transmission and valid:
                left.caption("Apply the current alignment and conditions before fitting calibrated OD parameters.")
    except ValueError as exc:
        left.warning(str(exc))
    st.button("Close", key="sas_close_precise", on_click=_dismiss_precise)


def _reset_alignment(initial):
    _set_alignment(initial)
    st.session_state[_REVERSE] = False
    st.session_state.pop("sas_profile_options", None)
    st.session_state.pop("sas_profile_result", None)


def _toggle_precise():
    st.session_state.sas_precise_open = True


def _file_controls(panel):
    row = panel.container(horizontal=True, gap="small")
    if row.button("Open CSV...", key="sas_open_csv"):
        _open_dialog()
    if _SCOPE in st.session_state:
        if row.button("OD Channel...", key="sas_od_channel"):
            st.session_state.sas_channel_open = True
        if st.session_state.get("sas_channel_open"):
            _channel_dialog()
    if has_csv() and row.button("Close CSV", key="sas_close_csv"):
        close_csv()
        st.rerun()


def render_fit_panel(panel, *, params=None, view=None):
    """Render the right column and return its aligned measurement for the shared plots."""
    panel.markdown("**Fit OD Data**")
    _file_controls(panel)
    upload = current_upload()
    if upload is None:
        if has_csv():
            panel.info("Choose the channel that measures the transmitted probe.")
        return None
    from gabes import od_measurement as measurement
    from gabes_ui import od_measurement as inputs

    filename = st.session_state[_FILE][0]
    panel.caption(filename)
    mode, unit = _interpretation(upload)
    flag = str(upload.metadata.get("experiment_denoised", True)).lower() not in ("0", "false")
    check = panel.container(horizontal=True, gap="small")
    noise = check.checkbox("Automatic noise correction", value=flag, key=_NOISE)
    if "sas_csv_pending_reverse" in st.session_state:
        st.session_state[_REVERSE] = st.session_state.pop("sas_csv_pending_reverse")
    reverse = check.checkbox("Invert X-axis", key=_REVERSE)
    try:
        trace = inputs.load_trace(upload, mode=mode, x_unit=unit, denoise=noise,
                                  **st.session_state.get("sas_csv_detector", {}))
    except ValueError as exc:
        panel.error(str(exc))
        return None
    readout = (view or {}).get("optical_readout", {})
    if not readout:
        return None
    generic = readout["x_unit"] == "MHz"
    tx = np.asarray(readout["x"], dtype=float) * (0.001 if generic else 1.)
    ty = np.asarray(readout["transmission"], dtype=float)
    scope = upload.scope
    acquisition = (scope.od, scope.start, scope.stop, scope.manual) if scope else None
    source = upload.sha256, upload.columns, acquisition
    incoming_shift, issue = inputs.native_reference_shift(upload, params) if unit != "arbitrary" else (0., None)
    try:
        initial = (AxisAlignment(shift=incoming_shift) if unit != "arbitrary" else
                   frame_alignment(trace.detuning, tx))
    except ValueError as exc:
        panel.warning(str(exc))
        return None
    state = st.session_state
    identity = source, unit, incoming_shift
    if state.get("sas_csv_source") != identity:
        state.sas_csv_source = identity
        _set_alignment(initial)
    alignment = replace(state.sas_csv_alignment, reverse=reverse)
    context = source, mode, unit, noise, incoming_shift, tuple(sorted(state.get("sas_csv_detector", {}).items()))
    model_key = hashlib.sha256(np.ascontiguousarray([tx, ty]).tobytes()).digest()
    if state.get("sas_csv_fit_context") != (context, model_key, reverse):
        state.sas_csv_fit_context = context, model_key, reverse
        state.pop("sas_csv_fast_result", None)
    actions = panel.container(horizontal=True, gap="small")
    if actions.button("Fast fitting", key="sas_fast_fit"):
        try:
            result = measurement.quick_measurement_alignment(trace, tx, ty, current=alignment, reverse=reverse)
        except ValueError as exc:
            panel.warning(str(exc))
        else:
            _set_alignment(result.alignment)
            alignment = result.alignment
            state.sas_csv_fast_result = result
    actions.button("Precise fitting", key="sas_open_precise", on_click=_toggle_precise, disabled=generic,
                   help="Choose a resolved Rb or Cs line for precise OD fitting." if generic else None)
    actions.button("Back to initial", key="sas_back_to_initial", on_click=_reset_alignment, args=(initial,))
    if state.get("sas_precise_open") and not generic:
        _precise_dialog(upload, params, trace, alignment, source, context, tx, ty, issue)
        # Dialog callbacks may have published a physical fit or a new display alignment.
    result = state.get("sas_csv_fast_result")
    if result is not None:
        if not result.matched:
            panel.caption("No clear shape match. Showing the measured span in the model frequency range.")
        elif result.candidates > 1:
            panel.caption(f"{result.candidates} similar matches. Showing the one nearest the previous alignment.")
    if issue:
        panel.warning(issue)
    try:
        x, y = alignment.transform_x(trace.detuning), alignment.transform_y(trace.transmission)
    except ValueError as exc:
        panel.error(str(exc))
        return None
    parts = measurement.gap_slices(trace)
    coverage = sorted(tuple(sorted((float(x[s.start]), float(x[s.stop - 1])))) for s in parts)
    model = inputs.plot_metadata(params, view.get("od_model_id"))
    if origin := measurement.native_frequency_reference(upload.metadata):
        model["experiment_frequency_reference"] = origin
    theory = {"detuning": tx.tolist(), "transmission": ty.tolist(), "n_eff": readout.get("n_eff")}
    for quantity in ("optical_depth", "od0"):
        if readout.get(quantity) is not None:
            theory[quantity] = np.asarray(readout[quantity]).tolist()
    diag = trace.correction_diagnostics
    warnings = [diag.warning] if diag.warning else []
    if trace.diagnostics.directional_merge_warning:
        warnings.append(trace.diagnostics.directional_merge_warning)
    for warning in warnings:
        panel.warning(warning)
    saved = state.get("sas_od_physical_fit")
    if saved:
        # The applied estimates can change exactly the conditions that were free.
        # Reject prior results when the file, calibration, fixed conditions or alignment changes.
        previous = saved[0]
        fixed = inputs.fit_conditions_key(params, previous[6])
        if previous[:5] != (context, fixed, alignment.scale, alignment.shift, alignment.reverse):
            state.pop("sas_od_physical_fit")
            saved = None
    if saved:
        model["fit_model_id"] = saved[1]["probe_model"]
    state.sas_od_export_context = (trace, tx, ty, alignment, mode, unit)
    return {
        "enabled": True, "kind": "experimental_csv_overlay", "measurement_workspace": True,
        "overlay_filename": filename, "source_fingerprint_sha256": upload.sha256,
        "coverage_ghz": coverage, "calibration_mode": diag.calibration_mode,
        "calibration_inputs": {"auto_correct": noise, "detector_gain": diag.calibration_gain,
                               "detector_offset": diag.calibration_offset,
                               "dark_signal": diag.calibration_offset if diag.calibration_mode == CALIBRATION_ABSOLUTE_DARK_REFERENCE else None,
                               "reference_signal": (diag.calibration_offset + diag.calibration_gain
                                                    if diag.calibration_mode == CALIBRATION_ABSOLUTE_DARK_REFERENCE else None)},
        "alignment": {**asdict(alignment), "invert": False}, "model": model,
        "alignment_fit": (None if result is None else {"success": result.matched,
                          "candidates": result.candidates, "physical_parameters_fitted": False}),
        "input_metadata": upload.metadata, "theory_trace": theory, "warnings": warnings,
        "comparison_trace": {"detuning": x.tolist(), "transmission": y.tolist(), "detuning_unit": "GHz",
                             "transmission_unit": "1", "transmission_label": trace.transmission_label},
        **({"physical_od_fit": saved[1]} if saved else {}),
    }


def render_comparison_export(panel, comparison):
    if not comparison or not comparison["model"].get("axis_reference"):
        return
    fmt = panel.selectbox("Comparison CSV format", ["Standard", "Origin"], key="sas_measurement_export_format")
    if panel.button("Prepare comparison CSV", key="sas_prepare_measurement_csv"):
        from gabes.od_measurement import export_comparison_csv

        trace, tx, ty, alignment, mode, unit = st.session_state.sas_od_export_context
        try:
            data = export_comparison_csv(current_upload(), trace, tx, ty, alignment=alignment,
                    metadata=comparison["model"], origin=fmt == "Origin", mode=mode, x_unit=unit)
        except ValueError as exc:
            panel.error(str(exc))
        else:
            panel.download_button("Download comparison CSV", data=data, file_name="od-comparison.csv",
                                  mime="text/csv", key="sas_download_measurement_csv", on_click="ignore")
