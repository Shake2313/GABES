"""CSV controls in the OD/SAS spectrum and the AutoOD-style precise-fit editor."""
from dataclasses import asdict, replace
import hashlib
import inspect

import numpy as np
import streamlit as st

from gabes.experimental_alignment import AxisAlignment, frame_alignment
from gabes.experimental_csv import (CALIBRATION_ABSOLUTE_DARK_REFERENCE,
                                    CALIBRATION_ABSOLUTE_GAIN_OFFSET, CALIBRATION_RELATIVE_EXTREMA)
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
    st.session_state.pop("sas_precise_open", None)
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


def _manual_alignment(alignment):
    keys = {name: skey("sas", key) for name, key in (
        ("scale", "_csv_x_scale"), ("shift", "_csv_x_shift"),
        ("signal_scale", "_csv_y_scale"), ("signal_offset", "_csv_y_offset"))}
    changed = st.session_state.get("sas_csv_manual_base") != alignment
    for name, key in keys.items():
        if changed or key not in st.session_state:
            st.session_state[key] = float(getattr(alignment, name))
    st.session_state.sas_csv_manual_base = alignment
    return keys


def _advanced_settings(upload, trace, alignment):
    from gabes_ui import od_measurement as inputs

    box = st.expander("Detector calibration and manual alignment")
    state = st.session_state
    if _interpretation(upload)[0] == "relative":
        options = {"Relative normalized transmission (extrema-scaled)": CALIBRATION_RELATIVE_EXTREMA,
                   "Absolute transmission (dark/reference)": CALIBRATION_ABSOLUTE_DARK_REFERENCE,
                   "Absolute transmission (gain/offset)": CALIBRATION_ABSOLUTE_GAIN_OFFSET}
        mode_key = skey("sas", "_csv_calibration_mode")
        if mode_key not in state:
            saved = state.get("sas_csv_detector")
            if saved:
                mode = saved["calibration_mode"]
                state[mode_key] = next(label for label, value in options.items() if value == mode)
                for parameter, key in (("dark_signal", "_csv_dark_signal"), ("reference_signal", "_csv_reference_signal"),
                                       ("gain", "_csv_detector_gain"), ("offset", "_csv_detector_offset")):
                    if parameter in saved:
                        state[skey("sas", key)] = saved[parameter]
            else:
                inputs.restore_detector_law("sas", upload, options)
        choice = box.selectbox("Transmission calibration", tuple(options), key=mode_key)
        mode = options[choice]
        detector = dict(calibration_mode=mode)
        if mode == CALIBRATION_ABSOLUTE_DARK_REFERENCE:
            detector["dark_signal"] = box.number_input("Dark signal", value=0., key=skey("sas", "_csv_dark_signal"))
            detector["reference_signal"] = box.number_input("Reference signal (T = 1)", value=1., key=skey("sas", "_csv_reference_signal"))
        elif mode == CALIBRATION_ABSOLUTE_GAIN_OFFSET:
            detector["gain"] = box.number_input("Detector gain", value=1., key=skey("sas", "_csv_detector_gain"))
            detector["offset"] = box.number_input("Detector offset", value=0., key=skey("sas", "_csv_detector_offset"))
        if box.button("Apply detector calibration", key="sas_apply_detector"):
            try:
                inputs.load_trace(upload, mode="relative", x_unit=_interpretation(upload)[1],
                                  denoise=state[_NOISE], **detector)
            except ValueError as exc:
                box.error(str(exc))
            else:
                state.sas_csv_detector = detector
                _set_alignment(replace(alignment, signal_scale=1., signal_offset=0.))
                st.rerun()

    keys = _manual_alignment(alignment)
    with box.form("sas_manual_alignment"):
        scale = st.number_input("X scale", key=keys["scale"], min_value=0., format="%.8g")
        shift = st.number_input("X shift (GHz)", key=keys["shift"], format="%.8g")
        gain = st.number_input("Y scale", key=keys["signal_scale"], min_value=0., format="%.8g",
                               disabled=trace.is_absolute_transmission)
        offset = st.number_input("Y offset", key=keys["signal_offset"], format="%.8g",
                                 disabled=trace.is_absolute_transmission)
        if st.form_submit_button("Apply alignment", key=skey("sas", "_csv_apply")):
            try:
                _set_alignment(replace(alignment, scale=scale, shift=shift, signal_scale=gain, signal_offset=offset))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.rerun()


@st.dialog("Precise fitting", width="large", on_dismiss=_dismiss_precise)
def _precise_dialog(upload, params, trace, alignment, source, context, tx, ty, origin_issue):
    from gabes_ui.od_profile_fit import render_profile_fit
    from gabes_ui.plotcard import _render_fit_progress, _render_physical_od_fit

    if not st.session_state.get("sas_precise_open"):
        st.rerun()
    _advanced_settings(upload, trace, alignment)
    if origin_issue:
        st.warning(origin_issue)
    else:
        try:
            render_profile_fit(st.container(), upload, params, trace, alignment, False, source, context, tx, ty,
                               _set_alignment, _render_fit_progress)
            if trace.is_absolute_transmission:
                pending = st.session_state.pop("sas_od_pending_fit_conditions", None)
                _render_physical_od_fit(st, trace, alignment, params, context, upload=upload,
                                        pending_conditions=pending)
        except ValueError as exc:
            st.warning(str(exc))
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
