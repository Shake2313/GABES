"""Joint frequency alignment and optional pump-free OD conditions for measured spectra."""
from concurrent.futures import CancelledError

import numpy as np
import streamlit as st

from gabes_ui.controls import current_params, skey
from gabes_ui.od_fit_job import ODFitJob, PROFILE_JOB
from gabes_ui.od_measurement import fit_conditions_key, workspace_params


_PREFIX = "sas_profile_"
_TOLERANCES = {"cell_mm": 1., "temp_c": 5., "probe_power_uw": 10., "rb85_fraction": 0.05}
_GROUP_LABELS = {"full": "Full spectrum", "85-F2": "⁸⁵Rb F=2", "85-F3": "⁸⁵Rb F=3",
                 "87-F1": "⁸⁷Rb F=1", "87-F2": "⁸⁷Rb F=2"}


def _setup(source):
    """Save dialog controls separately from Streamlit's transient widget state."""
    state = st.session_state
    if state.get("sas_profile_measurement") != source:
        for key in list(state):
            if key.startswith(_PREFIX):
                del state[key]
        state.sas_profile_measurement = source
    return state.setdefault("sas_profile_options", {})


def _rows(x, y, label, window, segments=None):
    """Keep local extrema visible when a scope record exceeds the chart width."""
    visible = np.flatnonzero((x >= window[0]) & (x <= window[1]))
    if len(visible) > 1600:
        chunks = np.array_split(visible, 800)
        indices = {visible[0], visible[-1]}
        for chunk in chunks:
            indices.update((chunk[np.argmin(y[chunk])], chunk[np.argmax(y[chunk])]))
        visible = sorted(indices)
    return [{"frequency": float(x[i]), "signal": float(y[i]), "trace": label,
             "segment": int(segments[i]) if segments is not None else 0} for i in visible]


def _apply_brush(chart_key, display_x):
    # Streamlit invokes chart callbacks before mounting the range slider.
    selected = st.session_state.get(chart_key, {}).get("selection", {}).get("samples", {})
    endpoints = selected.get("frequency")
    if endpoints is None or len(endpoints) != 2:
        return
    lo, hi = sorted(endpoints)
    inside = np.flatnonzero((display_x >= lo) & (display_x <= hi))
    if inside.size >= 2:
        st.session_state.sas_profile_roi = (int(inside[0]), int(inside[-1]))


def _chart(box, trace, alignment, invert, window, indices, theory_x, theory_y, result):
    if result is not None:
        alignment = result.alignment
        theory_x, theory_y = result.preview_frequency_ghz, result.preview_transmission
    x = alignment.transform_x(trace.detuning)
    y = alignment.transform_y(trace.transmission, invert=invert)
    from gabes.od_measurement import gap_slices
    segments = np.zeros(len(x), dtype=int)
    for index, section in enumerate(gap_slices(trace)):
        segments[section] = index
    values = _rows(x, y, "Measurement", window, segments)
    lo, hi = indices
    picked = slice(lo, hi + 1)
    values += _rows(x[picked], y[picked], "Selected samples", window, segments[picked])
    values += _rows(np.asarray(theory_x), np.asarray(theory_y), "OD model", window)
    if not values:
        box.caption("No samples fall within this spectral region.")
        return
    amplitudes = [row["signal"] for row in values]
    bottom, top = min(amplitudes), max(amplitudes)
    padding = max(0.03 * (top - bottom), 1e-6)
    key = "sas_profile_graph"
    # Streamlit's brush callback supports single-view Vega-Lite charts.
    box.vega_lite_chart(data=values, spec={
        "height": 300, "autosize": {"type": "fit-x", "contains": "padding"},
        "params": [{"name": "samples", "select": {"type": "interval", "encodings": ["x"]}}],
        "mark": {"type": "line", "clip": True}, "encoding": {
            "x": {"field": "frequency", "type": "quantitative", "title": "Detuning (GHz)",
                  "scale": {"domain": list(window), "nice": False}},
            "y": {"field": "signal", "type": "quantitative",
                  "title": "Transmission" if trace.is_absolute_transmission else "Relative signal",
                  "scale": {"domain": [bottom - padding, top + padding], "zero": False}},
            "detail": {"field": "segment", "type": "nominal"},
            "color": {"field": "trace", "type": "nominal", "title": None,
                      "scale": {"domain": ["OD model", "Measurement", "Selected samples"],
                                "range": ["#286f75", "#ad939b", "#a44360"]},
                      "legend": {"orient": "bottom"}},
        }}, key=key, use_container_width=True, on_select=lambda: _apply_brush(key, x),
        selection_mode="samples")


def _fit_bounds(box, params, model_id, settings):
    from gabes.od_profile_fit import profile_fit_parameters

    offered = profile_fit_parameters(params, model_id=model_id)
    state = st.session_state
    previous = state.get("sas_profile_vary", settings.get("vary", ()))
    state.sas_profile_vary = [name for name in previous if name in offered]
    vary = box.multiselect("Vary model conditions", tuple(offered), key="sas_profile_vary",
                           format_func=lambda name: offered[name][0],
                           help="Leave empty to refine only the frequency axis and, for relative signals, detector scaling.")
    settings["vary"] = tuple(vary)
    bounds = {}
    for name in vary:
        label, unit, lower, upper = offered[name]
        value = float(params[name])
        if not lower <= value <= upper:
            box.warning(f"{label} must be between {lower:g} and {upper:g} {unit} to vary it.")
            return None
        widget = _PREFIX + "delta_" + name
        default = settings.get(widget, _TOLERANCES[name])
        if widget not in state:
            state[widget] = float(default)
        delta = box.number_input(f"{label} ±" + (f" ({unit})" if unit else ""), key=widget,
                                 min_value=0., max_value=float(upper - lower), format="%.5g",
                                 step=float(_TOLERANCES[name] / 10.))
        settings[widget] = delta
        if delta <= 0:
            box.warning(f"Enter a positive range for {label}.")
            return None
        bounds[name] = max(lower, value - delta), min(upper, value + delta)
        a, b = bounds[name]
        box.caption(f"{value:g} {unit} · search {a:g} to {b:g} {unit}".strip())
    return bounds


def _select_samples(box, params, trace, alignment, invert, settings, theory_range, theory_x, theory_y):
    from gabes.od_profile_fit import profile_group_windows
    from gabes.od_measurement import quick_profile_alignment

    state = st.session_state
    groups = profile_group_windows(params, theory_range)
    group_key = "sas_profile_group"
    if state.get(group_key, settings.get("group")) not in groups:
        state[group_key] = "full"
    elif group_key not in state:
        state[group_key] = settings["group"]
    group = box.selectbox("Spectral region", tuple(groups), key=group_key,
                          format_func=lambda g: _GROUP_LABELS.get(g, g))
    window = groups[group]
    count = len(trace.detuning)
    roi_key = "sas_profile_roi"
    if group != settings.get("group"):
        x = alignment.transform_x(trace.detuning)
        selected = np.flatnonzero((x >= window[0]) & (x <= window[1]))
        default = (0, count - 1)
        if group != "full" and len(selected) >= 2:
            default = int(selected[0]), int(selected[-1])
        state[roi_key] = default
    elif roi_key not in state:
        state[roi_key] = settings["indices"]
    indices = box.slider("Measured samples (sorted by X)", 0, count - 1,
                         key=roi_key,
                         help="Drag across the preview or adjust these inclusive sample indices. Changing the frequency alignment preserves this selection.")
    settings.update(group=group, indices=tuple(indices))
    roi = tuple(float(trace.detuning[i]) for i in indices)
    frame_key = group, window, tuple(indices), alignment, invert, trace.is_absolute_transmission
    frame = settings.get("frame")
    if frame and frame[0] != frame_key:
        settings.pop("frame")
        frame = None
    controls = box.columns(2)
    if controls[0].button("Fast fitting", key="sas_profile_frame", disabled=indices[1] - indices[0] < 11,
                          help="Match the selected measured samples to the model in this spectral region."):
        result = quick_profile_alignment(trace, theory_x, theory_y, indices=indices, window=window,
                                         current=alignment, invert=invert)
        frame = settings["frame"] = (frame_key, result.alignment)
        if not result.matched:
            box.info("No clear shape match. The preview uses the selected frequency span.")
    if controls[1].button("Use current alignment", key="sas_profile_unframe", disabled=frame is None):
        settings.pop("frame")
        frame = None
    return group, window, indices, roi, frame[1] if frame else alignment


def _current_fit(box, identity):
    from gabes.od_spectrum_fit import ODFitCancelled

    state = st.session_state
    result_key = "sas_profile_result"
    saved = state.get(result_key)
    result = saved[1] if saved and saved[0] == identity else None
    if saved and result is None:
        state.pop(result_key)
    job = state.get(PROFILE_JOB)
    if job and job.identity != identity:
        box.info("Profile fit cancelled because its inputs changed.")
        job.cancel()
        state.pop(PROFILE_JOB)
        job = None
    if job and job.future.done():
        state.pop(PROFILE_JOB)
        try:
            result = job.future.result()
            state[result_key] = (identity, result)
        except (CancelledError, ODFitCancelled):
            box.info("Profile fit cancelled.")
        except Exception as exc:
            box.error(f"Profile fit failed: {exc}")
        job = None
    if state.pop("sas_profile_cancelled", False):
        box.info("Profile fit cancelled.")
    return job, result


def _review_result(box, result, trace, indices, params, model_id, upload, invert, apply_alignment):
    from gabes import schemes
    from gabes.od_profile_fit import profile_fit_parameters

    state = st.session_state
    fitted_x = result.alignment.transform_x(trace.detuning)
    low, high = sorted((fitted_x[indices[0]], fitted_x[indices[1]]))
    box.write(f"Aligned detuning · **{low:+.5g} to {high:+.5g} GHz**")
    box.caption(f"RMS {result.initial_rmse:.4g} → {result.rmse:.4g} · waveform correlation {result.correlation:.4g}")
    if result.relative:
        box.caption(f"Inferred Y scale {result.alignment.signal_scale:.5g} · offset {result.alignment.signal_offset:+.5g}")
    offered = profile_fit_parameters(params, model_id=model_id)
    for name, value in result.condition_values.items():
        label, unit = offered[name][:2]
        box.write(f"{label} · {params[name]:g} → **{value:.5g} {unit}**")
    if result.correlated_pairs:
        labels = {key: spec[0] for key, spec in offered.items()}
        labels.update(axis_left_ghz="lower frequency endpoint", axis_right_ghz="upper frequency endpoint")
        for first, second, rho in result.correlated_pairs:
            box.warning(f"Correlated estimates · {labels[first]} and {labels[second]} (ρ = {rho:+.3f}).")
    for warning in result.warnings:
        box.warning(warning)
    apply, discard = box.columns(2)
    if apply.button("Apply profile fit", key="sas_profile_apply", disabled=not result.identifiable):
        live = workspace_params(current_params("sas", schemes.get("sas")), state, upload)
        if fit_conditions_key(live) != fit_conditions_key(params):
            box.info("The model changed. Preview the fit again.")
            return
        apply_alignment(result.alignment, invert=invert)
        state.update({skey("sas", name): value for name, value in result.condition_values.items()})
        state.pop("sas_profile_result")
        state.pop("sas_precise_open", None)
        state.sas_profile_notice = "Applied profile alignment and selected model conditions." if result.condition_values else "Applied profile alignment."
        st.rerun()
    if discard.button("Discard preview", key="sas_profile_discard"):
        state.pop("sas_profile_result")
        st.rerun()


def render_profile_fit(panel, upload, params, trace, alignment, invert, source, fit_context,
                       theory_x, theory_y, apply_alignment, render_progress):
    """Run only on explicit request; opening the editor never changes main controls."""
    from gabes.od_model import ODConditions, od_unavailable

    state = st.session_state
    settings = _setup(source)
    from gabes.od_profile_fit import fit_od_profile

    box = panel
    conditions = ODConditions.from_params(params)
    reason = od_unavailable(conditions)
    if reason:
        state.pop("sas_profile_result", None)
        box.info(reason)
        return
    if len(trace.detuning) < 2:
        state.pop("sas_profile_result", None)
        box.info("The measurement needs at least two distinct X values.")
        return
    if notice := state.pop("sas_profile_notice", None):
        box.success(notice)
    if trace.is_absolute_transmission:
        box.caption("Refines the frequency scale and shift. The measured transmission calibration stays fixed.")
    else:
        box.caption("Fits relative detector gain and offset to the OD model. The fitted signal scale is model-dependent.")
    theory_range = (float(np.min(theory_x)), float(np.max(theory_x)))
    group, window, indices, roi, alignment = _select_samples(
        box, params, trace, alignment, invert, settings, theory_range, theory_x, theory_y)
    bounds = _fit_bounds(box, params, conditions.model_id, settings)
    identity = (source, fit_context, fit_conditions_key(params), alignment, invert, group, roi,
                None if bounds is None else tuple(bounds.items()))
    job, result = _current_fit(box, identity)
    _chart(box, trace, alignment, invert, window, indices, theory_x, theory_y, result)
    selected_count = 1 + indices[1] - indices[0]
    box.caption(f"{selected_count:,} of {len(trace.detuning):,} measured samples selected."
                + (" Select at least 12 to fit." if selected_count < 12 else ""))
    if job is None and box.button("Preview profile fit", key="sas_profile_run",
                                  disabled=bounds is None or selected_count < 12):
        state.pop("sas_profile_result", None)
        job = ODFitJob(identity, trace, params, fit_function=fit_od_profile, alignment=alignment,
                       roi=roi, bounds=bounds, group=group, model_id=conditions.model_id,
                       theory_range_ghz=theory_range, invert=invert)
        state[PROFILE_JOB] = job
    if job is not None:
        job.visible = True
        with box:
            render_progress(job, PROFILE_JOB)
        return
    if result is None:
        return
    _review_result(box, result, trace, indices, params, conditions.model_id, upload, invert, apply_alignment)
