"""Joint frequency alignment and optional pump-free OD conditions for measured spectra."""
from concurrent.futures import CancelledError
import hashlib

import numpy as np
import streamlit as st

from gabes_ui.controls import current_params
from gabes_ui.od_fit_job import ODFitJob, PROFILE_JOB
from gabes_ui.od_measurement import fit_conditions_key, native_reference_shift, workspace_params


_PREFIX = "sas_profile_"
# Display a 1/e² diameter and percent abundance; the evaluator takes radius and fraction.
# The last column is the initial tolerance, expressed in the displayed unit.
_FIELDS = (
    ("cell_mm", "Cell length (mm)", 1., 1.),
    ("temp_c", "Cell temperature (°C)", 1., 5.),
    ("probe_power_uw", "Probe power (µW)", 1., 10.),
    ("probe_waist_mm", "Beam diameter (1/e², mm)", 2., 0.2),
    ("rb85_fraction", "⁸⁵Rb fraction (%)", 100., 5.),
)
_GROUP_LABELS = {"full": "Full spectrum", "85-F2": "⁸⁵Rb F=2", "85-F3": "⁸⁵Rb F=3",
                 "87-F1": "⁸⁷Rb F=1", "87-F2": "⁸⁷Rb F=2"}


def clear_spectrum_draft():
    for key in list(st.session_state):
        if key.startswith("sas_precise_base_"):
            st.session_state.pop(key)


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
    # Streamlit's selectable Vega widget can retain the old Arrow table when
    # its replacement has the same row count. Remount on changed curve data;
    # a brush gesture alone keeps its widget and the native-sample ROI.
    revision = hashlib.sha256()
    for values_in_plot in (x, y, theory_x, theory_y, window):
        array = np.ascontiguousarray(values_in_plot, dtype=np.float64)
        revision.update(array.size.to_bytes(8, "little"))
        revision.update(array)
    key = "sas_profile_graph_" + revision.hexdigest()[:20]
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


def _condition_rows(box, params, settings):
    from gabes.od_model import ODConditions
    from gabes.od_profile_fit import profile_fit_parameters
    from gabes.od_spectrum_fit import PARAMETERS, CANDIDATE_PARAMETERS
    from gabes.schemes.sas import RB_COMPOSITION_CUSTOM, _isotope_composition

    state = st.session_state
    _, _, composition, fraction, _ = _isotope_composition(params)
    custom = composition == RB_COMPOSITION_CUSTOM
    fixed_fraction = None if custom else fraction
    preview = dict(params)
    columns = (2.25, 1, 1)
    headers = box.columns(columns, gap="small")
    for slot, title in zip(headers, ("Fit condition", "Base", "±")):
        slot.caption(title)
    # Resolve the same model ID that will evaluate the preview and fit. Widget
    # changes are already present in session state when this script reruns.
    for name, _, factor, _ in _FIELDS:
        if name != "rb85_fraction" or custom:
            value = state.get("sas_precise_base_" + name, params[name] * factor)
            if value != params[name] * factor:
                preview[name] = value / factor

    try:
        model_id = ODConditions.from_params(preview).model_id
        offered = profile_fit_parameters(preview, model_id=model_id)
    except ValueError:
        offered = {}  # Report the actual evaluator error next to the preview.

    vary, bounds = [], {}
    for name, label, factor, default_delta in _FIELDS:
        if name == "rb85_fraction" and fraction is None:
            continue
        field, base, tolerance = box.columns(columns, gap="small")
        fit_key = "sas_profile_vary_" + name
        eligible = name in offered
        if not eligible or fit_key not in state:
            state[fit_key] = eligible and name in settings.get("vary", ())
        help_text = None
        if not eligible:
            if name == "probe_waist_mm":
                help_text = "Fixed during fitting. Probe intensity depends on power divided by beam radius squared."
            elif name == "probe_power_uw":
                help_text = "Power estimation requires the finite-probe Rb D2 rate model."
            elif name == "rb85_fraction":
                help_text = "The isotope selection fixes this abundance. Use a custom Rb mixture to vary it."
        selected = field.checkbox(label, key=fit_key, disabled=not eligible, help=help_text)
        spec = (PARAMETERS if name in PARAMETERS else CANDIDATE_PARAMETERS)[name]
        lower, upper = spec[2] * factor, spec[3] * factor
        locked = name == "rb85_fraction" and not custom
        current = fixed_fraction if locked else params[name]
        key = "sas_precise_base_" + name
        if key not in state or locked:
            state[key] = current * factor
        base.number_input(label + " base", key=key, min_value=min(lower, state[key]),
                          max_value=max(upper, state[key]), disabled=locked,
                          step=default_delta / 10., format="%.5g", label_visibility="collapsed")
        key = _PREFIX + "delta_" + name
        if key not in state:
            state[key] = settings.get(key, default_delta / factor) * factor
        delta = tolerance.number_input(label + " tolerance", key=key, min_value=0.,
                                       max_value=upper - lower, disabled=not selected,
                                       step=default_delta / 10., format="%.5g", label_visibility="collapsed")
        settings[key] = delta / factor
        if selected:
            value = preview[name]
            vary.append(name)
            bounds[name] = max(spec[2], value - delta / factor), min(spec[3], value + delta / factor)
    settings["vary"] = tuple(vary)
    if any(lo >= hi for lo, hi in bounds.values()):
        box.warning("Enter a positive tolerance for each selected condition.")
        bounds = None
    return preview, bounds


def _select_samples(box, group_box, params, trace, alignment, settings, theory_range):
    from gabes.od_profile_fit import profile_group_windows

    state = st.session_state
    groups = profile_group_windows(params, theory_range)
    group_key = "sas_profile_group"
    if state.get(group_key, settings.get("group")) not in groups:
        state[group_key] = "full"
    elif group_key not in state:
        state[group_key] = settings["group"]
    group = group_box.selectbox("Ground-state group", tuple(groups), key=group_key,
                                format_func=lambda g: _GROUP_LABELS.get(g, g))
    window = groups[group]
    count = len(trace.detuning)
    roi_key = "sas_profile_roi"
    region_changed = group != settings.get("group") or (group != "full" and params["line"] != settings.get("line"))
    if region_changed:
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
    settings.update(group=group, indices=tuple(indices), line=params["line"])
    roi = tuple(float(trace.detuning[i]) for i in indices)
    return group, window, indices, roi


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


def _review_result(box, result, trace, indices, params, original, model_id, upload, invert, apply_alignment):
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
        if fit_conditions_key(live) != fit_conditions_key(original):
            box.info("The model changed. Preview the fit again.")
            return
        state.sas_profile_notice = "Applied profile alignment and selected model conditions." if result.condition_values else "Applied profile alignment."
        apply_alignment(result.alignment, params={**params, **result.condition_values}, invert=invert)
    if discard.button("Discard preview", key="sas_profile_discard"):
        state.pop("sas_profile_result")
        st.rerun()


@st.cache_data(max_entries=8, show_spinner=False)
def _preview_spectrum(params, cache_version):
    """Use the workspace solver's grid for the selected line, mixture, and beam."""
    from gabes import schemes
    from gabes.core import blas_single_thread
    from gabes_ui.od_measurement import evaluate_workspace

    scheme = schemes.get("sas")
    with blas_single_thread():
        raw = scheme.compute({key: params[key] for key in scheme.solve_param_keys(params)})
    evaluation = evaluate_workspace(params, raw)
    return raw["scan"] / (2. * np.pi) / 1.e9, evaluation.transmission


def render_profile_fit(panel, upload, params, trace, alignment, invert, source, fit_context,
                       theory_x, theory_y, apply_alignment, render_progress, *, controls,
                       render_alignment, origin_issue=None, detector_issue=None):
    """Preview a local set of conditions; only Apply publishes it to the workspace."""
    from gabes import schemes
    from gabes.od_model import ODConditions, od_unavailable
    from gabes.od_profile_fit import fit_od_profile

    state = st.session_state
    settings = _setup(source)
    controls.markdown("**Spectrum conditions**")
    line_box, group_box = controls.columns([1, 1.5], gap="small")
    line = line_box.selectbox("Transition", ("D1", "D2"), index=("D1", "D2").index(params["line"]),
                              key="sas_precise_base_line")
    preview, bounds = _condition_rows(controls, dict(params, line=line), settings)
    if notice := state.pop("sas_profile_notice", None):
        panel.success(notice)
    if len(trace.detuning) < 2:
        panel.info("Select at least two distinct measured X values for alignment.")
        return alignment, preview, False
    if line != params["line"] and upload.x_unit != "arbitrary":
        _, origin_issue = native_reference_shift(upload, preview)

    changed = preview != params
    reason = None
    try:
        conditions = ODConditions.from_params(preview)
        reason = od_unavailable(conditions)
        if changed and not reason:
            theory_x, theory_y = _preview_spectrum(preview, schemes.get("sas").cache_version)
    except ValueError as exc:
        reason = str(exc)

    theory_range = (float(np.min(theory_x)), float(np.max(theory_x)))
    chart_box = panel.container()
    group, window, indices, roi = _select_samples(
        panel, group_box, preview, trace, alignment, settings, theory_range)
    issue = detector_issue or origin_issue
    alignment = render_alignment(controls, trace, window, indices, theory_x,
                                 None if (changed and reason) or issue else theory_y)
    if issue or reason:
        panel.warning(issue or reason)
        if not (changed and reason):
            _chart(chart_box, trace, alignment, invert, window, indices, theory_x, theory_y, None)
        stale = state.pop(PROFILE_JOB, None)
        if stale is not None:
            stale.cancel()
        state.pop("sas_profile_result", None)
        return alignment, preview, not issue and not changed

    identity = (source, fit_context, fit_conditions_key(preview), alignment, invert, group, roi,
                None if bounds is None else tuple(bounds.items()))
    box = panel
    job, result = _current_fit(box, identity)
    _chart(chart_box, trace, alignment, invert, window, indices, theory_x, theory_y, result)
    selected_count = 1 + indices[1] - indices[0]
    box.caption(f"{selected_count:,} of {len(trace.detuning):,} measured samples selected."
                + (" Select at least 12 to fit." if selected_count < 12 else ""))
    if job is None and box.button("Preview profile fit", key="sas_profile_run",
                                  disabled=bounds is None or selected_count < 12):
        state.pop("sas_profile_result", None)
        job = ODFitJob(identity, trace, preview, fit_function=fit_od_profile, alignment=alignment,
                       roi=roi, bounds=bounds, group=group, model_id=conditions.model_id,
                       theory_range_ghz=theory_range, invert=invert)
        state[PROFILE_JOB] = job
    if job is not None:
        job.visible = True
        with box:
            render_progress(job, PROFILE_JOB)
    elif result is not None:
        _review_result(box, result, trace, indices, preview, params, conditions.model_id,
                       upload, invert, apply_alignment)
    return alignment, preview, True
