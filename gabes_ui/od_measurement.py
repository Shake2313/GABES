"""File selection and fit-result ownership for the pump-free OD workspace."""
from dataclasses import asdict, dataclass

import numpy as np
import streamlit as st

from gabes.od_model import (MODEL_KEY, MODELS, ODConditions, ODUnsupportedError, display_reference_offset_hz,
                            evaluate_od_display, od_fit_conditions, od_unavailable)
from gabes.schemes.sas import TWO_LEVEL_MODELS
from gabes_ui.controls import skey


def recorded_model_id(upload):
    """Only GABES comparison envelopes declare a replayable pump-free ID."""
    metadata = upload.metadata if upload is not None else {}
    # AutoOD column values, including any user-supplied headers, are strings.
    if metadata.get("generated_by") == "GABES" and metadata.get("format_version") in (1, 2):
        return metadata["model_id"]
    return None


def workspace_params(params, state, upload=None):
    """Derive a pump-free view without changing the active preset or theory controls."""
    derived = od_fit_conditions(params)
    if MODEL_KEY not in derived and state.get(skey("sas", MODEL_KEY)) is not None:
        derived[MODEL_KEY] = state[skey("sas", MODEL_KEY)]
    model_id = recorded_model_id(upload)
    if model_id is not None:
        if model_id not in MODELS:
            raise ODUnsupportedError(f"Unknown recorded pump-free OD model ID {model_id!r}.")
        derived[MODEL_KEY] = model_id
        conditions = ODConditions.from_params(derived)
        if MODELS[model_id].status == "implemented":
            reason = od_unavailable(conditions)
            if reason:
                raise ODUnsupportedError(reason)
        elif model_id in TWO_LEVEL_MODELS and conditions.probe_power_uw == 0:
            raise ODUnsupportedError("The recorded finite-power two-level model needs positive probe power.")
    return derived


@st.cache_data(show_spinner=False, max_entries=8)
def _spectrum(conditions, scan_rad_s):
    from gabes.core import blas_single_thread
    with blas_single_thread():
        return evaluate_od_display(conditions, scan_rad_s)


def evaluate_workspace(params, raw):
    """Check explicit requests before allocating figures; permit the displayed scheme-only fallback."""
    from gabes.schemes.sas import OD_MODEL_SETS
    requested = params.get(MODEL_KEY)
    if requested is not None and (not isinstance(requested, str) or not requested.strip()):
        raise ODUnsupportedError("Invalid OD model ID. Expected a non-blank string.")
    # A model set (pre_d20, d20, d20_d2sb) chooses per regime; only an explicit ID is a request.
    explicit = None if requested in (None, *OD_MODEL_SETS) else requested
    scheme_id = raw.get("probe_model")
    try:
        conditions = ODConditions.from_params(params)
    except ODUnsupportedError:
        if explicit:
            raise
        return None
    reason = od_unavailable(conditions)
    if reason:
        if explicit and explicit != scheme_id:
            raise ODUnsupportedError(reason)
        return None
    return _spectrum(conditions, np.asarray(raw["scan"]))


def measurement_model_view(view, raw, evaluation):
    """Use the same sampled evaluator for the displayed curve, downloaded curves, and fit."""
    from gabes_ui.plotcard import PLOT_LOCK
    if evaluation is None:
        return dict(view, experimental_workspace=True, od_model_id=raw.get("probe_model")), raw

    conditions = evaluation.conditions
    changed = evaluation.model_id != raw.get("probe_model")
    old = view["optical_readout"]
    readout = ({k: old[k] for k in ("schema", "x", "x_unit", "unit", "conventions",
                                      "cell_length_m", "line_strength")} if changed else dict(old))
    readout.update(transmission=evaluation.transmission, optical_depth=evaluation.optical_depth,
                   decadic_absorbance=evaluation.optical_depth / np.log(10.), od0=evaluation.od0,
                   n_eff=old.get("n_eff"), effective=conditions.probe_power_uw > 0.,
                   model_id=evaluation.model_id)
    accuracy = MODELS[evaluation.model_id].accuracy
    if accuracy:
        readout["model_accuracy"] = accuracy
    pages = []
    with PLOT_LOCK:
        for page in view.get("figure_views", []):
            axis = page["figure"].axes[0]
            line = axis.lines[0]
            quantity = ("transmission" if page["label"] == "Transmission" else
                        getattr(line, "_gabes_export_metadata", {}).get("quantity"))
            if quantity in ("transmission", "optical_depth", "decadic_absorbance"):
                line.set_ydata(readout[quantity])
                if changed:
                    axis.relim()
                    axis.autoscale_view(scalex=False)
                pages.append(page)
            elif not changed:
                pages.append(page)
            else:
                # This evaluator returns attenuation only, with no dispersion spectrum.
                import matplotlib.pyplot as plt
                plt.close(page["figure"])
    output_view = ({key: view[key] for key in ("figure", "comparison") if key in view} if changed else dict(view))
    output_view.update(experimental_workspace=True, od_model_id=evaluation.model_id,
                       optical_readout=readout, figure_views=pages)
    output_raw = ({"scan": raw["scan"], "markers": raw.get("markers", [])} if changed else dict(raw))
    output_raw.update(probe_model=evaluation.model_id, transmission=evaluation.transmission,
                      optical_depth=evaluation.optical_depth, od0=evaluation.od0,
                      frequency_reference=asdict(evaluation.frequency_reference),
                      offsets_hz=evaluation.offsets_hz, od_conditions=asdict(conditions))
    return output_view, output_raw


@dataclass(frozen=True)
class _UnresolvedScope:
    channels: tuple[str, ...]
    previews: tuple[tuple[float, ...], ...]
    source_id: str
    reason: str


@st.cache_data(show_spinner=False, max_entries=8)
def _project(content, channel=None, source_id=None):
    from gabes.od_measurement import ScopeChannelChoice, prepare_measurement_upload
    try:
        return prepare_measurement_upload(content, scope_channel=channel, scope_source_id=source_id)
    except ScopeChannelChoice as choice:
        # cache_data caches return values. Ambiguity is an ordinary import outcome.
        return _UnresolvedScope(choice.channels, choice.previews, choice.source_id, str(choice))


@st.cache_data(show_spinner=False, max_entries=8)
def load_trace(upload, **options):
    from gabes.od_measurement import load_measurement
    return load_measurement(upload, **options)


def _scope_waveforms(panel, labels, previews):
    """Draw the shared downsampled traces without bridging missing detector samples."""
    rows = []
    for label, samples in zip(labels, previews):
        segment = 0
        for i, value in enumerate(samples):
            if np.isfinite(value):
                rows.append({"channel": label, "fraction": i / max(1, len(samples) - 1),
                             "level": float(value), "segment": segment})
            else:
                segment += 1
        if not samples:
            panel.caption(f"{label} · No finite preview samples")
    if rows:
        panel.vega_lite_chart(spec={
            "data": {"values": rows},
            "mark": {"type": "line", "clip": True},
            "encoding": {
                "row": {"field": "channel", "type": "nominal", "sort": labels, "title": None,
                        "header": {"labelAngle": 0, "labelLimit": 100}},
                "x": {"field": "fraction", "type": "quantitative", "title": "Acquisition order",
                      "axis": {"format": "%", "values": [0, 0.5, 1]}, "scale": {"domain": [0, 1]}},
                "y": {"field": "level", "type": "quantitative", "axis": None, "scale": {"domain": [0, 1]}},
                "detail": {"field": "segment", "type": "nominal"},
            },
            "height": 42,
        }, use_container_width=True)
    panel.caption("Each channel has its own voltage scale. A manual selection uses the full recording.")


def _scope_options(detected):
    unresolved = isinstance(detected, _UnresolvedScope)
    scope = detected if unresolved else detected.scope
    labels = tuple(
        f"CH{label}" if label.isdecimal() else label for label in scope.channels
    )
    if len(set(labels)) < len(labels):
        labels = tuple(f"Column {i + 1} · {label}" for i, label in enumerate(labels))
    automatic = None if unresolved else f"Automatic · {labels[scope.od]}"
    return scope, labels, automatic


def _scope_projection(content, detected, scope, labels, choice, automatic):
    if choice == automatic and automatic is not None:
        return detected
    return _project(content, labels.index(choice), scope.source_id)


def _select_scope_channel(panel, name, detected, content, previous):
    scope, labels, automatic = _scope_options(detected)
    unresolved = automatic is None
    options = labels if unresolved else (automatic, *labels)
    _scope_waveforms(panel, labels, scope.previews)
    widget_key = skey(name, "_csv_scope_" + scope.source_id)
    if widget_key not in st.session_state and getattr(previous, "sha256", None) == scope.source_id:
        if previous.scope and previous.scope.manual:
            st.session_state[widget_key] = labels[previous.scope.od]
    choice = panel.selectbox("OD signal channel", options, index=None if unresolved else 0,
                                   key=widget_key)
    if choice is None:
        reason = f"{scope.reason} " if unresolved else ""
        panel.info(reason + "Choose the channel measuring the transmitted probe.")
        return None
    if choice == automatic:
        projected = detected
        roles = f"OD {labels[scope.od]} · SAS {labels[scope.sas]} · Trigger {labels[scope.trigger]}"
    else:
        try:
            projected = _scope_projection(content, detected, scope, labels, choice, automatic)
        except ValueError as exc:
            # An empty detector channel should not discard the remaining waveforms.
            panel.warning(str(exc))
            return None
        roles = f"OD {choice} (manual)"
    span = projected.scope
    region = (f"Full recording ({span.total:,} samples)" if span.manual else
              f"Samples {span.start + 1:,}–{span.stop:,} of {span.total:,}")
    panel.caption(f"{roles} · {region} · Raw detector signal")
    return projected


def plot_metadata(params, actual_model):
    from gabes import species
    from gabes.od_model import frequency_reference
    metadata = {"model_id": actual_model, "line": params.get("line"),
                "pump_power_mw": params.get("pump_power_mw"),
                "probe_power_uw": params.get("probe_power_uw")}
    try:
        conditions = ODConditions.from_params(params)
    except ODUnsupportedError:
        return metadata
    contract = frequency_reference(conditions)
    axis = asdict(contract)
    if conditions.species == species.RB87.label:
        axis["isotope"] = species.RB87.name
    metadata.update(model_id=actual_model or conditions.model_id,
                    conditions=asdict(conditions),
                    axis_reference=axis, contract_reference=asdict(contract),
                    display_reference_offset_hz=display_reference_offset_hz(conditions))
    if accuracy := MODELS[metadata["model_id"]].accuracy:
        metadata["model_accuracy"] = accuracy
    return metadata


def native_reference_shift(upload, params):
    """Convert a declared native fine-centroid origin to the current plot, in GHz."""
    from gabes.od_measurement import native_frequency_reference
    from gabes.od_model import frequency_reference
    from gabes.optical_frequency import reference_offset_hz

    source = native_frequency_reference(upload.metadata)
    if not source:
        return 0.0, None
    try:
        conditions = ODConditions.from_params(params)
        isotope, line = source["isotope"], source.get("line")
        if line != conditions.line or source.get("point", "fine_centroid") != "fine_centroid":
            raise ValueError
        if frequency_reference(conditions).isotope == "Cs133":
            if isotope != "Cs133":
                raise ValueError
            return 0.0, None
        shift = reference_offset_hz(isotope, line) - display_reference_offset_hz(conditions)
        return shift * 1.e-9, None
    except (ValueError, KeyError, TypeError, AttributeError):
        return 0.0, "The recorded measurement origin differs from this plot. Calibrate the frequency axis before fitting."


def fit_conditions_key(params, free=()):
    conditions = asdict(ODConditions.from_params(params))
    return tuple((key, value) for key, value in conditions.items() if key not in free)


APPLICABLE_FIT_CONDITIONS = ("temp_c", "probe_power_uw", "rb85_fraction")


def request_fitted_conditions(identity, displayed_conditions):
    """Defer the click until the current file, model and fit controls have been read."""
    st.session_state["sas_od_pending_fit_conditions"] = identity, displayed_conditions


def apply_fitted_conditions(saved, identity, pending, live_conditions):
    """Only apply the converged estimates from the fit the user just inspected."""
    if not saved or saved[0] != identity or pending != (identity, live_conditions):
        return False
    result = saved[1]
    if not result["identifiable"] or result["status"] != "converged":
        return False
    fields = [row for row in result["parameters"] if row["key"] in APPLICABLE_FIT_CONDITIONS]
    if not fields:
        return False
    st.session_state.update({skey("sas", row["key"]): row["value"] for row in fields})
    values = ", ".join(f"{row['name']} {row['value']:.4g} {row['unit']}".strip() for row in fields)
    st.session_state["sas_od_apply_notice"] = f"Applied fitted conditions · {values}."
    return True
