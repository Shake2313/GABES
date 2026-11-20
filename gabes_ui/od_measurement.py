"""File selection and fit-result ownership for the pump-free OD workspace."""
from dataclasses import asdict

import numpy as np
import streamlit as st

from gabes.od_model import (MODEL_KEY, MODELS, ODConditions, ODUnsupportedError, display_reference_offset_hz,
                            evaluate_od_display, od_fit_conditions, od_unavailable)
from gabes_ui.controls import skey


def workspace_params(params, state):
    """Honor a reserved model from existing session state until the SAS schema exposes it."""
    derived = od_fit_conditions(params)
    if MODEL_KEY not in derived and state.get(skey("sas", MODEL_KEY)) is not None:
        derived[MODEL_KEY] = state[skey("sas", MODEL_KEY)]
    return derived


@st.cache_data(show_spinner=False, max_entries=8)
def _spectrum(conditions, scan_rad_s):
    from gabes.core import blas_single_thread
    with blas_single_thread():
        return evaluate_od_display(conditions, scan_rad_s)


def evaluate_workspace(params, raw):
    """Check explicit requests before allocating figures; permit the displayed scheme-only fallback."""
    from gabes.schemes.sas import OD_MODELS_D20, OD_MODELS_PRE_D20
    requested = params.get(MODEL_KEY)
    if requested is not None and (not isinstance(requested, str) or not requested.strip()):
        raise ODUnsupportedError("Invalid OD model ID. Expected a non-blank string.")
    # A model set ("pre_d20", "d20") chooses per regime; only an explicit ID is a request.
    explicit = None if requested in (None, OD_MODELS_PRE_D20, OD_MODELS_D20) else requested
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
                   effective=conditions.probe_power_uw > 0., model_id=evaluation.model_id)
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
    output_view = ({"figure": view["figure"], "comparison": view["comparison"]} if changed else dict(view))
    output_view.update(experimental_workspace=True, od_model_id=evaluation.model_id,
                       optical_readout=readout, figure_views=pages)
    output_raw = ({"scan": raw["scan"], "markers": raw.get("markers", [])} if changed else dict(raw))
    output_raw.update(probe_model=evaluation.model_id, transmission=evaluation.transmission,
                      optical_depth=evaluation.optical_depth, od0=evaluation.od0,
                      frequency_reference=asdict(evaluation.frequency_reference),
                      offsets_hz=evaluation.offsets_hz, od_conditions=asdict(conditions))
    return output_view, output_raw


@st.cache_data(show_spinner=False, max_entries=8)
def _project(content, channel=None, source_id=None):
    from gabes.od_measurement import prepare_measurement_upload
    return prepare_measurement_upload(content, scope_channel=channel, scope_source_id=source_id)


@st.cache_data(show_spinner=False, max_entries=8)
def load_trace(upload, **options):
    from gabes.od_measurement import load_measurement
    return load_measurement(upload, **options)


def select_upload(panel, name, uploaded, forget):
    from gabes.od_measurement import ScopeChannelChoice

    key = skey(name, "_csv_retained_file")
    if uploaded is None:
        saved = st.session_state.get(key)
        if saved is None:
            return None
        filename, projection = saved
        panel.button("Remove retained CSV", key=skey(name, "_csv_remove_retained"),
                     on_click=forget, args=(name,))
        # Older sessions retained the original bytes until the first view switch.
        if isinstance(projection, bytes):
            projection = _project(projection)
            st.session_state[key] = filename, projection
        return filename, projection

    content = uploaded.getvalue()
    try:
        projection = _project(content)
        scope = projection.scope
        channels = scope.channels if scope else ()
        source_id = projection.sha256
        automatic = scope.od if scope else None
    except ScopeChannelChoice as choice:
        projection = None
        channels, source_id, automatic = choice.channels, choice.source_id, None
    if channels:
        channel = panel.selectbox("OD signal channel", channels, index=automatic,
                                  key=skey(name, "_csv_scope_" + source_id))
        if channel is None:
            panel.info("Choose the channel measuring the transmitted probe.")
            return None
        if automatic is None or channel != channels[automatic]:
            projection = _project(content, channel, source_id)
    st.session_state[key] = uploaded.name, projection
    return uploaded.name, projection


def restore_detector_law(name, upload, calibration_options):
    """Restore the declared detector law from a comparison file's native channel."""
    import math
    from gabes.experimental_csv import CALIBRATION_ABSOLUTE_DARK_REFERENCE, CALIBRATION_ABSOLUTE_GAIN_OFFSET
    mode = upload.metadata.get("experiment_calibration_mode")
    if mode not in (CALIBRATION_ABSOLUTE_DARK_REFERENCE, CALIBRATION_ABSOLUTE_GAIN_OFFSET):
        return
    try:
        gain = float(upload.metadata["experiment_calibration_gain"])
        offset = float(upload.metadata["experiment_calibration_offset"])
        if not math.isfinite(gain) or not gain or not math.isfinite(offset) or not math.isfinite(offset + gain):
            raise ValueError
    except (ValueError, TypeError, OverflowError, KeyError) as exc:
        raise ValueError("Invalid detector calibration in the comparison file.") from exc
    label = next(label for label, value in calibration_options.items() if value == mode)
    st.session_state[skey(name, "_csv_calibration_mode")] = label
    if mode == CALIBRATION_ABSOLUTE_GAIN_OFFSET:
        pair = {"_csv_detector_gain": gain, "_csv_detector_offset": offset}
    else:
        pair = {"_csv_dark_signal": offset, "_csv_reference_signal": offset + gain}
    for key, value in pair.items():
        st.session_state[skey(name, key)] = value


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
    reference = frequency_reference(conditions)
    metadata.update(model_id=actual_model or conditions.model_id,
                    conditions=asdict(conditions),
                    frequency_reference=asdict(reference),
                    display_reference_offset_hz=display_reference_offset_hz(conditions))
    if accuracy := MODELS[metadata["model_id"]].accuracy:
        metadata["model_accuracy"] = accuracy
    if conditions.species == species.RB87.label:
        metadata["frequency_reference"] = dict(asdict(reference), isotope=species.RB87.name)
    return metadata


def native_reference_shift(upload, params):
    """Convert a declared native fine-centroid origin to the current plot, in GHz."""
    from gabes.optical_frequency import reference_offset_hz
    from gabes.od_model import frequency_reference

    source = upload.metadata.get("experiment_frequency_reference")
    if not source:
        return 0.0, None
    try:
        conditions = ODConditions.from_params(params)
        if isinstance(source, str):
            source = {"isotope": source, "line": upload.metadata.get("line")}
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


def fit_conditions_key(params, free):
    conditions = asdict(ODConditions.from_params(params))
    return tuple((key, value) for key, value in conditions.items() if key not in free)


def request_temperature(identity):
    """Validate and apply after the current file and all form fields have been read."""
    st.session_state["sas_od_pending_temperature"] = identity
