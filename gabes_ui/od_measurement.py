"""File selection and fit-result ownership for the pump-free OD workspace."""
from dataclasses import asdict

import streamlit as st

from gabes.od_model import ODConditions, ODUnsupportedError, display_reference_offset_hz
from gabes_ui.controls import skey


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
