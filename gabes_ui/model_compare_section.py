"""The model spectra workspace. The component reads prepared samples only."""
import hashlib
import json

import numpy as np
import streamlit as st

from gabes_ui import export as ui_export


def comparison_context(scheme, params, schema):
    """Selection events from an older grid must not apply to a new run."""
    identity = [scheme.name, scheme.cache_version, schema, params]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _fit_dataset(dataset, view):
    """Keep the group windows and cursor on the samples used for the OD fit."""
    from gabes.model_comparison import sample_values
    from gabes.od_model import MODELS

    readout = view["optical_readout"]
    scale = {"GHz": 1000., "MHz": 1.}[readout["x_unit"]]
    if not np.array_equal(np.asarray(readout["x"]) * scale, dataset["x"]):
        raise ValueError("The OD fit and displayed model must share a frequency grid.")
    off, = dataset["series"]
    if off["power_mw"] != 0:
        raise ValueError("OD fitting requires the pump-off curve.")
    row = dict(off)
    for quantity in ("transmission", "optical_depth", "od0"):
        row[quantity] = sample_values(readout[quantity])
    row.update(effective=readout["effective"], delta=[0.] * len(dataset["x"]))
    source = dict(dataset["source"])
    model_id = view.get("od_model_id")
    if model_id in MODELS:
        model = MODELS[model_id]
        row.update(probe_model=model_id, probe_model_scope=model.description,
                   probe_model_note=readout.get("model_accuracy", model.accuracy))
        source.update(probe_model=model_id)
    if "resolution" not in readout:
        # These diagnostics belong to the original solver. A replay evaluator
        # that only emits attenuation cannot supply its resolution or dispersion.
        for field in ("resolution", "attenuation_scope", "weak_probe_profile", "axial_cell"):
            row.pop(field, None)
        source.pop("weak_probe_profile", None)
        source["method"] = "pump-free OD evaluator"
    return dict(dataset, series=[row], n_eff=readout["n_eff"], source=source)


def _measurement_samples(comparison):
    from gabes.model_comparison import sample_values

    if not comparison:
        return None
    trace = comparison["comparison_trace"]
    scale = {"MHz": 1., "GHz": 1000.}[trace.get("detuning_unit", "GHz")]
    x = np.asarray(trace["detuning"], dtype=float) * scale
    y = np.asarray(trace["transmission"], dtype=float)
    if not x.size:
        return None
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    coverage = np.asarray(comparison["coverage_ghz"], dtype=float) * 1000.
    absolute = comparison["calibration_mode"] != "relative_extrema"
    digest = hashlib.sha256()
    for values in (x, y, coverage):
        digest.update(values.tobytes())
    digest.update(bytes([absolute]))
    return dict(x=x.tolist(), y=sample_values(y), coverage=coverage.tolist(),
                absolute=absolute, revision=digest.hexdigest())


def comparison_plot_payloads(dataset, measurement=None):
    """Export the full native T and OD curves shown by the model workspace."""
    plots = []
    for axis_index, (key, title) in enumerate((
            ("transmission", "Transmission T"),
            ("optical_depth", "Optical depth OD"))):
        traces = []
        for curve_index, condition in enumerate(dataset["series"]):
            power = condition["power_mw"]
            label = "Pump off (0 mW)" if power == 0 else f"Pump {power} mW"
            traces.append({
                "axis_index": axis_index,
                "curve_index": curve_index,
                "figure_curve_label": label,
                "x_label": dataset["axis"]["label"],
                "x_unit": dataset["axis"]["unit"],
                "y_label": title,
                "y_unit": "1",
                "x_data": dataset["x"],
                "y_data": condition[key],
                "color": "#4c606c" if power == 0 else "#07655e",
                "quantity_metadata": {
                    "power_mw": power,
                    "frequency_reference": dataset["axis"]["reference"],
                    "sample_scope": "full native sampled range",
                },
            })
        if measurement and (key == "transmission" or measurement["absolute"]):
            values = measurement["y"]
            if key == "optical_depth":
                values = [-float(np.log(t)) if t is not None and t > 0 else None for t in values]
            signal = title if measurement["absolute"] else "Relative signal"
            traces.append({
                "axis_index": axis_index, "curve_index": len(traces),
                "figure_curve_label": "Measurement", "x_label": dataset["axis"]["label"],
                "x_unit": "MHz", "y_label": signal, "y_unit": "1",
                "x_data": measurement["x"], "y_data": values, "color": "#a23c5c",
                "quantity_metadata": {"frequency_reference": dataset["axis"]["reference"],
                                      "coverage_mhz": measurement["coverage"]},
            })
        plots.append({"label": title, "source": "model_comparison", "traces": traces})
    return plots


@st.fragment
def render_comparison(dataset, context, *, scheme, raw, params, view, readout_version, fit_view=None):
    from gabes.od_model import PUMP_GROUP
    from gabes_ui.conditions import render_power_inputs
    from gabes_ui.model_compare_plot import render_group_control, render_model_compare, render_readout_outlet
    from gabes_ui.od_fit_panel import current_upload, has_csv, render_comparison_export, render_fit_panel
    from gabes_ui.od_measurement import workspace_params

    key = f"model-compare-{scheme.name}"
    has_file = has_csv()
    specs = scheme.param_schema()
    if fit_view is not None:
        dataset = _fit_dataset(dataset, fit_view)
    with st.container(key="gabes_spectrum_toolbar"):
        if has_file:
            group_column, probe_column = st.columns([3.4, 1], gap="small")
            beam_columns = (probe_column,)
            specs = [sp for sp in specs if sp.group != PUMP_GROUP]
        else:
            group_column, *beam_columns = st.columns([2.4, 1, 1], gap="small")
        with group_column:
            render_group_control(dataset, context=context, key=key)
        current = render_power_inputs(scheme, specs, columns=beam_columns)
    if has_file:
        current = workspace_params(current, st.session_state, current_upload())
    if current != params:
        st.rerun(scope="app")
    charts, rail = st.columns([2.6, 1], gap="medium")
    with rail:
        render_readout_outlet(key, "point")
        comparison = render_fit_panel(st.container(), params=params, view=fit_view)
        render_readout_outlet(key, "details")
    if has_file != has_csv():
        st.rerun(scope="app")
    measurement = _measurement_samples(comparison)
    with charts:
        selection = render_model_compare(dataset, key=key, context=context, show_groups=False,
                                         split_readout=True, measurement=measurement)
    export_box = st.popover("Export", icon=":material/download:", type="tertiary")
    if comparison:
        render_comparison_export(export_box, comparison)
    ui_export.render_export(export_box, scheme=scheme, raw=raw, params=params,
                            view=view, plot_payloads=comparison_plot_payloads(dataset, measurement),
                            comparison_payload=comparison,
                            cache_version=scheme.cache_version, readout_cache_version=readout_version,
                            model_comparison={**dataset, "selection": selection})
