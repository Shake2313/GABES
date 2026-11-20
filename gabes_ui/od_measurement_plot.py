"""Browser-local inspection of measured points and the pump-free OD spectrum."""
import streamlit as st

from gabes.plot_style import PALETTE
from .component_runtime import missing_apptest_component, spectrum_component_js
from .paths import asset_text


_COMPONENT = st.components.v2.component(
    "gabes_od_measurement_plot",
    html='<div class="mc od-measurement"></div>',
    css=asset_text("ui/model_compare.css") + "\n" + asset_text("ui/od_measurement_plot.css"),
    js=spectrum_component_js("od_measurement_plot.js"),
)


def render_measurement_plot(comparison):
    theory = comparison["theory_trace"]
    measured = comparison["comparison_trace"]
    if len(theory["detuning"]) < 2 or not len(measured["detuning"]):
        return False
    x, y = measured["detuning"], measured["transmission"]
    if x[0] > x[-1]:  # Reverse alignment preserves acquisition direction in the export.
        x, y = x[::-1], y[::-1]
    payload = dict(
        grid=theory["detuning"], transmission=theory["transmission"],
        depth=theory.get("optical_depth"), measuredX=x,
        measuredY=y, coverage=comparison["coverage_ghz"],
        absolute=comparison["calibration_mode"] != "relative_extrema",
        source=comparison["source_fingerprint_sha256"],
        theoryColor=PALETTE["cyan"], measuredColor=PALETTE["rose"],
    )
    try:
        _COMPONENT(key="od-measurement-inspector", data=payload, width="stretch")
    except ValueError as exc:
        if not missing_apptest_component(exc, "gabes_od_measurement_plot"):
            raise
        return False
    return True
