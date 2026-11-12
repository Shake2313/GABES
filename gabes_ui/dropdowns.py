"""One browser policy for short, selection-only Streamlit dropdowns."""
import streamlit as st

from .component_runtime import missing_apptest_component
from .paths import asset_text


_COMPONENT = st.components.v2.component(
    "gabes_dropdown_policy",
    html='<span class="dropdown-policy" hidden></span>',
    js=asset_text("ui/dropdown_policy.js"),
)


def render_dropdown_policy():
    """Apply to native dropdowns across main panels, popovers and sidebars."""
    try:
        _COMPONENT(key="gabes-dropdown-policy", width="content")
    except ValueError as error:
        if not missing_apptest_component(error, "gabes_dropdown_policy"):
            raise
    except TypeError as error:
        # Streamlit's AppTest does not implement serializing v2 component JS.
        if "bad argument type for built-in operation" not in str(error):
            raise
