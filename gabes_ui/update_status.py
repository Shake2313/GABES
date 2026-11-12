"""Browser-visible update state; observes rendering without changing computation."""
import streamlit as st

from .component_runtime import missing_apptest_component
from .paths import asset_text

_COMPONENT = st.components.v2.component(
    "gabes_update_status",
    html='<div class="update-status" role="status" aria-live="polite" aria-atomic="true"></div>',
    css="""
    .update-status{min-height:1.4rem;font:400 .78rem/1.4 var(--g-font-sans,sans-serif);
      color:var(--g-muted,#5B6B80);font-variant-numeric:tabular-nums;}
    .update-status[data-phase="syncing"],.update-status[data-phase="preview"]{
      color:var(--g-accent-ink,#0369A1)}
    .update-status[data-phase="error"]{color:var(--g-danger-ink,#B91C1C)}
    """,
    js=asset_text("ui/update_status.js"),
)


def render_update_status(scheme_name):
    """Mount before the solve; the old component also observes the next rerun."""
    try:
        _COMPONENT(key="gabes-update-status", data={"scheme": scheme_name}, width="stretch")
    except TypeError:
        # AppTest does not implement bidirectional components.
        pass
    except ValueError as error:
        if not missing_apptest_component(error, "gabes_update_status"):
            raise
