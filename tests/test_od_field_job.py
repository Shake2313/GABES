"""User controls remain accessible while a finite-power field scan is pending."""
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event

import pytest


def test_full_app_pending_cancel_pump_switch_and_weak_readout(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from streamlit.testing.v1.element_tree import ButtonGroup
    from gabes import schemes, species
    from gabes.schemes.sas import FIELD_HISTORY_TRANSIT
    from gabes_ui import od_field_job

    monkeypatch.setattr(st, "_bidi_component", lambda *args, **kwargs: {})
    def indices(widget):
        values = widget.value if isinstance(widget.value, list) else [widget.value]
        return [widget.options.index(widget.format_func(v)) for v in values if v is not None]
    monkeypatch.setattr(ButtonGroup, "indices", property(indices))
    entered, resume = Event(), Event()
    def slow_worker(_scheme, _params, *, progress):
        entered.set()
        assert resume.wait(20)
        if not progress(("Rb87", 1, 10)):
            raise CancelledError()
        pytest.fail("obsolete field scan was allowed to finish")
    monkeypatch.setattr(od_field_job, "_solve", slow_worker)
    scheme = schemes.get("sas")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=40)
    app.query_params["scheme"] = "sas"
    app.session_state["sas___defaults_version"] = scheme.defaults_version
    params = dict(scheme.defaults(), species=species.RB87.label, line="D1", scan_points=201,
                  pump_power_mw=0., probe_power_uw=.12, od_field_x_ut=50.,
                  od_transit_model=FIELD_HISTORY_TRANSIT)
    for key, value in params.items():
        app.session_state["sas__" + key] = value
    app.run()
    assert not app.exception and not app.error
    assert "sas_od_field_history_job" not in app.session_state
    try:
        app.button(key="sas_start_field_history").click().run()
        assert entered.wait(5)
        job = app.session_state["sas_od_field_history_job"]
        assert app.button(key="sas_cancel_field_history")
        # The ordinary power editor must be available before a scan completes.
        app.text_input(key="power-input-sas-pump_power_mw").set_value("0.05").run()
        assert not app.exception and not app.error
        assert "sas_od_field_history_job" not in app.session_state
        assert not any(b.key == "sas_start_field_history" for b in app.button)
    finally:
        resume.set()
    with pytest.raises(CancelledError):
        job.future.result(timeout=5)
    # The selected field model is restored when the pump is turned off.
    app.text_input(key="power-input-sas-probe_power_uw").set_value("0").run()
    app.text_input(key="power-input-sas-pump_power_mw").set_value("0").run()
    assert not app.exception and not app.error
    assert not any(b.key == "sas_start_field_history" for b in app.button)
    assert app.session_state["sas__od_transit_model"] == FIELD_HISTORY_TRANSIT
