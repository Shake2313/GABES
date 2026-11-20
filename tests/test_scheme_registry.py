"""The scheme menu must not run unrelated physics imports at startup."""
from pathlib import Path
import subprocess
import sys
import textwrap

from gabes import schemes


def test_menu_and_sas_selection_load_only_the_requested_scheme():
    # Other tests import the registry during collection; inspect a fresh interpreter.
    subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import sys
        from gabes import schemes

        choices = schemes.choices()
        assert [c.name for c in choices] == ["sas", "lambda", "rydberg_eit", "magneto", "fwm", "psr"]
        assert {m for m in sys.modules if m.startswith("gabes.schemes.")} == {"gabes.schemes.base"}
        assert "numpy" not in sys.modules and "matplotlib" not in sys.modules

        sas = schemes.get("sas")
        assert sas is schemes.get(choices[0].name)
        assert type(sas) is schemes.SASScheme
        for module in ("absorption", "rydberg", "magneto", "fwm", "psr"):
            assert "gabes.schemes." + module not in sys.modules, module
    """)], check=True)


def test_lambda_deep_link_does_not_import_od_input_validation():
    subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import sys
        from unittest.mock import patch
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        with patch.object(st.components.v2, "component", return_value=lambda *args, **kwargs: {}):
            app = AppTest.from_file("streamlit_app.py", default_timeout=60)
            app.query_params["scheme"] = "lambda"
            app.run()
            assert not app.exception and not app.error
            assert "gabes.schemes.absorption" in sys.modules
            assert "gabes.schemes.sas" not in sys.modules
    """)], cwd=Path(__file__).resolve().parents[1], check=True, timeout=90)


def test_choices_match_instances_and_historical_registry(monkeypatch):
    choices, instances = schemes.choices(), schemes.all_schemes()
    assert [(c.name, c.title, c.cluster) for c in choices] == [
        (s.name, s.title, s.cluster) for s in instances]
    registry = schemes.REGISTRY
    assert isinstance(registry, dict)
    assert list(registry) == [c.name for c in choices] + ["eit", "at", "cpt", "hanle", "eia", "nmor"]
    assert all(registry[s.name] is s for s in instances)
    assert type(schemes.get("eit")) is schemes.LambdaScheme
    assert type(schemes.get("hanle")) is schemes.MagnetoScheme
    monkeypatch.setitem(registry, "sas", instances[-1])
    assert schemes.get("sas") is instances[-1]
    assert schemes.all_schemes()[0] is instances[0]
