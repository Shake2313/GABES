"""Exports use the same rendered data and preserve reproduction context."""
from io import BytesIO
import json
from types import SimpleNamespace
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gabes.collisions import REFERENCE_COLLISIONS
from gabes.core import blas_single_thread
from gabes.schemes import get
from gabes.schemes.absorption import LAMBDA_JOINT_MAXWELL, ODScheme
from gabes_ui import export, plotcard
from gabes_ui.spectral_compare import comparison_plots
from gabes_ui.spectral_sweep import assemble_dataset


@pytest.fixture(autouse=True)
def noninteractive_canvas():
    with matplotlib.rc_context({"backend": "Agg"}):
        yield


def test_named_native_curves_round_trip_from_exported_conditions():
    scheme = get("lambda")
    params = scheme.defaults()

    def render(condition):
        with blas_single_thread():
            raw = scheme.compute({key: condition[key] for key in scheme.recompute_keys()})
            view = scheme.observables(raw, condition)
        plots = export.collect_view_plot_payloads(view, view["figure"])
        for figure in [view["figure"], *(f for _, f in view.get("figures", ()))]:
            plt.close(figure)
        return raw, view, plots

    raw, view, plots = render(params)
    bundle = export.build_export_payload(scheme=scheme, raw=raw, params=params, view=view,
        plot_payloads=plots, comparison_payload=None, cache_version=scheme.cache_version,
        readout_cache_version="roundtrip-test")
    loaded = json.loads(json.dumps(bundle, allow_nan=False))
    _, _, replay = render(loaded["params"])
    saved, rerun = export.named_plot_arrays(loaded["plots"]), export.named_plot_arrays(replay)
    assert saved and saved.keys() == rerun.keys()
    for key in saved:
        for array in ("x_data", "y_data"):
            np.testing.assert_allclose(np.asarray(saved[key][array], float),
                                       np.asarray(rerun[key][array], float),
                                       rtol=1e-12, atol=1e-15, equal_nan=True)


def test_named_arrays_and_zip_preserve_duplicate_curve_labels():
    trace = dict(axis_index=0, figure_curve_label="branch", x_data=[1.], y_data=[2.])
    plots = [dict(label="Scan", traces=[trace, dict(trace, axis_index=1, y_data=[3.])])]
    named = export.named_plot_arrays(plots)
    assert [trace["y_data"] for trace in named.values()] == [[2.], [3.]]
    with zipfile.ZipFile(BytesIO(export._build_plot_zip_bytes(plots))) as archive:
        assert archive.namelist() == [name + ".csv" for name in named]


def test_lazy_extra_view_export_runs_only_when_requested(monkeypatch):
    from streamlit.testing.v1 import AppTest

    calls, downloads = [], []
    scheme = get("lambda")

    def render(raw, _params):
        figure, axis = plt.subplots()
        axis.plot([1., 2.], raw["signal"], label="signal")
        return figure

    extra = SimpleNamespace(key="Test scan", description="Test scan", param_keys=(),
                            render=render, render_with_params=True)
    monkeypatch.setattr(type(scheme), "extra_views", lambda _self: [extra])
    def compute(*_args, **_kwargs):
        calls.append("run")
        return {"signal": [3., 4.], "minus": {"branch": -1, "validation_level": "PREDICTED"}}
    def download(*_args, **kwargs):
        downloads.append(kwargs)
    monkeypatch.setattr(plotcard, "_cached_extra", compute)
    monkeypatch.setattr(export, "render_export", download)
    monkeypatch.setattr(plotcard.st, "pills", lambda *_args, **_kwargs: "Test scan")
    app = AppTest.from_string("""
        from gabes.schemes import get
        from gabes_ui.plotcard import render_more
        import streamlit as st
        scheme = get("lambda")
        st.session_state.setdefault("lambda__more", "Test scan")
        render_more({}, scheme, scheme.defaults(), scheme.cache_version, "roundtrip-test")
    """).run()
    assert not app.exception and not calls and not downloads
    app.button[0].click().run()
    assert not app.exception and calls == ["run"] and len(downloads) == 1
    result = downloads[0]
    assert result["plot_payloads"][0]["traces"][0]["y_data"] == [3., 4.]
    assert result["key_suffix"] == "extra-view"
    assert result["extra_view"]["branches"]["minus"]["validation_level"] == "PREDICTED"
    assert result["extra_view"]["status"] == "complete"
    assert result["extra_view"]["compute_params"]


def test_retained_extra_view_retries_unresolved_and_caches_success(monkeypatch):
    from streamlit.testing.v1 import AppTest

    calls, downloads = [], []
    scheme = get("sas")
    view_key = "Finite-probe pumping reference"
    timeout_reason = "The 30 s reference calculation limit was reached."

    def compute(_self, params):
        calls.append(params)
        provenance = {"validation": "numerical readiness", "peak_probe_intensity_w_m2": 0.0}
        if len(calls) == 1:
            return {"status": "unresolved", "reason": timeout_reason,
                    "provenance": provenance, "refinement": {"levels": 1}}
        return {"status": "converged", "x_ghz": [1., 2.], "transmission": [0.8, 0.9],
                "provenance": provenance, "refinement": {"levels": 2}}

    monkeypatch.setattr(type(scheme), "projected_probe_reference", compute)
    monkeypatch.setattr(export, "render_export", lambda *_args, **kwargs: downloads.append(kwargs))
    monkeypatch.setattr(plotcard.st, "pills", lambda *_args, **_kwargs: view_key)
    app = AppTest.from_string("""
        from gabes.schemes import get
        from gabes_ui.plotcard import render_on_demand
        scheme = get("sas")
        params = dict(scheme.defaults(), pump_power_mw=0.0)
        render_on_demand(scheme, params, "retained-retry-test", "roundtrip-test")
    """).run()
    run_key = f"run__sas__{view_key}"
    assert not app.exception and not calls and not downloads

    app.button(key=run_key).click().run()
    assert not app.exception and len(calls) == 1
    assert app.warning[0].value == timeout_reason
    assert downloads[-1]["extra_view"]["status"] == "unresolved"
    assert downloads[-1]["extra_view"]["refinement"] == {"levels": 1}
    assert not downloads[-1]["plot_payloads"]

    app.run()
    assert not app.exception and len(calls) == 1
    assert app.warning[0].value == timeout_reason

    app.button(key=run_key).click().run()
    assert not app.exception and len(calls) == 2 and calls[0] == calls[1]
    assert not app.warning
    assert downloads[-1]["extra_view"]["status"] == "converged"
    assert downloads[-1]["plot_payloads"][0]["traces"][0]["y_data"] == [0.8, 0.9]

    app.button(key=run_key).click().run()
    assert not app.exception and len(calls) == 2
    assert downloads[-1]["extra_view"]["refinement"] == {"levels": 2}


def test_sweep_exports_status_before_first_row_and_keeps_curve_provenance():
    curve = dict(id="branch", label="Lower branch", branch=-1, x=[1., 2.],
                 observables={"gain": [3., None]}, observable_status={"gain": ["available", "unresolved"]})
    template = dict(axis=dict(label="Frequency", unit="MHz"), series=[curve],
                    panels=[dict(title="Gain", y_label="Gain", curves=[dict(series_id="branch", observable="gain")])])
    dataset = assemble_dataset(template, dict(rows=[], status="failed", completed=0, error="no completed rows"))
    assert dataset["sweep"] == dict(status="failed", completed=0, error="no completed rows")
    metadata = comparison_plots(dataset)[0]["traces"][0]["quantity_metadata"]
    assert metadata["branch"] == -1
    assert metadata["observable_status"] == ["available", "unresolved"]


@pytest.mark.parametrize("scheme_name, reference, expected_key", [
    ("od", {}, None),
    ("od", dict(optical_collision_model=REFERENCE_COLLISIONS, ne_pressure_torr=10.0),
     "optical_collisions"),
    ("lambda", {}, None),
    ("lambda", dict(optical_collision_model=REFERENCE_COLLISIONS, ne_pressure_torr=10.0),
     "lambda_readout"),
    ("lambda", dict(angled_doppler_model=LAMBDA_JOINT_MAXWELL,
                    doppler="on", beam_angle_mrad=0.7), "lambda_readout"),
])
def test_optical_reference_exports_view_evidence_in_strict_json(scheme_name, reference, expected_key):
    scheme = ODScheme() if scheme_name == "od" else get(scheme_name)
    params = dict(scheme.defaults(), doppler="off")
    if scheme_name == "od":
        params["model"] = "single 2-level"
    params.update(reference)
    with blas_single_thread():
        raw = scheme.compute(params)
        view = scheme.headless_observables(raw, params)
    bundle = export.build_export_payload(
        scheme=scheme, raw=raw, params=params, view=view, plot_payloads=[],
        comparison_payload=None, cache_version=scheme.cache_version,
        readout_cache_version="reference-export-test")
    saved = json.loads(json.dumps(bundle, allow_nan=False))["view"]
    new_keys = {"optical_collisions", "lambda_readout"}
    assert new_keys.intersection(saved) == ({expected_key} if expected_key else set())
    if expected_key:
        assert saved[expected_key] == export.to_json_primitive(view[expected_key])
    if "angled_doppler_model" in reference:
        evidence = saved["lambda_readout"]["joint_doppler"]
        assert "difference_from_scalar" in evidence
        assert "transverse_refinement" in evidence
    elif "optical_collision_model" in reference:
        evidence = saved[expected_key] if scheme_name == "od" else saved[expected_key]["optical_collisions"]
        assert evidence["shift_mhz"] == pytest.approx(-11.0)
        assert evidence["shift_sources"][0]["source"].startswith("https://doi.org/")
