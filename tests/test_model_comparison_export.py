"""CSV downloads must contain the native curves displayed in the model workspace."""
import csv
from io import BytesIO, StringIO
import zipfile

import numpy as np
import pytest

from gabes import schemes
from gabes.core import blas_single_thread
from gabes_ui.export import _build_plot_zip_bytes
from gabes_ui.model_compare_section import _fit_dataset, _measurement_samples, comparison_plot_payloads


@pytest.mark.parametrize("power", [0., .83])
def test_comparison_csv_exports_current_and_baseline_samples_without_marker_traces(power):
    scheme = schemes.get("sas")
    params = {**scheme.defaults(), "pump_power_mw": power, "scan_points": 401}
    with blas_single_thread():
        raw = scheme.compute({key: params[key] for key in scheme.recompute_keys()})
        view = scheme.headless_observables(raw, params)
        model = scheme.model_comparison(
            params, current_raw=raw, current_view=view,
            resolve_raw=lambda values: scheme.compute({key: values[key] for key in scheme.recompute_keys()}))
    plots = comparison_plot_payloads(model)
    assert [p["label"] for p in plots] == ["Transmission T", "Optical depth OD"]
    assert all(p["source"] == "model_comparison" for p in plots)
    assert all(len(p["traces"]) == len(model["series"]) for p in plots)
    archive = zipfile.ZipFile(BytesIO(_build_plot_zip_bytes(plots)))
    assert len(archive.namelist()) == 2 * len(model["series"])
    assert not any("line-" in name or "Dispersion" in name or "absorbance" in name
                   for name in archive.namelist())
    for plot, key in zip(plots, ("transmission", "optical_depth")):
        for trace, condition in zip(plot["traces"], model["series"]):
            assert trace["quantity_metadata"]["power_mw"] == condition["power_mw"]
            assert trace["quantity_metadata"]["frequency_reference"] == model["axis"]["reference"]
            np.testing.assert_array_equal(trace["x_data"], model["x"])
            np.testing.assert_array_equal(trace["y_data"], condition[key])
    # Parse actual ZIP bytes, including the native MHz axis and both full curves.
    for name, key in (("Transmission-T/Pump-off-0-mW.csv", "transmission"),
                      ("Optical-depth-OD/Pump-off-0-mW.csv", "optical_depth")):
        rows = list(csv.reader(StringIO(archive.read(name).decode("utf-8"))))
        assert rows[0][0] == "Relative frequency [MHz]"
        assert len(rows) == 402
        np.testing.assert_array_equal([float(row[0]) for row in rows[1:]], model["x"])
        np.testing.assert_array_equal([float(row[1]) for row in rows[1:]], model["series"][0][key])


def test_replayed_model_and_native_csv_share_plots_and_exports_without_mutating_simulation():
    dataset = dict(x=[0., 100., 200.], axis={"label": "Frequency", "unit": "MHz", "reference": "Rb85"},
                   source={"weak_probe_profile": "old"},
                   series=[dict(power_mw=0., resolution="original", probe_model="original")])
    readout = dict(x=[0., .1, .2], x_unit="GHz", transmission=[1., .25, .9],
                   optical_depth=[0., 1.39, .1], od0=[0., 2., .2], n_eff=123., effective=True)
    model_id = "od_finite_rb_mf_rate_v1"
    model = _fit_dataset(dataset, dict(optical_readout=readout, od_model_id=model_id))
    assert model["series"][0]["probe_model"] == model_id
    assert model["series"][0]["probe_model_note"]
    assert "resolution" not in model["series"][0]
    assert "weak_probe_profile" not in model["source"]
    assert dataset["series"][0]["resolution"] == "original"
    comparison = dict(comparison_trace={"detuning": [.2,.1,0.], "transmission": [.88,.21,.98]},
                      coverage_ghz=[[0.,0.],[.1,.2]], calibration_mode="relative_extrema")
    measured = _measurement_samples(comparison)
    assert measured["x"] == [0.,100.,200.] and measured["y"] == [.98,.21,.88]
    assert measured["coverage"] == [[0.,0.],[100.,200.]]
    assert comparison["comparison_trace"]["detuning"][0] == .2
    plots = comparison_plot_payloads(model, measured)
    assert plots[0]["traces"][0]["y_data"] == readout["transmission"]
    assert plots[0]["traces"][1]["y_label"] == "Relative signal"
    assert len(plots[1]["traces"]) == 1
    absolute = _measurement_samples({**comparison, "calibration_mode": "absolute_dark_reference"})
    assert absolute["revision"] != measured["revision"]
    plots = comparison_plot_payloads(model, absolute)
    np.testing.assert_allclose(plots[1]["traces"][1]["y_data"], -np.log(measured["y"]))
    # Keep calibrated T above unity and signed optical depth. An unreported
    # detector value or non-positive transmission has no logarithmic value.
    comparison["comparison_trace"]["transmission"] = [1.1, None, 0.]
    absolute = _measurement_samples({**comparison, "calibration_mode": "absolute_gain_offset"})
    plots = comparison_plot_payloads(model, absolute)
    assert plots[0]["traces"][1]["y_data"] == [0., None, 1.1]
    assert plots[1]["traces"][1]["y_data"] == [None, None, pytest.approx(-np.log(1.1))]
