"""Frozen AutoOD 0.2.2 measurement contracts, without an external desktop checkout."""
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from gabes.experimental_alignment import AxisAlignment
from gabes.experimental_csv import ExperimentalCSVError, MAX_FILE_BYTES, load_experimental_csv
from gabes.od_measurement import (
    ScopeChannelChoice, export_comparison_csv, gap_slices, load_measurement,
    prepare_measurement_upload, quick_measurement_alignment,
)
from gabes.od_spectrum_fit import prepare_od_fit

META = {"model_id": "weak_probe_exact_maxwell_voigt", "axis_reference": "Rb85",
        "contract_reference": "Rb85", "display_reference_offset_hz": 0.,
        "line": "D2", "temperature_c": 25., "transit_khz": 100.}


def paired_csv(x, y, header=("frequency_GHz", "transmission")):
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(header)
    writer.writerows(zip(x, y, strict=True))
    return text.getvalue().encode("utf-8")


def theory():
    with np.load(Path(__file__).parent / "fixtures/autood_baseline_0_2_2/spectra.npz") as data:
        return (data["natural_d2_25c_75mm__scan_rad_s"] / (2 * np.pi * 1e9),
                data["natural_d2_25c_75mm__probe_0uw__transmission"])


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp949"])
def test_declared_transmission_before_reference_encoding_and_units(encoding):
    payload = ("frequency_MHz,reference intensity,transmission (%),메모\n"
               "-500,100,80,측정\n0,100,20,측정\n500,100,82,측정\n").encode(encoding)
    upload = prepare_measurement_upload(io.BytesIO(payload))
    trace = load_measurement(upload)
    assert upload.sha256 == hashlib.sha256(payload).hexdigest()
    assert upload.columns == ("frequency_MHz", "transmission (%)")
    assert (upload.x_unit, upload.mode) == ("MHz", "percent")
    assert upload.metadata["encoding"] == encoding
    np.testing.assert_array_equal(trace.detuning, [-.5, 0., .5])
    np.testing.assert_array_equal(trace.transmission, [.8, .2, .82])


@pytest.mark.parametrize("origin", [False, True])
def test_raw_pairs_and_order_survive_display_transform(origin):
    x = np.r_[np.arange(20), np.arange(20)[::-1]]
    y = np.repeat([.8, .2], 20)
    upload = prepare_measurement_upload(paired_csv(x, y))
    trace = load_measurement(upload, denoise=False)
    assert trace.diagnostics.branches_merged
    np.testing.assert_array_equal(trace.transmission, np.full(20, .5))
    tx, ty = theory()
    data = export_comparison_csv(upload, trace, tx, ty, metadata=META, origin=origin,
                                 alignment=AxisAlignment(scale=2, shift=3, reverse=True))
    restored = prepare_measurement_upload(data)
    measured = load_measurement(restored)
    np.testing.assert_array_equal(np.concatenate([b.detuning for b in measured.branches]), x)
    np.testing.assert_array_equal(np.concatenate([b.raw_signal for b in measured.branches]), y)
    assert measured.branches[0].stop_index == trace.branches[0].stop_index
    assert restored.metadata["input_file_sha256"] == upload.sha256
    assert restored.metadata["model_id"] == META["model_id"]
    assert restored.metadata["axis_reference"] == restored.metadata["contract_reference"] == "Rb85"
    assert restored.metadata["format_version"] == 2
    assert "frequency_reference" not in restored.metadata
    assert restored.metadata["experiment_alignment"]["reverse"]


def _historical_gabes_csv(reference, offset, origin, native=None):
    # Construct the v1 wire format independently of the current writer.
    headers = ["detuning_GHz", "theory_transmission", "experiment_input_x",
               "experiment_input_signal", "gabes_comparison_v1"]
    metadata = dict(format_version=1, generated_by="GABES", model_id="weak_probe_legacy",
                    frequency_reference=reference, display_reference_offset_hz=offset,
                    experiment_input_mode="fraction", experiment_x_unit="GHz", line="D2")
    if native is not None:
        metadata["experiment_frequency_reference"] = native
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    if origin:
        writer.writerow([key.replace("_", " ") for key in headers[:-1]] + headers[-1:])
        writer.writerows([["GHz", "1", "GHz", "1", ""], ["Comments", "", "", "", ""], []])
    else:
        writer.writerow(headers)
    writer.writerow([0., 1., 0.25, 0.5, json.dumps(metadata)])
    writer.writerow([1., 1., -0.75, 1., ""])
    return output.getvalue().encode("utf-8-sig")


@pytest.mark.parametrize("origin", [False, True])
@pytest.mark.parametrize("isotope,contract,offset", [("Rb87", "Rb85", 78_095_000.0),
                                                    ("Rb85", "Rb85", 0.), ("Cs133", "Cs133", 0.)])
def test_read_v1_and_reexport_v2_without_moving_native_samples(isotope, contract, offset, origin):
    axis = dict(isotope=isotope, line="D2", point="fine_centroid")
    native = dict(isotope="Rb85", line="D2") if isotope == "Rb87" else None
    upload = prepare_measurement_upload(_historical_gabes_csv(axis, offset, origin, native))
    assert upload.metadata["axis_reference"] == axis
    assert upload.metadata["contract_reference"] == dict(axis, isotope=contract)
    assert upload.metadata["display_reference_offset_hz"] == offset
    assert "frequency_reference" not in upload.metadata
    assert upload.metadata.get("experiment_frequency_reference") == native
    trace = load_measurement(upload, denoise=False)
    exported = export_comparison_csv(upload, trace, [0., 1.], [1., 1.],
                                     metadata=upload.metadata, origin=origin)
    assert b"gabes_comparison_v2" in exported.splitlines()[0]
    reopened = prepare_measurement_upload(exported)
    for key in ("axis_reference", "contract_reference", "display_reference_offset_hz", "model_id"):
        assert reopened.metadata[key] == upload.metadata[key]
    assert reopened.metadata.get("experiment_frequency_reference") == native
    assert reopened.payload == upload.payload


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("native,source_line,expected", [
    ("Rb85", "D2", {"isotope": "Rb85", "line": "D2"}),
    ("Rb85", None, {"isotope": "Rb85", "line": None}),
    ({"isotope": "Rb85"}, "D2", {"isotope": "Rb85"}),  # No line was declared in this object.
    (None, "D2", None),
])
def test_changing_theory_line_never_relabels_native_origin(version, native, source_line, expected):
    sample = _historical_gabes_csv("Rb85", 0., False, native=native)
    rows = list(csv.reader(io.StringIO(sample.decode("utf-8-sig"))))
    metadata = json.loads(rows[1][-1])
    if source_line is None:
        del metadata["line"]
    if version == 2:
        rows[0][-1] = "gabes_comparison_v2"
        metadata.pop("frequency_reference")
        metadata.update(format_version=2, axis_reference="Rb85", contract_reference="Rb85")
    rows[1][-1] = json.dumps(metadata)
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    uploaded = prepare_measurement_upload(output.getvalue().encode("utf-8"))
    assert uploaded.metadata.get("experiment_frequency_reference") == expected
    exported = export_comparison_csv(uploaded, load_measurement(uploaded), [0.], [1.],
                                     metadata={**META, "line": "D1"})
    reread = prepare_measurement_upload(exported)
    assert reread.metadata["line"] == "D1"
    assert reread.metadata.get("experiment_frequency_reference") == expected
    assert reread.payload == uploaded.payload


def test_cached_legacy_projection_keeps_original_native_line_on_reexport():
    native_key = "experiment_frequency_reference"
    uploaded = prepare_measurement_upload(paired_csv([-1., 0.], [.5, 1.]))
    uploaded = replace(uploaded, metadata={"line": "D2", native_key: "Rb85"})
    # A caller can copy the original shorthand into its new theory metadata.
    exported = export_comparison_csv(uploaded, load_measurement(uploaded), [0.], [1.],
                                     metadata={**META, "line": "D1", native_key: "Rb85"})
    reopened = prepare_measurement_upload(exported)
    assert reopened.metadata[native_key] == {"isotope": "Rb85", "line": "D2"}


def test_export_accepts_legacy_metadata_argument_without_deriving_native_origin():
    upload = prepare_measurement_upload(paired_csv([-1., 0.], [.5, 1.]))
    legacy = {"model_id": "weak_probe_legacy", "frequency_reference": "Rb87", "line": "D2"}
    exported = export_comparison_csv(upload, load_measurement(upload), [0.], [1.], metadata=legacy)
    assert "axis_reference" not in legacy
    reopened = prepare_measurement_upload(exported).metadata
    assert (reopened["axis_reference"], reopened["contract_reference"]) == ("Rb87", "Rb85")
    assert reopened["display_reference_offset_hz"] == pytest.approx(78_095_000.)
    assert "frequency_reference" not in reopened and "experiment_frequency_reference" not in reopened


def test_reference_fields_and_envelope_version_must_match():
    upload = prepare_measurement_upload(paired_csv([-1., 0.], [.5, 1.]))
    trace = load_measurement(upload)
    for field in ("axis_reference", "contract_reference", "display_reference_offset_hz"):
        with pytest.raises(ValueError, match="both theory references and a finite offset"):
            export_comparison_csv(upload, trace, [0.], [1.],
                                  metadata={k: v for k, v in META.items() if k != field})
    legacy = _historical_gabes_csv("Rb87", 78_095_000., False)
    with pytest.raises(ExperimentalCSVError, match="version"):
        prepare_measurement_upload(legacy.replace(b"gabes_comparison_v1", b"gabes_comparison_v2"))
    for field, value in (("axis_reference", []), ("axis_reference", True), ("axis_reference", "invalid"),
                         ("contract_reference", {"line": "D2"}), ("contract_reference", "Cs133"),
                         ("axis_reference", {"isotope": "Rb85", "line": "D1"}),
                         ("axis_reference", {"isotope": "Rb85", "point": "lower_ground_centroid"}),
                         ("display_reference_offset_hz", True), ("display_reference_offset_hz", 1.0),
                         ("display_reference_offset_hz", float("inf")),
                         ("display_reference_offset_hz", 10**309)):
        with pytest.raises(ValueError, match="both theory references and a finite offset"):
            export_comparison_csv(upload, trace, [0.], [1.], metadata={**META, field: value})


@pytest.mark.parametrize("extra", [{"display_reference_offset_hz": 10**309},
                                    {"axis_reference": ["Rb85"]}, {"frequency_reference": "Rb87"}])
def test_v2_reader_rejects_malformed_references_even_with_real_native_pairs(extra):
    sample = _historical_gabes_csv("Rb85", 0., False)
    rows = list(csv.reader(io.StringIO(sample.decode("utf-8-sig"))))
    rows[0][-1] = "gabes_comparison_v2"
    metadata = json.loads(rows[1][-1])
    del metadata["frequency_reference"]
    metadata.update(format_version=2, axis_reference="Rb85", contract_reference="Rb85")
    rows[1][-1] = json.dumps({**metadata, **extra})
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    with pytest.raises(ExperimentalCSVError, match="Invalid v2 comparison theory references or offset"):
        prepare_measurement_upload(output.getvalue().encode("utf-8"))


def test_v1_custom_reference_remains_readable_without_guessing_a_contract():
    upload = prepare_measurement_upload(_historical_gabes_csv("old instrument setting", 0., False))
    assert upload.metadata["axis_reference"] == "old instrument setting"
    assert "contract_reference" not in upload.metadata
    assert "experiment_frequency_reference" not in upload.metadata


@pytest.mark.parametrize("reverse", [False, True])
def test_export_leaves_open_gap_blank_in_mhz_input(reverse):
    x = np.r_[np.linspace(-3000, -2500, 100), np.linspace(2500, 3000, 100)]
    upload = prepare_measurement_upload(paired_csv(x, np.repeat([.2, .8], 100),
                                                  header=("frequency_MHz", "transmission")))
    trace = load_measurement(upload, denoise=False)
    assert gap_slices(trace) == (slice(0, 100), slice(100, 200))
    # AutoOD 0.2.2 default natural-Rb D2 axis at 90 C, 12.5 mm.
    tx = np.linspace(-4.706687420589813, 6.2687850735898145, 2001)
    ty = np.ones_like(tx)
    encoded = export_comparison_csv(upload, trace, tx, ty, metadata=META,
                                    alignment=AxisAlignment(reverse=reverse))
    rows = list(csv.DictReader(io.StringIO(encoded.decode("utf-8-sig"))))
    gap = [r for r in rows if -2.5 < float(r["detuning_GHz"]) < 2.5]
    assert len(gap) == 911
    assert all(row["experiment_display_transmission_on_theory_grid"] == "" for row in gap)
    restored = load_measurement(prepare_measurement_upload(encoded))
    np.testing.assert_array_equal(restored.detuning, trace.detuning)


def test_relative_quick_candidate_and_periodic_interference():
    tx, ty = theory()
    x = np.linspace(0., 1., 3841)
    for accepted, y in ((False, .5 + .45 * np.cos(2 * np.pi * 385 * x)),
                        (True, .7 + 3 * np.interp(-3.9 + x * 8.5, tx, ty))):
        upload = prepare_measurement_upload(paired_csv(x, y, header=("x", "detector")))
        trace = load_measurement(upload, denoise=False)
        result = quick_measurement_alignment(trace, tx, ty)
        assert result.matched == accepted
        assert bool(result.candidates) == accepted
        with pytest.raises(ValueError, match="absolute transmission"):
            prepare_od_fit(trace, result.alignment)
    data = export_comparison_csv(upload, trace, tx, ty, metadata=META, alignment=result.alignment)
    saved = prepare_measurement_upload(data)
    assert saved.mode == "relative"
    assert saved.metadata["experiment_display_scaling"] == "display_only"
    assert not load_measurement(saved).is_absolute_transmission


def test_raw_detector_calibration_is_separate_from_display_and_restored():
    x = np.linspace(.1, 1., 501)
    upload = prepare_measurement_upload(paired_csv(x, 1 + 2 * x, header=("frequency_GHz", "detector volts")))
    trace = load_measurement(upload, calibration_mode="dark_reference", dark_signal=1.,
                             reference_signal=3., denoise=False)
    data = prepare_od_fit(trace, AxisAlignment())
    assert data.source_points == 501
    np.testing.assert_allclose(data.transmission, x, atol=1e-15)
    saved = export_comparison_csv(upload, trace, *theory(), metadata=META,
                                  alignment=AxisAlignment(signal_scale=1.3, signal_offset=.2))
    restored = load_measurement(prepare_measurement_upload(saved), denoise=False)
    np.testing.assert_array_equal(restored.branches[0].transmission, trace.branches[0].transmission)
    assert restored.correction.calibration_source == trace.correction.calibration_source


@pytest.mark.parametrize("origin", [False, True])
def test_large_autood_worksheet_keeps_only_native_pairs_and_provenance(origin):
    # This represents AutoOD's sparse rectangular worksheet (metadata on its first row).
    n = 180000
    indices = np.arange(n)
    x = np.where(indices < n // 2, indices, n - 1 - indices)
    y = np.where(indices < n // 2, .8, .2)
    header = ["detuning_ghz", "theory_transmission", "experiment_input_x", "experiment_input_signal",
              "experiment_input_mode", "experiment_x_unit", "autood_version", "gabes_commit",
              "frequency_reference", "model", "experiment_display_transmission"]
    out = io.StringIO(newline="")
    w = csv.writer(out)
    if origin:
        w.writerow(["Detuning from Rb-85 D2 centroid", "Theory transmission", "Experiment input X",
                    "Experiment input signal", *header[4:]])
        w.writerow(["GHz", "1", "GHz", "1", *([""] * 7)])
        w.writerow(["Comments", *([""] * 10)])
        w.writerow([""] * len(header))
    else:
        w.writerow(header)
    for i, (frequency, transmission) in enumerate(zip(x, y, strict=True)):
        labels = ["fraction", "GHz", "0.2.2", "8846193", "Rb85", META["model_id"]] if i == 0 else [""] * 6
        w.writerow(["", "", frequency, transmission, *labels, f"{transmission:.48f}"])
    payload = out.getvalue().encode("utf-8-sig")
    assert MAX_FILE_BYTES < len(payload) < 16 * 1024**2
    restored = prepare_measurement_upload(payload)
    assert restored.sha256 == hashlib.sha256(payload).hexdigest()
    assert restored.metadata["frequency_reference"] == "Rb85"
    assert len(restored.payload) < 4_000_000
    trace = load_measurement(restored, denoise=False)
    assert len(trace.branches) == 2
    np.testing.assert_array_equal(np.concatenate([b.detuning for b in trace.branches]), x)
    np.testing.assert_array_equal(np.concatenate([b.raw_signal for b in trace.branches]), y)
    exported = export_comparison_csv(restored, trace, *theory(), metadata=META, origin=origin)
    again = load_measurement(prepare_measurement_upload(exported), denoise=False)
    for old, new in zip(trace.branches, again.branches, strict=True):
        assert (new.start_index, new.stop_index) == (old.start_index, old.stop_index)
        np.testing.assert_array_equal(new.raw_signal, old.raw_signal)
        np.testing.assert_array_equal(new.detuning, old.detuning)


def test_large_ordinary_file_and_malformed_comparison():
    ordinary = b"x,y\n" + b"1,2\n" * (MAX_FILE_BYTES // 4)
    with pytest.raises(ExperimentalCSVError, match="10 MiB"):
        prepare_measurement_upload(ordinary)
    with pytest.raises(ExperimentalCSVError, match="10 MiB"):
        load_experimental_csv(ordinary)
    upload = prepare_measurement_upload(paired_csv([0, 1], [.2, .9]))
    exported = export_comparison_csv(upload, load_measurement(upload), *theory(), metadata=META)
    header = exported.splitlines(keepends=True)[0]
    for line in (b"0" * 8193, b"0,1\n", b"0,1,2,3,4,5,6,7,{}\n"):
        with pytest.raises(ExperimentalCSVError):
            prepare_measurement_upload(header + line)


def test_scope_choice_keeps_only_selected_finite_acquisition_rows():
    header = "time,CH1,CH2,CH3\n"
    payload = (header + "".join(f"{i},{i},{2*i},{i+3}\n" for i in range(200))).encode()
    with pytest.raises(ScopeChannelChoice) as prompt:
        prepare_measurement_upload(payload)
    names = prompt.value.channels
    assert names == ("CH1", "CH2", "CH3")
    selected = prepare_measurement_upload(payload, scope_channel="CH3", scope_source_id=prompt.value.source_id)
    trace = load_measurement(selected, denoise=False)
    assert selected.scope.manual
    np.testing.assert_array_equal(trace.branches[0].raw_signal, np.arange(200) + 3)
    with pytest.raises(ScopeChannelChoice):
        prepare_measurement_upload(payload + b"200,200,400,203\n", scope_channel="CH3",
                                   scope_source_id=prompt.value.source_id)


def test_calibrated_501_point_partial_d1_keeps_physical_frequency_and_recovers_temperature():
    from gabes.od_spectrum_fit import fit_od_spectrum, od_transmission
    from gabes.schemes.sas import SASScheme
    params = {**SASScheme().defaults(), "species": "Rb (natural)", "line": "D1", "temp_c": 50.,
              "pump_power_mw": 0., "probe_power_uw": 0., "cell_mm": 12.5}
    x = np.linspace(-2.5, .3, 501)
    t = od_transmission(x * 1.e3, params)
    upload = prepare_measurement_upload(paired_csv(x, t))
    data = prepare_od_fit(load_measurement(upload, denoise=False), AxisAlignment())
    result = fit_od_spectrum(data, params, free=("temp_c",))
    assert data.source_points == 501
    np.testing.assert_allclose(data.frequency_mhz, x * 1.e3, atol=1.e-12)
    assert result["fit_residuals"]["rms"] < 1.e-12
    assert result["parameters"][0]["value"] == pytest.approx(50., abs=1.e-8)


def test_public_display_affine_accepts_samples_and_rejects_wrong_polarity():
    from gabes.experimental_alignment import fit_signal_affine
    assert fit_signal_affine([0, 1, 2], [1, 4, 7]) == pytest.approx((3, 1))
    assert fit_signal_affine([0, 1, 2], [7, 4, 1]) is None


def _scope_waveforms():
    gate_start, gate_stop = 260, 1340
    index = np.arange(1600)
    # Rounded display timestamps repeat at adjacent native acquisition samples.
    times = (index // 2 - 400) * 0.0001
    coordinate = (index - gate_start) / (gate_stop - gate_start)
    rng = np.random.default_rng(31)
    trigger = np.where((index >= gate_start) & (index < gate_stop), 0.4, 4.5)
    trigger += rng.normal(0, 0.006, len(trigger))
    broad = np.exp(-0.5 * ((coordinate - 0.37) / 0.11) ** 2)
    second = np.exp(-0.5 * ((coordinate - 0.83) / 0.10) ** 2)
    od = 0.011 - 0.009 * broad - 0.006 * second
    od += 0.0003 * coordinate + rng.normal(0, 0.000025, len(od))
    sas = 0.53 - 0.07 * broad - 0.04 * second
    for center, height in ((0.33, 0.019), (0.37, 0.026), (0.41, 0.014), (0.82, 0.010)):
        sas += height / (1 + ((coordinate - center) / 0.003) ** 2)
    sas += rng.normal(0, 0.0002, len(sas))
    return times, {"od": od, "sas": sas, "trigger": trigger}


def test_automatic_scope_keeps_od_gate_and_acquisition_provenance():
    time, signals = _scope_waveforms()
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(("time", "CH1", "CH2", "CH3"))
    writer.writerows(zip(time, signals["trigger"], signals["od"], signals["sas"], strict=True))
    upload = prepare_measurement_upload(text.getvalue().encode())
    assert (upload.scope.trigger, upload.scope.od, upload.scope.sas) == (0, 1, 2)
    assert (upload.scope.start, upload.scope.stop) == (422, 1178)
    trace = load_measurement(upload, denoise=False)
    np.testing.assert_array_equal(trace.raw_signal, signals["od"][422:1178])
    reopened = prepare_measurement_upload(export_comparison_csv(upload, trace, *theory(), metadata=META))
    assert reopened.metadata["scope"]["selected_samples"] == [422, 1178]
    assert reopened.metadata["scope"]["source_id"] == upload.sha256
