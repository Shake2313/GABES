"""Tabulated Fast absorption versus independent full-resolution trajectory solves."""
from dataclasses import replace

import numpy as np
import pytest

from gabes import _ballistic_reference_table as table
from gabes import ballistic_spectrum as history, beam, species
from gabes.od_model import ODConditions, evaluate_od_display
from gabes.schemes.sas import BALLISTIC_FAST


def conditions(temperature_c):
    return ODConditions(species.RB85.label, "D1", temperature_c, 12.5,
                        BALLISTIC_FAST, probe_waist_mm=0.53, probe_power_uw=100.)


def inputs(temperature_c):
    optics = history._optics(conditions(temperature_c))
    peak = beam.peak_intensity(160e-6, optics.waist)
    intensity = history._intensity_nodes(peak, history._REFERENCE)
    nodes, weights = history._panels(optics.width, history._REFERENCE.panel_order,
                                     history._REFERENCE.wing_widths)
    return optics, intensity, nodes, weights


@pytest.mark.parametrize(("temperature_c", "method"), [(99., "exact knot"), (99.25, "local cubic")])
def test_shipped_rest_rows_match_dense_trajectory_solutions(temperature_c, method):
    optics, intensity, nodes, weights = inputs(temperature_c)
    rest, provenance = table.lookup(optics, intensity, nodes, weights)
    assert provenance.get("rest_method") == "tabulated reference trajectories", provenance
    assert provenance["rest_temperature_interpolation"] == method
    knots = provenance["rest_temperature_nodes_c"]
    if method == "exact knot":
        assert knots == [temperature_c]
    else:
        assert len(knots) == 4 and knots[0] < temperature_c < knots[-1]
        assert temperature_c not in knots

    atom = history._atom(optics.temperature, optics.width)
    grid = history._path_grid(history._REFERENCE)
    centers = np.unique(atom.manifold.omega)
    indices = [0, *(np.argmin(abs(nodes - center)) for center in centers), len(nodes) - 1]
    selected = nodes[indices]
    weak = atom.weak_cross_sections(selected)
    for row in (0, 9, 12):
        for index, frequency, weak_cross_section in zip(indices, selected, weak):
            # Independent dense solve of the full 14-state system, including
            # the same finer time step when a resonant path loses positivity.
            mesh = grid
            while True:
                try:
                    direct = atom.cross_sections([frequency], [intensity[row]], optics.waist,
                                                 grid=mesh, accelerated=False)[0, 0]
                    break
                except ArithmeticError as error:
                    assert "Unresolved ballistic populations" in str(error)
                    assert mesh.nx < 16 * grid.nx
                    mesh = replace(mesh, nx=2 * mesh.nx)
            expected = direct - mesh.weak_factor() * weak_cross_section
            tolerance = 2e-12 if method == "exact knot" else 5e-7
            assert abs(rest[row, index] - expected) <= tolerance * weak_cross_section


def test_public_fit_axis_uses_actual_temperature_and_preserves_order():
    c = conditions(99.25)
    scan = 2*np.pi*1e9*np.array([1.2, -1.3, 1.2, 0.])
    spectrum = evaluate_od_display(c, scan)
    assert spectrum.diagnostics["rest_method"] == "tabulated reference trajectories"
    assert spectrum.diagnostics["rest_temperature_interpolation"] == "local cubic"
    assert spectrum.transmission[0] == spectrum.transmission[2]
    weak = evaluate_od_display(replace(c, probe_power_uw=0.), scan)
    np.testing.assert_array_equal(spectrum.od0, weak.od0)
    assert np.max(np.abs(spectrum.transmission - weak.transmission)) > 1e-3
    assert "rest_method" not in weak.diagnostics


def test_nonzero_microwatt_limit_propagates_and_approaches_analytical_weak_spectrum():
    # At 99 C this first offset has the largest weak-limit difference in the
    # original 2,001-point reference scan (6.887e-8 in transmission).
    scan = 2*np.pi*1e9*np.array([-0.4581963306552205, 0., 1.2])
    c = conditions(99.)
    weak = evaluate_od_display(replace(c, probe_power_uw=0.), scan)
    finite = evaluate_od_display(replace(c, probe_power_uw=1e-6), scan)
    assert finite.diagnostics["rest_method"] == "tabulated reference trajectories"
    assert finite.diagnostics["intensity_rows"] > 0 and weak.diagnostics["intensity_rows"] == 0
    np.testing.assert_array_equal(finite.od0, weak.od0)
    difference = finite.transmission - weak.transmission
    assert difference[0] == pytest.approx(6.886703390884819e-8, rel=0., abs=1e-12)
    assert np.max(np.abs(difference)) < 1e-7


def test_withheld_detector_floor_uses_reference_trajectory_classification():
    # The reduced 64-step/10-speed mesh returned 0.00100050508479 here.
    # A direct full-resolution solve put this wing below the detector floor.
    scan = 2*np.pi*1e6*np.array([2185.931699189985])
    result = evaluate_od_display(conditions(95.25), scan)
    assert result.diagnostics["rest_method"] == "tabulated reference trajectories"
    assert result.diagnostics["rest_temperature_interpolation"] == "local cubic"
    assert result.transmission[0] < 0.001
    assert result.transmission[0] == pytest.approx(0.0009998546550342572, rel=0., abs=1e-7)


def test_lookup_checks_geometry_and_quadrature_before_reusing_coefficients():
    optics, intensity, nodes, weights = inputs(99.25)
    # Different libm/SIMD implementations may round the same exp() knot one ULP apart.
    expected, _ = table.lookup(optics, intensity, nodes, weights)
    rounded = np.nextafter(intensity, np.inf)
    equivalent, provenance = table.lookup(optics, rounded, nodes, weights)
    assert provenance.get("rest_method") == "tabulated reference trajectories", provenance
    np.testing.assert_array_equal(equivalent, expected)
    roundtripped = replace(conditions(99.25), probe_waist_mm=np.nextafter(0.53, np.inf))
    rounded = history.evaluate_ballistic_spectrum(roundtripped, nodes[:1], fidelity="fast")
    assert rounded.diagnostics["rest_method"] == "tabulated reference trajectories"
    for query in (
        (replace(optics, waist=optics.waist * 1.05), intensity, nodes, weights),
        (optics, intensity * 1.01, nodes, weights),
        (optics, intensity, nodes + 1e6, weights),
        (optics, intensity, nodes, weights * 1.01),
    ):
        rest, details = table.lookup(*query)
        assert rest is None and details["rest_table_fallback_reason"]


def test_missing_corrupt_and_stale_tables_are_rejected(monkeypatch, tmp_path):
    from gabes import _ballistic_reference_table as contract

    shipped = table._TABLE
    absent = tmp_path / "missing.npz"
    monkeypatch.setattr(table, "_TABLE", absent)
    # Exercise the file loader without retaining these deliberately invalid tables
    # in the process cache used by a later test.
    load = table._load_table.__wrapped__
    data, reason = load()
    assert data is None and "unavailable" in reason
    absent.write_bytes(b"incomplete")
    data, reason = load()
    assert data is None and "checksum" in reason
    monkeypatch.setattr(table, "_TABLE", shipped)
    monkeypatch.setattr(contract, "source_signature", lambda: "different direct solver")
    data, reason = load()
    assert data is None and "model" in reason


def test_public_fast_reverts_to_full_reference_and_direct_reference_never_looks_up_table(monkeypatch):
    probes = []
    meshes = []

    def unavailable(*args):
        probes.append(True)
        return None, {"rest_table_fallback_reason": "Table unavailable for this test."}

    def direct(temperature, width, waist, intensity, resolution):
        meshes.append(resolution)
        nodes, _ = history._panels(width, resolution.panel_order, resolution.wing_widths)
        return np.zeros_like(nodes), 0, resolution.path_nodes[0]

    monkeypatch.setattr(table, "lookup", unavailable)
    monkeypatch.setattr(history, "_rest_excess", direct)
    scan = 2*np.pi*np.array([-1.1e9, 0., 1.9e9])
    c = conditions(99.)
    fast = history.evaluate_ballistic_spectrum(c, scan, fidelity="fast")
    assert probes and meshes and all(mesh is history._REFERENCE for mesh in meshes)
    assert fast.diagnostics["fidelity_requested"] == "fast"
    assert fast.diagnostics["fidelity_used"] == "reference"
    assert "Table unavailable" in fast.diagnostics["rest_table_fallback_reason"]

    probes.clear()
    meshes.clear()
    # Explicit v1 in an older recording must replay its original reduced mesh.
    legacy_id = "od_rb85_d1_ballistic_history_fast_v1"
    assert c.model_id == "od_rb85_d1_ballistic_history_fast_v2"
    saved = evaluate_od_display(replace(c, model_id=legacy_id), scan)
    assert saved.model_id == legacy_id
    assert not probes and meshes and all(mesh is history._FAST for mesh in meshes)
    meshes.clear()
    outside = replace(c, probe_waist_mm=1.)
    fallback = history.evaluate_ballistic_spectrum(outside, scan, fidelity="fast")
    reference = history.evaluate_ballistic_spectrum(outside, scan, fidelity="reference")
    assert not probes and meshes and all(mesh is history._REFERENCE for mesh in meshes)
    assert fallback.diagnostics["fidelity_used"] == "reference"
    for name in ("transmission", "optical_depth", "od0"):
        np.testing.assert_array_equal(getattr(fallback, name), getattr(reference, name))
