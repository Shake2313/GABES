"""Atomic cache reuse requires matching numerical dependencies and acceptance scope."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gabes_lab.pipelines import beam_self_action as pipeline


WAVELENGTH_M = 795e-9


class _Run:
    def __init__(self, directory):
        self.directory = directory
        self.directory.mkdir()

    def path(self, name):
        return self.directory / name


@pytest.fixture
def trusted_cache(tmp_path, monkeypatch):
    calls = []

    def mean(detuning_mhz, intensity_w_m2, **kwargs):
        calls.append((detuning_mhz, intensity_w_m2, kwargs))
        # Affine passive quadratures interpolate exactly. The fixture does not
        # approximate OBE physics; it exercises cache and acceptance bookkeeping.
        sign = np.sign(detuning_mhz)
        chi = sign * (-2e-6 + 2e-12 * intensity_w_m2)
        chi += 1j * (1e-6 - 1e-12 * intensity_w_m2)
        return dict(chi=complex(chi), class_count=2,
                    diagnostics=dict(trace_error=0., hermiticity_error=0.,
                                     min_eigenvalue=0., steady_state_relative_residual=0.),
                    omitted_maxwell_probability=2e-9)

    monkeypatch.setattr(pipeline, "solve_rb87_d1_mean", mean)
    args = pipeline.parse_args([])
    args.table_nodes = 5
    run = _Run(tmp_path / "original")
    tables, checks = pipeline.make_tables(args, run, WAVELENGTH_M)
    assert checks["passed"]
    assert calls
    calls.clear()
    return args, run, tables, checks, calls


def _reuse_args(args, run, **changes):
    return SimpleNamespace(**dict(vars(args),
        reuse_tables=run.path("susceptibility_tables.npz"), **changes))


def test_matching_audit_cache_reuses_without_any_atomic_solve(tmp_path, trusted_cache):
    args, original, original_tables, original_checks, calls = trusted_cache
    run = _Run(tmp_path / "same")
    tables, checks = pipeline.make_tables(_reuse_args(args, original), run, WAVELENGTH_M)
    assert calls == []
    assert checks == original_checks
    for label in ("red", "blue", "red_refined", "blue_refined"):
        np.testing.assert_array_equal(tables[label].intensity_w_m2,
                                      original_tables[label].intensity_w_m2)
        np.testing.assert_array_equal(tables[label].chi, original_tables[label].chi)
    node_health = json.loads(run.path("atomic_nodes.json").read_text(encoding="utf-8"))
    for label in ("red", "blue"):
        assert node_health[label]["reused"]
        assert node_health[label]["node_health_available"]
        assert len(node_health[label]["nodes"]) == args.table_nodes
    with np.load(run.path("atomic_refinement.npz"), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    assert metadata["cell_length_mm"] == args.cell_length_mm
    assert metadata["wavelength_m"] == WAVELENGTH_M


@pytest.mark.parametrize("changed", ["cell_length", "wavelength"])
def test_changed_phase_scale_rechecks_audit_but_reuses_atomic_table(
        tmp_path, trusted_cache, changed):
    args, original, original_tables, _, calls = trusted_cache
    changes = {"cell_length_mm": 2 * args.cell_length_mm} if changed == "cell_length" else {}
    wavelength = 2 * WAVELENGTH_M if changed == "wavelength" else WAVELENGTH_M
    reused_args = _reuse_args(args, original, **changes)
    run = _Run(tmp_path / "changed_scope")
    tables, checks = pipeline.make_tables(reused_args, run, wavelength)
    assert checks["passed"]
    # Two detunings, every held-out midpoint, plus 8 intensity probes at
    # baseline/order/cutoff settings. No original table-node solves recur.
    assert len(calls) == 2 * ((args.table_nodes - 1) + 8 * 3)
    health = json.loads(run.path("atomic_nodes.json").read_text(encoding="utf-8"))
    for label in ("red", "blue"):
        assert health[label]["reused"]
        np.testing.assert_array_equal(tables[label].chi, original_tables[label].chi)
    with np.load(run.path("atomic_refinement.npz"), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    assert metadata["cell_length_mm"] == reused_args.cell_length_mm
    assert metadata["wavelength_m"] == wavelength


@pytest.mark.parametrize("library", ["numpy", "scipy"])
def test_changed_numerical_library_rejects_raw_table_cache(
        tmp_path, trusted_cache, monkeypatch, library):
    args, original, _, _, calls = trusted_cache
    module = pipeline.np if library == "numpy" else pipeline.scipy
    monkeypatch.setattr(module, "__version__", module.__version__ + "+changed")
    run = _Run(tmp_path / "changed_library")
    with pytest.raises(ValueError, match="cached table settings/source hashes differ"):
        pipeline.make_tables(_reuse_args(args, original), run, WAVELENGTH_M)
    assert calls == []
    assert not run.path("susceptibility_tables.npz").exists()
