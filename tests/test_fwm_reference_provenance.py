"""Portable checks of the source-linked FWM literature evidence views."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "references" / "fwm_experiments"


@pytest.fixture(scope="module")
def evidence():
    return json.loads((ROOT / "database.json").read_text(encoding="utf-8"))


def experiment(evidence, paper_id, experiment_id):
    paper = next(p for p in evidence["papers"] if p["paper_id"] == paper_id)
    return next(e for e in paper["experiments"] if e["experiment_id"] == experiment_id)


def quantities(record):
    return {q["quantity"]: q for q in record["conditions"] + record["results"]}


def test_legacy_input_is_preserved_and_audit_links_resolve(evidence):
    audit = evidence["legacy_csv_audit"]
    assert audit == json.loads((ROOT / "legacy_csv_audit.json").read_text(encoding="utf-8"))
    csv_path = ROOT / audit["legacy_source"]["path"]
    pinned_hash = "028bd42e3d04f9811594312a24abe1fdeac67562f4a2e3f378f4a6a1731315b1"
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == audit["legacy_source"]["sha256"] == pinned_hash
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        legacy_columns = set(rows.fieldnames[5:])
        row_ids = {row["row_id"] for row in rows}

    papers = {p["paper_id"]: p for p in evidence["papers"] if p.get("legacy_column") in legacy_columns}
    assert len(papers) == len(legacy_columns) == 10
    assert audit["findings"].keys() == papers.keys()
    for paper_id, findings in audit["findings"].items():
        paper = papers[paper_id]
        assert findings
        for finding in findings:
            assert finding["legacy_rows"] and set(finding["legacy_rows"]) <= row_ids
            assert finding["experiment_ids"]
            for experiment_id in finding["experiment_ids"]:
                assert experiment(evidence, paper_id, experiment_id)
            source = next(s for s in paper["sources"] if s["source_id"] == finding["source_id"])
            assert source["url"] and source["version"] and finding["locator"]
            assert isinstance(finding["unresolved"], list)


def test_wu_output_observations_stay_with_their_own_seed(evidence):
    paper_id = "wu_2019_oe"
    with (ROOT / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
        observations = [row for row in csv.DictReader(stream) if row["paper_id"] == paper_id]

    for experiment_id, seed, output in (("fig2_high_seed", 6.4, 129), ("fig2_low_seed", 0.9, 17)):
        record = experiment(evidence, paper_id, experiment_id)
        q = quantities(record)
        assert record["seed_mode"] == "single_seed"
        assert q["seed_power"]["value"] == seed
        assert q["output_power_unspecified_arm"]["value"] == output
        assert q["detection_efficiency"]["value_type"] == "not_reported"
        assert "output_probe_power" not in q and "output_conjugate_power" not in q
        matching = [row for row in observations if row["experiment_id"] == experiment_id
                    and row["quantity"] == "output_power_unspecified_arm"]
        assert len(matching) == 1
        assert float(matching[0]["value"]) == output
        assert "not specified" in matching[0]["plane"]
        assert matching[0]["source_url"].endswith("/1810.09589")
        assert "Fig. 2" in matching[0]["locator"]


def test_comparison_contexts_preserve_measurement_classes(evidence):
    papers = {p["paper_id"]: p for p in evidence["papers"]}
    contexts = evidence["legacy_csv_audit"]["comparison_contexts"]
    expected = {
        "single_seed_direct_ids": ("intensity_difference", "single_seed", None),
        "dual_seed_ids": ("intensity_difference", "dual_seed", None),
        "vacuum_seeded_homodyne": ("quadrature_homodyne", "vacuum_seed", None),
        "before_fiber_ids": ("intensity_difference", "single_seed", "free_space"),
        "fiber_output_ids": ("intensity_difference", "single_seed", "fiber_output"),
    }
    assert {c["measurement_class"] for c in contexts} == expected.keys()
    for c in contexts:
        p = papers[c["paper_id"]]
        e = experiment(evidence, c["paper_id"], c["experiment_id"])
        scope = p["scope"]
        assert (scope["measurement"], e.get("seed_mode", scope["seed_mode"]),
                e.get("measurement_path")) == expected[c["measurement_class"]]


def test_source_numbers_and_derived_quantities_are_distinct(evidence):
    direct = quantities(experiment(evidence, "sim_2025_scirep", "fig3a_rb85_best_ids"))
    gain_scan = quantities(experiment(evidence, "sim_2025_scirep", "fig2_rb85_gain_scan"))
    slope = quantities(experiment(evidence, "sim_2025_scirep", "fig3b_power_slope"))
    assert (direct["probe_gain"]["value"], gain_scan["probe_gain"]["value"]) == (15, 16)
    assert (direct["seed_power"]["value"], gain_scan["seed_power"]["value"]) == (8, 10)
    assert direct["probe_output_seed_ratio"]["value"] == 111 / 8
    assert direct["conjugate_output_seed_ratio"]["value"] == 109 / 8
    assert direct["detection_efficiency"]["value"] == 86.5
    assert (direct["squeezing_measured"]["value"], slope["squeezing_slope"]["value"]) == (-7.8, -8.7)

    before = quantities(experiment(evidence, "jain_2025_ol", "before_fiber"))
    after = quantities(experiment(evidence, "jain_2025_ol", "after_fiber"))
    expected_db = 10 * math.log10(0.9 * 10 ** (before["squeezing_measured"]["value"] / 10) + 0.1)
    assert after["squeezing_expected_reported"]["value"] == -6.3
    assert after["squeezing_expected_vacuum_loss"]["value"] == pytest.approx(expected_db, abs=5e-6)
    assert after["squeezing_expected_vacuum_loss"]["value_type"] == "derived"
    assert after["squeezing_measured"]["value"] == -4.4

    mccormick = quantities(experiment(evidence, "mccormick_2007_ol", "red_seed_fig2"))
    assert mccormick["cell_length"]["value"] == 9
    assert mccormick["detection_efficiency"]["value"] == pytest.approx(65.6)

    monsa = quantities(experiment(evidence, "monsa_2026_preprint", "fig6_slope"))
    assert monsa["squeezing_slope"]["value"] == -8
    assert monsa["squeezing_slope_computed"]["value"] == pytest.approx(10 * math.log10(0.135))


def test_gold_receipts_keep_distinct_targets_and_exclude_noise(evidence):
    fits = {f["fit_class"]: f for f in evidence["legacy_csv_audit"]["historical_gold_fits"]}
    gain_fit = fits["gain_only"]
    mean_fit = fits["mean_output_only"]
    assert [(q["quantity"], q["value"]) for q in gain_fit["targets"]] == [("probe_gain", 15)]
    assert [(q["quantity"], q["value"]) for q in mean_fit["targets"]] == [
        ("output_probe_power", 111), ("output_conjugate_power", 109),
    ]
    assert gain_fit["source_url"] != mean_fit["source_url"]
    assert "conjugate_gain" in gain_fit["excluded"]
    assert all("squeezing_measured" in f["excluded"] and "squeezing_slope" in f["excluded"]
               for f in fits.values())
