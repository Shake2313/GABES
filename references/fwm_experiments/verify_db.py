"""Focused DB checks: source PDFs, schema, exports and invalid-data guards."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter

from build_db import ROOT, UNIT_ALIASES, quantities, read_records, readiness, validate


def main():
    papers = read_records()
    checks = []

    def check(name, okay, detail=None):
        checks.append({"id": name, "status": "PASS" if okay else "FAIL", "detail": detail})

    receipt = validate(papers)
    check("canonical_structure_sources_and_legacy", receipt["status"] == "PASS", receipt)

    import jsonschema
    validator = jsonschema.Draft202012Validator(json.loads((ROOT / "schema.json").read_text(encoding="utf-8")))
    schema_errors = [f"{p['paper_id']}: {e.message}" for p in papers for e in validator.iter_errors(p)]
    check("formal_json_schema", not schema_errors, schema_errors)

    import pymupdf
    source_paths = {s["path"]: s for p in papers for s in p["sources"] if s.get("path")}
    page_errors = []
    for path, source in source_paths.items():
        if path.lower().endswith(".pdf"):
            with pymupdf.open(ROOT / path) as document:
                if source.get("pages") != len(document):
                    page_errors.append({"path": path, "recorded": source.get("pages"), "actual": len(document)})
    check("independent_pdf_open_and_page_counts", not page_errors,
          {"unique_source_files": len(source_paths), "errors": page_errors})

    payload = json.loads((ROOT / "database.json").read_text(encoding="utf-8"))
    check("combined_json_lossless", payload["papers"] == papers)
    counts = {"papers.csv": len(papers), "experiments.csv": sum(len(p["experiments"]) for p in papers),
              "observations.csv": sum(1 for p in papers for e in p["experiments"] for _ in quantities(e))}
    tables = {}
    for name, expected in counts.items():
        with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
        check(name + "_row_count", len(tables[name]) == expected, {"expected": expected, "actual": len(tables[name])})
    rows = {r["observation_id"]: r for r in tables["observations.csv"]}
    roundtrip_errors = []
    for p in papers:
        for e in p["experiments"]:
            for role, index, q in quantities(e):
                row = rows.get(f"{p['paper_id']}/{e['experiment_id']}/{role}/{index}", {})
                expected = "" if q.get("value") is None else str(q["value"])
                if row.get("value") != expected or row.get("quantity") != q["quantity"]:
                    roundtrip_errors.append(row.get("observation_id"))
                if row.get("normalized_unit") != UNIT_ALIASES.get(q["unit"], q["unit"]):
                    roundtrip_errors.append("unit: " + row.get("observation_id", ""))
    check("observations_numeric_and_unit_roundtrip", not roundtrip_errors, roundtrip_errors)

    stale = []
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for path, digest in manifest["files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            stale.append(path)
    check("generated_manifest_fresh", not stale, stale)

    # Deliberately corrupt independent copies. A passing nominal record alone
    # would not show that the validator rejects dangerous ingestion errors.
    mutations = []
    bad = copy.deepcopy(papers)
    bad.append(copy.deepcopy(bad[0]))
    mutations.append(("duplicate_ids_rejected", bad, "duplicate"))
    for name, change, message in (
        ("source_checksum_rejected", lambda p: p[0]["sources"][0].update(sha256="0" * 64), "checksum mismatch"),
        ("unknown_source_rejected", lambda p: next(q for _, _, q in quantities(p[0]["experiments"][0]) if q["value_type"] != "not_reported").update(source_id="__invalid__"), "unknown source"),
        ("missing_numeric_value_rejected", lambda p: next(q for _, _, q in quantities(p[0]["experiments"][0]) if q["value_type"] == "reported").update(value=None), "absent value"),
        ("zero_is_not_missing", lambda p: next(q for _, _, q in quantities(p[0]["experiments"][0])).update(value_type="not_reported", value=0), "must be null"),
        ("derived_formula_required", lambda p: next(q for _, _, q in quantities(p[0]["experiments"][0])).update(value_type="derived", derivation={}), "lacks formula"),
        ("reversed_interval_rejected", lambda p: next(q for _, _, q in quantities(p[0]["experiments"][0])).update(value_type="range", value=None, lower=2, upper=1), "invalid interval"),
    ):
        bad = copy.deepcopy(papers)
        change(bad)
        mutations.append((name, bad, message))
    for name, bad, expected in mutations:
        result = validate(bad)
        check(name, result["status"] == "FAIL" and any(expected in s for s in result["errors"]))

    evaluated = [readiness(p, e) for p in papers for e in p["experiments"]]
    check("no_fit_or_untouched_holdout_claims", all(not e["fit_accepted"] and e["holdout_status"] == "inspected_literature_pool" for e in evaluated))
    wu = next(p for p in papers if p["paper_id"] == "wu_2019_oe")
    single = [readiness(wu, e)["single_seed_operating_point"] for e in wu["experiments"] if e.get("seed_mode") == "single_seed"]
    dual = [readiness(wu, e)["single_seed_operating_point"] for e in wu["experiments"] if e.get("seed_mode") == "dual_seed"]
    check("single_and_dual_seed_not_merged", bool(single) and all(single) and not any(dual))

    result = {"date": "2026-10-23", "origin": "AI focused data verification", "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
              "checks": checks, "value_type_counts": dict(Counter(q["value_type"] for p in papers for e in p["experiments"] for _, _, q in quantities(e))),
              "limits": "No simulation, fit, untouched prediction or exhaustive numerical digitization is tested."}
    (ROOT / "validation_detail.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [c["id"] for c in checks if c["status"] != "PASS"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
