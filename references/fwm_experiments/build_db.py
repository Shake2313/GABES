"""Validate primary JSON records and build portable literature DB views.

Run from any directory: python references/fwm_experiments/build_db.py
No simulation, fitting, or production imports are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALUE_TYPES = {"reported", "approximate", "range", "lower_bound", "upper_bound", "derived", "digitized", "not_reported"}
UNIT_ALIASES = {"µW": "uW", "μW": "uW", "µm": "um", "μm": "um", "°C": "degC", "%": "percent"}
GAIN_QUANTITIES = {"probe_gain", "conjugate_gain", "reported_gain"}
CANONICAL_INPUTS = {
    "opd_gabes", "tpd_gabes", "temperature", "cell_length", "pump_power",
    "seed_power", "pump_radius", "seed_radius", "crossing_angle",
}


def read_records():
    return [json.loads(p.read_text(encoding="utf-8-sig"))
            for p in sorted((ROOT / "papers").glob("*.json"))]


def quantities(exp):
    for role in ("conditions", "results"):
        for index, item in enumerate(exp.get(role, [])):
            yield role, index, item


def is_value(item):
    return item.get("value_type") != "not_reported" and (
        isinstance(item.get("value"), (int, float))
        or item.get("value_type") == "range")


def display_value(item):
    unit = item.get("unit", "")
    unit = {"degC": "°C", "um": "µm", "uW": "µW", "percent": "%", "1": ""}.get(unit, unit)
    kind = item.get("value_type")
    value = item.get("value")
    if kind == "not_reported" or value is None and kind != "range":
        return "미보고"
    if kind == "range":
        value = f"{item.get('lower', '?')} … {item.get('upper', '?')}"
    else:
        value = f"{value:g}" if isinstance(value, (float, int)) else str(value)
    prefix = {"approximate": "≈", "lower_bound": "≥", "upper_bound": "≤"}.get(kind, "")
    return f"{prefix}{value} {unit}".strip()


def readiness(paper, exp):
    present = {q["quantity"] for _, _, q in quantities(exp) if is_value(q)}
    gain = bool(present & GAIN_QUANTITIES)
    canonical = CANONICAL_INPUTS <= present
    scope = paper.get("scope", {})
    compatible = (scope.get("inclusion") == "primary"
                  and exp.get("seed_mode", scope.get("seed_mode")) == "single_seed"
                  and exp.get("kind") == "operating_point"
                  and exp.get("species", "85Rb") == "85Rb")
    # This is a missing-field inventory, not acceptance for fitting. A complete
    # record may still have an unknown gain plane, nuisance input or uncertainty.
    return {
        "has_numeric_gain": gain,
        "single_seed_operating_point": compatible,
        "canonical_inputs_complete": canonical,
        "missing_canonical_inputs": sorted(CANONICAL_INPUTS - present),
        "fit_accepted": False,
        "holdout_status": "inspected_literature_pool",
    }


def validate(papers):
    errors, warnings = [], []
    ids, experiments, total, sources = set(), set(), 0, 0
    for paper in papers:
        pid = paper.get("paper_id", "MISSING")
        prefix = pid
        if pid in ids:
            errors.append(f"duplicate paper_id {pid}")
        ids.add(pid)
        for key in ("title", "authors", "year", "publication", "scope", "sources", "experiments"):
            if not paper.get(key):
                errors.append(f"{prefix}: missing {key}")
        source_map = {s.get("source_id"): s for s in paper.get("sources", [])}
        if len(source_map) != len(paper.get("sources", [])):
            errors.append(f"{prefix}: duplicate source_id")
        for source in source_map.values():
            sources += 1
            if not source.get("url"):
                errors.append(f"{prefix}: source URL absent")
            if source.get("access") != "full_text":
                warnings.append(f"{prefix}: source {source.get('source_id')} access {source.get('access')}")
            path = source.get("path")
            if path:
                target = (ROOT / path).resolve()
                if not target.is_file():
                    errors.append(f"{prefix}: source missing {path}")
                    continue
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if source.get("sha256") != actual:
                    errors.append(f"{prefix}: source checksum mismatch {path}")
                if target.suffix.lower() == ".pdf" and not target.read_bytes()[:1024].lstrip().startswith(b"%PDF"):
                    errors.append(f"{prefix}: invalid PDF header {path}")
            elif source.get("access") == "full_text":
                warnings.append(f"{prefix}: full text has URL but no local snapshot")
        for exp in paper.get("experiments", []):
            eid = exp.get("experiment_id")
            label = f"{pid}/{eid}"
            if not eid or label in experiments:
                errors.append(f"duplicate/missing experiment_id {label}")
            experiments.add(label)
            for role, index, q in quantities(exp):
                total += 1
                labelq = f"{label}/{role}[{index}] {q.get('quantity')}"
                if not q.get("quantity") or "unit" not in q:
                    errors.append(f"{labelq}: missing quantity/unit")
                kind = q.get("value_type")
                if kind not in VALUE_TYPES:
                    errors.append(f"{labelq}: unknown value_type {kind}")
                if kind == "not_reported":
                    if q.get("value") is not None:
                        errors.append(f"{labelq}: missing value must be null")
                    continue
                if q.get("source_id") not in source_map:
                    errors.append(f"{labelq}: unknown source")
                if not q.get("locator"):
                    errors.append(f"{labelq}: missing locator")
                if kind == "range":
                    low, high = q.get("lower"), q.get("upper")
                    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or low > high:
                        errors.append(f"{labelq}: invalid interval")
                elif not isinstance(q.get("value"), (int, float, str, bool)):
                    errors.append(f"{labelq}: absent value")
                if isinstance(q.get("value"), float) and not math.isfinite(q["value"]):
                    errors.append(f"{labelq}: nonfinite value")
                if kind == "derived" and not q.get("derivation", {}).get("formula"):
                    errors.append(f"{labelq}: derived value lacks formula")
                if q.get("quantity") in {"opd_gabes", "tpd_gabes"} and not q.get("reference"):
                    errors.append(f"{labelq}: canonical detuning lacks convention")
    # Historical CSV retained only for the migrated ten-paper coverage check.
    legacy_path = ROOT.parents[1] / "archive" / "fwm_squeezing_paper_parameters.csv"
    with legacy_path.open(encoding="utf-8-sig", newline="") as stream:
        legacy = list(csv.reader(stream))[0][5:]
    retained = {p.get("legacy_column") for p in papers}
    missing = sorted(set(legacy) - retained)
    if missing:
        errors.append("legacy papers absent: " + ", ".join(missing))
    return {
        "schema_version": 1, "status": "PASS" if not errors else "FAIL",
        "validation_date": "2026-10-23", "paper_count": len(papers),
        "experiment_count": len(experiments), "quantity_count": total,
        "source_count": sources, "legacy_papers_expected": len(legacy),
        "legacy_papers_retained": len(set(legacy) & retained),
        "inclusion_counts": dict(Counter(p.get("scope", {}).get("inclusion") for p in papers)),
        "errors": errors, "warnings": warnings,
        "scope": "Structure, references, source bytes and legacy coverage; scientific spot checks recorded separately.",
    }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def export_tables(papers):
    paper_rows, experiment_rows, observation_rows = [], [], []
    for paper in papers:
        pid = paper["paper_id"]
        pub, scope = paper["publication"], paper["scope"]
        paper_rows.append(dict(paper_id=pid, title=paper["title"], authors="; ".join(paper["authors"]),
                               year=paper["year"], **pub, **scope,
                               legacy_column=paper.get("legacy_column"),
                               apparatus_group=paper.get("apparatus_group", "unassigned"),
                               experiment_count=len(paper["experiments"]),
                               source_urls=[s["url"] for s in paper["sources"]]))
        source_map = {s["source_id"]: s for s in paper["sources"]}
        for exp in paper["experiments"]:
            ready = readiness(paper, exp)
            row = dict(paper_id=pid, experiment_id=exp["experiment_id"], title=paper["title"],
                       label=exp["label"], kind=exp["kind"], inclusion=scope["inclusion"],
                       seed_mode=exp.get("seed_mode", scope["seed_mode"]), **ready, notes=exp.get("notes", []))
            for _, _, q in quantities(exp):
                key = q["quantity"]
                row[key] = " | ".join(filter(None, (row.get(key, ""), display_value(q))))
            experiment_rows.append(row)
            for role, index, q in quantities(exp):
                source = source_map.get(q.get("source_id"), {})
                observation_rows.append(dict(paper_id=pid, experiment_id=exp["experiment_id"],
                    observation_id=f"{pid}/{exp['experiment_id']}/{role}/{index}", role=role,
                    **q, normalized_unit=UNIT_ALIASES.get(q.get("unit"), q.get("unit")),
                    source_url=source.get("url"), source_path=source.get("path"),
                    source_sha256=source.get("sha256")))
    paper_fields = ["paper_id", "title", "authors", "year", "journal", "doi", "arxiv_id", "status", "inclusion", "measurement", "seed_mode", "isotope_composition", "legacy_column", "apparatus_group", "experiment_count", "source_urls", "reason"]
    base = ["paper_id", "experiment_id", "title", "label", "kind", "inclusion", "seed_mode", "has_numeric_gain", "single_seed_operating_point", "canonical_inputs_complete", "missing_canonical_inputs", "fit_accepted", "holdout_status"]
    preferred = ["opd_gabes", "opd_reported", "tpd_gabes", "tpd_reported", "temperature", "cell_length", "pump_power", "seed_power", "pump_radius", "seed_radius", "pump_beam_size", "seed_beam_size", "crossing_angle", "probe_gain", "conjugate_gain", "reported_gain", "squeezing_measured", "squeezing_electronic_noise_subtracted", "squeezing_loss_corrected", "squeezing_slope", "analysis_frequency", "detection_efficiency"]
    extra = sorted(set().union(*(r.keys() for r in experiment_rows)) - set(base + preferred + ["notes"])) if experiment_rows else []
    fields = ["paper_id", "experiment_id", "observation_id", "role", "quantity", "value", "lower", "upper", "unit", "normalized_unit", "value_type", "uncertainty", "basis", "reference", "plane", "locator", "evidence", "notes", "derivation", "source_id", "source_url", "source_path", "source_sha256"]
    write_csv(ROOT / "papers.csv", paper_rows, paper_fields)
    write_csv(ROOT / "experiments.csv", experiment_rows, base + preferred + extra + ["notes"])
    write_csv(ROOT / "observations.csv", observation_rows, fields)
    return experiment_rows


def build_html(papers, receipt):
    data = json.dumps(papers, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "viewer.html").read_text(encoding="utf-8")
    template = template.replace("__DB_DATA__", data).replace("__DB_RECEIPT__", json.dumps(receipt, ensure_ascii=False))
    (ROOT / "index.html").write_text(template, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without rewriting generated views")
    args = parser.parse_args()
    papers = read_records()
    receipt = validate(papers)
    if not args.check:
        payload = {"schema_version": 1, "as_of": "2026-10-23", "purpose": "Literature evidence, before calibration development", "unit_aliases": UNIT_ALIASES, "papers": papers}
        (ROOT / "database.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        export_tables(papers)
        build_html(papers, receipt)
        (ROOT / "validation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        members = sorted((ROOT / "papers").glob("*.json")) + [ROOT / n for n in (
            "SCHEMA.md", "schema.json", "build_db.py", "viewer.html", "README.md",
            "database.json", "papers.csv", "experiments.csv", "observations.csv", "index.html",
            "REVIEW.md", "REVIEW_recent.json", "SEARCH.md", "search_new.json", "verify_db.py")]
        manifest = {"as_of": "2026-10-23", "files": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in members if p.is_file()}, "sources": {
            s["path"]: s.get("sha256") for paper in papers for s in paper["sources"] if s.get("path")}}
        (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    raise SystemExit(0 if receipt["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
