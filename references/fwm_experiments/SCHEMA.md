# FWM literature database schema, version 1

## User-requested scope

85Rb D1 double-lambda four-wave mixing experiments measuring intensity-difference squeezing. Preserve every paper in the previous CSV, including explicitly identified context-only papers. Keep original publication titles. Rebuild evidence from primary papers; previous CSV values are discovery leads, not verified measurements.

## ⟦AI⟧ Record contract

Canonical records: `papers/<paper_id>.json`, UTF-8. Generated CSV, HTML and combined JSON are views of these records. Stable IDs use lowercase ASCII. One paper can contain several operating points, sweeps and apparatus variants. Do not merge optima or borrow an input from another paper.

```json
{
  "schema_version": 1,
  "paper_id": "author_year_shortname",
  "title": "Exact original title",
  "authors": ["Full Author Name"],
  "year": 2025,
  "publication": {"journal": "", "volume": "", "issue": "", "pages": "", "doi": "", "arxiv_id": "", "status": "published"},
  "legacy_column": null,
  "scope": {"species": "85Rb", "line": "D1", "configuration": "double_lambda", "measurement": "intensity_difference", "seed_mode": "single_seed", "inclusion": "primary", "reason": "", "isotope_composition": "not_reported"},
  "sources": [{"source_id": "main", "url": "https://...", "path": "sources/paper_id.pdf", "access": "full_text", "version": "published", "sha256": "", "pages": 4}],
  "experiments": [{
    "experiment_id": "main",
    "label": "Figure or operating-point name",
    "kind": "operating_point",
    "conditions": [],
    "results": [],
    "notes": []
  }],
  "ai_notes": []
}
```

Sources already in the repository use a path relative to this DB folder, e.g. `../oe-26-25-33366.pdf`. New downloads stay in `sources/`. Source checksums identify exactly which version was read. No duplicated downloads for an already accessible local primary source.

Each condition/result is an evidence-bearing quantity:

```json
{"quantity":"pump_power", "value":400, "unit":"mW", "value_type":"approximate", "basis":"total power", "reference":null, "plane":"cell_input", "source_id":"main", "locator":"PDF p. 2, Experimental setup", "evidence":"Paraphrase identifying the associated experiment.", "notes":""}
```

Allowed value types: `reported`, `approximate`, `range`, `lower_bound`, `upper_bound`, `derived`, `digitized`, `not_reported`. A range uses `value: null, lower: ..., upper: ...`. Missing quantities use `value: null, value_type: not_reported`; never zero or an assumed default. An optional `uncertainty` object gives `value`, `unit`, `kind` and `confidence` as reported; an unreported error bar stays absent.

Retain original definitions in `basis`, `reference` and `plane`. Gain quantities: `probe_gain`, `conjugate_gain`, `reported_gain`, `gain_coefficient`; do not treat a squeezing-derived ideal gain as a measured gain. Results: `squeezing_measured`, `squeezing_electronic_noise_subtracted`, `squeezing_loss_corrected`, `squeezing_slope`, `noise_ratio`, `output_probe_power`, `output_conjugate_power`, `bandwidth`, etc. Conditions: `opd_reported`, `tpd_reported`, `pump_probe_spacing`, `temperature`, `cell_length`, `pump_power`, `seed_power`, `pump_beam_size`, `seed_beam_size`, `crossing_angle`, `detector_qe`, `optical_transmission`, `detection_efficiency`, `analysis_frequency`, `resolution_bandwidth`, `video_bandwidth`, etc.

Derived quantities retain their input record and include `derivation: {"formula": "...", "inputs": ["quantity", "..."]}`. Standardized quantities use `opd_gabes` (GHz, pump minus 85Rb F=2→F′=3), `tpd_gabes` (MHz, seed=pump−3035.732439 MHz+δ), `pump_radius`/`seed_radius` (µm, 1/e² intensity radius). Convert beam diameter/waist only when its definition is established. Ambiguous excited-state reference or detuning sign does not receive a guessed canonical number.

Each numeric record needs a source and a page/section/figure locator. Evidence is concise paraphrase, not a substitute for the locator. Exact plot points are not invented from axis ranges. `kind: sweep` stores the actual joint sweep and fixed controls, not Cartesian combinations. `kind: literature_summary` means a statement about previous work, excluded from independent experimental observations.

Inclusion: `primary` for the requested system/observable, `context_only` for retained legacy papers with a different observable or seeding arrangement requiring separate treatment, `excluded` for screened nonmatching work. Preserve natural-isotope cells but label their mixture. `publication.status` distinguishes preprints. Related papers from the same apparatus use optional `apparatus_group` at paper level for later group-wise validation.

No fit, train/test assignment or claimed prediction is created in this data-building stage. All inspected papers are an explored literature pool. A future untouched test set requires a separate predeclared collection/freeze; leave-one-paper/apparatus-out cross-validation is possible but is not untouched external validation.
