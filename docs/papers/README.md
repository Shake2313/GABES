# Scientific documents

The current edition of each GABES scientific document. Each edition is self-contained.

| Document | File | Edition | Purpose |
|---|---|---|---|
| Squeezing report | [squeezing_report_v8.pdf](squeezing_report_v8.pdf) | v8 | Simulation–experiment agreement, parameter/frontier exploration, uncertainty and applicability |
| Analytic reconstruction | [squeezing_analytic_reconstruction_v3.pdf](squeezing_analytic_reconstruction_v3.pdf) | v3 | Definitions, derivations, physical assumptions and theoretical precision |
| Quotient structure | [fwm_quotient_structure_v3.2.html](fwm_quotient_structure_v3.2.html) | v3.2 | Logical dependencies, equivalences, partial orders and the conditions for each claim |

## Where the sources and history live

These files are byte copies of the published artifacts. The LaTeX/builder sources, earlier editions,
development records and the publication policy are kept in the GABES research archive
(`GABES-archive`, tag `pre-split-2026-11-09`), which owns `current_publications.json`.

| File | Archive artifact | Archive source | SHA-256 |
|---|---|---|---|
| `squeezing_report_v8.pdf` | `docs/squeezing_report/squeezing_report_v8.pdf` | `docs/squeezing_report/squeezing_report_v8.tex` | `87476483f4d18a9a…` |
| `squeezing_analytic_reconstruction_v3.pdf` | `docs/FWM physics and analytic reconstruction/squeezing_analytic_reconstruction_v3.pdf` | `docs/FWM physics and analytic reconstruction/squeezing_analytic_reconstruction_v3.tex` | `6a3b86c49847963b…` |
| `fwm_quotient_structure_v3.2.html` | `docs/FWM physics and analytic reconstruction/fwm_quotient_structure_v3.2.html` | `docs/FWM physics and analytic reconstruction/fwm_quotient_structure_v3.2_builder.py` | `7293b69e708b9521…` |

To publish a new edition: build it in the archive, update its `current_publications.json`,
then replace the file here and this table in the same main commit.
