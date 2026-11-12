"""Development package for GABES: reusable blocks, pipelines and tools.

Physics lives in `gabes`; this package holds what analyses and maintenance
scripts share around it. Run its modules from the repository root with
`python -m gabes_lab.<area>.<module>`.

- `blocks`: small helpers without physics (paths, provenance, io, figures,
  snapshot, runner, publish_to_archive).
- `pipelines`: reusable analysis workflows that call `gabes`.
- `tools`: developer tools (UI audit, benchmarks, repository maintenance).

Package map, block usage and rules: `gabes_lab/README.md`.
"""
