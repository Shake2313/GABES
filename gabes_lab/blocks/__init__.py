"""Small helpers without physics shared by pipelines and tools.

- `paths`: repository root (`REPO_ROOT`).
- `provenance`: file hashes, git state, portable paths, runtime versions.
- `io`: UTF-8 text and JSON files.
- `figures`: headless pyplot and multi-format figure saving.
- `snapshot`: before/after snapshots for refactors that must keep results.
- `runner`: run folders under `runs/` with a `run.json` record.
- `publish_to_archive`: copy a finished result into GABES-archive with a manifest.

Physics belongs in `gabes`, never here.
"""
