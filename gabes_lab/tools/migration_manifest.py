"""Write the S5 Migration Manifest: where each moved or removed file went, and why.

usage: python -m gabes_lab.tools.migration_manifest [--base COMMIT] [--head COMMIT] [--output PATH]

The manifest is generated from git, never written by hand. For every file that
the S5 consolidation renamed or deleted between `--base` and `--head` it
records the old and new path, the sha256 of the file bytes before and after,
the kind of change and the reason. Files that were only edited in place are
not migrations and are left out. Reasons come from MIGRATIONS below; add a row
there in the same commit as a new move.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from gabes_lab.blocks.io import write_json
from gabes_lab.blocks.paths import REPO_ROOT
from gabes_lab.blocks.provenance import sha256_bytes

ROOT = REPO_ROOT
BASE = "18b0b5f"  # main before S5 started (2026-11-12)
OUTPUT = ROOT / "docs" / "repo_restructure" / "s5_migration" / "migration_manifest.json"

# (old path prefix, decision, reason). The first matching prefix wins.
MIGRATIONS = [
    ("assets/", "R5", "UI assets belong to the UI package (gabes_ui/assets)."),
    ("sabes/components/", "R5", "SABES screen code moves to gabes_ui/sabes; the SABES model stays in sabes/."),
    ("sabes/layout/render.py", "R5", "Optical-table drawing spec is screen code; moved to gabes_ui/sabes/render.py."),
    ("sabes_page.py", "R5", "SABES page entry point moves to gabes_ui/sabes/page.py."),
    ("tools/", "R6", "Developer tools join the development package (gabes_lab/tools)."),
    ("analysis/rydberg_cell_heating/test_", "R6", "Package tests live in the root tests/ folder."),
    ("analysis/rydberg_cell_heating/", "R6", "Reusable heating workflow becomes a gabes_lab pipeline."),
    ("analysis/squeezing/", "R6", "Reusable squeezing scanners become gabes_lab pipelines."),
]

KINDS = {"R": "move", "D": "delete"}


def git(*args):
    return subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=ROOT,
                          capture_output=True, check=True).stdout


def blob_sha256(commit, path):
    return sha256_bytes(git("cat-file", "blob", f"{commit}:{path}"))


def reason_for(path):
    for prefix, decision, reason in MIGRATIONS:
        if path == prefix or path.startswith(prefix):
            return decision, reason
    return None, None


def entries(base, head):
    raw = git("diff", "-M", "--name-status", "-z", base, head).decode("utf-8").split("\0")
    rows, i = [], 0
    while i < len(raw) - 1:
        status = raw[i]
        if status.startswith("R"):
            old, new, i = raw[i + 1], raw[i + 2], i + 3
        else:
            old, new, i = raw[i + 1], None, i + 2
        if status[0] not in KINDS:
            continue
        before = blob_sha256(base, old)
        after = blob_sha256(head, new) if new else None
        kind = KINDS[status[0]]
        if kind == "move" and before != after:
            kind = "move+edit"
        decision, reason = reason_for(old)
        rows.append(dict(old_path=old, new_path=new, sha256_before=before, sha256_after=after,
                         kind=kind, similarity=int(status[1:]) if status.startswith("R") else None,
                         decision=decision, reason=reason))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    base = git("rev-parse", args.base).decode().strip()
    head = git("rev-parse", args.head).decode().strip()
    rows = entries(base, head)
    unexplained = [row["old_path"] for row in rows if row["reason"] is None]
    if unexplained:
        sys.exit("no MIGRATIONS reason for: " + ", ".join(unexplained))
    manifest = dict(schema="gabes.migration_manifest.v1", base_commit=base, head_commit=head,
                    generator="gabes_lab/tools/migration_manifest.py", count=len(rows), entries=rows)
    write_json(args.output, manifest, indent=1, allow_nan=True)
    kinds = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"{len(rows)} entries {kinds} -> {args.output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
