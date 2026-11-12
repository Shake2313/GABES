"""Copy a finished result into a GABES-archive working copy, with its provenance.

No physics. Results leave main only through this explicit step. The copy gets
`publish_manifest.json`: the main repository and commit that produced it, the
archive's engine pin, and the SHA-256 of every file. Nothing is committed in
the archive; review and commit there yourself.

    python -m gabes_lab.blocks.publish_to_archive runs/squeezing/delta_T/<run> \\
        --archive ../GABES-archive --dest analysis/squeezing/2026-11-12_delta_T [--dry-run]

A folder made by `runner` must have finished ("complete") and its files must
still match `run.json`; its git state is the one recorded before the run. Any
other folder is published with the git state at publish time. A dirty main
tree is refused unless `--allow-dirty`, which records it as dirty.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from . import io, provenance
from .paths import REPO_ROOT
from .runner import RUN_RECORD

MANIFEST = "publish_manifest.json"
SCHEMA = "gabes_lab.publish/1"


class PublishError(RuntimeError):
    pass


def check_archive(root: str | Path) -> Path:
    """`root` resolved, if it is a GABES-archive working copy (engine submodule)."""
    root = Path(root).expanduser().resolve()
    modules = root / ".gitmodules"
    if not (modules.is_file() and (root / "engine").is_dir()
            and "path = engine" in modules.read_text(encoding="utf-8")):
        raise PublishError(f"not a GABES-archive working copy (no engine submodule): {root}")
    return root


def _destination(archive: Path, dest: str) -> Path:
    rel = PurePosixPath(dest.replace("\\", "/"))
    if not rel.parts or rel.is_absolute() or ".." in rel.parts or rel.parts[0] == "engine":
        raise PublishError(f"--dest must be a relative archive path outside engine/: {dest!r}")
    return archive.joinpath(*rel.parts)


def _source_files(source: Path) -> tuple[list[str], dict[str, Any] | None]:
    record_path = source / RUN_RECORD
    if not record_path.is_file():
        files = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
        return files, None
    record = io.read_json(record_path)
    if record.get("status") != "complete":
        raise PublishError(f"run status is {record.get('status')!r}, not 'complete': {source}")
    for entry in record.get("outputs", []):
        path = source / entry["path"]
        if not path.is_file() or provenance.sha256_file(path) != entry["sha256"]:
            raise PublishError(f"{entry['path']} changed or vanished since the run finished")
    return [entry["path"] for entry in record["outputs"]] + [RUN_RECORD], record


def publish_to_archive(
    source: str | Path,
    archive: str | Path,
    dest: str,
    *,
    allow_dirty: bool = False,
    dry_run: bool = False,
    note: str | None = None,
    repo: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Copy `source` to `<archive>/<dest>/` and write the publish manifest there.

    Returns the manifest. With `dry_run`, only checks and returns it.
    """
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise PublishError(f"source folder not found: {source}")
    archive = check_archive(archive)
    target = _destination(archive, dest)
    if target.exists() and any(target.iterdir()):
        raise PublishError(f"destination exists and is not empty: {target}")

    files, record = _source_files(source)
    if not files:
        raise PublishError(f"nothing to publish in {source}")
    git = record["git"] if record else provenance.git_state(repo)
    if not git.get("commit"):
        raise PublishError("main commit unknown; run inside the GABES repository")
    if git.get("dirty") and not allow_dirty:
        raise PublishError("main working tree was dirty; commit first, or pass --allow-dirty "
                           "(allow_dirty=True) to record it as dirty")

    engine_pin = provenance.git("rev-parse", "HEAD:engine", root=archive)
    manifest = {
        "schema": SCHEMA,
        "published_utc": provenance.utc_now(),
        "source": {
            "repo": (record or {}).get("repo") or provenance.github_slug(repo),
            "commit": git["commit"],
            "branch": git.get("branch"),
            "dirty": bool(git.get("dirty")),
            "path": provenance.portable_path(source, repo),
        },
        "git_state_from": "run.json" if record else "publish time",
        "pipeline": record.get("pipeline") if record else None,
        "archive_engine_pin": engine_pin,
        "engine_pin_matches_source": engine_pin == git["commit"],
        "note": note,
        "files": [
            {"path": rel, "sha256": provenance.sha256_file(source / rel),
             "bytes": (source / rel).stat().st_size}
            for rel in files
        ],
    }
    if dry_run:
        return manifest

    for entry in manifest["files"]:
        copied = target / entry["path"]
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / entry["path"], copied)
        if provenance.sha256_file(copied) != entry["sha256"]:
            raise PublishError(f"copy check failed: {copied}")
    io.write_json(target / MANIFEST, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="run folder or other result folder")
    parser.add_argument("--archive", type=Path, required=True,
                        help="GABES-archive working copy")
    parser.add_argument("--dest", required=True, help="folder inside the archive")
    parser.add_argument("--note", help="free text stored in the manifest")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = publish_to_archive(args.source, args.archive, args.dest,
                                      allow_dirty=args.allow_dirty, dry_run=args.dry_run,
                                      note=args.note)
    except PublishError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {len(manifest['files'])} files to {args.dest} "
          f"(main {manifest['source']['commit'][:7]}, engine pin "
          f"{(manifest['archive_engine_pin'] or 'unknown')[:7]}, "
          f"match {manifest['engine_pin_matches_source']})")
    if not args.dry_run:
        print("Nothing is committed in the archive; review and commit there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
