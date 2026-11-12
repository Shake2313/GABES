"""Run folders for pipeline outputs. No physics.

Each run writes into its own folder, by default
`runs/<pipeline>/<UTC time>-<short commit>/` at the repository root (ignored by
git), and leaves `run.json` there: pipeline name, arguments, configuration,
git state captured before any output, runtime versions, timing, and the
SHA-256 of every output file. `publish_to_archive` reads that record.

    run = start_run("squeezing/delta_T", config={...})
    np.savez(run.path("map.npz"), ...)
    run.finish()

or, recording a failure automatically:

    with start_run("squeezing/delta_T") as run:
        ...
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from . import io, provenance
from .paths import REPO_ROOT

RUNS_ROOT = REPO_ROOT / "runs"
RUN_RECORD = "run.json"
SCHEMA = "gabes_lab.run/1"
# Files whose modification time is at least this much before the run started
# are marked as not written by the run (explicit output folders may be reused).
_MTIME_SLACK_NS = 2_000_000_000


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _pipeline_parts(name: str) -> tuple[str, ...]:
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or any(p in ("..", ".") for p in parts):
        raise ValueError(f"pipeline name must be a relative path without '..': {name!r}")
    return parts


def _fresh_folder(parent: Path, commit: str | None) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{stamp}-{commit[:7] if commit else 'nogit'}"
    candidate, n = parent / base, 1
    while candidate.exists():
        n += 1
        candidate = parent / f"{base}-{n}"
    return candidate


class Run:
    """An output folder with its run record. Create it with `start_run`."""

    def __init__(self, name: str, folder: Path, record: dict[str, Any]):
        self.name = name
        self.dir = folder
        self.record = record
        self._t0 = time.perf_counter()
        self._start_ns = time.time_ns()

    def path(self, *parts: str) -> Path:
        """A path inside the run folder (parent folders are created)."""
        target = self.dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _write(self) -> None:
        io.write_json(self.dir / RUN_RECORD, self.record, default=_json_default)

    def finish(self, status: str = "complete", error: str | None = None) -> dict[str, Any]:
        """Hash every file in the folder and write the final record."""
        outputs = []
        for file in sorted(p for p in self.dir.rglob("*") if p.is_file()):
            rel = file.relative_to(self.dir).as_posix()
            if rel == RUN_RECORD:
                continue
            entry = {"path": rel, "sha256": provenance.sha256_file(file),
                     "bytes": file.stat().st_size}
            if file.stat().st_mtime_ns < self._start_ns - _MTIME_SLACK_NS:
                entry["written_this_run"] = False
            outputs.append(entry)
        self.record.update(
            status=status,
            finished_utc=provenance.utc_now(),
            elapsed_s=round(time.perf_counter() - self._t0, 3),
            outputs=outputs,
        )
        if error is not None:
            self.record["error"] = error
        self._write()
        return self.record

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish()
        else:
            self.finish("failed", f"{exc_type.__name__}: {exc}")
        return False


def start_run(
    name: str,
    *,
    out: str | os.PathLike[str] | None = None,
    runs_root: Path = RUNS_ROOT,
    config: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
    repo: Path = REPO_ROOT,
) -> Run:
    """Create the output folder and write its first `run.json` (status "running").

    `out` replaces the default folder `runs_root/<name>/<time>-<commit>/`; an
    explicit folder may already exist. `argv` defaults to `sys.argv[1:]`.
    """
    parts = _pipeline_parts(name)
    git = provenance.git_state(repo)          # before anything is written
    if out is not None:
        folder = Path(out).expanduser().resolve()
    else:
        folder = _fresh_folder(runs_root.joinpath(*parts), git["commit"])
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": SCHEMA,
        "pipeline": "/".join(parts),
        "status": "running",
        "started_utc": provenance.utc_now(),
        "finished_utc": None,
        "elapsed_s": None,
        "argv": list(sys.argv[1:] if argv is None else argv),
        "config": dict(config) if config is not None else None,
        "repo": provenance.github_slug(repo),
        "git": git,
        "runtime": provenance.runtime_info(),
        "outputs": [],
    }
    run = Run("/".join(parts), folder, record)
    run._write()
    return run
