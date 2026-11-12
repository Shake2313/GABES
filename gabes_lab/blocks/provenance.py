"""Where a result came from: file hashes, repository state, portable paths, runtime.

No physics. Pipelines and tools record their sources with these helpers so that
every record uses the same hash, the same git query and the same path form.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

_CHUNK = 1024 * 1024


def utc_now() -> str:
    """Current UTC time in ISO 8601, whole seconds (e.g. 2026-11-12T03:04:05+00:00)."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: str | os.PathLike[str], root: Path = REPO_ROOT) -> str:
    """A path for records that names no drive, user folder or temporary directory.

    Files under `root` become POSIX paths relative to it. Other files keep a
    sanitized basename plus a short hash of their resolved location, so repeated
    references still match. Use the real Path for filesystem work.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        location_digest = hashlib.sha256(
            str(resolved).encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:16]
        basename = resolved.name or "external-path"
        safe_basename = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in basename
        )
        return f"external://{safe_basename}#path-sha256={location_digest}"


def file_record(path: str | os.PathLike[str], root: Path = REPO_ROOT) -> dict[str, Any]:
    """{"path", "sha256", "bytes"} for one file; `path` is `portable_path`."""
    path = Path(path)
    return {
        "path": portable_path(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def git(*args: str, root: Path = REPO_ROOT, timeout: float = 5) -> str | None:
    """Stripped stdout of `git <args>` in `root`, or None when git fails or is absent."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            check=False, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state(root: Path = REPO_ROOT) -> dict[str, Any]:
    """{"commit", "branch", "dirty"} of the repository at `root`.

    Capture it before writing outputs inside the working tree, so the outputs
    themselves do not make the tree look dirty.
    """
    return {
        "commit": git("rev-parse", "HEAD", root=root),
        "branch": git("branch", "--show-current", root=root),
        "dirty": bool(git("status", "--porcelain", root=root)),
    }


def github_slug(root: Path = REPO_ROOT, remote: str = "origin") -> str | None:
    """"owner/name" of a GitHub remote, or None for another host or no remote."""
    url = git("remote", "get-url", remote, root=root)
    if not url:
        return None
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return match.group(1) if match else None


def runtime_info() -> dict[str, str]:
    """Python, platform and NumPy versions."""
    import numpy as np

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
