"""Reading and writing result files. No physics.

Text is always UTF-8, written in text mode like `Path.write_text` (platform
newline), and parent folders are created on write. The JSON layout
options exist so that existing records keep their bytes; new code can use the
defaults (2-space indent, NaN refused, trailing newline).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def dumps_json(
    payload: Any,
    *,
    indent: int | None = 2,
    allow_nan: bool = False,
    default: Callable[[Any], Any] | None = None,
    trailing_newline: bool = True,
) -> str:
    """JSON text with non-ASCII characters kept as they are."""
    text = json.dumps(payload, indent=indent, ensure_ascii=False,
                      allow_nan=allow_nan, default=default)
    return text + "\n" if trailing_newline else text


def write_json(path: str | os.PathLike[str], payload: Any, **layout: Any) -> Path:
    """Write `payload` as UTF-8 JSON; `layout` goes to `dumps_json`."""
    return write_text(path, dumps_json(payload, **layout))


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
