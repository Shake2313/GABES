"""Before/after snapshots for refactors that must keep results. No physics.

AGENTS.md asks a consolidation to keep every caller's outputs: bit-identical,
or (user decision 2026-11-04) at most 1e-12 relative with identical displayed
readouts. `capture` flattens a nested result into named arrays, `save`/`load`
keep it as .npz, and `compare` reports which entries changed and by how much.

    python -m gabes_lab.blocks.snapshot compare before.npz after.npz [--rtol 1e-12]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RTOL = 1e-12
NONE_MARK = "#none"
EMPTY_MARK = "#empty"


def _key(part: Any) -> str:
    return str(part).replace("%", "%25").replace("/", "%2F")


def _join(prefix: str, part: Any) -> str:
    return f"{prefix}/{_key(part)}" if prefix else _key(part)


def capture(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten `value` into {"a/b/0": array}.

    Mappings, lists, tuples, dataclasses and objects with attributes are walked;
    numbers, strings, bools and numeric arrays become leaves. Object arrays are
    walked element by element. `None` and empty containers get marker keys
    (`#none`, `#empty`) so a change between them and a value is reported.
    """
    out: dict[str, np.ndarray] = {}
    _walk(value, prefix, out, set())
    return out


def _walk(value: Any, prefix: str, out: dict[str, np.ndarray], active: set[int]) -> None:
    if value is None:
        out[prefix + NONE_MARK] = np.zeros(0, dtype=np.int8)
        return
    if isinstance(value, (str, bytes, bool, int, float, complex, np.generic)):
        out[prefix] = np.asarray(value)
        return
    if isinstance(value, np.ndarray) and value.dtype != object:
        out[prefix] = value.copy()
        return
    if id(value) in active:
        raise ValueError(f"reference cycle at {prefix!r}")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            items = list(value.items())
        elif isinstance(value, np.ndarray):
            items = list(enumerate(value.tolist()))
        elif isinstance(value, (list, tuple)):
            items = list(enumerate(value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            items = [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
        elif isinstance(value, os.PathLike):
            out[prefix] = np.asarray(os.fspath(value))
            return
        elif hasattr(value, "__dict__"):
            items = [(k, v) for k, v in vars(value).items() if not k.startswith("_")]
        else:
            raise TypeError(f"cannot snapshot {type(value).__name__} at {prefix!r}")
        if not items:
            out[prefix + EMPTY_MARK] = np.zeros(0, dtype=np.int8)
        for part, item in items:
            _walk(item, _join(prefix, part), out, active)
    finally:
        active.discard(id(value))


def save(path: str | os.PathLike[str], snap: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **dict(snap))
    return path


def load(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@dataclasses.dataclass(frozen=True)
class Comparison:
    missing: tuple[str, ...]          # in before, not in after
    extra: tuple[str, ...]            # in after, not in before
    changed: tuple[tuple[str, float], ...]  # (key, max relative difference), largest first
    n_keys: int

    @property
    def identical(self) -> bool:
        """Same keys, and every array has the same dtype, shape and bytes."""
        return not (self.missing or self.extra or self.changed)

    def within(self, rtol: float = DEFAULT_RTOL) -> bool:
        """Same keys and every changed entry within `rtol` relative."""
        return not (self.missing or self.extra) and all(d <= rtol for _, d in self.changed)

    def max_relative(self) -> float:
        return max((d for _, d in self.changed), default=0.0)

    def report(self, limit: int = 20) -> str:
        if self.identical:
            return f"bit-identical: {self.n_keys} entries"
        lines = [f"entries {self.n_keys}, missing {len(self.missing)}, "
                 f"extra {len(self.extra)}, changed {len(self.changed)}, "
                 f"max relative {self.max_relative():.3e}"]
        lines += [f"  missing {key}" for key in self.missing[:limit]]
        lines += [f"  extra   {key}" for key in self.extra[:limit]]
        lines += [f"  changed {key}: {d:.3e}" for key, d in self.changed[:limit]]
        return "\n".join(lines)


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Largest |a-b| / max(|a|, |b|) over elements; 0 when bit-identical.

    Equal NaN or equal infinite entries count as equal. Shape or kind
    mismatches and unequal non-numeric entries give infinity.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes():
        return 0.0
    numeric = all(x.dtype.kind in "biufc" for x in (a, b))
    if a.shape != b.shape or not numeric:
        return float("inf")
    a = a.astype(np.complex128)
    b = b.astype(np.complex128)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(same, 0.0, np.where(finite & (scale > 0), diff / scale, np.inf))
    return float(rel.max()) if rel.size else 0.0


def compare(before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]) -> Comparison:
    keys_before, keys_after = set(before), set(after)
    changed = []
    for key in sorted(keys_before & keys_after):
        d = relative_difference(before[key], after[key])
        if d != 0.0 or before[key].dtype != after[key].dtype:
            changed.append((key, d))
    changed.sort(key=lambda item: (-item[1], item[0]))
    return Comparison(
        missing=tuple(sorted(keys_before - keys_after)),
        extra=tuple(sorted(keys_after - keys_before)),
        changed=tuple(changed),
        n_keys=len(keys_before | keys_after),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two snapshot .npz files.")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("before", type=Path)
    cmp_parser.add_argument("after", type=Path)
    cmp_parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args(argv)
    result = compare(load(args.before), load(args.after))
    print(result.report())
    if not result.identical:
        print(f"within rtol {args.rtol:g}: {result.within(args.rtol)}")
    return 0 if result.within(args.rtol) else 1


if __name__ == "__main__":
    sys.exit(main())
