"""Saving matplotlib figures from scripts. No physics.

Plot content stays in each pipeline; this block only fixes how figures are
written: a headless backend, one file per format, and resolution applied to
raster formats only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

RASTER_FORMATS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "webp"})


def pyplot() -> Any:
    """`matplotlib.pyplot` on the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(
    fig: Any,
    stem: str | os.PathLike[str],
    formats: Iterable[str] = ("png",),
    *,
    dpi: float | None = None,
    bbox_inches: str | None = "tight",
    close: bool = True,
) -> list[Path]:
    """Write `fig` to `<stem>.<format>` for each format and return the paths.

    `dpi` applies to raster formats; vector formats are written without it.
    The parent folder is created. The figure is closed afterwards unless
    `close=False`.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        path = stem.with_name(f"{stem.name}.{fmt}")
        fig.savefig(path, dpi=dpi if fmt in RASTER_FORMATS else None,
                    bbox_inches=bbox_inches)
        paths.append(path)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return paths
