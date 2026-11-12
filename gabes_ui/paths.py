"""Where the UI package's static files live, and the one reader for them.

Every UI module, the app entry point and the tests resolve asset files here,
so moving the asset folder changes one line.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
UI_ASSETS = ASSETS / "ui"


def asset_text(name):
    """UTF-8 text of an asset, `name` relative to `ASSETS` (e.g. "ui/gabes.css")."""
    return (ASSETS / name).read_text(encoding="utf-8")
