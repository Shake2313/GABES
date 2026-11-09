"""docs/papers holds the current scientific editions shipped with GABES main.

Repository split, user decision R4 (docs/repo_restructure/README.md): main
keeps only the latest editions; sources, history and the publication policy
live in the archive repository.
"""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "docs" / "papers"


def _listed():
    table = (PAPERS / "README.md").read_text(encoding="utf-8")
    return {unquote(m) for m in re.findall(r"\]\(([^)/]+\.(?:pdf|html))\)", table)}


def test_every_listed_edition_exists_and_nothing_else_ships():
    listed = _listed()
    assert len(listed) == 3
    present = {p.name for p in PAPERS.iterdir() if p.suffix in (".pdf", ".html")}
    assert present == listed


def test_readme_links_each_current_edition():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in _listed():
        assert f"(docs/papers/{name})" in readme
