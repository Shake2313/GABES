"""Keep GABES main light (docs/repo_restructure/README.md, S2).

Results, caches and dated records belong to the archive repository. This
guard fails when a tracked main-side file exceeds its size budget, so heavy
outputs cannot creep back into main. Before the split it checks only the
files that tools/split_boundary.py assigns to main; after the split every
tracked file is main.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE_LIMIT = 1_500_000
# Larger budgets for generated or curated artifacts that must ship with main.
ALLOWANCE = {"static/": 5_000_000}
TOTAL_LIMIT = 50_000_000


def _tracked():
    out = subprocess.run(["git", "-c", "core.quotepath=false", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [path for path in out.split("\0") if path]


def _main_side(paths):
    if not any(path.startswith("docs/grand_challenge/") for path in paths):
        return paths  # split done: everything tracked is main
    from tools.split_boundary import ARCHIVE, classify
    return [path for path in paths if classify(path)[0] != ARCHIVE]


def _budget(path):
    for prefix, limit in ALLOWANCE.items():
        if path.startswith(prefix):
            return limit
    return FILE_LIMIT


def test_main_files_stay_within_their_size_budget():
    sizes = {path: (ROOT / path).stat().st_size for path in _main_side(_tracked())
             if (ROOT / path).is_file()}
    oversized = {path: size for path, size in sizes.items() if size > _budget(path)}
    assert not oversized, (
        "Main-side files over budget (move results to the archive repository, or add an "
        f"explicit allowance with a reason): {oversized}")
    assert sum(sizes.values()) <= TOTAL_LIMIT, f"main total {sum(sizes.values())} B"
