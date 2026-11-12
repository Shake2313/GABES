"""Repository locations for development code.

The root is the first folder above the starting point that holds both the
`gabes` package and `pytest.ini`. Searching for that marker instead of counting
parent folders keeps a script working after it moves inside `gabes_lab`, and
when this repository is checked out as the archive's `engine/` submodule.
"""
from pathlib import Path


def find_repo_root(start=None):
    """Return the GABES repository root at or above `start` (default: this file)."""
    here = Path(start if start is not None else __file__).resolve()
    for folder in (here, *here.parents):
        if (folder / "gabes" / "__init__.py").is_file() and (folder / "pytest.ini").is_file():
            return folder
    raise FileNotFoundError(f"no GABES repository root (gabes/ and pytest.ini) above {here}")


REPO_ROOT = find_repo_root()
