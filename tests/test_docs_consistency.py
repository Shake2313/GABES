"""Cross-document claims that must change together."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _flat(relative_path):
    return " ".join(_read(relative_path).split())


def test_repository_visibility_wording_is_consistently_public():
    readme = _read("README.md")
    guide = _read("docs/Userguide/GABES_User_Guide_v2.html")
    assert "github.com/Shake2313/fwm-squeezing-app` (public)" in readme
    assert "로컬 설치용 공개 소스" in guide
    assert "fwm-squeezing-app` (private)" not in readme
    assert "소스 (private)" not in guide


def test_readme_preserves_biphoton_compatibility_and_coupling_limits():
    # Generic biphoton/API claims belong to the active project documentation;
    # the current analytic publication covers seeded double-Lambda theory.
    readme_flat = _flat("README.md")
    assert "Cs cascade channels land within ~30 %" not in readme_flat
    assert "OD/cell length do not directly scale the rate" in readme_flat
    assert "intrinsic source FWHM" in readme_flat
    assert (
        "legacy `fwhm_ns` API key is retained as an alias for the detected width"
    ) in readme_flat
    assert "source FWHM is about 0.17 ns" in readme_flat
    assert "detected FWHM to about 0.50 ns" in readme_flat
    assert "pair-rate scale remains reference-anchored" in readme_flat
    assert "That agreement is not an independent validation" in readme_flat
    assert "reference residual × additional mode-overlap penalty" in readme_flat
    assert "no unsupported numerical split of `0.74`" in readme_flat
    assert "separate from Ultra's normalized axial crossing-angle profile" in (
        readme_flat
    )
