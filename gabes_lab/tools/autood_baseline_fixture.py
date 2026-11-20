"""Freeze AutoOD 0.2.2 spectra as a GABES test fixture (OD-fit port, M0).

The AutoOD source comes from the sibling repository's tag ``baseline-0.2.2-src``
(A0) and runs on its pinned GABES engine snapshot, never on this checkout.
Numba stays disabled, as in the shipped AutoOD executable. The child process
writes one compressed ``.npz`` with the arrays and a JSON metadata string.

Usage:  python -m gabes_lab.tools.autood_baseline_fixture [--autood ../AutoOD] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTOOD = ROOT.parent / "AutoOD"
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "autood_baseline_0_2_2" / "spectra.npz"
TAG = "baseline-0.2.2-src"

# (case, AutoOD Settings keyword arguments). Probe power is in mW, as in AutoOD.
CASES = (
    ("natural_d2_25c_75mm", dict(line="D2", temperature_c=25.0, cell_mm=75.0, diameter_mm=2.0)),
    ("rb87_d1_90c_12p5mm", dict(line="D1", temperature_c=90.0, cell_mm=12.5, diameter_mm=2.0,
                                rb85_percent=0.0)),
    ("rb85_d2_90c_12p5mm", dict(line="D2", temperature_c=90.0, cell_mm=12.5, diameter_mm=2.0,
                                rb85_percent=100.0)),
)
PROBE_MW = {"natural_d2_25c_75mm": (0.0, 0.1), "rb87_d1_90c_12p5mm": (0.0, 0.1),
            "rb85_d2_90c_12p5mm": (0.0, 1.0)}

CHILD = r'''
import json, sys
snapshot, source, out = sys.argv[1:4]
sys.path[:0] = [snapshot, source]
import numpy as np
from autood import GABES_SHA, __version__, engine
import build_release
cases = json.loads(sys.argv[4]); powers = json.loads(sys.argv[5])
arrays, meta = {}, dict(autood_version=__version__, gabes_pin=GABES_SHA,
                        adapter_sha256=build_release.theory_adapter_sha256(build_release.ROOT),
                        numba_disabled=True, cases={})
for name, kwargs in cases:
    entries = []
    for probe_mw in powers[name]:
        settings = engine.Settings(probe_mw=probe_mw, **kwargs)
        spectrum = engine.calculate(settings)
        scan = engine._full_rb_scan(engine._parameters(settings))
        key = f"{name}__probe_{probe_mw * 1000:g}uw"
        arrays[f"{name}__scan_rad_s"] = scan
        arrays[f"{key}__transmission"] = spectrum.transmission
        arrays[f"{key}__optical_depth"] = spectrum.optical_depth
        arrays[f"{key}__od0"] = spectrum.od0
        params = engine._parameters(settings)
        entries.append(dict(key=key, probe_mw=probe_mw, model=spectrum.model,
                            transit_khz=params["transit_khz"],
                            frequency_reference=spectrum.frequency_reference))
    meta["cases"][name] = dict(settings=dict(kwargs), spectra=entries)
np.savez_compressed(out, metadata=np.array(json.dumps(meta, sort_keys=True)), **arrays)
'''


def _extract_tag(autood: Path, target: Path) -> str:
    commit = subprocess.run(["git", "-C", str(autood), "rev-parse", f"{TAG}^{{commit}}"],
                            capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(autood), "archive", "--format=tar", TAG],
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    return commit


def build(autood: Path, out: Path) -> str:
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "autood_src"
        commit = _extract_tag(autood, source)
        sys.path.insert(0, str(source))
        try:
            import build_release
            pin = build_release.release_config()["gabes_commit"]
            snapshot = build_release.source_engine_snapshot(ROOT, pin, root=Path(scratch))
        finally:
            sys.path.remove(str(source))
            sys.modules.pop("build_release", None)
        import json
        out.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, GABES_DISABLE_NUMBA="1", PYTHONHASHSEED="0")
        subprocess.run([sys.executable, "-I", "-c", CHILD, str(snapshot), str(source), str(out),
                        json.dumps(CASES), json.dumps(PROBE_MW)], check=True, env=env, cwd=scratch)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"AutoOD {TAG} = {commit}\nwrote {out}\nsha256 {digest}")
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--autood", type=Path, default=DEFAULT_AUTOOD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    build(args.autood, args.out)


if __name__ == "__main__":
    main()
