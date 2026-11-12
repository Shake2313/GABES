"""gabes_lab.blocks: provenance, io, figures, snapshot, runner, publish_to_archive."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gabes_lab.blocks import figures, io, provenance, runner, snapshot
from gabes_lab.blocks.paths import REPO_ROOT
from gabes_lab.blocks.publish_to_archive import (
    MANIFEST, PublishError, check_archive, main as publish_main, publish_to_archive,
)


# provenance -----------------------------------------------------------------

def test_sha256_file_streams_the_same_digest(tmp_path):
    data = bytes(range(256)) * 9000          # > 1 MiB, several chunks
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert provenance.sha256_file(path) == hashlib.sha256(data).hexdigest()
    assert provenance.sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_portable_path_hides_locations_outside_the_root(tmp_path):
    assert provenance.portable_path(REPO_ROOT / "gabes" / "core.py") == "gabes/core.py"
    outside = provenance.portable_path(tmp_path / "my data (1).csv")
    assert outside.startswith("external://my_data__1_.csv#path-sha256=")
    assert str(tmp_path) not in outside and len(outside.rsplit("=", 1)[1]) == 16


def test_file_record_and_utc_now(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    record = provenance.file_record(path, root=tmp_path)
    assert record == {"path": "a.txt", "sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}
    assert provenance.utc_now().endswith("+00:00") and "." not in provenance.utc_now()


def test_git_state_in_and_outside_a_repository(tmp_path):
    state = provenance.git_state()
    assert set(state) == {"commit", "branch", "dirty"}
    assert state["commit"] is None or len(state["commit"]) == 40
    assert provenance.git_state(tmp_path) == {"commit": None, "branch": None, "dirty": False}


def test_github_slug_parses_remote_urls(monkeypatch):
    for url, slug in (("https://github.com/Owner/Repo.git", "Owner/Repo"),
                      ("git@github.com:Owner/Repo.git", "Owner/Repo"),
                      ("https://github.com/Owner/Repo", "Owner/Repo"),
                      ("https://gitlab.com/Owner/Repo.git", None)):
        monkeypatch.setattr(provenance, "git", lambda *a, url=url, **k: url)
        assert provenance.github_slug() == slug


# io ---------------------------------------------------------------------------

def test_write_json_layouts(tmp_path):
    payload = {"name": "Γ", "x": [1, 2]}
    path = io.write_json(tmp_path / "deep" / "a.json", payload)
    # Text mode like Path.write_text: the platform newline is written.
    assert path.read_text(encoding="utf-8") == json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with pytest.raises(ValueError):
        io.write_json(tmp_path / "nan.json", {"x": math.nan})
    compact = io.write_json(tmp_path / "b.json", {"x": np.float64(1.5)}, indent=1,
                            allow_nan=True, default=float, trailing_newline=False)
    assert compact.read_text(encoding="utf-8") == '{\n "x": 1.5\n}'
    assert io.read_json(path) == payload


# figures ----------------------------------------------------------------------

def test_save_figure_writes_each_format_and_closes(tmp_path):
    plt = figures.pyplot()
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    paths = figures.save_figure(fig, tmp_path / "sub" / "fig_v1.2", ("png", "PDF", ".svg"), dpi=50)
    assert [p.name for p in paths] == ["fig_v1.2.png", "fig_v1.2.pdf", "fig_v1.2.svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert not plt.fignum_exists(fig.number)


# snapshot ---------------------------------------------------------------------

@dataclasses.dataclass
class _Result:
    xi: np.ndarray
    label: str
    extra: object = None


def test_snapshot_capture_save_load_and_compare(tmp_path):
    value = {"a": _Result(np.array([1.0, np.nan, 3.0]), "ok"), "b": [1, "s", True],
             "c/d": {}, "z": 2 + 1j, "objs": np.array([1, "x"], dtype=object)}
    snap = snapshot.capture(value)
    assert {"a/xi", "a/label", "a/extra#none", "b/0", "b/1", "b/2", "c%2Fd#empty", "z",
            "objs/0", "objs/1"} == set(snap)
    loaded = snapshot.load(snapshot.save(tmp_path / "s.npz", snap))
    same = snapshot.compare(snap, loaded)
    assert same.identical and same.within() and "bit-identical" in same.report()

    nudged = dict(value, a=_Result(np.array([1.0 + 1e-15, np.nan, 3.0]), "ok"))
    near = snapshot.compare(snap, snapshot.capture(nudged))
    assert not near.identical and near.within(1e-12) and not near.within(1e-16)

    other = dict(value, a=_Result(np.array([1.0, 2.0, 3.0]), "ok"), b=[1, "t", True])
    far = snapshot.compare(snap, snapshot.capture(other))
    assert dict(far.changed) == {"a/xi": math.inf, "b/1": math.inf}
    assert not far.within()

    renamed = snapshot.compare(snap, snapshot.capture(dict(value, a=None)))
    assert "a/xi" in renamed.missing and "a#none" in renamed.extra


def test_snapshot_relative_difference_rules():
    rd = snapshot.relative_difference
    assert rd(np.array([0.0, 2.0]), np.array([0.0, 2.0 * (1 + 1e-13)])) == pytest.approx(1e-13, rel=1e-3)
    assert rd(np.array([np.inf]), np.array([np.inf])) == 0.0
    assert rd(np.array([np.inf]), np.array([1.0])) == math.inf
    assert rd(np.array([1.0, 2.0]), np.array([1.0])) == math.inf
    assert rd(np.array(1), np.array(1.0)) == 0.0
    cmp = snapshot.compare({"k": np.array(1)}, {"k": np.array(1.0)})
    assert cmp.changed == (("k", 0.0),) and not cmp.identical and cmp.within()


def test_snapshot_rejects_cycles():
    loop: list = []
    loop.append(loop)
    with pytest.raises(ValueError, match="cycle"):
        snapshot.capture(loop)


def test_snapshot_cli(tmp_path, capsys):
    a = snapshot.save(tmp_path / "a.npz", snapshot.capture({"x": np.arange(3.0)}))
    b = snapshot.save(tmp_path / "b.npz", snapshot.capture({"x": np.arange(3.0) + 1}))
    assert snapshot.main(["compare", str(a), str(a)]) == 0
    assert snapshot.main(["compare", str(a), str(b)]) == 1
    assert "changed x" in capsys.readouterr().out


# runner -----------------------------------------------------------------------

def test_run_records_outputs_and_status(tmp_path):
    run = runner.start_run("demo/scan", runs_root=tmp_path, config={"T": np.float64(1.5)},
                           argv=["--quick"])
    assert run.dir.parent == tmp_path / "demo" / "scan"
    started = io.read_json(run.dir / runner.RUN_RECORD)
    assert started["status"] == "running" and started["config"] == {"T": 1.5}
    np.save(run.path("data", "x.npy"), np.arange(3))
    run.path("note.txt").write_text("hi", encoding="utf-8")
    record = run.finish()
    assert record == io.read_json(run.dir / runner.RUN_RECORD)
    assert record["status"] == "complete" and record["argv"] == ["--quick"]
    assert [o["path"] for o in record["outputs"]] == ["data/x.npy", "note.txt"]
    assert record["outputs"][1]["sha256"] == hashlib.sha256(b"hi").hexdigest()
    assert set(record["git"]) == {"commit", "branch", "dirty"}

    second = runner.start_run("demo/scan", runs_root=tmp_path)
    assert second.dir != run.dir and second.dir.parent == run.dir.parent


def test_run_context_manager_records_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with runner.start_run("demo", out=tmp_path / "explicit") as run:
            raise RuntimeError("boom")
    record = io.read_json(tmp_path / "explicit" / runner.RUN_RECORD)
    assert record["status"] == "failed" and record["error"] == "RuntimeError: boom"
    assert run.dir == (tmp_path / "explicit").resolve()


@pytest.mark.parametrize("name", ["", "../x", "/abs", "a/../b"])
def test_run_rejects_bad_pipeline_names(tmp_path, name):
    with pytest.raises(ValueError):
        runner.start_run(name, runs_root=tmp_path)


# publish_to_archive -------------------------------------------------------------

def _archive(tmp_path) -> Path:
    root = tmp_path / "GABES-archive"
    (root / "engine").mkdir(parents=True)
    (root / ".gitmodules").write_text('[submodule "engine"]\n\tpath = engine\n', encoding="utf-8")
    return root


def _finished_run(tmp_path, dirty=False) -> Path:
    run = runner.start_run("demo", out=tmp_path / "run")
    run.path("out", "x.txt").write_text("result", encoding="utf-8")
    run.record["git"] = {"commit": "a" * 40, "branch": "main", "dirty": dirty}
    run.finish()
    return run.dir


def test_publish_copies_files_with_manifest(tmp_path):
    archive, source = _archive(tmp_path), _finished_run(tmp_path)
    dry = publish_to_archive(source, archive, "analysis/demo", dry_run=True)
    assert not (archive / "analysis").exists()
    manifest = publish_to_archive(source, archive, "analysis/demo", note="test")
    target = archive / "analysis" / "demo"
    assert (target / "out" / "x.txt").read_text(encoding="utf-8") == "result"
    assert io.read_json(target / MANIFEST) == manifest
    assert [f["path"] for f in manifest["files"]] == ["out/x.txt", "run.json"]
    assert [f["path"] for f in dry["files"]] == ["out/x.txt", "run.json"]
    assert manifest["source"]["commit"] == "a" * 40 and manifest["git_state_from"] == "run.json"
    assert manifest["pipeline"] == "demo" and manifest["note"] == "test"
    with pytest.raises(PublishError, match="not empty"):
        publish_to_archive(source, archive, "analysis/demo")


def test_publish_refusals(tmp_path):
    archive = _archive(tmp_path)
    with pytest.raises(PublishError, match="not a GABES-archive"):
        check_archive(tmp_path)
    dirty = _finished_run(tmp_path / "d", dirty=True)
    with pytest.raises(PublishError, match="dirty"):
        publish_to_archive(dirty, archive, "x")
    assert publish_to_archive(dirty, archive, "x", allow_dirty=True)["source"]["dirty"] is True
    clean = _finished_run(tmp_path / "c")
    for dest in ("../out", "/abs", "engine/x", ""):
        with pytest.raises(PublishError, match="--dest"):
            publish_to_archive(clean, archive, dest)
    (clean / "out" / "x.txt").write_text("edited", encoding="utf-8")
    with pytest.raises(PublishError, match="changed"):
        publish_to_archive(clean, archive, "y")
    running = runner.start_run("demo", out=tmp_path / "r")
    with pytest.raises(PublishError, match="not 'complete'"):
        publish_to_archive(running.dir, archive, "z")


def test_publish_plain_folder_uses_publish_time_git_state(tmp_path, monkeypatch, capsys):
    archive = _archive(tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "fig.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(provenance, "git_state",
                        lambda root=None: {"commit": "b" * 40, "branch": "main", "dirty": False})
    assert publish_main([str(plain), "--archive", str(archive), "--dest", "docs/p"]) == 0
    manifest = io.read_json(archive / "docs" / "p" / MANIFEST)
    assert manifest["git_state_from"] == "publish time" and manifest["pipeline"] is None
    assert [f["path"] for f in manifest["files"]] == ["fig.pdf"]
    assert publish_main([str(plain), "--archive", str(tmp_path), "--dest", "q"]) == 2
    assert "refused" in capsys.readouterr().err
