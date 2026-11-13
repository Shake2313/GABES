"""GABES and SABES coordinate Matplotlib work across reruns."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gabes_ui import plotcard
from gabes_ui.sabes import page


@pytest.fixture(autouse=True)
def noninteractive_canvas():
    # The deployed app uses a noninteractive backend. Test Matplotlib work,
    # without asking a platform GUI event loop to run on worker threads.
    with matplotlib.rc_context({"backend": "Agg"}):
        yield


def test_concurrent_cold_and_warm_sabes_observables_use_gabes_lock(monkeypatch):
    entered = Event()
    inside_gabes_draw = Event()
    release_gabes_draw = Event()
    labels = []

    def draw(figure):
        assert plotcard.PLOT_LOCK._is_owned()
        labels.append(figure.axes[0].get_xlabel())
        if figure.axes[0].get_xlabel() == "GABES":
            inside_gabes_draw.set()
            assert release_gabes_draw.wait(5)
        figure.canvas.draw()

    def build(_scheme, _raw, params):
        assert plotcard.PLOT_LOCK._is_owned()
        entered.set()
        figure, axis = plt.subplots()
        axis.set_xlabel(params["label"])
        return {"figure": figure}

    def sabes_draw(label):
        cached = page._cached_observables({"input": label}, (("label", label),), "shared-lock-test")
        plotcard.render_fig(cached["figure"])

    monkeypatch.setattr(plotcard.st, "pyplot", draw)
    monkeypatch.setattr(page.fwm.FWMScheme, "observables", build)
    page._cached_observables.clear()
    try:
        with plotcard.PLOT_LOCK:
            figure, axis = plt.subplots()
            axis.set_xlabel("GABES")
        with ThreadPoolExecutor(max_workers=2) as pool:
            gabes = pool.submit(plotcard.render_fig, figure)
            assert inside_gabes_draw.wait(5)
            started = Event()

            def start_sabes():
                started.set()
                sabes_draw("SABES")

            sabes = pool.submit(start_sabes)
            try:
                assert started.wait(5)
                assert not entered.wait(.05)
            finally:
                release_gabes_draw.set()
            gabes.result(timeout=5)
            sabes.result(timeout=5)
        assert entered.is_set()
        entered.clear()
        sabes_draw("SABES")
        assert not entered.is_set()
        assert labels == ["GABES", "SABES", "SABES"]
    finally:
        page._cached_observables.clear()


def test_shared_lock_is_reentrant_and_released_after_draw_failure(monkeypatch):
    before = set(plt.get_fignums())

    def broken_draw(_figure):
        assert plotcard.PLOT_LOCK._is_owned()
        raise RuntimeError("draw failed")

    monkeypatch.setattr(plotcard.st, "pyplot", broken_draw)
    trace = SimpleNamespace(kind="line", x=np.array([0., 1.]), series={"signal": [1., 2.]},
                            x_label="Frequency", x_unit="Hz", y_label="Signal", y_unit="V")
    with pytest.raises(RuntimeError, match="draw failed"):
        page._render_trace(trace)
    assert set(plt.get_fignums()) == before
    with plotcard.PLOT_LOCK:
        figure, _axis = plt.subplots()
        with pytest.raises(RuntimeError, match="draw failed"):
            plotcard.render_fig(figure)
    assert set(plt.get_fignums()) == before
    with ThreadPoolExecutor(max_workers=1) as pool:
        def probe_lock():
            if plotcard.PLOT_LOCK.acquire(timeout=2):
                plotcard.PLOT_LOCK.release()
                return True
            return False
        assert pool.submit(probe_lock).result(timeout=3)
