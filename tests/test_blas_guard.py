"""The process-wide BLAS limit survives overlapping and nested guards."""
import threading

import pytest

from gabes import core

threadpoolctl = pytest.importorskip("threadpoolctl")


def blas_threads():
    return [info["num_threads"] for info in threadpoolctl.threadpool_info()
            if info["user_api"] == "blas"]


@pytest.fixture
def several_threads():
    original = blas_threads()
    if not original:
        pytest.skip("No BLAS library is loaded.")
    with threadpoolctl.threadpool_limits(limits=4, user_api="blas"):
        start = blas_threads()
        if max(start) < 2:
            pytest.skip("BLAS cannot run more than one thread here.")
        yield start
    assert blas_threads() == original


def test_interleaved_guards_restore_the_original_limit(several_threads):
    first, second = core.blas_single_thread(), core.blas_single_thread()
    first.__enter__()
    second.__enter__()
    assert max(blas_threads()) == 1
    first.__exit__(None, None, None)
    assert max(blas_threads()) == 1            # the second guard is still active
    second.__exit__(None, None, None)
    assert blas_threads() == several_threads


def test_nested_guard_and_exception_restore(several_threads):
    with pytest.raises(RuntimeError):
        with core.blas_single_thread():
            with core.blas_single_thread():
                assert max(blas_threads()) == 1
            assert max(blas_threads()) == 1
            raise RuntimeError
    assert blas_threads() == several_threads


def test_threads_overlapping_in_time(several_threads):
    inside, release = threading.Barrier(3), threading.Event()
    seen = []

    def work():
        with core.blas_single_thread():
            inside.wait()
            release.wait()
            seen.append(max(blas_threads()))

    workers = [threading.Thread(target=work) for _ in range(2)]
    for worker in workers:
        worker.start()
    inside.wait()
    release.set()
    for worker in workers:
        worker.join()
    assert seen == [1, 1]
    assert blas_threads() == several_threads
