"""The shared pole-solver executor is created once, even by overlapping first callers."""
import threading
import time

import numpy as np
import pytest

from gabes import pole_doppler


@pytest.fixture
def fresh_pool(monkeypatch):
    monkeypatch.setattr(pole_doppler, "_POOL", None)
    created = []
    yield created
    for pool in created:
        pool.shutdown(wait=True)


def counting_executor(created, *, delay=0.0, fail_first=False):
    real = pole_doppler.ThreadPoolExecutor
    attempts = []

    def factory(*args, **kwargs):
        attempts.append(1)
        if fail_first and len(attempts) == 1:
            raise RuntimeError("construction failed")
        if delay and len(attempts) == 1:
            time.sleep(delay)          # hold the first constructor inside the critical section
        pool = real(*args, **kwargs)
        created.append(pool)
        return pool

    return factory, attempts


def test_overlapping_first_callers_share_one_executor(monkeypatch, fresh_pool):
    factory, attempts = counting_executor(fresh_pool, delay=0.2)
    monkeypatch.setattr(pole_doppler, "ThreadPoolExecutor", factory)
    start = threading.Barrier(2)
    seen, results = [], []

    def caller(value):
        start.wait()
        pool = pole_doppler._pool()
        seen.append(pool)
        results.append(pool.submit(lambda: value * value).result(timeout=10))

    threads = [threading.Thread(target=caller, args=(value,)) for value in (3, 4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert len(attempts) == 1 and len(fresh_pool) == 1
    assert seen[0] is seen[1] is pole_doppler._POOL
    assert sorted(results) == [9, 16]


def test_failed_construction_is_retried(monkeypatch, fresh_pool):
    factory, attempts = counting_executor(fresh_pool, fail_first=True)
    monkeypatch.setattr(pole_doppler, "ThreadPoolExecutor", factory)
    with pytest.raises(RuntimeError, match="construction failed"):
        pole_doppler._pool()
    assert pole_doppler._POOL is None
    pool = pole_doppler._pool()
    assert pool is pole_doppler._POOL and len(attempts) == 2
    assert pool.submit(lambda: 7).result(timeout=10) == 7


def test_serial_and_pooled_eigen_solves_agree(monkeypatch):
    rng = np.random.default_rng(3)
    stacks = [rng.normal(size=(40, 6, 6)), rng.normal(size=(17, 6, 6))]
    monkeypatch.setattr(pole_doppler, "_POOL_WORKERS", 4)
    pooled = pole_doppler._eig_stacks(stacks)
    monkeypatch.setattr(pole_doppler, "_POOL_WORKERS", 1)

    def unexpected_pool():
        raise AssertionError("the single-worker path must not create the executor")

    monkeypatch.setattr(pole_doppler, "_pool", unexpected_pool)
    serial = pole_doppler._eig_stacks(stacks)
    for (lam_a, vec_a), (lam_b, vec_b) in zip(pooled, serial, strict=True):
        np.testing.assert_array_equal(lam_a, lam_b)
        np.testing.assert_array_equal(vec_a, vec_b)
