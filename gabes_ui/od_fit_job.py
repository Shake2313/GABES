"""One bounded fit worker shared by web sessions. Workers never touch Streamlit."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy
from threading import Event, Lock
from time import monotonic

from gabes.core import blas_single_thread

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gabes-od-fit")
SESSION_KEY = "sas_od_fit_job"
PROFILE_JOB = "sas_od_profile_job"


class ODFitJob:
    """Run a fit on its own prepared branch and a snapshot of the model inputs."""

    def __init__(self, identity, data, params, *, fit_function=None, **options):
        self.identity = identity
        self.visible = True
        self._cancel = Event()
        self._lock = Lock()
        self._last_seen = monotonic()
        self._progress = None
        self.future = _EXECUTOR.submit(self._run, fit_function, data, deepcopy(params), options)

    def _run(self, fit_function, data, params, options):
        if fit_function is None:
            from gabes.od_spectrum_fit import fit_od_spectrum as fit_function
        with self._lock:
            if not self._continue():
                raise CancelledError()
        with blas_single_thread():
            return fit_function(data, params, progress=self._report, **options)

    def _continue(self):
        return not self._cancel.is_set() and monotonic() - self._last_seen <= 120

    def _report(self, progress):
        with self._lock:
            self._progress = progress
            return self._continue()

    def observe(self):
        with self._lock:
            self._last_seen = monotonic()
            return self._progress

    def cancel(self):
        self._cancel.set()
        self.future.cancel()
