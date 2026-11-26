"""Keep finite-power field-history calculations off the Streamlit session thread."""
from concurrent.futures import CancelledError, ThreadPoolExecutor

import streamlit as st

from .od_fit_job import ODFitJob


_KEY = "sas_od_field_history_job"
# An exploratory coherent scan can run much longer than an ordinary CSV fit.
_FIELD_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gabes-od-field")


def cancel_field_job():
    job = st.session_state.pop(_KEY, None)
    if job is not None:
        job.cancel()


def _solve(scheme_name, params, *, progress):
    from gabes import schemes
    from gabes.field_spectrum import field_progress
    with field_progress(progress):
        return schemes.get(scheme_name).compute(params)


def needs_field_job(scheme_name, params):
    if scheme_name != "sas" or params.get("probe_power_uw", 0) <= 0 or params.get("cell_mm", 0) <= 0:
        return False
    from gabes.od_model import FIELD_HISTORY_REFERENCE
    from gabes.schemes.sas import ODModelSettingError, _od_model_setting, _pump_free_model_reason
    if _od_model_setting(params)[1] != FIELD_HISTORY_REFERENCE:
        return False
    reason = _pump_free_model_reason(params, FIELD_HISTORY_REFERENCE)
    if reason:
        raise ODModelSettingError(reason)
    return True


@st.fragment(run_every=1.0)
def _poll(job):
    if st.session_state.get(_KEY) is not job or job.future.done():
        st.rerun(scope="app")
    status = job.observe()
    if status is None:
        st.caption("Waiting for the field-history solver…")
    else:
        isotope, done, total = status
        st.progress(done / total, text=f"{isotope} · {done} / {total} rest frequencies")
    if st.button("Cancel calculation", key="sas_cancel_field_history"):
        cancel_field_job()
        st.rerun(scope="app")


def field_result(scheme, items, version, pending_controls):
    """Return completed raw data only for this exact set of solve inputs."""
    identity = (scheme.name, items, version)
    job = st.session_state.get(_KEY)
    if job is not None and job.identity != identity:
        cancel_field_job()
        job = None
    if job is not None and job.future.done():
        try:
            return job.future.result()
        except CancelledError:
            cancel_field_job()
        except Exception as error:
            cancel_field_job()
            st.error(f"Field-history calculation stopped: {error}")
        job = None
    pending_controls()
    if job is None:
        st.caption("A finite-power scan can take minutes to hours and runs in the background. "
                   "A new field, polarization or beam setting can require a new scan.")
        if st.button("Calculate field-history spectrum", key="sas_start_field_history"):
            job = ODFitJob(identity, scheme.name, dict(items), fit_function=_solve, executor=_FIELD_WORKER)
            st.session_state[_KEY] = job
    if job is not None:
        _poll(job)
    return None
