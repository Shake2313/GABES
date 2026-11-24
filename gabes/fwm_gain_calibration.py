"""Readout-only selection of practical FWM gain calibrations."""
from . import fwm_experiment_gain


MODEL_LAB = "2026-10-13 lab"
MODEL_LAB_CURRENT = "2026-10-13 lab (corrected atomic model)"
MODEL_LAB_REFIT = MODEL_LAB_CURRENT
_RETIRED_LAB_SELECTORS = (MODEL_LAB, "2026-10-13 lab (exact-source refit)")
MODEL_LITERATURE = "Multi-paper literature (experimental)"
MODEL_CHI = "Multi-paper χ propagation (experimental)"
MODEL_STARK = "Multi-paper pump/Stark + χ (experimental)"
MODEL_ZEEMAN = "Multi-paper Zeeman pump + χ (experimental)"
MODELS = (MODEL_LAB_CURRENT, MODEL_LITERATURE, MODEL_CHI, MODEL_STARK, MODEL_ZEEMAN)


def normalize_model(model=MODEL_LAB_CURRENT):
    """Resolve older saved lab selector labels to the current measured-data fit."""
    return MODEL_LAB_CURRENT if model in _RETIRED_LAB_SELECTORS else model


def _module(model):
    if model == MODEL_LAB_CURRENT:
        return fwm_experiment_gain
    if model == MODEL_LITERATURE:
        from . import fwm_literature_gain
        return fwm_literature_gain
    if model == MODEL_CHI:
        from . import fwm_chi_gain
        return fwm_chi_gain
    if model == MODEL_STARK:
        from . import fwm_stark_gain
        return fwm_stark_gain
    if model == MODEL_ZEEMAN:
        from . import fwm_zeeman_gain
        return fwm_zeeman_gain
    raise ValueError(f"Unknown FWM gain calibration model: {model!r}")


def profile(*, model=MODEL_LAB_CURRENT):
    model = normalize_model(model)
    result = _module(model).profile()
    result["calibration_model"] = model
    return result


def evaluate(raw, delta_mhz, *, enabled=True, model=MODEL_LAB_CURRENT):
    model = normalize_model(model)
    result = _module(model).evaluate(raw, delta_mhz, enabled=enabled)
    result["calibration_model"] = model
    result.setdefault("display_name", model)
    return result


def calibrate_spectrum(raw, *, enabled=False, selected_delta_mhz=None,
                       model=MODEL_LAB_CURRENT):
    model = normalize_model(model)
    result = _module(model).calibrate_spectrum(
        raw, enabled=enabled, selected_delta_mhz=selected_delta_mhz)
    result["calibration_model"] = model
    result.setdefault("display_name", model)
    return result
