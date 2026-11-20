"""Scheme registry and lightweight menu entries for the front-end."""
from dataclasses import dataclass
from importlib import import_module
from threading import RLock

from .base import ExtraView, ParamSpec, Preset, Scheme


@dataclass(frozen=True)
class _SchemeChoice:
    name: str
    title: str
    cluster: str
    module: str
    class_name: str


# The dropdown reads only these labels. Physics imports start at selection.
_CHOICES = (
    _SchemeChoice("sas", "Absorption spectroscopy (OD / SAS)", "A — Absorption",
                  "sas", "SASScheme"),
    _SchemeChoice("lambda", "Λ coherence (EIT / AT / CPT)", "A — Absorption",
                  "absorption", "LambdaScheme"),
    _SchemeChoice("rydberg_eit", "Rydberg-EIT electrometry", "A — Absorption",
                  "rydberg", "RydbergEITScheme"),
    _SchemeChoice("magneto", "Magneto-optics", "C - Magneto-optics",
                  "magneto", "MagnetoScheme"),
    _SchemeChoice("fwm", "Four-wave mixing (Squeezing / Biphoton)", "D — Wave mixing",
                  "fwm", "FWMScheme"),
    _SchemeChoice("psr", "Polarization squeezing (PSR)", "C — Magneto-optics",
                  "psr", "PSRScheme"),
)
_BY_NAME = {choice.name: choice for choice in _CHOICES}
_CLASS_MODULES = {choice.class_name: choice.module for choice in _CHOICES}
_CLASS_MODULES["ODScheme"] = "absorption"
# Single-regime physics entry points stay accessible outside the menu.
_ALIAS_PARENT = {"eit": "lambda", "at": "lambda", "cpt": "lambda",
                 "hanle": "magneto", "eia": "magneto", "nmor": "magneto"}
_instances = {}
_lock = RLock()


def choices():
    """Menu entries in display order, without loading their implementations."""
    return _CHOICES


def _instance(name):
    with _lock:
        if name not in _instances:
            choice = _BY_NAME[_ALIAS_PARENT.get(name, name)]
            factory = getattr(import_module("." + choice.module, __name__), choice.class_name)
            _instances[name] = factory(name) if name in _ALIAS_PARENT else factory()
        return _instances[name]


def all_schemes():
    """Return the concrete dropdown schemes in registration order."""
    return [_instance(choice.name) for choice in _CHOICES]


def get(name):
    """Resolve one scheme. Honor mutations to the published registry dictionary."""
    if "REGISTRY" in globals():
        return REGISTRY[name]
    return _instance(name)


def __getattr__(name):
    if name == "REGISTRY":
        # Preserve the historical, mutable dictionary and its order when requested.
        with _lock:
            if name not in globals():
                globals()[name] = {key: _instance(key) for key in (*_BY_NAME, *_ALIAS_PARENT)}
            return globals()[name]
    if name in _CLASS_MODULES:
        return getattr(import_module("." + _CLASS_MODULES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_CLASS_MODULES) | {"REGISTRY"})


__all__ = ["Scheme", "ParamSpec", "Preset", "ExtraView",
           "REGISTRY", "all_schemes", "choices", "get"]
