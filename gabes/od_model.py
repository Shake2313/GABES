"""Pump-free single-beam OD contract for the theory plot, OD fitting and AutoOD.

Decision D20 (docs/ui_redesign/decisions.md): the OD-fit view computes with the
pump off, while the theory view keeps its pump settings. One evaluator returns
transmission at exactly the requested optical frequencies, so a plotted scan
and fit residuals sample the same function.

Frequencies are offsets in Hz from a named reference: the Rb85 fine-structure
centroid of the selected D line for every Rb medium, as in
``optical_frequency.RbOpticalFrequency``, and the Cs133 centroid for Cs. The
SAS scheme draws pure Rb87 against the Rb87 centroid; add
``display_reference_offset_hz`` to that axis to obtain contract offsets.

M0 scope: the zero-probe-power models already in ``SASScheme`` (legacy velocity
grid and Exact Maxwell Voigt) on a uniform cell. Positive probe power and the
reserved model IDs raise ``ODUnsupportedError`` with the reason, until the shared
magnetic-sublevel model lands. Units follow the SAS controls (°C, mm, µW, kHz).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import math

import numpy as np

from . import observables, species
from ._arrays import readonly_float_array
from .optical_frequency import reference_offset_hz
from .schemes.sas import (
    EXACT_WEAK_PROBE, LEGACY_WEAK_PROBE, RB_COMPOSITION_CUSTOM, RB_COMPOSITION_NATURAL,
    SASScheme, _probe_power_mw, _weak_probe_model_id, _weak_probe_profile,
)


MODEL_KEY = "od_model"   # reserved parameter name for an explicit saved model ID
PUMP_GROUP = "Pump"
TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class ODModel:
    model_id: str
    description: str
    status: str                       # implemented | scheme_only | reserved
    finite_power: bool
    weak_probe_profile: str | None = None


MODELS = {model.model_id: model for model in (
    ODModel("weak_probe_legacy", "Legacy 1 m/s velocity grid, zero probe power",
            "implemented", False, LEGACY_WEAK_PROBE),
    ODModel("weak_probe_exact_maxwell_voigt",
            "Exact Maxwell Voigt without a transit width, zero probe power",
            "implemented", False, EXACT_WEAK_PROBE),
    ODModel("finite_power_2level_gaussian_propagation",
            "Current SAS finite-probe path; needs a uniform scan axis", "scheme_only", True),
    ODModel("od_weak_exact_voigt_transit_v1",
            "Exact Maxwell Voigt with the transit HWHM (user decision U1)", "reserved", False),
    ODModel("od_finite_rb_mf_rate_v1",
            "Magnetic-sublevel rate model for a linearly polarized Rb probe", "reserved", True),
)}


class ODUnsupportedError(ValueError):
    """The requested conditions or model are outside the implemented contract."""


@dataclass(frozen=True)
class FrequencyReference:
    isotope: str
    line: str
    point: str = "fine_centroid"

    @property
    def label(self):
        return f"{self.isotope} {self.line} {self.point.replace('_', ' ')}"


@dataclass(frozen=True)
class ODConditions:
    """Every input of a pump-free OD solve; there are no pump fields.

    ``doppler_temp_c`` None ties the Doppler temperature to the cold spot.
    ``rb_composition`` and ``rb85_fraction`` apply to "Rb (natural)" only.
    ``line_strength`` scales absorption at propagation, as in the SAS readout.
    """

    species: str
    line: str
    temp_c: float
    cell_mm: float
    model_id: str
    doppler_temp_c: float | None = None
    rb_composition: str = RB_COMPOSITION_NATURAL
    rb85_fraction: float = species.RB85.abundance
    probe_power_uw: float = 0.0
    probe_waist_mm: float = 1.0
    transit_khz: float = 100.0
    line_strength: float = 1.0
    ne_pressure_torr: float = 0.0
    optical_collision_model: str | None = None
    paraffin_coated: bool = False
    cs_self_broadening_reference: bool = False
    extra_homogeneous_fwhm_hz: float = 0.0

    def __post_init__(self):
        if self.species not in species.SPECIES_ORDER:
            raise ODUnsupportedError("Choose a resolved Rb or Cs medium.")
        if self.line not in ("D1", "D2"):
            raise ODUnsupportedError("Choose the D1 or D2 line.")
        if self.model_id not in MODELS:
            raise ODUnsupportedError(f"Unknown OD model ID {self.model_id!r}.")
        if self.rb_composition not in (RB_COMPOSITION_NATURAL, RB_COMPOSITION_CUSTOM):
            raise ODUnsupportedError("Rb isotope composition must be Natural or Custom.")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ODUnsupportedError(f"{item.name} must be finite.")
        if self.probe_power_uw < 0 or self.extra_homogeneous_fwhm_hz < 0 or self.line_strength <= 0:
            raise ODUnsupportedError("Probe power and extra width must be nonnegative, line strength positive.")

    @classmethod
    def from_params(cls, params, *, extra_homogeneous_fwhm_hz=0.0):
        """Read SAS parameters; pump settings are ignored, as in the OD-fit view.

        An explicit ``od_model`` entry wins. Without one, the model is the one the
        SAS scheme applies to these parameters today, so older saved settings
        replay unchanged.
        """
        params = {**SASScheme().defaults(), **od_fit_conditions(params)}
        if params.get("species") not in species.SPECIES_ORDER:
            raise ODUnsupportedError("Choose a resolved Rb or Cs medium.")
        if params.get("cell_propagation", "Uniform") != "Uniform":
            raise ODUnsupportedError("Choose the Uniform cell for OD fitting.")
        model_id = params.get(MODEL_KEY) or _scheme_model_id(params)
        tied = bool(params.get("constrain_doppler_temp", True))
        return cls(
            species=params["species"], line=params["line"], temp_c=float(params["temp_c"]),
            cell_mm=float(params["cell_mm"]), model_id=model_id,
            doppler_temp_c=None if tied else float(params.get("doppler_temp_c", params["temp_c"])),
            rb_composition=params.get("rb_composition", RB_COMPOSITION_NATURAL),
            rb85_fraction=float(params.get("rb85_fraction", species.RB85.abundance)),
            probe_power_uw=_probe_power_mw(params) * 1e3,
            probe_waist_mm=float(params.get("probe_waist_mm", 1.0)),
            transit_khz=float(params["transit_khz"]),
            line_strength=float(params.get("line_strength", 1.0)),
            ne_pressure_torr=float(params.get("ne_pressure_torr", 0.0)),
            optical_collision_model=params.get("optical_collision_model"),
            paraffin_coated=bool(params.get("paraffin_coated", False)),
            cs_self_broadening_reference=bool(params.get("cs_self_broadening_reference", False)),
            extra_homogeneous_fwhm_hz=float(extra_homogeneous_fwhm_hz),
        )

    def to_params(self):
        """SAS parameters for this solve: pump off, uniform cell, profile from the model."""
        params = od_fit_conditions(SASScheme().defaults())
        params.update(
            species=self.species, line=self.line, temp_c=self.temp_c, cell_mm=self.cell_mm,
            constrain_doppler_temp=self.doppler_temp_c is None,
            doppler_temp_c=self.temp_c if self.doppler_temp_c is None else self.doppler_temp_c,
            rb_composition=self.rb_composition, rb85_fraction=self.rb85_fraction,
            probe_power_uw=self.probe_power_uw, probe_waist_mm=self.probe_waist_mm,
            transit_khz=self.transit_khz, line_strength=self.line_strength,
            ne_pressure_torr=self.ne_pressure_torr, paraffin_coated=self.paraffin_coated,
            cs_self_broadening_reference=self.cs_self_broadening_reference,
            cell_propagation="Uniform",
        )
        if self.optical_collision_model is not None:
            params["optical_collision_model"] = self.optical_collision_model
        profile = MODELS[self.model_id].weak_probe_profile
        if profile is not None:
            params["weak_probe_profile"] = profile
        params[MODEL_KEY] = self.model_id
        return params


@dataclass(frozen=True)
class ODEvaluation:
    offsets_hz: np.ndarray
    transmission: np.ndarray
    optical_depth: np.ndarray
    od0: np.ndarray
    model_id: str
    frequency_reference: FrequencyReference
    conditions: ODConditions
    weak_probe_profile: dict | None = field(default=None, compare=False)


def od_fit_conditions(params):
    """Return a copy with the pump off and every other pump control at its default.

    Hidden pump settings then cannot change a pump-free solve or its cache key.
    The caller's mapping, for example Streamlit session state, is not modified.
    """
    result = dict(params)
    for spec in SASScheme().param_schema():
        if spec.group == PUMP_GROUP:
            result[spec.name] = 0.0 if spec.name == "pump_power_mw" else spec.default
    return result


def _scheme_model_id(params):
    if _probe_power_mw(params) > 0.0:
        return "finite_power_2level_gaussian_propagation"
    return _weak_probe_model_id(_weak_probe_profile(params))


def frequency_reference(conditions):
    if conditions.species == species.CS133.label:
        return FrequencyReference(species.CS133.name, conditions.line)
    return FrequencyReference(species.RB85.name, conditions.line)


def display_reference_offset_hz(conditions):
    """SAS display origin minus the contract reference, in Hz."""
    if conditions.species == species.RB87.label:
        return reference_offset_hz(species.RB87.name, conditions.line)
    return 0.0


def od_unavailable(conditions):
    """Reason the evaluator cannot run these conditions, or None."""
    model = MODELS[conditions.model_id]
    if model.status == "reserved":
        return f"Model {model.model_id} is reserved and not implemented yet."
    if model.finite_power or conditions.probe_power_uw > 0.0:
        return ("Finite probe power needs the shared magnetic-sublevel model, which is not "
                "implemented yet. Set the probe power to zero.")
    if _scheme_model_id(conditions.to_params()) != model.model_id:
        return f"{model.description} does not apply to these conditions."
    return None


def evaluate_od(conditions, offsets_hz):
    """Transmission and optical depth at exactly the requested offsets.

    Any order, repeated values and gaps are accepted; each value depends only
    on its own frequency. The result arrays are read-only.
    """
    if reason := od_unavailable(conditions):
        raise ODUnsupportedError(reason)
    offsets = np.asarray(offsets_hz, dtype=float)
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("Offsets must be a nonempty, finite 1-D array in Hz.")
    params = conditions.to_params()
    scan = (offsets - display_reference_offset_hz(conditions)) * TWO_PI
    raw = SASScheme()._compute_species(
        params, scan_rad_s=scan, extra_homogeneous_fwhm_hz=conditions.extra_homogeneous_fwhm_hz)
    if raw["probe_model"] != conditions.model_id:
        raise ODUnsupportedError(
            f"The SAS solve applied {raw['probe_model']}, not {conditions.model_id}.")
    alpha = raw["alpha_unit"] * conditions.line_strength
    length = conditions.cell_mm * 1e-3
    depth = readonly_float_array(observables.optical_depth(alpha, length))
    return ODEvaluation(
        offsets_hz=readonly_float_array(offsets),
        transmission=readonly_float_array(observables.transmission(alpha, length)),
        optical_depth=depth, od0=depth, model_id=conditions.model_id,
        frequency_reference=frequency_reference(conditions), conditions=conditions,
        weak_probe_profile=raw.get("weak_probe_profile"),
    )


def with_extra_width(conditions, extra_homogeneous_fwhm_hz):
    return replace(conditions, extra_homogeneous_fwhm_hz=float(extra_homogeneous_fwhm_hz))
