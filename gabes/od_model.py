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

Implemented models (``MODELS``): the zero-probe SAS profiles (legacy velocity
grid and Exact Maxwell Voigt), the Exact profile with the transit HWHM (user
decision U1), and the magnetic-sublevel rate model for a linearly polarized Rb
probe ported from AutoOD 0.2.2. Pure ⁸⁵Rb D1 also supports opt-in ballistic
beam-crossing histories for a pump-free, uncoated, buffer-free cell. ``from_params`` still selects today's SAS model
when no ``od_model`` is saved; the defaults change in a separate value commit.
The current two-level finite-probe SAS path is not evaluable on arbitrary
offsets. Units follow the SAS controls (°C, mm, µW, kHz).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
import math

import numpy as np

from . import constants, core, linear_hyperfine, observables, probe_saturation, species
from .transit import gaussian_transit_rate_s, rate_from_khz_over_2pi
from ._arrays import readonly_float_array
from .optical_frequency import reference_offset_hz
from .optical_pumping import MAGNETIC_PI, ProjectedProbe
from .schemes.sas import (
    EXACT_WEAK_PROBE, LEGACY_WEAK_PROBE, OD_MODEL_KEY, RB_COMPOSITION_CUSTOM,
    GAUSSIAN_TRANSIT_MODEL, GAUSSIAN_TRANSIT_D20_MODEL, GAUSSIAN_TRANSIT_MODELS, GAUSSIAN_TRANSIT,
    BALLISTIC_FAST, BALLISTIC_MODELS, BALLISTIC_REFERENCE, _ballistic_scope_reason,
    OD_TRANSIT_KEY, _species_path_density, _temperatures,
    RB_COMPOSITION_NATURAL, SASScheme, _isotope_composition, _probe_power_mw,
    _pump_free_model_id, _pump_free_model_reason, _rate_scope_reason,
)


MODEL_KEY = OD_MODEL_KEY   # SAS parameter: a model set or an explicit model ID
PUMP_GROUP = "Pump"
TWO_PI = 2.0 * math.pi
# User decision U2, 2026-11-20: keep the rate model and document its accuracy.
RATE_MODEL_ACCURACY = (
    "Rate equations drop coherences between optical links. Against the full "
    "optical Bloch model (local absorption at the beam centre and line centre, "
    "25 °C, transit 2π·100 kHz, 1 mm 1/e² radius) the rate model is higher by "
    "at most 0.59 % at 100 µW and 1.62 % up to 2 mW (⁸⁵Rb F=2 lines); ⁸⁷Rb stays "
    "within 0.87 % up to 2 mW. After Gaussian-beam averaging and propagation "
    "through 75 mm (pure isotopes, D2, line centres, OD0 1.1-1.8) the "
    "transmission differs by at most 1.85e-3 and the effective optical depth by "
    "at most 1.21 %; at 100 µW by at most 6.1e-4 and 0.20 %."
)


GAUSSIAN_TRANSIT_ACCURACY = (
    "The rate-equation comparison with the full optical Bloch model used a fixed "
    "γₜ/(2π) of 100 kHz. Its error bounds have not been established independently "
    "across the rates calculated from beam size and kinetic temperature."
)


BALLISTIC_ACCURACY = (
    "Thermal atoms enter a linearly polarized beam at zero magnetic field. "
    "The model integrates magnetic-sublevel populations along each crossing with intrinsic optical "
    "width and propagates the collected power through the cell. A Gaussian profile is assumed at each "
    "depth. At 99 °C and 100 µW, evolving the radial profile changed transmission by about "
    "0.6–0.8 percentage points at the two measured inner wings and 2.85 points at +2.275 GHz. "
    "Stray-field effects, radiation trapping and longitudinal atomic motion are omitted."
)


@dataclass(frozen=True)
class ODModel:
    model_id: str
    description: str
    status: str                       # implemented | scheme_only
    finite_power: bool
    weak_probe_profile: str | None = None
    engine: str = "sas"               # sas | rate | ballistic
    accuracy: str = ""                # documented accuracy, for readouts and exports
    # Measured Rb D2 self-broadening instead of the D1 value on D2 (_v2 IDs,
    # user decision 2026-11-24, checklist rb-d2-self-broadening-default-value).
    per_line_self_broadening: bool = False


MODELS = {model.model_id: model for model in (
    ODModel("weak_probe_legacy", "Legacy 1 m/s velocity grid, zero probe power",
            "implemented", False, LEGACY_WEAK_PROBE),
    ODModel("weak_probe_exact_maxwell_voigt",
            "Exact Maxwell Voigt without a transit width, zero probe power",
            "implemented", False, EXACT_WEAK_PROBE),
    ODModel("finite_power_2level_gaussian_propagation",
            "Current SAS finite-probe path; needs a uniform scan axis", "scheme_only", True),
    ODModel("od_weak_exact_voigt_transit_v1",
            "Exact Maxwell Voigt with the transit HWHM (user decision U1), zero probe power",
            "implemented", False, EXACT_WEAK_PROBE),
    ODModel("od_finite_rb_mf_rate_v1",
            "Magnetic-sublevel rate model for a linearly polarized Rb probe, any probe power",
            "implemented", True, engine="rate", accuracy=RATE_MODEL_ACCURACY),
    ODModel("od_weak_exact_voigt_transit_v2",
            "Exact Maxwell Voigt with the transit HWHM and the measured Rb D2 "
            "self-broadening, zero probe power",
            "implemented", False, EXACT_WEAK_PROBE, per_line_self_broadening=True),
    ODModel("finite_power_2level_gaussian_propagation_v2",
            "Current SAS finite-probe path with the measured Rb D2 self-broadening; "
            "needs a uniform scan axis", "scheme_only", True, per_line_self_broadening=True),
    ODModel("od_finite_rb_mf_rate_v2",
            "Magnetic-sublevel rate model for a linearly polarized Rb probe with the "
            "measured Rb D2 self-broadening, any probe power",
            "implemented", True, engine="rate", accuracy=RATE_MODEL_ACCURACY,
            per_line_self_broadening=True),
    ODModel(BALLISTIC_FAST, "Fast beam-crossing history for pure ⁸⁵Rb D1",
            "implemented", True, engine="ballistic", accuracy=BALLISTIC_ACCURACY, per_line_self_broadening=True),
    ODModel(BALLISTIC_REFERENCE, "Beam-crossing history for pure ⁸⁵Rb D1",
            "implemented", True, engine="ballistic", accuracy=BALLISTIC_ACCURACY, per_line_self_broadening=True),
    ODModel(GAUSSIAN_TRANSIT_MODEL,
            "Rb magnetic-sublevel probe model with Gaussian transit and measured Rb D2 self-broadening",
            "implemented", True, engine="rate", accuracy=GAUSSIAN_TRANSIT_ACCURACY, per_line_self_broadening=True),
    ODModel(GAUSSIAN_TRANSIT_D20_MODEL,
            "Rb magnetic-sublevel probe model with Gaussian transit and historical D2 self-broadening",
            "implemented", True, engine="rate", accuracy=GAUSSIAN_TRANSIT_ACCURACY),
)}


def is_rate_model(model_id):
    """Whether ``model_id`` is a magnetic-sublevel rate model (any self-broadening)."""
    model = MODELS.get(model_id)
    return model is not None and model.engine == "rate"


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

        ``od_model`` holds a model set ("pre_d20", "d20", "d20_d2sb") or an explicit model
        ID; the SAS scheme resolves it, so the evaluator and the plot agree.
        Without the key the pre-D20 choice applies, so older settings replay,
        exactly as in the SAS solve; the defaults merged below must not add it.
        """
        missing_model = MODEL_KEY not in params
        params = {**SASScheme().defaults(), **od_fit_conditions(params)}
        if missing_model:
            params.pop(MODEL_KEY, None)
        if params.get("species") not in species.SPECIES_ORDER:
            raise ODUnsupportedError("Choose a resolved Rb or Cs medium.")
        if params.get("cell_propagation", "Uniform") != "Uniform":
            raise ODUnsupportedError("Choose the Uniform cell for OD fitting.")
        try:
            model_id = _pump_free_model_id(params)
        except ValueError as error:
            raise ODUnsupportedError(str(error)) from None
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
        if self.model_id in GAUSSIAN_TRANSIT_MODELS:
            params[OD_TRANSIT_KEY] = GAUSSIAN_TRANSIT
        for mode, model_id in BALLISTIC_MODELS.items():
            if self.model_id == model_id:
                params[OD_TRANSIT_KEY] = mode
        return params

    def _transit_rate_s(self, isotope):
        if self.model_id in GAUSSIAN_TRANSIT_MODELS:
            kinetic_c = self.temp_c if self.doppler_temp_c is None else self.doppler_temp_c
            return gaussian_transit_rate_s(self.probe_waist_mm * 1e-3, kinetic_c + 273.15, isotope.mass)
        return rate_from_khz_over_2pi(self.transit_khz)

    def transit_rates_s(self):
        """Beam replacement rates in s⁻¹ for the isotopes present, keyed by isotope name."""
        if self.model_id in BALLISTIC_MODELS.values():
            return {}
        components, *_ = _isotope_composition(self.to_params())
        return {isotope.name: self._transit_rate_s(isotope) for isotope, _ in components}

    def transit_metadata(self):
        """Beam conditions and per-isotope rates for a pump-free OD calculation."""
        kinetic_c = self.temp_c if self.doppler_temp_c is None else self.doppler_temp_c
        return dict(mode=self.to_params()[OD_TRANSIT_KEY],
                    kinetic_temperature_k=kinetic_c + 273.15, probe_waist_m=self.probe_waist_mm * 1e-3,
                    rate_s_inverse=self.transit_rates_s())


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
    diagnostics: dict | None = field(default=None, compare=False)


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
    if model.status != "implemented":
        return (f"{model.description}. Use the magnetic-sublevel rate model "
                "(od_finite_rb_mf_rate_v1) for a finite probe on arbitrary frequencies.")
    if model.engine == "ballistic":
        return _ballistic_scope_reason(conditions.to_params())
    if model.engine == "rate":
        return _rate_model_reason(conditions)
    if conditions.probe_power_uw > 0.0:
        return (f"{model.description}. Set the probe power to zero or choose the "
                "magnetic-sublevel rate model.")
    if _pump_free_model_reason(conditions.to_params(), model.model_id):
        return f"{model.description} does not apply to these conditions."
    return None


def evaluate_od(conditions, offsets_hz):
    """Transmission and optical depth at exactly the requested offsets in Hz.

    Offsets are from the contract reference (``frequency_reference``). Any
    order, repeated values and gaps are accepted; each value depends only on
    its own frequency. The result arrays are read-only.
    """
    offsets = np.asarray(offsets_hz, dtype=float)
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("Offsets must be a nonempty, finite 1-D array in Hz.")
    shift = display_reference_offset_hz(conditions)
    return _evaluate(conditions, offsets, (offsets - shift) * TWO_PI)


def evaluate_od_display(conditions, scan_rad_s):
    """The same evaluation on the SAS display axis, angular frequency in rad/s.

    This is the axis of the scheme's ``raw["scan"]`` and of the OD fit's MHz
    input: pure Rb87 is measured from the Rb87 centroid. ``offsets_hz`` in the
    result is converted to the contract reference.
    """
    scan = np.asarray(scan_rad_s, dtype=float)
    if scan.ndim != 1 or not scan.size or not np.isfinite(scan).all():
        raise ValueError("The scan must be a nonempty, finite 1-D array in rad/s.")
    return _evaluate(conditions, scan / TWO_PI + display_reference_offset_hz(conditions), scan)


def _evaluate(conditions, offsets, scan):
    if reason := od_unavailable(conditions):
        raise ODUnsupportedError(reason)
    model = MODELS[conditions.model_id]
    reference = frequency_reference(conditions)
    if model.engine in ("rate", "ballistic"):
        diagnostics = None
        if model.engine == "ballistic":
            from .ballistic_spectrum import evaluate_ballistic_spectrum
            fidelity = "fast" if model.model_id == BALLISTIC_FAST else "reference"
            result = evaluate_ballistic_spectrum(conditions, scan, fidelity=fidelity)
            transmission, depth, od0 = result.transmission, result.optical_depth, result.od0
            diagnostics = result.diagnostics
        else:
            transmission, depth, od0 = _rate_model_evaluate(conditions, scan)
        return ODEvaluation(
            offsets_hz=readonly_float_array(offsets),
            transmission=readonly_float_array(transmission),
            optical_depth=readonly_float_array(depth), od0=readonly_float_array(od0),
            model_id=model.model_id, frequency_reference=reference, conditions=conditions, diagnostics=diagnostics)
    params = conditions.to_params()
    raw = SASScheme()._compute_species(
        params, scan_rad_s=scan, extra_homogeneous_fwhm_hz=conditions.extra_homogeneous_fwhm_hz)
    if raw["probe_model"] != model.model_id:
        raise ODUnsupportedError(f"The SAS solve applied {raw['probe_model']}, not {model.model_id}.")
    alpha = raw["alpha_unit"] * conditions.line_strength
    length = conditions.cell_mm * 1e-3
    depth = readonly_float_array(observables.optical_depth(alpha, length))
    return ODEvaluation(
        offsets_hz=readonly_float_array(offsets),
        transmission=readonly_float_array(observables.transmission(alpha, length)),
        optical_depth=depth, od0=depth, model_id=model.model_id,
        frequency_reference=reference, conditions=conditions,
        weak_probe_profile=raw.get("weak_probe_profile"),
    )


def with_extra_width(conditions, extra_homogeneous_fwhm_hz):
    return replace(conditions, extra_homogeneous_fwhm_hz=float(extra_homogeneous_fwhm_hz))


# ---------------------------------------------------------------------------
# Magnetic-sublevel rate model, ported from AutoOD 0.2.2 (tag baseline-0.2.2-src,
# autood/engine.py: _full_rb_scan, _population_response, calculate).
# Step 1 keeps AutoOD's algorithm: the nonlinear table is built on the
# requested uniform axis, so a value depends slightly on the axis sampling.
# It is the parity anchor for the sample-invariant evaluator.
# ---------------------------------------------------------------------------

def _rate_model_reason(conditions):
    return _rate_scope_reason(conditions.to_params(), geometry=conditions.model_id in GAUSSIAN_TRANSIT_MODELS)


def _rate_model_params(conditions):
    if reason := _rate_model_reason(conditions):
        raise ODUnsupportedError(reason)
    return conditions.to_params()


def rate_model_axis(conditions, points=2001):
    """AutoOD's native axis for these conditions, angular frequency in rad/s.

    Measured on the SAS display axis (the Rb85 centroid for "Rb (natural)")
    and spanning both Rb manifolds even at pure-isotope endpoints. Evaluating
    ``od_finite_rb_mf_rate_v1`` here with ``evaluate_od_display`` repeats
    AutoOD 0.2.2's calculation, so its own table nodes are sampled exactly.
    """
    return _rate_model_scan(conditions, points)


def _rate_model_scan(conditions, points=2001):
    """AutoOD's axis in rad/s: both Rb manifolds stay visible at pure endpoints."""
    params = _rate_model_params(conditions)
    components, reference, *_ = _isotope_composition(params)
    fractions = {isotope.name: fraction for isotope, fraction in components}
    line = params["line"]
    cold_k, temperature_k, _ = _temperatures(params)
    reference_hz = reference.line(line)[1]
    peak_intensity = probe_saturation.gaussian_peak_intensity(
        params["probe_power_uw"] * 1e-3, params["probe_waist_mm"])
    per_line = MODELS[conditions.model_id].per_line_self_broadening
    centers = []
    doppler_fwhm_max = 0.0
    broadened_fwhm_max = 0.0
    for isotope in (species.RB85, species.RB87):
        manifold = species.build_manifold(isotope, line, transit_rate=conditions._transit_rate_s(isotope))
        offset = 2 * np.pi * (manifold.nu0 - reference_hz)
        centers.append(manifold.omega + offset)
        sigma_v = np.sqrt(constants.KB * temperature_k / isotope.mass)
        doppler_fwhm_max = max(
            doppler_fwhm_max, np.sqrt(8 * np.log(2)) * manifold.k_vec * sigma_v)
        density = _species_path_density(isotope, cold_k, temperature_k, fractions.get(isotope.name, 0.0))
        gamma_effective = manifold.gamma + species.self_broadening(
            isotope, density, line, per_line=per_line)
        saturation_intensity = probe_saturation.closed_two_level_saturation_intensity(
            manifold.nu0, manifold.gamma)
        saturation = probe_saturation.peak_saturation_parameter(
            peak_intensity, saturation_intensity, manifold.gamma, gamma_effective)
        broadened_fwhm_max = max(
            broadened_fwhm_max, gamma_effective * np.sqrt(1.0 + saturation))
    centers = np.concatenate(centers)
    margin = max(3.5 * doppler_fwhm_max, 6.0 * broadened_fwhm_max)
    return np.linspace(centers.min() - margin, centers.max() + margin, int(points))


def _rate_model_optics(conditions):
    """Per-isotope manifold, density, offset, Doppler σ, optical HWHM and replacement rate."""
    params = _rate_model_params(conditions)
    isotope_fractions, reference, *_ = _isotope_composition(params)
    line = params["line"]
    cold_k, temperature, _ = _temperatures(params)
    extra_hwhm = np.pi * conditions.extra_homogeneous_fwhm_hz
    peak = probe_saturation.gaussian_peak_intensity(
        params["probe_power_uw"] * 1e-3, params["probe_waist_mm"])
    per_line = MODELS[conditions.model_id].per_line_self_broadening
    optics = []
    for isotope, fraction in isotope_fractions:
        density = _species_path_density(isotope, cold_k, temperature, fraction)
        transit = conditions._transit_rate_s(isotope)
        manifold = species.build_manifold(isotope, line)
        offset = 2 * np.pi * (manifold.nu0 - reference.line(line)[1])
        sigma = manifold.k_vec * np.sqrt(constants.KB * temperature / isotope.mass)
        gamma2 = 0.5 * (manifold.gamma + species.self_broadening(
            isotope, density, line, per_line=per_line)) + transit
        if extra_hwhm:
            gamma2 += extra_hwhm
        optics.append((manifold, density, offset, sigma, gamma2, transit))
    return optics, temperature, peak


def _rate_model_alpha0(optics, temperature, scan):
    """Weak absorption in m^-1 with the transit HWHM; pointwise in frequency."""
    alpha0 = np.zeros_like(scan)
    for manifold, density, offset, _sigma, gamma2, _transit in optics:
        weak, _ = linear_hyperfine.species_complex_voigt(
            scan, manifold, density, temperature, 2 * gamma2, offset=offset)
        alpha0 += weak
    return alpha0


def _rate_model_ratio(optics, peak, scan, alpha0, *, log_step=0.25,
                      propagation_step=0.02, frequency_stride=4):
    """log(alpha(I, nu)/alpha0) on every frequency_stride-th sample of ``scan``."""
    components = []
    pumping_scale = 0.0
    for manifold, density, offset, sigma, gamma2, transit in optics:
        probe = ProjectedProbe(manifold, gamma2, transit, resolution=MAGNETIC_PI)
        components.append((probe, density, offset, sigma))
        pumping_scale = max(pumping_scale, float(probe.peak_saturation(peak))
                            * (manifold.gamma + transit) / transit)
    # The tail begins well below the optical-pumping / transit crossover.
    xmax = max(12.0, np.log1p(pumping_scale) + 14)
    log_intensity = np.linspace(0, xmax, 1 + math.ceil(xmax / log_step))
    propagation_grid = np.linspace(0, xmax, 1 + math.ceil(xmax / propagation_step))
    indices = np.unique(np.r_[np.arange(0, scan.size, frequency_stride), scan.size - 1])
    coarse_scan = scan[indices]
    intensities = peak * np.exp(-log_intensity)
    coarse_alpha = np.zeros((intensities.size, indices.size))
    for probe, density, offset, sigma in components:
        coarse_alpha += density * probe.absorption_cross_section(intensities, coarse_scan - offset, sigma)
    if not np.all(np.isfinite(coarse_alpha)) or np.any(coarse_alpha <= 0):
        raise ArithmeticError("Nonpositive or nonfinite absorption in the probe population solve.")
    ratio = np.log(coarse_alpha / alpha0[indices])
    if np.max(np.abs(ratio[-1])) > 3e-5:
        raise ArithmeticError("The nonlinear propagation table did not reach its analytical weak tail.")
    # The final boundary matches the analytical continuation used by GABES.
    ratio[-1] = 0.0
    return log_intensity, propagation_grid, coarse_scan, ratio


def _rate_model_table(log_intensity, propagation_grid, frequency_spline, scan, alpha0, inside):
    """alpha(x, nu) on the propagation lattice; unsaturated outside the table span."""
    from scipy.interpolate import CubicSpline

    ratio = frequency_spline(scan)
    if not np.all(inside):
        ratio[:, ~inside] = 0.0
    return alpha0 * np.exp(CubicSpline(log_intensity, ratio, axis=0)(propagation_grid))


def _rate_model_tables(conditions, scan, *, log_step=0.25, propagation_step=0.02,
                       frequency_stride=4):
    """Weak absorption and, when driven, a local alpha(I, nu) table in m^-1.

    Step-1 algorithm (AutoOD 0.2.2): the table is built on ``scan`` itself.
    Unit line strength. The optical HWHM is (Gamma + Gamma_self)/2 + gamma_t,
    plus half of any extra homogeneous FWHM.
    """
    from scipy.interpolate import CubicSpline

    optics, temperature, peak = _rate_model_optics(conditions)
    alpha0 = _rate_model_alpha0(optics, temperature, scan)
    if peak == 0:
        return alpha0, None, None
    log_intensity, propagation_grid, coarse_scan, ratio = _rate_model_ratio(
        optics, peak, scan, alpha0, log_step=log_step,
        propagation_step=propagation_step, frequency_stride=frequency_stride)
    spline = CubicSpline(coarse_scan, ratio, axis=1)
    table = _rate_model_table(log_intensity, propagation_grid, spline, scan, alpha0,
                              np.ones(scan.size, dtype=bool))
    return alpha0, propagation_grid, table


def _propagate(conditions, alpha0, grid, table):
    strength = conditions.line_strength
    if strength != 1.0:
        alpha0 = alpha0 * strength
        table = None if table is None else table * strength
    length = conditions.cell_mm * 1e-3
    od0 = alpha0 * length
    if grid is None:
        return np.exp(-od0), od0, od0
    with core.blas_single_thread():
        transmission, effective_alpha = probe_saturation.propagate_gaussian_spectrum(
            grid, table, alpha0, length, radial_order=96)
    return transmission, effective_alpha * length, od0


def _rate_model_on_axis(conditions, scan_rad_s):
    """(T, OD_eff, OD0) of the step-1 rate model on an increasing uniform axis in rad/s.

    The axis is relative to the medium's SAS reference (the Rb85 centroid for
    "Rb (natural)"). ``line_strength`` scales absorption at propagation.
    """
    scan = np.asarray(scan_rad_s, dtype=float)
    with core.blas_single_thread():
        alpha0, grid, table = _rate_model_tables(conditions, scan)
    return _propagate(conditions, alpha0, grid, table)


def _table_key(conditions):
    """Conditions that change the optical table; length and strength enter later."""
    return replace(conditions, cell_mm=1.0, line_strength=1.0,
                   transit_khz=0.0 if conditions.model_id in GAUSSIAN_TRANSIT_MODELS else conditions.transit_khz)


@lru_cache(maxsize=8)
def _rate_model_spline(key):
    """Nonlinear ratio table on the model's own axis, fixed by the conditions.

    The axis is AutoOD's (_rate_model_scan, 2,001 points), so evaluating the
    model there repeats the step-1 computation exactly.
    """
    from scipy.interpolate import CubicSpline

    optics, temperature, peak = _rate_model_optics(key)
    if peak == 0:
        return None
    grid_scan = _rate_model_scan(key)
    with core.blas_single_thread():
        alpha0 = _rate_model_alpha0(optics, temperature, grid_scan)
        log_intensity, propagation_grid, coarse_scan, ratio = _rate_model_ratio(
            optics, peak, grid_scan, alpha0)
    spline = CubicSpline(coarse_scan, ratio, axis=1)
    return log_intensity, propagation_grid, spline, float(coarse_scan[0]), float(coarse_scan[-1])


def rate_model_display_tables(conditions, scan):
    """(alpha0, Re chi, x grid, alpha(x, nu)) at unit line strength on a display scan.

    The SAS solve propagates these exactly as ``evaluate_od`` does (radial
    order 96), so the theory plot and the evaluator agree point for point.
    The dispersion is the weak-probe transit Voigt's.
    """
    if reason := _rate_model_reason(conditions):
        raise ODUnsupportedError(reason)
    scan = np.asarray(scan, dtype=float)
    optics, temperature, _peak = _rate_model_optics(conditions)
    chi_real = np.zeros_like(scan)
    with core.blas_single_thread():
        alpha0 = _rate_model_alpha0(optics, temperature, scan)
        for manifold, density, offset, _sigma, gamma2, _transit in optics:
            chi_real += linear_hyperfine.species_complex_voigt(
                scan, manifold, density, temperature, 2 * gamma2, offset=offset)[1]
    cached = _rate_model_spline(_table_key(conditions))
    if cached is None:
        return alpha0, chi_real, None, None
    log_intensity, propagation_grid, spline, lo, hi = cached
    tolerance = 1e-9 * (hi - lo)
    inside = (scan >= lo - tolerance) & (scan <= hi + tolerance)
    with core.blas_single_thread():
        table = _rate_model_table(log_intensity, propagation_grid, spline, scan, alpha0, inside)
    return alpha0, chi_real, propagation_grid, table


def _rate_model_evaluate(conditions, scan):
    """Sample-invariant rate model at arbitrary scan values in rad/s.

    The ratio table comes from the model's own axis; weak absorption and
    propagation are evaluated at each requested frequency. Outside the
    table span the probe is treated as unsaturated.
    """
    optics, temperature, _peak = _rate_model_optics(conditions)
    with core.blas_single_thread():
        alpha0 = _rate_model_alpha0(optics, temperature, scan)
    cached = _rate_model_spline(_table_key(conditions))
    if cached is None:
        return _propagate(conditions, alpha0, None, None)
    log_intensity, propagation_grid, spline, lo, hi = cached
    # Offsets converted from Hz can round a few ulp past the table ends.
    tolerance = 1e-9 * (hi - lo)
    inside = (scan >= lo - tolerance) & (scan <= hi + tolerance)
    with core.blas_single_thread():
        table = _rate_model_table(log_intensity, propagation_grid, spline, scan, alpha0, inside)
    return _propagate(conditions, alpha0, propagation_grid, table)
