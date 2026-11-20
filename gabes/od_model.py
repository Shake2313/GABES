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

from . import constants, core, linear_hyperfine, observables, probe_saturation, species
from ._arrays import readonly_float_array
from .optical_frequency import reference_offset_hz
from .optical_pumping import MAGNETIC_PI, ProjectedProbe
from .schemes.sas import (
    EXACT_WEAK_PROBE, LEGACY_WEAK_PROBE, RB_COMPOSITION_CUSTOM, RB_COMPOSITION_NATURAL,
    SASScheme, _isotope_composition, _probe_power_mw, _weak_probe_model_id, _weak_probe_profile,
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


# ---------------------------------------------------------------------------
# Magnetic-sublevel rate model, ported from AutoOD 0.2.2 (tag baseline-0.2.2-src,
# autood/engine.py: _full_rb_scan, _population_response, calculate).
# Step 1 keeps AutoOD's algorithm: the nonlinear table is built on the
# requested uniform axis, so a value depends slightly on the axis sampling.
# It is the parity anchor for the sample-invariant evaluator.
# ---------------------------------------------------------------------------

def _rate_model_reason(conditions):
    if conditions.species not in ("Rb (natural)", species.RB85.label, species.RB87.label):
        return "The magnetic-sublevel rate model covers rubidium only."
    if conditions.doppler_temp_c is not None:
        return "The magnetic-sublevel rate model needs the Doppler temperature tied to the cell."
    if (conditions.ne_pressure_torr != 0.0 or conditions.paraffin_coated
            or conditions.optical_collision_model not in (None, SASScheme().defaults()["optical_collision_model"])):
        return "The magnetic-sublevel rate model covers uncoated cells without buffer gas."
    return None


def _rate_model_params(conditions):
    if reason := _rate_model_reason(conditions):
        raise ODUnsupportedError(reason)
    return conditions.to_params()


def _rate_model_scan(conditions, points=2001):
    """AutoOD's axis in rad/s: both Rb manifolds stay visible at pure endpoints."""
    params = _rate_model_params(conditions)
    components, reference, *_ = _isotope_composition(params)
    fractions = {isotope.name: fraction for isotope, fraction in components}
    line = params["line"]
    temperature_k = params["temp_c"] + 273.15
    reference_hz = reference.line(line)[1]
    transit_rate = 2 * np.pi * params["transit_khz"] * 1e3
    peak_intensity = probe_saturation.gaussian_peak_intensity(
        params["probe_power_uw"] * 1e-3, params["probe_waist_mm"])
    centers = []
    doppler_fwhm_max = 0.0
    broadened_fwhm_max = 0.0
    for isotope in (species.RB85, species.RB87):
        manifold = species.build_manifold(isotope, line, transit_rate=transit_rate)
        offset = 2 * np.pi * (manifold.nu0 - reference_hz)
        centers.append(manifold.omega + offset)
        sigma_v = np.sqrt(constants.KB * temperature_k / isotope.mass)
        doppler_fwhm_max = max(
            doppler_fwhm_max, np.sqrt(8 * np.log(2)) * manifold.k_vec * sigma_v)
        density = (species.number_density(isotope, temperature_k)
                   * fractions.get(isotope.name, 0.0))
        gamma_effective = manifold.gamma + species.self_broadening(isotope, density)
        saturation_intensity = probe_saturation.closed_two_level_saturation_intensity(
            manifold.nu0, manifold.gamma)
        saturation = probe_saturation.peak_saturation_parameter(
            peak_intensity, saturation_intensity, manifold.gamma, gamma_effective)
        broadened_fwhm_max = max(
            broadened_fwhm_max, gamma_effective * np.sqrt(1.0 + saturation))
    centers = np.concatenate(centers)
    margin = max(3.5 * doppler_fwhm_max, 6.0 * broadened_fwhm_max)
    return np.linspace(centers.min() - margin, centers.max() + margin, int(points))


def _rate_model_tables(conditions, scan, *, log_step=0.25, propagation_step=0.02,
                       frequency_stride=4):
    """Weak absorption and, when driven, a local alpha(I, nu) table in m^-1.

    Unit line strength. The optical HWHM is (Gamma + Gamma_self)/2 + gamma_t,
    plus half of any extra homogeneous FWHM. Known Voigt wings are factored
    out before spline interpolation; the propagation lattice stays on the
    requested frequency samples.
    """
    from scipy.interpolate import CubicSpline

    params = _rate_model_params(conditions)
    isotope_fractions, reference, *_ = _isotope_composition(params)
    line = params["line"]
    temperature = params["temp_c"] + 273.15
    transit = 2 * np.pi * params["transit_khz"] * 1e3
    extra_hwhm = np.pi * conditions.extra_homogeneous_fwhm_hz
    peak = probe_saturation.gaussian_peak_intensity(
        params["probe_power_uw"] * 1e-3, params["probe_waist_mm"])
    alpha0 = np.zeros_like(scan)
    components = []
    pumping_scale = 0.0
    for isotope, fraction in isotope_fractions:
        density = fraction * species.number_density(isotope, temperature)
        manifold = species.build_manifold(isotope, line)
        offset = 2 * np.pi * (manifold.nu0 - reference.line(line)[1])
        sigma = manifold.k_vec * np.sqrt(constants.KB * temperature / isotope.mass)
        gamma2 = 0.5 * (manifold.gamma + species.self_broadening(isotope, density)) + transit
        if extra_hwhm:
            gamma2 += extra_hwhm
        weak, _ = linear_hyperfine.species_complex_voigt(
            scan, manifold, density, temperature, 2 * gamma2, offset=offset)
        alpha0 += weak
        if peak > 0:
            probe = ProjectedProbe(manifold, gamma2, transit, resolution=MAGNETIC_PI)
            components.append((probe, density, offset, sigma))
            pumping_scale = max(pumping_scale, float(probe.peak_saturation(peak))
                                * (manifold.gamma + transit) / transit)
    if peak == 0:
        return alpha0, None, None

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
    ratio = CubicSpline(coarse_scan, ratio, axis=1)(scan)
    alpha_table = alpha0 * np.exp(CubicSpline(log_intensity, ratio, axis=0)(propagation_grid))
    return alpha0, propagation_grid, alpha_table


def _rate_model_on_axis(conditions, scan_rad_s):
    """(T, OD_eff, OD0) of the rate model on an increasing uniform axis in rad/s.

    The axis is relative to the medium's SAS reference (the Rb85 centroid for
    "Rb (natural)"). ``line_strength`` scales absorption at propagation.
    """
    scan = np.asarray(scan_rad_s, dtype=float)
    with core.blas_single_thread():
        alpha0, grid, table = _rate_model_tables(conditions, scan)
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
