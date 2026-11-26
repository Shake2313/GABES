"""
Cluster A — absorption spectroscopy (OD / SAS), one scheme.

A weak probe measures the vapor's absorption. A counter-propagating *pump* of
power P saturates and optically pumps the velocity class it is resonant with;
the probe then samples those prepared populations:

    pump  atom-frame detuning = Δ + k·v
    probe atom-frame detuning = Δ − k·v        (Δ = laser − line centre, the scan)

  * **Pump off (P = 0) → OD.** The probe sees the linear, Doppler-broadened
    multi-line absorption — the thermally normalized 85Rb D1 hyperfine spectrum
    (sublevel populations counted once) at pump = 0, generalised to any isotope/line.
  * **Pump on → SAS.** Velocity-selective hole burning + hyperfine optical
    pumping give Doppler-free Lamb dips and (enhanced / inverted) crossovers.

Model
-----
For a chosen isotope/line `gabes.species` builds the full {Fg}×{Fe} hyperfine
manifold (CG-branched decay + transit-time relaxation). The absorption is

    α(δ) = Σ_components Σ_(Fg→Fe)  A_(Fg,Fe) · ĝ_(Fg,Fe)(δ)

  A_t = ∫α_t dδ = ls·π·k·C_F²·|d|²·N/(ε₀ℏ)/(2(2I+1))   (AutoOD absolute area)
  ĝ_t(δ) = Σ_v f(v)·[(ρ_Fg − ρ_Fe)(Δ+k·v)/p_Fg]·L̂((δ − k·v) − ω_t)   (unit area at P=0)

ρ comes from one OBE steady state per velocity class; the pump Hamiltonian is
scan-independent (the scan enters only via Δ_eff = Δ + k·v), so it is solved once
on a fine Δ_eff grid and interpolated. At P=0 the bracket is 1, ĝ_t is a unit
Voigt, and α reduces to the thermally normalized OD spectrum (∫α_t = A_t).

A generic Γ-unit hole-burning toy (one ground + one/two excited states) remains
as an internal backwards-compatibility/validation path, but is not UI-exposed.

References:  Smith & Hughes, Am. J. Phys. 72, 631 (2004) (hyperfine pumping);
Preston, Am. J. Phys. 64, 1432 (1996); Steck alkali D-line data; the lab AutoOD
calculator (pump-off 85Rb D1 absolute scale).
"""
from functools import partial

import numpy as np

from .. import (atoms, collisions, constants, doppler, kernels, linear_hyperfine, observables,
               probe_saturation, species, vapor_cell)
from ..constants import GAMMA, K_VEC
from .. import core
from ..lineshape import (MIN_SUBDOPPLER_SAMPLES_PER_FWHM, subdoppler_feature,
                         subdoppler_refinement)
from ..optical_pumping import steady_populations as _pump_pops
from ..transit import rate_from_khz_over_2pi
from ..report import derived_table
from ..rydberg_experiment import effective_atom_number
from .base import ExtraView, ParamSpec, Scheme
from . import sas_axial

PROBE_RABI = 1e-3                       # weak probe, in units of Γ
PROBE_PASSIVITY_INTERIOR = 1e-8         # avoid an artificial exact-zero fixed point
PROBE_REFERENCE_RB_D2_ISOTOPES = frozenset(("Rb85", "Rb87"))
RB_COMPOSITION_NATURAL = "Natural"
RB_COMPOSITION_CUSTOM = "Custom"
AUTO_WEAK_PROBE = "Automatic"
LEGACY_WEAK_PROBE = "Legacy velocity grid"
EXACT_WEAK_PROBE = "Exact Maxwell Voigt"
PUMPED_MAXWELL = "Exact Voigt + SAS correction"
_RB_MEDIA = ("Rb (natural)", species.RB85.label, species.RB87.label)
PROJECTED_PROBE_VIEW = "Finite-probe pumping reference"
# OD model selection (decision D20). A model set chooses per regime; an explicit
# pump-free model ID fixes the pump-off solve and selects its set for the rest.
OD_MODEL_KEY = "od_model"
OD_MODELS_PRE_D20 = "pre_d20"
OD_MODELS_D20 = "d20"
# d20 with the measured Rb D2 self-broadening coefficient (user decision
# 2026-11-24, checklist rb-d2-self-broadening-default-value). Its model IDs
# carry a _v2 suffix; every older set and ID keeps BETA_SELF on D2 and replays.
OD_MODELS_D2SB = "d20_d2sb"
# User decision 2026-11-26: all Rb isotopes act as collision partners.
OD_MODELS_NTOTAL = "d20_ntotal"
OD_MODEL_SETS = (OD_MODELS_PRE_D20, OD_MODELS_D20, OD_MODELS_D2SB, OD_MODELS_NTOTAL)
OD_MODELS_DEFAULT = OD_MODELS_NTOTAL  # ParamSpec default; also an explicit None
LEGACY_MODEL = "weak_probe_legacy"
EXACT_MODEL = "weak_probe_exact_maxwell_voigt"
TRANSIT_WEAK_MODEL = "od_weak_exact_voigt_transit_v1"
TWO_LEVEL_MODEL = "finite_power_2level_gaussian_propagation"
RATE_MODEL = "od_finite_rb_mf_rate_v1"
PUMPED_MODEL = "weak_probe_maxwell_pumped_correction"
PUMPED_TRANSIT_MODEL = "sas_pumped_exact_voigt_transit_v1"
TRANSIT_WEAK_MODEL_V2 = "od_weak_exact_voigt_transit_v2"
TWO_LEVEL_MODEL_V2 = "finite_power_2level_gaussian_propagation_v2"
RATE_MODEL_V2 = "od_finite_rb_mf_rate_v2"
PUMPED_TRANSIT_MODEL_V2 = "sas_pumped_exact_voigt_transit_v2"
TRANSIT_WEAK_MODEL_V3 = "od_weak_exact_voigt_transit_v3"
TWO_LEVEL_MODEL_V3 = "finite_power_2level_gaussian_propagation_v3"
RATE_MODEL_V3 = "od_finite_rb_mf_rate_v3"
PUMPED_TRANSIT_MODEL_V3 = "sas_pumped_exact_voigt_transit_v3"
GRID_NTOTAL_MODEL = "weak_probe_grid_ntotal_v1"
TRANSIT_WEAK_MODELS = (TRANSIT_WEAK_MODEL, TRANSIT_WEAK_MODEL_V2, TRANSIT_WEAK_MODEL_V3)
TWO_LEVEL_MODELS = (TWO_LEVEL_MODEL, TWO_LEVEL_MODEL_V2, TWO_LEVEL_MODEL_V3)
GAUSSIAN_TRANSIT_MODEL = "od_finite_rb_mf_gaussian_transit_v1"
GAUSSIAN_TRANSIT_D20_MODEL = "od_finite_rb_mf_gaussian_transit_d20_v1"
GAUSSIAN_TRANSIT_NTOTAL_MODEL = "od_finite_rb_mf_gaussian_transit_v2"
GAUSSIAN_TRANSIT_MODELS = (GAUSSIAN_TRANSIT_MODEL, GAUSSIAN_TRANSIT_D20_MODEL, GAUSSIAN_TRANSIT_NTOTAL_MODEL)
# Versions of the same OD/SAS solvers that use total elemental Rb perturber density.
_NTOTAL_MODEL_IDS = {
    TRANSIT_WEAK_MODEL_V2: TRANSIT_WEAK_MODEL_V3,
    TWO_LEVEL_MODEL_V2: TWO_LEVEL_MODEL_V3,
    RATE_MODEL_V2: RATE_MODEL_V3,
    GAUSSIAN_TRANSIT_MODEL: GAUSSIAN_TRANSIT_NTOTAL_MODEL,
    LEGACY_MODEL: GRID_NTOTAL_MODEL,
}
BALLISTIC_REFERENCE = "od_rb85_d1_ballistic_history_v1"
BALLISTIC_TRANSIT = "ballistic_history"
BALLISTIC_FAST_V1 = "od_rb85_d1_ballistic_history_fast_v1"
BALLISTIC_FAST = "od_rb85_d1_ballistic_history_fast_v2"
BALLISTIC_FAST_TRANSIT = "ballistic_history_fast"
FIELD_HISTORY_REFERENCE = "od_rb_field_history_v1"
FIELD_HISTORY_TRANSIT = "ballistic_field_history"
FIELD_HISTORY_KEYS = ("od_field_x_ut", "od_field_y_ut", "od_field_z_ut",
                      "od_probe_azimuth_deg", "od_probe_ellipticity_deg")
BALLISTIC_MODELS = {BALLISTIC_TRANSIT: BALLISTIC_REFERENCE, BALLISTIC_FAST_TRANSIT: BALLISTIC_FAST,
                    FIELD_HISTORY_TRANSIT: FIELD_HISTORY_REFERENCE}
# An explicit prior Fast ID replays the original trajectory grid.
BALLISTIC_FIDELITIES = {BALLISTIC_REFERENCE: "reference", BALLISTIC_FAST_V1: "fast_v1", BALLISTIC_FAST: "fast"}
RB_HISTORY_REFERENCE = "od_rb_ballistic_history_v1"
RB_HISTORY_FAST = "od_rb_ballistic_history_fast_v1"
RB_HISTORY_MODELS = (RB_HISTORY_REFERENCE, RB_HISTORY_FAST)
RB_MIXTURE_HISTORY_MODELS = (*RB_HISTORY_MODELS, FIELD_HISTORY_REFERENCE)
BALLISTIC_FIDELITIES.update({RB_HISTORY_REFERENCE: "reference", RB_HISTORY_FAST: "fast",
                             FIELD_HISTORY_REFERENCE: "reference"})
OD_TRANSIT_KEY = "od_transit_model"
GAUSSIAN_TRANSIT = "gaussian_fwhm"
RATE_MODELS = (RATE_MODEL, RATE_MODEL_V2, RATE_MODEL_V3, *GAUSSIAN_TRANSIT_MODELS)
_EXPLICIT_OD_MODELS = {
    LEGACY_MODEL: OD_MODELS_PRE_D20, EXACT_MODEL: OD_MODELS_PRE_D20,
    TWO_LEVEL_MODEL: OD_MODELS_PRE_D20,
    TRANSIT_WEAK_MODEL: OD_MODELS_D20, RATE_MODEL: OD_MODELS_D20,
    TRANSIT_WEAK_MODEL_V2: OD_MODELS_D2SB, TWO_LEVEL_MODEL_V2: OD_MODELS_D2SB,
    RATE_MODEL_V2: OD_MODELS_D2SB, GAUSSIAN_TRANSIT_MODEL: OD_MODELS_D2SB,
    GAUSSIAN_TRANSIT_D20_MODEL: OD_MODELS_D20,
    **dict.fromkeys(_NTOTAL_MODEL_IDS.values(), OD_MODELS_NTOTAL),
    **dict.fromkeys(BALLISTIC_FIDELITIES, OD_MODELS_D2SB),
    **dict.fromkeys(RB_MIXTURE_HISTORY_MODELS, OD_MODELS_NTOTAL),
}
GEOMETRY_MODELS = (*GAUSSIAN_TRANSIT_MODELS, *BALLISTIC_FIDELITIES)
OD_MODEL_CHOICES = (*OD_MODEL_SETS, *_EXPLICIT_OD_MODELS)
# Atom / Isotope rows of the species knob (rendered as two selectors).
ATOM_MENU = (
    ("Rb", "Natural", "Rb (natural)", {"rb_composition": RB_COMPOSITION_NATURAL}),
    ("Rb", "85", "⁸⁵Rb", {"rb_composition": RB_COMPOSITION_NATURAL}),
    ("Rb", "87", "⁸⁷Rb", {"rb_composition": RB_COMPOSITION_NATURAL}),
    ("Rb", "Custom", "Rb (natural)", {"rb_composition": RB_COMPOSITION_CUSTOM}),
    ("Cs", "133", "¹³³Cs", {}),
)
GAMMA_MHZ = GAMMA / (2 * np.pi) / 1e6
GENERIC = "Generic (Γ units)"
PARAFFIN_REFERENCE_T1_S = 25.1e-3   # 87Rb population T1 at 300 K (Bandi et al.)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy ≥2.0 rename


def _temperatures(params):
    """(cold-spot T, Doppler T) in kelvin, plus whether the two are tied.

    ``temp_c`` is the **cold-spot** (reservoir) temperature: it fixes the
    saturated vapor pressure. The **Doppler-width** temperature is that of the
    atoms crossing the beam: it sets the thermal velocity distribution (Doppler
    width and velocity classes) and, at the cold-spot pressure, the ideal-gas
    density n = n_sat(T_cold)·T_cold/T_Doppler that sets the absorption scale
    and the self-broadening (vapor_cell.cold_spot_limited_density). It follows
    the cell temperature unless ``constrain_doppler_temp`` is released.
    Buffer-gas broadening is pressure-only here (no temperature enters), and the
    transit rate stays a user knob; see `collisional-coefficient-provenance-and-
    pressure-shift` in docs/checklist.json.

    A params dict from before these knobs existed stays tied, so legacy calls
    reproduce their previous spectra exactly.
    """
    density_c = float(params["temp_c"])
    tied = bool(params.get("constrain_doppler_temp", True))
    doppler_c = density_c if tied else float(
        params.get("doppler_temp_c", density_c))
    return density_c + 273.15, doppler_c + 273.15, tied


def _probe_power_mw(params):
    """Return probe power in mW, accepting the retired mW input as an alias.

    The UI uses µW because OD probes normally live well below one milliwatt.
    Parameter dictionaries saved before that UI migration contain only
    ``probe_power_mw``; that explicit legacy key wins if both are supplied, so
    old callers that start from new defaults still behave as requested without
    carrying two independently editable UI controls.
    """
    if "probe_power_mw" in params:
        return float(params["probe_power_mw"])
    return float(params.get("probe_power_uw", 0.0)) * 1e-3


def _species_path_density(isotope, cold_k, path_k, fraction):
    return vapor_cell.cold_spot_limited_density(
        species.number_density(isotope, cold_k), cold_k, path_k) * fraction


def _species_path_densities(isotope, cold_k, path_k, fraction, *, total_rb=False):
    """Absorber and self-broadening perturber densities in the optical path (m^-3)."""
    elemental = _species_path_density(isotope, cold_k, path_k, 1.0)
    absorber = elemental * fraction
    # The shared plotting axis includes an absent Rb isotope at enriched endpoints.
    # Leave that invisible component at zero density, including its scan padding.
    partner = elemental if total_rb and fraction > 0 and isotope in (species.RB85, species.RB87) else absorber
    return absorber, partner


def _pump_is_driven(power_mw):
    # Match the mW-to-watts conversion before the Rabi calculation.
    return float(power_mw) * 1e-3 > 0.0


def _weak_probe_profile(params):
    """Resolve the weak-probe backend and report explicit Voigt fallbacks."""
    requested = params.get("weak_probe_profile", AUTO_WEAK_PROBE)
    if requested == LEGACY_WEAK_PROBE:
        return None
    if requested not in (AUTO_WEAK_PROBE, EXACT_WEAK_PROBE):
        raise ValueError("Choose Automatic, Legacy velocity grid or Exact Maxwell Voigt.")
    eligible = _exact_voigt_eligible(params)
    power = float(params["pump_power_mw"])
    driven = _pump_is_driven(power)
    if eligible and power >= 0.0 and not driven:
        return dict(requested=requested, applied=EXACT_WEAK_PROBE, status="active",
                    scope="Numerical reference for uniform weak-probe Rb D1/D2.",
                    velocity_measure="Untruncated 1D Maxwell distribution")
    # A bare old parameter dictionary has no explicit profile choice. Its
    # zero-pump default was Exact; its pumped solver stayed on the legacy grid.
    if eligible and driven and params.get("weak_probe_profile") == AUTO_WEAK_PROBE:
        return dict(requested=requested, applied=PUMPED_MAXWELL, status="active",
                    scope="Exact Maxwell Voigt background with a population correction "
                          "integrated on a 1 m/s grid out to 5 thermal velocity standard deviations.",
                    velocity_measure="Untruncated 1D Maxwell background; pumped correction to 5σ")
    if requested == AUTO_WEAK_PROBE:
        return None
    return dict(
        requested=requested, applied=LEGACY_WEAK_PROBE, status="inactive",
        scope="Select uniform uncoated Rb D1/D2 with pump, probe and buffer "
              "pressure at zero and Legacy broadening to use Exact Maxwell Voigt.")


def _exact_voigt_eligible(params):
    """Uniform, uncoated, buffer-free Rb D1/D2 at zero probe power."""
    return (
        params.get("species", "Rb (natural)") in _RB_MEDIA
        and params["line"] in ("D1", "D2")
        and _probe_power_mw(params) == 0.0
        and params.get("cell_propagation", "Uniform") == "Uniform"
        and not params.get("paraffin_coated", False)
        and float(params.get("ne_pressure_torr", 0.0)) == 0.0
        and not _measured_collision_mode(params)
    )


class ODModelSettingError(ValueError):
    """An invalid ``od_model`` setting; shown to the user instead of a fallback."""


def _od_model_setting(params, *, driven=None):
    """(model set, explicit pump-free model or None).

    A missing key is pre-D20, so parameter dictionaries written before the
    setting existed replay unchanged. ``None`` is an unset session value and
    takes the current default, like a fresh session. A driven pump uses the SAS
    transit rate. Callers may resolve the pump-off preference independently.
    """
    if driven is None:
        driven = _pump_is_driven(params.get("pump_power_mw", 0.0))
    value = params.get(OD_MODEL_KEY, OD_MODELS_PRE_D20)
    if value is None:
        value = OD_MODELS_DEFAULT
    if isinstance(value, str) and value in OD_MODEL_SETS:
        transit = params.get(OD_TRANSIT_KEY, "fixed")
        if transit == "fixed":
            return value, None
        if isinstance(transit, str) and transit in BALLISTIC_MODELS:
            model = BALLISTIC_MODELS[transit]
            if transit in (BALLISTIC_TRANSIT, BALLISTIC_FAST_TRANSIT):
                # Older automatic selectors produced a spectrum only on pure 85Rb D1,
                # including a Custom 100% endpoint. Preserve those replay IDs.
                # Current Custom mixtures keep their common isotope-resolved ID at both endpoints.
                medium = params.get("species", "Rb (natural)")
                use_pure_history = (_pure_rb85_d1(params) and
                                    (value != OD_MODELS_NTOTAL or medium == species.RB85.label))
                if medium in _RB_MEDIA and not use_pure_history:
                    model = RB_HISTORY_REFERENCE if transit == BALLISTIC_TRANSIT else RB_HISTORY_FAST
        elif transit == GAUSSIAN_TRANSIT:
            model = _versioned_model(value, GAUSSIAN_TRANSIT_D20_MODEL, GAUSSIAN_TRANSIT_MODEL)
        else:
            raise ODModelSettingError(f"Unknown pump-off OD transit model {transit!r}.")
        return (value, None) if driven else (_EXPLICIT_OD_MODELS[model], model)
    if isinstance(value, str) and value in _EXPLICIT_OD_MODELS:
        model = None if driven and value in GEOMETRY_MODELS else value
        return _EXPLICIT_OD_MODELS[value], model
    raise ODModelSettingError(
        f"Invalid OD model ID {value!r}. Choose one of: {', '.join(OD_MODEL_CHOICES)}.")


def _versioned_model(model_set, original, measured_d2):
    if model_set == OD_MODELS_NTOTAL:
        if measured_d2 == PUMPED_TRANSIT_MODEL_V2:
            return PUMPED_TRANSIT_MODEL_V3
        return _NTOTAL_MODEL_IDS[measured_d2]
    return measured_d2 if model_set == OD_MODELS_D2SB else original


def _ntotal_uniform_rb(params, model_set):
    return (model_set == OD_MODELS_NTOTAL and params.get("species") in _RB_MEDIA
            and params.get("cell_propagation", "Uniform") == "Uniform")


def per_line_self_broadening(params):
    """Whether the finite-probe reference takes the measured Rb D2 self-broadening.

    Both d20_d2sb and its d20_ntotal successor select the line-specific coefficients.
    """
    return _od_model_setting(params)[0] in (OD_MODELS_D2SB, OD_MODELS_NTOTAL)


def _rate_scope_reason(params, *, geometry=False):
    """Why the magnetic-sublevel rate model cannot describe these parameters, or None."""
    if params.get("species") not in _RB_MEDIA or params.get("line") not in ("D1", "D2"):
        return "The magnetic-sublevel rate model covers rubidium D1 and D2 only."
    if geometry and _pump_is_driven(params.get("pump_power_mw", 0.0)):
        return "Gaussian transit describes single-probe OD with the pump off."
    if not geometry and not bool(params.get("constrain_doppler_temp", True)):
        return "The magnetic-sublevel rate model needs the Doppler temperature tied to the cell."
    if (float(params.get("ne_pressure_torr", 0.0)) != 0.0 or params.get("paraffin_coated", False)
            or _measured_collision_mode(params)):
        return "The magnetic-sublevel rate model covers uncoated cells without buffer gas."
    if params.get("cell_propagation", "Uniform") != "Uniform":
        return "The magnetic-sublevel rate model needs the Uniform cell."
    return None


def _pure_rb85_d1(params):
    pure = (params.get("species") == species.RB85.label or
            (params.get("species") == "Rb (natural)"
             and params.get("rb_composition") == RB_COMPOSITION_CUSTOM
             and params.get("rb85_fraction") == 1.0))
    return pure and params.get("line") == "D1"


def _ballistic_scope_reason(params, model=BALLISTIC_REFERENCE):
    if params.get("species", "Rb (natural)") not in _RB_MEDIA or params.get("line") not in ("D1", "D2"):
        return "Beam-crossing histories require rubidium on D1 or D2."
    if model not in RB_MIXTURE_HISTORY_MODELS and not _pure_rb85_d1(params):
        return "This recorded beam-crossing model requires pure ⁸⁵Rb on D1."
    if _pump_is_driven(params.get("pump_power_mw", 0.0)):
        return "Turn off the pump for the single-probe beam-crossing model."
    if params.get("probe_waist_mm", 1.0) <= 0 or params.get("cell_mm", 1.0) < 0:
        return "Beam-crossing histories require a positive beam radius and nonnegative cell length."
    return _rate_scope_reason(params, geometry=True)


def _pump_free_model_id(params):
    """Model of the pump-off solve for these parameters (explicit ID or model set)."""
    model_set, explicit = _od_model_setting(params, driven=False)
    if explicit is not None:
        return explicit
    if _probe_power_mw(params) > 0.0:
        if model_set != OD_MODELS_PRE_D20 and _rate_scope_reason(params) is None:
            return _versioned_model(model_set, RATE_MODEL, RATE_MODEL_V2)
        return _versioned_model(model_set, TWO_LEVEL_MODEL, TWO_LEVEL_MODEL_V2)
    model = _weak_probe_model_id(_weak_probe_profile(dict(params, pump_power_mw=0.0)))
    if model_set != OD_MODELS_PRE_D20 and model == EXACT_MODEL:
        return _versioned_model(model_set, TRANSIT_WEAK_MODEL, TRANSIT_WEAK_MODEL_V2)
    if model == LEGACY_MODEL and _ntotal_uniform_rb(params, model_set):
        return GRID_NTOTAL_MODEL
    return model


def _pump_free_model_reason(params, model):
    """Why ``model`` cannot run the pump-off solve of these parameters, or None."""
    probe = _probe_power_mw(params)
    if model in (LEGACY_MODEL, GRID_NTOTAL_MODEL, EXACT_MODEL, *TRANSIT_WEAK_MODELS):
        if probe > 0.0:
            return "This weak-probe model needs zero probe power."
        if model == GRID_NTOTAL_MODEL and not _ntotal_uniform_rb(params, OD_MODELS_NTOTAL):
            return "The total-Rb-density grid model requires uniform rubidium."
        if model not in (LEGACY_MODEL, GRID_NTOTAL_MODEL) and not _exact_voigt_eligible(params):
            return ("The Exact Maxwell Voigt models need uniform, uncoated Rb D1/D2 "
                    "without buffer gas and with Legacy broadening.")
        return None
    if model in BALLISTIC_FIDELITIES:
        return _ballistic_scope_reason(params, model)
    if model in RATE_MODELS:
        return _rate_scope_reason(params, geometry=model in GAUSSIAN_TRANSIT_MODELS)
    return None


def _weak_probe_model_id(profile):
    """Model ID for a weak readout, including an inactive Exact Voigt request."""
    if profile is not None:
        if profile["applied"] == EXACT_WEAK_PROBE:
            return "weak_probe_exact_maxwell_voigt"
        if profile["applied"] == PUMPED_MAXWELL:
            return "weak_probe_maxwell_pumped_correction"
    return "weak_probe_legacy"


def _isotope_composition(params):
    """Resolve elemental-density shares without moving the Rb frequency origin.

    Natural Rb keeps its historical weights. Custom Rb assigns complementary
    shares of the same total elemental vapor density; zero-share isotopes are
    skipped entirely. The mixture always uses the Rb85 centroid, including the
    enriched endpoints, so changing composition cannot translate the spectrum.
    """
    species_key = params["species"]
    if species_key != "Rb (natural)":
        components = species.SPECIES[species_key]
        reference = components[0][0]
        fraction = (1.0 if reference is species.RB85 else
                    0.0 if reference is species.RB87 else None)
        return components, reference, "Pure", fraction, True
    composition = params.get("rb_composition", RB_COMPOSITION_NATURAL)
    if composition not in (RB_COMPOSITION_NATURAL, RB_COMPOSITION_CUSTOM):
        raise ValueError("Rb isotope composition must be Natural or Custom.")
    fraction = species.RB85.abundance
    if composition == RB_COMPOSITION_CUSTOM:
        fraction = float(params.get("rb85_fraction", species.RB85.abundance))
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("Rb85 fraction must be finite and between 0 and 1.")
    components = [(iso, weight) for iso, weight in
                  ((species.RB85, fraction), (species.RB87, 1.0 - fraction))
                  if weight > 0.0]
    # Keep the natural abundance literal to preserve the legacy spectrum bit
    # for bit (1 - .7217 differs from the stored .2783 by roundoff).
    if composition == RB_COMPOSITION_NATURAL:
        components = species.SPECIES[species_key]
    reference_matched = fraction in (0.0, species.RB85.abundance, 1.0)
    return components, species.RB85, composition, fraction, reference_matched


def _probe_validation_tier(species_key, line):
    """Provenance tier for the AutoOD-style finite-probe approximation."""
    components = species.SPECIES.get(species_key, ())
    if (line == "D2" and components
            and all(iso.name in PROBE_REFERENCE_RB_D2_ISOTOPES
                    for iso, _weight in components)):
        return "reference_scope"
    return "generalized"


def _probe_reference_matched(raw, params):
    """Whether the displayed result keeps every AutoOD regression condition."""
    return (
        raw.get("probe_validation_tier") == "reference_scope"
        and bool(raw.get("composition_reference_matched", True))
        and not _measured_collision_mode(params)
        and float(params.get("pump_power_mw", 0.0)) == 0.0
        and float(raw.get("buffer_gamma", 0.0)) == 0.0
        and bool(raw.get("doppler_temp_tied", True))
        and float(params.get("line_strength", 1.0)) == 1.0
    )


def _probe_scope_text(raw, params):
    """Short status and one detailed finite-probe qualification."""
    pump_on = float(params.get("pump_power_mw", 0.0)) > 0.0
    reference_scope = raw.get("probe_validation_tier") == "reference_scope"
    reference_matched = _probe_reference_matched(raw, params)
    if not reference_scope:
        value = "Generalized estimate"
        help_text = (
            f"{raw['species']} {raw['line']} uses line-specific two-level scaling "
            "without a matching finite-power reference.")
        detail = (
            f"For {raw['species']} {raw['line']}, this is a generalized "
            "closed-two-level estimate using the selected line's frequency, "
            "natural linewidth, and AutoOD-style relative hyperfine scale; "
            "it has no matching finite-power reference and is qualitative. "
            "mF state and polarization dependence are not resolved.")
        if raw["line"] == "D1":
            detail += " D1 also has no closed hyperfine cycling transition."
        if raw.get("self_broadening", {}).get("status") == "unsupported":
            detail += " The Cs–Cs self-broadening contribution is omitted."
        if pump_on:
            detail += (
                " The pump-prepared populations are frozen before this "
                "correction; simultaneous finite pump + finite probe also "
                "omits probe-induced hyperfine repumping.")
    elif raw.get("od_model_set") == OD_MODELS_NTOTAL and len(raw.get("isotope_fractions", {})) > 1:
        value = "Rb D2 mixture"
        help_text = "Both Rb isotopes contribute to the self-broadening of each transition."
        detail = (help_text + " The recorded AutoOD software reference used isotope-partial widths. "
                  "Finite-power agreement for the total-density mixture requires separate validation.")
        if pump_on:
            detail += " This solve holds the pump-prepared populations fixed during probe propagation."
    elif reference_matched:
        value = "Rb D2 reference-matched"
        help_text = (
            "Matches the supplied pump-off Rb D2 software reference; this is "
            "not finite-power experimental validation.")
        detail = (
            f"This pump-off, unbuffered, one-temperature {raw['species']} "
            "D2 mode is regression-matched to the supplied AutoOD calculator's "
            "isotope-composition scope; its finite-power output is not "
            "experimentally validated.")
    else:
        value = "Rb D2 extrapolation"
        help_text = (
            "Custom isotope ratio, pump, buffer gas, split temperatures, or line-strength scaling is "
            "outside the regression-matched conditions.")
        if pump_on:
            detail = (
                "The selected Rb D2 ensemble is inside the supplied calculator's "
                "isotope-composition scope, but the pump-prepared populations are "
                "frozen before this correction; simultaneous finite pump + finite "
                "probe is qualitative and omits probe-induced hyperfine repumping.")
        else:
            detail = (
                "Buffer broadening, a released Doppler temperature, or a calibrated "
                "line-strength factor extrapolates the finite-probe correction "
                "beyond the supplied AutoOD reference.")
        if not raw.get("composition_reference_matched", True):
            detail = (
                "The selected custom isotope ratio is outside the existing natural-"
                "abundance and pure-isotope regression anchors. " + detail)
    if raw.get("probe_passivity_limited", False):
        detail += (
            " The frozen-pump residual became non-passive and was reduced "
            "to a nonnegative-absorption scale; interpret this "
            "passivity-limited result qualitatively.")
    return value, help_text, detail


class SASScheme(Scheme):
    name = "sas"
    cluster = "A — Absorption"
    title = "Absorption spectroscopy (OD / SAS)"
    cache_version = "22"           # Fast v2 uses the reference trajectory table
    defaults_version = "5"         # µW probe control + expanded visibility
    supports_headless_observables = True
    model_comparison_parameter = "pump_power_mw"
    workspace_inputs = ("pump_power_mw", "probe_power_uw")
    model_comparison_controls = ("species", "rb85_fraction", "line", "temp_c", "cell_mm", "probe_power_uw")
    caption = ("Absorption with a counter-propagating pump and optional finite-power "
               "probe saturation for the built-in alkali D lines. Pump off → "
               "linear Doppler-broadened OD (thermal 85Rb D1 hyperfine scale); "
               "pump on → Doppler-free saturated-absorption Lamb dips and crossovers "
               "with hyperfine optical pumping. ⁸⁵Rb / ⁸⁷Rb / ¹³³Cs · D1/D2 or natural Rb.")

    def atom_conditions(self, params):
        cell = []
        if params.get("paraffin_coated", False):
            cell.append("Paraffin")
        if params.get("ne_pressure_torr", 0.) > 0:
            cell.append("Buffer")
        if params.get("species") == species.CS133.label and params.get("cs_self_broadening_reference"):
            rate = collisions.CS_SELF_BROADENING[params.get("line", "D1")]
            cell.append(f"Cs width ±{rate.relative_uncertainty:.2%}")
        return dict(fields=("species", "rb_composition", "rb85_fraction", "line", "temp_c", "cell_mm",
                            "constrain_doppler_temp", "doppler_temp_c", "paraffin_coated", "ne_pressure_torr"),
                    line="line", cell=" ".join(cell),
                    labels={"rb85_fraction": "⁸⁵Rb fraction (Natural 0.7217)"})

    def param_schema(self):
        return [
            ParamSpec("pump_power_mw", "Pump beam power", "Pump", 0.5, 0.0, 2.0, 0.01,
                      "mW", endpoints=("◀ OD", "SAS ▶"),
                      help="Counter-propagating saturating beam. Pull to 0 → linear "
                      "absorption (OD); raise → Doppler-free SAS features. Converted "
                      "to a Rabi frequency via the beam waist and I_sat (see About)."),
            ParamSpec("probe_power_uw", "Probe beam power", "Probe", 0.0,
                      0.0, 100.0, 0.1, "µW",
                      endpoints=("◀ weak probe", "saturated ▶"),
                      visible_if={"species": tuple(species.SPECIES_ORDER)},
                      help="0 keeps the exact weak-probe OD/SAS result. At finite power "
                           "with the pump off, uniform uncoated Rb cells use the "
                           "magnetic-sublevel rate model (optical pumping among m "
                           "sublevels); other media and pumped SAS use the "
                           "independent-transition two-level estimate. Both propagate "
                           "the Gaussian beam self-consistently through the cell."),
            ParamSpec("species", "Atom", "Atomic", "Rb (natural)",
                      choices=tuple(species.SPECIES_ORDER), atom_menu=ATOM_MENU,
                      help="Natural: 72.17% ⁸⁵Rb and 27.83% ⁸⁷Rb by abundance. "
                           "Custom sets the ⁸⁵Rb fraction."),
            ParamSpec("rb_composition", "Isotope", "Atomic",
                      RB_COMPOSITION_NATURAL,
                      choices=(RB_COMPOSITION_NATURAL, RB_COMPOSITION_CUSTOM),
                      hidden=True,   # set by the species Atom / Isotope menu
                      help="Natural uses 72.17% ⁸⁵Rb and 27.83% ⁸⁷Rb. Custom "
                           "sets their complementary shares of the same total Rb "
                           "vapor density. 85 and 87 select a pure isotope."),
            ParamSpec("rb85_fraction", "⁸⁵Rb fraction", "Atomic",
                      species.RB85.abundance, 0.0, 1.0, 0.0001,
                      visible_if={"species": "Rb (natural)",
                                  "rb_composition": RB_COMPOSITION_CUSTOM},
                      format="%.4f",
                      help="⁸⁵Rb share of total Rb density; ⁸⁷Rb uses 1 minus "
                           "this fraction. 0.7217 is natural abundance. Mixture "
                           "frequencies always refer to the ⁸⁵Rb centroid."),
            ParamSpec("line", "Transition line", "Atomic", "D1",
                      choices=("D1", "D2"),
                      help="D1 (nP₁/₂) or D2 (nP₃/₂). Sets the excited hyperfine "
                      "manifold and line-specific probe saturation scale. The supplied "
                      "AutoOD regression anchors cover Rb D2; other built-in lines "
                      "use a clearly labelled closed-two-level generalization."),
            ParamSpec("temp_c", "Cell temperature", "Cell & beams", 40.0, 20.0, 200.0,
                      1.0, "°C",
                      help="Cold-spot temperature: it fixes the saturated vapor pressure, "
                      "hence the number density (absorption scale) and the density-"
                      "dependent self-broadening. It also sets the Doppler width unless "
                      "the Advanced Doppler-width temperature is released."),
            ParamSpec("cell_mm", "Cell length", "Cell & beams", 75.0, 0.5, 200.0, 0.5, "mm",
                      recompute=False),
            ParamSpec(
                "constrain_doppler_temp", "Tie Doppler width to cell temperature",
                "Cell & beams", True, advanced=True, control="checkbox",
                advanced_group="Cell temperatures",
                help="On (default): one temperature sets both the vapor density and "
                     "the Doppler width. Off: the cell temperature is the cold spot that "
                     "fixes the vapor pressure, and the Doppler-width temperature below "
                     "is the beam-path temperature: it sets the thermal velocity "
                     "distribution and the density n_sat(T_cold)·T_cold/T_Doppler.",
            ),
            ParamSpec("doppler_temp_c", "Doppler-width temperature", "Cell & beams",
                      40.0, 20.0, 300.0, 1.0, "°C", advanced=True,
                      advanced_group="Cell temperatures",
                      visible_if={"constrain_doppler_temp": False},
                      help="Temperature of the atoms crossing the beam: it sets the "
                      "thermal velocity distribution (Doppler width, velocity "
                      "classes) and, at the cold-spot vapor pressure, the density "
                      "n_sat(T_cold)·T_cold/T_Doppler. It must be at or above the cell "
                      "temperature. Ignored while the tie above is on."),
            ParamSpec(
                "paraffin_coated", "Cell type", "Cell & beams", False,
                advanced=True, choices=(False, True),
                choice_labels={False: "Uncoated", True: "Paraffin coated"},
                visible_if={"species": tuple(species.SPECIES_ORDER)},
                help="Preserve ground-hyperfine population between velocity-randomized "
                     "beam passages using a nominal coated-cell reservoir. This changes "
                     "atomic pumping only; it does not apply a coating transmission loss.",
            ),
            ParamSpec("cs_self_broadening_reference", "Cs self-broadening", "Cell & beams",
                      False, choices=(False, True), advanced=True,
                      choice_labels={False: "Off", True: "Jabbour reference (high uncertainty)"},
                      visible_if={"species": species.CS133.label},
                      help="Add a Cs density-dependent optical linewidth. The coefficient "
                           "uncertainty including inter-study spread exceeds 20%."),
            ParamSpec("optical_collision_model", "Optical collisions", "Cell & beams",
                      collisions.LEGACY_COLLISIONS, choices=collisions.COLLISION_CHOICES,
                      advanced=True,
                      help="Measured Ne width and shift with source uncertainty. The reference "
                           "uses uniform-cell propagation."),
            ParamSpec("ne_pressure_torr", "Ne buffer pressure", "Cell & beams", 0.0,
                      0.0, 200.0, 1.0, "Torr", advanced=True,
                      help="Ne partial pressure at the Doppler-width temperature."),
            ParamSpec("waist_mm", "Pump beam waist (1/e²)", "Pump", 1.0, 0.1, 5.0, 0.05,
                      "mm", advanced=True, help="Sets the pump intensity I = 2P/(πw²) "
                      "for the power→Rabi conversion."),
            ParamSpec("probe_waist_mm", "Probe beam waist (1/e²)", "Probe", 1.0,
                      0.1, 5.0, 0.05, "mm", advanced=True,
                      visible_if={"species": tuple(species.SPECIES_ORDER)},
                      help="Sets the Gaussian peak probe intensity I₀ = 2P/(πw²). "
                           "The Rb-D2 AutoOD reference's 2 mm 1/e² diameter is "
                           "the 1 mm waist used here. The model holds this waist "
                           "constant through the cell and assumes full output-beam "
                           "collection; check the Rayleigh range for small waists."),
            ParamSpec(OD_TRANSIT_KEY, "Pump-off OD transit", "Atomic", "fixed",
                      choices=("fixed", GAUSSIAN_TRANSIT, *BALLISTIC_MODELS),
                      choice_labels={"fixed": "Fixed rate", GAUSSIAN_TRANSIT: "Gaussian beam (experimental)",
                                     BALLISTIC_TRANSIT: "Beam-crossing history (reference)",
                                     BALLISTIC_FAST_TRANSIT: "Beam-crossing history (Fast)",
                                     FIELD_HISTORY_TRANSIT: "Beam-crossing history (static field)"},
                      advanced=True,
                      help="Turning on the pump uses the SAS transit relaxation rate for both comparison curves. "
                           "Returning to zero pump restores the selected OD model. "
                           "Gaussian beam estimates the replacement rate from the probe radius and kinetic "
                           "temperature in uniform, uncoated, buffer-free Rb. Beam-crossing history follows "
                           "each Rb isotope through a linearly polarized probe at zero field. "
                           "The static-field option accepts a Jones polarization. "
                           "The precomputed Fast table covers pure ⁸⁵Rb D1 at a 0.530 mm radius, 95–107 °C, up to 160 µW "
                           "and a cell up to 12.5 mm long. Other histories use direct trajectories. The zero-field D2 "
                           "history uses a population approximation; static-field D2 retains coherences."),
            *(ParamSpec(key, label, "Atomic", 0.0, low, high, step, unit,
                        advanced=True, help=help_text,
                        visible_if={OD_TRANSIT_KEY: FIELD_HISTORY_TRANSIT, "pump_power_mw": 0.0})
              for key, label, low, high, step, unit, help_text in (
                  ("od_field_x_ut", "Magnetic field Bₓ", None, None, 1.0, "µT",
                   "Static field along the probe’s transverse Jones x axis."),
                  ("od_field_y_ut", "Magnetic field Bᵧ", None, None, 1.0, "µT",
                   "Static field along the probe’s transverse Jones y axis."),
                  ("od_field_z_ut", "Magnetic field B_z", None, None, 1.0, "µT",
                   "Positive z points along probe propagation."),
                  ("od_probe_azimuth_deg", "Probe azimuth ψ", None, None, 1.0, "°",
                   "Major-axis angle from Jones x toward y. With zero ellipticity, 0° is x-linear."),
                  ("od_probe_ellipticity_deg", "Probe ellipticity ε", -45.0, 45.0, 1.0, "°",
                   "0° is linear, ±45° is circular. Positive ε gives positive S₃ = 2 Im(Eₓ* Eᵧ)."),
              )),
            ParamSpec("transit_khz", "Transit relaxation γ_t/2π", "Atomic", 100.0,
                      5.0, 2000.0, 5.0, "kHz", advanced=True,
                      help="Atoms leaving/entering the beam relax toward the thermal "
                      "ground state — regularises hyperfine pumping. Smaller γ_t → "
                      "stronger inverted crossovers. The experimental Gaussian OD model derives "
                      "this rate from beam radius and kinetic temperature instead."),
            ParamSpec("line_strength", "Line-strength factor", "Detection & scaling", 1.0,
                      0.01, 2.0, 0.01, "", advanced=True, recompute=False,
                      help="Scale the optical readout, including absorption used in propagation. "
                      "At a given local intensity, the atomic drive and populations are unchanged. "
                      "1.0 uses the thermal 85Rb D1 sublevel normalization at pump = 0."),
            ParamSpec("transitions", "Generic: transitions", "Atomic", "single line",
                      choices=("single line", "two lines (crossover)"), advanced=True,
                      hidden=True, help="Internal legacy Generic mode only."),
            ParamSpec("splitting", "Generic: excited splitting", "Atomic", 60.0,
                      5.0, 200.0, 1.0, "Γ", advanced=True, hidden=True,
                      help="Internal legacy Generic mode only."),
            ParamSpec("cell_propagation", "Cell propagation", "Cell & beams", "Uniform",
                      choices=("Uniform", sas_axial.SEGMENTED), advanced=True,
                      advanced_group="Axial reference",
                      help="Segmented reference uses the beam-path temperature as its "
                           "axial midpoint and requires 0 µW probe. Pumped coated cells "
                           "require the uniform model."),
            ParamSpec("axial_segments", "Axial segments", "Cell & beams", sas_axial.DEFAULT_SEGMENTS,
                      1, 16, 1, advanced=True, advanced_group="Axial reference",
                      visible_if={"cell_propagation": sas_axial.SEGMENTED},
                      help="The reference compares this count against twice as many segments."),
            ParamSpec("axial_temperature_span_c", "Axial temperature change", "Cell & beams", 0.0,
                      -60.0, 60.0, 0.5, "°C", advanced=True, advanced_group="Axial reference",
                      visible_if={"cell_propagation": sas_axial.SEGMENTED},
                      help="Linear probe-exit minus probe-entrance temperature. "
                           "Both endpoints must be at or above the cold spot."),
            ParamSpec("axial_center_field_ut", "Axial center field", "Cell & beams", 0.0,
                      -100.0, 100.0, 1.0, "µT", advanced=True, advanced_group="Axial reference",
                      visible_if={"cell_propagation": sas_axial.SEGMENTED},
                      help="Longitudinal field at the cell center. Magnetic line shifts "
                           "retain hyperfine-lumped populations."),
            ParamSpec("axial_field_span_ut", "Axial field change", "Cell & beams", 0.0,
                      -200.0, 200.0, 2.0, "µT", advanced=True, advanced_group="Axial reference",
                      visible_if={"cell_propagation": sas_axial.SEGMENTED},
                      help="Linear probe-exit minus probe-entrance magnetic field."),
            ParamSpec("axial_pump_depletion", "Axial pump depletion", "Pump", False,
                      control="checkbox", advanced=True, advanced_group="Axial reference",
                      visible_if={"cell_propagation": sas_axial.SEGMENTED},
                      help="Propagate the pump from the probe exit toward the probe entrance."),
            # User decision 2026-11-24: use Automatic for new calculations.
            # Retain the profile values in the schema for saved-model replay.
            ParamSpec("weak_probe_profile", "Weak-probe line profile", "Numerics",
                      AUTO_WEAK_PROBE, choices=(AUTO_WEAK_PROBE, LEGACY_WEAK_PROBE, EXACT_WEAK_PROBE),
                      hidden=True),
            ParamSpec("scan_points", "Scan points", "Numerics", 1401,
                      401, 4001, 100, "", advanced=True),
            ParamSpec(OD_MODEL_KEY, "OD model", "Numerics", OD_MODELS_DEFAULT,
                      choices=OD_MODEL_CHOICES, advanced=True, hidden=True,
                      help="d20_ntotal (default) uses all Rb isotopes as self-broadening partners in uniform cells. "
                           "d20_d2sb retains each isotope's own perturber density and uses "
                           "the measured Rb D2 self-broadening coefficient (1.10×10⁻⁷ Hz cm³) "
                           "in place of the D1 value. d20 adds the transit width to the "
                           "Exact profiles (pumped too) and uses the magnetic-sublevel rate "
                           "model for a finite Rb probe; pre_d20 keeps the models used before "
                           "the OD-fit port. Both replay older saved settings. An "
                           "explicit model ID fixes the pump-off solve."),
        ]

    def recommended_defaults(self, params):
        """Two labelled presets — 'OD default' (pump off) and 'SAS default' (pump
        on) — for the current atom/line. Same temperature & cell; they differ only
        in the pump power."""
        if params.get("species", "Rb (natural)") == GENERIC:
            base, sas_power = dict(temp_c=50.0, cell_mm=10.0), 0.5
        else:
            rec = species.recommended(params.get("species", "Rb (natural)"),
                                      params.get("line", "D1"))
            base = dict(temp_c=rec["temp_c"], cell_mm=rec["cell_mm"])
            sas_power = rec["pump_power_mw"]
        base["probe_power_uw"] = 0.0
        return {"OD default": {**base, "pump_power_mw": 0.0},
                "SAS default": {**base, "pump_power_mw": sas_power}}

    def info(self):
        return (
            "**Absorption spectroscopy (OD / SAS).** A probe measures the vapor "
            "absorption; a counter-propagating pump of power *P* saturates the "
            "resonant velocity class.\n\n"
            "- **Pump off (P = 0):** linear Doppler-broadened absorption (OD). For "
            "85Rb D1 at 90 C, the pump-off area and peak agree within 0.1 % "
            "with GABES's independent OD implementation under the same "
            "normalization. This is an internal numerical comparison.\n"
            "- **Pump on:** Doppler-free Lamb dips + crossovers; CG-branched decay "
            "pumps population into the other ground hyperfine state, enhancing / "
            "inverting the crossovers (the dominant feature of real alkali SAS).\n\n"
            "**Transit width.** Atoms crossing the beam relax at the transit rate "
            "γ_t, which also widens each optical line: the Exact Maxwell Voigt "
            "profiles use an optical HWHM of (Γ + Γ_self + Γ_buffer)/2 + γ_t, with "
            "and without the pump. Uniform, uncoated Rb with zero probe power and "
            "no buffer gas uses the Exact background under the standard collision "
            "model. A driven pump adds a velocity-resolved population correction. "
            "Grid-based configurations use the historical width without γ_t.\n\n"
            "**Finite probe power.** With the pump off, a nonzero probe on a uniform, "
            "uncoated, buffer-free Rb cell uses the magnetic-sublevel rate model: a "
            "linearly polarized probe at zero field drives π links between (F, |m|) "
            "populations, which relax by spontaneous decay and transit, followed by "
            "longitudinal depletion and Gaussian beam averaging. The same model "
            "draws the theory curve and fits measured OD. Rate equations drop "
            "coherences between optical links; against the full optical Bloch model "
            "the local absorption is higher by at most 0.59 % at 100 µW and 1.62 % "
            "up to 2 mW (1 mm beam radius); after beam averaging and propagation "
            "the transmission differs by at most 1.85×10⁻³. Other media, and a finite probe with "
            "the pump on, use independent-transition two-level power broadening; "
            "the Finite-probe model table reports its scope and limits.\n\n"
            "The **pump power** maps to the Rabi scale via I = 2P/(πw²) and "
            "Ω = Γ·√(I/2I_sat), using the Advanced pump waist and Steck's "
            "effective far-detuned π-polarized I_sat for the isotope and line "
            "([D-line tables](https://steck.us/alkalidata/)).\n\n"
            "The Advanced **Paraffin-coated cell** switch retains the two ground-"
            "hyperfine populations between beam passages while rethermalizing velocity. "
            "It uses the existing transit rate for beam exchange and a fixed reference "
            "population lifetime T1 = 25.1 ms, measured for a paraffin-coated 87Rb cell "
            "at 300 K. This is a quasi-static population-memory model, not a calibrated "
            "cell-geometry model; it preserves no Zeeman coherence and adds no window "
            "or coating throughput loss.\n\n"
            "**Two temperatures.** The **cell temperature** is the vapor-density "
            "temperature: it fixes the number density (absorption scale) and the "
            "density-dependent self-broadening. The **Doppler-width temperature** "
            "sets the thermal velocity distribution only. They are tied by default; "
            "release the Advanced tie to model a cold spot below the beam-path "
            "temperature. Buffer-gas broadening here is pressure-only (no "
            "temperature dependence) and the transit rate stays a separate knob.\n\n"
            "**Dispersion.** In the weak-probe limit, absorption and its "
            "Kramers-Kronig partner come from "
            "one complex line profile, so the **Dispersion** view shows the refractive "
            "index n = 1 + Re χ/2 (dilute vapor) built from the *same* velocity-"
            "resolved population difference as the absorption — under pumping it is "
            "the saturated medium's dispersion, not a weak-probe curve pasted next to "
            "a saturated spectrum. The reported **Peak phase shift** is a single-pass "
            "phase φ = k(n−1)L, not a group delay. At finite probe power, the "
            "reference defines collected transmission but no unique phase for a "
            "beam whose susceptibility varies across radius and cell depth, so the "
            "Dispersion view and phase metric are intentionally suppressed.\n\n"
            "The reported **Gaussian Doppler FWHM** is calculated from the thermal "
            "velocity distribution for one optical line; it is not a measured "
            "Voigt or multi-line envelope FWHM. Sub-Doppler widths use interpolated "
            "half-height edges and report samples per FWHM plus scan-edge clearance. "
            "A local lock slope is shown only when that resolution status is "
            "resolved.\n\n"
            "Atomic data (hyperfine A/B, line centres, masses, linewidths) from the "
            "Steck D-line data sheets; Wigner-6j/3j line strengths in the AutoOD "
            "convention. Rb densities use the CRC vapor pressure (AutoOD), Cs the "
            "Steck fit.\n\n"
            "**References**\n"
            "- D. A. Smith & I. G. Hughes, *Am. J. Phys.* **72**, 631 (2004).\n"
            "- D. W. Preston, *Am. J. Phys.* **64**, 1432 (1996).\n"
            "- T. Bandi, C. Affolderbach & G. Mileti, *J. Appl. Phys.* **111**, "
            "124906 (2012), https://doi.org/10.1063/1.4729925.\n"
            "- D. A. Steck, *Rubidium 85 / 87 & Cesium D Line Data*, http://steck.us/alkalidata."
        )

    # =================================================================
    # compute  (dispatch)
    # =================================================================
    def model_comparison(self, params, *, resolve_raw, current_raw=None, current_view=None):
        from .sas_comparison import pump_power_comparison
        return pump_power_comparison(self, params, resolve_raw=resolve_raw,
                                     current_raw=current_raw, current_view=current_view)

    def extra_views(self):
        return [ExtraView(
            key=PROJECTED_PROBE_VIEW,
            description="Calculate transmission with optical pumping driven by the probe. "
                        "Use a uniform, uncoated Rb cell with the pump off and no buffer gas.",
            compute=self.projected_probe_reference,
            render=self._render_projected_probe,
            param_keys=("cell_mm", "line_strength"),
        )]

    def extra_view_unavailable(self, key, params):
        if key != PROJECTED_PROBE_VIEW:
            return None
        if (params.get("species") not in _RB_MEDIA or params.get("line") not in ("D1", "D2")
                or float(params.get("pump_power_mw", 0.0)) != 0.0
                or params.get("cell_propagation", "Uniform") != "Uniform"
                or params.get("paraffin_coated", False)
                or float(params.get("ne_pressure_torr", 0.0)) != 0.0
                or _measured_collision_mode(params)):
            return ("Select Rb D1 or D2, turn off the pump, and use a uniform, uncoated cell "
                    "with zero buffer pressure and Legacy broadening.")
        if _od_model_setting(params)[1] in BALLISTIC_FIDELITIES:
            return "The stationary probe reference uses a replacement rate. Select Fixed rate or Gaussian beam to compare it."
        if not 0.0 <= _probe_power_mw(params) <= 0.1:
            return "Set the probe power between 0 and 100 µW."
        cold_k, path_k, _ = _temperatures(params)
        if not 293.15 <= cold_k <= 473.15 or not 293.15 <= path_k <= 573.15:
            return "Set the temperatures within the OD/SAS controls."
        if path_k < cold_k:
            return "Set the beam-path temperature at or above the cold-spot temperature."
        return None

    def projected_probe_reference(self, params):
        """Run the on-demand reference independently of the cached SAS solve."""
        from .sas_probe_reference import compute

        # This independently callable reference historically filled an omitted model
        # with d20_d2sb. Exported sessions supply an explicit set or ID.
        params = {**self.defaults(), OD_MODEL_KEY: OD_MODELS_D2SB, **params}
        unavailable = self.extra_view_unavailable(PROJECTED_PROBE_VIEW, params)
        if unavailable is not None:
            raise ValueError(unavailable)
        components, reference, *_ = _isotope_composition(params)
        cold_k, path_k, _ = _temperatures(params)
        total_rb = _ntotal_uniform_rb(params, _od_model_setting(params)[0])
        densities = [
            (iso, fraction, *_species_path_densities(iso, cold_k, path_k, fraction, total_rb=total_rb))
            for iso, fraction in components
        ]
        return compute(params, densities, reference, path_k, _probe_power_mw(params))

    @staticmethod
    def _render_projected_probe(data):
        from .sas_probe_reference import render

        return render(data)

    def normalize_params(self, params):
        """Select a live history for edits outside a recorded pure-85 D1 model.

        The UI retains the pinned choice and replays it again when the cell
        returns to its original composition. Explicit solver requests stay strict.
        """
        recorded = params.get(OD_MODEL_KEY)
        if (recorded not in (BALLISTIC_REFERENCE, BALLISTIC_FAST_V1, BALLISTIC_FAST)
                or params.get("species") not in _RB_MEDIA or params.get("line") not in ("D1", "D2")
                or _pure_rb85_d1(params)):
            return params
        reference = recorded == BALLISTIC_REFERENCE
        return {**params, OD_MODEL_KEY: RB_HISTORY_REFERENCE if reference else RB_HISTORY_FAST,
                OD_TRANSIT_KEY: BALLISTIC_TRANSIT if reference else BALLISTIC_FAST_TRANSIT}

    def migrate_default_params(self, params):
        # Saved snapshots without a profile predate the Exact Voigt default;
        # snapshots without an OD model predate the D20 model choices.
        migrated = {"weak_probe_profile": LEGACY_WEAK_PROBE, OD_MODEL_KEY: OD_MODELS_PRE_D20, **params}
        if migrated[OD_MODEL_KEY] == FIELD_HISTORY_REFERENCE:
            migrated[OD_TRANSIT_KEY] = FIELD_HISTORY_TRANSIT
        return migrated

    def solve_param_keys(self, params):
        keys = super().solve_param_keys(params)
        if ("cs_self_broadening_reference" not in params
                or params.get("species") != species.CS133.label):
            keys = tuple(key for key in keys if key != "cs_self_broadening_reference")
        if "weak_probe_profile" not in params:
            keys = tuple(key for key in keys if key != "weak_probe_profile")
        if OD_MODEL_KEY not in params:
            keys = tuple(key for key in keys if key != OD_MODEL_KEY)
        transit = params.get(OD_TRANSIT_KEY, "fixed")
        if transit == "fixed" or (transit in (GAUSSIAN_TRANSIT, *BALLISTIC_MODELS)
                                  and _pump_is_driven(params.get("pump_power_mw", 0.0))):
            keys = tuple(key for key in keys if key != OD_TRANSIT_KEY)
        # Invalid saved IDs are reported by compute inside the app's input-error handler.
        # Cache-key construction must also work before that handler runs.
        try:
            model = _od_model_setting(params)[1]
        except ODModelSettingError:
            model = None
        keys = tuple(key for key in keys if key not in FIELD_HISTORY_KEYS
                     or (model == FIELD_HISTORY_REFERENCE and key in params))
        if params.get("cell_propagation", "Uniform") == sas_axial.SEGMENTED:
            return keys + ("cell_mm", "line_strength")
        keys = tuple(key for key in keys if not key.startswith("axial_"))
        if model in BALLISTIC_FIDELITIES:
            return tuple(key for key in keys if key != "transit_khz") + ("cell_mm", "line_strength")
        return keys

    def compute(self, params):
        requested = _od_model_setting(params)[1]
        if requested in GEOMETRY_MODELS:
            reason = _pump_free_model_reason(params, requested)
            if reason:
                raise ODModelSettingError(reason)
        segmented = params.get("cell_propagation", "Uniform") == sas_axial.SEGMENTED
        if segmented and _measured_collision_mode(params):
            raise sas_axial.AxialInputError(
                "Measured Ne optical collisions require uniform-cell propagation. "
                "Select Legacy broadening to use the segmented reference.")
        if params.get("species", "Rb (natural)") == GENERIC:
            if segmented:
                raise sas_axial.AxialInputError("Segmented reference requires an isotope-resolved D-line species.")
            raw = self._compute_generic(params)
            profile = _weak_probe_profile(params)
            if profile is not None:
                raw["weak_probe_profile"] = profile
            return raw
        if segmented:
            components, reference, *_ = _isotope_composition(params)
            cs_reference = reference == species.CS133 and params.get("cs_self_broadening_reference", False)
            response = self._component_response
            if cs_reference:
                response = partial(response, cs_axial_reference=True)
            raw = sas_axial.compute(params, uniform=self._compute_species, components=components,
                                    reference=reference, response=response)
            if cs_reference:
                density = raw["axial_cell"]["density_m3"][reference.name]
                raw["axial_cell"]["cs_self_fwhm_rad_s"] = (
                    raw["self_broadening"]["coefficient_rad_s_m3"] * density)
            return raw
        return self._compute_species(params)

    def quantitative_export(self, params, optical_readout):
        """Check the selected feature on two denser, independently solved grids."""
        if params["pump_power_mw"] <= 0:
            return dict(status="not applicable", reason="Pump off.")
        x = optical_readout["x"]
        mhz_per_x = 1000.0 if optical_readout["x_unit"] == "GHz" else 1.0
        coarse = optical_readout["resolution"]
        center, width = coarse["center_mhz"], coarse["width_mhz"]
        evidence = dict(
            status="unresolved", linewidth_mhz=None, lock_slope_per_mhz=None,
            lock_detuning_mhz=None, coarse_grid_points=len(x),
            coarse_status=coarse["status"], coarse_center_mhz=center, checks=[],
        )
        if (center is None or width is None or width <= 0
                or coarse["status"] == "scan-edge-limited"):
            return dict(evidence, reason="The displayed scan has no usable feature identity.")
        center /= mhz_per_x
        width /= mhz_per_x
        window = (center - width, center + width)
        background_width = (float(x[-1] - x[0]) / 60.0)
        evidence.update(
            feature_window_mhz=[float(v * mhz_per_x) for v in window],
            background_median_width_mhz=background_width * mhz_per_x,
            width_relative_tolerance=0.05, slope_relative_tolerance=0.05,
            center_shift_tolerance_fwhm=0.25, coarse_center_tolerance_fwhm=0.5,
        )
        for points in (4001, 8001):
            condition = {**params, "scan_points": points}
            with core.blas_single_thread():
                raw = self.compute(condition)
                alpha, _chi, length = _collected_response(raw, condition)
            grid = raw["scan"] / (2 * np.pi) / (1e9 if mhz_per_x == 1000.0 else 1e6)
            refined = subdoppler_refinement(
                grid, observables.transmission(alpha, length), window, background_width,
                polarity=coarse["polarity"],
            )
            evidence["checks"].append(_quantitative_check(refined, grid, mhz_per_x))
        return _finish_quantitative_export(evidence)

    # ---- realistic multilevel OD/SAS model ----
    def _compute_species(self, params, *, scan_rad_s=None, extra_homogeneous_fwhm_hz=0.0):
        extra_fwhm = float(extra_homogeneous_fwhm_hz)
        if not np.isfinite(extra_fwhm) or extra_fwhm < 0.0:
            raise ValueError("Extra homogeneous FWHM must be finite and nonnegative.")
        line = params["line"]
        comps, iso_ref, composition, rb85_fraction, composition_matched = (
            _isotope_composition(params))
        T_density, T_doppler, tied = _temperatures(params)
        model_set, explicit_model = _od_model_setting(params)
        use_history = explicit_model in BALLISTIC_FIDELITIES
        field_history = explicit_model == FIELD_HISTORY_REFERENCE
        gt = 0.0 if use_history else rate_from_khz_over_2pi(params["transit_khz"])
        power, waist = params["pump_power_mw"], params["waist_mm"]
        probe_supported = (
            params["species"] in species.SPECIES_ORDER
            and line in ("D1", "D2")
        )
        probe_power = _probe_power_mw(params) if probe_supported else 0.0
        probe_validation_tier = _probe_validation_tier(
            params["species"], line)
        probe_waist = float(params.get("probe_waist_mm", 1.0))
        probe_peak_intensity = probe_saturation.gaussian_peak_intensity(
            probe_power, probe_waist)
        paraffin_coated = bool(params.get("paraffin_coated", False))
        # With no pump, coating memory cannot change a thermal OD spectrum. Keep
        # that limit on the legacy path exactly (and avoid an unnecessary solve).
        paraffin_memory = paraffin_coated and power > 0.0
        reference = _measured_collision_mode(params)
        cs_reference = iso_ref == species.CS133 and params.get("cs_self_broadening_reference", False)
        pump_free_model = None
        if probe_supported and not _pump_is_driven(power):
            pump_free_model = _pump_free_model_id(params)
            if reason := _pump_free_model_reason(params, pump_free_model):
                # A valid model outside its scope is an input error the app shows.
                raise ODModelSettingError(reason)
            if explicit_model in (LEGACY_MODEL, GRID_NTOTAL_MODEL, EXACT_MODEL, *TRANSIT_WEAK_MODELS):
                # An explicit weak-probe model fixes its velocity measure.
                params = dict(params, weak_probe_profile=(
                    LEGACY_WEAK_PROBE if explicit_model in (LEGACY_MODEL, GRID_NTOTAL_MODEL) else EXACT_WEAK_PROBE))
        weak_profile = None if use_history else _weak_probe_profile(params)
        applied_profile = weak_profile["applied"] if weak_profile is not None else None
        exact_weak_probe = applied_profile == EXACT_WEAK_PROBE
        pumped_maxwell = applied_profile == PUMPED_MAXWELL
        use_rate = pump_free_model in RATE_MODELS
        transit_weak = pump_free_model in TRANSIT_WEAK_MODELS
        pumped_transit = pumped_maxwell and model_set != OD_MODELS_PRE_D20
        # The measured Rb D2 self-broadening applies on the paths that record a
        # versioned line-specific ID. The weak grid keeps its historical coefficient.
        # The archived segmented solver and its uniform-limit scan retain isotope
        # densities and their historical coefficient on both D lines.
        total_rb = _ntotal_uniform_rb(params, model_set)
        per_line = model_set in (OD_MODELS_D2SB, OD_MODELS_NTOTAL) and (
            use_history or use_rate or transit_weak or pumped_transit or probe_peak_intensity > 0.0)
        pressure = params.get("ne_pressure_torr", 0.0)
        buffer_gamma = constants.neon_buffer_broadening(pressure) if not reference else 0.0
        collision_reports = {}
        self_reports = {}

        nu_ref = iso_ref.line(line)[1]

        built, omega_all = [], []
        dopp_fwhm, gamma_max, gamma_scan_max = 0.0, 0.0, 0.0
        geometry_rates = None
        if pump_free_model in GAUSSIAN_TRANSIT_MODELS:
            from ..od_model import ODConditions
            geometry_rates = ODConditions.from_params(params).transit_rates_s()
        for iso, weight in comps:
            transit = gt if geometry_rates is None else geometry_rates[iso.name]
            man = species.build_manifold(
                iso, line, transit_rate=0.0 if paraffin_memory else transit)
            offset = 2 * np.pi * (man.nu0 - nu_ref)
            sigma_v = constants.thermal_speed(T_doppler, iso.mass)
            dopp_fwhm = max(dopp_fwhm, np.sqrt(8 * np.log(2)) * man.k_vec * sigma_v)
            N, perturber_density = _species_path_densities(iso, T_density, T_doppler, weight, total_rb=total_rb)
            Op = species.pump_rabi_from_power(power, waist, man.gamma, iso=iso, line=line)
            pump_atom = man.atom
            if reference or cs_reference:
                self_fwhm, self_reports[iso.name] = collisions.self_broadening_reference(
                    iso.name, line, perturber_density, cs_reference=cs_reference)
            if reference:
                effects = collisions.pressure_effects(iso.name, line, pressure, T_doppler)
                buffer_gamma = max(buffer_gamma, effects.fwhm_rad_s)
                offset += effects.shift_rad_s
                gamma_eff = man.gamma + self_fwhm + effects.fwhm_rad_s
                pump_atom = pump_atom.with_optical_dephasing(
                    (self_fwhm + effects.fwhm_rad_s) / 2 + np.pi * extra_fwhm)
                collision_reports[iso.name] = effects.readout()
            elif cs_reference:
                gamma_eff = man.gamma + self_fwhm + buffer_gamma
                pump_atom = pump_atom.with_optical_dephasing(self_fwhm / 2)
            else:
                gamma_eff = (man.gamma + species.self_broadening(iso, perturber_density, line, per_line=per_line)
                             + buffer_gamma)
            if extra_fwhm:
                gamma_eff += 2.0 * np.pi * extra_fwhm
            saturation_intensity = (
                probe_saturation.closed_two_level_saturation_intensity(
                    man.nu0, man.gamma))
            probe_s0 = probe_saturation.peak_saturation_parameter(
                probe_peak_intensity, saturation_intensity,
                man.gamma, gamma_eff)
            gamma_max = max(gamma_max, gamma_eff)
            gamma_scan_max = max(
                gamma_scan_max, gamma_eff * np.sqrt(1.0 + probe_s0))
            built.append(dict(man=man, offset=offset, N=N, Op=Op, iso=iso,
                              pump_atom=pump_atom, gamma_eff=gamma_eff, transit_rate=transit,
                              paraffin_memory=paraffin_memory,
                              pumped_maxwell=pumped_maxwell,
                              probe_transit_hwhm=gt if pumped_transit else 0.0,
                              probe_saturation_max=probe_s0))
            omega_all.append(man.omega + offset)
        omega_all = np.concatenate(omega_all)

        margin = max(3.5 * dopp_fwhm, 6.0 * gamma_scan_max)
        if scan_rad_s is None:
            scan = np.linspace(omega_all.min() - margin, omega_all.max() + margin,
                               int(params["scan_points"]))
        else:
            scan = np.asarray(scan_rad_s, dtype=float)
            if scan.ndim != 1 or not scan.size or not np.isfinite(scan).all():
                raise ValueError("The frequency axis must be a nonempty, finite 1-D array.")
            if probe_power > 0.0 and not (use_rate or use_history):
                steps = np.diff(scan)
                if not steps.size or steps[0] <= 0.0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
                    raise ValueError("Finite-probe propagation requires an increasing, uniform frequency axis.")

        alpha = np.zeros(scan.size)
        chi_real = np.zeros(scan.size)
        markers = []
        probe_profiles = []
        ng_of = {id(b["man"]): len(b["man"].Fg) for b in built}
        for b in built:
            man, off = b["man"], b["offset"]
            if use_rate or field_history:
                pass   # the selected model supplies the weak-probe response below
            elif use_history or exact_weak_probe or transit_weak:
                # U1: the transit rate adds gamma_t to the optical HWHM.
                width = b["gamma_eff"] + 2.0 * gt if transit_weak else b["gamma_eff"]
                component_alpha, component_chi_real = linear_hyperfine.species_complex_voigt(
                    scan, man, b["N"], T_doppler, width, offset=off)
            else:
                component_alpha, component_chi_real, component_probe_profiles = (
                    self._component_response(b, scan, T_doppler))
                probe_profiles.extend(component_probe_profiles)
            if not (use_rate or field_history):
                alpha += component_alpha
                chi_real += component_chi_real
            ng = ng_of[id(man)]
            for t in range(man.omega.size):
                fg = man.Fg[man.g_idx[t]]
                fe = man.Fe[man.e_idx[t] - ng]
                markers.append((float((man.omega[t] + off) / (2 * np.pi) / 1e9),
                                f"{man.iso.label} {fg:g}→{fe:g}′"))

        probe_log_grid = None
        probe_alpha_table = None
        probe_saturation_max = max(
            (b["probe_saturation_max"] for b in built), default=0.0)
        if use_rate:
            from ..od_model import ODConditions, rate_model_display_tables
            conditions = ODConditions.from_params(
                params, extra_homogeneous_fwhm_hz=extra_fwhm)
            alpha, chi_real, probe_log_grid, probe_alpha_table = rate_model_display_tables(
                conditions, scan)
        if use_history:
            from ..od_model import ODConditions, evaluate_od_display
            conditions = ODConditions.from_params(params, extra_homogeneous_fwhm_hz=extra_fwhm)
            try:
                history = evaluate_od_display(conditions, scan)
            except ArithmeticError as error:
                raise ODModelSettingError(f"Beam-crossing reference did not converge: {error}") from error
            if field_history:
                scale = conditions.cell_mm * 1e-3 * conditions.line_strength
                alpha = history.od0 / scale if scale else np.zeros_like(scan)
                chi_real = None   # the field model returns collected intensity only
        if probe_peak_intensity > 0.0 and not (use_rate or use_history):
            probe_log_grid = probe_saturation.saturation_log_grid(
                probe_saturation_max)
            probe_background_table = np.zeros((probe_log_grid.size, scan.size))
            probe_residual_table = np.zeros((probe_log_grid.size, scan.size))
            spacing = float(scan[1] - scan[0])
            for profile in probe_profiles:
                saturation_t = (profile["saturation_max"]
                                * profile["rabi_rel"] ** 2)
                background = probe_saturation.power_broadened_profile_table(
                    profile["thermal"], spacing, profile["hwhm"],
                    saturation_t, probe_log_grid)
                residual = profile["pumped"] - profile["thermal"]
                prepared = probe_saturation.power_broadened_profile_table(
                    # Counter-propagating pump/probe kinematics map a probe
                    # homogeneous-width increment to half that scan-axis width
                    # for the narrow pump-prepared residual.
                    residual, spacing, 0.5 * profile["hwhm"],
                    saturation_t, probe_log_grid)
                probe_background_table += background
                probe_residual_table += prepared

            # The frozen-pump residual is a separable approximation, not a
            # jointly solved finite-pump/finite-probe OBE.  At strong settings
            # its independently broadened negative hole can exceed the passive
            # thermal background.  Enforce passivity per local-intensity row by
            # retaining a common residual scale eta in [0, 1], infinitesimally
            # inside the largest value for which background + eta*residual >= 0
            # at every frequency.  The interior margin avoids making a sampled
            # transparency point an artificial exact-zero ODE fixed point.  This
            # preserves the reference pump-off path (residual == 0), changes
            # well-behaved SAS settings not at all, and is reported to the user
            # whenever it materially limits the qualitative joint model.
            alpha_scale = max(
                float(np.max(alpha)),
                float(np.max(np.abs(probe_background_table))),
                1e-30,
            )
            tolerance = 1e-12 * alpha_scale
            probe_background_table = np.maximum(probe_background_table, 0.0)
            unbounded = probe_background_table + probe_residual_table
            residual_scales = np.ones(probe_log_grid.size, dtype=float)
            for row in range(probe_log_grid.size):
                material_gain = unbounded[row] < -tolerance
                if np.any(material_gain):
                    admissible = (
                        probe_background_table[row, material_gain]
                        / -probe_residual_table[row, material_gain]
                    )
                    residual_scales[row] = np.clip(
                        float(np.min(admissible))
                        * (1.0 - PROBE_PASSIVITY_INTERIOR),
                        0.0, 1.0)
            probe_alpha_table = (
                probe_background_table
                + residual_scales[:, None] * probe_residual_table
            )
            probe_alpha_table = np.maximum(probe_alpha_table, 0.0)
        else:
            residual_scales = None

        # Each component's own k converts its A_t into a susceptibility; the
        # single k reported here only turns that susceptibility into a phase, so
        # the reference isotope's value is enough (isotope shifts are ~1e-7 of k).
        k_ref = float(2 * np.pi * nu_ref / constants.C_LIGHT)
        result = dict(mode="species", scan=scan, alpha_unit=alpha,
                    chi_real_unit=chi_real, k_vec=k_ref,
                    dopp_fwhm=dopp_fwhm, buffer_gamma=buffer_gamma,
                    gamma_eff_max=gamma_max, markers=markers,
                    species=params["species"], line=line,
                    rb_composition=composition, rb85_fraction=rb85_fraction,
                    isotope_fractions={iso.name: float(weight)
                                       for iso, weight in comps},
                    pump_saturation_intensity_w_m2={
                        iso.name: species.pump_saturation_intensity(iso, line)
                        for iso, _ in comps},
                    frequency_reference=iso_ref.name,
                    composition_reference_matched=composition_matched,
                    density_temp_c=T_density - 273.15,
                    doppler_temp_c=T_doppler - 273.15,
                    doppler_temp_tied=tied,
                    probe_model=(pump_free_model if use_rate or use_history
                                 else _versioned_model(model_set, TWO_LEVEL_MODEL, TWO_LEVEL_MODEL_V2)
                                 if probe_peak_intensity > 0.0
                                 else pump_free_model if transit_weak
                                 else _versioned_model(model_set, PUMPED_TRANSIT_MODEL, PUMPED_TRANSIT_MODEL_V2)
                                 if pumped_transit
                                 else GRID_NTOTAL_MODEL if total_rb else _weak_probe_model_id(weak_profile)),
                    od_model_set=model_set,
                    probe_radial_order=96 if use_rate else probe_saturation.RADIAL_QUADRATURE_ORDER,
                    probe_saturation_supported=probe_supported,
                    probe_validation_tier=probe_validation_tier,
                    probe_power_mw=probe_power,
                    probe_peak_intensity=probe_peak_intensity,
                    probe_saturation_max=probe_saturation_max,
                    probe_intensity_log_grid=probe_log_grid,
                    probe_alpha_table_unit=probe_alpha_table,
                    probe_passivity_limited=(
                        residual_scales is not None
                        and bool(np.any(residual_scales < 1.0 - 1e-10))
                    ),
                    probe_residual_scale_min=(
                        float(np.min(residual_scales))
                        if residual_scales is not None else 1.0
                    ),
                    paraffin_coated=paraffin_coated,
                    paraffin_population_memory=(
                        dict(status="active" if paraffin_memory else "inactive-pump-off",
                             model="ground-hyperfine populations only",
                             t1_reference_s=PARAFFIN_REFERENCE_T1_S,
                             t1_reference_isotope="87Rb",
                             t1_reference_temperature_k=300.0,
                             t1_reference_doi="10.1063/1.4729925",
                             assumed_cycle_rate_s_1=gt if paraffin_memory else None,
                             cycle_rate_source="beam exchange uses the configured transit rate",
                             calibration_status="semi-quantitative; cell specific T1 and geometry uncalibrated",
                             optical_throughput_loss_applied=False)
                        if paraffin_coated else None),
                    paraffin_t1_s=(PARAFFIN_REFERENCE_T1_S
                                   if paraffin_coated else None))
        if use_history:
            result["ballistic_depth"] = history.optical_depth
            result["ballistic_diagnostics"] = history.diagnostics
            if field_history:
                result["field_history"] = dict(conditions=conditions.recorded_conditions(),
                                               offsets_hz=history.offsets_hz, transmission=history.transmission,
                                               od0=history.od0)
        if geometry_rates is not None:
            result["transit_rates_s"] = geometry_rates
        if weak_profile is not None:
            result["weak_probe_profile"] = weak_profile
        if reference:
            result["optical_collisions"] = collision_reports
            result["self_broadening_reference"] = self_reports
        if cs_reference:
            result["self_broadening"] = self_reports[iso_ref.name]
        elif iso_ref.name == species.CS133.name:
            result["self_broadening"] = {
                "status": "unsupported",
                "isotope": iso_ref.name,
                "line": line,
                "coefficient_rad_s_m3": None,
                "applied_fwhm_rad_s": 0.0,
                "source_doi": "10.1016/0022-4073(95)00115-2",
                "source_scope": "Jabbour et al. (1995), Table 2 and quoted uncertainties",
                "reason": "The uncertainty including inter-study spread exceeds 20%.",
            }
        return result

    def _component_response(self, b, scan, T, *, axial_optics=False, pump_attenuation=False,
                            cs_axial_reference=False):
        """(α, Re χ, finite-probe transition profiles) for one isotope.

        Both quadratures come from the same complex line profile

            χ_t(δ) = (A_t / k) · i / (π (Γ_eff/2 − i·δ_t)),

        whose imaginary part is the unit-area absorption profile ĝ_t used for α
        (∫ĝ dδ = 1) and whose real part is its Kramers-Kronig partner. Sign and
        normalisation follow `observables.chi_phys` (α = k·Im χ, absorptive for a
        passive transition), so n = 1 + Re χ/2 rises below resonance. The same
        velocity-resolved population difference and the same Doppler quadrature
        weight both parts in the weak-probe/pump-prepared response.  The optional
        finite-probe absorption table is built later; its spatially varying phase
        is intentionally not inferred from this weak-probe Re χ.
        """
        man, offset, N, Op = b["man"], b["offset"], b["N"], b["Op"]
        iso, k = b["iso"], man.k_vec
        ng = len(man.Fg)
        gamma_eff = b["gamma_eff"]
        pump_atom = b.get("pump_atom", man.atom)
        if cs_axial_reference:
            # The axial solver supplies this segment's density and legacy Ne width.
            self_fwhm, _ = collisions.self_broadening_reference(iso.name, man.line, N, cs_reference=True)
            gamma_eff += self_fwhm
            pump_atom = pump_atom.with_optical_dephasing(self_fwhm / 2)

        pumped_maxwell = b.get("pumped_maxwell", False)
        if pumped_maxwell:
            v, wt = doppler.velocity_grid(T, mass=iso.mass, dv=1.0, cutoff_sigma=5.0,
                                          normalize=False)
        else:
            v, wt = doppler.velocity_grid(T, mass=iso.mass, dv=1.0, cutoff_sigma=4.0)
        kv = k * v

        # Pump steady state ρ(Δ_eff): scan-independent H, one fine table.
        pump_off = Op <= 0.0
        om = man.omega
        DS = scan - offset
        if not pump_off:
            Hp = species.pump_hamiltonian(man, Op)
            covered_detunings = None
            if pumped_maxwell or b.get("paraffin_memory", False):
                covered_detunings = (
                    float(DS.min() + kv.min()), float(DS.max() + kv.max()))
            deff = _pump_detuning_axis(
                om, gamma_eff, covered_detunings=covered_detunings)
            if b.get("paraffin_memory", False):
                basis_pops = _basis_reset_pump_pops(
                    man, Hp, deff, b["transit_rate"], pump_atom=pump_atom)
            else:
                L0 = core.build_liouvillian(Hp, pump_atom)
                pops = _pump_pops(L0, deff, pump_atom.S_v, man.n_levels)

        # Homogeneous unit-area lineshape: the weak-probe 2-level absorption is a
        # Lorentzian of FWHM Γ_eff, ∫L̂ dδ = 1. Evaluated analytically at the probe
        # detuning (exact, no truncation); its Doppler sum is the Voigt.
        hwhm = gamma_eff / 2.0
        transit_hwhm = b.get("probe_transit_hwhm", 0.0)
        if transit_hwhm:
            hwhm = hwhm + transit_hwhm   # U3: probe Lorentzian of the pumped correction

        # Absolute per-line integrated absorption A_t (AutoOD normalisation, ls=1).
        Aline = species.line_integrated_alpha(iso, line=man.line, N=N)

        # Pump populations of the levels the lines connect, per reservoir source
        # (one unconditioned source unless the coated-cell reservoir is on).
        levels = sorted(set(man.g_idx.tolist()) | set(man.e_idx.tolist()))
        tables = reservoir = None
        if not pump_off:
            if b.get("paraffin_memory", False):
                reservoir = _coated_ground_populations(
                    man, basis_pops, deff, DS, kv, wt,
                    cycle_rate=b["transit_rate"])
                tables = basis_pops[:, :, levels].transpose(0, 2, 1)
            else:
                tables = pops[:, levels].T[None]
        finite_probe = b.get("probe_saturation_max", 0.0) > 0.0
        sums = kernels.sas_velocity_sums if kernels.available() else _velocity_sums
        ground = man.g_idx
        excited = man.e_idx
        if axial_optics:
            om, parent, weights = sas_axial.zeeman_components(man, b["field_ut"])
            ground, excited = ground[parent], excited[parent]
        sum_inputs = (DS, kv, wt, om, hwhm,
                      [levels.index(g) for g in ground], [levels.index(e) for e in excited],
                      np.asarray(man.p_ground)[ground])
        sum_options = dict(axis=None if pump_off else deff, tables=tables,
                           reservoir=reservoir, thermal_needed=finite_probe)
        if pumped_maxwell and pump_off:
            # The profile and the solve can disagree on whether a tiny power
            # drives the pump; with no pump the correction W − 1 is exactly zero
            # and only the exact Voigt background below remains.
            absorptive = dispersive = thermal = np.zeros((np.size(om), DS.size))
        else:
            if pumped_maxwell:
                from scipy.interpolate import CubicSpline
                # A cutoff on the pump-state table would reuse nonthermal endpoint
                # populations at far-detuned velocities. Interpolate over the full
                # detuning domain of the correction at the inherited table spacing.
                sum_options.update(population_spline=CubicSpline(deff, tables, axis=-1),
                                   population_delta=True)
                if not kernels.available():
                    sums = _chunked_pumped_velocity_sums
            absorptive, dispersive, thermal = sums(*sum_inputs, **sum_options)
        if axial_optics:
            pump_absorption = (sums(*sum_inputs, **sum_options, same_direction=True)[0]
                               if pump_attenuation and not pump_off else absorptive)
            integrated = weights * np.asarray([
                Aline[(man.Fg[g], man.Fe[e - ng])] for g, e in zip(ground, excited)])
            return (integrated @ absorptive, -(integrated / k) @ dispersive,
                    integrated @ pump_absorption)

        alpha = np.zeros(scan.size)
        chi_real = np.zeros(scan.size)
        probe_profiles = []
        for t in range(om.size):
            g, e = man.g_idx[t], man.e_idx[t]
            fg, fe = man.Fg[g], man.Fe[e - ng]
            A_t = Aline[(fg, fe)]
            transition_alpha = A_t * absorptive[t]
            alpha += transition_alpha
            chi_real -= (A_t / k) * dispersive[t]
            if finite_probe:
                # At pump=0 the thermal sum is the pumped one, so the finite-probe
                # pump-induced residual is exactly zero.
                probe_profiles.append(dict(
                    pumped=transition_alpha,
                    thermal=A_t * thermal[t],
                    hwhm=hwhm,
                    rabi_rel=float(man.rabi_rel[t]),
                    saturation_max=float(b["probe_saturation_max"]),
                ))
        if pumped_maxwell:
            background_alpha, background_chi = linear_hyperfine.species_complex_voigt(
                scan, man, N, T, gamma_eff + 2.0 * transit_hwhm if transit_hwhm else gamma_eff,
                offset=offset)
            alpha += background_alpha
            chi_real += background_chi
        return alpha, chi_real, probe_profiles

    # ---- generic Γ-unit hole-burning toy (pedagogical) ----
    def _compute_generic(self, params):
        two_lines = params["transitions"].startswith("two")
        if two_lines:
            split = params["splitting"] * GAMMA
            n_exc, offsets = 2, np.array([-split / 2, split / 2])
        else:
            n_exc, offsets = 1, np.array([0.0])

        T_density, T_doppler, tied = _temperatures(params)
        mode = params.get("optical_collision_model", collisions.LEGACY_COLLISIONS)
        effects = collisions.pressure_effects(
            "Rb85", "D1", params.get("ne_pressure_torr", 0.0), T_doppler, mode=mode)
        reference = _measured_collision_mode(params)
        buffer_gamma = effects.fwhm_rad_s
        gamma_eff = GAMMA + buffer_gamma
        radiative_gamma = GAMMA if reference else gamma_eff
        elastic_hwhm = effects.optical_dephasing_rad_s if reference else 0.0
        Op = species.pump_rabi_from_power(params["pump_power_mw"], params["waist_mm"], GAMMA)
        pump_off = Op <= 0.0
        atom = None
        L0_pump = None
        if not pump_off:
            atom = atoms.sas_atom(
                n_exc, gamma=radiative_gamma).with_optical_dephasing(elastic_hwhm)
            Hp = np.zeros((atom.n_levels, atom.n_levels), dtype=complex)
            for i, e in enumerate(atom.excited):
                Hp[e, e] = offsets[i]
                Hp[0, e] = Hp[e, 0] = Op / 2
            L0_pump = core.build_liouvillian(Hp, atom)

        sigma_v = constants.thermal_speed(T_doppler, constants.MASS_85RB)
        dopp_fwhm = np.sqrt(8 * np.log(2)) * K_VEC * sigma_v
        v, wt = doppler.velocity_grid(T_doppler, dv=3.0, cutoff_sigma=3.5)
        kv = K_VEC * v
        N = vapor_cell.cold_spot_limited_density(
            species.rb85_natural_density(T_density), T_density, T_doppler)

        off_span = float(np.abs(offsets).max())
        half = max(3.5 * dopp_fwhm, off_span + 0.4 * dopp_fwhm, 10 * gamma_eff)
        scan = np.linspace(-half, half, int(params["scan_points"]))

        two_lvl = atoms.two_level(gamma=radiative_gamma).with_optical_dephasing(elastic_hwhm)
        Hpr = np.zeros((2, 2), dtype=complex)
        Hpr[0, 1] = Hpr[1, 0] = PROBE_RABI * GAMMA / 2
        L0_probe = core.build_liouvillian(Hpr, two_lvl)
        kvmax = float(np.abs(kv).max())
        flo, fhi = scan.min() - kvmax - off_span, scan.max() + kvmax + off_span
        fine = np.linspace(flo, fhi, int((fhi - flo) / (gamma_eff / 20)) + 2)
        rho_pr = core.steady_state_batched(L0_probe, fine, two_lvl.S_v, 2)
        chi_pr = rho_pr[:, 1, 0] / (PROBE_RABI * GAMMA)
        # Both quadratures of one weak-probe susceptibility: α = k·Im χ is the
        # absorption table, Re χ its dispersive partner on the same fine axis.
        alpha_L, chi_L = observables.absorption_coefficient(chi_pr, K_VEC, N)
        chi_real_L = np.real(chi_L)

        ns, nv = scan.size, v.size
        if pump_off:
            rho_gg = 1.0
            excited_pops = (0.0,) * n_exc
        else:
            # Pump H₀ is scan-independent. Solve one effective-detuning table.
            dlo, dhi = scan.min() + kv.min(), scan.max() + kv.max()
            n_grid = int((dhi - dlo) / (gamma_eff / 8.0)) + 2
            deff_p = np.linspace(dlo, dhi, n_grid)
            pops = _pump_pops(
                L0_pump, deff_p, atom.S_v, atom.n_levels)
            dgrid = (scan[:, None] + kv[None, :]).ravel()
            rho_gg = np.interp(
                dgrid, deff_p, pops[:, 0]).reshape(ns, nv)
            excited_pops = tuple(
                np.interp(dgrid, deff_p, pops[:, e]).reshape(ns, nv)
                for e in atom.excited
            )
        probe_base = scan[:, None] - kv[None, :]
        alpha = np.zeros(ns)
        chi_real = np.zeros(ns)
        for i, pe in enumerate(excited_pops):
            probe_arg = (probe_base - offsets[i]).ravel()
            aL = np.interp(probe_arg, fine, alpha_L).reshape(ns, nv)
            xL = np.interp(probe_arg, fine, chi_real_L).reshape(ns, nv)
            alpha += ((rho_gg - pe) * aL * wt[None, :]).sum(axis=1)
            chi_real += ((rho_gg - pe) * xL * wt[None, :]).sum(axis=1)

        result = dict(mode="generic", scan=scan + effects.shift_rad_s, alpha_unit=alpha,
                    chi_real_unit=chi_real, k_vec=float(K_VEC),
                    dopp_fwhm=dopp_fwhm, buffer_gamma=buffer_gamma,
                    density_temp_c=T_density - 273.15,
                    doppler_temp_c=T_doppler - 273.15,
                    doppler_temp_tied=tied,
                    pump_saturation_intensity_w_m2={
                        "Rb85": species.pump_saturation_intensity(species.RB85, "D1")},
                    gamma_eff=gamma_eff, offsets=offsets + effects.shift_rad_s, two=(n_exc == 2))
        if reference:
            result["optical_collisions"] = {"Rb85": effects.readout()}
        return result

    # =================================================================
    # observables  (dispatch)
    # =================================================================
    def observables(self, raw, params, include_figures=True):
        alpha, chi_real, L = _collected_response(raw, params)
        if raw["mode"] == "species":
            return self._obs_species(
                raw, params, alpha, chi_real, L, include_figures=include_figures)
        return self._obs_generic(
            raw, params, alpha, chi_real, L, include_figures=include_figures)

    def _obs_species(self, raw, params, alpha, chi_real, L, include_figures=True):
        x = raw["scan"] / (2 * np.pi) / 1e9                  # GHz (relative)
        finite_probe = raw.get("probe_model") in (*TWO_LEVEL_MODELS, *RATE_MODELS)
        if "ballistic_depth" in raw or "transit_rates_s" in raw:
            finite_probe = raw["probe_power_mw"] > 0
        optical_readout = _optical_readout(
            x, "GHz", alpha, L, params, raw, effective=finite_probe)
        T_trans = optical_readout["transmission"]
        if "ballistic_depth" in raw:
            optical_readout["ballistic_diagnostics"] = raw["ballistic_diagnostics"]
        if "transit_rates_s" in raw:
            optical_readout["transit_rates_s"] = raw["transit_rates_s"]
        if "optical_collisions" in raw:
            optical_readout["optical_collisions"] = raw["optical_collisions"]
            optical_readout["self_broadening_reference"] = raw.get("self_broadening_reference", {})
        buffer_mhz = raw.get("buffer_gamma", 0.0) / (2 * np.pi) / 1e6
        pump = params["pump_power_mw"]
        pump_on = pump > 0
        coating = raw.get("paraffin_population_memory")
        if coating is not None:
            optical_readout["paraffin_population_memory"] = coating
        axial = sas_axial.readout(raw, params, x, T_trans) if "axial_cell" in raw else None
        if axial is not None:
            optical_readout["axial_cell"] = axial
            if axial["status"] != "exact_uniform_limit":
                optical_readout["attenuation_scope"] = (
                    "Axially integrated weak-probe Beer-Lambert intensity attenuation.")
                if axial["status"] == "refinement_limited":
                    optical_readout["attenuation_scope"] += " Axial refinement limit reached."

        fig = None
        figure_views = []
        if include_figures:
            import matplotlib.pyplot as plt

            title = ""
            if coating is not None:
                title = ("Paraffin: population memory only; reference T1 = 25.1 ms"
                         if coating["status"] == "active"
                         else "Paraffin: population memory inactive with pump off")
            xlabel = "Relative frequency  [GHz]  (ref: line centroid)"
            marker_x = [gx for gx, _lbl in raw["markers"]]
            fig, axT = plt.subplots(1, 1, figsize=(8.5, 4.35))
            axT.plot(x, T_trans, color="#0284C7", lw=1.3)
            axT.set_ylabel("Transmission")
            axT.set_xlabel(xlabel)
            axT.set_title(title)
            attenuation_views = _attenuation_figure_views(
                optical_readout, "Relative frequency (ref: line centroid)  [GHz]",
                title, linewidth=1.3)
            attenuation_axes = [item["figure"].axes[0]
                                for item in attenuation_views]
            for gx in marker_x:
                for ax in (axT, *attenuation_axes):
                    ax.axvline(gx, color="gray", ls=":", lw=0.5, alpha=0.6)
            fig.tight_layout()
            for item in attenuation_views:
                item["figure"].tight_layout()
            figure_views = [
                {"label": "Transmission", "figure": fig},
                *attenuation_views,
            ]
            if chi_real is not None:
                figure_views.append({
                    "label": "Dispersion",
                    "figure": _dispersion_figure(
                        x, chi_real, xlabel, title, markers=marker_x),
                })

        feature = subdoppler_feature(x, T_trans) if pump_on else None
        optical_readout["resolution"] = _feature_resolution(feature, 1000.0, "whole_spectrum")
        axial_limited = axial is not None and axial["status"] == "refinement_limited"
        if pump_on and axial_limited:
            optical_readout["resolution"].update(
                status="axial-refinement-limited", reason="Axial convergence tolerance was exceeded.",
                width_mhz=None)
        peak_metric, depth_metric = _attenuation_peak_metrics(optical_readout)
        broad_metric = _gaussian_doppler_metric(raw["dopp_fwhm"], raw)
        if pump_on:
            status_metric, feature_metrics = _subdoppler_metrics(
                feature, mhz_per_x=1000.0)
            if axial_limited:
                metrics = [dict(label="SAS resolution", value="axial refinement limited",
                                kind="status", tier="hero")]
            elif feature.resolved:
                metrics = _lock_readout_metrics(
                    x, T_trans, mhz_per_x=1000.0,
                    feature_at=feature.center, feature_fwhm=feature.fwhm)
                metrics[0]["tier"] = "hero"
                metrics.extend(feature_metrics)
                metrics.append(status_metric)
            else:
                status_metric["tier"] = "hero"
                metrics = [status_metric, *feature_metrics]
            metrics.extend([peak_metric, depth_metric, broad_metric])
        else:
            peak_metric["tier"] = "hero"
            metrics = [peak_metric, depth_metric, broad_metric]
        metrics.extend(_dispersion_metrics(chi_real, raw.get("k_vec"), L))
        metrics.extend(_probe_saturation_metrics(raw, params, L))
        metrics.extend(_temperature_metrics(raw))
        self_broadening = raw.get("self_broadening")
        if self_broadening:
            optical_readout["self_broadening"] = self_broadening
            if self_broadening["status"] == "opt-in reference":
                fwhm = self_broadening["applied_fwhm_rad_s"] / species.MHZ
                metric = dict(label="Self-broadening",
                              value=f"{fwhm:.4g} MHz (high uncertainty, ±{self_broadening['relative_uncertainty']:.2%})",
                              kind="status", help=self_broadening["warning"] +
                              " Added optical FWHM at the cell-center density.")
            else:
                metric = dict(label="Self-broadening", value=f"Unsupported for Cs {raw['line']}",
                              kind="status", help="The Cs–Cs width contribution is omitted. The natural width "
                              "and configured buffer-gas broadening still apply. " + self_broadening["reason"])
            metrics.append(metric)
        if raw.get("rb_composition") == RB_COMPOSITION_CUSTOM:
            fraction = float(raw["rb85_fraction"])
            metrics.append(dict(
                label="Rb isotope composition",
                value=f"{fraction * 100:.2f}% ⁸⁵Rb / {(1.0 - fraction) * 100:.2f}% ⁸⁷Rb",
                help="Shares of total elemental Rb vapor density; the frequency "
                     "axis uses the ⁸⁵Rb line centroid for every mixture."))
        metrics.extend(_collision_metrics(raw))
        if params.get("ne_pressure_torr", 0.0) != 0.0:
            metrics.append(dict(
                label="Buffer Gas Broadening", value=f"{buffer_mhz:.1f} MHz"))
        rows = "".join(f"| {lbl} | {gx*1e3:.1f} |\n" for gx, lbl in raw["markers"])
        table = ("Hyperfine transitions (Lamb-dip centres); crossovers appear at the "
                 "midpoint of any two sharing a ground state, enhanced/inverted by "
                 "hyperfine pumping.\n\n| Transition | Center [MHz] |\n|---|---|\n" + rows)
        tables = [{"title": "Hyperfine lines", "markdown": table}]
        if coating is not None:
            rate = coating["assumed_cycle_rate_s_1"]
            tables.append(derived_table([
                ("Paraffin model", coating["model"] + " (" + coating["status"] + ")"),
                ("Population lifetime reference", (
                    f"{1e3 * coating['t1_reference_s']:.1f} ms for coated 87Rb at "
                    f"{coating['t1_reference_temperature_k']:.0f} K "
                    f"(doi:{coating['t1_reference_doi']})")),
                ("Beam exchange cycle rate", (
                    f"{rate:.3g} s^-1 (configured transit rate)"
                    if rate is not None else "inactive with pump off")),
                ("Calibration scope", coating["calibration_status"]),
                ("Coating throughput correction", "none"),
            ]))
        if axial is not None:
            metrics.append(dict(label="Axial reference", value=axial["status"].replace("_", " "),
                                kind="status", help="Each refinement allows 1% in normalized spectra and "
                                "pump output, and 0.01 in absolute probe transmission."))
            details = [("Axial segments", str(axial["segment_count"])),
                       ("Pump depletion", "on" if axial["pump_depletion"] else "off"),
                       ("Minimum pump output / input", (
                           f"{axial['minimum_pump_output_fraction']:.3f}"
                           if axial["pump_depletion"] else None)),
                       ("Temperature profile", axial["temperature_source"]),
                       ("Field profile", axial["field_source"]),
                       ("Magnetic model", axial["field_model"])]
            for key, label in (("uniform", "Uniform"), ("undepleted", "Undepleted")):
                delta = axial.get(f"difference_from_{key}")
                if delta is not None:
                    details.append((f"{label} comparison: maximum transmission difference",
                                    f"{delta['transmission_absolute_max']:.4g}"))
                    if delta["matched_subdoppler_contrast_change"] is not None:
                        details.append((f"{label} comparison: sub-Doppler contrast change",
                                        f"{delta['matched_subdoppler_contrast_change']:+.4g}"))
                    if key == "undepleted":
                        dips = delta["matched_lamb_dips"]
                        if dips:
                            largest = max(dips, key=lambda dip: abs(dip["depth_change"]))
                            details.append((f"Lamb-dip depth change ({largest['transition']})",
                                            f"{largest['depth_change']:+.4g}"))
                        else:
                            details.append(("Lamb-dip depth change", "No matched resolved line-centered dip"))
            tables.append(derived_table(details, title="Segmented cell reference"))
        profile = raw.get("weak_probe_profile")
        if profile is not None:
            optical_readout["weak_probe_profile"] = profile
            metrics.append(_weak_probe_profile_metric(profile))
        if "ballistic_depth" in raw:
            from ..od_model import MODELS
            model = MODELS[raw["probe_model"]]
            diagnostics = raw["ballistic_diagnostics"]
            note = model.accuracy
            if model.model_id == FIELD_HISTORY_REFERENCE:
                bx, by, bz, azimuth, ellipticity = (params.get(key, 0.0) for key in FIELD_HISTORY_KEYS)
                note += (f" Applied B = ({bx:g}, {by:g}, {bz:g}) µT in the probe frame (+z along propagation), "
                         f"azimuth {azimuth:g}° and ellipticity {ellipticity:g}°. "
                         "Transition markers identify zero-field hyperfine centers.")
            if reason := diagnostics.get("fallback_reason"):
                note += " " + str(reason)
            optical_readout["model_accuracy"] = note
            metrics.append(dict(label="Probe model scope", value=model.description, help=note, kind="status"))
            tables.append({"title": "Beam-crossing history", "markdown": note})
        elif raw.get("probe_model") in RATE_MODELS:
            from ..od_model import MODELS
            model_id = raw["probe_model"]
            accuracy = MODELS[model_id].accuracy
            tables.append({
                "title": "Finite-probe model",
                "markdown": (
                    "Magnetic-sublevel rate model for a linearly polarized probe at zero "
                    f"field (`{model_id}`): π links between (F, |m|) "
                    "populations, spontaneous decay with all polarizations, transit "
                    "reset, self-consistent longitudinal depletion and full Gaussian "
                    f"power averaging. {accuracy} The beam waist is held "
                    "constant through the cell and the detector collects the full output "
                    "beam. Dispersion is unavailable for the nonlinear response."
                ),
            })
        elif finite_probe:
            _, _, qualifier = _probe_scope_text(raw, params)
            tables.append({
                "title": "Finite-probe model",
                "markdown": (
                    "Independent-transition two-level power broadening, "
                    "self-consistent longitudinal depletion, and full Gaussian "
                    f"power averaging (`{raw['probe_model']}`). "
                    f"{qualifier} The beam waist is held constant through the cell "
                    "and the detector collects the full output beam; diffraction, "
                    "clipping, and transverse atomic transport are omitted. "
                    "Dispersion is unavailable because this model does not define "
                    "a collected phase for the spatially varying "
                    "nonlinear susceptibility."
                ),
            })
        return dict(
            metrics=metrics,
            hero_count=1,
            figure=fig,
            figure_views=figure_views,
            optical_readout=optical_readout,
            tables=tables,
            comparison={
                "axis_index": 0,
                "x_unit": "GHz",
                "raw_x_unit": "Arb. unit",
                "raw_y_unit": "Arb. unit",
                "label": "Experimental CSV",
            },
        )

    def _obs_generic(self, raw, params, alpha, chi_real, L, include_figures=True):
        x = raw["scan"] / (2 * np.pi) / 1e6                  # MHz
        optical_readout = _optical_readout(x, "MHz", alpha, L, params, raw)
        T_trans = optical_readout["transmission"]
        if "optical_collisions" in raw:
            optical_readout["optical_collisions"] = raw["optical_collisions"]
            optical_readout["self_broadening_reference"] = raw.get("self_broadening_reference", {})
        offs_mhz = raw["offsets"] / (2 * np.pi) / 1e6
        buffer_mhz = raw.get("buffer_gamma", 0.0) / (2 * np.pi) / 1e6

        fig = None
        figure_views = []
        if include_figures:
            import matplotlib.pyplot as plt

            title = ""
            xlabel = "Probe detuning  [MHz]"
            fig, axT = plt.subplots(1, 1, figsize=(8.5, 4.35))
            axT.plot(x, T_trans, color="#0284C7", lw=1.6)
            axT.set_ylabel("Transmission")
            axT.set_xlabel(xlabel)
            axT.set_title(title)
            attenuation_views = _attenuation_figure_views(
                optical_readout, xlabel, title, linewidth=1.6)
            attenuation_axes = [item["figure"].axes[0]
                                for item in attenuation_views]
            for ax in (axT, *attenuation_axes):
                for off in offs_mhz:
                    ax.axvline(off, color="gray", ls=":", lw=0.7)
                if raw["two"]:
                    ax.axvline(0.0, color="green", ls=":", lw=0.7)
            fig.tight_layout()
            for item in attenuation_views:
                item["figure"].tight_layout()
            figure_views = [
                {"label": "Transmission", "figure": fig},
                *attenuation_views,
            ]
            if chi_real is not None:
                figure_views.append({
                    "label": "Dispersion",
                    "figure": _dispersion_figure(
                        x, chi_real, xlabel, title,
                        markers=list(offs_mhz) + ([0.0] if raw["two"] else [])),
                })

        pump_on = params["pump_power_mw"] > 0
        gamma_mhz = raw["gamma_eff"] / (2 * np.pi) / 1e6
        selected_line = float(offs_mhz[0])
        feature_window = (
            selected_line - 10.0 * gamma_mhz,
            selected_line + 10.0 * gamma_mhz,
        )
        feature = (
            subdoppler_feature(x, T_trans, search_window=feature_window)
            if pump_on else None
        )
        optical_readout["resolution"] = _feature_resolution(feature, 1.0, "model_feature_window")
        peak_metric, depth_metric = _attenuation_peak_metrics(optical_readout)
        broad_metric = _gaussian_doppler_metric(raw["dopp_fwhm"], raw)
        if pump_on:
            status_metric, feature_metrics = _subdoppler_metrics(
                feature, mhz_per_x=1.0)
            if feature.resolved:
                metrics = _lock_readout_metrics(
                    x, T_trans, mhz_per_x=1.0,
                    feature_at=feature.center, feature_fwhm=feature.fwhm)
                metrics[0]["tier"] = "hero"
                metrics.extend(feature_metrics)
                metrics.append(status_metric)
            else:
                status_metric["tier"] = "hero"
                metrics = [status_metric, *feature_metrics]
            metrics.extend([peak_metric, depth_metric, broad_metric])
        else:
            peak_metric["tier"] = "hero"
            metrics = [peak_metric, depth_metric, broad_metric]
        metrics.extend(_dispersion_metrics(chi_real, raw.get("k_vec"), L))
        metrics.extend(_temperature_metrics(raw))
        metrics.extend(_collision_metrics(raw))
        if params.get("ne_pressure_torr", 0.0) != 0.0:
            metrics.append(dict(
                label="Buffer Gas Broadening", value=f"{buffer_mhz:.1f} MHz"))
        note = ("Two transitions: Lamb dips at ±splitting/2 and a **crossover** dip "
                "at the midpoint (green)." if raw["two"]
                else "Single transition: one Lamb dip at line centre.")
        profile = raw.get("weak_probe_profile")
        if profile is not None:
            optical_readout["weak_probe_profile"] = profile
            metrics.append(_weak_probe_profile_metric(profile))
        return dict(
            metrics=metrics,
            hero_count=1,
            figure=fig,
            figure_views=figure_views,
            optical_readout=optical_readout,
            tables=[{"title": "Notes", "markdown": note}],
            comparison={
                "axis_index": 0,
                "x_unit": "MHz",
                "raw_x_unit": "Arb. unit",
                "raw_y_unit": "Arb. unit",
                "label": "Experimental CSV",
            },
        )


# =====================================================================
# helpers
# =====================================================================
def _measured_collision_mode(params):
    return params.get("optical_collision_model") == collisions.REFERENCE_COLLISIONS


def _weak_probe_profile_metric(profile):
    return dict(label="Weak-probe line profile", value=profile["applied"], kind="status",
                help=profile["scope"])


def _collision_metrics(raw):
    metrics = []
    for isotope, report in raw.get("optical_collisions", {}).items():
        metrics.append(dict(
            label=f"{isotope} Ne optical reference", kind="status", value=report["status"],
            help=(f"Line shift {report['shift_mhz']:+.3f} MHz. "
                  f"Added homogeneous FWHM {report['fwhm_mhz']:.3f} MHz. "
                  + " ".join(report["warnings"]))))
    return metrics


def _collected_response(raw, params):
    """The same transmission input for display and export refinement."""
    L = params["cell_mm"] * 1e-3
    alpha = raw["alpha_unit"] * params["line_strength"]
    chi_real_unit = raw.get("chi_real_unit")
    chi_real = None if chi_real_unit is None else chi_real_unit * params["line_strength"]
    alpha_table = raw.get("probe_alpha_table_unit") if raw["mode"] == "species" else None
    if "ballistic_depth" in raw and raw["probe_power_mw"] > 0:
        return (raw["ballistic_depth"] / L if L > 0 else alpha), None, L
    if alpha_table is not None:
        # The nonlinear model gives collected power but no unique phase index.
        _transmission, alpha = probe_saturation.propagate_gaussian_spectrum(
            raw["probe_intensity_log_grid"], alpha_table * params["line_strength"], alpha, L,
            radial_order=raw.get("probe_radial_order", probe_saturation.RADIAL_QUADRATURE_ORDER))
        chi_real = None
    return alpha, chi_real, L


def _quantitative_check(refined, grid, mhz_per_x):
    def frequency(key):
        value = refined[key]
        return float(value * mhz_per_x) if np.isfinite(value) else None

    check = dict(
        status=refined["status"], reason=refined["reason"],
        points=int(grid.size), grid_step_mhz=float((grid[1] - grid[0]) * mhz_per_x),
        center_mhz=frequency("center"), width_mhz=frequency("width"),
        left_half_height_mhz=frequency("left"), right_half_height_mhz=frequency("right"),
        samples_per_fwhm=refined["samples_per_fwhm"],
    )
    for key in ("polarity", "local_noise_proxy"):
        if key in refined:
            check[key] = refined[key]
    if "flanks" in refined:
        check["flanks"] = [dict(detuning_mhz=flank["detuning"] * mhz_per_x,
                                slope_per_mhz=flank["slope"] / mhz_per_x)
                            for flank in refined["flanks"]]
    return check


def _finish_quantitative_export(evidence):
    first, second = evidence["checks"]
    if not all(check["status"] == "resolved" for check in (first, second)):
        return dict(evidence, reason="A refined grid did not resolve the fixed feature and both flanks.")
    width = second["width_mhz"]
    width_error = abs(first["width_mhz"] - width) / width
    center_error = abs(first["center_mhz"] - second["center_mhz"]) / width
    coarse_error = abs(evidence["coarse_center_mhz"] - second["center_mhz"]) / width
    overlap = (max(first["left_half_height_mhz"], second["left_half_height_mhz"])
               < min(first["right_half_height_mhz"], second["right_half_height_mhz"]))
    slope_errors = [abs(a["slope_per_mhz"] - b["slope_per_mhz"]) / abs(b["slope_per_mhz"])
                    for a, b in zip(first["flanks"], second["flanks"])]
    evidence["comparison"] = dict(width_relative_error=width_error,
                                  center_shift_fwhm=center_error,
                                  coarse_center_shift_fwhm=coarse_error,
                                  half_height_regions_overlap=overlap,
                                  flank_slope_relative_errors=slope_errors)
    if (not overlap or center_error > evidence["center_shift_tolerance_fwhm"]
            or coarse_error > evidence["coarse_center_tolerance_fwhm"]
            or width_error > evidence["width_relative_tolerance"]
            or max(slope_errors) > evidence["slope_relative_tolerance"]):
        return dict(evidence, reason="Refined feature identity, linewidth or flank slopes did not converge.")
    flank = max(second["flanks"], key=lambda item: abs(item["slope_per_mhz"]))
    return dict(evidence, status="converged", linewidth_mhz=width,
                lock_slope_per_mhz=abs(flank["slope_per_mhz"]),
                lock_detuning_mhz=flank["detuning_mhz"],
                reason="Independent grids agree on the fixed feature and both derivative flanks.")


def _probe_atom_count(raw, params, length_m):
    """Gaussian intensity-weighted atom count for the same cell and probe."""
    if raw["mode"] != "species":
        return None  # The generic toy has no physical probe waist.
    if length_m == 0.0:
        return 0.0
    if axial := raw.get("axial_cell"):
        segment_densities = sum(axial["density_m3"].values())
        density = np.average(segment_densities, weights=np.diff(axial["z_edges_m"]))
    elif raw.get("isotope_fractions"):
        cold_k, path_k, _ = _temperatures(params)
        density = sum(_species_path_density(species.ISOTOPES[name], cold_k, path_k, fraction)
                      for name, fraction in raw["isotope_fractions"].items())
    else:
        return None
    return effective_atom_number(
        density, length_m=length_m,
        probe_radius_m=float(params.get("probe_waist_mm", 1.0)) * 1e-3).atoms


def _optical_readout(x, x_unit, alpha, L, params, raw, *, effective=False):
    """Typed attenuation values shared by plots, headless readout and export."""
    depth = observables.optical_depth(alpha, L)
    # For a finite probe this is the weak-input coefficient already used in
    # the propagation table, evaluated with the same pump and model.
    od0 = (observables.optical_depth(raw["alpha_unit"] * params["line_strength"], L)
           if effective else depth)
    return dict(
        schema="gabes-optical-attenuation-v1",
        x=x,
        x_unit=x_unit,
        transmission=observables.transmission(alpha, L),
        decadic_absorbance=observables.decadic_absorbance(alpha, L),
        optical_depth=depth, od0=od0, n_eff=_probe_atom_count(raw, params, L),
        cell_length_m=float(L),
        line_strength=float(params["line_strength"]),
        effective=bool(effective),
        attenuation_scope=(
            "Collected Gaussian power transmission; alpha_eff = -ln(T) / L."
            if effective else "Uniform Beer-Lambert intensity attenuation."),
        conventions={
            "transmission": "T = exp(-tau)",
            "decadic_absorbance": "A10 = -log10(T) = tau / ln(10)",
            "optical_depth": "tau = -ln(T) = alpha_eff * L",
        },
        unit="dimensionless",
    )


def _attenuation_peak_metrics(readout):
    prefix = "Effective peak" if readout["effective"] else "Peak"
    scope = readout["attenuation_scope"]
    return (
        dict(label=f"{prefix} absorbance A10",
             value=f"{np.nanmax(readout['decadic_absorbance']):.2f}",
             delta="A₁₀ = −log₁₀(T)",
             help=f"Decadic absorbance A10 = −log₁₀(T) = τ / ln(10), dimensionless. {scope}"),
        dict(label=f"{prefix} optical depth tau",
             value=f"{np.nanmax(readout['optical_depth']):.2f}",
             delta="τ = −ln(T)",
             help=f"Optical depth τ = −ln(T) = α_eff L, dimensionless. {scope}"),
    )


def _attenuation_figure_views(readout, xlabel, title, *, linewidth):
    import matplotlib.pyplot as plt

    views = []
    quantities = (
        ("decadic_absorbance", "Decadic absorbance A10", "#F43F5E",
         "Decadic absorbance A10 = -log10(T)"),
        ("optical_depth", "Optical depth tau", "#9333EA",
         "Optical depth tau = -ln(T)"),
    )
    for quantity, label, color, y_label in quantities:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.35))
        curve, = ax.plot(readout["x"], readout[quantity], color=color, lw=linewidth)
        curve._gabes_export_metadata = {
            "quantity": quantity,
            "y_label": y_label,
            "y_unit": "dimensionless",
            "definition": readout["conventions"][quantity],
            "effective": readout["effective"],
        }
        ax.set_ylabel(f"{y_label}  [dimensionless]")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        views.append({"label": label, "figure": fig})
    return views


def _pump_detuning_axis(omega, gamma_eff, covered_detunings=None):
    """Fine pump-state axis, optionally covering every requested detuning.

    The legacy path uses the historical ±14Γ extent. The Maxwell correction
    and long-lived coated reservoirs require the full scan+velocity domain
    because a clamped edge retains nonthermal populations.
    """
    omega = np.asarray(omega, dtype=float)
    gamma_eff = float(gamma_eff)
    de_lo = float(omega.min() - 14.0 * gamma_eff)
    de_hi = float(omega.max() + 14.0 * gamma_eff)
    if covered_detunings is not None:
        covered_lo, covered_hi = map(float, covered_detunings)
        de_lo = min(de_lo, covered_lo)
        de_hi = max(de_hi, covered_hi)
    count = int((de_hi - de_lo) / (gamma_eff / 8.0)) + 2
    return np.linspace(de_lo, de_hi, count)


def _velocity_correlated_average(prepared_population, probe_profile, weights):
    """Average population×probe response in the same velocity class."""
    return (prepared_population * probe_profile) @ weights


def _velocity_sums(detuning, kv, weights, omega, hwhm, g_pos, e_pos, p_ground,
                   axis=None, tables=None, reservoir=None, thermal_needed=False,
                   *, same_direction=False, population_spline=None,
                   population_delta=False):
    """NumPy path of `kernels.sas_velocity_sums`: same arguments, same sums."""
    probe_base = (detuning[:, None] + kv[None, :] if same_direction
                  else detuning[:, None] - kv[None, :])
    prepared = None
    if tables is not None:
        flat_deff = (detuning[:, None] + kv[None, :]).ravel()   # pump Δ_eff
        if population_spline is not None:
            values = population_spline(flat_deff).reshape(
                tables.shape[:2] + probe_base.shape)
            if reservoir is None:
                prepared = values[0]
            else:
                # Same source weighting as the compiled kernel.
                prepared = np.zeros(values.shape[1:])
                for source in range(values.shape[0]):
                    prepared += reservoir[:, source, None] * values[source]
        elif reservoir is None:
            prepared = [np.interp(flat_deff, axis, table).reshape(probe_base.shape)
                        for table in tables[0]]
        else:
            prepared = []
            for level in range(tables.shape[1]):
                local = np.zeros_like(probe_base)
                for source in range(tables.shape[0]):
                    conditioned = np.interp(
                        flat_deff, axis, tables[source, level]
                    ).reshape(probe_base.shape)
                    local += reservoir[:, source, None] * conditioned
                prepared.append(local)
    absorptive = np.empty((omega.size, detuning.size))
    dispersive = np.empty_like(absorptive)
    # Zero when unused, as the compiled kernel returns it.
    thermal = absorptive if prepared is None else np.zeros_like(absorptive)
    for t in range(omega.size):
        arg = probe_base - omega[t]
        denom = arg ** 2 + hwhm ** 2
        Lp = (hwhm / np.pi) / denom                          # unit-area Lorentzian
        Dp = (arg / np.pi) / denom                           # KK dispersive partner
        if prepared is None:                                 # pump off: weight 1
            absorptive[t] = Lp @ weights
            dispersive[t] = Dp @ weights
            continue
        w = (prepared[g_pos[t]] - prepared[e_pos[t]]) / p_ground[t]
        if population_delta:
            w -= 1.0
        absorptive[t] = _velocity_correlated_average(w, Lp, weights)
        dispersive[t] = _velocity_correlated_average(w, Dp, weights)
        if thermal_needed:
            thermal[t] = Lp @ weights
    return absorptive, dispersive, thermal


def _chunked_pumped_velocity_sums(detuning, kv, *args, reservoir=None, **kwargs):
    """Bound the Automatic pumped NumPy fallback's scan × velocity arrays."""
    rows = max(1, 100_000 // kv.size)
    parts = [_velocity_sums(detuning[i:i + rows], kv, *args, **kwargs,
                            reservoir=None if reservoir is None else reservoir[i:i + rows])
             for i in range(0, detuning.size, rows)]
    return tuple(np.concatenate(columns, axis=1) for columns in zip(*parts))


def _basis_reset_atom(atom, transit_rate, source_ground):
    """Clone a species manifold atom with transit reload into one ground F.

    The basis-reset steady states span the response to any incoherent incoming
    ground-hyperfine distribution. The uncoated path does not call this helper.
    """
    return atom.with_reset(
        float(transit_rate),
        [1.0 if g == source_ground else 0.0 for g in atom.ground],
        name=f"{atom.name}_reset_g{source_ground}")


def _basis_reset_pump_pops(man, pump_hamiltonian, deff_axis, transit_rate, *, pump_atom=None):
    """Pump populations conditioned on each incoming ground hyperfine state."""
    pump_atom = man.atom if pump_atom is None else pump_atom
    conditioned = []
    for source_ground in pump_atom.ground:
        atom = _basis_reset_atom(pump_atom, transit_rate, source_ground)
        L0 = core.build_liouvillian(pump_hamiltonian, atom)
        conditioned.append(
            _pump_pops(L0, deff_axis, atom.S_v, atom.n_levels)
        )
    return np.asarray(conditioned)


def _ground_decay_branching(man):
    """Return B[g,e], including only natural excited-state decay channels."""
    grounds = tuple(man.atom.ground)
    excited = tuple(man.atom.excited)
    g_pos = {level: i for i, level in enumerate(grounds)}
    e_pos = {level: i for i, level in enumerate(excited)}
    branching = np.zeros((len(grounds), len(excited)))
    for source, target, rate in man.atom.decay:
        if source in e_pos and target in g_pos:
            branching[g_pos[target], e_pos[source]] += float(rate)
    totals = branching.sum(axis=0)
    if np.any(totals <= 0.0):
        raise ValueError("every excited hyperfine state needs a ground decay path")
    return branching / totals[None, :]


def _coated_ground_transfer(man, basis_pops, deff_axis, detuning, kv, weights):
    """Velocity-averaged ground-F transfer matrix for one pump scan.

    ``transfer[scan, destination, source]`` includes spontaneous decay of any
    excited population after the atom exits the illuminated region, for the pump
    detuning ``detuning[scan] + kv[velocity]``. Velocity is averaged only for
    this dark reservoir map; the absorption calculation keeps each pump/probe
    velocity correlation intact.
    """
    grounds = np.asarray(man.atom.ground, dtype=int)
    excited = np.asarray(man.atom.excited, dtype=int)
    branching = _ground_decay_branching(man)
    exit_pops = basis_pops[:, :, grounds].copy()
    exit_pops += np.einsum(
        "sxe,ge->sxg", basis_pops[:, :, excited], branching)

    n_ground = grounds.size
    if kernels.available():
        # tables[source·n + destination] = exit_pops[source, :, destination]
        means = kernels.velocity_interp_mean(
            detuning, kv, weights, deff_axis,
            exit_pops.transpose(0, 2, 1).reshape(n_ground * n_ground, -1))
        transfer = means.reshape(n_ground, n_ground, -1).transpose(2, 1, 0).copy()
    else:
        deff_grid = detuning[:, None] + kv[None, :]
        transfer = np.empty((detuning.size, n_ground, n_ground))
        flat_deff = deff_grid.ravel()
        for source in range(n_ground):
            for destination in range(n_ground):
                local = np.interp(
                    flat_deff, deff_axis, exit_pops[source, :, destination]
                ).reshape(deff_grid.shape)
                transfer[:, destination, source] = local @ weights

    # Interpolation/linear-solve roundoff can make a probability about -1e-16.
    transfer = np.maximum(transfer, 0.0)
    column_sum = transfer.sum(axis=1, keepdims=True)
    if np.any(column_sum <= 0.0):
        raise ValueError("coated-cell ground transfer lost probability")
    return transfer / column_sum


def _stationary_ground_populations(
        transfer, thermal_populations, cycle_rate,
        wall_t1=PARAFFIN_REFERENCE_T1_S):
    """Solve the quasi-static coated-cell ground-population reservoir.

    The existing transit rate closes the one-zone model as both bright-region
    exit and return cadence. Wall relaxation drives the reservoir toward the
    thermal hyperfine populations at 1/T1 (not 2π/T1).
    """
    transfer = np.asarray(transfer, dtype=float)
    thermal = np.asarray(thermal_populations, dtype=float)
    n_ground = thermal.size
    identity = np.eye(n_ground)
    wall_rate = 1.0 / float(wall_t1)
    A = (float(cycle_rate) * (identity[None, :, :] - transfer)
         + wall_rate * identity[None, :, :])
    rhs = np.broadcast_to(wall_rate * thermal, transfer.shape[:1] + (n_ground,))
    populations = np.linalg.solve(A, rhs[..., None])[..., 0]
    populations = np.maximum(populations, 0.0)
    totals = populations.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("coated-cell reservoir has no ground population")
    return populations / totals


def _coated_ground_populations(
        man, basis_pops, deff_axis, detuning, kv, weights, cycle_rate):
    transfer = _coated_ground_transfer(
        man, basis_pops, deff_axis, detuning, kv, weights)
    return _stationary_ground_populations(
        transfer, man.p_ground, cycle_rate=cycle_rate)


def _dispersion_figure(x, chi_real, xlabel, title, markers=()):
    """Refractive index over the scan (ASCII axis strings — mathtext lock)."""
    import matplotlib.pyplot as plt

    index_ppm = (observables.refractive_index(chi_real) - 1.0) * 1e6
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.35))
    ax.plot(x, index_ppm, color="#7C3AED", lw=1.3)
    ax.axhline(0.0, color="gray", ls="--", lw=0.6, alpha=0.7)
    ax.set_ylabel("Refractive index  n - 1  [ppm]")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for gx in markers:
        ax.axvline(gx, color="gray", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def _dispersion_metrics(chi_real, k_vec, L):
    """Single-pass phase readout from the dispersive quadrature (or nothing)."""
    if chi_real is None or k_vec is None:
        return []
    phase = observables.single_pass_phase(chi_real, float(k_vec), float(L))
    peak = float(np.nanmax(np.abs(phase)))
    value = f"{peak:.2f} rad" if peak >= 1.0 else f"{peak * 1e3:.1f} mrad"
    return [dict(
        label="Peak phase shift",
        value=value,
        help="Largest single-pass |φ| = k·(n−1)·L from the dispersive (real) "
             "part of the same susceptibility that gives the absorption, with "
             "n = 1 + Re χ/2. A phase quantity — not a group delay. It is the "
             "medium's phase alone: a deeply absorbed spectral region can carry "
             "a large phase with almost no light left to measure it.",
    )]


def _probe_saturation_metrics(raw, params, length_m):
    """Finite-power model provenance and its strongest local saturation scale."""
    if "transit_rates_s" in raw:
        rates = ", ".join(f"{name}: {rate / (2 * np.pi * 1e3):.2f} kHz"
                          for name, rate in raw["transit_rates_s"].items())
        return [dict(label="Transit γₜ/2π", value=rates, kind="status",
                     help="Experimental Gaussian-beam estimate, per isotope, using the kinetic temperature "
                          "and the intensity FWHM diameter (Häupl et al., Eq. 8).")]
    if raw.get("probe_model") in RATE_MODELS:
        from ..od_model import RATE_MODEL_ACCURACY
        return [dict(label="Probe model", value="Magnetic-sublevel rate", kind="status",
                     help=RATE_MODEL_ACCURACY)]
    if raw.get("probe_model") not in TWO_LEVEL_MODELS:
        return []
    saturation = float(raw["probe_saturation_max"])
    intensity = float(raw["probe_peak_intensity"])
    scope_value, scope_help, _ = _probe_scope_text(raw, params)
    metrics = [dict(
        label="Peak probe saturation",
        value=f"{saturation:.2f}",
        help=("Strongest-transition on-axis resonant s₀ before propagation; "
              f"Gaussian peak intensity {intensity:.3g} W/m². The selected "
              "line's ν and Γ set I_sat = πhcΓ/(3λ³). Hyperfine transitions use "
              "the AutoOD-style s_t = s₀·C_F²/C_F,max² scale and local intensity "
              "depletion."),
    ), dict(
        label="Probe model scope",
        value=scope_value,
        help=scope_help,
    )]
    if raw.get("probe_passivity_limited", False):
        residual_scale = float(raw.get("probe_residual_scale_min", 0.0))
        metrics.append(dict(
            label="Finite-probe approximation",
            value="Passivity-limited",
            help=("The separately broadened frozen-pump residual would predict "
                  "negative absorption at this setting. It was multiplied, at "
                  "each local intensity, by a near-largest passive scale with a "
                  "small interior margin; minimum "
                  f"scale η = {residual_scale:.3f}. Treat simultaneous finite "
                  "pump + finite probe quantitatively only with a joint model."),
        ))

    length_ratio = probe_saturation.cell_to_rayleigh_ratio(
        length_m, params.get("probe_waist_mm", 1.0), raw["k_vec"])
    if length_ratio >= 0.5:
        metrics.append(dict(
            label="Probe beam geometry",
            value=f"L/z_R = {length_ratio:.2f}",
            help=("The finite-probe model holds the waist constant through the "
                  "cell and assumes complete output-beam collection. This cell is "
                  "at least half a Rayleigh range long, so diffraction, clipping, "
                  "and transverse atomic transport may matter."),
        ))
    return metrics


def _temperature_metrics(raw):
    """Report the Doppler temperature only when it is released from the cell."""
    if raw.get("doppler_temp_tied", True):
        return []
    return [dict(
        label="Doppler-width temperature",
        value=f"{float(raw['doppler_temp_c']):.0f} °C",
        help="Released from the cell temperature: it sets the thermal velocity "
             "distribution, and the beam-path density is the ideal gas at the "
             f"cold-spot vapor pressure of {float(raw['density_temp_c']):.0f} °C, "
             "n_sat(T_cold)·T_cold/T_Doppler.",
    )]


def _temperature_title(raw, params):
    """Plot-title fragment naming whichever temperatures are in play."""
    density_c = float(raw.get("density_temp_c", params["temp_c"]))
    if raw.get("doppler_temp_tied", True):
        return f"T = {density_c:.0f} °C"
    return (f"T_density = {density_c:.0f} °C, "
            f"T_Doppler = {float(raw['doppler_temp_c']):.0f} °C")


def _gaussian_doppler_metric(dopp_fwhm, raw=None):
    width_mhz = float(dopp_fwhm) / (2 * np.pi) / 1e6
    source = "the cell temperature"
    if raw is not None and not raw.get("doppler_temp_tied", True):
        source = ("the released Doppler-width temperature "
                  f"({float(raw['doppler_temp_c']):.0f} °C)")
    return dict(
        label="Gaussian Doppler FWHM",
        value=f"{width_mhz:.1f} MHz",
        help=f"Calculated thermal Gaussian FWHM for one optical line, from "
             f"{source}. It is not a measured Voigt or multi-line envelope FWHM.",
    )


def _feature_resolution(feature, mhz_per_x, scope):
    def measured(value, scale=1.0):
        return float(value * scale) if np.isfinite(value) else None

    return {
        "status": feature.status if feature else "not applicable",
        "scope": scope,
        "reason": feature.reason if feature else "Pump off; no sub-Doppler linewidth.",
        "center_mhz": measured(feature.center, mhz_per_x) if feature else None,
        "width_mhz": measured(feature.fwhm, mhz_per_x) if feature else None,
        "samples_per_fwhm": measured(feature.samples_per_fwhm) if feature else None,
        "polarity": feature.polarity if feature else None,
    }


def _subdoppler_metrics(feature, mhz_per_x):
    """User-facing resolution diagnostics for a :class:`SubdopplerFeature`."""
    status = dict(
        label="SAS resolution",
        value=feature.status,
        kind="status",
        help=feature.reason,
    )
    if not feature.detected:
        return status, []

    scale = float(mhz_per_x)
    metrics = [
        dict(
            label="Sub-Doppler FWHM",
            value=f"{feature.fwhm * scale:.2f} MHz",
            delta=(f"{feature.samples_per_fwhm:.1f} samples/FWHM · "
                   f"minimum {MIN_SUBDOPPLER_SAMPLES_PER_FWHM:g}"),
            status=status,
            help=f"Interpolated residual half-height width near "
                 f"{feature.center * scale:+.1f} MHz; report the linewidth only "
                 "when the SAS resolution status is resolved.",
        ),
        dict(
            label="Half-height edges",
            value=(f"{feature.left_half_height * scale:+.2f} to "
                   f"{feature.right_half_height * scale:+.2f} MHz"),
            help="Linear interpolation between the samples bracketing each "
                 "half-height crossing.",
        ),
        dict(
            label="Samples / FWHM",
            value=f"{feature.samples_per_fwhm:.1f}",
            help="Interpolated FWHM divided by the local median sample spacing; "
                 f"at least {MIN_SUBDOPPLER_SAMPLES_PER_FWHM:g} is required for resolved status.",
        ),
        dict(
            label="Scan-edge distance",
            value=f"{feature.scan_edge_distance * scale:.1f} MHz",
            help="Nearest distance from either interpolated half-height edge to "
                 "the displayed scan boundary; at least one FWHM is required.",
        ),
    ]
    return status, metrics


def _lock_readout_metrics(x, T_trans, mhz_per_x,
                          feature_at=None, feature_fwhm=None,
                          search_window=None):
    """Local finite-slope lock proxy for a *resolved* sub-Doppler feature.

    There is deliberately no full-spectrum fallback: a Doppler-envelope flank
    is not a sub-Doppler laser-lock discriminator.
    """
    if len(x) < 2:
        return []
    slope_per_x = np.gradient(T_trans, x)
    candidates = np.isfinite(slope_per_x)
    if search_window is None:
        valid_feature = (
            feature_at is not None
            and feature_fwhm is not None
            and np.isfinite(feature_at)
            and np.isfinite(feature_fwhm)
            and float(feature_fwhm) > 0.0
        )
        if not valid_feature:
            return []
        lo = float(feature_at) - float(feature_fwhm)
        hi = float(feature_at) + float(feature_fwhm)
    else:
        lo, hi = map(float, search_window)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return []
    local = (np.asarray(x) >= lo) & (np.asarray(x) <= hi)
    if np.any(candidates & local):
        candidates &= local
    else:
        return []
    if not candidates.any():
        return []
    indices = np.flatnonzero(candidates)
    i = int(indices[np.argmax(np.abs(slope_per_x[indices]))])
    slope_per_mhz = abs(float(slope_per_x[i])) / float(mhz_per_x)
    detuning_mhz = float(x[i]) * float(mhz_per_x)
    return [
        dict(label="Lock Slope", value=f"{slope_per_mhz:.4f} /MHz",
             help="Largest |dT/dΔ| within the resolved sub-Doppler feature "
                  "window; proxy for a laser-lock discriminator."),
        dict(label="Lock Detuning", value=f"{detuning_mhz:+.1f} MHz",
             help="Detuning where the lock-slope proxy is largest."),
    ]


