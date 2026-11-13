"""Pressure and density coefficients for the optical D-line reference.

Coefficients below are linear-frequency slopes. Pressure is Ne partial
pressure in Torr at the operating condition. A negative shift moves the
atomic resonance to lower frequency: delta_eff = delta_laser - shift.
"""
from dataclasses import dataclass
import math

from . import constants

LEGACY_COLLISIONS = "Legacy broadening"
REFERENCE_COLLISIONS = "Measured Ne reference"
COLLISION_CHOICES = (LEGACY_COLLISIONS, REFERENCE_COLLISIONS)
MAX_DEFAULT_RELATIVE_UNCERTAINTY = 0.20
_MHZ = 2 * math.pi * 1e6
_SARGSYAN = "https://doi.org/10.1134/S0030400X2108018X"
_OTTINGER = "https://doi.org/10.1103/PhysRevA.11.1815"
_ROTONDARO = "https://doi.org/10.1016/S0022-4073(96)00147-1"
_HA = "https://doi.org/10.1103/PhysRevA.103.022826"


@dataclass(frozen=True)
class Measurement:
    value: float
    uncertainty: float
    source: str
    locator: str


@dataclass(frozen=True)
class Coefficient:
    primary: Measurement
    comparisons: tuple[Measurement, ...] = ()

    @property
    def uncertainty(self):
        """Conservative spread bound about the declared reference slope."""
        return max(math.hypot(item.uncertainty, item.value - self.primary.value)
                   for item in (self.primary, *self.comparisons))

    @property
    def relative_uncertainty(self):
        return self.uncertainty / abs(self.primary.value)


@dataclass(frozen=True)
class PressureCoefficients:
    fwhm: Coefficient
    shift: Coefficient
    temperature_k: float
    pressure_torr: tuple[float, float]
    regime: str


def _rb_ne(line, *, rb87=False):
    d1 = line == "D1"
    width_comparisons = (
        Measurement(9.84 if d1 else 9.47, 0.10, _ROTONDARO, "Ne row"),
        Measurement(9.4 if d1 else 9.8, 1.3 if d1 else 1.7, _OTTINGER, "Ne line cores"),
    )
    if rb87 and d1:
        width_comparisons += (Measurement(12.2, 0.97, _HA, "Table I, 337 K"),)
    return PressureCoefficients(
        Coefficient(
            Measurement(10.0, 1.0, _SARGSYAN, "Figs. 3-6"),
            width_comparisons),
        Coefficient(
            Measurement(-1.1 if d1 else -2.1, 0.2, _SARGSYAN, "Figs. 3-6"),
            (Measurement(-0.90 if d1 else -2.44, 0.02, _ROTONDARO, "Ne row"),
             Measurement(-0.37 if d1 else -2.0, 0.3, _OTTINGER, "Ne line cores"))),
        393.15, (0.0, 20.0), "Half-wave nanocell, 120 C reservoir, nominal 0/6/20 Torr",
    )


_CS_D1 = "https://doi.org/10.1103/PhysRevA.80.062718"
_CS_D2 = "https://doi.org/10.1103/PhysRevA.82.042502"
_CS_BERNABEU = "https://doi.org/10.1103/PhysRevA.22.2690"
_CS_INOUE = "https://doi.org/10.1143/JPSJ.59.516"
PRESSURE_COEFFICIENTS = {
    ("Ne", "Rb85", "D1"): _rb_ne("D1"),
    ("Ne", "Rb87", "D1"): _rb_ne("D1", rb87=True),
    ("Ne", "Rb85", "D2"): _rb_ne("D2"),
    ("Ne", "Rb87", "D2"): _rb_ne("D2", rb87=True),
    ("Ne", "Cs133", "D1"): PressureCoefficients(
        Coefficient(Measurement(10.85, 0.02, _CS_D1, "Table II, 313 K"),
                    (Measurement(10.6, 0.41, _HA, "Table I, 320 K"),
                     Measurement(10.13, 0.86, _CS_BERNABEU, "295 K, compared in Pitz Table II"))),
        Coefficient(Measurement(-1.60, 0.01, _CS_D1, "Table III, 313 K"),
                    (Measurement(-2.88, 0.09, _CS_BERNABEU, "295 K, compared in Pitz Table III"),)),
        313.0, (0.0, 300.0), "Weak-probe line core in macroscopic absorption cell"),
    ("Ne", "Cs133", "D2"): PressureCoefficients(
        Coefficient(Measurement(9.81, 0.06, _CS_D2, "Table II, 313 K"),
                    (Measurement(12.39, 0.21, _CS_INOUE,
                                 "293 K, compared in Pitz Table II"),
                     Measurement(10.34, 1.05, _CS_BERNABEU, "295 K, compared in Pitz Table II"))),
        Coefficient(Measurement(-2.58, 0.01, _CS_D2, "Table III, 313 K"),
                    (Measurement(-2.59, 0.07, _CS_INOUE,
                                 "293 K, compared in Pitz Table III"),)),
        313.0, (0.0, 300.0), "Weak-probe line core in macroscopic absorption cell"),
}


_SELF_RB_D1 = Coefficient(Measurement(0.69e-13, 0.04e-13,
    "https://doi.org/10.1088/0953-4075/44/19/195006", "Sec. IV, up to 3e20 m^-3"))
_SELF_RB_D2 = Coefficient(Measurement(1.10e-13, 0.17e-13,
    "https://doi.org/10.1103/PhysRevA.73.062504", "Sec. V.B, 3.8e20-3.7e21 m^-3"))
SELF_BROADENING_COEFFICIENTS = {
    ("Rb85", "D1"): _SELF_RB_D1, ("Rb87", "D1"): _SELF_RB_D1,
    ("Rb85", "D2"): _SELF_RB_D2, ("Rb87", "D2"): _SELF_RB_D2,
}


@dataclass(frozen=True)
class PressureEffects:
    fwhm_rad_s: float
    shift_rad_s: float
    pressure_torr: float
    status: str
    coefficients: PressureCoefficients | None = None
    warnings: tuple[str, ...] = ()

    @property
    def optical_dephasing_rad_s(self):
        return self.fwhm_rad_s / 2.0

    def readout(self):
        """JSON-friendly optical provenance, with linear MHz for displayed rates."""
        payload = dict(status=self.status, pressure_torr=self.pressure_torr,
                       fwhm_mhz=self.fwhm_rad_s / _MHZ, shift_mhz=self.shift_rad_s / _MHZ,
                       optical_dephasing_mhz=self.optical_dephasing_rad_s / _MHZ,
                       warnings=list(self.warnings))
        if self.coefficients is not None:
            c = self.coefficients
            payload.update(reference_temperature_k=c.temperature_k, reference_pressure_torr=c.pressure_torr,
                           reference_regime=c.regime)
            for key in ("fwhm", "shift"):
                slope = getattr(c, key)
                payload[key + "_uncertainty_mhz"] = self.pressure_torr * slope.uncertainty
                payload[key + "_sources"] = [
                    dict(source=m.source, locator=m.locator, value_mhz_per_torr=m.value,
                         uncertainty_mhz_per_torr=m.uncertainty)
                    for m in (slope.primary, *slope.comparisons)]
        return payload


def pressure_effects(isotope, line, pressure_torr, temperature_k, *,
                     gas="Ne", mode=REFERENCE_COLLISIONS):
    """Return separate elastic optical width and resonance shift in rad/s."""
    p, temperature = float(pressure_torr), float(temperature_k)
    if not math.isfinite(p) or p < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Ne pressure must be finite and nonnegative, temperature finite and positive")
    if mode not in COLLISION_CHOICES:
        raise ValueError("unknown optical collision model")
    if p == 0:
        return PressureEffects(0.0, 0.0, p, "zero buffer pressure")
    if mode == LEGACY_COLLISIONS:
        if gas != "Ne":
            return PressureEffects(0.0, 0.0, p, "unsupported")
        return PressureEffects(constants.neon_buffer_broadening(p), 0.0, p,
                               "legacy fixed-Ne broadening")
    c = PRESSURE_COEFFICIENTS.get((gas, isotope, line))
    if c is None:
        return PressureEffects(0.0, 0.0, p, "unsupported",
                               warnings=("No matching gas/isotope/line coefficient was applied.",))
    warnings = []
    for label, coefficient in (("Broadening", c.fwhm), ("Shift", c.shift)):
        relative = coefficient.relative_uncertainty
        if relative > MAX_DEFAULT_RELATIVE_UNCERTAINTY:
            warnings.append(f"{label} coefficient uncertainty {relative:.0%} exceeds 20%.")
    if not c.pressure_torr[0] <= p <= c.pressure_torr[1]:
        warnings.append("Pressure outside the reference measurements.")
    if abs(temperature - c.temperature_k) > 0.5:
        warnings.append("Reference slope held fixed across temperature. Temperature transfer is unvalidated.")
    return PressureEffects(p * c.fwhm.primary.value * _MHZ, p * c.shift.primary.value * _MHZ,
                           p, "opt-in reference", c, tuple(warnings))


def self_broadening_reference(isotope, line, density_m3):
    """Measured density-linear FWHM [rad/s], with explicit support status."""
    density = float(density_m3)
    if not math.isfinite(density) or density < 0:
        raise ValueError("self-broadening density must be finite and nonnegative")
    coefficient = SELF_BROADENING_COEFFICIENTS.get((isotope, line))
    if coefficient is None:
        return 0.0, dict(status="unsupported", reason="No adopted self-broadening coefficient.")
    m = coefficient.primary
    fwhm = 2 * math.pi * m.value * density
    return fwhm, dict(status="reference coefficient", source=m.source, locator=m.locator,
                      density_m3=density, fwhm_rad_s=fwhm,
                      uncertainty_rad_s=2 * math.pi * coefficient.uncertainty * density)
