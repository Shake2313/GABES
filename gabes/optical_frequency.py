"""Rb D-line frequencies and elemental-density composition in one contract.

A frequency is stored as a fixed Rb85 fine-centroid reference plus an offset.
Changing isotope fractions never changes that frequency origin. Named atomic
references are coordinate conversions, not different optical frequencies.
Use offsets for detuning arithmetic: forming and subtracting optical absolute
frequencies loses sub-Hz precision. This module assigns no validity range to an
atomic response model; selecting a D line is not a claim of arbitrary-detuning
or simultaneous-D1/D2 physics support.
"""
from dataclasses import dataclass
import math

from . import constants, magneto_multilevel, species


REFERENCE_CHOICES = (
    "fine_centroid", "upper_ground_centroid", "lower_ground_centroid",
    "hyperfine_transition",
)


def _isotope_and_line(isotope, line):
    if isotope not in ("Rb85", "Rb87"):
        raise ValueError("isotope must be Rb85 or Rb87")
    if line not in ("D1", "D2"):
        raise ValueError("line must be D1 or D2")
    return species.ISOTOPES[isotope]


def reference_offset_hz(isotope, line, *, reference="fine_centroid",
                        ground_f=None, excited_f=None):
    """Named atomic reference minus this band's fixed Rb85 fine centroid.

    Ground-component centroids reuse the shared line-strength centroid. Exact
    hyperfine transitions reuse the shared level structure and reject forbidden
    dipole transitions. The output is an offset in ordinary Hz, not rad/s.
    """
    iso = _isotope_and_line(isotope, line)
    if reference not in REFERENCE_CHOICES:
        raise ValueError(f"reference must be one of {REFERENCE_CHOICES}")
    isotope_shift = iso.line(line)[1]-species.RB85.line(line)[1]
    if reference == "fine_centroid":
        if ground_f is not None or excited_f is not None:
            raise ValueError("fine_centroid does not accept hyperfine level selectors")
        return float(isotope_shift)
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    ground_levels = tuple(sorted({f for f, _ in atom.ground}))
    if reference in ("upper_ground_centroid", "lower_ground_centroid"):
        if ground_f is not None or excited_f is not None:
            raise ValueError("upper/lower centroid already selects its ground level")
        selected = max(ground_levels) if reference.startswith("upper") else min(ground_levels)
        return float(isotope_shift + 1e6*magneto_multilevel.component_centroid_mhz(atom, selected))
    if (isinstance(ground_f, bool) or isinstance(excited_f, bool)
            or ground_f not in ground_levels):
        raise ValueError("hyperfine_transition requires valid ground_f and excited_f")
    excited_levels = tuple(sorted({f for f, _ in atom.excited}))
    if excited_f not in excited_levels:
        raise ValueError("hyperfine_transition requires valid ground_f and excited_f")
    if species.line_strength(ground_f, excited_f, iso.I, iso.Jg, iso.line(line)[0]) <= 0:
        raise ValueError("selected hyperfine transition is dipole-forbidden")
    ground_energy = atom.ground_energy_mhz[ground_levels.index(ground_f)]
    excited_energy = atom.excited_energy_mhz[next(
        i for i, (f, _) in enumerate(atom.excited) if f == excited_f)]
    return float(isotope_shift + 1e6*(excited_energy-ground_energy))


@dataclass(frozen=True)
class RbOpticalFrequency:
    """A physical carrier as Rb85 fine-centroid Hz plus a stable offset Hz."""
    line: str
    offset_hz: float

    def __post_init__(self):
        _isotope_and_line("Rb85", self.line)
        offset = float(self.offset_hz)
        if not math.isfinite(offset) or self.reference_hz+offset <= 0:
            raise ValueError("offset_hz must be finite and give a positive physical frequency")
        object.__setattr__(self, "offset_hz", offset)

    @property
    def reference_hz(self):
        return float(species.RB85.line(self.line)[1])

    @property
    def absolute_hz(self):
        """Optical Hz for presentation/wavevector; use offsets for differences."""
        return self.reference_hz+self.offset_hz

    @property
    def wavelength_m(self):
        return constants.C_LIGHT/self.absolute_hz

    @classmethod
    def from_detuning(cls, isotope, line, detuning_hz, *, reference="fine_centroid",
                      ground_f=None, excited_f=None):
        detuning = float(detuning_hz)
        if not math.isfinite(detuning):
            raise ValueError("detuning_hz must be finite")
        origin = reference_offset_hz(isotope, line, reference=reference,
                                     ground_f=ground_f, excited_f=excited_f)
        return cls(line, origin+detuning)

    @classmethod
    def from_absolute_hz(cls, frequency_hz, line):
        """Import measured absolute Hz; cannot recover precision already lost."""
        _isotope_and_line("Rb85", line)
        value = float(frequency_hz)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        return cls(line, value-species.RB85.line(line)[1])

    def shifted_hz(self, shift_hz):
        """Move a carrier/sideband by ordinary Hz without optical subtraction."""
        shift = float(shift_hz)
        if not math.isfinite(shift):
            raise ValueError("shift_hz must be finite")
        return type(self)(self.line, self.offset_hz+shift)

    def detuning_hz(self, isotope, *, reference="fine_centroid", ground_f=None,
                    excited_f=None):
        origin = reference_offset_hz(isotope, self.line, reference=reference,
                                     ground_f=ground_f, excited_f=excited_f)
        return self.offset_hz-origin

    def detuning_mhz(self, isotope, **reference_options):
        return self.detuning_hz(isotope, **reference_options)/1e6

    def as_dict(self):
        return dict(line=self.line, reference_isotope="Rb85", reference_kind="fine_centroid",
                    reference_hz=self.reference_hz, offset_hz=self.offset_hz,
                    absolute_hz=self.absolute_hz, wavelength_m=self.wavelength_m,
                    detuning_sign="laser minus named atomic reference")


@dataclass(frozen=True)
class RbComposition:
    """Complementary isotope shares of ONE total elemental Rb number density.

    No abundance multiplies a pure-isotope response a second time. Zero-share
    isotopes are omitted, and isotope fraction does not select a frequency zero.
    The existing SAS parser remains unchanged; this contract follows its custom
    composition rule without changing the legacy natural/pure UI paths.
    """
    rb85_fraction: float

    def __post_init__(self):
        fraction = float(self.rb85_fraction)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("rb85_fraction must be finite and between 0 and 1")
        object.__setattr__(self, "rb85_fraction", fraction)

    @classmethod
    def natural(cls):
        return cls(species.RB85.abundance)

    @property
    def components(self):
        return tuple((name, fraction) for name, fraction in (
            ("Rb85", self.rb85_fraction), ("Rb87", 1-self.rb85_fraction)) if fraction > 0)

    def as_dict(self):
        return dict(rb85_fraction=self.rb85_fraction, rb87_fraction=1-self.rb85_fraction,
                    density_basis="shares of total elemental Rb number density",
                    frequency_reference="fixed Rb85 fine centroid of selected D line")
