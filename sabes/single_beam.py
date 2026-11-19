"""Selected pre-cell SABES light as an isolated single-beam calculation input.

This adapter does not turn the FWM output into a self-action prediction. It
selects one source-chain line and declares every other optical drive absent.
The source Gaussian is defined at the nominal cell centre, not its entrance.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

from gabes import species
from gabes.optical_frequency import RbOpticalFrequency
from gabes.psr_conventions import jones_polarization_angles

from .session_export import json_value


@dataclass(frozen=True)
class SingleBeamInput:
    """Physical entrance data; infinity denotes a planar wavefront in memory.

    ``radius_m`` is the entrance 1/e^2 intensity radius, not the source waist.
    A scalar field uses ``frequency.wavelength_m`` with this radius and the
    signed wavefront curvature. Positive curvature is a diverging input.
    ``excluded_power_w`` counts other lines in the selected arm; the other
    arm's independently excluded power is recorded separately.
    """
    frequency: RbOpticalFrequency
    power_w: float
    radius_m: float
    curvature_radius_m: float
    source_label: str
    reference_plane: str
    excluded_power_w: float
    source_m2: float
    arm: str
    spectral_offset_hz: float
    cell_length_m: float
    cell_centre_radius_m: float
    cell_centre_curvature_radius_m: float
    gaussian_geometry_wavelength_m: float
    source_jones: tuple[complex, complex]
    source_azimuth_rad: float
    excluded_other_arm_power_w: float

    def as_dict(self):
        """Strict JSON metadata without changing the physical infinity values."""
        values = asdict(self)
        values["frequency"] = self.frequency.as_dict()
        values["source_jones"] = [dict(real=v.real, imag=v.imag)
                                  for v in self.source_jones]
        values["planar_wavefront"] = math.isinf(self.curvature_radius_m)
        values["cell_centre_planar_wavefront"] = math.isinf(
            self.cell_centre_curvature_radius_m)
        values["source_reference_plane"] = "nominal cell centre; actual waist position unmeasured"
        values["polarization_scope"] = "linear polarization rotated to x at zero magnetic field"
        values["scope"] = (
            "isolated selected pre-cell line; other optical drives absent; "
            "no FWM output or coupled-beam prediction")
        return json_value(values)


def _spectral_power(beam):
    for line in beam.lines:
        if (not math.isfinite(line.offset_hz)
                or not math.isfinite(line.power_w) or line.power_w < 0.):
            raise ValueError("source lines require finite offsets and nonnegative finite powers")
    return math.fsum(line.power_w for line in beam.lines)


def _curvature_radius(mode):
    inverse_radius = (1. / mode.q).real
    return math.inf if inverse_radius == 0. else 1. / inverse_radius


def cell_beam_input(chain, arm, cell_length_m, *, offset_hz=None):
    """Select one pre-cell pump/seed line and map its nominal mode to entrance.

    The default is the pump carrier or the wanted seed sideband. An explicit
    offset selects one existing line within the source's 1 Hz lookup precision;
    its actual stored frequency and power are retained. Selecting a total
    spectrum or a generated probe/conjugate arm is unsupported.

    Existing Gaussian geometry defines the entrance radius and curvature by
    propagation through -L/2. Its compatibility wavelength is retained only as
    metadata; the new scalar field uses the selected physical optical frequency.
    M^2=1 and linear polarization are required by the current zero-field model.
    """
    if arm not in ("pump", "seed"):
        raise ValueError("single-beam input requires a pre-cell pump or seed arm")
    length = float(cell_length_m)
    if not math.isfinite(length) or length <= 0.:
        raise ValueError("cell_length_m must be finite and positive")
    if not isinstance(chain.carrier_frequency, RbOpticalFrequency):
        raise ValueError("source chain requires a structured physical carrier frequency")
    beam = getattr(chain, arm)
    other = chain.seed if arm == "pump" else chain.pump
    _spectral_power(beam)
    other_power = _spectral_power(other)
    target = (0. if arm == "pump" else chain.seed_offset_hz) if offset_hz is None else float(offset_hz)
    if not math.isfinite(target):
        raise ValueError("spectral offset must be finite")
    selected = [line for line in beam.lines if abs(line.offset_hz-target) <= 1.]
    if len(selected) != 1:
        raise ValueError("select exactly one existing spectral line; a multifrequency total is unsupported")
    line = selected[0]
    if line.power_w <= 0.:
        raise ValueError("selected spectral line must have positive power")

    mode = beam.mode
    if (not all(math.isfinite(v) for v in
                (mode.waist_m, mode.wavelength_m, mode.z_m, mode.m2))
            or mode.waist_m <= 0. or mode.wavelength_m <= 0.):
        raise ValueError("source Gaussian geometry must be finite with positive radius and wavelength")
    if not math.isclose(mode.m2, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("single-beam scalar propagation requires source M^2=1")
    azimuth, ellipticity = jones_polarization_angles(beam.jones)
    if abs(ellipticity) > 1e-12:
        raise ValueError("single-beam response supports linear polarization only at zero magnetic field")

    entrance = mode.propagated(-0.5*length)
    radius, curvature = entrance.radius_m, _curvature_radius(entrance)
    if not math.isfinite(radius) or radius <= 0.:
        raise ValueError("source Gaussian does not give a finite positive entrance radius")
    return SingleBeamInput(
        frequency=chain.carrier_frequency.shifted_hz(line.offset_hz),
        power_w=float(line.power_w), radius_m=float(radius),
        curvature_radius_m=float(curvature),
        source_label=f"{arm}: {line.label or 'selected line'}", reference_plane="cell entrance",
        excluded_power_w=math.fsum(item.power_w for item in beam.lines if item is not line),
        source_m2=float(mode.m2), arm=arm, spectral_offset_hz=float(line.offset_hz),
        cell_length_m=length, cell_centre_radius_m=float(mode.radius_m),
        cell_centre_curvature_radius_m=float(_curvature_radius(mode)),
        gaussian_geometry_wavelength_m=float(mode.wavelength_m),
        source_jones=tuple(complex(v) for v in beam.jones),
        source_azimuth_rad=azimuth, excluded_other_arm_power_w=other_power)


@dataclass(frozen=True)
class FrequencyReference:
    """One named detuning coordinate, independent of isotope composition."""
    key: str
    label: str
    isotope: str
    line: str
    reference: str
    ground_f: int | None = None
    excited_f: int | None = None

    def _options(self):
        return dict(reference=self.reference, ground_f=self.ground_f,
                    excited_f=self.excited_f)

    def frequency(self, detuning_mhz):
        """Create a physical frequency from laser-minus-reference detuning."""
        return RbOpticalFrequency.from_detuning(
            self.isotope, self.line, float(detuning_mhz)*1e6, **self._options())

    def detuning_mhz(self, frequency):
        """Display an existing same-band frequency without changing it."""
        if not isinstance(frequency, RbOpticalFrequency):
            raise TypeError("frequency must be an RbOpticalFrequency")
        if frequency.line != self.line:
            raise ValueError("a detuning reference must use the frequency's D line")
        return frequency.detuning_mhz(self.isotope, **self._options())

    def as_dict(self):
        return asdict(self)


@lru_cache(maxsize=2)
def frequency_references(line):
    """Immutable Rb85/Rb87 fine/component/allowed-transition reference choices."""
    if line not in ("D1", "D2"):
        raise ValueError("line must be D1 or D2")
    choices = []
    for isotope, display in (("Rb85", "\u2078\u2075Rb"), ("Rb87", "\u2078\u2077Rb")):
        iso = species.ISOTOPES[isotope]
        levels = species.level_structure(isotope, line)
        ground = sorted({f for f, _ in levels.ground})
        excited = sorted({f for f, _ in levels.excited})
        prefix = f"{isotope.lower()}_{line.lower()}"
        labels = (
            ("fine_centroid", "fine-structure centroid"),
            ("upper_ground_centroid", f"upper ground F={max(ground)} centroid"),
            ("lower_ground_centroid", f"lower ground F={min(ground)} centroid"),
        )
        for reference, label in labels:
            choices.append(FrequencyReference(
                f"{prefix}_{reference}", f"{display} {line} {label}", isotope, line, reference))
        for fg in ground:
            for fe in excited:
                if species.line_strength(fg, fe, iso.I, iso.Jg, iso.line(line)[0]) <= 0.:
                    continue
                choices.append(FrequencyReference(
                    f"{prefix}_f{fg}_fp{fe}",
                    f"{display} {line} F={fg} \u2192 F\u2032={fe}",
                    isotope, line, "hyperfine_transition", fg, fe))
    return tuple(choices)
