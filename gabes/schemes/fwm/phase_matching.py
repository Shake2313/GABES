"""Seeded and biphoton phase matching and temporal overlap."""
from .. import fwm
from .common import FieldSpec, NM, SEEDED_PHASE_ANGLE_DEG, SIDE_MINUS, SIDE_PLUS
from ...constants import OMEGA_HF
from ... import constants
import math
import numpy as np


def _wavevector_nm(wavelength_nm):
    return 2 * np.pi / (float(wavelength_nm) * NM)


def _side_sign(value):
    if isinstance(value, str):
        return 1.0 if value.strip() == SIDE_PLUS else -1.0
    return 1.0 if float(value) >= 0.0 else -1.0


def _side_label(value):
    return SIDE_PLUS if float(value) >= 0.0 else SIDE_MINUS


def transverse_matched_angle_deg(source_wavelength_nm, target_wavelength_nm,
                                 source_angle_deg):
    """Collection angle that cancels transverse k for two generated photons."""
    source_k = _wavevector_nm(source_wavelength_nm)
    target_k = _wavevector_nm(target_wavelength_nm)
    x = source_k / target_k * math.sin(math.radians(float(source_angle_deg)))
    return math.degrees(math.asin(float(np.clip(x, -1.0, 1.0))))


def _field_with(field, *, wavelength_nm=None, detuning_mhz=None, rabi_mhz=None,
                angle_deg=None, side_sign=None):
    return FieldSpec(
        role=field.role,
        lower=field.lower,
        upper=field.upper,
        wavelength_nm=field.wavelength_nm if wavelength_nm is None else wavelength_nm,
        detuning_mhz=field.detuning_mhz if detuning_mhz is None else detuning_mhz,
        rabi_mhz=field.rabi_mhz if rabi_mhz is None else rabi_mhz,
        phase_sign=field.phase_sign,
        direction=field.direction,
        angle_deg=field.angle_deg if angle_deg is None else angle_deg,
        side_sign=field.side_sign if side_sign is None else side_sign,
    )


def phase_mismatch(fields, *, signal_angle_deg=None, idler_angle_deg=None,
                   reference_delta_k=0.0):
    """Longitudinal four-field phase mismatch, with an optional reference offset."""
    total = 0.0
    for field in fields:
        angle = field.angle_deg
        if field.role == "signal" and signal_angle_deg is not None:
            angle = signal_angle_deg
        if field.role == "idler" and idler_angle_deg is not None:
            angle = idler_angle_deg
        total += field.phase_sign * field.k * math.cos(math.radians(angle))
    return total - (reference_delta_k or 0.0)


def phase_mismatch_vector(fields, *, signal_angle_deg=None, idler_angle_deg=None,
                          signal_side_sign=None, idler_side_sign=None,
                          reference_delta_k=0.0):
    """Biphoton vector mismatch: calibrated longitudinal, absolute transverse."""
    delta_k_z_absolute = 0.0
    delta_k_x = 0.0
    for field in fields:
        angle = field.angle_deg
        side = field.side_sign
        if field.role == "signal":
            if signal_angle_deg is not None:
                angle = signal_angle_deg
            if signal_side_sign is not None:
                side = signal_side_sign
        elif field.role == "idler":
            if idler_angle_deg is not None:
                angle = idler_angle_deg
            if idler_side_sign is not None:
                side = idler_side_sign
        angle_rad = math.radians(float(angle))
        delta_k_z_absolute += field.phase_sign * field.k * math.cos(angle_rad)
        delta_k_x += field.phase_sign * field.k * math.sin(angle_rad) * side
    delta_k_z_relative = delta_k_z_absolute - (reference_delta_k or 0.0)
    delta_k_vector = math.hypot(delta_k_z_relative, delta_k_x)
    return {
        "delta_k_z_relative": delta_k_z_relative,
        "delta_k_z_absolute": delta_k_z_absolute,
        "delta_k_x": delta_k_x,
        "delta_k_vector": delta_k_vector,
    }


def phase_matching_weight(delta_k, L):
    """sinc^2(delta_k L / 2), normalized to 1 at perfect phase matching."""
    x = 0.5 * np.asarray(delta_k, dtype=float) * L
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-12
    out[mask] = (np.sin(x[mask]) / x[mask]) ** 2
    return out


def _sinc_complex(x):
    """sinc(x) = sin(x)/x for complex argument (→1 at x→0). The longitudinal
    detuning function Φ = sinc(ρ̄)·e^{iρ̄} (Du et al. Eq. 14, Chen et al. Eq. 3)
    carries a complex ρ̄ when in-cell loss/dispersion (OD) is included."""
    x = np.asarray(x, dtype=complex)
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-9
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def _bandwidth_from_waveform_mhz(tau_s, psi):
    """Spectral FWHM [MHz] of a biphoton temporal waveform ψ(τ) via its FFT."""
    n = np.asarray(tau_s).size
    if n < 4:
        return float("nan")
    dt = float(tau_s[1] - tau_s[0])
    nfft = 4 * n
    spec = np.fft.fftshift(np.fft.fft(np.asarray(psi), n=nfft))
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, dt))     # Hz
    power = np.abs(spec)**2
    if power.max() <= 0:
        return float("nan")
    above = np.where(power >= 0.5 * power.max())[0]
    if above.size < 2:
        return float("nan")
    return float((freq[above[-1]] - freq[above[0]]) / 1e6)


def phase_mismatch_grid(fields, signal_axis_deg, idler_axis_deg,
                        reference_delta_k=0.0):
    """Vectorized signal/idler longitudinal mismatch grid."""
    signal_axis_deg = np.asarray(signal_axis_deg, dtype=float)
    idler_axis_deg = np.asarray(idler_axis_deg, dtype=float)
    sig, ide = np.meshgrid(signal_axis_deg, idler_axis_deg, indexing="ij")
    total = np.zeros_like(sig, dtype=float)
    for field in fields:
        if field.role == "signal":
            total += field.phase_sign * field.k * np.cos(np.deg2rad(sig))
        elif field.role == "idler":
            total += field.phase_sign * field.k * np.cos(np.deg2rad(ide))
        else:
            total += field.phase_sign * field.k * math.cos(math.radians(field.angle_deg))
    return total - (reference_delta_k or 0.0)


def phase_mismatch_vector_grid(fields, signal_axis_deg, idler_axis_deg,
                               reference_delta_k=0.0, *, precise_norm=False):
    """Vectorized signal/idler mismatch magnitude grid for biphoton collection."""
    signal_axis_deg = np.asarray(signal_axis_deg, dtype=float)
    idler_axis_deg = np.asarray(idler_axis_deg, dtype=float)
    sig, ide = np.meshgrid(signal_axis_deg, idler_axis_deg, indexing="ij")
    delta_k_z_absolute = np.zeros_like(sig, dtype=float)
    delta_k_x = np.zeros_like(sig, dtype=float)
    for field in fields:
        if field.role == "signal":
            angle_rad = np.deg2rad(sig)
            delta_k_z_absolute += field.phase_sign * field.k * np.cos(angle_rad)
            delta_k_x += (field.phase_sign * field.k * np.sin(angle_rad)
                          * field.side_sign)
        elif field.role == "idler":
            angle_rad = np.deg2rad(ide)
            delta_k_z_absolute += field.phase_sign * field.k * np.cos(angle_rad)
            delta_k_x += (field.phase_sign * field.k * np.sin(angle_rad)
                          * field.side_sign)
        else:
            angle_rad = math.radians(field.angle_deg)
            delta_k_z_absolute += field.phase_sign * field.k * math.cos(angle_rad)
            delta_k_x += (field.phase_sign * field.k * math.sin(angle_rad)
                          * field.side_sign)
    delta_k_z_relative = delta_k_z_absolute - (reference_delta_k or 0.0)
    if precise_norm:
        # Match scalar math.hypot exactly at deep sinc minima. Vectorized
        # np.hypot can differ by an ulp on the same components.
        return np.fromiter(map(math.hypot, delta_k_z_relative.flat, delta_k_x.flat),
                           dtype=float, count=delta_k_x.size).reshape(delta_k_x.shape)
    return np.hypot(delta_k_z_relative, delta_k_x)


def biphoton_phase_matching_map(fields, L, *, signal_angle_deg,
                                 idler_angle_deg, reference_delta_k=0.0,
                                 span_deg=3.0, points=121):
    """Build the optional 2-D signal/idler collection-acceptance map.

    The map is a presentation diagnostic, not an input to the biphoton solve.
    Keeping it in a standalone helper lets headless and batch callers skip the
    2-D allocation while the figure path can request exactly the same map.
    """
    span_deg = max(float(span_deg), 0.0)
    points = max(int(points), 2)
    signal_axis = np.linspace(max(float(signal_angle_deg) - span_deg, 0.0),
                              float(signal_angle_deg) + span_deg, points)
    idler_axis = np.linspace(max(float(idler_angle_deg) - span_deg, 0.0),
                             float(idler_angle_deg) + span_deg, points)
    delta_k = phase_mismatch_vector_grid(
        fields, signal_axis, idler_axis,
        reference_delta_k=reference_delta_k)
    return signal_axis, idler_axis, phase_matching_weight(delta_k, L)


def energy_mismatch_hz(fields):
    signs = {"pump": 1.0, "coupling": 1.0, "signal": -1.0, "idler": -1.0}
    return sum(signs.get(f.role, 0.0) * f.frequency_hz for f in fields)


def _optical_k_from_offset(offset_rad_s, *, legacy_centroid=False):
    # Scan offsets are relative to F=2 -> F'=3, whereas OMEGA_D1 is the
    # fine-structure centroid. The hyperfine degeneracies are 5:7 on both
    # manifolds: Eg(F=2)=-7*HF/12 and Ee(F'=3)=+5*excited_HF/12.
    origin = (0.0 if legacy_centroid else
              (7.0 * OMEGA_HF + 5.0 * fwm.OMEGA_EXCITED_HF) / 12.0)
    return (constants.OMEGA_D1 + origin
            + np.asarray(offset_rad_s, dtype=float)) / constants.C_LIGHT


def seeded_option_a_wavenumbers(D_GHz, probe_axis_GHz, *, legacy_centroid=False):
    """Bare wave numbers; scan offsets use the 85Rb F=2 -> F'=3 origin.

    ``legacy_centroid=True`` reproduces historical numerical fixtures only.
    It never changes the detuning in the atomic Hamiltonian.
    """
    pump_offset = 2 * np.pi * float(D_GHz) * 1e9
    seed_offset = 2 * np.pi * np.asarray(probe_axis_GHz, dtype=float) * 1e9
    conj_offset = 2.0 * pump_offset - seed_offset
    k_pump = _optical_k_from_offset(pump_offset, legacy_centroid=legacy_centroid)
    k_seed = _optical_k_from_offset(seed_offset, legacy_centroid=legacy_centroid)
    k_conj = _optical_k_from_offset(conj_offset, legacy_centroid=legacy_centroid)
    return k_pump, k_seed, k_conj


def seeded_phase_mismatch_z(D_GHz, probe_axis_GHz,
                            angle_deg=SEEDED_PHASE_ANGLE_DEG, *,
                            legacy_centroid=False):
    """Option-A vacuum/geometric mismatch ``2k_p-(k_s+k_c)cos(theta)``.

    Susceptibility, including its real dispersive part, stays in the diagonal
    Maxwell response. Refractive indices must not be folded into this mismatch.
    """
    theta = math.radians(float(angle_deg))
    k_pump, k_seed, k_conj = seeded_option_a_wavenumbers(
        D_GHz, probe_axis_GHz, legacy_centroid=legacy_centroid)
    return 2.0 * k_pump - (k_seed + k_conj) * math.cos(theta)


def _gaussian_overlap_profile(nseg, L, w_pump, w_probe, angle_deg):
    """Amplitude overlap of crossed Gaussian beams along the cell."""
    theta = math.radians(float(angle_deg))
    z = ((np.arange(nseg, dtype=float) + 0.5) / nseg - 0.5) * L
    separation = np.abs(z * math.tan(theta))
    waist_sq = max(float(w_pump) ** 2 + float(w_probe) ** 2, 1e-30)
    profile = np.exp(-(separation ** 2) / waist_sq)
    return profile / max(float(np.nanmax(profile)), 1e-30)
