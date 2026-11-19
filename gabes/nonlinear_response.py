"""Classical local single-beam nonlinear response for rubidium D1 and D2.

This adapter averages the same atomic mean field used by PSR, without building
RF response or Langevin kernels. It assumes a spatially local steady state;
transverse atomic transport, buffer gas and magnetic fields are not modeled.
"""
from copy import deepcopy
from dataclasses import replace
from math import erfc, sqrt
from time import perf_counter

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import constants, core, magneto_multilevel, species
from ._arrays import readonly_float_array
from .psr_full_obe import solve_psr_pump
from .optical_bloch import solve_optical_pump
from .optical_frequency import RbComposition, RbOpticalFrequency


MODEL_ID = "rb87_d1_maxwell_full_obe_mean_v1"


def _maxwell_mean(isotope, line, detuning_mhz, intensity_w_m2, *, temp_c=70.,
                  ground_relaxation_hz=1000., density_m3=None,
                  velocity_order=6, cutoff_sigma=6., legacy=False, laser_frequency_hz=None):
    """Return a fixed-order Maxwell average of the local complex susceptibility.

    Selected Rb isotope and D line, zero B, x-linear polarization. Detuning is
    laser frequency minus the upper-ground-F line-strength centroid in MHz;
    positive is blue. Intensity is cycle-averaged W/m^2. The reset rate is in
    s^-1, despite the inherited ``_hz`` suffix, and acts on the full density
    operator. Density is this isotope's number density in m^-3.

    ``velocity_order`` is Gauss-Legendre nodes per resonance-centered interval,
    not the total velocity count. The shared quadrature normalizes its finite
    Maxwell tail to unity. Cutoff and order convergence are deliberately
    unassessed here: callers must compare independently refined quadratures.
    ``chi`` is dimensionless, passive Im(chi)>0, and alpha=k*Im(chi) in m^-1.
    Only integrated means and state-health extrema are retained.
    """
    started = perf_counter()
    detuning, intensity, temperature, reset, cutoff = map(float, (
        detuning_mhz, intensity_w_m2, temp_c, ground_relaxation_hz, cutoff_sigma))
    if not np.isfinite([detuning, intensity, temperature, reset, cutoff]).all():
        raise ValueError("finite mean-response inputs required")
    if intensity < 0 or temperature <= -273.15 or reset <= 0 or cutoff <= 0:
        raise ValueError("nonnegative intensity, positive temperature, reset and cutoff required")
    if (isinstance(velocity_order, (bool, np.bool_)) or not np.isfinite(velocity_order)
            or int(velocity_order) != velocity_order or velocity_order < 2):
        raise ValueError("velocity_order must be an integer at least two")
    if density_m3 is not None:
        density_m3 = float(density_m3)
        if not np.isfinite(density_m3) or density_m3 < 0:
            raise ValueError("density_m3 must be finite and nonnegative")

    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    centroid = magneto_multilevel.component_centroid_mhz(atom)
    frequency_hz = atom.frequency_hz if laser_frequency_hz is None else laser_frequency_hz
    quadrature_atom = atom if laser_frequency_hz is None else replace(atom, frequency_hz=frequency_hz)
    shifts, weights, sigma = magneto_multilevel.velocity_quadrature(
        quadrature_atom, centroid + detuning, temperature + 273.15,
        order=int(velocity_order), cutoff_sigma=cutoff)
    inputs = dict(intensity_mw_cm2=intensity / 10., temp_c=temperature,
                  ground_relaxation_hz=reset, density_m3=density_m3, jones_xy=(1., 0.))
    mean = None
    health = dict(trace_error=0., hermiticity_error=0., min_eigenvalue=1.,
                  steady_state_relative_residual=0.)
    with core.blas_single_thread():
        for shift, weight in zip(shifts, weights, strict=True):
            velocity = float(shift * 1e6 * constants.C_LIGHT / frequency_hz)
            state = (solve_psr_pump(detuning, velocity_m_s=velocity, **inputs) if legacy else
                     solve_optical_pump(isotope, line, detuning, velocity_m_s=velocity,
                                        include_atomic_state=False, laser_frequency_hz=laser_frequency_hz, **inputs))
            if mean is None:
                mean = {key: np.zeros_like(state[key]) for key in
                        ("chi_probe", "alpha_per_m", "excited_fraction", "polarization_xy_c_m2")}
                conditions = deepcopy(state["conditions"])
            for key in mean:
                mean[key] += weight * state[key]
            for key in health:
                reducer = min if key == "min_eigenvalue" else max
                health[key] = reducer(health[key], state["diagnostics"][key])

    for key in ("velocity_m_s", "effective_detuning_mhz"):
        conditions.pop(key, None)
    conditions.update(
        intensity_w_m2=intensity, polarization="x-linear", doppler=True,
        velocity_distribution="independent one-dimensional Maxwell classes",
        velocity_measure="conditional finite-tail Maxwell distribution",
        velocity_sigma_m_s=float(sigma * 1e6 * constants.C_LIGHT / frequency_hz),
        detuning_reference="upper-ground-F line-strength centroid; positive is blue",
        spatial_response="local steady state; no transverse transport")
    health["solver_status"] = "all sampled velocity steady states physical"
    return dict(
        model_id=MODEL_ID if legacy else "rb_dline_maxwell_full_obe_v1",
        chi=complex(mean["chi_probe"]),
        alpha_per_m=float(mean["alpha_per_m"]),
        excited_fraction=float(mean["excited_fraction"]),
        polarization_xy_c_m2=mean["polarization_xy_c_m2"],
        conditions=conditions, diagnostics=health, class_count=len(weights),
        weight_sum=float(weights.sum()), velocity_order=int(velocity_order),
        cutoff_sigma=cutoff, omitted_maxwell_probability=erfc(cutoff / sqrt(2.)),
        convergence=dict(assessed=False, status="fixed quadrature; order and cutoff unassessed"),
        elapsed_seconds=perf_counter() - started)


def solve_rb87_d1_mean(detuning_mhz, intensity_w_m2, *, temp_c=70.,
                      ground_relaxation_hz=1000., density_m3=None,
                      velocity_order=6, cutoff_sigma=6.):
    """Compatibility reference: pure87 D1, upper-F centroid, isotope-centroid k.

    This preserves the original benchmark's frequency/wavevector convention
    and numerical path. New callers should use :func:`solve_rb_mean` with an
    explicit physical laser frequency.
    """
    return _maxwell_mean("Rb87", "D1", detuning_mhz, intensity_w_m2,
        temp_c=temp_c, ground_relaxation_hz=ground_relaxation_hz,
        density_m3=density_m3, velocity_order=velocity_order,
        cutoff_sigma=cutoff_sigma, legacy=True)


def solve_rb_mean(frequency, intensity_w_m2, *, rb85_fraction=species.RB85.abundance,
                  temp_c=70., ground_relaxation_hz=1000., density_m3=None,
                  velocity_order=6, cutoff_sigma=6., component_cache=None):
    """Full-OBE Maxwell response of any Rb85/Rb87 mixture at one laser frequency.

    ``frequency`` is an RbOpticalFrequency, not an ambiguous detuning scalar.
    ``density_m3`` is TOTAL elemental density, apportioned exactly once.
    Both isotopes see the same local W/m^2, with isotope-specific Doppler masses
    and optical detunings. Polarizations add before a common beam propagates.
    The reported absorption uses the physical laser k; the compatibility
    Rb87 D1 adapter retains its original centroid k.

    Scope: one CW x-linear beam, B=0, independent local velocity classes,
    fixed reset in s^-1, no inter-isotope collisions or transverse transport.
    All hyperfine/Zeeman states of the chosen D line are retained. Far enough
    from that line, the omitted other fine-structure manifold matters; this
    single-line rotating-wave model is not a broadband atomic polarizability.
    Returned convergence is unassessed until a caller refines the quadrature.
    Optional ``component_cache`` is an in-process mutable mapping; keys include
    the full atomic settings and density. It stores pure-isotope results at
    total density, then rescales polarization once for the chosen fractions.
    This is exact only because this model has no density-dependent collisions.
    Persistent reuse additionally requires caller-side source/version checks.
    """
    started = perf_counter()
    if not isinstance(frequency, RbOpticalFrequency):
        raise TypeError("frequency must be RbOpticalFrequency with an explicit reference")
    composition = RbComposition(rb85_fraction)
    intensity, temperature, reset, cutoff = map(float, (
        intensity_w_m2, temp_c, ground_relaxation_hz, cutoff_sigma))
    if (not np.isfinite([intensity, temperature, reset, cutoff]).all()
            or intensity < 0 or temperature <= -273.15 or reset <= 0 or cutoff <= 0):
        raise ValueError("finite nonnegative intensity and positive temperature/reset/cutoff required")
    if (isinstance(velocity_order, (bool, np.bool_)) or not np.isfinite(velocity_order)
            or int(velocity_order) != velocity_order or velocity_order < 2):
        raise ValueError("velocity_order must be an integer at least two")
    density = (species.number_density(species.RB85, temperature + 273.15)
               if density_m3 is None else float(density_m3))
    if not np.isfinite(density) or density < 0:
        raise ValueError("total density_m3 must be finite and nonnegative")
    chi, polarization, excited = 0j, np.zeros(2, complex), 0.
    components = {}
    health = dict(trace_error=0., hermiticity_error=0., min_eigenvalue=1.,
                  steady_state_relative_residual=0.)
    if density > 0:
        for isotope, fraction in composition.components:
            detuning = frequency.detuning_mhz(isotope, reference="upper_ground_centroid")
            key = ("rb_dline_maxwell_full_obe_v1", isotope, frequency.line,
                   detuning, intensity, temperature, reset, density,
                   int(velocity_order), cutoff, frequency.absolute_hz)
            result = None if component_cache is None else component_cache.get(key)
            if result is None:
                result = _maxwell_mean(isotope, frequency.line, detuning, intensity,
                    temp_c=temperature, ground_relaxation_hz=reset,
                    density_m3=density, velocity_order=velocity_order, cutoff_sigma=cutoff,
                    laser_frequency_hz=frequency.absolute_hz)
                if component_cache is not None:
                    component_cache[key] = deepcopy(result)
            result = deepcopy(result)
            result["chi"] *= fraction
            result["alpha_per_m"] = 2*np.pi/frequency.wavelength_m*result["chi"].imag
            result["polarization_xy_c_m2"] *= fraction
            result["conditions"]["density_m3"] = fraction*density
            chi += result["chi"]
            polarization += result["polarization_xy_c_m2"]
            excited += fraction*result["excited_fraction"]
            components[isotope] = result
            for key in health:
                reducer = min if key == "min_eigenvalue" else max
                health[key] = reducer(health[key], result["diagnostics"][key])
    health["solver_status"] = ("all sampled isotope/velocity states physical" if components
                               else "zero density; no atomic states sampled")
    k = 2*np.pi/frequency.wavelength_m
    return dict(model_id="rb_mixture_dline_maxwell_full_obe_v1", chi=complex(chi),
        alpha_per_m=float(k*chi.imag), excited_fraction=float(excited),
        polarization_xy_c_m2=polarization, components=components, diagnostics=health,
        conditions=dict(frequency=frequency.as_dict(), rb85_fraction=float(rb85_fraction),
            density_m3=float(density), isotope_density_m3={iso: fraction*density
                for iso, fraction in composition.components}, intensity_w_m2=intensity,
            temp_c=temperature, ground_relaxation_hz=reset, polarization="x-linear",
            spatial_response="local steady state; no transverse transport",
            carrier_wavevector="physical laser frequency for Doppler and propagation"),
        class_count=sum(value["class_count"] for value in components.values()),
        velocity_order=int(velocity_order), cutoff_sigma=cutoff,
        omitted_maxwell_probability=erfc(cutoff/sqrt(2.)),
        convergence=dict(assessed=False, status="fixed quadrature; order and cutoff unassessed"),
        elapsed_seconds=perf_counter()-started)


class ComplexSusceptibilityTable:
    """Shape-preserving local chi(I) interpolation on a supplied intensity axis.

    The first node must be I=0 with its separately evaluated weak-field chi.
    Real and imaginary quadratures are interpolated independently with PCHIP.
    No extrapolation is permitted: a beam exceeding the table's intensity
    ceiling must trigger a larger atomic table and a new convergence check.
    Node and interpolation accuracy remain the caller's responsibility.
    """

    def __init__(self, intensity_w_m2, chi):
        intensities = readonly_float_array(intensity_w_m2)
        values = np.array(chi, dtype=complex, copy=True)
        if (intensities.ndim != 1 or intensities.size < 2
                or values.shape != intensities.shape
                or not np.isfinite(intensities).all() or not np.isfinite(values).all()
                or intensities[0] != 0. or np.any(np.diff(intensities) <= 0.)):
            raise ValueError("finite matching chi/intensity axes with I[0]=0 and increasing positive nodes required")
        values.setflags(write=False)
        self.intensity_w_m2 = intensities
        self.chi = values
        self._real = PchipInterpolator(intensities, values.real, extrapolate=False)
        self._imag = PchipInterpolator(intensities, values.imag, extrapolate=False)

    @property
    def max_intensity_w_m2(self):
        return float(self.intensity_w_m2[-1])

    def __call__(self, intensity_w_m2):
        intensity = np.asarray(intensity_w_m2, dtype=float)
        if not np.isfinite(intensity).all() or np.any(intensity < 0.):
            raise ValueError("sample intensities must be finite and nonnegative")
        if np.any(intensity > self.max_intensity_w_m2):
            raise ValueError("intensity exceeds the susceptibility table; rebuild with a larger upper bound")
        result = self._real(intensity) + 1j * self._imag(intensity)
        return complex(result) if result.ndim == 0 else result
