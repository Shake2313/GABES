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

from . import constants, core, magneto_multilevel, pole_doppler, species
from ._arrays import readonly_float_array
from .psr_full_obe import solve_psr_pump
from .optical_bloch import (
    _assemble_pi_mirror_generator,
    _linear_coherence_per_half_rabi,
    _pi_mirror_model,
    _reduced_state_diagnostics,
    solve_optical_pump,
)
from .optical_frequency import RbComposition, RbOpticalFrequency


MODEL_ID = "rb87_d1_maxwell_full_obe_mean_v1"
_POLE_GUARD_RTOL = 1e-9


def _zero_drive_maxwell_mean(atom, detuning, shifts, weights, density, reset):
    """Exact analytic zero-drive susceptibility averaged on the supplied velocity nodes."""
    n_ground = len(atom.ground)
    centroid = magneto_multilevel.component_centroid_mhz(atom)
    laser_mhz = centroid + detuning - shifts
    # Sum_{g,e} |d_0[g,e]|^2 / (-(laser + E_g - E_e)*MHZ - i*(Gamma/2 + reset)) / n_ground
    hwhm = atom.gamma / 2.0 + reset
    d0_sq = np.abs(atom.dipoles[0]) ** 2
    chi_sum = 0j
    for ground_slice, energy in zip(atom.ground_slices, atom.ground_energy_mhz):
        line_weights = np.sum(d0_sq[ground_slice, :], axis=0)
        det = (laser_mhz[:, None] + energy - atom.excited_energy_mhz[None, :]) * species.MHZ
        denom = -det - 1j * hwhm
        chi_sum += np.sum(weights[:, None] * (line_weights[None, :] / (n_ground * denom)))
    prefactor = density * atom.dipole_sq / (constants.EPS_0 * constants.HBAR)
    return complex(prefactor * chi_sum)


def _pi_mirror_pole_maxwell_mean(isotope, line, atom, detuning, rabi, density,
                                 reset, shifts, weights):
    """Exact discrete-measure Maxwell average via M-even pi-frame pole_doppler.

    Reproduces the fixed quadrature nodes/weights (pole_doppler.discrete_mean)
    with a direct-solve guard at representative nodes. Falls back to batched
    reduced real solves across all velocity nodes if the pole decomposition
    fails or misses the guard.
    """
    model = _pi_mirror_model(isotope, line)
    centroid = magneto_multilevel.component_centroid_mhz(atom)
    laser_mhz = centroid + detuning
    a00 = _assemble_pi_mirror_generator(model, laser_mhz, rabi, reset)
    if not np.isfinite(a00).all():
        raise ValueError("settings exceed the finite full-OBE generator range")
    dim = a00.shape[0]
    scale = max(float(np.max(np.abs(a00))), 1.0)
    a00_tr = a00.copy()
    a00_tr[0, :] = model.trace_vec * scale
    s_red = model.s_doppler.copy()
    s_red[0, :] = 0.0
    rhs_tr = np.zeros(dim, dtype=float)
    rhs_tr[0] = scale

    prefactor = density * atom.dipole_sq / (constants.EPS_0 * constants.HBAR)
    w_chi = (prefactor / (0.5 * rabi)) * model.w_chi_unit
    readouts = np.vstack([w_chi, model.w_exc.astype(complex)])
    shift_rads = np.ascontiguousarray(shifts * species.MHZ, dtype=float)
    weight_arr = np.ascontiguousarray(weights, dtype=float)

    def _direct_check_nodes(indices):
        health = dict(trace_error=0., hermiticity_error=0., min_eigenvalue=1.,
                      steady_state_relative_residual=0.)
        d_vals = shift_rads[indices]
        mats = a00_tr[None, :, :] - d_vals[:, None, None] * s_red[None, :, :]
        rhs_b = np.broadcast_to(rhs_tr, (len(indices), dim))[..., None]
        xs = np.linalg.solve(mats, rhs_b)[..., 0]
        direct_out = xs @ readouts.T
        for k, d_val in enumerate(d_vals):
            l_node = a00 - d_val * model.s_doppler
            rho_vec = np.zeros(model.n_levels * model.n_levels, dtype=complex)
            rho_vec[model.dm0] = model.u_even @ xs[k]
            diag = _reduced_state_diagnostics(l_node, xs[k], rho_vec.reshape(model.n_levels, model.n_levels), reset)
            for key in health:
                reducer = min if key == "min_eigenvalue" else max
                health[key] = reducer(health[key], diag[key])
        return direct_out, health

    # Runtime pole-quality guard on up to 3 representative nodes (edges + nearest to rest).
    # Because discrete_mean evaluates scalar readouts directly from (lam, res, c0) without
    # forming x(D_j) on all velocity nodes, solving every node for a full residual would forfeit
    # the pole speedup. Checking 3 nodes directly catches ill-conditioned/defective eigenbases V
    # (which corrupt the rational function globally) at ~0.1 ms cost; 1e-9 matches the physicality
    # residual gate and is 1000x tighter than the repo's standard 1e-6 pole guard.
    n_shifts = len(shift_rads)
    mid_idx = int(np.argmin(np.abs(shift_rads)))
    guard_idx = sorted(set((0, mid_idx, n_shifts - 1)))

    try:
        system = pole_doppler.AffineShiftSystem(
            a00_tr, np.zeros_like(a00_tr), s_red, rhs_tr, readouts)
        residues = system.residues(np.array([0.0]))
        if (np.isfinite(residues.lam).all()
                and np.isfinite(residues.res).all()
                and np.isfinite(residues.c0).all()
                and residues.min_abs_imag > 0.0):
            direct_guard, health = _direct_check_nodes(guard_idx)
            guard_ok = True
            for k, idx in enumerate(guard_idx):
                pole_val = residues.value_at(float(shift_rads[idx]))[0]
                chi_scale = max(abs(direct_guard[k, 0]), 1e-30)
                exc_scale = max(abs(direct_guard[k, 1]), 1e-12)
                if (abs(pole_val[0] - direct_guard[k, 0]) > _POLE_GUARD_RTOL * chi_scale
                        or abs(pole_val[1] - direct_guard[k, 1]) > _POLE_GUARD_RTOL * exc_scale):
                    guard_ok = False
                    break
            if guard_ok:
                means = residues.discrete_mean(shift_rads, weight_arr)[0]
                if np.all(np.isfinite(means)):
                    return complex(means[0]), float(means[1].real), health
    except (ValueError, np.linalg.LinAlgError):
        pass

    # Fallback: exact batched reduced real solve over every velocity node
    direct_all, health = _direct_check_nodes(list(range(n_shifts)))
    means = weight_arr @ direct_all
    return complex(means[0]), float(means[1].real), health


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
    if legacy:
        inputs = dict(intensity_mw_cm2=intensity / 10., temp_c=temperature,
                      ground_relaxation_hz=reset, density_m3=density_m3, jones_xy=(1., 0.))
        mean = None
        health = dict(trace_error=0., hermiticity_error=0., min_eigenvalue=1.,
                      steady_state_relative_residual=0.)
        with core.blas_single_thread():
            for shift, weight in zip(shifts, weights, strict=True):
                velocity = float(shift * 1e6 * constants.C_LIGHT / frequency_hz)
                state = solve_psr_pump(detuning, velocity_m_s=velocity, **inputs)
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
    else:
        density = (species.number_density(species.ISOTOPES[isotope], temperature + 273.15)
                   if density_m3 is None else float(density_m3))
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("laser_frequency_hz must be finite and positive")
        intensity_mw_cm2 = intensity / 10.
        peak_field = float(np.sqrt(2 * intensity_mw_cm2 * 10 / (constants.EPS_0 * constants.C_LIGHT)))
        rabi = float(np.sqrt(atom.dipole_sq) * peak_field / constants.HBAR)
        wave_number = 2 * np.pi * frequency_hz / constants.C_LIGHT
        with core.blas_single_thread():
            if intensity == 0.:
                chi_val = _zero_drive_maxwell_mean(atom, detuning, shifts, weights, density, reset)
                exc_val = 0.0
                health = dict(trace_error=0., hermiticity_error=0., min_eigenvalue=0.,
                              steady_state_relative_residual=0.)
            else:
                chi_val, exc_val, health = _pi_mirror_pole_maxwell_mean(
                    isotope, line, atom, detuning, rabi, density, reset, shifts, weights)
        mean = {
            "chi_probe": complex(chi_val),
            "alpha_per_m": float(wave_number * chi_val.imag),
            "excited_fraction": float(exc_val),
            "polarization_xy_c_m2": np.array([constants.EPS_0 * peak_field * chi_val, 0j], dtype=complex),
        }
        conditions = {
            "isotope": isotope, "line": line, "b_ut": 0.,
            "detuning_mhz": detuning,
            "reference_centroid_mhz": centroid,
            "optical_frequency_hz": float(frequency_hz),
            "intensity_mw_cm2": intensity_mw_cm2,
            "jones_xy": np.array([1., 0.], dtype=complex),
            "peak_field_v_m": peak_field, "rabi_rad_s": rabi,
            "temp_c": temperature, "ground_relaxation_hz": reset, "density_m3": float(density),
            "density_source": "pure-isotope vapor-pressure density" if density_m3 is None else "specified isotope density",
            "ground_f_coherences": "retained",
            "emission_secularization": "final Fg and q resolved, excited F coherent",
            "susceptibility_basis": "finite-drive induced polarization" if intensity > 0 else "zero-drive linear-probe limit",
        }
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
