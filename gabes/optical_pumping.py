"""Stationary populations of an optically driven atom."""
import time

import numpy as np

from . import beam, constants, core, doppler, pole_doppler, species


def steady_populations(L0, deff_axis, S_v, n, chunk=1500):
    """Diagonal populations ρ_ii(Δ_eff) on a fine axis, in memory-safe chunks."""
    pops = np.empty((deff_axis.size, n))
    for s in range(0, deff_axis.size, chunk):
        sl = slice(s, s + chunk)
        rho = core.steady_state_batched(L0, deff_axis[sl], S_v, n)
        pops[sl] = np.einsum("vii->vi", rho).real
    return pops


class ProjectedProbe:
    """Single-probe saturation of a hyperfine manifold with uniform m_F populations.

    Each allowed pair retains its own optical coherence and shares the total
    F populations with every other pair. The coherence is driven by
    p_g - (g_g/g_e) p_e, where g_F = 2F+1. Eliminating it at steady state
    gives a population-rate system with stimulated detailed balance.
    Inter-transition coherences are outside this projection.
    """

    def __init__(self, manifold, optical_hwhm_rad_s, transit_rate):
        width = float(optical_hwhm_rad_s)
        transit = float(transit_rate)
        if not np.isfinite(transit) or transit <= 0.0:
            raise ValueError("The projected probe requires a positive finite transit rate.")
        if not np.isfinite(width) or width < 0.5 * manifold.gamma + transit:
            raise ValueError("Optical HWHM must include natural and transit decay.")

        self._manifold = manifold
        self._width = width
        self._transit = transit
        self._population_decay = manifold.gamma + transit
        self._frequencies = manifold.omega
        self._ground = manifold.g_idx
        self._excited = manifold.e_idx
        self._nground = len(manifold.Fg)
        self._nlevels = manifold.n_levels
        self._nlinks = len(self._frequencies)

        ground_f = np.asarray(manifold.Fg)[self._ground]
        excited_f = np.asarray(manifold.Fe)[self._excited - self._nground]
        self._degeneracy_ratio = (2.0 * ground_f + 1.0) / (2.0 * excited_f + 1.0)
        area = species.line_integrated_alpha(manifold.iso, manifold.line, 1.0)
        self._areas = np.array([area[fg, fe] for fg, fe in zip(ground_f, excited_f)])

        structure = species.level_structure(manifold.iso.name, manifold.line)
        dipole_sq = structure.dipole_sq
        area_per_strength = np.pi * manifold.k_vec * dipole_sq / (
            constants.EPS_0 * constants.HBAR
        )
        self._strengths = self._areas / (
            area_per_strength * manifold.p_ground[self._ground]
        )
        self._rabi_scale = np.sqrt(dipole_sq) * beam.field_amplitude(1.0) / constants.HBAR
        self._power_scale = constants.PLANCK * manifold.nu0 * self._population_decay

        # Start from the natural branching table. Transit is a separate
        # term even when the caller's manifold already has a transit reset.
        natural = species.build_manifold(manifold.iso, manifold.line)
        thermal = np.zeros(self._nlevels)
        thermal[:self._nground] = manifold.p_ground
        self._unforced = transit * (thermal[:, None] - np.eye(self._nlevels))
        for source, target, rate in natural.atom.decay:
            self._unforced[target, source] += rate
            self._unforced[source, source] -= rate

        self._base, self._drive, self._detuning, self._poles = self._affine_system()
        self._trace = self._base[0].copy()
        self._check_detuning = np.sort(self._frequencies)
        midpoints = 0.5 * (self._check_detuning[:-1] + self._check_detuning[1:])
        self._check_detuning = np.concatenate((self._check_detuning, midpoints))

    def _affine_system(self):
        """Pack the population and real optical-coherence equations."""
        dimension = self._nlevels + 2 * self._nlinks
        real = self._nlevels + 2 * np.arange(self._nlinks)
        imag = real + 1
        ground, excited = self._ground, self._excited
        coupling = np.sqrt(self._strengths)

        base = np.zeros((dimension, dimension))
        base[:self._nlevels, :self._nlevels] = self._unforced
        base[real, real] = base[imag, imag] = -self._width
        base[real, imag] = self._frequencies
        base[imag, real] = -self._frequencies
        detuning = np.zeros_like(base)
        detuning[real, imag] = 1.0
        detuning[imag, real] = -1.0
        drive = np.zeros_like(base)
        drive[ground, imag] = coupling
        drive[excited, imag] = -coupling
        drive[imag, ground] = -0.5 * coupling
        drive[imag, excited] = 0.5 * coupling * self._degeneracy_ratio

        # The pole helper expects an already imposed trace boundary condition.
        base[0] = drive[0] = 0.0
        base[0, :self._nlevels] = 1.0
        rhs = np.zeros(dimension)
        rhs[0] = 1.0
        readout = np.zeros(dimension)
        readout[self._nground:self._nlevels] = 1.0
        system = pole_doppler.AffineShiftSystem(base, drive, detuning, rhs, readout)
        return base, drive, detuning, system

    def peak_saturation(self, intensity_w_m2):
        """Resonant unit-dipole saturation scale Ω²/(Γ₁γ₂) at local intensity."""
        return (self._rabi_scale**2 * intensity_w_m2
                / (self._population_decay * self._width))

    def linear_cross_section(self, detuning_rad_s, doppler_sigma_rad_s):
        """Matching weak-probe cross section in m², including transit HWHM."""
        detuning, sigma = self._frequency_inputs(detuning_rad_s, doppler_sigma_rad_s)
        profile = doppler.voigt_profile(
            detuning[:, None] - self._frequencies, 2.0 * self._width, sigma
        )
        return profile @ self._areas

    @staticmethod
    def _frequency_inputs(detuning_rad_s, doppler_sigma_rad_s):
        detuning = np.asarray(detuning_rad_s, dtype=float)
        sigma = float(doppler_sigma_rad_s)
        if detuning.ndim != 1 or not np.all(np.isfinite(detuning)):
            raise ValueError("Probe detunings must be a finite one-dimensional axis.")
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Doppler sigma must be positive and finite.")
        return detuning, sigma

    def _stationary_populations(self, rabi, detuning):
        """Direct projected OBE solve for pole guards and fallback quadrature."""
        matrix = (
            self._base + np.asarray(rabi)[..., None, None] * self._drive
            - detuning[..., None, None] * self._detuning
        )
        state = core.stationary_solve(matrix, self._trace, overwrite_a=True)
        return state[..., :self._nlevels]

    def populations(self, intensity_w_m2, detuning_rad_s):
        """Stationary F populations per effective detuning, without Doppler averaging."""
        intensity = float(intensity_w_m2)
        detuning = np.asarray(detuning_rad_s, dtype=float)
        if not np.isfinite(intensity) or intensity < 0.0:
            raise ValueError("Probe intensity must be finite and nonnegative.")
        if detuning.ndim != 1 or not np.all(np.isfinite(detuning)):
            raise ValueError("Probe detunings must be a finite one-dimensional axis.")
        return self._stationary_populations(self._rabi_scale * np.sqrt(intensity), detuning)

    def _checked_maxwell_mean(self, rabi, detuning, sigma):
        """Excited population mean and a per-row pole validity flag."""
        shape = (rabi.size, detuning.size)
        try:
            poles = self._poles.residues(rabi)
        except np.linalg.LinAlgError:
            return np.zeros(shape), np.zeros(rabi.size, dtype=bool)

        lam, residues = poles.lam, poles.res[:, 0, :]
        constant = poles.c0[:, 0]
        valid = (np.all(np.isfinite(lam) & np.isfinite(residues), axis=1)
                 & np.isfinite(constant)
                 & (np.sum(lam.imag > 0.0, axis=1) == self._nlinks)
                 & (np.min(np.abs(lam.imag), axis=1) > self._width / 8.0))
        candidates = np.flatnonzero(valid)
        result = np.zeros(shape)
        if candidates.size == 0:
            return result, valid

        guard = self._check_detuning
        direct = self._stationary_populations(
            rabi[candidates, None], guard
        )[..., self._nground:].sum(axis=-1)
        pole_guard = constant[candidates, None] - np.sum(
            residues[candidates, :, None] / (lam[candidates, :, None] - guard), axis=1
        )
        valid[candidates] &= (
            np.max(np.abs(pole_guard - direct), axis=1)
            <= 1e-8 * np.max(direct, axis=1)
        )
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            return result, valid

        # The real generator supplies conjugate pairs. Contract the upper
        # half to avoid duplicate work at every scan frequency.
        upper = lam[candidates].imag > 0.0
        upper_poles = lam[candidates][upper].reshape(-1, self._nlinks)
        upper_residues = residues[candidates][upper].reshape(-1, self._nlinks)
        mean = doppler.maxwell_resolvent_mean(
            upper_poles[:, :, None], detuning[None, None, :], sigma
        )
        result[candidates] = constant[candidates, None].real - 2.0 * np.einsum(
            "ip,ipf->if", upper_residues, mean
        ).real
        valid[candidates] &= np.all(
            np.isfinite(result[candidates]) & (result[candidates] >= 0.0), axis=1
        )
        return result, valid

    @staticmethod
    def _check_deadline(deadline):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("The projected-probe model exceeded its solve time budget.")

    def _quadrature_mean(self, rabi, detuning, sigma, deadline=None):
        """Doppler quadrature of the direct OBE on one fine effective-detuning axis."""
        from scipy.interpolate import CubicSpline
        from scipy.signal import fftconvolve

        man = self._manifold
        thermal_temperature = (sigma / man.k_vec)**2 * man.iso.mass / constants.KB
        velocity, weights = doppler.maxwell_strip_grid(
            thermal_temperature, man.iso.mass, man.k_vec,
            self._width / 4.0, error=1e-9, cutoff_sigma=8.0,
        )
        step = man.k_vec * (velocity[1] - velocity[0])
        lo = np.min(detuning) + man.k_vec * velocity[0] - step
        hi = np.max(detuning) + man.k_vec * velocity[-1] + step
        if (hi - lo) / step > 1_000_000:
            raise ArithmeticError("Projected-probe fallback exceeded its detuning grid budget.")
        axis = lo + step * np.arange(int(np.ceil((hi - lo) / step)) + 1)
        population = np.empty(axis.size)
        for start in range(0, axis.size, 8192):
            self._check_deadline(deadline)
            part = slice(start, start + 8192)
            population[part] = self._stationary_populations(
                rabi, axis[part]
            )[:, self._nground:].sum(axis=1)
        mean = CubicSpline(axis, fftconvolve(population, weights, mode="same"))(detuning)
        if np.any(~np.isfinite(mean)) or np.any(mean < 0.0):
            raise ArithmeticError("Projected-probe fallback produced a nonphysical absorption.")
        return mean

    def absorption_cross_section(
        self, intensities_w_m2, detuning_rad_s, doppler_sigma_rad_s, *, deadline=None
    ):
        """Return the local Maxwell-averaged absorption [m²].

        An optional deadline is an absolute time from `time.perf_counter()`.
        """
        intensities = np.asarray(intensities_w_m2, dtype=float)
        detuning, sigma = self._frequency_inputs(detuning_rad_s, doppler_sigma_rad_s)
        if (intensities.ndim != 1 or not np.all(np.isfinite(intensities))
                or np.any(intensities < 0.0)):
            raise ValueError("Probe intensities must form a finite nonnegative axis.")
        result = np.empty((intensities.size, detuning.size))
        if not result.size:
            return result

        # The weak branch also handles inputs that are indistinguishable from
        # zero drive on the optical-pumping scale, before division by I.
        weak = (self.peak_saturation(intensities)
                <= np.finfo(float).eps * self._transit / self._population_decay)
        if np.any(weak):
            result[weak] = self.linear_cross_section(detuning, sigma)
        driven = np.flatnonzero(~weak)
        chunk = max(1, min(16, 100_000 // (detuning.size * self._nlinks)))
        for start in range(0, driven.size, chunk):
            self._check_deadline(deadline)
            rows = driven[start:start + chunk]
            rabi = self._rabi_scale * np.sqrt(intensities[rows])
            excitation, valid = self._checked_maxwell_mean(rabi, detuning, sigma)
            for index in np.flatnonzero(~valid):
                excitation[index] = self._quadrature_mean(
                    rabi[index], detuning, sigma, deadline
                )
            result[rows] = excitation * (self._power_scale / intensities[rows, None])
        self._check_deadline(deadline)
        return result
