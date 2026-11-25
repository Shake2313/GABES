"""Focused three-carrier Maxwell–Floquet reference for the seeded scheme.

Fixed paraxial Gaussian modes, one-dimensional Maxwell velocities, local
stationary atoms and the reduced D1 dipoles. No fitted output gain or detector
plane correction enters the propagation.
"""

from dataclasses import dataclass

import numpy as np

from ... import beam, constants, core, doppler, hyperfine, observables
from ...quantum.diffusion import ordered_jump_diffusion
from ...quantum.reservoirs import CollapseChannel, ExplicitReservoirs, thermal_reset_channels
from .common import (
    BRANCHES, EXCITED_STATES, GROUND_STATES, N_LEVELS, RABI_DIPOLE_C_M,
    TRANSITION_DIPOLE_SCALE, collisional_atom, seeded_floquet_frequency,
)
from .phase_matching import seeded_option_a_wavenumbers, seeded_phase_mismatch_z
from .seeded import branch_center_GHz
from .seeded_response import _ground_legs, pump_hamiltonian_at_Deff_zero

_CARRIERS = ("pump", "probe", "conjugate")


@dataclass(frozen=True)
class LongitudinalSeededInputs:
    """Lab detunings in GHz/MHz; cell geometry, power and reset rate in SI.

    The residual scales the Maxwell polarization of all three carriers.
    The physical density sets the collision rates independently.
    """

    D_GHz: float
    delta_mhz: float
    T: float
    L: float
    P_pump: float
    P_probe: float
    w_pump: float
    w_probe: float
    w_conj: float
    angle_deg: float
    transit_rate: float
    coupling_residual: float
    branch: int = -1

    def __post_init__(self):
        values = (self.D_GHz, self.delta_mhz, self.T, self.L, self.P_pump,
                  self.P_probe, self.w_pump, self.w_probe, self.w_conj,
                  self.angle_deg, self.transit_rate, self.coupling_residual)
        if (not np.isfinite(values).all() or self.branch not in BRANCHES
                or min(self.T, self.P_probe, self.w_pump, self.w_probe, self.w_conj) <= 0
                or min(self.L, self.P_pump, self.transit_rate, self.coupling_residual) < 0
                or abs(self.angle_deg) >= 90):
            raise ValueError("finite physical inputs and one seeded Raman branch required")


class LongitudinalSeededMedium:
    """Reciprocal local drives and Maxwell projections for three optical modes."""

    def __init__(self, inputs, *, velocity_step=2.5, velocity_cutoff=6.0,
                 transverse_order=3, floquet_order=3):
        self.inputs = inputs
        if (int(transverse_order) != transverse_order or transverse_order < 1
                or int(floquet_order) != floquet_order or floquet_order < 1
                or not np.isfinite([velocity_step, velocity_cutoff]).all()
                or min(velocity_step, velocity_cutoff) <= 0):
            raise ValueError("positive quadrature settings and integer Floquet order required")
        self.floquet_order = int(floquet_order)
        self.velocity_step = float(velocity_step)
        self.velocity_cutoff = float(velocity_cutoff)
        self.transverse_order = int(transverse_order)
        D = inputs.D_GHz
        probe = branch_center_GHz(D, inputs.branch) + inputs.delta_mhz * 1e-3
        self.k = np.array(seeded_option_a_wavenumbers(D, probe))
        self.photon_energy = constants.HBAR * constants.C_LIGHT * self.k
        self.mismatch = seeded_phase_mismatch_z(D, probe, inputs.angle_deg)
        self.beat = -seeded_floquet_frequency(2*np.pi*1e6*inputs.delta_mhz, inputs.branch)
        self.density = hyperfine.number_density(inputs.T)
        self.atom = collisional_atom(inputs.T, self.density, transit_rate=inputs.transit_rate)
        v, self.velocity_weights = doppler.velocity_grid(
            inputs.T, dv=velocity_step, cutoff_sigma=velocity_cutoff)
        # Same axial Doppler shift as the reduced production solve. Angular
        # Raman Doppler and transport between transverse nodes are separate.
        self.deff = 2*np.pi*1e9*D - constants.K_VEC*v
        self.density_coupled = self.density * inputs.coupling_residual
        self.rabi_per_sqrt_intensity = beam.two_level_rabi(1.0, constants.GAMMA, constants.I_SAT)
        self.maxwell = 0.5j*self.k*observables.chi_phys(
            1.0, self.density_coupled, dipole=RABI_DIPOLE_C_M, line_strength=1.0
        ) / self.rabi_per_sqrt_intensity
        self.waists = np.array([inputs.w_pump, inputs.w_probe, inputs.w_conj])
        self.areas = observables.gaussian_mode_area(self.waists)
        self.slopes = np.tan(np.radians(inputs.angle_deg))*np.array([0., 1., -1.])
        self.zero_pump = pump_hamiltonian_at_Deff_zero(0., 0.)
        probe_ground, conjugate_ground = _ground_legs(inputs.branch)
        lowering = np.zeros((3, N_LEVELS, N_LEVELS), complex)
        lowering[0] = TRANSITION_DIPOLE_SCALE
        lowering[1, probe_ground] = TRANSITION_DIPOLE_SCALE[probe_ground]
        lowering[2, conjugate_ground] = TRANSITION_DIPOLE_SCALE[conjugate_ground]
        self.lowering = lowering
        self.spontaneous = np.zeros(N_LEVELS)
        for excited, _, rate in self.atom.decay:
            self.spontaneous[excited] += rate
        self.transit = np.zeros(N_LEVELS)
        self.transit[list(EXCITED_STATES)] = self.atom.transit_reset_rate

        # Integrate dA against a mixture of the three normalized |u|²
        # distributions. Reflect y analytically, since every beam has y=0.
        x, wt = np.polynomial.hermite.hermgauss(self.transverse_order)
        y, wy = x[x >= 0], wt[x >= 0]*np.where(x[x >= 0] > 0, 2.0, 1.0)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        self.xy = np.column_stack([xx.ravel(), yy.ravel()])/np.sqrt(2)
        self.area_node_weights = np.tile((wt[:, None]*wy/np.pi).ravel()/3, 3)

    def _transverse(self, z):
        centers = (z - self.inputs.L/2)*self.slopes
        xy = self.waists[:, None, None]*self.xy
        xy[:, :, 0] += centers[:, None]
        x, y = xy.reshape(-1, 2).T
        r2 = (x[:, None]-centers)**2 + y[:, None]**2
        profiles = np.exp(-r2/self.waists**2)/np.sqrt(self.areas)
        weights = self.area_node_weights / np.mean(profiles**2, axis=1)
        # Discrete modal normalization also makes the weak passive absorption
        # independent of the overlap quadrature resolution.
        profiles /= np.sqrt(weights @ profiles**2)
        return profiles, weights

    def atomic_harmonics(self, rabi, *, dense=False):
        """Physical pump frame. Probe and conjugate occupy harmonics ±1."""
        pump, probe, conjugate = rabi
        pump_drive = 0.5*pump*self.lowering[0].conj().T
        h0 = self.zero_pump + pump_drive + pump_drive.conj().T
        h1 = 0.5*(probe*self.lowering[1].conj().T + conjugate.conjugate()*self.lowering[2])
        solve = core.floquet_solve_direct if dense else core.floquet_solve_truncated
        return solve(core.build_liouvillian(h0, self.atom), core.comm_super(h1),
                     core.comm_super(h1.conj().T), self.beat, self.deff, self.atom.S_v,
                     N_LEVELS, self.floquet_order, return_harmonics=True)

    def _local(self, z, envelopes, *, noise_state=False):
        rate = np.zeros(3, complex)
        loss = np.zeros(2)
        states = np.zeros((3, 3, N_LEVELS, N_LEVELS), complex) if noise_state else None
        if self.density_coupled == 0:
            return rate, loss, states
        profiles, weights = self._transverse(z)
        # Put the vacuum loop phase on the conjugate. The local Hamiltonian
        # already sees the pump's accumulated material phase in its envelope.
        phase = np.array([1., 1., np.exp(-1j*self.mismatch*z)])
        n = self.floquet_order
        for spatial, area in zip(profiles, weights):
            mode = spatial*phase
            rabi = self.rabi_per_sqrt_intensity * mode * envelopes
            rho = np.einsum("v,vhij->hij", self.velocity_weights,
                            self.atomic_harmonics(rabi))
            polarization = np.einsum("pij,pji->p", self.lowering, rho[[n, n+1, n-1]])
            rate += area * mode.conj() * polarization
            population = rho[n].diagonal().real
            loss += area*np.array([self.spontaneous @ population, self.transit @ population])
            if noise_state:
                states += np.einsum("i,j,ab->ijab", area*spatial, spatial, rho[n])
        return self.maxwell*rate, self.density_coupled*loss, states

    def carrier_rate(self, z, envelopes):
        """Derivative of the laboratory-frame sqrt(W) carrier envelopes."""
        return self._local(z, envelopes)[0]

    def _reservoirs(self):
        channels = []
        for e, g, rate in self.atom.decay:
            jump = np.zeros((N_LEVELS, N_LEVELS), complex)
            jump[g, e] = np.sqrt(rate)
            channels.append(CollapseChannel(f"radiative:{e}->{g}", jump, "radiative",
                                            "85Rb D1 hyperfine branching"))
        ground = self.atom.ground_collision_dephasing_rate
        dephasings = {(i, j): rate for i, j, rate in self.atom.dephasing}
        optical = dephasings.get((GROUND_STATES[0], EXCITED_STATES[0]), 0.)
        # A ground differential phase jump adds ground/4 to each optical
        # dephasing. Remove that part before adding the common optical jump.
        if optical < ground/4:
            raise ValueError("the declared ground/optical dephasings have no diagonal jump realization")
        phase_jump = np.diag([1., -1., 0., 0.])*np.sqrt(ground/2)
        optical_jump = np.diag([0., 0., 1., 1.])*np.sqrt(2*(optical-ground/4))
        channels.extend((
            CollapseChannel("ground-collision", phase_jump, "ground_collision", "Rb–Rb dephasing"),
            CollapseChannel("optical-collision", optical_jump, "optical_collision", "Rb D1 self-broadening"),
        ))
        channels.extend(thermal_reset_channels(
            self.atom.transit_reset_rate, self.atom.thermal_reset_populations,
            source="thermal replacement at the declared transit rate"))
        reservoirs = ExplicitReservoirs(N_LEVELS, tuple(channels))
        if not np.allclose(reservoirs.dissipator(), self.atom.lindblad, rtol=1e-12, atol=1e-9):
            raise ValueError("explicit jump channels differ from the mean-field dissipator")
        return reservoirs

    @staticmethod
    def _projected_atomic_diffusion(reservoirs, states, quadratures):
        """Assemble the ordered Gram matrix using the same transverse profiles.

        Ordered jump diffusion is linear in the state. For a block (a,b) its
        input state is the Maxwell average weighted by u_a*u_b over dA.
        """
        diffusion = np.zeros((len(reservoirs.channels), 6, 6), complex)
        for a in range(3):
            rows = slice(2*a, 2*a+2)
            for b in range(a, 3):
                cols = slice(2*b, 2*b+2)
                state = states[a, b]
                overlap = np.trace(state).real
                if overlap <= np.finfo(float).tiny:
                    continue
                local = ordered_jump_diffusion(reservoirs, state/overlap, quadratures)
                diffusion[:, rows, cols] = overlap*local[:, rows, cols]
                if a != b:
                    diffusion[:, cols, rows] = diffusion[:, rows, cols].conj().swapaxes(-1, -2)
        return diffusion

    def propagate(self, segments, *, noise_ledger=False):
        """Fixed RK4 of the local nonlinear equations, with a photon budget."""
        if isinstance(segments, bool) or int(segments) != segments or segments < 1:
            raise ValueError("positive integer segment count required")
        z = np.linspace(0, self.inputs.L, int(segments)+1)
        envelopes = np.empty((z.size, 3), complex)
        envelopes[0] = np.sqrt([self.inputs.P_pump, self.inputs.P_probe, 0.])
        loss = np.zeros((z.size, 2))
        dz = self.inputs.L/segments
        mid = (z[:-1]+z[1:])/2
        local_balance = np.zeros(int(segments))
        reservoirs = self._reservoirs() if noise_ledger and self.density_coupled else None
        sources = []
        quadratures = np.stack([self.lowering+self.lowering.conj().transpose(0, 2, 1),
                                -1j*(self.lowering-self.lowering.conj().transpose(0, 2, 1))],
                               axis=1).reshape(6, N_LEVELS, N_LEVELS)/np.sqrt(2)
        for index, location in enumerate(mid):
            a = envelopes[index]
            k1, l1, _ = self._local(z[index], a)
            k2, l2, _ = self._local(location, a+dz*k1/2)
            middle = a+dz*k2/2
            k3, l3, states = self._local(location, middle, noise_state=reservoirs is not None)
            k4, l4, _ = self._local(z[index+1], a+dz*k3)
            envelopes[index+1] = a+dz*(k1+2*k2+2*k3+k4)/6
            loss[index+1] = loss[index]+dz*(l1+2*l2+2*l3+l4)/6
            local_balance[index] = np.sum(2*np.real(middle.conj()*k3)/self.photon_energy)+l3.sum()
            if reservoirs is not None:
                sources.append(self._projected_atomic_diffusion(reservoirs, states, quadratures))
        power = np.abs(envelopes)**2
        flux = power/self.photon_energy
        residual = flux.sum(axis=1)-flux[0].sum()+loss.sum(axis=1)
        # Write the carrier power balance in pump-photon units plus an exact
        # sideband frequency offset term. The sinks are photon counts, not
        # a model for the absolute emitted photon energies or heat.
        spectral_power_change = (flux-flux[0]) @ (self.photon_energy-self.photon_energy[0])
        sink_equivalent_power = loss*self.photon_energy[0]
        power_residual = (power.sum(axis=1)-power[0].sum()
                          + sink_equivalent_power.sum(axis=1)-spectral_power_change)
        return {
            "z_m": z, "envelope_sqrt_W": envelopes, "power_W": power,
            "probe_gain": float(power[-1, 1]/self.inputs.P_probe),
            "conjugate_gain": float(power[-1, 2]/self.inputs.P_probe),
            "photon_flux_s_inverse": flux,
            "photon_sink_labels": ("spontaneous_emission", "excited_atoms_removed_by_reset"),
            "cumulative_photon_sinks_s_inverse": loss,
            "photon_balance_relative": residual/flux[0].sum(),
            "local_photon_balance_s_inverse_m_inverse": local_balance,
            "spectral_power_change_W": spectral_power_change,
            "photon_sinks_at_pump_energy_W": sink_equivalent_power,
            "power_balance_residual_W": power_residual,
            "noise_ledger": {
                "midpoint_z_m": mid,
                "basis": tuple(f"{name}:{q}" for name in _CARRIERS for q in ("X", "Y")),
                "reservoirs": tuple(c.name for c in reservoirs.channels) if reservoirs else (),
                "reservoir_kinds": tuple(c.kind for c in reservoirs.channels) if reservoirs else (),
                "reservoir_sources": tuple(c.source for c in reservoirs.channels) if reservoirs else (),
                "coupled_density_m3": self.density_coupled,
                "ordered_atomic_diffusion_s_inverse": np.array(sources).reshape(
                    (int(segments), -1, 6, 6)),
                "scope": ("cycle-averaged ordered atomic jump sources in the local phase gauge, "
                          "weighted by pairwise transverse modes and the Maxwell distribution; "
                          "optical covariance requires elimination and propagation"),
            },
        }

    def converged(self, segments=32, *, tolerance=0.005, noise_ledger=False):
        """Compare the nonlinear output fields and both powers at N and 2N."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("positive finite convergence tolerance required")
        coarse = self.propagate(segments)
        fine = self.propagate(2*segments, noise_ledger=noise_ledger)
        power_floor = self.inputs.P_probe*1e-9
        change = np.abs(fine["power_W"][-1]-coarse["power_W"][-1])/np.maximum(
            fine["power_W"][-1], power_floor)
        # Independent arm scaling keeps the pump from masking sideband errors.
        envelope = fine["envelope_sqrt_W"][-1]
        field_change = np.abs(envelope-coarse["envelope_sqrt_W"][-1])/np.maximum(
            np.abs(envelope), np.sqrt(power_floor))
        balance = float(np.max(np.abs(fine["photon_balance_relative"])))
        fine["longitudinal_convergence"] = {
            "segments": (segments, 2*segments), "power_relative_change": change,
            "field_relative_change": field_change, "max_photon_budget_relative_residual": balance,
            "tolerance": tolerance,
            "passed": bool(np.max(change) <= tolerance and np.max(field_change) <= tolerance
                           and balance <= tolerance),
            "scope": "longitudinal RK4 step size at fixed transverse, velocity and Floquet settings",
        }
        return fine
