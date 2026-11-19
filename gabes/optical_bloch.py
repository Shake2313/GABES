"""Saturated single-velocity optical Bloch response for Rb D1 and D2.

The complete rotating-wave Hamiltonian retains ground- and excited-hyperfine
coherences. Emission distinguishes final ground F and spherical polarization
but coherently sums excited-F and magnetic-sublevel paths, as in the weak PSR
pumping model. Thermal replacement acts on the entire density operator. The
state, Hamiltonian and named jumps are available for subsequent linearization.
"""
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from . import constants, core, magneto_multilevel, species
from .psr_conventions import atom_frame_detuning_mhz, circular_to_jones, jones_to_circular, stokes_rates
from .quantum.reservoirs import CollapseChannel, ExplicitReservoirs, thermal_reset_channels


MODEL_ID = "rb_dline_single_velocity_full_obe_v1"


@lru_cache(maxsize=8)
def _prepared_reservoirs(isotope, line, ground_relaxation_hz, legacy=False):
    """Keep the dissipator fixed while scanning the Hamiltonian or polarization."""
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    n_ground = len(atom.ground)
    n_levels = n_ground + len(atom.excited)
    channels = []
    for ground_slice in atom.ground_slices:
        ground_f = atom.ground[ground_slice.start][0]
        for q in (-1, 0, 1):
            jump = np.zeros((n_levels, n_levels), complex)
            jump[ground_slice, n_ground:] = np.sqrt(atom.gamma) * atom.dipoles[q][ground_slice]
            channels.append(CollapseChannel(
                f"emission:F={ground_f}:q={q}", jump, "spontaneous_emission",
                f"{isotope} {line} signed dipoles and natural width from species.level_structure",
            ))
    emission_channels = tuple(channels)
    thermal_populations = np.zeros(n_levels)
    thermal_populations[:n_ground] = 1.0 / n_ground
    channels.extend(thermal_reset_channels(
        ground_relaxation_hz, thermal_populations, name="reset",
        source=("Isotropic transit replacement into the "
                + ("eight" if n_ground == 8 else "twelve")
                + f" {isotope} ground sublevels"),
    ))
    reservoirs = ExplicitReservoirs(n_levels, tuple(channels))
    if legacy:
        dissipator = reservoirs.dissipator()
        dissipator.flags.writeable = False
    else:
        # Same named reset channels; use their shared closed-form sum rather
        # than hundreds of dense Kronecker products. The legacy PSR arithmetic
        # is retained above for byte-preserving compatibility.
        dissipator = ExplicitReservoirs(n_levels, emission_channels).dissipator()
        dissipator += core.thermal_reset_superoperator(ground_relaxation_hz, thermal_populations)
        dissipator = sparse.csc_matrix(dissipator)
    return reservoirs, dissipator


def _hamiltonian(atom, laser_from_centroid_mhz, rabi_rad_s, excitation):
    n_ground = len(atom.ground)
    ground_energy = np.empty(n_ground)
    for ground_slice, energy in zip(atom.ground_slices, atom.ground_energy_mhz):
        ground_energy[ground_slice] = energy
    excited_energy = atom.excited_energy_mhz - laser_from_centroid_mhz
    # Subtracting a scalar identity places the upper ground F at zero energy.
    energies = np.concatenate((ground_energy, excited_energy)) - ground_energy[-1]
    hamiltonian = np.diag(energies * species.MHZ).astype(complex)
    coupling = -0.5 * rabi_rad_s * excitation
    hamiltonian[n_ground:, :n_ground] = coupling
    hamiltonian[:n_ground, n_ground:] = coupling.conj().T
    return hamiltonian


@lru_cache(maxsize=4)
def _sparse_hamiltonian_generators(isotope, line):
    """Linear Hamiltonian basis mapped by the one shared commutator builder."""
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    ng, ne = len(atom.ground), len(atom.excited)
    zero = np.zeros((ne, ng), complex)
    base = _hamiltonian(atom, 0., 0., zero)
    detuning = np.diag(np.r_[np.zeros(ng), np.full(ne, -species.MHZ)])
    generators = [sparse.csc_matrix(core.comm_super(h)) for h in (base, detuning)]
    for q in (-1, 1):
        for phase in (1., 1j):
            coupling = _hamiltonian(atom, 0., 1., phase * atom.dipoles[q].T) - base
            generators.append(sparse.csc_matrix(core.comm_super(coupling)))
    return tuple(generators)


def _sparse_generator(atom, laser_mhz, rabi, circular, dissipator):
    basis = _sparse_hamiltonian_generators(atom.isotope, atom.line)
    result = dissipator + basis[0] + laser_mhz * basis[1]
    index = 2
    for q in (-1, 1):
        result = result + (rabi * circular[q].real) * basis[index]
        result = result + (rabi * circular[q].imag) * basis[index + 1]
        index += 2
    return result.tocsc()


def _sparse_stationary_state(generator, n):
    """Exact sparse LU of the full trace-one steady-state system."""
    matrix = generator.tolil()
    matrix[0, :] = core.population_trace(n)
    rhs = np.zeros(n*n, complex)
    rhs[0] = 1.
    try:
        vector = splu(matrix.tocsc()).solve(rhs)
    except RuntimeError as error:
        raise np.linalg.LinAlgError("sparse optical steady-state solve failed") from error
    return vector.reshape(n, n)


def _state_diagnostics(generator, rho, reset_rate):
    vector = rho.ravel()
    residual = float(np.linalg.norm(generator @ vector))
    # Scale by terms actually present in L rho. In particular, a large bare
    # hyperfine splitting acting on a zero coherence cannot mask a bad solve.
    scale = max(float(np.linalg.norm(np.abs(generator) @ np.abs(vector))),
                reset_rate * float(np.linalg.norm(vector)))
    diagnostics = {
        "trace_error": float(abs(np.trace(rho) - 1)),
        "hermiticity_error": float(np.linalg.norm(rho - rho.conj().T)),
        "min_eigenvalue": float(np.linalg.eigvalsh((rho + rho.conj().T) / 2)[0]),
        "steady_state_residual_s_inverse": residual,
        "steady_state_relative_residual": residual / scale,
    }
    if (not all(np.isfinite(list(diagnostics.values())))
            or diagnostics["trace_error"] > 1e-9
            or diagnostics["hermiticity_error"] > 1e-9
            or diagnostics["min_eigenvalue"] < -1e-9
            or diagnostics["steady_state_relative_residual"] > 1e-9):
        raise np.linalg.LinAlgError(f"Optical full-OBE steady-state check failed: {diagnostics}")
    diagnostics["solver_status"] = "physical steady state"
    return diagnostics


def _linear_coherence_per_half_rabi(atom, excitation, laser_mhz, reset_rate):
    """Exact first-order rho_eg / (Omega/2) at zero drive and positive reset."""
    n_ground = len(atom.ground)
    coherence = np.empty_like(excitation)
    for ground_slice, energy in zip(atom.ground_slices, atom.ground_energy_mhz):
        detuning = (laser_mhz + energy - atom.excited_energy_mhz) * species.MHZ
        denominator = -detuning - 1j * (atom.gamma / 2 + reset_rate)
        coherence[:, ground_slice] = excitation[:, ground_slice] / (n_ground * denominator[:, None])
    return coherence


def solve_optical_pump(isotope, line, detuning_mhz, *, intensity_mw_cm2=0.1, jones_xy=(1., 0.),
                   velocity_m_s=0., temp_c=70., ground_relaxation_hz=1000., density_m3=None, solver="auto",
                       include_atomic_state=True, laser_frequency_hz=None):
    """Solve a complete local stationary Rb D-line atom at zero B.

    Positive detuning is blue of the upper-ground-F line-strength centroid.
    Positive velocity is along the incident beam. Intensity is the total
    local cycle-averaged power per area and ``jones_xy`` is a complex unit
    vector. Temperature supplies the vapor density when ``density_m3`` is
    omitted. Each call describes one velocity, without a Maxwell average.
    An explicit ``laser_frequency_hz`` supplies the physical optical frequency
    for the Doppler shift and propagation rates; when omitted, the selected
    atomic line frequency is retained for compatibility. It does not change
    the centroid reference of ``detuning_mhz``.

    ``solver="auto"`` uses exact sparse LU, with an analytic thermal state at
    zero drive. ``"full"`` is the independent dense reference. Setting
    ``include_atomic_state=False`` omits rho/H/generator/reservoir arrays from
    the returned dictionary while retaining all physical mean outputs and
    full-state residual/positivity diagnostics.

    Ground states precede excited states, each in ``species.level_structure``
    order. Hamiltonian entries are angular frequencies in rad/s, the reset
    rate is in s^-1. ``polarization_xy_c_m2`` is the macroscopic peak phasor
    with physical polarization Re[P exp(-i omega t)]. At zero intensity the
    polarization vanishes and chi is its analytic infinitesimal-probe limit.
    Circular component ratios are None when the corresponding field is absent.
    Azimuth/ellipticity angle rates are undefined at exactly circular input.
    """
    if isotope not in ("Rb85", "Rb87") or line not in ("D1", "D2"):
        raise ValueError("full optical OBE supports Rb85/Rb87 on D1/D2")
    if solver not in ("auto", "full"):
        raise ValueError("solver must be auto or full")
    detuning, intensity, velocity, temperature, reset = map(float, (
        detuning_mhz, intensity_mw_cm2, velocity_m_s, temp_c, ground_relaxation_hz))
    if not np.all(np.isfinite([detuning, intensity, velocity, temperature, reset])):
        raise ValueError("finite scalar full-OBE drive, detuning, velocity, temperature and reset required")
    if intensity < 0 or temperature <= -273.15 or reset <= 0:
        raise ValueError("nonnegative intensity, positive temperature and positive thermal reset required")
    density = (species.number_density(species.ISOTOPES[isotope], temperature + 273.15)
               if density_m3 is None else float(density_m3))
    if not np.isfinite(density) or density < 0:
        raise ValueError("density_m3 must be finite and nonnegative")

    jones = np.asarray(jones_xy, dtype=complex)
    circular = jones_to_circular(jones)
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    optical_frequency = atom.frequency_hz if laser_frequency_hz is None else float(laser_frequency_hz)
    if not np.isfinite(optical_frequency) or optical_frequency <= 0:
        raise ValueError("laser_frequency_hz must be finite and positive")
    centroid = magneto_multilevel.component_centroid_mhz(atom)
    effective_detuning = atom_frame_detuning_mhz(
        detuning, velocity_m_s=velocity, optical_frequency_hz=optical_frequency)
    laser_mhz = centroid + effective_detuning
    # E is the peak field. 1 mW/cm² = 10 W/m², Omega = d0 E / hbar.
    peak_field = np.sqrt(2 * intensity * 10 / (constants.EPS_0 * constants.C_LIGHT))
    rabi = np.sqrt(atom.dipole_sq) * peak_field / constants.HBAR
    excitation = sum(circular[q] * atom.dipoles[q].T for q in (-1, 1))
    hamiltonian = _hamiltonian(atom, laser_mhz, rabi, excitation)
    legacy = solver == "full" and isotope == "Rb87" and line == "D1"
    reservoirs, dissipator = _prepared_reservoirs(isotope, line, reset, legacy)
    with core.blas_single_thread():
        if solver == "full":
            dense_dissipator = dissipator.toarray() if sparse.issparse(dissipator) else dissipator
            generator = core.comm_super(hamiltonian) + dense_dissipator
            if not np.isfinite(generator).all():
                raise ValueError("settings exceed the finite full-OBE generator range")
            rho = core.steady_state_from_liouvillian(generator, reservoirs.n_levels)
        else:
            generator = _sparse_generator(atom, laser_mhz, rabi, circular, dissipator)
            if not np.isfinite(generator.data).all():
                raise ValueError("settings exceed the finite full-OBE generator range")
            if intensity == 0.:
                rho = np.diag(np.r_[np.full(len(atom.ground), 1. / len(atom.ground)),
                                   np.zeros(len(atom.excited))]).astype(complex)
            else:
                rho = _sparse_stationary_state(generator, reservoirs.n_levels)
        diagnostics = _state_diagnostics(generator, rho, reset)

    n_ground = len(atom.ground)
    optical_eg = rho[n_ground:, :n_ground]
    coherence = (optical_eg / (0.5 * rabi) if intensity > 0 else
                 _linear_coherence_per_half_rabi(atom, excitation, laser_mhz, reset))
    prefactor = density * atom.dipole_sq / (constants.EPS_0 * constants.HBAR)
    # P_q / (epsilon0 * |E|), defined even when the incident E_q is absent.
    induced = {q: prefactor * np.einsum("ge,eg->", atom.dipoles[q], coherence) for q in (-1, 1)}
    chi = {q: complex(induced[q] / circular[q]) if abs(circular[q]) > 1e-12 else None for q in (-1, 1)}
    chi_probe = sum(np.conj(circular[q]) * induced[q] for q in (-1, 1))
    induced_xy = circular_to_jones(induced[1], induced[-1])
    wave_number = 2 * np.pi * optical_frequency / constants.C_LIGHT
    epsilon, rotation, ellipticity_rate, alpha = stokes_rates(jones, 0.5j * wave_number * induced_xy)
    result = {
        "model_id": MODEL_ID,
        "ground_levels": atom.ground, "excited_levels": atom.excited,
        "ground_populations": rho.diagonal()[:n_ground].real.copy(),
        "excited_populations": rho.diagonal()[n_ground:].real.copy(),
        "excited_fraction": float(np.trace(rho[n_ground:, n_ground:]).real),
        "chi_plus": chi[1], "chi_minus": chi[-1], "chi_probe": complex(chi_probe),
        "polarization_xy_c_m2": constants.EPS_0 * peak_field * induced_xy,
        "alpha_per_m": alpha, "ellipticity_rad": epsilon,
        "azimuth_rate_per_m": rotation, "ellipticity_rate_per_m": ellipticity_rate,
        "diagnostics": diagnostics,
        "conditions": {
            "isotope": isotope, "line": line, "b_ut": 0., "velocity_m_s": velocity,
            "detuning_mhz": detuning, "effective_detuning_mhz": effective_detuning,
            "reference_centroid_mhz": centroid, "optical_frequency_hz": optical_frequency,
            "intensity_mw_cm2": intensity, "jones_xy": jones.copy(),
            "peak_field_v_m": float(peak_field), "rabi_rad_s": float(rabi),
            "temp_c": temperature, "ground_relaxation_hz": reset, "density_m3": float(density),
            "density_source": "pure-isotope vapor-pressure density" if density_m3 is None else "specified isotope density",
            "ground_f_coherences": "retained", "emission_secularization": "final Fg and q resolved, excited F coherent",
            "susceptibility_basis": "finite-drive induced polarization" if intensity > 0 else "zero-drive linear-probe limit",
        },
    }

    if include_atomic_state:
        result.update(rho=rho, hamiltonian_rad_s=hamiltonian,
                      generator_s_inverse=generator.toarray() if sparse.issparse(generator) else generator,
                      reservoirs=reservoirs, optical_coherence_eg=optical_eg.copy())
    return result
