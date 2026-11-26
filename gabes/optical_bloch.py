"""Saturated single-velocity optical Bloch response for Rb D1 and D2.

The complete rotating-wave Hamiltonian retains ground- and excited-hyperfine
coherences. Emission distinguishes final ground F and spherical polarization
but coherently sums excited-F and magnetic-sublevel paths, as in the weak PSR
pumping model. Thermal replacement acts on the entire density operator. The
state, Hamiltonian and named jumps are available for subsequent linearization.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import splu

from . import constants, core, magneto_multilevel, species, zeeman
from .psr_conventions import atom_frame_detuning_mhz, circular_to_jones, jones_to_circular, stokes_rates
from .quantum.reservoirs import CollapseChannel, ExplicitReservoirs, thermal_reset_channels


MODEL_ID = "rb_dline_single_velocity_full_obe_v1"
_LINEAR_ELLIPTICITY_ATOL = 1e-13


@dataclass(frozen=True)
class _PiMirrorModel:
    """Exact real Hermitian M-even pi-frame representation at zero magnetic field."""

    n_levels: int
    n_ground: int
    dm0: np.ndarray
    u_even: np.ndarray
    rx_half_pi: np.ndarray
    m_levels: np.ndarray
    l_emiss: np.ndarray
    l_reset_unit: np.ndarray
    l_base: np.ndarray
    l_det: np.ndarray
    l_pi_unit: np.ndarray
    s_doppler: np.ndarray
    trace_vec: np.ndarray
    w_chi_unit: np.ndarray
    w_exc: np.ndarray


def _is_linear_jones(jones):
    """True when unit Jones vector (Ex, Ey) has zero circular Stokes component S3."""
    s3 = 2.0 * (jones[0].conjugate() * jones[1]).imag
    return abs(s3) <= _LINEAR_ELLIPTICITY_ATOL


def _build_rx_half_pi(atom):
    """Block-diagonal SO(3) rotation exp(-i * (pi/2) * F_x) on each hyperfine F manifold."""
    ng, ne = len(atom.ground), len(atom.excited)
    n = ng + ne
    rot = np.zeros((n, n), dtype=complex)
    for sl in atom.ground_slices:
        f = atom.ground[sl.start][0]
        fx, _, _ = zeeman.angular_momentum_matrices(f)
        rot[sl, sl] = expm(-0.5j * math.pi * fx)
    idx = ng
    while idx < n:
        f = atom.excited[idx - ng][0]
        dim = 2 * f + 1
        fx, _, _ = zeeman.angular_momentum_matrices(f)
        rot[idx:idx + dim, idx:idx + dim] = expm(-0.5j * math.pi * fx)
        idx += dim
    rot.flags.writeable = False
    return rot


def _pi_to_z_unitary(model, circular):
    """Exact unitary U mapping pi-quantized H_pi onto z-quantized linear H_z(circular)."""
    cp, cm = circular[1], circular[-1]
    ratio = cp / cm
    delta = 0.5 * math.atan2(ratio.imag, ratio.real)
    z0 = cp * np.exp(-1j * delta)
    gphase = math.atan2(z0.imag, z0.real)
    diag = np.exp(1j * delta * model.m_levels)
    diag[model.n_ground:] *= 1j * np.exp(1j * gphase)
    return diag[:, None] * model.rx_half_pi


def _build_pi_mirror_basis(atom):
    """Orthonormal real Hermitian M-even basis for the Delta_m = 0 sector."""
    ng, ne = len(atom.ground), len(atom.excited)
    n = ng + ne
    levels = list(atom.ground) + list(atom.excited)
    ms = np.array([m for _, m in levels], dtype=int)
    fs = np.array([f for f, _ in levels], dtype=int)
    is_exc = np.arange(n) >= ng
    index_of = {(bool(is_exc[i]), int(fs[i]), int(ms[i])): i for i in range(n)}
    dm0 = np.flatnonzero((ms[:, None] - ms[None, :]).ravel() == 0)
    cols = []
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # Diagonal populations |i><i| paired under m <-> -m
    for i in range(n):
        if ms[i] < 0:
            continue
        v = np.zeros((n, n), dtype=complex)
        if ms[i] == 0:
            v[i, i] = 1.0
        else:
            jm = index_of[(bool(is_exc[i]), int(fs[i]), int(-ms[i]))]
            v[i, i] = inv_sqrt2
            v[jm, jm] = inv_sqrt2
        cols.append(v.ravel()[dm0])

    # Off-diagonal Delta_m = 0 coherences paired under mirror reflection M
    for i in range(n):
        for j in range(i + 1, n):
            if ms[i] != ms[j] or ms[i] < 0:
                continue
            pi = -1 if (int(fs[i]) + int(is_exc[i])) % 2 else 1
            pj = -1 if (int(fs[j]) + int(is_exc[j])) % 2 else 1
            sgn = pi * pj
            if ms[i] == 0:
                if sgn != 1:
                    continue
                v_re = np.zeros((n, n), dtype=complex)
                v_re[i, j] = inv_sqrt2
                v_re[j, i] = inv_sqrt2
                v_im = np.zeros((n, n), dtype=complex)
                v_im[i, j] = 1j * inv_sqrt2
                v_im[j, i] = -1j * inv_sqrt2
                cols.append(v_re.ravel()[dm0])
                cols.append(v_im.ravel()[dm0])
            else:
                im = index_of[(bool(is_exc[i]), int(fs[i]), int(-ms[i]))]
                jm = index_of[(bool(is_exc[j]), int(fs[j]), int(-ms[j]))]
                v_re = np.zeros((n, n), dtype=complex)
                v_re[i, j] = 0.5
                v_re[j, i] = 0.5
                v_re[im, jm] = 0.5 * sgn
                v_re[jm, im] = 0.5 * sgn
                v_im = np.zeros((n, n), dtype=complex)
                v_im[i, j] = 0.5j
                v_im[j, i] = -0.5j
                v_im[im, jm] = 0.5j * sgn
                v_im[jm, im] = -0.5j * sgn
                cols.append(v_re.ravel()[dm0])
                cols.append(v_im.ravel()[dm0])

    u_even = np.ascontiguousarray(np.column_stack(cols), dtype=complex)
    return dm0, u_even, ms


@lru_cache(maxsize=4)
def _pi_mirror_model(isotope, line):
    """Precompute static reduced M-even pi-frame operators once per (isotope, line)."""
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    ng, ne = len(atom.ground), len(atom.excited)
    n = ng + ne
    dm0, u_even, ms = _build_pi_mirror_basis(atom)
    uh = u_even.conj().T

    def project_real(full_super):
        block = full_super[np.ix_(dm0, dm0)]
        proj = uh @ block @ u_even
        arr = np.ascontiguousarray(proj.real, dtype=float)
        arr.flags.writeable = False
        return arr

    # Spontaneous emission and unit thermal reset superoperators
    channels = []
    for ground_slice in atom.ground_slices:
        ground_f = atom.ground[ground_slice.start][0]
        for q in (-1, 0, 1):
            jump = np.zeros((n, n), complex)
            jump[ground_slice, ng:] = np.sqrt(atom.gamma) * atom.dipoles[q][ground_slice]
            channels.append(CollapseChannel(
                f"emission:F={ground_f}:q={q}", jump, "spontaneous_emission",
                f"{isotope} {line} signed dipoles and natural width from species.level_structure",
            ))
    thermal_populations = np.zeros(n)
    thermal_populations[:ng] = 1.0 / ng
    l_emiss = project_real(ExplicitReservoirs(n, tuple(channels)).dissipator())
    l_reset_unit = project_real(core.thermal_reset_superoperator(1.0, thermal_populations))

    zero = np.zeros((ne, ng), complex)
    base_h = _hamiltonian(atom, 0., 0., zero)
    det_h = np.diag(np.r_[np.zeros(ng), np.full(ne, -species.MHZ)])
    pi_h = _hamiltonian(atom, 0., 1., atom.dipoles[0].T) - base_h
    doppler_h = np.diag(np.r_[np.zeros(ng), np.ones(ne)])

    l_base = project_real(core.comm_super(base_h))
    l_det = project_real(core.comm_super(det_h))
    l_pi_unit = project_real(core.comm_super(pi_h))
    # H(D) = H(0) + D * P_e => comm_super(D * P_e) = -D * S_doppler
    s_doppler = project_real(-core.comm_super(doppler_h))

    trace_vec = np.ascontiguousarray((np.eye(n).ravel()[dm0] @ u_even).real, dtype=float)
    w_full = np.zeros((n, n), dtype=complex)
    w_full[:ng, ng:] = atom.dipoles[0]
    w_chi_unit = np.ascontiguousarray(w_full.T.ravel()[dm0] @ u_even, dtype=complex)
    w_exc = np.ascontiguousarray((np.diag(np.r_[np.zeros(ng), np.ones(ne)]).ravel()[dm0] @ u_even).real, dtype=float)
    rx_half_pi = _build_rx_half_pi(atom)
    for arr in (dm0, u_even, ms, trace_vec, w_chi_unit, w_exc):
        arr.flags.writeable = False
    return _PiMirrorModel(
        n_levels=n, n_ground=ng, dm0=dm0, u_even=u_even,
        rx_half_pi=rx_half_pi, m_levels=ms,
        l_emiss=l_emiss, l_reset_unit=l_reset_unit,
        l_base=l_base, l_det=l_det, l_pi_unit=l_pi_unit,
        s_doppler=s_doppler, trace_vec=trace_vec,
        w_chi_unit=w_chi_unit, w_exc=w_exc,
    )


def _assemble_pi_mirror_generator(model, laser_mhz, rabi, reset):
    """Assemble the real M-even pi-frame Liouvillian at (laser_mhz, rabi, reset)."""
    return (model.l_emiss + reset * model.l_reset_unit
            + model.l_base + laser_mhz * model.l_det + rabi * model.l_pi_unit)


def _solve_pi_mirror_state(model, l_even, reset):
    """Solve the reduced real M-even system and return (rho_pi, x_even, diagnostics)."""
    if not np.isfinite(l_even).all():
        raise ValueError("settings exceed the finite full-OBE generator range")
    dim = l_even.shape[0]
    scale = max(float(np.max(np.abs(l_even))), 1.0)
    matrix = l_even.copy()
    matrix[0, :] = model.trace_vec * scale
    rhs = np.zeros(dim, dtype=float)
    rhs[0] = scale
    x_even = np.linalg.solve(matrix, rhs)
    rho_vec = np.zeros(model.n_levels * model.n_levels, dtype=complex)
    rho_vec[model.dm0] = model.u_even @ x_even
    rho_pi = rho_vec.reshape(model.n_levels, model.n_levels)
    diagnostics = _reduced_state_diagnostics(l_even, x_even, rho_pi, reset)
    return rho_pi, x_even, diagnostics


def _reduced_state_diagnostics(l_even, x_even, rho_pi, reset_rate):
    """Exact unitary-invariant steady-state physicality diagnostics in the M-even basis."""
    residual = float(np.linalg.norm(l_even @ x_even))
    scale = max(float(np.linalg.norm(np.abs(l_even) @ np.abs(x_even))),
                reset_rate * float(np.linalg.norm(x_even)))
    diagnostics = {
        "trace_error": float(abs(np.trace(rho_pi) - 1)),
        "hermiticity_error": float(np.linalg.norm(rho_pi - rho_pi.conj().T)),
        "min_eigenvalue": float(np.linalg.eigvalsh((rho_pi + rho_pi.conj().T) / 2)[0]),
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
    use_pi_mirror = (solver == "auto" and intensity > 0. and _is_linear_jones(jones))
    with core.blas_single_thread():
        if solver == "full":
            reservoirs, dissipator = _prepared_reservoirs(isotope, line, reset, legacy)
            dense_dissipator = dissipator.toarray() if sparse.issparse(dissipator) else dissipator
            generator = core.comm_super(hamiltonian) + dense_dissipator
            if not np.isfinite(generator).all():
                raise ValueError("settings exceed the finite full-OBE generator range")
            rho = core.steady_state_from_liouvillian(generator, reservoirs.n_levels)
            diagnostics = _state_diagnostics(generator, rho, reset)
        elif use_pi_mirror:
            model = _pi_mirror_model(isotope, line)
            l_even = _assemble_pi_mirror_generator(model, laser_mhz, rabi, reset)
            try:
                rho_pi, _, diagnostics = _solve_pi_mirror_state(model, l_even, reset)
                u_rot = _pi_to_z_unitary(model, circular)
                rho = u_rot @ rho_pi @ u_rot.conj().T
                if include_atomic_state:
                    reservoirs, dissipator = _prepared_reservoirs(isotope, line, reset, False)
                    generator = _sparse_generator(atom, laser_mhz, rabi, circular, dissipator)
            except np.linalg.LinAlgError:
                reservoirs, dissipator = _prepared_reservoirs(isotope, line, reset, False)
                generator = _sparse_generator(atom, laser_mhz, rabi, circular, dissipator)
                if not np.isfinite(generator.data).all():
                    raise ValueError("settings exceed the finite full-OBE generator range")
                rho = _sparse_stationary_state(generator, reservoirs.n_levels)
                diagnostics = _state_diagnostics(generator, rho, reset)
        else:
            reservoirs, dissipator = _prepared_reservoirs(isotope, line, reset, False)
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
