"""Beam crossings in a static magnetic field with the full sublevel density matrix.

One Rb isotope and D line, a single transverse Gaussian beam, a uniform static
field B in any direction and any transverse Jones polarization. The quantization
axis is the beam axis k. Ground and excited Zeeman terms use g_F from
``species.lande_gf``. Spontaneous emission comes from
``optical_bloch._prepared_reservoirs`` with zero reset; a declared optical pure
dephasing adds the homogeneous HWHM increment (self-broadening by default in the
callers). Atoms enter with a uniform ground population and cross on straight
lines at a fixed rest-frame detuning. No reset acts during a crossing.

Scope: uncoated, buffer-free, one beam, pump off, no field gradients. The
zero-field population projection in ``ballistic_atom`` is the B = 0, linear
light special case of this model.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from . import constants, core, magneto_multilevel, species, zeeman
from .optical_bloch import _hamiltonian, _prepared_reservoirs, jones_to_circular

SDIRK_DIAGONAL = 1.0 - 1.0 / math.sqrt(2.0)


def _excited_gj(iso, line):
    return iso.gJ_P12 if line == "D1" else iso.gJ_P32


def _excited_j(line):
    return 0.5 if line == "D1" else 1.5


def _zeeman_block(states, g_values, field_t):
    """Block-diagonal Zeeman Hamiltonian [rad/s] for (F, m) states ordered m=-F..F."""
    n = len(states)
    result = np.zeros((n, n), complex)
    start = 0
    while start < n:
        f = states[start][0]
        size = int(round(2 * f + 1))
        block = zeeman.zeeman_hamiltonian(f, g_values[f], field_t)
        result[start:start + size, start:start + size] = block
        start += size
    return result


@dataclass(frozen=True)
class FieldCrossingOperators:
    """Sparse generators for one isotope, line, field and polarization."""

    isotope: str
    line: str
    base: sparse.csc_matrix      # free evolution + Zeeman + dissipation, no light
    driver: sparse.csc_matrix    # light coupling per unit Rabi frequency [rad/s]
    initial: np.ndarray          # vectorized uniform ground state
    excited_population: np.ndarray
    trace: np.ndarray
    population_rows: np.ndarray
    gamma: float
    dipole_sq: float


@lru_cache(maxsize=32)
def _operators(isotope, line, rest_mhz, field_ut, jones, optical_hwhm_increment_rad_s,
               quantization):
    iso = species.ISOTOPES[isotope]
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    ng, ne = len(atom.ground), len(atom.excited)
    n = ng + ne
    if quantization == "k":
        circular = jones_to_circular(np.asarray(jones, dtype=complex))
        excitation = sum(circular[q] * atom.dipoles[q].T for q in (-1, 1))
        # field_ut is in the Jones frame; see zeeman.field_from_jones_frame.
        field = np.asarray(zeeman.field_from_jones_frame(*field_ut), dtype=float) * 1e-6
    elif quantization == "E":
        # Linear light along x only: quantize along E (pi light). Right-handed
        # relabelling z' = x, x' = y, y' = z maps B = (Bx, By, Bz) to (By, Bz, Bx).
        if not np.allclose(np.asarray(jones, dtype=complex), (1.0, 0.0)):
            raise ValueError("E quantization is defined for x-linear light only.")
        excitation = atom.dipoles[0].T
        bx, by, bz = field_ut
        field = np.asarray((by, bz, bx), dtype=float) * 1e-6
    else:
        raise ValueError("quantization must be 'k' or 'E'.")

    h0 = _hamiltonian(atom, rest_mhz, 0.0, excitation)
    h1 = _hamiltonian(atom, rest_mhz, 1.0, excitation) - h0
    ground_g = {f: species.lande_gf(iso.gJ_S, iso.g_I, iso.I, iso.Jg, f)
                for f, _ in atom.ground}
    excited_g = {f: species.lande_gf(_excited_gj(iso, line), iso.g_I, iso.I, _excited_j(line), f)
                 for f, _ in atom.excited}
    if np.any(field):
        h0 = h0.copy()
        h0[:ng, :ng] += _zeeman_block(atom.ground, ground_g, field)
        h0[ng:, ng:] += _zeeman_block(atom.excited, excited_g, field)

    excited = np.arange(n) >= ng
    optical = (excited[:, None] != excited).ravel(order="F").astype(float)
    dissipator = _prepared_reservoirs(isotope, line, 0.0)[1]
    if optical_hwhm_increment_rad_s:
        dissipator = dissipator + sparse.diags(-optical_hwhm_increment_rad_s * optical)
    base = (sparse.csc_matrix(core.comm_super(h0)) + dissipator).tocsc()
    driver = sparse.csc_matrix(core.comm_super(h1))

    initial = np.diag(np.r_[np.full(ng, 1.0 / ng), np.zeros(ne)]).ravel(order="F").astype(complex)
    _, labels = connected_components(abs(base) + abs(driver), directed=False)
    use = np.flatnonzero(np.isin(labels, labels[initial != 0]))
    trace = np.eye(n).ravel(order="F")[use]
    return FieldCrossingOperators(
        isotope, line, base[use, :][:, use].tocsc(), driver[use, :][:, use].tocsc(),
        initial[use], np.diag(excited.astype(float)).ravel(order="F")[use], trace,
        np.flatnonzero(trace == 1.0), atom.gamma, atom.dipole_sq)


def operators(isotope, line, rest_mhz, *, field_ut=(0.0, 0.0, 0.0), jones=(1.0, 0.0),
              optical_hwhm_increment_rad_s=0.0, quantization="k"):
    """Sparse crossing generators; arguments are cached by value."""
    if isotope not in ("Rb85", "Rb87") or line not in ("D1", "D2"):
        raise ValueError("Field crossings support Rb85/Rb87 on D1/D2.")
    jones = tuple(complex(v) for v in jones)
    norm = math.sqrt(sum(abs(v) ** 2 for v in jones))
    if not norm:
        raise ValueError("The Jones vector must be nonzero.")
    jones = tuple(v / norm for v in jones)
    field = tuple(float(v) for v in field_ut)
    if len(field) != 3 or not all(math.isfinite(v) for v in field):
        raise ValueError("field_ut must be three finite components.")
    return _operators(isotope, line, float(rest_mhz), field, jones,
                      float(optical_hwhm_increment_rad_s), quantization)


def rabi_per_sqrt_intensity(ops):
    """Rabi frequency [rad/s] per sqrt(W/m^2) for the reduced dipole d0."""
    return math.sqrt(ops.dipole_sq * 2.0 / (constants.EPS_0 * constants.C_LIGHT)) / constants.HBAR


def crossing(ops, peak_intensity_w_m2, waist_m, speed_m_s, *, impact_over_waist=0.0,
             steps=1536, half_span_waist=4.0):
    """Absorbed photons for one straight crossing through a Gaussian beam.

    The field amplitude along the path is exp(-(x^2 + b^2)/w^2), x in waists.
    SDIRK-2 with two stage solves per step. Excited population left at the
    exit is counted as emitted.
    """
    if steps < 2 or speed_m_s <= 0 or waist_m <= 0 or peak_intensity_w_m2 < 0:
        raise ValueError("positive steps, speed and waist and nonnegative intensity required")
    rabi0 = rabi_per_sqrt_intensity(ops) * math.sqrt(peak_intensity_w_m2) \
        * math.exp(-impact_over_waist ** 2)
    d = SDIRK_DIAGONAL
    dx = 2.0 * half_span_waist / steps
    dt = waist_m / speed_m_s * dx
    x = -half_span_waist + dx * np.arange(steps)
    stage1 = np.exp(-(x + d * dx) ** 2)
    stage2 = np.exp(-(x + dx) ** 2)
    eye = sparse.eye(ops.base.shape[0], format="csc")
    state = ops.initial.copy()
    photons = 0.0
    trace_error = 0.0
    min_population = 1.0
    for a, b in zip(stage1, stage2):
        u = spsolve(eye - (d * dt) * (ops.base + (rabi0 * a) * ops.driver), state)
        q = spsolve(eye - (d * dt) * (ops.base + (rabi0 * b) * ops.driver),
                    state + (1.0 - d) / d * (u - state))
        photons += dt * ops.gamma * ((1.0 - d) * (ops.excited_population @ u).real
                                     + d * (ops.excited_population @ q).real)
        min_population = min(min_population, float(q[ops.population_rows].real.min()))
        trace_error = max(trace_error, float(abs(ops.trace @ q - 1.0)))
        state = q
    photons += float((ops.excited_population @ state).real)
    return {"photons": photons, "trace_error": trace_error, "min_population": min_population,
            "steps": steps}
