"""
Zeeman manifold support for the magneto-optics schemes (Hanle / EIA / NMOR).

A builder that turns (F_g, F_e) into an `AtomModel` plus the
polarization-resolved dipole couplings (σ⁺, σ⁻, π) and CG-branched spontaneous
emission. Clebsch-Gordan coefficients come from `angular` (re-exported here as
`zeeman.clebsch_gordan`).
"""
import functools
import math

import numpy as np

from . import constants, hyperfine, species
from .angular import clebsch_gordan
from .atoms import AtomModel


def angular_momentum_matrices(F):
    """Return (Fx, Fy, Fz) in the |F,m> basis ordered m=-F..F.

    Matrices are dimensionless: H_B = omega_L * (Bxhat Fx + Byhat Fy + Bzhat Fz).
    """
    mvals = np.arange(-F, F + 1, 1, dtype=float)
    n = mvals.size
    Fp = np.zeros((n, n), dtype=complex)
    for col, m in enumerate(mvals):
        mp = m + 1
        if mp > F:
            continue
        row = int(round(mp + F))
        Fp[row, col] = math.sqrt(F * (F + 1) - m * mp)
    Fm = Fp.conj().T
    Fx = 0.5 * (Fp + Fm)
    Fy = (Fp - Fm) / (2j)
    Fz = np.diag(mvals.astype(complex))
    return Fx, Fy, Fz


def field_from_jones_frame(bx, by, bz=0.0):
    """Map a field given in the light's Jones (x, y, k) frame onto these matrices' frame.

    The circular drive weights used with the signed dipoles
    (psr_conventions.jones_to_circular, magneto_multilevel._polarization,
    schemes.magneto._qwp_drive_weights) omit the Condon-Shortley sign of the
    q=+1 component. Jones-x light is therefore y-linear here: its coupling
    commutes with F_y, not F_x. A +90 degree rotation about the beam axis
    maps the Jones frame onto this frame: Jones x -> y, Jones y -> -x; the
    beam axis is unchanged. Zero components stay +0.0.
    """
    return (-by if by else 0.0, bx, bz)


def zeeman_hamiltonian(F, g_f, field_t, *, multiply_g_before_hbar=False):
    """Linear Zeeman Hamiltonian in angular-frequency units for a field in tesla.

    The single-pair reference preserves its historical scalar evaluation order.
    A one-ulp difference can move pole diagnostics and displayed near-zero values.
    """
    fx, fy, fz = angular_momentum_matrices(F)
    bx, by, bz = field_t
    if multiply_g_before_hbar:
        larmor_per_t = (constants.MU_B * g_f) / constants.HBAR
    else:
        larmor_per_t = (constants.MU_B / constants.HBAR) * g_f
    return larmor_per_t * (bx * fx + by * fy + bz * fz)


@functools.lru_cache(maxsize=16)
def zeeman_manifold(Fg, Fe, gamma=None, gamma_gg=None, g_ratio=1.0,
                    transit_rate=0.0, optical_dephasing=0.0):
    """
    Build the (F_g ↔ F_e) Zeeman manifold as an AtomModel with extra attributes:
      m_ground, m_excited : magnetic quantum numbers per level index
      g_ratio             : excited/ground Landé g-factor ratio (excited Zeeman)
      couplings[q]        : list of (ground_idx, excited_idx, CG) for q ∈ {-1,0,+1}
    Spontaneous emission is CG-branched (Γ·|CG|² per channel, Σ = Γ from each
    excited); ground-ground Zeeman coherences dephase at γ_gg. All excited levels
    are Doppler-shifted (optical line); ground Zeeman is Doppler-free.

    `optical_dephasing` is an added elastic decay rate for every ground-excited
    coherence. It does not alter excited-state populations, spontaneous-emission
    branching, or transfer of coherence (TOC). A collisional optical FWHM
    contribution therefore enters here at half that FWHM.

    `transit_rate` optionally adds isotropic population reload into the addressed
    ground manifold. It is a compact warm-vapor approximation for atoms entering
    and leaving the beam, useful for practical Hanle/EIA curves where optical
    pumping would otherwise trap all population.
    """
    gamma = constants.GAMMA if gamma is None else gamma
    gamma_gg = constants.GAMMA_GG if gamma_gg is None else gamma_gg

    mg = list(range(-Fg, Fg + 1))
    me = list(range(-Fe, Fe + 1))
    ng, ne = len(mg), len(me)
    n = ng + ne
    ground = tuple(range(ng))
    excited = tuple(range(ng, n))

    # Spontaneous emission as polarization-grouped jump operators Σ_q (q = m_e−m_g):
    #   Σ_q = √Γ Σ ⟨Fg m_g; 1 q | Fe m_e⟩ |Fg m_g⟩⟨Fe m_e|.
    # One operator per q (not one per channel) so that D[Σ_q] carries transfer of
    # coherence: a ground Zeeman coherence |g⟩⟨g'| is refed from an excited
    # coherence ρ_{ee'} sharing the same emitted-photon polarization. This is what
    # turns cycling-type transitions into EIA/LCA. Σ_q^†Σ_q sums to Γ·P_excited, so
    # excited decay and ground-population refilling are byte-identical to the old
    # per-channel form; only the off-diagonal (TOC) gain is new.
    sqrt_gamma = math.sqrt(gamma)
    emission = {q: np.zeros((n, n), dtype=complex) for q in (-1, 0, 1)}
    couplings = {-1: [], 0: [], +1: []}
    dipoles = species.signed_dipole_block(Fg, Fe)
    for ie, m_e in enumerate(me):
        eidx = ng + ie
        for ig, m_g in enumerate(mg):
            q = m_e - m_g
            if q not in (-1, 0, 1):
                continue
            cg = float(dipoles[q][ig, ie])
            if abs(cg) < 1e-14:
                continue
            emission[q][ig, eidx] = sqrt_gamma * cg      # |Fg m_g⟩⟨Fe m_e|
            couplings[q].append((ig, eidx, cg))          # drive ground→excited, pol q
    emission_ops = tuple(emission[q] for q in (-1, 0, 1))

    collapse_ops = []
    if gamma_gg and gamma_gg > 0:
        # Projective ground-state phase randomization is completely positive:
        # populations are unchanged, each ground-ground coherence decays at
        # gamma_gg, and each optical ground-excited coherence at gamma_gg / 2.
        sqrt_gamma_gg = math.sqrt(gamma_gg)
        for g in ground:
            projector = np.zeros((n, n), dtype=complex)
            projector[g, g] = sqrt_gamma_gg
            collapse_ops.append(projector)
    if optical_dephasing and optical_dephasing > 0:
        # One excited-block projector preserves populations and excited-state
        # Zeeman coherence while damping every optical coherence at the declared
        # elastic rate. It is not an emission/TOC operator.
        projector = np.zeros((n, n), dtype=complex)
        for e in excited:
            projector[e, e] = math.sqrt(2.0 * optical_dephasing)
        collapse_ops.append(projector)

    atom = AtomModel(
        name=f"zeeman_{Fg}_{Fe}", n_levels=n,
        labels=tuple(f"g{m:+d}" for m in mg) + tuple(f"e{m:+d}" for m in me),
        ground=ground, excited=excited,
        decay=(), dephasing=(), doppler_levels=excited,
        emission_ops=emission_ops, collapse_ops=tuple(collapse_ops),
    )
    if transit_rate and transit_rate > 0:
        # Incoherent, isotropic population reload (transit through the beam).
        atom = atom.with_reset(transit_rate, [1.0 / ng] * ng)
    atom.m_ground = np.array(mg, dtype=float)
    atom.m_excited = np.array(me, dtype=float)
    atom.g_ratio = g_ratio
    atom.couplings = couplings
    return atom


def rb85_d1_double_lambda_zeeman(gamma=None, gamma_gg=None):
    """Full 85Rb D1 double-lambda Zeeman manifold used by FWM Ultra diagnostics.

    Levels are ordered as all ground states F=2,3 followed by all excited states
    F'=2,3. Drive and spontaneous emission use the same signed C_q values.
    The separate Ultra diagnostic projects their strengths onto the reduced
    FWM normalization.
    """
    gamma = constants.GAMMA if gamma is None else gamma
    gamma_gg = constants.GAMMA_GG if gamma_gg is None else gamma_gg

    structure = species.level_structure("Rb85", "D1")
    sub_g, sub_e = structure.ground, structure.excited
    labels = tuple(f"gF{F}m{m:+d}" for F, m in sub_g) + tuple(f"eF{F}m{m:+d}" for F, m in sub_e)
    ng = len(sub_g)
    n = len(labels)
    ground = tuple(range(ng))
    excited = tuple(range(ng, n))

    couplings = {q: [] for q in structure.dipoles}
    strengths = {pair: [] for pair in hyperfine.CF2}
    for Fg, Fe in strengths:
        for gi, (fg, mg) in enumerate(sub_g):
            if fg != Fg:
                continue
            for ei, (fe, me) in enumerate(sub_e):
                q = me - mg
                if fe == Fe and q in structure.dipoles:
                    c = float(structure.dipoles[q][gi, ei])
                    if c:
                        couplings[q].append((gi, ng + ei, c))
                        strengths[(Fg, Fe)].append(c**2)
    # Comparing the sum over sublevels and polarizations divided by three
    # against 3*C_F^2 checks the reduced convention without rescaling a drive.
    block_sum = {pair: math.fsum(values)/3.0 for pair, values in strengths.items()}

    emission = []
    sqrt_gamma = math.sqrt(gamma)
    for dipole in structure.dipoles.values():
        jump = np.zeros((n, n), dtype=complex)
        jump[:ng, ng:] = sqrt_gamma * dipole
        emission.append(jump)

    dephasing = tuple((i, j, gamma_gg) for i in ground for j in ground if i != j)
    atom = AtomModel(
        name="rb85_d1_double_lambda_zeeman",
        n_levels=n,
        labels=labels,
        ground=ground,
        excited=excited,
        decay=(),
        dephasing=dephasing,
        doppler_levels=excited,
        emission_ops=tuple(emission),
    )
    atom.level_F = np.array([F for F, _ in sub_g + sub_e], dtype=int)
    atom.level_m = np.array([m for _, m in sub_g + sub_e], dtype=float)
    atom.couplings = couplings
    atom.lumped_strengths = block_sum
    ratios = [
        block_sum[key] / (3.0 * hyperfine.CF2[key])
        for key in hyperfine.CF2
    ]
    atom.lumped_strength_correction = float(np.mean(ratios))
    atom.ultra_status = (
        "24-level 85Rb D1 Zeeman manifold available for CG diagnostics")
    return atom
