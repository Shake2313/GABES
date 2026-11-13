"""
Cluster A — weak-probe absorption / dispersion schemes.

OD  : 2-level Doppler-broadened absorption (the validation backbone).
EIT : Λ system, weak probe + strong coupling → transparency window + dispersion.
AT  : same Λ in the strong-coupling regime → Autler-Townes doublet (split ≈ Ω_c).
CPT : same Λ, narrow two-photon scan → sub-natural dark resonance.

All share one engine: build a rotating-frame H₀ with a weak probe (and optional
strong coupling), solve the steady state per velocity class (core.steady_state_
batched), and read the probe coherence ρ_eg/Ω_p = χ̄. Co-propagating geometry is
the default: the optical line is Doppler-broadened and the Λ two-photon
resonance retains the small longitudinal wave-vector difference. Angled beams
can use a scalar residual shift or a joint Maxwell reference calculation.

Spectroscopic knobs are in units of Γ (natural linewidth) so the physics is
atom-agnostic and validation is direct (AT splitting reads Ω_c). Temperature sets
the Doppler width and the 85Rb number density (hence the absorption scale).
"""
import numpy as np

from .. import atoms, beam, constants, doppler, hyperfine, observables, species
from ..constants import GAMMA, K_VEC, OMEGA_D1
from .. import affine_doppler, core
from ..lineshape import MIN_SUBDOPPLER_EDGE_CLEARANCE_FWHM, fwhm_halfmax, level_crossing
from ..report import derived_table
from .base import ParamSpec, Preset, Scheme

PROBE_RABI = 1e-3              # weak probe, in units of Γ
GAMMA_MHZ = GAMMA / (2 * np.pi) / 1e6
MHZ = 2 * np.pi * 1e6
KHZ = 2 * np.pi * 1e3

# numpy ≥ 2.0 renamed np.trapz → np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


_TABLE_STEP = GAMMA / 30.0          # Δ_eff sampling for the Doppler χ̄ table


def _ne_buffer_gamma(params):
    return constants.neon_buffer_broadening(params.get("ne_pressure_torr", 0.0))


def _solve_chi_avg(atom, build_H0, probe_coh, scan, params, h_dep,
                   probe_omega=None, k_vec=None, mass=None, scan_shift=None):
    """
    χ̄(scan) = probe coherence ρ_eg / Ω_p, Doppler-averaged over the Maxwell
    velocity distribution (optical shift plus optional per-level residual shifts).

    Doppler off: a single v = 0 solve per scan point.
    Doppler on, scan-independent H (OD): one fine χ̄(Δ_eff) table + interpolated
        Doppler average — decouples velocity sampling from the solve (accurate
        Voigt without millions of solves), mirroring the FWM Δ_eff trick.
    Doppler on, scan-dependent H (Λ): the exact Maxwell average of every scan
        point (`AffineScan.maxwell_mean`). The Λ two-photon feature is Doppler-free
        unless atom.S_v carries an optional residual two-photon k-ratio.
    """
    e, g = probe_coh
    om_p = PROBE_RABI * GAMMA if probe_omega is None else probe_omega
    k_vec = K_VEC if k_vec is None else k_vec
    mass = constants.MASS_85RB if mass is None else mass
    scan_shift = atom.S_v if scan_shift is None else scan_shift
    n = atom.n_levels
    doppler_on = params.get("doppler", "off") == "on"

    if not doppler_on:
        if not h_dep:
            L0 = core.build_liouvillian(build_H0(0.0), atom)
            rho = core.steady_state_batched(L0, scan, atom.S_v, n)
            return rho[:, e, g] / om_p
        # Scan-dependent H, single (v = 0) class: Δ_eff = s.
        return affine_doppler.AffineScan(
            atom, build_H0, probe_coh, om_p, scan_shift).stationary(scan)

    T = params["temp_c"] + 273.15

    if not h_dep:
        # OD: scan-independent H solved once on a fine Δ_eff table, then averaged.
        # The solve cost is independent of the velocity grid, so keep it fine
        # (dv = 1) — this path feeds the AutoOD-validated absolute scale.
        v, w = doppler.velocity_grid(T, mass=mass, dv=1.0, cutoff_sigma=4.0)
        kvmax = k_vec * np.abs(v).max()
        L0 = core.build_liouvillian(build_H0(0.0), atom)
        lo, hi = scan.min() - kvmax, scan.max() + kvmax
        deff = np.linspace(lo, hi, int((hi - lo) / _TABLE_STEP) + 2)
        rho = core.steady_state_batched(L0, deff, atom.S_v, n)
        table = rho[:, e, g] / om_p
        return doppler.doppler_average_1d(table, deff, scan, v, w, k_vec=k_vec)

    # Scan-dependent H (Λ): the optical background and any optional residual
    # two-photon Doppler, per scan point.
    return affine_doppler.AffineScan(
        atom, build_H0, probe_coh, om_p, scan_shift).maxwell_mean(scan, T, mass, k_vec)


def _hyperfine_alpha(scan, params, line_strength=None):
    """
    Realistic 85Rb D1 absorption coefficient α(scan) [1/m]: an incoherent sum of
    the four hyperfine transitions (Fg∈{2,3}→Fe∈{2,3}), each a Doppler-broadened
    Voigt placed at its line-center detuning, weighted by the validated
    Clebsch-Gordan strength C_F² and thermal sublevel population, with self-broadening.

    The Voigt *shape* is produced once by the OBE Doppler kernel (a genuine OBE
    solve — no analytic wofz) on a fine grid, normalised to unit area, then
    shifted/scaled per line. Since C_F² sums initial magnetic sublevels,
    the thermal population is included exactly once:
        ∫α_F dδ = π · k · C_F² · |d|² · N / (ε₀ℏ) / (2(2I+1)).

    Returns (alpha, components, info) where `components` maps (Fg,Fe)→α_F(scan)
    and `info` carries N, Γ_eff, σ_v, Doppler FWHM for the readout.
    """
    T = params["temp_c"] + 273.15
    doppler_on = params.get("doppler", "on") == "on"
    ls = (params.get("line_strength", 1.0)
          if line_strength is None else float(line_strength))

    N = hyperfine.number_density(T)
    buffer_gamma = _ne_buffer_gamma(params)
    gamma_eff = hyperfine.self_broadened_gamma(N) + buffer_gamma
    atom = atoms.two_level(gamma=gamma_eff)

    def build_H0(_s):
        H = np.zeros((2, 2), dtype=complex)
        H[0, 1] = H[1, 0] = PROBE_RABI * GAMMA / 2
        return H

    sigma_v = np.sqrt(constants.KB * T / constants.MASS_85RB)
    dopp_fwhm = np.sqrt(8 * np.log(2)) * K_VEC * sigma_v
    width = dopp_fwhm if doppler_on else gamma_eff

    shifts = {k: 2 * np.pi * v for k, v in hyperfine.LINE_SHIFT_HZ.items()}
    smin, smax = min(shifts.values()), max(shifts.values())

    # One unit-area Voigt shape on a fine grid covering (scan − shift) ± wings.
    lo = scan.min() - smax - 12 * width
    hi = scan.max() - smin + 12 * width
    grid = np.linspace(lo, hi, int((hi - lo) / (width / 40.0)) + 2)
    chi = _solve_chi_avg(atom, build_H0, (1, 0), grid, params, h_dep=False)
    shape, _ = observables.absorption_coefficient(chi, K_VEC, 1.0, line_strength=1.0)
    S_unit = shape / _trapz(shape, grid)          # ∫ S_unit dδ = 1

    # Absolute per-line integrated absorption (AutoOD normalisation).
    K = (np.pi * K_VEC * hyperfine.DIPOLE_SQ * N
         / (constants.HBAR * constants.EPS_0) / hyperfine.N_GROUND_SUBLEVELS)

    alpha = np.zeros_like(scan)
    components = {}
    for (Fg, Fe) in hyperfine.TRANSITIONS:
        I_line = ls * K * hyperfine.CF2[(Fg, Fe)]
        line = I_line * np.interp(scan - shifts[(Fg, Fe)], grid, S_unit,
                                  left=0.0, right=0.0)
        alpha += line
        components[(Fg, Fe)] = line

    info = dict(N=N, gamma_eff=gamma_eff, buffer_gamma=buffer_gamma,
                sigma_v=sigma_v, dopp_fwhm=dopp_fwhm)
    return alpha, components, info


# =========================================================
# OD — 2-level Doppler-broadened absorption
# =========================================================
class ODScheme(Scheme):
    name = "od"
    cluster = "A — Absorption"
    title = "Optical absorption (OD)"
    supports_headless_observables = True
    caption = ("Weak-probe absorption. The full 85Rb D1 four-line hyperfine "
               "spectrum with thermal sublevel normalization, or a single "
               "bare 2-level Voigt. Doppler-broadened in a warm vapor; natural-"
               "linewidth Lorentzian when Doppler is off.")

    def param_schema(self):
        return [
            ParamSpec("model", "Absorption model", "Model", "85Rb D1 hyperfine",
                      choices=("85Rb D1 hyperfine", "single 2-level"),
                      help="Hyperfine: the full 85Rb D1 four-line spectrum (CG "
                      "strengths, ground populations, self-broadening) compared "
                      "against the normalization-corrected AutoOD calculator. Single: one bare 2-level "
                      "Voigt — the validation backbone the Λ schemes reduce to."),
            ParamSpec("temp_c", "Temperature", "Cell & beams", 50.0, 20.0, 200.0, 1.0, "°C",
                      help="Sets the 85Rb density (absorption scale) and Doppler width."),
            ParamSpec("cell_mm", "Cell length", "Cell & beams", 10.0, 0.5, 200.0, 0.5, "mm",
                      recompute=False),
            ParamSpec("line_strength", "Line-strength factor", "Detection & scaling", 1.0,
                      0.01, 2.0, 0.01, "", help="Effective |d|² calibration knob. Single: "
                      "=1.0 reproduces the textbook 3λ²/2π cross-section. Hyperfine: "
                      "=1.0 uses the lifetime-derived dipole and thermal sublevel populations.",
                      recompute=False),
            ParamSpec("doppler", "Doppler (vapor motion)", "Numerics", "on",
                      choices=("on", "off"), advanced=True),
            ParamSpec("ne_pressure_torr", "Ne buffer pressure", "Cell & beams", 0.0,
                      0.0, 200.0, 1.0, "Torr", advanced=True,
                      help="Fixed-neon pressure broadening only; pressure shift and "
                      "Dicke narrowing are not included."),
        ]

    def presets(self):
        return [
            Preset("85Rb D1 cell (90 °C, 12.5 mm)",
                   values=dict(model="85Rb D1 hyperfine", temp_c=90.0, cell_mm=12.5),
                   icon="🧪", help="Thermally normalized warm-cell spectrum."),
            Preset("Single line (cold)",
                   values=dict(model="single 2-level", temp_c=25.0, cell_mm=3.0,
                               doppler="off"), icon="📏"),
        ]

    # ---- hyperfine (validated full-D1) ----
    def _compute_hyperfine(self, params):
        T = params["temp_c"] + 273.15
        sigma_v = np.sqrt(constants.KB * T / constants.MASS_85RB)
        dopp_fwhm = np.sqrt(8 * np.log(2)) * K_VEC * sigma_v
        N = hyperfine.number_density(T)
        gamma_eff = hyperfine.self_broadened_gamma(N) + _ne_buffer_gamma(params)
        width = dopp_fwhm if params["doppler"] == "on" else gamma_eff

        shifts = np.array([2 * np.pi * v for v in hyperfine.LINE_SHIFT_HZ.values()])
        margin = max(2 * np.pi * 1.5e9, 6 * width)
        lo, hi = shifts.min() - margin, shifts.max() + margin
        n = int(np.clip((hi - lo) / (width / 8.0), 1201, 8000))
        scan = np.linspace(lo, hi, n)

        alpha, components, info = _hyperfine_alpha(
            scan, params, line_strength=1.0)
        return dict(model="hyperfine", scan=scan, alpha=alpha,
                    components={f"{Fg}-{Fe}": v for (Fg, Fe), v in components.items()},
                    L=params.get("cell_mm", 10.0) * 1e-3, T=T, **info)

    def _observables_hyperfine(self, raw, params, include_figures=True):
        x = raw["scan"] / (2 * np.pi) / 1e9                      # GHz
        alpha = raw["alpha"] * float(params.get("line_strength", 1.0))
        L = params["cell_mm"] * 1e-3            # navigate-only knob: read live
        OD = observables.optical_density(alpha, L)
        T_trans = observables.transmission(alpha, L)

        fig = None
        if include_figures:
            import matplotlib.pyplot as plt

            fig, (axT, axOD) = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
            axT.plot(x, T_trans, color="#1f77b4", lw=1.6)
            axT.set_ylabel("Transmission")
            axT.set_ylim(-0.02, 1.04)
            axT.set_title(f"85Rb D1 hyperfine:  T = {params['temp_c']:.0f} °C,  "
                          f"L = {params['cell_mm']:.1f} mm,  "
                          f"Doppler {params['doppler']}")
            axOD.plot(x, OD, color="#d62728", lw=1.6)
            axOD.set_ylabel("Optical density  (−log₁₀T)")
            axOD.set_xlabel("Detuning  [GHz]  (ref: 87Rb F=2→F′=2)")
            for (Fg, Fe), sh in hyperfine.LINE_SHIFT_HZ.items():
                for a in (axT, axOD):
                    a.axvline(sh / 1e9, color="gray", ls=":", lw=0.7)
                axOD.annotate(f"{Fg}→{Fe}′", (sh / 1e9, 0), xytext=(0, 2),
                              textcoords="offset points", ha="center", va="bottom",
                              fontsize=7, color="gray")
            fig.tight_layout()

        buffer_mhz = raw.get("buffer_gamma", 0.0) / (2 * np.pi) / 1e6
        self_gamma = raw["gamma_eff"] - GAMMA - raw.get("buffer_gamma", 0.0)
        sb_mhz = self_gamma / (2 * np.pi) / 1e6
        metrics = [
            dict(label="Peak OD", value=f"{np.nanmax(OD):.3f}",
                 help="Largest −log₁₀(T) across the four-line spectrum."),
            dict(label="Min transmission", value=f"{np.nanmin(T_trans):.3e}"),
            dict(label="Doppler FWHM", value=f"{raw['dopp_fwhm']/(2*np.pi)/1e6:.0f} MHz",
                 help="Gaussian (Doppler) width of each hyperfine line."),
        ]
        rows = "".join(
            f"| {Fg}→{Fe}′ | {hyperfine.LINE_SHIFT_HZ[(Fg, Fe)]/1e9:.3f} | "
            f"{hyperfine.CF2[(Fg, Fe)]:.4f} |\n"
            for (Fg, Fe) in hyperfine.TRANSITIONS)
        derived = derived_table([
            ("N(85Rb), pure cell", f"{raw['N']:.3e} /m³"),
            ("σ_v (1-D)", f"{raw['sigma_v']:.1f} m/s"),
            ("Doppler FWHM", f"{raw['dopp_fwhm']/(2*np.pi)/1e6:.1f} MHz"),
            ("Γ_eff / 2π (self + Ne broadened)",
             f"{raw['gamma_eff']/(2*np.pi)/1e6:.3f} MHz (+{sb_mhz:.3f})"),
            ("Ne buffer pressure", f"{params.get('ne_pressure_torr', 0.0):.0f} Torr"),
            ("Ne broadening / 2π", f"{buffer_mhz:.3f} MHz"),
            ("Natural Γ / 2π", f"{GAMMA_MHZ:.3f} MHz"),
        ])
        lines = ("| Line | Center [GHz] | C_F² |\n|---|---|---|\n" + rows)
        return dict(metrics=metrics, figure=fig, tables=[
            derived,
            {"title": "Hyperfine lines (relative strength)", "markdown": lines}])

    # ---- single bare 2-level line (validation backbone) ----
    def _compute_single(self, params):
        T = params["temp_c"] + 273.15
        sigma_v = np.sqrt(constants.KB * T / constants.MASS_85RB)
        dopp_fwhm = np.sqrt(8 * np.log(2)) * K_VEC * sigma_v
        buffer_gamma = _ne_buffer_gamma(params)
        gamma_eff = GAMMA + buffer_gamma
        half = (max(10 * gamma_eff, 3.5 * dopp_fwhm)
                if params["doppler"] == "on" else 12 * gamma_eff)
        scan = np.linspace(-half, half, 601)

        atom = atoms.two_level(gamma=gamma_eff)

        def build_H0(_s):
            H = np.zeros((2, 2), dtype=complex)
            H[0, 1] = H[1, 0] = PROBE_RABI * GAMMA / 2
            return H

        chi_bar = _solve_chi_avg(atom, build_H0, (1, 0), scan, params, h_dep=False)
        N = species.rb85_natural_density(T)
        return dict(model="single", scan=scan, chi_bar=chi_bar, N=N, T=T,
                    L=params.get("cell_mm", 10.0) * 1e-3, ls=1.0,
                    sigma_v=sigma_v, dopp_fwhm=dopp_fwhm,
                    gamma_eff=gamma_eff, buffer_gamma=buffer_gamma)

    def _observables_single(self, raw, params, include_figures=True):
        x = raw["scan"] / (2 * np.pi) / 1e6                      # MHz
        alpha, _ = observables.absorption_coefficient(
            raw["chi_bar"], K_VEC, raw["N"],
            line_strength=float(params.get("line_strength", 1.0)))
        L = params["cell_mm"] * 1e-3           # navigate-only knob: read live
        OD = observables.optical_density(alpha, L)
        T_trans = observables.transmission(alpha, L)

        fig = None
        if include_figures:
            import matplotlib.pyplot as plt

            fig, (axT, axOD) = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
            axT.plot(x, T_trans, color="#1f77b4", lw=1.8)
            axT.set_ylabel("Transmission")
            axT.set_ylim(-0.02, 1.02)
            axT.set_title(f"Single 2-level:  T = {params['temp_c']:.0f} °C,  "
                          f"L = {params['cell_mm']:.0f} mm,  "
                          f"Doppler {params['doppler']}")
            axOD.plot(x, OD, color="#d62728", lw=1.8)
            axOD.set_ylabel("Optical density  (−log₁₀T)")
            axOD.set_xlabel("Probe detuning  [MHz]")
            for a in (axT, axOD):
                a.axvline(0, color="gray", ls=":", lw=0.8)
            fig.tight_layout()

        fwhm_mhz = fwhm_halfmax(x, OD)
        buffer_mhz = raw.get("buffer_gamma", 0.0) / (2 * np.pi) / 1e6
        metrics = [
            dict(label="Peak OD", value=f"{np.nanmax(OD):.3f}",
                 help="On-line optical density −log₁₀(T)."),
            dict(label="Min transmission", value=f"{np.nanmin(T_trans):.3f}"),
            dict(label="Line FWHM", value=f"{fwhm_mhz:.1f} MHz",
                 help="Voigt width (Doppler on) or natural Γ (Doppler off)."),
        ]
        derived = derived_table([
            ("Ne buffer pressure", f"{params.get('ne_pressure_torr', 0.0):.0f} Torr"),
            ("Ne broadening / 2π", f"{buffer_mhz:.3f} MHz"),
            ("Γ_eff / 2π", f"{raw.get('gamma_eff', GAMMA)/(2*np.pi)/1e6:.3f} MHz"),
            ("N(85Rb), natural Rb", f"{raw['N']:.3e} /m³"),
            ("σ_v (1-D)", f"{raw['sigma_v']:.1f} m/s"),
            ("Doppler FWHM", f"{raw['dopp_fwhm']/(2*np.pi)/1e6:.1f} MHz"),
            ("Natural Γ / 2π", f"{GAMMA_MHZ:.3f} MHz"),
        ])
        return dict(metrics=metrics, figure=fig, tables=[derived])

    def compute(self, params):
        if params.get("model", "85Rb D1 hyperfine") == "single 2-level":
            return self._compute_single(params)
        return self._compute_hyperfine(params)

    def observables(self, raw, params, include_figures=True):
        if raw.get("model") == "single":
            return self._observables_single(
                raw, params, include_figures=include_figures)
        return self._observables_hyperfine(
            raw, params, include_figures=include_figures)

LAMBDA_GENERIC = "Generic"
# Atom / Isotope rows of the Λ species knob (rendered as two selectors).
LAMBDA_ATOM_MENU = (
    ("Rb", "Natural", "Rb (natural)", {}),
    ("Rb", "85", "⁸⁵Rb", {}),
    ("Rb", "87", "⁸⁷Rb", {}),
    ("Cs", "133", "¹³³Cs", {}),
    (LAMBDA_GENERIC, LAMBDA_GENERIC, LAMBDA_GENERIC, {}),
)


def _medium_from_params(params, default_temp_c=50.0):
    """Scalar D-line medium used by the Lambda/Rydberg linear readout models."""
    T = float(params.get("temp_c", default_temp_c)) + 273.15
    line = params.get("line", "D1")
    key = params.get("species", "Rb (natural)")
    if key == LAMBDA_GENERIC:
        return dict(
            key=key, label=key, line=line, T=T, N=species.rb85_natural_density(T),
            gamma=GAMMA, gamma_mhz=GAMMA_MHZ, k_vec=K_VEC, omega0=OMEGA_D1,
            dipole=constants.DIPOLE_D1, mass=constants.MASS_85RB,
            ground_hf_hz=constants.NU_GROUND_HF,
        )

    comps = species.SPECIES.get(key, species.SPECIES["Rb (natural)"])
    iso_ref = max(comps, key=lambda item: item[1])[0]
    Je, nu0, gamma_mhz, _, _ = iso_ref.line(line)
    gamma = gamma_mhz * MHZ
    lam = constants.C_LIGHT / nu0
    total_weight = sum(weight for _, weight in comps)
    N = sum(species.number_density(iso, T) * weight for iso, weight in comps)
    mass = sum(iso.mass * weight for iso, weight in comps) / max(total_weight, 1e-30)
    dipole = np.sqrt(species.reduced_dipole_sq(gamma, lam, iso_ref.Jg, Je))
    return dict(
        key=key, label=key, line=line, T=T, N=N, gamma=gamma,
        gamma_mhz=gamma_mhz, k_vec=2 * np.pi / lam, omega0=2 * np.pi * nu0,
        dipole=dipole, mass=mass,
        ground_hf_hz=(iso_ref.I + 0.5) * iso_ref.A_S * 1e6,
    )


def _mkey(params, new_key, old_key, scale, default):
    """Read a physical-unit key, with old Gamma-unit Lambda params as fallback."""
    if new_key in params:
        return float(params[new_key])
    if old_key in params:
        return float(params[old_key]) * scale
    return default


# =========================================================
# Λ system — EIT / AT / CPT (one engine, three presentations)
# =========================================================
LAMBDA_COUPLING_POWER_REF_MW = 1.0
LAMBDA_COUPLING_DIAMETER_REF_MM = 1.0
LAMBDA_JOINT_MAXWELL = "Joint Maxwell (reference)"
LAMBDA_MAX_ANGLE_MRAD = 20.0


class LambdaScheme(Scheme):
    cluster = "A — Absorption"
    cache_version = "physical-units-v7-joint-maxwell-reference"
    defaults_version = "physical-units-v3"
    supports_headless_observables = True
    model_comparison_parameter = "coupling_power_mw"
    model_comparison_controls = ("lambda_compare",)
    workspace_inputs = ("coupling_power_mw",)

    _TITLE = {
        "eit": "Electromagnetically induced transparency (EIT)",
        "at":  "Autler-Townes splitting (AT)",
        "cpt": "Coherent population trapping (CPT)",
    }
    _CAPTION = {
        "eit": ("Λ system: a strong coupling field opens a transparency "
                "window for a weak probe at two-photon resonance."),
        "at":  ("Strong coupling dresses the excited state; the weak-probe "
                "absorption line splits into a doublet separated by ≈ Ω_c."),
        "cpt": ("Narrow two-photon (Raman) scan: atoms are pumped into a "
                "dark state, giving a sub-natural absorption dip."),
    }

    # Per-regime defaults converted from the old Gamma-unit controls into lab units.
    _DEF = {
        "eit": dict(oc_mhz=3.0 * GAMMA_MHZ, gg_khz=0.01 * GAMMA_MHZ * 1e3,
                    temp=50.0, cell=15.0, dopp="on"),
        "at":  dict(oc_mhz=8.0 * GAMMA_MHZ, gg_khz=0.01 * GAMMA_MHZ * 1e3,
                    temp=25.0, cell=3.0, dopp="off"),
        "cpt": dict(oc_mhz=1.0 * GAMMA_MHZ, gg_khz=0.005 * GAMMA_MHZ * 1e3,
                    temp=25.0, cell=3.0, dopp="off"),
    }

    def __init__(self, mode=None):
        self.mode = mode                       # None → merged EIT/AT/CPT dropdown entry
        if mode is None:
            self.name = "lambda"
            self.title = "Λ coherence (EIT / AT / CPT)"
            self.caption = ("One Λ system, three regimes of the same coupling field Ω_c: a "
                            "weak coupling opens an EIT transparency window, a strong one "
                            "splits the line into an Autler-Townes doublet, and a narrow "
                            "two-photon scan reveals the CPT dark resonance.")
        else:
            self.name = mode
            self.title = self._TITLE[mode]
            self.caption = self._CAPTION[mode]

    def _mode(self, params):
        """Effective regime: pinned (alias instance) or the merged `view` knob."""
        return self.mode or str(params.get("view", "eit")).lower()

    def atom_conditions(self, params):
        fields = ("species", "line", "temp_c", "cell_mm", "buffer_ground_relax_khz")
        return dict(fields=tuple(name for name in fields if name not in self.comparison_input_names(params)),
                    line="line")

    def comparison_input_names(self, params):
        from .lambda_comparison import comparison_key
        return tuple(dict.fromkeys(("coupling_power_mw", comparison_key(params))))

    def param_schema(self):
        d = self._DEF[self.mode or "eit"]
        specs = []
        if self.mode is None:
            specs.append(ParamSpec(
                "view", "Regime", "Regime", "EIT", choices=("EIT", "AT", "CPT"),
                control="segmented", applies_defaults=True,
                help="EIT: weak-coupling transparency window. AT: strong-coupling "
                     "doublet (split ≈ Ω_c). CPT: narrow two-photon dark resonance. "
                     "One Λ engine — the presets set textbook values for each."))
        specs += [
            ParamSpec("lambda_compare", "Compare", "Comparison", "Coupling power",
                      choices=("Coupling power", "Ground relaxation", "Beam angle", "Detuning map"),
                      recompute=False,
                      help="Compare one physical control. Detuning map uses Δp × Δc; other CPT views use δ = Δp − Δc."),
            ParamSpec("species", "Atom", "Atomic", "Rb (natural)",
                      choices=tuple(species.SPECIES_ORDER) + (LAMBDA_GENERIC,),
                      atom_menu=LAMBDA_ATOM_MENU,
                      help="Scalar D-line medium. Natural Rb uses abundance-weighted density."),
            ParamSpec("line", "Transition line", "Atomic", "D1",
                      choices=("D1", "D2"),
                      help="Sets linewidth, wavelength, Doppler scale and dipole moment."),
            ParamSpec("coupling_detuning_mhz", "Coupling detuning", "Detunings", 0.0,
                      -80.0, 80.0, 0.1, "MHz"),
            ParamSpec("coupling_power_mw", "Coupling power", "Fields",
                      LAMBDA_COUPLING_POWER_REF_MW, 0.01, 50.0, 0.01, "mW",
                      help="Lab-facing coupling-field power. The effective Ω_c is "
                           "anchored by Coupling Rabi and scales as √P/d."),
            ParamSpec("coupling_diameter_mm", "Coupling beam diameter", "Fields",
                      LAMBDA_COUPLING_DIAMETER_REF_MM, 0.05, 5.0, 0.05, "mm",
                      help="1/e² coupling-beam diameter used for the √(P/d²) "
                           "Rabi scaling."),
            ParamSpec("beam_angle_mrad", "Beam angle mismatch", "Fields",
                      0.0, 0.0, LAMBDA_MAX_ANGLE_MRAD, 0.1, "mrad", advanced=True,
                      help="Included angle between probe and coupling beams. "
                           "At 0 mrad only the optical-frequency difference "
                           "contributes to the Raman Doppler width."),
            ParamSpec("coupling_rabi_mhz", "Coupling Rabi (anchor)", "Fields", d["oc_mhz"],
                      0.1, 120.0, 0.1, "MHz",
                      advanced=True,
                      help="Ω_c/2π at the reference coupling power and beam diameter. "
                           "Power and diameter sliders scale the effective Ω_c.",
                      endpoints=("weak EIT", "strong AT")),
            ParamSpec("buffer_ground_relax_khz", "Buffer ground relaxation", "Atomic",
                      d["gg_khz"], 0.0, 2000.0, 1.0, "kHz",
                      help="Ground-state coherence relaxation (buffer-gas collisions); "
                           "sets the EIT/CPT linewidth floor. Same quantity as the "
                           "magneto buffer ground relaxation."),
            ParamSpec("temp_c", "Temperature", "Cell & beams", d["temp"], 20.0, 200.0, 1.0, "°C"),
            ParamSpec("cell_mm", "Cell length", "Cell & beams", d["cell"], 0.5, 200.0, 0.5, "mm",
                      recompute=False),
            ParamSpec("line_strength", "Line-strength factor", "Detection & scaling", 1.0,
                      0.01, 2.0, 0.01, "", advanced=True, recompute=False),
            ParamSpec("doppler", "Doppler (vapor motion)", "Numerics", d["dopp"],
                      choices=("on", "off"), advanced=True),
            ParamSpec("angled_doppler_model", "Angled Doppler model", "Numerics", "Scalar",
                      choices=("Scalar", LAMBDA_JOINT_MAXWELL), advanced=True,
                      visible_if={"doppler": "on"},
                      help="The joint reference uses the same longitudinal velocity in both "
                           "optical and Raman shifts, and averages transverse motion independently. "
                           "Angled scans may take several seconds per condition."),
        ]
        return specs

    def presets(self):
        if self.mode is None:
            return []
        label = self._TITLE[self.mode].split("(")[-1].rstrip(")")
        return [Preset(f"{label} default", values=self._default_values(self.mode))]

    def _default_values(self, mode):
        d = self._DEF[mode]
        view = {"eit": "EIT", "at": "AT", "cpt": "CPT"}[mode]
        return dict(
            view=view, coupling_detuning_mhz=0.0,
            coupling_rabi_mhz=d["oc_mhz"], buffer_ground_relax_khz=d["gg_khz"],
            coupling_power_mw=LAMBDA_COUPLING_POWER_REF_MW,
            coupling_diameter_mm=LAMBDA_COUPLING_DIAMETER_REF_MM,
            beam_angle_mrad=0.0,
            temp_c=d["temp"], cell_mm=d["cell"], doppler=d["dopp"],
            line_strength=1.0,
        )

    @staticmethod
    def _coupling_rabi_from_params(params, gamma_mhz, default_mhz):
        anchor = _mkey(params, "coupling_rabi_mhz", "coupling_rabi",
                       gamma_mhz, default_mhz)
        p = float(params.get("coupling_power_mw", LAMBDA_COUPLING_POWER_REF_MW))
        d = float(params.get("coupling_diameter_mm", LAMBDA_COUPLING_DIAMETER_REF_MM))
        return beam.anchored_rabi_mhz(
            anchor, p, LAMBDA_COUPLING_POWER_REF_MW,
            diameter=d, ref_diameter=LAMBDA_COUPLING_DIAMETER_REF_MM)

    @staticmethod
    def _wave_numbers(medium):
        # Probe addresses the lower hyperfine ground level in this Lambda model.
        k_probe = medium["k_vec"]
        return k_probe, k_probe - 2.0 * np.pi * medium["ground_hf_hz"] / constants.C_LIGHT

    @classmethod
    def _two_photon_k_ratio(cls, params, medium):
        angle = float(params.get("beam_angle_mrad", 0.0))
        k_probe, k_coupling = cls._wave_numbers(medium)
        if angle == 0.0:
            return beam.collinear_residual_k_ratio(k_probe, k_coupling)
        # Scalar angle approximation. Joint transverse averaging is a separate model.
        return np.copysign(beam.angled_residual_k_ratio(
            k_probe, k_coupling, angle_mrad=angle), k_probe - k_coupling)

    def recommended_defaults(self, params):
        if self.mode is not None:
            return {self._TITLE[self.mode].split("(")[-1].rstrip(")"):
                    self._default_values(self.mode)}
        return {
            "EIT": self._default_values("eit"),
            "AT": self._default_values("at"),
            "CPT": self._default_values("cpt"),
        }

    def _scan(self, params, medium=None, mode=None, Oc_mhz=None, Dc_mhz=None,
              gg_khz=None):
        m = self._mode(params) if mode is None else mode
        medium = _medium_from_params(params) if medium is None else medium
        gamma_mhz = medium["gamma_mhz"]
        Oc = (self._coupling_rabi_from_params(params, gamma_mhz, self._DEF[m]["oc_mhz"])
              if Oc_mhz is None else Oc_mhz)
        Dc = (_mkey(params, "coupling_detuning_mhz", "coupling_detuning", gamma_mhz, 0.0)
              if Dc_mhz is None else Dc_mhz)
        gg_khz = (_mkey(params, "buffer_ground_relax_khz", "gamma_gg", gamma_mhz * 1e3,
                        self._DEF[m]["gg_khz"])
                  if gg_khz is None else gg_khz)
        gg = gg_khz / (gamma_mhz * 1e3)
        Oc_g = Oc / gamma_mhz
        Dc_g = Dc / gamma_mhz
        if m == "cpt":
            dark = gg + Oc_g**2          # power-broadened dark width (Gamma units)
            half = max(0.05, 12.0 * dark)
        elif m == "at":
            half = max(10.0, 3.0 * Oc_g)
        else:
            half = max(8.0, 4.0 * Oc_g)
        if m == "at":
            offsets = np.linspace(-half, half, 601)
        else:
            # Reserve half the points for the dark resonance, while keeping
            # the original scan extent to bracket the background and crossings.
            local = min(half, max(0.015, min(0.22, 2.0 * (gg + Oc_g**2))))
            wings = np.linspace(local, half, 151)[1:]
            offsets = np.concatenate((-wings[::-1], np.linspace(-local, local, 301), wings))
        return (Dc_g + offsets) * medium["gamma"]

    def compute(self, params):
        medium = _medium_from_params(params)
        gamma = medium["gamma"]
        gamma_mhz = medium["gamma_mhz"]
        mode = self._mode(params)
        Oc_mhz = self._coupling_rabi_from_params(
            params, gamma_mhz, self._DEF[mode]["oc_mhz"])
        Dc_mhz = _mkey(params, "coupling_detuning_mhz", "coupling_detuning", gamma_mhz, 0.0)
        gg_khz = _mkey(params, "buffer_ground_relax_khz", "gamma_gg", gamma_mhz * 1e3,
                       self._DEF[mode]["gg_khz"])
        angle_mrad = float(params.get("beam_angle_mrad", 0.0))
        joint_selected = params.get("angled_doppler_model", "Scalar") == LAMBDA_JOINT_MAXWELL
        doppler_on = params.get("doppler", "off") == "on"
        if joint_selected and doppler_on and not 0.0 <= angle_mrad <= LAMBDA_MAX_ANGLE_MRAD:
            raise ValueError(f"Joint Maxwell beam angle must be in [0, {LAMBDA_MAX_ANGLE_MRAD:g}] mrad.")
        two_photon_k_ratio = self._two_photon_k_ratio(params, medium)
        atom = atoms.lambda3(
            gamma_gg=gg_khz * KHZ, gamma=gamma,
            two_photon_doppler_ratio=two_photon_k_ratio)
        Oc = Oc_mhz * MHZ
        Dc = Dc_mhz * MHZ
        probe = PROBE_RABI * gamma

        def build_H0(s):
            H = np.zeros((3, 3), dtype=complex)
            H[1, 1] = Dc - s                      # two-photon detuning
            H[0, 2] = H[2, 0] = probe / 2
            H[1, 2] = H[2, 1] = Oc / 2
            return H

        scan = self._scan(
            params, medium=medium, mode=mode, Oc_mhz=Oc_mhz,
            Dc_mhz=Dc_mhz, gg_khz=gg_khz)
        scan_shift = core.comm_super(np.diag([0.0, 0.0, 1.0]))
        chi_bar = _solve_chi_avg(
            atom, build_H0, (2, 0), scan, params, h_dep=True,
            probe_omega=probe, k_vec=medium["k_vec"], mass=medium["mass"],
            scan_shift=scan_shift)
        raw = dict(scan=scan, chi_bar=chi_bar, N=medium["N"], T=medium["T"], Dc=Dc,
                    L=params.get("cell_mm", 10.0) * 1e-3, ls=1.0,
                    gamma=gamma, gamma_mhz=gamma_mhz, k_vec=medium["k_vec"],
                    omega0=medium["omega0"], dipole=medium["dipole"],
                    medium_label=medium["label"], line=medium["line"],
                    coupling_rabi_mhz=Oc_mhz, buffer_ground_relax_khz=gg_khz,
                    coupling_power_mw=float(params.get(
                        "coupling_power_mw", LAMBDA_COUPLING_POWER_REF_MW)),
                    coupling_diameter_mm=float(params.get(
                        "coupling_diameter_mm", LAMBDA_COUPLING_DIAMETER_REF_MM)),
                    beam_angle_mrad=angle_mrad, two_photon_k_ratio=two_photon_k_ratio)
        if joint_selected:
            if not doppler_on:
                raw["joint_doppler"] = dict(status="doppler_off")
            elif angle_mrad == 0.0:
                raw["joint_doppler"] = dict(status="collinear_limit")
            else:
                raw["scalar_chi_bar"] = chi_bar
                raw["chi_bar"], raw["joint_doppler"] = self._joint_doppler(
                    scan, medium, angle_mrad, build_H0,
                    gg_khz * KHZ, probe, scan_shift)
        return raw

    @classmethod
    def _joint_doppler(cls, scan, medium, angle_mrad, build_H0, ground_relaxation,
                       probe, scan_shift):
        k_probe, k_coupling = cls._wave_numbers(medium)
        theta = angle_mrad * 1e-3
        longitudinal = float(doppler.noncollinear_raman_shift_rad_s(
            0.0, 1.0, k_probe, k_coupling, theta))
        transverse = float(doppler.noncollinear_raman_shift_rad_s(
            1.0, 0.0, k_probe, k_coupling, theta))
        atom = atoms.lambda3(gamma_gg=ground_relaxation, gamma=medium["gamma"],
                             two_photon_doppler_ratio=longitudinal / k_probe)

        def conditional_scan(vx):
            def shifted_hamiltonian(detuning):
                hamiltonian = build_H0(detuning)
                hamiltonian[1, 1] += transverse * vx
                return hamiltonian

            return affine_doppler.AffineScan(
                atom, shifted_hamiltonian, (2, 0), probe, scan_shift)

        mean, evidence = affine_doppler.joint_maxwell_mean(
            scan, medium["T"], medium["mass"], k_probe, conditional_scan)
        evidence.update(raman_rms=doppler.noncollinear_raman_rms_budget(
            medium["T"], k_probe, k_coupling, theta, medium["mass"]),
            probe_k_rad_m=k_probe, coupling_k_rad_m=k_coupling,
            longitudinal_raman_k_rad_m=longitudinal, transverse_raman_k_rad_m=transverse)
        return mean, evidence

    def model_comparison(self, params, *, resolve_raw, current_raw=None, current_view=None):
        from .lambda_comparison import coupling_comparison
        return coupling_comparison(self, params, resolve_raw=resolve_raw,
                                   current_raw=current_raw, current_view=current_view)

    def observables(self, raw, params, include_figures=True):
        m = self._mode(params)
        x = raw["scan"] / (2 * np.pi) / 1e6                      # MHz
        alpha, xphys = observables.absorption_coefficient(
            raw["chi_bar"], raw.get("k_vec", K_VEC), raw["N"],
            dipole=raw.get("dipole"),
            line_strength=float(params.get("line_strength", 1.0)))
        L = params["cell_mm"] * 1e-3           # navigate-only knob: read live
        T_trans = observables.transmission(alpha, L)
        center = raw["Dc"] / (2 * np.pi) / 1e6                   # two-photon resonance, MHz

        fig = None
        if include_figures:
            import matplotlib.pyplot as plt

            fig, (axT, axD) = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
            axT.plot(x, T_trans, color="#1f77b4", lw=1.8)
            axT.set_ylabel("Transmission")
            axT.set_ylim(-0.02, 1.02)
            axD.plot(x, np.real(xphys), color="#9467bd", lw=1.6)
            axD.set_ylabel("Re chi  (dispersion)")
            axD.set_xlabel("Probe detuning  [MHz]")
            for a in (axT, axD):
                a.axvline(center, color="gray", ls=":", lw=0.8)
            fig.tight_layout()

        ic = int(np.argmin(np.abs(x - center)))
        window = _lambda_window_readout(x, T_trans, xphys, ic, raw, L) if m != "at" else None
        joint = raw.get("joint_doppler")
        joint_failed = joint is not None and joint["status"] == "not_converged"
        if joint_failed and window is not None:
            window.update(status="velocity_not_converged", width_mhz=None,
                          group_index=None, group_delay_s=None)
        metrics = self._metrics(m, x, alpha, T_trans, ic, center, raw, params, window,
                                joint_failed=joint_failed)
        details = [
            ("Medium", f"{raw.get('medium_label', '85Rb')} {raw.get('line', 'D1')}"),
            ("Γ / 2π", f"{raw.get('gamma_mhz', GAMMA_MHZ):.4f} MHz"),
            ("Ω_c / 2π", f"{raw['coupling_rabi_mhz']:.4f} MHz"),
            ("Coupling power", f"{raw['coupling_power_mw']:.3f} mW"),
            ("Coupling beam diameter", f"{raw['coupling_diameter_mm']:.3f} mm"),
            ("Beam angle mismatch", f"{raw.get('beam_angle_mrad', 0.0):.3f} mrad"),
            ("Scalar residual k / k" if "scalar_chi_bar" in raw else "Residual two-photon k / k",
             f"{raw.get('two_photon_k_ratio', 0.0):.3e}"),
            ("Buffer ground relax / 2π", f"{raw['buffer_ground_relax_khz']:.3f} kHz"),
            ("N", f"{raw['N']:.3e} /m³"),
        ]
        readout = dict(
            x_mhz=x, transmission=T_trans, absorption_per_m=alpha,
            dispersion=np.real(xphys), center_mhz=float(center),
            coupling_rabi_mhz=float(raw["coupling_rabi_mhz"]),
            coupling_power_mw=float(raw["coupling_power_mw"]),
            regime=m, experimental_visibility="not assessed",
            window=window,
        )
        if joint is not None:
            status = joint["status"]
            readout["joint_doppler"] = joint
            details.append(("Joint Maxwell reference", status.replace("_", " ")))
            if "scalar_chi_bar" in raw:
                scalar_alpha, _ = observables.absorption_coefficient(
                    raw["scalar_chi_bar"], raw["k_vec"], raw["N"], dipole=raw["dipole"],
                    line_strength=float(params.get("line_strength", 1.0)))
                scalar_transmission = observables.transmission(scalar_alpha, L)
                scalar_scale = max(float(np.max(np.abs(raw["scalar_chi_bar"]))),
                                   np.finfo(float).tiny)
                delta = dict(
                    complex_chi_relative_max=float(np.max(np.abs(
                        raw["chi_bar"] - raw["scalar_chi_bar"]))) / scalar_scale,
                    transmission_at_resonance=float(T_trans[ic] - scalar_transmission[ic]),
                    transmission_absolute_max=float(np.max(np.abs(T_trans - scalar_transmission))),
                )
                readout["joint_doppler"] = dict(joint, difference_from_scalar=delta)
                details.extend([
                    ("Joint minus scalar, T at resonance", f"{delta['transmission_at_resonance']:+.4f}"),
                    ("Raman Doppler RMS", f"{joint['raman_rms']['total_rms_hz'] / 1e3:.2f} kHz"),
                ])
        return dict(metrics=metrics, figure=fig, tables=[derived_table(details)],
                    lambda_readout=readout)

    def _metrics(self, mode, x, alpha, T_trans, ic, center, raw, params, window,
                 *, joint_failed=False):
        center_transmission = None if joint_failed else float(T_trans[ic])
        center_display = "—" if joint_failed else f"{center_transmission:.3f}"
        if mode == "at":
            # A real AT doublet needs two *resolved local maxima*. Taking the
            # largest sample on either side of centre always returns a nonzero
            # grid spacing, even in the weak-coupling single-peak limit.
            local = np.flatnonzero(
                (alpha[1:-1] > alpha[:-2]) & (alpha[1:-1] >= alpha[2:])) + 1
            left = local[x[local] < center]
            right = local[x[local] > center]
            resolved = False
            split = np.nan
            if not joint_failed and left.size and right.size:
                il = int(left[np.nanargmax(alpha[left])])
                ir = int(right[np.nanargmax(alpha[right])])
                valley = float(np.nanmin(alpha[il:ir + 1]))
                peak_l, peak_r = float(alpha[il]), float(alpha[ir])
                contrast_scale = max(
                    abs(peak_l), abs(peak_r), abs(valley), 1e-30)
                resolved = (
                    np.isfinite([peak_l, peak_r, valley]).all()
                    and min(peak_l - valley, peak_r - valley)
                    > 0.01 * contrast_scale
                )
                if resolved:
                    split = abs(float(x[ir] - x[il]))
            expected = raw.get("coupling_rabi_mhz",
                               params.get("coupling_rabi", 0.0) * GAMMA_MHZ)
            if not resolved:
                return [
                    dict(label="AT status", kind="status", tier="hero",
                         value=("velocity integration not converged" if joint_failed
                                else "doublet unresolved"),
                         help=("Joint Maxwell refinement exceeded its bound."
                               if joint_failed else "No pair of absorption maxima with "
                               "resolvable central contrast exists in the displayed spectrum.")),
                    dict(label="AT splitting", value="—",
                         numeric_value=None, unit="MHz",
                         help="Withheld because the spectrum does not contain a "
                              "resolved Autler-Townes doublet."),
                    dict(label="Expected ≈ Ω_c", value=f"{expected:.1f} MHz",
                         numeric_value=float(expected), unit="MHz",
                         help="Coupling Rabi frequency Ω_c/2π."),
                    dict(label="Transmission at center", value=center_display,
                         numeric_value=center_transmission, unit="1",
                         tier="hero"),
                ]
            return [
                dict(label="AT splitting", value=f"{split:.1f} MHz",
                     numeric_value=float(split), unit="MHz",
                     sample_positions=[float(x[il]), float(x[ir])],
                     help="Separation of the two absorption maxima.", tier="hero"),
                dict(label="Expected ≈ Ω_c", value=f"{expected:.1f} MHz",
                     numeric_value=float(expected), unit="MHz",
                     help="Coupling Rabi frequency Ω_c/2π."),
                dict(label="Transmission at center", value=center_display,
                     numeric_value=center_transmission, unit="1",
                     tier="hero"),
            ]
        # EIT / CPT: transparency at two-photon resonance + dark-resonance width.
        win_fwhm = window["width_mhz"]
        ng = window["group_index"]
        delay = window["group_delay_s"]
        unit = "kHz" if mode == "cpt" else "MHz"
        scale = 1e3 if mode == "cpt" else 1.0
        width_valid = window["status"] == "resolved"
        metrics = [
            dict(label="Transmission at resonance", value=center_display,
                 numeric_value=center_transmission, unit="1",
                 help="Probe transmission at two-photon resonance (transparency).",
                 tier="hero"),
            dict(label="Window FWHM",
                 value=(f"{win_fwhm*scale:.2f} {unit}" if width_valid else "—"),
                 numeric_value=float(win_fwhm*scale) if width_valid else None, unit=unit,
                 help=("Width of the transparency feature." if width_valid else
                       "Withheld until the numerical resolution checks pass."),
                 **({"tier": "hero"} if width_valid else {})),
            dict(label="Group index n_g", value=f"{ng:.3e}" if ng is not None else "—",
                 numeric_value=ng, help="Group index at two-photon resonance."),
            dict(label="Group delay", value=f"{delay * 1e9:.3g} ns" if delay is not None else "—",
                 numeric_value=delay, unit="s", help="Excess transit time relative to vacuum."),
            dict(label="Experimental visibility", value="Not assessed", kind="status",
                 help="A calibrated detector-noise threshold was not supplied."),
        ]
        if not width_valid:
            metrics.append(dict(
                label="Window status", value=window["status"].replace("_", " "), kind="status",
                help=("Joint Maxwell refinement exceeded its bound." if joint_failed else
                      "The current opacity/coupling settings do not produce a "
                      "measurable transparency-window FWHM in this scan."),
                tier="hero"))
        return metrics



def _lambda_window_readout(x, transmission, chi, center_index, raw, length):
    """Numerical support for the transparency-window and dispersion readouts."""
    out = dict(status="unresolved", width_mhz=None, half_height_mhz=None,
               samples_per_fwhm=None, edge_clearance_fwhm=None,
               group_index=None, group_delay_s=None)
    peak = float(transmission[center_index])
    floor = float(np.min(transmission))
    if not np.isfinite(peak - floor) or peak - floor <= np.finfo(float).eps:
        return out
    half = (peak + floor) * 0.5
    left, _ = level_crossing(x, transmission, center_index, -1, half)
    right, _ = level_crossing(x, transmission, center_index, 1, half)
    width = right - left
    if not np.isfinite(width) or width <= 0.0:
        return out
    steps = np.diff(x)
    spans = (x[1:] > left) & (x[:-1] < right)
    samples = width / float(np.max(steps[spans]))
    edge = min(left - x[0], x[-1] - right) / width
    out.update(width_mhz=float(width), half_height_mhz=[left, right],
               samples_per_fwhm=float(samples), edge_clearance_fwhm=float(edge))
    if edge < MIN_SUBDOPPLER_EDGE_CLEARANCE_FWHM:
        out["status"] = "scan_edge_limited"
    elif samples < 8.0:
        out["status"] = "under_resolved"
    else:
        ng = float(observables.group_index(
            chi, raw["scan"], raw.get("omega0", OMEGA_D1))[center_index])
        if np.isfinite(ng):
            out.update(status="resolved", group_index=ng,
                       group_delay_s=(ng - 1.0) * length / constants.C_LIGHT)
    return out
