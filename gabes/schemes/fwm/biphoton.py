"""Biphoton topology, generic solver and source cache."""
from .common import (
    BIPHOTON_PREDICTIVE, CS_CHANNEL_795, CS_CHANNEL_917, FieldSpec, GROUND_DEPHASING_FRAC,
    LevelSpec, MHZ_ANG, NM, PRED_RHO_CLIP, PRED_V_FWHM_TOL, PRED_V_MAX_POINTS, PRED_V_MAX_REFINE,
    PRED_V_OVERSAMPLE, TOPOLOGY_CS_BTW, TOPOLOGY_DIAMOND, TOPOLOGY_RB87_TELECOM, TopologySpec,
)
from .phase_matching import (
    _bandwidth_from_waveform_mhz, _field_with, _side_label, _side_sign, _sinc_complex,
    energy_mismatch_hz, phase_matching_weight, phase_mismatch, phase_mismatch_vector,
    transverse_matched_angle_deg,
)
from ... import beam, constants, doppler, species
from ...linear_hyperfine import line_absorption
import functools
import math
import numpy as np
from ...lineshape import fwhm_interp


def _raw_delta_k(fields):
    return phase_mismatch(fields, reference_delta_k=0.0)


def _with_reference_delta_k(spec):
    return TopologySpec(
        name=spec.name,
        label=spec.label,
        family=spec.family,
        isotope_name=spec.isotope_name,
        levels=spec.levels,
        fields=spec.fields,
        signal_role=spec.signal_role,
        idler_role=spec.idler_role,
        default_temp_c=spec.default_temp_c,
        default_cell_mm=spec.default_cell_mm,
        default_pump_uw=spec.default_pump_uw,
        default_coupling_mw=spec.default_coupling_mw,
        pair_rate_cps_per_mw=spec.pair_rate_cps_per_mw,
        emission_decay_ns=spec.emission_decay_ns,
        target_g2_peak=spec.target_g2_peak,
        reference_fwhm_ns=spec.reference_fwhm_ns,
        reference_od=spec.reference_od,
        reference_bandwidth_mhz=spec.reference_bandwidth_mhz,
        reference_width_ratio=spec.reference_width_ratio,
        reference_delta_k=_raw_delta_k(spec.fields),
        notes=spec.notes,
    )


def _rb87_telecom_spec():
    levels = (
        LevelSpec("5S1/2(F=2)", 0.0, 0.0),
        LevelSpec("5P3/2", constants.C_LIGHT / (780.24 * NM), 6.07),
        LevelSpec("4D5/2", constants.C_LIGHT / (780.24 * NM)
                  + constants.C_LIGHT / (1529.37 * NM), 0.66),
        LevelSpec("5P3/2 collection", constants.C_LIGHT / (780.24 * NM), 6.07),
    )
    fields = (
        FieldSpec("pump", 0, 1, 780.24, phase_sign=+1.0),
        FieldSpec("coupling", 1, 2, 1529.37, phase_sign=-1.0, direction=-1.0),
        FieldSpec("signal", 2, 3, 1529.37, phase_sign=-1.0, angle_deg=1.5,
                  side_sign=+1.0),
        FieldSpec("idler", 3, 0, 780.24, phase_sign=+1.0,
                  angle_deg=transverse_matched_angle_deg(1529.37, 780.24, 1.5),
                  side_sign=+1.0),
    )
    return _with_reference_delta_k(TopologySpec(
        name=TOPOLOGY_RB87_TELECOM,
        label="⁸⁷Rb telecom cascade (5S–5P–4D)",
        family="cascade",
        isotope_name="Rb87",
        levels=levels,
        fields=fields,
        signal_role="signal",
        idler_role="idler",
        default_temp_c=90.0,
        default_cell_mm=12.5,
        default_pump_uw=10.0,
        default_coupling_mw=1.0,
        pair_rate_cps_per_mw=38_000.0,
        emission_decay_ns=0.52,
        target_g2_peak=44.0,
        reference_fwhm_ns=0.56,
        reference_od=112.0,
        reference_bandwidth_mhz=300.0,
        notes=("Reference-calibrated cascade SFWM estimate for the telecom "
               "biphoton source in hot 87Rb."),
    ))


def _cs_btw_spec(channel):
    if channel == CS_CHANNEL_795:
        upper = "8S1/2"
        coupling_nm = 795.0
        upper_gamma_mhz = 1.7
        decay_ns = 1.35
        label = "¹³³Cs cascade (852/795 nm)"
    else:
        upper = "6D5/2"
        coupling_nm = 917.0
        upper_gamma_mhz = 2.6
        decay_ns = 4.1
        label = "¹³³Cs cascade (852/917 nm)"
    pump_nm = 852.35
    levels = (
        LevelSpec("6S1/2(F=4)", 0.0, 0.0),
        LevelSpec("6P3/2(F'=5)", constants.C_LIGHT / (pump_nm * NM), 5.23),
        LevelSpec(upper, constants.C_LIGHT / (pump_nm * NM)
                  + constants.C_LIGHT / (coupling_nm * NM), upper_gamma_mhz),
        LevelSpec("6P3/2 collection", constants.C_LIGHT / (pump_nm * NM), 5.23),
    )
    fields = (
        FieldSpec("pump", 0, 1, pump_nm, phase_sign=+1.0),
        FieldSpec("coupling", 1, 2, coupling_nm, phase_sign=-1.0, direction=-1.0),
        FieldSpec("signal", 2, 3, coupling_nm, phase_sign=-1.0, angle_deg=1.5,
                  side_sign=+1.0),
        FieldSpec("idler", 3, 0, pump_nm, phase_sign=+1.0,
                  angle_deg=transverse_matched_angle_deg(coupling_nm, pump_nm, 1.5),
                  side_sign=+1.0),
    )
    return _with_reference_delta_k(TopologySpec(
        name=TOPOLOGY_CS_BTW,
        label=label,
        family="cascade",
        isotope_name="Cs133",
        levels=levels,
        fields=fields,
        signal_role="signal",
        idler_role="idler",
        default_temp_c=75.0,
        default_cell_mm=12.5,
        default_pump_uw=20.0,
        default_coupling_mw=1.0,
        pair_rate_cps_per_mw=12_000.0,
        emission_decay_ns=decay_ns,
        target_g2_peak=18.0,
        reference_fwhm_ns=decay_ns * 0.42,
        reference_od=10.0,
        reference_width_ratio=3.0,
        notes=("Velocity-class coherent-sum model for the Cs biphoton temporal "
               "waveform comparison."),
    ))


def _diamond_generic_spec(params=None):
    params = params or {}
    pump_nm = float(params.get("diamond_pump_nm", 780.0))
    coupling_nm = float(params.get("diamond_coupling_nm", 776.0))
    signal_nm = float(params.get("diamond_signal_nm", 795.0))
    # Energy conservation: 1/λ_idler = 1/λ_pump + 1/λ_coupling − 1/λ_signal.
    # Guard the reciprocal sum — a zero/negative denominator (e.g. pump=coupling
    # with signal=pump/2) has no physical idler and would otherwise raise
    # ZeroDivisionError. Fall back to NaN; the live UI always supplies an explicit
    # diamond_idler_nm, so this default is only the degenerate-case fallback.
    _inv_idler = 1.0 / pump_nm + 1.0 / coupling_nm - 1.0 / signal_nm
    idler_default = 1.0 / _inv_idler if abs(_inv_idler) > 1e-9 else float("nan")
    idler_nm = float(params.get("diamond_idler_nm", idler_default))
    levels = (
        LevelSpec("g", 0.0, 0.0),
        LevelSpec("e1", constants.C_LIGHT / (pump_nm * NM), 6.0),
        LevelSpec("e2", constants.C_LIGHT / (coupling_nm * NM), 6.0),
        LevelSpec("u", constants.C_LIGHT / (pump_nm * NM)
                  + constants.C_LIGHT / (coupling_nm * NM), 1.0),
    )
    fields = (
        FieldSpec("pump", 0, 1, pump_nm, phase_sign=+1.0),
        FieldSpec("coupling", 0, 2, coupling_nm, phase_sign=+1.0),
        FieldSpec("signal", 3, 1, signal_nm, phase_sign=-1.0, angle_deg=2.0,
                  side_sign=+1.0),
        FieldSpec("idler", 3, 2, idler_nm, phase_sign=-1.0,
                  angle_deg=transverse_matched_angle_deg(signal_nm, idler_nm, 2.0),
                  side_sign=-1.0),
    )
    return _with_reference_delta_k(TopologySpec(
        name=TOPOLOGY_DIAMOND,
        label="Generic diamond SFWM",
        family="diamond",
        isotope_name="Rb87",
        levels=levels,
        fields=fields,
        signal_role="signal",
        idler_role="idler",
        default_temp_c=60.0,
        default_cell_mm=12.5,
        default_pump_uw=20.0,
        default_coupling_mw=1.0,
        pair_rate_cps_per_mw=5_000.0,
        emission_decay_ns=8.0,
        target_g2_peak=None,
        reference_fwhm_ns=None,
        reference_od=None,
        reference_bandwidth_mhz=None,
        reference_width_ratio=None,
        notes=("Generic template only; not tied to a validated diamond reference "
               "preset."),
    ))


def _optional_float(value):
    return None if value is None else float(value)


def _topology_cache_key(params):
    return (
        params.get("topology", TOPOLOGY_RB87_TELECOM),
        params.get("cs_channel", CS_CHANNEL_917),
        float(params.get("diamond_pump_nm", 780.0)),
        float(params.get("diamond_coupling_nm", 776.0)),
        float(params.get("diamond_signal_nm", 795.0)),
        _optional_float(params.get("diamond_idler_nm")),
    )


@functools.lru_cache(maxsize=32)
def _topology_from_key(topo, cs_channel, diamond_pump_nm, diamond_coupling_nm,
                       diamond_signal_nm, diamond_idler_nm):
    if topo == TOPOLOGY_CS_BTW:
        return _cs_btw_spec(cs_channel)
    if topo == TOPOLOGY_DIAMOND:
        params = dict(
            diamond_pump_nm=diamond_pump_nm,
            diamond_coupling_nm=diamond_coupling_nm,
            diamond_signal_nm=diamond_signal_nm,
        )
        if diamond_idler_nm is not None:
            params["diamond_idler_nm"] = diamond_idler_nm
        return _diamond_generic_spec(params)
    return _rb87_telecom_spec()


def topology_from_params(params):
    return _topology_from_key(*_topology_cache_key(params))


def _default_biphoton_geometry(params):
    spec = topology_from_params(params)
    signal = spec.field_map["signal"]
    idler = spec.field_map["idler"]
    return dict(
        signal_angle_deg=signal.angle_deg,
        idler_angle_deg=idler.angle_deg,
        signal_side=_side_label(signal.side_sign),
        idler_side=_side_label(idler.side_sign),
    )


def _longitudinal_transfer(omega, v_grid, weights, residual_k, two_det,
                           Oc, Gamma_e, gamma_g, od_phys):
    """Doppler-averaged longitudinal factor Φ(δ) for δ in rad/s."""
    d = omega[:, None]
    twoden = Oc**2 - 4.0 * (d + 1j * gamma_g) * (
        d + two_det - residual_k * v_grid + 0.5j * Gamma_e)
    rho = (0.5 * od_phys * Gamma_e) * ((d + 1j * gamma_g) / twoden * weights).sum(axis=1)
    rho = (np.clip(rho.real, -PRED_RHO_CLIP, PRED_RHO_CLIP)
           + 1j * np.clip(rho.imag, 0.0, PRED_RHO_CLIP))
    return _sinc_complex(rho) * np.exp(1j * rho)


class GenericFWMSolver:
    """Reference-calibrated v3 engine for generic SFWM biphoton estimates."""

    def __init__(self, topology):
        self.topology = topology

    def _fields_from_params(self, params):
        pump_power = params.get("pump_biphoton_uw", self.topology.default_pump_uw)
        coupling_power = params.get("coupling_mw", self.topology.default_coupling_mw)
        out = []
        for field in self.topology.fields:
            rabi = field.rabi_mhz
            detuning = field.detuning_mhz
            if field.role == "pump":
                rabi = beam.anchored_rabi_mhz(
                    2.0, pump_power, self.topology.default_pump_uw)
                detuning = params.get("pump_detuning_mhz", 0.0)
            elif field.role == "coupling":
                rabi = beam.anchored_rabi_mhz(
                    12.0, coupling_power, self.topology.default_coupling_mw)
                detuning = params.get("coupling_detuning_mhz", 0.0)
            out.append(_field_with(
                field,
                detuning_mhz=detuning,
                rabi_mhz=rabi,
                angle_deg=params.get(f"{field.role}_angle_deg", field.angle_deg),
                side_sign=_side_sign(params[f"{field.role}_side"])
                if field.role in ("signal", "idler")
                and f"{field.role}_side" in params else field.side_sign,
            ))
        return tuple(out)

    def _leg_optical_depth(self, density, L, gamma_mhz, wavelength_nm):
        """Homogeneous on-resonance OD scale for the subsequent velocity average.

        The longitudinal response integrates velocities separately, so this
        input uses the zero-Doppler limit with the natural linewidth.
        """
        gamma_nat = max(float(gamma_mhz), 1e-3) * MHZ_ANG
        lam = float(wavelength_nm) * NM
        k = 2.0 * np.pi / lam
        d2 = species.reduced_dipole_sq(gamma_nat, lam, 0.5, 0.5)
        area = np.pi * density * k * d2 / (constants.EPS_0 * constants.HBAR)
        return float(line_absorption(0.0, area, gamma_nat) * L)

    def _apply_longitudinal_response(self, kappa_tau, tau_s, v_grid, weights,
                                     residual_k, two_det, Oc, Gamma_e, gamma_g,
                                     od_phys):
        """Convolve the nonlinear response κ̃(τ) with the linear longitudinal
        function Φ̃(τ) (Du Eq. 15), done as a product in the conjugate domain:
        ψ = FFT⁻¹[FFT(κ̃)·Φ(δ)], with Φ(δ)=sinc(ρ̄)·e^{iρ̄} (Du Eq. 14) and ρ̄(δ)
        the OD-weighted EIT/slow-light phase (Chen Eq. 5). At OD→0, ρ̄→0, Φ→1 and
        ψ→κ̃ exactly (no reshaping)."""
        if od_phys <= 0:
            return np.asarray(kappa_tau)
        n = tau_s.size
        dt = float(tau_s[1] - tau_s[0])
        nfft = 4 * n
        K = np.fft.fft(np.asarray(kappa_tau), n=nfft)
        omega = 2.0 * np.pi * np.fft.fftfreq(nfft, dt)        # δ axis [rad/s]
        phi = _longitudinal_transfer(omega, v_grid, weights, residual_k, two_det,
                                     Oc, Gamma_e, gamma_g, od_phys)
        return np.fft.ifft(K * phi, n=nfft)[:n]

    def _predictive_waveform(self, params, fields, v_grid, weights, pump, coupling,
                             signal, idler, residual_k, density, L, pm_weight,
                             tau_axis_ns, od_value):
        """First-principles Doppler-averaged biphoton amplitude.

        Frequency-domain form of Chen et al. (Phys. Rev. Research 4, 023132 (2024))
        Eq. (3-5), equivalent to Kim et al. (Quantum Sci. Technol. 9, 045006 (2024))
        Eq. (2) and Du, Wen, Rubin (J. Opt. Soc. Am. B 25, C98 (2008)) Eq. (13-18):
        for each signal detuning δ the Maxwell velocity classes are coherently
        summed into the nonlinear coupling κ̃(δ) and the linear longitudinal
        function ρ̄(δ); the joint amplitude A(δ)=κ̃·sinc(ρ̄)·e^{iρ̄} is
        inverse-Fourier-transformed to relative time τ. The two-photon denominator
        carries the Ω_c² Autler-Townes term (no weak-coupling approximation), the
        decay envelope emerges from the transform (no injected lifetime), and the
        optical depth α enters κ̃/ρ̄ (group-delay / Sommerfeld-precursor reshaping).
        """
        Gamma_i = max(self.topology.levels[1].gamma_mhz, 0.1) * MHZ_ANG   # intermediate (idler leg)
        Gamma_e = max(self.topology.levels[2].gamma_mhz, 0.1) * MHZ_ANG   # excited Γ₃
        gamma_g = GROUND_DEPHASING_FRAC * Gamma_i                         # two-photon (ground) dephasing
        Op = pump.rabi_mhz * MHZ_ANG
        Oc = max(coupling.rabi_mhz, 1e-6) * MHZ_ANG
        dp = params.get("pump_detuning_mhz", 0.0) * MHZ_ANG
        two_det = (params.get("pump_detuning_mhz", 0.0)
                   + params.get("coupling_detuning_mhz", 0.0)) * MHZ_ANG

        # optical depth seen by the near-resonant idler leg → Chen's α. OD is a
        # measured quantity in these sources (like cell temperature), so use the
        # reference-anchored, density/L-scaled value where available; the in-cell
        # reshaping (ρ̄) it drives is still computed from first principles.
        od_phys = float(od_value)

        tau_s = tau_axis_ns * 1e-9

        # ---- Nonlinear response κ̃(τ): time-domain Kim et al. Eq. (2) ----
        # Per velocity class amp(v) = Ω_p Ω_c / [4·f₁·f₂ + Ω_c²] with the Ω_c²
        # Autler-Townes term in the two-photon denominator (no weak-coupling
        # approximation). The collective two-photon coherence is the coherent sum
        # over Maxwell velocity classes carrying the single-photon phase
        # e^{i k_P v τ}, ×natural decay e^{−Γ τ/2}, ×H(τ) (τ≥0 implicit). The
        # velocity-sum dephasing — not an injected lifetime — sets the BTW width.
        f1 = 0.5 * Gamma_i + 1j * (dp - pump.direction * pump.k * v_grid)
        f2 = 0.5 * Gamma_e + 1j * (two_det - residual_k * v_grid)
        amp_v = weights * (Op * Oc) / (4.0 * f1 * f2 + Oc**2)
        amp_v = amp_v * od_phys                          # κ ∝ α (OD) → rate ∝ OD²
        coherent = np.exp(1j * pump.k * v_grid[:, None] * tau_s[None, :])
        kappa_tau = (amp_v[:, None] * coherent).sum(axis=0)
        kappa_tau = kappa_tau * np.exp(-0.5 * Gamma_i * tau_s)

        # ---- Linear longitudinal response Φ̃(τ): Du Eq. (15) convolution ----
        # ρ̄(δ) (Chen Eq. 5) is the OD-weighted EIT / slow-light phase; the
        # longitudinal function Φ(δ)=sinc(ρ̄)·e^{iρ̄} (Du Eq. 14) reshapes the
        # waveform (group delay / Sommerfeld precursor) at high OD. Convolved with
        # κ̃(τ); at low OD ρ̄→0, Φ→1, ψ→κ̃ (no reshaping). OFF by default: the
        # lumped 4-level model overestimates the high-OD group-delay broadening
        # (it would smear the validated narrow telecom BTW), so the reshaping is a
        # diagnostic opt-in (`biphoton_od_reshaping`) rather than the default path.
        if params.get("biphoton_od_reshaping", False):
            psi_tau = self._apply_longitudinal_response(
                kappa_tau, tau_s, v_grid, weights, residual_k, two_det,
                Oc, Gamma_e, gamma_g, od_phys)
        else:
            psi_tau = kappa_tau
        psi_tau = psi_tau * math.sqrt(max(pm_weight, 0.0))   # transverse collection

        # Source spectral width from the waveform itself (predictive).
        bandwidth_mhz = _bandwidth_from_waveform_mhz(tau_s, psi_tau)

        # The phase slope at zero sideband detuning gives the delay in seconds.
        # Use the conjugate product to take the phase difference across a wrap.
        step = 1e-3 * min(gamma_g, Oc)
        phi_minus, phi_plus = _longitudinal_transfer(
            np.array([-step, step]), v_grid, weights, residual_k, two_det,
            Oc, Gamma_e, gamma_g, od_phys)
        tau_group = np.angle(phi_plus * phi_minus.conjugate()) / (2.0 * step)
        tau_rabi = 2.0 * np.pi / Oc
        regime = "group-delay" if tau_group > tau_rabi else "damped-Rabi"

        source_v = amp_v   # velocity-class source for the existing diagnostic plot

        return {
            "psi_tau": psi_tau,
            "source_v": source_v,
            "od_phys": float(od_phys),
            "bandwidth_mhz": float(bandwidth_mhz),
            "regime": regime,
            # Chen et al. ultimate spectral-brightness ceiling ≈ (π/2)·10⁶ pairs/s/MHz
            "brightness_limit_cps_per_mhz": float(0.5 * np.pi * 1e6),
        }

    def _predictive_velocity_step(self, params, pump, coupling, residual_k, T, iso):
        """Velocity-grid step that oversamples the velocity-space resonance of the
        nonlinear source |amp(v)| — the integrand of the velocity-class coherent
        sum in `_predictive_waveform`.

        The resonance is measured directly on a fine probe grid (so it follows the
        Autler-Townes / detuning broadening of the denominator, not just the bare
        linewidth), and the step is set to its FWHM / `PRED_V_OVERSAMPLE`. Falls
        back to the natural-linewidth width Γ/2k if the probe feature is
        ill-defined. This is only the *starting* step; `_converged_predictive_
        waveform` halves it further until the waveform width converges.
        """
        sigma_v = constants.thermal_speed(T, iso.mass, sqrt=math.sqrt)
        Gamma_i = max(self.topology.levels[1].gamma_mhz, 0.1) * MHZ_ANG
        Gamma_e = max(self.topology.levels[2].gamma_mhz, 0.1) * MHZ_ANG
        Oc = max(coupling.rabi_mhz, 1e-6) * MHZ_ANG
        Op = pump.rabi_mhz * MHZ_ANG
        dp = params.get("pump_detuning_mhz", 0.0) * MHZ_ANG
        two_det = (params.get("pump_detuning_mhz", 0.0)
                   + params.get("coupling_detuning_mhz", 0.0)) * MHZ_ANG
        kp = max(abs(pump.k), 1e-30)
        rk = max(abs(residual_k), 1e-30)
        # probe spacing resolves the narrowest bare-linewidth velocity scale
        narrow = min(Gamma_i / (2.0 * kp), Gamma_e / (2.0 * rk))
        dv_probe = max(narrow / 6.0, 1e-3)
        n_probe = int(min(2.0 * 3.2 * sigma_v / dv_probe + 1.0, 200000))
        v = np.linspace(-3.2 * sigma_v, 3.2 * sigma_v, max(n_probe, 64))
        w = np.exp(-v**2 / (2.0 * sigma_v**2))
        f1 = 0.5 * Gamma_i + 1j * (dp - pump.direction * pump.k * v)
        f2 = 0.5 * Gamma_e + 1j * (two_det - residual_k * v)
        integ = np.abs(w * (Op * Oc) / (4.0 * f1 * f2 + Oc**2))
        res_fwhm = fwhm_interp(v, integ)
        if not np.isfinite(res_fwhm) or res_fwhm <= 0:
            res_fwhm = Gamma_i / (2.0 * kp)
        return max(res_fwhm / PRED_V_OVERSAMPLE, 1e-3)

    def _converged_predictive_waveform(self, params, fields, pump, coupling, signal,
                                       idler, residual_k, density, L, pm_weight,
                                       tau_axis_ns, od_value, T, iso, user_step):
        """Predictive biphoton waveform on a velocity grid auto-refined to
        convergence.

        The coherent sum over Maxwell velocity classes is a discretized Fourier
        integral of the narrow source |amp(v)|; a step coarser than that resonance
        aliases it, so the navigate-only `biphoton_velocity_step` (sized for the
        σ_v-wide Doppler profile) is far too coarse and the absolute BTW width
        swings with it. Start from a step that oversamples the measured resonance —
        never coarser than the user step — then halve until the |ψ|² FWHM is stable
        within `PRED_V_FWHM_TOL`. Returns the converged waveform, its velocity grid
        and weights, the step used, and a convergence flag (False if the point cap
        `PRED_V_MAX_POINTS` is reached first → width is qualitative only).
        """
        def solve_on(v_grid, weights):
            wf = self._predictive_waveform(
                params, fields, v_grid, weights, pump, coupling, signal, idler,
                residual_k=residual_k, density=density, L=L, pm_weight=pm_weight,
                tau_axis_ns=tau_axis_ns, od_value=od_value)
            fw = float(fwhm_interp(tau_axis_ns, np.abs(wf["psi_tau"])**2))
            return wf, fw

        dv = min(float(user_step),
                 self._predictive_velocity_step(params, pump, coupling,
                                                residual_k, T, iso))
        v_grid, weights = doppler.velocity_grid(T, dv=dv, cutoff_sigma=3.0,
                                                mass=iso.mass)
        wf, fw = solve_on(v_grid, weights)
        converged = False
        for _ in range(PRED_V_MAX_REFINE):
            step2 = 0.5 * dv
            v2, w2 = doppler.velocity_grid(T, dv=step2, cutoff_sigma=3.0,
                                           mass=iso.mass)
            if v2.size > PRED_V_MAX_POINTS:
                break
            wf2, fw2 = solve_on(v2, w2)
            rel = abs(fw2 - fw) / max(fw2, 1e-12)
            v_grid, weights, wf, dv, fw = v2, w2, wf2, step2, fw2
            if np.isfinite(rel) and rel < PRED_V_FWHM_TOL:
                converged = True
                break
        return wf, v_grid, weights, float(dv), bool(converged)

    def compute_biphoton(self, params):
        T = params.get("biphoton_temp_c", self.topology.default_temp_c) + 273.15
        L = params.get("biphoton_cell_mm", self.topology.default_cell_mm) * 1e-3
        fields = self._fields_from_params(params)
        fmap = {f.role: f for f in fields}
        detail = params.get("phase_detail", "Balanced")
        phase_detail = "Fine" if str(detail).lower() == "fine" else "Balanced"
        iso = self.topology.isotope
        v_step = params.get("biphoton_velocity_step", 2.0)
        pump = fmap["pump"]
        coupling = fmap["coupling"]
        signal = fmap["signal"]
        idler = fmap["idler"]
        delta_k = phase_mismatch(
            fields,
            signal_angle_deg=signal.angle_deg,
            idler_angle_deg=idler.angle_deg,
            reference_delta_k=self.topology.reference_delta_k,
        )
        pm_weight_longitudinal = float(phase_matching_weight(np.array([delta_k]), L)[0])
        delta_k_absolute = phase_mismatch(
            fields,
            signal_angle_deg=signal.angle_deg,
            idler_angle_deg=idler.angle_deg,
            reference_delta_k=0.0,
        )
        pm_weight_absolute = float(phase_matching_weight(
            np.array([delta_k_absolute]), L)[0])
        vector_pm = phase_mismatch_vector(
            fields,
            signal_angle_deg=signal.angle_deg,
            idler_angle_deg=idler.angle_deg,
            reference_delta_k=self.topology.reference_delta_k,
        )
        pm_weight = float(phase_matching_weight(
            np.array([vector_pm["delta_k_vector"]]), L)[0])
        density = species.number_density(iso, T)
        default_density = species.number_density(iso, self.topology.default_temp_c + 273.15)
        residual_k = pump.direction * pump.k + coupling.direction * coupling.k
        if self.topology.reference_od is not None:
            od_estimate = (self.topology.reference_od
                           * density / max(default_density, 1e-30)
                           * L / max(self.topology.default_cell_mm * 1e-3, 1e-30))
        else:
            od_estimate = np.nan

        # Absolute pair rate is reference-anchored. Its current scalar law uses
        # pump power, sqrt(coupling power), vector phase matching and sqrt(N/N_ref);
        # OD/cell length shape the waveform path but do not directly scale this
        # rate. The lumped model does not pin the collection coefficient, so the
        # magnitude remains anchored like the squeezing `line_strength` residual.
        pump_mw = params.get("pump_biphoton_uw", self.topology.default_pump_uw) * 1e-3
        coupling_power_scale = max(
            params.get("coupling_mw", self.topology.default_coupling_mw), 0.0
        ) / max(self.topology.default_coupling_mw, 1e-12)
        pair_rate = (self.topology.pair_rate_cps_per_mw * pump_mw
                     * math.sqrt(coupling_power_scale) * pm_weight
                     * math.sqrt(max(density, 1e-30) / max(default_density, 1e-30)))

        model = params.get("biphoton_model", BIPHOTON_PREDICTIVE)
        predictive = model == BIPHOTON_PREDICTIVE
        tau_axis_ns = np.linspace(0.0, params.get("tau_max_ns", 12.0), 481)

        if predictive:
            od_value = (od_estimate if np.isfinite(od_estimate)
                        else self._leg_optical_depth(
                            density, L, self.topology.levels[1].gamma_mhz,
                            idler.wavelength_nm))
            # The velocity-class coherent sum aliases on a step coarser than the
            # narrow source resonance, so the predictive grid is auto-refined to
            # convergence (the user `biphoton_velocity_step` acts as an upper bound).
            wf, v_grid, weights, v_step, velocity_converged = (
                self._converged_predictive_waveform(
                    params, fields, pump, coupling, signal, idler,
                    residual_k=residual_k, density=density, L=L,
                    pm_weight=pm_weight, tau_axis_ns=tau_axis_ns,
                    od_value=od_value, T=T, iso=iso, user_step=v_step))
            psi_tau = wf["psi_tau"]
            source_v = wf["source_v"]
            od_estimate = wf["od_phys"]
            source_bandwidth_mhz = wf["bandwidth_mhz"]
            regime = wf["regime"]
            brightness_limit = wf["brightness_limit_cps_per_mhz"]
        else:
            v_grid, weights = doppler.velocity_grid(
                T, dv=v_step, cutoff_sigma=3.0, mass=iso.mass)
            lower_gamma = max(self.topology.levels[1].gamma_mhz, 0.1) * MHZ_ANG
            upper_gamma = max(self.topology.levels[2].gamma_mhz, 0.1) * MHZ_ANG
            pump_det = params.get("pump_detuning_mhz", 0.0) * MHZ_ANG
            two_det = (params.get("pump_detuning_mhz", 0.0)
                       + params.get("coupling_detuning_mhz", 0.0)) * MHZ_ANG
            lower = lower_gamma / 2.0 + 1j * (pump_det - pump.direction * pump.k * v_grid)
            upper = upper_gamma / 2.0 + 1j * (two_det - residual_k * v_grid)
            drive = (pump.rabi_mhz * coupling.rabi_mhz) * (MHZ_ANG ** 2)
            source_v = weights * drive / (lower * upper)
            source_v *= math.sqrt(max(pm_weight, 0.0))
            tau_s = tau_axis_ns * 1e-9
            phase_k = abs(idler.k)
            coherent = np.exp(1j * phase_k * v_grid[:, None] * tau_s[None, :])
            psi_tau = (source_v[:, None] * coherent).sum(axis=0)
            psi_tau *= np.exp(-tau_axis_ns / max(self.topology.emission_decay_ns, 1e-12))
            if np.max(np.abs(psi_tau)) > 0:
                psi_tau = psi_tau / np.max(np.abs(psi_tau))
            source_bandwidth_mhz = float(self.topology.reference_bandwidth_mhz or 300.0)
            regime = "calibrated"
            brightness_limit = float("nan")
            # Calibrated width is set by the injected emission lifetime, not the
            # velocity-sum dephasing, so it is stable on the user grid (no refine).
            velocity_converged = True

        return {
            "kind": "biphoton",
            "topology": self.topology,
            "fields": fields,
            "tau_axis_ns": tau_axis_ns,
            "psi_tau": psi_tau,
            "v_grid": v_grid,
            "velocity_weights": weights,
            "source_v": source_v,
            "angle_axis_deg": None,
            "phase_matching": None,
            # The optional 2-D map is built lazily by the figure path. Keep the
            # legacy keys as None so existing raw-result consumers fail soft.
            "signal_angle_axis_deg": None,
            "idler_angle_axis_2d_deg": None,
            "phase_matching_2d": None,
            "delta_k": float(delta_k),
            "delta_k_absolute": float(delta_k_absolute),
            "delta_k_z_relative": float(vector_pm["delta_k_z_relative"]),
            "delta_k_z_absolute": float(vector_pm["delta_k_z_absolute"]),
            "delta_k_x": float(vector_pm["delta_k_x"]),
            "delta_k_vector": float(vector_pm["delta_k_vector"]),
            "phase_match_weight": pm_weight,
            "phase_match_weight_vector": pm_weight,
            "phase_match_weight_longitudinal": pm_weight_longitudinal,
            "phase_match_weight_absolute": pm_weight_absolute,
            "phase_detail": phase_detail,
            "energy_mismatch_hz": float(energy_mismatch_hz(fields)),
            "pair_rate_cps": float(pair_rate),
            "density": float(density),
            "od_estimate": float(od_estimate),
            "source_bandwidth_mhz": float(source_bandwidth_mhz),
            "temperature_K": float(T),
            "cell_length_m": float(L),
            "residual_two_photon_k": float(residual_k),
            "biphoton_model": model,
            "predictive": bool(predictive),
            "regime": regime,
            "brightness_limit_cps_per_mhz": float(brightness_limit),
            "velocity_step_used": float(v_step),
            "velocity_converged": bool(velocity_converged),
            "notes": self.topology.notes,
        }


@functools.lru_cache(maxsize=32)
def _solver_from_topology_key(key):
    return GenericFWMSolver(_topology_from_key(*key))


_BIPHOTON_READOUT_PARAM_KEYS = frozenset({
    "signal_eff_pct", "idler_eff_pct", "dark_signal_cps", "dark_idler_cps",
    "coincidence_window_ns", "timing_jitter_ns", "filter_bandwidth_mhz",
    "phase_detail",
})


def _biphoton_source_items(params):
    return tuple(sorted(
        (key, value) for key, value in params.items()
        if key not in _BIPHOTON_READOUT_PARAM_KEYS))


@functools.lru_cache(maxsize=8)
def _cached_biphoton_source(param_items):
    params = dict(param_items)
    return _solver_from_topology_key(
        _topology_cache_key(params)).compute_biphoton(params)
