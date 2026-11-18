"""Streamlit-facing FWM scheme, controls, plots and extra views."""
from .. import fwm
from .common import (
    BIPHOTON_MODELS, BIPHOTON_MODEL_LABELS, BIPHOTON_PREDICTIVE, CS_CHANNEL_795, CS_CHANNEL_917,
    CS_CHANNEL_LABELS, DEFAULT_BRANCH, FIDELITY_FAST, FIDELITY_LABELS, FWM_FIDELITY, L_CELL,
    MODE_BIPHOTON, MODE_LABELS, MODE_SEEDED, PHASE_LEGACY, PHASE_ULTRA, PRED_V_FWHM_TOL,
    PROBE_WAVELENGTH_NM, QE_DETECTOR, RESPONSE_POLE, SCAN_START_STRIDE, SCAN_TOLERANCE_DB,
    SCAN_TOLERANCE_LOG10_GAIN, SEEDED_FLOQUET_ORDER, SEEDED_PHASE_ANGLE_DEG,
    SEEDED_POST_CELL_LOSS_PCT, SEEDED_REFERENCE_RESIDUAL, SIDE_CHOICES, SIDE_PLUS, TOPOLOGY_CS_BTW,
    TOPOLOGY_DIAMOND, TOPOLOGY_LABELS, TOPOLOGY_RB87_TELECOM, TPD_LIMIT_MHZ, WINDOW_GHZ, W_PROBE,
    W_PUMP, _tier_response_method, normalize_fidelity,
)
from ..base import ExtraView, ParamSpec, Preset, Scheme
from ...fwm_gain_calibration import (
    MODELS as GAIN_CALIBRATION_MODELS, MODEL_CHI, MODEL_LAB, MODEL_LAB_REFIT, MODEL_LITERATURE, MODEL_STARK,
    MODEL_ZEEMAN, calibrate_spectrum as calibrate_experimental_gain,
)
from .readout import (
    NOISE_EXPERIMENTAL_JASPERSE, NOISE_PASSIVE_L1, SQUEEZING_INTENSITY_DIFFERENCE,
    SQUEEZING_QUADRATURE_PHASE, _detection_efficiency_from_params, _label_noise_axis,
    _measured_gain_context, _plot_reference, _power_gain_metric, _seeded_noise_caption,
    _seeded_passive_l1_readout, _seeded_sidebar_readouts,
)
from .biphoton import (
    _biphoton_source_items, _cached_biphoton_source, _default_biphoton_geometry,
    _solver_from_topology_key, _topology_cache_key, topology_from_params,
)
from .phase_matching import (
    _side_label, phase_matching_weight, phase_mismatch_vector_grid, transverse_matched_angle_deg,
)
from ...biphoton_evidence import biphoton_output_evidence
from .seeded import branch_center_GHz
from ... import detection, fwm_gain_closure, fwm_measured_gain, observables
from ...fwm_quadrature import experimental_quadrature_phase_scan
import numpy as np


class FWMScheme(Scheme):
    name = "fwm"
    cluster = "D — Wave mixing"
    title = "Four-wave mixing (Squeezing / Biphoton)"
    cache_version = "fwm-experimental-jasperse-gain-v30-faddeeva-absorption"
    defaults_version = "fwm-experimental-jasperse-v4"
    cache_observables = True
    supports_headless_observables = True
    # Dev note: Gain diagnostic is the original 85Rb double-Lambda seeded-gain
    # model (regression-anchored); Biphoton is a newer, less-calibrated
    # spontaneous-FWM source estimate shared across cascade/diamond level
    # schemes rather than fit per atom/transition.
    caption = ("Seeded ⁸⁵Rb D1 mean-field Squeezing indicator and spontaneous "
               "biphoton estimates.")

    def atom_conditions(self, params):
        if params.get("mode", MODE_SEEDED) != MODE_BIPHOTON:
            fixed = {"Atom": "Rb", "Isotope": "85", "Transition line": "D1"}
        else:
            topology = topology_from_params(params)
            atom = topology.isotope_name.rstrip("0123456789")
            fixed = {"Atom": atom, "Isotope": topology.isotope_name[len(atom):]}
            if topology.name != TOPOLOGY_DIAMOND:
                fixed["Transition line"] = "D2"
        return dict(fields=("temp_c", "cell_mm", "biphoton_temp_c", "biphoton_cell_mm"), fixed=fixed)

    def migrate_default_params(self, params):
        replay_flags = ("gain_closure_enabled", "legacy_gain_calibration_fixture")
        migrated = dict(params)
        # Saved settings without a model selector used the original lab fit.
        migrated.setdefault("gain_calibration_model", MODEL_LAB)
        for name in replay_flags:
            if name in migrated and not isinstance(migrated[name], bool):
                raise ValueError(f"{name} must be true or false.")
        fwm_gain_closure.reject_retired_replay(
            *(migrated.pop(name, False) for name in replay_flags))
        return migrated

    def param_schema(self):
        seeded = {"mode": MODE_SEEDED}
        biphoton = {"mode": MODE_BIPHOTON}
        cs_btw = {"mode": MODE_BIPHOTON, "topology": TOPOLOGY_CS_BTW}
        diamond = {"mode": MODE_BIPHOTON, "topology": TOPOLOGY_DIAMOND}
        return [
            ParamSpec("mode", "Mode", "Mode", MODE_SEEDED,
                      choices=(MODE_SEEDED, MODE_BIPHOTON),
                      choice_labels=MODE_LABELS,
                      control="segmented", applies_defaults=True,
                      help="Switches between seeded squeezing and a biphoton "
                           "source model, then loads that mode's defaults."),
            ParamSpec("squeezing_view", "Squeezing view", "Readout",
                      SQUEEZING_INTENSITY_DIFFERENCE,
                      choices=(SQUEEZING_INTENSITY_DIFFERENCE, SQUEEZING_QUADRATURE_PHASE),
                      control="required_segmented", recompute=False, visible_if=seeded,
                      help="Intensity difference scans detuning. Quadrature phase scan rotates "
                           "both homodyne LO phases together at the selected detuning, using "
                           "the effective Jasperse model with shot-noise seed phase. Gain stays fixed."),
            ParamSpec("topology", "Topology", "Model", TOPOLOGY_RB87_TELECOM,
                      choices=(TOPOLOGY_RB87_TELECOM, TOPOLOGY_CS_BTW, TOPOLOGY_DIAMOND),
                      choice_labels=TOPOLOGY_LABELS,
                      visible_if=biphoton, applies_defaults=True,
                      help="Atomic level scheme for the biphoton source."),
            ParamSpec("cs_channel", "Cs wavelength pair", "Model", CS_CHANNEL_917,
                      choices=(CS_CHANNEL_917, CS_CHANNEL_795), visible_if=cs_btw,
                      choice_labels=CS_CHANNEL_LABELS,
                      applies_defaults=True,
                      help="Selects the two wavelengths in the Cs cascade."),
            ParamSpec("biphoton_model", "Source model", "Model", BIPHOTON_PREDICTIVE,
                      choices=BIPHOTON_MODELS, control="segmented",
                      choice_labels=BIPHOTON_MODEL_LABELS,
                      visible_if=biphoton, advanced=True,
                      help="Reduced model calculates the Doppler waveform; its pair-"
                           "rate scale remains reference-anchored. Reference model "
                           "reproduces the stored literature calibration."),
            ParamSpec("opd", "One-photon detuning Δ", "Detunings", 0.9,
                      -3.0, 3.0, 0.1, "GHz",
                      visible_if=seeded,
                      help="Pump detuning from the F=2→F′=3 transition."),
            ParamSpec("tpd", "Two-photon Detuning", "Detunings", -8.0,
                      -TPD_LIMIT_MHZ, TPD_LIMIT_MHZ, 1.0, "MHz", recompute=False,
                      visible_if=seeded,
                      help="Selects the operating point on the cached curve."),
            ParamSpec("temp_c", "Temperature", "Cell", 121.0,
                      60.0, 150.0, 1.0, "°C", visible_if=seeded),
            ParamSpec("cell_mm", "Cell length", "Cell", 12.5,
                      1.0, 100.0, 0.5, "mm", visible_if=seeded,
                      help="Vapor-cell length L. Enters the Maxwell-Bloch "
                           "propagation exp(M·L), so it recomputes the gain."),
            ParamSpec("transit_rate_khz", "Transit rate", "Cell", 100.0,
                      1.0, 1000.0, 1.0, "kHz", visible_if=seeded, advanced=True,
                      advanced_group="Model provenance",
                      help="Atom replacement rate; 100 kHz is an inherited estimate."),
            ParamSpec("pump_mw", "Pump power", "Beams", 600.0,
                      50.0, 1200.0, 10.0, "mW", visible_if=seeded),
            ParamSpec("probe_uw", "Seed power", "Beams", 8.0,
                      1.0, 200.0, 1.0, "µW", visible_if=seeded,
                      help="Seed power entering the cell."),
            ParamSpec("eom_residual_carrier_uw", "Residual EOM carrier power",
                      "Beams", 0.0, 0.0, 5000.0, 0.1, "µW",
                      visible_if=seeded, advanced=True, hidden=True,
                      advanced_group="EOM spectrum provenance",
                      help="Cell-plane residual carrier power. It is recorded and "
                           "claim-gated but not applied because its coupling, phase, "
                           "polarization, and noise transfer are uncalibrated."),
            ParamSpec("eom_other_sidebands_uw", "Other EOM sidebands power",
                      "Beams", 0.0, 0.0, 5000.0, 0.1, "µW",
                      visible_if=seeded, advanced=True, hidden=True,
                      advanced_group="EOM spectrum provenance",
                      help="Total cell-plane power outside the wanted sideband and "
                           "carrier. Recorded as unapplied provenance only."),
            ParamSpec("pump_waist_um", "Pump waist", "Beams",
                      W_PUMP * 1e6, 50.0, 2000.0, 10.0, "µm",
                      visible_if=seeded,
                      help="Pump 1/e² radius at the cell; sets pump intensity and "
                           "spatial overlap."),
            ParamSpec("probe_waist_um", "Seed waist", "Beams",
                      W_PROBE * 1e6, 50.0, 2000.0, 10.0, "µm",
                      visible_if=seeded,
                      help="Seed 1/e² radius at the cell; used in the seeded drive "
                           "and spatial overlap."),
            ParamSpec("seeded_angle_deg", "Pump–seed angle", "Beams",
                      SEEDED_PHASE_ANGLE_DEG, 0.0, 2.0, 0.05, "°",
                      visible_if=seeded,
                      help="Crossing angle used in the longitudinal phase mismatch."),
            ParamSpec("experimental_gain_calibration", "Empirical gain calibration",
                      "Detection & scaling", False, control="checkbox", recompute=False,
                      visible_if=seeded,
                      help="Apply the selected empirical gain model. Correction continues "
                           "outside its data range and after changing operating conditions."),
            ParamSpec("exact_readout_coordinates", "Exact source coordinates", "Numerics", False,
                      control="checkbox", recompute=False, advanced=True, visible_if=seeded,
                      help="Use a cached dense reference for focused gain and noise, including the "
                           "selected detuning and the 10/13 lab model's mapped source points. "
                           "The first calculation takes longer. The adaptive atomic scan stays separate. "
                           "Other empirical gain models keep their existing sampling."),
            ParamSpec("gain_calibration_model", "Gain calibration model",
                      "Detection & scaling", MODEL_LAB_REFIT,
                      choices=GAIN_CALIBRATION_MODELS, recompute=False,
                      visible_if=dict(seeded, experimental_gain_calibration=True),
                      help="The 10/13 refit uses exact source coordinates. The original lab fit is available for replay. "
                           "Literature options correct gains or complex susceptibility; "
                           "pump/Stark also recomputes the atomic response before propagation. "
                           "Zeeman pump uses a cached 24-state response with optical pumping, "
                           "conditional on zero field and orthogonal linear polarizations. "
                           "These options report grouped gain "
                           "validation separately from noise."),
            ParamSpec("noise_model", "Noise model", "Detection & scaling", NOISE_EXPERIMENTAL_JASPERSE,
                      choices=(NOISE_EXPERIMENTAL_JASPERSE, NOISE_PASSIVE_L1), recompute=False,
                      visible_if=dict(seeded, squeezing_view=SQUEEZING_INTENSITY_DIFFERENCE),
                      help="Experimental Jasperse includes both distributed loss reservoirs using effective parameters. "
                           "Passive L1 retains the restricted canonical-transfer diagnostic."),
            ParamSpec("jasperse_loss_pct", "Jasperse common cell loss", "Detection & scaling",
                      0.0, 0.0, 95.0, 1.0, "%", recompute=False, visible_if=seeded,
                      advanced=True,
                      help="Extra common in-cell loss assumed when matching the two gains. Zero uses the "
                           "minimum asymmetric loss; this is an effective model assumption."),
            ParamSpec("measured_gain_calibration", "Measured-data reference only",
                      "Detection & scaling", False, control="checkbox", recompute=False,
                      visible_if=seeded, hidden=True,
                      help="Show a separate measured-data interpolation at the matching 2026-10-13 lab preset. "
                           "Agreement at the 23 measured nodes is by construction, not independent validation. "
                           "Atomic gains and NF are unchanged; this is not a squeezing calibration."),
            ParamSpec("seed_input_assumption", "Measured-reference seed input assumption",
                      "Detection & scaling", "", hidden=True, recompute=False,
                      visible_if=seeded),
            ParamSpec("seed_classical_rin_scaled", "Seed classical noise a", "Detection & scaling",
                      0.0, 0.0, 1000.0, 0.1, "10⁻¹⁴ Hz⁻¹", recompute=False,
                      visible_if=seeded, format="%.3f",
                      help="One-sided classical relative-intensity-noise PSD at the seed input. "
                           "a and the calculated shot RIN b give input Fano factor 1+a/b; "
                           "a is not an additive gain-proportional noise coefficient."),
            ParamSpec("seed_shot_noise_b", "Seed shot noise b", "Detection & scaling",
                      None, 0.0, 10000.0, 0.0001, "10⁻¹⁴ Hz⁻¹", control="derived_slider",
                      recompute=False, visible_if=seeded, format="%.4f",
                      help="Calculated one-sided shot RIN b=2ℏω_seed/P_seed. "
                           "An integrated variance additionally requires measurement bandwidth."),
            ParamSpec("after_cell_eff_pct", "After-cell optical efficiency η_after cell",
                      "Detection & scaling", 100.0-SEEDED_POST_CELL_LOSS_PCT,
                      0.0, 100.0, 0.1, "%", recompute=False, visible_if=seeded,
                      help="Measured optical power transmission after the cell; excludes cell loss and detector QE."),
            ParamSpec("qe_pct", "Detector quantum efficiency η_QE", "Detection & scaling",
                      QE_DETECTOR*100.0, 0.0, 100.0, 0.1, "%", recompute=False, visible_if=seeded),
            ParamSpec("eta_probe_pct", "Total Probe Efficiency η_probe", "Detection & scaling",
                      None, 0.0, 100.0, 0.01, "%", control="derived_slider", recompute=False,
                      visible_if=seeded, help="Calculated pumped-cell absorption transmission × after-cell transmission × QE. Stimulated emission is retained separately. Cell loss is already included in NF propagation."),
            ParamSpec("eta_conjugate_pct", "Total Conjugate Efficiency η_conjugate", "Detection & scaling",
                      None, 0.0, 100.0, 0.01, "%", control="derived_slider", recompute=False,
                      visible_if=seeded, help="Calculated pumped-cell absorption transmission × after-cell transmission × QE. Positive absorption is separated from stimulated emission with the same reduced Lindblad model."),
            ParamSpec("loss_pct", "Loss after cell", "Detection & scaling",
                      SEEDED_POST_CELL_LOSS_PCT,
                      0.0, 50.0, 0.5, "%", visible_if=seeded, hidden=True, recompute=False),
            ParamSpec("line_strength", "Inherited residual factor",
                      "Detection & scaling", SEEDED_REFERENCE_RESIDUAL,
                      0.2, 5.0, 0.01, "×",
                      visible_if=seeded, advanced=True, hidden=True,
                      advanced_group="Seeded coupling factorization",
                      help="Backward-compatible dimensionless residual. The physical "
                           "macroscopic normalization — Rb85 D1 hyperfine Clebsch-Gordan "
                           "strengths × 1/[2(2I+1)] — is computed from first principles. "
                           "The trace-normalized rho_ss supplies the manifold population "
                           "exactly once. The inherited 0.74 residual has not been "
                           "refitted after the normalization and Maxwell-convention "
                           "corrections, so it does not anchor measured gain or squeezing. "
                           "It remains separate "
                           "because no measurements identify a unique split of 0.74 among "
                           "overlap, polarization, and Zeeman participation; the three "
                           "unit-default penalties below represent only additional loss "
                           "relative to that anchored setup."),
            ParamSpec("mode_overlap_penalty", "Additional mode-overlap penalty",
                      "Detection & scaling", 1.0,
                      0.0, 1.0, 0.01, "×", visible_if=seeded, advanced=True,
                      hidden=True,
                      advanced_group="Seeded coupling factorization",
                      help="One-sided effective-coupling penalty for unresolved "
                           "transverse pump-seed mode mismatch beyond what the 0.74 "
                           "reference anchor already absorbs. Unity adds no penalty. "
                           "This is separate from Ultra's normalized axial crossing-angle "
                           "profile, so do not use it to count that geometry twice."),
            ParamSpec("polarization_penalty", "Additional polarization penalty",
                      "Detection & scaling", 1.0, 0.0, 1.0, 0.01, "×",
                      visible_if=seeded, advanced=True, hidden=True,
                      advanced_group="Seeded coupling factorization",
                      help="One-sided effective-coupling penalty for extra leakage out "
                           "of the intended double-Lambda polarization channel relative "
                           "to the anchored reference. It is not a raw Stokes or "
                           "optical-power purity."),
            ParamSpec("zeeman_participation_penalty",
                      "Additional Zeeman-participation penalty",
                      "Detection & scaling", 1.0, 0.0, 1.0, 0.01, "×",
                      visible_if=seeded, advanced=True, hidden=True,
                      advanced_group="Seeded coupling factorization",
                      help="One-sided effective-coupling penalty for additional loss of "
                           "addressed m_F pathways relative to the anchored reference. "
                           "Unity keeps the lumped hyperfine model; this scalar does not "
                           "turn on the 24-level Zeeman diagnostic or a full Floquet solve."),
            ParamSpec("biphoton_temp_c", "Temperature", "Cell & beams", 90.0,
                      30.0, 160.0, 1.0, "°C", visible_if=biphoton),
            ParamSpec("biphoton_cell_mm", "Cell length", "Cell & beams", 12.5,
                      1.0, 100.0, 0.5, "mm", visible_if=biphoton,
                      advanced=True),
            ParamSpec("pump_biphoton_uw", "Pump power", "Fields", 10.0,
                      0.1, 200.0, 0.1, "µW", visible_if=biphoton),
            ParamSpec("coupling_mw", "Coupling strength", "Fields", 1.0,
                      0.01, 50.0, 0.01, "×", visible_if=biphoton,
                      help="Relative drive amplitude; this is a scale factor, not "
                           "an absolute power."),
            ParamSpec("pump_detuning_mhz", "Pump detuning", "Detunings", 0.0,
                      -2000.0, 2000.0, 10.0, "MHz", visible_if=biphoton),
            ParamSpec("two_photon_detuning_mhz", "Two-photon detuning", "Detunings", 0.0,
                      -2000.0, 2000.0, 10.0, "MHz", visible_if=biphoton,
                      help="Delta_p + Delta_c. Internally this sets the coupling "
                           "detuning relative to the pump detuning."),
            ParamSpec("coupling_detuning_mhz", "Coupling detuning", "Detunings", 0.0,
                      -2000.0, 2000.0, 10.0, "MHz", visible_if=biphoton,
                      hidden=True, recompute=False),
            ParamSpec("signal_angle_deg", "Signal collection angle", "Phase matching", 1.5,
                      0.0, 10.0, 0.1, "°", visible_if=biphoton),
            ParamSpec("idler_angle_offset_deg", "Idler angle offset", "Phase matching",
                      0.0, -5.0, 5.0, 0.05, "°", visible_if=biphoton,
                      help="Offset from the transverse phase-matched idler angle "
                           "derived from the selected topology and signal angle."),
            ParamSpec("idler_angle_deg", "Idler angle", "Phase matching",
                      transverse_matched_angle_deg(1529.37, 780.24, 1.5),
                      0.0, 10.0, 0.1, "°", visible_if=biphoton,
                      hidden=True, recompute=False),
            ParamSpec("signal_side", "Signal side", "Phase matching", SIDE_PLUS,
                      choices=SIDE_CHOICES, visible_if=biphoton,
                      hidden=True,
                      help="Transverse collection side used in vector phase matching."),
            ParamSpec("idler_side", "Idler side", "Phase matching", SIDE_PLUS,
                      choices=SIDE_CHOICES, visible_if=biphoton,
                      hidden=True,
                      help="Opposite side flips the idler transverse wavevector."),
            ParamSpec("diamond_pump_nm", "Diamond pump wavelength", "Fields", 780.0,
                      300.0, 2000.0, 1.0, "nm", visible_if=diamond,
                      advanced=True),
            ParamSpec("diamond_coupling_nm", "Diamond coupling wavelength", "Fields", 776.0,
                      300.0, 2000.0, 1.0, "nm", visible_if=diamond,
                      advanced=True),
            ParamSpec("diamond_signal_nm", "Diamond signal wavelength", "Fields", 795.0,
                      300.0, 2000.0, 1.0, "nm", visible_if=diamond,
                      advanced=True),
            ParamSpec("diamond_idler_nm", "Diamond idler wavelength", "Fields", 761.702,
                      300.0, 2500.0, 0.001, "nm", visible_if=diamond,
                      advanced=True),
            ParamSpec("signal_eff_pct", "Signal efficiency", "Detection & scaling", 10.0,
                      0.1, 95.0, 0.1, "%", visible_if=biphoton,
                      advanced=True, recompute=False),
            ParamSpec("idler_eff_pct", "Idler efficiency", "Detection & scaling", 10.0,
                      0.1, 95.0, 0.1, "%", visible_if=biphoton,
                      advanced=True, recompute=False),
            ParamSpec("dark_signal_cps", "Signal background", "Detection & scaling", 2000.0,
                      0.0, 100000.0, 100.0, "cps", visible_if=biphoton,
                      advanced=True, recompute=False),
            ParamSpec("dark_idler_cps", "Idler background", "Detection & scaling", 2000.0,
                      0.0, 100000.0, 100.0, "cps", visible_if=biphoton,
                      advanced=True, recompute=False),
            ParamSpec("coincidence_window_ns", "Coincidence window", "Detection & scaling", 1.0,
                      0.01, 100.0, 0.01, "ns", visible_if=biphoton,
                      recompute=False),
            ParamSpec("timing_jitter_ns", "Timing jitter FWHM", "Detection & scaling", 0.55,
                      0.0, 5.0, 0.01, "ns", visible_if=biphoton,
                      advanced=True, recompute=False,
                      help="Net Gaussian signal-idler timing-difference response "
                           "FWHM. It broadens the detected correlation only; the "
                           "intrinsic source FWHM is reported separately."),
            ParamSpec("filter_bandwidth_mhz", "Filter bandwidth", "Detection & scaling", 300.0,
                      1.0, 5000.0, 1.0, "MHz", visible_if=biphoton,
                      recompute=False),
            ParamSpec("tau_max_ns", "Temporal window", "Numerics", 12.0,
                      1.0, 100.0, 1.0, "ns", visible_if=biphoton, advanced=True),
            ParamSpec("biphoton_velocity_step", "Velocity step", "Numerics", 2.0,
                      0.5, 20.0, 0.5, "m/s", visible_if=biphoton, advanced=True,
                      help="Maxwell velocity-grid step. The calibrated source model "
                           "uses it directly; the predictive model treats it as an "
                           "upper bound and auto-refines finer until the biphoton "
                           "width converges (a coarse step aliases the velocity-"
                           "class coherent sum)."),
            ParamSpec("resolution", "Solver detail", "Numerics", FIDELITY_FAST,
                      choices=tuple(FWM_FIDELITY.keys()), advanced=True,
                      choice_labels=FIDELITY_LABELS, visible_if=seeded,
                      help="Controls the scan grid and propagation refinements."),
            ParamSpec("velocity_measure", "Maxwell velocity measure", "Numerics", "exact",
                      choices=("exact", "truncated"), advanced=True, visible_if=seeded,
                      choice_labels={"exact": "Exact Maxwell", "truncated": "Historical 4σ"},
                      help="Use the untruncated pole mean or replay the previous 4-sigma grid."),
            ParamSpec("floquet_order", "Floquet order", "Numerics",
                      SEEDED_FLOQUET_ORDER, choices=(3, 4, 5), advanced=True,
                      visible_if=seeded, hidden=True,
                      help="Retained harmonics -N_F,...,+N_F. Every reported scan "
                           "is compared with N_F-1 over all complex response and "
                           "transfer coefficients; a failed gate is returned as "
                           "UNCONVERGED."),
            ParamSpec("phase_detail", "Phase-matching view", "Phase matching", "Balanced",
                      choices=("Balanced", "Fine"), visible_if=biphoton,
                      choice_labels={"Balanced": "1D", "Fine": "1D + 2D"},
                      control="segmented", advanced=True, hidden=True,
                      recompute=False,
                      help="Adds the 2D signal–idler acceptance map without rerunning "
                           "the source model."),
        ]

    def presets(self):
        context = fwm_measured_gain.nominal_context()
        # A default is a complete input snapshot: switching back to the Sim
        # reference also clears experiment-specific calibration and provenance.
        gold = dict(self.defaults(), **self._squeezing_defaults())
        values = dict(gold)
        values.update(
            opd=context["D_GHz"], temp_c=context["T"]-273.15,
            pump_mw=context["P_pump"]*1e3, probe_uw=context["P_probe"]*1e6,
            cell_mm=context["L"]*1e3, pump_waist_um=context["w_pump"]*1e6,
            probe_waist_um=context["w_probe"]*1e6,
            seeded_angle_deg=context["pump_probe_angle_deg"],
            tpd=float(fwm_measured_gain.internal_delta_mhz_from_eom(3034.0)),
            after_cell_eff_pct=97.09, loss_pct=2.91,
            qe_pct=100.0 * detection.photodiode_quantum_efficiency(
                0.58, PROBE_WAVELENGTH_NM * 1e-9),
            seed_input_assumption=fwm_measured_gain.SEED_INPUT_ASSUMPTION,
        )
        practical = dict(values, experimental_gain_calibration=True,
                         gain_calibration_model=MODEL_LAB_REFIT,
                         noise_model=NOISE_EXPERIMENTAL_JASPERSE, jasperse_loss_pct=0.0)
        return [
            Preset("2026-10-13 experiment", practical,
                   help="113 °C, nominal blue OPD ≈1 GHz, pump 294.804 mW. "
                        "Experimental Jasperse noise with empirical peak/width/gain correction. "
                        "Correction continues outside the measured range and at changed conditions. "
                        "Seed power uses the pre-window incident proxy P_off/(0.8046×0.9709)."),
            Preset("Sim Gold Reference", gold,
                   help="Sim reference: 121 °C, OPD 0.9 GHz, pump 600 mW and seed 8 µW. "
                        "Resets empirical calibration and measured-reference provenance."),
        ]

    def recommended_defaults(self, params):
        return {
            MODE_SEEDED: self._squeezing_defaults(),
            MODE_BIPHOTON: self._biphoton_defaults(params),
        }

    def _squeezing_defaults(self):
        return dict(mode=MODE_SEEDED, opd=0.9, tpd=-8.0, temp_c=121.0,
                    cell_mm=12.5, pump_mw=600.0, probe_uw=8.0,
                    after_cell_eff_pct=100.0-SEEDED_POST_CELL_LOSS_PCT,
                    seed_classical_rin_scaled=0.0,
                    measured_gain_calibration=False, seed_input_assumption="",
                    experimental_gain_calibration=False,
                    gain_calibration_model=MODEL_LAB_REFIT,
                    noise_model=NOISE_EXPERIMENTAL_JASPERSE, jasperse_loss_pct=0.0,
                    loss_pct=SEEDED_POST_CELL_LOSS_PCT,
                    transit_rate_khz=100.0,
                    eom_residual_carrier_uw=0.0,
                    eom_other_sidebands_uw=0.0,
                    pump_waist_um=W_PUMP * 1e6, probe_waist_um=W_PROBE * 1e6,
                    qe_pct=QE_DETECTOR * 100.0,
                    line_strength=SEEDED_REFERENCE_RESIDUAL,
                    mode_overlap_penalty=1.0,
                    polarization_penalty=1.0,
                     zeeman_participation_penalty=1.0,
                     resolution=FIDELITY_FAST,
                     floquet_order=SEEDED_FLOQUET_ORDER,
                     seeded_angle_deg=SEEDED_PHASE_ANGLE_DEG)

    def _biphoton_defaults(self, params):
        topology = params.get("topology", TOPOLOGY_RB87_TELECOM)
        base = dict(
            mode=MODE_BIPHOTON,
            topology=topology,
            biphoton_model=params.get("biphoton_model", BIPHOTON_PREDICTIVE),
            biphoton_cell_mm=12.5,
            pump_detuning_mhz=0.0,
            two_photon_detuning_mhz=0.0,
            coupling_detuning_mhz=0.0,
            signal_angle_deg=1.5,
            idler_angle_offset_deg=0.0,
            idler_angle_deg=transverse_matched_angle_deg(1529.37, 780.24, 1.5),
            signal_side=SIDE_PLUS,
            idler_side=SIDE_PLUS,
            signal_eff_pct=10.0,
            idler_eff_pct=10.0,
            dark_signal_cps=2000.0,
            dark_idler_cps=2000.0,
            coincidence_window_ns=1.0,
            filter_bandwidth_mhz=300.0,
            timing_jitter_ns=0.55,
            tau_max_ns=12.0,
            biphoton_velocity_step=2.0,
            phase_detail="Balanced",
        )
        if topology == TOPOLOGY_CS_BTW:
            base.update(cs_channel=params.get("cs_channel", CS_CHANNEL_917),
                        biphoton_temp_c=75.0, pump_biphoton_uw=20.0,
                        coupling_mw=1.0)
        elif topology == TOPOLOGY_DIAMOND:
            base.update(biphoton_temp_c=60.0, pump_biphoton_uw=20.0,
                        coupling_mw=1.0, diamond_pump_nm=780.0,
                        diamond_coupling_nm=776.0, diamond_signal_nm=795.0,
                        diamond_idler_nm=761.702)
        else:
            base.update(topology=TOPOLOGY_RB87_TELECOM, biphoton_temp_c=90.0,
                        pump_biphoton_uw=10.0, coupling_mw=1.0)
        base.update(_default_biphoton_geometry(base))
        return base

    def _biphoton_runtime_params(self, params):
        """Map the compact lab-facing controls onto the backend parameters."""
        out = dict(params)
        if out.get("mode", MODE_SEEDED) != MODE_BIPHOTON:
            return out

        topology = topology_from_params(out)
        signal = topology.field_map["signal"]
        idler = topology.field_map["idler"]
        signal_angle = float(out.get("signal_angle_deg", signal.angle_deg))
        out["signal_angle_deg"] = signal_angle

        if "two_photon_detuning_mhz" in out:
            two_det = float(out.get("two_photon_detuning_mhz", 0.0))
            pump_det = float(out.get("pump_detuning_mhz", 0.0))
            out["coupling_detuning_mhz"] = two_det - pump_det
        else:
            out["two_photon_detuning_mhz"] = (
                float(out.get("pump_detuning_mhz", 0.0))
                + float(out.get("coupling_detuning_mhz", 0.0))
            )

        if "idler_angle_offset_deg" in out:
            matched_idler = transverse_matched_angle_deg(
                signal.wavelength_nm, idler.wavelength_nm, signal_angle)
            out["idler_angle_deg"] = float(np.clip(
                matched_idler + float(out.get("idler_angle_offset_deg", 0.0)),
                0.0, 10.0))
        return out

    def info(self):
        return (
            "**Seeded FWM.** Computes mean-field seed and conjugate gain for the "
            "⁸⁵Rb D1 double-Λ system. The optional 2026-10-13 experimental calibration "
            "corrects peak position, width and arm gains and continues across detuning and operating conditions. "
            "Outside the measured interval it is an empirical model extrapolation. "
            "Experimental Jasperse noise uses both distributed vacuum loss reservoirs and an explicit "
            "common-loss assumption to match the displayed gain pair. "
            "Linear noise figure NF is "
            "Var(N_probe−N_conjugate)/mean(N_probe+N_conjugate); "
            "Squeezing [dB] is 10 log₁₀(NF). The alternative Passive L1 readout uses the canonical "
            "transfer before the pump-energy cap. It assumes a static annihilation-only "
            "vacuum completion and a bright seed. Points without this completion are "
            "unavailable. Without atomic Langevin covariance it is not a physical "
            "squeezing spectrum or a general lower bound for correlated atomic noise. "
            "Balanced directly samples the scan; Fast's L1 interpolation error "
            "has not been independently certified. "
            "Historical caches without a transfer map retain their legacy algebraic "
            "gain approximation. A pump-energy cap prevents unbounded "
            "small-signal gain but is not a depleted three-field solve. Numerical "
            "checks and solver provenance are listed under Model diagnostics. "
            "**Seed noise.** a is one-sided classical RIN [Hz⁻¹]; b=2ℏω_seed/P_seed "
            "is the calculated shot RIN, so the input Fano factor is 1+a/b. "
            "These are per-Hz PSDs; integrated noise needs the effective bandwidth. "
            "The former constant N and gain-linear a models are removed. "
            "Cell absorption, after-cell optical transmission and detector QE are "
            "reported separately. Cell loss already in propagation is never applied twice.\n\n"
            "**Biphoton.** The Reduced model calculates a Doppler-averaged waveform "
            "and vector phase matching. Absolute widths remain approximate and the "
            "pair-rate scale is anchored to a literature reference. The Reference "
            "model reproduces the stored calibration; agreement by construction is "
            "not an independent validation.\n\n"
            "**References**\n"
            "- G. Sim, H. Kim, H. S. Moon, *Sci. Rep.* **15**, 7727 (2025) "
            "(seeded 85Rb D1 double-Lambda gain & squeezing, regression anchor).\n"
            "- H. Kim, H. Jeong, H. S. Moon, [*Quantum Sci. Technol.* **9**, "
            "045006 (2024)](https://arxiv.org/abs/2402.06872) (Cs cascade BTW, Eq. 2).\n"
            "- H. Jeong, H. Kim, H. S. Moon, *Adv. Quantum Technol.* **7**, "
            "2300108 (2024) (87Rb telecom).\n"
            "- S. Du, J. Wen, M. H. Rubin, [*J. Opt. Soc. Am. B* **25**, C98 "
            "(2008)](https://arxiv.org/abs/0804.3981) (biphoton = nonlinear ⊛ "
            "linear response, Eq. 15).\n"
            "- Chen *et al.*, [*Phys. Rev. Research* **4**, 023132 (2022)]"
            "(https://arxiv.org/abs/2109.09062) (Doppler-averaged hot-vapor SFWM, Eq. 3-5).\n"
            "- J. Park, T. Jeong, H. S. Moon, [*Sci. Rep.* **10**, 16413 "
            "(2020)](https://www.nature.com/articles/s41598-020-73610-2) "
            "(cascade-type warm-atom biphoton waveform)."
        )

    def compute(self, params):
        fwm_gain_closure.reject_retired_replay(
            params.get("gain_closure_enabled", False),
            params.get("legacy_gain_calibration_fixture", False))
        if params.get("mode", MODE_SEEDED) == MODE_BIPHOTON:
            params = self._biphoton_runtime_params(params)
            return _solver_from_topology_key(
                _topology_cache_key(params)).compute_biphoton(params)
        inputs = self._seeded_compute_inputs(params)
        detuning = inputs.pop("D_GHz")
        raw = fwm.compute_spectrum(detuning, **inputs)
        if "stark_recompute_context" in raw:
            raw["focused_readout_reference"] = inputs["velocity_measure"] == "exact"
        return raw

    def _seeded_compute_inputs(self, params):
        center = branch_center_GHz(params["opd"], -1)
        fidelity = normalize_fidelity(params["resolution"])
        res = FWM_FIDELITY[fidelity]
        velocity_measure = params.get("velocity_measure", "exact")
        wanted_seed_uw = params["probe_uw"]
        alias_seed_uw = params.get("seed_wanted_sideband_uw", wanted_seed_uw)
        if not np.isclose(alias_seed_uw, wanted_seed_uw, rtol=0.0, atol=1e-12):
            raise ValueError(
                "probe_uw and seed_wanted_sideband_uw disagree; use one "
                "wanted-sideband power at the bridge boundary")
        return dict(
            D_GHz=params["opd"],
            T=params["temp_c"] + 273.15,
            P_pump=params["pump_mw"] * 1e-3,
            P_probe=wanted_seed_uw * 1e-6,
            line_strength=params["line_strength"],
            mode_overlap_penalty=params.get("mode_overlap_penalty", 1.0),
            polarization_penalty=params.get("polarization_penalty", 1.0),
            zeeman_participation_penalty=params.get(
                "zeeman_participation_penalty", 1.0),
            L=params["cell_mm"] * 1e-3,
            w_pump=params.get("pump_waist_um", W_PUMP * 1e6) * 1e-6,
            w_probe=params.get("probe_waist_um", W_PROBE * 1e6) * 1e-6,
            qe=(params.get("qe_pct", QE_DETECTOR * 100.0) / 100.0
                if params.get("legacy_gain_calibration_fixture", False) else 1.0),
            loss_frac=(params.get("loss_pct", SEEDED_POST_CELL_LOSS_PCT) / 100.0
                       if params.get("legacy_gain_calibration_fixture", False) else 0.0),
            detection_efficiency=(_detection_efficiency_from_params(params)
                                  if params.get("legacy_gain_calibration_fixture", False) else 1.0),
            coarse_points=res["coarse_points"], fine_points=0,
            scan_min=center - WINDOW_GHZ, scan_max=center + WINDOW_GHZ,
            velocity_step=res["velocity_step"],
            velocity_cutoff=res["velocity_cutoff"],
            velocity_measure=velocity_measure,
            phase_detail=res["phase_detail"],
            pump_probe_angle_deg=params.get("seeded_angle_deg", SEEDED_PHASE_ANGLE_DEG),
            model_fidelity=fidelity,
            gain_closure_enabled=params.get("gain_closure_enabled", False),
            legacy_gain_calibration_fixture=params.get("legacy_gain_calibration_fixture", False),
            legacy_mean_field_fixture=params.get("legacy_mean_field_fixture", False),
            floquet_order=params.get("floquet_order", SEEDED_FLOQUET_ORDER),
            enforce_floquet_convergence=True,
            response_method=_tier_response_method(fidelity, velocity_measure),
            scan_tolerance_dB=res.get("scan_tolerance_dB", SCAN_TOLERANCE_DB),
            scan_tolerance_log10_gain=res.get(
                "scan_tolerance_log10_gain", SCAN_TOLERANCE_LOG10_GAIN),
            scan_start_stride=res.get("scan_start_stride", SCAN_START_STRIDE),
            transit_rate=(2.0 * np.pi
                          * (params.get("transit_rate_khz", 100.0) * 1e3)),
            eom_residual_carrier_power=(
                params.get("eom_residual_carrier_uw", 0.0) * 1e-6),
            eom_other_sidebands_power=(
                params.get("eom_other_sidebands_uw", 0.0) * 1e-6),
            eom_seed_spectrum_provenance=params.get(
                "eom_seed_spectrum_provenance", "direct FWM input"),
            eom_seed_spectrum_status=params.get(
                "eom_seed_spectrum_status", "not supplied"),
            eom_seed_spectrum_application=params.get(
                "eom_seed_spectrum_application", "unapplied"),
            branch=DEFAULT_BRANCH,
        )

    def observables(self, raw, params, include_figures=True):
        if raw.get("kind") == "biphoton":
            params = self._biphoton_runtime_params(params)
            return self._biphoton_observables(
                raw, params, include_figures=include_figures)
        return self._seeded_observables(
            raw, params, include_figures=include_figures)

    def _seeded_observables(self, raw, params, include_figures=True):
        tpd = params["tpd"]
        squeezing_view = params.get("squeezing_view", SQUEEZING_INTENSITY_DIFFERENCE)
        phase_view = squeezing_view == SQUEEZING_QUADRATURE_PHASE
        noise_db = np.asarray(raw["gain_referred_noise_dB"])
        measured_reference = fwm._measured_gain_reference_readout(
            raw, enabled=params.get("measured_gain_calibration", False),
            seed_input_assumption=params.get("seed_input_assumption", ""),
            selected_delta_mhz=tpd)
        gain_model = params.get("gain_calibration_model", MODEL_LAB)
        literature_gain = gain_model in (MODEL_LITERATURE, MODEL_CHI, MODEL_STARK, MODEL_ZEEMAN)
        chi_gain = gain_model in (MODEL_CHI, MODEL_STARK, MODEL_ZEEMAN)
        stark_gain = gain_model == MODEL_STARK
        zeeman_gain = gain_model == MODEL_ZEEMAN
        selected = fwm.seeded_operating_readout(raw, params, include_passive=True)
        op = selected["atomic_operating_point"]
        gain_point = selected["gain_operating_point"]
        experimental_gain = selected["experimental_gain_calibration"]
        gain_active = experimental_gain["applied"]
        gain_curve = selected["gain_curve"]
        gain_axis = gain_curve["internal_delta_mhz"]
        displayed_gs = gain_curve["G_s"]
        displayed_gc = gain_curve["G_c"]
        displayed_point = experimental_gain["selected_point"] if gain_active else op
        noise_model = selected["noise_model"]
        experimental_noise = selected["experimental_jasperse_readout"]
        passive_readout = selected["passive_l1_readout"]
        passive_op = selected["passive_l1_operating_point"]
        noise_point = selected["noise_operating_point"]
        selected_noise_db = op["gain_referred_noise_dB"]
        noise_label = "Squeezing [dB]"
        noise_help = (
            "10 log10(NF), where NF=Var(N_probe-N_conjugate)/mean(N_probe+N_conjugate). "
            "Legacy data without a canonical map retain only their gain approximation.")
        noise_delta = "legacy data; physical squeezing unavailable"
        sidebar_readouts = {}
        cell_absorption = None
        if passive_readout is not None:
            noise_db = passive_readout["NF_dB"]
            selected_noise_db = passive_op["NF_dB"]
            noise_help = (
                "10 log10(NF), with seed classical RIN a and calculated shot RIN b. "
                "The input Fano factor is 1+a/b. Cell propagation is already included; "
                "only after-cell optical transmission and detector QE act downstream. "
                "Conditional annihilation-only vacuum completion, not microscopic RF noise.")
            noise_delta = (
                "conditional passive completion; physical squeezing unavailable"
                if passive_op["passive_l1_available"] else
                "passive completion unavailable; physical squeezing unavailable")
            cell_absorption = fwm._seeded_absorption_readout(raw, op["probe_GHz"])
            sidebar_readouts = _seeded_sidebar_readouts(
                raw, params, passive_readout, op["probe_GHz"], cell_absorption)
        if experimental_noise is not None:
            noise_db = experimental_noise["NF_dB"]
            selected_noise_db = noise_point["NF_dB"]
            noise_help = (
                "10 log10(NF) from an effective Jasperse distributed gain/loss model with both vacuum reservoirs. "
                "The two gains and assumed common cell loss set the effective parameters. "
                "This experimental approximation is separate from microscopic atomic-noise theory.")
            noise_delta = "effective Jasperse model"
            if gain_active:
                noise_delta += "; calibrated-gain approximation"
                if literature_gain:
                    noise_delta += "; gain validation does not validate noise"
            if noise_point["inference_approximate"]:
                noise_delta += "; approximate gain match"
            if not sidebar_readouts:
                sidebar_readouts = _seeded_sidebar_readouts(
                    raw, params, experimental_noise, op["probe_GHz"], cell_absorption)
        d_axis = (raw["probe_axis_GHz"] - raw["raman_center_minus_GHz"]) * 1e3
        claim_gate = raw.get("claim_gate", {})
        floquet_convergence = raw.get("floquet_convergence", {})
        if measured_reference["requested"]:
            noise_delta += "; measured gain interpolation does not recalibrate squeezing"

        phase_scan = (experimental_quadrature_phase_scan(experimental_noise, gain_axis, tpd)
                      if phase_view else None)
        fig = None
        if include_figures and phase_view:
            import matplotlib.pyplot as plt
            from ...plot_style import PALETTE

            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.plot(phase_scan["phase_degrees"], phase_scan["noise_a_db"],
                    color=PALETTE["cyan"], lw=1.8, label="Quadrature A")
            ax.plot(phase_scan["phase_degrees"], phase_scan["noise_b_db"],
                    color=PALETTE["rose"], lw=1.8, ls="--", label="Quadrature B (+90 deg)")
            ax.axhline(0.0, color="black", ls="--", lw=1.0, label="Shot noise (0 dB)")
            ax.set_xlabel("Common LO phase [deg]")
            ax.set_ylabel("Quadrature noise / shot noise [dB]")
            ax.set_xlim(0.0, 180.0)
            ax.set_xticks(np.arange(0.0, 181.0, 30.0))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
            if not phase_scan["available"]:
                ax.text(.5, .5, "Quadrature estimate unavailable at this operating point",
                        ha="center", va="center", transform=ax.transAxes, wrap=True)
            fig.tight_layout()
        elif include_figures:
            import matplotlib.pyplot as plt

            fig, (axG, axS) = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
            for ax in (axG, axS):
                ax.grid(alpha=0.3)
            axG.plot(gain_axis, displayed_gs, color="#1f77b4", lw=1.8,
                     label="Calibrated probe" if gain_active else "Probe")
            axG.axvline(tpd, color="crimson", ls="--", lw=1.2)
            _plot_reference(axG, 1.0, "G = 1 (unity gain)")
            axG.plot(gain_axis, displayed_gc, color="#ff7f0e", lw=1.4,
                     ls="-" if gain_active else "--",
                     label="Calibrated conjugate" if gain_active else "Conjugate")
            if gain_active and not literature_gain:
                bounds = sorted(fwm_measured_gain.internal_delta_mhz_from_eom(
                    np.array(fwm_measured_gain.EOM_SUPPORT_MHZ)))
                axG.axvspan(*bounds, facecolor=(1.0, .75, .28, .12),
                            edgecolor="#8a5d21", linestyle="--", linewidth=.9,
                            label="Measured RF interval; outside: gain extrapolation")
            axG.legend(fontsize=8)
            axG.scatter([tpd], [displayed_point["G_s"]], color="crimson", zorder=5)
            axG.set_ylabel("Zeeman pump + chi propagation gain" if gain_active and zeeman_gain else
                           "Stark + chi propagation gain" if gain_active and stark_gain else
                           "Chi-corrected propagation gain" if gain_active and chi_gain else
                           "Literature-corrected power gain" if gain_active and literature_gain else
                           "Calibrated pump-off-reference gain" if gain_active else
                           "Atomic probe / conjugate gain" if measured_reference["requested"] else
                           "Gain")
            if max(np.nanmax(displayed_gs), np.nanmax(displayed_gc)) > 50:
                axG.set_yscale("log")
            noise_axis = gain_axis if experimental_noise is not None else d_axis
            plot_noise = noise_db
            if passive_readout is not None or experimental_noise is not None:
                # Include the exact selected point so marker, curve and metric
                # share log10 of the selected linearly interpolated NF.
                idx = int(np.searchsorted(noise_axis, tpd))
                if idx == noise_axis.size or not np.isclose(noise_axis[idx], tpd, rtol=0.0, atol=1e-9):
                    noise_axis = np.insert(noise_axis, idx, tpd)
                    plot_noise = np.insert(noise_db, idx, selected_noise_db)
            axS.plot(noise_axis, plot_noise, color="#2ca02c", lw=1.8)
            axS.axvline(tpd, color="crimson", ls="--", lw=1.2)
            _label_noise_axis(axS, noise_point is not None)
            axS.scatter([tpd], [selected_noise_db],
                        color="crimson", zorder=5)
            axS.set_xlabel("Two-photon Detuning [MHz]")
            axS.set_xlim(-TPD_LIMIT_MHZ, TPD_LIMIT_MHZ)
            fig.tight_layout()

        metrics = [
            dict(label=noise_label,
                 value=(f"{selected_noise_db:.2f} dB" if np.isfinite(selected_noise_db)
                        else "unavailable"),
                 numeric_value=float(selected_noise_db) if np.isfinite(selected_noise_db) else None,
                 quantity="intensity_difference_noise_db",
                 delta=noise_delta, help=noise_help, tier="hero"),
            _power_gain_metric("Seed gain G_s", op["G_s"],
                 delta=("semi-empirical estimate" if raw.get(
                     "gain_closure", {}).get("applied") else None),
                 help="Probe power at the atomic-medium exit / seed power at its entrance. "
                      "The retired empirical gain correction is disabled in current runs.",
                 tier="hero"),
            _power_gain_metric("Conjugate gain G_c", op["G_c"],
                 help="Generated conjugate power gain at the selected detuning."),
        ]
        measured_rows = ""
        if measured_reference["requested"]:
            point = measured_reference["selected_point"]
            profile = measured_reference["profile"]
            if point["available"]:
                metrics[1] = _power_gain_metric(
                    "Probe gain, pump-off reference", point["measured_gain_probe"], tier="hero",
                    delta="conditional measured-data interpolation",
                    help="Interpolated pump-on probe output / measured pump-off seed detector output. "
                         "Distinct from the atomic input/output gain and from NF.")
                metrics[2] = _power_gain_metric(
                    "Conjugate gain, pump-off reference", point["measured_gain_conjugate"],
                    help="Interpolated conjugate output / the same pump-off seed detector output.")
            else:
                metrics.append(dict(label="Measured gain reference", value="unavailable",
                                    delta=point["status"], kind="status"))
            metrics.extend([
                _power_gain_metric("Atomic probe gain G_s", op["G_s"],
                     help="Unchanged mean-field power gain at the atomic-medium boundaries."),
                _power_gain_metric("Atomic conjugate gain G_c", op["G_c"],
                     help="Unchanged mean-field conjugate power gain at the atomic-medium boundaries."),
            ])
            probe_text = f"{point['measured_gain_probe']:.6g}" if point["available"] else "unavailable"
            conjugate_text = f"{point['measured_gain_conjugate']:.6g}" if point["available"] else "unavailable"
            measured_rows = (
                f"| Measured-reference status | {point['status']} |\n"
                f"| Measured-reference EOM frequency | {point['eom_MHz']:.6f} MHz |\n"
                f"| Probe gain, pump-off reference | {probe_text} |\n"
                f"| Conjugate gain, pump-off reference | {conjugate_text} |\n"
                f"| Gain denominator plane | {profile['reference_plane']} |\n"
                f"| Reference seed detector level | {profile['reference_seed_voltage_mV']:.6g} mV |\n"
                f"| Reference method | {profile['method']} |\n"
                "| Measured-node agreement | By construction; not independent validation |\n"
                f"| Predictive-fit development record | {profile['development_status']} |\n"
                f"| Independent validation | {profile['independent_validation']} |\n"
                f"| Seed proxy assumption | {measured_reference['seed_input_assumption'] or 'not adopted'} |\n"
                f"| Reference input-seed proxy | {profile['input_seed_proxy_W']*1e6:.6g} µW, "
                "external pre-window estimate assumed as atomic input; vapor input not independently measured |\n"
                f"| Atomic probe / conjugate gain | {op['G_s']:.6g} / {op['G_c']:.6g} |\n"
                "| NF scope | Gain-reference interpolation does not modify canonical transfer, noise NF, or detector power propagation |\n"
            )
            if measured_reference["reasons"]:
                measured_rows += "| Reference context mismatch | " + "; ".join(measured_reference["reasons"]) + " |\n"
        if gain_active:
            point = experimental_gain["selected_point"]
            gain_profile = experimental_gain.get("profile", {})
            if literature_gain:
                status_label = experimental_gain.get(
                    "validation_label", gain_profile.get(
                        "validation_label", "Experimental · grouped prediction not established"))
                if not point.get("within_training_domain", experimental_gain.get(
                        "within_training_domain", False)):
                    status_label += " · condition extrapolation"
                gain_help = (
                    "The conditional zero-field, orthogonal-linear 24-state response includes "
                    "pump optical pumping and is recomputed before propagation; "
                    if zeeman_gain else
                    "The pump Hamiltonian self-energy is corrected and the atomic response recomputed; "
                    if stark_gain else "") + (
                    "The full Zeeman diagonal response is retained while both off-diagonal "
                    "components share one residual scale before segmented propagation; "
                    "vacuum geometry and the pump refractive phase are kept separate. "
                    if zeeman_gain else
                    "Complex off-diagonal and full diagonal susceptibility are corrected "
                    "before segmented propagation; vacuum phase mismatch is retained. "
                    if chi_gain else "Power gain corrected using multiple experimental papers. ") + (
                    "Grouped prediction results apply to gain only; noise still uses "
                    "the assumed effective Jasperse closure.")
            else:
                status_label = ("Measured RF range" if point["within_measured_support"] else "RF extrapolation")
                if point["context_changed"]:
                    status_label += " · changed conditions"
                if experimental_gain["source_model_differences"]:
                    status_label += (" · different source settings" if gain_model == MODEL_LAB_REFIT
                                     else " · updated source model")
                gain_help = (
                    "Empirically corrected pump-on output / pump-off seed reference. "
                    + ("The refit uses the Ultra model at exact source coordinates. "
                       if gain_model == MODEL_LAB_REFIT else "The frozen fit used an older atomic model. ")
                    + "The correction continues beyond the 10/13 measured interval.")
            for index, key, arm in ((1, "G_s", "probe"), (2, "G_c", "conjugate")):
                metrics[index] = _power_gain_metric(
                    (f"Zeeman pump + χ {arm} gain" if zeeman_gain else
                           f"Stark + χ {arm} gain" if stark_gain else
                           f"χ-propagated {arm} gain" if chi_gain else
                           f"Literature-corrected {arm} gain" if literature_gain else
                           f"Calibrated {arm} gain"), point[key],
                    approximate=True, tier="hero" if index == 1 else "secondary",
                    delta=status_label, help=gain_help)
            if not measured_reference["requested"]:
                metrics.extend([
                    _power_gain_metric("Atomic probe gain G_s", op["G_s"]),
                    _power_gain_metric("Atomic conjugate gain G_c", op["G_c"]),
                ])
            if literature_gain:
                measured_rows += (
                    f"| Experimental gain calibration | {point['status']} |\n"
                    "| Calibration data | Multiple 85Rb D1 experimental papers |\n"
                    f"| Gain model | {experimental_gain.get('display_name', gain_model)} |\n"
                    f"| Grouped gain validation | {experimental_gain.get('validation_status', gain_profile.get('validation_status', 'not established'))} |\n"
                    f"| Correction | {gain_profile.get('method', 'Shared literature gain correction')} |\n"
                    f"| Correction formula | {gain_profile.get('formula', 'not supplied')} |\n"
                    f"| Calibrated probe / conjugate gain | {point['G_s']:.6g} / {point['G_c']:.6g} |\n"
                    f"| Calibrated gain basis | {experimental_gain.get('gain_quantity', 'Effective source power gain')} |\n"
                    "| Continuation | Applied across changed conditions; training-domain extrapolation is identified |\n"
                    "| Noise scope | Grouped gain validation does not validate a noise covariance or squeezing prediction |\n"
                )
                if chi_gain:
                    measured_rows += (
                        f"| χ off-diagonal scale κ | {point['offdiagonal_scale']:.6g} |\n"
                        f"| χ diagonal scale d (absorption and dispersion) | {point['diagonal_susceptibility_scale']:.6g} |\n"
                        "| Propagation | Corrected complex susceptibility; segmented overlap and pump depletion |\n"
                        "| Geometric phase mismatch | Retained independently of susceptibility scales |\n"
                    )
                if stark_gain:
                    stark_response = experimental_gain.get("pump_stark_response", {})
                    stark_response = stark_response if isinstance(stark_response, dict) else {}
                    measured_rows += (
                        f"| Pump/Stark self-energy fraction | {experimental_gain.get('pump_stark_fraction', point.get('pump_stark_fraction', 'not supplied'))} |\n"
                        f"| Pump self-energy counterterm [MHz] | {stark_response.get('counterterm_mhz', 'not supplied')} |\n"
                        f"| Corrected pump-dressed gap shift [MHz] | {stark_response.get('corrected_gap_shift_mhz', 'not supplied')} |\n"
                        f"| Corrected atomic Floquet status | {experimental_gain.get('corrected_atomic_floquet_status', 'not supplied')} |\n"
                        "| Canonical/noise scope | Raw canonical transfer and noise certification retained separately |\n"
                    )
                if zeeman_gain:
                    zeeman_populations = experimental_gain.get("zeeman_pump_populations", {})
                    zeeman_populations = zeeman_populations if isinstance(zeeman_populations, dict) else {}
                    zeeman_diagnostics = experimental_gain.get("corrected_atomic_response_diagnostics", {})
                    zeeman_diagnostics = zeeman_diagnostics if isinstance(zeeman_diagnostics, dict) else {}
                    zeeman_provenance = experimental_gain.get("zeeman_response_provenance", {})
                    zeeman_provenance = zeeman_provenance if isinstance(zeeman_provenance, dict) else {}
                    pump_phase = zeeman_provenance.get("pump_phase", {})
                    pump_phase = pump_phase if isinstance(pump_phase, dict) else {}
                    measured_rows += (
                        "| Zeeman atomic response | 24-state D1 with pump optical pumping |\n"
                        "| Polarization / magnetic field | Conditional zero field; orthogonal linear pump and seed |\n"
                        f"| Pump populations Fg2 / Fg3 | {zeeman_populations.get('ground_F2', 'not supplied')} / {zeeman_populations.get('ground_F3', 'not supplied')} |\n"
                        f"| Pump excited-state population | {zeeman_populations.get('excited_total', 'not supplied')} |\n"
                        f"| Pump density-matrix trace | {zeeman_populations.get('trace', 'not supplied')} |\n"
                        f"| Zeeman response finite | {zeeman_diagnostics.get('finite', 'not supplied')} |\n"
                        "| Effective phase mismatch | Vacuum geometry plus pump refractive phase from the same steady state |\n"
                        f"| Pump-index mismatch contribution [1/m] | {pump_phase.get('pump_index_added_mismatch_per_m', 'not supplied')} |\n"
                        "| Pump phase coefficient | Fixed physical coefficient 1; not fitted |\n"
                        "| Canonical/noise scope | Raw canonical transfer and noise certification retained separately |\n"
                    )
            else:
                measured_rows += (
                    f"| Experimental gain calibration | {point['status']} |\n"
                    "| Calibration data | 2026-10-13 measurements, 23 points, RF 3022–3044 MHz |\n"
                    "| Correction | Frozen peak/width and gain correction applied to the current model |\n"
                    f"| Calibrated probe / conjugate gain | {point['G_s']:.6g} / {point['G_c']:.6g} |\n"
                    "| Calibrated gain basis | Pump-on arm output / measured pump-off seed output |\n"
                    "| Continuation | Applied outside measurement support and across changed operating conditions |\n"
                )
                if gain_model == MODEL_LAB_REFIT:
                    measured_rows += (
                        "| Fit source | Ultra, untruncated Maxwell, exact arm-specific coordinates |\n"
                        "| Fit scope | All 23 points trained the empirical residual |\n")
                elif experimental_gain["source_model_differences"]:
                    measured_rows += (
                        "| Source model | Updated atomic model with the frozen fit coefficients |\n"
                        "| Fit scope | Exact agreement at the 23 training points applies to the frozen source arrays |\n"
                    )
                deviation = experimental_gain["nominal_node_deviation"]
                if deviation is not None:
                    errors = deviation["max_relative_error"]
                    measured_rows += (
                        f"| Max deviation at the {deviation['point_count']} training points "
                        f"(probe / conjugate) | {errors['probe']:.2%} / {errors['conjugate']:.2%} |\n"
                    )
                exact_deviation = experimental_gain.get("exact_nominal_node_deviation")
                if exact_deviation is not None:
                    errors = exact_deviation["max_relative_error"]
                    measured_rows += (
                        f"| Exact-coordinate deviation at the {exact_deviation['point_count']} training points "
                        f"(probe / conjugate) | {errors['probe']:.2%} / {errors['conjugate']:.2%} |\n"
                    )
        if phase_view:
            def quadrature_metric(label, key):
                value = phase_scan[key]
                return dict(
                    label=label,
                    value=f"{value:.2f} dB" if np.isfinite(value) else "unavailable",
                    tier="hero", approximate=True,
                    delta=(noise_delta if key == "minimum_db" else "Shot-noise seed phase assumed"),
                    help="Equal-weight collective difference mode (probe minus conjugate). "
                         "Both LO phases rotate together; B is A rotated by 90 degrees. "
                         "0 dB is shot noise; negative values indicate squeezing. "
                         "Static effective covariance, not a microscopic finite-RF prediction.")

            gain_metrics = [dict(metric, tier="secondary") for metric in metrics[1:3]]
            for metric, key, arm in zip(gain_metrics, ("G_s", "G_c"), ("Probe", "Conjugate")):
                value = gain_point[key]
                metric["numeric_value"] = float(value) if np.isfinite(value) else None
                metric["value"] = f"{value:.2f}" if np.isfinite(value) else "unavailable"
                if measured_reference["requested"] and not gain_active:
                    metric.update(label=f"{arm} gain {key}",
                                  delta="Current operating point",
                                  help="Atomic-medium power gain; independent of homodyne LO phase.")
            metrics = [
                quadrature_metric("Minimum quadrature noise", "minimum_db"),
                quadrature_metric("Maximum quadrature noise", "maximum_db"),
                *gain_metrics,
                dict(label="Operating detuning", value=f"{tpd:.3f} MHz", tier="secondary",
                     help="Fixed two-photon detuning during the common LO phase scan."),
                *[dict(metric, tier="secondary") for metric in metrics[3:]],
            ]
        cap = raw.get("pump_depletion_cap", float("inf"))
        small_signal = raw.get("G_s_smallsignal_peak", op["G_s"])
        depletion_limited = small_signal > 1.1 * cap
        solver = raw.get("atomic_solver_provenance", {})
        pump_reference = raw.get(
            "pump_weak_response_reference_provenance", {})
        noncollinear_reference = raw.get(
            "noncollinear_doppler_reference_provenance", {})
        solver_detail = FIDELITY_LABELS.get(
            raw.get("model_fidelity"), raw.get("model_fidelity", "unavailable"))
        phase_rows = ""
        if raw.get("phase_detail", PHASE_LEGACY) != PHASE_LEGACY:
            phase_rows = (
                f"| Solver detail | {solver_detail} |\n"
                f"| Phase model | {raw['phase_detail']} |\n"
                f"| Pump–seed angle | {raw['pump_probe_angle_deg']:.3f}° |\n"
                f"| Operating Δk_z | {op.get('delta_k_z', np.nan):.3e} 1/m |\n"
                f"| Vacuum Δk_z | {op.get('delta_k_z_vacuum', np.nan):.3e} 1/m |\n"
                f"| Propagation segments | {raw.get('phase_segments', 1)} |\n"
                f"| Segment absorption OD estimate | {raw.get('segment_absorption_od', 0.0):.3f} |\n"
            )
            if raw.get("phase_detail") == PHASE_ULTRA:
                phase_rows += (
                    f"| Dispersion convention | Option A (diagonal chi only) |\n"
                    f"| Dynamic depletion | {raw.get('ultra_dynamic_depletion', False)} |\n"
                    f"| Min pump remaining | {raw.get('ultra_pump_remaining_min', np.nan):.3e} W |\n"
                    f"| Distributed atomic Langevin covariance | unavailable |\n"
                    f"| Min Gaussian overlap | {raw.get('ultra_spatial_overlap_min', 1.0):.4f} |\n"
                    f"| Zeeman status | {raw.get('zeeman_status', 'inactive')} |\n"
                    f"| Zeeman CG-sum diagnostic | {raw.get('zeeman_correction', 1.0):.4f} |\n"
                )
        depletion_warning = (
            "\n⚠️ **Pump-depletion limited:** the small-signal gain exceeds the "
            "pump-energy budget, so the displayed gain is capped. Lower the "
            "temperature or raise the seed power to return to the linear regime.\n"
            if depletion_limited else ""
        )
        noise_rows = ""
        active_noise = experimental_noise if experimental_noise is not None else passive_readout
        if active_noise is not None:
            seed = active_noise["seed_noise"]
            axis = gain_curve["probe_axis_GHz"] if experimental_noise is not None else raw["probe_axis_GHz"]
            at_point = lambda value: float(np.interp(op["probe_GHz"], axis, value))
            noise_rows += (
                f"| Noise model | {noise_model} |\n"
                f"| NF | {noise_point['NF']:.6g} |\n"
                f"| {'Intensity-difference squeezing' if phase_view else 'Squeezing'} [dB] = 10 log10(NF) | {selected_noise_db:.6g} |\n"
                f"| Seed classical RIN a | {at_point(seed['classical_rin_per_Hz']):.6g} Hz⁻¹ |\n"
                f"| Seed shot RIN b = 2ℏω/P | {at_point(seed['shot_rin_per_Hz']):.6g} Hz⁻¹ |\n"
                f"| Input seed Fano factor = 1+a/b | {at_point(seed['seed_fano']):.6g} |\n"
                "| Noise convention | One-sided PSD per Hz; integrated variance requires effective bandwidth |\n"
                + ("| Transfer gain basis | Effective Jasperse fit to displayed gains; assumed common cell loss |\n"
                   if experimental_noise is not None else
                   "| Transfer gain basis | Canonical photon flux before pump cap; no current empirical correction |\n")
                +
                "| Cell losses in NF | Already in propagation; total efficiencies are not applied a second time |\n"
            )
            cell = raw.get("dressed_cell_transmission", {})
            for arm in ("probe", "conjugate"):
                transmission = cell.get(f"in_cell_{arm}_transmission")
                if transmission is not None:
                    value = float(np.interp(op["probe_GHz"], raw["probe_axis_GHz"], transmission))
                    noise_rows += f"| Pumped-cell {arm} net transmission (mixing off) | {value:.6g} |\n"
                if cell_absorption and arm in cell_absorption:
                    rates = cell_absorption[arm]
                    noise_rows += (
                        f"| Cell {arm} absorption coefficient | {rates['alpha_abs_per_m']:.6g} m⁻¹ |\n"
                        f"| Cell {arm} stimulated-emission coefficient | {rates['alpha_em_per_m']:.6g} m⁻¹ |\n"
                        f"| Cell {arm} absorption transmission η_in cell | {rates['in_cell_absorption_transmission']:.6g} |\n"
                        f"| Total {arm} efficiency | {sidebar_readouts[f'eta_{arm}_pct']['value']/100:.6g} |\n")
            if cell_absorption and cell_absorption.get("reason"):
                noise_rows += f"| Cell absorption status | {cell_absorption['reason']} |\n"
            elif cell_absorption and "probe" in cell_absorption:
                noise_rows += (
                    "| Efficiency scope | Uniform pump/input-seed state; absorption only; generated/depleted fields not fed back |\n"
                    "| Absorption vs net response | Emission/gain is separate; these total efficiencies do not replace the NF channel |\n")
            if experimental_noise is not None:
                noise_rows += (
                    f"| Assumed Jasperse common loss | {params.get('jasperse_loss_pct', 0.0):.6g}% |\n"
                    f"| Effective squeeze parameter | {at_point(experimental_noise['effective_squeeze']):.6g} |\n"
                    f"| Effective probe / conjugate transmission | "
                    f"{at_point(experimental_noise['probe_transmission']):.6g} / "
                    f"{at_point(experimental_noise['conjugate_transmission']):.6g} |\n"
                    f"| Effective gain fit relative residual | {at_point(experimental_noise['fit_relative_residual']):.3g} |\n"
                    f"| Effective gain match | {'Approximate; assumed loss cannot reproduce this pair exactly' if noise_point['inference_approximate'] else 'Matched'} |\n"
                    f"| Effective model probe / conjugate gain | {noise_point['G_s']:.6g} / {noise_point['G_c']:.6g} |\n"
                    "| Experimental noise scope | Effective uniform distributed mixing and two-arm vacuum loss; "
                    "not a microscopic RF atomic-noise calculation |\n")
        if phase_view:
            noise_rows += (
                f"| Squeezing view | {squeezing_view} |\n"
                f"| Quadrature status | {phase_scan['status']} |\n"
                "| Phase axis | Common homodyne LO rotation, 0–180°; B is A + 90° |\n"
                "| Collective mode | Q=(X_probe−X_conjugate)/√2; P=(P_probe−P_conjugate)/√2 |\n"
                "| Phase zero | Real phase-matched effective-model reference; not a measured optical phase |\n"
                f"| Seed phase noise | {phase_scan['seed_phase_assumption']} |\n"
                f"| Minimum / maximum quadrature noise | {phase_scan['minimum_db']:.6g} / {phase_scan['maximum_db']:.6g} dB |\n"
                f"| Orthogonal-curve crossing | {phase_scan['crossing_db']:.6g} dB at 45° and 135° |\n"
                "| Shot-noise normalization | Vacuum covariance I; 10 log10(V), 0 dB reference |\n"
                "| Gain during phase scan | Current operating-point values; independent of LO phase |\n"
                f"| Quadrature model scope | {phase_scan['scope']} |\n"
            )
        gain_budget_rows = ""
        if "gain_budget" in op:
            labels = {"small_signal": "Small-signal", "pre_cap": "Propagation output before cap",
                      "final_cap": "After pump cap", "displayed": "Displayed atomic-model"}
            for stage, label in labels.items():
                gains = op["gain_budget"][stage]
                gain_budget_rows += f"| {label} G_s / G_c | {gains['G_s']:.6g} / {gains['G_c']:.6g} |\n"
        operating_table = (
            f"| Quantity | Value |\n|---|---|\n"
            f"| ⁸⁵Rb density | {raw['N_atoms']:.3e} /m³ |\n"
            f"| Thermal velocity σ_v | {raw['sigma_v']:.1f} m/s |\n"
            f"| Velocity classes | {raw['n_velocity']} |\n"
            f"| Ω_pump / 2π | {raw['Op_A_2pi_GHz']:.3f} GHz |\n"
            f"| Ω_seed / 2π | {raw['Os_2pi_MHz']:.3f} MHz |\n"
            f"| (−) Raman line (probe axis) | {raw['raman_center_minus_GHz']:.3f} GHz |\n"
            f"| After-cell transmission × QE | {_detection_efficiency_from_params(params):.4f} |\n"
            f"{noise_rows}"
            f"{measured_rows}"
            f"| Operating probe detuning | {op['probe_GHz']:.4f} GHz |\n"
            f"{gain_budget_rows}"
            f"| Cell length | {raw.get('cell_length_m', L_CELL)*1e3:.1f} mm |\n"
            f"| Pump waist | {raw.get('w_pump_m', W_PUMP)*1e6:.0f} µm |\n"
            f"| Seed waist | {raw.get('w_probe_m', W_PROBE)*1e6:.0f} µm |\n"
            f"| Pump–seed angle | {raw.get('pump_probe_angle_deg', SEEDED_PHASE_ANGLE_DEG):.3f}° |\n"
            + depletion_warning
        )
        claim_reasons = "; ".join(claim_gate.get("reasons", ())) or "none recorded"
        closure = raw.get("gain_closure", {})
        closure_rows = (
            f"| Gain correction | {closure.get('status', 'disabled')} |\n"
        )
        if closure.get("applied"):
            closure_rows += (
                f"| Cross-coupling multiplier | {closure['coupling_multiplier']:.6f} |\n"
                "| Participation source | one fitted effective mixing coefficient; contributions unresolved |\n"
                "| Gain convention | P_probe,out / P_seed,in; no detection correction |\n"
                "| Calibration target | Sim 2025 representative 15.5 (range 15–16) |\n"
                "| Separate rounded-power ratios | probe 111/8 = 13.875; conjugate 109/8 = 13.625 |\n"
                "| Held-out comparison | Liu: 6.445 vs 8; McCormick: 1.926 vs 9 (4.7× low); conditional |\n"
            )
        diagnostics_table = (
            f"| Check | Value |\n|---|---|\n"
            f"| Model scope | Gain-only diagnostic (physical squeezing unavailable) |\n"
            f"| Validation level | {claim_gate.get('level', 'unavailable')} |\n"
            f"| Claim-gate reasons | {claim_reasons} |\n"
            f"| Quantitative gain claim | "
            f"{'supported' if claim_gate.get('quantitative_gain_supported') else 'UNSUPPORTED'} |\n"
            f"| Physical squeezing claim | "
            f"{'supported' if claim_gate.get('physical_squeezing_prediction') else 'UNAVAILABLE'} |\n"
            f"| Coupling norm 1/[2(2I+1)] (rho_ss supplies population) | "
            f"{raw.get('coupling_norm', float('nan')):.4f} |\n"
            f"| Canonical-mode status | {raw.get('canonical_mode_status', 'unavailable')} |\n"
            f"| Probe / conjugate effective area | "
            f"{raw.get('canonical_mode_area_probe_m2', np.nan):.3e} / "
            f"{raw.get('canonical_mode_area_conjugate_m2', np.nan):.3e} m² |\n"
            f"| Small-signal photon-flux gap | "
            f"{op.get('photon_flux_gap_smallsignal', np.nan):.6f} |\n"
            f"| Bare-map commutator defect max | "
            f"{op.get('commutator_defect_max_smallsignal', np.nan):.3e} |\n"
            f"| Atomic solver | {solver.get('solver_id', 'unavailable')} |\n"
            f"| Pump-only weak-response reference | "
            f"{pump_reference.get('solver_id', 'unavailable')} (separate/slow) |\n"
            f"| Pump-reference supported branch | "
            f"{pump_reference.get('supported_branches', ('unavailable',))} |\n"
            f"| 2-D Raman-Doppler reference | "
            f"{noncollinear_reference.get('solver_id', 'unavailable')} "
            f"(separate/slow; production 1-D) |\n"
            f"| 2-D reference velocity geometry | "
            f"{noncollinear_reference.get('velocity_geometry', 'unavailable')} |\n"
            f"| Floquet order | N_F={solver.get('floquet_order', 'unavailable')} |\n"
            f"| Floquet adjacent-order gate | "
            f"{floquet_convergence.get('status', 'NOT_EVALUATED')} vs "
            f"N_F={floquet_convergence.get('comparison_order', 'unavailable')} |\n"
            f"| Floquet full-scan points | "
            f"{floquet_convergence.get('full_scan_points', 0)} |\n"
            f"| Response estimator | "
            f"{raw.get('response_estimator', {}).get('method', 'unavailable')} |\n"
            f"| Solved / displayed probe detunings | "
            f"{raw.get('response_estimator', {}).get('solved_probe_points', '—')} / "
            f"{raw.get('response_estimator', {}).get('display_probe_points', '—')} |\n"
            f"| Maxwell average | "
            f"{raw.get('response_estimator', {}).get('velocity_average', 'unavailable')} |\n"
            f"| Atomic readout source coordinates | {selected['readout_coordinate_status']} |\n"
            f"| Displayed readout scope | {selected['readout_certification_scope']} |\n"
            f"| Noise-trace status | {raw.get('squeezing_status', 'unavailable')} |\n"
            f"| Transit reset γ_t / 2π | "
            f"{raw.get('transit_reset_rate_rad_s', np.nan)/(2*np.pi*1e3):.3f} kHz |\n"
            f"| Combined residual coupling | "
            f"{raw.get('combined_line_strength_residual', raw.get('line_strength_residual', float('nan'))):.4f}× |\n"
            f"| Effective coupling scale | {raw.get('effective_line_strength', float('nan')):.4f} |\n"
            f"| Pump-depletion cap on G_s (Manley-Rowe) | {cap:.3e} |\n"
            f"| Small-signal peak G_s (pre-saturation) | {small_signal:.3e} |\n"
            + closure_rows + phase_rows
        )
        if passive_readout is not None:
            valid = np.asarray(passive_readout["passive_completion_valid"], dtype=bool)
            diagnostics_table += (
                f"| Passive L1 selected status | {passive_op['passive_l1_status']} |\n"
                f"| L1 sampling scope | {passive_readout['passive_l1_sampling_status']} |\n"
                f"| Passive completion feasible samples | {np.count_nonzero(valid)} / {valid.size} |\n"
                f"| L1 source photon-flux G_s / G_c | {passive_op['passive_l1_G_s']:.6g} / {passive_op['passive_l1_G_c']:.6g} |\n"
                f"| L1 source D / T | {passive_op['passive_l1_D']:.6g} / {passive_op['passive_l1_T']:.6g} |\n"
                f"| Passive completion minimum eigenvalue | {passive_op['passive_completion_min_eig']:.3e} |\n"
                "| L1 scope | Static annihilation-only vacuum completion; no RF spectrum or bound for correlated phase-sensitive atomic noise |\n"
                "| Legacy gain diagnostic | Retained raw gain_referred_noise_dB / S_dB; not used for current L1 curve |\n"
            )
        return {
            "metrics": metrics,
            "figure": fig,
            "squeezing_view": squeezing_view,
            "quadrature_phase_scan": phase_scan,
            "gain_operating_point": gain_point,
            "passive_l1_readout": passive_readout,
            "passive_l1_operating_point": passive_op,
            "jasperse_readout": active_noise,
            "experimental_jasperse_readout": experimental_noise,
            "experimental_gain_calibration": experimental_gain,
            "noise_operating_point": noise_point,
            "noise_model": noise_model,
            "readout_coordinate_status": selected["readout_coordinate_status"],
            "readout_certification_scope": selected["readout_certification_scope"],
            "dressed_cell_absorption": cell_absorption,
            "sidebar_readouts": sidebar_readouts,
            "measured_gain_reference": measured_reference,
            "tables": [
                {"title": "Operating point", "markdown": operating_table},
                {"title": "Model diagnostics", "markdown": diagnostics_table},
            ],
        }

    def _biphoton_observables(self, raw, params, include_figures=True):
        topo = raw["topology"]
        predictive = raw.get("predictive", False)
        velocity_converged = raw.get("velocity_converged", True)
        requested_detail = params.get(
            "phase_detail", raw.get("phase_detail", "Balanced"))
        phase_detail = (
            "Fine" if str(requested_detail).lower() == "fine" else "Balanced")
        # Predictive: g²_SI(τ) comes from the computed waveform |ψ|² and the
        # physical accidentals (no target-g² forcing). Calibrated: legacy
        # added-accidental anchoring to the reference g² peak.
        stats = observables.biphoton_stats(
            raw["tau_axis_ns"], raw["psi_tau"], raw["pair_rate_cps"],
            signal_eff=params["signal_eff_pct"] / 100.0,
            idler_eff=params["idler_eff_pct"] / 100.0,
            dark_signal_cps=params["dark_signal_cps"],
            dark_idler_cps=params["dark_idler_cps"],
            coincidence_window_ns=params["coincidence_window_ns"],
            timing_jitter_ns=params["timing_jitter_ns"],
            filter_bandwidth_mhz=params["filter_bandwidth_mhz"],
            source_bandwidth_mhz=raw["source_bandwidth_mhz"],
            target_g2_peak=None if predictive else topo.target_g2_peak,
        )
        evidence = biphoton_output_evidence(
            raw, stats, params, velocity_fwhm_tolerance=PRED_V_FWHM_TOL)

        def display(key, value, suffix=""):
            if not evidence["observables"][key]["display_numeric"]:
                return "unconverged"
            return f"{value}{suffix}"

        width_is_reportable = velocity_converged or not predictive
        source_width_value = (
            f"{stats['source_fwhm_ns']:.2f} ns"
            if width_is_reportable else "unconverged")
        detected_width_value = (
            f"{stats['detected_fwhm_ns']:.2f} ns"
            if width_is_reportable else "unconverged")
        source_width_table = (
            f"{stats['source_fwhm_ns']:.3f} ns"
            if width_is_reportable else "unconverged")
        detected_width_table = (
            f"{stats['detected_fwhm_ns']:.3f} ns"
            if width_is_reportable else "unconverged")

        fig = None
        extra_figures = []
        if include_figures:
            import matplotlib.pyplot as plt

            fig, (axG2, axPM) = plt.subplots(2, 1, figsize=(8.5, 6.4))
            axG2.plot(stats["tau_axis_ns"], stats["g2_SI_tau"],
                      color="#1f77b4", lw=1.8)
            axG2.axhline(2.0, color="black", lw=0.7, ls=":")
            axG2.set_ylabel(r"$g^{(2)}_{SI}(\tau)$")
            rate_kind = evidence["observables"]["source_pair_rate_cps"]["origin"]
            title_tag = ("Reduced model" if predictive else "Reference model")
            title_tag += (" · assumed template rate" if rate_kind == "assumed_template"
                          else " · reference-anchored rate")
            if not width_is_reportable:
                title_tag += " · qualitative, velocity check failed"
            axG2.set_title(f"{topo.label}: {title_tag}")
            axG2.grid(alpha=0.3)

            angle_axis = raw.get("angle_axis_deg")
            phase_matching = raw.get("phase_matching")
            if angle_axis is None or phase_matching is None:
                angle_axis = np.linspace(max(params["idler_angle_deg"] - 4.0, 0.0),
                                         params["idler_angle_deg"] + 4.0, 181)
                signal = next(field for field in raw["fields"] if field.role == "signal")
                angle_dk = phase_mismatch_vector_grid(
                    raw["fields"], [signal.angle_deg], angle_axis,
                    reference_delta_k=topo.reference_delta_k, precise_norm=True,
                )[0]
                phase_matching = phase_matching_weight(angle_dk, raw["cell_length_m"])

            axPM.plot(angle_axis, phase_matching,
                      color="#2ca02c", lw=1.8)
            axPM.axvline(params["idler_angle_deg"], color="crimson", lw=1.1, ls="--")
            axPM.set_xlabel("Idler collection angle [deg]")
            axPM.set_ylabel(r"$\mathrm{sinc}^2(|\Delta\mathbf{k}| L / 2)$")
            axPM.grid(alpha=0.3)
            fig.tight_layout()

            amp = np.abs(raw["source_v"])
            amp = amp / np.nanmax(amp) if np.nanmax(amp) > 0 else amp
            phase = np.unwrap(np.angle(raw["source_v"]))
            figV, (axA, axP) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
            axA.plot(raw["v_grid"], amp, color="#ff7f0e", lw=1.5)
            axA.set_ylabel("Velocity source amplitude")
            axA.grid(alpha=0.3)
            axP.plot(raw["v_grid"], phase, color="#9467bd", lw=1.3)
            axP.set_xlabel("Atomic velocity [m/s]")
            axP.set_ylabel("Velocity source phase [rad]")
            axP.grid(alpha=0.3)
            figV.tight_layout()

            extra_figures = [("Velocity-class coherent source", figV)]
            if phase_detail == "Fine":
                signal_axis, idler_axis, pm2d = fwm.biphoton_phase_matching_map(
                    raw["fields"], raw["cell_length_m"],
                    signal_angle_deg=params["signal_angle_deg"],
                    idler_angle_deg=params["idler_angle_deg"],
                    reference_delta_k=topo.reference_delta_k)
                figPM2, ax2 = plt.subplots(1, 1, figsize=(7.2, 5.2))
                im = ax2.imshow(
                    pm2d.T, origin="lower", aspect="auto",
                    extent=[
                        signal_axis[0], signal_axis[-1],
                        idler_axis[0], idler_axis[-1],
                    ],
                    cmap="viridis", vmin=0.0, vmax=1.0)
                ax2.scatter([params["signal_angle_deg"]], [params["idler_angle_deg"]],
                            color="crimson", s=28, zorder=5)
                ax2.set_xlabel("Signal angle [deg]")
                ax2.set_ylabel("Idler angle [deg]")
                ax2.set_title("2D phase-matching acceptance")
                figPM2.colorbar(
                    im, ax=ax2,
                    label=r"$\mathrm{sinc}^2(|\Delta\mathbf{k}| L / 2)$")
                figPM2.tight_layout()
                extra_figures.append(("2D phase matching", figPM2))

        if predictive:
            g2_help = ("Peak signal-idler cross-correlation from the computed "
                       "delay density and accidental rate density; independent "
                       "of coincidence gate width (not forced).")
            rate_help = ("Reference-anchored pair rate after pump, coupling, density, "
                         "and phase-matching scaling.")
            source_fwhm_help = (
                "Intrinsic |ψ(τ)|² FWHM on an auto-refined velocity grid. The "
                "absolute width remains approximate."
                if velocity_converged else
                "Width withheld because velocity-grid refinement hit its point cap.")
            detected_fwhm_help = (
                "Intrinsic waveform convolved with the signal–idler timing response."
                if velocity_converged else
                "Width withheld because the intrinsic waveform did not converge.")
            status_value = (f"Reduced model · {raw.get('regime', '—')}"
                            if velocity_converged
                            else f"Reduced model · {raw.get('regime', '—')} · "
                                 "width unconverged")
            status = dict(
                label="Model status", value=status_value,
                help="Calculates the Doppler waveform and vector phase matching; "
                     "the pair-rate scale remains reference-anchored.")
        else:
            g2_help = ("Peak normalized signal–idler correlation from the stored "
                       "reference calibration.")
            rate_help = "Pair rate from the stored reference calibration."
            source_fwhm_help = (
                "Intrinsic FWHM of the modeled |ψ(τ)|² source waveform before "
                "detector timing response.")
            detected_fwhm_help = (
                "FWHM after convolution with the net Gaussian signal-idler timing-"
                "difference response.")
            status = dict(
                label="Model status", value="Reference model",
                help="Reproduces stored calibration values; it is not an independent "
                     "prediction.")
        if evidence["observables"]["source_pair_rate_cps"]["origin"] == "assumed_template":
            rate_help = ("Assumed template pair-rate coefficient; no literature rate "
                         "anchor or held-out validation. Scalar filter throughput applied.")
            status["value"] = ("Assumed diamond template" if width_is_reportable
                               else "Assumed diamond template · unconverged")
            status["help"] = "Generic template parameters are assumed; no validated diamond preset."
        elif not evidence["observables"]["pair_rate_cps"]["display_numeric"]:
            rate_help += " Filter-passed rate withheld because the waveform check failed."
        source_fwhm_help += " Velocity refinement tests FWHM only; it is not experimental validation."
        detected_fwhm_help += " Timing-response reuse is not independent width validation."
        metrics = [
            status,
            dict(label="Peak signal–idler correlation",
                 value=display("g2_peak", f"{stats['g2_peak']:.2f}"), help=g2_help,
                 tier="hero"),
            dict(label="CAR", value=display("CAR", f"{stats['CAR']:.1f}"),
                 help="Captured true coincidences divided by accidentals in the "
                      "symmetric gate [-W/2, W/2] about zero delay."),
            dict(label="Pair rate", value=display("pair_rate_cps", f"{stats['pair_rate_cps']:.1f}", " cps"),
                 help=rate_help, tier="hero"),
            dict(label="Intrinsic biphoton width",
                 value=source_width_value,
                 help=source_fwhm_help),
            dict(label="Detected biphoton width",
                 value=detected_width_value,
                 help=detected_fwhm_help),
            dict(label="Source bandwidth", value=display(
                "source_bandwidth_mhz", f"{raw['source_bandwidth_mhz']:.0f}", " MHz"),
                 help=("Spectral FWHM from the reduced waveform; spectral/time-grid convergence "
                       "has not been certified." if predictive else "Reference or assumed source bandwidth.")),
            dict(label="Phase-matching efficiency",
                 value=f"{raw['phase_match_weight']:.3f}",
                 help="Vector sinc^2 phase-matching collection weight."),
        ]
        for metric in metrics:
            if metric["value"] == "unconverged":
                metric["kind"] = "status"

        field_rows = "".join(
            f"| {f.role} | {raw['topology'].levels[f.lower].name} -> "
            f"{raw['topology'].levels[f.upper].name} | {f.wavelength_nm:.2f} nm | "
            f"{f.angle_deg:.2f}° | {_side_label(f.side_sign) if f.side_sign else '0'} |\n"
            for f in raw["fields"]
        )
        pm_warning = (
            "| Warning | Vector phase match < 1e-3; collection geometry is not "
            "physically phase matched. |\n"
            if raw["phase_match_weight"] < 1e-3 else ""
        )
        topology_table = (
            f"| Quantity | Value |\n|---|---|\n"
            f"| Topology | {topo.label} |\n"
            f"| Family | {topo.family} |\n"
            f"| Density | {raw['density']:.3e} /m^3 |\n"
            f"| Cell length | {raw['cell_length_m']*1e3:.2f} mm |\n"
            f"| Delta k z relative | {raw['delta_k_z_relative']:.3e} 1/m |\n"
            f"| Delta k z absolute | {raw['delta_k_z_absolute']:.3e} 1/m |\n"
            f"| Delta k x transverse | {raw['delta_k_x']:.3e} 1/m |\n"
            f"| Delta k vector | {raw['delta_k_vector']:.3e} 1/m |\n"
            f"| Vector phase match | {raw['phase_match_weight']:.3f} |\n"
            f"| Longitudinal phase match | {raw['phase_match_weight_longitudinal']:.3f} |\n"
            f"| Vacuum phase match | {raw['phase_match_weight_absolute']:.3f} |\n"
            + pm_warning +
            f"| Phase-matching view | "
            f"{'1D + 2D' if phase_detail == 'Fine' else '1D'} |\n"
            f"| Energy mismatch | {raw['energy_mismatch_hz']/1e6:.3f} MHz |\n"
            f"| Residual two-photon Doppler k | {raw['residual_two_photon_k']:.3e} 1/m |\n"
            f"| Velocity step{' (auto-refined)' if predictive else ''} | "
            f"{raw.get('velocity_step_used', float('nan')):.3f} m/s"
            f"{'' if velocity_converged else ' · UNCONVERGED'} |\n\n"
            f"| Field | Transition | Wavelength | Angle | Side |\n|---|---|---:|---:|---:|\n"
            + field_rows
        )
        detection_table = (
            "Source estimate with detector/background model:\n\n"
            f"| Quantity | Value |\n|---|---|\n"
            f"| Source estimate before filter | {raw['pair_rate_cps']:.3f} cps |\n"
            f"| Signal singles | {display('singles_signal_cps', f'{stats["singles_signal_cps"]:.2f}', ' cps')} |\n"
            f"| Idler singles | {display('singles_idler_cps', f'{stats["singles_idler_cps"]:.2f}', ' cps')} |\n"
            f"| All-time true coincidence | {display('all_time_coincidence_cps', f'{stats["all_time_coincidence_cps"]:.3f}', ' cps')} |\n"
            f"| Gate-captured true coincidence | {display('coincidence_cps', f'{stats["coincidence_cps"]:.3f}', ' cps')} |\n"
            f"| Coincidence gate | symmetric [-W/2, W/2], W = {params['coincidence_window_ns']:.3f} ns |\n"
            f"| Gate capture fraction | {display('gate_capture_fraction', f'{stats["gate_capture_fraction"]:.6f}')} |\n"
            f"| Accidental coincidence | {display('accidental_cps', f'{stats["accidental_cps"]:.3e}', ' cps')} |\n"
            f"| Raw accidental before reference calibration | {display('raw_accidental_cps', f'{stats["raw_accidental_cps"]:.3e}', ' cps')} |\n"
            f"| Added unmodelled accidental/background | {display('added_accidental_cps', f'{stats["added_accidental_cps"]:.3e}', ' cps')} |\n"
            f"| Raw g2 peak before reference calibration | {display('raw_g2_peak', f'{stats["raw_g2_peak"]:.2f}')} |\n"
            f"| Heralding signal | {display('heralding_signal', f'{stats["heralding_signal"]:.3e}')} |\n"
            f"| Heralding idler | {display('heralding_idler', f'{stats["heralding_idler"]:.3e}')} |\n"
            f"| Thermal-arm comparison R (assumed g²_SS = g²_II = 2) | {display('cauchy_schwarz_R', f'{stats["cauchy_schwarz_R"]:.2f}')} |\n"
            f"| Physical nonclassicality | unavailable: auto-correlations/covariance not supplied |\n"
            f"| Intrinsic biphoton width | {source_width_table} |\n"
            f"| Detected biphoton width | {detected_width_table} |\n"
            f"| Net timing-difference response FWHM | {params['timing_jitter_ns']:.3f} ns |\n"
            f"| Filter transmission estimate | {display('filter_transmission', f'{stats["filter_transmission"]:.3f}')} |\n"
            f"| Filter model | Scalar throughput; no temporal reshaping |\n"
            f"| Delay density normalization | Finite source-delay axis; full Gaussian timing tails |\n\n"
            f"{raw['notes']}"
        )
        validation_table = self._reference_validation_table(raw, params, stats)
        evidence_labels = {
            "source_pair_rate_cps": "Source rate before filter",
            "source_fwhm_ns": "Intrinsic width",
            "detected_fwhm_ns": "Detected width",
            "source_bandwidth_mhz": "Source bandwidth",
            "pair_rate_cps": "Filter-passed rate",
            "g2_peak": "Peak g²",
            "CAR": "Gate CAR",
            "coincidence_cps": "Gate-captured coincidence",
            "phase_match_weight": "Phase matching",
            "cauchy_schwarz_R": "Assumed thermal-arm comparison",
        }
        evidence_table = (
            "Conditional model estimates; no same-condition measurements or held-out validation. "
            "Velocity refinement checks intrinsic FWHM only, not the whole waveform, "
            "spectral bandwidth or time-window convergence. Seeded-FWM validation does "
            "not validate this spontaneous-source model.\n\n"
            "| Output | Basis | Status |\n|---|---|---|\n"
            + "".join(f"| {label} | {evidence['observables'][key]['origin'].replace('_', ' ')} | "
                      f"{evidence['observables'][key]['status'].replace('_', ' ')} |\n"
                      for key, label in evidence_labels.items()))
        return {
            "metrics": metrics,
            "biphoton_readout": stats,
            "biphoton_evidence": evidence,
            "figure": fig,
            "figures": extra_figures,
            "figure_controls": ["phase_detail"],
            "tables": [
                {"title": "Source geometry", "markdown": topology_table},
                {"title": "Detection estimates", "markdown": detection_table},
                {"title": "Literature comparison", "markdown": validation_table},
                {"title": "Observable evidence", "markdown": evidence_table},
            ],
        }

    def _reference_validation_table(self, raw, params, stats):
        topo = raw["topology"]
        predictive = raw.get("predictive", False)
        velocity_converged = raw.get("velocity_converged", True)

        def verdict(ok, kind="physical"):
            # Calibration anchors use the reference value, so pass/fail would be
            # circular. Waveform quantities are estimates and are marked predicted.
            if kind == "calibrated":
                return "by construction"
            if kind == "predicted":
                return "predicted (approx)"
            if kind == "setting":
                return "setting only"
            if kind == "geometry":
                return "conditional geometry" if ok else "geometry CHECK"
            if kind == "assembly":
                return "assembly PASS" if ok else "assembly CHECK"
            return "PASS" if ok else "CHECK"

        def row(name, calc, ref, ok, note="", kind="physical"):
            return f"| {name} | {calc} | {ref} | {verdict(ok, kind)} | {note} |\n"

        rows = []
        if topo.name == TOPOLOGY_RB87_TELECOM:
            pump_mw = max(params["pump_biphoton_uw"] * 1e-3, 1e-30)
            rate_per_mw = raw["pair_rate_cps"] / pump_mw
            rows.append(row(
                "Source pair rate / pump", f"{rate_per_mw:.0f} cps/mW",
                "38000 cps/mW", abs(rate_per_mw / 38000.0 - 1.0) < 0.15,
                "anchored source rate before filter and detection losses", kind="calibrated"))
            rows.append(row(
                "Filter-passed pair rate / pump",
                ("unconverged" if predictive and not velocity_converged
                 and params.get("filter_bandwidth_mhz", 0.0) > 0 else
                 f"{stats['pair_rate_cps'] / pump_mw:.0f} cps/mW"),
                "no independent filter-matched reference", False,
                "source anchor times the scalar filter-transmission estimate",
                kind="predicted"))
            rows.append(row(
                "g2 peak", ("unconverged" if predictive and not velocity_converged
                            else f"{stats['g2_peak']:.2f} (raw {stats['raw_g2_peak']:.1f})"),
                "44(3)", abs(stats["g2_peak"] - 44.0) <= 3.0,
                ("predicted from waveform |ψ|² + physical accidentals"
                 if predictive else
                 "anchored: forced to the target via added-accidental calibration"),
                kind="predicted" if predictive else "calibrated"))
            if predictive and not velocity_converged:
                rows.append(row(
                    "Detected BTW FWHM", "unconverged",
                    "0.56(4) ns", False,
                    "velocity-grid auto-refine hit the point cap — absolute width "
                    "not converged, qualitative only (shape/ordering still indicative)",
                    kind="predicted"))
            else:
                rows.append(row(
                    "Detected BTW FWHM", f"{stats['detected_fwhm_ns']:.3f} ns",
                    "0.56(4) ns",
                    abs(stats["detected_fwhm_ns"] - 0.56) <= 0.04,
                    ("velocity-converged source waveform + net timing response; "
                     "absolute ns carries per-source calibration uncertainty"
                     if predictive else "model waveform + net timing response"),
                    kind="predicted" if predictive else "calibrated"))
            rows.append(row(
                "OD estimate", f"{raw['od_estimate']:.1f}",
                "112(3)", abs(raw["od_estimate"] - 112.0) <= 3.0,
                "anchored: density/cell scaling of the reference OD", kind="calibrated"))
            rows.append(row(
                "Bandwidth setting", f"{params['filter_bandwidth_mhz']:.0f} MHz",
                "about 300 MHz", abs(params["filter_bandwidth_mhz"] - 300.0) <= 40.0,
                "input setting, not a calculated source-bandwidth validation",
                kind="setting"))
        elif topo.name == TOPOLOGY_CS_BTW:
            other_channel = CS_CHANNEL_795 if params.get("cs_channel") == CS_CHANNEL_917 else CS_CHANNEL_917
            other_params = dict(params)
            other_params["cs_channel"] = other_channel
            other_params = self._biphoton_runtime_params(other_params)
            other_raw = _cached_biphoton_source(
                _biphoton_source_items(other_params))
            other_stats = observables.biphoton_stats(
                other_raw["tau_axis_ns"], other_raw["psi_tau"], other_raw["pair_rate_cps"],
                signal_eff=params["signal_eff_pct"] / 100.0,
                idler_eff=params["idler_eff_pct"] / 100.0,
                dark_signal_cps=params["dark_signal_cps"],
                dark_idler_cps=params["dark_idler_cps"],
                coincidence_window_ns=params["coincidence_window_ns"],
                timing_jitter_ns=params["timing_jitter_ns"],
                filter_bandwidth_mhz=params["filter_bandwidth_mhz"],
                source_bandwidth_mhz=other_raw["source_bandwidth_mhz"],
                target_g2_peak=other_raw["topology"].target_g2_peak,
            )
            ratio = max(
                stats["detected_fwhm_ns"], other_stats["detected_fwhm_ns"]
            ) / max(
                min(stats["detected_fwhm_ns"], other_stats["detected_fwhm_ns"]),
                1e-30)
            rows.append(row(
                "Detected BTW width ratio", (
                    "unconverged" if predictive and (not velocity_converged
                    or not other_raw.get("velocity_converged", True)) else f"{ratio:.2f}"),
                "about 3", abs(ratio - 3.0) <= 0.5,
                ("predicted ordering (917 narrower than 795); absolute ratio approximate"
                 if predictive else
                 "medium model only; full Cs BTW theory is not yet included"),
                kind="predicted" if predictive else "calibrated"))
            rows.append(row(
                "OD estimate", f"{raw['od_estimate']:.1f}",
                "about 10", abs(raw["od_estimate"] - 10.0) <= 2.0,
                "anchored: density/cell scaling of the reference OD", kind="calibrated"))
        else:
            rows.append(row(
                "Reference validation", "generic diamond template",
                "no paper anchor", False,
                "configure wavelengths manually; no validated default"))

        rows.append(row(
            "Phase matching", f"{raw['phase_match_weight']:.3f}",
            "> 0.90", raw["phase_match_weight"] > 0.90,
            "reference-geometry offset + vector sinc^2; not independent validation",
            kind="geometry"))
        rows.append(row(
            "Energy conservation", f"{raw['energy_mismatch_hz']/1e6:.3f} MHz",
            "near 0 MHz", abs(raw["energy_mismatch_hz"]) < 1e6,
            "wavelength bookkeeping, not experimental validation", kind="assembly"))

        if topo.name == TOPOLOGY_DIAMOND:
            intro = (
                "**Assumed diamond template.** Rate and lifetime coefficients are "
                "assumed inputs with no paper anchor or held-out validation.\n\n")
        elif predictive:
            intro = (
                "**Reduced model.** Waveform-derived widths are approximate. Pair-"
                "rate rows are reference-anchored. Velocity refinement checks "
                "intrinsic FWHM only.\n\n")
        else:
            intro = (
                "**Reference model.** Lifetime and optional added-background anchors "
                "are comparison inputs, not independent width or correlation validation.\n\n")
        intro += (
            "Phase matching uses a reference-geometry offset; energy conservation "
            "checks wavelength bookkeeping. Neither is held-out experimental validation.\n\n")
        return (
            intro
            + "| Check | Calculated | Reference | Verdict | Note |\n|---|---:|---:|---|---|\n"
            + "".join(rows)
        )

    def _finite_seed_angular_view(self):
        def compute(params):
            physical = self._seeded_compute_inputs(params)
            physical.update(coarse_points=81, fine_points=0, response_method=RESPONSE_POLE,
                            velocity_measure="exact")
            center = branch_center_GHz(physical["D_GHz"], -1)
            physical.update(scan_min=center - .05, scan_max=center + .05)
            baseline = fwm.compute_spectrum(**physical)
            reference = fwm.compute_spectrum(**physical, angular_transverse_order=18)
            refined = fwm.compute_spectrum(**physical, angular_transverse_order=24)
            tail = fwm.compute_spectrum(**physical, angular_transverse_order=24,
                                    angular_transverse_cutoff=5.5)
            return {
                "delta_mhz": (reference["probe_axis_GHz"] - center) * 1e3,
                "collinear_atomic_model": {key: baseline[key] for key in ("G_s", "G_c")},
                "angular": {key: refined[key] for key in ("G_s", "G_c")},
                "quadrature_delta": {
                    "transverse_18_to_24": {key: float(np.max(np.abs(reference[key] - refined[key])
                        / np.maximum(1.0, np.abs(refined[key])))) for key in ("G_s", "G_c")},
                    "cutoff_5_to_5p5_sigma": {key: float(np.max(np.abs(tail[key] - refined[key])
                        / np.maximum(1.0, np.abs(refined[key])))) for key in ("G_s", "G_c")},
                },
                "floquet_convergence": refined["floquet_convergence"],
                "angular_provenance": refined["finite_seed_angular_reference"],
            }

        def render(data):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            for arm, label, color in (("G_s", "Probe", "#1f77b4"),
                                      ("G_c", "Conjugate", "#d64771")):
                ax.plot(data["delta_mhz"], data["angular"][arm], color=color,
                        label=f"{label}, angular reference")
                ax.plot(data["delta_mhz"], data["collinear_atomic_model"][arm], color=color,
                        linestyle="--", label=f"{label}, current atomic model")
            ax.set(xlabel="Two-photon detuning [MHz]", ylabel="Atomic-medium power gain")
            errors = data["quadrature_delta"]
            max_order = max(errors["transverse_18_to_24"].values())
            max_tail = max(errors["cutoff_5_to_5p5_sigma"].values())
            status = data["floquet_convergence"]["status"]
            label = f"Floquet {status} | transverse refinement {max_order:.2g} | cutoff {max_tail:.2g}"
            ax.set_title(label)
            ax.legend()
            ax.grid(alpha=.3)
            fig.tight_layout()
            return fig

        return ExtraView(
            key="Finite-seed angular Doppler reference",
            description="Minus Raman branch, ±50 MHz. Compare the finite-seed angular Maxwell "
                        "average with the current atomic model. This calculation takes several seconds.",
            compute=compute, render=render,
        )

    def extra_views(self):
        def _compute_full(params):
            fidelity = normalize_fidelity(params.get("resolution", FIDELITY_FAST))
            fidelity_settings = FWM_FIDELITY[fidelity]
            velocity_measure = params.get("velocity_measure", "exact")
            return fwm.full_spectrum(
                params["opd"], params["temp_c"] + 273.15,
                params["pump_mw"], params["probe_uw"], params["line_strength"],
                params["loss_pct"], L=params["cell_mm"] * 1e-3,
                mode_overlap_penalty=params.get("mode_overlap_penalty", 1.0),
                polarization_penalty=params.get("polarization_penalty", 1.0),
                zeeman_participation_penalty=params.get(
                    "zeeman_participation_penalty", 1.0),
                w_pump=params.get("pump_waist_um", W_PUMP * 1e6) * 1e-6,
                w_probe=params.get("probe_waist_um", W_PROBE * 1e6) * 1e-6,
                qe=(params.get("qe_pct", QE_DETECTOR * 100.0) / 100.0
                    if params.get("legacy_gain_calibration_fixture", False) else 1.0),
                detection_efficiency=(_detection_efficiency_from_params(params)
                                      if params.get("legacy_gain_calibration_fixture", False) else 1.0),
                floquet_order=params.get("floquet_order", SEEDED_FLOQUET_ORDER),
                phase_detail=fidelity_settings["phase_detail"],
                pump_probe_angle_deg=params.get(
                    "seeded_angle_deg", SEEDED_PHASE_ANGLE_DEG),
                model_fidelity=fidelity,
                gain_closure_enabled=params.get("gain_closure_enabled", False),
                legacy_gain_calibration_fixture=params.get("legacy_gain_calibration_fixture", False),
                legacy_mean_field_fixture=params.get(
                    "legacy_mean_field_fixture", False),
                velocity_step=fidelity_settings["velocity_step"],
                velocity_cutoff=fidelity_settings["velocity_cutoff"],
                velocity_measure=velocity_measure,
                # The 0.4 MHz resonance windows need a solve at every display point.
                response_method=_tier_response_method(fidelity, velocity_measure, full_scan=True),
                scan_tolerance_dB=fidelity_settings.get(
                    "scan_tolerance_dB", SCAN_TOLERANCE_DB),
                scan_tolerance_log10_gain=fidelity_settings.get(
                    "scan_tolerance_log10_gain", SCAN_TOLERANCE_LOG10_GAIN),
                scan_start_stride=fidelity_settings.get(
                    "scan_start_stride", SCAN_START_STRIDE),
                transit_rate=(2.0 * np.pi
                              * params.get("transit_rate_khz", 100.0) * 1e3),
                eom_residual_carrier_power=(
                    params.get("eom_residual_carrier_uw", 0.0) * 1e-6),
                eom_other_sidebands_power=(
                    params.get("eom_other_sidebands_uw", 0.0) * 1e-6),
                eom_seed_spectrum_provenance=params.get(
                    "eom_seed_spectrum_provenance", "direct FWM input"),
                eom_seed_spectrum_status=params.get(
                    "eom_seed_spectrum_status", "not supplied"),
                eom_seed_spectrum_application=params.get(
                    "eom_seed_spectrum_application", "unapplied"))

        def _render_full(full, params=None):
            import matplotlib.pyplot as plt
            params = params or {}
            gain_model = params.get("gain_calibration_model", MODEL_LAB)
            literature_gain = gain_model in (MODEL_LITERATURE, MODEL_CHI, MODEL_STARK, MODEL_ZEEMAN)
            chi_gain = gain_model in (MODEL_CHI, MODEL_STARK, MODEL_ZEEMAN)
            stark_gain = gain_model == MODEL_STARK
            zeeman_gain = gain_model == MODEL_ZEEMAN
            figF, (aG, aS) = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
            for ax in (aG, aS):
                ax.grid(alpha=0.3)
            styles = {
                "minus": dict(color="#1f77b4", label="minus Raman branch"),
                "plus": dict(color="#ff7f0e", label="plus Raman branch"),
            }
            has_passive_readout = any(
                "T_canonical_small_signal" in full[key] for key in styles)
            measured_readouts = {}
            experimental_gains, experimental_noises = {}, {}
            for key, style in styles.items():
                spec = full[key]
                if params.get("measured_gain_calibration", False):
                    measured_readouts[key] = fwm._measured_gain_reference_readout(
                        spec, enabled=True,
                        seed_input_assumption=params.get("seed_input_assumption", ""))
                calibration = None
                if params.get("experimental_gain_calibration", False):
                    calibration = calibrate_experimental_gain(
                        spec, enabled=True, model=gain_model)
                    experimental_gains[key] = calibration
                displayed = calibration["G_s"] if calibration is not None else spec["G_s"]
                aG.plot(spec["probe_axis_GHz"], displayed, lw=1.4, **style)
                if params.get("noise_model", NOISE_PASSIVE_L1) == NOISE_EXPERIMENTAL_JASPERSE:
                    experimental_noises[key] = fwm.experimental_jasperse_readout(spec, params, gains=calibration)
                    noise_db = experimental_noises[key]["NF_dB"]
                elif "T_canonical_small_signal" in spec:
                    noise_db = _seeded_passive_l1_readout(spec, params)[
                        "passive_l1_with_excess_noise_dB"]
                else:
                    noise_db = spec["gain_referred_noise_dB"]
                aS.plot(spec["probe_axis_GHz"], noise_db, lw=1.4, **style)
            _plot_reference(aG, 1.0, "G = 1 (unity gain)")
            aG.set_ylabel("Zeeman pump + chi propagation gain" if experimental_gains and zeeman_gain else
                           "Stark + chi propagation gain" if experimental_gains and stark_gain else
                           "Chi-corrected propagation gain" if experimental_gains and chi_gain else
                           "Literature-corrected power gain" if experimental_gains and literature_gain else
                           "Calibrated pump-off-reference gain" if experimental_gains else
                           "Atomic probe gain G_s" if measured_readouts else "Seed gain G_s")
            stored_detail = full["minus"].get("model_fidelity", "unknown")
            title = f"Solver detail: {FIDELITY_LABELS.get(stored_detail, stored_detail)}"
            title += f",  {_seeded_noise_caption(params)}"
            if experimental_gains and literature_gain:
                literature_metadata = experimental_gains["minus"]
                title += (",  experimental Zeeman pump + chi propagation\n" if zeeman_gain else
                          ",  experimental Stark + chi propagation\n" if stark_gain else
                          ",  experimental chi propagation\n" if chi_gain else
                          ",  experimental literature gain\n") + literature_metadata.get(
                    "validation_label", "Experimental · grouped prediction not established")
            aG.set_title(title)
            if max(np.nanmax(full[key]["G_s"]) for key in styles) > 50:
                aG.set_yscale("log")
            _label_noise_axis(aS, bool(experimental_noises) or has_passive_readout)
            if experimental_gains and literature_gain and experimental_noises:
                aS.set_title("Gain validation does not validate noise; effective Jasperse model", fontsize=8)
            aS.set_xlabel("Probe detuning from F=2 -> F'=3 [GHz]")
            for a in (aG, aS):
                a.axvline(full["D_GHz"], color="gray", ls=":", lw=0.8)
            aG.legend(fontsize=9)
            aS.legend(fontsize=9)
            figF.tight_layout()
            # Reference ratios have another denominator and are never drawn as
            # the atomic gain curve in this view. Keep their separate provenance.
            figF._gabes_measured_gain_reference = measured_readouts
            figF._gabes_experimental_gain_calibration = experimental_gains
            figF._gabes_gain_calibration_model = gain_model
            figF._gabes_experimental_jasperse_readout = experimental_noises
            return figF

        return [ExtraView(
            key="Full probe scan (−8 to 12 GHz, slow)",
            description="Runs both Raman branches using the selected geometry and "
                        "solver detail.",
            compute=_compute_full, render=_render_full,
            render_with_params=True,
        ), self._finite_seed_angular_view()]

    def internal_measured_gain_comparison(self):
        """Historical bounded interpolation diagnostic, excluded from the UI.

        The current empirical gain correction has its own fit/evidence record.
        This comparison remains available for internal provenance checks only.
        """
        def _compute_measured_comparison(params):
            observations = fwm_measured_gain.observations()
            profile = fwm_measured_gain.profile()
            lower, upper = profile["eom_support_MHz"]
            frequency = np.linspace(lower, upper, 441)
            raw_context = {
                "D_GHz": params.get("opd"),
                "temperature_K": params.get("temp_c", np.nan)+273.15,
                "input_pump_power_W": params.get("pump_mw", np.nan)*1e-3,
                "input_seed_power_W": params.get("probe_uw", np.nan)*1e-6,
                "w_pump_m": params.get("pump_waist_um", np.nan)*1e-6,
                "w_probe_m": params.get("probe_waist_um", np.nan)*1e-6,
                "cell_length_m": params.get("cell_mm", np.nan)*1e-3,
                "pump_probe_angle_deg": params.get("seeded_angle_deg"),
                "branch": DEFAULT_BRANCH,
                "legacy_mean_field_fixture": params.get("legacy_mean_field_fixture", False),
                "gain_closure": {"historical_fixture_requested": params.get(
                    "legacy_gain_calibration_fixture", False)},
            }
            gate = fwm_measured_gain.context_eligibility(
                _measured_gain_context(raw_context, params.get("seed_input_assumption", "")),
                enabled=params.get("measured_gain_calibration", False))
            return {"observations": observations,
                    "curve": fwm_measured_gain.reference_curve(frequency),
                    "profile": profile, "current_context": gate}

        def _render_measured_comparison(data, params=None):
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(8.5, 4.8))
            measurements, curve = data["observations"], data["curve"]
            for arm, color, marker in (("probe", "#1f77b4", "o"),
                                       ("conjugate", "#ff7f0e", "s")):
                axis.scatter(measurements["eom_MHz"], measurements[f"{arm}_ratio"],
                             color=color, marker=marker, s=28,
                             label=f"{arm.capitalize()} measurements")
                axis.plot(curve["eom_MHz"], curve[f"measured_gain_{arm}"],
                          color=color, lw=1.6, label=f"{arm.capitalize()} interpolation")
            axis.set_xlim(*data["profile"]["eom_support_MHz"])
            axis.set_yscale("log")
            axis.ticklabel_format(axis="x", useOffset=False)
            axis.set_xlabel("EOM frequency [MHz]")
            axis.set_ylabel("Pump-on output / pump-off seed detector reference")
            axis.set_title("2026-10-13 measured gain reference\n"
                           "Node agreement by construction; no independent prediction validation")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=9, ncol=2)
            figure.tight_layout()
            figure._gabes_measured_gain_reference = data
            return figure

        return ExtraView(
            key="2026-10-13 measured gain comparison",
            description="23 measured pump-off detector-reference ratios and their log-PCHIP interpolation, "
                        "3022–3044 MHz only. Reference apparatus: 113 °C, nominal blue OPD ≈1 GHz, "
                        "pump 294.804 mW. Node agreement is by construction; no atomic gain or NF calibration.",
            compute=_compute_measured_comparison, render=_render_measured_comparison,
            param_keys=("measured_gain_calibration", "seed_input_assumption"),
            render_with_params=True,
        )
