"""Compare the shipped FWM preset with a focused three-carrier calculation.

Run `python -m gabes_lab.pipelines.squeezing.compare_longitudinal_fwm kim-probe`.
All inputs come from the named preset. Results go to runs/squeezing/longitudinal.
"""

import argparse
from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from gabes import constants, doppler, observables
from gabes.schemes import fwm
from gabes.schemes.fwm.longitudinal import LongitudinalSeededInputs, LongitudinalSeededMedium
from gabes_lab.blocks import io, runner

_PRESET_PREFIXES = {
    "gold": "Sim Gold Reference",
    "kim-probe": "Kim & Marino 2018 \u2014 singly resonant probe",
    "kim-conjugate": "Kim & Marino 2018 \u2014 singly resonant conjugate",
}


def _fixed_pump_references(raw, inputs):
    """Independent resolvent and matrix propagation, holding pump fixed."""
    D, delta, T, L, wp, ws = (
        inputs.D_GHz, inputs.delta_mhz, inputs.T, inputs.L, inputs.w_pump, inputs.w_probe)
    probe = fwm.branch_center_GHz(D, inputs.branch) + delta*1e-3
    kp, ks, kc = fwm.seeded_option_a_wavenumbers(D, probe)
    dk = fwm.seeded_phase_mismatch_z(D, probe, inputs.angle_deg)
    pump, seed = fwm.rabi_freq(inputs.P_pump, wp), fwm.rabi_freq(inputs.P_probe, ws)
    delta_rad, detuning = 2*np.pi*1e6*delta, 2*np.pi*1e9*D
    atom = fwm.collisional_atom(T, transit_rate=inputs.transit_rate)
    profile = fwm._gaussian_overlap_profile(64, L, wp, ws, inputs.angle_deg)

    def propagated(response):
        ss, cs, sc, cc = map(np.atleast_1d, response)
        drift = observables._gain_matrix_from_chi(
            ss, sc, cs, cc, ks, kc, raw["N_atoms"], constants.DIPOLE_D1,
            raw["effective_line_strength"], delta_k_z=dk)[0]
        transfer = np.eye(2, dtype=complex)
        for scale in profile:
            local = drift.copy()
            local[0, 1] *= scale
            local[1, 0] *= scale
            transfer = expm(local*L/len(profile)) @ transfer
        return np.abs(transfer[:, 0])**2 * np.array([1., inputs.w_conj**2/ws**2])

    result = {"arm_order": ("probe", "conjugate"), "velocity_cutoff_sigma": 6.0}
    for velocity_step in (2.5, 1.25):
        v, weights = doppler.velocity_grid(T, dv=velocity_step, cutoff_sigma=6.0)
        direct = fwm.pump_only_weak_response_reference(
            pump, pump, [delta_rad], detuning-constants.K_VEC*v, atom=atom, branch=inputs.branch)
        result[f"weak_resolvent_1d_step_{velocity_step}_m_per_s"] = propagated(
            tuple(weights @ getattr(direct, key)[0, 0]
                  for key in ("chi_ss", "chi_cs", "chi_sc", "chi_cc")))
    for nodes in (24, 32):
        for name, angle in (("axial", 0.0), ("angular", inputs.angle_deg)):
            ref = fwm.finite_seed_angular_reference(
                pump, pump, seed, seed, [delta_rad], detuning, T=T,
                pump_k_rad_m=kp, probe_k_axis_rad_m=ks, crossing_angle_rad=np.radians(angle),
                atom=atom, transverse_order=nodes, branch=inputs.branch)
            result[f"finite_seed_{name}_{nodes}_nodes"] = {
                order: propagated(response) for order, response in ref["responses"].items()
            }
    return result


def _conditional_sensitivity(raw, inputs):
    """One-factor perturbations on the production source, with no fitting."""
    axis = raw["raman_center_minus_GHz"] + 1e-3*np.array(
        [inputs.delta_mhz, inputs.delta_mhz+0.01])
    context = raw["stark_recompute_context"]
    baseline = raw["parameter_provenance"]
    factors = {
        key: {key: baseline[key]["value"]*0.9} for key in
        ("mode_overlap_penalty", "polarization_penalty", "zeeman_participation_penalty")
    }
    variants = {
        "baseline": {},
        "ell_minus_10pct": {"line_strength": baseline["ell_s"]["value"]*0.9},
        **{key+"_minus_10pct": settings for key, settings in factors.items()},
        "transit_minus_10pct": {"transit_rate": inputs.transit_rate*0.9},
        "transit_plus_10pct": {"transit_rate": inputs.transit_rate*1.1},
        "scatter_kappa_zero": {"excess_noise_model": {"pump_scatter_kappa": 0.}},
        "scatter_kappa_twice": {"excess_noise_model": {
            "pump_scatter_kappa": 2*baseline["kappa"]["value"]}},
    }
    result = {}
    for name, overrides in variants.items():
        sampled = fwm.compute_spectrum(
            **(context | overrides), _readout_probe_points_GHz=axis)
        result[name] = {
            key: float(sampled[key][0]) for key in ("G_s", "G_c", "gain_referred_noise_dB")
        }
    return result


def _pump_controls(medium):
    """Isolate pump phase and pump amplitude at otherwise identical inputs."""
    initial = np.sqrt([medium.inputs.P_pump, medium.inputs.P_probe, 0.]).astype(complex)
    result = {}
    for name in ("coupled", "fixed_phase", "fixed_complex_pump"):
        def derivative(z, a):
            rate = medium.carrier_rate(z, a)
            if name == "fixed_phase":
                rate[0] = rate[0].real
            elif name == "fixed_complex_pump":
                rate[0] = 0.
            return rate

        solution = solve_ivp(derivative, (0., medium.inputs.L), initial, method="DOP853",
                             rtol=2e-8, atol=2e-12, max_step=medium.inputs.L/8)
        if not solution.success:
            raise RuntimeError(solution.message)
        output = np.abs(solution.y[:, -1])**2
        result[name] = {
            "output_powers_W": output,
            "gain_pair": output[1:]/medium.inputs.P_probe,
            "pump_accumulated_phase_rad": float(np.unwrap(np.angle(solution.y[0]))[-1]),
            "right_hand_side_evaluations": solution.nfev,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=_PRESET_PREFIXES)
    parser.add_argument("--segments", type=int, default=32, help="coarse axial grid, doubled once")
    parser.add_argument("--velocity-step", type=float, default=2.5, help="m/s")
    parser.add_argument("--transverse-order", type=int, default=3)
    parser.add_argument("--floquet-order", type=int, default=3)
    parser.add_argument("--point-audit", action="store_true",
                        help="compare fixed-pump references and one-factor sensitivities")
    parser.add_argument("--pump-controls", action="store_true",
                        help="isolate the effects of pump phase and pump depletion")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    scheme = fwm.FWMScheme()
    preset = next(p for p in scheme.presets() if p.name == _PRESET_PREFIXES[args.preset])
    params = preset.values
    raw = scheme.compute(params)
    focus = fwm.seeded_operating_readout(raw, params)
    point = focus["atomic_operating_point"]
    context = raw["stark_recompute_context"]
    inputs = LongitudinalSeededInputs(
        **{k: context[k] for k in ("D_GHz", "T", "L", "P_pump", "P_probe",
                                  "w_pump", "w_probe", "transit_rate", "branch")},
        delta_mhz=params["tpd"], w_conj=raw["w_conjugate_m"],
        angle_deg=context["pump_probe_angle_deg"],
        coupling_residual=raw["combined_line_strength_residual"],
    )
    config = dict(vars(args), inputs=asdict(inputs), preset_name=preset.name,
                  calibration_selected=params["gain_calibration_model"],
                  calibration_enabled=params["experimental_gain_calibration"])
    with runner.start_run("squeezing/longitudinal", out=args.out, config=config) as run:
        axis_mhz = (raw["probe_axis_GHz"]-raw["raman_center_minus_GHz"])*1e3
        peaks = {key: int(np.nanargmax(raw[key])) for key in ("G_s", "G_c")}
        summary = dict(
            inputs=asdict(inputs),
            shipped={
                key: point[key] for key in ("G_s", "G_c", "G_s_smallsignal", "G_c_smallsignal")
            },
            preset_calibration_enabled=params["experimental_gain_calibration"],
            preset_calibration_selected=params["gain_calibration_model"],
            preset_seed_provenance=params["seed_input_assumption"],
            preset_readout_scope=focus["readout_certification_scope"],
            native_raw_gain_peaks={
                key: {"delta_mhz": float(axis_mhz[i]), "gain": float(raw[key][i])}
                for key, i in peaks.items()
            },
            native_raw_gain_scan=dict(points=int(axis_mhz.size),
                                      span_mhz=[float(axis_mhz[0]), float(axis_mhz[-1])]),
        )
        if args.point_audit:
            summary["fixed_pump_reference"] = _fixed_pump_references(raw, inputs)
            summary["parameter_sensitivity"] = _conditional_sensitivity(raw, inputs)
            summary["parameter_provenance"] = raw["parameter_provenance"]
            summary["coupling_normalization_ledger"] = raw["coupling_normalization_ledger"]
            summary["reported_detection_ledger"] = focus["experimental_jasperse_readout"][
                "detection_efficiency_ledger"]
        if args.pump_controls or not args.point_audit:
            medium = LongitudinalSeededMedium(
                inputs, velocity_step=args.velocity_step, transverse_order=args.transverse_order,
                floquet_order=args.floquet_order,
            )
            summary["three_carrier_discretization"] = {
                "transverse_order": medium.transverse_order,
                "velocity_step_m_per_s": medium.velocity_step,
                "velocity_cutoff_sigma": medium.velocity_cutoff,
                "velocity_classes": medium.deff.size,
                "floquet_order": medium.floquet_order,
            }
            if args.pump_controls:
                summary["pump_controls"] = _pump_controls(medium)
            else:
                solution = medium.converged(args.segments, noise_ledger=True)
                noise = solution["noise_ledger"]
                summary.update(
                    local_gain_pair=[solution["probe_gain"], solution["conjugate_gain"]],
                    output_powers_W=solution["power_W"][-1],
                    cumulative_photon_sinks_s_inverse=solution["cumulative_photon_sinks_s_inverse"][-1],
                    photon_sink_labels=solution["photon_sink_labels"],
                    max_power_balance_residual_W=float(np.max(np.abs(solution["power_balance_residual_W"]))),
                    longitudinal_convergence=solution["longitudinal_convergence"],
                    noise_ledger_scope=noise["scope"], noise_basis=noise["basis"],
                    noise_reservoirs=noise["reservoirs"], noise_reservoir_kinds=noise["reservoir_kinds"],
                    noise_reservoir_sources=noise["reservoir_sources"],
                    noise_coupled_density_m3=noise["coupled_density_m3"],
                )
                np.savez_compressed(
                    run.path("trajectory.npz"),
                    **{k: v for k, v in solution.items() if isinstance(v, np.ndarray)},
                    noise_midpoint_z_m=noise["midpoint_z_m"],
                    ordered_atomic_diffusion_s_inverse=noise["ordered_atomic_diffusion_s_inverse"],
                )
        io.write_json(run.path("summary.json"), summary, default=lambda value: value.tolist())
        print(io.dumps_json(summary, default=lambda value: value.tolist()), flush=True)


if __name__ == "__main__":
    main()
