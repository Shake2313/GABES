"""Fit the 2026-10-13 FWM pump-off gain ratios on the exact Ultra source.

Run python -m gabes_lab.pipelines.fwm_lab_gain_refit [--out runs/...] .
Calibration data and fit results are development data, not a held-out test.
"""
import argparse
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

from gabes import fwm_experiment_gain, fwm_measured_gain as measured
from gabes.schemes import fwm
from gabes_lab.blocks import io, runner


def exact_gains(raw, delta_mhz):
    """Return solved rows at the specified source positions in input order."""
    delta_mhz = np.asarray(delta_mhz)
    probe = raw["raman_center_minus_GHz"] + delta_mhz * 1e-3
    context = raw["stark_recompute_context"]
    dense = fwm.compute_spectrum(**context, _readout_probe_points_GHz=np.unique(probe))
    method = dense["response_estimator"]
    if (method["method"] != fwm.RESPONSE_POLE or method["interpolation"] != "none"
            or method["pole_guard_fallback_rows"]):
        raise RuntimeError("The fit requires the exact pole solution at every source node.")
    index = np.searchsorted(dense["probe_axis_GHz"], probe)
    np.testing.assert_array_equal(dense["probe_axis_GHz"][index], probe)
    return {key: dense[key][index] for key in ("G_s", "G_c")}


def fit_arm(raw, nodes, data, key):
    """Optimize the warp and affine gain on solved coordinates, including its Jacobian."""
    native = 1e3 * (raw["probe_axis_GHz"] - raw["raman_center_minus_GHz"])
    peak = float(native[np.argmax(raw[key])])
    target = float(nodes[data >= np.max(data) * (1 - 1e-12)].mean())
    displacement = nodes - target
    derivative_step = 0.0001  # MHz, symmetric numerical derivative of the atomic gain

    @lru_cache(maxsize=1)
    def samples(shift, width):
        positions = peak + shift + width * displacement
        x = np.array([positions, positions - derivative_step, positions + derivative_step])
        current, left, right = exact_gains(raw, x)[key]
        return current, (right - left) / (2 * derivative_step)

    def residual(params):
        shift, width, log_scale, offset = params
        gains, _ = samples(shift, width)
        return np.log(np.exp(log_scale) * gains + offset) - np.log(data)

    def jacobian(params):
        shift, width, log_scale, offset = params
        gains, slope = samples(shift, width)
        amp = np.exp(log_scale)
        derivatives = np.column_stack((amp*slope, amp*slope*displacement,
                                       amp*gains, np.ones_like(gains)))
        return derivatives / (amp*gains + offset)[:, None]

    fit = least_squares(
        residual, [0., 1., np.log(np.max(data) / np.max(raw[key])), .001],
        jac=jacobian, bounds=([-8., .25, -12., 0.], [8., 4., 3., 1.]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1500)
    if not fit.success:
        raise RuntimeError(f"{key} fit failed: {fit.message}")
    shift, width, log_scale, offset = fit.x
    positions = peak + shift + width * displacement
    gain = exact_gains(raw, positions)[key]
    if not np.all(np.isfinite(gain) & (gain > 0)):
        raise ValueError(f"{key} requires finite positive reference gains")
    affine = np.exp(log_scale) * gain + offset
    arm = dict(model_peak_delta_mhz=peak, measured_peak_delta_mhz=target,
               source_center_delta_mhz=float(peak + shift), frequency_scale=float(width),
               gain_scale=float(np.exp(log_scale)), gain_offset=float(offset),
               affine_max_relative_error=float(np.max(np.abs(affine / data - 1))),
               affine_rms_log_error=float(np.sqrt(np.mean(np.log(affine / data)**2))),
               log_total_coefficient_at_nodes=np.log(data / gain).tolist())
    training = dict(model_gain_at_exact_nodes=gain.tolist(), source_delta_mhz=positions.tolist(),
                    measured_ratios=data.tolist(), affine_predictions=affine.tolist(),
                    optimizer_status=int(fit.status), objective_evaluations=int(fit.nfev),
                    optimality=float(fit.optimality), message=fit.message)
    return arm, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="output directory; default is an independent runs/ folder")
    args = parser.parse_args()
    scheme = fwm.FWMScheme()
    lab = next(p.values for p in scheme.presets() if p.name == "2026-10-13 experiment")
    params = dict(lab, resolution=fwm.FIDELITY_ULTRA, velocity_measure="exact",
                  floquet_order=3, experimental_gain_calibration=False)
    source = measured.observations()
    nodes = measured.internal_delta_mhz_from_eom(source["eom_MHz"])
    with runner.start_run("fwm_lab_gain_refit", out=args.out, config=params) as run:
        raw = scheme.compute(params)
        profile = dict(model_id=fwm_experiment_gain.CURRENT_MODEL_ID,
                       source_workbook_sha256=measured.SOURCE_WORKBOOK_SHA256,
                       frozen_source_model=dict(raw["gain_source_model"]), arms={})
        training = dict(observation_eom_mhz=source["eom_MHz"].tolist(),
                        internal_delta_mhz=nodes.tolist(), arms={})
        for arm, key in (("probe", "G_s"), ("conjugate", "G_c")):
            profile["arms"][arm], training["arms"][arm] = fit_arm(
                raw, nodes, source[arm+"_ratio"], key)
        io.write_json(run.path("profile.json"), profile)
        io.write_json(run.path("training.json"), training)
        np.savez(run.path("native-reference.npz"), probe_axis_GHz=raw["probe_axis_GHz"],
                 G_s=raw["G_s"], G_c=raw["G_c"])
        print(run.dir)


if __name__ == "__main__":
    main()
