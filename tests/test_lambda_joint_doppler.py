"""Angled Lambda joint Maxwell reference and scalar compatibility."""
import numpy as np
import pytest

from gabes import atoms, constants, core, doppler
from gabes.schemes.absorption import (
    KHZ, MHZ, PROBE_RABI, LAMBDA_JOINT_MAXWELL, LambdaScheme, _medium_from_params,
)


@pytest.mark.parametrize("mode", ["eit", "at", "cpt"])
@pytest.mark.parametrize("doppler_mode,angle,status", [
    ("off", 10., "doppler_off"), ("on", 0., "collinear_limit"),
])
def test_joint_mode_recovers_inactive_limits_exactly(mode, doppler_mode, angle, status):
    scheme = LambdaScheme()
    params = dict(scheme._default_values(mode), doppler=doppler_mode, beam_angle_mrad=angle)
    automatic = scheme.compute(params)
    scalar = scheme.compute(dict(params, angled_doppler_model="Scalar"))
    assert automatic.keys() == scalar.keys()
    for key, expected in scalar.items():
        np.testing.assert_array_equal(automatic[key], expected)
    opt_in = scheme.compute(dict(params, angled_doppler_model=LAMBDA_JOINT_MAXWELL))
    assert opt_in.pop("joint_doppler") == {"status": status}
    assert scalar.keys() == opt_in.keys()
    for key, expected in scalar.items():
        np.testing.assert_array_equal(opt_in[key], expected)


def _direct_cartesian_mean(scan, raw, params, *, step_z, order_x, cutoff):
    """Independent tensor sum using the full Hamiltonian at each velocity."""
    medium = _medium_from_params(params)
    kp = medium["k_vec"]
    kc = kp - 2*np.pi*medium["ground_hf_hz"] / constants.C_LIGHT
    theta = params["beam_angle_mrad"] * 1e-3
    vz, wz = doppler.velocity_grid(medium["T"], medium["mass"], step_z, cutoff)
    vx, wx = doppler.maxwell_legendre_grid(medium["T"], medium["mass"], order_x, cutoff)
    longitudinal_raman = (kp - kc*np.cos(theta)) * vz
    transverse_raman = -kc*np.sin(theta) * vx
    atom = atoms.lambda3(gamma_gg=raw["buffer_ground_relax_khz"]*KHZ,
                         gamma=medium["gamma"])
    probe = PROBE_RABI*medium["gamma"]
    hamiltonian = np.array([[0., 0., probe/2],
                            [0., 0., raw["coupling_rabi_mhz"]*MHZ/2],
                            [probe/2, raw["coupling_rabi_mhz"]*MHZ/2, 0.]], dtype=complex)
    base = core.build_liouvillian(hamiltonian, atom)
    optical = core.comm_super(np.diag([0., 0., 1.]))
    ground = core.comm_super(np.diag([0., 1., 0.]))
    velocity_z = (longitudinal_raman[:, None, None] * ground +
                  (kp*vz)[:, None, None] * optical)
    result = []
    with core.blas_single_thread():
        for detuning in scan:
            shifted = (base + (raw["Dc"] - detuning) * ground - detuning * optical
                       + velocity_z)
            average = 0j
            for x_shift, weight in zip(transverse_raman, wx):
                rho = core.steady_state_from_liouvillian(shifted + x_shift * ground, 3)
                average += weight * np.dot(wz, rho[:, 2, 0]) / probe
            result.append(average)
    return np.asarray(result)


def test_angled_default_matches_direct_held_out_points_and_reports_scalar_delta():
    scheme = LambdaScheme()
    params = dict(scheme.defaults(), beam_angle_mrad=0.7,
                  coupling_detuning_mhz=2.4, species="\u00b9\u00b3\u00b3Cs", line="D2")
    raw = scheme.compute(params)
    assert raw["joint_doppler"]["status"] == "converged"
    assert raw["joint_doppler"]["transverse_cutoff_check"]["cutoff_sigma"] == 7.
    indices = [183, 286, 313, 397]
    scan = raw["scan"][indices]
    coarse = _direct_cartesian_mean(scan, raw, params, step_z=0.75, order_x=64, cutoff=7.)
    fine_z = _direct_cartesian_mean(scan, raw, params, step_z=0.375, order_x=64, cutoff=7.)
    fine = _direct_cartesian_mean(scan, raw, params, step_z=0.375, order_x=96, cutoff=7.)
    scale = np.max(np.abs(fine))
    np.testing.assert_allclose(coarse, fine_z, atol=1e-5*scale, rtol=0)
    np.testing.assert_allclose(fine_z, fine, atol=1e-5*scale, rtol=0)
    np.testing.assert_allclose(raw["chi_bar"][indices], fine, atol=1e-5*scale, rtol=0)
    assert np.max(np.abs(raw["chi_bar"] - raw["scalar_chi_bar"])) > 1e-3*scale
    view = scheme.headless_observables(raw, params)
    evidence = view["lambda_readout"]["joint_doppler"]
    assert evidence["difference_from_scalar"]["complex_chi_relative_max"] > 0.
    assert evidence["longitudinal_raman_k_rad_m"] > 0.
    assert evidence["transverse_raman_k_rad_m"] < 0.
    assert evidence["raman_rms"]["total_rms_hz"] == pytest.approx(
        np.hypot(evidence["longitudinal_raman_k_rad_m"], evidence["transverse_raman_k_rad_m"])
        * evidence["raman_rms"]["sigma_velocity_m_s"] / (2*np.pi))


def test_default_unconverged_transverse_average_withholds_cpt_width_and_group_index():
    scheme = LambdaScheme()
    params = dict(scheme._default_values("cpt"), doppler="on", beam_angle_mrad=5.)
    raw = scheme.compute(params)
    evidence = raw["joint_doppler"]
    assert evidence["status"] == "not_converged"
    assert evidence["transverse_refinement"][-1]["order"] == 256
    window = scheme.headless_observables(raw, params)["lambda_readout"]["window"]
    assert window["status"] == "velocity_not_converged"
    assert window["width_mhz"] is None
    assert window["group_index"] is None
    assert window["group_delay_s"] is None


@pytest.mark.parametrize("angle", [-0.1, 20.1, float("nan")])
@pytest.mark.parametrize("model", [None, LAMBDA_JOINT_MAXWELL])
def test_joint_mode_rejects_angles_outside_its_control_range(angle, model):
    scheme = LambdaScheme()
    params = dict(scheme.defaults(), beam_angle_mrad=angle)
    if model is not None:
        params["angled_doppler_model"] = model
    with pytest.raises(ValueError, match="beam angle"):
        scheme.compute(params)


@pytest.mark.parametrize("mode,joint", [("eit", False), ("at", False), ("cpt", True)])
def test_capped_velocity_row_withholds_metrics_comparisons_and_json(monkeypatch, mode, joint):
    import json
    from gabes import affine_doppler
    from gabes_ui.export import build_export_payload

    def capped(self, scan, temperature, mass, k_vec, *, fallback_rows=None):
        row = len(scan) // 2
        response = np.full(scan.shape, -1e-10j)
        response[row] = complex(np.nan, np.nan)
        fallback_rows.append(dict(row=row, status="unresolved", classes=351811,
                                  total_classes=500000, relative_change=.01, refinements=14))
        return response

    monkeypatch.setattr(affine_doppler.AffineScan, "maxwell_mean", capped)
    scheme = LambdaScheme()
    params = dict(scheme.defaults(), **scheme._default_values(mode))
    params.update(doppler="on", beam_angle_mrad=1. if joint else 0.)
    raw = scheme.compute(params)
    view = scheme.headless_observables(raw, params)
    data = view["lambda_readout"]
    if joint:
        evidence = data["joint_doppler"]
        assert evidence["status"] == "not_converged"
        assert all(r["status"] == "unresolved" for r in evidence["longitudinal_fallback"])
        assert evidence["difference_from_scalar"]["complex_chi_relative_max"] is None
    else:
        assert data["doppler_fallback"] == raw["doppler_fallback"]
    metrics = {m["label"]: m for m in view["metrics"]}
    center = "Transmission at center" if mode == "at" else "Transmission at resonance"
    assert metrics[center]["numeric_value"] is None
    if mode == "at":
        assert metrics["AT splitting"]["numeric_value"] is None
    else:
        assert data["window"]["status"] == "velocity_not_converged"
        assert data["window"]["width_mhz"] is None
        assert data["window"]["group_index"] is None

    comparison = scheme.model_comparison(params, resolve_raw=scheme.compute,
                                         current_raw=raw, current_view=view)
    for curve in comparison["series"]:
        center_row = len(curve["x"]) // 2
        assert curve["observables"]["transmission"][center_row] is None
        values = {r["key"]: r["value"] for r in curve["readouts"]}
        assert values["central_transmission"] is None
        assert values["doublet_spacing" if mode == "at" else "window_width"] is None
        if mode != "at":
            assert values["contrast"] is None
        assert ("joint_doppler" if joint else "doppler_fallback") in curve
    bundle = build_export_payload(scheme=scheme, raw=raw, params=params, view=view,
                                  plot_payloads=[], comparison_payload=None,
                                  cache_version=scheme.cache_version, readout_cache_version="test")
    saved = json.loads(json.dumps(bundle, allow_nan=False))["view"]["lambda_readout"]
    assert saved["transmission"][len(raw["scan"]) // 2] is None
    assert ("joint_doppler" if joint else "doppler_fallback") in saved


def test_unresolved_scalar_comparison_keeps_converged_joint_readout(monkeypatch):
    from gabes import affine_doppler

    first_call = True

    def means(self, scan, temperature, mass, k_vec, *, fallback_rows=None):
        nonlocal first_call
        response = -1e-10j * (1 + .5 * scan / np.max(np.abs(scan)))
        if first_call:
            response[0] = complex(np.nan, np.nan)
            fallback_rows.append(dict(row=0, status="unresolved", classes=351811,
                                      total_classes=500000, relative_change=.01, refinements=14))
            first_call = False
        return response

    monkeypatch.setattr(affine_doppler.AffineScan, "maxwell_mean", means)
    scheme = LambdaScheme()
    params = dict(scheme.defaults(), doppler="on", beam_angle_mrad=1.)
    raw = scheme.compute(params)
    readout = scheme.headless_observables(raw, params)["lambda_readout"]
    joint = readout["joint_doppler"]
    assert joint["status"] == "converged"
    assert joint["scalar_reference_fallback"][0]["row"] == 0
    assert joint["difference_from_scalar"]["complex_chi_relative_max"] is None
    assert joint["difference_from_scalar"]["transmission_at_resonance"] is not None
    assert np.isfinite(readout["transmission"]).all()
