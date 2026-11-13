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
    scalar = scheme.compute(params)
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


def test_joint_reference_matches_direct_held_out_points_and_reports_scalar_delta():
    scheme = LambdaScheme()
    params = dict(scheme._default_values("eit"), beam_angle_mrad=0.7,
                  coupling_detuning_mhz=2.4, species="\u00b9\u00b3\u00b3Cs", line="D2",
                  angled_doppler_model=LAMBDA_JOINT_MAXWELL)
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


def test_unconverged_transverse_average_withholds_cpt_width_and_group_index():
    scheme = LambdaScheme()
    params = dict(scheme._default_values("cpt"), doppler="on", beam_angle_mrad=5.,
                  angled_doppler_model=LAMBDA_JOINT_MAXWELL)
    raw = scheme.compute(params)
    evidence = raw["joint_doppler"]
    assert evidence["status"] == "not_converged"
    assert evidence["transverse_refinement"][-1]["order"] == 256
    window = scheme.headless_observables(raw, params)["lambda_readout"]["window"]
    assert window["status"] == "velocity_not_converged"
    assert window["width_mhz"] is None
    assert window["group_index"] is None
    assert window["group_delay_s"] is None


def test_joint_mode_rejects_angles_outside_its_control_range():
    scheme = LambdaScheme()
    params = dict(scheme._default_values("eit"), beam_angle_mrad=20.1,
                  angled_doppler_model=LAMBDA_JOINT_MAXWELL)
    with pytest.raises(ValueError, match="beam angle"):
        scheme.compute(params)
