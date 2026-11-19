"""Independent scalar-line and physical balance checks for all Rb D-line OBEs.

The weak reference uses integrated hyperfine strengths, not the solver's
Zeeman dipole matrices or its Hamiltonian/coherence assembly.
"""
import numpy as np
import pytest

from gabes import constants, core, species
from gabes.optical_bloch import solve_optical_pump


CASES = [("Rb85", "D1", 4), ("Rb85", "D2", 6),
         ("Rb87", "D1", 4), ("Rb87", "D2", 6)]
DENSITY_M3 = 1e15
RESET_PER_S = 5e5


def _scalar_line_data(isotope, line):
    """Independent F-level centroid and integrated strengths from atomic data."""
    iso = species.ISOTOPES[isotope]
    je, frequency, gamma_mhz, ae, be = iso.line(line)
    upper_f = max(species.f_values(iso.I, iso.Jg))
    upper_energy = species.hf_energy_mhz(iso.A_S, 0., iso.I, iso.Jg, upper_f)
    excited_f = species.f_values(iso.I, je)
    strengths = np.array([species.line_strength(upper_f, fe, iso.I, iso.Jg, je)
                          for fe in excited_f])
    excited_energy = np.array([species.hf_energy_mhz(ae, be, iso.I, je, fe)
                              for fe in excited_f])
    centroid = float(strengths @ excited_energy / strengths.sum() - upper_energy)
    integrated = species.line_integrated_alpha(iso, line, DENSITY_M3)
    offsets = {
        (fg, fe): (species.hf_energy_mhz(ae, be, iso.I, je, fe)
                   - species.hf_energy_mhz(iso.A_S, 0., iso.I, iso.Jg, fg)
                   - centroid)
        for fg, fe in integrated
    }
    return dict(centroid=centroid, offsets=offsets, integrated=integrated,
                frequency=frequency, gamma=gamma_mhz * species.MHZ,
                k=2 * np.pi * frequency / constants.C_LIGHT)


def _weak_reference(data, detuning_mhz):
    # An isolated Lorentzian integrates to K in angular-frequency units.
    # Full thermal replacement dephases an optical coherence at reset, so
    # HWHM = Gamma/2 + reset. This also fixes the dispersive sign independently.
    hwhm = data["gamma"] / 2 + RESET_PER_S
    return sum(
        data["integrated"][transition] / (np.pi * data["k"])
        / ((offset - detuning_mhz) * species.MHZ - 1j * hwhm)
        for transition, offset in data["offsets"].items()
    )


@pytest.mark.parametrize(("isotope", "line", "transition_count"), CASES)
def test_weak_field_matches_scalar_spectrum_at_every_allowed_line(
        isotope, line, transition_count):
    data = _scalar_line_data(isotope, line)
    assert len(data["offsets"]) == transition_count
    # Nonzero drive exercises the actual steady-state solve. One natural
    # HWHM off each line samples both quadratures without requiring an
    # expensive frequency scan or hiding Im(chi) behind the larger real part.
    for transition, resonance in data["offsets"].items():
        detuning = resonance + data["gamma"] / (2 * species.MHZ)
        state = solve_optical_pump(
            isotope, line, detuning, intensity_mw_cm2=1e-9,
            density_m3=DENSITY_M3, ground_relaxation_hz=RESET_PER_S)
        expected = _weak_reference(data, detuning)
        label = f"{isotope} {line} F={transition[0]} -> F'={transition[1]}"
        np.testing.assert_allclose(state["chi_probe"].real, expected.real,
                                   rtol=1e-6, atol=1e-17, err_msg=label)
        np.testing.assert_allclose(state["chi_probe"].imag, expected.imag,
                                   rtol=1e-6, atol=1e-17, err_msg=label)
        assert state["chi_probe"].imag > 0., label
        np.testing.assert_allclose(state["chi_plus"], state["chi_minus"],
                                   rtol=1e-8, atol=1e-17, err_msg=label)


@pytest.mark.parametrize(("isotope", "line", "transition_count"), CASES)
def test_zero_drive_is_thermal_with_correct_scalar_limit(isotope, line, transition_count):
    data = _scalar_line_data(isotope, line)
    detuning = min(data["offsets"].values()) - 250.
    state = solve_optical_pump(
        isotope, line, detuning, intensity_mw_cm2=0.,
        density_m3=DENSITY_M3, ground_relaxation_hz=RESET_PER_S)
    iso = species.ISOTOPES[isotope]
    je = iso.line(line)[0]
    ng = int((2 * iso.Jg + 1) * (2 * iso.I + 1))
    ne = int((2 * je + 1) * (2 * iso.I + 1))
    assert state["rho"].shape == (ng + ne, ng + ne)
    np.testing.assert_allclose(state["rho"], np.diag([1 / ng] * ng + [0.] * ne),
                               rtol=0., atol=1e-12)
    np.testing.assert_array_equal(state["polarization_xy_c_m2"], 0.)
    np.testing.assert_allclose(state["chi_probe"], _weak_reference(data, detuning),
                               rtol=2e-11, atol=1e-18)
    assert state["conditions"]["reference_centroid_mhz"] == pytest.approx(
        data["centroid"], abs=2e-10)


@pytest.fixture(scope="module", params=CASES, ids=lambda item: "-".join(item[:2]))
def driven_state(request):
    isotope, line, _ = request.param
    data = _scalar_line_data(isotope, line)
    transition = max(data["offsets"], key=lambda item: (item[0], item[1]))
    detuning = data["offsets"][transition] + data["gamma"] / species.MHZ
    inputs = dict(intensity_mw_cm2=35., ground_relaxation_hz=RESET_PER_S,
                  jones_xy=(3 / 5, 4j / 5), density_m3=DENSITY_M3)
    state = solve_optical_pump(isotope, line, detuning, **inputs)
    return isotope, line, data, detuning, inputs, state


def test_finite_drive_obeys_lindblad_and_absorbed_power_balance(driven_state):
    isotope, line, data, _, inputs, state = driven_state
    rho = state["rho"]
    ng = len(state["ground_levels"])
    np.testing.assert_allclose(rho, rho.conj().T, rtol=0., atol=1e-10)
    assert rho.trace() == pytest.approx(1., abs=1e-10)
    assert np.linalg.eigvalsh(rho).min() >= -1e-10
    excited = float(rho[ng:, ng:].trace().real)
    assert 0. < excited < 1.
    assert excited == pytest.approx(state["excited_fraction"], rel=1e-11)

    # Apply named jump operators directly, independent of vectorization and
    # any optimized parity/sparse path used by the production steady solve.
    hamiltonian = state["hamiltonian_rad_s"]
    residual = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    emission_metric = np.zeros_like(rho)
    with core.blas_single_thread():
        for channel in state["reservoirs"].channels:
            jump = channel.operator
            metric = jump.conj().T @ jump
            residual += jump @ rho @ jump.conj().T - .5 * (metric @ rho + rho @ metric)
            if channel.kind == "spontaneous_emission":
                emission_metric += metric
    projector = np.diag([0.] * ng + [1.] * (rho.shape[0] - ng))
    np.testing.assert_allclose(emission_metric / data["gamma"], projector,
                               rtol=0., atol=2e-13)
    assert np.linalg.norm(residual) < 1e-8 * RESET_PER_S

    assert state["alpha_per_m"] > 0.
    assert state["alpha_per_m"] == pytest.approx(
        data["k"] * state["chi_probe"].imag, rel=2e-10)
    absorbed = state["alpha_per_m"] * (10 * inputs["intensity_mw_cm2"])
    photon_energy = 2 * np.pi * constants.HBAR * data["frequency"]
    decay = DENSITY_M3 * photon_energy * (data["gamma"] + RESET_PER_S) * excited
    assert absorbed == pytest.approx(decay, rel=2e-8)


def test_finite_drive_saturates_each_dline_response(driven_state):
    _, _, data, detuning, _, state = driven_state
    linear = _weak_reference(data, detuning)
    assert 0. < state["alpha_per_m"] < .5 * data["k"] * linear.imag
    assert abs(state["chi_probe"] - linear) > .1 * abs(linear)


def test_polarization_rotation_covariance_at_zero_magnetic_field(driven_state):
    isotope, line, _, detuning, inputs, state = driven_state
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    rotated_inputs = dict(inputs, jones_xy=rotation @ np.asarray(inputs["jones_xy"]))
    rotated = solve_optical_pump(isotope, line, detuning, **rotated_inputs)
    np.testing.assert_allclose(rotated["chi_probe"], state["chi_probe"],
                               rtol=2e-9, atol=1e-17)
    np.testing.assert_allclose(rotated["polarization_xy_c_m2"],
                               rotation @ state["polarization_xy_c_m2"],
                               rtol=2e-9, atol=1e-25)
    assert rotated["excited_fraction"] == pytest.approx(state["excited_fraction"], rel=2e-9)


def test_optimized_full_system_matches_dense_reference(driven_state):
    isotope, line, _, detuning, inputs, state = driven_state
    reference = solve_optical_pump(isotope, line, detuning, **inputs, solver="full")
    np.testing.assert_allclose(state["rho"], reference["rho"], rtol=2e-9, atol=2e-12)
    np.testing.assert_allclose(state["chi_probe"], reference["chi_probe"],
                               rtol=2e-9, atol=1e-17)
    np.testing.assert_allclose(state["polarization_xy_c_m2"],
                               reference["polarization_xy_c_m2"],
                               rtol=2e-9, atol=1e-25)
