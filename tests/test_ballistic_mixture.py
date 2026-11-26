"""Analytical weak absorption and common axial depletion for isotope-resolved Rb histories."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import voigt_profile

from gabes import ballistic_spectrum as history, beam, constants, doppler, species, vapor_cell
from gabes.od_model import ODConditions


SCAN = 2*np.pi*np.array([0.41e9, -2.31e9, 0.41e9, 3.025e9, -1.08e9])


def conditions(medium="Rb (natural)", line="D1", **changes):
    base = ODConditions(species=medium, line=line, temp_c=79., cell_mm=12.5,
                        probe_waist_mm=.53, model_id="od_rb_ballistic_history_v1")
    return replace(base, **changes)


def analytical_weak(c, parts, reference, profile=voigt_profile):
    cold = c.temp_c + 273.15
    kinetic = cold if c.doppler_temp_c is None else c.doppler_temp_c + 273.15
    spectrum = np.zeros_like(SCAN)
    for iso, fraction in parts:
        atom = species.build_manifold(iso, c.line)
        total = vapor_cell.cold_spot_limited_density(species.number_density(iso, cold), cold, kinetic)
        density = fraction*total
        width = .5*(atom.gamma + species.self_broadening(iso, total, c.line, per_line=True))
        width += np.pi*c.extra_homogeneous_fwhm_hz
        sigma = atom.k_vec * np.sqrt(constants.KB*kinetic/iso.mass)
        offset = 2*np.pi*(atom.nu0 - reference.line(c.line)[1])
        for (fg, fe), area in species.line_integrated_alpha(iso, c.line, density).items():
            center = atom.e_excited[atom.Fe.index(fe)] - atom.e_ground[atom.Fg.index(fg)]
            spectrum += area * profile(SCAN - offset - center, sigma, width)
    return spectrum * c.cell_mm * 1e-3 * c.line_strength


@pytest.mark.parametrize("line", ["D1","D2"])
@pytest.mark.parametrize("medium,fraction", [(species.RB85.label, 1.), (species.RB87.label, 0.),
                                            ("Rb (natural)", 0.38), ("Rb (natural)", None)])
def test_weak_thermal_areas_density_and_frequency_frame(medium, fraction, line):
    custom = medium == "Rb (natural)" and fraction is not None
    c = conditions(medium, line, rb_composition="Custom" if custom else "Natural",
                   rb85_fraction=fraction if custom else species.RB85.abundance,
                   doppler_temp_c=94., extra_homogeneous_fwhm_hz=0.8e6, line_strength=1.21)
    reference = species.RB87 if medium == species.RB87.label else species.RB85
    f85 = species.RB85.abundance if fraction is None else fraction
    parts = [(species.RB85, f85), (species.RB87, 1-f85)]
    expected = analytical_weak(c, parts, reference)
    result = history.evaluate_rb_ballistic_spectrum(c, SCAN)
    # In the weak wings compare absolute OD. The independent SciPy line
    # profile and the product's complex Faddeeva expansion have different
    # real-part roundoff. Resolved absorption also has a relative bound.
    np.testing.assert_allclose(result.od0, expected, rtol=5e-12, atol=2e-12)
    np.testing.assert_allclose(result.transmission, np.exp(-expected), rtol=0, atol=2e-12)
    # Isolate isotope weights and offsets using the same native Voigt kernel
    # in an independently assembled transition sum, including the far wings.
    def native_profile(delta, sigma, hwhm):
        return doppler.voigt_profile(delta, 2*hwhm, sigma)
    expected_native = analytical_weak(c, parts, reference, native_profile)
    np.testing.assert_allclose(result.od0, expected_native, rtol=5e-12, atol=0)
    np.testing.assert_array_equal(result.od0, result.optical_depth)
    assert result.transmission[0] == result.transmission[2]
    assert not result.optical_depth.flags.writeable


@pytest.mark.parametrize("line", ["D1", "D2"])
@pytest.mark.parametrize("iso, fraction", [(species.RB85, 1.), (species.RB87, 0.)])
def test_enriched_endpoint_keeps_rb85_display_origin(iso, fraction, line):
    endpoint = conditions(line=line, rb_composition="Custom", rb85_fraction=fraction)
    pure = replace(endpoint, species=iso.label)
    # SCAN is measured from Rb85; pure Rb87 displays from its own centroid.
    shift = 2*np.pi*(iso.line(line)[1] - species.RB85.line(line)[1])
    a = history.evaluate_rb_ballistic_spectrum(endpoint, SCAN)
    b = history.evaluate_rb_ballistic_spectrum(pure, SCAN-shift)
    np.testing.assert_allclose(a.od0, b.od0, rtol=5e-14, atol=0)


def test_saturated_isotopes_share_one_propagating_beam(monkeypatch):
    c = conditions(line="D2", rb_composition="Custom", rb85_fraction=.38, probe_power_uw=10.)
    peak = beam.peak_intensity(c.probe_power_uw * 1e-6, c.probe_waist_mm * 1e-3)
    column = c.cell_mm * 1e-3 * species.number_density(species.RB85, c.temp_c+273.15)
    exponent = 0.2
    intrinsic = {species.RB85: 2., species.RB87: 1.2}
    sample_scale = 2*np.pi*0.53e9

    def local(optics, scan, intensities, resolution, **kwargs):
        scale = intrinsic[optics.isotope]*(1 + .15*np.sin(scan/sample_scale)) / column
        table = scale[None, :] * (intensities[:, None]/peak)**(-exponent)
        return table[0], table

    monkeypatch.setattr(history, "_thermal_table", local)
    result = history.evaluate_rb_ballistic_spectrum(c, SCAN)
    # For depth D, dD/d(z/L) = (A85+A87)*exp(exponent*D) with D(0) = 0.
    # The cell never reaches the lowest sampled power, where this test law ends.
    scale = []
    for iso, frac in [(species.RB85, .38), (species.RB87, .62)]:
        shift = 2*np.pi*(iso.line("D2")[1]-species.RB85.line("D2")[1])
        scale.append(frac*intrinsic[iso]*(1 + .15*np.sin((SCAN-shift)/sample_scale)))
    def depth(strength):
        return -np.log1p(-exponent*strength)/exponent
    expected = depth(scale[0]+scale[1])
    np.testing.assert_allclose(result.optical_depth, expected, rtol=0, atol=5e-4)
    np.testing.assert_allclose(result.transmission, np.exp(-expected), rtol=0, atol=2e-4)
    separate = np.exp(-depth(scale[0])-depth(scale[1]))
    assert np.max(np.abs(separate-result.transmission)) > .005


@pytest.mark.parametrize("iso,line", [(species.RB85,"D1"), (species.RB87,"D1"),
                                    (species.RB85,"D2"), (species.RB87,"D2")])
def test_live_finite_transverse_trajectories_reduce_thermal_absorption(iso, line):
    c = conditions(iso.label, line)
    part, _, shift = history._rb_components(c)[0]
    assert shift == 0
    atom = species.build_manifold(iso, line)
    scan = np.array([atom.omega[0], atom.omega[-1], atom.omega[0]])
    mesh = history._Resolution(path_nodes=(32,4,6), panel_order=2, wing_widths=64)
    weak, finite = history._thermal_table(part, scan, [10.], mesh)
    assert np.all(finite > 0) and np.all(finite[0] < weak)
    assert finite[0, 0] == pytest.approx(finite[0, 2], rel=1e-14, abs=0)


@pytest.mark.parametrize("medium,line", [(species.RB87.label, "D1"), ("Rb (natural)", "D2")])
@pytest.mark.parametrize("model_id", ["od_rb_ballistic_history_v1", "od_rb_ballistic_history_fast_v1"])
def test_public_contract_and_csv_recover_isotope_origin_and_history(medium, line, model_id):
    from gabes.od_measurement import export_comparison_csv, load_measurement, prepare_measurement_upload
    from gabes.od_model import evaluate_od, evaluate_od_display
    from gabes.schemes.sas import SASScheme
    from gabes_ui.od_measurement import geometry_replay_params, plot_metadata, workspace_params

    c = conditions(medium, line, model_id=model_id,
                   rb_composition="Custom", rb85_fraction=.38)
    display = np.sort(np.unique(SCAN))
    shift = species.RB87.line(line)[1]-species.RB85.line(line)[1] if medium == species.RB87.label else 0.
    offsets = display/(2*np.pi) + shift
    direct = evaluate_od(c, offsets)
    shown = evaluate_od_display(c, display)
    assert direct.frequency_reference.isotope == species.RB85.name
    assert direct.model_id == model_id
    if model_id.endswith("fast_v1"):
        assert direct.diagnostics["fidelity_used"] == "reference"
    np.testing.assert_allclose(shown.offsets_hz, offsets, rtol=1e-15, atol=1e-6)
    np.testing.assert_allclose(shown.transmission, direct.transmission, rtol=1e-13, atol=1e-15)

    p = c.to_params()
    metadata = plot_metadata(p, c.model_id)
    ghz = display/(2*np.pi*1e9)
    pairs = "\n".join(f"{x:.17g},{y:.17g}" for x,y in zip(ghz, shown.transmission))
    upload = prepare_measurement_upload(("frequency_GHz,transmission\n"+pairs+"\n").encode("ascii"))
    trace = load_measurement(upload, mode="fraction", x_unit="GHz")
    saved = export_comparison_csv(upload, trace, ghz, shown.transmission, metadata=metadata)
    reopened = prepare_measurement_upload(saved)
    assert reopened.metadata["model_id"] == c.model_id
    assert reopened.metadata["transit"]["rate_s_inverse"] == {}
    assert reopened.metadata["display_reference_offset_hz"] == shift
    defaults = dict(SASScheme().defaults(), **geometry_replay_params(reopened))
    replay = ODConditions.from_params(workspace_params(defaults, {}, reopened))
    assert (replay.model_id, replay.species, replay.line, replay.rb85_fraction) == (
        c.model_id, c.species, c.line, c.rb85_fraction)
    np.testing.assert_array_equal(evaluate_od(replay, offsets).transmission, direct.transmission)
