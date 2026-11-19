"""The isolated profiler input must retain its actual line and physical plane."""
from dataclasses import FrozenInstanceError, replace
import json
import math

import numpy as np
import pytest

from gabes import constants
from sabes.beamline import build_source_chain
from sabes.beamstate import SpectralLine, linear_jones
from sabes.presets import LAB_PROFILE, PAPER_PROFILE, profile_calibration, profile_settings
from sabes.single_beam import cell_beam_input


@pytest.fixture(scope="module")
def chain():
    return build_source_chain()


@pytest.mark.parametrize("profile", (PAPER_PROFILE, LAB_PROFILE))
@pytest.mark.parametrize("arm", ("pump", "seed"))
def test_current_source_profiles_select_actual_pre_cell_component(profile, arm):
    settings, _ = profile_settings(profile)
    chain = build_source_chain(settings, profile_calibration(profile))
    result = cell_beam_input(chain, arm, .0125)
    beam = getattr(chain, arm)
    offset = 0. if arm == "pump" else chain.seed_offset_hz
    assert result.power_w == beam.power_at(offset)
    assert result.frequency.offset_hz == chain.carrier_frequency.offset_hz+offset
    assert result.reference_plane == "cell entrance"
    assert result.source_m2 == 1.
    assert result.cell_centre_radius_m == beam.mode.radius_m
    assert result.excluded_other_arm_power_w == math.fsum(
        line.power_w for line in (chain.seed if arm == "pump" else chain.pump).lines)
    if arm == "seed":
        assert result.power_w < beam.total_power_w
        assert result.excluded_power_w > 0.
        assert result.power_w+result.excluded_power_w == pytest.approx(beam.total_power_w, rel=1e-15)
    else:
        assert result.excluded_power_w == 0.


def test_explicit_sideband_selection_uses_stored_offset_and_own_power(chain):
    # The established beam lookup precision is 1 Hz. Do not turn a lookup
    # tolerance into an unintended change of the physical optical frequency.
    line = next(line for line in chain.seed.lines if line.offset_hz == 0.)
    result = cell_beam_input(chain, "seed", .0125, offset_hz=.25)
    assert result.power_w == line.power_w
    assert result.spectral_offset_hz == 0.
    assert result.frequency == chain.carrier_frequency
    assert result.power_w != chain.seed.total_power_w
    assert "seed" in result.source_label


@pytest.mark.parametrize("centre_z", (-.12, 0., .18))
def test_cell_centre_to_entrance_preserves_radius_and_signed_curvature(chain, centre_z):
    mode = replace(chain.pump.mode, waist_m=220e-6, z_m=centre_z)
    modified = replace(chain, pump=chain.pump.with_mode(mode))
    result = cell_beam_input(modified, "pump", .04)
    z = centre_z-.02
    rayleigh = math.pi*(220e-6)**2/mode.wavelength_m
    expected_radius = 220e-6*math.sqrt(1+(z/rayleigh)**2)
    expected_curvature = (z*z+rayleigh*rayleigh)/z
    assert result.radius_m == pytest.approx(expected_radius, rel=3e-15)
    assert result.curvature_radius_m == pytest.approx(expected_curvature, rel=3e-15)
    assert math.copysign(1., result.curvature_radius_m) == math.copysign(1., z)
    assert result.frequency.wavelength_m == constants.C_LIGHT/result.frequency.absolute_hz
    assert result.frequency.wavelength_m != result.gaussian_geometry_wavelength_m
    assert modified.pump.mode == mode


def test_planar_entrance_keeps_infinity_in_physics_and_exports_strict_metadata(chain):
    mode = replace(chain.pump.mode, z_m=.01)
    result = cell_beam_input(replace(chain, pump=chain.pump.with_mode(mode)), "pump", .02)
    assert math.isinf(result.curvature_radius_m)
    metadata = result.as_dict()
    assert metadata["curvature_radius_m"] is None
    assert metadata["planar_wavefront"]
    assert not metadata["cell_centre_planar_wavefront"]
    assert metadata["frequency"] == result.frequency.as_dict()
    assert "other optical drives absent" in metadata["scope"]
    assert "unmeasured" in metadata["source_reference_plane"]
    encoded = json.dumps(metadata, allow_nan=False)
    assert json.loads(encoded)["source_jones"][0]["imag"] == 0.
    with pytest.raises(FrozenInstanceError):
        result.power_w = 0.


def test_linear_polarization_rotation_and_common_phase_do_not_change_physics(chain):
    baseline = cell_beam_input(chain, "pump", .0125)
    phase = np.exp(.73j)
    jones = tuple(phase*v for v in linear_jones(71.))
    rotated = cell_beam_input(replace(chain, pump=chain.pump.with_jones(jones)), "pump", .0125)
    assert rotated.power_w == baseline.power_w
    assert rotated.radius_m == baseline.radius_m
    assert rotated.curvature_radius_m == baseline.curvature_radius_m
    assert rotated.frequency == baseline.frequency
    assert rotated.source_azimuth_rad == pytest.approx(math.radians(71.), rel=1e-14)


@pytest.mark.parametrize("jones", ((1/math.sqrt(2), 1j/math.sqrt(2)),
                                   (math.sqrt(.99), .1j)))
def test_elliptical_polarization_is_rejected_without_silent_linearization(chain, jones):
    with pytest.raises(ValueError, match="linear polarization only"):
        cell_beam_input(replace(chain, seed=chain.seed.with_jones(jones)), "seed", .0125)


@pytest.mark.parametrize("m2", (.8, 1.2, float("nan")))
def test_nonideal_or_invalid_beam_quality_cannot_become_a_pure_gaussian(chain, m2):
    beam = chain.pump.with_mode(replace(chain.pump.mode, m2=m2))
    with pytest.raises(ValueError, match=r"M\^2=1|geometry"):
        cell_beam_input(replace(chain, pump=beam), "pump", .0125)


@pytest.mark.parametrize("arm", ("probe", "conjugate", "common", "total"))
def test_generated_or_multibeam_paths_are_not_a_pre_cell_single_beam(chain, arm):
    with pytest.raises(ValueError, match="pre-cell pump or seed"):
        cell_beam_input(chain, arm, .0125)


@pytest.mark.parametrize("length", (0., -.01, float("nan"), float("inf")))
def test_cell_length_must_be_physical(chain, length):
    with pytest.raises(ValueError, match="cell_length_m"):
        cell_beam_input(chain, "pump", length)


def test_zero_power_missing_line_and_ambiguous_lines_are_rejected(chain):
    dark = replace(chain, pump=chain.pump.scaled(0.))
    with pytest.raises(ValueError, match="positive power"):
        cell_beam_input(dark, "pump", .0125)
    with pytest.raises(ValueError, match="exactly one"):
        cell_beam_input(chain, "pump", .0125, offset_hz=1e9)
    duplicate = chain.pump.with_lines(chain.pump.lines+(SpectralLine(.5, 1e-3, "nearby"),))
    with pytest.raises(ValueError, match="exactly one"):
        cell_beam_input(replace(chain, pump=duplicate), "pump", .0125)


def test_missing_structured_frequency_is_not_inferred_from_legacy_geometry(chain):
    with pytest.raises(ValueError, match="structured physical carrier"):
        cell_beam_input(replace(chain, carrier_frequency=None), "pump", .0125)


@pytest.mark.parametrize("line", (SpectralLine(0., -1.), SpectralLine(0., float("nan")),
                                  SpectralLine(float("inf"), .001)))
def test_unphysical_spectral_data_are_rejected(chain, line):
    with pytest.raises(ValueError, match="source lines"):
        cell_beam_input(replace(chain, pump=chain.pump.with_lines((line,))), "pump", .0125)


@pytest.mark.parametrize("line,transition_count", (("D1", 8), ("D2", 12)))
def test_named_frequency_references_cover_all_and_only_allowed_transitions(line, transition_count):
    from gabes import species
    from sabes.single_beam import frequency_references

    references = frequency_references(line)
    assert isinstance(references, tuple)
    assert len(references) == 6+transition_count
    assert len({reference.key for reference in references}) == len(references)
    assert all(reference.key.isascii() for reference in references)
    for isotope in ("Rb85", "Rb87"):
        selected = [reference for reference in references if reference.isotope == isotope]
        centroids = {reference.reference for reference in selected
                     if reference.ground_f is None}
        assert centroids == {"fine_centroid", "upper_ground_centroid", "lower_ground_centroid"}
        structure = species.level_structure(isotope, line)
        expected = {(fg, fe) for fg, _ in structure.ground for fe, _ in structure.excited
                    if abs(fg-fe) <= 1 and (fg != 0 or fe != 0)}
        actual = {(reference.ground_f, reference.excited_f) for reference in selected
                  if reference.reference == "hyperfine_transition"}
        assert actual == expected
    with pytest.raises(FrozenInstanceError):
        references[0].label = "changed"


@pytest.mark.parametrize("line", ("D1", "D2"))
def test_changing_reference_preserves_physical_frequency_and_sub_hz_offset(line):
    from gabes.optical_frequency import RbOpticalFrequency
    from sabes.single_beam import frequency_references

    original = RbOpticalFrequency(line, 123456789.125)
    for reference in frequency_references(line):
        detuning = reference.detuning_mhz(original)
        restored = reference.frequency(detuning)
        assert restored.offset_hz == pytest.approx(original.offset_hz, rel=0., abs=2e-6)
        assert reference.detuning_mhz(reference.frequency(-172.25)) == pytest.approx(
            -172.25, rel=0., abs=2e-12)
        assert json.loads(json.dumps(reference.as_dict()))["key"] == reference.key


def test_named_frequency_reference_rejects_cross_band_reinterpretation():
    from gabes.optical_frequency import RbOpticalFrequency
    from sabes.single_beam import frequency_references

    reference = frequency_references("D1")[0]
    with pytest.raises(ValueError, match="D line"):
        reference.detuning_mhz(RbOpticalFrequency("D2", 0.))
    with pytest.raises(ValueError, match="D1 or D2"):
        frequency_references("D3")
