"""Physical Rb frequency identity, precision, mixture policy and SABES compatibility."""
from types import SimpleNamespace

import numpy as np
import pytest

from gabes import constants, magneto_multilevel, species
from gabes.optical_frequency import RbComposition, RbOpticalFrequency, reference_offset_hz


@pytest.mark.parametrize("line,shift", [("D1", 77.690e6), ("D2", 78.095e6)])
def test_isotope_shift_and_sub_hz_offsets_do_not_move_with_composition(line, shift):
    frequency = RbOpticalFrequency.from_detuning("Rb85", line, .001)
    assert frequency.offset_hz == .001
    assert frequency.detuning_hz("Rb85") == .001
    assert frequency.detuning_hz("Rb87") == pytest.approx(.001-shift, rel=0, abs=1e-9)
    # Optical float spacing exceeds the stored offset; detuning arithmetic
    # must retain it instead of subtracting two optical-frequency floats.
    assert frequency.absolute_hz == frequency.reference_hz
    assert frequency.shifted_hz(.002).offset_hz == pytest.approx(.003, abs=1e-18)
    for fraction in (0., .15, .7217, .9, 1.):
        composition = RbComposition(fraction)
        assert sum(weight for _, weight in composition.components) == 1.
        assert frequency.reference_hz == species.RB85.line(line)[1]


@pytest.mark.parametrize("isotope", ["Rb85", "Rb87"])
@pytest.mark.parametrize("line", ["D1", "D2"])
def test_every_allowed_hyperfine_transition_uses_one_absolute_frequency(isotope, line):
    manifold = species.build_manifold(species.ISOTOPES[isotope], line)
    isotope_shift = manifold.nu0-species.RB85.line(line)[1]
    for g, e, omega in zip(manifold.g_idx, manifold.e_idx, manifold.omega):
        fg = manifold.Fg[int(g)]
        fe = manifold.Fe[int(e)-len(manifold.Fg)]
        transition = dict(reference="hyperfine_transition", ground_f=fg, excited_f=fe)
        frequency = RbOpticalFrequency.from_detuning(isotope, line, 12345.25, **transition)
        expected = isotope_shift+omega/(2*np.pi)+12345.25
        assert frequency.offset_hz == pytest.approx(expected, rel=1e-15, abs=2e-6)
        assert frequency.detuning_hz(isotope, **transition) == pytest.approx(12345.25, rel=0, abs=1e-6)
        for reference in ("upper_ground_centroid", "lower_ground_centroid", "fine_centroid"):
            detuning = frequency.detuning_hz(isotope, reference=reference)
            rebuilt = RbOpticalFrequency.from_detuning(isotope, line, detuning, reference=reference)
            assert rebuilt.offset_hz == pytest.approx(frequency.offset_hz, rel=0, abs=2e-6)
        assert frequency.wavelength_m*frequency.absolute_hz == pytest.approx(constants.C_LIGHT)


@pytest.mark.parametrize("line", ["D1", "D2"])
@pytest.mark.parametrize("isotope", ["Rb85", "Rb87"])
def test_component_centroids_match_existing_atomic_reference(line, isotope):
    atom = magneto_multilevel.build_multilevel_atom(isotope, line)
    for reference, ground_f in (("upper_ground_centroid", max(f for f, _ in atom.ground)),
                                 ("lower_ground_centroid", min(f for f, _ in atom.ground))):
        frequency = RbOpticalFrequency.from_detuning(isotope, line, 1500e6, reference=reference)
        expected = magneto_multilevel.component_centroid_mhz(atom, ground_f)+1500.
        assert frequency.detuning_mhz(isotope) == pytest.approx(expected, rel=1e-15, abs=1e-12)
        assert frequency.detuning_mhz(isotope, reference=reference) == 1500.


def test_reference_transition_anchor_and_serialized_offset_are_explicit():
    frequency = RbOpticalFrequency.from_detuning(
        "Rb85", "D1", .9e9, reference="hyperfine_transition", ground_f=2, excited_f=3)
    assert reference_offset_hz("Rb85", "D1", reference="hyperfine_transition",
                                ground_f=2, excited_f=3) == pytest.approx(1921.50267275e6, rel=0, abs=1e-6)
    record = frequency.as_dict()
    assert record["reference_isotope"] == "Rb85"
    assert record["reference_kind"] == "fine_centroid"
    assert RbOpticalFrequency(record["line"], record["offset_hz"]) == frequency
    assert RbOpticalFrequency.from_absolute_hz(frequency.absolute_hz, "D1").absolute_hz == frequency.absolute_hz


def test_natural_custom_and_pure_shares_follow_existing_density_policy():
    assert RbComposition(0.).components == (("Rb87", 1.),)
    assert RbComposition(1.).components == (("Rb85", 1.),)
    assert RbComposition(.25).components == (("Rb85", .25), ("Rb87", .75))
    assert RbComposition.natural().components == tuple(
        (isotope.name, weight) for isotope, weight in species.SPECIES["Rb (natural)"])
    assert "total elemental" in RbComposition(.3).as_dict()["density_basis"]


@pytest.mark.parametrize("fraction", [-.1, 1.1, np.nan, np.inf])
def test_invalid_compositions_are_rejected(fraction):
    with pytest.raises(ValueError, match="fraction"):
        RbComposition(fraction)


@pytest.mark.parametrize("kwargs", [
    dict(isotope="Cs133", line="D1"), dict(isotope="Rb85", line="D3"),
    dict(isotope="Rb85", line="D1", reference="unknown"),
    dict(isotope="Rb87", line="D2", reference="hyperfine_transition", ground_f=1, excited_f=3),
    dict(isotope="Rb85", line="D1", reference="hyperfine_transition", ground_f=1, excited_f=3),
    dict(isotope="Rb85", line="D1", reference="hyperfine_transition"),
    dict(isotope="Rb85", line="D1", reference="upper_ground_centroid", ground_f=3),
])
def test_invalid_references_and_forbidden_transitions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RbOpticalFrequency.from_detuning(detuning_hz=0., **kwargs)


def test_sabes_corrects_carrier_origin_but_preserves_gaussian_and_fwm_inputs(monkeypatch):
    from sabes import beamline, bridge
    from sabes.instruments import Wavemeter

    result = bridge.run(solve=False)
    chain = result.chain
    expected = RbOpticalFrequency.from_detuning(
        "Rb85", "D1", result.settings.opd_ghz*1e9,
        reference="hyperfine_transition", ground_f=2, excited_f=3)
    assert chain.carrier_frequency == expected
    assert chain.pump.carrier_hz == expected.absolute_hz
    assert chain.seed.carrier_hz == expected.absolute_hz
    measured = Wavemeter().measure(chain.pump)
    assert measured.quantity("Optical frequency").value*1e12 == pytest.approx(expected.absolute_hz, rel=0, abs=.1)
    assert chain.frequency_metadata["carrier"]["wavelength_m"] == expected.wavelength_m
    assert chain.frequency_metadata["gaussian_geometry_wavelength_m"] == constants.WAVELENGTH_D1_85RB
    assert chain.pump.mode.wavelength_m != expected.wavelength_m

    # Recreate the historical carrier origin only; every model quantity that
    # feeds Gaussian optics or the FWM solve must remain bit-identical.
    monkeypatch.setattr(beamline, "RbOpticalFrequency", SimpleNamespace(
        from_detuning=lambda isotope, line, detuning_hz, **kwargs: RbOpticalFrequency(line, detuning_hz)))
    legacy = bridge.run(solve=False)
    assert result.params == legacy.params
    assert result.geometry == legacy.geometry
    assert len(chain.stages) == len(legacy.chain.stages)
    for actual, previous in zip(chain.stages, legacy.chain.stages):
        assert actual.beam.lines == previous.beam.lines
        assert actual.beam.mode == previous.beam.mode
        assert actual.beam.jones == previous.beam.jones
        assert actual.beam.carrier_hz-previous.beam.carrier_hz == pytest.approx(1921.50267275e6, rel=0, abs=.1)
