"""Independent cylinder eigenvalues, rate provenance and public routing."""

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from gabes import buffer_relaxation
from gabes.schemes.magneto_unified import BUFFER, UnifiedMagnetoScheme


def _finite_volume_radial_eigenvalue(radius, cells):
    step = radius / cells
    faces = np.linspace(0.0, radius, cells + 1)
    centers = (faces[:-1] + faces[1:]) / 2
    # Cell-center finite volumes, reflecting at r=0 and zero polarization
    # at the outer half-step. Rescale by sqrt(volume) for a symmetric matrix.
    diagonal = (faces[:-1] + faces[1:]) / (centers * step ** 2)
    diagonal[-1] += radius / (centers[-1] * step ** 2)
    off_diagonal = -faces[1:-1] / (np.sqrt(centers[:-1] * centers[1:]) * step ** 2)
    return eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(0, 0), eigvals_only=True,
    )[0]


def test_cylinder_matches_independent_radial_and_axial_discretizations():
    D = 1.8e-3
    R = 0.005
    L = 0.018
    radial, axial = buffer_relaxation.quenching_cylinder_mode(D, R, L)
    errors = []
    for n in (12, 24, 48):
        radial_fd = D * _finite_volume_radial_eigenvalue(R, n)
        step = L / (n + 1)
        axial_fd = 4 * D / step ** 2 * np.sin(np.pi / (2 * (n + 1))) ** 2
        errors.append(max(abs(radial_fd / radial - 1), abs(axial_fd / axial - 1)))
    assert errors[2] < errors[1] < errors[0] < .05
    radial_long, axial_long = buffer_relaxation.quenching_cylinder_mode(D, R, L * 1000)
    assert radial_long == radial
    assert axial_long == pytest.approx(axial / 1e6)
    radial_wide, axial_wide = buffer_relaxation.quenching_cylinder_mode(D, R * 1000, L)
    assert radial_wide == pytest.approx(radial / 1e6)
    assert axial_wide == axial


def _budget(pressure=20., radius=.007, beam=.001, wall="quenching", gas="Ne"):
    return buffer_relaxation.cylinder_budget(
        "Rb87", gas, pressure, 297.15, beam, radius, .02, wall=wall)


def test_budget_uses_opposite_pressure_scaling_and_uncertainty_sources():
    p20 = _budget()
    p40 = _budget(40.)
    wall20 = p20["wall_relaxation"]
    spin20 = p20["spin_destruction"]
    assert p40["wall_relaxation"]["rate_s_inverse"] == pytest.approx(wall20["rate_s_inverse"] / 2)
    assert p40["spin_destruction"]["rate_s_inverse"] == pytest.approx(spin20["rate_s_inverse"] * 2)
    assert p20["total_s_inverse"] == pytest.approx(wall20["rate_s_inverse"] + spin20["rate_s_inverse"])
    assert p20["uncertainty_s_inverse"] == pytest.approx(
        wall20["uncertainty_s_inverse"] + spin20["uncertainty_s_inverse"])
    assert p20["relative_uncertainty"] > .2
    assert any("exceeds 20%" in warning for warning in p20["warnings"])
    assert len(p20["diffusion"]["sources"]) == 3
    wide_beam = _budget(beam=.002)
    assert wide_beam["total_s_inverse"] == p20["total_s_inverse"]
    assert wide_beam["diffusion"]["beam_crossing_time_s"] == pytest.approx(
        4 * p20["diffusion"]["beam_crossing_time_s"])


@pytest.mark.parametrize("changes", [
    {"pressure": 0.}, {"pressure": .005}, {"radius": 0.}, {"beam": .010},
    {"wall": "coated"}, {"gas": "Ar"},
])
def test_unsupported_or_incomplete_budget_has_no_derived_rate(changes):
    invalid = _budget(**changes)
    assert invalid["status"] == "insufficient calibration"
    assert invalid["reason"]
    assert "total_s_inverse" not in invalid


def test_public_geometry_reference_keeps_manual_override_and_invalidates_length():
    scheme = UnifiedMagnetoScheme()
    params = dict(scheme.defaults(), cell_type=BUFFER, doppler="off", scan_points=3,
                  ground_relaxation_hz=123., ne_pressure_torr=20., buffer_radius_mm=7.,
                  buffer_beam_radius_mm=1., cell_mm=20., buffer_wall="Fully quenching")
    assert "cell_mm" not in scheme.solve_param_keys(params)
    raw_manual = scheme.compute(params)
    assert raw_manual["conditions"]["ground_relaxation_hz"] == 123.
    params["buffer_ground_model"] = buffer_relaxation.QUENCHING_CYLINDER
    assert "cell_mm" in scheme.solve_param_keys(params)
    raw = scheme.compute(params)
    derived = raw["buffer_relaxation"]
    assert raw["conditions"]["ground_relaxation_hz"] == derived["total_s_inverse"]
    assert derived["applied_rate_source"].startswith("calculated")
    assert not np.array_equal(raw["backend"]["chi_probe"], raw_manual["backend"]["chi_probe"])
    view = scheme.headless_observables(raw, params)["magneto_readout"]
    assert view["buffer_relaxation"] == derived

    insufficient = scheme.compute(dict(params, buffer_wall="Unspecified"))
    fallback = insufficient["buffer_relaxation"]
    assert fallback["status"] == "insufficient calibration"
    assert "total_s_inverse" not in fallback
    assert fallback["applied_rate_source"] == "configured manual reset rate"
    assert fallback["applied_rate_s_inverse"] == 123.
