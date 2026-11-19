"""Doppler wavevector override changes widths, never atomic normalization."""
import numpy as np
import pytest

from gabes import linear_hyperfine, species


@pytest.mark.parametrize("isotope", [species.RB85, species.RB87])
@pytest.mark.parametrize("line", ["D1", "D2"])
def test_default_centroid_wavevector_retains_exact_arithmetic(isotope, line):
    manifold=species.build_manifold(isotope,line)
    axis=np.linspace(-7e9,7e9,73)*2*np.pi
    args=(axis,manifold,1e16,350.,manifold.gamma+2e5)
    default=linear_hyperfine.species_complex_voigt(*args)
    explicit=linear_hyperfine.species_complex_voigt(*args,doppler_wavevector_per_m=manifold.k_vec)
    for actual,expected in zip(explicit,default):
        np.testing.assert_array_equal(actual,expected)


def test_array_wavevectors_change_only_gaussian_width_not_line_areas_or_chi_scale():
    manifold=species.build_manifold(species.RB85,"D2")
    axis=np.array([[-1600.,-1450.,-1300.],[1600.,1750.,1900.]])*species.MHZ
    wavevector=manifold.k_vec*np.array([[.7],[1.4]])
    actual=linear_hyperfine.species_complex_voigt(
        axis,manifold,1e16,320.,manifold.gamma+2e5,doppler_wavevector_per_m=wavevector)
    for row,factor in enumerate((.7,1.4)):
        expected=linear_hyperfine.species_complex_voigt(
            axis[row],manifold,1e16,320.*factor**2,manifold.gamma+2e5)
        for values,reference in zip(actual,expected):
            np.testing.assert_allclose(values[row],reference,rtol=3e-14,atol=0)


@pytest.mark.parametrize("wavevector", [0.,-1.,np.nan,np.inf,[1.,np.nan,2.]])
def test_doppler_wavevector_must_be_finite_and_positive(wavevector):
    manifold=species.build_manifold(species.RB87,"D1")
    with pytest.raises(ValueError,match="finite and positive"):
        linear_hyperfine.species_complex_voigt(
            np.zeros(3),manifold,1e16,300.,manifold.gamma,doppler_wavevector_per_m=wavevector)


def test_wavevector_cannot_expand_or_mismatch_the_scan_shape():
    manifold=species.build_manifold(species.RB87,"D1")
    for wavevector in ([1.,2.],np.ones((3,1))):
        with pytest.raises(ValueError,match="scan shape"):
            linear_hyperfine.species_complex_voigt(
                np.zeros(3),manifold,1e16,300.,manifold.gamma,doppler_wavevector_per_m=wavevector)
