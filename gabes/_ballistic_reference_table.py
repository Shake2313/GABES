"""Rest-frame 85Rb D1 excess absorption from the reference trajectory solver.

Only the nonlinear per-atom cross-section correction is tabulated. The Maxwell
weights, analytic weak absorption, density, propagation and readout use the
requested physical conditions, even between tabulated temperatures.
"""
from functools import lru_cache
from hashlib import sha256
import inspect
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


_TABLE = Path(__file__).with_name('data') / 'rb85_d1_ballistic_fast_v2.npz'
_TABLE_SHA256 = 'cef73b4dae5f3845f9e234fd14eb82b5f1777d71098662fa7ec7bf4455bce59d'


@lru_cache(maxsize=1)
def source_signature():
    """SHA-256 of the coefficient solver and its effective physical inputs.

    The table reader, interpolation, and subsequent Maxwell convolution and
    propagation do not generate these coefficients. They can change independently.
    """
    from . import _ballistic_kernel, ballistic_pumping, ballistic_spectrum, beam, species
    from .od_model import ODConditions
    from .schemes.sas import BALLISTIC_FAST

    history = ballistic_spectrum
    digest = sha256(b"GABES Rb85 D1 ballistic rest-frame signature v2\0")
    for source in (_ballistic_kernel, ballistic_pumping, history._optics, history._atom,
                   history._path_grid, history._panels, history._rest_excess, history._intensity_nodes):
        # inspect reads Python source in universal-newline mode, including for
        # lru_cache-decorated functions. Do not use checkout paths or Git HEAD.
        digest.update(inspect.getsource(source).replace("\r\n", "\n").encode("utf-8"))
        digest.update(b"\0")

    def add(values):
        # Ten significant digits tolerate libm/BLAS roundoff across platforms.
        # The trajectory and panel source above still gets a byte-exact check.
        array = np.asarray(values)
        digest.update(str(array.shape).encode("ascii"))
        for value in array.flat:
            digest.update(format(float(value) if value else 0., ".10g").encode("ascii"))
            digest.update(b"\0")

    grid = history._path_grid(history._REFERENCE)
    for values in ballistic_pumping._geometry(grid, ()):
        add(values)
    for temp_c in (95., 101., 107.):
        conditions = ODConditions(species.RB85.label, "D1", temp_c, 12.5,
                                  BALLISTIC_FAST, probe_waist_mm=0.53)
        optics = history._optics(conditions)
        atom = history._atom(optics.temperature, optics.width)
        add([optics.temperature, optics.waist, optics.width, atom.manifold.k_vec,
             atom.photon_energy_j, atom.speed_sigma])
        nodes, weights = history._panels(optics.width, history._REFERENCE.panel_order,
                                         history._REFERENCE.wing_widths)
        add(nodes)
        add(weights)
        if temp_c == 95.:
            add(history._intensity_nodes(beam.peak_intensity(160e-6, optics.waist), history._REFERENCE))
            add(atom.thermal)
            add(atom._free)
            for values in (atom._probe._ground, atom._probe._excited, atom._probe._strengths,
                           atom._probe._frequencies, atom._probe._rabi_scale):
                add(values)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_table():
    try:
        content = _TABLE.read_bytes()
        if sha256(content).hexdigest() != _TABLE_SHA256:
            return None, 'Trajectory table checksum mismatch.'
        with np.load(BytesIO(content), allow_pickle=False) as archive:
            if str(archive['source_signature']) != source_signature():
                return None, 'The direct trajectory model differs from the tabulated model.'
            data = {k: archive[k] for k in ('temperatures', 'intensities', 'widths', 'nodes',
                                             'weights', 'rest_excess', 'refinements')}
        return data, None
    except (OSError, TypeError, ValueError, KeyError, BadZipFile) as error:
        return None, f'Trajectory table unavailable ({type(error).__name__}).'


def lookup(optics, intensities, nodes, weights):
    """Return tabulated or interpolated reference rows, or a reason to solve live."""
    data, reason = _load_table()
    if reason:
        return None, {'rest_table_fallback_reason': reason}
    t = optics.temperature - 273.15
    axis = data['temperatures']
    n = len(intensities)
    radius_tolerance = 8*np.spacing(0.53)*1e-3 + np.spacing(optics.waist)
    if (not axis[0] <= t <= axis[-1] or abs(optics.waist - 0.53 * 1e-3) > radius_tolerance
            or n > len(data['intensities']) or not np.allclose(
                intensities, data['intensities'][:n], rtol=8*np.finfo(float).eps, atol=0)):
        return None, {'rest_table_fallback_reason': 'Temperature, radius or intensity outside the table.'}

    right = int(np.searchsorted(axis, t))
    if axis[right] == t:
        indices = np.array([right])
        factors = [1.]
    else:
        first = max(0, min(right-2, len(axis)-4))
        indices = np.arange(first, first+4)
        knots = axis[indices]
        # Four local Lagrange weights on physical temperature. Each column
        # follows the same Gauss node on the same hyperfine integration panel.
        factors = []
        for i, knot in enumerate(knots):
            other = np.delete(knots, i)
            factors.append(np.prod((t-other)/(knot-other)))
    # Throughout the certified band all panel boundaries are affine in HWHM.
    # Reconstruct the live node positions and quadrature weights in the current
    # width to detect any change of panel topology or optical broadening.
    widths = data['widths']
    lower = max(0, min(right-1, len(axis)-2))
    fraction = (optics.width-widths[lower]) / (widths[lower+1]-widths[lower])
    for key, live in (('nodes', nodes), ('weights', weights)):
        a, b = data[key][lower:lower+2]
        if (a.shape != live.shape or not -1e-10 <= fraction <= 1+1e-10
                or not np.allclose(live, a + fraction*(b-a), rtol=1e-11, atol=1e-4)):
            return None, {'rest_table_fallback_reason': 'Live rest-frame quadrature differs from the table.'}
    rows = data['rest_excess']
    if len(indices) == 1:
        result = rows[right, :n]
    else:
        result = sum(w * rows[k, :n] for k, w in zip(indices, factors))
    refinements = data['refinements'][indices, :n]
    return result, {
        'rest_method': 'tabulated reference trajectories',
        'rest_table_sha256': _TABLE_SHA256,
        'rest_temperature_nodes_c': axis[indices].tolist(),
        'rest_temperature_interpolation': 'exact knot' if len(indices) == 1 else 'local cubic',
        'rest_table_refined_rows_at_knots': refinements[:, :, 0].sum(axis=1).tolist(),
        'rest_table_max_steps_at_knots': refinements[:, :, 1].max(axis=1).tolist(),
    }
