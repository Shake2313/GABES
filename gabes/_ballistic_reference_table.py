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
_MULTIWAIST_TABLE = Path(__file__).with_name('data') / 'rb85_d1_ballistic_multiwaist_v1.npz'
_MULTIWAIST_TABLE_SHA256 = 'd764bed116e2d15447e9843bd3cece91d790af48df03b39c7ef6b74573b50023'
_MULTIWAIST_W_REF_M = 0.53e-3
_MULTIWAIST_MIN_MM = 0.265
_MULTIWAIST_MAX_MM = 2.12
# Fluence cap equivalent to the v2 validated range (160 uW at w_ref=0.53 mm),
# matching the exact 160 uW cap at the 0.53 mm seam with an upper guard knot.
_MULTIWAIST_V2_MAX_POWER_UW = 160.0
_MULTIWAIST_MAX_PHI_W_M2 = (2.0 * _MULTIWAIST_V2_MAX_POWER_UW * 1e-6) / (np.pi * _MULTIWAIST_W_REF_M ** 2)


def multiwaist_max_power_uw(waist_mm):
    """Maximum eligible probe power [uW] at ``waist_mm`` (v2-equivalent fluence cap: 160 uW * w/0.53)."""
    return float(_MULTIWAIST_V2_MAX_POWER_UW * (float(waist_mm) / (_MULTIWAIST_W_REF_M * 1e3)))


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
            recorded, live = str(archive['source_signature']), source_signature()
            # The isotope-aware solver regenerated all 25 x 18 v2 rows, both
            # panel arrays and all refinement counts byte for byte. Retain the
            # published table checksum and reject any other source change.
            verified_refactor = (recorded, live) == (
                '3259ae5104dff3e33a081b5d8b844930e482a2fa6480155552524fca4fbb157b',
                '8cd7ddbe96a64edba7f843258cbcf4598a124ab1eccff3e3a39f270dbc01a6c0',
            )
            if recorded != live and not verified_refactor:
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


@lru_cache(maxsize=1)
def _load_multiwaist_table():
    try:
        content = _MULTIWAIST_TABLE.read_bytes()
        if sha256(content).hexdigest() != _MULTIWAIST_TABLE_SHA256:
            return None, 'Multi-waist trajectory table checksum mismatch.'
        with np.load(BytesIO(content), allow_pickle=False) as archive:
            recorded, live = str(archive['source_signature']), source_signature()
            if recorded != live:
                return None, 'The direct trajectory model differs from the multi-waist table.'
            data = {k: archive[k] for k in (
                'temperatures', 'waists_mm', 'fluence_intensities', 'widths',
                'nodes', 'weights', 'rest_excess', 'refinements',
            )}
        return data, None
    except (OSError, TypeError, ValueError, KeyError, BadZipFile) as error:
        return None, f'Multi-waist trajectory table unavailable ({type(error).__name__}).'


def _lagrange_weights(axis, value):
    right = int(np.searchsorted(axis, value))
    if right < len(axis) and abs(axis[right] - value) <= 8 * np.spacing(max(abs(axis[right]), 1.0)):
        return np.array([right]), [1.0], True
    if right > 0 and abs(axis[right - 1] - value) <= 8 * np.spacing(max(abs(axis[right - 1]), 1.0)):
        return np.array([right - 1]), [1.0], True
    first = max(0, min(right - 2, len(axis) - 4))
    indices = np.arange(first, first + 4)
    knots = axis[indices]
    factors = []
    for i, knot in enumerate(knots):
        other = np.delete(knots, i)
        factors.append(float(np.prod((value - other) / (knot - other))))
    return indices, factors, False


def lookup_multiwaist(optics, peak_intensity, nodes, weights):
    """Interpolate rest-frame excess in (T, log w) on exact fluence intensities ``I = Phi * w_ref / w``.

    Returns ``(intensities, rest_excess, provenance)`` or ``(None, None, provenance)`` on fallback.
    """
    data, reason = _load_multiwaist_table()
    if reason:
        return None, None, {'rest_table_fallback_reason': reason}
    t = optics.temperature - 273.15
    w_mm = optics.waist * 1e3
    t_axis = data['temperatures']
    w_axis = data['waists_mm']
    phi_axis = data['fluence_intensities']
    t_tol = 8 * np.spacing(max(abs(t), 1.0))
    w_lo_tol = 8 * np.spacing(w_axis[0])
    w_hi_tol = 8 * np.spacing(w_axis[-1])
    if not (t_axis[0] - t_tol <= t <= t_axis[-1] + t_tol
            and w_axis[0] - w_lo_tol <= w_mm <= w_axis[-1] + w_hi_tol):
        return None, None, {'rest_table_fallback_reason': 'Temperature or radius outside the multi-waist table.'}

    i_grid = phi_axis * (_MULTIWAIST_W_REF_M / optics.waist)
    max_i = min(float(i_grid[-2]), _MULTIWAIST_MAX_PHI_W_M2 * (_MULTIWAIST_W_REF_M / optics.waist))
    if peak_intensity > max_i * (1.0 + 8 * np.finfo(float).eps):
        return None, None, {'rest_table_fallback_reason': 'Intensity exceeds the multi-waist validated fluence cap.'}
    hi = max(2, min(len(i_grid), int(np.searchsorted(i_grid, peak_intensity, side='right')) + 2))
    i_slice = i_grid[:hi]

    # Verify panel alignment against live optical width across the enclosing temperature interval.
    widths = data['widths']
    t_clamped = float(np.clip(t, t_axis[0], t_axis[-1]))
    t_right = int(np.searchsorted(t_axis, t_clamped))
    lower = max(0, min(t_right - 1, len(t_axis) - 2))
    fraction = (optics.width - widths[lower]) / (widths[lower + 1] - widths[lower])
    for key, live in (('nodes', nodes), ('weights', weights)):
        a, b = data[key][lower:lower + 2]
        if (a.shape != live.shape or not -1e-10 <= fraction <= 1.0 + 1e-10
                or not np.allclose(live, a + fraction * (b - a), rtol=1e-11, atol=1e-4)):
            return None, None, {'rest_table_fallback_reason': 'Live rest-frame quadrature differs from the multi-waist table.'}

    t_idx, t_fac, t_exact = _lagrange_weights(t_axis, t_clamped)
    w_clamped = float(np.clip(w_mm, w_axis[0], w_axis[-1]))
    w_idx, w_fac, w_exact = _lagrange_weights(np.log(w_axis), np.log(w_clamped))
    rows = data['rest_excess']
    # Separable interpolation: cubic Lagrange in T, then cubic Lagrange in log(w) at fixed fluence.
    t_interp = sum(tf * rows[ti, :, :hi] for ti, tf in zip(t_idx, t_fac))
    result = sum(wf * t_interp[wi] for wi, wf in zip(w_idx, w_fac))
    ref_sub = data['refinements'][np.ix_(t_idx, w_idx, np.arange(hi))]
    return i_slice, result, {
        'rest_method': 'tabulated multi-waist fluence reference trajectories',
        'rest_table_sha256': _MULTIWAIST_TABLE_SHA256,
        'rest_temperature_nodes_c': t_axis[t_idx].tolist(),
        'rest_temperature_interpolation': 'exact knot' if t_exact else 'local cubic',
        'rest_waist_nodes_mm': w_axis[w_idx].tolist(),
        'rest_waist_interpolation': 'exact knot' if w_exact else 'cubic log-waist',
        'rest_table_refined_rows_at_knots': ref_sub[:, :, :, 0].sum(axis=(1, 2)).tolist(),
        'rest_table_max_steps_at_knots': ref_sub[:, :, :, 1].max(axis=(1, 2)).tolist(),
    }
