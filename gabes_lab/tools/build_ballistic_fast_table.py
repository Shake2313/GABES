"""Rebuild the pure 85Rb D1 Fast rest-frame absorption coefficients from trajectories.

Run from the repository root with python -m gabes_lab.tools.build_ballistic_fast_table.
No measured spectra, frequency-axis samples or fitted parameters enter this calculation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from gabes import ballistic_spectrum as history, beam, species
from gabes._ballistic_reference_table import source_signature
from gabes.od_model import ODConditions
from gabes.schemes.sas import BALLISTIC_FAST
from gabes_lab.blocks.paths import REPO_ROOT


def _solve_temperature(temperature_c):
    conditions = ODConditions(species.RB85.label, 'D1', temperature_c, 12.5, BALLISTIC_FAST,
                              probe_waist_mm=0.53)
    optics = history._optics(conditions)
    peak = beam.peak_intensity(160e-6, optics.waist)
    intensities = history._intensity_nodes(peak, history._REFERENCE)
    rows = [history._rest_excess(optics.temperature, optics.width, optics.waist, float(intensity),
                                 history._REFERENCE) for intensity in intensities]
    rest = np.stack([row[0] for row in rows])
    refinements = np.array([(count, max_steps) for _, count, max_steps in rows], dtype=np.int32)
    nodes, weights = history._panels(optics.width, history._REFERENCE.panel_order, history._REFERENCE.wing_widths)
    return optics.width, nodes, weights, intensities, rest, refinements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO_ROOT/'gabes/data/rb85_d1_ballistic_fast_v2.npz')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    temperatures = np.arange(95., 107.01, 0.5)
    if args.workers == 1:
        rows = list(map(_solve_temperature, temperatures))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_solve_temperature, temperatures))
    widths, nodes, weights, intensities, rest, refinements = map(np.stack, zip(*rows))
    if not np.all(intensities == intensities[0]):
        raise ValueError('Shared physical incident-intensity knots required.')
    with NamedTemporaryFile(dir=args.output.parent, suffix='.npz', delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, temperatures=temperatures, widths=widths, nodes=nodes,
                            weights=weights, intensities=intensities[0], rest_excess=rest,
                            refinements=refinements, source_signature=source_signature())
        digest = sha256(temp.read_bytes()).hexdigest()
        temp.replace(args.output)
        print(f'{digest}  {args.output.name}')
    finally:
        temp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
