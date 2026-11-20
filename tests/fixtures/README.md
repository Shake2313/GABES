# Test fixtures

⟦AI⟧ Copies of archive-side records that main tests still need after the repository split
(docs/repo_restructure/README.md, S2). User decision 2026-11-09: keep the magneto frozen-reference
regression tests in main by copying their reference code. The sources remain unchanged in the
archive repository at tag `pre-split-2026-11-09`; the record folders there are the authority.

Rules: copy, do not edit, unless the table names the adaptation; `.gitattributes` keeps bytes.
Record-generation entry points in the copied `validation.py` modules still name archive paths and are
not used by the tests.

| Fixture | Source at `pre-split-2026-11-09` | Source SHA-256 | Copy | Used for |
|---|---|---|---|---|
| `atomic_qrt.py` | `analysis/grand_challenge/reference/atomic_qrt.py` | `517f7abbb9b45d40…` | byte copy | tests/test_psr_spectrum.py: independent QRT ordered-spectrum reference |
| `fwm_experiment_20261022/gain_profile.json` | `analysis/fwm_experiment_20261022/gain_profile.json` | `837d13630d1ff2eb…` | byte copy | tests/test_fwm_experiment_gain.py: reference model and fitted arm coefficients |
| `magneto_speed_20261030/before_magneto_multilevel.py` | `analysis/magneto_speed_20261030/before_magneto_multilevel.py` | `c4c647b0792ef9dd…` | byte copy | frozen pre-speed-patch backend |
| `magneto_speed_20261030/before_magneto_unified.py` | `analysis/magneto_speed_20261030/before_magneto_unified.py` | `8df979dd13c06001…` | byte copy | frozen pre-speed-patch scheme |
| `magneto_speed_20261030/validation.py` | `analysis/magneto_speed_20261030/validation.py` | `a1365d698de6cca1…` | byte copy | loader and comparison helpers |
| `magneto_optical_assembly_20261030/before_magneto_multilevel.py` | `analysis/magneto_optical_assembly_20261030/before_magneto_multilevel.py` | `bf50907543b272bf…` | byte copy | frozen pre-assembly backend |
| `magneto_optical_assembly_20261030/before_magneto_unified.py` | `analysis/magneto_optical_assembly_20261030/before_magneto_unified.py` | `234092eeaeec29e9…` | byte copy | frozen pre-assembly scheme |
| `magneto_optical_assembly_20261030/validation.py` | `analysis/magneto_optical_assembly_20261030/validation.py` | `4df60897e59fe5c9…` | one line: imports fixtures.magneto_speed_20261030.validation instead of analysis.magneto_speed_20261030.validation | loader and comparison helpers |

Magneto tests using these: `tests/test_magneto_fast_solver.py`, `tests/test_magneto_optical_assembly.py`,
`tests/test_magneto_reference_isolation.py`. Import them as `fixtures.<name>`: pytest puts `tests/` on
`sys.path` for the test modules in that folder.

## AutoOD 0.2.2 spectra (OD-fit port, M0)

⟦AI⟧ `autood_baseline_0_2_2/spectra.npz` freezes AutoOD output for the D20 port
(`docs/ui_redesign/od_sas_experimental_csv_port.md`). It is generated, not copied:
`python -m gabes_lab.tools.autood_baseline_fixture` runs the AutoOD source at tag
`baseline-0.2.2-src` (commit `1bd66154c36a…` in the sibling `AutoOD` repository) on its pinned
GABES snapshot `8846193ffcef…`, with Numba disabled as in the shipped executable. Theory adapter
fingerprint `7063e3471c71…`. SHA-256 of the committed file `25393b1006c3e712…`; a rerun gives
equal arrays but different zip bytes, so replace the hash in `tests/test_od_model.py` only on purpose.

| Case | Settings | Probe powers |
|---|---|---|
| `natural_d2_25c_75mm` | natural Rb, D2, 25 °C, 75 mm, 2 mm 1/e² diameter | 0, 100 µW |
| `rb87_d1_90c_12p5mm` | ⁸⁵Rb 0 %, D1, 90 °C, 12.5 mm, 2 mm | 0, 100 µW |
| `rb85_d2_90c_12p5mm` | ⁸⁵Rb 100 %, D2, 90 °C, 12.5 mm, 2 mm | 0, 1 mW |

Arrays per case: `scan_rad_s` (from the ⁸⁵Rb centroid), and per power `transmission`,
`optical_depth`, `od0`. A JSON `metadata` string lists settings, model IDs and the transit rate.
Used by `tests/test_od_model.py`; the finite-power spectra serve the later rate-model parity step.
