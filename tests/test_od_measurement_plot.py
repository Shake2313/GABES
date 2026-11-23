"""Gap-aware OD inspection preserves the native point while navigating the overview."""
import json
import shutil
import subprocess

import pytest

from gabes_ui import od_measurement_plot as plot
from gabes_ui.component_runtime import spectrum_component_js


def test_reversed_display_axis_is_sorted_for_inspection_without_modifying_source(monkeypatch):
    calls = []
    monkeypatch.setattr(plot, "_COMPONENT", lambda **kwargs: calls.append(kwargs["data"]))
    measured = {"detuning": [3.0, 2.8, -2.5], "transmission": [0.91, 0.3, 0.89]}
    assert plot.render_measurement_plot({
        "theory_trace": {"detuning": [-3, 0, 3], "transmission": [1, 0, 1],
                        "od0": [0, 1200, 0], "optical_depth": [0, 900, 0], "n_eff": 1.2e9},
        "comparison_trace": measured, "coverage_ghz": [[-2.5, -2.5], [2.8, 3.0]],
        "source_fingerprint_sha256": "example", "calibration_mode": "relative_extrema",
    })
    assert calls[0]["measuredX"] == [-2.5, 2.8, 3.0]
    assert calls[0]["measuredY"] == [0.89, 0.3, 0.91]
    assert calls[0]["depth"][1] == 900 and calls[0]["od0"][1] == 1200
    assert calls[0]["transmission"][1] == 0 and calls[0]["nEff"] == 1.2e9
    assert not calls[0]["absolute"]
    assert measured["detuning"] == [3.0, 2.8, -2.5]


def test_inspector_keeps_window_and_native_samples_across_overview_history(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the browser spectrum renderer")
    module = tmp_path / "od-inspector.mjs"
    module.write_text(spectrum_component_js("od_measurement_plot.js") + """
const x = Array.from({length: 202}, (_, i) =>
  i < 101 ? -3 + i/200 : 2.5 + (i-101)/200);
const y = x.map(v => v === 2.8 ? 0.2 : 1);
const coverage = [[-3,-2.5],[2.5,3]];
const reduced = measurementPath(x,y,coverage,[-3,3],4);
const curve = {...reduced,sampleX:x,sampleY:y,coverage};
const detail = [2.5,3], state = {position:0, views:[[-4,4]],detail,index:161};
chooseOverview(state,detail);
const nav = [];
for (const pos of [0,1]) {state.position=pos;nav.push(state.views[pos]);}
chooseOverview(state,[-4,4]);
console.log(JSON.stringify({nav, selected:x[state.index],detail:state.detail,
  peak:curveSample(curve,x,2.8),gap:curveSample(curve,x,0),
  breakCount:reduced.y.filter(v=>v===null).length,hasDip:reduced.y.includes(0.2),
  missingDepth:opticalDepth(0), validDepth:opticalDepth(0.2)}));
""", encoding="utf-8")
    result = subprocess.run([node, str(module)], text=True, capture_output=True, check=True)
    data = json.loads(result.stdout)
    assert data["nav"] == [[-4, 4], [2.5, 3]]
    assert data["selected"] == pytest.approx(2.8)
    assert data["detail"] == [2.5, 3]
    assert data["peak"] == {"x": 2.8, "y": 0.2}
    assert data["gap"] == {"x": None, "y": None}
    assert data["breakCount"] == 1 and data["hasDip"]
    assert data["missingDepth"] is None
    assert data["validDepth"] == pytest.approx(1.6094379124341003)
