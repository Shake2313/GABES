"""Direct SAS window edits reject invalid limits before mutating navigation."""
import base64
import re
import shutil
import subprocess

import pytest

from gabes_ui.paths import UI_ASSETS


def test_sas_window_input_validation_and_recovery():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for spectrum range tests")
    source = (UI_ASSETS / "model_compare.js").read_text(encoding="utf-8")
    body = re.search(r"function applyRange\(\)\{(.*?)\n  \}", source, re.S).group(1)
    encoded = base64.b64encode(body.encode()).decode()
    script = r"""
import assert from 'node:assert/strict';
const body=Buffer.from('BODY','base64').toString('utf8');
const from={valueAsNumber:20},to={valueAsNumber:80,validity:'',reports:0,
  setCustomValidity(message){this.validity=message;},
  reportValidity(){this.reports++;}};
const state={sample_index:1},d={x:[0,50,100]},full=[0,100];
let applied=null,refreshes=0,published=0;
const edit=new Function('from','to','minWidth','format','applyWindow','d','state',
  'refresh','publish','full',body);
const apply=()=>edit(from,to,2,String,value=>{applied=value;},d,state,
  ()=>refreshes++,()=>published++,full);
for(const [a,b] of [[NaN,80],[20,NaN],[Infinity,80],[20,-Infinity],
                    [-1,80],[20,101],[101,110],[-10,-1],[80,20],[20,21]]) {
  from.valueAsNumber=a;to.valueAsNumber=b;
  apply();
  assert.equal(applied,null);
  assert.equal(refreshes,0);
  assert.equal(published,0);
  assert.ok(to.validity.length>0,`${a}, ${b} must explain its rejection`);
}
assert.equal(to.reports,10);
// A valid edit after an invalid one clears the error and commits exactly once.
from.valueAsNumber=0;to.valueAsNumber=100;
apply();
assert.equal(to.validity,'');
assert.deepEqual(applied,{window:[0,100],cursor:50});
assert.equal(refreshes,1);
assert.equal(published,1);
assert.equal(to.reports,10);
""".replace("BODY", encoded)
    result = subprocess.run([node, "--input-type=module"], input=script,
                            text=True, encoding="utf-8", capture_output=True,
                            timeout=20)
    assert result.returncode == 0, result.stderr[-4000:]


def test_od_peak_and_sas_reference_follow_native_samples():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required for spectrum readout tests")
    module = (UI_ASSETS / "model_compare.js").read_text(encoding="utf-8")
    assertions = r"""
import assert from 'node:assert/strict';
assert.deepEqual(absorptionPeaks([7,6,4,3]), []);
assert.deepEqual(absorptionPeaks([0,2,2,2,1,5,4,7]), [2,5]);
assert.deepEqual(absorptionPeaks([0,2,2,null,0,4,NaN,2]), []);
const marker=(kind,label,x_mhz)=>({kind,label,x_mhz});
const groups=[
  {id:'Rb87',markers:[marker('crossover_candidate','CO candidate',4),
                      marker('transition','a',8),marker('transition','out',12)]},
  {id:'Rb85',markers:[marker('transition','b',8)]}
];
const all=modelReferences(groups,null,[4,8]);
const rb=modelReferences(groups,'Rb87',[4,8]);
assert.equal(all.length,3);
assert.equal(rb.length,2);
const nearest=chooseReference(rb,null,6);
assert.equal(nearest.label,'a');  // equally distant candidate and allowed line
assert.equal(chooseReference(rb,nearest,4).label,'a');  // retain reference as cursor moves
assert.equal(chooseReference(all,all[2],4).group_id,'Rb85');  // coincident isotope labels
assert.equal(chooseReference(rb,all[2],4).label,'CO candidate');  // another group
assert.equal(chooseReference(rb,{...nearest,x_mhz:4},4).label,'CO candidate');
// A custom window with no reference begins empty. An existing anchor remains
// usable there only if its full identity belongs to the current solved scan.
assert.equal(chooseReference(modelReferences(groups,null,[9,11]),null,10),undefined);
assert.deepEqual(modelReferences(groups,null,[9,11],nearest,[3,13]),[nearest]);
assert.equal(modelReferences(groups,null,[9,11],nearest,[9,13]).length,0);
assert.equal(modelReferences(groups,null,[9,11],{...nearest,label:'forged'},[3,13]).length,0);
assert.equal(modelReferences(groups,'Rb85',[9,11],nearest,[3,13]).length,0);
const shifted=modelReferences(groups,null,[11,13],nearest,[3,13]);
assert.deepEqual(shifted.map(m=>m.label),['a','out']);
assert.equal(chooseReference(shifted,nearest,12).label,'a');
"""
    result = subprocess.run([node, "--input-type=module"], input=module + assertions,
                            text=True, encoding="utf-8", capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr[-4000:]
