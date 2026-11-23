// SAS-specific controls over the shared sampled-spectrum renderer.
const sessions = new Map();

// A sampled maximum needs lower neighbours on both sides. Use one native
// sample near the middle of a plateau and leave clipped or missing edges out.
function absorptionPeaks(values) {
  const peaks=[];
  for (let first=1; first<values.length-1; first++) {
    const value=values[first], left=values[first-1];
    if (!Number.isFinite(value) || !Number.isFinite(left) || value<=left) continue;
    let last=first;
    while (values[last+1]===value) last++;
    const right=values[last+1];
    if (Number.isFinite(right) && right<value) peaks.push(Math.floor((first+last)/2));
    first=last;
  }
  return peaks;
}

function sameReference(marker, saved) {
  return saved && marker.group_id===saved.group_id && marker.kind===saved.kind
    && marker.label===saved.label && marker.x_mhz===saved.x_mhz;
}

function modelReferences(groups, groupId, [low,high], saved, [first,last]=[low,high]) {
  return groups.filter(g=>!groupId || g.id===groupId).flatMap(g=>
    g.markers.filter(m=>Number.isFinite(m.x_mhz) && m.x_mhz>=first && m.x_mhz<=last
      && ['transition','crossover_candidate'].includes(m.kind) && typeof m.label==='string' && m.label)
      .map(m=>({group_id:g.id,kind:m.kind,label:m.label,x_mhz:m.x_mhz})))
    .filter(m=>(m.x_mhz>=low && m.x_mhz<=high) || (!groupId && sameReference(m,saved)));
}

function chooseReference(markers, saved, x) {
  const retained=markers.find(m=>sameReference(m,saved));
  if (retained) return retained;
  let nearest, distance=Infinity;
  for (const m of markers) {
    const delta=Math.abs(x-m.x_mhz);
    if (delta<distance || (delta===distance && m.kind==='transition' && nearest?.kind!=='transition')) {
      nearest=m; distance=delta;
    }
  }
  return nearest;
}

function signedReadout(value, digits) {
  if (!Number.isFinite(value)) return '—';
  return (value>0?'+':'')+format(value,digits);
}

export default function mount({parentElement, data, key, setStateValue}) {
  const root = parentElement.querySelector('.mc');
  if (!root) return;
  const d=data.dataset, previous=sessions.get(key);
  // Server acknowledgement of a prior cursor position can arrive after another
  // local gesture. Preserve the newest choice while this dataset is mounted.
  const revision=data.navigation_revision || 0;
  const sameNavigation=previous?.context===data.context && (previous.navigationRevision||0)===revision;
  if (previous?.root === root && sameNavigation) return;
  previous?.cleanup();
  const state = {...(sameNavigation ? previous.state : data.selection)};
  const base = d.series.find(c=>c.power_mw===0);
  const transmissionRange=valueRange(d.series,'transmission',0,d.x.length-1,1);
  const full=[d.x[0],d.x[d.x.length-1]];
  root.innerHTML = `
    <div class="mc-tools">
      <fieldset class="mc-group-field"><legend>Doppler group</legend><div class="mc-groups"></div></fieldset>
    </div>
    <div class="mc-legend" aria-label="Curve legend">
      <span><i class="off"></i>Pump off · 0 mW</span><span class="mc-on-legend"><i></i><span></span></span>
    </div><p class="mc-frequency-reference mc-muted"></p>
    <div class="mc-layout"><div class="mc-graphs"><div class="mc-overview"></div>
      <div class="mc-local"><div class="mc-zoom"></div><div class="mc-od"></div></div></div>
      <aside class="mc-reader mc-od-reader" aria-label="Spectrum readout">
        <h3></h3><p class="mc-loc"></p>
        <label class="mc-pick"><span></span><select></select></label>
        <dl class="mc-readings mc-sas-readings" hidden>
          <dt>Detuning from reference</dt><dd class="mc-offset"></dd>
          <dt>Pump on − off, ΔT</dt><dd class="mc-delta"></dd>
        </dl>
        <div class="mc-cursor"><label>Sample frequency (MHz)<input type="number" step="any" inputmode="decimal"></label>
        <button type="button" class="prev" aria-label="Previous sample">←</button>
        <button type="button" class="next" aria-label="Next sample">→</button></div>
        <button type="button" class="mc-go-ref" hidden>Go to reference</button>
        <dl class="mc-readings mc-od-readings" hidden></dl>
        <details class="mc-model"><summary>Model details</summary>
          <table class="mc-values"><caption></caption><thead><tr>
            <th scope="col">Pump</th><th scope="col">Transmission T</th><th scope="col">OD τ</th>
          </tr></thead><tbody></tbody></table>
          <p class="mc-feature"></p><p class="mc-probe" hidden></p><p class="mc-basis"></p>
          <p class="mc-marker-note"></p><p class="mc-condition-note"></p>
          <button type="button" class="mc-export">Download comparison JSON</button>
        </details><output class="mc-sr" aria-live="polite"></output>
      </aside></div>`;
  const q=s=>root.querySelector(s), groups=q('.mc-groups');
  q('.mc-tools').hidden=data.show_groups===false;
  const freq=q('.mc-cursor input');
  q('.mc-frequency-reference').textContent='Frequency reference: '+d.axis.reference+'.';
  const groupFor = () => d.groups.find(g=>g.id===state.group_id);
  const curveFor = () => d.series.find(c=>c.power_mw===d.default_power_mw) || base;
  const active=curveFor(), driven=active.power_mw*1e-3>0;
  const peaks=driven ? [] : absorptionPeaks(active.optical_depth);
  const picker=q('.mc-pick select');
  let pickerWindow='', references=[];
  q('.mc-od-reader h3').textContent=driven ? 'SAS detuning' : 'Optical depth';
  q('.mc-pick span').textContent=driven ? 'Model reference' : 'Absorption peak';
  q('.mc-sas-readings').hidden=!driven;
  q('.mc-go-ref').hidden=!driven;
  q('.mc-od-readings').hidden=driven;
  const onLabel='Pump '+format(active.power_mw,8)+' mW';
  q('.mc-legend .off').classList.toggle('mc-off-only',active===base);
  q('.mc-on-legend').hidden=active===base;
  q('.mc-on-legend span').textContent=onLabel;
  const bounds = () => state.window_mhz || groupFor()?.window_mhz || full;
  const visibleMarkers = () => {
    const group=groupFor();
    if(group)return group.markers;
    const [low,high]=bounds();
    return d.groups.flatMap(g=>g.markers.filter(m=>m.x_mhz>=low && m.x_mhz<=high));
  };
  const minWidth=Math.min(full[1]-full[0],2*(d.x[1]-d.x[0]));
  const navControls=element('details',undefined,'mc-range');
  navControls.innerHTML='<summary>Zoom window</summary><p>Drag the shaded window to pan, drag an edge to resize, or click outside to recenter. Drag the dotted line to read a sample.</p>'+
    '<form><label>From (MHz) <input type="number" step="any" required></label> '+
    '<label>To (MHz) <input type="number" step="any" required></label> '+
    '<button type="button" class="mc-full">Full scan</button></form>';
  q('.mc-overview').after(navControls);
  const [from,to]=navControls.querySelectorAll('input');
  for(const input of [from,to]){
    input.min=String(full[0]);input.max=String(full[1]);
    input.style.width='8.5em';input.oninput=()=>to.setCustomValidity('');
  }
  function applyRange(){
    const a=from.valueAsNumber,b=to.valueAsNumber;
    const error=!Number.isFinite(a)||!Number.isFinite(b)
      ? 'Enter both window limits as finite numbers.'
      : a<full[0]||a>full[1]||b<full[0]||b>full[1]
      ? 'Keep both limits inside '+format(full[0],8)+' to '+format(full[1],8)+' MHz.'
      : b-a<minWidth ? 'Use an interval at least '+format(minWidth,5)+' MHz wide.' : '';
    to.setCustomValidity(error);
    if(error){to.reportValidity();return;}
    applyWindow({window:[a,b],cursor:d.x[state.sample_index]});refresh();publish();
  }
  navControls.querySelector('form').onsubmit=e=>{e.preventDefault();applyRange();};
  for(const input of [from,to]){
    input.onchange=applyRange;
    input.onkeydown=e=>{
      if(e.key==='Escape'){e.preventDefault();from.value=bounds()[0];to.value=bounds()[1];to.setCustomValidity('');}
      else if(e.key==='Enter'){e.preventDefault();applyRange();}
    };
  }
  navControls.querySelector('.mc-full').onclick=()=>{
    applyWindow({window:full,cursor:d.x[state.sample_index]});refresh();publish();
  };
  function getIndices(){
    const [lo,hi]=bounds();let a=nearestSample(d.x,lo),b=nearestSample(d.x,hi);
    if (a<b && d.x[a]<lo) a++;
    if (b>a && d.x[b]>hi) b--;
    return [a,b];
  }
  function pickSample(x) {
    const [a,b]=getIndices();
    state.sample_index=Math.min(b,Math.max(a,nearestSample(d.x,x)));
  }
  function applyWindow(result) {
    const exact=d.groups.find(g=>g.window_mhz.every((v,j)=>Math.abs(v-result.window[j])<1e-8));
    state.group_id=exact?.id ?? null;
    state.window_mhz=exact ? [...exact.window_mhz] : result.window;
    pickSample(result.cursor);
  }
  function snapshot() {
    const {reference,detuning_mhz,...selection}=state, x=d.x[state.sample_index];
    return reference && driven
      ? {...selection,x_mhz:x,reference,detuning_mhz:x-reference.x_mhz}
      : {...selection,x_mhz:x};
  }
  function publish() {
    const value = {...snapshot(),context:data.context,navigation_revision:revision}, stamp=JSON.stringify(value);
    if (stamp===entry.lastSent) return;
    entry.lastSent=stamp;
    setStateValue('selection',value);
  }
  function selectGroup(id) {
    state.group_id=id;
    state.window_mhz=[...groupFor().window_mhz];
    const [a,b]=bounds();pickSample((a+b)/2);
    refresh();publish();
  }
  for (const g of d.groups) {
    const label=element('label',undefined,'mc-choice'), radio=element('input');
    radio.type='radio'; radio.name='doppler-group';radio.value=g.id;
    radio.onchange = () => { if (radio.checked) selectGroup(g.id); };
    label.append(radio,element('span',g.label));groups.append(label);
  }
  if (!d.groups.length) groups.append(element('span','Full scan · no grouped transition markers','mc-muted'));
  function navigateToFrequency(value) {
    if (!Number.isFinite(value)) return;
    const [a,b]=bounds(), sample=d.x[nearestSample(d.x,value)];
    if (value<a || value>b || sample<a || sample>b) {
      applyWindow(navigateSpectrumWindow('center',{window:[a,b],x:value,cursor:d.x[state.sample_index]},
                                         value,full,minWidth));
    } else pickSample(value);
  }
  function goFrequency() {
    navigateToFrequency(freq.valueAsNumber);
    refresh();publish();
  }
  freq.onchange=goFrequency;
  freq.onkeydown=e=>{
    if (e.key==='Enter') {e.preventDefault();goFrequency();}
    else if (e.key==='ArrowDown'||e.key==='ArrowUp') {
      e.preventDefault();step(e.key==='ArrowUp'?1:-1);
    }
  };
  function step(delta) {
    const [a,b]=getIndices();
    const index=Math.min(b,Math.max(a,state.sample_index+delta));
    if (index===state.sample_index) return;
    state.sample_index=index;
    refresh();publish();
  }
  q('.prev').onclick=()=>step(-1);q('.next').onclick=()=>step(1);
  picker.onchange=()=>{
    if (driven) {
      state.reference=references[picker.selectedIndex];
      pickerWindow='';  // Remove a former off-screen anchor from the options.
    } else if (picker.value) state.sample_index=Number(picker.value);
    refresh();publish();
  };
  q('.mc-go-ref').onclick=()=>{
    if (state.reference) {navigateToFrequency(state.reference.x_mhz);refresh();publish();}
  };
  q('.mc-export').onclick=()=>{
    const value={...d,selection:snapshot()};
    const url=URL.createObjectURL(new Blob([JSON.stringify(value)],{type:'application/json'}));
    const link=element('a');link.href=url;link.download='GABES-'+d.scheme+'-model-comparison.json';
    link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  function seriesFor(key) {
    const off={y:base[key],color:ink.off,dash:active===base ? '' : '6 4',label:'Pump off'};
    return active===base ? [off] : [off,{y:active[key],color:ink.on,label:onLabel}];
  }
  function draw() {
    const group=groupFor(),limit=bounds();
    const curves=seriesFor('transmission');
    const marks=visibleMarkers();
    const spec={grid:d.x,index:state.sample_index,unit:'MHz',xlim:limit,ylim:transmissionRange,
                series:curves,yLabel:'Transmission T'};
    const panel=(selector,config)=>{
      const host=q(selector);host.replaceChildren();spectrum(host,{...spec,...config});
    };
    panel('.mc-overview',{title:'Full transmission spectrum',xlim:full,unit:d.axis.overview_unit,
      band:limit,description:'Full transmission. Drag the shaded zoom window, its edges or the dotted sample cursor. Click outside the window to recenter it.'});
    panel('.mc-zoom',{title:'Transmission',marks,keyboardInspector:true,
      description:'Transmission in the selected frequency window. Hover to read the computed samples.'});
    const [first,last]=getIndices();
    panel('.mc-od',{title:active.effective?'Effective optical depth (OD)':'Optical depth (OD)',
      yLabel:'Optical depth τ',ylim:valueRange(d.series,'optical_depth',first,last,0),
      series:seriesFor('optical_depth'),marks,keyboardInspector:true,
      description:'Natural-log optical depth. The vertical scale fits both curves inside this frequency window. Hover to read the computed samples.'});
  }
  function updatePicker(group, x) {
    const limits=bounds(), windowKey=JSON.stringify([state.group_id,...limits]);
    if (pickerWindow!==windowKey) {
      pickerWindow=windowKey;
      picker.replaceChildren();
      if (driven) {
        references=modelReferences(d.groups,group?.id,limits,state.reference,full);
        for (const m of references) {
          const name=group ? m.label : d.groups.find(g=>g.id===m.group_id).label+' · '+m.label;
          const outside=m.x_mhz<limits[0] || m.x_mhz>limits[1];
          picker.append(element('option',name+' · '+format(m.x_mhz,8)+' MHz'+(outside?' · outside window':'')));
        }
        if (!references.length) picker.append(element('option','No named reference in window'));
        picker.disabled=q('.mc-go-ref').disabled=!references.length;
      } else {
        const inWindow=peaks.filter(i=>d.x[i]>=limits[0] && d.x[i]<=limits[1]);
        const cursor=element('option',inWindow.length ? 'Choose a peak…' : 'No sampled peak in window');
        cursor.value='';picker.append(cursor);
        for (const i of inWindow) {
          const option=element('option',format(d.x[i],8)+' MHz');
          option.value=String(i);picker.append(option);
        }
        picker.disabled=!inWindow.length;
      }
    }
    if (driven) {
      state.reference=chooseReference(references,state.reference,x);
      const offscreen=state.reference && (state.reference.x_mhz<limits[0] || state.reference.x_mhz>limits[1]);
      q('.mc-pick span').textContent=offscreen ? 'Model reference · outside window' : 'Model reference';
      if (state.reference) picker.selectedIndex=references.indexOf(state.reference);
    } else {
      picker.value=String(state.sample_index);
      if (picker.selectedIndex<0) picker.value='';
    }
  }

  function showPoint(i) {
    freq.value=Number(d.x[i].toPrecision(12)).toString();
    const delta=active.delta?.[i], ref=state.reference;
    if (driven) {
      q('.mc-offset').textContent=ref ? signedReadout(d.x[i]-ref.x_mhz,6)+' MHz' : '—';
      q('.mc-delta').textContent=Number.isFinite(delta) ? signedReadout(delta*100,5)+' pp' : '—';
    } else {
      renderOdReadings(q('.mc-od-readings'),{
        od0:active.od0?.[i], odEff:active.optical_depth[i], nEff:d.n_eff,
        transmission:active.transmission[i]
      });
    }
  }

  function refresh() {
    const group=groupFor(),curve=curveFor(),i=state.sample_index,[a,b]=getIndices();
    state.x_mhz=d.x[i];
    for (const radio of groups.querySelectorAll('input')) radio.checked=radio.value===state.group_id;
    freq.min=String(full[0]);freq.max=String(full[1]);
    q('.prev').setAttribute('aria-disabled',String(i<=a));
    q('.next').setAttribute('aria-disabled',String(i>=b));
    q('.mc-loc').textContent=group?.label||(bounds()[0]===full[0]&&bounds()[1]===full[1]?'Full scan':'Custom window');
    updatePicker(group,d.x[i]);
    showPoint(i);
    from.value=bounds()[0];to.value=bounds()[1];to.setCustomValidity('');
    q('.mc-values caption').textContent='T and optical depth at the selected frequency';
    const tbody=q('tbody');tbody.replaceChildren();
    for (const c of (active===base ? [base] : [base,active])) {
      const row=element('tr');row.dataset.selected=String(c===curve);
      const name=element('th',c===base ? 'Off · 0 mW' : format(c.power_mw,8)+' mW');name.scope='row';row.append(name);
      for (const y of [c.transmission[i],c.optical_depth[i]]) row.append(element('td',format(y,6)));
      tbody.append(row);
    }
    const resolution=curve.resolution || {}, status=resolution.status;
    const scope=resolution.scope==='model_feature_window'?'Representative feature in model search window':'Representative full-spectrum feature';
    const feature=[scope+' for '+format(curve.power_mw)+' mW'];
    if (Number.isFinite(resolution.center_mhz)) feature.push('at '+format(resolution.center_mhz,7)+' MHz');
    if (Number.isFinite(resolution.center_mhz) &&
        (resolution.center_mhz < bounds()[0] || resolution.center_mhz > bounds()[1])) {
      feature.push(group?'outside this group':'outside this window');
    }
    if (Number.isFinite(resolution.width_mhz)) feature.push('FWHM '+format(resolution.width_mhz,5)+' MHz');
    if (Number.isFinite(resolution.samples_per_fwhm)) feature.push(format(resolution.samples_per_fwhm,3)+' samples/FWHM (minimum 6)');
    if (status) feature.push(status);
    q('.mc-feature').textContent=status?feature.join(' · '):'';
    const probe=q('.mc-probe');
    probe.hidden=!curve.probe_model_scope;
    probe.textContent=curve.probe_model_scope ? 'Probe model: '+curve.probe_model_scope + (curve.probe_model_note ? '. '+curve.probe_model_note : '') : '';
    q('.mc-basis').textContent=driven
      ? 'Detuning is the sample frequency minus the model reference. Go to reference selects the nearest solved sample. ΔT is the pump-on minus pump-off transmission in percentage points.'
      : 'OD₀ is the weak-probe optical depth. Effective OD equals −ln(T) at the selected probe power. The effective atom count includes atoms in the Gaussian probe volume across all isotopes.';
    const markerText=visibleMarkers().map(m=>m.label+' at '+format(m.x_mhz,8)+' MHz').join(' · ');
    let markerNote = markerText ? markerText+'. CO? marks a candidate crossover at the midpoint of two allowed transitions.' : 'No named transition markers in this window.';
    if (group?.overlaps?.length) markerNote += ' Group windows overlap. Curves show total transmission.';
    q('.mc-marker-note').textContent=markerNote;
    const fixed=d.fixed || {}, conditions=[fixed.species,fixed.line];
    if (Number.isFinite(fixed.temp_c)) conditions.push(fixed.temp_c+' °C');
    if (Number.isFinite(fixed.cell_mm)) conditions.push(fixed.cell_mm+' mm cell');
    if (Number.isFinite(fixed.probe_power_uw)) conditions.push(fixed.probe_power_uw+' µW probe');
    conditions.push(d.x.length+' solved samples per curve',d.axis.reference);
    q('.mc-condition-note').textContent=conditions.filter(Boolean).join(' · ')+'. Zoom retains the original samples.';
    const ref=state.reference;
    q('output').textContent='Sample '+format(d.x[i],8)+' MHz. '+(driven && ref
      ? 'Detuning '+q('.mc-offset').textContent+' from '+ref.label+'.'
      : 'Transmission '+format(curve.transmission[i],6)+', optical depth '+format(curve.optical_depth[i],6)+'.');
    draw();
  }
  let width=0, frame=0;
  const observer=new ResizeObserver(records=>{
    const next=records[0].contentRect.width;
    if (Math.abs(width-next)<.5) return;
    width=next;cancelAnimationFrame(frame);frame=requestAnimationFrame(draw);
  });
  const pointer=sameNavigation && previous.pointer ? previous.pointer : {drag:null};
  const graphs=q('.mc-graphs');
  function hit(svg,e){
    const n=svg._navigation,{x,y}=spectrumPointer(svg,e,n.w,n.h);
    if(x<n.left||x>n.right||y<n.top||y>n.bottom)return null;
    const window=bounds(),cursor=d.x[state.sample_index],frequency=n.toFrequency(x);
    let action='cursor';
    if(n.overview){
      const edge=Math.min(Math.abs(x-n.sx(window[0])),Math.abs(x-n.sx(window[1])));
      const tolerance=Math.min(e.pointerType==='touch'?12:6,Math.max(2,(n.sx(window[1])-n.sx(window[0]))/5));
      if(Math.abs(x-n.sx(cursor))<=tolerance && (edge>tolerance || y>n.top+10))action='cursor';
      else if(edge<=tolerance)action=Math.abs(x-n.sx(window[0]))<Math.abs(x-n.sx(window[1]))?'left':'right';
      else if(frequency>=window[0]&&frequency<=window[1])action='pan';
      else action='center';
    }
    const slot=svg.closest('.mc-overview')?'.mc-overview':svg.closest('.mc-od')?'.mc-od':'.mc-zoom';
    return {action,slot,startX:e.clientX,window:[...window],x:frequency,cursor};
  }
  const stopPointer=bindSpectrumGesture(graphs,pointer,{
    begin(e){
      const svg=e.composedPath().find(node=>node._navigation);
      return svg ? hit(svg,e) : null;
    },
    preview(drag,e) {
      const svg=q(drag.slot+' svg'), n=svg?._navigation;
      if(!n) return;
      const {x}=spectrumPointer(svg,e,n.w,n.h);
      const result=navigateSpectrumWindow(drag.action,drag,n.toFrequency(x),full,minWidth);
      applyWindow(result);
      for(const svg of graphs.querySelectorAll('svg'))if(svg._navigation)previewMarkers(svg,bounds(),state.sample_index);
      for(const radio of groups.querySelectorAll('input'))radio.checked=radio.value===state.group_id;
      root.dataset.gesture='preview';
      showPoint(state.sample_index);
      q('.mc-loc').textContent='Preview · '+format(d.x[state.sample_index],8)+' MHz · window '+bounds().map(v=>format(v,7)).join(' to ')+' MHz';
    },
    commit(drag,e) {
      if(drag.action==='pan' && Math.abs(e.clientX-drag.startX)<2)
        applyWindow({window:drag.window,cursor:drag.x});
      root.dataset.gesture='ready';refresh();publish();
    },
    cancel(drag){
      applyWindow({window:drag.window,cursor:drag.cursor});
      root.dataset.gesture='ready';refresh();
    }
  });
  graphs.onpointermove=e=>{
    if(pointer.drag)return;
    const svg=e.composedPath().find(node=>node._navigation);
    if(!svg)return;
    const action=hit(svg,e)?.action;
    svg.style.cursor=action==='pan'?'grab':action==='left'||action==='right'||action==='cursor'?'ew-resize':'crosshair';
  };
  const entry={root,state,pointer,context:data.context,navigationRevision:revision,lastSent:'',cleanup:()=>{observer.disconnect();cancelAnimationFrame(frame);stopPointer();}};
  sessions.set(key,entry);
  entry.lastSent=JSON.stringify({...snapshot(),context:data.context,navigation_revision:revision});
  refresh();observer.observe(root);
  // If the tab unmounts, detach its observer. Keep only the small selection
  // while the same context may remount; do not retain arrays or detached DOM.
  return ()=>{
    entry.cleanup();
    if(sessions.get(key)!==entry)return;
    const saved={context:entry.context,navigationRevision:revision,state:{...state},pointer,cleanup:()=>{}};
    sessions.set(key,saved);
    queueMicrotask(()=>{
      if(sessions.get(key)!==saved || !pointer.drag || root.isConnected)return;
      // A genuine unmount cancels uncommitted motion. A replacement mount can
      // adopt the gesture before this check and continue from its new SVG.
      applyWindow({window:pointer.drag.window,cursor:pointer.drag.cursor});
      saved.state={...state}; pointer.drag=null;
    });
  };
}
