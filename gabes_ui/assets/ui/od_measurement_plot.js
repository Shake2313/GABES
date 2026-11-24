// Overview navigation changes the viewport only. The detail window and its
// selected native theory sample are shared by both detail plots and the readout.
export function chooseOverview(state, limits) {
  if (state.views[state.position].every((v, i) => v === limits[i])) return;
  state.views.splice(state.position + 1);
  state.views.push([...limits]);
  if (state.views.length > 40) state.views.splice(1, 1);
  state.position = state.views.length - 1;
}

export default function mount({parentElement, data:d}) {
  const root = parentElement.querySelector('.od-measurement');
  const x = d.grid, domain = [x[0], x.at(-1)], width = domain[1] - domain[0];
  const stateKey = JSON.stringify([d.source, ...domain, x.length]);
  const centre = (domain[0] + domain[1]) / 2;
  const state = root._odKey === stateKey ? root._odState : {
    views:[domain], position:0, detail:[centre-width/8, centre+width/8], index:nearestSample(x,centre)
  };
  root._odState = state; root._odKey = stateKey;
  root.innerHTML = `
    <div class="mc-legend"><span><i></i>Theory</span><span><i class="off"></i>Measurement</span></div>
    <div class="mc-graphs"><div class="od-overview"></div>
      <div class="od-history" role="group" aria-label="Overview navigation">
        <button type="button" data-action="zoom">Zoom overview to window</button>
        <button type="button" data-action="home">Home</button><button type="button" data-action="back">Back</button>
        <button type="button" data-action="forward">Forward</button></div>
      <div class="od-controls"><label>Window start (GHz)<input type="number" step="any" name="low"></label>
        <label>Window end (GHz)<input type="number" step="any" name="high"></label>
        <label>Selected frequency (GHz)<input type="number" step="any" name="point"></label></div>
      <div class="mc-layout"><div class="mc-local"><div class="od-detail"></div><div class="od-depth"></div></div>
      <section class="mc-reader" aria-label="Selected point">
        <h3>Model at selected frequency</h3><p class="mc-loc"></p>
        <dl class="mc-readings" aria-live="polite"></dl>
        <h4>Nearest measured sample</h4><output aria-live="polite"></output>
      </section></div></div>`;
  const q = query => root.querySelector(query);
  q('.mc-legend i').style.borderColor = d.theoryColor;
  q('.mc-legend .off').style.borderColor = d.measuredColor;
  const mx = d.measuredX, my = d.measuredY;
  const depth = d.depth || d.transmission.map(opticalDepth);
  const measuredDepth = d.absolute ? my.map(opticalDepth) : null;
  const native = {sampleX:mx, sampleY:my, coverage:d.coverage};
  function pick(value) {
    state.index = nearestSample(x, value);
    const picked = x[state.index], [lo, hi] = state.detail;
    if (picked < lo || picked > hi) {
      const low = Math.max(domain[0], Math.min(picked - (hi-lo)/2, domain[1]-(hi-lo)));
      state.detail = [low, low+hi-lo];
    }
  }
  function refresh() {
    const view = state.views[state.position], selected = x[state.index];
    for (const [name, value] of [['low',state.detail[0]], ['high',state.detail[1]], ['point',selected]]) {
      const input = q('[name='+name+']'); input.value=Number(value.toPrecision(12)); input.setCustomValidity('');
    }
    q('[data-action=back]').disabled = state.position === 0;
    q('[data-action=forward]').disabled = state.position === state.views.length-1;
    q('[data-action=home]').disabled = view.every((v,i)=>v===domain[i]);
    const nearest = curveSample(native, x, selected);
    q('.mc-loc').textContent = format(selected,8)+' GHz';
    renderOdReadings(q('.mc-readings'), {od0:d.od0?.[state.index], odEff:depth[state.index],
      nEff:d.nEff, transmission:d.transmission[state.index]});
    const measured = nearest.x === null ? 'No measured point at this frequency'
      : format(nearest.x,8)+' GHz · '+(d.absolute
        ? 'T '+format(100*nearest.y,6)+'% · OD '+format(opticalDepth(nearest.y),6)
        : 'Relative signal '+format(nearest.y,6));
    q('output').textContent = measured;
    function panel(selector,title,limits,isDepth,overview) {
      const host=q(selector);
      const oldChart=host.querySelector('svg');
      const active=oldChart && host.getRootNode().activeElement===oldChart;
      host.replaceChildren();
      const theory={label:'Theory',color:d.theoryColor,y:isDepth?depth:d.transmission};
      const curves=[theory], samplesY=isDepth?measuredDepth:my;
      if(samplesY) curves.push({...measurementPath(mx,my,d.coverage,limits,host.clientWidth || 500),
        label:'Measurement',color:d.measuredColor,dash:'5 3',sampleX:mx,sampleY:samplesY,coverage:d.coverage});
      if(isDepth && samplesY) curves[1].y=curves[1].y.map(opticalDepth);
      spectrum(host,{title,description:title+' versus frequency',grid:x,series:curves,
        xlim:limits,ylim:plotLimits(curves,x,limits,isDepth?0:1),yLabel:isDepth?'Optical depth':d.absolute?'Transmission':'Transmission / relative signal',
        yUnit:'1',unit:'GHz',divisor:1,compact:!overview,band:overview?state.detail:null,index:state.index});
      const svg=host.querySelector('svg'); svg.setAttribute('tabindex','0');
      if(active) svg.focus({preventScroll:true});
    }
    panel('.od-overview','Spectrum overview',view,false,true);
    panel('.od-detail','Detail · transmission',state.detail,false,false);
    panel('.od-depth','Detail · optical depth',state.detail,true,false);
  }
  q('.od-history').onclick=e=>{
    const action=e.target.dataset.action;
    if(action==='zoom') chooseOverview(state,state.detail);
    if(action==='home') chooseOverview(state,domain);
    if(action==='back') state.position=Math.max(0,state.position-1);
    if(action==='forward') state.position=Math.min(state.views.length-1,state.position+1);
    refresh();
  };
  q('.od-controls').onchange=e=>{
    const input=e.target, v=input.valueAsNumber;
    const next=[q('[name=low]').valueAsNumber,q('[name=high]').valueAsNumber];
    let valid=Number.isFinite(v) && v>=domain[0] && v<=domain[1];
    if(input.name==='point' && valid) pick(v);
    else if(input.name!=='point') {
      valid=valid && next.every(Number.isFinite) && next[0]<next[1] &&
        x[nearestSample(x,next[1])]>=next[0] && x[nearestSample(x,next[0])]<=next[1];
      if(valid) {
        state.detail=next;
        let i=nearestSample(x,Math.max(next[0],Math.min(next[1],x[state.index])));
        if(x[i]<next[0])i++; else if(x[i]>next[1])i--;
        state.index=i;
      }
    }
    if(!valid){input.setCustomValidity('Choose a frequency within the spectrum and a window containing a sample.');input.reportValidity();return;}
    refresh();
  };
  const graphs=q('.mc-graphs');
  graphs.onkeydown=e=>{
    if(e.target.tagName.toLowerCase()!=='svg' || !['ArrowLeft','ArrowRight'].includes(e.key))return;
    e.preventDefault(); pick(x[Math.max(0,Math.min(x.length-1,state.index+(e.key==='ArrowLeft'?-1:1)))]); refresh();
  };
  const pointer={drag:null};
  function locate(svg,e) {
    const n=svg._navigation, {x:px,y:py}=spectrumPointer(svg,e,n.w,n.h);
    return {px,py,frequency:n.toFrequency(px),n};
  }
  const stop=bindSpectrumGesture(graphs,pointer,{
    begin(e) {
      const svg=e.composedPath().find(node=>node._navigation); if(!svg)return null;
      const {px,py,frequency,n}=locate(svg,e);
      if(px<n.left || px>n.right || py<n.top || py>n.bottom)return null;
      let action='cursor';
      if(n.overview) {
        const [a,b]=state.detail.map(n.sx), tolerance=Math.min(e.pointerType==='touch'?12:6,Math.max(2,(b-a)/5));
        if(Math.abs(px-a)<=tolerance)action='left';
        else if(Math.abs(px-b)<=tolerance)action='right';
        else if(px>a && px<b)action='pan';
      }
      return {action,svg,x:frequency,window:[...state.detail],cursor:x[state.index],start:e.clientX};
    },
    preview(drag,e) {
      const {frequency}=locate(drag.svg,e);
      const result=navigateSpectrumWindow(drag.action,drag,frequency,domain,width/(x.length-1));
      if(drag.action==='cursor')pick(frequency);
      else {state.detail=result.window; pick(result.cursor);}
      for(const chart of graphs.querySelectorAll('svg'))previewMarkers(chart,state.detail,state.index);
    },
    commit(drag,e) {
      if(drag.action==='pan' && Math.abs(e.clientX-drag.start)<2)pick(drag.x);
      refresh();
    },
    cancel(drag){state.detail=drag.window; state.index=nearestSample(x,drag.cursor);refresh();}
  });
  let lastWidth=root.clientWidth;
  const resize=new ResizeObserver(()=>{if(root.clientWidth!==lastWidth){lastWidth=root.clientWidth;refresh();}});
  refresh();resize.observe(root);
  return ()=>{stop();resize.disconnect();};
}
