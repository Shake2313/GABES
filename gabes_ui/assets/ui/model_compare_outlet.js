// Only component-owned DOM crosses these two shadow roots. The fit controls
// between them remain ordinary Streamlit widgets with their native lifecycle.
export default function mount({parentElement, data}) {
  const host=parentElement.querySelector('.mc-outlet');
  const event='gabes:od-readout:'+data.group;
  const connect=()=>window.dispatchEvent(new CustomEvent(event, {
    detail:{part:data.part, host}
  }));
  window.addEventListener(event+':ready',connect);
  connect();
  return ()=>window.removeEventListener(event+':ready',connect);
}
