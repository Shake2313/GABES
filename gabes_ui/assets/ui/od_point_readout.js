// Both OD inspectors display the same model quantities at a native sample.
const atomCountFormat = new Intl.NumberFormat('en-US', {maximumSignificantDigits: 4});

function renderOdReadings(dl, {od0, odEff, nEff, transmission}) {
  if (!dl.firstElementChild) {
    dl.innerHTML = `
      <dt>Weak-probe OD<sub>0</sub></dt><dd data-quantity="od0"></dd>
      <dt>Effective OD<sub>eff</sub></dt><dd data-quantity="odEff"></dd>
      <dt>Probe atoms N<sub>eff</sub></dt><dd data-quantity="nEff"></dd>
      <dt>Transmission T</dt><dd data-quantity="transmission"></dd>`;
  }
  const set = (key, value) => {
    const field = dl.querySelector('[data-quantity="'+key+'"]');
    field.textContent = value ?? '—';
    if (value == null) field.setAttribute('aria-label', 'Unavailable for this model');
    else field.removeAttribute('aria-label');
  };
  set('od0', Number.isFinite(od0) ? format(od0, 6) : null);
  set('odEff', Number.isFinite(odEff) ? format(odEff, 6) : null);
  set('nEff', Number.isFinite(nEff) ? atomCountFormat.format(nEff) : null);
  set('transmission', Number.isFinite(transmission) ? format(100 * transmission, 6) + '%' : null);
}
