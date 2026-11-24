function opticalDepth(t) { return Number.isFinite(t) && t > 0 ? -Math.log(t) : null; }

// Preserve endpoints and vertical extrema in display bins for each measured
// interval. Nearest-point inspection uses the native arrays, including on zoom.
export function measurementPath(x, y, intervals, limits, pixels) {
  const outX = [], outY = [];
  for (const [low, high] of intervals) {
    const left = Math.max(limits[0], low), right = Math.min(limits[1], high);
    if (left > right) continue;
    let first = nearestSample(x, left), last = nearestSample(x, right);
    if (x[first] < low) first++;
    if (x[last] > high) last--;
    if (first > last) continue;
    if (outX.length) { outX.push((outX.at(-1) + x[first]) / 2); outY.push(null); }
    const take = i => { outX.push(x[i]); outY.push(y[i]); };
    const stride = Math.max(1, Math.ceil((last - first + 1) / Math.max(1, pixels)));
    for (let i = first; i <= last; i += stride) {
      const end = Math.min(last, i + stride - 1);
      let min = i, max = i;
      const indices = new Set([i, end]);
      for (let j = i + 1; j <= end; j++) {
        if (Number.isFinite(y[j]) !== Number.isFinite(y[j-1])) {
          indices.add(j-1); indices.add(j);
        }
        if (!Number.isFinite(y[j])) continue;
        if (!Number.isFinite(y[min]) || y[j] < y[min]) min = j;
        if (!Number.isFinite(y[max]) || y[j] > y[max]) max = j;
      }
      indices.add(min); indices.add(max);
      for (const j of Array.from(indices).sort((a,b)=>a-b)) take(j);
    }
  }
  return {x:outX, y:outY};
}

function plotLimits(curves, grid, limits, floorMax) {
  let lo = 0, hi = floorMax;
  for (const curve of curves) {
    const x = curve.x || grid;
    for (let i = nearestSample(x, limits[0]); i < x.length && x[i] <= limits[1]; i++) {
      const y = curve.y[i];
      if (Number.isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
    }
  }
  return [lo, hi + .04 * (hi-lo || 1)];
}
