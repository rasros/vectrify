// Pencil: a freehand stroke, sampled as the pointer moves, fitted to a few
// cubic curves. Everything is measured in screen pixels, so a stroke comes
// out alike at any zoom: the curves stay within the smoothing distance of the
// stroke, and smoothing more gives fewer points. Pure, so it can be checked
// outside the browser; the preview and the path drawn are one fit.
//
// The fit follows Redraw outline's freehand fit (refine/redraw.py): the
// stroke resampled evenly, its sharp turns kept as corners, smoothed between
// them, then fitted with as few cubics as stay within the tolerance, meeting
// smoothly except at the corners (Schneider's error-bounded fit).

// The smoothing distance, in screen pixels: by default, and its range.
export const SMOOTHING = Object.freeze({initial: 2, least: 1, most: 20});
// How far apart a stroke's samples are kept, in screen pixels.
export const SAMPLE_STEP = 1;
// A stroke shorter than this, in screen pixels, draws nothing.
export const SHORTEST = 2;
// The most samples a fit reads: longer strokes are resampled more sparsely.
const MOST_SAMPLES = 2000;
// A turn sharper than this, in degrees, over a few pixels either side, stays a
// corner.
const CORNER = 60;
// The smoothing share the stroke is blurred by before fitting.
const BLUR = 0.5;
// Reparameterisation steps tried on a cubic nearly within tolerance.
const STEPS = 4;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const scale = (a, s) => [a[0] * s, a[1] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const length = a => Math.hypot(a[0], a[1]);
const finite = p => Number.isFinite(p[0]) && Number.isFinite(p[1]);

// *a* as a unit vector, or *fallback* when it has no length.
function unit(a, fallback = null) {
  const size = length(a);
  return size > 1e-9 ? scale(a, 1 / size) : fallback;
}

// The smoothing distance *value* asks for, kept within its range.
export function smoothingDistance(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return SMOOTHING.initial;
  return Math.min(SMOOTHING.most, Math.max(SMOOTHING.least, number));
}

// *line* with points every *step* along it, its ends kept.
function resampled(line, step) {
  const along = [0];
  for (let i = 1; i < line.length; i++) along.push(along[i - 1] + length(sub(line[i], line[i - 1])));
  const total = along.at(-1), count = Math.max(2, Math.ceil(total / step) + 1);
  const points = [line[0]];
  let k = 1;
  for (let i = 1; i < count - 1; i++) {
    const at = total * i / (count - 1);
    while (along[k] < at) k++;
    const span = along[k] - along[k - 1], t = span > 0 ? (at - along[k - 1]) / span : 0;
    points.push(add(line[k - 1], scale(sub(line[k], line[k - 1]), t)));
  }
  points.push(line.at(-1));
  return points;
}

// Where *line* turns sharply, more than CORNER degrees over *span* points
// either side: the sharpest point of each turn.
function corners(line, span) {
  const turns = [];
  for (let i = span; i < line.length - span; i++) {
    const before = sub(line[i], line[i - span]), after = sub(line[i + span], line[i]);
    const size = length(before) * length(after);
    if (size < 1e-12) continue;
    const turn = Math.acos(Math.max(-1, Math.min(1, dot(before, after) / size))) * 180 / Math.PI;
    if (turn >= CORNER) turns.push([turn, i]);
  }
  turns.sort((a, b) => b[0] - a[0]);
  const found = [];
  for (const [, i] of turns) if (found.every(j => Math.abs(i - j) > span)) found.push(i);
  return found.sort((a, b) => a - b);
}

// *line* blurred along its length over *sigma* points, each piece between the
// *kept* points on its own, those staying.
function smoothed(line, sigma, kept) {
  if (sigma < 0.3) return line;
  const radius = Math.ceil(3 * sigma), weights = [];
  for (let k = -radius; k <= radius; k++) weights.push(Math.exp(-k * k / (2 * sigma * sigma)));
  const smooth = line.map(p => [...p]);
  for (let piece = 1; piece < kept.length; piece++) {
    const from = kept[piece - 1], to = kept[piece];
    for (let i = from + 1; i < to; i++) {
      let x = 0, y = 0, sum = 0;
      for (let k = -radius; k <= radius; k++) {
        // Past the piece's ends, its end points repeat.
        const p = line[Math.max(from, Math.min(to, i + k))], w = weights[k + radius];
        x += w * p[0]; y += w * p[1]; sum += w;
      }
      smooth[i] = [x / sum, y / sum];
    }
  }
  return smooth;
}

function bezier([a, b, c, d], t) {
  const u = 1 - t;
  return add(add(scale(a, u * u * u), scale(b, 3 * u * u * t)), add(scale(c, 3 * u * t * t), scale(d, t * t * t)));
}

// The cubic from points[0] to its last point leaving along *leave* and
// arriving back along *arrive*, its handle lengths fitted by least squares to
// the points at parameters *u*.
function cubic(points, u, leave, arrive) {
  const first = points[0], last = points.at(-1), chord = length(sub(last, first));
  const c = [[0, 0], [0, 0]], x = [0, 0];
  for (let i = 0; i < points.length; i++) {
    const t = u[i], v = 1 - t;
    const a0 = scale(leave, 3 * v * v * t), a1 = scale(arrive, 3 * v * t * t);
    c[0][0] += dot(a0, a0); c[0][1] += dot(a0, a1); c[1][1] += dot(a1, a1);
    const rest = sub(points[i], add(scale(first, v * v * v + 3 * v * v * t), scale(last, 3 * v * t * t + t * t * t)));
    x[0] += dot(a0, rest); x[1] += dot(a1, rest);
  }
  c[1][0] = c[0][1];
  const det = c[0][0] * c[1][1] - c[1][0] * c[0][1];
  let left = det ? (x[0] * c[1][1] - x[1] * c[0][1]) / det : 0;
  let right = det ? (c[0][0] * x[1] - c[1][0] * x[0]) / det : 0;
  // Handles that would point backwards, or are too short to tell, fall back
  // to a third of the chord.
  const least = 1e-6 * chord;
  if (!Number.isFinite(left) || !Number.isFinite(right) || left < least || right < least) left = right = chord / 3;
  return [first, add(first, scale(leave, left)), add(last, scale(arrive, right)), last];
}

// Each point's parameter, by its share of the distance along them.
function chordParameters(points) {
  const u = [0];
  for (let i = 1; i < points.length; i++) u.push(u[i - 1] + length(sub(points[i], points[i - 1])));
  const total = u.at(-1);
  return total > 0 ? u.map(v => v / total) : u.map((_, i) => i / (points.length - 1));
}

// The farthest any point lies from the curve at its parameter, squared, and
// which point that is.
function farthest(curve, points, u) {
  let most = 0, at = Math.floor(points.length / 2);
  for (let i = 1; i < points.length - 1; i++) {
    const offset = sub(bezier(curve, u[i]), points[i]), distance = dot(offset, offset);
    if (distance >= most) { most = distance; at = i; }
  }
  return [most, at];
}

// Better parameters for *points* on *curve*: one Newton step towards the
// nearest place on it for each.
function reparameterised(curve, points, u) {
  const [a, b, c, d] = curve;
  const first = [scale(sub(b, a), 3), scale(sub(c, b), 3), scale(sub(d, c), 3)];
  const second = [scale(sub(first[1], first[0]), 2), scale(sub(first[2], first[1]), 2)];
  return u.map((t, i) => {
    if (i === 0 || i === u.length - 1) return t;
    const v = 1 - t, offset = sub(bezier(curve, t), points[i]);
    const slope = add(add(scale(first[0], v * v), scale(first[1], 2 * v * t)), scale(first[2], t * t));
    const bend = add(scale(second[0], v), scale(second[1], t));
    const denominator = dot(slope, slope) + dot(offset, bend);
    const next = Math.abs(denominator) > 1e-12 ? t - dot(offset, slope) / denominator : t;
    return Number.isFinite(next) ? Math.max(0, Math.min(1, next)) : t;
  });
}

// The direction *points* leave in from point *i*, read a little way along
// them towards *towards*.
function direction(points, i, towards, reach) {
  const step = Math.sign(towards - i), j = i + step * Math.max(1, Math.min(reach, Math.abs(towards - i)));
  return unit(sub(points[j], points[i])) ?? unit(sub(points[towards], points[i]));
}

// As few cubics as stay within *tolerance* (squared) of *points*, leaving
// along *leave* and arriving along *arrive*, split where they stray most and
// smooth across each split.
function fitted(points, leave, arrive, tolerance, reach, out) {
  if (points.length === 2) {
    const chord = length(sub(points[1], points[0])) / 3;
    out.push([points[0], add(points[0], scale(leave, chord)), add(points[1], scale(arrive, chord)), points[1]]);
    return;
  }
  let u = chordParameters(points), curve = cubic(points, u, leave, arrive), [most, at] = farthest(curve, points, u);
  if (most <= tolerance) { out.push(curve); return; }
  if (most <= 4 * tolerance) {
    for (let step = 0; step < STEPS; step++) {
      u = reparameterised(curve, points, u);
      curve = cubic(points, u, leave, arrive); [most, at] = farthest(curve, points, u);
      if (most <= tolerance) { out.push(curve); return; }
    }
  }
  // Split where the curve strays most, the two halves meeting smoothly.
  at = Math.max(1, Math.min(points.length - 2, at));
  const across = unit(sub(points[Math.max(0, at - reach)], points[Math.min(points.length - 1, at + reach)]))
    ?? unit(sub(points[at - 1], points[at + 1])) ?? scale(leave, -1);
  fitted(points.slice(0, at + 1), leave, across, tolerance, reach, out);
  fitted(points.slice(at), scale(across, -1), arrive, tolerance, reach, out);
}

// The stroke *points* (document units, as drawn) fitted at *zoom* screen
// pixels per unit to curves within *smoothing* screen pixels of it, closed
// back to its start with *closed*. {start, curves} in document units, or null
// for a stroke too short to draw.
export function fitStroke(points, {zoom = 1, smoothing = SMOOTHING.initial, closed = false} = {}) {
  if (!(zoom > 0) || !Number.isFinite(zoom)) return null;
  const tolerance = smoothingDistance(smoothing);
  // In screen pixels, without repeats and anything not a number.
  const line = [];
  for (const p of points) {
    const q = [p[0] * zoom, p[1] * zoom];
    if (finite(q) && (!line.length || length(sub(q, line.at(-1))) > 1e-6)) line.push(q);
  }
  if (line.length < 2) return null;
  let total = 0;
  for (let i = 1; i < line.length; i++) total += length(sub(line[i], line[i - 1]));
  if (total < SHORTEST) return null;
  // Closed, the stroke runs back to where it started.
  if (closed && length(sub(line.at(-1), line[0])) > 1e-6) line.push(line[0]);
  const step = Math.max(SAMPLE_STEP, (total + (closed ? length(sub(line.at(-1), line.at(-2))) : 0)) / MOST_SAMPLES);
  const samples = resampled(line, step);
  const reach = Math.max(1, Math.round(Math.max(3, tolerance) / step)), last = samples.length - 1;
  const kept = [0, ...corners(samples, reach).filter(i => i > 0 && i < last), last];
  const smooth = smoothed(samples, tolerance * BLUR / step, kept);
  const curves = [], near = Math.max(1, Math.round(2 / step));
  for (let piece = 1; piece < kept.length; piece++) {
    const part = smooth.slice(kept[piece - 1], kept[piece] + 1);
    const leave = direction(part, 0, part.length - 1, near) ?? [1, 0];
    const arrive = direction(part, part.length - 1, 0, near) ?? scale(leave, -1);
    fitted(part, leave, arrive, tolerance * tolerance, near, curves);
  }
  return {start: scale(curves[0][0], 1 / zoom), curves: curves.map(c => c.slice(1).map(p => scale(p, 1 / zoom))), closed};
}

// A number to the decimal place finer than a tenth of a screen pixel at
// *zoom*, as the shape tools round theirs.
function rounded(value, zoom) {
  const digits = Math.max(0, Math.ceil(Math.log10(10 * zoom)));
  return Number(value.toFixed(digits)) + 0;
}

// A fit as path data at *zoom*: a moveto, a cubic for each curve, and with
// closed a Z, the last curve ending exactly on the first point. Curves that
// round to nothing are left out.
export function strokePath(fit, zoom = 1) {
  if (!fit) return '';
  const at = p => [rounded(p[0], zoom), rounded(p[1], zoom)];
  const start = at(fit.start);
  let d = `M${start.join(' ')}`, end = start, count = 0;
  fit.curves.forEach((curve, i) => {
    const [b, c] = curve.slice(0, 2).map(at), last = fit.closed && i === fit.curves.length - 1 ? start : at(curve[2]);
    if ([b, c, last].every(p => p[0] === end[0] && p[1] === end[1])) return;
    d += ` C${b.join(' ')} ${c.join(' ')} ${last.join(' ')}`; end = last; count++;
  });
  if (!count) return '';
  return fit.closed ? `${d} Z` : d;
}

// How many points a fit's path has, as the status line counts them.
export const strokeNodes = d => (d.match(/[MC]/g) || []).length - (d.endsWith('Z') ? 1 : 0);

// What the status line says while a stroke is drawn.
export function strokeStatus(d) {
  const count = strokeNodes(d);
  return `Pencil · ${count} point${count === 1 ? '' : 's'}`;
}
