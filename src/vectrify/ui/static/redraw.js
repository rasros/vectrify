// Redraw outline: where a stroke attaches to the selected path and which
// stretch it replaces. Pure functions over plain points, so they can be
// checked outside the browser; the server decides the same way.

// How near, in screen pixels, a stroke's end attaches to a point of the
// outline, else to the outline itself.
export const NODE_REACH = 6, REACH = 10;
const SAMPLES = 16;

function bezier(controls, t) {
  if (controls.length === 2) {
    const [a, b] = controls;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  const u = 1 - t, [a, b, c, d] = controls;
  const w = [u*u*u, 3*u*u*t, 3*u*t*t, t*t*t];
  return [0, 1].map(k => w[0]*a[k] + w[1]*b[k] + w[2]*c[k] + w[3]*d[k]);
}

// Each contour as a polyline in the frame *map* takes local points to. Every
// sample names the segment it is on, by the node the segment ends at, its t
// there and its distance along the contour. A closed contour's closing line
// is named by its moveto and left out when the last node is on the first.
export function contourLines(geometry, map = p => p) {
  return geometry.subpaths.map(subpath => {
    const nodes = subpath.nodes, segments = [];
    for (let i = 1; i < nodes.length; i++) {
      const node = nodes[i], v = node.values;
      const handles = node.command === 'C' ? [[v[0], v[1]], [v[2], v[3]]] : [];
      segments.push({id: node.id, controls: [nodes[i-1].values.slice(-2), ...handles, v.slice(-2)]});
    }
    const first = nodes[0].values, last = nodes.at(-1).values;
    if (subpath.closed && nodes.length > 1 && (first[0] !== last.at(-2) || first[1] !== last.at(-1)))
      segments.push({id: nodes[0].id, controls: [last.slice(-2), first.slice()]});
    const points = [];
    let s = 0;
    segments.forEach((segment, i) => {
      // A line is drawn by its ends alone.
      const controls = segment.controls.map(map), count = controls.length === 2 ? 1 : SAMPLES;
      for (let k = i ? 1 : 0; k <= count; k++) {
        const t = k / count, [x, y] = bezier(controls, t), previous = points.at(-1);
        if (previous) s += Math.hypot(x - previous.x, y - previous.y);
        points.push({x, y, node: segment.id, t, s, end: k === count || (k === 0 && !subpath.closed)});
      }
    });
    return {id: subpath.id, closed: subpath.closed, points, length: s};
  }).filter(line => line.points.length > 1);
}

// Where (x, y) attaches: the nearest point of the outline within NODE_REACH
// screen pixels, else the nearest place on it within REACH, on *only*'s
// contour when given. *pixel* is a screen pixel's size in the lines' frame.
export function attach(lines, x, y, pixel, only = null) {
  let node = null, place = null;
  for (const line of lines) {
    if (only && line.id !== only) continue;
    const points = line.points;
    for (const p of points) {
      const distance = Math.hypot(p.x - x, p.y - y);
      if (p.end && distance <= NODE_REACH * pixel && (!node || distance < node.distance))
        node = {contour: line.id, node: p.node, t: p.t, x: p.x, y: p.y, s: p.s, distance};
    }
    for (let i = 1; i < points.length; i++) {
      const a = points[i-1], b = points[i], dx = b.x - a.x, dy = b.y - a.y, size = dx*dx + dy*dy;
      const along = size ? Math.max(0, Math.min(1, ((x - a.x) * dx + (y - a.y) * dy) / size)) : 0;
      const px = a.x + along * dx, py = a.y + along * dy, distance = Math.hypot(px - x, py - y);
      if (distance > REACH * pixel || (place && distance >= place.distance)) continue;
      // The place's t on its segment; a sample ending a segment starts the next.
      const t = b.node === a.node ? a.t + along * (b.t - a.t) : along * b.t;
      place = {contour: line.id, node: b.node, t, x: px, y: py, s: a.s + along * (b.s - a.s), distance};
    }
  }
  return node || place;
}

// The part of *line* between attachments *a* and *b* that the stroke
// replaces, as points from a to b: along an open contour the one between
// them; round a closed one the shorter way, or the longer with *longWay*.
export function stretch(line, a, b, longWay = false) {
  if (!a || !b || a.contour !== line.id || b.contour !== line.id || Math.abs(a.s - b.s) < 1e-9) return null;
  const total = line.length;
  let forward = b.s >= a.s;
  if (line.closed) {
    const onward = ((b.s - a.s) % total + total) % total;
    forward = (onward <= total - onward) !== longWay;
  }
  const inside = (s, from, to) => from <= to ? s > from && s < to : s > from || s < to;
  const [from, to] = forward ? [a.s, b.s] : [b.s, a.s];
  // A closed contour's last sample is its first again.
  const middle = line.points.filter(p => line.closed ? p.s < total && inside(p.s, from, to) : p.s > Math.min(from, to) && p.s < Math.max(from, to));
  // Round a closed contour the stretch may pass its start: order by distance
  // from where it begins.
  const offset = p => line.closed ? ((p.s - from) % total + total) % total : p.s - from;
  middle.sort((p, q) => offset(p) - offset(q));
  const points = [[a.x, a.y], ...(forward ? middle : middle.reverse()).map(p => [p.x, p.y]), [b.x, b.y]];
  return points;
}
