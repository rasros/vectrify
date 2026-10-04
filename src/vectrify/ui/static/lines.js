// Which line edits a contour's selected points allow: breaking it at a point,
// deleting the segment between two, joining two free ends. Pure, so it can be
// checked outside the browser; the server makes the edits.

const same = (a, b) => a[0] === b[0] && a[1] === b[1];
const end = node => node.values.slice(-2);

// The contour's points in order, as node IDs. A closed contour ending on its
// moveto shows one point for the two nodes: the last stands for both.
export function contourPoints(contour) {
  const ids = contour.nodes.map(node => node.id);
  const twins = contour.closed && ids.length > 2 && same(end(contour.nodes[0]), end(contour.nodes.at(-1)));
  return twins ? ids.slice(1) : ids;
}

// An open contour's first or last point: a free end of a line.
export function freeEnd(contour, id) {
  return !contour.closed && contour.nodes.length > 1 && (contour.nodes[0].id === id || contour.nodes.at(-1).id === id);
}

// Whether the contour can be broken at the point: anywhere on a closed
// contour, between the ends of an open one.
export function breakable(contour, id) {
  return contour.closed || (contour.nodes.some(node => node.id === id) && !freeEnd(contour, id));
}

// Whether some segment of the contour has both its points among *ids*.
export function segmentAmong(contour, ids) {
  const points = contourPoints(contour), twins = points.length < contour.nodes.length;
  const first = contour.nodes[0].id, last = points.at(-1);
  const chosen = new Set([...ids].map(id => twins && id === first ? last : id));
  const pairs = points.slice(1).map((id, i) => [points[i], id]);
  if (contour.closed && points.length > 1) pairs.push([points.at(-1), points[0]]);
  return pairs.some(([a, b]) => a !== b && chosen.has(a) && chosen.has(b));
}
