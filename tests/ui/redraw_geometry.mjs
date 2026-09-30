// Checks where a redraw stroke attaches and what it replaces; run by
// test_redraw.py.
import assert from 'node:assert/strict';
import {attach, contourLines, stretch} from '../../src/vectrify/ui/static/redraw.js';

const node = (id, command, ...values) => ({id, command, values});
const square = {subpaths: [{id: 's', closed: true, nodes: [
  node('a', 'M', 0, 0), node('b', 'L', 100, 0), node('c', 'L', 100, 100), node('d', 'L', 0, 100)]}]};
const [line] = contourLines(square);
assert.equal(line.length, 400);

// Near a point the stroke attaches to it, as the end of its segment; the
// start of a closed contour is the end of its closing line.
assert.deepEqual(pick(attach([line], 103, 2, 1)), {node: 'b', t: 1, x: 100, y: 0});
assert.deepEqual(pick(attach([line], 2, -1, 1)), {node: 'a', t: 1, x: 0, y: 0});
// Else it attaches to the nearest place on the outline, within reach.
assert.deepEqual(pick(attach([line], 50, 8, 1)), {node: 'b', t: 0.5, x: 50, y: 0});
assert.deepEqual(pick(attach([line], 0, 25, 1)), {node: 'a', t: 0.75, x: 0, y: 25});
assert.equal(attach([line], 50, 12, 1), null);
// Reach is in screen pixels: zoomed out, a screen pixel spans more.
assert.equal(attach([line], 50, 12, 2).node, 'b');
assert.equal(attach([line], 50, 8, 1, 'other'), null);

// Round a closed contour the shorter way is replaced, else the longer.
const top = attach([line], 50, 1, 1), right = attach([line], 99, 50, 1);
assert.deepEqual(stretch(line, top, right), [[50, 0], [100, 0], [100, 50]]);
const long = stretch(line, top, right, true);
assert.deepEqual([long[0], long.at(-1)], [[50, 0], [100, 50]]);
assert.ok(long.some(([x, y]) => x === 0 && y === 0) && !long.some(([x, y]) => x === 100 && y === 0));
// Drawn the other way, it runs from the stroke's start to its end still.
assert.deepEqual(stretch(line, right, top), [[100, 50], [100, 0], [50, 0]]);
// Across the contour's start.
const left = attach([line], 1, 50, 1);
assert.deepEqual(stretch(line, left, top), [[0, 50], [0, 0], [50, 0]]);
assert.equal(stretch(line, top, top), null);

// Along an open contour only the part between the two places.
const open = contourLines({subpaths: [{id: 'o', closed: false, nodes: [
  node('p', 'M', 0, 0), node('q', 'L', 100, 0), node('r', 'L', 200, 0)]}]});
assert.deepEqual(pick(attach(open, 1, 1, 1)), {node: 'q', t: 0, x: 0, y: 0});
const from = attach(open, 150, 3, 1), to = attach(open, 50, 3, 1);
assert.deepEqual(stretch(open[0], from, to), [[150, 0], [100, 0], [50, 0]]);
assert.deepEqual(stretch(open[0], from, to, true), stretch(open[0], from, to));

// Curves are followed through the frame the path is drawn in.
const curve = contourLines({subpaths: [{id: 'c', closed: false, nodes: [
  node('p', 'M', 0, 0), node('q', 'C', 0, 10, 10, 10, 10, 0)]}]}, ([x, y]) => [2 * x + 5, 2 * y]);
const peak = attach(curve, 15, 16, 1);
assert.equal(peak.node, 'q');
assert.ok(Math.abs(peak.t - 0.5) < 0.01 && Math.abs(peak.y - 15) < 0.01);

function pick(found) {
  return found && {node: found.node, t: Math.round(found.t * 1e9) / 1e9, x: found.x, y: found.y};
}
