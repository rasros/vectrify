// Checks which line edits selected points allow; run by test_line_edits.py.
import assert from 'node:assert/strict';
import {breakable, contourPoints, freeEnd, segmentAmong} from '../../src/vectrify/ui/static/lines.js';

const node = (id, command, ...values) => ({id, command, values});
const line = {closed: false, nodes: [node('a', 'M', 0, 0), node('b', 'L', 10, 0), node('c', 'L', 20, 0)]};
const square = {closed: true, nodes: [node('a', 'M', 0, 0), node('b', 'L', 10, 0), node('c', 'L', 10, 10)]};
// A closed contour ending on its moveto: one point for the two nodes.
const ring = {closed: true, nodes: [node('a', 'M', 0, 0), node('b', 'L', 10, 0), node('c', 'L', 10, 10), node('d', 'L', 0, 0)]};

assert.deepEqual(contourPoints(line), ['a', 'b', 'c']);
assert.deepEqual(contourPoints(ring), ['b', 'c', 'd']);
// A line's ends are free; a closed contour has none.
assert.ok(freeEnd(line, 'a') && freeEnd(line, 'c') && !freeEnd(line, 'b'));
assert.ok(!freeEnd(square, 'a'));
// It breaks between its ends, and a closed contour anywhere.
assert.ok(breakable(line, 'b') && !breakable(line, 'a') && !breakable(line, 'c'));
assert.ok(breakable(square, 'a'));
// A segment needs both its points, neighbours on the contour.
assert.ok(segmentAmong(line, new Set(['a', 'b'])));
assert.ok(!segmentAmong(line, new Set(['a', 'c'])));
assert.ok(!segmentAmong(line, new Set(['b'])));
// The closing line joins the last point to the first.
assert.ok(segmentAmong(square, new Set(['c', 'a'])));
// The moveto of a contour ending on it is its last point.
assert.ok(segmentAmong(ring, new Set(['a', 'b'])));
assert.ok(segmentAmong(ring, new Set(['c', 'a'])));

// Large point selections must read a contour's ids once, rather than
// constructing its complete id list again for every selected point.
let idReads = 0;
const dense = {closed: false, nodes: Array.from({length: 2000}, (_, i) => ({
  get id() { idReads++; return `n${i}`; }, values: [i, 0],
}))};
assert.ok(segmentAmong(dense, new Set(Array.from({length: 2000}, (_, i) => `n${i}`))));
assert.ok(idReads < 4000, `read ids ${idReads} times for 2000 nodes`);
