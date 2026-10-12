// Checks which line edits selected points allow; run by test_line_edits.py.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
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

// Add node stays available at either end of an open path. A lone point has
// no segment to subdivide.
const source = readFileSync(new URL('../../src/vectrify/ui/static/app.js', import.meta.url), 'utf8');
const commandStart = source.indexOf("{id: 'split-edge'");
let selected = ['a'], contour = line;
const context = vm.createContext({
  noPoints: () => !selected.length && 'Select points first',
  selectedPoints: () => selected,
  contourAt: () => contour,
  noSelection: () => false,
  level: () => 'points',
  joinCandidates: () => [{id: 'a'}, {id: 'b'}],
});
const command = vm.runInContext(`(${source.slice(commandStart, source.indexOf(",\n  {id: 'delete-contour'", commandStart))})`, context);
assert.ok(!command.disabled());
selected = ['c'];
assert.ok(!command.disabled());
selected = ['a', 'b'];
assert.ok(!command.disabled());
contour = {closed: false, nodes: [node('a', 'M', 0, 0)]};
selected = ['a'];
assert.equal(command.disabled(), 'This contour has no edge to add a node on');
selected = [];
assert.equal(command.disabled(), 'Select points first');

// Grouping one object and combining the selected paths from Nodes are useful
// object operations too, without requiring a tool or selection change.
const groupStart = source.indexOf("{id: 'group'");
const group = vm.runInContext(`(${source.slice(groupStart, source.indexOf(",\n  {id: 'ungroup'", groupStart))})`, context);
assert.ok(!group.disabled());
const combineStart = source.indexOf("{id: 'combine-paths'");
const combine = vm.runInContext(`(${source.slice(combineStart, source.indexOf(",\n  {id: 'convert-lines'", combineStart))})`, context);
assert.ok(!combine.disabled());
