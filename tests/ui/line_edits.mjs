// Checks which line edits selected points allow; run by test_line_edits.py.
import assert from 'node:assert/strict';
import {createCommandRegistryController} from '../../src/vectrify/ui/static/app_command_registry.js';
import {createJoinController} from '../../src/vectrify/ui/static/app_join.js';
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
let selected = ['a'], contour = line;
const context = ({
  noPoints: () => !selected.length && 'Select points first',
  selectedPoints: () => selected,
  contourAt: () => contour,
  noSelection: () => false,
  level: () => 'points',
  joinCandidates: () => [{id: 'a'}, {id: 'b'}],
});
const registry = createCommandRegistryController(context, {
  noPoints:context.noPoints, selectedPoints:context.selectedPoints, contourAt:context.contourAt,
  noSelection:context.noSelection, level:context.level, joinCandidates:context.joinCandidates,
});
const command = registry.commandById.get('split-edge');
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
const group = registry.commandById.get('group');
assert.ok(!group.disabled());
const combine = registry.commandById.get('combine-paths');
assert.ok(!combine.disabled());

// A filled compound path can already be one object: Join must remain visible
// and union its contours directly, without routing to line ends or a colour dialog.
const edits = [];
let candidates = [{id: 'a'}], fills = candidates, lines = [], points = false;
const joinContext = ({
  joinCandidates: () => candidates,
  fillPaths: () => fills,
  linePaths: () => lines,
  twoEnds: () => points,
  level: () => 'objects',
  action: (...args) => edits.push(args),
  pointPairs: () => [['a', 'n1'], ['a', 'n2']],
  joinEnds: () => edits.push(['join_ends']),
  openJoin: () => edits.push(['dialog']),
});
joinContext.zoom = 1;
const joining = createJoinController(joinContext, {
  action:joinContext.action, fillPaths:joinContext.fillPaths, joinCandidates:joinContext.joinCandidates,
  linePaths:joinContext.linePaths, openJoin:joinContext.openJoin, twoEnds:joinContext.twoEnds,
  pointPairs:joinContext.pointPairs, plural:() => '', toast() {},
});
const joinRegistry = createCommandRegistryController(joinContext, {
  ...joinContext, join:joining.join, joinOpensDialog:joining.joinOpensDialog,
});
const joinCommand = joinRegistry.commandById.get('join');
assert.ok(!joinCommand.disabled());
assert.equal(joinCommand.label(), 'Join');
joinCommand.run();
assert.equal(edits.at(-1)[0], 'join_paths');
candidates = [{id: 'a'}, {id: 'b'}]; fills = candidates;
assert.equal(joinCommand.label(), 'Join…');
joinCommand.run(); assert.equal(edits.at(-1)[0], 'dialog');
candidates = [{id: 'a'}]; fills = []; lines = candidates;
joinCommand.run(); assert.equal(edits.at(-1)[0], 'join_ends');
points = true;
joinCommand.run(); assert.equal(edits.at(-1)[0], 'join_two_ends');
points = false; candidates = []; lines = [];
assert.ok(joinCommand.disabled());
