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

// Each join names what it does. A filled compound path can already be one
// object: Merge shapes unions its contours directly, without a colour dialog;
// two or more paths merge through the dialog. Lines join at their ends, and
// two selected points join each other, each from a command of its own.
const edits = [];
let candidates = [{id: 'a'}], fills = candidates, lines = [], points = [], pointLevel = false;
const joinContext = ({
  joinCandidates: () => candidates,
  fillPaths: () => fills,
  linePaths: () => lines,
  twoEnds: () => points.length === 2,
  selectedPoints: () => points,
  noPoints: () => !points.length && 'Select points first',
  level: () => pointLevel ? 'points' : 'objects',
  action: (...args) => edits.push(args),
  pointPairs: () => points.map(key => key.split(' ')),
  openJoin: () => edits.push(['dialog']),
  state: {selection: {objects: ['a']}},
});
joinContext.zoom = 1;
const joining = createJoinController(joinContext, {
  action:joinContext.action, fillPaths:joinContext.fillPaths, joinCandidates:joinContext.joinCandidates,
  linePaths:joinContext.linePaths, openJoin:joinContext.openJoin, plural:() => '', toast() {},
});
const joinRegistry = createCommandRegistryController(joinContext, {
  ...joinContext, joinEnds:joining.joinEnds, mergeShapes:joining.mergeShapes, joinOpensDialog:joining.joinOpensDialog,
});
const merge = joinRegistry.commandById.get('merge-shapes');
const lineEnds = joinRegistry.commandById.get('join-line-ends');
const joinPoints = joinRegistry.commandById.get('join-points');
const shortcut = joinRegistry.commandById.get('join');
assert.equal(joinRegistry.commandRefusal(merge), '');
assert.equal(joinRegistry.commandName(merge), 'Merge shapes');
merge.run();
assert.equal(edits.at(-1)[0], 'join_paths');
assert.equal(joinRegistry.commandRefusal(lineEnds), 'Select stroked lines without a fill, or groups that contain them');
candidates = [{id: 'a'}, {id: 'b'}]; fills = candidates;
assert.equal(joinRegistry.commandName(merge), 'Merge shapes…');
merge.run(); assert.equal(edits.at(-1)[0], 'dialog');
assert.equal(joinRegistry.commandName(shortcut), 'Merge shapes…');
// Lines alone do not merge by area: Join line ends joins them, and Ctrl/⌘ J
// picks it.
candidates = [{id: 'a'}, {id: 'b'}]; fills = []; lines = candidates;
assert.equal(joinRegistry.commandRefusal(merge), 'These are lines without a fill: use Join line ends');
assert.equal(joinRegistry.commandRefusal(lineEnds), '');
assert.equal(joinRegistry.commandName(shortcut), 'Join line ends');
shortcut.run(); assert.equal(edits.at(-1)[0], 'join_ends');
// Two selected points join in a point tool, whatever paths hold them.
pointLevel = true;
assert.equal(joinRegistry.commandRefusal(joinPoints), 'Select points first');
assert.equal(joinRegistry.commandRefusal(shortcut), 'Select points first');
points = ['a n1', 'b n2'];
assert.equal(joinRegistry.commandRefusal(joinPoints), '');
assert.equal(joinRegistry.commandName(shortcut), 'Join points');
shortcut.run(); assert.deepEqual(edits.at(-1).slice(0, 2), ['join_two_ends', {points: [['a', 'n1'], ['b', 'n2']]}]);
points = ['a n1'];
assert.equal(joinRegistry.commandRefusal(joinPoints), 'Select exactly two points to join');
// With nothing to join, the shortcut says what would be.
pointLevel = false; points = []; candidates = []; lines = []; fills = [];
assert.equal(joinRegistry.commandRefusal(shortcut), 'Select paths to merge, or lines whose ends to join');
