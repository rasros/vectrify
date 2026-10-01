// Checks the two-level selection's transitions; run by test_selection_model.py.
import assert from 'node:assert/strict';
import {boxSelect, clickPoint, dragBox, escapeStep, instancePoints, pickTarget, pointInside, pointKey, pointOwners, pointTargets, pointerTarget, rectInside, scopeChain, selectionStatus, splitKey, switchTool} from '../../src/vectrify/ui/static/selection.js';

// The drawing holds group trees (group pines (a, b), c) and d.
const parents = new Map([['trees', 'root'], ['pines', 'trees'], ['a', 'pines'], ['b', 'pines'], ['c', 'trees'], ['d', 'root']]);

assert.deepEqual(splitKey(pointKey('a', 'node_1')), ['a', 'node_1']);

// Leaving a point tool hides the points and remembers them with the objects.
const hidden = switchTool({objects: ['b', 'a'], points: ['a n1', 'b n2']}, 'nodes', 'select');
assert.deepEqual(hidden.points, []);
assert.deepEqual(hidden.memory, {objects: 'a b', points: ['a n1', 'b n2']});
// Back to a point tool with the same objects, in any order, restores them.
assert.deepEqual(switchTool({objects: ['a', 'b'], points: [], memory: hidden.memory}, 'select', 'nodes'), {points: ['a n1', 'b n2'], memory: null});
// With other objects selected meanwhile, the points are cleared.
assert.deepEqual(switchTool({objects: ['a'], points: [], memory: hidden.memory}, 'knife', 'redraw'), {points: [], memory: null});
// Between two point tools, or two object tools, nothing changes.
assert.deepEqual(switchTool({objects: ['a'], points: ['a n1']}, 'nodes', 'redraw'), {points: ['a n1'], memory: null});
assert.deepEqual(switchTool({objects: ['a'], points: [], memory: hidden.memory}, 'select', 'knife'), {points: [], memory: hidden.memory});

// Clicks pick the outermost group, or within the entered one.
assert.deepEqual(pickTarget('a', null, parents, 'root'), {id: 'trees', scope: null});
assert.deepEqual(pickTarget('a', 'trees', parents, 'root'), {id: 'pines', scope: 'trees'});
assert.deepEqual(pickTarget('a', 'pines', parents, 'root'), {id: 'a', scope: 'pines'});
assert.deepEqual(pickTarget('d', null, parents, 'root'), {id: 'd', scope: null});
// A click outside the entered group leaves it.
assert.deepEqual(pickTarget('d', 'pines', parents, 'root'), {id: 'd', scope: null});
assert.deepEqual(scopeChain('pines', parents, 'root'), ['trees', 'pines']);

// Escape: points to their paths, a path to its group, out of the entered
// group, then to nothing.
const up = state => escapeStep(state, parents, 'root');
assert.deepEqual(up({objects: ['a', 'c'], points: ['a n1'], scope: 'pines'}), {objects: ['a', 'c'], points: [], scope: 'pines'});
assert.deepEqual(up({objects: ['a'], points: [], scope: 'pines'}), {objects: ['pines'], points: [], scope: 'trees'});
assert.deepEqual(up({objects: ['pines'], points: [], scope: 'trees'}), {objects: ['trees'], points: [], scope: null});
assert.deepEqual(up({objects: ['trees', 'd'], points: [], scope: null}), {objects: [], points: [], scope: null});
assert.deepEqual(up({objects: ['a'], points: [], scope: null}), {objects: ['pines'], points: [], scope: null});
assert.deepEqual(up({objects: [], points: [], scope: 'pines'}), {objects: [], points: [], scope: 'trees'});
assert.deepEqual(up({objects: [], points: [], scope: 'trees'}), {objects: [], points: [], scope: null});

// A point click replaces the points, or with Shift toggles one.
assert.deepEqual(clickPoint(['a n1', 'b n2'], 'a n3', false), ['a n3']);
assert.deepEqual(clickPoint(['a n1'], 'b n2', true), ['a n1', 'b n2']);
assert.deepEqual(clickPoint(['a n1', 'b n2'], 'a n1', true), ['b n2']);

// Box select takes what lies wholly inside; Shift toggles it.
const box = dragBox({x: 50, y: 40}, {x: 10, y: 0});
assert.deepEqual(box, {left: 10, top: 0, right: 50, bottom: 40});
assert.ok(rectInside({left: 10, top: 5, right: 50, bottom: 40}, box));
assert.ok(!rectInside({left: 5, top: 5, right: 20, bottom: 20}, box));
assert.ok(pointInside(10, 40, box) && !pointInside(51, 20, box));
assert.deepEqual(boxSelect(['a'], ['b', 'c', 'b'], false), ['b', 'c']);
assert.deepEqual(boxSelect(['a', 'b'], ['b', 'c'], true), ['a', 'c']);

// The status names the level and count.
assert.equal(selectionStatus('objects', ['a', 'b', 'c'], []), 'Objects · 3 selected');
assert.equal(selectionStatus('objects', [], []), 'Objects · nothing selected');
assert.equal(selectionStatus('points', ['a', 'b', 'c'], ['a 1', 'a 2', 'b 1', 'c 1', 'c 2']), 'Points · 5 points in 3 paths');
assert.equal(selectionStatus('points', ['a'], ['a 1']), 'Points · 1 point in 1 path');
assert.equal(selectionStatus('points', ['a', 'b'], []), 'Points · none selected in 2 paths');
assert.equal(selectionStatus('points', [], []), 'Points · select a path');

// Point tools show the points of selected paths and of every path inside a
// selected group, however deep; instances are counted apart.
const drawing = [
  {id: 'trees', tag: 'g', parent: 'root'}, {id: 'pines', tag: 'g', parent: 'trees'},
  {id: 'a', tag: 'path', parent: 'pines'}, {id: 'b', tag: 'path', parent: 'pines'},
  {id: 'c', tag: 'use', parent: 'trees'}, {id: 'd', tag: 'path', parent: 'root'},
  {id: 'defs', tag: 'defs', parent: 'root', resource: true}, {id: 'e', tag: 'path', parent: 'defs', resource: true},
];
assert.deepEqual(pointTargets(['trees'], drawing), {paths: ['a', 'b'], instances: ['c']});
assert.deepEqual(pointTargets(['d', 'pines'], drawing), {paths: ['a', 'b', 'd'], instances: []});
assert.deepEqual(pointTargets(['a', 'pines'], drawing), {paths: ['a', 'b'], instances: []});
assert.deepEqual(pointTargets(['defs', 'e'], drawing), {paths: [], instances: []});
assert.deepEqual(pointTargets([], drawing), {paths: [], instances: []});

// Paths a and b draw geometry G; c its own. A node picked in b is selected in
// b alone, and a twin in a.
const geometryOf = id => ({a: 'G', b: 'G', c: 'H'}[id]);
const shown = [{id: 'a', geometry: 'G', nodes: ['n1', 'n2']}, {id: 'b', geometry: 'G', nodes: ['n1', 'n2']}, {id: 'c', geometry: 'H', nodes: ['m1']}];
const picked = pointOwners(['b n1', 'c m1'], geometryOf);
assert.deepEqual(instancePoints(shown, new Set(['n1', 'm1']), picked), {strong: ['b n1', 'c m1'], twins: ['a n1']});
// A node the server selected that was not picked (a split's new point) goes
// with the path last picked in for its geometry.
assert.deepEqual(instancePoints(shown, new Set(['n1', 'n2']), picked), {strong: ['b n1', 'b n2'], twins: ['a n1', 'a n2']});
// With nothing picked, the first path showing the geometry has it.
assert.deepEqual(instancePoints(shown, new Set(['n2']), pointOwners([], geometryOf)), {strong: ['a n2'], twins: ['b n2']});
// Picked in both, it is selected in both.
assert.deepEqual(instancePoints(shown, new Set(['n1']), pointOwners(['a n1', 'b n1'], geometryOf)), {strong: ['a n1', 'b n1'], twins: []});
// When the path it was picked in is no longer shown, another one has it.
assert.deepEqual(instancePoints(shown.slice(0, 1), new Set(['n1']), picked), {strong: ['a n1'], twins: []});

// Tools act on the path under the pointer; a selection narrows that.
{
  const item = (id, parent, extra = {}) => ({id, parent, tag: 'path', resource: false, locks: [], inherited_locks: [], ...extra});
  const objects = [
    {id: 'g', parent: 'root', tag: 'g', resource: false, locks: [], inherited_locks: []},
    item('a', 'g'), item('b', 'root'), item('locked', 'root', {locks: ['geometry']}), item('def', 'root', {resource: true}),
  ];
  assert.equal(pointerTarget(['b', 'a'], objects, []), 'b');
  assert.equal(pointerTarget(['locked', 'def', 'a'], objects, []), 'a');
  assert.equal(pointerTarget(['b', 'a'], objects, ['g']), 'a');
  assert.equal(pointerTarget(['b'], objects, ['g']), null);
  assert.equal(pointerTarget(['b', 'a'], objects, [], 'g'), 'a');
  assert.equal(pointerTarget([], objects, []), null);
}
