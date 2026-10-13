import {createEditorState} from '../../src/vectrify/ui/static/app_state.js';
import {createTreeDragController} from '../../src/vectrify/ui/static/app_tree_drag.js';
// Checks where rows dragged in the Objects tree land; run by test_tree_drop.py.
import assert from 'node:assert/strict';

import {dropIndex, dropRefusal, dropTarget, replayDrop} from '../../src/vectrify/ui/static/tree.js';

// The drawing holds g (a, b) then c, back to front; rows are 20px tall.
const spec = [['g', 'root', 0, true], ['a', 'g', 1, false], ['b', 'g', 1, false], ['c', 'root', 0, false]];
const rows = spec.map(([id, parent, depth, container], i) => ({id, parent, depth, container, top: i * 20, height: 20}));

// The upper half of a row drops before it; the middle of a group row into it.
assert.deepEqual(dropTarget(rows, 9, 22, 9), {parent: 'g', before: 'a', line: {y: 20, depth: 1}});
assert.deepEqual(dropTarget(rows, 0, 10, 9), {parent: 'g', into: true});
// Just below a group row is the back of the group.
assert.deepEqual(dropTarget(rows, 0, 18, 9), {parent: 'g', before: 'a', line: {y: 20, depth: 1}});
// Where the group ends, the pointer's depth keeps the drop inside or leaves it.
assert.deepEqual(dropTarget(rows, 9, 58, 9), {parent: 'g', after: 'b', line: {y: 60, depth: 1}});
assert.deepEqual(dropTarget(rows, 0, 58, 9), {parent: 'root', after: 'g', line: {y: 60, depth: 0}});
// Past the ends of the list: the drawing's front and back.
assert.deepEqual(dropTarget(rows, 0, 200, 9), {parent: 'root', after: 'c', line: {y: 80, depth: 0}});
assert.deepEqual(dropTarget(rows, 0, 1, 9), {parent: 'root', before: 'g', line: {y: 0, depth: 0}});
assert.equal(dropTarget([], 0, 0, 9), null);
// A collapsed group's hidden children are absent from the measured rows:
// its middle still accepts objects, and its edges restack beside the group.
const collapsed = [rows[0], {...rows[3], top: 20}];
assert.deepEqual(dropTarget(collapsed, 0, 10, 9), {parent: 'g', into: true});
assert.deepEqual(dropTarget(collapsed, 9, 18, 9), {parent: 'root', after: 'g', line: {y: 20, depth: 0}});
assert.deepEqual(dropTarget(collapsed, 0, 2, 9), {parent: 'root', before: 'g', line: {y: 0, depth: 0}});

const parents = new Map(spec.map(([id, parent]) => [id, parent]));
assert.match(dropRefusal({parent: 'g', into: true}, new Set(['g']), parents, new Set()), /itself/);
assert.match(dropRefusal({parent: 'a', into: true}, new Set(['g']), parents, new Set()), /itself/);
assert.match(dropRefusal({parent: 'g', into: true}, new Set(['c']), parents, new Set(['g'])), /Definitions/);
assert.equal(dropRefusal({parent: 'g', into: true}, new Set(['c']), parents, new Set()), '');
assert.match(dropRefusal(null, new Set(['c']), parents, new Set()), /Drop/);

// Indices count only the parent's children that are not dragged.
assert.equal(dropIndex({parent: 'g', into: true}, ['a', 'b'], new Set(['c'])), 2);
assert.equal(dropIndex({parent: 'g', after: 'a'}, ['a', 'b'], new Set(['a'])), 0);
assert.equal(dropIndex({parent: 'g', before: 'b'}, ['a', 'b'], new Set(['c'])), 1);
assert.equal(dropIndex({parent: 'root', after: 'c'}, ['g', 'c'], new Set(['g'])), 1);

// A drop made while an edit ran lands by the rows it was next to, on the tree
// the edit left, or is refused if those rows are gone or moved.
const objects = spec.map(([id, parent]) => ({id, parent, resource: false}));
const without = (...ids) => objects.filter(item => !ids.includes(item.id));
assert.deepEqual(replayDrop({parent: 'g', after: 'a'}, new Set(['c']), without('b')),
  {ids: new Set(['c']), target: {parent: 'g', after: 'a'}, refusal: ''});
assert.equal(replayDrop({parent: 'g', into: true}, new Set(['c']), without('a', 'b')).refusal, '');
assert.match(replayDrop({parent: 'g', after: 'a'}, new Set(['c']), without('a')).refusal, /changed/);
assert.match(replayDrop({parent: 'g', after: 'a'}, new Set(['c']), objects.map(item => item.id === 'a' ? {...item, parent: 'root'} : item)).refusal, /changed/);
assert.match(replayDrop({parent: 'g', into: true}, new Set(['c']), without('g', 'a', 'b')).refusal, /group/);
assert.equal(replayDrop({parent: 'root', before: 'g'}, new Set(['c']), objects).refusal, '');
// Objects removed meanwhile are left out; with none left, nothing moves.
assert.deepEqual(replayDrop({parent: 'root', after: 'c'}, new Set(['a', 'x']), objects).ids, new Set(['a']));
assert.match(replayDrop({parent: 'root', after: 'c'}, new Set(['x']), objects).refusal, /removed/);
// The tree the edit left is checked too: a group into itself, definitions.
assert.match(replayDrop({parent: 'a', into: true}, new Set(['g']), objects).refusal, /itself/);
assert.match(replayDrop({parent: 'g', into: true}, new Set(['c']), objects.map(item => item.id === 'g' ? {...item, resource: true} : item)).refusal, /Definitions/);

// Tree-row dragging scrolls at the visible edges, but leaving the tree in
// any direction keeps its viewport still while the pointer moves elsewhere.
const list = {scrollTop: 240, classList: {add() {}}, getBoundingClientRect: () => ({left: 50, right: 250, top: 100, bottom: 500})};
const context = Object.assign(createEditorState(), {
  $: () => list, state: {epoch: 'drawing', revision: 0, selection: {objects: ['c']}},
  object: id => objects.find(item => item.id === id),
  treeDropAt: () => ({target: null, refusal: ''}), showTreeDrop() {},
});

context.treeDrag = {item: 'c', x: 150, y: 300, active: false};
const context_tree_drag = createTreeDragController(context, {
  $: (...args) => context.$(...args),
  action: (...args) => context.action(...args),
  later: (...args) => context.later(...args),
  object: (...args) => context.object(...args),
  showTreeDrop: (...args) => context.showTreeDrop(...args),
  toast: (...args) => context.toast(...args),
  treeDropAt: (...args) => context.treeDropAt(...args),
  treeDropLine: context.treeDropLine,
  window: context.window,
});
context.moveTreeDrag = context_tree_drag.moveTreeDrag;
const moveTreeDrag = (context.moveTreeDrag);
const move = (clientX, clientY) => moveTreeDrag({clientX, clientY});
move(150, 200);
assert.equal(list.scrollTop, 240, 'dragging in the middle does not scroll');
move(150, 110);
assert.equal(list.scrollTop, 228, 'dragging near the top scrolls up');
move(150, 490);
assert.equal(list.scrollTop, 240, 'dragging near the bottom scrolls down');
for (const [x, y] of [[40, 110], [260, 490], [150, 90], [150, 510]]) {
  move(x, y);
  assert.equal(list.scrollTop, 240, 'dragging outside the tree does not scroll');
}
