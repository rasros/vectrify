// Checks which tool strip controls collapse into "⋯"; run by test_strip.py.
import assert from 'node:assert/strict';
import {overflowLayout} from '../../src/vectrify/ui/static/strip.js';

const items = [{width: 100, priority: 1}, {width: 60, priority: 2}, {width: 80, priority: 3}, {width: 40, priority: 3}, {width: 50, priority: 2}];
// Everything fits: nothing collapses, and no "⋯" is needed.
assert.deepEqual(overflowLayout(items, 330, 30), []);
assert.deepEqual(overflowLayout(items, 1000, 30), []);
// Just too narrow: the rightmost of the least important goes, and room is
// made for the "⋯" button too.
assert.deepEqual(overflowLayout(items, 320, 30), [3]);
// Narrower: both of the least important, then the rightmost of the next.
assert.deepEqual(overflowLayout(items, 240, 30), [2, 3]);
assert.deepEqual(overflowLayout(items, 190, 30), [2, 3, 4]);
// The most important goes last.
assert.deepEqual(overflowLayout(items, 60, 30), [0, 1, 2, 3, 4]);
assert.deepEqual(overflowLayout(items, 130, 30), [1, 2, 3, 4]);
// No controls, nothing to collapse.
assert.deepEqual(overflowLayout([], 10, 30), []);
