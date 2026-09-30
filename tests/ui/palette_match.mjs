// Checks how the command palette finds commands; run by test_palette.py.
import assert from 'node:assert/strict';
import {matchCommands, matchScore, moveHighlight} from '../../src/vectrify/ui/static/palette.js';

const commands = [
  {name: 'Select tool', group: 'Tools'},
  {name: 'Send to back', group: 'Arrange'},
  {name: 'Bring to front', group: 'Arrange'},
  {name: 'Tidy (Optimize nodes)…', group: 'Reference', keywords: 'simplify snap'},
  {name: 'Join paths…', group: 'Actions', keywords: 'merge union'},
  {name: 'Export SVG', group: 'File'},
];
const names = query => matchCommands(commands, query).map(c => c.name);

// An empty query lists everything in order.
assert.deepEqual(names(''), commands.map(c => c.name));
// Starts of the name rank above words inside it, then letters in order.
assert.deepEqual(names('to'), ['Select tool', 'Send to back', 'Bring to front', 'Tidy (Optimize nodes)…']);
assert.deepEqual(names('optim'), ['Tidy (Optimize nodes)…']);
assert.deepEqual(names('EXPORT'), ['Export SVG']);
assert.deepEqual(names('jp'), ['Join paths…']);
// Words in any order, and keywords and groups.
assert.deepEqual(names('front bring'), ['Bring to front']);
assert.deepEqual(names('merge'), ['Join paths…']);
assert.deepEqual(names('arrange'), ['Send to back', 'Bring to front']);
assert.deepEqual(names('zzz'), []);
assert.equal(matchScore('Fit colours…', 'fit colours…'), 100);
assert.ok(matchScore('Déplacer', 'depl') > 0);

// The highlight wraps round the rows.
assert.equal(moveHighlight(0, -1, 3), 2);
assert.equal(moveHighlight(2, 1, 3), 0);
assert.equal(moveHighlight(0, 1, 0), -1);
