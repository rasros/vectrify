import assert from 'node:assert/strict';
import {errorPixels, errorSize} from '../../src/vectrify/ui/static/reference_render.js';

const actual = new Uint8ClampedArray([
  0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 255,
  0, 255, 0, 255, 0, 0, 255, 255, 64, 0, 0, 255,
]);
const expected = new Uint8ClampedArray(actual.length); expected.fill(0);
for (let p = 3; p < expected.length; p += 4) expected[p] = 255;
const original = actual.slice();
assert.equal(errorPixels(actual, expected), actual, 'reuse the pixel buffer');
assert.deepEqual([...actual], [
  0, 0, 0, 255, 255, 255, 255, 255, 255, 186, 0, 255,
  255, 186, 0, 255, 255, 186, 0, 255, 110, 0, 0, 255,
]);
assert.deepEqual(errorPixels(expected.slice(), original), actual, 'distance is symmetric');
assert.throws(() => errorPixels(new Uint8ClampedArray(4), new Uint8ClampedArray(8)), /same size/);
assert.deepEqual(errorSize([0, 0, 1672, 941], [0, 0, 1672, 941], [1672, 941]), [1672, 941]);
assert.deepEqual(errorSize([0, 0, 6000, 4000], [0, 0, 6000, 4000], [6000, 4000]), [2048, 1365]);
assert.deepEqual(errorSize([40, 20, 120, 80], [0, 0, 1200, 800], [1200, 800]), [120, 80], 'shrinking the artboard keeps reference resolution');
assert.deepEqual(errorSize([0, 0, 5, 10], [0, 0, 5, 10], [5, 5]), [2, 5], 'match Python ties-to-even rounding');
assert.deepEqual(errorSize([0, 0, 1e9, 1e9], [0, 0, 1, 1], [2048, 2048]), [2048, 2048], 'cap a grown artboard');
