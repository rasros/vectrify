// Checks the selection frame's parts and the scale a drag gives; run by test_resize_frame.py.
import assert from 'node:assert/strict';
import {CURSORS, frameHandle, nearestEdge, resizeScale} from '../../src/vectrify/ui/static/resize.js';

const box = {left: 100, top: 100, right: 300, bottom: 200};
// Edges and corners within reach, either side of the line; inside; outside.
assert.equal(frameHandle(box, 104, 150, 6), 'w');
assert.equal(frameHandle(box, 95, 150, 6), 'w');
assert.equal(frameHandle(box, 306, 150, 6), 'e');
assert.equal(frameHandle(box, 200, 99, 6), 'n');
assert.equal(frameHandle(box, 200, 205, 6), 's');
assert.equal(frameHandle(box, 98, 103, 6), 'nw');
assert.equal(frameHandle(box, 303, 98, 6), 'ne');
assert.equal(frameHandle(box, 302, 204, 6), 'se');
assert.equal(frameHandle(box, 100, 200, 6), 'sw');
assert.equal(frameHandle(box, 200, 150, 6), 'inside');
assert.equal(frameHandle(box, 107, 150, 6), 'inside');
assert.equal(frameHandle(box, 93, 150, 6), null);
assert.equal(frameHandle(box, 200, 210, 6), null);
// Past the end of an edge, beyond the corner's reach, is outside.
assert.equal(frameHandle(box, 104, 90, 6), null);
// A box narrower than the reach on both sides: the nearer edge wins.
const thin = {left: 100, top: 100, right: 104, bottom: 200};
assert.equal(frameHandle(thin, 101, 150, 6), 'w');
assert.equal(frameHandle(thin, 103, 150, 6), 'e');
assert.equal(frameHandle(thin, 98, 150, 6), 'w');
// Every part has its cursor: arrows along the axes it resizes.
assert.equal(CURSORS.e, 'ew-resize'); assert.equal(CURSORS.w, 'ew-resize');
assert.equal(CURSORS.n, 'ns-resize'); assert.equal(CURSORS.s, 'ns-resize');
assert.equal(CURSORS.nw, 'nwse-resize'); assert.equal(CURSORS.se, 'nwse-resize');
assert.equal(CURSORS.ne, 'nesw-resize'); assert.equal(CURSORS.sw, 'nesw-resize');
assert.equal(CURSORS.inside, 'move');

const near = (actual, expected) => {
  for (const [key, value] of Object.entries(expected)) {
    if (typeof value === 'object') near(actual[key], value);
    else assert.ok(Math.abs(actual[key] - value) < 1e-9, `${key}: ${actual[key]} != ${value}`);
  }
};
// An edge scales its axis only, about the opposite edge.
near(resizeScale(box, 'e', {x: 400, y: 0}), {sx: 1.5, sy: 1, anchor: [100, 150], box: {left: 100, right: 400, top: 100, bottom: 200}});
near(resizeScale(box, 'w', {x: 200, y: 0}), {sx: .5, sy: 1, anchor: [300, 150], box: {left: 200, right: 300}});
near(resizeScale(box, 'n', {x: 0, y: 50}), {sx: 1, sy: 1.5, anchor: [200, 200], box: {top: 50, bottom: 200}});
// A corner scales both, about the opposite corner.
near(resizeScale(box, 'se', {x: 500, y: 250}), {sx: 2, sy: 1.5, anchor: [100, 100], box: {right: 500, bottom: 250}});
near(resizeScale(box, 'nw', {x: 200, y: 150}), {sx: .5, sy: .5, anchor: [300, 200]});
// Shift keeps the ratio: at a corner by the axis that changed most, at an
// edge about the middle of the other axis.
near(resizeScale(box, 'se', {x: 500, y: 210}, {keepRatio: true}), {sx: 2, sy: 2, anchor: [100, 100], box: {right: 500, bottom: 300}});
near(resizeScale(box, 'se', {x: 310, y: 300}, {keepRatio: true}), {sx: 2, sy: 2});
near(resizeScale(box, 'e', {x: 500, y: 0}, {keepRatio: true}), {sx: 2, sy: 2, anchor: [100, 150], box: {top: 50, bottom: 250}});
// Alt resizes from the centre: the opposite edge moves the other way.
near(resizeScale(box, 'e', {x: 350, y: 0}, {fromCentre: true}), {sx: 1.5, sy: 1, anchor: [200, 150], box: {left: 50, right: 350}});
near(resizeScale(box, 'sw', {x: 0, y: 250}, {fromCentre: true, keepRatio: true}), {sx: 2, sy: 2, anchor: [200, 150], box: {left: 0, right: 400, top: 50, bottom: 250}});
// Dragged past the anchor, the box keeps the minimum size instead of flipping.
near(resizeScale(box, 'e', {x: 50, y: 0}, {minimum: 2}), {sx: .01, box: {left: 100, right: 102}});
// An axis with no size does not scale.
const line = {left: 0, top: 10, right: 100, bottom: 10};
near(resizeScale(line, 'se', {x: 200, y: 50}), {sx: 2, sy: 1});
near(resizeScale(line, 'se', {x: 200, y: 50}, {keepRatio: true}), {sx: 2, sy: 1});

// Snapping takes the nearest edge within the radius.
assert.equal(nearestEdge(103, [0, 100, 105, 300], 4), 105);
assert.equal(nearestEdge(103, [0, 100, 108], 4), 100);
assert.equal(nearestEdge(150, [0, 100, 200], 4), null);
