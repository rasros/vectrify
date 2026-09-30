// The tool strip keeps to one row: when its controls do not fit, the least
// important go into a "⋯" menu. Pure, so it can be checked outside the
// browser; app.js measures and moves the controls.

// Which controls to collapse, as their indexes in order, given each one's
// *width* and *priority* (1 is kept longest), the *available* width and the
// width the "⋯" button takes when shown. The highest priority number goes
// first and, among equals, the one furthest right.
export function overflowLayout(items, available, moreWidth) {
  const total = items.reduce((sum, item) => sum + item.width, 0);
  if (total <= available) return [];
  const order = items.map((item, index) => ({...item, index}))
    .sort((a, b) => b.priority - a.priority || b.index - a.index);
  const hidden = [];
  let width = total + moreWidth;
  for (const item of order) {
    if (width <= available) break;
    hidden.push(item.index);
    width -= item.width;
  }
  return hidden.sort((a, b) => a - b);
}
