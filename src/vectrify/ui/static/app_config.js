
export const NS = 'http://www.w3.org/2000/svg';

export const SNAP_RADIUS = 8;

// How near, in screen pixels, the selection's frame is grabbed to resize it.
export const FRAME_REACH = 6;

// Two clicks at one spot within this many milliseconds are a double-click.
export const DOUBLE_CLICK = 400;

export const names = {select: 'Select', move: 'Move', nodes: 'Nodes', path: 'Draw path', rect: 'Rectangle', ellipse: 'Ellipse', line: 'Line', knife: 'Knife', redraw: 'Redraw outline', hand: 'Pan'};

// A one-line hint for the tools with few controls of their own.
export const hints = {select: 'Click to select · Drag to box select · Shift adds or removes', move: 'Drag the selection to move · Drag frame edges to resize · Shift adds or removes', nodes: '', path: 'Click for corners · Drag for curves · Click the first point to close · Enter finishes', rect: 'Drag to draw · Shift: square · Alt: from the centre · Click places the size set here · Escape cancels', ellipse: 'Drag to draw · Shift: circle · Alt: from the centre · Click places the size set here · Escape cancels', line: 'Drag to draw · Shift: 45° steps · Alt: from the centre · Click places the length set here · Escape cancels', knife: 'Drag a line across shapes to cut them, only the selected ones if any are · Shift snaps to 15°', redraw: 'Draw along the reference edge from a path\'s outline back to it, a selected path if any are · Shift replaces the longer way round · Escape cancels', hand: 'Drag to pan · Scroll to zoom'};

// Dragging rows in the tree restacks them, or moves them into a group. The
// drag carries the whole selection when it starts on a selected row.
export const TREE_INDENT = 9;

// How far apart, in screen pixels, two line ends may be for Join line ends: zoom
// out to join wider gaps.
export const JOIN_REACH = 12;

// The deepest zoom, 25600%, enough to place points within a hundredth of a
// pixel.
export const MAX_ZOOM = 256;

// How near, in screen pixels, a press in Nodes must be to a point or handle
// to take it: the nearest one within reach wins, so a point need not be hit
// exactly. *nearPoint* is the one under the pointer, drawn larger.
export const POINT_REACH = 10;

// The least length, in screen pixels, a handle is drawn at.
export const HANDLE_SPREAD = 16;

export const AGENT_POLL_MS = 5000, AGENT_RETRY_MS = 120;

// The objects an agent's change touched flash briefly once the page shows it:
// the person's selection stays theirs, never the agent's.
export const FLASH_MS = 1500;