import {pointOwners} from './selection.js';
// Session data and transient editor state. Controllers access these fields live.
export function createEditorState() {
    return {
      state: undefined,
      session: undefined,
      tool: 'select',
      zoom: 1,
      pan: {x: 0, y: 0},
      drag: null,
      reference: null,
      referenceVersion: null,
      showClipping: false,
      clipConsumers: new Map(),
      geometries: new Map(),
      hoverPath: null,
      focusPoint: null,
      pointMemory: null,
      activeHandle: null,
      owners: pointOwners([], () => null),
      scope: null,
      clickCycle: null,
      lastPick: null,
      pathDraft: [],
      pathHover: null,
      region: null,
      regionHover: null,
      eyedropperHover: null,
      redrawHover: null,
      selectionBox: null,
      keepRatio: false,
      // Align's reference, its key object and the selection it last saw.
      align: {reference: 'selection', key: null, seen: []},
      pending: 0,
      queue: Promise.resolve(),
      dirty: false,
      space: false,
      svgElements: new Map(),
      // The Objects tree: folded groups and a row being dragged.
      tree: {collapsed: new Set(), drag: null, dragEnded: false},
      nearPoint: null,
      nearAnchor: null,
      // Agents: their status, the change last caught up with, and the flash
      // of what they touched.
      agent: {
        status: {enabled: false, connected: false, last_action: null, changes: 0, touched: []},
        seen: 0,
        flash: {ids: [], start: 0, timer: null, seen: null},
      },
      pathHoles: new Map(),
      overlayFrame: null,
      holding: null,
      referenceView: 'overlay',
    };
}
