import {AGENT_POLL_MS, AGENT_RETRY_MS, FLASH_MS} from './app_config.js';

// Agents owns its controls; editor contains the live state shared with other controllers.
export function createAgentsController(editor, {
  DOMPoint = globalThis.DOMPoint,
  applyState,
  drawOverlay,
  getComputedStyle = globalThis.getComputedStyle,
  localToOverlay,
  request,
  showAgent,
  stage,
  svgElement,
  toast,
  window = globalThis.window,
  xmlElement,
} = {}) {
  let nextPulse = null;

  let pulsing = false;

  let viewTimer = null;

  let agentPoll = null;

  function flashTouched(touched) {
    const ids = [...new Set(touched.filter(t => t.change > editor.agent.flash.seen).flatMap(t => t.ids))];
    editor.agent.flash.seen = Math.max(editor.agent.flash.seen, ...touched.map(t => t.change));
    if (!ids.length) return;
    clearTimeout(editor.agent.flash.timer);
    editor.agent.flash = {...editor.agent.flash, ids, start: performance.now(), timer: setTimeout(() => { editor.agent.flash.ids = []; drawOverlay(); }, FLASH_MS)};
    drawOverlay();
  }

  function drawFlash() {
    if (!editor.agent.flash.ids.length) return;
    // Redrawing the overlay keeps the fade where it was.
    const delay = `${-Math.min(FLASH_MS, performance.now() - editor.agent.flash.start)}ms`;
    for (const id of editor.agent.flash.ids) {
      const element = svgElement(id), matrix = localToOverlay(element);
      if (!element?.getBBox || !matrix) continue;
      try {
        const b = element.getBBox();
        if (!b.width && !b.height) continue;
        const style = getComputedStyle(element);
        const pad = style.stroke === 'none' ? 0 : (parseFloat(style.strokeWidth) || 0) / 2;
        const pts = [[b.x-pad,b.y-pad], [b.x+b.width+pad,b.y-pad], [b.x+b.width+pad,b.y+b.height+pad], [b.x-pad,b.y+b.height+pad]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(matrix));
        const shape = xmlElement('polygon', {points: pts.map(p => `${p.x},${p.y}`).join(' '), class: 'agent-flash', 'data-object': id, 'aria-hidden': 'true'});
        shape.style.animationDelay = delay;
        editor.overlayFrame.content.append(shape);
      } catch { /* Resource elements have no display bounds. */ }
    }
  }

  // What the window shows, for an agent's describe(): the visible part of the
  // drawing in its own units [x, y, w, h], the zoom (screen pixels per unit),
  // the canvas size, the tool, the entered group and the reference view.
  function viewReport() {
    if (!editor.state) return null;
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return null;
    return {region: [editor.state.bounds[0] - editor.pan.x / editor.zoom, editor.state.bounds[1] - editor.pan.y / editor.zoom, w / editor.zoom, h / editor.zoom], zoom: editor.zoom, pixels: [w, h], tool: editor.tool, entered: editor.scope, reference_view: editor.reference ? editor.referenceView : null, reference_opacity: editor.reference ? editor.reference.opacity : null};
  }

  function reportView() {
    if (!editor.agent.status.enabled) return;
    clearTimeout(viewTimer);
    viewTimer = setTimeout(pollAgent, 250);
  }

  async function pulse(result) {
    if (!editor.state) return;
    if (pulsing) { nextPulse = result; return; }
    editor.agent.status = result.agent; showAgent();
    // What agents did before the page opened is not news.
    if (editor.agent.flash.seen === null) editor.agent.flash.seen = editor.agent.status.changes;
    // An edit the page did not make: take the session's state as it is now,
    // once the person is not mid-gesture (until then, look again soon).
    const behind = editor.agent.status.changes !== editor.agent.seen || result.epoch !== editor.state.epoch || result.project_version !== editor.state.project_version;
    if (!behind) return;
    if (editor.drag || editor.pending > 0) { schedulePoll(AGENT_RETRY_MS); return; }
    editor.agent.seen = editor.agent.status.changes;
    const touched = editor.agent.status.touched || [];
    pulsing = true;
    editor.queue = editor.queue.then(async () => {
      const next = await request('/api/session', {session: editor.session});
      await applyState(next);
      flashTouched(touched);
    }).catch(error => toast(error.message, true));
    try { await editor.queue; } finally {
      pulsing = false;
      const waiting = nextPulse; nextPulse = null;
      if (waiting) pulse(waiting);
    }
  }

  function schedulePoll(ms) {
    clearTimeout(agentPoll);
    if (editor.agent.status.enabled) agentPoll = setTimeout(pollAgent, ms);
  }

  async function pollAgent() {
    clearTimeout(agentPoll);
    // Looked again by now unless a pulse wants a sooner look.
    agentPoll = null;
    try { await pulse(await request('/api/poll', {view: viewReport()})); } catch { /* The next poll tries again. */ }
    if (agentPoll === null) schedulePoll(AGENT_POLL_MS);
  }

  function wire() {
    // The desktop window runs this with each pulse.
    window.vectrifyPulse = result => { if (result?.session === editor.session) pulse(result); };
      return api;
  }
  const api = {flashTouched, drawFlash, viewReport, reportView, pulse, schedulePoll, pollAgent, wire};
  return api;
}
