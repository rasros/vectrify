import {inputQueue} from './input.js';

// Transport owns its controls; editor contains the live state shared with other controllers.
export function createTransportController(editor, {
  $,
  document = globalThis.document,
  fetch = globalThis.fetch,
  location = globalThis.location,
  window = globalThis.window,
} = {}) {
  let toastTimer = undefined;

  function toast(message, error = false) {
    clearTimeout(toastTimer); $('toast-message').textContent = message;
    $('toast').classList.toggle('error', error); $('toast').hidden = false;
    if (!error) toastTimer = setTimeout(() => $('toast').hidden = true, 4500);
  }

  function setBusy(label, delta) {
    editor.pending += delta; $('busy').hidden = editor.pending === 0;
    if (label) $('busy-label').textContent = label;
    document.body.setAttribute('aria-busy', String(editor.pending > 0 || input.waiting > 0));
  }

  // Input while an edit runs, or while a canvas drag is under way, waits for it:
  // keys, clicks and commands run in order afterwards, on the state it leaves.
  // Only event handlers queue input; what they run calls the editor directly.
  const idle = () => new Promise(resolve => {
    const check = () => { if (!editor.pending && !editor.drag) resolve(); else setTimeout(check, 25); };
    check();
  });

  const input = inputQueue(() => editor.pending > 0 || !!editor.drag, idle);

  function later(fn) { return queued(input.run(fn)); }

  // Like later(), but merging into input of the same *kind* still waiting
  // (inputQueue's coalesce): repeated keys then make one edit, not a backlog.
  function laterMerged(kind, value, merge, fn) { return queued(input.coalesce(kind, value, merge, fn)); }

  function queued(result) {
    document.body.setAttribute('aria-busy', String(editor.pending > 0 || input.waiting > 0));
    Promise.resolve(result).catch(() => {}).finally(() => document.body.setAttribute('aria-busy', String(editor.pending > 0 || input.waiting > 0)));
    return result;
  }

  // In the desktop window (vectrify[desktop]) the page calls Python through
  // pywebview's bridge instead of the local server; the bridge appears a moment
  // after the page loads.
  const desktop = new URLSearchParams(location.search).has('desktop');

  const bridge = desktop ? new Promise(resolve => {
    if (window.pywebview?.api) resolve(window.pywebview.api);
    else window.addEventListener('pywebviewready', () => resolve(window.pywebview.api), {once: true});
  }) : null;

  async function request(path, data = {}) {
    if (bridge) {
      const {status, body} = await (await bridge).request(path, data, editor.session || '');
      if (status >= 400) throw new Error(body.error || `Request failed (${status})`);
      return body;
    }
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Vectrify-Session': editor.session || ''}, body: JSON.stringify(data)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    return result;
  }

  function wire() {
    $('toast-close').onclick = () => $('toast').hidden = true;
      return api;
  }
  const api = {toast, setBusy, idle, input, later, laterMerged, desktop, bridge, request, wire};
  return api;
}
