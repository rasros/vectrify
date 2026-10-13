

// Actions owns its controls; editor contains the live state shared with other controllers.
export function createActionsController(editor, {
  $,
  applyState,
  drawOverlay,
  renderDrawing,
  renderInspector,
  request,
  setBusy,
  toast,
  window = globalThis.window,
} = {}) {

  function action(command, data = {}, label = 'Applying edit…') {
    // Busy from now, so input meanwhile waits for this edit.
    setBusy('', 1);
    editor.queue = editor.queue.then(async () => {
      if (command !== 'select') editor.clickCycle = null;
      $('busy-label').textContent = label;
      try {
        const ids = command === 'undo' ? editor.state.undo_ids?.slice(-1) : command === 'redo' ? editor.state.redo_ids?.slice(0, 1) : undefined;
        const payload = {command, ...data, ...(ids ? {ids} : {}), selection: editor.state.selection, epoch: editor.state.epoch, revision: editor.state.revision};
        let result = await request('/api/action', payload);
        if (command === 'delete' && result.confirmation === 'delete_pinned') {
          if (!window.confirm(result.message)) return false;
          result = await request('/api/action', {...payload, confirm_pinned: true});
        }
        if (command === 'paste') { editor.scope = null; editor.focusPoint = null; editor.pointMemory = null; }
        await applyState(result);
        return true;
      } catch (error) {
        toast(error.message, true);
        // A genuine overlap can be refused. Catch up automatically so the
        // person can keep editing the live drawing without a manual refresh.
        try { await applyState(await request('/api/session', {session: editor.session})); } catch { /* Keep the last state if disconnected. */ }
        // Discard optimistic dragging even when the backend rejects the command.
        if (editor.state?.svg) renderDrawing();
        renderInspector(); drawOverlay();
        return false;
      } finally { setBusy('', -1); }
    });
    return editor.queue;
  }

  function wire() {
      return api;
  }
  const api = {action, wire};
  return api;
}
