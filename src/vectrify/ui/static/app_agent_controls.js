

// Agent_controls owns its controls; editor contains the live state shared with other controllers.
export function createAgentControlsController(editor, {
  $,
  EventSource = globalThis.EventSource,
  desktop,
  document = globalThis.document,
  pollAgent,
  pulse,
  request,
  toast,
  window = globalThis.window,
} = {}) {
  let agentEvents = null;

  function showAgent() {
    const s = editor.agentStatus, button = $('agent-toggle');
    button.setAttribute('aria-pressed', String(s.enabled));
    button.classList.toggle('connected', s.enabled && s.connected);
    const text = !s.enabled ? 'Agents off' : s.connected ? `Agent connected${s.last_action ? ' · ' + s.last_action.replace(/^Agent: /, '') : ''}` : 'Agents allowed';
    $('agent-status').textContent = text;
    button.title = s.enabled ? 'Agents may edit this drawing: the MCP server and how to add it' : 'Allow agents to edit this drawing';
    $('agent-enabled').checked = s.enabled;
    $('agent-popover-status').textContent = !s.enabled ? 'Turn this on to let an MCP client such as Claude Code look at and edit this drawing. Each edit shows here. Undo and redo affect only your own changes.' : text + '.';
    $('agent-setup').hidden = !(s.enabled && s.apps);
    $('agent-mcp').hidden = $('agent-claude-code').hidden = !(s.enabled && s.mcp);
    const apps = s.apps || {};
    const fields = {'agent-url': s.mcp?.url, 'agent-command': s.mcp?.command, 'agent-codex': apps.codex, 'agent-codex-command': apps.codex_command, 'agent-claude-desktop': apps.claude_desktop};
    // Keep a selection the person is making.
    for (const [id, value] of Object.entries(fields)) if ($(id).value !== (value || '')) $(id).value = value || '';
    $('agent-codex-config').textContent = apps.codex_config || '~/.codex/config.toml';
    $('agent-claude-desktop-config').textContent = apps.claude_desktop_config || '';
    $('agent-mcp-error').hidden = !(s.enabled && s.mcp_error);
    $('agent-mcp-error').textContent = s.mcp_error || '';
    // The port clients were given was taken: they must be given the new URL.
    const moved = s.enabled ? s.mcp?.moved : null;
    $('agent-mcp-moved').hidden = !moved;
    $('agent-mcp-moved').textContent = moved ? `The MCP server's URL changed: ${moved.old} was taken, so it is at ${moved.new} now. Update clients added with the old URL (copy the command again).` : '';
    $('agent-toggle').classList.toggle('moved', !!moved);
    listenAgents();
  }

  // The push channel, open while agents are allowed.
  function listenAgents() {
    if (desktop) return;
    if (editor.agentStatus.enabled && !agentEvents && editor.session) {
      agentEvents = new EventSource(`/api/events?session=${encodeURIComponent(editor.session)}`);
      agentEvents.onmessage = event => { try { pulse(JSON.parse(event.data)); } catch { /* The poll catches up. */ } };
    } else if (!editor.agentStatus.enabled && agentEvents) {
      agentEvents.close(); agentEvents = null;
    }
  }

  async function setAgents(body) {
    try {
      editor.agentStatus = await request('/api/agent', body);
      editor.agentSeen = editor.flash.seen = editor.agentStatus.changes; showAgent();
      if (editor.agentStatus.enabled) pollAgent();
    } catch (error) { toast(error.message, true); }
  }

  async function toggleAgents() {
    await setAgents({enabled: !editor.agentStatus.enabled});
    if (editor.agentStatus.enabled) openAgentPopover();
  }

  function openAgentPopover() {
    const popover = $('agent-popover'), rect = $('agent-toggle').getBoundingClientRect();
    popover.hidden = false;
    $('agent-toggle').setAttribute('aria-expanded', 'true');
    const width = popover.offsetWidth;
    popover.style.left = `${Math.max(16, Math.min(window.innerWidth - width - 16, rect.right - width))}px`;
    popover.style.bottom = `${window.innerHeight - rect.top + 8}px`;
    $(editor.agentStatus.enabled && editor.agentStatus.mcp ? 'agent-copy' : 'agent-enabled').focus();
  }

  function closeAgentPopover() {
    $('agent-popover').hidden = true;
    $('agent-toggle').setAttribute('aria-expanded', 'false');
  }

  function wire() {
    $('agent-toggle').onclick = async () => {
      if (!$('agent-popover').hidden) { closeAgentPopover(); return; }
      if (!editor.agentStatus.enabled) await setAgents({enabled: true});
      openAgentPopover();
    };

    $('agent-enabled').onchange = () => setAgents({enabled: $('agent-enabled').checked});

    // Each app's setup is text to copy: the editor never edits an app's config.
    for (const button of document.querySelectorAll('#agent-popover .agent-copy')) button.onclick = async () => {
      const field = $(button.dataset.copy);
      try { await navigator.clipboard.writeText(field.value); }
      catch { field.select(); document.execCommand('copy'); }
      toast(`Copied. Once ${button.dataset.app} has it and is restarted, it reaches this window whenever Agents is on.`);
    };

    $('agent-regenerate').onclick = () => setAgents({regenerate: true}).then(() => toast('New token. Clients added with the old one are refused: copy the command again.'));

    for (const id of ['agent-url', 'agent-command', 'agent-codex', 'agent-codex-command', 'agent-claude-desktop']) $(id).onfocus = () => $(id).select();

    $('agent-popover').addEventListener('keydown', event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeAgentPopover(); $('agent-toggle').focus(); } });
      return api;
  }
  const api = {showAgent, listenAgents, setAgents, toggleAgents, openAgentPopover, closeAgentPopover, wire};
  return api;
}
