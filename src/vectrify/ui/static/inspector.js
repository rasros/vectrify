// Navigation belongs to the user/tool switch, never to a selection render.
export function selectInspectorTab(document, name, focus = false) {
  for (const tab of document.querySelectorAll('[data-inspector-tab]')) {
    const active = tab.dataset.inspectorTab === name;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
    if (active && focus) tab.focus();
  }
}

export function wireInspector(document) {
  const tabs = [...document.querySelectorAll('[data-inspector-tab]')];
  for (const [index, tab] of tabs.entries()) {
    tab.onclick = () => selectInspectorTab(document, tab.dataset.inspectorTab);
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault(); event.stopPropagation();
      selectInspectorTab(document, tabs[next].dataset.inspectorTab, true);
    });
  }
}

// Palette and keyboard commands can focus fields in a closed tab/disclosure.
export function revealInspectorField(document, field) {
  selectInspectorTab(document, 'object');
  for (let parent = field.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName === 'DETAILS') parent.open = true;
  }
  field.focus();
}
