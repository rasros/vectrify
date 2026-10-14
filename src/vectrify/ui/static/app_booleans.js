import {BOOLEANS, booleanDescription, booleanDone, booleanOperands, booleanRefusal} from './booleans.js';

// Booleans owns Union, Subtract, Intersect and Exclude; editor contains the
// live state shared with other controllers. Before anything changes, each
// command's tooltip names the target and the cutters, and while one is
// hovered or focused the canvas outlines them: the target solid, the
// cutters dashed.
export function createBooleansController(editor, {
  action,
  document = globalThis.document,
  drawOverlay,
  filledPath,
  localToOverlay,
  noSelection,
  selectionContour,
  svgElement,
  toast,
  xmlElement,
} = {}) {
  // The selected objects in stacking order, back to front.
  const selectedObjects = () => editor.state.objects.filter(item => editor.state.selection.objects.includes(item.id));

  const refusal = () => noSelection() || booleanRefusal(selectedObjects(), item => filledPath(item.id));

  const description = operation => booleanDescription(operation, editor.state && !refusal() ? selectedObjects() : []);

  async function run(operation) {
    const items = selectedObjects();
    if (await action(operation, {}, `${BOOLEANS[operation].verb}…`)) toast(booleanDone(operation, items));
  }

  // The operands of the hovered command, outlined by role.
  function drawRoles() {
    if (!editor.booleanHover || refusal()) return;
    const {target, cutters} = booleanOperands(selectedObjects());
    for (const [item, role] of [[target, 'target'], ...cutters.map(item => [item, 'cutter'])]) {
      const element = svgElement(item.id), matrix = element && localToOverlay(element), contour = matrix && selectionContour(element);
      if (!contour) continue;
      contour.removeAttribute('transform');
      const group = xmlElement('g', {transform: matrix.toString(), class: `boolean-${role}`, 'aria-hidden': 'true'});
      group.append(contour);
      editor.overlayFrame.content.append(group);
    }
  }

  function hover(operation) {
    if (editor.booleanHover === operation) return;
    editor.booleanHover = operation;
    if (editor.state) drawOverlay();
  }

  const commandAt = event => {
    const id = event.target?.closest?.('[data-command]')?.dataset.command;
    return BOOLEANS[id] ? id : null;
  };

  function wire() {
    for (const name of ['pointerover', 'focusin']) document.addEventListener(name, event => hover(commandAt(event)));
    for (const name of ['pointerout', 'focusout']) document.addEventListener(name, event => { if (commandAt(event)) hover(null); });
    // Running the command, or a menu closing under the pointer, ends the hover.
    document.addEventListener('pointerdown', () => hover(null), true);
    return api;
  }
  const api = {selectedObjects, refusal, description, run, drawRoles, wire};
  return api;
}
