// The largest artboard side the dialogs accept, in document units.
export const MAX_SIDE = 100000;

// Why an artboard [x, y, width, height] cannot be used, or '' when it can.
export function artboardRefusal([x, y, width, height]) {
  if (![x, y, width, height].every(Number.isFinite)) return 'Enter a number in every field';
  if (width <= 0 || height <= 0) return 'The width and height must be more than 0';
  if (width > MAX_SIDE || height > MAX_SIDE) return `The width and height can be at most ${MAX_SIDE.toLocaleString('en')}`;
  return '';
}

// A size or position as a field shows it: no float noise.
const field = value => String(Math.round(value * 1000) / 1000);

// Document owns New drawing's and the Artboard's dialogs; editor contains the
// live state shared with other controllers.
//
// New drawing replaces the drawing as opening a file does (projectFiles). The
// artboard is the drawing's page, the root's viewBox: changing it keeps the
// artwork and the reference where they are, so nothing moves, scales or is
// cropped, and it undoes as one step.
export function createDocumentController(editor, {
  $,
  action,
  files,
  toast,
} = {}) {

  function showError(dialog, message) { $(`${dialog}-error`).textContent = message; $(`${dialog}-error`).hidden = !message; }

  function openNewDrawing() {
    if (!editor.state) return;
    const [, , width, height] = editor.state.bounds;
    $('new-width').value = field(width); $('new-height').value = field(height);
    showError('new', '');
    $('new-dialog').showModal();
  }

  async function confirmNewDrawing() {
    const size = [0, 0, Number($('new-width').value), Number($('new-height').value)];
    const reason = artboardRefusal(size);
    if (reason) { showError('new', reason); return; }
    $('new-confirm').disabled = true;
    try {
      if (await files.newDrawing(size[2], size[3])) $('new-dialog').close();
    } finally { $('new-confirm').disabled = false; }
  }

  function openArtboard() {
    if (!editor.state) return;
    const [x, y, width, height] = editor.state.bounds;
    $('artboard-x').value = field(x); $('artboard-y').value = field(y);
    $('artboard-width').value = field(width); $('artboard-height').value = field(height);
    showError('artboard', '');
    $('artboard-dialog').showModal();
  }

  // Apply the artboard the fields give, or with *fit* the artwork's painted
  // bounds; either is one undoable edit.
  async function applyArtboard(fit = false) {
    if (editor.pending) return;
    const box = ['x', 'y', 'width', 'height'].map(name => Number($(`artboard-${name}`).value));
    const reason = fit ? '' : artboardRefusal(box);
    if (reason) { showError('artboard', reason); return; }
    const buttons = ['artboard-confirm', 'artboard-fit', 'artboard-cancel'].map($);
    for (const button of buttons) button.disabled = true;
    try {
      if (await action('artboard', fit ? {fit: 'artwork'} : {box}, fit ? 'Fitting artboard…' : 'Changing artboard…')) {
        $('artboard-dialog').close();
        const [, , width, height] = editor.state.bounds;
        toast(`Artboard ${field(width)} × ${field(height)}. The artwork and the reference stay where they are.`);
      } else showError('artboard', $('toast-message').textContent);
    } finally { for (const button of buttons) button.disabled = false; }
  }

  function fitArtboard() { return applyArtboard(true); }

  function wire() {
    $('new-drawing').onclick = openNewDrawing;
    $('new-cancel').onclick = () => $('new-dialog').close();
    $('new-confirm').onclick = confirmNewDrawing;
    $('dimensions').onclick = openArtboard;
    $('artboard-cancel').onclick = () => $('artboard-dialog').close();
    $('artboard-confirm').onclick = () => applyArtboard();
    $('artboard-fit').onclick = fitArtboard;
    $('artboard-dialog').addEventListener('cancel', event => { if (editor.pending) event.preventDefault(); });
    // Enter in a field applies, as the dialog's primary button.
    for (const [dialog, run] of [['new-dialog', confirmNewDrawing], ['artboard-dialog', () => applyArtboard()]])
      $(dialog).addEventListener('keydown', event => { if (event.key === 'Enter' && event.target.tagName === 'INPUT') { event.preventDefault(); run(); } });
    return api;
  }
  const api = {openNewDrawing, confirmNewDrawing, openArtboard, applyArtboard, fitArtboard, wire};
  return api;
}
