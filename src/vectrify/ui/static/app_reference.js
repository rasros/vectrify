import {errorView as createErrorView, loadErrorImage} from './reference_error.js';

// Reference owns its controls; editor contains the live state shared with other controllers.
export function createReferenceController(editor, {
  $,
  errorView = createErrorView,
  FileReader = globalThis.FileReader,
  action,
  document = globalThis.document,
  fit,
  loadReference,
  localStorage = globalThis.localStorage,
  renderInspector,
  reportView,
  request,
  scheduleStrip,
  stage,
  toast,
  viewReport,
} = {}) {

  // The Reference panel, below the objects, holds the reference image, the
  // view and the tools that match the drawing to it. The view is the drawing
  // alone, the reference over it, the reference alone, or their pixel error.
  const VIEWS = ['drawing', 'overlay', 'reference', 'error'];

  const errorStatus = {loading: 'Reference error · loading…', ready: 'Reference error · black matches; red → white differs', failed: 'Reference error · unavailable'};

  const referenceError = errorView($('reference-error'), phase => {
    $('reference-status').textContent = errorStatus[phase];
  }, error => toast(error.message, true), async () => {
    const {image} = await request('/api/reference-error', {epoch: editor.state.epoch, revision: editor.state.revision, view: viewReport()});
    return loadErrorImage(image);
  });

  function refreshErrorView() {
    const reference = editor.reference && editor.referenceView === 'error' ? {data_url: editor.reference.data_url, box: editor.reference.box} : null;
    referenceError.update(editor.state?.svg, editor.state?.bounds, reference);
  }

  // The reference lies over its own place in the drawing, its box: the
  // artboard it was fitted to. After the artboard changes it stays there,
  // under the artwork, and shows only where it overlaps the artboard.
  function placeReference() {
    const image = $('reference-image'), box = editor.reference?.box, bounds = editor.state?.bounds;
    if (!box || !bounds) { for (const name of ['left', 'top', 'width', 'height', 'clipPath']) image.style[name] = ''; return; }
    const [ax, ay, aw, ah] = bounds, [bx, by, bw, bh] = box;
    const percent = value => `${value * 100}%`;
    image.style.left = percent((bx - ax) / aw); image.style.top = percent((by - ay) / ah);
    image.style.width = percent(bw / aw); image.style.height = percent(bh / ah);
    // Insets, as shares of the image, trim what lies off the artboard.
    const inset = [ay - by, bx + bw - ax - aw, by + bh - ay - ah, ax - bx].map((value, side) => percent(Math.max(0, value) / (side % 2 ? bw : bh)));
    image.style.clipPath = inset.some(value => value !== '0%') ? `inset(${inset.join(' ')})` : '';
  }

  function showReference() {
    reportView();
    placeReference();
    $('reference-controls').hidden=!editor.reference; $('remove-reference').hidden=!editor.reference;
    $('reference-name').textContent=editor.reference?.name || 'No reference';
    $('reference-name').title=editor.reference?.name || '';
    $('add-reference').textContent=editor.reference ? 'Replace…' : 'Load…';
    $('reference-thumb').hidden=!editor.reference; $('reference-thumb-empty').hidden=!!editor.reference;
    const only = !!editor.reference && editor.referenceView === 'reference', error = !!editor.reference && editor.referenceView === 'error';
    const enabled = !!editor.reference && editor.referenceView !== 'drawing';
    const visible = enabled && (only || error || editor.reference.opacity > 0);
    $('reference-image').hidden=!(only || (enabled && !error));
    // Keep SVG clipping resources visible to the selection overlay, even when
    // the artwork is hidden. Visibility would also hide the clips' geometry.
    $('drawing').style.opacity = only || error ? '0' : '';
    $('drawing').style.pointerEvents = only ? 'none' : '';
    stage.classList.toggle('reference-only-visible', only);
    stage.classList.toggle('reference-error-visible', error);
    const toggle = $('reference-toggle');
    toggle.hidden=!editor.reference;
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.title=`Show ${{drawing: 'the reference overlay', overlay: 'the reference only', reference: 'the reference error', error: 'the drawing only'}[editor.referenceView]} (W; Shift+W cycles backward)`;
    toggle.setAttribute('aria-label', toggle.title);
    if (!error) $('reference-status').textContent=only ? 'Reference only' : enabled ? `Reference overlay · ${Math.round(editor.reference.opacity*100)}%` : 'Drawing only';
    $('reference-state').textContent=editor.reference ? {drawing: 'Hidden', overlay: `${Math.round(editor.reference.opacity*100)}%`, reference: 'Alone', error: 'Error'}[editor.referenceView] : 'None';
    for (const button of document.querySelectorAll('[data-view]')) button.setAttribute('aria-pressed', String(button.dataset.view === editor.referenceView));
    stage.classList.toggle('reference-visible', visible);
    if(editor.reference){$('reference-image').src=editor.reference.data_url;$('reference-thumb').src=editor.reference.data_url;$('reference-image').style.opacity=only ? 1 : editor.reference.opacity;$('reference-opacity').value=Math.round(editor.reference.opacity*100);$('reference-percent').textContent=`${Math.round(editor.reference.opacity*100)}%`;}
    else{$('reference-image').removeAttribute('src');$('reference-thumb').removeAttribute('src');}
    refreshErrorView();
    scheduleStrip();
  }

  function setView(view) { editor.referenceView = view; showReference(); }

  // W cycles forward through the views; Shift+W cycles backward.
  function cycleReference(direction = 1) {
    if (editor.reference) setView(VIEWS[(VIEWS.indexOf(editor.referenceView) + direction + VIEWS.length) % VIEWS.length]);
  }

  // The panel folds to its heading; a newly loaded reference opens it again.
  function foldReference(folded) {
    $('reference-panel').classList.toggle('folded', folded);
    $('reference-body').hidden = folded;
    $('reference-fold').setAttribute('aria-expanded', String(!folded));
    try { localStorage.setItem('vectrify-reference-folded', folded ? '1' : ''); } catch { /* Folding is only remembered where storage works. */ }
  }

  async function removeReference(){if(await action('reference',{reference:null})){editor.reference=null;showReference();renderInspector();}}

  function wire() {
    for (const button of document.querySelectorAll('[data-view]')) button.onclick = () => setView(button.dataset.view);

    try { if (localStorage.getItem('vectrify-reference-folded')) foldReference(true); } catch { /* Open by default. */ }

    $('reference-fold').onclick = () => foldReference(!$('reference-body').hidden);

    $('reference-toggle').onclick=()=>cycleReference();

    $('add-reference').onclick=()=>$('reference-file').click();

    $('reference-file').onchange=async event=>{
      const file=event.target.files[0];event.target.value='';if(!file)return;
      const reader=new FileReader();reader.onload=async()=>{const value={name:file.name,data_url:reader.result,opacity:.5};if(await action('reference',{reference:value},'Loading reference…')){
        // The editor fits the image to the artboard: it may pad it, or size an
        // empty artboard to it, so show what it kept.
        editor.referenceView='overlay';foldReference(false);await loadReference();fit();}};reader.readAsDataURL(file);
    };

    $('remove-reference').onclick=removeReference;

    $('reference-opacity').oninput=event=>{if(editor.reference){editor.reference.opacity=Number(event.target.value)/100;if(editor.referenceView==='drawing')editor.referenceView='overlay';showReference();}};

    $('reference-opacity').onchange=()=>{if(editor.reference)action('reference',{reference: editor.reference});};
      return api;
  }
  const api = {VIEWS, errorStatus, referenceError, refreshErrorView, placeReference, showReference, setView, cycleReference, foldReference, removeReference, wire};
  return api;
}
