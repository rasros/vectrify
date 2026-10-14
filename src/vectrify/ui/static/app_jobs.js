// Operation dialogs own their controls and job lifetimes. Editor data is read
// through accessors when a dialog opens or a request runs.
import {soleObject} from './util.js';

const $ = id => document.getElementById(id);

// A dialog around one operation job: start, poll, stop, preview, choose, apply.
// Elements are found by id as `${prefix}-name`; missing optional ones are skipped.
export function createJobDialog(prefix, {start, describe, applied, choiceLabel = null},
  {getState, operation, applyResult, toast}) {
  let context = null, poll = null, runLabel = '';
  const show = id => $(prefix+'-'+id);
  const fail = error => { show('error').textContent = error.message; show('error').hidden = false; };
  const ready = () => { show('settings').disabled = false; show('run').disabled = false; show('stop').hidden = true; };
  // A run that failed leaves nothing to discard; show only the error.
  const failedRun = error => { show('progress').hidden = true; show('close').textContent = 'Cancel'; fail(error); ready(); };
  function choose(index) {
    const result = context?.results[index]; if (!result) return;
    context.choice = index;
    for (const [key, url] of Object.entries(result.previews)) { const img = show(key); if (img) img.src = url; }
    show('metrics').textContent = describe(result);
    show('apply').hidden = !result.changed;
  }
  function showChoices() {
    const box = show('choices'); if (!box) return;
    box.replaceChildren();
    const results = context.results;
    if (!choiceLabel || results.length < 2) return;
    results.forEach((result, index) => {
      if (index && !result.changed) return;
      const label = document.createElement('label'); label.className = 'toggle';
      const input = document.createElement('input');
      input.type = 'radio'; input.name = prefix+'-choice'; input.checked = index === 0;
      const current = context;
      input.onchange = () => { if (current === context) choose(index); };
      label.append(input, ' '+choiceLabel(result, index));
      box.append(label);
    });
  }
  async function refresh() {
    const current = context; if (!current?.job) return;
    try {
      const job = await operation('status', {job:current.job, preview:true});
      if (current !== context) return;
      show('meter').max = job.steps || 1; show('meter').value = job.step; show('status').textContent = job.message;
      if (job.status === 'running') { poll = setTimeout(refresh, 700); return; }
      ready();
      if (job.status === 'failed') throw new Error(job.error);
      if (job.status !== 'ready') return;
      current.results = [job.result, ...(job.alternatives || [])];
      showChoices(); choose(0);
      show('previews').hidden = false;
      show('close').textContent = current.results.some(result => result.changed) ? 'Discard' : 'Close';
      show('run').textContent = `${runLabel} again`;
    } catch (error) { if (current === context) failedRun(error); }
  }
  return {
    open(summary) {
      const state = getState();
      context = {epoch:state.epoch, revision:state.revision, job:null, results:[], choice:0};
      if (show('summary') && summary !== undefined) show('summary').textContent = summary;
      for (const id of ['error','previews','progress','apply','stop']) show(id).hidden = true;
      show('apply').disabled = false;
      show('run').textContent = runLabel; show('close').textContent = 'Cancel'; ready();
      show('dialog').showModal();
    },
    async run() {
      const current = context; if (!current) return;
      for (const id of ['error','apply','previews']) show(id).hidden = true;
      show('settings').disabled = true; show('run').disabled = true;
      // Quick methods finish inside the start request; say so while it runs.
      show('progress').hidden = false; show('meter').removeAttribute('value'); show('status').textContent = 'Working…';
      try {
        if (current.job) { await operation('discard', {job:current.job}); current.job = null; }
        if (current !== context) return;
        const job = await operation('start', {epoch:current.epoch, revision:current.revision, selection:current.selection, ...start()});
        if (current !== context) { await operation('discard', {job:job.id}); return; }
        current.job = job.id; show('progress').hidden = false; show('stop').hidden = job.status !== 'running';
        show('close').textContent = 'Cancel & discard';
        await refresh();
      } catch (error) { if (current === context) failedRun(error); }
    },
    wire() {
      runLabel = show('run').textContent;
      show('run').onclick = () => this.run();
      show('stop').onclick = async () => {
        const current = context; if (!current?.job) return;
        try {
          await operation('stop', {job:current.job});
          if (current === context) show('stop').hidden = true;
        } catch (error) { if (current === context) fail(error); }
      };
      show('apply').onclick = async () => {
        const current = context; if (!current?.job) return;
        show('apply').disabled = true;
        try {
          const result = await operation('apply', {job:current.job, choice:current.choice});
          current.job = null;
          // A successful server edit still belongs in the drawing if its
          // dialog closed meanwhile; it must not close a newer dialog.
          await applyResult(result);
          if (current === context) show('dialog').close();
          toast(applied);
        } catch (error) { if (current === context) fail(error); }
        finally { if (current === context || !context) show('apply').disabled = false; }
      };
      show('close').onclick = () => show('dialog').close();
      show('dialog').addEventListener('close', () => {
        clearTimeout(poll); const job = context?.job; context = null;
        if (job) operation('discard', {job}).catch(error => toast(error.message, true));
      });
      return this;
    },
  };
}

export function operationDialogs({getState, getObject, waitForEdits,
  selectionBounds, viewReport, operation, applyResult, toast}) {
  const jobDialog = (prefix, config) => createJobDialog(prefix, config,
    {getState, operation, applyResult, toast});
  // Tidy combines edge-seeking and gradient fitting in Fit path, alongside
  // Simplify (the operation's method is still called nodes).
  const NODE_STEPS = ['shape', 'detail', 'simplify'];
  const NODE_OVERRIDES = ['tolerance', 'movement', 'detail-gain', 'margin', 'budget', 'rounds', 'workers', 'steps', 'gain', 'seconds', 'allowance'];
  const STEP_NAMES = {shape:'fit', snap:'snap', simplify:'simplify'};
  const nodeSteps = () => {
    const steps = Object.fromEntries(NODE_STEPS.map(step => [step, $('nodes-'+step).checked]));
    return {...steps, detail:steps.shape && steps.detail, snap:steps.shape};
  };
  function syncNodeSteps() {
    const steps = nodeSteps();
    const customRun = $('nodes-custom-run').checked;
    for (const setting of NODE_OVERRIDES) $('nodes-'+setting).disabled = !customRun;
    $('nodes-shared').disabled = !customRun;
    $('nodes-detail').disabled = !steps.shape || !getState().reference;
    // A step that is off keeps its options visible but dimmed.
    for (const card of document.querySelectorAll('#nodes-settings .step-card[data-step]')) {
      const on = steps[card.dataset.step];
      card.classList.toggle('off', !on);
      for (const input of card.querySelectorAll('.two-fields input')) if (input.id !== 'nodes-detail') input.disabled = !customRun || !on;
    }
    $('nodes-detail-gain').disabled = !customRun || !steps.shape || !steps.detail;
    $('nodes-steps').disabled = !customRun || !steps.shape;
    $('nodes-apply').hidden = true; $('nodes-previews').hidden = true;
  }
  const nodesDialog = jobDialog('nodes', {
    start: () => {
      const steps = nodeSteps();
      const customRun = $('nodes-custom-run').checked;
      const region = $('nodes-in-view').checked ? viewReport()?.region : null;
      return {action:'improve', method:'nodes', scope:'selection',
        permissions:{geometry:true, structure:(steps.shape && steps.detail) || steps.simplify, paint:true},
        settings:{...steps, ...(customRun ? {tolerance:Number($('nodes-tolerance').value),
          movement:Number($('nodes-movement').value),
          steps:Number($('nodes-steps').value), workers:Number($('nodes-workers').value),
          detail_gain:Number($('nodes-detail-gain').value), gain:Number($('nodes-gain').value), margin:Number($('nodes-margin').value),
          allowance:Number($('nodes-allowance').value), budget:Number($('nodes-budget').value),
          shared:$('nodes-shared').checked,
          seconds:Number($('nodes-seconds').value)} : {}), ...(region ? {region} : {})},
        budget:customRun ? {steps:Number($('nodes-rounds').value)} : {}};
    },
    describe: ({changed, metrics}) => {
      if (!changed) return 'No step improved the paths within these settings. They are unchanged.';
      const points = `${metrics.before.nodes.toLocaleString()} → ${metrics.after.nodes.toLocaleString()} points`;
      const fit = metrics.reference ? `difference from the reference ${errorChange(metrics, 'difference')}` : 'the look is kept within the tolerance';
      const order = metrics.steps.map(step => STEP_NAMES[step] || step).join(' → ');
      const skipped = Object.values(metrics.skipped || {});
      const note = skipped.length ? ` Some paths were not fitted: ${[...new Set(skipped)].join('; ')}.` : '';
      const late = metrics.out_of_time ? ' Stopped at the time limit.' : '';
      const followed = metrics.followed ? ` ${metrics.followed} neighbouring path${metrics.followed === 1 ? '' : 's'} moved along shared edges.` : '';
      const linked = metrics.linked ? ' Linked outlines, instances and clipping keep following their shared geometry.' : '';
      return `${points} · ${fit} · ${order}.${followed}${linked}${late}${note} Apply keeps this result as one undoable edit.`;
    },
    applied: 'Paths tidied. Undo restores them.',
  }).wire();
  for (const step of NODE_STEPS) $('nodes-'+step).addEventListener('change', syncNodeSteps);
  $('nodes-custom-run').addEventListener('change', () => {
    if (!$('nodes-custom-run').checked) {
      for (const setting of NODE_OVERRIDES) {
        const input = $('nodes-'+setting);
        input.value = setting === 'tolerance' && !getState().reference ? '1' : input.defaultValue;
      }
      $('nodes-shared').checked = $('nodes-shared').defaultChecked;
    }
    syncNodeSteps();
  });
  for (const id of ['nodes-tolerance', 'nodes-rounds', 'nodes-workers', 'nodes-steps', 'nodes-movement', 'nodes-detail-gain', 'nodes-gain', 'nodes-margin', 'nodes-seconds', 'nodes-allowance', 'nodes-budget', 'nodes-in-view', 'nodes-shared']) $(id).addEventListener('input', () => { $('nodes-apply').hidden = true; $('nodes-previews').hidden = true; });
  // Whether paths, or groups that may hold them, are selected for Tidy.
  const tidyTargets = () => getState().selection.objects.some(id => ['path', 'use', 'g'].includes(getObject(id)?.tag));
  async function openTidy() {
    await waitForEdits();
    const reference = Boolean(getState().reference);
    // With nothing selected Tidy works on what is in view.
    const targets = tidyTargets();
    if (!targets) $('nodes-in-view').checked = true;
    $('nodes-in-view').disabled = !targets;
    // Fitting comes back on when a reference is added.
    if (reference && $('nodes-shape').disabled) {
      $('nodes-shape').checked = true;
      $('nodes-detail').checked = true;
      $('nodes-simplify').checked = false;
    }
    $('nodes-shape').disabled = !reference;
    if (!reference) $('nodes-shape').checked = false;
    if (!reference) { $('nodes-detail').checked = false; $('nodes-simplify').checked = true; }
    // With a reference the error budget decides and the tolerance is a cap;
    // without one the tolerance decides alone, so its default is tighter.
    const tolerance = $('nodes-tolerance');
    if (tolerance.value === (reference ? '1' : '3')) tolerance.value = reference ? '3' : '1';
    $('nodes-reference-caption').textContent = reference ? 'Reference' : 'Original';
    $('nodes-description').textContent = reference
      ? 'Clean up paths against the reference. Preview the result before applying.'
      : 'Simplify paths while keeping their shape. Add a reference to enable fitting.';
    syncNodeSteps();
    nodesDialog.open(targets ? selectionSummary() : 'The paths painting in view.');
  }

  const contactDialog = jobDialog('contact', {
    start: () => ({action:'snap', method:'edges', bounds:selectionBounds(), permissions:{geometry:true, structure:true},
      settings:{tolerance:Number($('contact-distance').value)}}),
    describe: ({changed, metrics}) => changed
      ? `${metrics.edges} touching edge spans. Apply snaps their nodes and curve handles together as one undoable edit.`
      : metrics.edges ? 'These edges already meet exactly.' : 'No touching edges within this distance. Try a larger contact distance.',
    applied: 'Edges snapped. The paths stay independent: editing one no longer moves the other.',
  }).wire();
  async function openSnapEdges() { await waitForEdits(); contactDialog.open(); }
  // Changing a setting invalidates the preview shown for the old one.
  for (const [prefix, ids] of [['contact', ['contact-distance']]]) {
    for (const id of ids) $(id).addEventListener('input', () => {
      $(prefix+'-apply').hidden = true; $(prefix+'-previews').hidden = true;
    });
  }

  // Generate: new shapes from the reference, placed as one group.
  const generateSettings = {
    cel: () => ({regions:Number($('cel-regions').value), tolerance:Number($('cel-tolerance').value), line_width:Number($('cel-line-width').value), strokes:$('cel-strokes').checked, outline:$('cel-outline').checked, fit_colours:$('cel-fit-colours').checked, gradients:$('cel-gradients').checked}),
    'colour-regions': () => {
      const outlines = $('regions-outlines').value;
      return {colours:Number($('regions-colours').value), min_pixels:Number($('regions-min-pixels').value), tolerance:Number($('regions-tolerance').value),
        preserve_outlines:outlines !== 'none', outline_style:outlines === 'none' ? 'preserve' : outlines, geometry_cleanup:$('regions-cleanup').checked};
    },
  };
  function showGenerateMethod() {
    for (const panel of document.querySelectorAll('[data-generate]')) panel.hidden = panel.dataset.generate !== $('generate-method').value;
  }
  $('generate-method').onchange = showGenerateMethod;
  const generateDialog = jobDialog('generate', {
    start: () => {
      const name = $('generate-method').value;
      return {action:'generate', method:name, scope:$('generate-scope').value, permissions:{structure:true}, settings:generateSettings[name]()};
    },
    describe: ({changed, metrics}) => changed ? `${metrics.shapes.toLocaleString()} shapes · reference error ${errorChange(metrics)}. Apply adds them as one undoable edit.` : 'Nothing was generated. Try other settings.',
    applied: 'Generated shapes added. Undo removes them.',
    choiceLabel: (result, index) => `Drawing ${index + 1} · error ${result.metrics.after.error.toFixed(5)}`,
  }).wire();
  async function openGenerate() {
    await waitForEdits();
    const group = oneObject()?.tag === 'g';
    $('generate-scope').value = group ? 'selection' : 'drawing';
    $('generate-scope').options[1].disabled = !group;
    showGenerateMethod(); generateDialog.open('');
  }

  function errorChange(metrics, key = 'error') {
    const before = metrics.before[key], after = metrics.after[key];
    const change = before > 0 ? 100*(before-after)/before : 0;
    return change >= 0 ? `reduced ${change.toFixed(1)}%` : `increased ${(-change).toFixed(1)}%`;
  }
  const oneObject = () => soleObject(getState().selection, getObject);
  const selectionSummary = () => oneObject()?.label || `${getState().selection.objects.length} selected objects`;
  const coloursDialog = jobDialog('colours', {
    start: () => ({action:'improve', method:'colours', permissions:{paint:true},
      settings:{passes:Number($('colours-passes').value), resolution:Number($('colours-resolution').value), fill:$('colours-fill').value}}),
    describe: result => {
      const m = result.metrics, gradients = m.gradients ? ` (${m.gradients} as ${m.gradients === 1 ? 'a gradient' : 'gradients'})` : '';
      return result.changed ? `${m.objects} of ${m.considered} fills changed${gradients} · reference error ${errorChange(m)}. Apply keeps it as one undoable edit.` : 'The colours already fit the reference.';
    },
    applied: 'Fills fitted. Undo restores the previous fills.',
  }).wire();
  // Fit colours and Fit gradient are one dialog, opened with its fill kind chosen.
  const showColoursFill = () => { $('colours-title').textContent = $('colours-fill').value === 'linear' ? 'Fit gradient' : 'Fit colours'; };
  $('colours-fill').onchange = showColoursFill;
  async function openColours(fill = 'flat') { await waitForEdits(); $('colours-fill').value = fill; showColoursFill(); coloursDialog.open(selectionSummary()); }
  const cleanupDialog = jobDialog('cleanup', {
    start: () => ({action:'simplify', method:'cleanup', bounds:selectionBounds(), permissions:{geometry:true, structure:true}}),
    describe: result => {
      const c = result.metrics.cleanup;
      return result.changed ? `${c.paths_before} → ${c.paths_after} paths (${c.paths_merged} merged, ${c.duplicate_paths_removed + c.empty_paths_removed} removed) · ${c.vertices_removed} redundant vertices removed. Apply keeps it as one undoable edit.` : 'Nothing to clean up in the selection.';
    },
    applied: 'Geometry cleaned up. Undo restores the original paths.',
  }).wire();
  async function openCleanup() { await waitForEdits(); cleanupDialog.open(selectionSummary()); }
  return {tidyTargets, openTidy, openSnapEdges, openGenerate, openColours, openCleanup};
}
