

// Paint owns its controls; editor contains the live state shared with other controllers.
export function createPaintController(editor, {
  $,
  CSS = globalThis.CSS,
  action,
  colorHex,
  document = globalThis.document,
  drawing,
  getComputedStyle = globalThis.getComputedStyle,
  later,
  object,
  oneObject,
  renderCommands,
  renderFillGradient,
  renderNodeInspector,
  renderStrokeControls,
  renderTransform,
  scheduleStrip,
  selectObject,
  svgElement,
} = {}) {

  function paintReference(element) {
    const href = element?.getAttribute('href') || element?.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
    return element?.localName === 'use' && href?.startsWith('#') ? drawing.querySelector(`[id="${CSS.escape(href.slice(1))}"]`) : null;
  }

  function resolvedPaint(id, attr, fallback = '') {
    const element = svgElement(id);
    if (!element) return fallback;
    const target = paintReference(element);
    return getComputedStyle(target?.hasAttribute(attr) ? target : element).getPropertyValue(attr).trim() || fallback;
  }

  // Any CSS colour as the browser computes it, rgb(), which colorHex reads.
  function cssColour(value) {
    const probe = document.createElement('span'); probe.style.color = value; document.body.append(probe);
    const colour = getComputedStyle(probe).color; probe.remove(); return colour;
  }

  // A fill or stroke of url(#id) names a linear gradient in the drawing's defs.
  function paintGradient(value) {
    const id = value?.match(/^url\(\s*["']?#([^"')\s]+)["']?\s*\)$/)?.[1];
    // Paint in the drawing names its ID with the art- prefix; a document ID has none.
    const element = id && (drawing.querySelector(`[id="${CSS.escape(id)}"]`) || svgElement(id));
    return element?.localName === 'linearGradient' ? element : null;
  }

  function gradientStops(gradient) {
    return [...gradient.querySelectorAll('stop')].map(stop => {
      const offset = stop.getAttribute('offset') || '0';
      const value = offset.endsWith('%') ? parseFloat(offset) / 100 : Number(offset);
      return {offset: Math.max(0, Math.min(1, value || 0)), colour: getComputedStyle(stop).stopColor || stop.getAttribute('stop-color') || 'black', opacity: Number(stop.getAttribute('stop-opacity') ?? 1)};
    });
  }

  // The gradient as a left-to-right CSS gradient, for swatches.
  function gradientCss(value) {
    const gradient = paintGradient(value); if (!gradient) return null;
    const stops = gradientStops(gradient);
    return stops.length ? `linear-gradient(90deg, ${stops.map(s => {
      const hex = colorHex(s.colour);
      const alpha = hex?.length === 9 ? parseInt(hex.slice(7), 16) / 255 : 1;
      const colour = hex ? hex.slice(0, 7) + Math.round(alpha * s.opacity * 255).toString(16).padStart(2, '0') : s.colour;
      return `${colour} ${(s.offset*100).toFixed(1)}%`;
    }).join(', ')})` : null;
  }

  // One colour where a single one is shown: a gradient's first stop.
  function paintColour(value) {
    const gradient = paintGradient(value);
    return gradient ? gradientStops(gradient)[0]?.colour || 'none' : value;
  }

  function paintSource(id, kind) {
    let element = svgElement(id);
    const target = paintReference(element);
    if (target?.hasAttribute(kind)) return `From referenced ${object(target.dataset.objectId)?.label || 'shape'}`;
    while (element && drawing.contains(element)) {
      if (element.hasAttribute(kind)) {
        const source = element.dataset.objectId;
        return source === id ? 'Set on this object' : `Inherited from ${object(source)?.label || 'SVG root'}`;
      }
      element = element.parentElement;
    }
    return 'SVG default';
  }

  function swatchPaint(element) {
    const style = getComputedStyle(element);
    // A referenced path can supply its own paint over the use's inherited paint.
    const target = paintReference(element);
    const fill = target?.getAttribute('fill') || style.fill;
    const stroke = target?.getAttribute('stroke') || style.stroke;
    return {fill, stroke};
  }

  function paintSwatch(swatch, item) {
    const element = svgElement(item.id); if (!element) return;
    const own = item.tag === 'linearGradient' && gradientCss(`url(#${item.id})`);
    if (own) {
      swatch.style.background = own;
      swatch.title = 'Linear gradient: paints the shapes that use it';
      swatch.setAttribute('aria-hidden', 'true');
      return;
    }
    if (item.resource) {
      swatch.classList.add('resource-symbol');
      swatch.textContent = item.tag === 'defs' ? '◇' : element.closest('clipPath') ? '▧' : '⌁';
      swatch.title = 'Definition only — not painted on the canvas';
      swatch.setAttribute('aria-hidden', 'true');
      return;
    }
    const paint = swatchPaint(element);
    const display = color => colorHex(color) || color;
    if (item.tag === 'g') {
      const colors = [...new Set([...element.querySelectorAll('path,rect,circle,ellipse,line,polyline,polygon,use')]
        .filter(el=>!el.closest('defs,clipPath')).map(el=>{const p=swatchPaint(el);return paintColour(p.fill !== 'none' ? p.fill : p.stroke);}).filter(c=>c !== 'none'))];
      const shown = colors.slice(0,4);
      swatch.classList.add('group-swatch');
      swatch.style.background = shown.length === 1 ? shown[0] : shown.length ? `conic-gradient(${shown.map((c,i)=>`${c} ${i*100/shown.length}% ${(i+1)*100/shown.length}%`).join(',')})` : 'transparent';
      swatch.title = colors.length ? `Group colors: ${shown.map(display).join(', ')}${colors.length > shown.length ? ` (+${colors.length-shown.length} more)` : ''}` : 'No paint';
    } else {
      swatch.classList.toggle('no-paint', paint.fill === 'none' && paint.stroke === 'none');
      const ramp = gradientCss(paint.fill);
      if (ramp) swatch.style.background = ramp;
      else swatch.style.backgroundColor = paint.fill === 'none' ? 'transparent' : paint.fill;
      if (paint.stroke !== 'none') { swatch.style.borderColor = paintColour(paint.stroke); swatch.style.borderWidth = '3px'; }
      const named = color => paintGradient(color) ? 'linear gradient' : display(color);
      swatch.title = `Fill: ${named(paint.fill)} (${paintSource(item.id,'fill')}) · Stroke: ${named(paint.stroke)} (${paintSource(item.id,'stroke')})`;
    }
    swatch.setAttribute('aria-hidden', 'true');
  }

  function objectContext(item) {
    const element = svgElement(item.id);
    const source = object(paintReference(element)?.dataset.objectId);
    const clipId = item.attributes['clip-path']?.match(/^url\(#(.+)\)$/)?.[1];
    const clip = object(clipId);
    const inClip = !!element?.closest('clipPath');
    let role = '';
    if (item.tag === 'defs') role = 'Not drawn · reusable geometry';
    else if (item.tag === 'linearGradient') role = 'Linear gradient · paint, not drawn';
    else if (item.tag === 'stop') role = 'Gradient stop';
    else if (item.tag === 'clipPath') role = 'Clipping boundary · not drawn';
    else if (item.tag === 'use') role = `${inClip ? 'Clip contour' : item.resource ? 'Shared instance' : 'Instance'} of ${source?.label || 'missing source'}`;
    else if (inClip) role = 'Clip contour · not drawn';
    else if (item.resource) role = 'Shared geometry · not drawn';
    else if (clip) role = `Clipped by ${clip.label}`;
    return {source, clip, inClip, role};
  }

  function paintValue(attr, fallback = '') {
    const values = editor.state.selection.objects.map(id => resolvedPaint(id, attr, fallback));
    return values.length && values.every(v => v === values[0]) ? values[0] : '';
  }

  function renderRelationships(item) {
    const section = $('object-relationships'), links = $('relationship-links');
    links.replaceChildren(); section.hidden = true;
    const context = item && objectContext(item);
    const clipOnly = editor.state.selection.objects.some(id => {
      const selected = object(id);
      return selected?.tag === 'defs' || objectContext(selected).inClip;
    });
    $('paint-section').hidden = clipOnly;
    $('paint-heading').textContent = item?.resource ? 'Shared paint' : 'Paint';
    if (!item) return;
    $('selection-kind').textContent = item.tag === 'linearGradient' ? 'Shared gradient' : item.tag === 'defs' ? 'Definitions' : context.inClip ? 'Clipping' : item.resource ? 'Shared geometry' : item.tag === 'use' ? 'Instance' : item.tag;
    const consumers = editor.state.objects.filter(candidate => {
      const relation = objectContext(candidate);
      return relation.source?.id === item.id || relation.clip?.id === item.id;
    });
    if (!context.role && !consumers.length) return;
    section.hidden = false;
    let explanation = context.role;
    if (item.tag === 'defs') explanation = 'Reusable definitions. These entries do not draw anything by themselves.';
    else if (context.inClip) explanation = 'This defines a clipping boundary. Shapes using it are only visible inside the blue outline. Use Nodes (S) to edit a clipping path’s points, or Show clips in the footer to reveal other boundaries.';
    else if (item.tag === 'linearGradient') explanation = 'This gradient supplies paint to the shapes that use it. Editing it updates all of them.';
    else if (item.resource) explanation = 'This geometry is stored for reuse, not drawn directly. Its instances supply the visible paint unless shared paint is set here.';
    else if (context.source) explanation = 'This draws an instance of the shared geometry below, using this instance’s paint and position.';
    else if (context.clip) explanation = 'This artwork is clipped: only the parts inside its clipping boundary are visible.';
    $('relationship-description').textContent = explanation;
    function link(label, target) {
      const button = document.createElement('button');
      button.textContent = `${label}: ${target.label}`;
      button.onclick = () => later(() => object(target.id) && selectObject(target.id, false, true));
      links.append(button);
    }
    if (context.source) link('Source geometry', context.source);
    if (context.clip) link('Clipping boundary', context.clip);
    if (context.inClip && item.tag !== 'clipPath') {
      const parent = object(svgElement(item.id).closest('clipPath').dataset.objectId);
      if (parent) link('Clipping boundary', parent);
    }
    for (const consumer of consumers) link(objectContext(consumer).inClip ? 'Used for clipping by' : consumer.tag === 'use' ? 'Drawn by' : 'Clips', consumer);
  }

  // Disable a control with the reason as its tooltip, or enable it with its own.
  function enable(element, reason) {
    const button = typeof element === 'string' ? $(element) : element;
    button.dataset.title ??= button.title;
    button.disabled = !!reason;
    button.title = reason || button.dataset.title;
  }

  // The right panel shows the selection's properties: with points selected their
  // coordinates and handles first, then the objects' name, paint and locks, and
  // the actions on them.
  function renderInspector() {
    const selected = editor.state.selection.objects, item = oneObject();
    $('object-name-section').hidden = !item;
    $('object-name').value = item?.name || '';
    $('object-name').placeholder = item?.label || 'Automatic name';
    $('object-name').dataset.objectId = item?.id || '';
    $('empty-inspector').hidden = !!selected.length;
    $('properties').hidden = !selected.length;
    $('selection-kind').textContent = item?.tag || (selected.length ? 'Multiple' : 'Drawing');
    for (const controls of document.querySelectorAll('[data-tools]')) controls.hidden = !controls.dataset.tools.split(' ').includes(editor.tool);
    renderNodeInspector(false);
    $('empty-reference-hint').hidden = !!editor.state.reference;
    renderRelationships(item);
    // Empty selections have no paint to resolve; keep the remaining controls reset.
    for (const kind of selected.length ? ['fill', 'stroke'] : []) {
      const value = paintValue(kind, kind === 'fill' ? 'black' : 'none');
      // A gradient shows as its ramp; picking a colour makes the paint flat.
      const ramp = gradientCss(value);
      const hex = colorHex(ramp ? cssColour(paintColour(value)) : value);
      $(`${kind}-value`).value = ramp ? '' : hex || value;
      $(`${kind}-value`).placeholder = ramp ? 'Linear gradient' : selected.length ? 'Mixed colors' : '';
      const picker = $(`${kind}-color`);
      picker.value = hex?.slice(0,7) || '#000000';
      picker.classList.toggle('no-paint', value === 'none');
      picker.classList.toggle('mixed-paint', !value);
      picker.classList.toggle('gradient-paint', !!ramp);
      picker.style.background = ramp || '';
      const sources = [...new Set(selected.map(id=>paintSource(id,kind)))];
      const source = !value ? 'Mixed colors' : sources.length === 1 ? sources[0] : 'Different paint sources';
      $(`${kind}-source`).textContent = ramp ? ` ${paintGradient(value)?.getAttribute('data-vectrify-paint-owner') ? 'Private linear gradient' : 'Shared linear gradient'} · ${source}. Pick a colour to make it flat.`.trim() : source;
      picker.title = `${kind === 'fill' ? 'Fill' : 'Stroke'}: ${ramp ? 'linear gradient' : hex || value || 'mixed'} · ${source}`;
    }
    renderFillGradient(item);
    renderStrokeControls();
    renderTransform();
    const strokeWidth = paintValue('stroke-width', '1');
    $('stroke-width').value = strokeWidth ? parseFloat(strokeWidth) : '';
    const opacity = paintValue('opacity', '1'); $('opacity').value = opacity === '' ? '' : Math.round(Number(opacity) * 100);
    document.querySelectorAll('[data-lock]').forEach(input => {
      input.disabled = !item; input.checked = item?.locks.includes(input.dataset.lock) || false;
      input.parentElement.title = item ? '' : 'Select one object to change its locks';
    });
    renderCommands();
    scheduleStrip();
  }

  // Paint and offsets apply in order with other input, after any edit under way.
  const paint = changes => later(() => action('paint', {changes}));

  function wire() {
    for (const kind of ['fill','stroke']) {
      $(`${kind}-value`).onchange=event=>paint({[kind]:event.target.value.trim()||null});
      $(`${kind}-color`).onchange=event=>paint({[kind]:event.target.value});
      $(`${kind}-none`).onclick=()=>paint({[kind]:'none'});
    }

    $('stroke-width').onchange=event=>paint({'stroke-width':event.target.value||null});

    $('opacity').onchange=event=>paint({opacity:String(Number(event.target.value)/100)});

    document.querySelectorAll('[data-lock]').forEach(box=>box.onchange=()=>{
      const item=oneObject(), locks=[...document.querySelectorAll('[data-lock]:checked')].map(el=>el.dataset.lock);
      if(item)later(()=>object(item.id)&&action('locks',{object:item.id,locks}));
    });

    $('object-name').onchange = event => {
      const input = event.target, item = object(input.dataset.objectId);
      const name = input.value;
      if (item && name.trim() !== item.name) later(() => object(item.id) && action('rename', {object:item.id, name}, 'Renaming object…'));
    };

    $('object-name').onkeydown = event => {
      if (event.key === 'Enter') {event.preventDefault();event.target.blur();}
      if (event.key === 'Escape') {event.preventDefault();event.target.value=oneObject()?.name || '';event.target.blur();}
    };
      return api;
  }
  const api = {paintReference, resolvedPaint, cssColour, paintGradient, gradientStops, gradientCss, paintColour, paintSource, swatchPaint, paintSwatch, objectContext, paintValue, renderRelationships, enable, renderInspector, paint, wire};
  return api;
}
