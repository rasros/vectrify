import {loadErrorImage} from './reference_error.js';

// One lookup per squared RGB distance. Packing through a Uint8 array keeps
// the byte order correct on either endian architecture.
let colours;
function errorColours() {
  if (colours) return colours;
  const bytes = new Uint8Array((3 * 255 ** 2 + 1) * 4);
  for (let sum = 0; sum <= 3 * 255 ** 2; sum++) {
    const d = Math.sqrt(sum / 3) / 255 * 3, offset = sum * 4;
    bytes[offset] = Math.min(1, d) * 255;
    bytes[offset + 1] = Math.max(0, Math.min(1, d - 1)) * 255;
    bytes[offset + 2] = Math.max(0, Math.min(1, d - 2)) * 255;
    bytes[offset + 3] = 255;
  }
  colours = new Uint32Array(bytes.buffer);
  return colours;
}

export function errorPixels(actual, expected) {
  if (actual.length !== expected.length || actual.length % 4) throw new Error('Comparison images must have the same size');
  const lookup = errorColours(), packed = new Uint32Array(actual.buffer, actual.byteOffset, actual.length / 4);
  for (let p = 0; p < actual.length; p += 4) {
    const r = actual[p] - expected[p], g = actual[p + 1] - expected[p + 1], b = actual[p + 2] - expected[p + 2];
    packed[p / 4] = lookup[r * r + g * g + b * b];
  }
  return actual;
}

// Match the server's artboard sampling and its reference resolution cap.
function roundPixels(value) {
  const rounded = Math.round(value);
  return value === rounded - 0.5 && rounded % 2 ? rounded - 1 : rounded;
}

export function errorSize(bounds, box, sourceSize) {
  const [, , aw, ah] = bounds, [, , bw, bh] = box;
  const width = aw * (sourceSize[0] / bw), height = ah * (sourceSize[1] / bh);
  const shrink = Math.min(1, Math.sqrt(64_000_000 / (width * height)));
  const side = Math.min(2048, Math.max(1, roundPixels(width * shrink), roundPixels(height * shrink)));
  const scale = side / Math.max(aw, ah);
  return [Math.max(1, roundPixels(aw * scale)), Math.max(1, roundPixels(ah * scale))];
}

export function createErrorRenderer({
  document = globalThis.document,
  DOMParser = globalThis.DOMParser,
  XMLSerializer = globalThis.XMLSerializer,
  URL = globalThis.URL,
  Blob = globalThis.Blob,
  load = loadErrorImage,
} = {}) {
  let picture = null, target = null;

  function canvas(size) {
    const image = document.createElement('canvas');
    [image.width, image.height] = size;
    return image;
  }

  async function prepare(bounds, reference) {
    if (picture?.source !== reference.data_url) picture = {source: reference.data_url, image: load(reference.data_url)};
    const image = await picture.image;
    const box = reference.box || bounds;
    const size = errorSize(bounds, box, [image.naturalWidth, image.naturalHeight]);
    const key = [bounds.join(), box.join(), size.join()].join(';');
    if (target?.key !== key || target?.source !== reference.data_url) {
      const expected = canvas(size), ctx = expected.getContext('2d', {willReadFrequently: true});
      ctx.fillStyle = 'white'; ctx.fillRect(0, 0, ...size);
      const [ax, ay, aw, ah] = bounds, [bx, by, bw, bh] = box;
      ctx.drawImage(image, (bx - ax) * size[0] / aw, (by - ay) * size[1] / ah, bw * size[0] / aw, bh * size[1] / ah);
      target = {key, source: reference.data_url, pixels: ctx.getImageData(0, 0, ...size).data};
    }
    errorColours();
    return {size, pixels: target.pixels};
  }

  async function render(svg, bounds, reference) {
    // Capture the expected pixels before yielding to the SVG decoder.
    const {size, pixels: expected} = await prepare(bounds, reference);
    // The exported SVG already has valid path data and definitions. Reframe
    // only its opening tag instead of reparsing and serializing every path.
    const head = /^\s*<svg\b[^>]*>/.exec(svg)?.[0];
    const empty = head?.endsWith('/>');
    const source = head ? head + (empty ? '' : '</svg>') : svg;
    const root = new DOMParser().parseFromString(source, 'image/svg+xml').documentElement;
    root.setAttribute('width', size[0]); root.setAttribute('height', size[1]);
    root.setAttribute('viewBox', bounds.join(' ')); root.setAttribute('preserveAspectRatio', 'none');
    const framed = new XMLSerializer().serializeToString(root);
    const parts = head && !empty ? [framed.replace(/\/>$/, '>'), svg.slice(head.length)] : [framed];
    const url = URL.createObjectURL(new Blob(parts, {type: 'image/svg+xml'}));
    let drawing;
    try { drawing = await load(url); }
    finally { URL.revokeObjectURL(url); }
    const result = canvas(size), ctx = result.getContext('2d', {willReadFrequently: true});
    ctx.fillStyle = 'white'; ctx.fillRect(0, 0, ...size); ctx.drawImage(drawing, 0, 0);
    const pixels = ctx.getImageData(0, 0, ...size);
    errorPixels(pixels.data, expected); ctx.putImageData(pixels, 0, 0);
    return result;
  }
  render.prepare = prepare;
  return render;
}
