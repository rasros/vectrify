

// Colors owns its controls; editor contains the live state shared with other controllers.
export function createColorsController(editor, {} = {}) {

  function colorHex(value) {
    const rgb = value.match(/^rgba?\(([^)]+)\)$/i);
    if (!rgb) return /^#[a-f\d]{6}$/i.test(value) ? value : null;
    const channels = rgb[1].split(/[, /]+/).map(Number);
    if (channels.length < 3 || channels.some(n => !Number.isFinite(n))) return null;
    return '#' + channels.slice(0,3).map(n=>Math.round(n).toString(16).padStart(2,'0')).join('') +
      (channels.length > 3 && channels[3] < 1 ? Math.round(channels[3]*255).toString(16).padStart(2,'0') : '');
  }

  function wire() {
      return api;
  }
  const api = {colorHex, wire};
  return api;
}
