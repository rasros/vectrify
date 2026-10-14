import {DRAW_TOOLS} from './shapes.js';

export const GENERAL_TOOLS = [...DRAW_TOOLS, 'knife', 'region', 'gradient', 'hand'];

// The secondary tool menu uses the same tool setter and shortcuts as the rail.
export function wireToolMenuKeys(menu, document) {
  menu.addEventListener('keydown', event => {
    const items = [...menu.querySelectorAll('[role="menuitemradio"]')];
    const index = items.indexOf(document.activeElement);
    let next;
    if (event.key === 'ArrowDown') next = (index + 1) % items.length;
    if (event.key === 'ArrowUp') next = (index + items.length - 1) % items.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = items.length - 1;
    if (next === undefined || !items.length) return;
    event.preventDefault(); event.stopPropagation(); items[next].focus();
  });
}
