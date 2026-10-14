// Small helpers shared by the editor's modules.

const counted = format => (count, one, many = `${one}s`) => `${format(count)} ${count === 1 ? one : many}`;

// A count and the noun for it: plural(3, 'path') is '3 paths'.
export const plural = counted(String);

// The same with the count written for the locale, thousands separated.
export const pluralLocale = counted(count => count.toLocaleString());

// The selected objects of *state* in stacking order, back to front.
export const selectedObjects = state => state.objects.filter(item => state.selection.objects.includes(item.id));

// The one selected object, looked up by *object*, or null.
export const soleObject = (selection, object) => selection.objects.length === 1 ? object(selection.objects[0]) : null;
