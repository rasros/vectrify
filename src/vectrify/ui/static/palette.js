// Command palette matching: pure functions over plain command records, so
// they can be checked outside the browser; app.js holds the commands and the
// dialog.

// How well *query* matches *text*, higher is better, or 0 for no match: the
// whole text, then its start, the start of a word, anywhere, and last its
// letters in order unless *fuzzy* is false. Case and accents are ignored.
export function matchScore(text, query, fuzzy = true) {
  const fold = value => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const t = fold(text), q = fold(query).trim();
  if (!q) return 1;
  if (t === q) return 100;
  if (t.startsWith(q)) return 80;
  const at = t.indexOf(q);
  if (at > 0 && /[^a-z0-9]/.test(t[at - 1])) return 60;
  if (at > 0) return 40;
  // Every word of the query at the start of a word of the text, in any order.
  const words = t.split(/[^a-z0-9]+/).filter(Boolean), parts = q.split(/\s+/);
  if (parts.length > 1 && parts.every(part => words.some(word => word.startsWith(part)))) return 30;
  if (!fuzzy) return 0;
  let i = 0;
  for (const letter of t) if (letter === q[i]) i++;
  return i === q.length ? 10 : 0;
}

// The commands matching *query*, best first; ties keep their order. A command
// matches on its name, or less well on its group and keywords.
export function matchCommands(commands, query) {
  return commands.map((command, index) => {
    const name = matchScore(command.name, query);
    const other = Math.max(matchScore(command.group || '', query, false), matchScore(command.keywords || '', query, false)) / 2;
    return {command, index, score: Math.max(name, other)};
  }).filter(item => item.score > 0).sort((a, b) => b.score - a.score || a.index - b.index).map(item => item.command);
}

// The row to highlight after moving *step* rows from *current* among *count*,
// wrapping round.
export function moveHighlight(current, step, count) {
  if (!count) return -1;
  return ((current + step) % count + count) % count;
}
