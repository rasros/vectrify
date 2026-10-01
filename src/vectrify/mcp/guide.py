"""How an agent works with Vectrify: the server's instructions and its guide."""

_LOOP = """\
Vectrify edits SVG drawings, usually traced from a reference image. Work in a
loop: describe() the drawing, look with render() (overlay="side" to set it
beside the reference) and compare() (error and a heat map of where it
differs), make one edit, render or compare again, and undo() an edit that
made it worse. Every edit is one undo step labelled "Agent: ..." in the
editor's history. Every edit names its targets: ids are object ids from
describe(), points are [object id, node id] pairs from points(); no tool acts
on the current selection. If an edit is refused because the
drawing changed, describe() again: someone else edited it."""

INSTRUCTIONS = (
    _LOOP
    + """ open(path) edits
a file headlessly; with the editor running and "Agents" allowed in its
footer, the server attaches to that window (or call connect()). Read the
vectrify://guide resource for the details."""
)

# The server the editor hosts: always on the window that allows agents.
WINDOW_INSTRUCTIONS = (
    _LOOP
    + """ This server is the
editor itself: every tool works on the drawing in the window that allows
agents, and the person sees each edit as you make it. Read the
vectrify://guide resource for the details."""
)

GUIDE = """\
# Working on a drawing with Vectrify

## Targets

- **A file.** `open(path)` loads an `.svg` or `.vectrify` project into a
  headless editor in the server. `save()` writes it back (`.vectrify` keeps
  locks, pins, the reference and the selection; `.svg` is plain SVG);
  `save(path)` or `export_svg(path)` write elsewhere.
- **The editor window.** When the person has the editor open and has turned
  on *Agents* in its footer, the server attaches to that window by itself
  (or call `connect()`). The editor also hosts this server itself, at the
  URL its Agents popover shows; a client added there is always on that
  window, and has no `open` or `connect`. Every edit shows in the window
  as you make it and lands in its undo history, and the objects you
  touched flash briefly. The person's selection stays theirs: your edits
  never change it.
- Both have the same drawing tools. `describe()` says which one you are on.

## The loop

1. `describe()`: the artboard, the reference, the person's selection (for
   context only), and the objects
   (id, label, tag, parent, paint, bounds, locks), a page at a time
   (`page`, `page_size`, or `within` a group id).
2. Look: `render()` is the drawing; `render(overlay="side")` puts the
   reference beside it, `overlay="over"` blends them. `region=[x, y, w, h]`
   in document units zooms into one part at the same pixel budget, which is
   how to inspect detail. `reference(region)` is the reference alone.
3. Measure: `compare(region)` gives the mean squared error against the
   reference, a heat map (black agrees, red to white differs) and the worst
   cells of a 4 x 4 grid over the region, each a region to look at next.
4. Edit, one change at a time. Each call is one undo step.
5. Look again. If it got worse, `undo()`. `history()` lists the steps,
   newest first, with who made each (person or agent).

## Editing

- Objects: `paint`, `rename`, `locks`, `move`, `resize` (scale about an
  anchor, or fit to a box), `reorder`, `move_into`, `group`, `ungroup`,
  `join` (merge outlines), `join_ends` (close gaps between line ends),
  `split_parts`, `cut_hole` (two paths: the inner one cuts the outer),
  `fill_holes` and `holes_to_shapes` (holes from `holes(id)`), `detach`,
  `convert` (fill to centre line, stroke to fill), `delete`, `knife`.
- New shapes: `add_path(d, fill, stroke, ...)` with one SVG subpath.
- Points: `points(id)` lists a path's contours and nodes. Node values are
  the SVG command's numbers: `[x, y]` for M and L, `[c1x, c1y, c2x, c2y, x,
  y]` for C. `set_points({id: {node: values}})` moves them; `handles`,
  `pin`, `break_points`, `delete_segment`, `split_edge`, `delete_points`,
  `delete_contours`, `join_points`.
- `redraw_outline(id, points)` redraws the stretch of an outline a stroke
  runs along, as the Redraw tool does; its ends must lie on the outline.
- Every edit takes its targets: `ids` for object tools (at least one),
  `points` for point tools, `contours` for holes. There is no selection to
  set or rely on. An edit's answer gives `result`, the objects (and points)
  it left to work on next, such as the new group or the cut pieces, and
  `created` and `removed`.
- `knife` without `ids` cuts every unlocked path it crosses (`within` a
  group if given).
- Locked properties and pinned points are enforced: a
  refusal is a tool error with the editor's reason. Do not work around it;
  tell the person if a lock is in the way.

## Operations

`generate(method, settings, group)` traces the reference into new shapes,
over the whole drawing or into the area of `group` (`cel` for flat colour
with ink lines, `colour-regions` for posterised regions; `samvg` needs a
GPU). `tidy(ids)`, `fit_colours(ids, fill="flat"|"linear")`,
`snap_edges(ids)` and `cleanup(ids)` improve existing paths. Each starts a
job:
`job_status(id, wait_seconds)` waits for it and returns its metrics and
before/after previews, then `apply(id)` keeps the result as one undo step
or `discard(id)` drops it. Look at the previews and metrics before applying.

## Budgets

Renders are PNGs of at most 2048 pixels on the long side (1024 by default);
ask for a smaller `max_side` for a quick look and a region for a close one.
Repeated renders of an unchanged drawing are cached.
"""
