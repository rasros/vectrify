"""How an agent works with Vectrify: the server's instructions and its guide."""

_LOOP = """\
Vectrify edits SVG drawings, usually traced from a reference image. Work in a
loop: look, make one edit, look again, and undo() an edit that made it worse.
Every edit is one undo step labelled "Agent: ..." in the editor's history and
one revision. Every edit names its targets: ids are object ids, points are
[object id, node id] pairs; no tool acts on the current selection. All
coordinates are document coordinates and regions are [x, y, w, h]. If an
edit is refused because the drawing changed, describe() again: someone else
edited it.

For X use Y:
- what the person is looking at: view() (their selection, zoom, visible
  region); render(region="view") shows it
- what is under a spot: pick(x, y, radius); in an area: describe(region=...)
  (objects and the contours of figure-wide paths that paint there)
- a path's nodes: points(id, region=...) or points(id, contours=[i])
- how the drawing looks: render(region, overlay="side", grid=true); every
  image's text says how its pixels map to document coordinates
- the reference alone: render(region, overlay="reference")
- where it differs from the reference: compare(region); colours at a spot:
  pick(x, y) (its colour)
- the shape to match: trace_reference(region) outlines the reference's dark
  (or colour=...) areas as path data
- paint, name, locks: properties(ids, ...); move or scale: transform;
  stacking and groups: arrange
- join anything (points, line ends, filled outlines): join
- take a piece out of a path: extract(region); delete it: delete(region=...);
  delete objects, points or contours: delete(ids | points, contours=...)
- holes: points(id) marks them (hole=true); holes(contours, action=...)
- move nodes: set_points; handles and pins: point_style; new shapes: add_path
- jobs (generate, tidy, ...): job(id, action="status" | "apply" | ...)
- save as SVG or project: save(path)"""

INSTRUCTIONS = (
    _LOOP
    + """

open(path) edits a file headlessly; with the editor running and "Agents"
allowed in its footer, the server attaches to that window (or call
connect()). Read the vectrify://guide resource for the details."""
)

# The server the editor hosts: always on the window that allows agents.
WINDOW_INSTRUCTIONS = (
    _LOOP
    + """

This server is the editor itself: every tool works on the drawing in the
window that allows agents, and the person sees each edit as you make it.
Read the vectrify://guide resource for the details."""
)

GUIDE = """\
# Working on a drawing with Vectrify

## Targets

- **A file.** `open(path)` loads an `.svg` or `.vectrify` project into a
  headless editor in the server. `save()` writes it back (`.vectrify` keeps
  locks, pins, the reference and the selection; `.svg` is plain SVG);
  `save(path)` writes elsewhere: plain SVG to a `.svg` path, a project to a
  `.vectrify` one. `load_reference(path)` loads the reference image;
  `load_reference()` removes it.
- **The editor window.** When the person has the editor open and has turned
  on *Agents* in its footer, the server attaches to that window by itself
  (or call `connect()`). The editor also hosts this server itself, at the
  URL its Agents popover shows; a client added there is always on that
  window, and has no `open` or `connect`. Every edit shows in the window
  as you make it and lands in its undo history, and the objects you
  touched flash briefly. The person's selection stays theirs: your edits
  never change it.
- Both have the same drawing tools. `describe()` says which one you are on.

## Coordinates

Everything is in document coordinates (the root's user space): regions are
`[x, y, w, h]`, points `[x, y]`. A path's own coordinates differ when it or
its groups have a transform (a cel trace's group usually does);
`points()` and `set_points()` work in document coordinates unless you ask
for `coords="local"`.

## The loop

1. `describe()`: the artboard, the reference, the person's selection (for
   context only), and the objects
   (id, label, tag, parent, paint, bounds, locks), a page at a time
   (`page`, `page_size`, or `within` a group id).
2. Ask what the person is looking at: `view()` gives their selection
   (objects, points), and in the editor window the visible region, the
   zoom, the active tool, the entered group and the reference view.
   `render(region="view")` renders exactly that. You never change them.
3. Find things by place, not by paging: `pick(x, y, radius)` lists what
   paints at a spot, front to back, each with its groups and, for a path,
   the contours there (index, first node, node count, bounds). Strokes
   count with their width, through transforms; a stroke's bounding box
   does not. `describe(region=[x, y, w, h])` does the same for an area.
   Traced line art is often a few paths spanning the whole figure, each
   holding hundreds of contours: these are how to find the ones you want.
4. Look: `render()` is the drawing; `render(overlay="side")` puts the
   reference beside it, `overlay="over"` blends them. `region` zooms into
   one part at the same pixel budget. `grid=true` draws lines labelled with
   document coordinates. Each image's text gives the mapping: which document
   point the top-left pixel is and the units per pixel, so a pixel (px, py)
   is (x + px * ux, y + py * uy). `overlay="reference"` is the reference
   alone.
5. Measure: `compare(region)` gives the mean squared error against the
   reference, a heat map (black agrees, red to white differs) and the worst
   cells of a 4 x 4 grid over the region, each a region to look at next.
   `pick(x, y, radius)` also gives the drawing's and the reference's mean
   colour there and their difference.
6. Edit, one change at a time. Each call is one undo step and one revision.
7. Look again. If it got worse, `undo()`. `history()` lists the steps,
   newest first, with who made each (person or agent).

## Editing

- Objects: `properties(ids, fill, stroke, ..., name, locks)` (paint, a
  name for one object, locks; one step), `transform(ids, dx, dy, scale,
  anchor)` (move and/or scale) or `transform(ids, box=...)` (fit to a box),
  `arrange(ids, to="front")` (restack) or `arrange(ids, parent, index)`
  (move into a group), `group`, `ungroup`, `split_parts`, `cut_hole` (two
  paths: the inner one cuts the outer), `convert(ids, to)` (`line`: fill to
  centre line, `fill`: stroke to fill, `path`: an instance or shared
  geometry into an editable path of its own), `delete(ids)`, `knife`.
- `join` is the editor's Join: `join(points=[a, b])` joins two points;
  `join(ids)` joins stroked lines at their ends within `reach`, or merges
  filled paths into one outline (`color_source` picks the paint). `joined`
  in the answer says which it did.
- Holes: `points(id)` marks each contour of a filled path that is a hole
  (`hole=true`, with its `area`); `holes(contours=[[path, contour id]],
  action="fill")` fills them (`delete_enclosed` also deletes the shapes
  inside), `action="shape"` makes them shapes of their own.
- New shapes: `add_path(d, fill, stroke, ...)`, path data in document
  coordinates (more subpaths are holes). It goes into the group of what is
  drawn under it, just above that, so a fix lands among the shapes it
  fixes; `placed` in the answer says where. Give `parent` and `index` to
  choose.
- Points: `points(id)` lists a path's contours (index, id, first node,
  count, closed, bounds) and their nodes (id, i, command, values, pinned),
  a page at a time; a `more` field says when there is another page.
  `region` keeps the contours crossing it and the nodes inside it,
  `contours=[i, ...]` those contours, `nodes=false` the summaries only.
  Node values are the SVG command's numbers: `[x, y]` for M and L, `[c1x,
  c1y, c2x, c2y, x, y]` for C. `set_points({id: {node: values}})` moves
  them; `point_style(points, handles=0|1|2, pinned=...)`, `break_points`
  (given a segment's two end points, it deletes that segment, as Break
  does), `split_edge`, `delete(points=...)` (with `contours=true` the whole
  contours they are on), `join(points=[a, b])`.
- Pieces: `extract(region)` takes the contours inside a region (a
  rectangle or a polygon `[[x, y], ...]`) out of every unlocked path
  painting there (or the paths `ids`) into new paths with the same paint,
  just above them in their group. `cut=true` (the default) cuts strokes
  where they cross the region's edge and splits fills crossing it along
  the edge; a fill merely around the region is left alone. `cut=false`
  takes whole contours only. `group=true` puts the new paths in one new
  group. Shapes (rect, circle, ellipse, line) are cut as paths; instances
  (use) need `detach=true`. `delete(region=...)` deletes them instead
  (whole contours unless `cut=true`). Each is one undo step.
- The shape to match: `trace_reference(region)` outlines the reference's
  dark areas there (`colour="#..."` for areas of one colour; `tolerance`
  and `min_area` tune it) as closed path data in document coordinates,
  largest first. Compare it with `points(id, region)` and move nodes, or
  draw it with `add_path`.
- `redraw_outline(id, points)` redraws the stretch of an outline a stroke
  runs along, as the Redraw tool does; its ends must lie on the outline.
- Every edit takes its targets: `ids` for object tools (at least one),
  `points` for point tools, `contours` for `holes`. There is no selection to
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
with ink lines, `colour-regions` for posterised regions). `tidy(ids)`,
`fit_colours(ids, fill="flat"|"linear")` (linear fills are private to each
shape and described in its `fill_gradient` properties), `snap_edges(ids)` and
`cleanup(ids)` improve existing paths; `tidy(region=[x, y, w, h])` touches
up only the points inside an area, of every path painting there or of
`ids`. Each starts a job:
`job(id, wait_seconds=...)` waits for it and returns its metrics and
before/after previews, then `job(id, action="apply")` keeps the result as
one undo step or `job(id, action="discard")` drops it; `action="stop"`
stops it early. Look at the previews and metrics before applying.

## Budgets

Renders are PNGs of at most 2048 pixels on the long side (1024 by default);
ask for a smaller `max_side` for a quick look and a region for a close one.
Repeated renders of an unchanged drawing are cached. `points` lists 300
nodes a page by default; narrow it with `region` or `contours` on a big
path.
"""
