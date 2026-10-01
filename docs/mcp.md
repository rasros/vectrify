# MCP server (sketch)

Status: design agreed, nothing built yet. It replaces the removed LLM
features: instead of Vectrify calling a model, an agent (Claude Code, Claude
Desktop or any MCP client) calls Vectrify.

## Goals

- An agent can look at the drawing and the reference, and edit the drawing
  with the same commands a person has, at the same granularity.
- Every edit goes through `Session.action` / `Session.operation`, so
  selection scope, locks, pins, permissions and revision checks hold exactly
  as in the editor, and each tool call is one undoable step.
- The agent can work on a drawing the user has open and is watching, or on a
  file headlessly.

Not goals: a second editing model, free-form SVG replacement (lenient replay
is gone), or running models inside Vectrify.

## Shape

`vectrify-mcp` is a stdio MCP server (Python, the `mcp` SDK), started by the
client. It holds a *target*, one of:

1. **A file, headless.** `open(path)` loads an `.svg` or `.vectrify` into its
   own `Backend`/`Session` in process; `save()` writes it back. No UI.
2. **The running editor, live.** The editor opens a local channel for it, and
   the MCP server joins the same `Session` the window shows, so edits appear
   as they happen and land in the window's undo history.

Live is the useful one for "fix this part while I watch", so it is the
default when an editor is running.

### The live channel

The desktop app has no port today (pywebview calls `Backend` in process).
The editor serves an agent endpoint over HTTP on localhost: in `--serve`
mode on its own port, and in the desktop app on a port it opens when
"Allow agents to edit" is turned on for the window. It writes
`{url, token}` to `~/.local/state/vectrify/editor.json`, owner-only; every
request carries the token. The calls are the existing
`Backend.handle(path, data, session)` plus one to find the window's session
id. The footer shows when an agent is connected and what it last did.

Images are the bulk of the traffic (an agent renders after most edits), so
renders are PNG bytes in the HTTP response body, not base64 inside JSON, and
the MCP server passes them on as image content. Renders are capped in size
and can be cropped to a region, so a close look at one part does not cost a
full-size image; a render of an unchanged revision and region is cached.

## Tools

Grouped; each is a thin wrapper over existing session calls. Ids are object
ids as in the tree; points are `[object, node]` pairs.

**Looking**
- `describe()`: document size, reference loaded or not, selection, and the
  object tree (id, name, tag, parent, paint, bounds, locks), paged.
- `render(region?, overlay?)`: PNG of the drawing (optionally a region in
  document units, optionally side by side with or over the reference), as an
  MCP image. The main way the agent sees results.
- `reference(region?)`: PNG of the reference.
- `compare(region?)`: reference error (MSE) of the region and a diff heat map.
- `get_svg(ids?)`: SVG of objects, for exact geometry.
- `points(id)`: the path's nodes, handles and pins.

**Selecting**
- `select(objects?, points?)`.

**History**
- `history()`: the undo and redo stacks, each entry with its label, who made
  it (the person or the agent) and the revision after it.
- `undo(steps?)`, `redo(steps?)`; the agent's own edits are labelled
  "Agent: …" in the editor's history too.

**Editing** (one undo step each; refusals come back as tool errors with the
editor's message)
- `paint(ids, fill?, stroke?, stroke_width?, opacity?)`, `rename`, `locks`.
- `move(ids, dx, dy)`, `resize(ids, box)`, `reorder(ids, to)`.
- `group`, `ungroup`, `join`, `split_parts`, `cut_hole`, `delete`.
- `add_path(d, paint, parent?, index?)`, `set_points(id, changes)`,
  `handles(points, count)`, `break`, `join_points`, `split_edge`,
  `delete_points`.
- `knife(line)`, `redraw_outline(id, stroke)`.

**Operations** (jobs: start, then poll or wait; the agent decides to apply)
- `generate(method, settings, scope)`, `tidy(ids, settings)`,
  `fit_colours(ids, fill)`, `snap_edges(ids)`, `cleanup(ids)`.
- `job_status(id)`, `apply(id)`, `discard(id)`.

**Files** (headless target only, or with the user's consent live)
- `open(path)`, `save(path?)`, `export_svg(path)`, `load_reference(path)`.

## Safety

- The session is the enforcement: an agent can do nothing a person couldn't.
- Revisions: every edit carries the revision the agent last saw; a stale one
  is refused, so it never overwrites a person's concurrent edit. The agent
  re-reads with `describe()`.
- Live editing only when the window allows it; the token keeps other local
  processes out; file writes outside the opened file need consent.
- Budgets: renders are capped in size; `describe` pages large trees.

## Decisions

1. Both targets: headless files and the live editor, with the same tools.
2. The live channel is HTTP on localhost with a token; images travel as
   PNG bodies.
3. The agent's edits are labelled in the history, and it can inspect the
   history and undo or redo.
4. Typed tools only, no generic `action(command, payload)`; together they
   cover nearly everything the UI can do. A test checks every editor command
   is reachable from some tool, or listed as deliberately left out.
5. A `vectrify://guide` resource (and the server's instructions) explains the
   workflow: describe, render and compare, edit, render again, undo what
   made it worse.

## First prototype

Headless target only: `open`, `describe`, `render`, `reference`, `compare`,
`select`, `paint`, `add_path`, `set_points`, `delete`, `undo`, `generate` +
`apply`, `save`. Tested with an in-process MCP client and from Claude Code.
Then the live channel.
