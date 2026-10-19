# Agents

How to let an AI agent look at the drawing and edit it, over MCP.

## What an agent can do

vectrify is an MCP server. An agent such as Claude Code, Codex or Claude
Desktop can describe and render the drawing, compare it with the
reference, and edit it with the editor's own commands at the same
granularity a person has. It can do nothing a person couldn't: selection
scope, locks, pins and revision checks hold exactly as in the editor.

It runs two ways, with the same tools:

- Hosted by the editor, over HTTP, while Agents is on in a window's footer.
  The agent edits that window's drawing live.
- `vectrify-mcp` over stdio, started by the client. It edits a file
  headlessly, or attaches to an editor window that has Agents on.

Both need the `mcp` extra.

## Hosting from the editor

Turn on Agents in the footer (or Allow agents to edit in the palette). The
editor starts the server at `http://127.0.0.1:8770/mcp` and opens a
popover with the command to add it to Claude Code:

```bash
claude mcp add --transport http --scope user vectrify http://127.0.0.1:8770/mcp \
  --header "Authorization: Bearer <token>"
```

`--scope user` makes it available in every project; leave it out for the
current one only. Add it once: the token and the port stay the same from
one run to the next.

- The token is kept in `$XDG_STATE_HOME/vectrify/agent-token`. Regenerate
  token in the popover replaces it, and clients with the old one are
  refused.
- The port is remembered in `$XDG_STATE_HOME/vectrify/mcp-port`, and
  `vectrify --mcp-port N` asks for another. If the port is taken the
  server moves to the next free one and the popover warns in amber with
  the old and new URL; update clients added with the old one.
- The server binds to 127.0.0.1 and accepts requests only for that host
  and with the token. Turning Agents off, or quitting, stops it.

## Running over stdio

```bash
uv tool install "vectrify[mcp]"
claude mcp add vectrify -- vectrify-mcp
# without installing:
claude mcp add vectrify -- uvx --from "vectrify[mcp]" vectrify-mcp
# from a source checkout:
claude mcp add vectrify -- uv run --directory /path/to/vectrify --extra mcp vectrify-mcp
```

`vectrify-mcp drawing.svg` opens that file at start.

## Other apps

The popover's Add to an app section also shows what Codex and Claude
Desktop need to start `vectrify-mcp`, with this install's own path. These
need no token: started without a file, the server attaches to the window
with Agents on. Restart the app after adding it.

Codex reads `~/.codex/config.toml` (or `$CODEX_HOME/config.toml`).
`startup_timeout_sec` gives the server time to import its vision stack:

```toml
[mcp_servers.vectrify]
command = "/path/to/.venv/bin/vectrify-mcp"
startup_timeout_sec = 30
```

Claude Desktop reads `claude_desktop_config.json`, in
`~/Library/Application Support/Claude/` on macOS and `%APPDATA%\Claude\`
on Windows. Merge this into its `mcpServers`:

```json
{"mcpServers": {"vectrify": {"command": "/path/to/.venv/bin/vectrify-mcp"}}}
```

## Working alongside an agent

While Agents is on, the agent's edits appear within a few tens of
milliseconds, the objects each edit touched flash briefly, and the footer
reads Agent connected with its last action. If you are mid-drag, the edit
shows when you release.

- Your selection, entered group and zoom stay yours. Every agent edit
  names its own targets; none acts on your selection.
- Agent edits land in the undo history labelled "Agent: …". Your Undo and
  Redo affect only your own edits.
- Edits you and the agent make at the same time merge, even to different
  properties of one object. A real overlap refuses the agent's edit and
  says which field conflicted.

Only one window per editor takes agents at a time. Turning Agents on in
another window moves the agent there, and it must look again before it
edits.

## Targets

The stdio server edits one target at a time:

- A file: `open(path)` loads an `.svg` or `.vectrify` file, `save()` writes
  it back, and `save(path)` writes elsewhere. A `.vectrify` path keeps
  locks, pins and the reference.
- An editor window: with no file open, the first call attaches to the
  window with Agents on; `connect()` does so explicitly.

The hosted server always edits its own window and has no `open` or
`connect`. `describe()` says which target is in use.

## Tools

Object ids are those `describe()` lists; points are `[object, node]` pairs
from `points()`. All coordinates are document coordinates, and a region is
`[x, y, w, h]` or, for `extract` and `delete`, a polygon `[[x, y], ...]`.
Every edit names its targets; there is no lasting selection and no
`select` tool. Each edit is one undo step and answers with `edit_id`,
`result` (what it left selected for itself), `created` and `removed`.

### Looking

Looking changes nothing.

- `describe(page?, page_size?, within?, region?, objects?)`: the target,
  artboard, reference, the person's selection and view (visible region,
  zoom, tool, reference view), and the objects with their paint, painted
  bounds, transform, locks and hidden state, a page at a time. With
  `region`, only what paints there. `objects=false` gives just the
  context.
- `pick(x, y, radius?)`: what paints at a point, front to back, with each
  path's contours there, the drawing's and reference's colour, and their
  difference.
- `render(region?, overlay?, max_side?, grid?)`: a PNG of the drawing.
  `region="view"` is exactly what the window shows. `overlay` is `side`,
  `over` or `reference`, and `grid=true` labels document coordinates.
- `compare(region?, max_side?, grid?, edge_aware?)`: mean squared error
  against the reference, the worst cells, and a heat map. See
  [Edge comparison](#edge-comparison).
- `trace_reference(region?, colour?, dark?, tolerance?, min_area?)`: the
  reference's dark areas, or those near a colour, as closed path data, to
  use with `add_path`.
- `points(id, region?, contours?, coords?, nodes?)`: a path's contours,
  holes and nodes, 300 nodes a page.
- `get_svg(ids?)`: the SVG of the drawing or some objects.

Every image comes with a `mapping` from its pixels to document
coordinates.

### Editing

- `properties(ids, ...)`: fill, stroke, width, opacities, line caps and
  joins, arrows, tapers, name, locks and `hidden`, in one step. For
  example `properties(ids=["line"], end_taper=20, end_taper_width=0)`
  narrows the last 20 units to a point.
- `transform(ids, dx?, dy?, scale?, anchor?, box?, rotate?, flip?)`: move,
  scale, rotate and mirror about one anchor, or fit the painted bounds to
  `box`. Strokes keep their width.
- `arrange(ids, to)` (front, back, forward, backward) or
  `arrange(ids, parent, index?)` to move into a group.
- `group(ids, action?)`: `create` or `dissolve`.
- `add_path(d, fill?, stroke?, stroke_width?, parent?, index?, name?)`:
  without `parent` it goes just above the object under its centre, and
  `placed` says where.
- `delete(ids? | points? | region?, contours?, cut?, detach?)`.
- `join(ids? | points?, reach?, bridge?, color_source?)`: the editor's
  Join, which picks points, line ends or Merge shapes.
- `combine`, `split_parts`, `cut_hole`, `boolean(ids, operation)`,
  `holes(contours, action?)` and `convert(ids, to?)`, as the Path commands.
- `knife(start, end, ids?)` and `redraw_outline(id, points, long_way?)`.
- `extract(region, ids?, cut?, group?, detach?)`: the Region tool's
  Extract; `group=true` gathers the pieces into one new group.
- `clip(ids, boundary, linked?, inset?)` and `clip(ids, action="remove")`:
  Set clip and Remove clip. `inset` shrinks the clip, for example by half
  an outline's stroke width to keep shading off it.
- Points: `set_points`, `point_style(points, handles?, pinned?)`,
  `break_points`, `split_edge`.
- `load_reference(path?)` loads or removes the reference.

The agent cannot change the artboard, open a file in the person's window,
or set view controls; it reads the view with `describe()`.

### History

`history(limit?)` lists both authors' changes, newest first, with stable
ids. `undo(ids)` and `redo(ids)` must name exact entries, can restore
either author's, and keep unrelated later edits. A conflict refuses the
whole batch.

### Operations

`generate(method?, settings?, group?)`, `tidy(ids?, settings?, rounds?,
region?)`, `fit_colours(ids, fill?, passes?, resolution?)`,
`snap_edges(ids, tolerance?)` and `cleanup(ids)` start a job.
`job(id, action?)` follows it: `status` waits up to 120 s and returns the
metrics and previews, `apply` keeps the result, `discard` drops it and
`stop` stops it early.

`settings_schema(action, method)` returns a method's settings as JSON
schema; unknown keys and invalid values are refused before the job
starts. Each job reply reports its `effective_settings` and `diagnostics`
of what the proposal changes.

### Previews

`edit_batch(edits=[{tool, args, as?}], close_region?)` stages up to 100
edits without changing the drawing and returns before, after, reference
and difference views. `as` names a result for later edits as `$name`.
`edit_batch(action="apply", id=...)` commits it as one undo step;
`discard` drops it.

### Features and outlines

- `linked_outline(id, colour?, width?, layer?)` makes a stroke-only path
  sharing a fill's geometry, in a layer above the fills.
- `protect_features(points, kind)` marks tips, corners or junctions that
  Tidy and other edits must keep (`kind="none"` releases them).
  `trace_reference(protected_features=...)` keeps such points exact in a
  trace.
- `isolate_feature(region | seed)` numbers the contours in an area so the
  agent can name which ones form a feature, then stages their extraction
  into a named group.

### Edge comparison

`compare(edge_aware=true, edge_threshold?, edge_tolerance?)` adds edge
precision and recall and boundary displacement in document units, and an
annotated view: missing reference edges red, extra drawing edges orange,
matches green, duplicates blue. `feature_checks` checks that given points
sit where expected and still turn sharply.

## The guide

The server's instructions give a short "for X use Y" list, and the
`vectrify://guide` resource explains the working loop: look, edit, look
again, and undo what made it worse.

## Limits

- Renders are PNG, 1024 pixels on the long side by default and at most
  2048. They use Cairo and match the page closely but not pixel for pixel.
- `describe` pages large drawings, `pick` lists at most 20 objects, and
  `trace_reference` at most 40 shapes; ask for a smaller region for more
  detail.
- The view in `describe` is as of the window's last report, a quarter
  second after it changes.
- A job works on the objects it started with.
- Two clients with no session id and the same name and version share one
  record of what they have seen.

## For developers

- `vectrify/mcp/server.py` registers every tool for both servers;
  `vectrify/mcp/hosted.py` hosts it in the editor; `vectrify/ui/agent.py`
  answers calls on a session, so live and headless targets give identical
  answers.
- With Agents on, the editor writes `{url, token, pid, mcp}` to
  `$XDG_STATE_HOME/vectrify/editor.json` (mode 0600). The stdio server's
  `connect()` posts `{tool, args}` to `/agent/call` there and fetches images
  from `/agent/image/<key>`.
- The page hears of each agent call by server-sent events from
  `/api/events` with `--serve`, or by a script call in the desktop window,
  and polls `/api/poll` every 5 s as a fallback.
- A tool that runs several session commands squashes them into one history
  entry and one revision. Edits planned on an older revision are merged
  into the live drawing; see
  [Transactions](12-document-model.md#transactions).
- `tests/mcp/test_coverage.py` fails unless every session command is
  reachable from a tool or listed in `LEFT_OUT` in `vectrify/ui/agent.py`
  with a reason.

Next: [Shortcuts and limits](10-shortcuts.md)
